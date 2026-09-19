"""End-to-end inference pipeline for a single breast ultrasound image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from bus_pipeline.adaptive_roi import AdaptiveROI

import cv2
import numpy as np
import torch

from bus_pipeline.classification_models import build_classifier
from bus_pipeline.config import load_config
from bus_pipeline.constants import CLASS_NAMES
from bus_pipeline.explainability import GradCAM, overlay_gradcam
from bus_pipeline.localization import YOLOLocalizer, crop_roi_from_box, draw_yolo_box
from bus_pipeline.preprocessing import (
    build_classification_transforms,
    build_segmentation_transforms,
    preprocess_ultrasound_image,
    read_grayscale_image,
    to_three_channel,
)
from bus_pipeline.segmentation_models import build_segmentation_model
from bus_pipeline.size_estimation import estimate_tumor_size
from bus_pipeline.utils import ensure_dir, get_device, load_model_state, save_json
from bus_pipeline.visualization import draw_label, overlay_mask, save_inference_panel


@dataclass(frozen=True)
class InferencePaths:
    """Paths to trained artifacts."""

    yolo_weights: Path
    segmentation_checkpoint: Path
    classifier_checkpoint: Path


class BreastTumorAnalysisPipeline:
    """Run localization, segmentation, classification, sizing, and explainability."""

    def __init__(self, config: dict[str, Any], artifact_paths: InferencePaths | None = None) -> None:
        self.config = config
        self.device = get_device(config.get("device", "cuda"))
        path_cfg = config.get("paths", {})
        self.artifact_paths = artifact_paths or InferencePaths(
            yolo_weights=Path(path_cfg.get("yolo_weights", "outputs/yolo/best.pt")),
            segmentation_checkpoint=Path(path_cfg.get("segmentation_checkpoint", "outputs/checkpoints/segmentation_best.pt")),
            classifier_checkpoint=Path(path_cfg.get("classifier_checkpoint", "outputs/checkpoints/classifier_best.pt")),
        )

        self.localizer = YOLOLocalizer(self.artifact_paths.yolo_weights)

        self.segmenter = build_segmentation_model(config.get("segmentation", {}).get("model", {})).to(self.device)
        load_model_state(self.segmenter, self.artifact_paths.segmentation_checkpoint, self.device)
        self.segmenter.eval()

        self.classifier = build_classifier(config.get("classification", {}).get("model", {})).to(self.device)
        load_model_state(self.classifier, self.artifact_paths.classifier_checkpoint, self.device)
        self.classifier.eval()

        seg_size = config.get("segmentation", {}).get("image_size", 256)
        cls_size = config.get("classification", {}).get("image_size", 300)
        self.segmentation_transform = build_segmentation_transforms(seg_size, augment=False)
        self.classification_transform = build_classification_transforms(cls_size, augment=False)

    @classmethod
    def from_config_file(cls, config_path: str | Path, artifact_paths: InferencePaths | None = None) -> "BreastTumorAnalysisPipeline":
        return cls(load_config(config_path), artifact_paths=artifact_paths)

    @torch.no_grad()
    def _segment_roi(self, roi: np.ndarray, full_shape: tuple[int, int], roi_bbox: tuple[int, int, int, int]) -> np.ndarray:
        transformed = self.segmentation_transform(image=roi, mask=np.zeros_like(roi, dtype=np.uint8))
        tensor = transformed["image"].unsqueeze(0).to(self.device).float()
        probability = torch.sigmoid(self.segmenter(tensor)).squeeze().detach().cpu().numpy()
        x1, y1, x2, y2 = roi_bbox
        roi_mask = cv2.resize(probability, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        full_mask = np.zeros(full_shape, dtype=np.float32)
        full_mask[y1:y2, x1:x2] = roi_mask
        return roi_mask, full_mask

    @torch.no_grad()
    def _classify_roi(self, roi: np.ndarray) -> tuple[str, float, np.ndarray, torch.Tensor]:
        transformed = self.classification_transform(image=to_three_channel(roi))
        tensor = transformed["image"].unsqueeze(0).to(self.device).float()
        logits = self.classifier(tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        class_index = int(np.argmax(probabilities))
        return CLASS_NAMES[class_index], float(probabilities[class_index]), probabilities, tensor

    def _gradcam(self, roi: np.ndarray, tensor: torch.Tensor, class_index: int) -> np.ndarray:
        target_model = self.classifier
        target_layer = getattr(target_model, "features")[-1]
        cam = GradCAM(target_model, target_layer)
        try:
            heatmap = cam(tensor, class_idx=class_index)
        finally:
            cam.close()
        return overlay_gradcam(roi, heatmap)

    def predict(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        pixel_spacing: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Run the complete pipeline for one image and write visual outputs."""

        output_dir = ensure_dir(output_dir)
        image_path = Path(image_path)
        stem = image_path.stem
        original = read_grayscale_image(image_path)
        preprocessed = preprocess_ultrasound_image(original, self.config.get("preprocessing", {}))

        localization_cfg = self.config.get("localization", {})
        detection = self.localizer.predict(
            image_path,
            conf=localization_cfg.get("conf_threshold", 0.25),
            iou=localization_cfg.get("iou_threshold", 0.7),
        )
        ##########################################################
# Adaptive ROI Refinement
##########################################################

raw_box = detection.best_box

adaptive = AdaptiveROI(
    initial_margin=0.20,
    margin_step=0.10,
    max_margin=0.40,
    border_width=5,
    pixel_threshold=30,
    max_iterations=3,
)

margin = adaptive.initial_margin

for i in range(adaptive.max_iterations):

    roi, roi_bbox = crop_roi_from_box(
        preprocessed,
        raw_box,
        margin=margin,
    )

    mask_probability = self._segment_roi(
        roi,
        preprocessed.shape[:2],
        roi_bbox,
    )

    mask = (
        mask_probability
        >= self.config.get("segmentation", {}).get("threshold", 0.5)
    ).astype(np.uint8)

    border = adaptive.touches_border(mask)

    # Stop if the tumor is well inside the ROI
    if not any(border.values()):
        print(f"Adaptive ROI finished after {i+1} iteration(s).")
        break

    print(f"Border touched -> increasing ROI margin to {margin + adaptive.margin_step:.2f}")

    margin += adaptive.margin_step

    if margin > adaptive.max_margin:
        print("Maximum ROI margin reached.")
        break

##########################################################
# Classification
##########################################################

label, confidence, probabilities, classification_tensor = self._classify_roi(roi)

class_index = int(np.argmax(probabilities))

        gradcam_roi = self._gradcam(roi, classification_tensor, class_index)
        full_gradcam = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = roi_bbox
        full_gradcam[y1:y2, x1:x2] = gradcam_roi

        localization_image = draw_yolo_box(preprocessed, raw_box, detection.best_score)
        localization_image = draw_label(localization_image, label, confidence)
        mask_overlay = overlay_mask(preprocessed, mask)

        spacing_x = pixel_spacing[0] if pixel_spacing is not None else None
        spacing_y = pixel_spacing[1] if pixel_spacing is not None else None
        size = estimate_tumor_size(mask, spacing_x_mm=spacing_x, spacing_y_mm=spacing_y)

        paths = {
            "localization_image": str(output_dir / f"{stem}_localization.png"),
            "segmentation_mask": str(output_dir / f"{stem}_mask.png"),
            "segmentation_overlay": str(output_dir / f"{stem}_mask_overlay.png"),
            "gradcam": str(output_dir / f"{stem}_gradcam.png"),
            "panel": str(output_dir / f"{stem}_panel.png"),
            "summary": str(output_dir / f"{stem}_summary.json"),
        }
        cv2.imwrite(paths["localization_image"], localization_image)
        cv2.imwrite(paths["segmentation_mask"], (mask * 255).astype(np.uint8))
        cv2.imwrite(paths["segmentation_overlay"], mask_overlay)
        cv2.imwrite(paths["gradcam"], full_gradcam)
        save_inference_panel(
            paths["panel"],
            localization_image,
            mask_overlay,
            full_gradcam,
            title=f"{label} ({confidence:.3f})",
        )

        result = {
            "image_path": str(image_path),
            "bbox_xyxy": list(raw_box) if raw_box is not None else None,
            "bbox_confidence": detection.best_score,
            "classification": label,
            "confidence": confidence,
            "class_probabilities": {name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)},
            "tumor_size": size,
            "outputs": paths,
        }
        save_json(result, paths["summary"])
        return result
