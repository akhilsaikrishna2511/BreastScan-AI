"""YOLOv11 localization wrappers and ROI helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    """Detection output in xyxy pixel coordinates."""

    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

    @property
    def best_box(self) -> tuple[int, int, int, int] | None:
        if self.boxes_xyxy.size == 0:
            return None
        index = int(np.argmax(self.scores))
        return tuple(self.boxes_xyxy[index].round().astype(int).tolist())  # type: ignore[return-value]

    @property
    def best_score(self) -> float:
        if self.scores.size == 0:
            return 0.0
        return float(self.scores.max())


class YOLOLocalizer:
    """Thin wrapper around Ultralytics YOLO for tumor localization."""

    def __init__(self, model_path: str | Path = "yolo11n.pt") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install ultralytics to use YOLO localization: pip install ultralytics") from exc
        self.model = YOLO(str(model_path))

    def train(
        self,
        data_yaml: str | Path,
        epochs: int,
        image_size: int,
        batch_size: int,
        project: str | Path,
        name: str = "yolo11_breast_tumor",
        device: str | int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Train YOLO on bounding-box annotations."""

        return self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=str(project),
            name=name,
            device=device,
            **kwargs,
        )

    def validate(self, data_yaml: str | Path, image_size: int, batch_size: int, **kwargs: Any) -> Any:
        """Validate YOLO on the configured validation split."""

        return self.model.val(data=str(data_yaml), imgsz=image_size, batch=batch_size, **kwargs)

    def predict(self, image: str | Path | np.ndarray, conf: float = 0.25, iou: float = 0.7) -> DetectionResult:
        """Run detection and return pixel-coordinate boxes."""

        results = self.model.predict(source=image, conf=conf, iou=iou, verbose=False)
        if not results:
            return DetectionResult(np.zeros((0, 4), dtype=np.float32), np.array([]), np.array([]))
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return DetectionResult(np.zeros((0, 4), dtype=np.float32), np.array([]), np.array([]))
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        return DetectionResult(boxes_xyxy=boxes, scores=scores, class_ids=class_ids)


def crop_roi_from_box(
    image: np.ndarray,
    box_xyxy: tuple[int, int, int, int] | None,
    margin: float = 0.12,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop an image to a detected lesion, falling back to the full image."""

    height, width = image.shape[:2]
    if box_xyxy is None:
        return image, (0, 0, width, height)
    x1, y1, x2, y2 = box_xyxy
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    pad_x = int(round(box_w * margin))
    pad_y = int(round(box_h * margin))
    x1 = max(x1 - pad_x, 0)
    y1 = max(y1 - pad_y, 0)
    x2 = min(x2 + pad_x, width)
    y2 = min(y2 + pad_y, height)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def bbox_xyxy_to_yolo(
    box_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert xyxy box coordinates to normalized YOLO xywh format."""

    x1, y1, x2, y2 = box_xyxy
    center_x = ((x1 + x2) / 2.0) / image_width
    center_y = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return center_x, center_y, width, height


def draw_yolo_box(image: np.ndarray, box: tuple[int, int, int, int] | None, score: float | None = None) -> np.ndarray:
    """Draw a localization box on an image."""

    output = image.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 255), 2)
        if score is not None:
            cv2.putText(
                output,
                f"tumor {score:.2f}",
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
    return output
