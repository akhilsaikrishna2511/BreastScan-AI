"""Morphological, texture, PyRadiomics, and deep feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from skimage.feature import graycomatrix, graycoprops

from bus_pipeline.classification_models import EfficientNetB3CBAM
from bus_pipeline.preprocessing import build_classification_transforms, to_three_channel
from bus_pipeline.size_estimation import estimate_tumor_size


def extract_morphological_features(mask: np.ndarray) -> dict[str, float]:
    """Extract contour-based morphological features from a binary mask."""

    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "circularity": 0.0,
            "aspect_ratio": 0.0,
            "solidity": 0.0,
            "major_axis_length": 0.0,
            "minor_axis_length": 0.0,
        }

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    circularity = float((4.0 * np.pi * area) / (perimeter**2 + 1e-7))
    x, y, width, height = cv2.boundingRect(contour)
    aspect_ratio = float(width / max(height, 1))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / (hull_area + 1e-7))

    major_axis = minor_axis = 0.0
    if len(contour) >= 5:
        (_, _), axes, _ = cv2.fitEllipse(contour)
        major_axis = float(max(axes))
        minor_axis = float(min(axes))
    else:
        rect = cv2.minAreaRect(contour)
        major_axis = float(max(rect[1]))
        minor_axis = float(min(rect[1]))

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "major_axis_length": major_axis,
        "minor_axis_length": minor_axis,
    }


def extract_texture_features(image: np.ndarray, mask: np.ndarray | None = None, levels: int = 32) -> dict[str, float]:
    """Extract GLCM texture features from the lesion ROI."""

    gray = image.astype(np.float32)
    if mask is not None and (mask > 0).any():
        ys, xs = np.where(mask > 0)
        gray = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    gray = cv2.normalize(gray, None, alpha=0, beta=levels - 1, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(
        gray,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    probabilities = glcm / (glcm.sum(axis=(0, 1), keepdims=True) + 1e-12)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12), axis=(0, 1))
    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_entropy": float(entropy.mean()),
    }


def extract_pyradiomics_features(
    image: np.ndarray,
    mask: np.ndarray,
    pixel_spacing: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Extract PyRadiomics features when PyRadiomics and SimpleITK are installed."""

    if not (mask > 0).any():
        return {}
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError("Install pyradiomics and SimpleITK to use PyRadiomics features.") from exc

    image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray((mask > 0).astype(np.uint8))
    if pixel_spacing is not None:
        spacing_x, spacing_y = pixel_spacing
        image_sitk.SetSpacing((float(spacing_x), float(spacing_y)))
        mask_sitk.SetSpacing((float(spacing_x), float(spacing_y)))
    extractor = featureextractor.RadiomicsFeatureExtractor()
    values = extractor.execute(image_sitk, mask_sitk)
    return {
        str(key): float(value)
        for key, value in values.items()
        if key.startswith("original_") and np.isscalar(value)
    }


class EfficientNetB3FeatureExtractor:
    """Deep feature extractor that returns EfficientNet-B3 CBAM embeddings."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        image_size: int = 300,
        device: str | torch.device = "cuda",
        pretrained: bool = True,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.model = EfficientNetB3CBAM(num_classes=3, pretrained=pretrained).to(self.device)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=False)
        self.model.eval()
        self.transforms = build_classification_transforms(image_size=image_size, augment=False)

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        rgb = to_three_channel(image)
        tensor = self.transforms(image=rgb)["image"].unsqueeze(0).to(self.device).float()
        _, embedding = self.model(tensor, return_features=True)
        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)


def extract_all_features(
    image: np.ndarray,
    mask: np.ndarray,
    deep_extractor: EfficientNetB3FeatureExtractor | None = None,
    pixel_spacing: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Extract morphological, texture, tumor-size, and optional deep features."""

    features: dict[str, float] = {}
    features.update(extract_morphological_features(mask))
    features.update(extract_texture_features(image, mask))
    spacing_x = pixel_spacing[0] if pixel_spacing is not None else None
    spacing_y = pixel_spacing[1] if pixel_spacing is not None else None
    size = estimate_tumor_size(mask, spacing_x_mm=spacing_x, spacing_y_mm=spacing_y)
    features.update({f"size_{key}": 0.0 if value is None else float(value) for key, value in size.items()})
    if deep_extractor is not None:
        deep = deep_extractor.extract(image)
        features.update({f"deep_{index:04d}": float(value) for index, value in enumerate(deep)})
    return features
