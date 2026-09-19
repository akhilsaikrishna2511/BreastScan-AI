"""Ultrasound-specific preprocessing and augmentation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def read_grayscale_image(path: str | Path) -> np.ndarray:
    """Read an image as single-channel grayscale."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def normalize_grayscale(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    out_dtype: type[np.uint8] | type[np.float32] = np.uint8,
) -> np.ndarray:
    """Percentile clip and min-max normalize a grayscale ultrasound image."""

    array = image.astype(np.float32)
    low, high = np.percentile(array, [lower_percentile, upper_percentile])
    if high <= low:
        low, high = float(array.min()), float(array.max())
    array = np.clip(array, low, high)
    denom = max(float(array.max() - array.min()), 1e-6)
    array = (array - array.min()) / denom
    if out_dtype == np.uint8:
        return (array * 255.0).round().astype(np.uint8)
    return array.astype(np.float32)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply contrast-limited adaptive histogram equalization."""

    image_u8 = normalize_grayscale(image, out_dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(image_u8)


def reduce_speckle_noise(
    image: np.ndarray,
    method: str = "median",
    kernel_size: int = 5,
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 50.0,
    bilateral_sigma_space: float = 50.0,
) -> np.ndarray:
    """Reduce ultrasound speckle noise using median or bilateral filtering."""

    if method == "median":
        ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.medianBlur(image, ksize)
    if method == "bilateral":
        return cv2.bilateralFilter(
            image,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )
    if method in {"none", None}:
        return image
    raise ValueError(f"Unsupported speckle reduction method: {method}")


def preprocess_ultrasound_image(image: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    """Run grayscale normalization, CLAHE, and speckle reduction."""

    cfg = config or {}
    normalized = normalize_grayscale(
        image,
        lower_percentile=cfg.get("lower_percentile", 1.0),
        upper_percentile=cfg.get("upper_percentile", 99.0),
        out_dtype=np.uint8,
    )
    if cfg.get("use_clahe", True):
        normalized = apply_clahe(
            normalized,
            clip_limit=cfg.get("clahe_clip_limit", 2.0),
            tile_grid_size=tuple(cfg.get("clahe_tile_grid_size", [8, 8])),
        )
    return reduce_speckle_noise(
        normalized,
        method=cfg.get("speckle_method", "median"),
        kernel_size=cfg.get("median_kernel_size", 5),
        bilateral_d=cfg.get("bilateral_d", 7),
        bilateral_sigma_color=cfg.get("bilateral_sigma_color", 50.0),
        bilateral_sigma_space=cfg.get("bilateral_sigma_space", 50.0),
    )


def to_three_channel(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image to RGB-like three-channel format."""

    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 1:
        return np.repeat(image, 3, axis=-1)
    return image


def build_segmentation_transforms(image_size: int, augment: bool = False) -> A.Compose:
    """Build transforms for segmentation images and masks."""

    transforms: list[Any] = [A.Resize(image_size, image_size)]
    if augment:
        transforms.extend(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.12,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.75,
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            ]
        )
    transforms.extend([A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])
    return A.Compose(transforms)


def build_classification_transforms(image_size: int, augment: bool = False) -> A.Compose:
    """Build transforms for EfficientNet-B3 classification."""

    transforms: list[Any] = [A.Resize(image_size, image_size)]
    if augment:
        transforms.extend(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.12,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.75,
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            ]
        )
    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)
