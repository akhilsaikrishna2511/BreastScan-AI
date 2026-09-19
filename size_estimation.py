"""Tumor size estimation from binary segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _scaled_contour(contour: np.ndarray, spacing_x_mm: float, spacing_y_mm: float) -> np.ndarray:
    points = contour.astype(np.float32).copy()
    points[:, 0, 0] *= spacing_x_mm
    points[:, 0, 1] *= spacing_y_mm
    return points


def estimate_tumor_size(
    mask: np.ndarray,
    spacing_x_mm: float | None = None,
    spacing_y_mm: float | None = None,
) -> dict[str, float | None]:
    """Estimate area, perimeter, and diameters in pixels and optionally millimeters."""

    binary = (mask > 0).astype(np.uint8)
    contour = _largest_contour(binary)
    area_px = float(binary.sum())
    if contour is None or len(contour) < 2:
        return {
            "area_px": area_px,
            "perimeter_px": 0.0,
            "maximum_diameter_px": 0.0,
            "minimum_diameter_px": 0.0,
            "area_mm2": None,
            "perimeter_mm": None,
            "maximum_diameter_mm": None,
            "minimum_diameter_mm": None,
        }

    perimeter_px = float(cv2.arcLength(contour, closed=True))
    rect = cv2.minAreaRect(contour)
    width_px, height_px = rect[1]
    max_diameter_px = float(max(width_px, height_px))
    min_diameter_px = float(min(width_px, height_px))

    area_mm2 = perimeter_mm = max_diameter_mm = min_diameter_mm = None
    if spacing_x_mm is not None and spacing_y_mm is not None:
        scaled = _scaled_contour(contour, spacing_x_mm, spacing_y_mm)
        area_mm2 = float(binary.sum() * spacing_x_mm * spacing_y_mm)
        perimeter_mm = float(cv2.arcLength(scaled, closed=True))
        rect_mm = cv2.minAreaRect(scaled)
        width_mm, height_mm = rect_mm[1]
        max_diameter_mm = float(max(width_mm, height_mm))
        min_diameter_mm = float(min(width_mm, height_mm))

    return {
        "area_px": area_px,
        "perimeter_px": perimeter_px,
        "maximum_diameter_px": max_diameter_px,
        "minimum_diameter_px": min_diameter_px,
        "area_mm2": area_mm2,
        "perimeter_mm": perimeter_mm,
        "maximum_diameter_mm": max_diameter_mm,
        "minimum_diameter_mm": min_diameter_mm,
    }
