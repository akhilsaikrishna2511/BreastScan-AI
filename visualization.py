"""Visualization helpers for outputs and evaluation reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35, color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """Overlay a binary segmentation mask on an image."""

    output = image.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    colored = np.zeros_like(output)
    colored[mask > 0] = color
    return cv2.addWeighted(output, 1.0, colored, alpha, 0)


def draw_label(
    image: np.ndarray,
    label: str,
    confidence: float,
    origin: tuple[int, int] = (12, 24),
) -> np.ndarray:
    """Draw classification label and confidence."""

    output = image.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        output,
        f"{label}: {confidence:.3f}",
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"{label}: {confidence:.3f}",
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return output


def save_inference_panel(
    output_path: str | Path,
    localization: np.ndarray,
    mask_overlay: np.ndarray,
    gradcam: np.ndarray,
    title: str,
) -> None:
    """Save a compact three-panel inference summary."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for axis, image, name in zip(
        axes,
        [localization, mask_overlay, gradcam],
        ["Localization", "Segmentation", "Grad-CAM"],
    ):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(name)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Save a confusion matrix heatmap."""

    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Save one-vs-rest ROC curves for multiclass classification."""

    fig, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(class_names):
        RocCurveDisplay.from_predictions((y_true == index).astype(int), y_prob[:, index], name=name, ax=axis)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
