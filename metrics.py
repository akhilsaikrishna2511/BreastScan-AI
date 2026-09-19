"""Evaluation metrics for segmentation and classification."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_bool(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.asarray(mask) > threshold


def dice_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """Compute the Dice coefficient."""

    true = _as_bool(y_true, threshold)
    pred = _as_bool(y_pred, threshold)
    intersection = np.logical_and(true, pred).sum()
    return float((2.0 * intersection + eps) / (true.sum() + pred.sum() + eps))


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """Compute intersection over union."""

    true = _as_bool(y_true, threshold)
    pred = _as_bool(y_pred, threshold)
    intersection = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()
    return float((intersection + eps) / (union + eps))


def precision_score_binary(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, eps: float = 1e-7) -> float:
    true = _as_bool(y_true, threshold)
    pred = _as_bool(y_pred, threshold)
    tp = np.logical_and(true, pred).sum()
    fp = np.logical_and(~true, pred).sum()
    return float((tp + eps) / (tp + fp + eps))


def recall_score_binary(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, eps: float = 1e-7) -> float:
    true = _as_bool(y_true, threshold)
    pred = _as_bool(y_pred, threshold)
    tp = np.logical_and(true, pred).sum()
    fn = np.logical_and(true, ~pred).sum()
    return float((tp + eps) / (tp + fn + eps))


def hausdorff_distance(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute symmetric Hausdorff distance between binary mask boundaries."""

    true = _as_bool(y_true, threshold)
    pred = _as_bool(y_pred, threshold)
    if not true.any() and not pred.any():
        return 0.0
    if not true.any() or not pred.any():
        return float("inf")

    true_boundary = _mask_boundary(true)
    pred_boundary = _mask_boundary(pred)
    true_points = np.column_stack(np.nonzero(true_boundary)).astype(np.float32)
    pred_points = np.column_stack(np.nonzero(pred_boundary)).astype(np.float32)
    hd_true_to_pred = _directed_hausdorff(true_points, pred_points)
    hd_pred_to_true = _directed_hausdorff(pred_points, true_points)
    return float(max(hd_true_to_pred, hd_pred_to_true))


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return a binary boundary map without requiring SciPy or OpenCV."""

    mask_bool = mask.astype(bool)
    padded = np.pad(mask_bool, 1, mode="constant", constant_values=False)
    eroded = mask_bool.copy()
    for y_offset in range(3):
        for x_offset in range(3):
            eroded &= padded[y_offset : y_offset + mask.shape[0], x_offset : x_offset + mask.shape[1]]
    return np.logical_and(mask_bool, ~eroded)


def _directed_hausdorff(source: np.ndarray, target: np.ndarray, chunk_size: int = 2048) -> float:
    """Compute directed Hausdorff distance with bounded temporary memory."""

    if source.size == 0 or target.size == 0:
        return float("inf")
    max_min_distance = 0.0
    for start in range(0, len(source), chunk_size):
        chunk = source[start : start + chunk_size]
        distances = np.sqrt(((chunk[:, None, :] - target[None, :, :]) ** 2).sum(axis=2))
        max_min_distance = max(max_min_distance, float(distances.min(axis=1).max()))
    return max_min_distance


def segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute the requested segmentation metrics."""

    return {
        "dice": dice_score(y_true, y_pred, threshold),
        "iou": iou_score(y_true, y_pred, threshold),
        "precision": precision_score_binary(y_true, y_pred, threshold),
        "recall": recall_score_binary(y_true, y_pred, threshold),
        "hausdorff": hausdorff_distance(y_true, y_pred, threshold),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute multiclass classification metrics."""

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    labels = list(range(len(class_names))) if class_names else sorted(np.unique(y_true).tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    output: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    names = class_names or [str(label) for label in labels]
    for index, name in enumerate(names):
        output["per_class"][name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    if y_prob is not None:
        try:
            output["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
            )
        except ValueError:
            output["roc_auc_ovr_macro"] = None
    return output
