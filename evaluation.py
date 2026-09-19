"""Reusable validation loops for segmentation and classification."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from bus_pipeline.constants import CLASS_NAMES
from bus_pipeline.metrics import classification_metrics, segmentation_metrics


@torch.no_grad()
def evaluate_segmentation_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate segmentation model and average per-image metrics."""

    model.eval()
    metric_values: dict[str, list[float]] = defaultdict(list)
    for batch in tqdm(dataloader, desc="validating segmentation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        probabilities = torch.sigmoid(model(images)).detach().cpu().numpy()
        targets = masks.detach().cpu().numpy()
        for target, probability in zip(targets, probabilities):
            values = segmentation_metrics(target.squeeze(), probability.squeeze(), threshold=threshold)
            for key, value in values.items():
                metric_values[key].append(value)

    summary: dict[str, float] = {}
    for key, values in metric_values.items():
        finite_values = [value for value in values if np.isfinite(value)]
        summary[key] = float(np.mean(finite_values)) if finite_values else float("inf")
    return summary


@torch.no_grad()
def evaluate_classifier_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate an image classifier."""

    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []

    for batch in tqdm(dataloader, desc="validating classifier", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_predictions.extend(predictions.detach().cpu().numpy().tolist())
        all_probabilities.extend(probabilities.detach().cpu().numpy())

    return classification_metrics(
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
        class_names=class_names or CLASS_NAMES,
    )
