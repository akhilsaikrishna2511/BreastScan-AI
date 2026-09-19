"""Validate a trained EfficientNet-B3 CBAM classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bus_pipeline.classification_models import build_classifier
from bus_pipeline.config import load_config
from bus_pipeline.constants import CLASS_NAMES
from bus_pipeline.datasets import ClassificationDataset, filter_records, load_metadata
from bus_pipeline.evaluation import evaluate_classifier_model
from bus_pipeline.utils import get_device, load_model_state, save_json
from bus_pipeline.visualization import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="outputs/classification_validation_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config.get("device", "cuda"))
    paths = config.get("paths", {})
    cls_cfg = config.get("classification", {})
    data_cfg = config.get("data", {})

    records = filter_records(load_metadata(paths.get("metadata_csv", "data/metadata.csv"), root_dir=data_cfg.get("root_dir")), args.split)
    dataset = ClassificationDataset(
        records,
        image_size=cls_cfg.get("image_size", 300),
        preprocess_config=config.get("preprocessing", {}),
        augment=False,
        crop_rois=cls_cfg.get("crop_rois", True),
    )
    loader = DataLoader(dataset, batch_size=cls_cfg.get("training", {}).get("batch_size", 8), shuffle=False)
    model = build_classifier(cls_cfg.get("model", {})).to(device)
    checkpoint = args.checkpoint or Path(paths.get("checkpoint_dir", "outputs/checkpoints")) / "classifier_best.pt"
    load_model_state(model, checkpoint, device)
    metrics = evaluate_classifier_model(model, loader, device, class_names=CLASS_NAMES)
    save_json(metrics, args.output)
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, str(Path(args.output).with_suffix(".png")))
    print(metrics)


if __name__ == "__main__":
    main()
