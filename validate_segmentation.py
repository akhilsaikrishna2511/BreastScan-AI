"""Validate a trained Attention U-Net++ segmentation checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bus_pipeline.config import load_config
from bus_pipeline.datasets import SegmentationDataset, filter_records, load_metadata
from bus_pipeline.evaluation import evaluate_segmentation_model
from bus_pipeline.segmentation_models import build_segmentation_model
from bus_pipeline.utils import get_device, load_model_state, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="outputs/segmentation_validation_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config.get("device", "cuda"))
    paths = config.get("paths", {})
    seg_cfg = config.get("segmentation", {})
    data_cfg = config.get("data", {})

    records = filter_records(load_metadata(paths.get("metadata_csv", "data/metadata.csv"), root_dir=data_cfg.get("root_dir")), args.split)
    dataset = SegmentationDataset(
        records,
        image_size=seg_cfg.get("image_size", 256),
        preprocess_config=config.get("preprocessing", {}),
        augment=False,
        crop_rois=seg_cfg.get("crop_rois", True),
    )
    loader = DataLoader(dataset, batch_size=seg_cfg.get("training", {}).get("batch_size", 8), shuffle=False)
    model = build_segmentation_model(seg_cfg.get("model", {})).to(device)
    checkpoint = args.checkpoint or Path(paths.get("checkpoint_dir", "outputs/checkpoints")) / "segmentation_best.pt"
    load_model_state(model, checkpoint, device)
    metrics = evaluate_segmentation_model(model, loader, device, threshold=seg_cfg.get("threshold", 0.5))
    save_json(metrics, args.output)
    print(metrics)


if __name__ == "__main__":
    main()
