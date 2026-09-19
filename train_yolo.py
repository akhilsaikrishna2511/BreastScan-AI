"""Train YOLOv11 tumor localization."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bus_pipeline.config import load_config
from bus_pipeline.localization import YOLOLocalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="YOLO model checkpoint, for example yolo11n.pt.")
    parser.add_argument("--data", default=None, help="Ultralytics dataset YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    yolo_cfg = config.get("localization", {})
    paths = config.get("paths", {})
    model_name = args.model or yolo_cfg.get("model", "yolo11n.pt")
    data_yaml = args.data or paths.get("yolo_data_yaml", "data/yolo/dataset.yaml")
    project = yolo_cfg.get("project", "outputs/yolo")
    device = "0" if config.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu"

    localizer = YOLOLocalizer(model_name)
    localizer.train(
        data_yaml=data_yaml,
        epochs=yolo_cfg.get("epochs", 100),
        image_size=yolo_cfg.get("image_size", 640),
        batch_size=yolo_cfg.get("batch_size", 8),
        project=project,
        name=yolo_cfg.get("run_name", "yolo11_breast_tumor"),
        device=device,
        patience=yolo_cfg.get("patience", 20),
    )
    print(f"YOLO run saved under {Path(project).resolve()}")


if __name__ == "__main__":
    main()
