"""Run the complete pipeline on a single ultrasound image."""

from __future__ import annotations

import argparse
from pathlib import Path

from bus_pipeline.config import load_config
from bus_pipeline.inference import BreastTumorAnalysisPipeline, InferencePaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--segmentation-checkpoint", default=None)
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--spacing-x-mm", type=float, default=None)
    parser.add_argument("--spacing-y-mm", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    artifact_paths = InferencePaths(
        yolo_weights=Path(args.yolo_weights or paths_cfg.get("yolo_weights", "outputs/yolo/best.pt")),
        segmentation_checkpoint=Path(
            args.segmentation_checkpoint or paths_cfg.get("segmentation_checkpoint", "outputs/checkpoints/segmentation_best.pt")
        ),
        classifier_checkpoint=Path(
            args.classifier_checkpoint or paths_cfg.get("classifier_checkpoint", "outputs/checkpoints/classifier_best.pt")
        ),
    )
    spacing = None
    if args.spacing_x_mm is not None and args.spacing_y_mm is not None:
        spacing = (args.spacing_x_mm, args.spacing_y_mm)
    pipeline = BreastTumorAnalysisPipeline(config, artifact_paths=artifact_paths)
    result = pipeline.predict(args.image, output_dir=args.output_dir, pixel_spacing=spacing)
    print(result)


if __name__ == "__main__":
    main()
