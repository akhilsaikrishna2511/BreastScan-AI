"""Build a normalized radiomics plus deep-feature table from metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from bus_pipeline.config import load_config
from bus_pipeline.datasets import load_metadata
from bus_pipeline.feature_extraction import EfficientNetB3FeatureExtractor
from bus_pipeline.fusion import build_feature_table
from bus_pipeline.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/features/fused_features.csv")
    parser.add_argument("--scaler", default="outputs/features/fusion_scaler.joblib")
    parser.add_argument("--no-deep", action="store_true", help="Skip EfficientNet embeddings and use radiomics only.")
    parser.add_argument("--checkpoint", default=None, help="Optional classifier checkpoint for deep embeddings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config.get("paths", {})
    data_cfg = config.get("data", {})
    records = load_metadata(paths.get("metadata_csv", "data/metadata.csv"), root_dir=data_cfg.get("root_dir"))
    extractor = None
    if not args.no_deep:
        checkpoint = args.checkpoint or paths.get("classifier_checkpoint")
        if checkpoint is not None and not Path(checkpoint).exists():
            checkpoint = None
        extractor = EfficientNetB3FeatureExtractor(
            checkpoint_path=checkpoint,
            image_size=config.get("classification", {}).get("image_size", 300),
            device=get_device(config.get("device", "cuda")),
            pretrained=config.get("classification", {}).get("model", {}).get("pretrained", True),
        )
    frame = build_feature_table(
        records,
        output_csv=args.output,
        preprocess_config=config.get("preprocessing", {}),
        deep_extractor=extractor,
        normalize=True,
        scaler_path=args.scaler,
    )
    print(f"Wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
