"""Export metadata CSV annotations into an Ultralytics YOLO dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import yaml

from bus_pipeline.datasets import bbox_from_mask, load_metadata
from bus_pipeline.localization import bbox_xyxy_to_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="CSV with image_path, label, split, and bbox or mask columns.")
    parser.add_argument("--output-dir", required=True, help="Destination YOLO dataset directory.")
    parser.add_argument("--root-dir", default=None, help="Base directory for relative paths in metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_metadata(args.metadata, root_dir=args.root_dir)
    output_dir = Path(args.output_dir)
    splits = sorted({record.split or "train" for record in records})

    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for record in records:
        split = record.split or "train"
        image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {record.image_path}")
        height, width = image.shape[:2]
        image_target = output_dir / "images" / split / record.image_path.name
        label_target = output_dir / "labels" / split / f"{record.image_path.stem}.txt"
        shutil.copy2(record.image_path, image_target)

        bbox = record.bbox
        if bbox is None and record.mask_path is not None:
            mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {record.mask_path}")
            bbox = bbox_from_mask(mask)

        lines: list[str] = []
        if bbox is not None and record.label != "normal":
            x_center, y_center, box_width, box_height = bbox_xyxy_to_yolo(bbox, width, height)
            lines.append(f"0 {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}")
        label_target.write_text("\n".join(lines), encoding="utf-8")

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val" if "val" in splits else "images/train",
        "test": "images/test" if "test" in splits else None,
        "names": {0: "tumor"},
    }
    with (output_dir / "dataset.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump({key: value for key, value in dataset_yaml.items() if value is not None}, file, sort_keys=False)
    print(f"Wrote YOLO dataset to {output_dir}")


if __name__ == "__main__":
    main()
