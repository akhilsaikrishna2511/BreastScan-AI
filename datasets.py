"""Dataset loaders for localization, segmentation, classification, and fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from bus_pipeline.constants import CLASS_TO_INDEX
from bus_pipeline.preprocessing import (
    build_classification_transforms,
    build_segmentation_transforms,
    preprocess_ultrasound_image,
    read_grayscale_image,
    to_three_channel,
)


@dataclass(frozen=True)
class BreastUltrasoundRecord:
    """One metadata row describing an image, optional mask, label, and lesion box."""

    image_path: Path
    label: str
    mask_path: Path | None = None
    split: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    spacing_x_mm: float | None = None
    spacing_y_mm: float | None = None
    patient_id: str | None = None


def _resolve_path(value: Any, root: Path) -> Path | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return None
    return float(value)


def _bbox_from_row(row: pd.Series) -> tuple[float, float, float, float] | None:
    bbox_cols = ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
    if all(col in row and not pd.isna(row[col]) for col in bbox_cols):
        return tuple(float(row[col]) for col in bbox_cols)  # type: ignore[return-value]
    return None


def load_metadata(metadata_csv: str | Path, root_dir: str | Path | None = None) -> list[BreastUltrasoundRecord]:
    """Load the metadata CSV used by all stages."""

    metadata_path = Path(metadata_csv)
    root = Path(root_dir) if root_dir is not None else metadata_path.parent
    frame = pd.read_csv(metadata_path)
    required = {"image_path", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")

    records: list[BreastUltrasoundRecord] = []
    for _, row in frame.iterrows():
        label = str(row["label"]).lower().strip()
        if label not in CLASS_TO_INDEX:
            raise ValueError(f"Unsupported label '{label}'. Expected one of {sorted(CLASS_TO_INDEX)}")
        records.append(
            BreastUltrasoundRecord(
                image_path=_resolve_path(row["image_path"], root) or Path(""),
                label=label,
                mask_path=_resolve_path(row.get("mask_path"), root),
                split=str(row.get("split")).lower().strip() if "split" in row and not pd.isna(row["split"]) else None,
                bbox=_bbox_from_row(row),
                spacing_x_mm=_optional_float(row.get("spacing_x_mm")),
                spacing_y_mm=_optional_float(row.get("spacing_y_mm")),
                patient_id=str(row.get("patient_id")) if "patient_id" in row and not pd.isna(row["patient_id"]) else None,
            )
        )
    return records


def filter_records(records: Iterable[BreastUltrasoundRecord], split: str | None) -> list[BreastUltrasoundRecord]:
    """Filter metadata records by split."""

    if split is None:
        return list(records)
    return [record for record in records if record.split == split]


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Derive an xyxy bounding box from a binary mask."""

    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return x, y, x + width, y + height


def crop_to_bbox(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    margin: float = 0.12,
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int]]:
    """Crop an image and optional mask to a lesion ROI."""

    height, width = image.shape[:2]
    if bbox is None and mask is not None:
        bbox = bbox_from_mask(mask)
    if bbox is None:
        return image, mask, (0, 0, width, height)

    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    pad_x = box_w * margin
    pad_y = box_h * margin
    x1_i = max(int(np.floor(x1 - pad_x)), 0)
    y1_i = max(int(np.floor(y1 - pad_y)), 0)
    x2_i = min(int(np.ceil(x2 + pad_x)), width)
    y2_i = min(int(np.ceil(y2 + pad_y)), height)

    cropped_image = image[y1_i:y2_i, x1_i:x2_i]
    cropped_mask = mask[y1_i:y2_i, x1_i:x2_i] if mask is not None else None
    return cropped_image, cropped_mask, (x1_i, y1_i, x2_i, y2_i)


def _load_mask(record: BreastUltrasoundRecord, shape: tuple[int, int]) -> np.ndarray:
    if record.mask_path is None:
        return np.zeros(shape, dtype=np.uint8)
    mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {record.mask_path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


class SegmentationDataset(Dataset):
    """PyTorch dataset returning lesion ROI images and binary masks."""

    def __init__(
        self,
        records: list[BreastUltrasoundRecord],
        image_size: int = 256,
        preprocess_config: dict[str, Any] | None = None,
        augment: bool = False,
        crop_rois: bool = True,
        roi_margin: float = 0.12,
    ) -> None:
        self.records = records
        self.preprocess_config = preprocess_config or {}
        self.crop_rois = crop_rois
        self.roi_margin = roi_margin
        self.transforms = build_segmentation_transforms(image_size=image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = read_grayscale_image(record.image_path)
        image = preprocess_ultrasound_image(image, self.preprocess_config)
        mask = _load_mask(record, image.shape[:2])
        roi_bbox = (0, 0, image.shape[1], image.shape[0])
        if self.crop_rois:
            image, mask, roi_bbox = crop_to_bbox(image, mask, record.bbox, margin=self.roi_margin)

        transformed = self.transforms(image=image, mask=mask)
        image_tensor = transformed["image"].float()
        mask_tensor = transformed["mask"].unsqueeze(0).float()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": CLASS_TO_INDEX[record.label],
            "image_path": str(record.image_path),
            "roi_bbox": torch.tensor(roi_bbox, dtype=torch.float32),
        }


class ClassificationDataset(Dataset):
    """PyTorch dataset returning ROI or full image tensors for lesion classification."""

    def __init__(
        self,
        records: list[BreastUltrasoundRecord],
        image_size: int = 300,
        preprocess_config: dict[str, Any] | None = None,
        augment: bool = False,
        crop_rois: bool = True,
        roi_margin: float = 0.12,
    ) -> None:
        self.records = records
        self.preprocess_config = preprocess_config or {}
        self.crop_rois = crop_rois
        self.roi_margin = roi_margin
        self.transforms = build_classification_transforms(image_size=image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = read_grayscale_image(record.image_path)
        image = preprocess_ultrasound_image(image, self.preprocess_config)
        mask = _load_mask(record, image.shape[:2]) if record.mask_path is not None else None
        roi_bbox = (0, 0, image.shape[1], image.shape[0])
        if self.crop_rois:
            image, _, roi_bbox = crop_to_bbox(image, mask, record.bbox, margin=self.roi_margin)

        image = to_three_channel(image)
        transformed = self.transforms(image=image)
        return {
            "image": transformed["image"].float(),
            "label": torch.tensor(CLASS_TO_INDEX[record.label], dtype=torch.long),
            "image_path": str(record.image_path),
            "roi_bbox": torch.tensor(roi_bbox, dtype=torch.float32),
        }


class FeatureTableDataset(Dataset):
    """Dataset for fused radiomics/deep features stored in a CSV file."""

    def __init__(self, feature_csv: str | Path, label_column: str = "label") -> None:
        frame = pd.read_csv(feature_csv)
        if label_column not in frame.columns:
            raise ValueError(f"Feature table is missing label column: {label_column}")
        self.labels = torch.tensor(
            [CLASS_TO_INDEX[str(label).lower().strip()] for label in frame[label_column]], dtype=torch.long
        )
        drop_columns = {label_column, "image_path", "mask_path", "split", "patient_id"}
        feature_columns = [col for col in frame.columns if col not in drop_columns]
        self.feature_names = feature_columns
        self.features = torch.tensor(frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index], "label": self.labels[index]}
