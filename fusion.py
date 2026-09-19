"""Feature-fusion utilities for radiomics and deep embeddings."""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from bus_pipeline.datasets import BreastUltrasoundRecord, load_metadata
from bus_pipeline.feature_extraction import EfficientNetB3FeatureExtractor, extract_all_features
from bus_pipeline.preprocessing import preprocess_ultrasound_image, read_grayscale_image


class FeatureFusionTransformer:
    """Normalize handcrafted and deep features before concatenation."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    def fit_transform(self, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        self.feature_columns = feature_columns
        transformed = frame.copy()
        transformed[feature_columns] = self.scaler.fit_transform(frame[feature_columns].fillna(0.0))
        return transformed

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        transformed[self.feature_columns] = self.scaler.transform(frame[self.feature_columns].fillna(0.0))
        return transformed

    def save(self, path: str | Path) -> None:
        joblib.dump({"scaler": self.scaler, "feature_columns": self.feature_columns}, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureFusionTransformer":
        payload = joblib.load(path)
        transformer = cls()
        transformer.scaler = payload["scaler"]
        transformer.feature_columns = payload["feature_columns"]
        return transformer


def build_feature_table(
    records: list[BreastUltrasoundRecord],
    output_csv: str | Path,
    preprocess_config: dict | None = None,
    deep_extractor: EfficientNetB3FeatureExtractor | None = None,
    normalize: bool = True,
    scaler_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create a CSV containing fused radiomics and deep features."""

    rows: list[dict] = []
    for record in tqdm(records, desc="extracting features"):
        image = read_grayscale_image(record.image_path)
        image = preprocess_ultrasound_image(image, preprocess_config or {})
        if record.mask_path is not None:
            mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {record.mask_path}")
            mask = (mask > 0).astype("uint8")
        else:
            mask = (image * 0).astype("uint8")
        pixel_spacing = None
        if record.spacing_x_mm is not None and record.spacing_y_mm is not None:
            pixel_spacing = (record.spacing_x_mm, record.spacing_y_mm)
        features = extract_all_features(image, mask, deep_extractor=deep_extractor, pixel_spacing=pixel_spacing)
        rows.append(
            {
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path) if record.mask_path else "",
                "label": record.label,
                "split": record.split or "",
                "patient_id": record.patient_id or "",
                **features,
            }
        )

    frame = pd.DataFrame(rows)
    metadata_cols = {"image_path", "mask_path", "label", "split", "patient_id"}
    feature_columns = [col for col in frame.columns if col not in metadata_cols]
    if normalize:
        transformer = FeatureFusionTransformer()
        frame = transformer.fit_transform(frame, feature_columns)
        if scaler_path is not None:
            transformer.save(scaler_path)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def build_feature_table_from_metadata(
    metadata_csv: str | Path,
    output_csv: str | Path,
    root_dir: str | Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load metadata and create a fused feature table."""

    records = load_metadata(metadata_csv, root_dir=root_dir)
    return build_feature_table(records, output_csv, **kwargs)
