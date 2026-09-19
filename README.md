# Breast Ultrasound Tumor Analysis Pipeline

This project is an end-to-end Python/PyTorch pipeline for breast ultrasound tumor analysis:

- Preprocess ultrasound images with grayscale normalization, CLAHE, and speckle reduction.
- Localize lesions with Ultralytics YOLOv11.
- Crop the detected ROI and segment it with Attention U-Net++.
- Extract morphological, GLCM texture, PyRadiomics, and EfficientNet-B3 deep features.
- Classify images as `normal`, `benign`, or `malignant` with EfficientNet-B3 + CBAM.
- Estimate tumor area, perimeter, maximum diameter, and minimum diameter in pixels or mm.
- Generate evaluation metrics, overlays, and Grad-CAM explanations.

This is research code, not a diagnostic medical device. Validate rigorously on your target scanner, population, and labeling protocol before any clinical use.

## Project Layout

```text
bus_pipeline/
  preprocessing.py          # normalization, CLAHE, speckle filtering, augmentations
  datasets.py               # metadata-driven segmentation and classification datasets
  localization.py           # YOLOv11 training/prediction wrapper
  segmentation_models.py    # Attention U-Net++
  classification_models.py  # EfficientNet-B3 + CBAM and optional fusion classifier
  feature_extraction.py     # morphology, GLCM, PyRadiomics, deep embeddings
  fusion.py                 # feature normalization and feature-table generation
  metrics.py                # Dice, IoU, precision, recall, Hausdorff, ROC-AUC, F1
  inference.py              # complete single-image inference pipeline
scripts/
  export_yolo_dataset.py
  train_yolo.py
  train_segmentation.py
  validate_segmentation.py
  train_classifier.py
  validate_classifier.py
  build_feature_table.py
  infer_single.py
configs/default.yaml
examples/metadata_example.csv
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch wheel that matches your GPU from the official PyTorch selector when needed. `ultralytics` will use model names such as `yolo11n.pt`; if you are offline, place the YOLO weights locally and point `configs/default.yaml` to that file.

## Dataset Contract

Create `data/metadata.csv` with these columns:

```csv
image_path,mask_path,label,split,bbox_xmin,bbox_ymin,bbox_xmax,bbox_ymax,spacing_x_mm,spacing_y_mm,patient_id
```

Required columns are `image_path` and `label`. Labels must be `normal`, `benign`, or `malignant`. `split` should be `train`, `val`, or `test`. Masks are optional for normal images. Bounding boxes can be supplied directly; when missing, the YOLO export script can derive boxes from masks.

Relative paths are resolved from `data.root_dir` in `configs/default.yaml`.

## Training Workflow

1. Export YOLO annotations:

```bash
python scripts/export_yolo_dataset.py --metadata data/metadata.csv --root-dir data --output-dir data/yolo
```

2. Train YOLOv11 localization:

```bash
python scripts/train_yolo.py --config configs/default.yaml
```

3. Train Attention U-Net++ segmentation:

```bash
python scripts/train_segmentation.py --config configs/default.yaml
```

4. Train EfficientNet-B3 + CBAM classification:

```bash
python scripts/train_classifier.py --config configs/default.yaml
```

5. Optionally build fused radiomics plus deep feature table:

```bash
python scripts/build_feature_table.py --config configs/default.yaml --output outputs/features/fused_features.csv
```

TensorBoard logs are written to `outputs/tensorboard`:

```bash
tensorboard --logdir outputs/tensorboard
```

## Validation

```bash
python scripts/validate_segmentation.py --config configs/default.yaml --split val
python scripts/validate_classifier.py --config configs/default.yaml --split val
```

Segmentation metrics include Dice, IoU, precision, recall, and Hausdorff distance. Classification metrics include accuracy, macro precision, macro recall, macro F1, ROC-AUC when valid, and a confusion matrix.

## Single-Image Inference

```bash
python scripts/infer_single.py ^
  --config configs/default.yaml ^
  --image data/images/test/example.png ^
  --output-dir outputs/inference ^
  --spacing-x-mm 0.08 ^
  --spacing-y-mm 0.08
```

For each image the pipeline writes:

- `{image}_localization.png`
- `{image}_mask.png`
- `{image}_mask_overlay.png`
- `{image}_gradcam.png`
- `{image}_panel.png`
- `{image}_summary.json`

The JSON contains bounding-box coordinates, class probabilities, confidence score, tumor dimensions, and paths to generated visualizations.

## Notes

- Normal images are supported with empty masks and empty YOLO label files.
- The segmentation model trains on lesion ROIs when bounding boxes or masks are available.
- Pixel spacing is optional. Without it, size estimates are reported in pixels only.
- PyRadiomics extraction is available through `bus_pipeline.feature_extraction.extract_pyradiomics_features`.
- Keep test patients separate from train/validation patients to avoid leakage.

