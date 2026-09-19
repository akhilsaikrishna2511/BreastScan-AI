from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# CHANGE THIS TO YOUR DATASET FOLDER
# =====================================================
DATASET_ROOT = Path("data")

# =====================================================
# COLLECT IMAGES
# =====================================================

records = []

for label in ["benign", "malignant", "normal"]:

    class_folder = DATASET_ROOT / label

    if not class_folder.exists():
        print(f"Folder not found: {class_folder}")
        continue

    for image_file in class_folder.glob("*.png"):

        # Skip mask files
        if "_mask" in image_file.stem.lower():
            continue

        # BUSI mask naming convention
        mask_file = image_file.with_name(
            image_file.stem + "_mask.png"
        )

        records.append({
            "image_path": str(image_file).replace("\\", "/"),
            "mask_path": str(mask_file).replace("\\", "/") if mask_file.exists() else "",
            "label": label
        })

# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(records)

if len(df) == 0:
    raise ValueError("No images found. Check DATASET_ROOT path.")

print(f"Total images found: {len(df)}")

# =====================================================
# TRAIN / VAL / TEST SPLIT
# =====================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

metadata = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

# =====================================================
# ADD OPTIONAL COLUMNS REQUIRED BY PIPELINE
# =====================================================

metadata["bbox_xmin"] = ""
metadata["bbox_ymin"] = ""
metadata["bbox_xmax"] = ""
metadata["bbox_ymax"] = ""

metadata["spacing_x_mm"] = ""
metadata["spacing_y_mm"] = ""

metadata["patient_id"] = ""

# =====================================================
# SAVE CSV
# =====================================================

output_csv = DATASET_ROOT / "metadata.csv"

metadata.to_csv(output_csv, index=False)

print(f"\nMetadata saved to:")
print(output_csv)

print("\nFirst 5 rows:")
print(metadata.head())