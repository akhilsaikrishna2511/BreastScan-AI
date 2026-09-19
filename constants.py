"""Shared constants for the breast ultrasound pipeline."""

CLASS_NAMES = ["normal", "benign", "malignant"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}

MASK_THRESHOLD = 0.5
