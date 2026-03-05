"""
preprocessor.py
---------------
Step 1 of the data pipeline.

What this module does:
  1. Scans the raw dataset folder
  2. Detects and removes duplicate images (using MD5 hash)
  3. Splits data into Train / Validation / Test sets using stratified sampling
  4. Copies images into the splits folder with proper structure

Run this ONCE before training:
    python src/data/preprocessor.py
"""

import os
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


# =============================================================================
# Step 1: Duplicate Detection & Removal
# =============================================================================

def compute_md5(file_path: str) -> str:
    """Compute MD5 hash of a file to detect exact duplicates."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def remove_duplicates(data_dir: str) -> dict:
    """
    Scan all images in data_dir, remove exact duplicates.

    Args:
        data_dir: Path to raw dataset root (contains one folder per class).

    Returns:
        dict: {class_name: duplicate_count} — summary of removed duplicates.
    """
    logger.info(f"Scanning for duplicates in: {data_dir}")
    seen_hashes = {}          # hash → first file path
    duplicates_removed = defaultdict(int)
    total_removed = 0

    for class_name in sorted(os.listdir(data_dir)):
        class_path = Path(data_dir) / class_name
        if not class_path.is_dir():
            continue

        for img_file in os.listdir(class_path):
            img_path = class_path / img_file
            if not img_path.is_file():
                continue

            file_hash = compute_md5(str(img_path))

            if file_hash in seen_hashes:
                # Duplicate found — remove it
                img_path.unlink()
                duplicates_removed[class_name] += 1
                total_removed += 1
            else:
                seen_hashes[file_hash] = str(img_path)

    logger.info(f"Duplicate removal complete. Total removed: {total_removed}")
    for cls, count in duplicates_removed.items():
        logger.info(f"  {cls}: {count} duplicates removed")

    return dict(duplicates_removed)


# =============================================================================
# Step 2: Dataset Summary
# =============================================================================

def get_class_distribution(data_dir: str) -> pd.DataFrame:
    """
    Count how many images exist per class.

    Args:
        data_dir: Path to dataset root.

    Returns:
        pd.DataFrame with columns ['Class', 'Image Count']
    """
    records = []
    for class_name in sorted(os.listdir(data_dir)):
        class_path = Path(data_dir) / class_name
        if class_path.is_dir():
            count = len([f for f in os.listdir(class_path) if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
            records.append({"Class": class_name, "Image Count": count})

    df = pd.DataFrame(records)
    logger.info(f"\nDataset distribution:\n{df.to_string(index=False)}")
    return df


# =============================================================================
# Step 3: Train / Val / Test Split
# =============================================================================

def build_file_list(data_dir: str) -> tuple:
    """
    Collect all image paths and their labels.

    Returns:
        (file_paths, labels): Two parallel lists.
    """
    file_paths, labels = [], []

    for class_name in sorted(os.listdir(data_dir)):
        class_path = Path(data_dir) / class_name
        if not class_path.is_dir():
            continue
        for img_file in os.listdir(class_path):
            if Path(img_file).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                file_paths.append(str(class_path / img_file))
                labels.append(class_name)

    return file_paths, labels


def split_dataset(data_dir: str, split_dir: str,
                  train_ratio: float = 0.60,
                  val_ratio: float = 0.20,
                  test_ratio: float = 0.20,
                  seed: int = 42) -> None:
    """
    Split the dataset into train/val/test and copy images to split_dir.

    Directory structure created:
        split_dir/
            train/
                Bacterialblight/  (images...)
                Blast/
                ...
            val/
                ...
            test/
                ...

    Args:
        data_dir:    Source directory with raw images.
        split_dir:   Target directory for splits.
        train_ratio: Fraction of data for training (default 0.60).
        val_ratio:   Fraction for validation (default 0.20).
        test_ratio:  Fraction for test (default 0.20).
        seed:        Random seed for reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    set_seed(seed)
    file_paths, labels = build_file_list(data_dir)
    total = len(file_paths)
    logger.info(f"Total images found: {total}")

    # --- First split: separate test set ---
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed
    )

    # --- Second split: separate val from the remaining train+val ---
    # Recompute val fraction relative to remaining data
    val_fraction = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=seed
    )

    splits = {
        "train": (train_paths, train_labels),
        "val":   (val_paths,   val_labels),
        "test":  (test_paths,  test_labels),
    }

    logger.info(f"Split sizes — Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # --- Copy images to split directories ---
    for split_name, (paths, lbls) in splits.items():
        for src_path, label in zip(paths, lbls):
            dst_dir = Path(split_dir) / split_name / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / Path(src_path).name
            if not dst_path.exists():  # Skip if already copied
                shutil.copy2(src_path, dst_path)

    logger.info(f"Images copied to: {split_dir}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg.project.random_seed)

    # Step 1 — Remove duplicates from raw data
    remove_duplicates(cfg.data.raw_dir)

    # Step 2 — Print distribution
    get_class_distribution(cfg.data.raw_dir)

    # Step 3 — Create train/val/test splits
    split_dataset(
        data_dir=cfg.data.raw_dir,
        split_dir=cfg.data.split_dir,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.project.random_seed,
    )

    logger.info("Preprocessing complete. Ready for training.")
