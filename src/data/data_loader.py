"""
data_loader.py
--------------
Creates TensorFlow/Keras data generators for training, validation, and test.

Key design decisions:
  - Training data: augmented (rotation, flip, zoom, etc.)
  - Validation/Test data: NO augmentation (only rescaling)
  - Uses flow_from_directory so no need to load all images into RAM
  - Supports class weight computation for imbalanced datasets

Usage:
    from src.data.data_loader import get_data_generators
    train_gen, val_gen, test_gen = get_data_generators(cfg)
"""

import os
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_data_generators(cfg, split_dir: str = None):
    """
    Build and return train, validation, and test data generators.

    Args:
        cfg:       Project config loaded with load_config().
        split_dir: Override the split directory path (optional).

    Returns:
        Tuple: (train_generator, val_generator, test_generator)
    """
    split_dir = split_dir or cfg.data.split_dir
    image_size = tuple(cfg.data.image_size)   # (224, 224)
    batch_size = cfg.data.batch_size
    aug = cfg.augmentation

    # -----------------------------------------------------------------
    # Training generator — WITH augmentation
    # -----------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,                       # Normalize pixel values to [0, 1]
        rotation_range=aug.rotation_range,        # Random rotation up to 40°
        width_shift_range=aug.width_shift_range,  # Random horizontal shift
        height_shift_range=aug.height_shift_range,# Random vertical shift
        shear_range=aug.shear_range,              # Shear transformation
        zoom_range=aug.zoom_range,                # Random zoom
        horizontal_flip=aug.horizontal_flip,      # Mirror images horizontally
        brightness_range=aug.brightness_range,    # Vary image brightness
        fill_mode=aug.fill_mode,                  # Fill empty pixels after transforms
    )

    # -----------------------------------------------------------------
    # Validation & Test generator — NO augmentation (only rescaling)
    # -----------------------------------------------------------------
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # -----------------------------------------------------------------
    # Create generators from directory structure
    # -----------------------------------------------------------------
    train_generator = train_datagen.flow_from_directory(
        directory=str(Path(split_dir) / "train"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_generator = val_test_datagen.flow_from_directory(
        directory=str(Path(split_dir) / "val"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,   # No shuffle for evaluation — keep order consistent
    )

    test_generator = val_test_datagen.flow_from_directory(
        directory=str(Path(split_dir) / "test"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    logger.info(f"Classes detected: {list(train_generator.class_indices.keys())}")
    logger.info(f"Train: {train_generator.samples} images | "
                f"Val: {val_generator.samples} images | "
                f"Test: {test_generator.samples} images")

    return train_generator, val_generator, test_generator


def compute_class_weights(train_generator) -> dict:
    """
    Compute class weights to handle class imbalance.

    Classes with fewer images get higher weights so the model
    pays equal attention to all classes during training.

    Args:
        train_generator: Keras ImageDataGenerator training generator.

    Returns:
        dict: {class_index: weight} mapping.
    """
    labels = train_generator.classes
    class_names = list(train_generator.class_indices.keys())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )

    class_weights_dict = dict(enumerate(weights))

    logger.info("Class weights computed:")
    for idx, name in enumerate(class_names):
        logger.info(f"  {name}: {class_weights_dict[idx]:.4f}")

    return class_weights_dict


def get_fold_generators(train_paths, train_labels, val_paths, val_labels,
                        fold_dir: str, fold_num: int, cfg):
    """
    Build generators for a specific cross-validation fold.
    Images are copied temporarily to fold_dir for this fold.

    Args:
        train_paths:  List of image paths for this fold's training set.
        train_labels: Corresponding class labels.
        val_paths:    Image paths for validation.
        val_labels:   Corresponding class labels.
        fold_dir:     Base directory to store fold data.
        fold_num:     Current fold number (1-based, used for directory naming).
        cfg:          Project config.

    Returns:
        Tuple: (train_generator, val_generator)
    """
    import shutil

    fold_train_dir = Path(fold_dir) / f"fold_{fold_num}" / "train"
    fold_val_dir   = Path(fold_dir) / f"fold_{fold_num}" / "val"

    # Copy images for this fold (only if not already there)
    for path, label in zip(train_paths, train_labels):
        dst = fold_train_dir / label / Path(path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(path, dst)

    for path, label in zip(val_paths, val_labels):
        dst = fold_val_dir / label / Path(path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(path, dst)

    aug = cfg.augmentation
    image_size = tuple(cfg.data.image_size)
    batch_size = cfg.data.batch_size

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=aug.rotation_range,
        width_shift_range=aug.width_shift_range,
        height_shift_range=aug.height_shift_range,
        shear_range=aug.shear_range,
        zoom_range=aug.zoom_range,
        horizontal_flip=aug.horizontal_flip,
        brightness_range=aug.brightness_range,
        fill_mode=aug.fill_mode,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        str(fold_train_dir), target_size=image_size,
        batch_size=batch_size, class_mode="categorical",
        shuffle=True, seed=42,
    )
    val_gen = val_datagen.flow_from_directory(
        str(fold_val_dir), target_size=image_size,
        batch_size=batch_size, class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen
