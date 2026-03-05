"""
build_model.py
--------------
Builds any of the 4 supported CNN architectures with transfer learning.

Supported models:
    - VGG16
    - ResNet50
    - InceptionV3
    - Xception

Design pattern (same for all models):
    1. Load pre-trained base (ImageNet weights, no top layer)
    2. Freeze all base layers initially
    3. Add custom classification head (GAP → Dense → Dropout → Softmax)
    4. Unfreeze the last N layers for fine-tuning
    5. Compile with Adam optimizer

Why this 2-stage freeze/unfreeze approach?
    - Freezing first prevents the pre-trained weights from being destroyed
      during early training when the new head has random weights.
    - Unfreezing a few top layers afterwards allows fine-tuning the most
      task-specific features in the base model.

Usage:
    from src.models.build_model import build_model
    model = build_model("vgg16", num_classes=4)
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maps string names → Keras application functions
SUPPORTED_MODELS = {
    "vgg16":       VGG16,
    "resnet50":    ResNet50,
    "inceptionv3": InceptionV3,
    "xception":    Xception,
}


def build_model(
    model_name: str,
    num_classes: int,
    input_shape: tuple = (224, 224, 3),
    dense_units: int = 1024,
    dropout_rate: float = 0.5,
    unfreeze_last_n_layers: int = 4,
    learning_rate: float = 0.0001,
) -> tf.keras.Model:
    """
    Build and compile a transfer learning model.

    Args:
        model_name:             One of: 'vgg16', 'resnet50', 'inceptionv3', 'xception'.
        num_classes:            Number of output classes (e.g., 4 for 4 diseases).
        input_shape:            Input image shape (H, W, C). Default: (224, 224, 3).
        dense_units:            Number of neurons in the custom dense layer.
        dropout_rate:           Dropout probability for regularization (0–1).
        unfreeze_last_n_layers: Number of base model layers to unfreeze for fine-tuning.
        learning_rate:          Adam optimizer learning rate.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
    model_name = model_name.lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(SUPPORTED_MODELS.keys())}"
        )

    logger.info(f"Building model: {model_name.upper()} | Classes: {num_classes} | "
                f"Unfreeze last {unfreeze_last_n_layers} layers")

    # ------------------------------------------------------------------
    # 1. Load pre-trained base model (ImageNet weights, no classification head)
    # ------------------------------------------------------------------
    base_model = SUPPORTED_MODELS[model_name](
        weights="imagenet",
        include_top=False,         # Remove the original 1000-class top layer
        input_shape=input_shape,
    )

    # ------------------------------------------------------------------
    # 2. Freeze ALL base model layers (we train only our new head first)
    # ------------------------------------------------------------------
    for layer in base_model.layers:
        layer.trainable = False

    # ------------------------------------------------------------------
    # 3. Add custom classification head
    # ------------------------------------------------------------------
    x = base_model.output
    x = GlobalAveragePooling2D()(x)          # Reduces feature maps to a 1D vector
    x = Dense(dense_units, activation="relu")(x)  # Learns task-specific features
    x = Dropout(dropout_rate)(x)             # Prevents overfitting
    output = Dense(num_classes, activation="softmax")(x)  # Probability per class

    model = Model(inputs=base_model.input, outputs=output)

    # ------------------------------------------------------------------
    # 4. Unfreeze the last N layers of the base model for fine-tuning
    # ------------------------------------------------------------------
    if unfreeze_last_n_layers > 0:
        for layer in base_model.layers[-unfreeze_last_n_layers:]:
            layer.trainable = True

    total_layers = len(model.layers)
    trainable_layers = sum(1 for l in model.layers if l.trainable)
    logger.info(f"Total layers: {total_layers} | Trainable: {trainable_layers}")

    # ------------------------------------------------------------------
    # 5. Compile
    # ------------------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def get_callbacks(model_name: str, save_dir: str = "outputs/models", cfg=None):
    """
    Create standard training callbacks.

    Callbacks:
        - EarlyStopping:     Stops training if val_loss stops improving.
        - ModelCheckpoint:   Saves the best model weights automatically.
        - ReduceLROnPlateau: Reduces learning rate when stuck.

    Args:
        model_name: Used for naming the saved checkpoint file.
        save_dir:   Directory to save model checkpoints.
        cfg:        Project config (for patience values etc.).

    Returns:
        List of Keras callbacks.
    """
    from pathlib import Path
    import tensorflow as tf

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Default values (overridden by config if provided)
    es_patience  = getattr(cfg.training, "early_stopping_patience", 10) if cfg else 10
    lr_patience  = getattr(cfg.training, "reduce_lr_patience", 5)       if cfg else 5
    lr_factor    = getattr(cfg.training, "reduce_lr_factor", 0.2)        if cfg else 0.2
    min_lr       = getattr(cfg.training, "min_lr", 1e-6)                 if cfg else 1e-6

    checkpoint_path = str(Path(save_dir) / f"best_{model_name}.keras")

    callbacks = [
        # Stop training when val_loss hasn't improved for `patience` epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save the model with the lowest validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Reduce LR by `factor` when val_loss is stuck for `patience` epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
    ]

    return callbacks
