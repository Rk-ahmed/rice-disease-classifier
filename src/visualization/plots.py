"""
plots.py
--------
All visualization functions for the thesis.

Generates:
    1. Training curves (accuracy, loss, precision, recall)
    2. Confusion matrix heatmap
    3. ROC-AUC curves (one-vs-rest)
    4. Model comparison bar chart
    5. Class distribution charts
    6. Grad-CAM (explainability heatmaps)

Usage:
    from src.visualization.plots import plot_training_curves, plot_confusion_matrix
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Consistent style across all thesis figures
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})
PALETTE = sns.color_palette("Set2", 8)


# =============================================================================
# 1. Training Curves
# =============================================================================

def plot_training_curves(history: dict, model_name: str, save_dir: str = "outputs/plots") -> None:
    """
    Plot 4 training metrics over epochs: Loss, Accuracy, Precision, Recall.

    Args:
        history:    Keras history.history dict (or loaded from JSON).
        model_name: Used in title and filename.
        save_dir:   Where to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training History — {model_name.upper()}", fontsize=16, fontweight="bold")

    metrics = [
        ("loss",      "val_loss",      "Loss",      "Loss"),
        ("accuracy",  "val_accuracy",  "Accuracy",  "Accuracy"),
        ("precision", "val_precision", "Precision", "Precision"),
        ("recall",    "val_recall",    "Recall",    "Recall"),
    ]

    for ax, (tr_key, val_key, title, ylabel) in zip(axes.flat, metrics):
        if tr_key not in history:
            ax.set_visible(False)
            continue

        epochs = range(1, len(history[tr_key]) + 1)
        ax.plot(epochs, history[tr_key],  color=PALETTE[0], label="Train",      linewidth=2)
        ax.plot(epochs, history[val_key], color=PALETTE[1], label="Validation", linewidth=2, linestyle="--")

        # Mark best epoch
        if "loss" in tr_key:
            best_epoch = int(np.argmin(history[val_key])) + 1
            best_val   = min(history[val_key])
        else:
            best_epoch = int(np.argmax(history[val_key])) + 1
            best_val   = max(history[val_key])

        ax.axvline(best_epoch, color="red", linestyle=":", alpha=0.7, label=f"Best epoch={best_epoch}")
        ax.scatter([best_epoch], [best_val], color="red", s=80, zorder=5)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_dir, f"training_curves_{model_name}.png")


# =============================================================================
# 2. Confusion Matrix
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names: list,
                           model_name: str, save_dir: str = "outputs/plots") -> None:
    """
    Plot a normalized and raw confusion matrix side by side.

    Args:
        y_true:       True labels (integers).
        y_pred:       Predicted labels (integers).
        class_names:  List of class name strings.
        model_name:   Used in title and filename.
        save_dir:     Save directory.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix — {model_name.upper()}", fontsize=15, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title("Raw Counts")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    # Normalized (shows per-class accuracy)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax2,
                vmin=0, vmax=1)
    ax2.set_title("Normalized (Row %)")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")

    plt.tight_layout()
    _save_figure(fig, save_dir, f"confusion_matrix_{model_name}.png")


# =============================================================================
# 3. ROC-AUC Curves
# =============================================================================

def plot_roc_curves(y_true, y_pred_proba, class_names: list,
                    model_name: str, save_dir: str = "outputs/plots") -> None:
    """
    Plot ROC curves for each class (One-vs-Rest approach).

    Args:
        y_true:         True integer labels.
        y_pred_proba:   Probability predictions (n_samples, n_classes).
        class_names:    List of class name strings.
        model_name:     For title and filename.
        save_dir:       Save directory.
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, cls_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i], linewidth=2,
                label=f"{cls_name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (One-vs-Rest) — {model_name.upper()}")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_dir, f"roc_curves_{model_name}.png")


# =============================================================================
# 4. Model Comparison Bar Chart
# =============================================================================

def plot_model_comparison(comparison_df: pd.DataFrame,
                           save_dir: str = "outputs/plots") -> None:
    """
    Bar chart comparing all models across key metrics.

    Args:
        comparison_df: DataFrame with models as index, metrics as columns.
        save_dir:      Save directory.
    """
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc"]
    metrics = [m for m in metrics if m in comparison_df.columns]
    labels  = [m.replace("test_", "").replace("_", " ").title() for m in metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(comparison_df)

    for i, (model_name, row) in enumerate(comparison_df.iterrows()):
        offset = (i - len(comparison_df) / 2 + 0.5) * width
        values = [row.get(m, 0) for m in metrics]
        bars = ax.bar(x + offset, values, width, label=model_name.upper(), color=PALETTE[i])

        # Add value labels on top
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison on Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([max(0, comparison_df[metrics].values.min() - 0.05), 1.02])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_dir, "model_comparison.png")


# =============================================================================
# 5. Class Distribution
# =============================================================================

def plot_class_distribution(class_counts: dict, save_dir: str = "outputs/plots") -> None:
    """
    Bar + Pie chart of class distribution.

    Args:
        class_counts: {class_name: count}
        save_dir:     Save directory.
    """
    names  = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dataset Class Distribution", fontsize=15, fontweight="bold")

    # Bar chart
    bars = ax1.bar(names, counts, color=PALETTE[:len(names)])
    ax1.set_xlabel("Disease Class")
    ax1.set_ylabel("Number of Images")
    ax1.set_title("Image Count per Class")
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha="center", va="bottom", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Pie chart
    ax2.pie(counts, labels=names, colors=PALETTE[:len(names)],
            autopct="%1.1f%%", startangle=140)
    ax2.set_title("Class Distribution (%)")

    plt.tight_layout()
    _save_figure(fig, save_dir, "class_distribution.png")


# =============================================================================
# 6. Grad-CAM (Explainability)
# =============================================================================

def generate_gradcam(model, img_array: np.ndarray, class_idx: int,
                     last_conv_layer_name: str = None) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a single image.

    Grad-CAM shows WHICH parts of the image the model focused on
    when making its prediction. This is critical for thesis explainability.

    Args:
        model:               Trained Keras model.
        img_array:           Preprocessed image array (1, H, W, 3).
        class_idx:           The predicted class index.
        last_conv_layer_name: Name of the last conv layer. Auto-detected if None.

    Returns:
        np.ndarray: Heatmap of shape (H, W), values in [0, 1].
    """
    import tensorflow as tf

    # Find the last convolutional layer automatically
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Conv layers have 4D output
                last_conv_layer_name = layer.name
                break

    # Create a model that outputs: [last_conv_output, final_predictions]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Focus on the predicted class score
        loss = predictions[:, class_idx]

    # Gradient of the class score w.r.t. conv layer outputs
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def plot_gradcam(model, img_path: str, class_names: list,
                 save_dir: str = "outputs/plots", last_conv_layer: str = None) -> None:
    """
    Generate and save a Grad-CAM visualization.

    Args:
        model:           Trained Keras model.
        img_path:        Path to the input image file.
        class_names:     List of class name strings.
        save_dir:        Save directory.
        last_conv_layer: Name of the conv layer. Auto-detected if None.
    """
    import cv2
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array_batch, verbose=0)
    class_idx   = int(np.argmax(predictions[0]))
    confidence  = float(predictions[0][class_idx])
    class_name  = class_names[class_idx]

    heatmap = generate_gradcam(model, img_array_batch, class_idx, last_conv_layer)

    # Resize heatmap to image size and overlay
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose on original image
    overlay = (img_array * 255).astype(np.uint8)
    superimposed = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Grad-CAM: Predicted = {class_name} ({confidence:.1%})",
                 fontsize=14, fontweight="bold")

    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    stem = Path(img_path).stem
    _save_figure(fig, save_dir, f"gradcam_{class_name}_{stem}.png")


# =============================================================================
# Helper
# =============================================================================

def _save_figure(fig, save_dir: str, filename: str) -> None:
    """Save figure and close it to free memory."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_dir) / filename
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Figure saved: {filepath}")
