"""
train.py
--------
Main training script. Trains all enabled models with Stratified K-Fold
cross-validation and saves results.

What this script does:
    1. Loads config and sets random seed
    2. Preprocesses data (if not done already)
    3. For each enabled model (VGG16, ResNet50, InceptionV3, Xception):
        a. Run Stratified K-Fold cross-validation
        b. Save per-fold metrics
        c. Train final model on full train+val data
        d. Evaluate on held-out test set
    4. Save a comparison table of all models

Run from project root:
    python src/models/train.py

To train only one model:
    python src/models/train.py --model vgg16
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.data_loader import (
    compute_class_weights,
    get_data_generators,
    get_fold_generators,
)
from src.data.preprocessor import build_file_list
from src.models.build_model import build_model, get_callbacks
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__, log_to_file=True)


# =============================================================================
# Cross-Validation Training
# =============================================================================

def run_cross_validation(model_name: str, cfg) -> list:
    """
    Run Stratified K-Fold cross-validation for one model.

    Why K-Fold?
        - Gives a reliable estimate of model performance.
        - Each sample is used for both training and validation exactly once.
        - Stratified ensures each fold has the same class distribution.

    Args:
        model_name: e.g. 'vgg16'
        cfg:        Project config.

    Returns:
        List of dicts — one per fold with metrics:
        [{'fold': 1, 'val_loss': ..., 'val_accuracy': ..., ...}, ...]
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Cross-validation: {model_name.upper()} | {cfg.training.n_folds} folds")
    logger.info(f"{'='*60}")

    # Load all train+val file paths and labels (test set excluded)
    all_paths, all_labels = build_file_list(cfg.data.split_dir + "/train")
    val_paths,  val_lbs   = build_file_list(cfg.data.split_dir + "/val")
    all_paths  += val_paths
    all_labels += val_lbs

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)

    skf = StratifiedKFold(
        n_splits=cfg.training.n_folds,
        shuffle=True,
        random_state=cfg.project.random_seed,
    )

    model_cfg = getattr(cfg.models, model_name)
    fold_results = []
    fold_dir = str(Path(cfg.data.processed_dir) / "cv_folds" / model_name)

    for fold_num, (train_idx, val_idx) in enumerate(
        skf.split(all_paths, all_labels), start=1
    ):
        logger.info(f"\n--- Fold {fold_num}/{cfg.training.n_folds} ---")

        fold_train_paths  = all_paths[train_idx].tolist()
        fold_train_labels = all_labels[train_idx].tolist()
        fold_val_paths    = all_paths[val_idx].tolist()
        fold_val_labels   = all_labels[val_idx].tolist()

        train_gen, val_gen = get_fold_generators(
            fold_train_paths, fold_train_labels,
            fold_val_paths,   fold_val_labels,
            fold_dir, fold_num, cfg,
        )

        num_classes = len(train_gen.class_indices)

        # Build a fresh model for each fold
        model = build_model(
            model_name=model_name,
            num_classes=num_classes,
            input_shape=(*cfg.data.image_size, 3),
            dense_units=model_cfg.dense_units,
            dropout_rate=model_cfg.dropout_rate,
            unfreeze_last_n_layers=model_cfg.unfreeze_last_n_layers,
            learning_rate=cfg.training.learning_rate,
        )

        class_weights = compute_class_weights(train_gen) if cfg.training.use_class_weights else None

        fold_save_dir = str(Path(cfg.outputs.models_dir) / model_name / "folds")
        fold_checkpoint = str(Path(fold_save_dir) / f"best_{model_name}_fold{fold_num}.keras")
        Path(fold_save_dir).mkdir(parents=True, exist_ok=True)

        import tensorflow as tf
        fold_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.training.early_stopping_patience,
                restore_best_weights=True, verbose=0,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=fold_checkpoint, monitor="val_loss",
                save_best_only=True, verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=cfg.training.reduce_lr_factor,
                patience=cfg.training.reduce_lr_patience,
                min_lr=cfg.training.min_lr, verbose=0,
            ),
        ]

        model.fit(
            train_gen,
            epochs=cfg.training.epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=fold_callbacks,
            verbose=1,
        )

        # Evaluate on this fold's validation set
        results = model.evaluate(val_gen, verbose=0)
        metrics = dict(zip(model.metrics_names, results))

        fold_result = {
            "fold":          fold_num,
            "val_loss":      metrics.get("loss", None),
            "val_accuracy":  metrics.get("accuracy", None),
            "val_precision": metrics.get("precision", None),
            "val_recall":    metrics.get("recall", None),
        }
        fold_results.append(fold_result)

        logger.info(
            f"Fold {fold_num} results — "
            f"Loss: {fold_result['val_loss']:.4f} | "
            f"Accuracy: {fold_result['val_accuracy']:.4f} | "
            f"Precision: {fold_result['val_precision']:.4f} | "
            f"Recall: {fold_result['val_recall']:.4f}"
        )

        # Free memory before next fold
        del model
        import tensorflow as tf
        tf.keras.backend.clear_session()

    # Print CV summary
    accs = [r["val_accuracy"] for r in fold_results]
    logger.info(f"\nCV Summary for {model_name.upper()}:")
    logger.info(f"  Mean Accuracy:  {np.mean(accs):.4f}")
    logger.info(f"  Std Accuracy:   {np.std(accs):.4f}")
    logger.info(f"  Min: {np.min(accs):.4f} | Max: {np.max(accs):.4f}")

    return fold_results


# =============================================================================
# Final Model Training
# =============================================================================

def train_final_model(model_name: str, cfg):
    """
    Train the final model on the full train+val data, then evaluate on test set.

    This is separate from cross-validation. After CV tells us the model is
    stable, we train one final model on all available non-test data for
    the best possible weights before deployment.

    Args:
        model_name: e.g. 'vgg16'
        cfg:        Project config.

    Returns:
        dict: Final test set metrics.
    """
    logger.info(f"\nTraining FINAL model: {model_name.upper()}")

    train_gen, val_gen, test_gen = get_data_generators(cfg)
    num_classes = len(train_gen.class_indices)

    model_cfg = getattr(cfg.models, model_name)
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        input_shape=(*cfg.data.image_size, 3),
        dense_units=model_cfg.dense_units,
        dropout_rate=model_cfg.dropout_rate,
        unfreeze_last_n_layers=model_cfg.unfreeze_last_n_layers,
        learning_rate=cfg.training.learning_rate,
    )

    class_weights = compute_class_weights(train_gen) if cfg.training.use_class_weights else None
    callbacks = get_callbacks(
        model_name=model_name,
        save_dir=str(Path(cfg.outputs.models_dir) / model_name),
        cfg=cfg,
    )

    history = model.fit(
        train_gen,
        epochs=cfg.training.epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Evaluate on test set ---
    logger.info(f"Evaluating {model_name.upper()} on test set...")
    test_gen.reset()
    results = model.evaluate(test_gen, verbose=1)
    metrics = dict(zip(model.metrics_names, results))

    # Additional metrics
    import tensorflow as tf
    from sklearn.metrics import f1_score, roc_auc_score, classification_report

    test_gen.reset()
    y_true = test_gen.classes
    y_pred_proba = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    f1  = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr")

    final_metrics = {
        "model":          model_name,
        "test_loss":      metrics.get("loss"),
        "test_accuracy":  metrics.get("accuracy"),
        "test_precision": metrics.get("precision"),
        "test_recall":    metrics.get("recall"),
        "test_f1":        f1,
        "test_auc":       auc,
    }

    logger.info(f"\nFinal test results for {model_name.upper()}:")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    # Save classification report
    report_dir = Path(cfg.outputs.reports_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(train_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(report_dir / f"classification_report_{model_name}.txt", "w") as f:
        f.write(report)

    # Save history for plotting later
    history_dir = Path(cfg.outputs.reports_dir)
    with open(history_dir / f"history_{model_name}.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    # Save model
    model_save_path = str(Path(cfg.outputs.models_dir) / model_name / f"final_{model_name}.keras")
    model.save(model_save_path)
    logger.info(f"Model saved: {model_save_path}")

    return final_metrics, history


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train rice disease classification models")
    parser.add_argument("--model", type=str, default=None,
                        help="Train a specific model only (e.g. vgg16). Default: all enabled models.")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip cross-validation and go straight to final training.")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg.project.random_seed)

    # Determine which models to train
    all_models = ["vgg16", "resnet50", "inceptionv3", "xception"]
    if args.model:
        models_to_train = [args.model.lower()]
    else:
        models_to_train = [m for m in all_models if getattr(cfg.models, m).enabled]

    logger.info(f"Models to train: {models_to_train}")

    all_cv_results    = {}
    all_final_metrics = []

    for model_name in models_to_train:
        # --- Cross-validation ---
        if cfg.training.use_cross_validation and not args.skip_cv:
            cv_results = run_cross_validation(model_name, cfg)
            all_cv_results[model_name] = cv_results

            # Save CV results
            cv_save_path = Path(cfg.outputs.reports_dir) / f"cv_results_{model_name}.json"
            cv_save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cv_save_path, "w") as f:
                json.dump(cv_results, f, indent=2)

        # --- Final model training ---
        final_metrics, history = train_final_model(model_name, cfg)
        all_final_metrics.append(final_metrics)

    # --- Comparison table ---
    if len(all_final_metrics) > 1:
        comparison_df = pd.DataFrame(all_final_metrics)
        comparison_df = comparison_df.set_index("model")
        comparison_df = comparison_df.round(4)

        comparison_path = Path(cfg.outputs.reports_dir) / "model_comparison.csv"
        comparison_df.to_csv(comparison_path)
        logger.info(f"\nModel Comparison:\n{comparison_df.to_string()}")
        logger.info(f"Saved to: {comparison_path}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
