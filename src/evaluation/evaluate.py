"""
evaluate.py
-----------
Comprehensive model evaluation module.

Computes:
    - Accuracy, Precision, Recall, F1, AUC-ROC
    - Confusion Matrix
    - Per-class Classification Report
    - Statistical Tests:
        * McNemar's Test  — compares two models on the same test set
        * Wilcoxon Signed-Rank Test — compares CV fold results
        * Paired t-test — parametric comparison of model pairs
        * 95% Confidence Intervals — for each metric

Why statistical tests?
    High accuracy alone is not enough for a Master's thesis.
    We need to prove that VGG16 is STATISTICALLY SIGNIFICANTLY better
    than the other models — not just by chance.

Usage:
    from src.evaluation.evaluate import evaluate_model, compare_models_statistically
"""

import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Single Model Evaluation
# =============================================================================

def evaluate_model(model, test_generator, class_names: list = None) -> dict:
    """
    Compute all evaluation metrics for a trained model.

    Args:
        model:           Trained Keras model.
        test_generator:  Keras ImageDataGenerator test generator (shuffle=False).
        class_names:     List of class names. Auto-detected if None.

    Returns:
        dict with all computed metrics.
    """
    if class_names is None:
        class_names = list(test_generator.class_indices.keys())

    # Get true labels and predicted probabilities
    test_generator.reset()
    y_true = test_generator.classes
    y_pred_proba = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # --- Core metrics ---
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc       = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro")

    # --- Confidence intervals (bootstrapped) ---
    ci_accuracy = bootstrap_confidence_interval(y_true, y_pred, accuracy_score)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)

    # --- Per-class report ---
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        "accuracy":             accuracy,
        "precision_macro":      precision,
        "recall_macro":         recall,
        "f1_macro":             f1,
        "auc_roc":              auc,
        "accuracy_ci_lower":    ci_accuracy[0],
        "accuracy_ci_upper":    ci_accuracy[1],
        "confusion_matrix":     cm.tolist(),
        "classification_report": report,
        "y_true":               y_true.tolist(),
        "y_pred":               y_pred.tolist(),
        "y_pred_proba":         y_pred_proba.tolist(),
    }

    logger.info(f"  Accuracy:  {accuracy:.4f} (95% CI: [{ci_accuracy[0]:.4f}, {ci_accuracy[1]:.4f}])")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  AUC-ROC:   {auc:.4f}")

    return metrics


# =============================================================================
# Confidence Intervals via Bootstrap
# =============================================================================

def bootstrap_confidence_interval(y_true, y_pred, metric_fn,
                                   n_iterations: int = 1000,
                                   confidence: float = 0.95) -> tuple:
    """
    Compute a bootstrap confidence interval for any metric.

    Bootstrap resampling: randomly sample the test set WITH replacement
    many times, compute the metric each time, then take the percentile range.

    Args:
        y_true:       True labels.
        y_pred:       Predicted labels.
        metric_fn:    A function like accuracy_score(y_true, y_pred).
        n_iterations: Number of bootstrap samples (1000 is standard).
        confidence:   Confidence level (default: 0.95 for 95% CI).

    Returns:
        (lower_bound, upper_bound) as floats.
    """
    rng = np.random.RandomState(42)
    scores = []
    n = len(y_true)

    for _ in range(n_iterations):
        indices = rng.randint(0, n, n)
        score = metric_fn(np.array(y_true)[indices], np.array(y_pred)[indices])
        scores.append(score)

    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return lower, upper


# =============================================================================
# Statistical Tests for Model Comparison
# =============================================================================

def mcnemar_test(y_true, y_pred_model_a, y_pred_model_b) -> dict:
    """
    McNemar's Test: Are two models making the SAME errors?

    This is the gold-standard statistical test for comparing classifiers
    on the same test set. It checks whether the difference in error rates
    between two models is statistically significant.

    Null hypothesis: Both models have the same error rate.
    If p-value < 0.05, we REJECT H0 → the models are significantly different.

    Args:
        y_true:          True labels.
        y_pred_model_a:  Predicted labels from model A.
        y_pred_model_b:  Predicted labels from model B.

    Returns:
        dict with 'statistic', 'p_value', and 'significant' (bool).
    """
    # Build contingency table:
    #   n00: both wrong
    #   n01: A wrong, B correct
    #   n10: A correct, B wrong
    #   n11: both correct
    correct_a = (np.array(y_pred_model_a) == np.array(y_true))
    correct_b = (np.array(y_pred_model_b) == np.array(y_true))

    n01 = np.sum(~correct_a & correct_b)   # A wrong, B right
    n10 = np.sum(correct_a & ~correct_b)   # A right, B wrong

    # McNemar statistic with continuity correction
    if (n01 + n10) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "n01": int(n01), "n10": int(n10)}

    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value   = stats.chi2.sf(statistic, df=1)

    return {
        "statistic":   float(statistic),
        "p_value":     float(p_value),
        "significant": bool(p_value < 0.05),
        "n01":         int(n01),   # Times A wrong, B right
        "n10":         int(n10),   # Times A right, B wrong
        "interpretation": (
            f"A is correct when B is wrong {n10} times. "
            f"B is correct when A is wrong {n01} times. "
            f"{'Significant difference (p<0.05).' if p_value < 0.05 else 'No significant difference (p≥0.05).'}"
        )
    }


def wilcoxon_test(scores_a: list, scores_b: list) -> dict:
    """
    Wilcoxon Signed-Rank Test on cross-validation fold scores.

    Non-parametric alternative to paired t-test. Better for small samples
    (like 5 CV folds) where we can't assume normality.

    Args:
        scores_a: Metric values from model A across k folds.
        scores_b: Metric values from model B across k folds.

    Returns:
        dict with 'statistic', 'p_value', 'significant'.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Both score lists must have the same length (one per fold).")

    try:
        statistic, p_value = stats.wilcoxon(scores_a, scores_b)
    except ValueError as e:
        # Happens when all differences are zero
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "note": str(e)}

    return {
        "statistic":   float(statistic),
        "p_value":     float(p_value),
        "significant": bool(p_value < 0.05),
    }


def paired_ttest(scores_a: list, scores_b: list) -> dict:
    """
    Paired t-test comparing mean metric across CV folds.

    Parametric test — assumes differences are normally distributed.
    Use Wilcoxon for small samples; t-test as a supplement.

    Args:
        scores_a: Metric values from model A across k folds.
        scores_b: Metric values from model B across k folds.

    Returns:
        dict with 'statistic', 'p_value', 'significant'.
    """
    statistic, p_value = stats.ttest_rel(scores_a, scores_b)
    return {
        "statistic":   float(statistic),
        "p_value":     float(p_value),
        "significant": bool(p_value < 0.05),
    }


# =============================================================================
# Full Model Comparison with Statistical Tests
# =============================================================================

def compare_models_statistically(
    model_results: dict,
    cv_results: dict = None,
    save_dir: str = "outputs/reports",
) -> pd.DataFrame:
    """
    Run all statistical tests across all model pairs.

    Args:
        model_results: {model_name: {'y_true': [...], 'y_pred': [...]}}
        cv_results:    {model_name: [{'val_accuracy': ..., ...}, ...]}
                       — CV fold results for Wilcoxon/t-test.
        save_dir:      Where to save the results table.

    Returns:
        pd.DataFrame with pairwise statistical test results.
    """
    model_names = list(model_results.keys())
    rows = []

    for model_a, model_b in combinations(model_names, 2):
        y_true = model_results[model_a]["y_true"]
        pred_a = model_results[model_a]["y_pred"]
        pred_b = model_results[model_b]["y_pred"]

        # McNemar's test on test set predictions
        mcn = mcnemar_test(y_true, pred_a, pred_b)

        row = {
            "Model A":              model_a,
            "Model B":              model_b,
            "McNemar Statistic":    round(mcn["statistic"], 4),
            "McNemar p-value":      round(mcn["p_value"], 4),
            "McNemar Significant":  mcn["significant"],
        }

        # Wilcoxon + t-test on CV fold accuracies
        if cv_results and model_a in cv_results and model_b in cv_results:
            acc_a = [r["val_accuracy"] for r in cv_results[model_a]]
            acc_b = [r["val_accuracy"] for r in cv_results[model_b]]

            wil = wilcoxon_test(acc_a, acc_b)
            tt  = paired_ttest(acc_a, acc_b)

            row.update({
                "Wilcoxon p-value":     round(wil["p_value"], 4),
                "Wilcoxon Significant": wil["significant"],
                "t-test p-value":       round(tt["p_value"], 4),
                "t-test Significant":   tt["significant"],
            })

        rows.append(row)

    df = pd.DataFrame(rows)

    save_path = Path(save_dir) / "statistical_tests.csv"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"\nStatistical test results saved: {save_path}")
    logger.info(f"\n{df.to_string(index=False)}")

    return df
