"""Comprehensive evaluation metrics for multiclass fault classification."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import LabelBinarizer


def compute_multiclass_brier_score(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    classes: list[str] | np.ndarray,
) -> float:
    """Compute the multi-class Brier score (mean squared error over class probability vectors).

    Formula:
        Brier = (1 / N) * sum_{i=1}^N sum_{k=1}^K (p_{ik} - y_{ik})^2
    where y_{ik} is 1 if sample i is of class k, else 0.
    """
    lb = LabelBinarizer()
    lb.fit(classes)
    y_true_bin = lb.transform(y_true)
    if y_true_bin.shape[1] == 1 and len(classes) == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    brier = np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1))
    return float(brier)


def compute_expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    classes: list[str] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute top-label Expected Calibration Error (ECE) with equal-width probability bins.

    Formula:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    classes_arr = np.asarray(classes)
    y_true_arr = np.asarray(y_true)

    confidences = np.max(y_prob, axis=1)
    predictions = classes_arr[np.argmax(y_prob, axis=1)]
    accuracies = predictions == y_true_arr

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true_arr)

    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_model_performance(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    classes: list[str] | np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive multiclass performance metrics.

    Primary metric: macro_f1.
    """
    classes_list = [str(c) for c in classes]
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # Per-class classification report
    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=classes_list,
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics: dict[str, dict[str, float]] = {}
    for cls in classes_list:
        if cls in report_dict:
            per_class_metrics[cls] = {
                "precision": round(float(report_dict[cls]["precision"]), 4),
                "recall": round(float(report_dict[cls]["recall"]), 4),
                "f1_score": round(float(report_dict[cls]["f1-score"]), 4),
                "support": int(report_dict[cls]["support"]),
            }

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes_list)

    metrics: dict[str, Any] = {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "accuracy": round(acc, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classes": classes_list,
    }

    if y_prob is not None:
        try:
            ll = float(log_loss(y_true_arr, y_prob, labels=classes_list))
            metrics["log_loss"] = round(ll, 4)
        except Exception:
            metrics["log_loss"] = None

        brier = compute_multiclass_brier_score(y_true_arr, y_prob, classes_list)
        metrics["brier_score"] = round(brier, 4)

        ece = compute_expected_calibration_error(y_true_arr, y_prob, classes_list)
        metrics["expected_calibration_error"] = round(ece, 4)

    return metrics
