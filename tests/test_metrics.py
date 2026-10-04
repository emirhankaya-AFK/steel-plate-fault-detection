"""Tests for multiclass classification metrics, Brier score, and calibration error."""

import numpy as np
import pytest

from steel_faults.evaluation.metrics import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    evaluate_model_performance,
)


def test_brier_score_handcrafted():
    """Verify Brier score on hand-crafted known values."""
    classes = ["A", "B"]
    y_true = np.array(["A", "B"])

    # Perfect prediction: [1.0, 0.0] and [0.0, 1.0] -> Brier = 0.0
    perfect_probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert compute_multiclass_brier_score(y_true, perfect_probs, classes) == pytest.approx(0.0)

    # 50/50 prediction -> for each sample: (1-0.5)^2 + (0-0.5)^2 = 0.50 -> Mean = 0.50
    half_probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert compute_multiclass_brier_score(y_true, half_probs, classes) == pytest.approx(0.50)

    # Completely wrong prediction -> for each sample: (1-0)^2 + (0-1)^2 = 2.0 -> Mean = 2.0
    wrong_probs = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert compute_multiclass_brier_score(y_true, wrong_probs, classes) == pytest.approx(2.0)


def test_expected_calibration_error_handcrafted():
    """Verify Expected Calibration Error calculation."""
    classes = ["A", "B"]

    # 1. Perfectly calibrated predictions: confidence 0.8 on 5 samples, exactly 4 correct (80% acc)
    # ECE should be ~0.0
    y_true = np.array(["A", "A", "A", "A", "B"])
    probs = np.array(
        [
            [0.8, 0.2],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.8, 0.2],  # Confident A (0.8), but actually B -> 4/5 correct = 0.8
        ]
    )
    ece = compute_expected_calibration_error(y_true, probs, classes, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-5)

    # 2. Overconfident completely wrong: confidence 1.0, accuracy 0.0 -> ECE = 1.0
    y_wrong = np.array(["B", "B"])
    probs_wrong = np.array([[1.0, 0.0], [1.0, 0.0]])
    ece_wrong = compute_expected_calibration_error(y_wrong, probs_wrong, classes, n_bins=5)
    assert ece_wrong == pytest.approx(1.0, abs=1e-5)


def test_evaluate_model_performance_structure():
    """Verify that evaluate_model_performance outputs all required metrics keys."""
    classes = ["Class1", "Class2", "Class3"]
    y_true = np.array(["Class1", "Class2", "Class3", "Class1"])
    y_pred = np.array(["Class1", "Class2", "Class1", "Class1"])
    y_prob = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.5, 0.2, 0.3],
            [0.9, 0.05, 0.05],
        ]
    )

    results = evaluate_model_performance(y_true, y_pred, y_prob, classes)

    assert "macro_f1" in results
    assert "weighted_f1" in results
    assert "balanced_accuracy" in results
    assert "accuracy" in results
    assert "per_class" in results
    assert "confusion_matrix" in results
    assert "log_loss" in results
    assert "brier_score" in results
    assert "expected_calibration_error" in results

    assert len(results["per_class"]) == 3
    for cls in classes:
        assert "precision" in results["per_class"][cls]
        assert "recall" in results["per_class"][cls]
        assert "f1_score" in results["per_class"][cls]
        assert "support" in results["per_class"][cls]
