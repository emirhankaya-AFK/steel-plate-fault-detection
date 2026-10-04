"""Tests for probability calibration and abstention selective classification."""

import json
from pathlib import Path

import numpy as np
import pytest

from steel_faults.evaluation.metrics import evaluate_model_performance
from steel_faults.models.abstention import AbstentionPolicy
from steel_faults.models.calibration import calibrate_on_validation


def test_calibration_produces_valid_probabilities(trained_strong_model, split_data):
    """Verify that calibration on validation set outputs valid probabilities."""
    calibrated = calibrate_on_validation(
        trained_strong_model,
        split_data.X_val,
        split_data.y_val,
        method="sigmoid",
    )
    probs = calibrated.predict_proba(split_data.X_test)

    assert probs.shape == (len(split_data.X_test), len(split_data.classes))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5)


def test_abstention_policy_single_sample_evaluation():
    """Verify single-sample abstention review flagging."""
    policy = AbstentionPolicy(confidence_threshold=0.70)
    classes = ["Pastry", "Z_Scratch", "Bumps"]

    # Sample 1: Low confidence (max 0.45 < 0.70) -> flag review
    p_low = np.array([0.45, 0.35, 0.20])
    pred_cls, pred_idx, conf, review = policy.evaluate_sample(p_low, classes)
    assert pred_cls == "Pastry"
    assert pred_idx == 0
    assert conf == pytest.approx(0.45)
    assert review is True

    # Sample 2: High confidence (max 0.85 >= 0.70) -> no review needed
    p_high = np.array([0.05, 0.85, 0.10])
    pred_cls2, pred_idx2, conf2, review2 = policy.evaluate_sample(p_high, classes)
    assert pred_cls2 == "Z_Scratch"
    assert pred_idx2 == 1
    assert conf2 == pytest.approx(0.85)
    assert review2 is False


def test_selective_classification_coverage_monotonicity():
    """Verify that coverage decreases monotonically as confidence threshold increases."""
    classes = ["A", "B", "C"]
    y_true = np.array(["A", "B", "C", "A", "B", "A", "C", "B"])
    probs = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.10, 0.80, 0.10],
            [0.20, 0.30, 0.50],
            [0.40, 0.35, 0.25],
            [0.05, 0.90, 0.05],
            [0.60, 0.20, 0.20],
            [0.10, 0.20, 0.70],
            [0.30, 0.40, 0.30],
        ]
    )

    thresholds = [0.0, 0.40, 0.60, 0.80, 0.90]
    points = AbstentionPolicy.compute_selective_curve(y_true, probs, classes, thresholds)

    coverages = [p.coverage for p in points]
    # Verify non-increasing coverage
    for i in range(len(coverages) - 1):
        assert coverages[i] >= coverages[i + 1], f"Coverage violated monotonicity: {coverages}"


def test_invalid_abstention_threshold():
    """Verify that invalid threshold values raise ValueError."""
    with pytest.raises(ValueError, match="confidence_threshold must be in"):
        AbstentionPolicy(confidence_threshold=1.5)
    with pytest.raises(ValueError, match="confidence_threshold must be in"):
        AbstentionPolicy(confidence_threshold=-0.1)


def test_zero_threshold_selective_metrics_equal_deployed_calibrated_metrics(
    calibrated_model, split_data
):
    """Regression test: verify zero-threshold selective metrics equal deployed calibrated metrics."""
    cal_probs = calibrated_model.predict_proba(split_data.X_test)
    cal_preds = calibrated_model.predict(split_data.X_test)
    cal_metrics = evaluate_model_performance(
        split_data.y_test, cal_preds, cal_probs, split_data.classes
    )

    policy = AbstentionPolicy(confidence_threshold=0.50)
    selective_points = policy.compute_selective_curve(
        split_data.y_test, cal_probs, split_data.classes, thresholds=[0.0]
    )
    p0 = selective_points[0]

    assert p0.threshold == 0.0
    assert p0.coverage == 1.0
    assert p0.covered_samples == len(split_data.X_test)
    assert p0.abstained_samples == 0
    assert p0.selective_accuracy == pytest.approx(cal_metrics["accuracy"], abs=1e-4)
    assert p0.selective_macro_f1 == pytest.approx(cal_metrics["macro_f1"], abs=1e-4)


def test_regression_metrics_json_zero_threshold_matches_calibrated_not_uncalibrated():
    """Regression test: verify metrics.json selective curve uses calibrated RF metrics."""
    metrics_path = Path("metrics/metrics.json")
    if not metrics_path.exists():
        pytest.skip("metrics/metrics.json not found")

    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)

    cal_metrics = data["test_metrics"]["random_forest_calibrated"]
    uncal_metrics = data["test_metrics"]["random_forest_uncalibrated"]
    selective_curve = data["abstention_selective_analysis"]["selective_curve"]
    p0 = selective_curve[0]

    assert p0["threshold"] == 0.0
    assert p0["coverage"] == 1.0
    assert p0["abstained_samples"] == 0
    assert p0["covered_samples"] == data["dataset_summary"]["test_samples"]

    # Must strictly equal deployed calibrated metrics
    assert p0["selective_accuracy"] == pytest.approx(cal_metrics["accuracy"], abs=1e-4)
    assert p0["selective_macro_f1"] == pytest.approx(cal_metrics["macro_f1"], abs=1e-4)

    # Must NOT match uncalibrated predictions (regression against audit issue)
    assert p0["selective_macro_f1"] != uncal_metrics["macro_f1"]
