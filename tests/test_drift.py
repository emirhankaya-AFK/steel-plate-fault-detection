"""Tests for distribution drift monitoring (PSI and KS test)."""

import numpy as np

from steel_faults.monitoring.drift import (
    DriftDetector,
    calculate_categorical_psi,
    calculate_psi,
)


def test_psi_identical_distributions():
    """Verify PSI is near zero for samples drawn from the same distribution."""
    rng = np.random.RandomState(42)
    s1 = rng.normal(10.0, 2.0, size=1000)
    s2 = rng.normal(10.0, 2.0, size=1000)

    psi = calculate_psi(s1, s2)
    assert psi < 0.05, f"Expected small PSI for identical distribution, got {psi}"


def test_psi_drifted_distribution():
    """Verify PSI is high (>= 0.20) for heavily shifted distribution."""
    rng = np.random.RandomState(42)
    ref = rng.normal(10.0, 2.0, size=1000)
    shifted = rng.normal(18.0, 2.0, size=1000)

    psi = calculate_psi(ref, shifted)
    assert psi >= 0.20, f"Expected large PSI for shifted distribution, got {psi}"


def test_categorical_psi():
    """Verify categorical PSI on class counts."""
    classes = ["A", "B", "C"]
    ref_counts = {"A": 100, "B": 100, "C": 100}
    same_counts = {"A": 98, "B": 102, "C": 100}
    drift_counts = {"A": 250, "B": 20, "C": 30}

    psi_same = calculate_categorical_psi(ref_counts, same_counts, classes)
    assert psi_same < 0.05

    psi_drift = calculate_categorical_psi(ref_counts, drift_counts, classes)
    assert psi_drift > 0.20


def test_drift_detector_end_to_end(split_data):
    """Verify DriftDetector on undisturbed test features vs shifted features."""
    detector = DriftDetector(reference_features=split_data.X_train, classes=split_data.classes)

    # 1. Clean test features should have minimal drift
    reports_clean, overall_drift_clean = detector.evaluate_feature_drift(split_data.X_test)
    assert len(reports_clean) == 27
    assert overall_drift_clean is False

    # 2. Artificially shifted features
    shifted_test = split_data.X_test.copy()
    for col in shifted_test.columns:
        shifted_test[col] = shifted_test[col] * 5.0 + 1000.0

    reports_shifted, overall_drift_shifted = detector.evaluate_feature_drift(shifted_test)
    assert overall_drift_shifted is True
    drifted_count = sum(1 for r in reports_shifted if r.is_drift)
    assert drifted_count >= 5
