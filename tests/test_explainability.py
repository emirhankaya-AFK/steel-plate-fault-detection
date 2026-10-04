"""Tests for model explainability and feature importance extraction."""

import pytest

from steel_faults.data.schema import FEATURE_NAMES
from steel_faults.evaluation.explainability import (
    compute_permutation_importance,
    extract_native_feature_importance,
)


def test_permutation_importance_structure(trained_strong_model, split_data):
    """Verify permutation importance returns ranked features for all 27 inputs."""
    importances = compute_permutation_importance(
        trained_strong_model,
        split_data.X_val,
        split_data.y_val,
        scoring="f1_macro",
        n_repeats=2,
        random_state=42,
    )

    assert len(importances) == len(FEATURE_NAMES)
    for item in importances:
        assert "feature" in item
        assert "importance_mean" in item
        assert "importance_std" in item
        assert item["feature"] in FEATURE_NAMES

    # Ensure sorted descending
    means = [item["importance_mean"] for item in importances]
    assert means == sorted(means, reverse=True)


def test_native_feature_importance(trained_strong_model):
    """Verify tree-native Gini feature importances extract and sum to ~1.0."""
    native = extract_native_feature_importance(trained_strong_model, list(FEATURE_NAMES))
    assert native is not None
    assert len(native) == len(FEATURE_NAMES)

    total_importance = sum(item["importance"] for item in native)
    assert total_importance == pytest.approx(1.0, abs=0.01)
