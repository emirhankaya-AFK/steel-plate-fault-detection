"""Tests for baseline and strong classification models."""

import numpy as np

from steel_faults.models.baseline import build_baseline_model
from steel_faults.models.strong_model import (
    build_hist_gradient_boosting_model,
    build_random_forest_model,
)


def test_baseline_logistic_regression(split_data):
    """Test training and inference with baseline Logistic Regression model."""
    model = build_baseline_model(max_iter=500, random_state=42)
    model.fit(split_data.X_train, split_data.y_train)

    preds = model.predict(split_data.X_test)
    probs = model.predict_proba(split_data.X_test)

    assert len(preds) == len(split_data.X_test)
    assert probs.shape == (len(split_data.X_test), len(split_data.classes))
    assert model.named_steps["classifier"].class_weight == "balanced"


def test_random_forest_strong_model(split_data):
    """Test training and inference with Random Forest model."""
    model = build_random_forest_model(n_estimators=30, random_state=42, n_jobs=1)
    model.fit(split_data.X_train, split_data.y_train)

    preds = model.predict(split_data.X_test)
    probs = model.predict_proba(split_data.X_test)

    assert len(preds) == len(split_data.X_test)
    assert probs.shape == (len(split_data.X_test), len(split_data.classes))
    assert model.named_steps["classifier"].class_weight == "balanced"


def test_hist_gradient_boosting_model(split_data):
    """Test training and inference with HistGradientBoosting model."""
    model = build_hist_gradient_boosting_model(max_iter=30, random_state=42)
    model.fit(split_data.X_train, split_data.y_train)

    preds = model.predict(split_data.X_test)
    probs = model.predict_proba(split_data.X_test)

    assert len(preds) == len(split_data.X_test)
    assert probs.shape == (len(split_data.X_test), len(split_data.classes))


def test_model_reproducibility(split_data):
    """Verify deterministic training outputs under fixed random state."""
    m1 = build_random_forest_model(n_estimators=30, random_state=999, n_jobs=1)
    m1.fit(split_data.X_train, split_data.y_train)
    p1 = m1.predict_proba(split_data.X_test)

    m2 = build_random_forest_model(n_estimators=30, random_state=999, n_jobs=1)
    m2.fit(split_data.X_train, split_data.y_train)
    p2 = m2.predict_proba(split_data.X_test)

    np.testing.assert_array_equal(p1, p2)
