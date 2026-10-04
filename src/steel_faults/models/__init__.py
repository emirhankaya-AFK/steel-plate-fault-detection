"""Model package exposing baseline, strong classifiers, calibration, and abstention."""

from steel_faults.models.abstention import AbstentionPolicy, SelectiveMetricPoint
from steel_faults.models.baseline import build_baseline_model
from steel_faults.models.calibration import CalibratedModelWrapper, calibrate_on_validation
from steel_faults.models.strong_model import (
    build_hist_gradient_boosting_model,
    build_random_forest_model,
    build_strong_model,
)

__all__ = [
    "build_baseline_model",
    "build_strong_model",
    "build_random_forest_model",
    "build_hist_gradient_boosting_model",
    "CalibratedModelWrapper",
    "calibrate_on_validation",
    "AbstentionPolicy",
    "SelectiveMetricPoint",
]
