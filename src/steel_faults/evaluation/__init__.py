"""Evaluation package providing metrics calculation and model explainability."""

from steel_faults.evaluation.explainability import (
    compute_permutation_importance,
    extract_native_feature_importance,
)
from steel_faults.evaluation.metrics import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    evaluate_model_performance,
)

__all__ = [
    "evaluate_model_performance",
    "compute_multiclass_brier_score",
    "compute_expected_calibration_error",
    "compute_permutation_importance",
    "extract_native_feature_importance",
]
