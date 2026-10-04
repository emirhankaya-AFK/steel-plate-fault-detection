"""Explainability module using permutation importance and model-native importances."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from steel_faults.data.schema import FEATURE_NAMES

logger = logging.getLogger(__name__)


def compute_permutation_importance(
    estimator: Any,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    scoring: str = "f1_macro",
    n_repeats: int = 5,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Compute model-agnostic feature importances by evaluating validation performance drops.

    Note:
        Uses scikit-learn's official permutation_importance. We explicitly state this
        methodology and do not make inflated or synthetic SHAP approximation claims.

    Args:
        estimator: Fitted scikit-learn estimator or pipeline.
        X_val: Validation feature set.
        y_val: Validation ground truth targets.
        scoring: Metric to track during permutation (default 'f1_macro').
        n_repeats: Number of times to permute each feature.
        random_state: Seed for reproducible feature permutations.

    Returns:
        list[dict[str, Any]]: Sorted list of feature importance dictionaries.
    """
    logger.info("Computing permutation importance with %d repeats (scoring=%s)", n_repeats, scoring)
    result = permutation_importance(
        estimator,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    feature_names = list(X_val.columns) if isinstance(X_val, pd.DataFrame) else list(FEATURE_NAMES)

    importances: list[dict[str, Any]] = []
    for idx, feature in enumerate(feature_names):
        importances.append(
            {
                "feature": str(feature),
                "importance_mean": round(float(result.importances_mean[idx]), 5),
                "importance_std": round(float(result.importances_std[idx]), 5),
            }
        )

    # Sort descending by mean importance
    importances.sort(key=lambda x: x["importance_mean"], reverse=True)
    return importances


def extract_native_feature_importance(
    estimator: Any,
    feature_names: list[str] | None = None,
) -> list[dict[str, Any]] | None:
    """Extract model-native tree feature importances (Mean Decrease in Impurity / Gini).

    Args:
        estimator: Fitted pipeline or model.
        feature_names: Optional custom list of feature names.

    Returns:
        list[dict[str, Any]] | None: Sorted importance list or None if not supported.
    """
    if feature_names is None:
        feature_names = list(FEATURE_NAMES)

    clf = estimator
    if hasattr(estimator, "named_steps") and "classifier" in estimator.named_steps:
        clf = estimator.named_steps["classifier"]
    elif hasattr(estimator, "base_estimator"):
        clf = estimator.base_estimator
        if hasattr(clf, "named_steps") and "classifier" in clf.named_steps:
            clf = clf.named_steps["classifier"]

    if not hasattr(clf, "feature_importances_"):
        return None

    raw_importances = clf.feature_importances_
    results = [
        {"feature": name, "importance": round(float(imp), 5)}
        for name, imp in zip(feature_names, raw_importances, strict=False)
    ]
    results.sort(key=lambda x: x["importance"], reverse=True)
    return results
