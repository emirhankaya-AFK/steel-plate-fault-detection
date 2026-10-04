"""Strong CPU-friendly models for steel plate fault classification."""

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from steel_faults.data.preprocessing import SteelPlatePreprocessor


def build_random_forest_model(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    class_weight: str = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Build a Random Forest ensemble model with balanced class weighting.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples required to split a node.
        class_weight: 'balanced' computes weights inversely proportional to class frequencies.
        random_state: Deterministic random seed.
        n_jobs: Parallel CPU threads (-1 uses all available cores).

    Returns:
        Pipeline: Preprocessor (no scaling needed for trees) + RandomForestClassifier.
    """
    return Pipeline(
        [
            ("preprocessor", SteelPlatePreprocessor(with_scaling=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    class_weight=class_weight,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting_model(
    max_iter: int = 200,
    min_samples_leaf: int = 5,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Pipeline:
    """Build a Histogram-based Gradient Boosting model for fast CPU training.

    Args:
        max_iter: Maximum number of boosting iterations.
        min_samples_leaf: Minimum samples per leaf.
        class_weight: 'balanced' class weights.
        random_state: Deterministic random seed.

    Returns:
        Pipeline: Preprocessor + HistGradientBoostingClassifier.
    """
    return Pipeline(
        [
            ("preprocessor", SteelPlatePreprocessor(with_scaling=False)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_strong_model(
    model_type: str = "random_forest",
    **kwargs: Any,
) -> Pipeline:
    """Factory helper to build selected strong model."""
    if model_type == "random_forest":
        return build_random_forest_model(**kwargs)
    elif model_type in ("hist_gradient_boosting", "hgb"):
        return build_hist_gradient_boosting_model(**kwargs)
    else:
        raise ValueError(f"Unknown strong model type: {model_type}")
