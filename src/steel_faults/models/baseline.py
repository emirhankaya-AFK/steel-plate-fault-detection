"""Baseline classification model for steel plate faults."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from steel_faults.data.preprocessing import SteelPlatePreprocessor


def build_baseline_model(
    max_iter: int = 2000,
    class_weight: str = "balanced",
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Build a standardized Logistic Regression baseline with balanced class weights.

    Args:
        max_iter: Maximum solver iterations.
        class_weight: Imbalance mitigation strategy ('balanced' computes weights inversely
                      proportional to class frequencies).
        C: Inverse regularization strength.
        random_state: Deterministic random seed.

    Returns:
        Pipeline: Preprocessor (StandardScaler) + LogisticRegression classifier.
    """
    return Pipeline(
        [
            ("preprocessor", SteelPlatePreprocessor(with_scaling=True)),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
