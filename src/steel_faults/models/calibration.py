"""Probability calibration module fitted strictly on validation split."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import sklearn.calibration
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class CalibratedModelWrapper:
    """Wraps an estimator and calibrates its output probabilities using validation data."""

    def __init__(self, base_estimator: Pipeline | Any, method: str = "sigmoid") -> None:
        self.base_estimator = base_estimator
        self.method = method
        self.calibrator: CalibratedClassifierCV | None = None
        self.is_calibrated: bool = False

    @property
    def classes_(self) -> np.ndarray:
        """Return the target class labels."""
        if self.calibrator is not None:
            return self.calibrator.classes_
        if hasattr(self.base_estimator, "classes_"):
            return self.base_estimator.classes_
        if hasattr(self.base_estimator, "named_steps"):
            return self.base_estimator.named_steps["classifier"].classes_
        raise AttributeError("Underlying estimator does not have classes_ attribute")

    def fit_calibration(
        self, X_val: pd.DataFrame | np.ndarray, y_val: pd.Series | np.ndarray
    ) -> "CalibratedModelWrapper":
        """Calibrate probabilities strictly on validation split."""
        logger.info("Fitting probability calibrator (%s) on validation data", self.method)

        # Handle scikit-learn >= 1.4 FrozenEstimator vs legacy cv='prefit'
        if hasattr(sklearn.calibration, "FrozenEstimator"):
            frozen = sklearn.calibration.FrozenEstimator(self.base_estimator)
            self.calibrator = CalibratedClassifierCV(frozen, method=self.method)
            self.calibrator.fit(X_val, y_val)
        else:
            self.calibrator = CalibratedClassifierCV(
                self.base_estimator, method=self.method, cv="prefit"
            )
            self.calibrator.fit(X_val, y_val)

        self.is_calibrated = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.is_calibrated and self.calibrator is not None:
            return self.calibrator.predict(X)
        return self.base_estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict calibrated or uncalibrated class probabilities."""
        if self.is_calibrated and self.calibrator is not None:
            return self.calibrator.predict_proba(X)
        return self.base_estimator.predict_proba(X)


def calibrate_on_validation(
    estimator: Pipeline | Any,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    method: str = "sigmoid",
) -> CalibratedModelWrapper:
    """Convenience helper to create and fit a CalibratedModelWrapper on validation data."""
    wrapper = CalibratedModelWrapper(estimator, method=method)
    return wrapper.fit_calibration(X_val, y_val)
