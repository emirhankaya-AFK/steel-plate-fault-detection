"""Leakage-safe preprocessing and scaling pipeline for steel plate features."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from steel_faults.data.schema import FEATURE_NAMES


class SteelPlatePreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor that fits scaling parameters strictly on training data."""

    def __init__(self, with_scaling: bool = True) -> None:
        self.with_scaling = with_scaling
        self.scaler: StandardScaler | None = StandardScaler() if with_scaling else None
        self.feature_names_: list[str] = list(FEATURE_NAMES)
        self.is_fitted_: bool = False

    def _validate_input(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Validate feature column order and absence of nulls."""
        if isinstance(X, np.ndarray):
            if X.shape[1] != len(self.feature_names_):
                raise ValueError(
                    f"Expected array with {len(self.feature_names_)} columns, got {X.shape[1]}"
                )
            df = pd.DataFrame(X, columns=self.feature_names_)
        elif isinstance(X, pd.DataFrame):
            missing = [col for col in self.feature_names_ if col not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame is missing required features: {missing}")
            # Ensure correct column order
            df = X[self.feature_names_].copy()
        else:
            raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")

        if df.isna().any().any():
            raise ValueError("Input features contain missing values (NaN)")

        return df

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> Self:
        """Fit preprocessor parameters strictly on training features."""
        df = self._validate_input(X)
        if self.scaler is not None:
            self.scaler.fit(df)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Apply fitted preprocessing to features without updating internal statistics."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform can be called.")
        df = self._validate_input(X)
        if self.scaler is not None:
            scaled = self.scaler.transform(df)
            return pd.DataFrame(scaled, columns=self.feature_names_, index=df.index)
        return df

    def fit_transform(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None
    ) -> pd.DataFrame:
        """Fit preprocessor on X and return transformed features."""
        return self.fit(X, y).transform(X)
