"""Stratified dataset splitting with deterministic seeds."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from steel_faults.data.schema import FEATURE_NAMES, TARGET_NAMES


@dataclass
class SplitData:
    """Container holding partitioned train, validation, and test datasets."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    classes: list[str]


class StratifiedSplitter:
    """Performs leakage-safe, reproducible stratified train/validation/test splits."""

    def __init__(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        if not (0.0 < val_size < 1.0):
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
        if test_size + val_size >= 1.0:
            raise ValueError(
                f"Sum of test_size and val_size must be < 1.0, got {test_size + val_size}"
            )

        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed

    def extract_labels(self, df: pd.DataFrame) -> pd.Series:
        """Convert 7 binary defect indicator columns into a single categorical label."""
        missing = [t for t in TARGET_NAMES if t not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns for label extraction: {missing}")

        target_df = df[list(TARGET_NAMES)]
        # Validate that each row has at least one positive defect label
        row_sums = target_df.sum(axis=1)
        if (row_sums == 0).any():
            zero_count = (row_sums == 0).sum()
            raise ValueError(f"Found {zero_count} rows with no defect label assigned")

        labels = target_df.idxmax(axis=1)
        labels.name = "defect_label"
        return labels

    def split(self, df: pd.DataFrame) -> SplitData:
        """Partition dataframe into stratified train, validation, and test sets.

        Returns:
            SplitData: Dataclass containing split features and series labels.
        """
        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        X = df[list(FEATURE_NAMES)].copy()
        y = self.extract_labels(df)

        # First split: separate untouched final test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y,
        )

        # Second split: separate train and validation sets
        # val_size relative to remaining train_val portion
        relative_val_size = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val_size,
            random_state=self.random_seed,
            stratify=y_train_val,
        )

        # Verify no index overlaps
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(val_idx), "Train and Validation indices overlap!"
        assert train_idx.isdisjoint(test_idx), "Train and Test indices overlap!"
        assert val_idx.isdisjoint(test_idx), "Validation and Test indices overlap!"

        classes = sorted(y.unique())
        return SplitData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            classes=classes,
        )
