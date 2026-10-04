"""Tests ensuring strict data leakage prevention and preprocessor integrity."""

import numpy as np
import pytest

from steel_faults.data.preprocessing import SteelPlatePreprocessor
from steel_faults.data.schema import FEATURE_NAMES


def test_preprocessor_no_leakage_on_test_transform(split_data):
    """Ensure transforming test set does not alter training-fitted scaling parameters."""
    preprocessor = SteelPlatePreprocessor(with_scaling=True)
    preprocessor.fit(split_data.X_train)

    train_mean_before = np.copy(preprocessor.scaler.mean_)
    train_var_before = np.copy(preprocessor.scaler.var_)

    # Transform test set
    transformed_test = preprocessor.transform(split_data.X_test)

    train_mean_after = preprocessor.scaler.mean_
    train_var_after = preprocessor.scaler.var_

    # Verify parameters are strictly identical (zero leakage from test set)
    np.testing.assert_array_equal(train_mean_before, train_mean_after)
    np.testing.assert_array_equal(train_var_before, train_var_after)
    assert transformed_test.shape == split_data.X_test.shape


def test_preprocessor_extreme_test_outliers_do_not_leak(split_data):
    """Ensure that extreme test values do not corrupt the trained model scaler."""
    preprocessor = SteelPlatePreprocessor(with_scaling=True)
    preprocessor.fit(split_data.X_train)

    train_mean = np.copy(preprocessor.scaler.mean_)

    # Create synthetic corrupted test sample with huge outliers
    corrupted_test = split_data.X_test.copy()
    corrupted_test.iloc[0, 0] = 999999999.0
    corrupted_test.iloc[0, 1] = 999999999.0

    _ = preprocessor.transform(corrupted_test)

    # Scaler mean must remain unchanged
    np.testing.assert_array_equal(preprocessor.scaler.mean_, train_mean)


def test_transform_unfitted_raises_runtime_error(split_data):
    """Test that calling transform prior to fit raises RuntimeError."""
    preprocessor = SteelPlatePreprocessor(with_scaling=True)
    with pytest.raises(RuntimeError, match="must be fitted before transform"):
        preprocessor.transform(split_data.X_test)


def test_preprocessor_missing_feature_raises_value_error(split_data):
    """Test that missing required feature in input raises ValueError."""
    preprocessor = SteelPlatePreprocessor(with_scaling=True)
    preprocessor.fit(split_data.X_train)

    incomplete_df = split_data.X_test.drop(columns=[FEATURE_NAMES[0]])
    with pytest.raises(ValueError, match="missing required features"):
        preprocessor.transform(incomplete_df)


def test_preprocessor_preserves_feature_names_and_ordering(split_data):
    """Test that preprocessor outputs DataFrame with identical feature column names and order."""
    preprocessor = SteelPlatePreprocessor(with_scaling=True)
    transformed_train = preprocessor.fit_transform(split_data.X_train)

    assert list(transformed_train.columns) == list(FEATURE_NAMES)
    assert transformed_train.index.equals(split_data.X_train.index)
