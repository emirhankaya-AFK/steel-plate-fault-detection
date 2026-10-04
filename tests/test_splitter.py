"""Tests for stratified dataset splitting."""

import pytest

from steel_faults.data.splitter import StratifiedSplitter


def test_stratified_splitter_proportions_and_disjoint(synthetic_dataframe):
    """Verify split proportions and strict disjointness of indices."""
    splitter = StratifiedSplitter(test_size=0.15, val_size=0.15, random_seed=42)
    splits = splitter.split(synthetic_dataframe)

    n_total = len(synthetic_dataframe)
    assert len(splits.X_test) == int(round(n_total * 0.15))
    assert len(splits.X_train) + len(splits.X_val) + len(splits.X_test) == n_total

    # Verify complete index disjointness
    train_idx = set(splits.X_train.index)
    val_idx = set(splits.X_val.index)
    test_idx = set(splits.X_test.index)

    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0


def test_stratified_splitter_preserves_classes(synthetic_dataframe):
    """Verify all 7 defect classes are represented across train, val, and test partitions."""
    splitter = StratifiedSplitter(test_size=0.15, val_size=0.15, random_seed=42)
    splits = splitter.split(synthetic_dataframe)

    assert set(splits.y_train.unique()) == set(splits.classes)
    assert set(splits.y_val.unique()) == set(splits.classes)
    assert set(splits.y_test.unique()) == set(splits.classes)


def test_stratified_splitter_deterministic(synthetic_dataframe):
    """Verify that identical random seed produces identical split partitions."""
    s1 = StratifiedSplitter(random_seed=123).split(synthetic_dataframe)
    s2 = StratifiedSplitter(random_seed=123).split(synthetic_dataframe)

    assert s1.X_train.index.equals(s2.X_train.index)
    assert s1.X_test.index.equals(s2.X_test.index)


def test_stratified_splitter_invalid_sizes():
    """Verify that invalid partition sizes raise ValueError."""
    with pytest.raises(ValueError, match="Sum of test_size and val_size must be < 1.0"):
        StratifiedSplitter(test_size=0.6, val_size=0.5)
