"""Tests for the deterministic offline synthetic fixture."""

from pathlib import Path

import pandas as pd

from steel_faults.data.schema import FEATURE_NAMES, TARGET_NAMES


def test_synthetic_fixture_integrity():
    """Verify that offline synthetic fixture exists and conforms to dataset schema."""
    fixture_path = Path("tests/fixtures/synthetic_steel_faults.csv")
    assert fixture_path.exists(), f"Synthetic fixture missing at {fixture_path}"

    df = pd.read_csv(fixture_path)
    assert len(df) >= 100, f"Expected >= 100 samples, got {len(df)}"

    # Check columns
    expected_cols = list(FEATURE_NAMES) + list(TARGET_NAMES)
    assert list(df.columns) == expected_cols

    # Check missing values
    assert df.isna().sum().sum() == 0

    # Check targets: each row must have exactly 1 active defect label
    target_sums = df[list(TARGET_NAMES)].sum(axis=1)
    assert (target_sums == 1).all()

    # Check each class has balanced representation
    for t in TARGET_NAMES:
        assert df[t].sum() >= 10, f"Class {t} has insufficient support in fixture"
