"""Tests for multi-label binary to multiclass categorical target mapping."""

import pandas as pd
import pytest

from steel_faults.data.schema import TARGET_NAMES
from steel_faults.data.splitter import StratifiedSplitter


def test_extract_labels_valid():
    """Verify that one-hot target columns are mapped to exact categorical defect names."""
    df = pd.DataFrame(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        columns=list(TARGET_NAMES),
    )
    splitter = StratifiedSplitter()
    labels = splitter.extract_labels(df)

    assert list(labels) == ["Pastry", "Z_Scratch", "Stains", "Other_Faults"]
    assert labels.name == "defect_label"


def test_extract_labels_empty_row_raises():
    """Verify that a row with all zeros raises ValueError."""
    df = pd.DataFrame(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],  # No label
        ],
        columns=list(TARGET_NAMES),
    )
    splitter = StratifiedSplitter()
    with pytest.raises(ValueError, match="Found 1 rows with no defect label"):
        splitter.extract_labels(df)


def test_extract_labels_missing_target_col_raises():
    """Verify that missing one of the 7 target columns raises ValueError."""
    df = pd.DataFrame([[1, 0, 0, 0, 0, 0]], columns=list(TARGET_NAMES)[:-1])
    splitter = StratifiedSplitter()
    with pytest.raises(ValueError, match="Missing target columns"):
        splitter.extract_labels(df)
