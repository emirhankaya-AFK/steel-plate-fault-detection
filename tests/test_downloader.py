"""Tests for dataset downloader and integrity validator."""

from pathlib import Path

import pytest

from steel_faults.data.downloader import DatasetDownloader
from steel_faults.data.schema import FEATURE_NAMES, TARGET_NAMES


def test_downloader_parse_from_disk(tmp_path: Path):
    """Test parsing column names and tab-delimited records from disk."""
    var_file = tmp_path / "Faults27x7_var"
    data_file = tmp_path / "Faults.NNA"

    # Write column names
    all_cols = list(FEATURE_NAMES) + list(TARGET_NAMES)
    var_file.write_text("\n".join(all_cols), encoding="utf-8")

    # Write 2 rows of tab-delimited values (27 features + 7 target columns = 34 values)
    row1 = [10.0] * 27 + [1, 0, 0, 0, 0, 0, 0]
    row2 = [20.0] * 27 + [0, 1, 0, 0, 0, 0, 0]
    data_lines = [
        "\t".join(str(v) for v in row1),
        "\t".join(str(v) for v in row2),
    ]
    data_file.write_text("\n".join(data_lines), encoding="utf-8")

    downloader = DatasetDownloader(cache_dir=tmp_path)
    df = downloader._parse_from_disk(var_file, data_file)

    assert len(df) == 2
    assert list(df.columns) == all_cols
    assert df["Pastry"].iloc[0] == 1
    assert df["Z_Scratch"].iloc[1] == 1


def test_downloader_missing_column_file_raises(tmp_path: Path):
    """Test that missing column file raises FileNotFoundError."""
    downloader = DatasetDownloader(cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        downloader._parse_from_disk(tmp_path / "missing_var", tmp_path / "missing_data")


def test_downloader_column_mismatch_raises(tmp_path: Path):
    """Test that mismatched column count in var file raises ValueError."""
    var_file = tmp_path / "Faults27x7_var"
    data_file = tmp_path / "Faults.NNA"

    # Only write 10 columns instead of 34
    var_file.write_text("\n".join(["col1", "col2"]), encoding="utf-8")
    data_file.write_text("1\t2\n", encoding="utf-8")

    downloader = DatasetDownloader(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Expected 34 columns"):
        downloader._parse_from_disk(var_file, data_file)
