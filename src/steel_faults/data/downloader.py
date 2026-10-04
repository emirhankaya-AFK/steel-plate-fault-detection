"""Downloader module for the UCI Steel Plates Faults dataset."""

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from steel_faults.data.schema import FEATURE_NAMES, TARGET_NAMES

logger = logging.getLogger(__name__)

DEFAULT_UCI_URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"


class DatasetDownloader:
    """Handles downloading and caching the UCI Steel Plates Faults dataset."""

    def __init__(
        self,
        url: str = DEFAULT_UCI_URL,
        cache_dir: str | Path = "data/raw",
        timeout: int = 45,
    ) -> None:
        self.url = url
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout

    def download_and_extract(self, force_download: bool = False) -> pd.DataFrame:
        """Download and parse the UCI Steel Plates dataset.

        Returns:
            pd.DataFrame: Complete dataset with 27 features and 7 defect targets.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        var_file = self.cache_dir / "Faults27x7_var"
        data_file = self.cache_dir / "Faults.NNA"

        if not force_download and var_file.exists() and data_file.exists():
            logger.info("Found cached dataset files in %s", self.cache_dir)
            return self._parse_from_disk(var_file, data_file)

        logger.info("Downloading UCI Steel Plates Faults dataset from %s", self.url)
        try:
            response = requests.get(self.url, timeout=self.timeout)
            response.raise_for_status()
        except Exception as exc:
            logger.error("Failed to download dataset from %s: %s", self.url, exc)
            raise RuntimeError(
                f"Failed to download UCI Steel Plates Faults dataset: {exc}"
            ) from exc

        zip_bytes = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_bytes) as z:
            z.extractall(self.cache_dir)

        return self._parse_from_disk(var_file, data_file)

    def _parse_from_disk(self, var_file: Path, data_file: Path) -> pd.DataFrame:
        """Parse the column header file and NNA tabular data from disk."""
        if not var_file.exists():
            raise FileNotFoundError(f"Missing column specification file: {var_file}")
        if not data_file.exists():
            raise FileNotFoundError(f"Missing data file: {data_file}")

        with open(var_file, encoding="utf-8", errors="ignore") as f:
            column_names = [line.strip() for line in f if line.strip()]

        expected_total_cols = len(FEATURE_NAMES) + len(TARGET_NAMES)
        if len(column_names) != expected_total_cols:
            raise ValueError(
                f"Expected {expected_total_cols} columns in {var_file}, found {len(column_names)}"
            )

        df = pd.read_csv(data_file, sep=r"\s+", header=None, names=column_names)
        logger.info("Successfully loaded dataset with shape %s", df.shape)

        self._validate_dataset(df)
        return df

    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate structure and label integrity of the loaded dataset."""
        expected_cols = list(FEATURE_NAMES) + list(TARGET_NAMES)
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        # Check for missing values
        if df.isna().any().any():
            nan_counts = df.isna().sum()
            raise ValueError(
                f"Dataset contains unexpected NaN values:\n{nan_counts[nan_counts > 0]}"
            )

        # In the official UCI dataset, each row has exactly 1 defect marked
        target_sums = df[list(TARGET_NAMES)].sum(axis=1)
        if not (target_sums == 1).all():
            non_single_count = (target_sums != 1).sum()
            logger.warning(
                "%d rows do not have exactly 1 active defect label. Normalizing to single label.",
                non_single_count,
            )
