"""Pytest fixtures for offline and deterministic test execution."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from steel_faults.api.app import create_app
from steel_faults.data.splitter import SplitData, StratifiedSplitter
from steel_faults.data.synthetic import generate_synthetic_steel_data
from steel_faults.models.baseline import build_baseline_model
from steel_faults.models.calibration import calibrate_on_validation
from steel_faults.models.strong_model import build_random_forest_model


@pytest.fixture(scope="session")
def synthetic_dataframe() -> pd.DataFrame:
    """Session-scoped synthetic steel faults dataset conforming to UCI schema."""
    fixture_path = Path("tests/fixtures/synthetic_steel_faults.csv")
    if fixture_path.exists():
        return pd.read_csv(fixture_path)
    return generate_synthetic_steel_data(n_samples=210, random_seed=42)


@pytest.fixture(scope="session")
def split_data(synthetic_dataframe: pd.DataFrame) -> SplitData:
    """Stratified train, val, test splits from synthetic data."""
    splitter = StratifiedSplitter(test_size=0.20, val_size=0.20, random_seed=42)
    return splitter.split(synthetic_dataframe)


@pytest.fixture(scope="session")
def trained_baseline(split_data: SplitData):
    """Fitted baseline logistic regression model."""
    model = build_baseline_model(max_iter=500, random_state=42)
    model.fit(split_data.X_train, split_data.y_train)
    return model


@pytest.fixture(scope="session")
def trained_strong_model(split_data: SplitData):
    """Fitted random forest model."""
    model = build_random_forest_model(n_estimators=50, random_state=42, n_jobs=1)
    model.fit(split_data.X_train, split_data.y_train)
    return model


@pytest.fixture(scope="session")
def calibrated_model(trained_strong_model, split_data: SplitData):
    """Calibrated random forest model fitted on validation data."""
    return calibrate_on_validation(
        trained_strong_model,
        split_data.X_val,
        split_data.y_val,
        method="sigmoid",
    )


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """FastAPI TestClient instance with fallback in-memory model."""
    app_instance = create_app(
        model_bundle_path="nonexistent_bundle_to_trigger_fallback.joblib",
        reference_path="nonexistent_ref_to_trigger_fallback.joblib",
    )
    with TestClient(app_instance) as client:
        yield client
