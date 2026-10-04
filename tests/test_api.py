"""Tests for FastAPI endpoints."""

from steel_faults.data.schema import FEATURE_NAMES


def get_sample_record(synthetic_dataframe):
    """Helper returning a single feature dict from synthetic data."""
    return synthetic_dataframe[list(FEATURE_NAMES)].iloc[0].to_dict()


def test_api_health_endpoint(test_client):
    """Test /health endpoint."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert len(data["classes"]) == 7


def test_api_predict_endpoint(test_client, synthetic_dataframe):
    """Test /predict endpoint with valid features."""
    sample = get_sample_record(synthetic_dataframe)
    resp = test_client.post("/predict", json=sample)
    assert resp.status_code == 200
    data = resp.json()

    assert "predicted_class" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 7
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "review_recommended" in data
    assert isinstance(data["review_recommended"], bool)


def test_api_predict_invalid_data_returns_422(test_client, synthetic_dataframe):
    """Test that schema violations return HTTP 422."""
    bad_sample = get_sample_record(synthetic_dataframe)
    bad_sample["X_Minimum"] = 1500
    bad_sample["X_Maximum"] = 100  # Invalid
    resp = test_client.post("/predict", json=bad_sample)
    assert resp.status_code == 422


def test_api_predict_missing_feature_returns_422(test_client, synthetic_dataframe):
    """Test that missing required features return HTTP 422."""
    bad_sample = get_sample_record(synthetic_dataframe)
    del bad_sample["X_Minimum"]
    resp = test_client.post("/predict", json=bad_sample)
    assert resp.status_code == 422


def test_api_batch_predict_endpoint(test_client, synthetic_dataframe):
    """Test /batch-predict endpoint."""
    records = synthetic_dataframe[list(FEATURE_NAMES)].iloc[:5].to_dict(orient="records")
    payload = {"samples": records, "abstention_threshold": 0.55}

    resp = test_client.post("/batch-predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_samples"] == 5
    assert len(data["predictions"]) == 5
    assert 0 <= data["review_count"] <= 5
    assert 0.0 <= data["review_rate"] <= 1.0


def test_api_drift_endpoint(test_client, synthetic_dataframe):
    """Test /drift endpoint with production batch."""
    records = synthetic_dataframe[list(FEATURE_NAMES)].iloc[:15].to_dict(orient="records")
    payload = {"production_samples": records}

    resp = test_client.post("/drift", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["features_evaluated"] == 27
    assert "overall_drift_detected" in data
    assert isinstance(data["overall_drift_detected"], bool)
    assert len(data["feature_reports"]) == 27
