"""Smoke test script verifying all FastAPI endpoints."""

import sys

from fastapi.testclient import TestClient

from steel_faults.api.app import app
from steel_faults.data.schema import FEATURE_NAMES
from steel_faults.data.synthetic import generate_synthetic_steel_data


def run_smoke_test() -> None:
    with TestClient(app) as client:
        print("1. Testing GET /health...")
        health_resp = client.get("/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["model_loaded"] is True
        assert len(health_data["classes"]) == 7
        print("   -> /health OK: status=healthy, classes=", health_data["classes"])

        print("\n2. Testing POST /predict...")
        syn_df = generate_synthetic_steel_data(n_samples=20, random_seed=99)
        sample_dict = syn_df[list(FEATURE_NAMES)].iloc[0].to_dict()

        pred_resp = client.post("/predict", json=sample_dict)
        assert pred_resp.status_code == 200, f"Predict failed: {pred_resp.text}"
        pred_data = pred_resp.json()
        assert "predicted_class" in pred_data
        assert "confidence" in pred_data
        assert "review_recommended" in pred_data
        assert 0.0 <= pred_data["confidence"] <= 1.0
        print(
            f"   -> /predict OK: class={pred_data['predicted_class']}, "
            f"conf={pred_data['confidence']:.3f}, review={pred_data['review_recommended']}"
        )

        print("\n3. Testing POST /predict validation error handling (422)...")
        invalid_sample = sample_dict.copy()
        invalid_sample["X_Minimum"] = 1000
        invalid_sample["X_Maximum"] = 100  # Invalid: X_Min > X_Max
        bad_resp = client.post("/predict", json=invalid_sample)
        assert bad_resp.status_code == 422, (
            f"Expected 422 for invalid coordinate, got {bad_resp.status_code}"
        )
        print("   -> Validation error handling OK: returned 422 as expected")

        print("\n4. Testing POST /batch-predict...")
        batch_records = syn_df[list(FEATURE_NAMES)].iloc[:5].to_dict(orient="records")
        batch_payload = {"samples": batch_records, "abstention_threshold": 0.60}
        batch_resp = client.post("/batch-predict", json=batch_payload)
        assert batch_resp.status_code == 200, f"Batch predict failed: {batch_resp.text}"
        batch_data = batch_resp.json()
        assert batch_data["total_samples"] == 5
        assert len(batch_data["predictions"]) == 5
        assert 0 <= batch_data["review_count"] <= 5
        assert 0.0 <= batch_data["review_rate"] <= 1.0
        print(
            f"   -> /batch-predict OK: total={batch_data['total_samples']}, "
            f"reviews={batch_data['review_count']}, rate={batch_data['review_rate']}"
        )

        print("\n5. Testing POST /drift...")
        drift_records = syn_df[list(FEATURE_NAMES)].iloc[:15].to_dict(orient="records")
        drift_payload = {"production_samples": drift_records}
        drift_resp = client.post("/drift", json=drift_payload)
        assert drift_resp.status_code == 200, f"Drift check failed: {drift_resp.text}"
        drift_data = drift_resp.json()
        assert drift_data["features_evaluated"] == 27
        assert "overall_drift_detected" in drift_data
        assert "feature_reports" in drift_data
        print(
            f"   -> /drift OK: evaluated={drift_data['features_evaluated']} features, "
            f"drifted={drift_data['features_drifted_count']}, overall={drift_data['overall_drift_detected']}"
        )

        print("\n==========================================")
        print("ALL API SMOKE TESTS PASSED SUCCESSFULLY!")
        print("==========================================")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)
