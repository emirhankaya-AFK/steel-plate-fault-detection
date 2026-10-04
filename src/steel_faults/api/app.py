"""FastAPI application for real-time steel fault prediction and drift monitoring."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from steel_faults import __version__
from steel_faults.data.schema import (
    FEATURE_NAMES,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftCheckRequest,
    DriftCheckResponse,
    PredictionResult,
    SteelPlateFeatures,
)
from steel_faults.data.synthetic import generate_synthetic_steel_data
from steel_faults.models.abstention import AbstentionPolicy
from steel_faults.models.strong_model import build_random_forest_model
from steel_faults.monitoring.drift import DriftDetector

logger = logging.getLogger(__name__)


def create_app(
    model_bundle_path: str | Path = "artifacts/models/best_model.joblib",
    reference_path: str | Path = "artifacts/reference_distribution.joblib",
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Load trained model bundle and drift reference at startup."""
        m_path = Path(model_bundle_path)
        ref_path = Path(reference_path)

        if m_path.exists():
            try:
                bundle = joblib.load(m_path)
                app.state.model = bundle["model"]
                app.state.classes = list(bundle["classes"])
                app.state.feature_names = list(bundle.get("feature_names", FEATURE_NAMES))
                app.state.abstention_policy = AbstentionPolicy(
                    confidence_threshold=bundle.get("abstention_threshold", 0.50)
                )
                app.state.is_loaded = True
                logger.info("Successfully loaded model bundle from %s", m_path)
            except Exception as e:
                logger.error("Failed to load model from %s: %s", m_path, e)

        if ref_path.exists():
            try:
                ref_dict = joblib.load(ref_path)
                app.state.drift_detector = DriftDetector.from_dict(ref_dict)
                logger.info("Successfully loaded drift reference from %s", ref_path)
            except Exception as e:
                logger.error("Failed to load drift reference from %s: %s", ref_path, e)

        # Fallback to fast in-memory synthetic model if no artifacts exist yet
        if not app.state.is_loaded:
            logger.warning(
                "No artifact found at %s. Initializing in-memory fallback model.", m_path
            )
            syn_df = generate_synthetic_steel_data(n_samples=210, random_seed=42)
            from steel_faults.data.splitter import StratifiedSplitter
            from steel_faults.models.calibration import calibrate_on_validation

            splitter = StratifiedSplitter(random_seed=42)
            splits = splitter.split(syn_df)
            fallback_model = build_random_forest_model(n_estimators=30, random_state=42)
            fallback_model.fit(splits.X_train, splits.y_train)
            calibrated_fallback = calibrate_on_validation(
                fallback_model, splits.X_val, splits.y_val, method="sigmoid"
            )

            app.state.model = calibrated_fallback
            app.state.classes = list(splits.classes)
            app.state.drift_detector = DriftDetector(
                reference_features=splits.X_train,
                reference_predictions=calibrated_fallback.predict(splits.X_train),
                classes=splits.classes,
            )
            app.state.is_loaded = True

        yield

    app = FastAPI(
        title="Steel Plate Fault Detection API",
        description="Production-grade API for steel plate defect inference, abstention filtering, and distribution drift detection.",
        version=__version__,
        lifespan=lifespan,
    )

    app.state.model = None
    app.state.classes = []
    app.state.feature_names = list(FEATURE_NAMES)
    app.state.abstention_policy = AbstentionPolicy(confidence_threshold=0.50)
    app.state.drift_detector = None
    app.state.is_loaded = False

    @app.get("/health", status_code=status.HTTP_200_OK)
    def health_check() -> dict[str, Any]:
        """Health check endpoint returning system status and model readiness."""
        return {
            "status": "healthy",
            "model_loaded": app.state.is_loaded,
            "model_type": type(app.state.model).__name__ if app.state.model else None,
            "classes": app.state.classes,
            "version": __version__,
        }

    @app.post("/predict", response_model=PredictionResult, status_code=status.HTTP_200_OK)
    def predict_single(features: SteelPlateFeatures) -> PredictionResult:
        """Generate defect classification and abstention recommendation for a single steel plate."""
        if not app.state.is_loaded or app.state.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded",
            )

        row_df = pd.DataFrame([features.to_dict()])[app.state.feature_names]
        probs = app.state.model.predict_proba(row_df)[0]
        pred_class, pred_idx, confidence, review_rec = app.state.abstention_policy.evaluate_sample(
            probs, app.state.classes
        )

        prob_dict = {
            str(cls): round(float(p), 4) for cls, p in zip(app.state.classes, probs, strict=False)
        }

        return PredictionResult(
            predicted_class=pred_class,
            predicted_class_index=pred_idx,
            probabilities=prob_dict,
            confidence=round(confidence, 4),
            review_recommended=review_rec,
            abstention_threshold=app.state.abstention_policy.confidence_threshold,
        )

    @app.post(
        "/batch-predict", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK
    )
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Batch defect classification with selective abstention flagging."""
        if not app.state.is_loaded or app.state.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded",
            )

        records = [s.to_dict() for s in request.samples]
        batch_df = pd.DataFrame(records)[app.state.feature_names]
        all_probs = app.state.model.predict_proba(batch_df)

        policy = (
            AbstentionPolicy(confidence_threshold=request.abstention_threshold)
            if request.abstention_threshold is not None
            else app.state.abstention_policy
        )

        predictions, confidences, review_flags = policy.evaluate_batch(all_probs, app.state.classes)

        results: list[PredictionResult] = []
        for i, _s in enumerate(request.samples):
            probs_i = all_probs[i]
            prob_dict = {
                str(cls): round(float(p), 4)
                for cls, p in zip(app.state.classes, probs_i, strict=False)
            }
            results.append(
                PredictionResult(
                    predicted_class=str(predictions[i]),
                    predicted_class_index=int(np.argmax(probs_i)),
                    probabilities=prob_dict,
                    confidence=round(float(confidences[i]), 4),
                    review_recommended=bool(review_flags[i]),
                    abstention_threshold=policy.confidence_threshold,
                )
            )

        review_count = int(np.sum(review_flags))
        total_samples = len(request.samples)
        review_rate = float(review_count / total_samples) if total_samples > 0 else 0.0

        return BatchPredictionResponse(
            predictions=results,
            total_samples=total_samples,
            review_count=review_count,
            review_rate=round(review_rate, 4),
        )

    @app.post("/drift", response_model=DriftCheckResponse, status_code=status.HTTP_200_OK)
    def check_drift(request: DriftCheckRequest) -> DriftCheckResponse:
        """Evaluate feature and prediction distribution drift against the baseline reference."""
        prod_records = [s.to_dict() for s in request.production_samples]
        prod_df = pd.DataFrame(prod_records)[app.state.feature_names]

        # Use custom reference if provided, otherwise baseline detector
        if request.reference_samples is not None and len(request.reference_samples) >= 10:
            ref_records = [s.to_dict() for s in request.reference_samples]
            ref_df = pd.DataFrame(ref_records)[app.state.feature_names]
            detector = DriftDetector(reference_features=ref_df, classes=app.state.classes)
        elif app.state.drift_detector is not None:
            detector = app.state.drift_detector
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Reference drift baseline is not available",
            )

        feature_reports, overall_drift = detector.evaluate_feature_drift(prod_df)

        # Check prediction drift if model is available
        pred_drift_report = None
        if app.state.model is not None:
            prod_preds = app.state.model.predict(prod_df)
            pred_drift_report = detector.evaluate_prediction_drift(prod_preds)

        drifted_count = sum(1 for r in feature_reports if r.is_drift)

        return DriftCheckResponse(
            features_evaluated=len(feature_reports),
            features_drifted_count=drifted_count,
            overall_drift_detected=overall_drift,
            feature_reports=feature_reports,
            prediction_drift=pred_drift_report,
        )

    return app


app = create_app()
