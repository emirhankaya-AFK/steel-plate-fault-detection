"""End-to-end training, validation tuning, calibration, and test evaluation pipeline."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import yaml

from steel_faults.data.downloader import DatasetDownloader
from steel_faults.data.schema import FEATURE_NAMES
from steel_faults.data.splitter import StratifiedSplitter
from steel_faults.data.synthetic import generate_synthetic_steel_data
from steel_faults.evaluation.explainability import (
    compute_permutation_importance,
    extract_native_feature_importance,
)
from steel_faults.evaluation.metrics import evaluate_model_performance
from steel_faults.models.abstention import AbstentionPolicy
from steel_faults.models.baseline import build_baseline_model
from steel_faults.models.calibration import calibrate_on_validation
from steel_faults.models.strong_model import build_strong_model
from steel_faults.monitoring.drift import DriftDetector

logger = logging.getLogger(__name__)


def run_pipeline(
    config_path: str | Path = "configs/config.yaml",
    offline_synthetic: bool = False,
    output_metrics_path: str | Path | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Execute the complete reproducible ML pipeline.

    Steps:
        1. Ingest dataset (official UCI download or offline synthetic fixture).
        2. Perform stratified train/validation/test split.
        3. Train baseline (Logistic Regression) and strong model (Random Forest).
        4. Calibrate probabilities strictly on validation split.
        5. Evaluate models on untouched test set once for final reporting.
        6. Compute coverage vs selective accuracy under abstention policy.
        7. Compute feature permutation importances on validation split.
        8. Initialize drift monitoring baseline and persist artifacts.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Dataset Ingestion
    if offline_synthetic:
        logger.info("Using offline synthetic dataset generator for CI/offline mode.")
        raw_df = generate_synthetic_steel_data(
            n_samples=cfg["data"].get("synthetic_sample_size", 210),
            random_seed=cfg["data"]["random_seed"],
        )
    else:
        downloader = DatasetDownloader(
            url=cfg["data"]["raw_data_url"],
            cache_dir=cfg["data"]["raw_data_dir"],
        )
        raw_df = downloader.download_and_extract()

    # 2. Stratified Partitioning
    splitter = StratifiedSplitter(
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_seed=cfg["data"]["random_seed"],
    )
    splits = splitter.split(raw_df)

    logger.info(
        "Partition sizes - Train: %d, Val: %d, Test: %d (Classes: %s)",
        len(splits.X_train),
        len(splits.X_val),
        len(splits.X_test),
        splits.classes,
    )

    # 3. Train Baseline Model (Logistic Regression with balanced weights)
    logger.info("Training baseline model (Logistic Regression)...")
    baseline = build_baseline_model(
        max_iter=cfg["model"]["baseline"]["max_iter"],
        class_weight=cfg["model"]["baseline"]["class_weight"],
        C=cfg["model"]["baseline"]["C"],
        random_state=cfg["model"]["random_seed"],
    )
    baseline.fit(splits.X_train, splits.y_train)

    # 4. Train Strong Model (Random Forest with balanced weights)
    logger.info("Training strong model (Random Forest)...")
    strong_model = build_strong_model(
        model_type=cfg["model"].get("selected_model", "random_forest"),
        n_estimators=cfg["model"]["random_forest"]["n_estimators"],
        max_depth=cfg["model"]["random_forest"]["max_depth"],
        min_samples_split=cfg["model"]["random_forest"]["min_samples_split"],
        class_weight=cfg["model"]["random_forest"]["class_weight"],
        random_state=cfg["model"]["random_seed"],
        n_jobs=cfg["model"]["random_forest"].get("n_jobs", -1),
    )
    strong_model.fit(splits.X_train, splits.y_train)

    # 5. Probability Calibration strictly on Validation split
    logger.info("Calibrating strong model on validation set...")
    calibrated_strong = calibrate_on_validation(
        strong_model,
        splits.X_val,
        splits.y_val,
        method=cfg["calibration"]["method"],
    )

    # 6. Evaluation on Validation Set
    val_baseline_preds = baseline.predict(splits.X_val)
    val_baseline_probs = baseline.predict_proba(splits.X_val)
    val_baseline_metrics = evaluate_model_performance(
        splits.y_val, val_baseline_preds, val_baseline_probs, splits.classes
    )

    val_rf_preds = strong_model.predict(splits.X_val)
    val_rf_probs = strong_model.predict_proba(splits.X_val)
    val_rf_metrics = evaluate_model_performance(
        splits.y_val, val_rf_preds, val_rf_probs, splits.classes
    )

    val_cal_preds = calibrated_strong.predict(splits.X_val)
    val_cal_probs = calibrated_strong.predict_proba(splits.X_val)
    val_cal_metrics = evaluate_model_performance(
        splits.y_val, val_cal_preds, val_cal_probs, splits.classes
    )

    # 7. Final Untouched Test Set Evaluation (Used ONCE)
    logger.info("Evaluating all candidate models on untouched test set...")
    test_baseline_preds = baseline.predict(splits.X_test)
    test_baseline_probs = baseline.predict_proba(splits.X_test)
    test_baseline_metrics = evaluate_model_performance(
        splits.y_test, test_baseline_preds, test_baseline_probs, splits.classes
    )

    test_rf_preds = strong_model.predict(splits.X_test)
    test_rf_probs = strong_model.predict_proba(splits.X_test)
    test_rf_metrics = evaluate_model_performance(
        splits.y_test, test_rf_preds, test_rf_probs, splits.classes
    )

    test_cal_preds = calibrated_strong.predict(splits.X_test)
    test_cal_probs = calibrated_strong.predict_proba(splits.X_test)
    test_cal_metrics = evaluate_model_performance(
        splits.y_test, test_cal_preds, test_cal_probs, splits.classes
    )

    # 8. Abstention & Selective Classification Analysis on Test Set
    abstention_policy = AbstentionPolicy(
        confidence_threshold=cfg["abstention"]["default_confidence_threshold"]
    )
    selective_points = abstention_policy.compute_selective_curve(
        splits.y_test,
        test_cal_probs,
        splits.classes,
        thresholds=cfg["abstention"]["thresholds_to_evaluate"],
    )
    selective_curve_data = [point.__dict__ for point in selective_points]

    # 9. Explainability (Permutation importance on Validation Split)
    logger.info("Computing validation permutation importances...")
    perm_importances = compute_permutation_importance(
        strong_model,
        splits.X_val,
        splits.y_val,
        scoring="f1_macro",
        n_repeats=5,
        random_state=cfg["model"]["random_seed"],
    )
    native_importances = extract_native_feature_importance(strong_model, list(FEATURE_NAMES))

    # 10. Drift Detector Baseline Initialization
    train_preds = calibrated_strong.predict(splits.X_train)
    drift_detector = DriftDetector(
        reference_features=splits.X_train,
        reference_predictions=train_preds,
        classes=splits.classes,
        psi_drift_threshold=cfg["drift"]["psi_threshold_drift"],
        psi_warning_threshold=cfg["drift"]["psi_threshold_warning"],
        ks_alpha=cfg["drift"]["ks_alpha"],
    )

    # Assemble comprehensive results
    results: dict[str, Any] = {
        "dataset_summary": {
            "source": "synthetic" if offline_synthetic else "UCI Machine Learning Repository",
            "total_samples": len(raw_df),
            "train_samples": len(splits.X_train),
            "val_samples": len(splits.X_val),
            "test_samples": len(splits.X_test),
            "num_features": len(FEATURE_NAMES),
            "num_classes": len(splits.classes),
            "classes": splits.classes,
        },
        "validation_metrics": {
            "baseline_logistic_regression": val_baseline_metrics,
            "random_forest_uncalibrated": val_rf_metrics,
            "random_forest_calibrated": val_cal_metrics,
        },
        "test_metrics": {
            "baseline_logistic_regression": test_baseline_metrics,
            "random_forest_uncalibrated": test_rf_metrics,
            "random_forest_calibrated": test_cal_metrics,
        },
        "abstention_selective_analysis": {
            "default_confidence_threshold": cfg["abstention"]["default_confidence_threshold"],
            "selective_curve": selective_curve_data,
        },
        "explainability": {
            "methodology": "Scikit-Learn permutation_importance on validation split (scoring=f1_macro) and native Gini MDI. No fake SHAP approximations.",
            "top_10_permutation_features": perm_importances[:10],
            "top_10_native_features": native_importances[:10] if native_importances else [],
        },
    }

    # Save artifacts if requested
    if save_artifacts:
        model_path = Path(cfg["serving"]["model_artifact_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # We save a deployment bundle with the best calibrated model and metadata
        deployment_bundle = {
            "model": calibrated_strong,
            "uncalibrated_model": strong_model,
            "classes": splits.classes,
            "feature_names": list(FEATURE_NAMES),
            "abstention_threshold": cfg["abstention"]["default_confidence_threshold"],
        }
        joblib.dump(deployment_bundle, model_path)
        logger.info("Saved deployment model bundle to %s", model_path)

        ref_path = Path(cfg["serving"]["reference_data_path"])
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(drift_detector.to_dict(), ref_path)
        logger.info("Saved drift reference baseline to %s", ref_path)

    # Save metrics JSON artifact
    metrics_file = (
        Path(output_metrics_path) if output_metrics_path else Path(cfg["serving"]["metrics_path"])
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Committed verified metrics to %s", metrics_file)

    return results
