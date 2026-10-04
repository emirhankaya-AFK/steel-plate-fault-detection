"""Feature and prediction distribution drift monitoring using PSI and KS test."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from steel_faults.data.schema import (
    FEATURE_NAMES,
    FeatureDriftReport,
    PredictionDriftReport,
)

logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    num_bins: int | None = None,
) -> float:
    """Calculate the Population Stability Index (PSI) between two continuous distributions.

    Uses adaptive quantile binning and Laplace smoothing to handle varying sample sizes safely.

    PSI Interpretation:
        < 0.10: Insignificant drift (distribution is stable)
        0.10 - 0.20: Moderate drift (warning)
        >= 0.20: Significant drift (action required)
    """
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)

    if len(exp_arr) == 0 or len(act_arr) == 0:
        return 0.0

    if num_bins is None:
        num_bins = min(10, max(3, len(act_arr) // 10))

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(exp_arr, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(exp_arr, bins=bin_edges)
    act_counts, _ = np.histogram(act_arr, bins=bin_edges)

    # Standard Laplace smoothing to prevent infinite log under small or zero counts
    n_bins = len(exp_counts)
    exp_pct = (exp_counts + 1.0) / (np.sum(exp_counts) + n_bins)
    act_pct = (act_counts + 1.0) / (np.sum(act_counts) + n_bins)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(max(0.0, psi_val))


def calculate_categorical_psi(
    ref_counts: dict[str, int | float],
    cur_counts: dict[str, int | float],
    classes: list[str],
) -> float:
    """Calculate PSI between categorical class frequency distributions with Laplace smoothing."""
    n_classes = len(classes)
    ref_total = sum(ref_counts.get(c, 0.0) for c in classes) + n_classes
    cur_total = sum(cur_counts.get(c, 0.0) for c in classes) + n_classes

    psi_val = 0.0
    for c in classes:
        p_ref = (ref_counts.get(c, 0.0) + 1.0) / ref_total
        p_cur = (cur_counts.get(c, 0.0) + 1.0) / cur_total
        psi_val += (p_cur - p_ref) * np.log(p_cur / p_ref)

    return float(max(0.0, psi_val))


class DriftDetector:
    """Monitors incoming steel plate batches for feature and target prediction drift."""

    def __init__(
        self,
        reference_features: pd.DataFrame,
        reference_predictions: pd.Series | np.ndarray | None = None,
        classes: list[str] | None = None,
        psi_drift_threshold: float = 0.20,
        psi_warning_threshold: float = 0.10,
        ks_alpha: float = 0.05,
    ) -> None:
        self.reference_features = reference_features.copy()
        self.reference_predictions = (
            np.asarray(reference_predictions) if reference_predictions is not None else None
        )
        self.classes = classes or []
        self.psi_drift_threshold = psi_drift_threshold
        self.psi_warning_threshold = psi_warning_threshold
        self.ks_alpha = ks_alpha

        # Compute baseline target distribution if predictions provided
        self.reference_pred_dist: dict[str, float] = {}
        if self.reference_predictions is not None and len(self.reference_predictions) > 0:
            total = len(self.reference_predictions)
            unique, counts = np.unique(self.reference_predictions, return_counts=True)
            self.reference_pred_dist = {
                str(u): round(float(c / total), 4) for u, c in zip(unique, counts, strict=False)
            }

    def evaluate_feature_drift(
        self, current_features: pd.DataFrame
    ) -> tuple[list[FeatureDriftReport], bool]:
        """Evaluate feature drift across all 27 numerical features.

        Returns:
            (reports, overall_drift_detected)
        """
        reports: list[FeatureDriftReport] = []
        features_drifted = 0

        for col in FEATURE_NAMES:
            if col not in current_features.columns or col not in self.reference_features.columns:
                continue

            ref_vals = self.reference_features[col].to_numpy()
            cur_vals = current_features[col].to_numpy()

            psi_val = calculate_psi(ref_vals, cur_vals)
            ks_res = stats.ks_2samp(ref_vals, cur_vals)
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)

            # Sample-size-aware drift flag: for small samples (N < 100), statistical significance
            # (KS p-value < alpha) prevents false positives from histogram noise
            is_drift = (
                psi_val >= self.psi_drift_threshold
                and (ks_pval < self.ks_alpha or len(cur_vals) >= 100)
            ) or (ks_pval < self.ks_alpha and ks_stat > 0.25)

            if is_drift:
                severity = "Severe"
            elif psi_val >= self.psi_warning_threshold or ks_pval < self.ks_alpha:
                severity = "Moderate"
            else:
                severity = "None"

            if is_drift:
                features_drifted += 1

            reports.append(
                FeatureDriftReport(
                    feature_name=col,
                    psi=round(psi_val, 4),
                    ks_statistic=round(ks_stat, 4),
                    ks_pvalue=round(ks_pval, 6),
                    is_drift=is_drift,
                    severity=severity,
                )
            )

        # Flag overall drift if >= 5 features show drift (>18% of input feature space)
        overall_drift = features_drifted >= 5
        return reports, overall_drift

    def evaluate_prediction_drift(
        self, current_predictions: pd.Series | np.ndarray
    ) -> PredictionDriftReport | None:
        """Evaluate drift in the predicted class label distribution."""
        if not self.reference_pred_dist:
            return None

        cur_arr = np.asarray(current_predictions)
        total = len(cur_arr)
        if total == 0:
            return None

        unique, counts = np.unique(cur_arr, return_counts=True)
        cur_dist = {
            str(u): round(float(c / total), 4) for u, c in zip(unique, counts, strict=False)
        }

        classes = list(set(list(self.reference_pred_dist.keys()) + list(cur_dist.keys())))
        psi_val = calculate_categorical_psi(self.reference_pred_dist, cur_dist, classes)

        return PredictionDriftReport(
            psi=round(psi_val, 4),
            reference_distribution=self.reference_pred_dist,
            production_distribution=cur_dist,
            is_drift=psi_val >= self.psi_drift_threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize reference baseline for offline loading."""
        return {
            "reference_features": self.reference_features.to_dict(orient="list"),
            "reference_predictions": (
                self.reference_predictions.tolist()
                if self.reference_predictions is not None
                else None
            ),
            "classes": self.classes,
            "reference_pred_dist": self.reference_pred_dist,
            "psi_drift_threshold": self.psi_drift_threshold,
            "psi_warning_threshold": self.psi_warning_threshold,
            "ks_alpha": self.ks_alpha,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DriftDetector":
        """Reconstruct DriftDetector from serialized dictionary."""
        ref_df = pd.DataFrame(data["reference_features"])
        ref_preds = data.get("reference_predictions")
        return cls(
            reference_features=ref_df,
            reference_predictions=ref_preds,
            classes=data.get("classes", []),
            psi_drift_threshold=data.get("psi_drift_threshold", 0.20),
            psi_warning_threshold=data.get("psi_warning_threshold", 0.10),
            ks_alpha=data.get("ks_alpha", 0.05),
        )
