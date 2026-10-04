"""Abstention and low-confidence selective classification policy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class SelectiveMetricPoint:
    """Performance metrics at a specific confidence abstention threshold."""

    threshold: float
    coverage: float
    covered_samples: int
    abstained_samples: int
    selective_accuracy: float
    selective_macro_f1: float


class AbstentionPolicy:
    """Manages low-confidence review flagging and selective classification analysis."""

    def __init__(self, confidence_threshold: float = 0.50) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
            )
        self.confidence_threshold = confidence_threshold

    def evaluate_sample(
        self, probabilities: np.ndarray, classes: list[str] | np.ndarray
    ) -> tuple[str, int, float, bool]:
        """Evaluate a single sample probability vector.

        Returns:
            (predicted_class, predicted_class_idx, confidence, review_recommended)
        """
        probs = np.asarray(probabilities)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = str(classes[pred_idx])
        review_recommended = confidence < self.confidence_threshold
        return pred_class, pred_idx, confidence, review_recommended

    def evaluate_batch(
        self,
        probabilities: np.ndarray,
        classes: list[str] | np.ndarray,
        custom_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a batch of probability vectors.

        Returns:
            (predictions, confidences, review_flags)
        """
        threshold = self.confidence_threshold if custom_threshold is None else custom_threshold
        probs = np.asarray(probabilities)
        pred_indices = np.argmax(probs, axis=1)
        classes_arr = np.asarray(classes)
        predictions = classes_arr[pred_indices]
        confidences = np.max(probs, axis=1)
        review_flags = confidences < threshold
        return predictions, confidences, review_flags

    @staticmethod
    def compute_selective_curve(
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        classes: list[str] | np.ndarray,
        thresholds: list[float] | None = None,
    ) -> list[SelectiveMetricPoint]:
        """Compute coverage vs selective accuracy curve across confidence thresholds.

        Args:
            y_true: Ground truth class labels.
            y_prob: Predicted probability array of shape (N, n_classes).
            classes: Array of class labels corresponding to columns of y_prob.
            thresholds: List of confidence thresholds to evaluate (0.0 to 1.0).

        Returns:
            list[SelectiveMetricPoint]: Metrics computed at each threshold.
        """
        if thresholds is None:
            thresholds = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        y_true_arr = np.asarray(y_true)
        probs = np.asarray(y_prob)
        classes_arr = np.asarray(classes)

        pred_indices = np.argmax(probs, axis=1)
        predictions = classes_arr[pred_indices]
        confidences = np.max(probs, axis=1)
        n_total = len(y_true_arr)

        points: list[SelectiveMetricPoint] = []
        for th in thresholds:
            mask = confidences >= th
            n_covered = int(np.sum(mask))
            n_abstained = n_total - n_covered
            coverage = float(n_covered / n_total) if n_total > 0 else 0.0

            if n_covered > 0:
                sel_acc = float(accuracy_score(y_true_arr[mask], predictions[mask]))
                sel_f1 = float(
                    f1_score(y_true_arr[mask], predictions[mask], average="macro", zero_division=0)
                )
            else:
                sel_acc = 0.0
                sel_f1 = 0.0

            points.append(
                SelectiveMetricPoint(
                    threshold=float(th),
                    coverage=round(coverage, 4),
                    covered_samples=n_covered,
                    abstained_samples=n_abstained,
                    selective_accuracy=round(sel_acc, 4),
                    selective_macro_f1=round(sel_f1, 4),
                )
            )

        return points
