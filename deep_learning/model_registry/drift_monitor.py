"""
Model Drift Monitoring & Retraining Engine
Calculates Population Stability Index (PSI) and checks for data / concept drift.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class DriftMonitor:
    def __init__(self, psi_threshold: float = 0.25):
        self.psi_threshold = psi_threshold

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI) between baseline and production feature distributions.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change, review recommended
        PSI >= 0.25: Significant change (Drift detected, trigger retraining)
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        percentiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(expected, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_pct = np.maximum(expected_counts / max(1, len(expected)), 1e-4)
        actual_pct = np.maximum(actual_counts / max(1, len(actual)), 1e-4)

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi_val)

    def evaluate_feature_drift(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """
        Evaluates drift across all features and returns drift status.
        """
        psi_scores = {}
        drifted_features = []

        for col in feature_cols:
            if col in baseline_df.columns and col in current_df.columns:
                psi = self.calculate_psi(baseline_df[col].dropna().values, current_df[col].dropna().values)
                psi_scores[col] = round(psi, 4)
                if psi >= self.psi_threshold:
                    drifted_features.append(col)

        mean_psi = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0
        is_drift_detected = (mean_psi >= self.psi_threshold) or (len(drifted_features) > len(feature_cols) * 0.3)

        return {
            "is_drift_detected": is_drift_detected,
            "mean_psi": round(mean_psi, 4),
            "drifted_features": drifted_features,
            "feature_psi": psi_scores
        }
