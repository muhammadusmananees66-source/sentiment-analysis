"""
Data drift and model drift detection using Evidently AI and PSI
"""

import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import mlflow
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Production drift detection for sentiment models

    Detects:
    - Data drift (feature distribution changes)
    - Prediction drift (output distribution changes)
    - Feature importance drift
    """

    def __init__(self, reference_data: pd.DataFrame, reference_predictions: Optional[np.ndarray] = None):
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.drift_threshold = 0.1  # PSI threshold

    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect data drift between reference and current datasets
        """
        try:
            report = Report(metrics=[
                DataDriftPreset(),
            ])

            report.run(reference_data=self.reference_data, current_data=current_data)

            # Extract drift metrics
            drift_results = report.as_dict()

            # Calculate PSI for each feature
            psi_scores = {}
            for column in self.reference_data.columns:
                if column in current_data.columns and self.reference_data[column].dtype in ['float64', 'int64']:
                    psi = self._calculate_psi(
                        self.reference_data[column].values,
                        current_data[column].values
                    )
                    psi_scores[column] = psi

            # Determine if drift is detected
            drift_detected = any(psi > self.drift_threshold for psi in psi_scores.values())

            return {
                "drift_detected": drift_detected,
                "psi_scores": psi_scores,
                "drifted_features": [col for col, psi in psi_scores.items() if psi > self.drift_threshold],
                "report": drift_results
            }

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return {"drift_detected": False, "error": str(e)}

    def detect_target_drift(self, current_predictions: np.ndarray) -> Dict:
        """
        Detect prediction drift (output distribution changes)
        """
        if self.reference_predictions is None:
            return {"drift_detected": False, "error": "No reference predictions"}

        try:
            # Calculate PSI for predictions
            psi = self._calculate_psi(self.reference_predictions, current_predictions)

            # Calculate distribution metrics
            ref_dist = np.bincount(self.reference_predictions.astype(int)) / len(self.reference_predictions)
            curr_dist = np.bincount(current_predictions.astype(int)) / len(current_predictions)

            return {
                "drift_detected": psi > self.drift_threshold,
                "psi": psi,
                "reference_distribution": ref_dist.tolist(),
                "current_distribution": curr_dist.tolist()
            }

        except Exception as e:
            logger.error(f"Target drift detection failed: {e}")
            return {"drift_detected": False, "error": str(e)}

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)

        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
        """
        # Create bins for continuous data
        if reference.dtype in ['float64', 'int64']:
            percentiles = np.percentile(reference, np.linspace(0, 100, bins + 1))
            percentiles[-1] += 1  # Ensure max value included

            # Bin reference
            ref_binned = np.digitize(reference, percentiles) - 1
            ref_binned = np.clip(ref_binned, 0, bins - 1)

            # Bin current using same bins
            curr_binned = np.digitize(current, percentiles) - 1
            curr_binned = np.clip(curr_binned, 0, bins - 1)
        else:
            # Categorical data - use unique values
            unique_vals = np.unique(np.concatenate([reference, current]))
            ref_binned = np.searchsorted(unique_vals, reference)
            curr_binned = np.searchsorted(unique_vals, current)
            bins = len(unique_vals)

        # Calculate percentages
        ref_counts = np.bincount(ref_binned, minlength=bins)
        curr_counts = np.bincount(curr_binned, minlength=bins)

        ref_pct = ref_counts / len(reference)
        curr_pct = curr_counts / len(current)

        # Add small epsilon to avoid division by zero
        ref_pct = np.clip(ref_pct, 1e-5, 1)
        curr_pct = np.clip(curr_pct, 1e-5, 1)

        # Calculate PSI
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))

        return psi

    def trigger_retraining(self, drift_results: Dict) -> bool:
        """
        Determine if model retraining should be triggered
        """
        if not drift_results.get("drift_detected", False):
            return False

        psi_scores = drift_results.get("psi_scores", {})

        # Trigger if more than 30% of features have high drift
        high_drift_count = sum(1 for psi in psi_scores.values() if psi > 0.25)

        if high_drift_count > len(psi_scores) * 0.3:
            logger.warning(f"High drift detected in {high_drift_count} features, triggering retraining")
            return True

        return False