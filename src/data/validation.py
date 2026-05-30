"""
Data validation for sentiment analysis
Simple, reliable validation without external dependencies
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Production data validation for sentiment analysis
    Validates text and label columns for quality checks
    """

    def __init__(self, suite_name: str = "sentiment_validation_suite"):
        self.suite_name = suite_name
        self.suite = None

    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Validate DataFrame against expectations
        
        Args:
            df: DataFrame to validate
            dataset_name: Name for logging purposes
            
        Returns:
            Dictionary with validation results
        """
        return self._fallback_validation(df, dataset_name)

    def _fallback_validation(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Validate DataFrame with comprehensive checks
        """
        failures = []
        
        # Check if DataFrame is empty
        if df.empty:
            failures.append("DataFrame is empty")
            logger.warning(f"Validation failed for {dataset_name}: DataFrame is empty")
            return {
                "success": False,
                "failures": failures,
                "statistics": {
                    "evaluated_expectations": 1,
                    "successful_expectations": 0,
                    "unsuccessful_expectations": 1,
                    "success_percent": 0.0
                }
            }
        
        # Check required columns
        required_cols = ["text", "label"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            failures.append(f"Missing columns: {missing_cols}")
        
        # Check null values
        if "text" in df.columns and df["text"].isnull().any():
            null_count = df["text"].isnull().sum()
            failures.append(f"Null values found in 'text' column: {null_count} rows")
        
        if "label" in df.columns and df["label"].isnull().any():
            null_count = df["label"].isnull().sum()
            failures.append(f"Null values found in 'label' column: {null_count} rows")
        
        # Check text length
        if "text" in df.columns and not df["text"].isnull().all():
            text_lengths = df["text"].dropna().str.len()
            if len(text_lengths) > 0:
                if text_lengths.min() < 1:
                    failures.append("Empty strings found in 'text' column")
                if text_lengths.max() > 10000:
                    failures.append(f"Text length exceeds 10000 chars: {text_lengths.max()} chars")
        
        # Check label values
        if "label" in df.columns and not df["label"].isnull().all():
            valid_labels = {0, 1, 2}
            invalid_labels = set(df["label"].dropna().unique()) - valid_labels
            if invalid_labels:
                failures.append(f"Invalid label values: {invalid_labels}")
        
        success = len(failures) == 0
        
        if not success:
            logger.warning(f"Validation failed for {dataset_name}: {failures}")
        else:
            logger.info(f"Validation passed for {dataset_name}")
        
        return {
            "success": success,
            "failures": failures,
            "statistics": {
                "evaluated_expectations": len(failures) + 1 if failures else 1,
                "successful_expectations": 1 if success else 0,
                "unsuccessful_expectations": len(failures),
                "success_percent": 100.0 if success else 0.0
            }
        }

    def get_expectation_suite(self):
        """Return the expectation suite (compatibility method)"""
        return self.suite

    def save_suite(self, filepath: str):
        """Save expectation suite to JSON file"""
        import json
        suite_dict = {
            "suite_name": self.suite_name,
            "expectations": [
                "columns_exist: text, label",
                "no_null_values",
                "text_length_between_1_and_10000",
                "label_values_in_{0,1,2}"
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(suite_dict, f, indent=2)
        logger.info(f"Saved expectation suite to {filepath}")

    def load_suite(self, filepath: str):
        """Load expectation suite from JSON file"""
        import json
        try:
            with open(filepath, 'r') as f:
                suite_dict = json.load(f)
            self.suite_name = suite_dict.get("suite_name", self.suite_name)
            logger.info(f"Loaded expectation suite from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Suite file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading suite: {e}")