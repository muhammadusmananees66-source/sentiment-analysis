"""
Data validation using Great Expectations (v0.18+)
"""

import great_expectations as ge
import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Production data validation with Great Expectations (v0.18+)

    Uses the modern GX API with ExpectationSuite and ExpectationConfiguration
    from the core module (still supported but with updated syntax)

    Validates:
    - No missing values in critical columns
    - Text length within bounds
    - Label values are valid (0,1,2)
    - Data types are correct
    """

    def __init__(self, suite_name: str = "sentiment_validation_suite"):
        self.suite_name = suite_name
        self.context = None
        self.suite = None
        self._initialize_suite()

    def _initialize_suite(self):
        """Initialize the expectation suite using modern GX approach"""
        try:
            # Method 1: Using core module (works in v0.18+)
            from great_expectations.core import ExpectationSuite, ExpectationConfiguration

            self.suite = ExpectationSuite(
                expectation_suite_name=self.suite_name,
                expectations=[]
            )

            # Add expectations using the modern API
            expectations = self._create_expectations()
            for expectation_config in expectations:
                self.suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type=expectation_config["type"],
                        kwargs=expectation_config["kwargs"]
                    )
                )

            logger.info(f"Initialized expectation suite: {self.suite_name}")

        except ImportError:
            # Method 2: Fallback for v0.18+ alternative import
            try:
                from great_expectations.expectations.core import ExpectColumnValuesToNotBeNull
                from great_expectations.expectations.core import ExpectColumnValuesToBeInSet
                from great_expectations.expectations.core import ExpectColumnValueLengthsToBeBetween
                from great_expectations.expectations.core import ExpectTableColumnsToMatchSet

                # Create expectations using the new classes
                self.suite = self._create_suite_with_classes()
                logger.info(f"Initialized expectation suite using class-based approach")

            except ImportError:
                # Method 3: Simplest fallback - create via context
                self._create_suite_via_context()

    def _create_expectations(self) -> list:
        """Create expectation configurations in modern format"""
        expectations = [
            # Column existence expectations
            {
                "type": "expect_table_columns_to_match_set",
                "kwargs": {
                    "column_set": ["text", "label"],
                    "exact_match": False
                }
            },
            # No null values in text column
            {
                "type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "text"}
            },
            # No null values in label column
            {
                "type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "label"}
            },
            # Text length validations (min 1 char, max 10000 chars)
            {
                "type": "expect_column_value_lengths_to_be_between",
                "kwargs": {
                    "column": "text",
                    "min_value": 1,
                    "max_value": 10000
                }
            },
            # Label valid values (0=negative, 1=positive, 2=neutral)
            {
                "type": "expect_column_values_to_be_in_set",
                "kwargs": {
                    "column": "label",
                    "value_set": [0, 1, 2]
                }
            },
            # Data type validation
            {
                "type": "expect_column_values_to_be_of_type",
                "kwargs": {
                    "column": "label",
                    "type_": "int"
                }
            },
            # Text should not be empty after stripping
            {
                "type": "expect_column_values_to_match_regex",
                "kwargs": {
                    "column": "text",
                    "regex": r".+"
                }
            }
        ]
        return expectations

    def _create_suite_with_classes(self):
        """Alternative: Create suite using expectation classes (v0.18+ alternative)"""
        from great_expectations.core import ExpectationSuite
        from great_expectations.expectations import (
            ExpectTableColumnsToMatchSet,
            ExpectColumnValuesToNotBeNull,
            ExpectColumnValueLengthsToBeBetween,
            ExpectColumnValuesToBeInSet
        )

        suite = ExpectationSuite(expectation_suite_name=self.suite_name)

        # Add expectations using classes
        suite.add_expectation(
            ExpectTableColumnsToMatchSet(
                column_set=["text", "label"],
                exact_match=False
            )
        )

        suite.add_expectation(
            ExpectColumnValuesToNotBeNull(column="text")
        )

        suite.add_expectation(
            ExpectColumnValuesToNotBeNull(column="label")
        )

        suite.add_expectation(
            ExpectColumnValueLengthsToBeBetween(
                column="text",
                min_value=1,
                max_value=10000
            )
        )

        suite.add_expectation(
            ExpectColumnValuesToBeInSet(
                column="label",
                value_set=[0, 1, 2]
            )
        )

        return suite

    def _create_suite_via_context(self):
        """Last resort: Create suite using GX context (most robust)"""
        try:
            import great_expectations as gx

            # Create a temporary context
            context = gx.get_context(mode="ephemeral")

            # Create expectation suite
            self.suite = context.add_expectation_suite(self.suite_name)

            # Add expectations
            expectations = self._create_expectations()
            for exp in expectations:
                self.suite.add_expectation(
                    ge.core.ExpectationConfiguration(
                        expectation_type=exp["type"],
                        kwargs=exp["kwargs"]
                    )
                )

            logger.info(f"Created suite via GX context: {self.suite_name}")

        except Exception as e:
            logger.error(f"Failed to create expectation suite: {e}")
            # Create minimal suite that won't break
            self.suite = self._create_minimal_suite()

    def _create_minimal_suite(self):
        """Create a minimal fallback suite if everything else fails"""
        from great_expectations.core import ExpectationSuite, ExpectationConfiguration

        suite = ExpectationSuite(expectation_suite_name=self.suite_name)

        # Only basic validations to ensure compatibility
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "text"}
            )
        )

        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "label"}
            )
        )

        return suite

    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Validate DataFrame against expectations

        Args:
            df: DataFrame to validate
            dataset_name: Name for logging purposes

        Returns:
            Dictionary with validation results
        """
        try:
            # Modern approach: Create validator from DataFrame
            from great_expectations.dataset import PandasDataset

            # Convert DataFrame to GX Dataset
            ge_df = PandasDataset(df, expectation_suite=self.suite)

            # Run validation
            results = ge_df.validate()

            # Convert results to serializable dict
            validation_result = {
                "success": results.get("success", False),
                "statistics": results.get("statistics", {}),
                "results": results.get("results", [])
            }

            # Log failures
            if not validation_result["success"]:
                failed_expectations = [
                    exp for exp in validation_result["results"]
                    if not exp.get("success", False)
                ]

                logger.warning(
                    f"Validation failed for {dataset_name}: "
                    f"{len(failed_expectations)} failures"
                )

                for exp in failed_expectations[:5]:  # Log first 5 failures
                    expectation_type = exp.get("expectation_config", {}).get("expectation_type", "unknown")
                    result_info = exp.get("result", {})
                    logger.warning(f"  - {expectation_type}: {result_info}")

            return validation_result

        except Exception as e:
            # Fallback validation when GX fails
            logger.error(f"GX validation failed with error: {e}, using fallback validation")
            return self._fallback_validation(df, dataset_name)

    def _fallback_validation(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Fallback validation when Great Expectations is not available
        This ensures the pipeline continues to work
        """
        failures = []

        # Check required columns
        required_cols = ["text", "label"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            failures.append(f"Missing columns: {missing_cols}")

        # Check null values
        if "text" in df.columns and df["text"].isnull().any():
            failures.append("Null values found in 'text' column")

        if "label" in df.columns and df["label"].isnull().any():
            failures.append("Null values found in 'label' column")

        # Check text length
        if "text" in df.columns:
            text_lengths = df["text"].str.len()
            if text_lengths.min() < 1:
                failures.append("Empty strings found in 'text' column")
            if text_lengths.max() > 10000:
                failures.append(f"Text length exceeds 10000 chars: {text_lengths.max()}")

        # Check label values
        if "label" in df.columns:
            invalid_labels = set(df["label"].unique()) - {0, 1, 2}
            if invalid_labels:
                failures.append(f"Invalid label values: {invalid_labels}")

        success = len(failures) == 0

        if not success:
            logger.warning(f"Fallback validation failed for {dataset_name}: {failures}")

        return {
            "success": success,
            "failures": failures,
            "statistics": {
                "evaluated_expectations": len(failures) + 1,
                "successful_expectations": 1 if success else 0,
                "unsuccessful_expectations": len(failures),
                "success_percent": 100.0 if success else 0.0
            }
        }

    def get_expectation_suite(self):
        """Return the expectation suite for use elsewhere"""
        return self.suite

    def save_suite(self, filepath: str):
        """Save expectation suite to JSON file"""
        import json

        suite_dict = {
            "suite_name": self.suite_name,
            "expectations": []
        }

        for expectation in self.suite.expectations:
            suite_dict["expectations"].append({
                "expectation_type": expectation.expectation_type,
                "kwargs": expectation.kwargs
            })

        with open(filepath, 'w') as f:
            json.dump(suite_dict, f, indent=2)

        logger.info(f"Saved expectation suite to {filepath}")

    def load_suite(self, filepath: str):
        """Load expectation suite from JSON file"""
        import json
        from great_expectations.core import ExpectationConfiguration, ExpectationSuite

        with open(filepath, 'r') as f:
            suite_dict = json.load(f)

        self.suite = ExpectationSuite(expectation_suite_name=suite_dict["suite_name"])

        for exp in suite_dict["expectations"]:
            self.suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type=exp["expectation_type"],
                    kwargs=exp["kwargs"]
                )
            )

        logger.info(f"Loaded expectation suite from {filepath}")