"""Tests for data validation module"""
import pytest
import pandas as pd
import numpy as np
from src.data.validation import DataValidator


class TestDataValidator:
    """Test cases for DataValidator"""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing"""
        return pd.DataFrame({
            'text': ['Great product!', 'Terrible service', 'Okay experience'],
            'label': [1, 0, 2]
        })

    @pytest.fixture
    def invalid_dataframe(self):
        """Create invalid dataframe for testing"""
        return pd.DataFrame({
            'text': ['Good', None, 'Bad'],
            'label': [1, 0, 5]  # 5 is invalid label
        })

    def test_validator_initialization(self):
        """Test that validator initializes"""
        validator = DataValidator()
        assert validator.suite_name == "sentiment_validation_suite"
        # Suite can be None in simplified version
        assert validator.suite is None

    def test_validate_valid_dataframe(self, sample_dataframe):
        """Test validation passes for valid data"""
        validator = DataValidator()
        result = validator.validate_dataframe(sample_dataframe, "test_dataset")
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True
        assert len(result["failures"]) == 0

    def test_validate_missing_columns(self):
        """Test validation fails for missing columns"""
        validator = DataValidator()
        df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        result = validator.validate_dataframe(df, "test_dataset")
        assert "success" in result
        assert result["success"] is False
        assert len(result["failures"]) > 0
        # Verify the failure message mentions missing columns
        assert any("Missing columns" in failure for failure in result["failures"])

    def test_validate_invalid_dataframe(self, invalid_dataframe):
        """Test validation fails for invalid data"""
        validator = DataValidator()
        result = validator.validate_dataframe(invalid_dataframe, "test_dataset")
        assert result["success"] is False
        assert len(result["failures"]) > 0
        # Should catch both null values and invalid labels
        assert any("Null values" in failure for failure in result["failures"])
        assert any("Invalid label" in failure for failure in result["failures"])

    def test_fallback_validation(self, sample_dataframe):
        """Test fallback validation works"""
        validator = DataValidator()
        result = validator._fallback_validation(sample_dataframe, "test")
        assert "success" in result
        assert result["success"] is True
        assert "failures" in result
        assert len(result["failures"]) == 0

    def test_save_and_load_suite(self, tmp_path):
        """Test saving and loading expectation suite"""
        validator = DataValidator()
        filepath = tmp_path / "suite.json"
        validator.save_suite(str(filepath))
        assert filepath.exists()
        
        new_validator = DataValidator()
        new_validator.load_suite(str(filepath))
        assert new_validator.suite_name == validator.suite_name

    def test_statistics_in_result(self, sample_dataframe):
        """Test that validation result includes statistics"""
        validator = DataValidator()
        result = validator.validate_dataframe(sample_dataframe, "test_dataset")
        assert "statistics" in result
        stats = result["statistics"]
        assert "evaluated_expectations" in stats
        assert "successful_expectations" in stats
        assert "unsuccessful_expectations" in stats
        assert "success_percent" in stats
        assert stats["successful_expectations"] == 1
        assert stats["unsuccessful_expectations"] == 0
        assert stats["success_percent"] == 100.0

    def test_empty_dataframe_validation(self):
        """Test validation fails for empty DataFrame"""
        validator = DataValidator()
        df = pd.DataFrame(columns=['text', 'label'])
        result = validator.validate_dataframe(df, "test_dataset")
        assert result["success"] is False
        assert any("empty" in failure.lower() for failure in result["failures"])