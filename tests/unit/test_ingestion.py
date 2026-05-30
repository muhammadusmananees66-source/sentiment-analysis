"""Tests for data ingestion module"""
import pytest
import pandas as pd
from src.data.ingestion import HuggingFaceIngestion, DataIngestionConfig


class TestHuggingFaceIngestion:
    """Test cases for HuggingFaceIngestion"""

    def test_config_creation(self):
        """Test that config can be created"""
        config = DataIngestionConfig(
            dataset_name="imdb",
            split_ratios=(0.7, 0.15, 0.15),
            seed=42
        )
        assert config.dataset_name == "imdb"
        assert config.seed == 42

    def test_supported_datasets_exists(self):
        """Test that SUPPORTED_DATASETS is defined"""
        assert hasattr(HuggingFaceIngestion, 'SUPPORTED_DATASETS')
        assert "imdb" in HuggingFaceIngestion.SUPPORTED_DATASETS

    def test_ingestion_initialization(self):
        """Test that ingestion class initializes"""
        config = DataIngestionConfig(dataset_name="imdb")
        ingestion = HuggingFaceIngestion(config)
        assert ingestion.config == config
        assert ingestion.dataset is None