"""
Production-grade data ingestion from Hugging Face with:
- Automatic dataset downloading/caching
- Data validation
- Train/validation/test splitting
- Streaming support for large datasets
"""

import logging  # Used to record events, errors, and info messages while running code (helps debugging and monitoring).
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass   # Used to create simple classes for storing data without writing boilerplate code like __init__.
from datetime import datetime
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split    # Used for tracking ML experiments, logging metrics, and managing model versions in MLOps pipelines.
import mlflow

logger = logging.getLogger(__name__)

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    dataset_name: str = "imdb"  # Hugging Face dataset identifier
    subset: Optional[str] = None  # Subset for datasets with configs
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)  # train/val/test
    max_samples: Optional[int] = None  # Limit for testing
    text_column: str = "text"
    label_column: str = "label"
    seed: int = 42
    streaming: bool = False  # Use streaming for large datasets
    cache_dir: str = "/tmp/huggingface_cache"

class HuggingFaceIngestion:
    """
    Enterprise data ingestion from Hugging Face Hub

    Supports:
    - IMDB, SST-2, Twitter Sentiment, Amazon Reviews, etc.
    - Streaming for datasets > memory
    - Automatic versioning with MLflow
    """

    SUPPORTED_DATASETS = {
        "imdb": {"text": "text", "label": "label", "type": "binary"},
        "sst2": {"text": "sentence", "label": "label", "type": "binary"},
        "twitter_sentiment": {"text": "text", "label": "label", "type": "binary"},
        "amazon_polarity": {"text": "content", "label": "label", "type": "binary"},
        "tweet_eval": {"subset": "sentiment", "text": "text", "label": "label", "type": "binary"},
    }

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.dataset = None
        self.metadata = {}

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load dataset from Hugging Face and prepare splits

        Returns:
            Dict with 'train', 'validation', 'test' DataFrames
        """
        logger.info(f"Loading dataset: {self.config.dataset_name}")

        with mlflow.start_run(run_name="data_ingestion", nested=True):
            # Load from Hugging Face
            dataset_kwargs = {
                "path": self.config.dataset_name,
                "cache_dir": self.config.cache_dir,
                "streaming": self.config.streaming
            }

            if self.config.subset:
                dataset_kwargs["name"] = self.config.subset
            elif self.config.dataset_name in self.SUPPORTED_DATASETS:
                subset = self.SUPPORTED_DATASETS[self.config.dataset_name].get("subset")
                if subset:
                    dataset_kwargs["name"] = subset

            # Load dataset
            raw_dataset = load_dataset(**dataset_kwargs)

            # Handle different split structures
            splits = self._extract_splits(raw_dataset)

            # Convert to DataFrames
            dataframes = {}
            for split_name, dataset_split in splits.items():
                if self.config.streaming:
                    # Convert streaming to pandas (limit if needed)
                    df = self._streaming_to_dataframe(dataset_split)
                else:
                    df = dataset_split.to_pandas()

                # Validate columns
                text_col = self._get_text_column()
                label_col = self._get_label_column()

                if text_col not in df.columns:
                    raise ValueError(f"Text column '{text_col}' not found. Available: {df.columns}")

                # Rename to standard names
                df = df.rename(columns={
                    text_col: "text",
                    label_col: "label"
                })

                # Basic cleaning
                df = self._clean_dataframe(df)

                dataframes[split_name] = df

            # If only one split provided, create train/val/test splits
            if len(dataframes) == 1 and "train" in dataframes:
                dataframes = self._create_splits(dataframes["train"])

            # Apply max_samples limit if specified
            if self.config.max_samples:
                for split in dataframes:
                    dataframes[split] = dataframes[split].head(self.config.max_samples)

            # Log metadata
            self._log_ingestion_metadata(dataframes)

            return dataframes

    def _extract_splits(self, dataset: DatasetDict) -> Dict[str, Any]:
        """Extract available splits from HF dataset"""
        splits = {}

        # Common split names
        possible_splits = ["train", "validation", "test", "val"]

        for split in possible_splits:
            if split in dataset:
                splits[split] = dataset[split]

        # If only 'train' exists, it might contain all data
        if not splits and "train" in dataset:
            splits["train"] = dataset["train"]

        return splits

    def _create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/val/test splits from single DataFrame"""
        train_ratio, val_ratio, test_ratio = self.config.split_ratios

        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=test_ratio,
            random_state=self.config.seed,
            stratify=df.get("label", None)
        )

        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            random_state=self.config.seed,
            stratify=train_val.get("label", None)
        )

        return {"train": train, "validation": val, "test": test}

    def _streaming_to_dataframe(self, streaming_dataset, limit: int = 100000) -> pd.DataFrame:
        """Convert streaming dataset to pandas with memory limit"""
        data = []
        for i, sample in enumerate(streaming_dataset):
            if i >= limit:
                break
            data.append(sample)
        return pd.DataFrame(data)

    def _get_text_column(self) -> str:
        """Get text column name for current dataset"""
        if self.config.dataset_name in self.SUPPORTED_DATASETS:
            return self.SUPPORTED_DATASETS[self.config.dataset_name]["text"]
        return self.config.text_column

    def _get_label_column(self) -> str:
        """Get label column name for current dataset"""
        if self.config.dataset_name in self.SUPPORTED_DATASETS:
            return self.SUPPORTED_DATASETS[self.config.dataset_name]["label"]
        return self.config.label_column

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning"""
        # Handle missing values
        df = df.dropna(subset=["text"])

        # Convert labels to integer if needed
        if "label" in df.columns:
            # Handle string labels (positive/negative -> 1/0)
            if df["label"].dtype == "object":
                label_map = {
                    "positive": 1, "pos": 1, "POS": 1,
                    "negative": 0, "neg": 0, "NEG": 0,
                    "neutral": 2
                }
                df["label"] = df["label"].map(label_map).fillna(0).astype(int)
            else:
                df["label"] = df["label"].astype(int)

        return df

    def _log_ingestion_metadata(self, dataframes: Dict[str, pd.DataFrame]):
        """Log metadata to MLflow"""
        for split_name, df in dataframes.items():
            mlflow.log_metric(f"{split_name}_samples", len(df))
            if "label" in df.columns:
                positive_pct = (df["label"] == 1).mean() * 100
                mlflow.log_metric(f"{split_name}_positive_pct", positive_pct)

        mlflow.log_param("dataset_name", self.config.dataset_name)
        mlflow.log_param("streaming", self.config.streaming)

        self.metadata = {
            "ingestion_timestamp": datetime.now().isoformat(),
            "dataset": self.config.dataset_name,
            "splits": {k: len(v) for k, v in dataframes.items()}
        }