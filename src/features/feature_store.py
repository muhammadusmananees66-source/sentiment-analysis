"""
Feature store integration using Feast
"""

from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SentimentFeatureStore:
    """
    Feature store for sentiment analysis using Feast

    Stores and serves features for both training and inference
    """

    def __init__(self, repo_path: str = "feature_repo"):
        self.repo_path = repo_path
        self.fs = None

    def initialize(self):
        """Initialize feature store connection"""
        try:
            self.fs = FeatureStore(repo_path=self.repo_path)
            logger.info("Feature store initialized")
        except Exception as e:
            logger.warning(f"Feature store not configured: {e}")
            self.fs = None

    def get_features_for_inference(self, text: str) -> pd.DataFrame:
        """Get real-time features for inference"""
        # For now, return basic features
        # In production, this would query from the feature store
        return pd.DataFrame([{
            "text": text,
            "text_length": len(text),
            "word_count": len(text.split())
        }])