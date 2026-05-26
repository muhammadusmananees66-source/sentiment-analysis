"""
Advanced feature engineering for sentiment analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import hashlib

class SentimentFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering for sentiment analysis with:
    - TF-IDF vectorization
    - N-gram extraction (unigrams, bigrams, trigrams)
    - Dimensionality reduction (SVD)
    - Text statistics (length, punctuation count, etc.)
    - Sentiment lexicon features
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        use_tfidf: bool = True,
        use_svd: bool = True,
        svd_components: int = 256,
        use_text_stats: bool = True,
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_tfidf = use_tfidf
        self.use_svd = use_svd
        self.svd_components = svd_components
        self.use_text_stats = use_text_stats
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        # Initialize vectorizer
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=sublinear_tf,
                analyzer='word'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                analyzer='word'
            )

        self.svd = TruncatedSVD(n_components=svd_components, random_state=42)
        self.is_fitted = False

        # Sentiment lexicon (simplified AFINN)
        self.sentiment_lexicon = self._load_sentiment_lexicon()

    def _load_sentiment_lexicon(self) -> Dict[str, int]:
        """Load sentiment lexicon (simplified AFINN)"""
        # Common positive/negative words
        lexicon = {}
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                         'awesome', 'love', 'like', 'enjoy', 'best', 'perfect', 'nice',
                         'happy', 'pleased', 'satisfied', 'recommend']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'hate',
                         'worst', 'poor', 'waste', 'sad', 'angry', 'frustrating', 'annoying',
                         'useless', 'broken', 'fail', 'issue', 'problem']

        for word in positive_words:
            lexicon[word] = 2
        for word in negative_words:
            lexicon[word] = -2

        return lexicon

    def fit(self, X: pd.Series, y=None):
        """Fit vectorizer and SVD"""
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0] if len(X.columns) == 1 else X

        # Fit vectorizer
        self.vectorizer.fit(X)

        # Transform and fit SVD
        if self.use_svd:
            X_vectorized = self.vectorizer.transform(X)
            self.svd.fit(X_vectorized)

        self.is_fitted = True
        return self

    def transform(self, X: pd.Series) -> np.ndarray:
        """Transform text to feature matrix"""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")

        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0] if len(X.columns) == 1 else X

        features_list = []

        # 1. TF-IDF or Count features
        X_vectorized = self.vectorizer.transform(X)

        if self.use_svd:
            X_reduced = self.svd.transform(X_vectorized)
            features_list.append(X_reduced)
        else:
            features_list.append(X_vectorized.toarray())

        # 2. Text statistics features
        if self.use_text_stats:
            text_stats = self._extract_text_statistics(X)
            features_list.append(text_stats)

        # Combine all features
        X_combined = np.hstack(features_list) if len(features_list) > 1 else features_list[0]

        return X_combined

    def _extract_text_statistics(self, X: pd.Series) -> np.ndarray:
        """Extract statistical features from text"""
        stats = []

        for text in X:
            words = text.split()
            text_lower = text.lower()

            features = [
                len(words),                          # word count
                sum(len(w) for w in words) / max(len(words), 1),  # avg word length
                text.count('!'),                     # exclamation count
                text.count('?'),                     # question count
                sum(1 for w in words if w.isupper()), # uppercase word count
                sum(1 for w in words if len(w) > 6),  # long word count
                sum(text_lower.count(w) * s for w, s in self.sentiment_lexicon.items()),  # sentiment score
                text.count('...'),                   # ellipsis count
                text.count('\"') // 2,               # quote count
                len([c for c in text if c.isdigit()]) # digit count
            ]
            stats.append(features)

        return np.array(stats)

    def get_feature_names(self) -> List[str]:
        """Get feature names for explainability"""
        names = []

        if self.use_tfidf:
            names.extend([f"tfidf_{f}" for f in self.vectorizer.get_feature_names_out()])
        else:
            names.extend([f"count_{f}" for f in self.vectorizer.get_feature_names_out()])

        if self.use_text_stats:
            stat_names = ['word_count', 'avg_word_length', 'exclamation_count', 'question_count',
                         'uppercase_count', 'long_word_count', 'lexicon_score', 'ellipsis_count',
                         'quote_count', 'digit_count']
            names.extend(stat_names)

        return names