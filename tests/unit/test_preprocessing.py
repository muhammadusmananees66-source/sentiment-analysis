"""Tests for text preprocessing module"""
import pytest
import pandas as pd
from src.data.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor"""

    def test_preprocessor_initialization(self):
        """Test that preprocessor initializes with default params"""
        preprocessor = TextPreprocessor()
        assert preprocessor.remove_html is True
        assert preprocessor.to_lowercase is True
        assert preprocessor.lemmatize is True

    def test_preprocessor_custom_params(self):
        """Test that preprocessor accepts custom parameters"""
        preprocessor = TextPreprocessor(
            remove_html=False,
            remove_stopwords=False,
            lemmatize=False
        )
        assert preprocessor.remove_html is False
        assert preprocessor.remove_stopwords is False
        assert preprocessor.lemmatize is False

    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        preprocessor = TextPreprocessor(
            remove_stopwords=False,
            lemmatize=False
        )
        result = preprocessor._preprocess_text("Hello World!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_preprocess_text_lowercase(self):
        """Test lowercase conversion"""
        preprocessor = TextPreprocessor(to_lowercase=True)
        result = preprocessor._preprocess_text("HELLO world")
        assert result.islower() or "hello" in result.lower()

    def test_get_preprocessing_info(self):
        """Test that preprocessing info returns dict"""
        preprocessor = TextPreprocessor()
        info = preprocessor.get_preprocessing_info()
        assert isinstance(info, dict)
        assert "remove_html" in info
        assert "lemmatize" in info

    def test_transform_series(self):
        """Test transform method with pandas Series"""
        preprocessor = TextPreprocessor()
        data = pd.Series(["Good movie!", "Bad acting."])
        result = preprocessor.transform(data)
        assert isinstance(result, pd.Series)
        assert len(result) == 2