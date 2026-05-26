"""
Advanced text preprocessing for sentiment analysis
"""

import re
import nltk
import emoji
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Callable
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
import contractions

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive text preprocessing pipeline for sentiment analysis

    Features:
    - HTML/URL removal
    - Contraction expansion (don't -> do not)
    - Emoji conversion to text
    - Lowercasing and punctuation removal
    - Stopword removal (configurable)
    - Stemming or Lemmatization
    - Numeric replacement options
    """

    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = True,
        expand_contractions: bool = True,
        convert_emoji: bool = True,
        to_lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        lemmatize: bool = True,  # False = use stemming
        min_word_length: int = 2,
        max_text_length: Optional[int] = 512
    ):
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.expand_contractions = expand_contractions
        self.convert_emoji = convert_emoji
        self.to_lowercase = to_lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.custom_stopwords = custom_stopwords or []
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        self.max_text_length = max_text_length

        # Initialize tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) | set(self.custom_stopwords)

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Apply preprocessing to text series"""
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0] if len(X.columns) == 1 else X

        return X.apply(self._preprocess_text)

    def _preprocess_text(self, text: str) -> str:
        """Apply all preprocessing steps"""
        if not isinstance(text, str):
            text = str(text)

        # 1. HTML removal
        if self.remove_html:
            text = BeautifulSoup(text, "html.parser").get_text()

        # 2. URL removal
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # 3. Contraction expansion
        if self.expand_contractions:
            text = contractions.fix(text)

        # 4. Emoji conversion
        if self.convert_emoji:
            text = emoji.demojize(text, delimiters=(" ", " "))
            # Clean up emoji names (remove underscores)
            text = re.sub(r':([a-z_]+):', lambda m: m.group(1).replace('_', ' '), text)

        # 5. Lowercase
        if self.to_lowercase:
            text = text.lower()

        # 6. Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        # 7. Remove numbers (optional)
        if self.remove_numbers:
            text = re.sub(r'\d+', ' ', text)

        # 8. Tokenize
        tokens = nltk.word_tokenize(text)

        # 9. Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # 10. Remove short words
        tokens = [t for t in tokens if len(t) >= self.min_word_length]

        # 11. Stemming or Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        else:
            tokens = [self.stemmer.stem(t) for t in tokens]

        # 12. Rejoin
        text = ' '.join(tokens)

        # 13. Truncate if needed
        if self.max_text_length and len(text) > self.max_text_length:
            text = text[:self.max_text_length]

        return text.strip()

    def get_preprocessing_info(self) -> dict:
        """Return preprocessing configuration for tracking"""
        return {
            "remove_html": self.remove_html,
            "remove_urls": self.remove_urls,
            "expand_contractions": self.expand_contractions,
            "convert_emoji": self.convert_emoji,
            "to_lowercase": self.to_lowercase,
            "remove_punctuation": self.remove_punctuation,
            "remove_stopwords": self.remove_stopwords,
            "stopword_count": len(self.stop_words),
            "lemmatize": self.lemmatize
        }