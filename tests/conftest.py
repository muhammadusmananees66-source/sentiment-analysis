"""pytest configuration and fixtures"""
import pytest
import nltk


@pytest.fixture(scope="session", autouse=True)
def setup_nltk():
    """Download required NLTK data once before all tests"""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    yield