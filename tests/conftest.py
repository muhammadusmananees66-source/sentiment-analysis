"""pytest configuration file"""
import sys
from pathlib import Path
import pytest
import nltk
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Disable NLTK download logs in CI
logging.getLogger('nltk').setLevel(logging.ERROR)


@pytest.fixture(scope="session", autouse=True)
def setup_nltk():
    """Download required NLTK data once before all tests with CI handling"""
    
    resources = ['punkt_tab', 'punkt', 'stopwords', 'wordnet', 'omw-1.4']
    
    for resource in resources:
        try:
            # Check if already downloaded
            if resource in ['punkt_tab', 'punkt']:
                nltk.data.find(f'tokenizers/{resource}')
            elif resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'wordnet':
                nltk.data.find('tokenizers/wordnet')
            else:
                nltk.data.find(resource)
        except LookupError:
            try:
                # Download with quiet mode and timeout
                nltk.download(resource, quiet=True, raise_on_error=False)
            except Exception:
                # Skip this resource if download fails (CI environment)
                pass
    
    yield