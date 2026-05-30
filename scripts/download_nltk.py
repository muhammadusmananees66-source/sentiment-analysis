import nltk
import sys

resources = ['punkt_tab', 'punkt', 'stopwords', 'wordnet', 'omw-1.4']

for resource in resources:
    try:
        print(f'Downloading {resource}...')
        nltk.download(resource, quiet=True)
        print(f'Successfully downloaded {resource}')
    except Exception as e:
        print(f'Failed to download {resource}: {e}')
        sys.exit(1)

print('All NLTK resources downloaded successfully')
