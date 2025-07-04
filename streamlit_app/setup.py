import subprocess
import sys
import os

def download_spacy_model():
    """Download spaCy English model if not already present"""
    try:
        import spacy
        try:
            # Try to load the model
            nlp = spacy.load("en_core_web_sm")
            print("spaCy model already available")
        except OSError:
            # Model not found, download it
            print("Downloading spaCy English model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("spaCy model downloaded successfully")
    except ImportError:
        print("spaCy not installed, skipping model download")

def download_textblob_corpora():
    """Download TextBlob corpora if not already present"""
    try:
        import nltk
        import textblob
        print("Downloading TextBlob corpora...")
        nltk.download('punkt', quiet=True)
        nltk.download('brown', quiet=True)
        print("TextBlob corpora downloaded successfully")
    except ImportError:
        print("TextBlob/NLTK not installed, skipping corpora download")

if __name__ == "__main__":
    download_spacy_model()
    download_textblob_corpora()
