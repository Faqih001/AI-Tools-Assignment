import subprocess
import sys
import ssl
import os

def setup_ssl_context():
    """Setup SSL context for downloads"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

def download_spacy_model():
    """Download spaCy English model if not already present"""
    try:
        import spacy
        try:
            # Try to load the model
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model already available")
            return True
        except OSError:
            # Model not found, download it
            print("📥 Downloading spaCy English model...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("✅ spaCy model downloaded successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to download spaCy model: {e}")
                return False
    except ImportError:
        print("❌ spaCy not installed, skipping model download")
        return False

def download_textblob_corpora():
    """Download TextBlob corpora and NLTK data if not already present"""
    try:
        import nltk
        import textblob
        
        # Setup SSL context for NLTK downloads
        setup_ssl_context()
        
        print("📥 Downloading NLTK/TextBlob data...")
        data_packages = ['punkt', 'brown', 'vader_lexicon', 'stopwords']
        
        for package in data_packages:
            try:
                if package == 'punkt':
                    nltk.data.find(f'tokenizers/{package}')
                elif package in ['brown', 'stopwords']:
                    nltk.data.find(f'corpora/{package}')
                elif package == 'vader_lexicon':
                    nltk.data.find(f'vader_lexicon/{package}')
                print(f"✅ {package} already available")
            except LookupError:
                try:
                    print(f"📥 Downloading {package}...")
                    nltk.download(package, quiet=True)
                    print(f"✅ {package} downloaded successfully")
                except Exception as e:
                    print(f"⚠️ Could not download {package}: {e}")
        
        print("✅ NLTK data download completed")
        return True
    except ImportError:
        print("❌ TextBlob/NLTK not installed, skipping corpora download")
        return False
    except Exception as e:
        print(f"⚠️ Error during NLTK download: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up AI Tools Assignment models...")
    spacy_success = download_spacy_model()
    textblob_success = download_textblob_corpora()
    
    if spacy_success and textblob_success:
        print("🎉 All models setup successfully!")
    else:
        print("⚠️ Some models failed to download, but the app will work with fallbacks.")
