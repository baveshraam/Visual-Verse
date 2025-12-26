"""
Model Download Script
Downloads required NLP models for VisualVerse.
"""
import subprocess
import sys


def download_spacy_model():
    """Download SpaCy English model."""
    print("Downloading SpaCy English model...")
    try:
        subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ], check=True)
        print("✅ SpaCy model downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download SpaCy model: {e}")


def download_nltk_data():
    """Download NLTK required data."""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")


def download_sentence_transformers():
    """Pre-download sentence transformer model."""
    print("Downloading Sentence Transformer model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence Transformer model downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download Sentence Transformer: {e}")


def main():
    """Download all required models."""
    print("=" * 60)
    print("VisualVerse Model Downloader")
    print("=" * 60)
    print()
    
    download_spacy_model()
    print()
    
    download_nltk_data()
    print()
    
    download_sentence_transformers()
    print()
    
    print("=" * 60)
    print("Model download complete!")
    print("You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
