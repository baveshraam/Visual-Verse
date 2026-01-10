# VisualVerse 🎨⚡

> **AI-Powered Text-to-Visual Transformation** — Automatically convert text into **Comic Strips** or **Mind-Maps** using deep learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI Text Classification** | Fine-tuned DistilBERT classifier with 100% validation accuracy |
| 📖 **Comic Generation** | Local GPU-powered image generation using Stable Diffusion |
| 🗺️ **Mind-Map Generation** | Interactive concept graphs with semantic relationship extraction |
| ⚡ **Auto Mode** | AI automatically routes text to the optimal visualization |
| 🎨 **Premium UI** | Futuristic glassmorphism design with animations |

---

## 🧠 AI Classification System

### DistilBERT Classifier
- **Model**: Fine-tuned `distilbert-base-uncased`
- **Training Data**: 
  - Indic folktales (Tamil/Malayalam → English translations)
  - WikiHow articles
  - arXiv abstracts
- **Validation Accuracy**: **100%** (F1: 1.0000)
- **Labels**: `narrative` → Comic | `informational` → Mind-Map

### Fallback System
If the ML model fails to load, the system automatically falls back to a rule-based classifier using:
- POS tag distribution analysis
- Named entity recognition
- Dialogue pattern detection
- Domain-specific marker words

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/baveshraam/Visual-Verse.git
cd Visual-Verse
pip install -r requirements.txt
```

### 2. Download NLP Models

```bash
python models/download_models.py
# Or manually:
python -m spacy download en_core_web_sm
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥️ GPU Setup (For Image Generation)

VisualVerse uses local GPU for comic panel generation:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

**Supported Models:**
- `dreamshaper` - Best for comic art (default)
- `cartoon` - Anime/cartoon style
- `sd15` - Classic Stable Diffusion 1.5

---

## 📁 Project Structure

```
Visual-Verse/
├── app.py                          # Streamlit application (main entry)
├── config.py                       # Configuration settings
├── requirements.txt                # Dependencies
│
├── core/
│   ├── classifier.py               # Rule-based text classifier
│   └── router.py                   # Pipeline routing logic
│
├── nlp_models/
│   ├── base.py                     # Base class for trainable models
│   └── classifier/
│       ├── model.py                # DomainClassifier (DistilBERT)
│       ├── train.py                # Training script
│       └── dataset.py              # Dataset utilities
│
├── models/nlp_models/classifier/
│   └── checkpoint/
│       ├── best/                   # Best validation checkpoint
│       └── final/                  # Final trained model
│           ├── model.pt            # PyTorch weights (267 MB)
│           ├── config.json         # Model configuration
│           └── vocab.txt           # Tokenizer vocabulary
│
├── pipelines/
│   ├── comic/                      # Comic generation pipeline
│   │   ├── segmenter.py            # Story → scenes
│   │   ├── extractor.py            # Scene details extraction
│   │   ├── prompt_builder.py       # Image prompt construction
│   │   ├── image_generator.py      # Stable Diffusion generation
│   │   └── layout.py               # Comic strip layout
│   │
│   └── mindmap/                    # Mind-map generation pipeline
│       ├── keyphrase.py            # Keyphrase extraction
│       ├── relation_extractor.py   # Semantic relationships
│       ├── graph_builder.py        # Concept graph construction
│       └── visualizer.py           # Interactive visualization
│
├── api/
│   └── routes.py                   # FastAPI endpoints
│
├── data/classifier/
│   └── train.json                  # Training dataset
│
├── reports/
│   └── classifier_training_report.txt  # Training metrics
│
├── scripts/
│   └── create_classifier_dataset.py    # Dataset creation utility
│
└── tests/
    ├── test_classifier.py          # Classifier tests
    ├── test_comic_pipeline.py      # Comic pipeline tests
    └── test_mindmap_pipeline.py    # Mind-map pipeline tests
```

---

## 🔌 API Endpoints

Run the API server:
```bash
uvicorn api.routes:app --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/classify` | POST | Classify text as narrative/informational |
| `/api/process` | POST | Auto-route and generate visualization |
| `/api/comic` | POST | Generate comic strip from narrative |
| `/api/mindmap` | POST | Generate mind-map from informational text |

API Docs: `http://localhost:8000/docs`

---

## 🏋️ Training the Classifier

To retrain the DistilBERT classifier:

```bash
# 1. Prepare dataset
python scripts/create_classifier_dataset.py

# 2. Train model
python nlp_models/classifier/train.py

# Output: models/nlp_models/classifier/checkpoint/final/
```

**Training Configuration:**
- Epochs: 3
- Batch Size: 16
- Learning Rate: 2e-5
- Max Length: 512 tokens

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Deep Learning** | PyTorch, Transformers, DistilBERT |
| **NLP** | spaCy, NLTK, Sentence-Transformers |
| **Keyphrase Extraction** | KeyBERT, YAKE |
| **Image Generation** | Stable Diffusion, DreamShaper |
| **Visualization** | NetworkX, PyVis |
| **Backend** | FastAPI, Python 3.9+ |
| **Frontend** | Streamlit |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Classification Accuracy | 100% |
| Classification F1 Score | 1.0000 |
| GPU Inference Time | ~20-50ms |
| Model Size | 267 MB |

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Bavesh Raam**
- GitHub: [@baveshraam](https://github.com/baveshraam)

---

<p align="center">
  <strong>⚡ VisualVerse — Transform Words into Worlds ⚡</strong>
</p>
