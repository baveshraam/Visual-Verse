# VisualVerse 🎨

A dual-mode NLP system that automatically converts text into **Comic Strips** (for narrative content) or **Mind-Maps** (for informational content).

## Features

- **Automatic Text Classification**: Detects whether input is narrative or informational
- **Comic Generation Pipeline**: Converts stories into visual comic panels
- **Mind-Map Generation Pipeline**: Transforms concepts into interactive mind maps
- **Interactive Web Interface**: Streamlit-based UI for easy interaction

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLP Models

```bash
python models/download_models.py
```

Or manually:

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Run Streamlit App

```bash
streamlit run app.py
```

### Run FastAPI Server

```bash
uvicorn api.routes:app --reload
```

Then access the API at `http://localhost:8000/docs`

## Project Structure

```
Visual-Verse/
├── app.py                    # Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Dependencies
│
├── core/
│   ├── classifier.py         # Text classification
│   └── router.py             # Pipeline routing
│
├── pipelines/
│   ├── comic/                # Comic generation
│   │   ├── segmenter.py
│   │   ├── extractor.py
│   │   ├── prompt_builder.py
│   │   ├── image_generator.py
│   │   └── layout.py
│   │
│   └── mindmap/              # Mind-map generation
│       ├── keyphrase.py
│       ├── relation_extractor.py
│       ├── graph_builder.py
│       └── visualizer.py
│
├── api/
│   └── routes.py             # FastAPI endpoints
│
└── models/
    └── download_models.py    # Model downloader
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/classify` | POST | Classify text type |
| `/api/process` | POST | Auto-route and process |
| `/api/comic` | POST | Generate comic strip |
| `/api/mindmap` | POST | Generate mind map |

## Technology Stack

- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **NLP**: SpaCy, NLTK, Sentence-Transformers
- **Keyphrases**: KeyBERT, YAKE
- **Visualization**: NetworkX, PyVis
- **Image**: Pillow, Stable Diffusion (optional)

## License

MIT License
"# Visual-Verse" 
