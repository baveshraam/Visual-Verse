# VisualVerse - How to Run

## Quick Start

```bash
# 1. Navigate to project directory
cd c:\Bavesh\Sem6\NLP\Visual-Verse

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Node.js dependencies  
cd desktop-app
npm install

# 4. Run the application
npm run electron:dev
```

---

## Prerequisites

### Required Software
- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **Git** (for cloning)

### Optional (for better performance)
- **CUDA 11.8+** for GPU acceleration
- **8GB+ RAM** recommended

---

## Installation Steps

### Step 1: Install Python Dependencies

```bash
cd c:\Bavesh\Sem6\NLP\Visual-Verse
pip install -r requirements.txt
```

Key packages installed:
- `transformers` - Hugging Face transformer models
- `torch` - PyTorch deep learning framework
- `langdetect` - Language detection
- `spacy` - NLP processing
- `sentence-transformers` - Semantic embeddings

### Step 2: Install spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### Step 3: Install Node Dependencies

```bash
cd desktop-app
npm install
```

---

## Running the Application

### Option 1: Full Desktop App (Recommended)

```bash
cd desktop-app
npm run electron:dev
```

This starts:
- **Backend** on `http://localhost:8000`
- **Frontend** on `http://localhost:5173`
- **Electron** desktop window

### Option 2: Backend Only

```bash
cd c:\Bavesh\Sem6\NLP\Visual-Verse
python -m uvicorn api.routes:app --reload --port 8000
```

### Option 3: Frontend Only (for development)

```bash
cd desktop-app
npm run dev
```

---

## Supported Languages

VisualVerse supports **22 Indian languages** via IndicTrans2:

| Language | Code | Script |
|----------|------|--------|
| Hindi | hi | देवनागरी |
| Bengali | bn | বাংলা |
| Tamil | ta | தமிழ் |
| Telugu | te | తెలుగు |
| Marathi | mr | मराठी |
| Gujarati | gu | ગુજરાતી |
| Kannada | kn | ಕನ್ನಡ |
| Malayalam | ml | മലയാളം |
| Odia | or | ଓଡ଼ିଆ |
| Punjabi | pa | ਪੰਜਾਬੀ |
| Urdu | ur | اردو |
| Assamese | as | অসমীয়া |
| Nepali | ne | नेपाली |
| Sanskrit | sa | संस्कृतम् |
| + 8 more | ... | ... |

---

## Troubleshooting

### "Backend not connected"
```bash
# Start backend manually
python -m uvicorn api.routes:app --reload --port 8000
```

### "GPU not detected"
- Install CUDA toolkit from NVIDIA
- Reinstall PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "langdetect not installed"
```bash
pip install langdetect
```

### "IndicTransToolkit not installed"  
```bash
pip install IndicTransToolkit
```

### No comic generated
- Check backend logs for errors
- Ensure spaCy model is installed
- Try with simpler/shorter text

---

## Debugging

### Log Files

All API calls and pipeline steps are logged to files in the `logs/` directory:

| File | Contains |
|------|----------|
| `logs/api.log` | All API calls with timestamps |
| `logs/pipeline.log` | Step-by-step pipeline execution |
| `logs/error.log` | All errors with stack traces |

### View Logs in Real-Time

```bash
# Windows PowerShell
Get-Content logs/pipeline.log -Wait -Tail 50

# Or just open the file
notepad logs/pipeline.log
```

### Pipeline Steps Logged

**Comic Pipeline:**
1. SEGMENT - Segmenting story into panels
2. EXTRACT - Extracting scene details
3. PROMPTS - Building image generation prompts
4. GENERATE - Generating images
5. LAYOUT - Creating comic strip layout

**Mindmap Pipeline:**
1. KEYPHRASE - Extracting keyphrases
2. RELATIONS - Finding relationships
3. GRAPH - Building knowledge graph
4. VISUALIZE - Creating visualization

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/classify` | POST | Classify text type |
| `/api/comic` | POST | Generate comic |
| `/api/mindmap` | POST | Generate mind-map |
| `/api/process` | POST | Auto-route processing |

---

## Project Structure

```
Visual-Verse/
├── api/routes.py          # FastAPI backend
├── core/                   # Core NLP modules
├── pipelines/              # Comic & Mind-map pipelines
├── nlp_models/             # 5 trainable NLP models
├── utils/                  # Utilities (multilingual processor)
├── desktop-app/            # React + Electron frontend
└── requirements.txt        # Python dependencies
```

---

## Contact

For issues, check the logs or create an issue in the repository.
