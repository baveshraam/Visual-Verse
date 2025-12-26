"""
VisualVerse Configuration Settings
"""
import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = BASE_DIR / "models" / "cached"

# NLP Model Settings
SPACY_MODEL = "en_core_web_sm"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# Classification Thresholds
NARRATIVE_THRESHOLD = 0.6  # Score above this = Narrative, below = Informational

# Narrative Markers (words indicating story-like content)
NARRATIVE_MARKERS = [
    "once", "upon", "suddenly", "then", "finally", "meanwhile",
    "later", "afterward", "next", "eventually", "soon",
    "said", "whispered", "shouted", "replied", "asked",
    "walked", "ran", "looked", "felt", "thought"
]

# Informational Markers (words indicating factual content)
INFORMATIONAL_MARKERS = [
    "defined", "refers", "consists", "includes", "contains",
    "therefore", "however", "because", "consequently", "thus",
    "according", "research", "study", "analysis", "data",
    "process", "system", "method", "approach", "technique"
]

# Comic Pipeline Settings
MAX_SCENES_PER_COMIC = 6
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# Mind-Map Settings
MAX_KEYWORDS = 8  # Reduced for cleaner mind maps
MIN_KEYWORD_SCORE = 0.3

# Image Generation API (placeholder - configure with actual API)
IMAGE_API_URL = os.getenv("IMAGE_API_URL", "http://localhost:7860/sdapi/v1/txt2img")
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "")

# Stable Diffusion Default Settings
SD_DEFAULT_PARAMS = {
    "steps": 30,
    "cfg_scale": 7.5,
    "width": IMAGE_WIDTH,
    "height": IMAGE_HEIGHT,
    "sampler_name": "DPM++ 2M Karras"
}
