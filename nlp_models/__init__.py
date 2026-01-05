"""
Multilingual NLP Models Package for VisualVerse
Contains 5 trainable deep learning models for text understanding.

Models:
1. DomainClassifier - DistilBERT for narrative vs informational
2. SceneSegmenter - BERT-CRF for scene boundary detection
3. EntityExtractor - BERT NER with coreference resolution
4. Scene2PromptGenerator - T5 for structured visual descriptions
5. RelationExtractor - SciBERT for knowledge triples

Supported Languages: English, Tamil, Malayalam, Kannada, Telugu, Hindi
(via IndicTrans2 translation preprocessing)
"""
import logging

logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
__author__ = "VisualVerse Team"

# Import base classes
try:
    from .base import BaseNLPModel, ModelConfig, TrainingMetrics
except ImportError as e:
    logger.warning(f"Could not import base classes: {e}")
    BaseNLPModel = None
    ModelConfig = None
    TrainingMetrics = None

# Import models with graceful fallbacks
try:
    from .classifier import DomainClassifier
except ImportError as e:
    logger.warning(f"DomainClassifier not available: {e}")
    DomainClassifier = None

try:
    from .segmenter import SceneSegmenter
except ImportError as e:
    logger.warning(f"SceneSegmenter not available: {e}")
    SceneSegmenter = None

try:
    from .extractor import EntityExtractor
except ImportError as e:
    logger.warning(f"EntityExtractor not available: {e}")
    EntityExtractor = None

try:
    from .scene2prompt import Scene2PromptGenerator
except ImportError as e:
    logger.warning(f"Scene2PromptGenerator not available: {e}")
    Scene2PromptGenerator = None

try:
    from .relation import RelationExtractor
except ImportError as e:
    logger.warning(f"RelationExtractor not available: {e}")
    RelationExtractor = None

# Export all
__all__ = [
    'BaseNLPModel',
    'ModelConfig', 
    'TrainingMetrics',
    'DomainClassifier',
    'SceneSegmenter',
    'EntityExtractor',
    'Scene2PromptGenerator',
    'RelationExtractor',
]


def get_available_models():
    """Return list of available model classes."""
    models = {}
    if DomainClassifier:
        models['classifier'] = DomainClassifier
    if SceneSegmenter:
        models['segmenter'] = SceneSegmenter
    if EntityExtractor:
        models['extractor'] = EntityExtractor
    if Scene2PromptGenerator:
        models['scene2prompt'] = Scene2PromptGenerator
    if RelationExtractor:
        models['relation'] = RelationExtractor
    return models


def check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append('torch')
    
    try:
        import transformers
    except ImportError:
        missing.append('transformers')
    
    try:
        import sklearn
    except ImportError:
        missing.append('scikit-learn')
    
    if missing:
        logger.warning(f"Missing dependencies: {', '.join(missing)}")
        logger.warning("Install with: pip install " + " ".join(missing))
        return False
    
    return True
