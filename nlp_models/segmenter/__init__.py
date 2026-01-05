"""
Scene Segmenter Package - BERT-CRF for Story Scene Boundary Detection
Supports: English, Tamil, Malayalam, Kannada, Telugu, Hindi
"""
import logging

logger = logging.getLogger(__name__)

try:
    from .model import SceneSegmenter
except ImportError as e:
    logger.warning(f"Could not import SceneSegmenter: {e}")
    SceneSegmenter = None

try:
    from .dataset import SegmenterDataset
except ImportError as e:
    logger.warning(f"Could not import SegmenterDataset: {e}")
    SegmenterDataset = None

__all__ = ['SceneSegmenter', 'SegmenterDataset']
