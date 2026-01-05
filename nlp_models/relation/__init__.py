"""
Relation Extractor Package - SciBERT for Knowledge Triple Extraction
Supports: English, Tamil, Malayalam, Kannada, Telugu, Hindi
"""
import logging

logger = logging.getLogger(__name__)

try:
    from .model import RelationExtractor, Triple
except ImportError as e:
    logger.warning(f"Could not import RelationExtractor: {e}")
    RelationExtractor = None
    Triple = None

__all__ = ['RelationExtractor', 'Triple']
