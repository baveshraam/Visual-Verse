"""
Classifier Package - Multilingual-Aware Domain Classifier
Supports: English, Tamil, Malayalam, Kannada, Telugu, Hindi
"""
import logging

logger = logging.getLogger(__name__)

try:
    from .model import DomainClassifier
except ImportError as e:
    logger.warning(f"Could not import DomainClassifier: {e}")
    DomainClassifier = None

try:
    from .dataset import ClassifierDataset, MultilingualClassifierDataset
except ImportError as e:
    logger.warning(f"Could not import datasets: {e}")
    ClassifierDataset = None
    MultilingualClassifierDataset = None

__all__ = ['DomainClassifier', 'ClassifierDataset', 'MultilingualClassifierDataset']
