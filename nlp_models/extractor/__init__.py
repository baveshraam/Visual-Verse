"""
Entity Extractor Package - Custom NER with Coreference Resolution
Supports: English, Tamil, Malayalam, Kannada, Telugu, Hindi
"""
import logging

logger = logging.getLogger(__name__)

try:
    from .model import EntityExtractor, Entity
except ImportError as e:
    logger.warning(f"Could not import EntityExtractor: {e}")
    EntityExtractor = None
    Entity = None

try:
    from .coref import CoreferenceResolver, CoreferenceCluster, Mention
except ImportError as e:
    logger.warning(f"Could not import CoreferenceResolver: {e}")
    CoreferenceResolver = None
    CoreferenceCluster = None
    Mention = None

__all__ = [
    'EntityExtractor', 
    'Entity',
    'CoreferenceResolver', 
    'CoreferenceCluster', 
    'Mention'
]
