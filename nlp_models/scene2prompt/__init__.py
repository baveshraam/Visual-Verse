"""
Scene-to-Prompt Generator Package - T5 for Structured Visual Descriptions
Supports: English, Tamil, Malayalam, Kannada, Telugu, Hindi
"""
import logging

logger = logging.getLogger(__name__)

try:
    from .model import Scene2PromptGenerator, SceneDescription
except ImportError as e:
    logger.warning(f"Could not import Scene2PromptGenerator: {e}")
    Scene2PromptGenerator = None
    SceneDescription = None

__all__ = ['Scene2PromptGenerator', 'SceneDescription']
