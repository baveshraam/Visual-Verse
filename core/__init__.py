"""
VisualVerse Core Module
Contains text classification and routing logic.
"""
from .classifier import TextClassifier
from .router import PipelineRouter

__all__ = ["TextClassifier", "PipelineRouter"]
