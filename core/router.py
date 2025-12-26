"""
Pipeline Router Module
Routes classified text to the appropriate pipeline (Comic or Mind-Map).
"""
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

from .classifier import TextClassifier, TextType, ClassificationResult


@dataclass
class RouteResult:
    """Result of pipeline routing."""
    pipeline: str
    classification: ClassificationResult
    text: str
    
    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "classification": self.classification.to_dict(),
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text
        }


class PipelineRouter:
    """
    Routes text to the appropriate processing pipeline based on classification.
    
    - Narrative text → Comic Pipeline
    - Informational text → Mind-Map Pipeline
    """
    
    PIPELINE_COMIC = "comic"
    PIPELINE_MINDMAP = "mindmap"
    
    def __init__(self, classifier: Optional[TextClassifier] = None):
        """
        Initialize the router.
        
        Args:
            classifier: TextClassifier instance (creates one if not provided)
        """
        self.classifier = classifier or TextClassifier()
        self._comic_pipeline = None
        self._mindmap_pipeline = None
    
    def set_comic_pipeline(self, pipeline: Any) -> None:
        """Set the comic generation pipeline."""
        self._comic_pipeline = pipeline
    
    def set_mindmap_pipeline(self, pipeline: Any) -> None:
        """Set the mind-map generation pipeline."""
        self._mindmap_pipeline = pipeline
    
    def route(self, text: str) -> RouteResult:
        """
        Classify text and determine which pipeline to use.
        
        Args:
            text: Input text to route
            
        Returns:
            RouteResult with pipeline name and classification details
        """
        # Classify the text
        classification = self.classifier.classify(text)
        
        # Determine pipeline
        if classification.text_type == TextType.NARRATIVE:
            pipeline = self.PIPELINE_COMIC
        else:
            pipeline = self.PIPELINE_MINDMAP
        
        return RouteResult(
            pipeline=pipeline,
            classification=classification,
            text=text
        )
    
    def route_and_process(self, text: str) -> Dict[str, Any]:
        """
        Route text and process through the appropriate pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with pipeline result and metadata
        """
        route_result = self.route(text)
        
        result = {
            "routing": route_result.to_dict(),
            "output": None,
            "success": False,
            "error": None
        }
        
        try:
            if route_result.pipeline == self.PIPELINE_COMIC:
                if self._comic_pipeline is not None:
                    result["output"] = self._comic_pipeline.process(text)
                    result["success"] = True
                else:
                    result["error"] = "Comic pipeline not configured"
                    
            elif route_result.pipeline == self.PIPELINE_MINDMAP:
                if self._mindmap_pipeline is not None:
                    result["output"] = self._mindmap_pipeline.process(text)
                    result["success"] = True
                else:
                    result["error"] = "Mind-map pipeline not configured"
                    
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def get_pipeline_for_text(self, text: str) -> str:
        """
        Get the pipeline name for given text without full classification details.
        
        Args:
            text: Input text
            
        Returns:
            Pipeline name string
        """
        return self.route(text).pipeline
    
    def force_pipeline(self, text: str, pipeline: str) -> RouteResult:
        """
        Force text to a specific pipeline (bypassing classification).
        
        Args:
            text: Input text
            pipeline: Pipeline to force ("comic" or "mindmap")
            
        Returns:
            RouteResult with forced pipeline
        """
        if pipeline not in [self.PIPELINE_COMIC, self.PIPELINE_MINDMAP]:
            raise ValueError(f"Unknown pipeline: {pipeline}")
        
        # Still classify for the metadata
        classification = self.classifier.classify(text)
        
        return RouteResult(
            pipeline=pipeline,
            classification=classification,
            text=text
        )


# Example usage
if __name__ == "__main__":
    router = PipelineRouter()
    
    # Test routing
    narrative = "Once upon a time, a brave knight said 'I will save the kingdom!'"
    informational = "Machine learning is a subset of AI that uses algorithms to learn from data."
    
    print("Narrative text routing:")
    result1 = router.route(narrative)
    print(f"  Pipeline: {result1.pipeline}")
    print(f"  Confidence: {result1.classification.confidence:.2%}")
    
    print("\nInformational text routing:")
    result2 = router.route(informational)
    print(f"  Pipeline: {result2.pipeline}")
    print(f"  Confidence: {result2.classification.confidence:.2%}")
