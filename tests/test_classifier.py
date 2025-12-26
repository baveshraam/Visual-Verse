"""
Test suite for the Text Classifier module.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.classifier import TextClassifier, TextType, ClassificationResult


class TestTextClassifier:
    """Test cases for TextClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return TextClassifier(use_semantic=False)
    
    def test_classifier_initialization(self, classifier):
        """Test that classifier initializes correctly."""
        assert classifier is not None
        assert classifier.use_semantic == False
    
    def test_classify_narrative_text(self, classifier):
        """Test classification of narrative/story text."""
        narrative_text = """
        Once upon a time, in a small village, there lived a young girl named Maya.
        She had always dreamed of adventure. One day, she said "I will explore!"
        She walked into the forest and suddenly found a hidden path.
        """
        
        result = classifier.classify(narrative_text)
        
        assert isinstance(result, ClassificationResult)
        assert result.text_type == TextType.NARRATIVE
        assert result.narrative_score > result.informational_score
        assert 0 <= result.confidence <= 1
    
    def test_classify_informational_text(self, classifier):
        """Test classification of informational/factual text."""
        informational_text = """
        Machine learning is a subset of artificial intelligence.
        It includes supervised learning, unsupervised learning, and reinforcement learning.
        Deep learning uses neural networks to process complex data patterns.
        Therefore, these techniques are essential for modern AI systems.
        """
        
        result = classifier.classify(informational_text)
        
        assert isinstance(result, ClassificationResult)
        assert result.text_type == TextType.INFORMATIONAL
        assert result.informational_score > result.narrative_score
    
    def test_extract_linguistic_features(self, classifier):
        """Test feature extraction."""
        text = "The quick brown fox jumped over the lazy dog."
        
        features = classifier.extract_linguistic_features(text)
        
        assert isinstance(features, dict)
        assert 'word_count' in features
        assert 'narrative_marker_ratio' in features
        assert 'informational_marker_ratio' in features
    
    def test_classification_result_to_dict(self, classifier):
        """Test result serialization."""
        text = "Once upon a time, a knight saved the kingdom."
        
        result = classifier.classify(text)
        result_dict = result.to_dict()
        
        assert 'text_type' in result_dict
        assert 'confidence' in result_dict
        assert 'features' in result_dict
    
    def test_classify_batch(self, classifier):
        """Test batch classification."""
        texts = [
            "Once upon a time in a kingdom far away...",
            "The process consists of three main steps."
        ]
        
        results = classifier.classify_batch(texts)
        
        assert len(results) == 2
        assert all(isinstance(r, ClassificationResult) for r in results)
    
    def test_empty_text_handling(self, classifier):
        """Test handling of empty or minimal text."""
        result = classifier.classify("Short text")
        
        assert isinstance(result, ClassificationResult)
        # Should still produce a result even for short text


class TestClassificationResult:
    """Test ClassificationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a classification result."""
        result = ClassificationResult(
            text_type=TextType.NARRATIVE,
            confidence=0.85,
            narrative_score=0.85,
            informational_score=0.15,
            features={"test": 0.5}
        )
        
        assert result.text_type == TextType.NARRATIVE
        assert result.confidence == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
