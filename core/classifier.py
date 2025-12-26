"""
Text Classifier Module
Determines whether input text is Narrative (story-based) or Informational (concept-based).
Uses a multi-feature hybrid approach combining linguistic and semantic features.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import sys
sys.path.append('..')
from config import (
    SPACY_MODEL,
    SENTENCE_TRANSFORMER_MODEL,
    NARRATIVE_THRESHOLD,
    NARRATIVE_MARKERS,
    INFORMATIONAL_MARKERS
)


class TextType(Enum):
    """Enumeration for text classification types."""
    NARRATIVE = "narrative"
    INFORMATIONAL = "informational"


@dataclass
class ClassificationResult:
    """Result of text classification."""
    text_type: TextType
    confidence: float
    narrative_score: float
    informational_score: float
    features: Dict[str, float]
    
    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "text_type": self.text_type.value,
            "confidence": round(self.confidence, 3),
            "narrative_score": round(self.narrative_score, 3),
            "informational_score": round(self.informational_score, 3),
            "features": {k: round(v, 3) for k, v in self.features.items()}
        }


class TextClassifier:
    """
    Classifies text as Narrative or Informational using linguistic features.
    
    Features analyzed:
    - POS tag distribution (verb/pronoun ratio vs noun ratio)
    - Named Entity types (PERSON vs ORG/GPE)
    - Verb tense (past tense for narrative)
    - Discourse markers (narrative vs informational patterns)
    - Dialogue presence
    - Sentence structure patterns
    """
    
    def __init__(self, use_semantic: bool = True):
        """
        Initialize the text classifier.
        
        Args:
            use_semantic: If True, use sentence transformers for semantic features
        """
        self.use_semantic = use_semantic and TRANSFORMERS_AVAILABLE
        self.nlp = None
        self.sentence_model = None
        
        # Initialize models lazily
        self._models_loaded = False
        
    def _load_models(self):
        """Load NLP models (lazy loading)."""
        if self._models_loaded:
            return
            
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(SPACY_MODEL)
            except OSError:
                print(f"SpaCy model '{SPACY_MODEL}' not found. Run: python -m spacy download {SPACY_MODEL}")
                # Fallback to basic features only
                self.nlp = None
        
        if self.use_semantic and TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
            except Exception as e:
                print(f"Could not load sentence transformer: {e}")
                self.sentence_model = None
        
        self._models_loaded = True
    
    def _count_pos_tags(self, doc) -> Dict[str, int]:
        """Count POS tags in document."""
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        return pos_counts
    
    def _get_pos_ratio(self, doc, pos_tag: str) -> float:
        """Calculate ratio of specific POS tag."""
        if len(doc) == 0:
            return 0.0
        count = sum(1 for token in doc if token.pos_ == pos_tag)
        return count / len(doc)
    
    def _count_entity_type(self, doc, entity_type: str) -> int:
        """Count named entities of specific type."""
        return sum(1 for ent in doc.ents if ent.label_ == entity_type)
    
    def _get_entity_ratio(self, doc, entity_type: str) -> float:
        """Calculate ratio of entity type to total tokens."""
        if len(doc) == 0:
            return 0.0
        return self._count_entity_type(doc, entity_type) / len(doc)
    
    def _count_past_tense(self, doc) -> int:
        """Count past tense verbs (VBD tag)."""
        return sum(1 for token in doc if token.tag_ == 'VBD')
    
    def _get_past_tense_ratio(self, doc) -> float:
        """Calculate ratio of past tense verbs."""
        verb_count = sum(1 for token in doc if token.pos_ == 'VERB')
        if verb_count == 0:
            return 0.0
        return self._count_past_tense(doc) / verb_count
    
    def _count_markers(self, text: str, markers: List[str]) -> int:
        """Count occurrences of marker words in text."""
        text_lower = text.lower()
        count = 0
        for marker in markers:
            count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
        return count
    
    def _get_marker_ratio(self, text: str, markers: List[str]) -> float:
        """Calculate ratio of marker words."""
        words = text.split()
        if len(words) == 0:
            return 0.0
        return self._count_markers(text, markers) / len(words)
    
    def _has_dialogue(self, text: str) -> bool:
        """Check if text contains dialogue patterns."""
        # Check for quoted speech
        has_quotes = bool(re.search(r'["\'][^"\']+["\']', text))
        # Check for speech verbs near quotes
        has_speech_verbs = bool(re.search(
            r'(said|asked|replied|whispered|shouted|exclaimed|muttered)',
            text.lower()
        ))
        return has_quotes or has_speech_verbs
    
    def _get_dialogue_score(self, text: str) -> float:
        """Calculate dialogue presence score."""
        quotes = re.findall(r'["\'][^"\']+["\']', text)
        speech_verbs = re.findall(
            r'\b(said|asked|replied|whispered|shouted|exclaimed|muttered|answered|called)\b',
            text.lower()
        )
        
        # Normalize by text length
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        dialogue_indicators = len(quotes) + len(speech_verbs)
        return min(dialogue_indicators / (len(words) / 50), 1.0)  # Cap at 1.0
    
    def _get_sentence_length_variance(self, text: str) -> float:
        """Calculate variance in sentence lengths (narrative tends to have more variance)."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        lengths = [len(s.split()) for s in sentences]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        
        # Normalize variance
        return min(variance / 100, 1.0)  # Cap at 1.0
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Basic text stats
        words = text.split()
        features['word_count'] = len(words)
        
        # Marker-based features (works without SpaCy)
        features['narrative_marker_ratio'] = self._get_marker_ratio(text, NARRATIVE_MARKERS)
        features['informational_marker_ratio'] = self._get_marker_ratio(text, INFORMATIONAL_MARKERS)
        features['dialogue_score'] = self._get_dialogue_score(text)
        features['sentence_variance'] = self._get_sentence_length_variance(text)
        
        # SpaCy-based features
        if self.nlp is not None:
            doc = self.nlp(text)
            
            # POS ratios
            features['verb_ratio'] = self._get_pos_ratio(doc, 'VERB')
            features['pronoun_ratio'] = self._get_pos_ratio(doc, 'PRON')
            features['noun_ratio'] = self._get_pos_ratio(doc, 'NOUN')
            features['adj_ratio'] = self._get_pos_ratio(doc, 'ADJ')
            
            # Entity ratios
            features['person_entity_ratio'] = self._get_entity_ratio(doc, 'PERSON')
            features['org_entity_ratio'] = self._get_entity_ratio(doc, 'ORG')
            features['gpe_entity_ratio'] = self._get_entity_ratio(doc, 'GPE')
            
            # Tense analysis
            features['past_tense_ratio'] = self._get_past_tense_ratio(doc)
        else:
            # Fallback values when SpaCy is not available
            features['verb_ratio'] = 0.0
            features['pronoun_ratio'] = 0.0
            features['noun_ratio'] = 0.0
            features['adj_ratio'] = 0.0
            features['person_entity_ratio'] = 0.0
            features['org_entity_ratio'] = 0.0
            features['gpe_entity_ratio'] = 0.0
            features['past_tense_ratio'] = 0.0
        
        return features
    
    def _compute_narrative_score(self, features: Dict[str, float]) -> float:
        """
        Compute narrative score from extracted features.
        
        Uses weighted combination of features to determine if text is narrative.
        Higher score = more narrative, Lower score = more informational.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Score between 0 and 1 (higher = more narrative)
        """
        # Feature weights (positive = narrative, negative = informational)
        weights = {
            'narrative_marker_ratio': 3.0,
            'informational_marker_ratio': -3.0,
            'dialogue_score': 2.5,
            'sentence_variance': 1.0,
            'verb_ratio': 1.5,
            'pronoun_ratio': 2.0,
            'noun_ratio': -0.5,
            'adj_ratio': -0.5,
            'person_entity_ratio': 2.0,
            'org_entity_ratio': -1.0,
            'gpe_entity_ratio': -0.5,
            'past_tense_ratio': 2.5,
        }
        
        # Calculate weighted score
        score = 0.5  # Start at neutral
        
        for feature, weight in weights.items():
            if feature in features:
                score += weight * features[feature]
        
        # Clamp score to [0, 1]
        return max(0.0, min(1.0, score))
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text as Narrative or Informational.
        
        Args:
            text: Input text to classify
            
        Returns:
            ClassificationResult with type, confidence, and features
        """
        # Ensure models are loaded
        self._load_models()
        
        # Extract features
        features = self.extract_linguistic_features(text)
        
        # Compute scores
        narrative_score = self._compute_narrative_score(features)
        informational_score = 1.0 - narrative_score
        
        # Determine classification
        if narrative_score >= NARRATIVE_THRESHOLD:
            text_type = TextType.NARRATIVE
            confidence = narrative_score
        else:
            text_type = TextType.INFORMATIONAL
            confidence = informational_score
        
        return ClassificationResult(
            text_type=text_type,
            confidence=confidence,
            narrative_score=narrative_score,
            informational_score=informational_score,
            features=features
        )
    
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of ClassificationResults
        """
        return [self.classify(text) for text in texts]
    
    def get_classification_explanation(self, result: ClassificationResult) -> str:
        """
        Generate human-readable explanation of classification.
        
        Args:
            result: Classification result to explain
            
        Returns:
            Explanation string
        """
        explanation_parts = []
        
        explanation_parts.append(
            f"Text classified as: {result.text_type.value.upper()} "
            f"(confidence: {result.confidence:.1%})"
        )
        
        # Highlight key features
        key_features = []
        
        if result.features.get('dialogue_score', 0) > 0.1:
            key_features.append("presence of dialogue")
        if result.features.get('past_tense_ratio', 0) > 0.3:
            key_features.append("past tense narrative style")
        if result.features.get('narrative_marker_ratio', 0) > 0.02:
            key_features.append("narrative transition words")
        if result.features.get('informational_marker_ratio', 0) > 0.02:
            key_features.append("informational/definitional language")
        if result.features.get('person_entity_ratio', 0) > 0.01:
            key_features.append("character references")
        
        if key_features:
            explanation_parts.append(f"Key indicators: {', '.join(key_features)}")
        
        return "\n".join(explanation_parts)


# Example usage and testing
if __name__ == "__main__":
    classifier = TextClassifier(use_semantic=False)
    
    # Test narrative text
    narrative_text = """
    Once upon a time, in a small village nestled between two mountains, 
    there lived a young girl named Maya. She had always dreamed of exploring 
    the world beyond the peaks. One day, she decided to embark on a journey.
    "I will find what lies beyond," she whispered to herself as she packed her bag.
    The next morning, Maya said goodbye to her family and walked toward the sunrise.
    """
    
    # Test informational text
    informational_text = """
    Machine learning is a subset of artificial intelligence that enables 
    computers to learn from data. It consists of three main types: supervised 
    learning, unsupervised learning, and reinforcement learning. Supervised 
    learning uses labeled data to train models, while unsupervised learning 
    discovers patterns in unlabeled data. Therefore, the choice of approach 
    depends on the available data and the problem being solved.
    """
    
    print("=" * 60)
    print("NARRATIVE TEXT TEST")
    print("=" * 60)
    result1 = classifier.classify(narrative_text)
    print(classifier.get_classification_explanation(result1))
    print(f"Features: {result1.features}")
    
    print("\n" + "=" * 60)
    print("INFORMATIONAL TEXT TEST")
    print("=" * 60)
    result2 = classifier.classify(informational_text)
    print(classifier.get_classification_explanation(result2))
    print(f"Features: {result2.features}")
