"""
Keyphrase Extractor - PROPERLY FIXED VERSION
Extracts MEANINGFUL multi-word technical phrases, not single generic words.
"""
from dataclasses import dataclass
from typing import List, Set
import re

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class Keyphrase:
    """Represents an extracted keyphrase."""
    phrase: str
    score: float
    source: str
    
    def to_dict(self) -> dict:
        return {
            "phrase": self.phrase,
            "score": round(self.score, 4),
            "source": self.source
        }


class KeyphraseExtractor:
    """
    Extracts MEANINGFUL keyphrases - compound terms, not single words.
    
    Example: "Machine Learning" not "Machine" or "Learning" separately.
    """
    
    # Single words that are TOO GENERIC to be useful
    GENERIC_WORDS = {
        'machine', 'learning', 'data', 'model', 'models', 'using',
        'based', 'type', 'types', 'form', 'forms', 'main', 'common',
        'approach', 'approaches', 'method', 'methods', 'technique', 'techniques',
        'system', 'systems', 'process', 'processes', 'used', 'uses',
        'include', 'includes', 'including', 'works', 'working',
        'artificial', 'intelligence', 'network', 'networks',
        'algorithm', 'algorithms', 'task', 'tasks', 'agent', 'agents',
        'environment', 'action', 'actions', 'pattern', 'patterns',
        'layer', 'layers', 'recognition', 'processing', 'computer',
        'computers', 'explicitly', 'programmed', 'labeled', 'unlabeled'
    }
    
    # Known compound terms - use PLURAL form only to avoid duplicates
    COMPOUND_TERMS = [
        'machine learning', 'deep learning', 'artificial intelligence',
        'neural networks', 'decision trees', 'random forests', 
        'natural language processing', 'speech recognition', 'image recognition', 
        'computer vision', 'supervised learning', 'unsupervised learning', 
        'reinforcement learning', 'dimensionality reduction', 'game playing', 
        'labeled data', 'unlabeled data'
    ]
    
    # Singular to plural mapping for deduplication
    SINGULAR_TO_PLURAL = {
        'neural network': 'neural networks',
        'decision tree': 'decision trees',
        'random forest': 'random forests',
    }
    
    def __init__(self, max_keywords: int = 8):
        self.max_keywords = max_keywords
        self._keybert = None
        self._nlp = None
        self._loaded = False
    
    def _load_models(self):
        """Load models lazily."""
        if self._loaded:
            return
        
        if KEYBERT_AVAILABLE:
            try:
                self._keybert = KeyBERT('all-MiniLM-L6-v2')
            except:
                pass
        
        if SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except:
                pass
        
        self._loaded = True
    
    def _extract_compound_terms(self, text: str) -> List[Keyphrase]:
        """Extract known compound terms from text."""
        text_lower = text.lower()
        found = []
        
        for term in self.COMPOUND_TERMS:
            if term in text_lower:
                # Count occurrences for scoring
                count = text_lower.count(term)
                found.append(Keyphrase(
                    phrase=term.title(),
                    score=min(0.9, 0.6 + count * 0.1),
                    source="compound"
                ))
        
        return found
    
    def _extract_noun_phrases(self, text: str) -> List[Keyphrase]:
        """Extract noun phrases using SpaCy."""
        if not self._nlp:
            return []
        
        try:
            doc = self._nlp(text)
            phrases = {}
            
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip()
                
                # Clean the phrase
                phrase = re.sub(r'^(the|a|an|this|these|that|those)\s+', '', phrase.lower())
                phrase = phrase.strip()
                
                # Must be 2+ words
                words = phrase.split()
                if len(words) < 2:
                    continue
                
                # Skip if all words are generic
                if all(w in self.GENERIC_WORDS for w in words):
                    continue
                
                # Skip if too long
                if len(words) > 4:
                    continue
                
                # Title case
                phrase = phrase.title()
                
                if phrase not in phrases:
                    phrases[phrase] = 0.7
                else:
                    phrases[phrase] = min(0.95, phrases[phrase] + 0.1)
            
            return [
                Keyphrase(phrase=p, score=s, source="spacy")
                for p, s in phrases.items()
            ]
        except:
            return []
    
    def _extract_with_keybert(self, text: str) -> List[Keyphrase]:
        """Extract with KeyBERT - but only 2-3 word phrases."""
        if not self._keybert:
            return []
        
        try:
            # Only extract 2-3 word phrases
            keywords = self._keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(2, 3),  # ONLY multi-word
                stop_words='english',
                top_n=self.max_keywords * 2,
                use_mmr=True,
                diversity=0.7
            )
            
            results = []
            for phrase, score in keywords:
                if score < 0.3:
                    continue
                
                # Skip if all generic words
                words = phrase.lower().split()
                if all(w in self.GENERIC_WORDS for w in words):
                    continue
                
                results.append(Keyphrase(
                    phrase=phrase.title(),
                    score=score,
                    source="keybert"
                ))
            
            return results
        except:
            return []
    
    def _merge_and_dedupe(self, all_phrases: List[Keyphrase]) -> List[Keyphrase]:
        """Merge and deduplicate phrases, normalizing singular/plural."""
        seen = {}
        
        for kp in all_phrases:
            key = kp.phrase.lower()
            
            # Normalize singular to plural
            if key in self.SINGULAR_TO_PLURAL:
                key = self.SINGULAR_TO_PLURAL[key]
                kp = Keyphrase(
                    phrase=key.title(),
                    score=kp.score,
                    source=kp.source
                )
            
            # Skip single generic words
            if key in self.GENERIC_WORDS:
                continue
            
            # Skip single words entirely
            if ' ' not in key:
                continue
            
            if key not in seen or kp.score > seen[key].score:
                seen[key] = kp
        
        # Sort by score
        sorted_phrases = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return sorted_phrases[:self.max_keywords]
    
    def extract(self, text: str) -> List[Keyphrase]:
        """
        Extract clean, meaningful keyphrases.
        
        Returns multi-word technical terms like:
        - Machine Learning
        - Deep Learning
        - Neural Networks
        - Supervised Learning
        """
        self._load_models()
        
        all_phrases = []
        
        # 1. Extract known compound terms first (highest priority)
        all_phrases.extend(self._extract_compound_terms(text))
        
        # 2. Extract noun phrases with SpaCy
        all_phrases.extend(self._extract_noun_phrases(text))
        
        # 3. Extract with KeyBERT (multi-word only)
        all_phrases.extend(self._extract_with_keybert(text))
        
        # Merge and return
        return self._merge_and_dedupe(all_phrases)
    
    def extract_topics(self, text: str, n_topics: int = 5) -> List[str]:
        """Extract main topics as strings."""
        return [kp.phrase for kp in self.extract(text)[:n_topics]]


# Test
if __name__ == "__main__":
    text = """
    Machine Learning is a subset of Artificial Intelligence that enables computers 
    to learn from data. There are three types: supervised learning, unsupervised 
    learning, and reinforcement learning. Deep learning uses neural networks with 
    many layers. Decision trees and random forests are popular algorithms.
    """
    
    extractor = KeyphraseExtractor(max_keywords=8)
    keyphrases = extractor.extract(text)
    
    print("Extracted keyphrases:")
    for kp in keyphrases:
        print(f"  [{kp.source}] {kp.phrase}: {kp.score:.3f}")
