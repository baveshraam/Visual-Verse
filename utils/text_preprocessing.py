"""
Text Preprocessing Utilities
Handles text cleaning and normalization for NLP processing.
"""
import re
from typing import List, Optional


class TextPreprocessor:
    """Utility class for text preprocessing operations."""
    
    def __init__(self):
        # Common contractions for expansion
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am"
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text ready for NLP processing
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """
        Remove special characters from text.
        
        Args:
            text: Input text
            keep_punctuation: If True, keeps basic punctuation marks
            
        Returns:
            Text with special characters removed
        """
        if keep_punctuation:
            # Keep letters, numbers, spaces, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
        else:
            # Keep only letters, numbers, and spaces
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting using regex
        # Handles ., !, ? as sentence terminators
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraphs
        """
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def normalize_whitespace(self, text: str) -> str:
        """Replace multiple whitespace characters with single space."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def extract_dialogue(self, text: str) -> List[str]:
        """
        Extract quoted dialogue from text.
        
        Args:
            text: Input text
            
        Returns:
            List of dialogue strings found in the text
        """
        # Match text within quotes
        dialogue = re.findall(r'"([^"]*)"', text)
        dialogue += re.findall(r"'([^']*)'", text)
        return dialogue
    
    def has_dialogue(self, text: str) -> bool:
        """Check if text contains dialogue (quoted speech)."""
        return bool(re.search(r'["\'][^"\']+["\']', text))
    
    def preprocess_for_classification(self, text: str) -> str:
        """
        Full preprocessing pipeline for text classification.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text ready for classification
        """
        text = self.clean_text(text)
        text = self.normalize_whitespace(text)
        return text
    
    def preprocess_for_ner(self, text: str) -> str:
        """
        Preprocessing pipeline optimized for Named Entity Recognition.
        Keeps more original structure for better NER accuracy.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text ready for NER
        """
        text = self.clean_text(text)
        # Keep contractions for NER as they don't affect entity recognition
        return text
