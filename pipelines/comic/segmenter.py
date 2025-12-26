"""
Story Segmenter Module
Breaks narrative text into distinct "story beats" or scenes for comic panel generation.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

import sys
sys.path.append('../..')
from config import SPACY_MODEL, MAX_SCENES_PER_COMIC


@dataclass
class StorySegment:
    """Represents a single story beat/scene."""
    id: int
    text: str
    sentences: List[str]
    start_char: int
    end_char: int
    segment_type: str = "scene"  # scene, dialogue, transition, action
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "sentences": self.sentences,
            "segment_type": self.segment_type,
            "char_range": (self.start_char, self.end_char)
        }


class StorySegmenter:
    """
    Segments narrative text into story beats for comic panel generation.
    
    Segmentation strategies:
    1. Scene breaks based on setting/time changes
    2. Dialogue exchanges
    3. Action sequences
    4. Paragraph-based fallback
    """
    
    # Patterns indicating scene transitions
    TRANSITION_PATTERNS = [
        r'\b(later|afterward|next\s+day|the\s+following|meanwhile|suddenly|then)\b',
        r'\b(morning|evening|night|dawn|dusk|sunset|sunrise)\b',
        r'\b(arrived|entered|left|walked\s+into|stepped\s+into|came\s+to)\b',
    ]
    
    # Patterns indicating dialogue
    DIALOGUE_PATTERN = r'["\'][^"\']+["\']'
    
    def __init__(self, max_segments: int = MAX_SCENES_PER_COMIC):
        """
        Initialize the story segmenter.
        
        Args:
            max_segments: Maximum number of segments to create
        """
        self.max_segments = max_segments
        self.nlp = None
        self._model_loaded = False
        
    def _load_model(self):
        """Load SpaCy model for advanced segmentation."""
        if self._model_loaded:
            return
            
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(SPACY_MODEL)
            except OSError:
                print(f"SpaCy model not found. Using fallback segmentation.")
                self.nlp = None
        
        self._model_loaded = True
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback regex-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_transition(self, sentence: str) -> bool:
        """Check if sentence indicates a scene transition."""
        for pattern in self.TRANSITION_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False
    
    def _detect_dialogue(self, sentence: str) -> bool:
        """Check if sentence contains dialogue."""
        return bool(re.search(self.DIALOGUE_PATTERN, sentence))
    
    def _classify_sentence(self, sentence: str) -> str:
        """Classify sentence type."""
        if self._detect_dialogue(sentence):
            return "dialogue"
        elif self._detect_transition(sentence):
            return "transition"
        else:
            return "action"
    
    def _group_sentences_by_type(
        self, 
        sentences: List[str]
    ) -> List[Tuple[str, List[str]]]:
        """Group consecutive sentences of similar type."""
        if not sentences:
            return []
        
        groups = []
        current_type = self._classify_sentence(sentences[0])
        current_group = [sentences[0]]
        
        for sentence in sentences[1:]:
            sent_type = self._classify_sentence(sentence)
            
            # Start new group on transition or type change
            if sent_type == "transition" or (sent_type != current_type and 
                                              current_type != "transition"):
                groups.append((current_type, current_group))
                current_type = sent_type
                current_group = [sentence]
            else:
                current_group.append(sentence)
        
        # Add last group
        if current_group:
            groups.append((current_type, current_group))
        
        return groups
    
    def _merge_small_segments(
        self, 
        segments: List[StorySegment],
        min_sentences: int = 2
    ) -> List[StorySegment]:
        """Merge very small segments with adjacent ones."""
        if len(segments) <= 1:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # If segment is too small and there's a next segment
            if len(current.sentences) < min_sentences and i + 1 < len(segments):
                next_seg = segments[i + 1]
                # Merge with next segment
                merged_sentences = current.sentences + next_seg.sentences
                merged_text = " ".join(merged_sentences)
                
                merged.append(StorySegment(
                    id=len(merged),
                    text=merged_text,
                    sentences=merged_sentences,
                    start_char=current.start_char,
                    end_char=next_seg.end_char,
                    segment_type=next_seg.segment_type
                ))
                i += 2  # Skip next segment as it's merged
            else:
                current.id = len(merged)
                merged.append(current)
                i += 1
        
        return merged
    
    def _limit_segments(self, segments: List[StorySegment]) -> List[StorySegment]:
        """Limit number of segments to max_segments."""
        if len(segments) <= self.max_segments:
            return segments
        
        # Select evenly distributed segments
        step = len(segments) / self.max_segments
        selected = []
        
        for i in range(self.max_segments):
            idx = int(i * step)
            segment = segments[idx]
            segment.id = i
            selected.append(segment)
        
        return selected
    
    def segment(self, text: str) -> List[StorySegment]:
        """
        Segment narrative text into story beats.
        
        For short texts: creates one panel per sentence
        For longer texts: groups related sentences together
        
        Args:
            text: Narrative text to segment
            
        Returns:
            List of StorySegment objects
        """
        self._load_model()
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # For short stories (5 or fewer sentences), create one panel per sentence
        # This ensures proper comic panel distribution
        if len(sentences) <= self.max_segments:
            segments = []
            char_pos = 0
            
            for i, sentence in enumerate(sentences):
                start_char = text.find(sentence, char_pos)
                if start_char == -1:
                    start_char = char_pos
                end_char = start_char + len(sentence)
                char_pos = end_char
                
                segment_type = self._classify_sentence(sentence)
                
                segments.append(StorySegment(
                    id=i,
                    text=sentence,
                    sentences=[sentence],
                    start_char=start_char,
                    end_char=end_char,
                    segment_type=segment_type
                ))
            
            return segments
        
        # For longer texts, group sentences intelligently
        groups = self._group_sentences_by_type(sentences)
        
        # Create segments
        segments = []
        char_pos = 0
        
        for segment_type, sent_group in groups:
            segment_text = " ".join(sent_group)
            start_char = text.find(sent_group[0], char_pos)
            if start_char == -1:
                start_char = char_pos
            end_char = start_char + len(segment_text)
            char_pos = end_char
            
            segments.append(StorySegment(
                id=len(segments),
                text=segment_text,
                sentences=sent_group,
                start_char=start_char,
                end_char=end_char,
                segment_type=segment_type
            ))
        
        # If we still have too few segments, split the largest ones
        while len(segments) < min(self.max_segments, len(sentences)):
            # Find the segment with the most sentences
            max_idx = max(range(len(segments)), key=lambda i: len(segments[i].sentences))
            seg = segments[max_idx]
            
            if len(seg.sentences) <= 1:
                break  # Can't split further
            
            # Split in half
            mid = len(seg.sentences) // 2
            first_sentences = seg.sentences[:mid]
            second_sentences = seg.sentences[mid:]
            
            first_text = " ".join(first_sentences)
            second_text = " ".join(second_sentences)
            
            # Replace the original with two new segments
            new_seg1 = StorySegment(
                id=max_idx,
                text=first_text,
                sentences=first_sentences,
                start_char=seg.start_char,
                end_char=seg.start_char + len(first_text),
                segment_type=seg.segment_type
            )
            new_seg2 = StorySegment(
                id=max_idx + 1,
                text=second_text,
                sentences=second_sentences,
                start_char=seg.start_char + len(first_text) + 1,
                end_char=seg.end_char,
                segment_type=seg.segment_type
            )
            
            segments = segments[:max_idx] + [new_seg1, new_seg2] + segments[max_idx+1:]
        
        # Re-number segments
        for i, seg in enumerate(segments):
            seg.id = i
        
        # Limit to max segments
        segments = self._limit_segments(segments)
        
        return segments
    
    def segment_by_paragraph(self, text: str) -> List[StorySegment]:
        """
        Simple segmentation based on paragraphs.
        Fallback method for simpler texts.
        
        Args:
            text: Text to segment
            
        Returns:
            List of StorySegment objects
        """
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        segments = []
        char_pos = 0
        
        for i, para in enumerate(paragraphs):
            start_char = text.find(para, char_pos)
            if start_char == -1:
                start_char = char_pos
            end_char = start_char + len(para)
            char_pos = end_char
            
            sentences = self._split_sentences(para)
            
            segments.append(StorySegment(
                id=i,
                text=para,
                sentences=sentences,
                start_char=start_char,
                end_char=end_char,
                segment_type="scene"
            ))
        
        return self._limit_segments(segments)


# Example usage
if __name__ == "__main__":
    segmenter = StorySegmenter(max_segments=4)
    
    sample_story = """
    Once upon a time, in a small village, there lived a young girl named Maya. 
    She spent her days exploring the nearby forest and dreaming of adventure.
    
    One sunny morning, Maya discovered a hidden path behind the old oak tree.
    "This must lead somewhere magical!" she exclaimed with excitement.
    
    She followed the path deeper into the woods. The trees grew taller and the 
    sunlight filtered through the leaves creating dancing shadows.
    
    Suddenly, she arrived at a clearing. In the center stood an ancient stone 
    tower covered in ivy. A mysterious light glowed from the highest window.
    """
    
    segments = segmenter.segment(sample_story)
    
    print(f"Found {len(segments)} segments:\n")
    for seg in segments:
        print(f"[{seg.id}] {seg.segment_type.upper()}")
        print(f"    Text: {seg.text[:100]}...")
        print()
