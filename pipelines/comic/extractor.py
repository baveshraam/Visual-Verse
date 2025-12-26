"""
Scene Extractor Module
Extracts characters, settings, and actions from story segments for image generation.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

import sys
sys.path.append('../..')
from config import SPACY_MODEL


@dataclass
class Character:
    """Represents a character in the scene."""
    name: str
    description: str = ""
    role: str = "character"  # protagonist, antagonist, supporting
    attributes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "attributes": self.attributes
        }


@dataclass
class Setting:
    """Represents the scene setting/location."""
    location: str
    time_of_day: str = ""
    atmosphere: str = ""
    details: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "time_of_day": self.time_of_day,
            "atmosphere": self.atmosphere,
            "details": self.details
        }


@dataclass
class Action:
    """Represents the main action in the scene."""
    description: str
    verb: str = ""
    participants: List[str] = field(default_factory=list)
    emotion: str = ""
    
    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "verb": self.verb,
            "participants": self.participants,
            "emotion": self.emotion
        }


@dataclass
class SceneDetails:
    """Complete extracted details for a scene."""
    segment_id: int
    original_text: str
    characters: List[Character]
    setting: Setting
    actions: List[Action]
    dialogue: List[str]
    mood: str = "neutral"
    
    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "characters": [c.to_dict() for c in self.characters],
            "setting": self.setting.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "dialogue": self.dialogue,
            "mood": self.mood
        }


class SceneExtractor:
    """
    Extracts visual elements from story segments.
    
    Extracts:
    - Characters: Names, physical descriptions
    - Settings: Locations, environment details
    - Actions: What's happening in the scene
    - Dialogue: Spoken text
    - Mood: Emotional tone of the scene
    """
    
    # Location indicator patterns
    LOCATION_PATTERNS = [
        r'\b(in|at|inside|outside|near|by|on)\s+(the\s+)?([a-zA-Z\s]+)',
        r'\b(forest|village|castle|tower|house|room|garden|mountain|river|city|town|cave|beach|ocean|sky)\b',
    ]
    
    # Time indicators
    TIME_PATTERNS = {
        "morning": ["morning", "dawn", "sunrise", "early"],
        "afternoon": ["afternoon", "midday", "noon"],
        "evening": ["evening", "dusk", "sunset"],
        "night": ["night", "midnight", "dark", "moonlight"]
    }
    
    # Mood/emotion patterns
    MOOD_PATTERNS = {
        "happy": ["happy", "joy", "excited", "laughed", "smiled", "bright", "sunny"],
        "sad": ["sad", "cry", "tears", "sorrow", "gloomy", "dark"],
        "mysterious": ["mysterious", "strange", "curious", "wonder", "magical", "ancient"],
        "tense": ["fear", "afraid", "danger", "threat", "worried", "anxious"],
        "peaceful": ["peaceful", "calm", "quiet", "serene", "gentle", "soft"]
    }
    
    # Character description adjectives
    CHARACTER_ADJECTIVES = [
        "young", "old", "tall", "short", "beautiful", "handsome", "wise",
        "brave", "kind", "fierce", "gentle", "mysterious", "ancient"
    ]
    
    def __init__(self):
        """Initialize the scene extractor."""
        self.nlp = None
        self._model_loaded = False
        
    def _load_model(self):
        """Load SpaCy model."""
        if self._model_loaded:
            return
            
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(SPACY_MODEL)
            except OSError:
                print(f"SpaCy model not found.")
                self.nlp = None
        
        self._model_loaded = True
    
    def _extract_characters(self, text: str, doc=None) -> List[Character]:
        """Extract characters from text."""
        characters = []
        found_names = set()
        
        # Use SpaCy NER if available
        if doc is not None:
            for ent in doc.ents:
                if ent.label_ == "PERSON" and ent.text not in found_names:
                    # Try to find description near the name
                    description = self._find_character_description(text, ent.text)
                    characters.append(Character(
                        name=ent.text,
                        description=description
                    ))
                    found_names.add(ent.text)
        
        # Fallback: Look for capitalized names
        if not characters:
            # Pattern for names (capitalized words not at sentence start)
            name_pattern = r'(?<=[.!?]\s)([A-Z][a-z]+)|(?:named|called)\s+([A-Z][a-z]+)'
            matches = re.findall(name_pattern, text)
            for match in matches:
                name = match[0] or match[1]
                if name and name not in found_names:
                    characters.append(Character(name=name))
                    found_names.add(name)
        
        # Look for pronouns if no characters found
        if not characters:
            if re.search(r'\b(she|her)\b', text.lower()):
                characters.append(Character(name="Woman", description="female character"))
            elif re.search(r'\b(he|him)\b', text.lower()):
                characters.append(Character(name="Man", description="male character"))
        
        return characters
    
    def _find_character_description(self, text: str, name: str) -> str:
        """Find description adjacent to character name."""
        # Pattern to find adjectives before name
        pattern = rf'((?:\w+\s+){{0,3}}){re.escape(name)}'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            prefix = match.group(1).strip()
            # Check if prefix contains descriptive adjectives
            for adj in self.CHARACTER_ADJECTIVES:
                if adj in prefix.lower():
                    return prefix
        
        return ""
    
    def _extract_setting(self, text: str, doc=None) -> Setting:
        """Extract setting/location from text."""
        location = "unknown location"
        time_of_day = ""
        atmosphere = ""
        details = []
        
        # Extract location using NER
        if doc is not None:
            for ent in doc.ents:
                if ent.label_ in ["LOC", "GPE", "FAC"]:
                    location = ent.text
                    break
        
        # Fallback: use patterns
        if location == "unknown location":
            for pattern in self.LOCATION_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if len(match.groups()) >= 3:
                        location = match.group(3).strip()
                    else:
                        location = match.group(0).strip()
                    break
        
        # Extract time of day
        text_lower = text.lower()
        for time, keywords in self.TIME_PATTERNS.items():
            if any(kw in text_lower for kw in keywords):
                time_of_day = time
                break
        
        # Extract atmosphere from adjectives
        if doc is not None:
            for token in doc:
                if token.pos_ == "ADJ" and token.text.lower() in [
                    "dark", "bright", "misty", "foggy", "sunny", "gloomy", 
                    "mysterious", "ancient", "magical", "quiet"
                ]:
                    details.append(token.text)
        
        return Setting(
            location=location,
            time_of_day=time_of_day,
            atmosphere=atmosphere,
            details=details
        )
    
    def _extract_actions(self, text: str, doc=None) -> List[Action]:
        """Extract main actions from text."""
        actions = []
        
        if doc is not None:
            # Find main verbs and their subjects/objects
            for token in doc:
                if token.pos_ == "VERB" and token.dep_ in ["ROOT", "conj"]:
                    # Get the subject
                    subjects = [child.text for child in token.children 
                               if child.dep_ in ["nsubj", "nsubjpass"]]
                    
                    # Get the object or complement
                    objects = [child.text for child in token.children 
                              if child.dep_ in ["dobj", "pobj", "attr"]]
                    
                    # Build action description
                    verb = token.lemma_
                    participants = subjects + objects
                    
                    action_text = f"{' '.join(subjects)} {token.text}"
                    if objects:
                        action_text += f" {' '.join(objects)}"
                    
                    actions.append(Action(
                        description=action_text.strip(),
                        verb=verb,
                        participants=participants
                    ))
        
        # Fallback: simple verb extraction
        if not actions:
            # Find sentences and extract main action
            sentences = re.split(r'[.!?]', text)
            for sent in sentences[:3]:  # Limit to first 3 sentences
                if sent.strip():
                    actions.append(Action(description=sent.strip()))
        
        return actions[:3]  # Limit to 3 main actions
    
    def _extract_dialogue(self, text: str) -> List[str]:
        """Extract quoted dialogue from text."""
        dialogue = re.findall(r'"([^"]*)"', text)
        dialogue += re.findall(r"'([^']*)'", text)
        return dialogue
    
    def _detect_mood(self, text: str) -> str:
        """Detect the emotional mood of the scene."""
        text_lower = text.lower()
        
        mood_scores = {}
        for mood, keywords in self.MOOD_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                mood_scores[mood] = score
        
        if mood_scores:
            return max(mood_scores, key=mood_scores.get)
        return "neutral"
    
    def extract(self, segment_id: int, text: str) -> SceneDetails:
        """
        Extract all visual details from a story segment.
        
        Args:
            segment_id: ID of the segment
            text: Text of the story segment
            
        Returns:
            SceneDetails object with all extracted information
        """
        self._load_model()
        
        # Process with SpaCy if available
        doc = self.nlp(text) if self.nlp else None
        
        # Extract all elements
        characters = self._extract_characters(text, doc)
        setting = self._extract_setting(text, doc)
        actions = self._extract_actions(text, doc)
        dialogue = self._extract_dialogue(text)
        mood = self._detect_mood(text)
        
        return SceneDetails(
            segment_id=segment_id,
            original_text=text,
            characters=characters,
            setting=setting,
            actions=actions,
            dialogue=dialogue,
            mood=mood
        )
    
    def extract_batch(
        self, 
        segments: List[Any]
    ) -> List[SceneDetails]:
        """
        Extract details from multiple segments.
        
        Args:
            segments: List of StorySegment objects
            
        Returns:
            List of SceneDetails objects
        """
        return [self.extract(seg.id, seg.text) for seg in segments]


# Example usage
if __name__ == "__main__":
    extractor = SceneExtractor()
    
    sample_text = """
    Maya, a young girl with bright eyes, stood at the edge of the dark forest.
    She took a deep breath and whispered "I'm not afraid."
    The ancient trees loomed overhead as she walked down the misty path.
    """
    
    details = extractor.extract(0, sample_text)
    
    print("Extracted Scene Details:")
    print(f"Characters: {[c.name for c in details.characters]}")
    print(f"Setting: {details.setting.location}")
    print(f"Actions: {[a.description for a in details.actions]}")
    print(f"Dialogue: {details.dialogue}")
    print(f"Mood: {details.mood}")
