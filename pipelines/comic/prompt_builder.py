"""
Prompt Builder Module - ENHANCED VERSION
Converts story sentences directly into rich, descriptive image generation prompts.
The key insight: use the actual story sentence as the base, not abstract extractions.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict
import re

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .extractor import SceneDetails


@dataclass
class ImagePrompt:
    """Generated prompt for image generation."""
    scene_id: int
    positive_prompt: str
    negative_prompt: str
    style_tags: List[str]
    seed: Optional[int] = None  # For character consistency
    
    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "style_tags": self.style_tags,
            "seed": self.seed
        }
    
    def get_full_prompt(self) -> str:
        """Get combined prompt with style tags."""
        tags = ", ".join(self.style_tags)
        return f"{self.positive_prompt}, {tags}"


class PromptBuilder:
    """
    Builds rich, story-accurate prompts for Stable Diffusion.
    
    Key approach:
    1. Use the ACTUAL story sentence as the base
    2. Enhance with scene descriptions
    3. Add consistent character description across panels
    4. Use proper negative prompts to avoid unwanted elements
    """
    
    # Style presets
    STYLE_PRESETS = {
        "western": [
            "digital illustration",
            "vibrant colors",
            "detailed artwork",
            "professional quality",
            "single scene"
        ],
        "manga": [
            "manga style",
            "anime art",
            "clean linework",
            "expressive",
            "Japanese illustration"
        ],
        "cartoon": [
            "children's book illustration",
            "colorful cartoon style",
            "cute and friendly",
            "storybook art",
            "soft colors"
        ],
        "realistic": [
            "realistic digital art",
            "detailed illustration",
            "professional artwork",
            "cinematic lighting"
        ]
    }
    
    # Mood to visual style mapping
    MOOD_VISUALS = {
        "happy": "bright, cheerful, sunny, warm colors",
        "sad": "muted colors, soft lighting, melancholic",
        "scared": "dark, shadowy, misty, worried expression",
        "peaceful": "calm, serene, soft lighting, gentle",
        "excited": "dynamic, energetic, bright colors",
        "mysterious": "dark shadows, misty, dramatic lighting",
        "friendly": "warm colors, welcoming, soft lighting"
    }
    
    # Default negative prompts
    DEFAULT_NEGATIVE = (
        "blurry, bad quality, ugly, deformed, disfigured, "
        "multiple panels, comic strip, collage, grid layout, "
        "text, watermark, signature, logo"
    )
    
    def __init__(self, style: str = "cartoon"):
        """Initialize with art style."""
        self.style = style
        self.style_tags = self.STYLE_PRESETS.get(style, self.STYLE_PRESETS["cartoon"])
        self.base_negative = self.DEFAULT_NEGATIVE
        
        # Character description - will be set based on story
        self.character_description = None
        
        # Consistent seed for character appearance
        self.use_consistent_seed = True
        self.seed = 42  # Fixed seed for consistency
        
        # Load spacy for NLP processing
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                pass
    
    def set_comic_style(self, style: str):
        """Set the art style."""
        self.style = style
        self.style_tags = self.STYLE_PRESETS.get(style, self.STYLE_PRESETS["cartoon"])
    
    def _detect_main_character(self, scenes: List[SceneDetails]) -> str:
        """Detect the main character from all scenes and create consistent description."""
        # Count character mentions
        char_counts = {}
        for scene in scenes:
            for char in scene.characters:
                name = char.name.lower()
                if name not in ["unknown", "person", "it", "they"]:
                    char_counts[name] = char_counts.get(name, 0) + 1
        
        if not char_counts:
            return ""
        
        # Get most common character
        main_char = max(char_counts, key=char_counts.get)
        
        # Create rich description based on character type
        char_descriptions = {
            "rabbit": "a small fluffy brown rabbit with big curious eyes and long floppy ears",
            "bunny": "a cute little bunny with soft fur and twitchy nose",
            "fox": "a clever orange fox with a bushy tail",
            "bear": "a friendly brown bear with a gentle expression",
            "girl": "a young girl with bright eyes",
            "boy": "a young boy with curious expression",
            "cat": "a cute cat with whiskers",
            "dog": "a friendly dog with wagging tail",
            "bird": "a colorful bird with pretty feathers",
            "mouse": "a tiny mouse with big ears",
        }
        
        # Return matching description or generic
        for key, desc in char_descriptions.items():
            if key in main_char:
                return desc
        
        return f"a {main_char}"
    
    def _detect_mood(self, text: str) -> str:
        """Detect the emotional mood of a scene."""
        text_lower = text.lower()
        
        mood_keywords = {
            "scared": ["scared", "afraid", "fear", "lost", "worried", "nervous", "frightened"],
            "happy": ["happy", "joy", "glad", "smile", "laugh", "cheerful", "delighted"],
            "sad": ["sad", "cry", "tears", "unhappy", "sorrow", "grief"],
            "peaceful": ["peaceful", "calm", "quiet", "serene", "gentle", "rest"],
            "excited": ["excited", "eager", "thrill", "adventure", "discover"],
            "friendly": ["helped", "friend", "together", "kind", "love", "care", "guide"]
        }
        
        for mood, keywords in mood_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return mood
        
        return "peaceful"
    
    def _detect_setting(self, text: str) -> str:
        """Detect the setting/location from text."""
        text_lower = text.lower()
        
        settings = {
            "forest": ["forest", "woods", "trees", "woodland"],
            "home": ["home", "house", "cottage", "den", "burrow"],
            "meadow": ["meadow", "field", "grass", "clearing"],
            "river": ["river", "stream", "water", "lake", "pond"],
            "mountain": ["mountain", "hill", "cliff"],
            "village": ["village", "town", "street"],
            "garden": ["garden", "flowers", "plants"]
        }
        
        for setting, keywords in settings.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return setting
        
        return "outdoor scene"
    
    def _detect_other_characters(self, text: str) -> List[str]:
        """Detect other animals/characters mentioned."""
        text_lower = text.lower()
        
        animals = {
            "deer": ["deer", "fawn", "doe"],
            "birds": ["bird", "birds", "sparrow", "robin"],
            "squirrel": ["squirrel", "squirrels"],
            "fox": ["fox", "foxes"],
            "bear": ["bear", "bears"],
            "owl": ["owl", "owls"],
            "butterfly": ["butterfly", "butterflies"],
            "mouse": ["mouse", "mice"],
        }
        
        found = []
        for animal, keywords in animals.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found.append(animal)
                    break
        
        return found
    
    def _enhance_sentence_to_prompt(self, sentence: str, scene_num: int) -> str:
        """
        Convert a story sentence into a rich, descriptive image prompt.
        This is the core function that makes prompts work.
        """
        # Detect scene elements
        mood = self._detect_mood(sentence)
        setting = self._detect_setting(sentence)
        other_chars = self._detect_other_characters(sentence)
        mood_visual = self.MOOD_VISUALS.get(mood, "")
        
        # Start with character description
        prompt_parts = []
        
        if self.character_description:
            prompt_parts.append(self.character_description)
        
        # Add the story action - convert sentence to visual description
        action = self._sentence_to_action(sentence)
        if action:
            prompt_parts.append(action)
        
        # Add other characters if mentioned
        if other_chars:
            other_str = " and ".join(other_chars)
            prompt_parts.append(f"with {other_str}")
        
        # Add setting
        if setting:
            prompt_parts.append(f"in a {setting}")
        
        # Add mood visuals
        if mood_visual:
            prompt_parts.append(mood_visual)
        
        # Combine into final prompt
        prompt = ", ".join(prompt_parts)
        
        return prompt
    
    def _sentence_to_action(self, sentence: str) -> str:
        """Convert a sentence into a clear visual action description."""
        sentence = sentence.strip()
        
        # Replace pronouns with character reference
        replacements = [
            (r'\bit\b', 'the rabbit'),
            (r'\bIt\b', 'The rabbit'),
            (r'\bits\b', "the rabbit's"),
            (r'\bhe\b', 'the rabbit'),
            (r'\bshe\b', 'the rabbit'),
            (r'\bthey\b', 'the animals'),
        ]
        
        for pattern, replacement in replacements:
            sentence = re.sub(pattern, replacement, sentence)
        
        # Common story patterns to visual descriptions
        patterns = [
            # "lived near" -> "sitting near"
            (r'lived near ([^.]+)', r'sitting happily near \1'),
            # "helped...find" -> "showing...where to find"
            (r'helped (.+) find (.+)', r'helping \1 find \2, friendly scene'),
            # "got lost" -> "looking lost"
            (r'got lost', 'looking lost and worried, alone'),
            # "felt scared" -> "scared expression"
            (r'felt scared', 'with a scared expression, anxious'),
            # "came together" -> "gathering together"
            (r'came together', 'gathering together as a group'),
            # "guided...home" -> "leading...back home"
            (r'guided (.+) home', r'leading \1 back to home, helpful scene'),
            # "learned that" -> focus on character
            (r'learned that (.+)', r'with a wise, content expression, understanding'),
        ]
        
        for pattern, replacement in patterns:
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def _get_scene_negative_prompt(self, sentence: str) -> str:
        """Get appropriate negative prompt for this scene."""
        negatives = [self.base_negative]
        
        sentence_lower = sentence.lower()
        
        # If story is about animals, exclude humans
        if any(animal in sentence_lower for animal in ['rabbit', 'bunny', 'fox', 'bear', 'deer', 'bird']):
            negatives.append("humans, people, person, man, woman, child, human hands")
        
        # If outdoor scene, exclude indoor
        if any(place in sentence_lower for place in ['forest', 'meadow', 'woods', 'outside']):
            negatives.append("indoor, interior, room, house interior, building interior")
        
        # If single character moment, exclude crowds
        if 'alone' in sentence_lower or 'lost' in sentence_lower:
            negatives.append("crowd, group, many characters")
        
        return ", ".join(negatives)
    
    def build_prompt(self, scene: SceneDetails) -> ImagePrompt:
        """Build a single prompt from scene details."""
        sentence = scene.original_text.strip()
        
        # Enhance sentence to rich prompt
        positive = self._enhance_sentence_to_prompt(sentence, scene.segment_id)
        
        # Get appropriate negative prompt
        negative = self._get_scene_negative_prompt(sentence)
        
        return ImagePrompt(
            scene_id=scene.segment_id,
            positive_prompt=positive,
            negative_prompt=negative,
            style_tags=self.style_tags.copy(),
            seed=self.seed if self.use_consistent_seed else None
        )
    
    def build_prompts(self, scenes: List[SceneDetails]) -> List[ImagePrompt]:
        """Build prompts for all scenes with character consistency."""
        # First, detect main character for consistency
        self.character_description = self._detect_main_character(scenes)
        
        print(f"Detected main character: {self.character_description}")
        
        # Build prompts for each scene
        prompts = []
        for scene in scenes:
            prompt = self.build_prompt(scene)
            prompts.append(prompt)
            print(f"Panel {scene.segment_id + 1} prompt: {prompt.positive_prompt[:100]}...")
        
        return prompts


# Test
if __name__ == "__main__":
    from .extractor import SceneDetails, Character, Setting, Action
    
    # Test with rabbit story
    scenes = [
        SceneDetails(
            segment_id=0,
            original_text="A little rabbit lived near a green forest.",
            characters=[Character(name="rabbit", description="little")],
            setting=Setting(location="forest"),
            actions=[],
            dialogue=[],
            mood="peaceful"
        ),
        SceneDetails(
            segment_id=1,
            original_text="Every morning, it helped other animals find food and water.",
            characters=[Character(name="rabbit")],
            setting=Setting(location="forest"),
            actions=[Action(description="helping")],
            dialogue=[],
            mood="happy"
        ),
        SceneDetails(
            segment_id=2,
            original_text="One day, the rabbit got lost and felt scared.",
            characters=[Character(name="rabbit")],
            setting=Setting(location="forest"),
            actions=[],
            dialogue=[],
            mood="scared"
        ),
        SceneDetails(
            segment_id=3,
            original_text="The animals it had helped came together and guided it home.",
            characters=[Character(name="rabbit"), Character(name="animals")],
            setting=Setting(location="home"),
            actions=[],
            dialogue=[],
            mood="happy"
        ),
    ]
    
    builder = PromptBuilder(style="cartoon")
    prompts = builder.build_prompts(scenes)
    
    print("\n=== Generated Prompts ===\n")
    for p in prompts:
        print(f"Panel {p.scene_id + 1}:")
        print(f"  Prompt: {p.get_full_prompt()}")
        print(f"  Negative: {p.negative_prompt[:80]}...")
        print()
