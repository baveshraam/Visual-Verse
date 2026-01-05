"""
Scene-to-Prompt Generator
T5-based text-to-text model for converting story scenes to structured visual descriptions.

Input: "A girl climbed the hill at dawn."
Output: "{subject: girl, action: climbing, location: hill, time: dawn, mood: peaceful}"

The structured output is then converted to Stable Diffusion prompts.

Supported Languages: English, Tamil, Malayalam, Kannada, Telugu, Hindi
(via IndicTrans2 translation preprocessing)

Novelty: Language-agnostic semantic scene parsing for visualization.
"""
import json
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle imports gracefully
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Install with: pip install transformers")

# Import base class
try:
    from ..base import BaseNLPModel, ModelConfig
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from nlp_models.base import BaseNLPModel, ModelConfig


@dataclass
class SceneDescription:
    """Structured scene description for image generation."""
    subject: str
    action: str
    location: Optional[str] = None
    time: Optional[str] = None
    mood: Optional[str] = None
    objects: Optional[List[str]] = None
    style_hints: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        result = {
            'subject': self.subject,
            'action': self.action
        }
        if self.location:
            result['location'] = self.location
        if self.time:
            result['time'] = self.time
        if self.mood:
            result['mood'] = self.mood
        if self.objects:
            result['objects'] = self.objects
        if self.style_hints:
            result['style_hints'] = self.style_hints
        return result
    
    def to_prompt(self, style: str = "western") -> str:
        """Convert to Stable Diffusion prompt."""
        parts = []
        
        # Subject + action
        parts.append(f"{self.subject} {self.action}")
        
        # Location
        if self.location:
            parts.append(f"in {self.location}")
        
        # Time
        if self.time:
            parts.append(f"at {self.time}")
        
        # Mood (converted to visual style)
        mood_visuals = {
            'peaceful': 'calm, serene lighting, soft colors',
            'excited': 'dynamic, energetic, bright colors',
            'sad': 'muted colors, soft shadows, melancholic',
            'scared': 'dark, shadowy, tense atmosphere',
            'happy': 'bright, cheerful, warm lighting',
            'mysterious': 'misty, dramatic shadows, ethereal'
        }
        if self.mood and self.mood in mood_visuals:
            parts.append(mood_visuals[self.mood])
        
        # Objects
        if self.objects:
            parts.append(f"with {', '.join(self.objects)}")
        
        # Style
        style_tags = {
            'western': 'digital illustration, vibrant colors, professional quality',
            'manga': 'anime style, manga art, clean linework',
            'cartoon': "children's book illustration, colorful, cute",
            'realistic': 'realistic digital art, detailed, cinematic lighting'
        }
        parts.append(style_tags.get(style, style_tags['western']))
        
        return ', '.join(parts)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SceneDescription':
        """Parse from JSON string."""
        try:
            data = json.loads(json_str)
            return cls(
                subject=data.get('subject', 'person'),
                action=data.get('action', 'standing'),
                location=data.get('location'),
                time=data.get('time'),
                mood=data.get('mood'),
                objects=data.get('objects'),
                style_hints=data.get('style_hints')
            )
        except json.JSONDecodeError:
            # Fallback parsing for malformed output
            return cls(subject='person', action='standing')


class Scene2PromptGenerator(BaseNLPModel):
    """
    T5-based generator for converting scene text to structured descriptions.
    
    Uses seq2seq architecture to generate structured JSON from natural language.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                model_name='scene2prompt',
                base_model='t5-small',  # Use t5-base for better quality
                num_labels=0,  # Not applicable for seq2seq
                max_length=256,
                batch_size=8,
                learning_rate=3e-4,
                num_epochs=10,
                label_names=[]
            )
        super().__init__(config)
        
        self.build_model()
        self.load_tokenizer()
    
    def build_model(self):
        """Build T5 model."""
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.base_model
        )
        self.model.to(self.device)
    
    def load_tokenizer(self):
        """Load T5 tokenizer."""
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.base_model,
            model_max_length=self.config.max_length
        )
    
    def preprocess(
        self, 
        texts: List[str],
        targets: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input for T5.
        
        Args:
            texts: Source scene descriptions
            targets: Optional target structured outputs
            
        Returns:
            Dictionary with input_ids, attention_mask, (optional) labels
        """
        # Add task prefix
        prefixed = [f"parse scene: {t}" for t in texts]
        
        encoded = self.tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
        if targets:
            target_encoded = self.tokenizer(
                targets,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            # Replace padding token id with -100 for loss calculation
            labels = target_encoded['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result['labels'] = labels
        
        return result
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch.get('labels')
        )
        
        result = {}
        if outputs.loss is not None:
            result['loss'] = outputs.loss
        result['logits'] = outputs.logits
        
        return result
    
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute seq2seq loss."""
        return outputs.get('loss', torch.tensor(0.0))
    
    def predict(self, texts: List[str]) -> List[SceneDescription]:
        """
        Generate structured scene descriptions.
        
        Args:
            texts: List of scene texts
            
        Returns:
            List of SceneDescription objects
        """
        self.model.eval()
        
        # Preprocess
        encoded = self.preprocess(texts)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                max_length=128,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode
        generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Parse to SceneDescription
        descriptions = []
        for gen in generated:
            desc = SceneDescription.from_json(gen)
            descriptions.append(desc)
        
        return descriptions
    
    def scene_to_prompt(
        self, 
        scene_text: str,
        style: str = "western",
        cultural_context: Optional[str] = None
    ) -> str:
        """
        Convert scene text directly to SD prompt.
        
        Args:
            scene_text: Natural language scene description
            style: Art style for the prompt
            cultural_context: Optional cultural context to append
            
        Returns:
            Stable Diffusion prompt string
        """
        descriptions = self.predict([scene_text])
        prompt = descriptions[0].to_prompt(style)
        
        if cultural_context:
            prompt = f"{prompt}, {cultural_context}"
        
        return prompt
    
    def batch_scene_to_prompts(
        self, 
        scenes: List[str],
        style: str = "western"
    ) -> List[str]:
        """Generate prompts for multiple scenes."""
        descriptions = self.predict(scenes)
        return [d.to_prompt(style) for d in descriptions]


# Test
if __name__ == '__main__':
    generator = Scene2PromptGenerator()
    
    scenes = [
        "A young girl named Maya climbed the misty hill at dawn.",
        "The old wizard stood before the ancient tower, holding his staff.",
        "Children played happily in the sunny meadow near the river."
    ]
    
    print("Scene to Prompt Generation:")
    for scene in scenes:
        prompt = generator.scene_to_prompt(scene, style="cartoon")
        print(f"\nScene: {scene}")
        print(f"Prompt: {prompt}")
