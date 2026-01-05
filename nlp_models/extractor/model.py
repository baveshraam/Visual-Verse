"""
Entity Extractor Model
Custom NER model with extended labels for narrative entities.

Labels:
- CHARACTER: Named characters or pronouns referring to characters
- LOCATION: Places, settings
- OBJECT: Important objects in the narrative
- TIME: Temporal expressions
- CONCEPT: Key concepts (for informational text)

Supported Languages: English, Tamil, Malayalam, Kannada, Telugu, Hindi
(via IndicTrans2 translation preprocessing)

Trained on WritingPrompts + translated Indic stories.
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle imports gracefully
try:
    from transformers import BertTokenizer, BertForTokenClassification
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
class Entity:
    """Extracted entity with metadata."""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'label': self.label,
            'start': self.start_char,
            'end': self.end_char,
            'confidence': self.confidence
        }


class EntityExtractor(BaseNLPModel):
    """
    BERT-based NER for extracting narrative entities.
    
    Extended entity types for story understanding:
    - CHARACTER, LOCATION, OBJECT, TIME, CONCEPT
    
    Uses BIO tagging scheme.
    """
    
    # Extended entity labels (BIO format)
    ENTITY_TYPES = ['CHARACTER', 'LOCATION', 'OBJECT', 'TIME', 'CONCEPT']
    LABEL_NAMES = ['O'] + [f'{prefix}-{etype}' for etype in ENTITY_TYPES for prefix in ['B', 'I']]
    
    LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_NAMES)}
    ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                model_name='entity_extractor',
                base_model='bert-base-uncased',
                num_labels=len(self.LABEL_NAMES),
                max_length=512,
                batch_size=16,
                learning_rate=3e-5,
                num_epochs=5,
                label_names=self.LABEL_NAMES
            )
        super().__init__(config)
        
        self.build_model()
        self.load_tokenizer()
    
    def build_model(self):
        """Build BERT token classifier."""
        self.model = BertForTokenClassification.from_pretrained(
            self.config.base_model,
            num_labels=self.config.num_labels
        )
        self.model.to(self.device)
    
    def load_tokenizer(self):
        """Load BERT tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(self.config.base_model)
    
    def preprocess(
        self, 
        texts: List[str],
        labels: Optional[List[List[str]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts with subword alignment.
        
        Args:
            texts: List of texts
            labels: Optional list of label sequences (word-level)
            
        Returns:
            Dictionary with input_ids, attention_mask, (optional) labels
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'offset_mapping': encoded['offset_mapping']
        }
        
        if labels:
            # Align labels to subword tokens
            aligned_labels = self._align_labels(texts, labels, encoded)
            result['labels'] = aligned_labels
        
        return result
    
    def _align_labels(
        self,
        texts: List[str],
        labels: List[List[str]],
        encoded
    ) -> torch.Tensor:
        """Align word-level labels to subword tokens."""
        batch_labels = []
        
        for i, (text, word_labels) in enumerate(zip(texts, labels)):
            token_labels = []
            word_ids = encoded.word_ids(i)
            
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    token_labels.append(-100)  # Special tokens
                elif word_idx != previous_word_idx:
                    if word_idx < len(word_labels):
                        label = word_labels[word_idx]
                        token_labels.append(self.LABEL_TO_ID.get(label, 0))
                    else:
                        token_labels.append(0)
                else:
                    # Inside token - use I- if previous was B-
                    if word_idx < len(word_labels):
                        prev_label = word_labels[word_idx]
                        if prev_label.startswith('B-'):
                            label = 'I-' + prev_label[2:]
                            token_labels.append(self.LABEL_TO_ID.get(label, 0))
                        else:
                            token_labels.append(self.LABEL_TO_ID.get(prev_label, 0))
                    else:
                        token_labels.append(0)
                
                previous_word_idx = word_idx
            
            batch_labels.append(token_labels)
        
        # Pad to same length
        max_len = max(len(l) for l in batch_labels)
        batch_labels = [l + [-100] * (max_len - len(l)) for l in batch_labels]
        
        return torch.tensor(batch_labels)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch.get('labels')
        )
        
        result = {'logits': outputs.logits}
        if outputs.loss is not None:
            result['loss'] = outputs.loss
        
        return result
    
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss."""
        return outputs.get('loss', torch.tensor(0.0))
    
    def predict(self, texts: List[str]) -> List[List[Entity]]:
        """
        Extract entities from texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of entity lists (one per text)
        """
        self.model.eval()
        
        # Preprocess
        encoded = self.preprocess(texts)
        encoded = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in encoded.items()}
        
        # Forward
        with torch.no_grad():
            outputs = self.forward(encoded)
        
        # Decode predictions
        predictions = torch.argmax(outputs['logits'], dim=-1)
        
        all_entities = []
        
        for i, (text, pred, offset) in enumerate(zip(
            texts, 
            predictions, 
            encoded['offset_mapping']
        )):
            entities = self._decode_entities(text, pred.cpu().tolist(), offset.cpu().tolist())
            all_entities.append(entities)
        
        return all_entities
    
    def _decode_entities(
        self,
        text: str,
        predictions: List[int],
        offsets: List[Tuple[int, int]]
    ) -> List[Entity]:
        """Decode BIO predictions to entities."""
        entities = []
        current_entity = None
        current_start = None
        current_end = None
        
        for pred_id, (start, end) in zip(predictions, offsets):
            if start == end:  # Special token
                if current_entity:
                    entities.append(Entity(
                        text=text[current_start:current_end],
                        label=current_entity,
                        start_char=current_start,
                        end_char=current_end
                    ))
                    current_entity = None
                continue
            
            label = self.ID_TO_LABEL.get(pred_id, 'O')
            
            if label.startswith('B-'):
                # New entity
                if current_entity:
                    entities.append(Entity(
                        text=text[current_start:current_end],
                        label=current_entity,
                        start_char=current_start,
                        end_char=current_end
                    ))
                current_entity = label[2:]
                current_start = start
                current_end = end
            elif label.startswith('I-') and current_entity == label[2:]:
                # Continue entity
                current_end = end
            else:
                # End current entity
                if current_entity:
                    entities.append(Entity(
                        text=text[current_start:current_end],
                        label=current_entity,
                        start_char=current_start,
                        end_char=current_end
                    ))
                    current_entity = None
        
        # Don't forget last entity
        if current_entity:
            entities.append(Entity(
                text=text[current_start:current_end],
                label=current_entity,
                start_char=current_start,
                end_char=current_end
            ))
        
        return entities
    
    def extract_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities grouped by type.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity type to list of entity texts
        """
        entities = self.predict([text])[0]
        
        grouped = {etype: [] for etype in self.ENTITY_TYPES}
        for entity in entities:
            if entity.label in grouped:
                grouped[entity.label].append(entity.text)
        
        return grouped


# Test
if __name__ == '__main__':
    extractor = EntityExtractor()
    
    text = """
    Maya walked through the enchanted forest at dawn. 
    She picked up an ancient golden key from the mossy ground.
    The old wizard Merlin appeared before her with a mysterious smile.
    """
    
    entities = extractor.predict([text])
    
    print("Extracted Entities:")
    for entity in entities[0]:
        print(f"  {entity.label}: '{entity.text}'")
