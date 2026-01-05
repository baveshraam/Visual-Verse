"""
Relation Extractor Model
SciBERT-based model for extracting (subject, predicate, object) triples from text.

Used for:
- Building knowledge graphs from educational content
- Mind-map relation extraction
- Cross-lingual knowledge construction

Example:
Input: "Photosynthesis requires sunlight to convert CO2 to glucose."
Output: [
    ("Photosynthesis", "requires", "sunlight"),
    ("Photosynthesis", "converts", "CO2"),
    ("Photosynthesis", "produces", "glucose")
]

Supported Languages: English, Tamil, Malayalam, Kannada, Telugu, Hindi
(via IndicTrans2 translation preprocessing)

Novelty: Cross-lingual knowledge graph construction from Indic educational content.
"""
import logging
import itertools
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle imports gracefully
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
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
class Triple:
    """A knowledge triple (subject, predicate, object)."""
    subject: str
    predicate: str
    obj: str  # 'object' is reserved keyword
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.obj,
            'confidence': self.confidence
        }
    
    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.obj})"


class RelationClassifier(nn.Module):
    """
    Classifier for relation types between entity pairs.
    
    Architecture:
        SciBERT(text with [E1]...[/E1] [E2]...[/E2] markers) → [CLS] → Linear → Relation Class
    """
    
    def __init__(
        self,
        model_name: str = 'allenai/scibert_scivocab_uncased',
        num_relations: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_relations)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        result = {'logits': logits}
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            result['loss'] = loss
        
        return result


class RelationExtractor(BaseNLPModel):
    """
    SciBERT-based relation extractor for knowledge triple extraction.
    
    Approach:
    1. Extract entities/terms from text
    2. For each entity pair, classify the relation type
    3. Filter high-confidence relations
    """
    
    # Common relation types for educational content
    RELATION_TYPES = [
        'NO_RELATION',
        'is_a',
        'part_of',
        'has_property',
        'caused_by',
        'causes',
        'requires',
        'produces',
        'used_for',
        'located_in'
    ]
    
    RELATION_TO_ID = {r: i for i, r in enumerate(RELATION_TYPES)}
    ID_TO_RELATION = {i: r for r, i in RELATION_TO_ID.items()}
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                model_name='relation_extractor',
                base_model='allenai/scibert_scivocab_uncased',
                num_labels=len(self.RELATION_TYPES),
                max_length=256,
                batch_size=16,
                learning_rate=2e-5,
                num_epochs=5,
                label_names=self.RELATION_TYPES
            )
        super().__init__(config)
        
        self.build_model()
        self.load_tokenizer()
    
    def build_model(self):
        """Build SciBERT relation classifier."""
        self.model = RelationClassifier(
            model_name=self.config.base_model,
            num_relations=self.config.num_labels,
            dropout=self.config.dropout
        )
        self.model.to(self.device)
    
    def load_tokenizer(self):
        """Load SciBERT tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        
        # Add special entity markers
        special_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # Resize embeddings
        self.model.encoder.resize_token_embeddings(len(self.tokenizer))
    
    def _mark_entities(
        self, 
        text: str, 
        entity1: Tuple[int, int, str],
        entity2: Tuple[int, int, str]
    ) -> str:
        """
        Insert entity markers into text.
        
        Args:
            text: Original text
            entity1: (start, end, text) for first entity
            entity2: (start, end, text) for second entity
            
        Returns:
            Text with [E1]...[/E1] [E2]...[/E2] markers
        """
        # Sort entities by position (handle overlapping)
        entities = sorted([(entity1, 'E1'), (entity2, 'E2')], key=lambda x: x[0][0])
        
        result = text
        offset = 0
        
        for (start, end, _), marker in entities:
            start += offset
            end += offset
            
            result = result[:start] + f'[{marker}]' + result[start:end] + f'[/{marker}]' + result[end:]
            offset += len(f'[{marker}]') + len(f'[/{marker}]')
        
        return result
    
    def preprocess(
        self, 
        texts: List[str],
        labels: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts for relation classification."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
        if labels:
            result['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return result
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch.get('labels')
        )
    
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss."""
        return outputs.get('loss', torch.tensor(0.0))
    
    def _extract_noun_phrases(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Extract noun phrases as candidate entities.
        
        Uses simple heuristics. For production, use spaCy NP chunking.
        """
        import re
        
        # Simple pattern: capitalized words or quoted phrases
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Capitalized phrases
            r'"([^"]+)"',  # Quoted text
            r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b',  # Acronyms
        ]
        
        entities = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                entities.append((match.start(), match.end(), match.group(1)))
        
        # Also include simple noun-like words
        # This is a placeholder - use spaCy for real NP extraction
        
        return entities
    
    def predict(self, texts: List[str]) -> List[List[Triple]]:
        """
        Extract relation triples from texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of triple lists (one per text)
        """
        self.model.eval()
        all_triples = []
        
        for text in texts:
            # Extract candidate entities
            entities = self._extract_noun_phrases(text)
            
            if len(entities) < 2:
                all_triples.append([])
                continue
            
            triples = []
            
            # Check all entity pairs
            for e1, e2 in itertools.combinations(entities, 2):
                # Mark entities in text
                marked_text = self._mark_entities(text, e1, e2)
                
                # Classify relation
                encoded = self.preprocess([marked_text])
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                with torch.no_grad():
                    outputs = self.forward(encoded)
                
                probs = torch.softmax(outputs['logits'], dim=-1)
                pred_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_id].item()
                
                # Skip NO_RELATION
                if pred_id != 0 and confidence > 0.5:
                    relation = self.ID_TO_RELATION[pred_id]
                    triples.append(Triple(
                        subject=e1[2],
                        predicate=relation,
                        obj=e2[2],
                        confidence=confidence
                    ))
            
            all_triples.append(triples)
        
        return all_triples
    
    def extract_from_text(self, text: str) -> List[Triple]:
        """Extract triples from a single text."""
        return self.predict([text])[0]
    
    def build_knowledge_graph(
        self, 
        texts: List[str]
    ) -> Dict[str, Any]:
        """
        Build a knowledge graph from multiple texts.
        
        Returns:
            Dictionary with nodes and edges
        """
        all_triples = []
        for text in texts:
            all_triples.extend(self.extract_from_text(text))
        
        # Build graph structure
        nodes = set()
        edges = []
        
        for triple in all_triples:
            nodes.add(triple.subject)
            nodes.add(triple.obj)
            edges.append({
                'source': triple.subject,
                'target': triple.obj,
                'relation': triple.predicate,
                'weight': triple.confidence
            })
        
        return {
            'nodes': list(nodes),
            'edges': edges,
            'triples': [t.to_dict() for t in all_triples]
        }


# Test
if __name__ == '__main__':
    extractor = RelationExtractor()
    
    text = """
    Machine Learning is a subset of Artificial Intelligence. 
    Deep Learning uses Neural Networks with multiple layers.
    Supervised Learning requires labeled data for training.
    """
    
    triples = extractor.extract_from_text(text)
    
    print("Extracted Triples:")
    for triple in triples:
        print(f"  {triple}")
    
    # Build knowledge graph
    kg = extractor.build_knowledge_graph([text])
    print(f"\nKnowledge Graph: {len(kg['nodes'])} nodes, {len(kg['edges'])} edges")
