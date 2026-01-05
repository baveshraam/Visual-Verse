"""
Domain Classifier Model
Fine-tuned DistilBERT for classifying text as narrative vs informational.

Trained on:
- Translated Indic folktales (Tamil/Malayalam → English)
- WikiHow articles
- arXiv abstracts

Novelty: First classifier evaluated on Indic-origin English translations.

Supported Languages: English, Tamil, Malayalam, Kannada, Telugu, Hindi
(via IndicTrans2 translation preprocessing)
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle imports gracefully
try:
    from transformers import (
        DistilBertTokenizer, 
        DistilBertForSequenceClassification,
        DistilBertModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Install with: pip install transformers")

# Import base class
try:
    from ..base import BaseNLPModel, ModelConfig
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from nlp_models.base import BaseNLPModel, ModelConfig


class DomainClassifier(BaseNLPModel):
    """
    DistilBERT-based classifier for narrative vs informational text.
    
    Architecture:
        DistilBERT → [CLS] pooling → Dropout → Linear(768, 2)
    
    Labels:
        0 = informational (explanatory)
        1 = narrative (story)
    """
    
    LABEL_NAMES = ['informational', 'narrative']
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                model_name='domain_classifier',
                base_model='distilbert-base-uncased',
                num_labels=2,
                max_length=512,
                batch_size=16,
                learning_rate=2e-5,
                num_epochs=3,
                label_names=self.LABEL_NAMES
            )
        super().__init__(config)
        
        self.build_model()
        self.load_tokenizer()
    
    def build_model(self):
        """Build DistilBERT classifier."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for DomainClassifier. "
                "Install with: pip install transformers"
            )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config.base_model,
            num_labels=self.config.num_labels
        )
        self.model.to(self.device)
    
    def load_tokenizer(self):
        """Load DistilBERT tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for tokenizer. "
                "Install with: pip install transformers"
            )
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.config.base_model
        )
    
    def preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts for DistilBERT.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary with input_ids, attention_mask
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        return encoded
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through classifier.
        
        Args:
            batch: Dictionary with input_ids, attention_mask, (optional) labels
            
        Returns:
            Dictionary with loss (if labels) and logits
        """
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
        """Compute cross-entropy loss."""
        if 'loss' in outputs:
            return outputs['loss']
        
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(outputs['logits'], labels)
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify texts as narrative or informational.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predictions with label and confidence
        """
        self.model.eval()
        
        # Preprocess
        encoded = self.preprocess(texts)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(encoded)
        
        # Get predictions
        probs = torch.softmax(outputs['logits'], dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            label = self.LABEL_NAMES[pred.item()]
            confidence = prob[pred].item()
            results.append({
                'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                'label': label,
                'label_id': pred.item(),
                'confidence': confidence,
                'narrative_score': prob[1].item(),
                'informational_score': prob[0].item()
            })
        
        return results
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Classify a single text."""
        return self.predict([text])[0]
    
    def is_narrative(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text is narrative (story-like)."""
        result = self.predict_single(text)
        return result['narrative_score'] >= threshold
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get [CLS] token embeddings for texts.
        
        Useful for downstream tasks or visualization.
        """
        self.model.eval()
        
        encoded = self.preprocess(texts)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            # Get base model outputs (before classification head)
            base_model = self.model.distilbert
            outputs = base_model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            # [CLS] token is first token
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings


# Test the model
if __name__ == '__main__':
    # Initialize classifier
    classifier = DomainClassifier()
    
    # Test texts
    narrative = """
    Once upon a time, in a small village, there lived a young girl named Maya.
    She dreamed of exploring the world beyond the mountains. One sunny morning,
    she discovered a hidden path behind the old oak tree. "This must lead somewhere
    magical!" she exclaimed with excitement.
    """
    
    informational = """
    Machine learning is a subset of artificial intelligence that enables computers
    to learn from data. It consists of three main types: supervised learning,
    unsupervised learning, and reinforcement learning. Supervised learning uses
    labeled data to train models, while unsupervised learning discovers patterns
    in unlabeled data.
    """
    
    # Predict
    results = classifier.predict([narrative, informational])
    
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Label: {result['label']} ({result['confidence']:.2%})")
        print(f"Narrative: {result['narrative_score']:.2%}, Informational: {result['informational_score']:.2%}")
