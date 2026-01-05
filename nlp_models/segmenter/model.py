"""
Scene Segmenter Model
BERT + BiLSTM + CRF for sequence labeling to detect story scene boundaries.

Trained on:
- VIST dataset
- Translated Indic stories with manual scene annotations

Labels:
- B-SCENE: Beginning of a scene
- I-SCENE: Inside a scene (continuation)
- E-SCENE: End of a scene  
- O: Outside (sentence not part of structured scene)

Supported Languages: English, Tamil, Malayalam, Kannada, Telugu, Hindi
(via IndicTrans2 translation preprocessing)

Novelty: First scene segmenter evaluated on Indic-origin narratives.
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle imports gracefully
try:
    from transformers import BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Install with: pip install transformers")

try:
    from torchcrf import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    logger.warning("pytorch-crf not installed. Install with: pip install pytorch-crf")

# Import base class
try:
    from ..base import BaseNLPModel, ModelConfig
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from nlp_models.base import BaseNLPModel, ModelConfig


class BertCRFModel(nn.Module):
    """
    BERT + BiLSTM + CRF for sequence labeling.
    
    Architecture:
        BERT → BiLSTM → Linear → CRF
    """
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        num_labels: int = 4,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.1
    ):
        # Check dependencies before initializing
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for BertCRFModel. "
                "Install with: pip install transformers"
            )
        if not CRF_AVAILABLE:
            raise ImportError(
                "pytorch-crf library is required for BertCRFModel. "
                "Install with: pip install pytorch-crf"
            )
        
        super().__init__()
        
        self.num_labels = num_labels
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Projection to label space
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Optional labels for training [batch, seq_len]
            
        Returns:
            Dictionary with 'loss' (if labels) and 'emissions'
        """
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_outputs.last_hidden_state  # [batch, seq, hidden]
        
        # BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # [batch, seq, lstm_hidden*2]
        
        # Projection
        lstm_output = self.dropout(lstm_output)
        emissions = self.classifier(lstm_output)  # [batch, seq, num_labels]
        
        result = {'emissions': emissions}
        
        if labels is not None:
            # Compute CRF loss (negative log-likelihood)
            # CRF expects mask as bool
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            result['loss'] = loss
        
        return result
    
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[List[int]]:
        """
        Decode best label sequences using Viterbi.
        
        Returns:
            List of label sequences
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            emissions = outputs['emissions']
            mask = attention_mask.bool()
            
            # Viterbi decoding
            predictions = self.crf.decode(emissions, mask=mask)
        
        return predictions


class SceneSegmenter(BaseNLPModel):
    """
    BERT-CRF based scene segmenter for stories.
    
    Segments stories into coherent scenes by labeling each sentence
    with scene boundary tags.
    """
    
    LABEL_NAMES = ['O', 'B-SCENE', 'I-SCENE', 'E-SCENE']
    LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_NAMES)}
    ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                model_name='scene_segmenter',
                base_model='bert-base-uncased',
                num_labels=4,
                max_length=512,
                batch_size=8,
                learning_rate=3e-5,
                num_epochs=5,
                label_names=self.LABEL_NAMES
            )
        super().__init__(config)
        
        self.build_model()
        self.load_tokenizer()
    
    def build_model(self):
        """Build BERT-CRF model."""
        self.model = BertCRFModel(
            bert_model_name=self.config.base_model,
            num_labels=self.config.num_labels,
            dropout=self.config.dropout
        )
        self.model.to(self.device)
    
    def load_tokenizer(self):
        """Load BERT tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(self.config.base_model)
    
    def preprocess(
        self, 
        sentences: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize sentences for BERT-CRF.
        
        Args:
            sentences: List of sentences (one story)
            labels: Optional list of labels per sentence
            
        Returns:
            Dictionary with input_ids, attention_mask, (optional) labels
        """
        # Tokenize all sentences together with sentence boundaries
        # We need to track which tokens belong to which sentence
        
        all_input_ids = []
        all_attention_mask = []
        all_labels = [] if labels else None
        
        # Add [CLS] token
        input_ids = [self.tokenizer.cls_token_id]
        attention_mask = [1]
        label_ids = [self.LABEL_TO_ID['O']] if labels else None
        
        for i, sent in enumerate(sentences):
            # Tokenize sentence
            encoded = self.tokenizer.encode(sent, add_special_tokens=False)
            
            # Check if we need to truncate
            if len(input_ids) + len(encoded) + 1 > self.config.max_length:
                break
            
            input_ids.extend(encoded)
            attention_mask.extend([1] * len(encoded))
            
            if labels:
                # All tokens in sentence get same label
                label_id = self.LABEL_TO_ID.get(labels[i], 0)
                label_ids.extend([label_id] * len(encoded))
        
        # Add [SEP] token
        input_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)
        if labels:
            label_ids.append(self.LABEL_TO_ID['O'])
        
        # Pad to max_length
        padding_length = self.config.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        if labels:
            label_ids.extend([self.LABEL_TO_ID['O']] * padding_length)
        
        result = {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }
        
        if labels:
            result['labels'] = torch.tensor([label_ids])
        
        return result
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
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
        """Compute CRF loss."""
        return outputs.get('loss', torch.tensor(0.0))
    
    def predict(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Segment a story into scenes.
        
        Args:
            sentences: List of sentences forming a story
            
        Returns:
            List of dicts with sentence and predicted label
        """
        self.model.eval()
        
        # Preprocess
        encoded = self.preprocess(sentences)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Decode with CRF
        predictions = self.model.decode(
            encoded['input_ids'],
            encoded['attention_mask']
        )[0]  # Single batch
        
        # Map back to sentences
        # This is simplified - in practice need to track token-to-sentence mapping
        results = []
        
        # Skip [CLS] and [SEP], map predictions back
        pred_idx = 1  # Start after [CLS]
        
        for sent in sentences:
            sent_tokens = self.tokenizer.encode(sent, add_special_tokens=False)
            
            if pred_idx < len(predictions):
                # Get majority label for this sentence's tokens
                sent_preds = predictions[pred_idx:pred_idx + len(sent_tokens)]
                if sent_preds:
                    majority_label = max(set(sent_preds), key=sent_preds.count)
                else:
                    majority_label = 0
                
                label = self.ID_TO_LABEL.get(majority_label, 'O')
            else:
                label = 'O'
            
            results.append({
                'sentence': sent,
                'label': label,
                'is_scene_boundary': label.startswith('B-') or label.startswith('E-')
            })
            
            pred_idx += len(sent_tokens)
        
        return results
    
    def segment_story(self, story: str) -> List[Dict[str, Any]]:
        """
        Segment a story text into scenes.
        
        Args:
            story: Full story text
            
        Returns:
            List of scenes with their sentences
        """
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', story.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Predict labels
        predictions = self.predict(sentences)
        
        # Group into scenes
        scenes = []
        current_scene = []
        
        for pred in predictions:
            if pred['label'] == 'B-SCENE':
                if current_scene:
                    scenes.append(current_scene)
                current_scene = [pred]
            elif pred['label'] == 'E-SCENE':
                current_scene.append(pred)
                scenes.append(current_scene)
                current_scene = []
            else:
                current_scene.append(pred)
        
        if current_scene:
            scenes.append(current_scene)
        
        return scenes


# Test
if __name__ == '__main__':
    segmenter = SceneSegmenter()
    
    story = """
    Once upon a time, in a small village, there lived a young girl named Maya.
    She had always dreamed of exploring the world beyond the mountains.
    One sunny morning, Maya discovered a hidden path behind the old oak tree.
    "This must lead somewhere magical!" she exclaimed.
    She followed the path deeper into the woods.
    The trees grew taller and the sunlight filtered through the leaves.
    Suddenly, she arrived at a clearing.
    In the center stood an ancient stone tower covered in ivy.
    A mysterious light glowed from the highest window.
    Maya knew her adventure was just beginning.
    """
    
    scenes = segmenter.segment_story(story)
    
    print("Story Segmentation Results:")
    for i, scene in enumerate(scenes):
        print(f"\n=== Scene {i+1} ===")
        for sent in scene:
            print(f"  [{sent['label']}] {sent['sentence'][:60]}...")
