"""
Dataset Classes for Scene Segmenter
Handles loading and preprocessing of scene-annotated story data.
"""
import json
import random
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset


class SegmenterDataset(Dataset):
    """
    Dataset for scene segmentation (sequence labeling).
    
    Expected data format (JSON):
    [
        {
            "story_id": "story_1",
            "sentences": [
                {"text": "...", "label": "B-SCENE"},
                {"text": "...", "label": "I-SCENE"},
                ...
            ],
            "source_lang": "ta",
            "is_translated": true
        },
        ...
    ]
    """
    
    LABEL_MAP = {'O': 0, 'B-SCENE': 1, 'I-SCENE': 2, 'E-SCENE': 3}
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        data: Optional[List[Dict]] = None,
        tokenizer=None,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = self._load_data(data_path)
        else:
            self.data = []
    
    def _load_data(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        sentences = item['sentences']
        
        if not self.tokenizer:
            raise ValueError("Tokenizer required for SegmenterDataset")
        
        # Build input sequence with labels aligned to tokens
        input_ids = [self.tokenizer.cls_token_id]
        labels = [self.LABEL_MAP['O']]  # [CLS] gets O label
        
        for sent_item in sentences:
            text = sent_item['text']
            label = sent_item['label']
            label_id = self.LABEL_MAP.get(label, 0)
            
            # Tokenize sentence
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Check truncation
            if len(input_ids) + len(tokens) + 1 > self.max_length:
                break
            
            input_ids.extend(tokens)
            labels.extend([label_id] * len(tokens))
        
        # Add [SEP]
        input_ids.append(self.tokenizer.sep_token_id)
        labels.append(self.LABEL_MAP['O'])
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad
        padding_length = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        labels.extend([self.LABEL_MAP['O']] * padding_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def split(
        self, 
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple['SegmenterDataset', 'SegmenterDataset']:
        """Split dataset into train and validation sets."""
        data = self.data.copy()
        if shuffle:
            random.seed(seed)
            random.shuffle(data)
        
        split_idx = int(len(data) * train_ratio)
        
        return (
            SegmenterDataset(data=data[:split_idx], tokenizer=self.tokenizer, max_length=self.max_length),
            SegmenterDataset(data=data[split_idx:], tokenizer=self.tokenizer, max_length=self.max_length)
        )
    
    @classmethod
    def from_stories(
        cls,
        stories: List[Dict[str, List[str]]],
        tokenizer=None,
        max_length: int = 512
    ) -> 'SegmenterDataset':
        """
        Create dataset from stories with scene annotations.
        
        Args:
            stories: List of {'sentences': [...], 'scene_breaks': [indices]}
        """
        data = []
        
        for i, story in enumerate(stories):
            sentences = story['sentences']
            breaks = set(story.get('scene_breaks', []))
            
            labeled_sents = []
            for j, sent in enumerate(sentences):
                if j in breaks:
                    label = 'B-SCENE'
                elif j + 1 in breaks or j == len(sentences) - 1:
                    label = 'E-SCENE'
                else:
                    label = 'I-SCENE'
                
                labeled_sents.append({'text': sent, 'label': label})
            
            data.append({
                'story_id': f'story_{i}',
                'sentences': labeled_sents,
                'source_lang': story.get('source_lang', 'en'),
                'is_translated': story.get('is_translated', False)
            })
        
        return cls(data=data, tokenizer=tokenizer, max_length=max_length)
