"""
Dataset Classes for Domain Classifier
Handles loading and preprocessing of training data.
"""
import json
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ClassifierDataset(Dataset):
    """
    Dataset for domain classification (narrative vs informational).
    
    Expected data format (JSON):
    [
        {"text": "...", "label": "narrative"},
        {"text": "...", "label": "informational"},
        ...
    ]
    """
    
    LABEL_MAP = {'informational': 0, 'narrative': 1}
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        data: Optional[List[Dict]] = None,
        tokenizer=None,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON data file
            data: Or provide data directly as list of dicts
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = self._load_data(data_path)
        else:
            self.data = []
    
    def _load_data(self, path: str) -> List[Dict]:
        """Load data from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item['text']
        label = self.LABEL_MAP.get(item['label'], 0)
        
        # Tokenize
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    @classmethod
    def from_texts_and_labels(
        cls,
        texts: List[str],
        labels: List[str],
        tokenizer=None,
        max_length: int = 512
    ) -> 'ClassifierDataset':
        """Create dataset from lists of texts and labels."""
        data = [{'text': t, 'label': l} for t, l in zip(texts, labels)]
        return cls(data=data, tokenizer=tokenizer, max_length=max_length)
    
    def split(
        self, 
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple['ClassifierDataset', 'ClassifierDataset']:
        """Split dataset into train and validation sets."""
        data = self.data.copy()
        if shuffle:
            random.seed(seed)
            random.shuffle(data)
        
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        return (
            ClassifierDataset(data=train_data, tokenizer=self.tokenizer, max_length=self.max_length),
            ClassifierDataset(data=val_data, tokenizer=self.tokenizer, max_length=self.max_length)
        )


class MultilingualClassifierDataset(ClassifierDataset):
    """
    Extended dataset that includes source language metadata.
    
    Expected data format:
    [
        {"text": "...", "label": "narrative", "source_lang": "ta", "is_translated": true},
        ...
    ]
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        data: Optional[List[Dict]] = None,
        tokenizer=None,
        max_length: int = 512,
        include_lang_token: bool = False
    ):
        super().__init__(data_path, data, tokenizer, max_length)
        self.include_lang_token = include_lang_token
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item['text']
        
        # Optionally prepend language token
        if self.include_lang_token and 'source_lang' in item:
            lang = item.get('source_lang', 'en')
            text = f"[{lang.upper()}] {text}"
        
        label = self.LABEL_MAP.get(item['label'], 0)
        
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            result = {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
            # Include metadata
            if 'source_lang' in item:
                result['source_lang'] = item['source_lang']
            if 'is_translated' in item:
                result['is_translated'] = item['is_translated']
            
            return result
        else:
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long),
                'source_lang': item.get('source_lang', 'en'),
                'is_translated': item.get('is_translated', False)
            }
    
    def get_language_distribution(self) -> Dict[str, int]:
        """Get distribution of source languages."""
        lang_counts = {}
        for item in self.data:
            lang = item.get('source_lang', 'en')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        return lang_counts
    
    def filter_by_language(self, lang: str) -> 'MultilingualClassifierDataset':
        """Filter dataset to specific source language."""
        filtered = [d for d in self.data if d.get('source_lang', 'en') == lang]
        return MultilingualClassifierDataset(
            data=filtered,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_lang_token=self.include_lang_token
        )
