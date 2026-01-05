"""
Base NLP Model Classes
Provides common functionality for all trainable NLP models in VisualVerse.
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for NLP models."""
    model_name: str
    base_model: str
    num_labels: int = 2
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "checkpoints"
    label_names: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelConfig':
        return cls(**d)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ModelConfig':
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class BaseNLPModel(ABC):
    """
    Abstract base class for all trainable NLP models.
    
    Provides:
    - Model loading/saving
    - Training loop structure
    - Evaluation framework
    - Logging and checkpointing
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.device)
        self.training_history: List[TrainingMetrics] = []
        
        # Set random seeds
        self._set_seeds(config.seed)
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @abstractmethod
    def build_model(self):
        """Build the model architecture. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def load_tokenizer(self):
        """Load the tokenizer. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Preprocess text for model input."""
        pass
    
    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def compute_loss(self, outputs: Dict, labels: torch.Tensor) -> torch.Tensor:
        """Compute training loss."""
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[Any]:
        """Make predictions on new texts."""
        pass
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward(batch)
            loss = self.compute_loss(outputs, batch.get('labels'))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(
        self, 
        eval_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.forward(batch)
                loss = self.compute_loss(outputs, batch.get('labels'))
                
                total_loss += loss.item()
                
                # Get predictions
                preds = outputs.get('logits', outputs.get('predictions'))
                if preds is not None:
                    if len(preds.shape) > 1:
                        preds = torch.argmax(preds, dim=-1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(batch['labels'].cpu().numpy().tolist())
        
        avg_loss = total_loss / len(eval_loader)
        metrics = self._compute_metrics(all_preds, all_labels)
        
        return avg_loss, metrics
    
    def _compute_metrics(
        self, 
        predictions: List, 
        labels: List
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Flatten if needed
        if isinstance(predictions[0], list):
            predictions = [p for sublist in predictions for p in sublist]
            labels = [l for sublist in labels for l in sublist]
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted'),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0)
        }
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        save_best: bool = True
    ) -> List[TrainingMetrics]:
        """
        Full training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            save_best: Whether to save the best model
            
        Returns:
            List of training metrics per epoch
        """
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.batch_size
            )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            logger.info(f"  Train Loss: {train_loss:.4f}")
            
            metrics = TrainingMetrics(epoch=epoch + 1, train_loss=train_loss)
            
            # Evaluate
            if val_loader:
                val_loss, val_metrics = self.evaluate(val_loader)
                metrics.val_loss = val_loss
                metrics.accuracy = val_metrics.get('accuracy')
                metrics.f1_score = val_metrics.get('f1')
                metrics.precision = val_metrics.get('precision')
                metrics.recall = val_metrics.get('recall')
                
                logger.info(f"  Val Loss: {val_loss:.4f}, Accuracy: {metrics.accuracy:.4f}")
                
                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save(os.path.join(self.config.output_dir, 'best'))
                    logger.info("  Saved best model!")
            
            self.training_history.append(metrics)
        
        return self.training_history
    
    def save(self, path: str):
        """Save model and config to path."""
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        
        # Save config
        self.config.save(os.path.join(path, 'config.json'))
        
        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from path."""
        # Load config
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)
            self.config = ModelConfig.from_dict(config_dict)
        
        # Rebuild model
        self.build_model()
        
        # Load weights
        weights_path = os.path.join(path, 'model.pt')
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        # Load tokenizer
        self.load_tokenizer()
        if hasattr(self.tokenizer, 'from_pretrained'):
            try:
                self.tokenizer = self.tokenizer.from_pretrained(path)
            except Exception as e:
                logger.debug(f"Could not load tokenizer from {path}: {e}")
        
        logger.info(f"Model loaded from {path}")
    
    def to(self, device: str):
        """Move model to device."""
        self.device = torch.device(device)
        if self.model:
            self.model.to(self.device)
        return self
