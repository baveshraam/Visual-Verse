"""
Training Script for Domain Classifier
Fine-tunes DistilBERT on narrative vs informational classification.
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import DistilBertTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from nlp_models.classifier.model import DomainClassifier
from nlp_models.classifier.dataset import ClassifierDataset, MultilingualClassifierDataset
from nlp_models.base import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_classifier(
    data_path: str,
    output_dir: str = 'checkpoints/classifier',
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_split: float = 0.2,
    seed: int = 42
):
    """
    Train the domain classifier.
    
    Args:
        data_path: Path to training data JSON
        output_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        val_split: Validation split ratio
        seed: Random seed
    """
    logger.info("=" * 60)
    logger.info("TRAINING DOMAIN CLASSIFIER")
    logger.info("=" * 60)
    
    # Initialize config
    config = ModelConfig(
        model_name='domain_classifier',
        base_model='distilbert-base-uncased',
        num_labels=2,
        max_length=512,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs,
        seed=seed,
        output_dir=output_dir,
        label_names=['informational', 'narrative']
    )
    
    # Initialize model
    logger.info(f"Initializing model with base: {config.base_model}")
    classifier = DomainClassifier(config)
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config.base_model)
    
    # Load and prepare dataset
    logger.info(f"Loading data from: {data_path}")
    
    if data_path.endswith('.json'):
        with open(data_path) as f:
            data = json.load(f)
        
        # Check if multilingual format
        if data and 'source_lang' in data[0]:
            dataset = MultilingualClassifierDataset(
                data=data,
                tokenizer=tokenizer,
                max_length=config.max_length
            )
            logger.info(f"Language distribution: {dataset.get_language_distribution()}")
        else:
            dataset = ClassifierDataset(
                data=data,
                tokenizer=tokenizer,
                max_length=config.max_length
            )
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Split into train/val
    train_dataset, val_dataset = dataset.split(
        train_ratio=1 - val_split,
        shuffle=True,
        seed=seed
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Train
    logger.info(f"Starting training for {epochs} epochs...")
    history = classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_best=True
    )
    
    # Save final model
    final_path = os.path.join(output_dir, 'final')
    classifier.save(final_path)
    logger.info(f"Final model saved to: {final_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump([m.to_dict() for m in history], f, indent=2)
    logger.info(f"Training history saved to: {history_path}")
    
    # Print final metrics
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    if history:
        final = history[-1]
        logger.info(f"Final Train Loss: {final.train_loss:.4f}")
        if final.val_loss:
            logger.info(f"Final Val Loss: {final.val_loss:.4f}")
        if final.accuracy:
            logger.info(f"Final Accuracy: {final.accuracy:.4f}")
        if final.f1_score:
            logger.info(f"Final F1 Score: {final.f1_score:.4f}")
    
    return classifier


def main():
    parser = argparse.ArgumentParser(description='Train Domain Classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to training data JSON')
    parser.add_argument('--output', type=str, default='checkpoints/classifier', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_classifier(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
