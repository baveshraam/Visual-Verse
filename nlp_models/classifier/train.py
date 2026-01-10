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
from typing import Optional, List, Dict
from datetime import datetime

import torch
import numpy as np
from transformers import DistilBertTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from nlp_models.classifier.model import DomainClassifier
from nlp_models.classifier.dataset import ClassifierDataset, MultilingualClassifierDataset
from nlp_models.base import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_training_report(
    classifier: DomainClassifier,
    val_dataset: ClassifierDataset,
    history: List,
    output_path: str,
    sample_count: int = 5
):
    """
    Generate a comprehensive training report.
    
    Args:
        classifier: Trained classifier
        val_dataset: Validation dataset
        history: Training history
        output_path: Path to save report
        sample_count: Number of sample predictions to include
    """
    from sklearn.metrics import confusion_matrix, classification_report
    from torch.utils.data import DataLoader
    
    logger.info("Generating training report...")
    
    # Create reports directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Collect predictions on validation set
    classifier.model.eval()
    all_preds = []
    all_labels = []
    all_texts = []
    all_probs = []
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch_device = {k: v.to(classifier.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
            
            outputs = classifier.forward(batch_device)
            probs = torch.softmax(outputs['logits'], dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch['labels'].numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
            # Get texts for sample predictions
            batch_start = i * 16
            for j in range(len(preds)):
                if batch_start + j < len(val_dataset.data):
                    all_texts.append(val_dataset.data[batch_start + j]['text'])
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Find correct and incorrect predictions
    correct_indices = [i for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]
    incorrect_indices = [i for i in range(len(all_preds)) if all_preds[i] != all_labels[i]]
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("DISTILBERT CLASSIFIER TRAINING REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    
    # Device info
    report_lines.append("\n## TRAINING CONFIGURATION")
    report_lines.append(f"Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        report_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    report_lines.append(f"Model: distilbert-base-uncased")
    report_lines.append(f"Epochs: {len(history)}")
    report_lines.append(f"Validation samples: {len(val_dataset)}")
    
    # Final metrics
    report_lines.append("\n## FINAL VALIDATION METRICS")
    report_lines.append(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    if history:
        final = history[-1]
        report_lines.append(f"Final Train Loss: {final.train_loss:.4f}")
        if final.val_loss:
            report_lines.append(f"Final Val Loss: {final.val_loss:.4f}")
        if final.f1_score:
            report_lines.append(f"F1 Score: {final.f1_score:.4f}")
    
    # Confusion Matrix
    report_lines.append("\n## CONFUSION MATRIX")
    report_lines.append("                  Predicted")
    report_lines.append("                  Info    Narr")
    report_lines.append(f"Actual Info       {cm[0][0]:5d}   {cm[0][1]:5d}")
    report_lines.append(f"Actual Narr       {cm[1][0]:5d}   {cm[1][1]:5d}")
    
    # Per-class metrics
    report_lines.append("\n## CLASSIFICATION REPORT")
    labels = ['informational', 'narrative']
    report_lines.append(classification_report(all_labels, all_preds, target_names=labels))
    
    # Training loss curve data
    report_lines.append("\n## TRAINING LOSS CURVE DATA")
    report_lines.append("Epoch | Train Loss | Val Loss   | Accuracy")
    report_lines.append("-" * 50)
    for m in history:
        val_loss = f"{m.val_loss:.4f}" if m.val_loss else "N/A"
        acc = f"{m.accuracy:.4f}" if m.accuracy else "N/A"
        report_lines.append(f"  {m.epoch:2d}  |   {m.train_loss:.4f}   |   {val_loss}   |  {acc}")
    
    # Sample predictions - Correct
    report_lines.append(f"\n## SAMPLE CORRECT PREDICTIONS ({min(sample_count, len(correct_indices))})")
    for i in correct_indices[:sample_count]:
        if i < len(all_texts):
            true_label = labels[all_labels[i]]
            pred_label = labels[all_preds[i]]
            conf = max(all_probs[i])
            text_preview = all_texts[i][:100] + "..." if len(all_texts[i]) > 100 else all_texts[i]
            report_lines.append(f"\n[{i}] Predicted: {pred_label} (conf: {conf:.2%})")
            report_lines.append(f"    Text: {text_preview}")
    
    # Sample predictions - Incorrect
    report_lines.append(f"\n## SAMPLE INCORRECT PREDICTIONS ({min(sample_count, len(incorrect_indices))})")
    for i in incorrect_indices[:sample_count]:
        if i < len(all_texts):
            true_label = labels[all_labels[i]]
            pred_label = labels[all_preds[i]]
            conf = max(all_probs[i])
            text_preview = all_texts[i][:100] + "..." if len(all_texts[i]) > 100 else all_texts[i]
            report_lines.append(f"\n[{i}] Predicted: {pred_label}, Actual: {true_label} (conf: {conf:.2%})")
            report_lines.append(f"    Text: {text_preview}")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)
    
    # Write report
    report_content = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Training report saved to: {output_path}")
    return report_content


def train_classifier(
    data_path: str,
    output_dir: str = 'models/nlp_models/classifier/checkpoint',
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_split: float = 0.2,
    seed: int = 42,
    report_path: str = 'reports/classifier_training_report.txt'
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
        report_path: Path for training report
    """
    logger.info("=" * 60)
    logger.info("TRAINING DOMAIN CLASSIFIER")
    logger.info("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("⚠ CUDA not available. Training on CPU (slower).")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
        with open(data_path, encoding='utf-8') as f:
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
    
    # Count label distribution
    label_counts = {'informational': 0, 'narrative': 0}
    for item in dataset.data:
        label_counts[item['label']] = label_counts.get(item['label'], 0) + 1
    logger.info(f"Label distribution: {label_counts}")
    
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
    
    # Generate training report
    generate_training_report(
        classifier=classifier,
        val_dataset=val_dataset,
        history=history,
        output_path=report_path
    )
    
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
    
    logger.info(f"\nOutputs:")
    logger.info(f"  - Model: {final_path}")
    logger.info(f"  - History: {history_path}")
    logger.info(f"  - Report: {report_path}")
    
    return classifier


def main():
    parser = argparse.ArgumentParser(description='Train Domain Classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to training data JSON')
    parser.add_argument('--output', type=str, default='models/nlp_models/classifier/checkpoint', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--report', type=str, default='reports/classifier_training_report.txt', help='Report output path')
    
    args = parser.parse_args()
    
    train_classifier(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        report_path=args.report
    )


if __name__ == '__main__':
    main()
