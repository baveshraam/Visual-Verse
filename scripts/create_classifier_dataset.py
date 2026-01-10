"""
Script to create a balanced training dataset for text classification.
Combines AG News (informational) with Children's Stories (narrative).
"""

import json
import random
from datasets import load_dataset

def filter_by_length(text: str, min_len: int = 50, max_len: int = 500) -> bool:
    """Check if text length is within the specified range."""
    return min_len <= len(text) <= max_len

def load_ag_news_samples(num_samples: int = 1500) -> list[dict]:
    """
    Load AG News dataset and get balanced samples from all 4 categories.
    Categories: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
    """
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news", split="train")
    
    # Get samples per category (balanced)
    samples_per_category = num_samples // 4
    extra = num_samples % 4
    
    category_samples = {0: [], 1: [], 2: [], 3: []}
    
    for item in dataset:
        cat = item['label']
        text = item['text']
        if filter_by_length(text) and len(category_samples[cat]) < samples_per_category + (1 if cat < extra else 0):
            category_samples[cat].append({
                "text": text,
                "label": "informational"
            })
    
    # Combine all categories
    all_samples = []
    for cat_samples in category_samples.values():
        all_samples.extend(cat_samples)
    
    print(f"Loaded {len(all_samples)} AG News samples")
    return all_samples

def load_children_stories_samples(filepath: str, num_samples: int = 1500) -> list[dict]:
    """
    Load children's stories and extract suitable text samples.
    Each paragraph/line that meets length criteria becomes a sample.
    """
    print(f"Loading children's stories from {filepath}...")
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text and filter_by_length(text):
                # Skip lines that are just titles (all caps or very short)
                if text.isupper() and len(text) < 50:
                    continue
                # Skip chapter markers and decorative text
                if text.startswith('[Picture:') or text.startswith('*'):
                    continue
                samples.append({
                    "text": text,
                    "label": "narrative"
                })
    
    print(f"Found {len(samples)} suitable narrative samples")
    
    # Randomly select the required number of samples
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    
    print(f"Selected {len(samples)} narrative samples")
    return samples

def main():
    random.seed(42)  # For reproducibility
    
    # Paths
    stories_path = "Dataset-NLP-Models/cleaned_merged_fairy_tales_without_eos.txt"
    output_path = "data/classifier/train.json"
    
    # Load datasets
    ag_news_samples = load_ag_news_samples(1500)
    narrative_samples = load_children_stories_samples(stories_path, 1500)
    
    # Combine datasets
    combined = ag_news_samples + narrative_samples
    print(f"\nTotal combined samples: {len(combined)}")
    
    # Shuffle
    random.shuffle(combined)
    print("Dataset shuffled")
    
    # Verify balance
    info_count = sum(1 for s in combined if s['label'] == 'informational')
    narr_count = sum(1 for s in combined if s['label'] == 'narrative')
    print(f"\nDataset composition:")
    print(f"  - Informational: {info_count}")
    print(f"  - Narrative: {narr_count}")
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset saved to {output_path}")
    
    # Show some examples
    print("\n--- Sample entries ---")
    for i, sample in enumerate(combined[:3]):
        print(f"\nExample {i+1} ({sample['label']}):")
        print(f"  Text: {sample['text'][:100]}...")

if __name__ == "__main__":
    main()
