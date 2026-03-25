"""
Save validation split from training data.

This script loads your training data, splits it using the same logic as train.py,
and saves the validation portion to a separate JSON file.
"""

import argparse
import json
import random
import numpy as np
import torch
from torch.utils.data import random_split
from data import KnapsackDataset

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_validation_split(train_path, val_path, val_split=0.1, seed=1337):
    """
    Split training data and save validation portion to file.
    
    Args:
        train_path: Path to training JSON file
        val_path: Path to save validation JSON file
        val_split: Fraction of data for validation (default 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    # Set seed for reproducible split
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Load full dataset
    print(f"\nLoading training data from: {train_path}")
    full_dataset = KnapsackDataset(train_path)
    total_size = len(full_dataset)
    print(f"Total instances: {total_size}")
    
    # Calculate split sizes
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    print(f"Train size: {train_size} ({100*(1-val_split):.1f}%)")
    print(f"Val size:   {val_size} ({100*val_split:.1f}%)")
    
    # Split dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Get indices of validation instances
    val_indices = val_dataset.indices
    print(f"\nValidation indices: {val_indices[:10]}... (showing first 10)")
    
    # Load original JSON to extract validation instances
    with open(train_path, 'r') as f:
        original_data = json.load(f)
    
    original_instances = original_data['instances']
    
    # Extract validation instances
    val_instances = [original_instances[i] for i in val_indices]
    
    # Create validation JSON
    val_data = {
        'instances': val_instances
    }
    
    # Save validation file
    print(f"\nSaving validation data to: {val_path}")
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"✓ Saved {len(val_instances)} validation instances")
    
    # Print sample
    print(f"\nSample validation instance (first one):")
    sample = val_instances[0]
    print(f"  Number of steps: {len(sample['steps'])}")
    print(f"  First step k: {sample['steps'][0].get('k', 1)}")
    print(f"  Last step k: {sample['steps'][-1].get('k', len(sample['steps']))}")


def main():
    parser = argparse.ArgumentParser(description='Save validation split from training data')
    parser.add_argument('--train', type=str, required=True,
                       help='Path to training JSON file')
    parser.add_argument('--val', type=str, required=True,
                       help='Path to save validation JSON file')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1 = 10%%)')
    parser.add_argument('--seed', type=int, default=1337,
                       help='Random seed (must match training seed!)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0 < args.val_split < 1):
        raise ValueError("val_split must be between 0 and 1")
    
    # Save validation split
    save_validation_split(
        train_path=args.train,
        val_path=args.val,
        val_split=args.val_split,
        seed=args.seed
    )
    
    print("\n" + "="*60)
    print("IMPORTANT: Use the SAME seed when training!")
    print(f"  python train.py --train {args.train} --seed {args.seed}")
    print("="*60)


if __name__ == '__main__':
    main()