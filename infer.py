import argparse
import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F

from model import *
from data import *
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Inference for Knapsack GRU Solver')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model weights (.pt file)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file (Schema B - unlabeled)')
    parser.add_argument('--out', type=str, required=True,
                        help='Path to output predictions JSON file (Schema C)')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for reproducibility')
    parser.add_argument('--deterministic', type=int, default=0, choices=[0, 1],
                        help='Enable deterministic mode (0 or 1)')
    
    # Optional model hyperparameters (must match training)
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size for GRU (must match training)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GRU layers (must match training)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (must match training)')
    
    args = parser.parse_args()
    
    # Set seed and determinism
    set_seed(args.seed, args.deterministic == 1)
    print(f"Random seed set to: {args.seed}")
    print(f"Deterministic mode: {'ON' if args.deterministic == 1 else 'OFF'}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(
        args.model,
        device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Load test data
    instances = load_test_data(args.input)
    
    # Run inference
    predictions = run_inference(model, instances, device)
    
    # Validate predictions
    validate_predictions(predictions, instances)
    
    # Save predictions
    save_predictions(predictions, args.out)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.out}")
    print(f"Instances processed: {len(instances)}")


if __name__ == "__main__":
    main()