import argparse
import os
import time
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

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True



def load_model(model_path, device, hidden_size=256, num_layers=2, dropout=0.1):
    print(f"Loading model from: {model_path}")
    
    # Create model
    model = KnapsackGRUSolver(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')
    print(f"Model loaded (epoch: {epoch}, val_loss: {val_loss})")
    
    return model


def load_test_data(input_path):
    print(f"Loading test data from: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    instances = data['instances']
    print(f"✓ Loaded {len(instances)} test instances")
    
    return instances


def predict_instance(model, instance, device):

    steps_data = instance['steps']
    n = len(steps_data)
    
    # Extract values and weights
    values = torch.zeros(1, n, dtype=torch.float32)
    weights = torch.zeros(1, n, dtype=torch.float32)
    step_indices = torch.zeros(1, n, dtype=torch.long)
    
    for i, step in enumerate(steps_data):
        values[0, i] = step['v']
        weights[0, i] = step['w']
        step_indices[0, i] = step.get('k', i + 1)
    
    # Move to device
    values = values.to(device)
    weights = weights.to(device)
    step_indices = step_indices.to(device)
    
    # Predict
    with torch.no_grad():
        dp_rows, opts, ys = model.predict_step(values, weights, step_indices, apply_mask=True)
    
    # Convert to numpy
    dp_rows = dp_rows[0].cpu().numpy()  # (n, 513)
    opts = opts[0].cpu().numpy()  # (n,)
    ys = ys[0].cpu().numpy()  # (n, 1024)
    
    # Build predictions for each step
    pred_steps = []
    for i, step in enumerate(steps_data):
        pred_step = {
            'k': step.get('k', i + 1),
            'w': step['w'],
            'v': step['v'],
            'dp_row': dp_rows[i].tolist(),
            'opt': float(opts[i]),
            'y': (ys[i] > 0.5).astype(int).tolist()
        }
        pred_steps.append(pred_step)
    
    return pred_steps


def run_inference(model, instances, device):

    print("\nRunning inference...")
    predictions = {'instances': []}
    
    for idx, instance in enumerate(instances):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Processing instance {idx + 1}/{len(instances)}")
        
        pred_steps = predict_instance(model, instance, device)
        predictions['instances'].append({'steps': pred_steps})
    
    print(f"✓ Inference complete for {len(instances)} instances")
    
    return predictions


def save_predictions(predictions, output_path):
    print(f"\nSaving predictions to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print("✓ Predictions saved successfully")


def validate_predictions(predictions, instances):
    print("\nValidating predictions...")
    
    # Check instance count
    assert len(predictions['instances']) == len(instances), \
        f"Instance count mismatch: {len(predictions['instances'])} != {len(instances)}"
    
    # Check each instance
    for idx, (pred_inst, true_inst) in enumerate(zip(predictions['instances'], instances)):
        pred_steps = pred_inst['steps']
        true_steps = true_inst['steps']
        
        # Check step count
        assert len(pred_steps) == len(true_steps), \
            f"Step count mismatch in instance {idx}: {len(pred_steps)} != {len(true_steps)}"
        
        # Check each step
        for step_idx, pred_step in enumerate(pred_steps):
            # Check required fields
            assert 'k' in pred_step, f"Missing 'k' in instance {idx}, step {step_idx}"
            assert 'w' in pred_step, f"Missing 'w' in instance {idx}, step {step_idx}"
            assert 'v' in pred_step, f"Missing 'v' in instance {idx}, step {step_idx}"
            assert 'dp_row' in pred_step, f"Missing 'dp_row' in instance {idx}, step {step_idx}"
            assert 'opt' in pred_step, f"Missing 'opt' in instance {idx}, step {step_idx}"
            assert 'y' in pred_step, f"Missing 'y' in instance {idx}, step {step_idx}"
            
            # Check lengths
            assert len(pred_step['dp_row']) == 513, \
                f"dp_row must have 513 values in instance {idx}, step {step_idx}, got {len(pred_step['dp_row'])}"
            assert len(pred_step['y']) == 1024, \
                f"y must have 1024 values in instance {idx}, step {step_idx}, got {len(pred_step['y'])}"
    
    print("✓ All validation checks passed")