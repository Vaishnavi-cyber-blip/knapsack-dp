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

class KnapsackDataset(Dataset):
    def __init__(self, json_path, n_max=1024, w_max=512):

        self.n_max = n_max
        self.w_max = w_max
        
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.instances = data['instances']
        print(f"Loaded {len(self.instances)} instances from {json_path}")
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):

        instance = self.instances[idx]
        steps = instance['steps']
        n = len(steps)
        
        # Extract data
        values = torch.zeros(n, dtype=torch.float32)
        weights = torch.zeros(n, dtype=torch.float32)
        step_indices = torch.zeros(n, dtype=torch.long)
        dp_rows = torch.zeros(n, self.w_max + 1, dtype=torch.float32)
        opts = torch.zeros(n, dtype=torch.float32)
        ys = torch.zeros(n, self.n_max, dtype=torch.float32)
        
        for i, step in enumerate(steps):
            values[i] = step['v']
            weights[i] = step['w']
            step_indices[i] = step['k']
            dp_rows[i] = torch.tensor(step['dp_row'], dtype=torch.float32)
            opts[i] = step['opt']
            ys[i] = torch.tensor(step['y'], dtype=torch.float32)
        
        return {
            'values': values,
            'weights': weights,
            'step_indices': step_indices,
            'dp_rows': dp_rows,
            'opts': opts,
            'ys': ys,
            'n': n
        }


def collate_fn(batch):

    # Find max length in this batch
    max_n = max(item['n'] for item in batch)
    
    batch_size = len(batch)
    w_max = batch[0]['dp_rows'].shape[1] - 1  # 512
    n_max = batch[0]['ys'].shape[1]  # 1024
    
    # Initialize padded tensors
    values = torch.zeros(batch_size, max_n)
    weights = torch.zeros(batch_size, max_n)
    step_indices = torch.zeros(batch_size, max_n, dtype=torch.long)
    dp_rows = torch.zeros(batch_size, max_n, w_max + 1)
    opts = torch.zeros(batch_size, max_n)
    ys = torch.zeros(batch_size, max_n, n_max)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill in data
    for i, item in enumerate(batch):
        n = item['n']
        values[i, :n] = item['values']
        weights[i, :n] = item['weights']
        step_indices[i, :n] = item['step_indices']
        dp_rows[i, :n] = item['dp_rows']
        opts[i, :n] = item['opts']
        ys[i, :n] = item['ys']
        lengths[i] = n
    
    return {
        'values': values,
        'weights': weights,
        'step_indices': step_indices,
        'dp_rows': dp_rows,
        'opts': opts,
        'ys': ys,
        'lengths': lengths
    }