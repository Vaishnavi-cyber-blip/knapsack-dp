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

class KnapsackGRUSolver(nn.Module):
    def __init__(
        self,
        input_size=2,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        n_max=1024,
        w_max=512
    ):
        super(KnapsackGRUSolver, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_max = n_max
        self.w_max = w_max
        
        # Input encoding
        self.item_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(n_max + 1, hidden_size)
        
        # GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Three decoder heads
        self.dp_decoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, w_max + 1)
        )
        
        self.opt_decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.y_decoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_max),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, values, weights, step_indices=None):
        """
        Args:
            values: [batch_size, seq_len]
            weights: [batch_size, seq_len]
            step_indices: [batch_size, seq_len]
        
        Returns:
            dp_rows: [batch_size, seq_len, w_max+1]
            opts: [batch_size, seq_len, 1]
            ys: [batch_size, seq_len, n_max]
        """
        batch_size, seq_len = values.shape
        device = values.device
        
        # Encode items
        items = torch.stack([values, weights], dim=-1)
        encoded = self.item_encoder(items)
        
        # Add positional encoding
        if step_indices is None:
            step_indices = torch.arange(1, seq_len + 1, device=device)
            step_indices = step_indices.unsqueeze(0).expand(batch_size, -1)
        
        pos_enc = self.pos_embedding(step_indices)
        encoded = encoded + pos_enc
        
        # GRU processing
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        gru_out, _ = self.gru(encoded, h0)
        
        # Decode outputs
        dp_rows = self.dp_decoder(gru_out)
        opts = self.opt_decoder(gru_out)
        ys = self.y_decoder(gru_out)
        
        # CRITICAL: Apply causal mask to Y
        # At step k, only items 0..k-1 can be predicted
        # Items k..n_max-1 must be masked to 0
        ys = self._apply_causal_mask(ys, step_indices)
        
        return dp_rows, opts, ys
    
    def _apply_causal_mask(self, ys, step_indices):
        """
        Apply causal mask to Y predictions.
        
        At step k, we've only seen items 0 to k-1, so:
        - y[0:k] can be 0 or 1 (predicted)
        - y[k:n_max] must be 0 (masked)
        
        Args:
            ys: [batch_size, seq_len, n_max] - raw Y predictions
            step_indices: [batch_size, seq_len] - step numbers (1-indexed)
        
        Returns:
            ys_masked: [batch_size, seq_len, n_max] - masked Y predictions
        """
        batch_size, seq_len, n_max = ys.shape
        device = ys.device
        
        # Create causal mask
        # For step k (1-indexed), items 0 to k-1 are valid
        # So we mask positions >= k
        
        # Create position indices [0, 1, 2, ..., n_max-1]
        positions = torch.arange(n_max, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, n_max]
        
        # Expand step_indices to match: [batch_size, seq_len, 1]
        step_k = step_indices.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Create mask: position < k (since k is 1-indexed, position k-1 is the last valid)
        # At step k=1: positions 0 is valid (position < 1 = False, but we've seen item 0)
        # Actually: at step k, we've seen items 0 to k-1, so valid positions are < k
        mask = positions < step_k  # [batch_size, seq_len, n_max]
        
        # Apply mask (set invalid positions to 0)
        ys_masked = ys * mask.float()
        
        return ys_masked
    
    def predict_step(self, values, weights, step_indices=None, apply_mask=True):
        """
        Prediction step for inference.
        
        Args:
            values: [batch_size, seq_len] - item values
            weights: [batch_size, seq_len] - item weights
            step_indices: [batch_size, seq_len] - step indices (optional)
            apply_mask: bool - whether to apply masking (always True now)
        
        Returns:
            dp_rows: [batch_size, seq_len, w_max+1]
            opts: [batch_size, seq_len] - squeezed from [batch_size, seq_len, 1]
            ys: [batch_size, seq_len, n_max]
        """
        # Call forward pass (masking is now applied inside forward)
        dp_rows, opts, ys = self.forward(values, weights, step_indices)
        
        # Squeeze opts dimension to match expected output
        opts = opts.squeeze(-1)  # [batch_size, seq_len, 1] -> [batch_size, seq_len]
        
        return dp_rows, opts, ys