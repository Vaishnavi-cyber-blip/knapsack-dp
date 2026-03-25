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
        
        return dp_rows, opts, ys
    
    def predict_step(self, values, weights, step_indices=None, apply_mask=False):
        """
        Prediction step for inference.
        
        Args:
            values: [batch_size, seq_len] - item values
            weights: [batch_size, seq_len] - item weights
            step_indices: [batch_size, seq_len] - step indices (optional)
            apply_mask: bool - whether to apply masking (optional)
        
        Returns:
            dp_rows: [batch_size, seq_len, w_max+1]
            opts: [batch_size, seq_len] - squeezed from [batch_size, seq_len, 1]
            ys: [batch_size, seq_len, n_max]
        """
        # Call forward pass
        dp_rows, opts, ys = self.forward(values, weights, step_indices)
        
        # Squeeze opts dimension to match expected output
        opts = opts.squeeze(-1)  # [batch_size, seq_len, 1] -> [batch_size, seq_len]
        
        # Apply masking if requested (currently not implemented)
        if apply_mask:
            # Optional: Add masking logic here if needed
            pass
        
        return dp_rows, opts, ys