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
from utils import *

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=0.001,
        save_path='weights.pt',
        time_limit=14400  # 4 hours in seconds
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path
        self.time_limit = time_limit
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # Best validation loss for checkpointing
        self.best_val_loss = float('inf')
    
    def compute_loss(self, dp_pred, opt_pred, y_pred, dp_true, opt_true, y_true, lengths):

        # Mask
        max_len = dp_pred.shape[1]
        mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.float()
        
        # DP Loss - MSE
        dp_loss = ((dp_pred - dp_true) ** 2).mean(dim=-1)
        dp_loss = (dp_loss * mask).sum() / mask.sum()
        
        # OPT Loss - MAE
        opt_loss = torch.abs(opt_pred - opt_true)
        opt_loss = (opt_loss * mask).sum() / mask.sum()
        
        # Y Loss - BCE
        y_loss = self.bce_loss(y_pred * mask.unsqueeze(-1), y_true * mask.unsqueeze(-1))
        
        w1 = 0.000002
        w2 = 0.0022
        w3 = 9.8
        # Total weighted loss
        total_loss = w1 * dp_loss + w2 * opt_loss + w3 * y_loss
        
        return total_loss, dp_loss*w1, opt_loss*w2, y_loss*w3
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_dp_loss = 0
        total_opt_loss = 0
        total_y_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            values = batch['values'].to(self.device)
            weights = batch['weights'].to(self.device)
            step_indices = batch['step_indices'].to(self.device)
            dp_true = batch['dp_rows'].to(self.device)
            opt_true = batch['opts'].to(self.device)
            y_true = batch['ys'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # Forward pass
            dp_pred, opt_pred, y_pred = self.model(values, weights, step_indices)
            opt_pred = opt_pred.squeeze(-1)
            
            # Compute loss
            loss, dp_loss, opt_loss, y_loss = self.compute_loss(
                dp_pred, opt_pred, y_pred, dp_true, opt_true, y_true, lengths
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_dp_loss += dp_loss.item()
            total_opt_loss += opt_loss.item()
            total_y_loss += y_loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'dp_loss': total_dp_loss / n_batches,
            'opt_loss': total_opt_loss / n_batches,
            'y_loss': total_y_loss / n_batches
        }
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_dp_loss = 0
        total_opt_loss = 0
        total_y_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                values = batch['values'].to(self.device)
                weights = batch['weights'].to(self.device)
                step_indices = batch['step_indices'].to(self.device)
                dp_true = batch['dp_rows'].to(self.device)
                opt_true = batch['opts'].to(self.device)
                y_true = batch['ys'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                
                dp_pred, opt_pred, y_pred = self.model(values, weights, step_indices)
                opt_pred = opt_pred.squeeze(-1)
                
                loss, dp_loss, opt_loss, y_loss = self.compute_loss(
                    dp_pred, opt_pred, y_pred, dp_true, opt_true, y_true, lengths
                )
                
                total_loss += loss.item()
                total_dp_loss += dp_loss.item()
                total_opt_loss += opt_loss.item()
                total_y_loss += y_loss.item()
        
        n_batches = len(self.val_loader)
        return {
            'loss': total_loss / n_batches,
            'dp_loss': total_dp_loss / n_batches,
            'opt_loss': total_opt_loss / n_batches,
            'y_loss': total_y_loss / n_batches
        }
    
    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, self.save_path)
        print(f"Checkpoint saved to {self.save_path}")
    
    def train(self, num_epochs):
        print("="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        start_time = time.time()
        
        # Save initial checkpoint
        self.save_checkpoint(0, float('inf'))
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Check time limit (stop 10 minutes before limit)
            elapsed = time.time() - start_time
            if elapsed > self.time_limit - 600:
                print(f"\nApproaching time limit ({elapsed/3600:.2f}h). Stopping training.")
                break
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Elapsed time: {elapsed/3600:.2f}h / {self.time_limit/3600:.1f}h")
            print("-" * 80)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            epoch_time = time.time() - epoch_start
            
            # Print metrics
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} (DP: {train_metrics['dp_loss']:.4f}, "
                  f"OPT: {train_metrics['opt_loss']:.4f}, Y: {train_metrics['y_loss']:.4f})")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} (DP: {val_metrics['dp_loss']:.4f}, "
                  f"OPT: {val_metrics['opt_loss']:.4f}, Y: {val_metrics['y_loss']:.4f})")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics['loss'])
                print(f"  ✓ New best validation loss: {self.best_val_loss:.4f}")
                print("Saving best checkpoint")
            
            # Periodic save (every 5 epochs)
            # if epoch % 5 == 0:
            #     self.save_checkpoint(epoch, val_metrics['loss'])
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total time: {(time.time() - start_time)/3600:.2f}h")
        print("="*80)


def set_seed(seed, deterministic=False):
    """Set random seeds for reproducibility."""
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


def main():
    """Main training script with argparse."""
    parser = argparse.ArgumentParser(description='Train Knapsack GRU Solver')
    
    # Required arguments
    parser.add_argument('--train', type=str, required=True,
                        help='Path to training JSON file (Schema A)')
    parser.add_argument('--save', type=str, required=True,
                        help='Path to save model weights')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for reproducibility')
    parser.add_argument('--deterministic', type=int, default=0, choices=[0, 1],
                        help='Enable deterministic mode (0 or 1)')
    
    # Optional hyperparameters (can be hardcoded or in config)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size for GRU')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Set seed and determinism
    set_seed(args.seed, args.deterministic == 1)
    print(f"Random seed set to: {args.seed}")
    print(f"Deterministic mode: {'ON' if args.deterministic == 1 else 'OFF'}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading training data from: {args.train}")
    full_dataset = KnapsackDataset(args.train)
    
    # Create train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print(f"\nCreating model...")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    
    model = KnapsackGRUSolver(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        save_path=args.save
    )
    
    # Train
    trainer.train(args.num_epochs)
    
    print(f"\n✓ Training complete. Weights saved to: {args.save}")


if __name__ == "__main__":
    main()