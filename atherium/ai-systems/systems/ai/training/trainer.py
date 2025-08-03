"""
Training Utilities

This module contains the main training loop and related utilities.
"""
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm

class Trainer:
    """Handles the training and evaluation of models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: callable,
        device: torch.device,
        log_dir: str = 'runs',
        checkpoint_dir: str = 'checkpoints',
        save_every: int = 1000,
        log_every: int = 100,
        grad_clip: Optional[float] = None
    ):
        """Initialize the trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer to use
            loss_fn: Loss function
            device: Device to train on
            log_dir: Directory to save logs
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every n steps
            log_every: Log metrics every n steps
            grad_clip: Gradient clipping value
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.step = 0
        self.grad_clip = grad_clip
        self.save_every = save_every
        self.log_every = log_every
        
        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up logging
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir
        
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = tuple(t.to(self.device) for t in batch)
        inputs, targets = batch[:-1], batch[-1]
        
        # Forward pass
        outputs = self.model(*inputs)
        loss = self.loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        # Log metrics
        metrics = {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        self.step += 1
        return metrics
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        """Evaluate the model on the given data loader.
        
        Args:
            data_loader: DataLoader to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = tuple(t.to(self.device) for t in batch)
                inputs, targets = batch[:-1], batch[-1]
                
                outputs = self.model(*inputs)
                loss = self.loss_fn(outputs, targets)
                
                batch_size = inputs[0].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        return {'loss': total_loss / total_samples}
    
    def save_checkpoint(self, name: str = 'checkpoint.pt') -> None:
        """Save a checkpoint of the model and optimizer.
        
        Args:
            name: Name of the checkpoint file
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint.
        
        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
    
    def close(self) -> None:
        """Close the writer and clean up."""
        self.writer.close()
