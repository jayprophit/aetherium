"""
Training Loop

This module contains the main training loop and related utilities.
"""
from typing import Dict, List, Tuple, Optional, Any, Callable
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import os

from .trainer import Trainer


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'runs',
    save_every: int = 1000,
    log_every: int = 100,
    grad_clip: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, Callable]] = None,
) -> Dict[str, List[float]]:
    """Train a model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        save_every: Save checkpoint every n steps
        log_every: Log metrics every n steps
        grad_clip: Gradient clipping value
        early_stopping_patience: Number of epochs to wait before early stopping
        scheduler: Learning rate scheduler
        metrics: Dictionary of metric functions to compute during evaluation
        
    Returns:
        Dictionary of training and validation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Set up loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        save_every=save_every,
        log_every=log_every,
        grad_clip=grad_clip,
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': {},
        'val_metrics': {},
    }
    
    if metrics:
        for metric_name in metrics:
            history['train_metrics'][metric_name] = []
            history['val_metrics'][metric_name] = []
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {name: 0.0 for name in metrics or {}}
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            if metrics:
                with torch.no_grad():
                    for metric_name, metric_fn in metrics.items():
                        train_metrics[metric_name] += metric_fn(
                            outputs.detach(),
                            targets.detach()
                        ).item()
        
        # Compute average metrics for the epoch
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        if metrics:
            for metric_name in metrics:
                train_metrics[metric_name] /= len(train_loader)
                history['train_metrics'][metric_name].append(train_metrics[metric_name])
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_metrics = {name: 0.0 for name in metrics or {}}
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    
                    if metrics:
                        for metric_name, metric_fn in metrics.items():
                            val_metrics[metric_name] += metric_fn(
                                outputs,
                                targets
                            ).item()
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            if metrics:
                for metric_name in metrics:
                    val_metrics[metric_name] /= len(val_loader)
                    history['val_metrics'][metric_name].append(val_metrics[metric_name])
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save best model
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
            else:
                epochs_without_improvement += 1
                
                # Early stopping
                if (early_stopping_patience is not None and 
                    epochs_without_improvement >= early_stopping_patience):
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}", end="")
        if val_loader is not None:
            print(f", Val Loss: {val_loss:.4f}")
        else:
            print()
            
        # Step the scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and val_loader is not None:
                scheduler.step(val_loss)
            else:
                scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_model.pt'))
    
    return history
