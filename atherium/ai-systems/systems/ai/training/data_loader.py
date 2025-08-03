"""
Data Loading Utilities

This module contains utilities for loading and preprocessing data for training.
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from pathlib import Path

class BaseDataset(Dataset):
    """Base dataset class for all datasets."""
    
    def __init__(self, data: List[Any], transform=None):
        """Initialize the dataset.
        
        Args:
            data: List of data samples
            transform: Optional transform to be applied to samples
        """
        self.data = data
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input, target)
        """
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        drop_last: bool = False,
        **kwargs
    ) -> DataLoader:
        """Create a data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            drop_last: Whether to drop the last incomplete batch
            **kwargs: Additional arguments to pass to DataLoader
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            **kwargs
        )
    
    @classmethod
    def train_val_split(
        cls,
        dataset: Dataset,
        val_split: float = 0.1,
        batch_size: int = 32,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """Split a dataset into training and validation sets.
        
        Args:
            dataset: Dataset to split
            val_split: Fraction of data to use for validation
            batch_size: Batch size for both loaders
            **kwargs: Additional arguments to pass to create_dataloader
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = cls.create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        
        val_loader = cls.create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )
        
        return train_loader, val_loader
