"""
Model Predictor

This module contains the base predictor class for making predictions with trained models.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from pathlib import Path
import os

class Predictor:
    """Base class for making predictions with trained models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        preprocess_fn: Optional[callable] = None,
        postprocess_fn: Optional[callable] = None,
    ):
        """Initialize the predictor.
        
        Args:
            model: Trained PyTorch model
            device: Device to run inference on (default: GPU if available, else CPU)
            preprocess_fn: Function to preprocess input data
            postprocess_fn: Function to postprocess model outputs
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.postprocess_fn = postprocess_fn or (lambda x: x)
    
    @classmethod
    def from_checkpoint(
        cls,
        model_class: type,
        checkpoint_path: Union[str, Path],
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'Predictor':
        """Create a predictor from a saved checkpoint.
        
        Args:
            model_class: Model class to instantiate
            checkpoint_path: Path to the saved model checkpoint
            model_config: Model configuration dictionary
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Predictor instance
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Initialize model
        if model_config is None:
            model_config = {}
        model = model_class(**model_config, **kwargs)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return cls(model=model, **kwargs)
    
    def preprocess(self, inputs: Any) -> Any:
        """Preprocess input data.
        
        Args:
            inputs: Raw input data
            
        Returns:
            Preprocessed input data
        """
        return self.preprocess_fn(inputs)
    
    def postprocess(self, outputs: Any) -> Any:
        """Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Postprocessed outputs
        """
        return self.postprocess_fn(outputs)
    
    def predict(self, inputs: Any, batch_size: int = 32, **kwargs) -> Any:
        """Make predictions on input data.
        
        Args:
            inputs: Input data to make predictions on
            batch_size: Batch size for inference
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Model predictions
        """
        # Preprocess inputs
        processed_inputs = self.preprocess(inputs)
        
        # Convert to tensor if not already
        if not isinstance(processed_inputs, torch.Tensor):
            processed_inputs = torch.tensor(processed_inputs, device=self.device)
        
        # Move to device
        processed_inputs = processed_inputs.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            if batch_size is not None and len(processed_inputs) > batch_size:
                # Process in batches
                outputs = []
                for i in range(0, len(processed_inputs), batch_size):
                    batch = processed_inputs[i:i + batch_size]
                    batch_outputs = self.model(batch, **kwargs)
                    outputs.append(batch_outputs.cpu())
                outputs = torch.cat(outputs, dim=0)
            else:
                # Process all at once
                outputs = self.model(processed_inputs, **kwargs)
        
        # Postprocess outputs
        return self.postprocess(outputs)
    
    def __call__(self, inputs: Any, **kwargs) -> Any:
        """Make predictions (same as predict method)."""
        return self.predict(inputs, **kwargs)
    
    def to(self, device: Union[str, torch.device]) -> 'Predictor':
        """Move the predictor to a different device.
        
        Args:
            device: Device to move to ('cpu' or 'cuda' or torch.device)
            
        Returns:
            self
        """
        self.device = torch.device(device)
        self.model.to(self.device)
        return self
