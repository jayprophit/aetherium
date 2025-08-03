"""
Base Model Architecture

Defines the base class for all AI models in the system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Base class for all AI models in the system.
    
    This class provides common functionality and interface for all models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the base model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.get("force_cpu", False) 
            else "cpu"
        )
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass of the model.
        
        Must be implemented by all subclasses.
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model to
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'BaseModel':
        """Load a model from disk.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(checkpoint['config'], **kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        """Get an optimizer for this model.
        
        Args:
            **kwargs: Additional arguments to pass to the optimizer
            
        Returns:
            Optimizer instance
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            **kwargs
        )
