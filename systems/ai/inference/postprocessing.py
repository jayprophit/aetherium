"""
Postprocessing Utilities

This module contains utilities for postprocessing model outputs.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np

class PostProcessor:
    """Base class for postprocessing model outputs."""
    
    def __init__(self, **kwargs):
        """Initialize the postprocessor.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        self.config = kwargs
    
    def __call__(self, outputs: Any, **kwargs) -> Any:
        """Process the model outputs.
        
        Args:
            outputs: Raw model outputs
            **kwargs: Additional arguments for processing
            
        Returns:
            Processed outputs
        """
        return self.process(outputs, **kwargs)
    
    def process(self, outputs: Any, **kwargs) -> Any:
        """Process the model outputs (to be implemented by subclasses).
        
        Args:
            outputs: Raw model outputs
            **kwargs: Additional arguments for processing
            
        Returns:
            Processed outputs
        """
        raise NotImplementedError


class ClassificationPostProcessor(PostProcessor):
    """Postprocessor for classification model outputs."""
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        **kwargs
    ):
        """Initialize the classification postprocessor.
        
        Args:
            class_names: List of class names (optional)
            threshold: Probability threshold for binary classification
            top_k: Number of top predictions to return (None for all above threshold)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.class_names = class_names or []
        self.threshold = threshold
        self.top_k = top_k
    
    def process(
        self,
        outputs: Union[torch.Tensor, np.ndarray],
        return_probs: bool = False,
        **kwargs
    ) -> Union[List[Dict[str, Any]], np.ndarray]:
        """Process classification model outputs.
        
        Args:
            outputs: Model outputs (logits or probabilities)
            return_probs: Whether to return probabilities with predictions
            **kwargs: Additional arguments for processing
            
        Returns:
            If return_probs is True, returns a list of dicts with 'class', 'score', and 'label'.
            Otherwise, returns the predicted class indices.
        """
        # Convert to numpy if needed
        if torch.is_tensor(outputs):
            outputs = outputs.detach().cpu().numpy()
        
        # Handle single sample
        if len(outputs.shape) == 1:
            outputs = outputs[np.newaxis, :]
        
        # Get probabilities (apply softmax if needed)
        if outputs.shape[-1] > 1 and not (outputs.min() >= 0 and outputs.max() <= 1):
            # Apply softmax to get probabilities
            probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=-1, keepdims=True)
        else:
            probs = outputs
        
        # Get predictions
        if self.top_k is not None and self.top_k > 0:
            # Get top-k predictions
            top_k = min(self.top_k, probs.shape[-1])
            top_indices = np.argpartition(probs, -top_k, axis=-1)[:, -top_k:]
            top_probs = np.take_along_axis(probs, top_indices, axis=-1)
            
            # Sort by probability
            sort_indices = np.argsort(-top_probs, axis=-1)
            top_indices = np.take_along_axis(top_indices, sort_indices, axis=-1)
            top_probs = np.take_along_axis(top_probs, sort_indices, axis=-1)
            
            if return_probs:
                # Return list of dicts with class info
                results = []
                for i in range(len(top_indices)):
                    result = []
                    for j in range(len(top_indices[i])):
                        class_idx = top_indices[i, j]
                        prob = top_probs[i, j]
                        result.append({
                            'class': int(class_idx),
                            'score': float(prob),
                            'label': self.class_names[class_idx] if class_idx < len(self.class_names) else str(class_idx)
                        })
                    results.append(result)
                return results if len(results) > 1 else results[0]
            else:
                # Return just the top class indices
                return top_indices[:, 0] if top_indices.shape[0] > 1 else top_indices[0, 0]
        else:
            # Get all predictions above threshold
            if return_probs:
                results = []
                for i in range(len(probs)):
                    result = []
                    for j in range(len(probs[i])):
                        if probs[i, j] >= self.threshold:
                            result.append({
                                'class': j,
                                'score': float(probs[i, j]),
                                'label': self.class_names[j] if j < len(self.class_names) else str(j)
                            })
                    # Sort by score in descending order
                    result.sort(key=lambda x: x['score'], reverse=True)
                    results.append(result)
                return results if len(results) > 1 else results[0]
            else:
                # Return class with highest probability
                return np.argmax(probs, axis=-1)


class DetectionPostProcessor(PostProcessor):
    """Postprocessor for object detection model outputs."""
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        **kwargs
    ):
        """Initialize the detection postprocessor.
        
        Args:
            class_names: List of class names (optional)
            confidence_threshold: Minimum confidence score to keep a detection
            nms_threshold: Non-maximum suppression threshold
            max_detections: Maximum number of detections to return per image
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.class_names = class_names or []
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
    
    def process(
        self,
        outputs: Union[torch.Tensor, np.ndarray, Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process detection model outputs.
        
        Args:
            outputs: Model outputs (format depends on the model)
            **kwargs: Additional arguments for processing
            
        Returns:
            List of detections, where each detection is a dict with:
            - 'bbox': [x1, y1, x2, y2] coordinates
            - 'score': Confidence score
            - 'class': Class index
            - 'label': Class name (if class_names provided)
        """
        raise NotImplementedError("Detection postprocessing not implemented yet")
