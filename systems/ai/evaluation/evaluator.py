"""
Model Evaluator

This module contains the ModelEvaluator class for evaluating trained models.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import json

from .metrics import get_metrics_dict

class ModelEvaluator:
    """A class for evaluating trained models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        metrics: Optional[List[str]] = None,
        average: str = 'weighted',
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize the model evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on (default: GPU if available, else CPU)
            metrics: List of metrics to compute (default: ['accuracy', 'precision', 'recall', 'f1'])
            average: Averaging strategy for multiclass metrics
            output_dir: Directory to save evaluation results (optional)
            **kwargs: Additional arguments
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.metrics = metrics or ['accuracy', 'precision', 'recall', 'f1']
        self.average = average
        self.output_dir = output_dir
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(
        self,
        data_loader: DataLoader,
        return_predictions: bool = False,
        return_metrics: bool = True,
        progress_bar: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate the model on the given data loader.
        
        Args:
            data_loader: DataLoader for evaluation data
            return_predictions: Whether to return model predictions
            return_metrics: Whether to compute and return metrics
            progress_bar: Whether to show a progress bar
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing evaluation results
        """
        self.model.eval()
        
        all_outputs = []
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", disable=not progress_bar):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                elif isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    targets = batch.get('labels', None)
                    if targets is not None:
                        targets = targets.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                # Forward pass
                outputs = self.model(inputs) if not isinstance(inputs, dict) else self.model(**inputs)
                
                # Get predictions
                if isinstance(outputs, (list, tuple)):
                    logits = outputs[0]
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                if logits is not None:
                    if len(logits.shape) > 1 and logits.shape[1] > 1:
                        # Multi-class classification
                        probs = torch.softmax(logits, dim=-1)
                        preds = torch.argmax(logits, dim=-1)
                    else:
                        # Binary classification or regression
                        probs = torch.sigmoid(logits) if logits.shape[-1] == 1 else logits
                        preds = (probs > 0.5).long().squeeze()
                    
                    all_outputs.append(probs.cpu().numpy())
                    all_predictions.append(preds.cpu().numpy())
                
                if targets is not None:
                    all_targets.append(targets.cpu().numpy())
        
        # Concatenate all batches
        results = {}
        
        if len(all_outputs) > 0:
            all_outputs = np.concatenate(all_outputs, axis=0)
            all_predictions = np.concatenate(all_predictions, axis=0)
            
            if return_predictions:
                results['outputs'] = all_outputs
                results['predictions'] = all_predictions
        
        if len(all_targets) > 0:
            all_targets = np.concatenate(all_targets, axis=0)
            
            if return_metrics:
                # Compute metrics
                metrics = get_metrics_dict(
                    all_targets,
                    all_predictions,
                    all_outputs if all_outputs is not None else None,
                    metrics=self.metrics,
                    average=self.average,
                    **kwargs
                )
                results['metrics'] = metrics
                
                # Print metrics
                print("\nEvaluation Metrics:")
                for name, value in metrics.items():
                    print(f"{name}: {value:.4f}")
                
                # Save metrics to file
                if self.output_dir is not None:
                    metrics_file = os.path.join(self.output_dir, 'metrics.json')
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    print(f"\nMetrics saved to {metrics_file}")
        
        return results
    
    def predict(
        self,
        data_loader: DataLoader,
        return_probabilities: bool = False,
        progress_bar: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Generate predictions for the given data loader.
        
        Args:
            data_loader: DataLoader for prediction data
            return_probabilities: Whether to return probabilities or class predictions
            progress_bar: Whether to show a progress bar
            **kwargs: Additional arguments
            
        Returns:
            Array of predictions or probabilities
        """
        results = self.evaluate(
            data_loader,
            return_predictions=True,
            return_metrics=False,
            progress_bar=progress_bar,
            **kwargs
        )
        
        if return_probabilities and 'outputs' in results:
            return results['outputs']
        elif 'predictions' in results:
            return results['predictions']
        else:
            raise ValueError("No predictions found in results")
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str = 'evaluation_results.json',
        **kwargs
    ) -> str:
        """Save evaluation results to a file.
        
        Args:
            results: Evaluation results dictionary
            filename: Name of the output file
            **kwargs: Additional arguments
            
        Returns:
            Path to the saved file
        """
        if self.output_dir is None:
            raise ValueError("output_dir must be specified to save results")
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                serializable_results[k] = v.tolist()
            elif isinstance(v, dict):
                serializable_results[k] = {
                    k2: v2.tolist() if isinstance(v2, np.ndarray) else v2
                    for k2, v2 in v.items()
                }
            else:
                serializable_results[k] = v
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, **kwargs)
        
        return filepath
