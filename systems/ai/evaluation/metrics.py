"""
Evaluation Metrics

This module contains functions for computing various evaluation metrics.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sk_confusion_matrix,
    roc_auc_score as sk_roc_auc_score,
    average_precision_score as sk_average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc
)

def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert input to numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def accuracy(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    **kwargs
) -> float:
    """Compute accuracy score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        **kwargs: Additional arguments to pass to sklearn's accuracy_score
        
    Returns:
        Accuracy score
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    return float(accuracy_score(y_true_np, y_pred_np, **kwargs))

def precision(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    average: str = 'weighted',
    **kwargs
) -> float:
    """Compute precision score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('micro', 'macro', 'weighted', 'samples')
        **kwargs: Additional arguments to pass to sklearn's precision_score
        
    Returns:
        Precision score
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    return float(precision_score(y_true_np, y_pred_np, average=average, **kwargs, zero_division=0))

def recall(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    average: str = 'weighted',
    **kwargs
) -> float:
    """Compute recall score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('micro', 'macro', 'weighted', 'samples')
        **kwargs: Additional arguments to pass to sklearn's recall_score
        
    Returns:
        Recall score
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    return float(recall_score(y_true_np, y_pred_np, average=average, **kwargs, zero_division=0))

def f1_score(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    average: str = 'weighted',
    **kwargs
) -> float:
    """Compute F1 score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('micro', 'macro', 'weighted', 'samples')
        **kwargs: Additional arguments to pass to sklearn's f1_score
        
    Returns:
        F1 score
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    return float(f1_score(y_true_np, y_pred_np, average=average, **kwargs, zero_division=0))

def confusion_matrix(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    normalize: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        **kwargs: Additional arguments to pass to sklearn's confusion_matrix
        
    Returns:
        Confusion matrix
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    
    cm = sk_confusion_matrix(y_true_np, y_pred_np, **kwargs)
    
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
    
    return cm

def roc_auc_score(
    y_true: Union[torch.Tensor, np.ndarray],
    y_score: Union[torch.Tensor, np.ndarray],
    average: str = 'weighted',
    multi_class: str = 'ovr',
    **kwargs
) -> float:
    """Compute ROC AUC score.
    
    Args:
        y_true: Ground truth labels
        y_score: Target scores (probability estimates)
        average: Averaging strategy ('macro', 'weighted', 'samples')
        multi_class: Multiclass mode ('ovr' or 'ovo')
        **kwargs: Additional arguments to pass to sklearn's roc_auc_score
        
    Returns:
        ROC AUC score
    """
    y_true_np = _to_numpy(y_true)
    y_score_np = _to_numpy(y_score)
    
    # Handle binary case
    if len(y_score_np.shape) == 1 or y_score_np.shape[1] == 1:
        if len(np.unique(y_true_np)) > 2:
            raise ValueError("Binary format is not supported for multiclass ROC AUC.")
        return float(sk_roc_auc_score(y_true_np, y_score_np, **kwargs))
    
    return float(sk_roc_auc_score(
        y_true_np, y_score_np, average=average, multi_class=multi_class, **kwargs
    ))

def average_precision_score(
    y_true: Union[torch.Tensor, np.ndarray],
    y_score: Union[torch.Tensor, np.ndarray],
    average: str = 'weighted',
    **kwargs
) -> float:
    """Compute average precision score.
    
    Args:
        y_true: Ground truth labels
        y_score: Target scores (probability estimates)
        average: Averaging strategy ('micro', 'macro', 'weighted', 'samples')
        **kwargs: Additional arguments to pass to sklearn's average_precision_score
        
    Returns:
        Average precision score
    """
    y_true_np = _to_numpy(y_true)
    y_score_np = _to_numpy(y_score)
    
    # Handle binary case
    if len(y_score_np.shape) == 1 or y_score_np.shape[1] == 1:
        if len(np.unique(y_true_np)) > 2:
            raise ValueError("Binary format is not supported for multiclass average precision.")
        return float(sk_average_precision_score(y_true_np, y_score_np, **kwargs))
    
    return float(sk_average_precision_score(
        y_true_np, y_score_np, average=average, **kwargs
    ))

def get_metrics_dict(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    y_score: Optional[Union[torch.Tensor, np.ndarray]] = None,
    metrics: Optional[List[str]] = None,
    average: str = 'weighted',
    **kwargs
) -> Dict[str, float]:
    """Compute multiple metrics at once.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Target scores (probability estimates, required for some metrics)
        metrics: List of metrics to compute (default: ['accuracy', 'precision', 'recall', 'f1'])
        average: Averaging strategy for multiclass metrics
        **kwargs: Additional arguments to pass to metric functions
        
    Returns:
        Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    results = {}
    
    for metric in metrics:
        if metric == 'accuracy':
            results['accuracy'] = accuracy(y_true, y_pred, **kwargs)
        elif metric == 'precision':
            results['precision'] = precision(y_true, y_pred, average=average, **kwargs)
        elif metric == 'recall':
            results['recall'] = recall(y_true, y_pred, average=average, **kwargs)
        elif metric == 'f1':
            results['f1'] = f1_score(y_true, y_pred, average=average, **kwargs)
        elif metric in ['roc_auc', 'auroc']:
            if y_score is None:
                raise ValueError("y_score is required for ROC AUC calculation")
            results['roc_auc'] = roc_auc_score(
                y_true, y_score, average=average, **kwargs
            )
        elif metric in ['average_precision', 'ap']:
            if y_score is None:
                raise ValueError("y_score is required for average precision calculation")
            results['average_precision'] = average_precision_score(
                y_true, y_score, average=average, **kwargs
            )
    
    return results
