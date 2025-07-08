"""
Visualization Utilities

This module contains functions for visualizing model evaluation results.
"""
from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    PrecisionRecallDisplay,
    RocCurveDisplay
)
import seaborn as sns
import os

# Set the style for the plots
plt.style.use('seaborn')
sns.set_style("whitegrid")

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: str = 'true',
    cmap: str = 'Blues',
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Plot a confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        cmap: Colormap for the plot
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        **kwargs: Additional arguments to pass to seaborn.heatmap
        
    Returns:
        Matplotlib figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
        fmt = '.2f'
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar=True,
        square=True,
        ax=ax,
        **kwargs
    )
    
    # Set labels
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    
    # Set class names if provided
    if class_names is not None:
        tick_marks = np.arange(len(class_names)) + 0.5
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show plot if requested
    if show:
        plt.show()
    
    return fig

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
    n_classes: Optional[int] = None,
    title: str = 'ROC Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Plot ROC curve(s).
    
    Args:
        y_true: Ground truth labels (binary or multiclass)
        y_score: Target scores (probability estimates)
        class_names: List of class names (for multiclass)
        n_classes: Number of classes (if y_true is not provided as one-hot)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        **kwargs: Additional arguments to pass to RocCurveDisplay
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle binary case
    if len(y_score.shape) == 1 or y_score.shape[1] == 1:
        if n_classes is not None and n_classes > 2:
            raise ValueError("Binary format is not supported for multiclass ROC.")
        
        # Binary ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, **kwargs)
        display.plot(ax=ax, name='ROC curve (AUC = %0.2f)' % roc_auc)
    else:
        # Multiclass ROC curve (one-vs-rest)
        if n_classes is None:
            n_classes = y_score.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        for i in range(n_classes):
            name = f'Class {i} (AUC = {roc_auc[i]:.2f})'
            if class_names is not None and i < len(class_names):
                name = f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
            
            plt.plot(
                fpr[i],
                tpr[i],
                lw=2,
                label=name,
                **kwargs
            )
        
        # Plot random guess line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set labels and title
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show plot if requested
    if show:
        plt.show()
    
    return fig

def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
    n_classes: Optional[int] = None,
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Plot precision-recall curve(s).
    
    Args:
        y_true: Ground truth labels (binary or multiclass)
        y_score: Target scores (probability estimates)
        class_names: List of class names (for multiclass)
        n_classes: Number of classes (if y_true is not provided as one-hot)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        **kwargs: Additional arguments to pass to PrecisionRecallDisplay
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle binary case
    if len(y_score.shape) == 1 or y_score.shape[1] == 1:
        if n_classes is not None and n_classes > 2:
            raise ValueError("Binary format is not supported for multiclass PR curve.")
        
        # Binary PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        
        display = PrecisionRecallDisplay(
            precision=precision,
            recall=recall,
            average_precision=avg_precision,
            **kwargs
        )
        display.plot(ax=ax, name=f'Precision-Recall (AP = {avg_precision:.2f})')
    else:
        # Multiclass PR curve (one-vs-rest)
        if n_classes is None:
            n_classes = y_score.shape[1]
        
        # Compute PR curve and average precision for each class
        precision = dict()
        recall = dict()
        avg_precision = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true == i, y_score[:, i]
            )
            avg_precision[i] = average_precision_score(
                y_true == i, y_score[:, i]
            )
        
        # Plot all PR curves
        for i in range(n_classes):
            name = f'Class {i} (AP = {avg_precision[i]:.2f})'
            if class_names is not None and i < len(class_names):
                name = f'{class_names[i]} (AP = {avg_precision[i]:.2f})'
            
            plt.plot(
                recall[i],
                precision[i],
                lw=2,
                label=name,
                **kwargs
            )
        
        # Set labels and title
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="upper right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show plot if requested
    if show:
        plt.show()
    
    return fig

def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Plot training history (loss and metrics over epochs).
    
    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot (default: all metrics in history)
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        **kwargs: Additional arguments to pass to plt.plot
        
    Returns:
        Matplotlib figure object
    """
    if metrics is None:
        metrics = [k for k in history.keys() if not k.startswith('val_') and k != 'epoch']
    
    num_plots = len(metrics)
    if num_plots == 0:
        raise ValueError("No metrics found in history")
    
    # Create subplots
    fig, axes = plt.subplots(1, num_plots, figsize=(figsize[0] * num_plots, figsize[1]))
    
    if num_plots == 1:
        axes = [axes]
    
    # Get number of epochs
    epochs = range(1, len(history[metrics[0]]) + 1)
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot training metric
        ax.plot(epochs, history[metric], 'b-', label=f'Training {metric}', **kwargs)
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(epochs, history[val_metric], 'r-', label=f'Validation {metric}', **kwargs)
        
        ax.set_title(f'Training and Validation {metric.upper()}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show plot if requested
    if show:
        plt.show()
    
    return fig
