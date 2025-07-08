"""
Model Evaluation Module

This module contains utilities for evaluating trained models.
"""

from .metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    get_metrics_dict
)

from .evaluator import ModelEvaluator
from .visualization import plot_confusion_matrix, plot_roc_curve, plot_pr_curve

__all__ = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    'roc_auc_score',
    'average_precision_score',
    'get_metrics_dict',
    'ModelEvaluator',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_pr_curve'
]
