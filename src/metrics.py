# SPDX-License-Identifier: MIT
"""
Evaluation metrics for classification.

Implements accuracy, macro-F1, per-class F1, and confusion matrix.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 score."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6) -> dict:
    """
    Compute per-class F1 scores.

    Returns:
        Dictionary mapping class index to F1 score
    """
    f1_scores = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    return {i: float(f1_scores[i]) for i in range(num_classes)}


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=range(num_classes))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6) -> dict:
    """
    Compute all metrics.

    Returns:
        Dictionary with accuracy, macro_f1, per_class_f1, confusion_matrix
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "per_class_f1": compute_per_class_f1(y_true, y_pred, num_classes),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred, num_classes).tolist(),
    }
