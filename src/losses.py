# SPDX-License-Identifier: MIT
"""
Loss functions for training.

Supports cross-entropy with label smoothing and optional focal loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Args:
        smoothing: Label smoothing factor (default: 0.05)
        weight: Class weights (optional)
    """

    def __init__(self, smoothing: float = 0.05, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [batch_size, num_classes]
            target: Target labels [batch_size]

        Returns:
            Scalar loss
        """
        num_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)

        # One-hot encode targets with label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Apply class weights if provided
        if self.weight is not None:
            true_dist = true_dist * self.weight.unsqueeze(0)

        loss = (-true_dist * log_probs).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    Args:
        alpha: Weighting factor (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        weight: Class weights (optional)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [batch_size, num_classes]
            target: Target labels [batch_size]

        Returns:
            Scalar loss
        """
        ce_loss = F.cross_entropy(pred, target, reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_loss_function(
    loss_type: str = "cross_entropy",
    label_smoothing: float = 0.05,
    weight: torch.Tensor | None = None,
) -> nn.Module:
    """
    Get loss function by name.

    Args:
        loss_type: Type of loss ("cross_entropy", "label_smoothing", "focal")
        label_smoothing: Label smoothing factor
        weight: Class weights

    Returns:
        Loss function module
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=weight)
    elif loss_type == "focal":
        return FocalLoss(weight=weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
