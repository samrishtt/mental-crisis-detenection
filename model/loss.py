"""
MindGuard Loss Functions
========================
Asymmetric loss functions that penalize false negatives on crisis classes
more heavily than false positives — because missing a crisis is worse
than a false alarm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class AsymmetricCrisisLoss(nn.Module):
    """
    Weighted cross-entropy loss with asymmetric false-negative penalty.
    
    Design philosophy:
        Missing a student in crisis (false negative) is FAR WORSE than
        flagging a student who is okay (false positive). This loss function
        reflects that ethical priority.
    
    Components:
        1. Base weighted CE loss — handles class imbalance
        2. Asymmetric FN penalty — extra cost for missing severe cases
        3. Ordinal smoothing — adjacent misclassifications less costly
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        fn_weight: float = 3.0,      # Extra penalty for false negatives on crisis
        fp_weight: float = 1.0,      # Standard penalty for false positives
        ordinal_penalty: float = 0.1, # Cost for distance between predicted and true
        num_classes: int = 4,
        label_smoothing: float = 0.05, # Slight label smoothing for calibration
    ):
        super().__init__()
        
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.ordinal_penalty = ordinal_penalty
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        # Weighted cross-entropy
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction='none',  # Per-sample loss for asymmetric weighting
        )
    
    def forward(
        self,
        logits: torch.Tensor,     # (batch_size, num_classes)
        labels: torch.Tensor,     # (batch_size,)
    ) -> torch.Tensor:
        """
        Compute asymmetric crisis-aware loss.
        
        The loss has three components:
        1. Weighted CE: Standard classification loss with class weights
        2. FN penalty: Extra cost when predicting LOW for TRUE HIGH severity
        3. Ordinal: Cost proportional to distance between predicted and true level
        """
        # Component 1: Weighted cross-entropy (per sample)
        ce = self.ce_loss(logits, labels)
        
        # Get predictions
        preds = torch.argmax(logits, dim=-1)
        
        # Component 2: Asymmetric false-negative penalty
        # If true label is SEVERE (3) but predicted LOW — heavy penalty
        # If true label is LOW but predicted HIGH — smaller penalty
        fn_mask = (preds < labels).float()       # Under-predicted (false negative direction)
        fp_mask = (preds > labels).float()        # Over-predicted (false positive direction)
        
        # Scale FN penalty by severity of the true label
        severity_scale = labels.float() / (self.num_classes - 1)  # 0 to 1
        
        asymmetric_weight = (
            1.0 + 
            fn_mask * self.fn_weight * (1.0 + severity_scale) +  # Worse for higher true severity
            fp_mask * self.fp_weight * 0.5  # Mild penalty for over-predicting
        )
        
        # Component 3: Ordinal distance penalty
        ordinal_dist = torch.abs(preds.float() - labels.float())
        ordinal_cost = self.ordinal_penalty * ordinal_dist
        
        # Combine
        total_loss = (ce * asymmetric_weight + ordinal_cost).mean()
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.
    Down-weights easy examples and focuses on hard ones.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        labels_one_hot = F.one_hot(labels, num_classes=logits.size(-1)).float()
        
        p_t = (probs * labels_one_hot).sum(dim=-1)
        focal_weight = (1 - p_t) ** self.gamma
        
        ce = F.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        loss = focal_weight * ce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedCrisisLoss(nn.Module):
    """
    Combined loss: Asymmetric CE + Focal Loss.
    Best of both worlds for crisis detection.
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        fn_weight: float = 3.0,
        gamma: float = 2.0,
        alpha_asymmetric: float = 0.7,
        alpha_focal: float = 0.3,
    ):
        super().__init__()
        self.asymmetric = AsymmetricCrisisLoss(
            class_weights=class_weights, fn_weight=fn_weight
        )
        self.focal = FocalLoss(alpha=class_weights, gamma=gamma)
        self.alpha_asymmetric = alpha_asymmetric
        self.alpha_focal = alpha_focal
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_asym = self.asymmetric(logits, labels)
        loss_focal = self.focal(logits, labels)
        return self.alpha_asymmetric * loss_asym + self.alpha_focal * loss_focal
