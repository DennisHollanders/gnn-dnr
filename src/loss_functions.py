
import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================================
# Loss Functions
# ============================================================================

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        # pos_weight only affects the positive class
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight),
            reduction='mean'
        )
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets.float())

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        # use logits → stable bce
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)                   # pt = sigmoid(logits) if target=1
        focal = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal.mean()