
import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================================
# Loss Functions
# ============================================================================

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        targets = targets.float()
        if logits.dim() == 2 and logits.size(1) == 2:
            binary_logits = logits[:, 1] - logits[:, 0]  
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.pos_weight, device=logits.device),
                reduction='mean'
            )
            return loss_fn(binary_logits, targets)
        

        else:
            if logits.dim() == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
                
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.pos_weight, device=logits.device),
                reduction='mean'
            )
            return loss_fn(logits, targets)

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        if logits.dim() == 2 and logits.size(1) == 2:
            binary_logits = logits[:, 1] - logits[:, 0] 
            bce_loss = F.binary_cross_entropy_with_logits(
                binary_logits, targets, reduction='none'
            )
            pt = torch.exp(-bce_loss)
            focal = self.alpha * (1 - pt) ** self.gamma * bce_loss
            return focal.mean()
        else:
            if logits.dim() == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
                
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )
            pt = torch.exp(-bce_loss)
            focal = self.alpha * (1 - pt) ** self.gamma * bce_loss
            return focal.mean()