import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Dict
class FocalCrossEntropy(nn.Module):
    """
    Multi-class focal loss (logits) with optional alpha:
      - alpha=None: no class prior reweighting
      - alpha=float: scalar in [0,1]
      - alpha=torch.Tensor([C]): per-class alpha weights (sum unconstrained)
    """


    def __init__(self, gamma: float = 1.5, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha  # set at creation; match device later
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits [B,C], target [B] (class indices)
        logp = torch.log_softmax(logits, dim=1)  # [B,C]
        p = torch.exp(logp)  # softmax probs
        pt = p[torch.arange(p.size(0), device=logits.device), target]  # [B]
        logpt = logp[torch.arange(logp.size(0), device=logits.device), target]
        focal_weight = (1.0 - pt).clamp(0, 1).pow(self.gamma)  # [B]

        if self.alpha is None:
            loss = - focal_weight * logpt
        else:
            if self.alpha.ndim == 0:
                a = self.alpha.to(logits.device).clamp(0, 1)
                alpha_t = a  # scalar to all
            else:
                a = self.alpha.to(logits.device).float()  # [C]
                alpha_t = a[target]  # [B]
            loss = - alpha_t * focal_weight * logpt

        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss