import torch
import numpy as np
from typing import Tuple, Optional

# -------------------------------- metrics / helpers ---------------------------------
def topk(logits: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> Tuple[float, float]:
    """Returns (top1%, top5%) for this batch."""
    maxk = max(ks)
    bsz = target.size(0)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None].expand_as(pred))
    res = []
    for k in ks:
        res.append((correct[:k].reshape(-1).float().sum(0) * (100.0 / bsz)).item())
    return float(res[0]), float(res[1])


def compute_class_weights_invfreq(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


def build_alpha(alpha_mode: str, train_labels: np.ndarray, scalar_alpha: float, num_classes: int) -> Optional[torch.Tensor]:
    if alpha_mode == "none":
        return None
    if alpha_mode == "scalar":
        return torch.tensor(float(scalar_alpha), dtype=torch.float32)
    if alpha_mode == "invfreq":
        return compute_class_weights_invfreq(train_labels, num_classes=num_classes)
    raise ValueError("alpha_mode must be one of {none, scalar, invfreq}")

