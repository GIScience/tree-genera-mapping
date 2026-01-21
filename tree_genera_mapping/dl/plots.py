import pandas as pd
from typing import Dict, List
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

# -------------------------------- plots (matplotlib) --------------------------------
def plot_confusion(cm: np.ndarray, classes: Dict[int, str], out_png: Path, normalize: bool = True) -> None:

    cm = np.asarray(cm, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_row = cm / cm.sum(axis=1, keepdims=True)
        cm_row[np.isnan(cm_row)] = 0.0

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(cm_row if normalize else cm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row-normalized" if normalize else "Count", rotation=-90, va="bottom")

    tick = np.arange(len(classes))
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    ax.set_xticklabels([classes[i] for i in range(len(classes))], rotation=45, ha="right")
    ax.set_yticklabels([classes[i] for i in range(len(classes))])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = (cm_row.max() if normalize else cm.max()) * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = 100.0 * cm_row[i, j]
            text = f"{count}\n{pct:.1f}%"
            val = cm_row[i, j] if normalize else cm[i, j]
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_history_curves(history: List[dict], out_dir: Path) -> None:
    if not history:
        return

    df = pd.DataFrame(history)
    ep = df["epoch"].to_numpy()

    # losses
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, df["train_loss"], label="train_loss", linewidth=2)
    ax1.plot(ep, df["val_loss"], label="val_loss", linewidth=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Losses")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "losses.png", dpi=220)
    plt.close(fig1)

    # accuracy
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(ep, df["train_top1"], label="train_top1", linewidth=2)
    ax2.plot(ep, df["val_top1"], label="val_top1", linewidth=2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("top1 (%)")
    ax2.set_title("Top-1 Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "accuracy.png", dpi=220)
    plt.close(fig2)

    # summary 2x2
    fig3, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0, 0].plot(ep, df["train_loss"], label="train_loss", linewidth=2)
    axs[0, 0].plot(ep, df["val_loss"], label="val_loss", linewidth=2)
    axs[0, 0].set_title("Loss")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].plot(ep, df["train_top1"], label="train_top1", linewidth=2)
    axs[0, 1].plot(ep, df["val_top1"], label="val_top1", linewidth=2)
    axs[0, 1].set_title("Top-1 (%)")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    axs[1, 0].plot(ep, df["train_top5"], label="train_top5", linewidth=2)
    axs[1, 0].plot(ep, df["val_top5"], label="val_top5", linewidth=2)
    axs[1, 0].set_title("Top-5 (%)")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    axs[1, 1].plot(ep, df["time_s"], label="sec/epoch", linewidth=2)
    axs[1, 1].set_title("Time per epoch (s)")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    for ax in axs.ravel():
        ax.set_xlabel("epoch")

    fig3.tight_layout()
    fig3.savefig(out_dir / "results.png", dpi=220)
    plt.close(fig3)

# ---------------- Tree Visualization ----------------
def _to_rgb(image: torch.Tensor) -> np.ndarray:
    """
    image: [C,H,W] float tensor, expected in [0,1] or close.
    Returns: HxWx3 float numpy clipped to [0,1]
    """
    if image.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(image.shape)}")
    c, h, w = image.shape
    if c < 3:
        raise ValueError("Need at least 3 channels to visualize as RGB.")
    rgb = image[:3].detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(rgb, 0.0, 1.0)


def draw_boxes(
    ax,
    boxes: np.ndarray,
    color: str,
    label: Optional[str] = None,
    scores: Optional[np.ndarray] = None,
    score_fmt: str = "{:.2f}",
    linewidth: float = 1.2,
) -> None:
    """
    boxes: Nx4 in xyxy pixel coords
    """
    if boxes is None or len(boxes) == 0:
        return

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [float(x) for x in b]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # optional text
        if scores is not None and i < len(scores):
            txt = score_fmt.format(float(scores[i]))
        elif label is not None:
            txt = str(label)
        else:
            txt = None

        if txt:
            ax.text(
                x1,
                max(0.0, y1 - 2.0),
                txt,
                color="white",
                fontsize=8,
                bbox=dict(facecolor=color, alpha=0.55, edgecolor="none", boxstyle="round,pad=0.2"),
            )


@torch.no_grad()
def save_prediction_grid(
    *,
    model: torch.nn.Module,
    dataset,
    out_png: Path,
    device: torch.device,
    n: int = 9,
    cols: int = 3,
    score_thresh: float = 0.05,
    show_gt: bool = True,
    seed: int = 0,
) -> None:
    """
    dataset must return (image, target) where:
      image: torch.Tensor [C,H,W]
      target: dict with "boxes" (Tensor Nx4), optional "labels"

    Saves a grid to out_png.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    n = int(min(n, len(dataset)))
    cols = int(max(1, cols))
    rows = (n + cols - 1) // cols

    rng = random.Random(seed)
    idxs = list(range(len(dataset)))
    rng.shuffle(idxs)
    idxs = idxs[:n]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(rows, cols)

    for k, idx in enumerate(idxs):
        r = k // cols
        c = k % cols
        ax = axes[r, c]

        img, tgt = dataset[idx]
        rgb = _to_rgb(img)

        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(f"idx={idx}")

        # GT (green)
        if show_gt and isinstance(tgt, dict) and "boxes" in tgt and tgt["boxes"].numel() > 0:
            gt_boxes = tgt["boxes"].detach().cpu().numpy()
            draw_boxes(ax, gt_boxes, color="g", label="GT", linewidth=1.6)

        # Pred (red)
        out = model([img.to(device)])[0]
        boxes = out.get("boxes", torch.empty((0, 4))).detach().cpu().numpy()
        scores = out.get("scores", torch.empty((0,))).detach().cpu().numpy()

        keep = scores >= float(score_thresh)
        boxes = boxes[keep] if len(boxes) else boxes
        scores = scores[keep] if len(scores) else scores

        draw_boxes(ax, boxes, color="r", scores=scores, linewidth=1.2)

    # blank any unused axes
    for k in range(n, rows * cols):
        r = k // cols
        c = k % cols
        axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)