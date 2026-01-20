import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

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

