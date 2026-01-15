import os, json, time, argparse, re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import cv2
from tifffile import imread, TiffFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import (
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import random

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# from research_code.models.data import FiveBandImageDataset
# from research_code.models.classification_model import make_resnet_5ch, MultiModalResNet

# -------- logging helpers --------
import logging, csv
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def setup_logging(out_dir: Path, use_tensorboard: bool):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers[:] = []
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler();
    ch.setFormatter(fmt);
    logger.addHandler(ch)
    fh = logging.FileHandler(out_dir / "train.log");
    fh.setFormatter(fmt);
    logger.addHandler(fh)

    csv_path = out_dir / "metrics_yolo.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["epoch", "time", "train/loss", "metrics/accuracy_top1", "metrics/accuracy_top5",
                    "val/loss", "lr/pg0", "lr/pg1", "lr/pg2"])

    tb = SummaryWriter(str(out_dir / "tb")) if use_tensorboard and SummaryWriter else None
    return logger, csv_f, csv_w, tb


def log_epoch_csv(csv_w, epoch: int, elapsed_s: float,
                  tr_loss: float, tr_top1: float, tr_top5: float,
                  va_loss: float, lrs: list):
    # YOLO logs accuracies as fractions (0..1); you compute percents → convert
    t1 = tr_top1 / 100.0
    t5 = tr_top5 / 100.0
    lr0 = lrs[0] if len(lrs) > 0 else ""
    lr1 = lrs[1] if len(lrs) > 1 else ""
    lr2 = lrs[2] if len(lrs) > 2 else ""
    csv_w.writerow([epoch, round(elapsed_s, 3), tr_loss, t1, t5, va_loss, lr0, lr1, lr2])


# ----------------------- CHANNEL STATS (computed over training set) -----------------------
CHANNEL_MEAN = torch.tensor([0.3603, 0.3813, 0.3470, 0.6094, 0.1883])
CHANNEL_STD = torch.tensor([0.1393, 0.1402, 0.1255, 0.1900, 0.0853])

# ------------------------------- fixed 10-class mapping -------------------------------
ID_TO_CLASS = {
    0: 'Acer', 1: 'Aesculus', 2: 'Carpinus', 3: 'Coniferous', 4: 'Fagus',
    5: 'OtherDeciduous', 6: 'Platanus', 7: 'Prunus', 8: 'Quercus', 9: 'Tilia'
}
CLASS_TO_ID = {v: k for k, v in ID_TO_CLASS.items()}
NUM_CLASSES = 10


# ------------------------------------- IO utils --------------------------------------
def read_5band_tiff(path: str, out_size: int) -> np.ndarray:
    arr = imread(path)
    if arr.ndim == 2:
        with TiffFile(path) as tf:
            pages = [p.asarray() for p in tf.pages[:5]]
        arr = np.stack(pages, axis=0)
    if arr.ndim != 3:
        raise ValueError(f"{path}: unexpected ndim={arr.ndim}")
    if arr.shape[0] == 5:
        chw = arr
    elif arr.shape[-1] == 5:
        chw = np.transpose(arr, (2, 0, 1))
    elif arr.shape[0] > 5:
        chw = arr[:5]
    elif arr.shape[-1] > 5:
        chw = np.transpose(arr[..., :5], (2, 0, 1))
    else:
        raise ValueError(f"{path}: cannot infer 5 bands from {arr.shape}")

    chw = chw.astype(np.float32)
    H = W = int(out_size)
    bands = [cv2.resize(chw[k], (W, H), interpolation=cv2.INTER_AREA) for k in range(5)]
    chw = np.stack(bands, axis=0)

    all_zero = np.all(chw == 0, axis=0)
    out = np.zeros_like(chw, dtype=np.float32)
    for k in range(5):
        b = chw[k]
        b = np.where(all_zero, np.nan, b)
        if np.isfinite(b).any():
            vmin, vmax = np.nanpercentile(b, 2), np.nanpercentile(b, 98)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = np.nanmin(b), np.nanmax(b)
        else:
            vmin, vmax = 0.0, 1.0
        if vmax > vmin:
            b = np.clip((b - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            b = np.zeros_like(b, dtype=np.float32)
        if np.isnan(b).any():
            med = float(np.nanmedian(b)) if np.isfinite(b).any() else 0.0
            b = np.nan_to_num(b, nan=med)
        out[k] = b
    return out  # [5,H,W] in [0,1]


# ---------------------------------- Augment -----------------------------------------
class NormalizeCH(nn.Module):
    def __init__(self, channels=5):
        super().__init__()
        if channels not in [5, 6]:
            raise ValueError("Only 5 or 6 channels are supported.")
        # Means and stds only for first 5 channels
        self.mean = torch.tensor([0.3603, 0.3813, 0.3470, 0.6094, 0.1883]).view(5, 1, 1)
        self.std = torch.tensor([0.1393, 0.1402, 0.1255, 0.1900, 0.0853]).view(5, 1, 1)
        self.channels = channels

    def forward(self, x):
        # Normalize first 5 channels
        x_img = (x[:5] - self.mean) / self.std
        if x.shape[0] == 6:
            # Append mask channel unchanged
            return torch.cat([x_img, x[5:].unsqueeze(0) if x[5:].dim() == 2 else x[5:]], dim=0)
        return x_img


class ResizeCH(nn.Module):
    def __init__(self, size=(224, 224)):
        super().__init__()
        self.size = size

    def forward(self, x):
        # Resize image channels
        img = x[:5].unsqueeze(0)
        img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)
        if x.shape[0] == 6:
            # Resize mask with nearest-neighbor interpolation
            mask = x[5].unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, size=self.size, mode='nearest')
            return torch.cat([img.squeeze(0), mask.squeeze(0)], dim=0)
        return img.squeeze(0)


class GeometricAugmentCH(nn.Module):
    def __init__(self, size=(224, 224), hflip=True, vflip=True, rotation=True, crop=True, scale=True):
        super().__init__()
        self.size = size
        self.hflip = hflip
        self.vflip = vflip
        self.rotation = rotation
        self.crop = crop
        self.scale = scale

    def forward(self, x):
        c, h, w = x.shape
        img = x[:5]
        mask = x[5:] if c == 6 else None

        # Random crop
        if self.crop and random.random() < 0.5:
            crop_size = int(min(h, w) * 0.8)
            i, j, th, tw = RandomCrop.get_params(img, output_size=(crop_size, crop_size))
            img = TF.resized_crop(img, i, j, th, tw, size=self.size, interpolation=TF.InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.resized_crop(mask, i, j, th, tw, size=self.size, interpolation=TF.InterpolationMode.NEAREST)

        # Horizontal/Vertical flips
        if self.hflip and random.random() < 0.5:
            img = TF.hflip(img)
            if mask is not None: mask = TF.hflip(mask)
        if self.vflip and random.random() < 0.5:
            img = TF.vflip(img)
            if mask is not None: mask = TF.vflip(mask)

        # Rotation
        if self.rotation and random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)
            if mask is not None: mask = TF.rotate(mask, angle)

        # Scale
        if self.scale and random.random() < 0.5:
            scale_factor = random.uniform(0.9, 1.1)
            img = TF.affine(img, angle=0, translate=[0, 0], scale=scale_factor, shear=[0.0, 0.0])
            if mask is not None:
                mask = TF.affine(mask, angle=0, translate=[0, 0], scale=scale_factor, shear=[0.0, 0.0])

        return torch.cat([img, mask], dim=0) if mask is not None else img


class PhotometricAugmentCH(nn.Module):
    def __init__(self, brightness_factor=(0.8, 1.2), rgb_noise=0.05, height_noise=0.02):
        super().__init__()
        self.brightness_factor = brightness_factor
        self.rgb_noise = rgb_noise
        self.height_noise = height_noise

    def forward(self, x):
        img_rgbnir = x[:4]
        height = x[4:5]
        mask = x[5:] if x.shape[0] == 6 else None

        if random.random() < 0.5:
            factor = random.uniform(*self.brightness_factor)
            img_rgbnir = torch.clamp(img_rgbnir * factor, 0, 1)

        if random.random() < 0.5:
            noise = torch.randn_like(img_rgbnir) * self.rgb_noise
            img_rgbnir = torch.clamp(img_rgbnir + noise, 0, 1)

        if random.random() < 0.3:
            noise_height = torch.randn_like(height) * self.height_noise
            height = torch.clamp(height + noise_height, 0, 1)

        if mask is not None:
            return torch.cat([img_rgbnir, height, mask], dim=0)
        return torch.cat([img_rgbnir, height], dim=0)


# ---------------------------------- dataset -----------------------------------------
class FiveBandImageDataset(Dataset):
    """
    merged_df: dataframe containing image_path, tree_id, class_id (+ tabular columns if used)
    """

    def __init__(
            self,
            merged_df: pd.DataFrame,
            img_size: int = 128,
            augment: bool = True,
            use_tabular: bool = False,
            tabular_cols: Optional[List[str]] = None,
    ):
        self.df = merged_df.copy().reset_index(drop=True)
        self.img_size = img_size
        self.augment = bool(augment)
        self.use_tabular = bool(use_tabular)
        self.tabular_cols = list(tabular_cols or [])

        # Map class names -> ids if needed
        if "class_id" not in self.df.columns:
            if "class_name" in self.df.columns:
                self.df["class_id"] = self.df["class_name"].map(CLASS_TO_ID).astype(int)
            else:
                raise ValueError("Need class_id or class_name after merge.")

        # Prepare tabular median for imputation (once)
        if self.use_tabular and len(self.tabular_cols) > 0:
            tab_vals = self.df[self.tabular_cols].to_numpy(dtype=np.float32)
            self.tab_median = np.nanmedian(tab_vals, axis=0)
        else:
            self.tab_median = None

        # Compose transforms
        self.resize = ResizeCH(size=(self.img_size, self.img_size))
        self.geo_aug = GeometricAugmentCH(size=(self.img_size, self.img_size))
        self.photo_aug = PhotometricAugmentCH()
        self.norm = NormalizeCH(channels=5)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        # read -> float32 -> [5,H,W] in [0,1]
        x_img = read_5band_tiff(r["image_path"], out_size=self.img_size)
        x_img = torch.from_numpy(x_img).float()  # [5,H,W]

        # always ensure desired size (read_5band_tiff already resizes, but keep safe)
        x_img = self.resize(x_img)

        # augmentations (train only)
        if self.augment:
            x_img = self.geo_aug(x_img)
            x_img = self.photo_aug(x_img)

        # normalize ALWAYS (train/val/test)
        x_img = self.norm(x_img)

        y = int(r["class_id"])

        if self.use_tabular and len(self.tabular_cols) > 0:
            tab = r[self.tabular_cols].to_numpy(dtype=np.float32)
            if np.isnan(tab).any():
                # median impute using precomputed medians
                m = self.tab_median
                mask = np.isnan(tab)
                tab[mask] = m[mask]
            x_tab = torch.from_numpy(tab).float()
            return x_img, x_tab, y

        return x_img, y


# --------------------------------- models -------------------------------------------
def make_resnet_5ch(backbone: str, num_classes: int) -> nn.Module:
    """
    Build a 5-channel ResNet (34, 50, 101) with pretrained ImageNet weights.
    Channels:
      [0: Red, 1: Green, 2: Blue, 3: NIR (copied from Red), 4: Height (He init)]
    """
    # ---- Load pretrained backbone ----
    if backbone == "resnet34":
        m = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif backbone == "resnet101":
        m = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    else:
        raise ValueError("backbone must be one of {resnet34,resnet50,resnet101}")

    old_conv = m.conv1

    # ---- Create new 5-channel conv ----
    new_conv = nn.Conv2d(
        in_channels=5,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # ---- Fill weights ----
    with torch.no_grad():
        # Copy pretrained RGB
        new_conv.weight[:, :3, :, :] = old_conv.weight

        # NIR (4th channel): use Red channel weights
        new_conv.weight[:, 3, :, :] = old_conv.weight[:, 0, :, :]

        # Height (5th channel): random He initialization
        nn.init.kaiming_normal_(
            new_conv.weight[:, 4:5, :, :],
            mode="fan_out", nonlinearity="relu"
        )

    # ---- Replace and finish setup ----
    m.conv1 = new_conv
    m.fc = nn.Linear(m.fc.in_features, num_classes)

    return m


class TabularMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=256, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(p),
            nn.Linear(hidden, out_dim), nn.ReLU(True), nn.Dropout(p)
        )

    def forward(self, x): return self.net(x)


class MultiModalResNet(nn.Module):
    def __init__(self, backbone: str, num_classes: int, tabular_dim: int,
                 tab_hidden=256, fused_hidden=512, p=0.2):
        super().__init__()
        # Select ResNet backbone
        if backbone == "resnet34":
            base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif backbone == "resnet101":
            base = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        else:
            raise ValueError("backbone must be one of {resnet34,resnet50,resnet101}")

        old = base.conv1
        new = nn.Conv2d(
            in_channels=5,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )
        # ---- Fill weights ----
        with torch.no_grad():
            new.weight[:, :3, :, :] = old.weight  # RGB
            new.weight[:, 3, :, :] = old.weight[:, 0, :, :]  # NIR <- Red
            nn.init.kaiming_normal_(new.weight[:, 4:5, :, :], mode="fan_out", nonlinearity="relu")  # Height <- He

        base.conv1 = new

        fdim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.tabular = TabularMLP(tabular_dim, hidden=tab_hidden, out_dim=256, p=p)
        self.head = nn.Sequential(
            nn.Linear(fdim + 256, fused_hidden),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(fused_hidden, num_classes)
        )

    def forward(self, x_img, x_tab):
        fi = self.backbone(x_img)
        ft = self.tabular(x_tab)
        return self.head(torch.cat([fi, ft], dim=1))


# --------------------------------- losses -------------------------------------------
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


# ------------------------------- metrics / helpers -----------------------------------
def topk(output, target, ks=(1,)):
    maxk = max(ks);
    B = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None].expand_as(pred))
    res = []
    for k in ks:
        res.append((correct[:k].reshape(-1).float().sum(0) * (100.0 / B)).item())
    return res


def compute_class_weights_invfreq(labels: np.ndarray, num_classes=NUM_CLASSES) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


def build_alpha(mode: str, train_labels: np.ndarray, scalar_alpha: float) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    if mode == "scalar":
        return torch.tensor(scalar_alpha, dtype=torch.float32)
    if mode == "invfreq":
        return compute_class_weights_invfreq(train_labels)  # reuse invfreq as alpha_vec
    raise ValueError("alpha_mode must be one of {none,scalar,invfreq}")


def plot_confusion(cm: np.ndarray, classes: Dict[int, str], out_png: Path, normalize: bool = True):
    """
    Saves a confusion-matrix PNG with value annotations.
    If normalize=True, colors reflect row-normalized values; annotations show both count and %.
    """
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=np.float64)
    # row-normalized (per-class recall) for coloring
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_row = cm / cm.sum(axis=1, keepdims=True)
        cm_row[np.isnan(cm_row)] = 0.0

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(cm_row if normalize else cm, interpolation='nearest', cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Row-normalized' if normalize else 'Count', rotation=-90, va="bottom")

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks);
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([classes[i] for i in range(len(classes))], rotation=45, ha="right")
    ax.set_yticklabels([classes[i] for i in range(len(classes))])
    ax.set_xlabel("Predicted");
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Annotate each cell: "count\nxx.x%"
    thresh = (cm_row.max() if normalize else cm.max()) * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = 100.0 * cm_row[i, j]
            text = f"{count}\n{pct:.1f}%"
            val_for_color = cm_row[i, j] if normalize else cm[i, j]
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if val_for_color > thresh else "black",
                    fontsize=9)

    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_history_curves(hist: List[Dict], out_dir: Path):
    """
    Saves:
      - losses.png (train/val loss vs epoch)
      - accuracy.png (train/val top1 vs epoch)
      - results.png (2x2 grid: loss, top1, top5, sec/epoch)
    """
    if not hist:
        return
    df = pd.DataFrame(hist)
    ep = df["epoch"].to_numpy()

    # --- losses ---
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, df["train_loss"], label="train_loss", linewidth=2)
    ax1.plot(ep, df["val_loss"], label="val_loss", linewidth=2)
    ax1.set_xlabel("epoch");
    ax1.set_ylabel("loss");
    ax1.set_title("Losses")
    ax1.grid(True, alpha=0.3);
    ax1.legend()
    fig1.tight_layout();
    fig1.savefig(out_dir / "losses.png", dpi=220);
    plt.close(fig1)

    # --- top1 accuracy ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(ep, df["train_top1"], label="train_top1", linewidth=2)
    ax2.plot(ep, df["val_top1"], label="val_top1", linewidth=2)
    ax2.set_xlabel("epoch");
    ax2.set_ylabel("top1 (%)");
    ax2.set_title("Top-1 Accuracy")
    ax2.grid(True, alpha=0.3);
    ax2.legend()
    fig2.tight_layout();
    fig2.savefig(out_dir / "accuracy.png", dpi=220);
    plt.close(fig2)

    # --- YOLO-like summary (2x2) ---
    fig3, axs = plt.subplots(2, 2, figsize=(10, 7))
    # (1) losses
    axs[0, 0].plot(ep, df["train_loss"], label="train_loss", linewidth=2)
    axs[0, 0].plot(ep, df["val_loss"], label="val_loss", linewidth=2)
    axs[0, 0].set_title("Loss");
    axs[0, 0].grid(True, alpha=0.3);
    axs[0, 0].legend()
    # (2) top1
    axs[0, 1].plot(ep, df["train_top1"], label="train_top1", linewidth=2)
    axs[0, 1].plot(ep, df["val_top1"], label="val_top1", linewidth=2)
    axs[0, 1].set_title("Top-1 (%)");
    axs[0, 1].grid(True, alpha=0.3);
    axs[0, 1].legend()
    # (3) top5
    axs[1, 0].plot(ep, df["train_top5"], label="train_top5", linewidth=2)
    axs[1, 0].plot(ep, df["val_top5"], label="val_top5", linewidth=2)
    axs[1, 0].set_title("Top-5 (%)");
    axs[1, 0].grid(True, alpha=0.3);
    axs[1, 0].legend()
    # (4) time per epoch
    axs[1, 1].plot(ep, df["time_s"], label="sec/epoch", linewidth=2)
    axs[1, 1].set_title("Time per epoch (s)");
    axs[1, 1].grid(True, alpha=0.3);
    axs[1, 1].legend()

    for ax in axs.ravel():
        ax.set_xlabel("epoch")
    fig3.suptitle("Training Summary", y=0.98)
    fig3.tight_layout()
    fig3.savefig(out_dir / "results.png", dpi=220)
    plt.close(fig3)


# -------------------------------- tabular columns -----------------------------------
def autodetect_tabular_cols(ndvi_df: pd.DataFrame) -> List[str]:
    cols = []
    lower = {c: c.lower() for c in ndvi_df.columns}
    # canopy columns (tolerate typos)
    for c in ndvi_df.columns:
        lc = lower[c]
        if lc in ("canopywidt", "canopywidth"):
            cols.append(c)
        if lc in ("canopyheigt", "canopyheight", "canopyheig"):
            cols.append(c)
    # monthly m01..m12 (case-insensitive exact names)
    for c in ndvi_df.columns:
        if re.fullmatch(r"m(0[1-9]|1[0-2])", c.lower()):
            cols.append(c)
    # order: canopy*, then months sorted
    canopy = [c for c in cols if c.lower().startswith("canopy")]
    months = sorted([c for c in cols if re.fullmatch(r"m(0[1-9]|1[0-2])", c.lower())], key=str.lower)
    return canopy + months


# -------------------------------- data split ----------------------------------------
def split_train_val(df: pd.DataFrame, val_frac=0.2, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df["class_id"].value_counts().min() >= 2:
        train_df = df.groupby("class_id", group_keys=False).apply(
            lambda x: x.sample(frac=(1 - val_frac), random_state=seed)
        )
    else:
        train_df = df.sample(frac=(1 - val_frac), random_state=seed)
    val_df = df.drop(train_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# -------------------------------- training loops ------------------------------------
class EarlyStop:
    """
    Early stopping on a scalar metric.
    If monitor='val_loss' -> lower is better.
    If monitor='val_top1' -> higher is better.
    """

    def __init__(self, patience=10, min_delta=0.0, monitor="val_loss"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.best = None
        self.bad = 0

    def step(self, current: float) -> bool:
        if self.best is None:
            self.best = current
            return False
        improved = (current < self.best - self.min_delta) if self.monitor == "val_loss" \
            else (current > self.best + self.min_delta)
        if improved:
            self.best = current
            self.bad = 0
        else:
            self.bad += 1
        return self.bad > self.patience


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None, use_amp=False):
    model.train()
    loss_m = top1m = top5m = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if len(batch) == 3:
            xi, xt, y = [b.to(device, non_blocking=True) for b in batch]
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(xi, xt);
                loss = loss_fn(out, y)
        else:
            xi, y = [b.to(device, non_blocking=True) for b in batch]
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(xi);
                loss = loss_fn(out, y)

        if use_amp:
            scaler.scale(loss).backward();
            scaler.step(optimizer);
            scaler.update()
        else:
            loss.backward();
            optimizer.step()

        t1, t5 = topk(out.detach(), y, ks=(1, 5))
        n = y.size(0)
        loss_m += loss.item() * n;
        top1m += t1 * n / 100.0;
        top5m += t5 * n / 100.0
    N = len(loader.dataset)
    return loss_m / N, (top1m / N) * 100.0, (top5m / N) * 100.0


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_m = top1m = top5m = 0.0
    preds_all = [];
    y_all = []
    for batch in loader:
        if len(batch) == 3:
            xi, xt, y = [b.to(device, non_blocking=True) for b in batch]
            out = model(xi, xt)
        else:
            xi, y = [b.to(device, non_blocking=True) for b in batch]
            out = model(xi)
        loss = loss_fn(out, y)
        t1, t5 = topk(out, y, ks=(1, 5))
        n = y.size(0)
        loss_m += loss.item() * n;
        top1m += t1 * n / 100.0;
        top5m += t5 * n / 100.0
        preds_all.append(out.argmax(1).cpu().numpy());
        y_all.append(y.cpu().numpy())
    N = len(loader.dataset)
    y_true = np.concatenate(y_all) if y_all else np.array([])
    y_pred = np.concatenate(preds_all) if preds_all else np.array([])
    return loss_m / N, (top1m / N) * 100.0, (top5m / N) * 100.0, y_true, y_pred


# --------------------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=False, help="Dataset directory",
                    default="/mnt/sds-hd/sd17f001/ygrin/silverways/greenspaces/experiments_datasets/tree_genera_512_pol")
    ap.add_argument("--ndvi_csv", required=False,
                    help="tree_id,x,y,class_name,class_id,canopyheigt,canopyWidt,m01..m12",
                    default="/home/hd/hd_hd/hd_wn297/greenspaces/data/per_tree_ndvi_stats.csv")
    ap.add_argument("--experiment", choices=["image_only", "multimodal"], default="multimodal")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out_dir",
                    default="/mnt/sds-hd/sd17f001/ygrin/silverways/greenspaces/experiments_checkpoints/resnet5ch_runs")

    # --- NEW backbone argument ---
    ap.add_argument("--backbone", choices=["resnet34", "resnet50", "resnet101"],
                    default="resnet101", help="Choose ResNet variant")

    # NEW: imbalance + loss knobs
    ap.add_argument("--sampler", choices=["none", "weighted"], default="none",
                    help="Use WeightedRandomSampler based on class counts.")
    ap.add_argument("--class_weights", choices=["off", "invfreq"], default="off",
                    help="Apply class weights in CrossEntropy (ignored for focal).")
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal_gamma", type=float, default=1.8)
    ap.add_argument("--alpha_mode", choices=["none", "scalar", "invfreq"], default="none")
    ap.add_argument("--alpha", type=float, default=0.25, help="Scalar alpha if alpha_mode=scalar")

    # New: Logging
    ap.add_argument("--tensorboard", action="store_true", help="Write TensorBoard logs to out_dir/tb")
    # Early Stoping
    ap.add_argument("--early_stop_patience", type=int, default=10,
                    help="Epochs to wait after no improvement before stopping.")
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0,
                    help="Minimum improvement to reset patience.")
    ap.add_argument("--early_stop_monitor", choices=["val_loss", "val_top1"], default="val_loss",
                    help="Metric to monitor for early stopping.")

    # Read Args
    args = ap.parse_args()
    out_dir = Path(args.out_dir);
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    # logging init
    logger, csv_f, csv_w, tb = setup_logging(out_dir, args.tensorboard)
    logger.info("Args:\n" + json.dumps(vars(args), indent=2))

    # -------------------------- load / merge data --------------------------
    # img_df  = pd.read_csv(args.images_csv)
    img_dir = Path(args.images_dir)
    imgs = img_dir.rglob("*.tif")
    img_records = []
    for img_path in imgs:
        tree_id = img_path.stem.split("_")[0]
        tree_class_name = img_path.parts[-2]  # assuming structure .../<split>/<class_name>/<image>.tiff
        tree_split = "train" if "train" in img_path.parts else "val" if "val" in img_path.parts else "unknown"
        img_records.append({"image_path": str(img_path),
                            "tree_id": tree_id,
                            "class_name": tree_class_name,
                            "split": tree_split
                            })
    img_df = pd.DataFrame(columns=["image_path", "tree_id", "class_name", "split"], data=img_records)
    ndvi_df = pd.read_csv(args.ndvi_csv)

    if "tree_id" not in img_df.columns:   raise ValueError("images_csv must have tree_id")
    if "image_path" not in img_df.columns: raise ValueError("images_csv must have image_path")
    if "tree_id" not in ndvi_df.columns:   raise ValueError("ndvi_csv must have tree_id")
    for col in ['canopyHeig', 'x', 'y']:
        if col in ndvi_df.columns:
            ndvi_df = ndvi_df.drop(columns=[col])

    merged = img_df.merge(ndvi_df, on=["tree_id", 'class_name'], how="left", suffixes=("", "_ndvi"))

    if "class_id" not in merged.columns:
        if "class_name" in merged.columns:
            merged["class_id"] = merged["class_name"].map(CLASS_TO_ID).astype(int)
        elif "class_id_ndvi" in merged.columns:
            merged["class_id"] = merged["class_id_ndvi"].astype(int)
        elif "class_name_ndvi" in merged.columns:
            merged["class_id"] = merged["class_name_ndvi"].map(CLASS_TO_ID).astype(int)
        else:
            raise ValueError("No class labels found in merged data.")

    # Split
    train_df = merged[merged['split'] == 'train'].reset_index(drop=True)
    val_df = merged[merged['split'] == 'val'].reset_index(drop=True)
    # Tabular cols (for multimodal)
    tab_cols = autodetect_tabular_cols(ndvi_df)
    use_tabular = (args.experiment == "multimodal")

    # Datasets
    train_set = FiveBandImageDataset(train_df, img_size=args.img_size, augment=True,
                                     use_tabular=use_tabular, tabular_cols=tab_cols)
    val_set = FiveBandImageDataset(val_df, img_size=args.img_size, augment=False,
                                   use_tabular=use_tabular, tabular_cols=tab_cols)

    # Sampler (optional)
    if args.sampler == "weighted":
        y_train = train_df["class_id"].to_numpy()
        class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
                                        num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(1, args.num_workers // 2), pin_memory=True)
    # --- Model creation (modified) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    if args.experiment == "multimodal":
        tab_dim = len(tab_cols)
        model = MultiModalResNet(args.backbone, NUM_CLASSES, tabular_dim=tab_dim)
    else:
        model = make_resnet_5ch(args.backbone, NUM_CLASSES)
    model = model.to(device)

    # Loss
    if args.loss == "ce":
        if args.class_weights == "invfreq":
            y_train = train_df["class_id"].to_numpy()
            w = compute_class_weights_invfreq(y_train)  # mean≈1
            loss_fn = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.05)
            cw_desc = "invfreq"
        else:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
            cw_desc = "off"
        focal_desc = "off"
    else:
        # focal
        y_train = train_df["class_id"].to_numpy()
        alpha_vec = build_alpha(args.alpha_mode, y_train, args.alpha)
        loss_fn = FocalCrossEntropy(gamma=args.focal_gamma, alpha=alpha_vec, reduction="mean")
        cw_desc = "n/a (focal)"
        focal_desc = f"on | gamma={args.focal_gamma} | alpha={args.alpha_mode}"

    # Optim + sched
    warmup_epochs = 5
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Logs
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Loss: {args.loss} (class_weights={cw_desc}; focal={focal_desc})")
    if use_tabular: logger.info(f"Tabular cols: {tab_cols}")

    best_top1 = 0.0
    hist = []
    # === Early stopping setup ===
    ckpt = out_dir / f"{args.experiment}_best.pt"  # you already have this name below; keep it here
    best_state = None

    es = EarlyStop(patience=args.early_stop_patience,
                   min_delta=args.early_stop_min_delta,
                   monitor=args.early_stop_monitor)

    time_cum = 0.0
    try:
        for ep in range(1, args.epochs + 1):
            t0 = time.time()

            # --- train & validate ---
            tr_loss, tr_t1, tr_t5 = train_one_epoch(model, train_loader, opt, loss_fn, device, scaler, use_amp)
            va_loss, va_t1, va_t5, y_true, y_pred = evaluate(model, val_loader, loss_fn, device)

            # --- step scheduler per iteration-equivalent (keeps your current behavior) ---
            for _ in range(len(train_loader)):
                sched.step()

            # --- timing & history row ---
            dt = time.time() - t0
            time_cum += dt
            hist.append({
                "epoch": ep, "train_loss": tr_loss, "train_top1": tr_t1, "train_top5": tr_t5,
                "val_loss": va_loss, "val_top1": va_t1, "val_top5": va_t5, "time_s": dt
            })

            # --- console/file log ---
            logger.info(
                f"Epoch {ep:03d}/{args.epochs} | "
                f"train {tr_loss:.4f}/{tr_t1:.1f}/{tr_t5:.1f} | "
                f"val {va_loss:.4f}/{va_t1:.1f}/{va_t5:.1f} | {dt:.1f}s"
            )

            # --- CSV (YOLO-style) ---
            lrs = [pg["lr"] for pg in opt.param_groups]
            log_epoch_csv(csv_w, ep, time_cum, tr_loss, tr_t1, tr_t5, va_loss, lrs)

            # --- TensorBoard (optional) ---
            if tb is not None:
                tb.add_scalar("loss/train", tr_loss, ep)
                tb.add_scalar("loss/val", va_loss, ep)
                tb.add_scalar("acc_top1/train", tr_t1, ep)
                tb.add_scalar("acc_top1/val", va_t1, ep)
                tb.add_scalar("acc_top5/train", tr_t5, ep)
                tb.add_scalar("acc_top5/val", va_t5, ep)
                for i, lr in enumerate(lrs):
                    tb.add_scalar(f"lr/pg{i}", lr, ep)

            # === unified save-best (according to early-stop monitor) ===
            monitored_value = va_loss if es.monitor == "val_loss" else va_t1
            is_better = (
                    es.best is None
                    or (es.monitor == "val_loss" and monitored_value < es.best - es.min_delta)
                    or (es.monitor == "val_top1" and monitored_value > es.best + es.min_delta)
            )
            if is_better:
                best_state = {
                    "model": model.state_dict(),
                    "classes": ID_TO_CLASS,
                    "args": vars(args),
                    "tabular_cols": tab_cols
                }
                torch.save(best_state, ckpt)

            # (optional) track best_top1 just for reporting
            best_top1 = max(best_top1, va_t1)

            # --- per-epoch exports (keep BEFORE early-stop break) ---
            if y_true.size and y_pred.size:
                report = classification_report(
                    y_true, y_pred,
                    labels=list(range(NUM_CLASSES)),
                    target_names=[ID_TO_CLASS[i] for i in range(NUM_CLASSES)],
                    digits=3, zero_division=0, output_dict=True
                )
                (out_dir / f"class_report_epoch{ep:03d}.csv").write_text(
                    pd.DataFrame(report).to_csv(index=True)
                )
                cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
                plot_confusion(cm, ID_TO_CLASS, out_dir / f"confusion_epoch{ep:03d}.png")

            # --- save running curves ---
            pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)

            # === early-stop check (AFTER exports) ===
            if es.step(monitored_value):
                logger.info(f"Early stopping at epoch {ep} (monitor={es.monitor}, best={es.best:.4f}).")
                break
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
    finally:
        # === Restore best weights (from early-stopping) BEFORE final plots/logs ===
        if 'best_state' in locals() and best_state is None and ckpt.exists():
            best_state = torch.load(ckpt, map_location="cpu")
        if 'best_state' in locals() and best_state is not None:
            model.load_state_dict(best_state["model"])
            logger.info(f"Restored best weights (monitor={es.monitor}, best={es.best:.4f}) from {ckpt}")

        # Plot history
        plot_history_curves(hist, out_dir)
        logger.info(f"Done. Best top1={best_top1:.2f}. Saved: {ckpt}")
        csv_f.close()
        if tb is not None:
            tb.close()


if __name__ == "__main__":
    main()
