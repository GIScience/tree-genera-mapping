import os, json, time, argparse, re, math, random, csv, logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import cv2
from tifffile import imread, TiffFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision.models import (
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ------------------- TensorBoard (optional) -------------------
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# ========================= Logging ============================
def setup_logging(out_dir: Path, use_tensorboard: bool):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers[:] = []
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(out_dir / "train.log"); fh.setFormatter(fmt); logger.addHandler(fh)

    csv_path = out_dir / "metrics_yolo.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["epoch","time","train/loss","metrics/accuracy_top1","metrics/accuracy_top5",
                    "val/loss","lr/pg0","lr/pg1","lr/pg2","gen_gap"])

    tb = SummaryWriter(str(out_dir / "tb")) if use_tensorboard and SummaryWriter else None
    return logger, csv_f, csv_w, tb

def log_epoch_csv(csv_w, epoch:int, elapsed_s:float,
                  tr_loss:float, tr_top1:float, tr_top5:float,
                  va_loss:float, lrs:list, gen_gap:float):
    # YOLO logs as fractions; we log as fractions too for compatibility
    t1 = tr_top1 / 100.0
    t5 = tr_top5 / 100.0
    lr0 = lrs[0] if len(lrs) > 0 else ""
    lr1 = lrs[1] if len(lrs) > 1 else ""
    lr2 = lrs[2] if len(lrs) > 2 else ""
    csv_w.writerow([epoch, round(elapsed_s, 3), tr_loss, t1, t5, va_loss, lr0, lr1, lr2, gen_gap/100.0])

# =============== Dataset / normalization =====================
CHANNEL_MEAN = torch.tensor([0.3603, 0.3813, 0.3470, 0.6094, 0.1883])
CHANNEL_STD  = torch.tensor([0.1393, 0.1402, 0.1255, 0.1900, 0.0853])

ID_TO_CLASS = {
    0: 'Acer', 1: 'Aesculus', 2: 'Carpinus', 3: 'Coniferous', 4: 'Fagus',
    5: 'OtherDeciduous', 6: 'Platanus', 7: 'Prunus', 8: 'Quercus', 9: 'Tilia'
}
CLASS_TO_ID = {v: k for k, v in ID_TO_CLASS.items()}
NUM_CLASSES = 10

def read_5band_tiff(path: str, out_size: int) -> np.ndarray:
    arr = imread(path)
    if arr.ndim == 2:
        with TiffFile(path) as tf:
            pages = [p.asarray() for p in tf.pages[:5]]
        arr = np.stack(pages, axis=0)
    if arr.ndim != 3:
        raise ValueError(f"{path}: unexpected ndim={arr.ndim}")

    if arr.shape[0] == 5: chw = arr
    elif arr.shape[-1] == 5: chw = np.transpose(arr, (2, 0, 1))
    elif arr.shape[0] > 5:  chw = arr[:5]
    elif arr.shape[-1] > 5: chw = np.transpose(arr[..., :5], (2, 0, 1))
    else: raise ValueError(f"{path}: cannot infer 5 bands from {arr.shape}")

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

# ==================== Augmentations ==========================
class NormalizeCH(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = CHANNEL_MEAN.view(5, 1, 1)
        self.std  = CHANNEL_STD.view(5, 1, 1)
    def forward(self, x):
        return (x[:5] - self.mean) / self.std

class ResizeCH(nn.Module):
    def __init__(self, size=(224, 224)):
        super().__init__()
        self.size = size
    def forward(self, x):
        img = x[:5].unsqueeze(0)
        img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)
        return img.squeeze(0)

class GeometricAugmentCH(nn.Module):
    def __init__(self, size=(224,224), hflip=True, vflip=True, rotation=True, crop=True, scale=True):
        super().__init__()
        self.size = size; self.hflip=hflip; self.vflip=vflip; self.rotation=rotation; self.crop=crop; self.scale=scale
    def forward(self, x):
        img = x[:5]
        c, h, w = img.shape
        if self.crop and random.random() < 0.5:
            crop_size = int(min(h, w) * 0.8)
            i, j, th, tw = RandomCrop.get_params(img, output_size=(crop_size, crop_size))
            img = TF.resized_crop(img, i, j, th, tw, size=self.size, interpolation=TF.InterpolationMode.BILINEAR)
        if self.hflip and random.random() < 0.5:
            img = TF.hflip(img)
        if self.vflip and random.random() < 0.5:
            img = TF.vflip(img)
        if self.rotation and random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)
        if self.scale and random.random() < 0.5:
            s = random.uniform(0.9, 1.1)
            img = TF.affine(img, angle=0, translate=[0,0], scale=s, shear=[0.0,0.0])
        return img

class PhotometricAugmentCH(nn.Module):
    def __init__(self, brightness_factor=(0.8,1.2), rgb_noise=0.05, height_noise=0.02):
        super().__init__()
        self.brightness_factor=brightness_factor; self.rgb_noise=rgb_noise; self.height_noise=height_noise
    def forward(self, x):
        img_rgbnir = x[:4]; height = x[4:5]
        if random.random() < 0.5:
            factor = random.uniform(*self.brightness_factor)
            img_rgbnir = torch.clamp(img_rgbnir * factor, 0, 1)
        if random.random() < 0.5:
            noise = torch.randn_like(img_rgbnir) * self.rgb_noise
            img_rgbnir = torch.clamp(img_rgbnir + noise, 0, 1)
        if random.random() < 0.3:
            noise_h = torch.randn_like(height) * self.height_noise
            height = torch.clamp(height + noise_h, 0, 1)
        return torch.cat([img_rgbnir, height], dim=0)

class RandomErasingCH(nn.Module):
    """Simple random erasing on 5-channel tensors, applied after normalization."""
    def __init__(self, p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)):
        super().__init__()
        self.p=p; self.scale=scale; self.ratio=ratio
    def forward(self, x):
        if random.random() > self.p: return x
        c,h,w = x.shape
        area = h*w
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            er_h = int(round(math.sqrt(target_area * aspect)))
            er_w = int(round(math.sqrt(target_area / aspect)))
            if er_h < h and er_w < w:
                i = random.randint(0, h-er_h)
                j = random.randint(0, w-er_w)
                x[:, i:i+er_h, j:j+er_w] = 0.0
                return x
        return x

# ==================== Dataset ================================
class FiveBandImageDataset(Dataset):
    def __init__(self, merged_df: pd.DataFrame, img_size=128, augment=True, use_tabular=False, tabular_cols: Optional[List[str]]=None,
                 rand_erase_p: float = 0.0):
        self.df = merged_df.copy().reset_index(drop=True)
        self.img_size = img_size
        self.augment = bool(augment)
        self.use_tabular = bool(use_tabular)
        self.tabular_cols = list(tabular_cols or [])
        self.resize = ResizeCH(size=(self.img_size, self.img_size))
        self.geo_aug = GeometricAugmentCH(size=(self.img_size, self.img_size))
        self.photo_aug = PhotometricAugmentCH()
        self.norm = NormalizeCH()
        self.rand_erase = RandomErasingCH(p=rand_erase_p)

        if "class_id" not in self.df.columns:
            if "class_name" in self.df.columns:
                self.df["class_id"] = self.df["class_name"].map(CLASS_TO_ID).astype(int)
            else:
                raise ValueError("Need class_id or class_name after merge.")

        if self.use_tabular and len(self.tabular_cols) > 0:
            tab_vals = self.df[self.tabular_cols].to_numpy(dtype=np.float32)
            self.tab_median = np.nanmedian(tab_vals, axis=0)
        else:
            self.tab_median = None

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        x_img = read_5band_tiff(r["image_path"], out_size=self.img_size)
        x_img = torch.from_numpy(x_img).float()
        x_img = self.resize(x_img)
        if self.augment:
            x_img = self.geo_aug(x_img)
            x_img = self.photo_aug(x_img)
        x_img = self.norm(x_img)
        if self.augment:
            x_img = self.rand_erase(x_img)

        y = int(r["class_id"])

        if self.use_tabular and len(self.tabular_cols) > 0:
            tab = r[self.tabular_cols].to_numpy(dtype=np.float32)
            if np.isnan(tab).any():
                m = self.tab_median
                mask = np.isnan(tab); tab[mask] = m[mask]
            x_tab = torch.from_numpy(tab).float()
            return x_img, x_tab, y
        return x_img, y

# =================== Models ==================================
def _load_resnet(backbone: str):
    if backbone == "resnet34":
        base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif backbone == "resnet101":
        base = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    else:
        raise ValueError("backbone must be one of {resnet34,resnet50,resnet101}")
    return base

def make_resnet_5ch(backbone: str, num_classes: int, dropout: float = 0.0) -> nn.Module:
    m = _load_resnet(backbone)
    old_conv = m.conv1
    new_conv = nn.Conv2d(5, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3, :, :] = old_conv.weight[:, 0, :, :]
        nn.init.kaiming_normal_(new_conv.weight[:, 4:5, :, :], mode="fan_out", nonlinearity="relu")
    m.conv1 = new_conv
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(*( [nn.Dropout(p=dropout)] if dropout>0 else [] ),
                         nn.Linear(in_feat, num_classes))
    return m

class TabularMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(p),
            nn.Linear(hidden, out_dim), nn.ReLU(True), nn.Dropout(p)
        )
    def forward(self, x): return self.net(x)

class MultiModalResNet(nn.Module):
    def __init__(self, backbone: str, num_classes: int, tabular_dim: int,
                 tab_hidden=256, fused_hidden=512, p=0.3, dropout_head=0.3):
        super().__init__()
        base = _load_resnet(backbone)
        old = base.conv1
        new = nn.Conv2d(5, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3, :, :] = old.weight
            new.weight[:, 3, :, :] = old.weight[:, 0, :, :]
            nn.init.kaiming_normal_(new.weight[:, 4:5, :, :], mode="fan_out", nonlinearity="relu")
        base.conv1 = new

        fdim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.tabular = TabularMLP(tabular_dim, hidden=tab_hidden, out_dim=256, p=p)
        self.head = nn.Sequential(
            nn.Linear(fdim + 256, fused_hidden), nn.ReLU(True), nn.Dropout(dropout_head),
            nn.Linear(fused_hidden, num_classes)
        )

    def forward(self, x_img, x_tab):
        fi = self.backbone(x_img)
        ft = self.tabular(x_tab)
        return self.head(torch.cat([fi, ft], dim=1))

# =================== Losses / helpers ========================
class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma: float = 1.5, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma); self.alpha = alpha; self.reduction = reduction
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        pt = p[torch.arange(p.size(0), device=logits.device), target]
        logpt = logp[torch.arange(logp.size(0), device=logits.device), target]
        focal_weight = (1.0 - pt).clamp(0, 1).pow(self.gamma)
        if self.alpha is None:
            loss = -focal_weight * logpt
        else:
            a = self.alpha.to(logits.device).float()
            if self.alpha.ndim == 0: alpha_t = a
            else: alpha_t = a[target]
            loss = - alpha_t * focal_weight * logpt
        return loss.mean() if self.reduction=="mean" else loss.sum() if self.reduction=="sum" else loss

def cross_entropy_soft_targets(logits: torch.Tensor, soft_targets: torch.Tensor, label_smoothing: float=0.0):
    # soft_targets: [B,C], sums to 1
    if label_smoothing > 0:
        num_classes = logits.size(1)
        smooth = label_smoothing / num_classes
        soft_targets = (1 - label_smoothing) * soft_targets + smooth
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()

def topk(output, target, ks=(1,)):
    maxk = max(ks); B = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None].expand_as(pred))
    return [ (correct[:k].reshape(-1).float().sum(0) * (100.0 / B)).item() for k in ks ]

def compute_class_weights_invfreq(labels: np.ndarray, num_classes=NUM_CLASSES) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)

def build_alpha(mode: str, train_labels: np.ndarray, scalar_alpha: float) -> Optional[torch.Tensor]:
    if mode == "none": return None
    if mode == "scalar": return torch.tensor(scalar_alpha, dtype=torch.float32)
    if mode == "invfreq": return compute_class_weights_invfreq(train_labels)
    raise ValueError("alpha_mode must be one of {none,scalar,invfreq}")

# ================ MixUp / CutMix =============================
def mixup_cutmix(images, targets, mixup_alpha: float=0.0, cutmix_alpha: float=0.0, num_classes: int=NUM_CLASSES):
    """
    Returns possibly augmented images and soft targets [B,C].
    Priority: CutMix if both p > 0 (randomly chosen).
    """
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        # one-hot targets
        B = targets.size(0)
        y = torch.zeros(B, num_classes, device=targets.device, dtype=torch.float32)
        y.scatter_(1, targets.view(-1,1), 1.0)
        return images, y

    B = images.size(0)
    indices = torch.randperm(B, device=images.device)
    y1 = torch.zeros(B, num_classes, device=images.device)
    y1.scatter_(1, targets.view(-1,1), 1.0)
    y2 = y1[indices]

    use_cutmix = (cutmix_alpha > 0) and (random.random() < 0.5)
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        H, W = images.size(2), images.size(3)
        cut_rat = math.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1_ = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2_ = np.clip(cy + cut_h // 2, 0, H)
        images[:, :, y1_:y2_, x1:x2] = images[indices, :, y1_:y2_, x1:x2]
        lam = 1 - ((x2 - x1) * (y2_ - y1_)) / (W * H + 1e-8)
        y = lam * y1 + (1 - lam) * y2
        return images, y
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = lam * images + (1 - lam) * images[indices]
        y = lam * y1 + (1 - lam) * y2
        return images, y

# =================== Confusion / Curves ======================
def plot_confusion(cm: np.ndarray, classes: Dict[int, str], out_png: Path, normalize: bool = True):
    cm = np.asarray(cm, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_row = cm / cm.sum(axis=1, keepdims=True)
        cm_row[np.isnan(cm_row)] = 0.0
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(cm_row if normalize else cm, interpolation='nearest', cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Row-normalized' if normalize else 'Count', rotation=-90, va="bottom")
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
    ax.set_xticklabels([classes[i] for i in range(len(classes))], rotation=45, ha="right")
    ax.set_yticklabels([classes[i] for i in range(len(classes))])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    thresh = (cm_row.max() if normalize else cm.max()) * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j]); pct = 100.0 * cm_row[i, j]
            val_for_color = cm_row[i, j] if normalize else cm[i, j]
            ax.text(j, i, f"{count}\n{pct:.1f}%",
                    ha="center", va="center",
                    color="white" if val_for_color > thresh else "black", fontsize=9)
    plt.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_history_curves(hist: List[Dict], out_dir: Path):
    if not hist: return
    df = pd.DataFrame(hist); ep = df["epoch"].to_numpy()
    fig1, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(ep, df["train_loss"], label="train_loss", linewidth=2)
    ax1.plot(ep, df["val_loss"], label="val_loss", linewidth=2)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Losses")
    ax1.grid(True, alpha=0.3); ax1.legend()
    fig1.tight_layout(); fig1.savefig(out_dir/"losses.png", dpi=220); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.plot(ep, df["train_top1"], label="train_top1", linewidth=2)
    ax2.plot(ep, df["val_top1"], label="val_top1", linewidth=2)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("top1 (%)"); ax2.set_title("Top-1 Accuracy")
    ax2.grid(True, alpha=0.3); ax2.legend()
    fig2.tight_layout(); fig2.savefig(out_dir/"accuracy.png", dpi=220); plt.close(fig2)

    fig3, axs = plt.subplots(2,2, figsize=(10,7))
    axs[0,0].plot(ep, df["train_loss"]); axs[0,0].plot(ep, df["val_loss"]); axs[0,0].set_title("Loss"); axs[0,0].grid(True, alpha=0.3)
    axs[0,1].plot(ep, df["train_top1"]); axs[0,1].plot(ep, df["val_top1"]); axs[0,1].set_title("Top-1"); axs[0,1].grid(True, alpha=0.3)
    axs[1,0].plot(ep, df["train_top5"]); axs[1,0].plot(ep, df["val_top5"]); axs[1,0].set_title("Top-5"); axs[1,0].grid(True, alpha=0.3)
    axs[1,1].plot(ep, df["time_s"]); axs[1,1].set_title("Sec/Epoch"); axs[1,1].grid(True, alpha=0.3)
    for ax in axs.ravel(): ax.set_xlabel("epoch")
    fig3.suptitle("Training Summary", y=0.98); fig3.tight_layout(); fig3.savefig(out_dir/"results.png", dpi=220); plt.close(fig3)

# =================== Tabular autodetect ======================
def autodetect_tabular_cols(ndvi_df: pd.DataFrame) -> List[str]:
    cols = []; lower = {c: c.lower() for c in ndvi_df.columns}
    for c in ndvi_df.columns:
        lc = lower[c]
        if lc in ("canopywidt", "canopywidth"): cols.append(c)
        if lc in ("canopyheigt", "canopyheight", "canopyheig"): cols.append(c)
    for c in ndvi_df.columns:
        if re.fullmatch(r"m(0[1-9]|1[0-2])", c.lower()): cols.append(c)
    canopy = [c for c in cols if c.lower().startswith("canopy")]
    months = sorted([c for c in cols if re.fullmatch(r"m(0[1-9]|1[0-2])", c.lower())], key=str.lower)
    return canopy + months

# =================== Early stopping ==========================
class EarlyStop:
    def __init__(self, patience=10):
        self.best = -1e9; self.bad = 0; self.patience = patience
    def step(self, val_acc):
        if val_acc > self.best + 1e-4:
            self.best = val_acc; self.bad = 0
        else:
            self.bad += 1
        return self.bad > self.patience

# =================== EMA =====================================
class ModelEMA:
    def __init__(self, model: nn.Module, decay=0.9999):
        self.ema = type(model) if isinstance(model, nn.DataParallel) else model.__class__
        self.module = copy_model(model).eval()
        self.decay = decay
        for p in self.module.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict(); esd = self.module.state_dict()
        for k in esd.keys():
            esd[k].mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
    def state_dict(self): return self.module.state_dict()

def copy_model(model: nn.Module):
    import copy
    m = copy.deepcopy(model)
    return m

# =================== Train/Eval loops ========================
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None, use_amp=False,
                    mixup_alpha=0.0, cutmix_alpha=0.0, label_smoothing=0.0,
                    grad_clip: float = 0.0):
    model.train()
    loss_m=top1m=top5m=0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if len(batch) == 3:
            xi, xt, y = [b.to(device, non_blocking=True) for b in batch]
            # build soft targets with mixup/cutmix
            xi, y_soft = mixup_cutmix(xi, y, mixup_alpha, cutmix_alpha, NUM_CLASSES)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xi, xt)
                loss = cross_entropy_soft_targets(logits, y_soft, label_smoothing=label_smoothing)
        else:
            xi, y = [b.to(device, non_blocking=True) for b in batch]
            xi, y_soft = mixup_cutmix(xi, y, mixup_alpha, cutmix_alpha, NUM_CLASSES)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xi)
                loss = cross_entropy_soft_targets(logits, y_soft, label_smoothing=label_smoothing)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # metrics computed on *hard* labels
        with torch.no_grad():
            preds = logits.detach()
            t1,t5 = topk(preds, y, ks=(1,5))
            n = y.size(0)
            loss_m += loss.item()*n; top1m += t1*n/100.0; top5m += t5*n/100.0

    N = len(loader.dataset)
    return loss_m/N, (top1m/N)*100.0, (top5m/N)*100.0

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_soft_ce=False, label_smoothing=0.0):
    model.eval()
    loss_m=top1m=top5m=0.0
    preds_all=[]; y_all=[]
    for batch in loader:
        if len(batch) == 3:
            xi, xt, y = [b.to(device, non_blocking=True) for b in batch]
            logits = model(xi, xt)
        else:
            xi, y = [b.to(device, non_blocking=True) for b in batch]
            logits = model(xi)

        if use_soft_ce:
            # hard labels to one-hot for loss consistency
            B = y.size(0)
            y_soft = torch.zeros(B, NUM_CLASSES, device=y.device)
            y_soft.scatter_(1, y.view(-1,1), 1.0)
            loss = cross_entropy_soft_targets(logits, y_soft, label_smoothing=label_smoothing)
        else:
            loss = loss_fn(logits, y)

        t1,t5 = topk(logits, y, ks=(1,5))
        n = y.size(0)
        loss_m += loss.item()*n; top1m += t1*n/100.0; top5m += t5*n/100.0
        preds_all.append(logits.argmax(1).cpu().numpy()); y_all.append(y.cpu().numpy())
    N = len(loader.dataset)
    y_true = np.concatenate(y_all) if y_all else np.array([])
    y_pred = np.concatenate(preds_all) if preds_all else np.array([])
    return loss_m/N, (top1m/N)*100.0, (top5m/N)*100.0, y_true, y_pred

# =================== Main ====================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--ndvi_csv",   default=None)
    ap.add_argument("--experiment", choices=["image_only","multimodal"], default="image_only")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--backbone", choices=["resnet34","resnet50","resnet101"], default="resnet101")
    # imbalance / loss
    ap.add_argument("--sampler", choices=["none","weighted"], default="none")
    ap.add_argument("--class_weights", choices=["off","invfreq"], default="off")
    ap.add_argument("--loss", choices=["ce","focal"], default="ce")
    ap.add_argument("--focal_gamma", type=float, default=1.8)
    ap.add_argument("--alpha_mode", choices=["none","scalar","invfreq"], default="none")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    # anti-overfitting
    ap.add_argument("--dropout", type=float, default=0.3, help="Dropout before head (image-only) or in fusion head (multimodal)")
    ap.add_argument("--mixup", type=float, default=0.4, help="MixUp alpha; 0 disables")
    ap.add_argument("--cutmix", type=float, default=0.0, help="CutMix alpha; 0 disables")
    ap.add_argument("--rand_erase_p", type=float, default=0.25, help="Random erasing prob after normalization")
    ap.add_argument("--early_stop_patience", type=int, default=8)
    ap.add_argument("--ema", action="store_true", help="Enable model EMA for eval/checkpointing")
    ap.add_argument("--ema_decay", type=float, default=0.9998)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # logging
    ap.add_argument("--tensorboard", action="store_true")

    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"args.json").write_text(json.dumps(vars(args), indent=2))
    logger, csv_f, csv_w, tb = setup_logging(out_dir, args.tensorboard)
    logger.info("Args:\n" + json.dumps(vars(args), indent=2))

    # --------- load dataset lists ----------
    img_dir = Path(args.images_dir)
    imgs = img_dir.rglob("*.tif")
    img_records = []
    for img_path in imgs:
        tree_id = img_path.stem.split("_")[0]
        tree_class_name = img_path.parts[-2]  # .../<split>/<class_name>/<image>.tif
        tree_split = "train" if "train" in img_path.parts else "val" if "val" in img_path.parts else "unknown"
        img_records.append({"image_path": str(img_path),
                            "tree_id": tree_id,
                            "class_name": tree_class_name,
                            "split": tree_split})
    img_df = pd.DataFrame(img_records, columns=["image_path","tree_id","class_name","split"])
    ndvi_df = pd.read_csv(args.ndvi_csv)
    if "tree_id" not in img_df.columns:   raise ValueError("images must include tree_id")
    if "tree_id" not in ndvi_df.columns:  raise ValueError("ndvi_csv must include tree_id")

    for col in ['canopyHeig','x','y']:
        if col in ndvi_df.columns: ndvi_df = ndvi_df.drop(columns=[col])
    merged = img_df.merge(ndvi_df, on=["tree_id","class_name"], how="left", suffixes=("", "_ndvi"))

    if "class_id" not in merged.columns:
        if "class_name" in merged.columns: merged["class_id"] = merged["class_name"].map(CLASS_TO_ID).astype(int)
        elif "class_id_ndvi" in merged.columns: merged["class_id"] = merged["class_id_ndvi"].astype(int)
        elif "class_name_ndvi" in merged.columns: merged["class_id"] = merged["class_name_ndvi"].map(CLASS_TO_ID).astype(int)
        else: raise ValueError("No class labels found in merged data.")

    train_df = merged[merged['split'] == 'train'].reset_index(drop=True)
    val_df   = merged[merged['split'] == 'val'].reset_index(drop=True)
    tab_cols = autodetect_tabular_cols(ndvi_df)
    use_tabular = (args.experiment == "multimodal")

    train_set = FiveBandImageDataset(train_df, img_size=args.img_size, augment=True,
                                     use_tabular=use_tabular, tabular_cols=tab_cols,
                                     rand_erase_p=args.rand_erase_p)
    val_set   = FiveBandImageDataset(val_df, img_size=args.img_size, augment=False,
                                     use_tabular=use_tabular, tabular_cols=tab_cols,
                                     rand_erase_p=0.0)

    if args.sampler == "weighted":
        y_train = train_df["class_id"].to_numpy()
        class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
                                        num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None; shuffle = True

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=max(1, args.num_workers//2), pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    if args.experiment == "multimodal":
        model = MultiModalResNet(args.backbone, NUM_CLASSES, tabular_dim=len(tab_cols),
                                 tab_hidden=256, fused_hidden=512, p=0.2, dropout_head=args.dropout)
    else:
        model = make_resnet_5ch(args.backbone, NUM_CLASSES, dropout=args.dropout)
    model = model.to(device)

    # Loss
    if args.loss == "ce":
        if args.class_weights == "invfreq":
            y_train = train_df["class_id"].to_numpy()
            w = compute_class_weights_invfreq(y_train)
            base_ce = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=args.label_smoothing)
        else:
            base_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        loss_fn = base_ce
        cw_desc = args.class_weights; focal_desc = "off"
    else:
        y_train = train_df["class_id"].to_numpy()
        alpha_vec = build_alpha(args.alpha_mode, y_train, args.alpha)
        loss_fn = FocalCrossEntropy(gamma=args.focal_gamma, alpha=alpha_vec, reduction="mean")
        cw_desc = "n/a"; focal_desc = f"on | gamma={args.focal_gamma} | alpha={args.alpha_mode}"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # Logs
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Loss: {args.loss} (class_weights={cw_desc}; focal={focal_desc})")
    if use_tabular: logger.info(f"Tabular cols: {tab_cols}")

    best_top1 = 0.0
    hist = []
    ckpt = out_dir / f"{args.experiment}_best.pt"
    stopper = EarlyStop(patience=args.early_stop_patience)
    time_cum = 0.0

    try:
        for ep in range(1, args.epochs+1):
            t0 = time.time()
            tr_loss, tr_t1, tr_t5 = train_one_epoch(
                model, train_loader, opt, loss_fn, device, scaler, use_amp,
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                label_smoothing=args.label_smoothing,
                grad_clip=args.grad_clip
            )
            if ema: ema.update(model)

            # Evaluate both raw model and EMA; choose EMA if enabled
            eval_model = ema.module if ema else model
            va_loss, va_t1, va_t5, y_true, y_pred = evaluate(
                eval_model, val_loader,
                loss_fn if args.loss!="ce" else loss_fn, device,
                use_soft_ce=(args.loss=="ce"), label_smoothing=args.label_smoothing
            )

            for _ in range(len(train_loader)): sched.step()
            dt = time.time() - t0; time_cum += dt
            gen_gap = tr_t1 - va_t1

            hist.append({"epoch":ep, "train_loss":tr_loss, "train_top1":tr_t1, "train_top5":tr_t5,
                         "val_loss":va_loss, "val_top1":va_t1, "val_top5":va_t5, "time_s":dt})
            logger.info(f"Epoch {ep:03d}/{args.epochs} | "
                        f"train {tr_loss:.4f}/{tr_t1:.1f}/{tr_t5:.1f} | "
                        f"val {va_loss:.4f}/{va_t1:.1f}/{va_t5:.1f} | "
                        f"gap {gen_gap:.1f} | {dt:.1f}s")

            lrs = [pg["lr"] for pg in opt.param_groups]
            log_epoch_csv(csv_w, ep, time_cum, tr_loss, tr_t1, tr_t5, va_loss, lrs, gen_gap)

            if tb is not None:
                tb.add_scalar("loss/train", tr_loss, ep)
                tb.add_scalar("loss/val", va_loss, ep)
                tb.add_scalar("acc_top1/train", tr_t1, ep)
                tb.add_scalar("acc_top1/val", va_t1, ep)
                tb.add_scalar("acc_top5/train", tr_t5, ep)
                tb.add_scalar("acc_top5/val", va_t5, ep)
                tb.add_scalar("generalization_gap", gen_gap, ep)
                for i, lr in enumerate(lrs): tb.add_scalar(f"lr/pg{i}", lr, ep)

            # Save best (EMA if on)
            if va_t1 > best_top1:
                best_top1 = va_t1
                torch.save({"model": eval_model.state_dict(),
                            "classes": ID_TO_CLASS,
                            "args": vars(args),
                            "tabular_cols": tab_cols}, ckpt)

            # Per-epoch metrics
            if y_true.size and y_pred.size:
                report = classification_report(
                    y_true, y_pred,
                    labels=list(range(NUM_CLASSES)),
                    target_names=[ID_TO_CLASS[i] for i in range(NUM_CLASSES)],
                    digits=3, zero_division=0, output_dict=True)
                (out_dir / f"class_report_epoch{ep:03d}.csv").write_text(
                    pd.DataFrame(report).to_csv(index=True))
                cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
                plot_confusion(cm, ID_TO_CLASS, out_dir / f"confusion_epoch{ep:03d}.png")

            pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)

            # Early stop on val top-1
            if stopper.step(va_t1):
                logger.info(f"Early stopping triggered at epoch {ep}. Best val top1={best_top1:.2f}")
                break

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
    finally:
        plot_history_curves(hist, out_dir)
        logger.info(f"Done. Best top1={best_top1:.2f}. Saved: {ckpt}")
        csv_f.close()
        if tb is not None: tb.close()

if __name__ == "__main__":
    main()
