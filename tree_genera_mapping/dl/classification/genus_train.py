#!/usr/bin/env python3
"""
genus_train.py

Train genus classification models:
- image_only (default): ResNet classifier on 3/4/5-channel TIFF inputs
- multimodal: image + tabular features (NDVI/canopy/etc.) via MultiModalResNet

Expected dataset layout
-----------------------
IMAGES_DIR/
  train/
    Acer/*.tif
    Aesculus/*.tif
    ...
  val/
    Acer/*.tif
    ...
Optional test/ is ignored here.

Works with:
- GeoTIFF
- plain TIFF
- HWC or CHW arrays
- 3 / 4 / 5 channel inputs

Outputs (out_dir):
- args.json
- train.log
- metrics.csv
- history.csv
- channel_stats.json
- losses.png / accuracy.png / results.png
- confusion_epochXXX.png
- class_report_epochXXX.csv
- <experiment>_best.pt

Checkpoint format:
{
  "model": state_dict,
  "classes": {id:int -> name:str},
  "args": training args,
  "tabular_cols": list[str],
  "norm_mean": list[float],
  "norm_std": list[float]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tree_genera_mapping.dl.losses import FocalCrossEntropy
from tree_genera_mapping.dl.metrics import topk, compute_class_weights_invfreq, build_alpha
from tree_genera_mapping.dl.plots import plot_confusion, plot_history_curves
from tree_genera_mapping.dl.utils import load_labels_csv
from tree_genera_mapping.dl.classification.genus_model import build_resnet_classifier, MultiModalResNet

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ---------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------
DEFAULT_MEAN = {
    3: [0.3996, 0.4186, 0.3855],
    4: [0.3996, 0.4186, 0.3855, 0.6243],
    5: [0.3996, 0.4186, 0.3855, 0.6243, 0.1899],
}
DEFAULT_STD = {
    3: [0.1900, 0.1741, 0.1641],
    4: [0.1900, 0.1741, 0.1641, 0.2313],
    5: [0.1900, 0.1741, 0.1641, 0.2313, 0.1500],
}


# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
def setup_logging(out_dir: Path, use_tensorboard: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("genus_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers[:] = []

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    csv_path = out_dir / "metrics.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(
        ["epoch", "time_s", "train_loss", "train_top1", "train_top5", "val_loss", "val_top1", "val_top5", "lr"]
    )

    tb = SummaryWriter(str(out_dir / "tb")) if use_tensorboard and SummaryWriter else None
    return logger, csv_f, csv_w, tb


def log_epoch_csv(csv_w, epoch: int, time_s: float, tr, va, lr: float):
    tr_loss, tr_t1, tr_t5 = tr
    va_loss, va_t1, va_t5 = va
    csv_w.writerow([epoch, round(time_s, 3), tr_loss, tr_t1, tr_t5, va_loss, va_t1, va_t5, lr])


# ---------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# early stop
# ---------------------------------------------------------------------
class EarlyStop:
    def __init__(self, patience: int, min_delta: float, monitor: str):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = str(monitor)
        self.best: Optional[float] = None
        self.bad = 0

    def step(self, current: float) -> bool:
        if self.best is None:
            self.best = current
            return False

        if self.monitor == "val_loss":
            improved = current < (self.best - self.min_delta)
        else:
            improved = current > (self.best + self.min_delta)

        if improved:
            self.best = current
            self.bad = 0
        else:
            self.bad += 1

        return self.bad > self.patience


# ---------------------------------------------------------------------
# path / split helpers
# ---------------------------------------------------------------------
def _infer_split_from_path(p: Path) -> str:
    parts = {x.lower() for x in p.parts}
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    return "unknown"


def build_image_index(images_dir: Path) -> pd.DataFrame:
    records = []
    for ext in ("*.tif", "*.tiff"):
        for img_path in images_dir.rglob(ext):
            tree_id = img_path.stem
            class_name = img_path.parent.name
            split = _infer_split_from_path(img_path)
            records.append(
                {
                    "image_path": str(img_path),
                    "tree_id": str(tree_id),
                    "class_name": str(class_name),
                    "split": split,
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError(f"No TIFF files found under: {images_dir}")
    return df


def split_train_val_fallback(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)

    if df["class_name"].nunique() > 1 and df.groupby("class_name").size().min() >= 2:
        train_df = df.groupby("class_name", group_keys=False).apply(
            lambda x: x.sample(frac=(1 - val_frac), random_state=seed)
        )
    else:
        train_df = df.sample(frac=(1 - val_frac), random_state=seed)

    val_df = df.drop(train_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------
# image reading / preprocessing
# ---------------------------------------------------------------------
def read_any_tiff(path: str | Path) -> np.ndarray:
    """
    Read GeoTIFF or plain TIFF and return H,W,C float32.
    Georeferencing is ignored.
    """
    path = str(path)

    try:
        with rasterio.open(path) as src:
            arr = src.read()  # C,H,W
        arr = np.transpose(arr, (1, 2, 0))  # H,W,C
        return arr.astype(np.float32, copy=False)
    except Exception:
        pass

    arr = tifffile.imread(path)

    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.ndim == 3 and arr.shape[0] <= 6 and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))

    return arr.astype(np.float32, copy=False)


def ensure_hwc_channels(arr: np.ndarray, in_channels: int) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    if arr.shape[0] <= 6 and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] < in_channels:
        raise ValueError(f"Image has {arr.shape[-1]} channels but in_channels={in_channels}")

    if arr.shape[-1] > in_channels:
        arr = arr[..., :in_channels]

    return arr


def scale_image_for_training(arr: np.ndarray) -> np.ndarray:
    """
    Scale uint8-like inputs to 0..1.
    Leaves already normalized float inputs alone.
    """
    arr = arr.astype(np.float32, copy=False)
    maxv = np.nanmax(arr)
    if maxv > 1.5:
        arr = arr / 255.0
    return arr


def percentile_normalize_hwc(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[-1]):
        band = arr[..., c]
        lo, hi = np.percentile(band, [2, 98])
        if hi <= lo:
            out[..., c] = band
        else:
            out[..., c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def resize_chw(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(
        x.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


# ---------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------
def estimate_channel_stats(
    df: pd.DataFrame,
    in_channels: int,
    percentile_normalize: bool,
    max_samples: int = 2000,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """
    Estimate mean/std from training images only.
    Uses random subset for speed.
    """
    if len(df) == 0:
        raise ValueError("Cannot estimate channel stats on empty dataframe")

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    sum_c = np.zeros(in_channels, dtype=np.float64)
    sumsq_c = np.zeros(in_channels, dtype=np.float64)
    count_c = np.zeros(in_channels, dtype=np.float64)

    for _, row in df.iterrows():
        arr = read_any_tiff(row["image_path"])
        arr = ensure_hwc_channels(arr, in_channels)
        arr = scale_image_for_training(arr)

        if percentile_normalize:
            arr = percentile_normalize_hwc(arr)

        flat = arr.reshape(-1, in_channels)
        sum_c += flat.sum(axis=0)
        sumsq_c += (flat ** 2).sum(axis=0)
        count_c += flat.shape[0]

    mean = sum_c / np.maximum(count_c, 1)
    var = (sumsq_c / np.maximum(count_c, 1)) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-12))

    return mean.tolist(), std.tolist()


# ---------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------
class GenusImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int,
        augment: bool,
        in_channels: int,
        class_to_id: Dict[str, int],
        percentile_normalize: bool = False,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.in_channels = int(in_channels)
        self.class_to_id = class_to_id
        self.percentile_normalize = bool(percentile_normalize)

        if norm_mean is None:
            norm_mean = DEFAULT_MEAN[in_channels]
        if norm_std is None:
            norm_std = DEFAULT_STD[in_channels]

        self.norm_mean = torch.tensor(norm_mean, dtype=torch.float32).view(in_channels, 1, 1)
        self.norm_std = torch.tensor(norm_std, dtype=torch.float32).view(in_channels, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return x

        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])  # vertical
        if random.random() < 0.5:
            k = random.randint(0, 3)
            x = torch.rot90(x, k=k, dims=[1, 2])

        return x

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        class_name = row["class_name"]

        arr = read_any_tiff(img_path)
        arr = ensure_hwc_channels(arr, self.in_channels)
        arr = scale_image_for_training(arr)

        if self.percentile_normalize:
            arr = percentile_normalize_hwc(arr)

        x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()  # C,H,W
        x = resize_chw(x, self.img_size)
        x = self._augment(x)
        x = (x - self.norm_mean) / self.norm_std

        y = torch.tensor(self.class_to_id[class_name], dtype=torch.long)
        return x, y


class GenusTabularDataset(Dataset):
    """
    Wrap image dataset and add tabular features from dataframe columns.
    """
    def __init__(self, image_dataset: GenusImageDataset, df: pd.DataFrame, tabular_cols: List[str]):
        self.image_dataset = image_dataset
        self.df = df.reset_index(drop=True).copy()
        self.tabular_cols = list(tabular_cols)

        for c in self.tabular_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing tabular column: {c}")

        self.df[self.tabular_cols] = self.df[self.tabular_cols].apply(pd.to_numeric, errors="coerce")
        self.df[self.tabular_cols] = self.df[self.tabular_cols].fillna(self.df[self.tabular_cols].median())

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx: int):
        x_img, y = self.image_dataset[idx]
        row = self.df.iloc[idx]
        x_tab = torch.tensor(row[self.tabular_cols].values.astype(np.float32))
        return x_img, x_tab, y


def autodetect_tabular_cols(df: pd.DataFrame) -> List[str]:
    ignore = {"tree_id", "image_path", "class_name", "split"}
    cols = []
    for c in df.columns:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# ---------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> Tuple[float, float, float]:
    model.train()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        if len(batch) == 3:
            x_img, x_tab, y = batch
            x_img = x_img.to(device, non_blocking=True)
            x_tab = x_tab.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x_img, x_tab)
                loss = loss_fn(logits, y)
        else:
            x_img, y = batch
            x_img = x_img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x_img)
                loss = loss_fn(logits, y)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        n = y.size(0)
        t1, t5 = topk(logits.detach(), y, ks=(1, 5))
        loss_sum += float(loss.item()) * n
        top1_sum += t1 * n
        top5_sum += t5 * n

    N = len(loader.dataset)
    return loss_sum / N, top1_sum / N, top5_sum / N


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    for batch in loader:
        if len(batch) == 3:
            x_img, x_tab, y = batch
            x_img = x_img.to(device, non_blocking=True)
            x_tab = x_tab.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x_img, x_tab)
        else:
            x_img, y = batch
            x_img = x_img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x_img)

        loss = loss_fn(logits, y)
        n = y.size(0)
        t1, t5 = topk(logits, y, ks=(1, 5))

        loss_sum += float(loss.item()) * n
        top1_sum += t1 * n
        top5_sum += t5 * n

        y_true_all.append(y.cpu().numpy())
        y_pred_all.append(logits.argmax(1).cpu().numpy())

    N = len(loader.dataset)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    return loss_sum / N, top1_sum / N, top5_sum / N, y_true, y_pred


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train genus classification (image-only or multimodal)")

    ap.add_argument("--images-dir", required=True, help="Directory containing train/ and val/ class folders.")
    ap.add_argument("--labels-csv", default=None, help="CSV with class_id,class_name.")
    ap.add_argument("--ndvi-csv", default=None, help="Optional per-tree tabular CSV for multimodal.")

    ap.add_argument("--experiment", choices=["image_only", "multimodal"], default="image_only")
    ap.add_argument("--in-channels", type=int, choices=[3, 4, 5], default=5)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        default="resnet50",
    )

    ap.add_argument("--percentile-normalize", action="store_true")

    # normalization
    ap.add_argument(
        "--norm-mode",
        choices=["default", "estimate"],
        default="default",
        help="default = use built-in channel stats, estimate = compute stats from train split",
    )
    ap.add_argument("--stats-max-samples", type=int, default=2000)
    ap.add_argument("--norm-stats-json", default=None, help="Optional JSON with mean/std to reuse at inference")

    # imbalance + loss
    ap.add_argument("--sampler", choices=["none", "weighted"], default="none")
    ap.add_argument("--class-weights", choices=["off", "invfreq"], default="off")
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal-gamma", type=float, default=1.8)
    ap.add_argument("--alpha-mode", choices=["none", "scalar", "invfreq"], default="none")
    ap.add_argument("--alpha", type=float, default=0.25)

    # split fallback
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # logging / early stopping
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--early-stop-monitor", choices=["val_loss", "val_top1"], default="val_loss")

    return ap.parse_args()


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    logger, csv_f, csv_w, tb = setup_logging(out_dir, args.tensorboard)
    logger.info("Args:\n" + json.dumps(vars(args), indent=2))

    # labels
    if args.labels_csv is None:
        default_labels = Path(__file__).resolve().parents[3] / "conf" / "genus_labels.csv"
        labels_csv = default_labels
    else:
        labels_csv = Path(args.labels_csv)

    id_to_class, class_to_id = load_labels_csv(labels_csv)
    num_classes = len(id_to_class)

    ids_sorted = sorted(id_to_class.keys())
    if ids_sorted != list(range(len(ids_sorted))):
        raise ValueError(f"Labels in {labels_csv} must be contiguous 0..K-1. Got ids: {ids_sorted}")

    logger.info(f"Loaded labels: K={num_classes} from {labels_csv}")

    # image index
    images_dir = Path(args.images_dir)
    img_df = build_image_index(images_dir)

    unknown = sorted(set(img_df["class_name"]) - set(class_to_id.keys()))
    if unknown:
        raise ValueError(
            "Found class folders not present in labels CSV:\n"
            + "\n".join(f"- {u}" for u in unknown)
            + f"\nFix folder names or update {labels_csv}."
        )

    if set(img_df["split"].unique()) >= {"train", "val"}:
        train_df = img_df[img_df["split"] == "train"].reset_index(drop=True)
        val_df = img_df[img_df["split"] == "val"].reset_index(drop=True)
    else:
        logger.info("No explicit train/val folders detected -> using fallback split.")
        train_df, val_df = split_train_val_fallback(img_df, val_frac=args.val_frac, seed=args.seed)

    logger.info(f"Images: train={len(train_df)} val={len(val_df)} classes={img_df['class_name'].nunique()}")

    # tabular
    use_tabular = args.experiment == "multimodal"
    tab_cols: List[str] = []

    if use_tabular:
        if args.ndvi_csv is None:
            raise ValueError("--experiment multimodal requires --ndvi-csv")

        ndvi_df = pd.read_csv(args.ndvi_csv)
        if "tree_id" not in ndvi_df.columns:
            raise ValueError("ndvi_csv must contain 'tree_id' column")

        tab_cols = autodetect_tabular_cols(ndvi_df)
        if not tab_cols:
            raise ValueError("Could not autodetect tabular columns in ndvi_csv")

        train_df = train_df.merge(ndvi_df, on="tree_id", how="left")
        val_df = val_df.merge(ndvi_df, on="tree_id", how="left")
        logger.info(f"Multimodal enabled. Tabular cols: {tab_cols}")

    # normalization stats
    if args.norm_stats_json:
        stats_path = Path(args.norm_stats_json)
        stats = json.loads(stats_path.read_text())
        norm_mean = stats["mean"]
        norm_std = stats["std"]
        logger.info(f"Loaded normalization stats from {stats_path}")
    elif args.norm_mode == "estimate":
        logger.info("Estimating channel stats from train split...")
        norm_mean, norm_std = estimate_channel_stats(
            train_df,
            in_channels=args.in_channels,
            percentile_normalize=args.percentile_normalize,
            max_samples=args.stats_max_samples,
            seed=args.seed,
        )
        logger.info(f"Estimated mean={norm_mean}")
        logger.info(f"Estimated std={norm_std}")
    else:
        norm_mean = DEFAULT_MEAN[args.in_channels]
        norm_std = DEFAULT_STD[args.in_channels]
        logger.info(f"Using default mean/std for {args.in_channels} channels")

    (out_dir / "channel_stats.json").write_text(
        json.dumps({"mean": norm_mean, "std": norm_std, "in_channels": args.in_channels}, indent=2)
    )

    # datasets
    train_base = GenusImageDataset(
        train_df,
        img_size=args.img_size,
        augment=True,
        in_channels=args.in_channels,
        class_to_id=class_to_id,
        percentile_normalize=args.percentile_normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    val_base = GenusImageDataset(
        val_df,
        img_size=args.img_size,
        augment=False,
        in_channels=args.in_channels,
        class_to_id=class_to_id,
        percentile_normalize=args.percentile_normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    if use_tabular:
        train_set = GenusTabularDataset(train_base, train_df, tabular_cols=tab_cols)
        val_set = GenusTabularDataset(val_base, val_df, tabular_cols=tab_cols)
    else:
        train_set = train_base
        val_set = val_base

    # loaders
    if args.sampler == "weighted":
        y_train = train_df["class_name"].map(class_to_id).to_numpy()
        class_counts = np.bincount(y_train, minlength=num_classes)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    # model
    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if use_tabular:
        model = MultiModalResNet(
            backbone=args.backbone,
            num_classes=num_classes,
            tabular_dim=len(tab_cols),
            pretrained=True,
        )
    else:
        model = build_resnet_classifier(
            model_name=args.backbone,
            num_classes=num_classes,
            in_channels=args.in_channels,
            pretrained=True,
            extra_channel_init="copy",
        )

    model = model.to(device)
    logger.info(f"Device: {device} | AMP: {use_amp} | Model: {args.backbone} | in_channels={args.in_channels}")

    # loss
    y_train_ids = train_df["class_name"].map(class_to_id).to_numpy()

    if args.loss == "ce":
        if args.class_weights == "invfreq":
            w = compute_class_weights_invfreq(y_train_ids, num_classes=num_classes).to(device)
            loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
            logger.info("Loss: CrossEntropy (invfreq weights ON)")
        else:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
            logger.info("Loss: CrossEntropy")
    else:
        alpha_vec = build_alpha(args.alpha_mode, y_train_ids, args.alpha, num_classes=num_classes)
        loss_fn = FocalCrossEntropy(gamma=args.focal_gamma, alpha=alpha_vec, reduction="mean")
        logger.info(f"Loss: Focal (gamma={args.focal_gamma}, alpha_mode={args.alpha_mode})")

    # optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # training
    ckpt_path = out_dir / f"{args.experiment}_best.pt"
    es = EarlyStop(args.early_stop_patience, args.early_stop_min_delta, args.early_stop_monitor)

    history: List[dict] = []
    best_state = None
    best_top1 = 0.0
    time_cum = 0.0

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, use_amp)
            va = evaluate(model, val_loader, loss_fn, device)

            for _ in range(len(train_loader)):
                scheduler.step()

            dt = time.time() - t0
            time_cum += dt

            tr_loss, tr_t1, tr_t5 = tr
            va_loss, va_t1, va_t5, y_true, y_pred = va
            lr = float(optimizer.param_groups[0]["lr"])
            best_top1 = max(best_top1, va_t1)

            history.append(
                dict(
                    epoch=epoch,
                    train_loss=tr_loss,
                    train_top1=tr_t1,
                    train_top5=tr_t5,
                    val_loss=va_loss,
                    val_top1=va_t1,
                    val_top5=va_t5,
                    time_s=dt,
                    lr=lr,
                )
            )
            pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

            logger.info(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss={tr_loss:.4f} top1={tr_t1:.1f} top5={tr_t5:.1f} | "
                f"val loss={va_loss:.4f} top1={va_t1:.1f} top5={va_t5:.1f} | "
                f"{dt:.1f}s | lr={lr:.2e}"
            )

            log_epoch_csv(csv_w, epoch, time_cum, tr, (va_loss, va_t1, va_t5), lr)

            if tb is not None:
                tb.add_scalar("loss/train", tr_loss, epoch)
                tb.add_scalar("loss/val", va_loss, epoch)
                tb.add_scalar("acc_top1/train", tr_t1, epoch)
                tb.add_scalar("acc_top1/val", va_t1, epoch)
                tb.add_scalar("acc_top5/train", tr_t5, epoch)
                tb.add_scalar("acc_top5/val", va_t5, epoch)
                tb.add_scalar("lr", lr, epoch)

            monitored_value = va_loss if es.monitor == "val_loss" else va_t1
            is_better = (
                es.best is None
                or (es.monitor == "val_loss" and monitored_value < es.best - es.min_delta)
                or (es.monitor == "val_top1" and monitored_value > es.best + es.min_delta)
            )

            if is_better:
                best_state = {
                    "model": model.state_dict(),
                    "classes": id_to_class,
                    "args": vars(args),
                    "tabular_cols": tab_cols,
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                }
                torch.save(best_state, ckpt_path)

            if y_true.size and y_pred.size:
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=list(range(num_classes)),
                    target_names=[id_to_class[i] for i in range(num_classes)],
                    digits=3,
                    zero_division=0,
                    output_dict=True,
                )
                (out_dir / f"class_report_epoch{epoch:03d}.csv").write_text(
                    pd.DataFrame(report).to_csv(index=True)
                )

                cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
                plot_confusion(cm, id_to_class, out_dir / f"confusion_epoch{epoch:03d}.png")

            if es.step(monitored_value):
                logger.info(f"Early stopping at epoch {epoch} (monitor={es.monitor}, best={es.best:.4f}).")
                break

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

    finally:
        if best_state is None and ckpt_path.exists():
            best_state = torch.load(ckpt_path, map_location="cpu")

        if best_state is not None:
            model.load_state_dict(best_state["model"])
            logger.info(f"Restored best weights from {ckpt_path}")

        plot_history_curves(history, out_dir)
        logger.info(f"Done. Best val top1={best_top1:.2f}. Saved best checkpoint: {ckpt_path}")

        csv_f.close()
        if tb is not None:
            tb.close()


if __name__ == "__main__":
    main()