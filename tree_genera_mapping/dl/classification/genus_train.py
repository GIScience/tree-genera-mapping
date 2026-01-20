#!/usr/bin/env python3
"""
genus_train.py

Train genus classification models:
- image_only (default): ResNet classifier on 3/4/5-channel TIFF inputs
- multimodal: image + tabular features (NDVI/canopy/etc.) via MultiModalResNet

Expected dataset layout:
  <images_dir>/**/*.tif
class_name inferred from parent folder:
  .../<split>/<class_name>/<tree_id>_*.tif
split inferred from path parts containing "train" or "val".
If no split folders exist, script falls back to a random stratified split.

Outputs (out_dir):
- args.json
- train.log
- metrics.csv
- history.csv
- losses.png / accuracy.png / results.png
- confusion_epochXXX.png
- class_report_epochXXX.csv
- <experiment>_best.pt

Checkpoint format:
{
  "model": state_dict,
  "classes": {id:int -> name:str},
  "args": training args,
  "tabular_cols": list[str]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tree_genera_mapping.dl.losses import FocalCrossEntropy
from tree_genera_mapping.dl.metrics import topk, compute_class_weights_invfreq, build_alpha
from tree_genera_mapping.dl.plots import plot_confusion, plot_history_curves
from tree_genera_mapping.dl.utils import load_labels_csv
from tree_genera_mapping.dl.classification.genus_dataset import GenusImageDataset, GenusTabularDataset, autodetect_tabular_cols
from tree_genera_mapping.dl.classification.genus_model import build_resnet_classifier, MultiModalResNet

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ----------------------------- logging -----------------------------
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


# ----------------------------- early stop -----------------------------
class EarlyStop:
    """Early stopping on val_loss (lower better) or val_top1 (higher better)."""

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


# ----------------------------- data indexing -----------------------------
def _infer_split_from_path(p: Path) -> str:
    parts = {x.lower() for x in p.parts}
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    return "unknown"


def build_image_index(images_dir: Path) -> pd.DataFrame:
    records = []
    for img_path in images_dir.rglob("*.tif"):
        tree_id = img_path.stem.split("_")[0]
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
        raise ValueError(f"No .tif files found under: {images_dir}")
    return df


def split_train_val_fallback(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)
    # stratify by class_name if possible
    if df["class_name"].nunique() > 1 and df.groupby("class_name").size().min() >= 2:
        train_df = df.groupby("class_name", group_keys=False).apply(
            lambda x: x.sample(frac=(1 - val_frac), random_state=seed)
        )
    else:
        train_df = df.sample(frac=(1 - val_frac), random_state=seed)

    val_df = df.drop(train_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ----------------------------- train / eval -----------------------------
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


# ----------------------------- CLI -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train genus classification (image-only or multimodal)")

    ap.add_argument("--images-dir", required=True, help="Directory containing dataset images (expects train/ and val/).")
    ap.add_argument("--labels-csv", default=None, help="CSV with class_id,class_name (default: conf/genus_labels.csv)")
    ap.add_argument("--ndvi-csv", default=None, help="Optional per-tree tabular CSV (for multimodal).")

    ap.add_argument("--experiment", choices=["image_only", "multimodal"], default="image_only")
    ap.add_argument("--in-channels", type=int, choices=[3, 4, 5], default=5)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", required=True, help="Output directory for checkpoints and logs.")

    ap.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        default="resnet50",
        help="Torchvision ResNet backbone name.",
    )

    ap.add_argument(
        "--percentile-normalize",
        action="store_true",
        help="Apply per-image percentile normalization in dataset reader (usually OFF).",
    )

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


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    logger, csv_f, csv_w, tb = setup_logging(out_dir, args.tensorboard)
    logger.info("Args:\n" + json.dumps(vars(args), indent=2))

    # --------------------- labels ---------------------
    if args.labels_csv is None:
        # repo_root/.../tree_genera_mapping/dl/classification/genus_train.py -> repo_root/conf/genus_labels.csv
        default_labels = Path(__file__).resolve().parents[3] / "conf" / "genus_labels.csv"
        labels_csv = default_labels
    else:
        labels_csv = Path(args.labels_csv)

    id_to_class, class_to_id = load_labels_csv(labels_csv)
    num_classes = len(id_to_class)

    # validate mapping is contiguous 0..K-1 (recommended)
    ids_sorted = sorted(id_to_class.keys())
    if ids_sorted != list(range(len(ids_sorted))):
        raise ValueError(f"Labels in {labels_csv} must be contiguous 0..K-1. Got ids: {ids_sorted}")

    logger.info(f"Loaded labels: K={num_classes} from {labels_csv}")

    # --------------------- build image index ---------------------
    images_dir = Path(args.images_dir)
    img_df = build_image_index(images_dir)

    # drop unknown classes early
    unknown = sorted(set(img_df["class_name"]) - set(class_to_id.keys()))
    if unknown:
        raise ValueError(
            "Found class folders not present in labels CSV:\n"
            + "\n".join(f"- {u}" for u in unknown)
            + f"\nFix folder names or update {labels_csv}."
        )

    # split logic
    if set(img_df["split"].unique()) >= {"train", "val"}:
        train_df = img_df[img_df["split"] == "train"].reset_index(drop=True)
        val_df = img_df[img_df["split"] == "val"].reset_index(drop=True)
    else:
        logger.info("No explicit train/val folders detected -> using fallback split.")
        train_df, val_df = split_train_val_fallback(img_df, val_frac=args.val_frac, seed=args.seed)

    logger.info(f"Images: train={len(train_df)} val={len(val_df)} classes={img_df['class_name'].nunique()}")

    # --------------------- tabular (optional) ---------------------
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

    # --------------------- datasets ---------------------
    train_base = GenusImageDataset(
        train_df,
        img_size=args.img_size,
        augment=True,
        in_channels=args.in_channels,
        class_to_id=class_to_id,
        percentile_normalize=args.percentile_normalize,
    )
    val_base = GenusImageDataset(
        val_df,
        img_size=args.img_size,
        augment=False,
        in_channels=args.in_channels,
        class_to_id=class_to_id,
        percentile_normalize=args.percentile_normalize,
    )

    if use_tabular:
        train_set = GenusTabularDataset(train_base, train_df, tabular_cols=tab_cols)
        val_set = GenusTabularDataset(val_base, val_df, tabular_cols=tab_cols)
    else:
        train_set = train_base
        val_set = val_base

    # --------------------- loaders ---------------------
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

    # --------------------- model ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(torch.cuda.is_available())
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

    # --------------------- loss ---------------------
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
        # NOTE: if your build_alpha signature differs, adjust this line accordingly.
        alpha_vec = build_alpha(args.alpha_mode, y_train_ids, args.alpha, num_classes=num_classes)
        loss_fn = FocalCrossEntropy(gamma=args.focal_gamma, alpha=alpha_vec, reduction="mean")
        logger.info(f"Loss: Focal (gamma={args.focal_gamma}, alpha_mode={args.alpha_mode})")

    # --------------------- optimizer / scheduler ---------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # --------------------- training loop ---------------------
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

            # keep your old behavior: scheduler step "per-iteration-equivalent"
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
                }
                torch.save(best_state, ckpt_path)

            # per-epoch reports
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
                (out_dir / f"class_report_epoch{epoch:03d}.csv").write_text(pd.DataFrame(report).to_csv(index=True))

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
