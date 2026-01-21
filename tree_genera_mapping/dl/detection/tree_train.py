#!/usr/bin/env python3
"""
tree_train.py

Train-only script for tree crown detection (Faster R-CNN) on a YOLO-style dataset:

  dataset_root/
    images/{train,val}/*.tif
    labels/{train,val}/*.txt

This file intentionally contains ONLY:
- training loop
- checkpointing
- minimal logging

Evaluation/visualization live elsewhere (tree_eval.py / tree_viz.py).

Checkpoint format (saved each epoch + best_model.pth):
{
  "epoch": int,
  "model_state_dict": state_dict,
  "optimizer_state_dict": state_dict,
  "train_loss": float,
  "best_loss": float,
  "args": dict
}
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from tree_genera_mapping.dl.detection.tree_dataset import (
    TreeDetectionDataset,
    detection_collate_fn,
    basic_hflip_transform,
)
from tree_genera_mapping.dl.detection.tree_model import build_detector


# ------------------------ utils ------------------------
def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    best_loss: float,
    args: Dict,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": float(train_loss),
            "best_loss": float(best_loss),
            "args": dict(args),
        },
        path,
    )


def try_resume(
    resume_path: Optional[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    """
    Returns (start_epoch, best_loss).
    start_epoch means: training will continue at start_epoch+1.
    """
    if not resume_path:
        return 0, float("inf")

    ckpt = torch.load(resume_path, map_location="cpu")

    # expected dict checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                # optimizer state may not match if hyperparams changed; continue safely
                pass
        start_epoch = int(ckpt.get("epoch", 0))
        best_loss = float(ckpt.get("best_loss", float("inf")))
        print(f"✅ Resumed: {resume_path} | start_epoch={start_epoch} | best_loss={best_loss:.6f}")
        return start_epoch, best_loss

    # allow raw state_dict
    if isinstance(ckpt, dict) and any(k.startswith("backbone") or k.endswith("weight") for k in ckpt.keys()):
        model.load_state_dict(ckpt, strict=True)
        print(f"✅ Resumed model weights (raw state_dict): {resume_path}")
        return 0, float("inf")

    raise ValueError(f"Unsupported checkpoint format: {resume_path}")


# ------------------------ training ------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    n_batches = 0

    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item())
        n_batches += 1

    return loss_sum / max(1, n_batches)


# ------------------------ CLI ------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train Faster R-CNN tree detector (train-only).")

    ap.add_argument("--dataset-root", required=True, help="Root with images/{train,val} and labels/{train,val}")
    ap.add_argument("--save-dir", default="cache/checkpoints", help="Where to write checkpoints/logs")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)

    ap.add_argument("--in-channels", type=int, default=5, choices=[3, 4, 5])
    ap.add_argument("--num-classes", type=int, default=2, help="Including background (usually 2: bg + tree)")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu (default: auto)")

    # model / norm (optional)
    ap.add_argument("--backbone", choices=["resnet50", "resnet101", "resnet152"], default="resnet50")
    ap.add_argument("--pretrained-backbone", action="store_true", help="Use torchvision pretrained backbone")
    ap.add_argument("--norm-mean", default=None, help="Comma-separated mean per channel (len=in_channels)")
    ap.add_argument("--norm-std", default=None, help="Comma-separated std per channel (len=in_channels)")
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--max-size", type=int, default=1333)

    # resume
    ap.add_argument("--resume", default=None, help="Path to checkpoint .pth to resume from")

    return ap.parse_args()


def _parse_csv_floats(s: Optional[str]) -> Optional[list]:
    if s is None:
        return None
    vals = [x.strip() for x in str(s).split(",") if x.strip() != ""]
    return [float(x) for x in vals] if vals else None


# ------------------------ main ------------------------
def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed, device)

    dataset_root = Path(args.dataset_root)
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    # persist args for reproducibility
    save_json(save_dir / "train_args.json", vars(args))

    # dirs
    train_img_dir = dataset_root / "images" / "train"
    train_lbl_dir = dataset_root / "labels" / "train"
    val_img_dir = dataset_root / "images" / "val"
    val_lbl_dir = dataset_root / "labels" / "val"

    if not train_img_dir.exists() or not train_lbl_dir.exists():
        raise FileNotFoundError(f"Missing train dirs: {train_img_dir} / {train_lbl_dir}")

    # (train-only, but warn if val missing)
    if not val_img_dir.exists() or not val_lbl_dir.exists():
        print(f"⚠️ Val dirs missing (ok for train-only): {val_img_dir} / {val_lbl_dir}")

    # dataset / loader (TRAIN ONLY)
    train_dataset = TreeDetectionDataset(
        str(train_img_dir),
        str(train_lbl_dir),
        transform=basic_hflip_transform,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {train_img_dir}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # model factory (from tree_model.py)
    norm_mean = _parse_csv_floats(args.norm_mean)
    norm_std = _parse_csv_floats(args.norm_std)
    if (norm_mean is not None) != (norm_std is not None):
        raise ValueError("Provide BOTH --norm-mean and --norm-std, or neither.")
    if norm_mean is not None and (len(norm_mean) != args.in_channels or len(norm_std) != args.in_channels):
        raise ValueError("--norm-mean/--norm-std must have len == --in-channels")

    model = build_detector(
        detector="fasterrcnn",
        num_classes=int(args.num_classes),
        in_channels=int(args.in_channels),
        backbone=args.backbone,
        norm_mean=norm_mean,
        norm_std=norm_std,
        pretrained_backbone=bool(args.pretrained_backbone),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    # resume (optional)
    start_epoch, best_loss = try_resume(args.resume, model=model, optimizer=optimizer)

    # log csv
    log_path = save_dir / "training_log.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["epoch", "train_loss", "sec"])

        for epoch in range(start_epoch + 1, int(args.epochs) + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            dt = time.time() - t0

            print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f} | {dt:.1f}s")
            w.writerow([epoch, f"{train_loss:.6f}", f"{dt:.3f}"])
            f.flush()

            # epoch checkpoint
            save_checkpoint(
                save_dir / f"model_epoch_{epoch:03d}.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                best_loss=best_loss,
                args=vars(args),
            )

            # best checkpoint by train loss (since train-only script)
            if train_loss < best_loss:
                best_loss = float(train_loss)
                save_checkpoint(
                    save_dir / "best_model.pth",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    train_loss=train_loss,
                    best_loss=best_loss,
                    args=vars(args),
                )
                print(f"✅ New best (train_loss={best_loss:.6f}) saved: {save_dir / 'best_model.pth'}")

    print(f"Done. Logs: {log_path} | Best train loss: {best_loss:.6f} | Checkpoints: {save_dir}")


if __name__ == "__main__":
    main()
