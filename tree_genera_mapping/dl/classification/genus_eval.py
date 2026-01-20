#!/usr/bin/env python3
"""
genus_eval.py

Evaluate a saved genus classification checkpoint on the VAL split.

Expected dataset layout:
  <images_dir>/val/<class_name>/*.tif
(or valid/validation instead of val)

Outputs (in out_dir):
- classification_report.csv
- confusion_best_val.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from tree_genera_mapping.dl.plots import plot_confusion
from tree_genera_mapping.dl.utils import load_labels_csv
from tree_genera_mapping.dl.classification.genus_dataset import GenusImageDataset
from tree_genera_mapping.dl.classification.genus_model import build_resnet_classifier


# ------------------------- data collection -------------------------
def _is_val_path(p: Path) -> bool:
    parts = {x.lower() for x in p.parts}
    return ("val" in parts) or ("valid" in parts) or ("validation" in parts)


def collect_val_images(images_dir: Path) -> pd.DataFrame:
    """
    Build dataframe for VAL split:
      image_path, class_name
    Assumes class_name is parent folder of the tif file.
    """
    records: List[dict] = []
    for p in images_dir.rglob("*.tif"):
        if not _is_val_path(p):
            continue
        class_name = p.parent.name
        records.append({"image_path": str(p), "class_name": class_name})

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError(f"No VAL .tif files found under: {images_dir}")
    return df


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())

    return np.concatenate(y_true) if y_true else np.array([]), np.concatenate(y_pred) if y_pred else np.array([])


def _maybe_get(d: dict, keys: Sequence[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a genus classification checkpoint on VAL split")

    ap.add_argument("--out-dir", required=True, help="Run directory that contains <experiment>_best.pt")
    ap.add_argument("--images-dir", required=True, help="Dataset root (expects val/<class_name>/*.tif)")
    ap.add_argument("--labels-csv", default="conf/genus_labels.csv", help="Label mapping CSV")

    ap.add_argument("--experiment", choices=["image_only", "multimodal"], default="image_only")
    ap.add_argument("--ckpt-name", default=None, help="Override checkpoint filename (default: <experiment>_best.pt)")

    # Optional overrides (normally read from checkpoint args)
    ap.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"], default=None)
    ap.add_argument("--in-channels", type=int, choices=[3, 4, 5], default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--percentile-normalize", action="store_true", help="Override to enable percentile normalization")
    ap.add_argument("--band-indices", default=None,
                    help="Comma-separated band indices (0-based) to read from source TIFF. Example: '0,1,2,3,4'")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = Path(args.images_dir)
    labels_csv = Path(args.labels_csv)

    ckpt_name = args.ckpt_name or f"{args.experiment}_best.pt"
    ckpt_path = out_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # labels
    id_to_class, class_to_id = load_labels_csv(labels_csv)
    num_classes = len(id_to_class)

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    # resolve eval params (prefer CLI override -> checkpoint args -> safe default)
    backbone = args.backbone or _maybe_get(ckpt_args, ["backbone"], "resnet50")
    in_channels = args.in_channels or _maybe_get(ckpt_args, ["in_channels", "in-channels"], 5)
    img_size = args.img_size or _maybe_get(ckpt_args, ["img_size", "img-size"], 128)

    # band indices (optional)
    band_indices: Optional[List[int]] = None
    if args.band_indices is not None:
        band_indices = [int(x) for x in args.band_indices.split(",") if x.strip() != ""]
    else:
        # allow checkpoint arg (if you decide to save it later)
        bi = _maybe_get(ckpt_args, ["band_indices", "band-indices"], None)
        if isinstance(bi, (list, tuple)) and bi:
            band_indices = [int(x) for x in bi]

    # percentile normalize: CLI flag overrides to ON, otherwise use checkpoint arg if present
    percentile_normalize = bool(args.percentile_normalize) or bool(
        _maybe_get(ckpt_args, ["percentile_normalize", "percentile-normalize"], False)
    )

    # collect val df
    val_df = collect_val_images(images_dir)
    val_df["class_id"] = val_df["class_name"].map(class_to_id)

    if val_df["class_id"].isna().any():
        missing = val_df[val_df["class_id"].isna()]["class_name"].value_counts()
        raise ValueError("Some class folders are not present in labels CSV:\n" + missing.to_string())

    val_df["class_id"] = val_df["class_id"].astype(int)

    # dataset / loader
    if args.experiment != "image_only":
        raise NotImplementedError(
            "Multimodal evaluation is not wired here yet (needs NDVI merge + GenusTabularDataset). "
            "Run with --experiment image_only for now."
        )

    val_set = GenusImageDataset(
        val_df,
        img_size=int(img_size),
        augment=False,
        in_channels=int(in_channels),
        class_to_id=class_to_id,  # class_id is already present, but ok
        band_indices=band_indices,
        percentile_normalize=percentile_normalize,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # model
    model = build_resnet_classifier(
        model_name=str(backbone),
        num_classes=num_classes,
        in_channels=int(in_channels),
        pretrained=False,
    )
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Checkpoint {ckpt_path} does not look like expected dict with key 'model'.")
    model.load_state_dict(ckpt["model"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # predict
    y_true, y_pred = predict(model, val_loader, device=device)

    # report
    class_names = [id_to_class[i] for i in sorted(id_to_class.keys())]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T
    report_csv = out_dir / "classification_report.csv"
    report_df.to_csv(report_csv, index=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_png = out_dir / "confusion_best_val.png"
    plot_confusion(cm, classes=id_to_class, out_png=cm_png, normalize=True)

    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Eval params: backbone={backbone} in_channels={in_channels} img_size={img_size} percentile_norm={percentile_normalize}")
    if band_indices is not None:
        print(f"Band indices: {band_indices}")
    print(f"VAL samples: {len(y_true)} | Accuracy: {acc:.4f}")
    print(f"Saved: {report_csv}")
    print(f"Saved: {cm_png}")


if __name__ == "__main__":
    main()
