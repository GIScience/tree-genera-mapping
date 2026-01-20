"""
genus_eval.py

Evaluate genus classification checkpoint on VAL split.

- Uses conf/genera_labels.csv as source of truth for class mapping.
- Expects images_dir to contain split folders (train/val) and class folders:
    <images_dir>/val/<class_name>/*.tif

Outputs (in out_dir):
- classification_report.csv
- confusion_best_val.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
def collect_val_images(images_dir: Path) -> pd.DataFrame:
    """
    Build dataframe for VAL split:
      image_path, class_name

    Assumes folder layout: .../val/<class_name>/*.tif
    """
    records = []
    for p in images_dir.rglob("*.tif"):
        if "val" not in p.parts:
            continue
        class_name = p.parent.name
        records.append({"image_path": str(p), "class_name": class_name})

    df = pd.DataFrame(records)
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

    return np.concatenate(y_true), np.concatenate(y_pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Run directory that contains <experiment>_best.pt")
    ap.add_argument("--images_dir", required=True, help="Dataset root (expects val/<class_name>/*.tif)")
    ap.add_argument("--labels_csv", default="conf/genera_labels.csv", help="Label mapping CSV")

    ap.add_argument("--experiment", choices=["image_only", "multimodal"], default="image_only")
    ap.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"], default="resnet50")

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = Path(args.images_dir)
    labels_csv = Path(args.labels_csv)

    ckpt_path = out_dir / f"{args.experiment}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    id_to_class, class_to_id = load_labels_csv(labels_csv)
    num_classes = len(id_to_class)

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # build VAL dataframe
    val_df = collect_val_images(images_dir)
    val_df["class_id"] = val_df["class_name"].map(class_to_id)

    if val_df["class_id"].isna().any():
        missing = val_df[val_df["class_id"].isna()]["class_name"].value_counts()
        raise ValueError(
            "Some class folders are not present in labels CSV:\n" + missing.to_string()
        )
    val_df["class_id"] = val_df["class_id"].astype(int)

    # dataset / loader (image-only dataset API)
    val_set = GenusImageDataset(
        val_df,
        img_size=args.img_size,
        augment=False,
        class_to_id=class_to_id,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # model (image_only uses 5ch input as per dataset)
    model = build_resnet_classifier(
        model_name=args.backbone,
        num_classes=num_classes,
        in_channels=5,
        pretrained=False,  # weights come from checkpoint
    )
    model.load_state_dict(ckpt["model"])

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

    # confusion + plot (use your dl/plots.py)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_png = out_dir / "confusion_best_val.png"
    plot_confusion(cm, classes=id_to_class, out_png=cm_png, normalize=True)

    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    print(f"Checkpoint: {ckpt_path}")
    print(f"VAL samples: {len(y_true)} | Accuracy: {acc:.4f}")
    print(f"Saved: {report_csv}")
    print(f"Saved: {cm_png}")


if __name__ == "__main__":
    main()
