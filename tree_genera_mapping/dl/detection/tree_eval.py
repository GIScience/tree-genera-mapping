#!/usr/bin/env python3
"""
tree_eval.py

Evaluate a trained Faster R-CNN tree detector on a YOLO-style VAL split.

Expected dataset layout:
  dataset_root/
    images/val/*.tif
    labels/val/*.txt

Notes:
- Uses TreeDetectionDataset to read images + YOLO txt labels.
- Exports COCO-style GT/DT to temp files and runs COCOeval.
- Optionally saves a few visualizations via tree_viz.py.

Outputs (in --out-dir):
- coco_eval_summary.json
- preds_preview.png (optional)
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from tree_genera_mapping.dl.detection.tree_dataset import (
    TreeDetectionDataset,
    detection_collate_fn,
)
from tree_genera_mapping.dl.detection.tree_model import get_faster_rcnn_5ch

from tree_genera_mapping.dl.plots import save_prediction_grid


def build_coco_from_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    score_thresh: float,
) -> Tuple[Dict, List[Dict]]:
    """
    Returns:
      coco_gt_dict, coco_dt_list
    """
    model.eval()

    coco_gt: Dict = {"images": [], "annotations": [], "categories": []}
    coco_dt: List[Dict] = []

    # COCO category IDs start at 1 (0 reserved for background)
    for cid in range(1, len(class_names)):
        coco_gt["categories"].append({"id": cid, "name": class_names[cid]})

    ann_id = 1
    image_id = 0

    with torch.no_grad():
        for images, targets in loader:
            # this loader should be batch=1 for COCOeval sanity
            img = images[0].to(device, non_blocking=True)
            tgt = targets[0]

            # image dims
            _, h, w = img.shape

            out = model([img])[0]

            coco_gt["images"].append(
                {"id": image_id, "height": int(h), "width": int(w), "file_name": f"{image_id}.tif"}
            )

            # --- GT ---
            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.cpu().numpy()

            for b, lab in zip(gt_boxes, gt_labels):
                xmin, ymin, xmax, ymax = [float(x) for x in b]
                bw = max(0.0, xmax - xmin)
                bh = max(0.0, ymax - ymin)
                coco_gt["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(lab),
                        "bbox": [xmin, ymin, bw, bh],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            # --- DT ---
            dt_boxes = out.get("boxes", torch.empty((0, 4))).detach().cpu().numpy()
            dt_scores = out.get("scores", torch.empty((0,))).detach().cpu().numpy()
            dt_labels = out.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().numpy()

            for b, s, lab in zip(dt_boxes, dt_scores, dt_labels):
                s = float(s)
                if s < score_thresh:
                    continue
                xmin, ymin, xmax, ymax = [float(x) for x in b]
                bw = max(0.0, xmax - xmin)
                bh = max(0.0, ymax - ymin)
                coco_dt.append(
                    {
                        "image_id": image_id,
                        "category_id": int(lab),
                        "bbox": [xmin, ymin, bw, bh],
                        "score": s,
                    }
                )

            image_id += 1

    return coco_gt, coco_dt


def run_coco_eval(coco_gt: Dict, coco_dt: List[Dict]) -> Dict:
    if len(coco_dt) == 0:
        return {
            "warning": "No predictions above threshold; COCOeval skipped.",
            "stats": None,
        }

    with tempfile.NamedTemporaryFile("w", delete=False) as f_gt:
        json.dump(coco_gt, f_gt)
        gt_path = f_gt.name
    with tempfile.NamedTemporaryFile("w", delete=False) as f_dt:
        json.dump(coco_dt, f_dt)
        dt_path = f_dt.name

    coco = COCO(gt_path)
    coco_pred = coco.loadRes(dt_path)
    coco_eval = COCOeval(coco, coco_pred, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = [float(x) for x in coco_eval.stats.tolist()]  # 12 metrics
    names = [
        "AP@[.5:.95]",
        "AP@.5",
        "AP@.75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR@1",
        "AR@10",
        "AR@100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]

    # Optional "mean IoU" (rough): COCOeval stores ious per image/category.
    mean_iou = None
    ious = coco_eval.eval.get("ious")
    if ious and isinstance(ious, list) and len(ious) > 0:
        arrs = []
        for x in ious:
            if hasattr(x, "size") and x.size > 0:
                arrs.append(np.nanmean(x))
        if arrs:
            mean_iou = float(np.nanmean(arrs))

    return {
        "stats": {k: v for k, v in zip(names, stats)},
        "mean_iou_proxy": mean_iou,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate Faster R-CNN tree detector on val split (COCOeval).")

    ap.add_argument("--dataset-root", required=True, help="Root with images/val and labels/val")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (best_model.pth or model_epoch_XXX.pth)")
    ap.add_argument("--out-dir", required=True, help="Directory to write eval outputs")

    ap.add_argument("--in-channels", type=int, default=5, choices=[3, 4, 5])
    ap.add_argument("--num-classes", type=int, default=2, help="Including background (usually 2)")
    ap.add_argument("--score-thresh", type=float, default=0.05)

    ap.add_argument("--batch-size", type=int, default=1, help="Keep 1 for clean COCOeval")
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu (default: auto)")
    ap.add_argument("--save-preview", action="store_true", help="Save a small prediction grid PNG")
    ap.add_argument("--preview-n", type=int, default=9, help="Number of images for preview grid")
    ap.add_argument("--preview-cols", type=int, default=3)

    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    img_dir = dataset_root / "images" / "val"
    lbl_dir = dataset_root / "labels" / "val"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing val dirs: {img_dir} / {lbl_dir}")

    # dataset/loader
    ds = TreeDetectionDataset(str(img_dir), str(lbl_dir), transform=None)
    if len(ds) == 0:
        raise ValueError(f"No validation samples found in {img_dir}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # model
    model = get_faster_rcnn_5ch(num_classes=args.num_classes, in_channels=args.in_channels).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        # allow your other checkpoint format
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        # allow raw state_dict
        model.load_state_dict(ckpt, strict=True)

    class_names = ["__background__", "tree"]

    # coco build + eval
    coco_gt, coco_dt = build_coco_from_loader(
        model=model,
        loader=loader,
        device=device,
        class_names=class_names,
        score_thresh=float(args.score_thresh),
    )
    summary = run_coco_eval(coco_gt, coco_dt)

    # save outputs
    (out_dir / "coco_eval_summary.json").write_text(json.dumps(summary, indent=2))

    # optional preview
    if args.save_preview:
        save_prediction_grid(
            model=model,
            dataset=ds,
            out_png=out_dir / "preds_preview.png",
            device=device,
            n=int(args.preview_n),
            cols=int(args.preview_cols),
            score_thresh=float(args.score_thresh),
            show_gt=True,
        )

    # console
    print(f"Checkpoint: {args.ckpt}")
    print(f"VAL images: {len(ds)} | score_thresh={args.score_thresh}")
    if summary.get("stats"):
        print("COCOeval:")
        for k, v in summary["stats"].items():
            print(f"  {k:>12}: {v:.4f}")
    else:
        print(summary.get("warning", "No stats"))
    if summary.get("mean_iou_proxy") is not None:
        print(f"mean_iou_proxy: {summary['mean_iou_proxy']:.4f}")
    print(f"Saved: {out_dir / 'coco_eval_summary.json'}")
    if args.save_preview:
        print(f"Saved: {out_dir / 'preds_preview.png'}")


if __name__ == "__main__":
    main()
