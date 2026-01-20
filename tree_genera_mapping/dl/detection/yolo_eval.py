#!/usr/bin/env python3
"""
yolo_eval.py

Minimal, repo-ready YOLO evaluation script for object detection.

What it does
- Loads a YOLO model checkpoint (Ultralytics)
- Loads a YOLO data.yaml (val split + class names)
- Runs prediction on the val images
- Computes detection metrics at a chosen (conf, IoU) threshold:
    TP / FP / FN, Precision, Recall, F1
- Computes a (num_classes+1) x (num_classes+1) confusion matrix:
    rows = predicted class, cols = true class
    last row/col = background (missed GT / false detections)

Notes
- This is NOT mAP. It’s threshold-based evaluation, useful for model selection & debugging.
- Assumes YOLO labels in *.txt format with normalized xywh.
- Assumes labels live in a "labels" folder mirroring the "images" folder (common YOLO layout).

Example
python jobs/yolo_eval.py \
  --data conf/data.yaml \
  --weights runs/detect/train/weights/best.pt \
  --imgsz 1024 \
  --device 0 \
  --conf 0.25 \
  --iou 0.25 \
  --save-cm cache/yolo_cm.npy
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml

# Ultralytics + Torch
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO


# ----------------------------
# Data.yaml + dataset helpers
# ----------------------------
def load_data_yaml(data_yaml: str) -> Tuple[List[str], List[str]]:
    """
    Returns:
      val_images: list of image paths for validation
      class_names: list of class names indexed by class id
    """
    with open(data_yaml, "r") as f:
        y = yaml.safe_load(f)

    base = y.get("path", "") or ""
    val_rel = y.get("val")
    if val_rel is None:
        raise ValueError(f"`val` not found in {data_yaml}")

    # class names can be dict {0:'a',1:'b'} or list ['a','b']
    names = y.get("names", None)
    if names is None:
        raise ValueError(f"`names` not found in {data_yaml}")

    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names.keys(), key=int)]
    elif isinstance(names, list):
        class_names = names
    else:
        raise ValueError("`names` must be a list or a dict in data.yaml")

    val_path = Path(base) / val_rel if base else Path(val_rel)

    # Support: val points to a directory or a text file listing images
    if val_path.is_file() and val_path.suffix.lower() in {".txt"}:
        with open(val_path, "r") as f:
            val_images = [line.strip() for line in f if line.strip()]
        # Make relative lines absolute if base exists
        if base:
            val_images = [str(Path(base) / p) if not Path(p).is_absolute() else p for p in val_images]
    else:
        # Directory: collect common image formats
        exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
        val_images = []
        for e in exts:
            val_images.extend(glob.glob(str(val_path / "**" / e), recursive=True))
        val_images = sorted(val_images)

    if not val_images:
        raise ValueError(f"No validation images found for val={val_path}")

    return val_images, class_names


def image_to_label_path(img_path: str) -> str:
    """
    Typical YOLO structure:
      .../images/val/xxx.jpg  -> .../labels/val/xxx.txt
    If the path does not include /images/, we fall back to same folder with .txt.
    """
    p = Path(img_path)
    parts = list(p.parts)

    if "images" in parts:
        i = parts.index("images")
        parts[i] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
    else:
        lbl = p.with_suffix(".txt")

    return str(lbl)


def load_yolo_labels_xyxy(label_file: str, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read YOLO labels: cls x y w h (normalized, 0..1)
    Return:
      gt_boxes_xyxy: (N,4) float32 in pixel coords
      gt_cls: (N,) int64
    """
    lf = Path(label_file)
    if not lf.exists():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

    boxes = []
    clss = []
    with open(lf, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c, x, y, w, h = map(float, line.split())
            cx, cy, ww, hh = x * img_w, y * img_h, w * img_w, h * img_h
            x1, y1 = cx - ww / 2.0, cy - hh / 2.0
            x2, y2 = cx + ww / 2.0, cy + hh / 2.0
            boxes.append([x1, y1, x2, y2])
            clss.append(int(c))

    return np.asarray(boxes, np.float32), np.asarray(clss, np.int64)


# ----------------------------
# Matching + metrics
# ----------------------------
def greedy_match(
    pred_xyxy: np.ndarray,
    pred_conf: np.ndarray,
    gt_xyxy: np.ndarray,
    iou_thr: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int], List[float]]:
    """
    One-to-one greedy matching (highest conf first).
    Returns:
      matches: list of (pred_idx, gt_idx)
      unmatched_preds: list of pred indices not matched
      unmatched_gts: list of gt indices not matched
      matched_ious: list of IoUs for matches
    """
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return [], list(range(len(pred_xyxy))), list(range(len(gt_xyxy))), []

    ious = box_iou(torch.tensor(pred_xyxy), torch.tensor(gt_xyxy)).cpu().numpy()
    order = np.argsort(-pred_conf)

    gt_used = set()
    matches = []
    iou_vals = []
    unmatched_preds = []

    for pi in order:
        gi = int(np.argmax(ious[pi]))
        best_iou = float(ious[pi, gi])
        if best_iou >= iou_thr and gi not in gt_used:
            gt_used.add(gi)
            matches.append((int(pi), gi))
            iou_vals.append(best_iou)
        else:
            unmatched_preds.append(int(pi))

    unmatched_gts = [gi for gi in range(len(gt_xyxy)) if gi not in gt_used]
    return matches, unmatched_preds, unmatched_gts, iou_vals


def summarize_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def compute_confusion_matrix(
    records: List[Dict],
    num_classes: int,
    conf_thr: float,
    iou_thr: float,
) -> np.ndarray:
    """
    Confusion matrix (num_classes+1)x(num_classes+1)
    rows = predicted class, cols = true class
    last index = background
      - unmatched prediction -> (pred_class, background)
      - missed ground truth  -> (background, gt_class)
    Matched pairs -> (pred_class, gt_class)
    """
    bg = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for r in records:
        m = r["p_conf"] >= conf_thr
        p_xyxy = r["p_xyxy"][m]
        p_conf = r["p_conf"][m]
        p_cls = r["p_cls"][m]

        g_xyxy = r["gt_xyxy"]
        g_cls = r["gt_cls"]

        matches, unp, ung, _ = greedy_match(p_xyxy, p_conf, g_xyxy, iou_thr)

        for pi, gi in matches:
            cm[int(p_cls[pi]), int(g_cls[gi])] += 1

        for pi in unp:
            cm[int(p_cls[pi]), bg] += 1

        for gi in ung:
            cm[bg, int(g_cls[gi])] += 1

    return cm


# ----------------------------
# Main evaluation
# ----------------------------
@torch.no_grad()
def evaluate_yolo_threshold(
    model: YOLO,
    val_images: List[str],
    class_names: List[str],
    conf_thr: float,
    iou_thr: float,
    imgsz: int,
    device: str,
) -> Dict:
    """
    Runs predictions and computes TP/FP/FN and confusion matrix at fixed thresholds.
    """
    num_classes = len(class_names)
    records: List[Dict] = []

    # Predict per image (simple + reliable; batch predict is also possible)
    for img_path in val_images:
        res = model.predict(
            img_path,
            conf=0.001,          # predict low, apply conf_thr ourselves for evaluation
            iou=0.25,            # NMS IoU (not the eval IoU)
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]

        # Predictions
        if res.boxes is None or len(res.boxes) == 0:
            p_xyxy = np.zeros((0, 4), np.float32)
            p_conf = np.zeros((0,), np.float32)
            p_cls = np.zeros((0,), np.int64)
        else:
            p_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            p_conf = res.boxes.conf.cpu().numpy().astype(np.float32)
            p_cls = res.boxes.cls.cpu().numpy().astype(np.int64)

        # Image size for label decoding
        # Ultralytics stores orig_shape as (h,w)
        H, W = res.orig_shape
        label_path = image_to_label_path(img_path)
        gt_xyxy, gt_cls = load_yolo_labels_xyxy(label_path, img_w=W, img_h=H)

        records.append(
            {
                "img_path": img_path,
                "p_xyxy": p_xyxy,
                "p_conf": p_conf,
                "p_cls": p_cls,
                "gt_xyxy": gt_xyxy,
                "gt_cls": gt_cls,
            }
        )

    # Aggregate counts (class-agnostic detection)
    TP = FP = FN = 0
    for r in records:
        m = r["p_conf"] >= conf_thr
        p_xyxy = r["p_xyxy"][m]
        p_conf = r["p_conf"][m]

        matches, unp, ung, _ = greedy_match(p_xyxy, p_conf, r["gt_xyxy"], iou_thr)
        TP += len(matches)
        FP += len(unp)
        FN += len(ung)

    metrics = summarize_counts(TP, FP, FN)
    cm = compute_confusion_matrix(records, num_classes=num_classes, conf_thr=conf_thr, iou_thr=iou_thr)

    return {
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "metrics": metrics,
        "confusion_matrix": cm,
        "class_names": class_names,
        "conf_thr": conf_thr,
        "iou_thr": iou_thr,
    }


def _build_argparser():
    ap = argparse.ArgumentParser(description="Threshold-based YOLO evaluation (not mAP).")
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--weights", required=True, help="Path to YOLO weights (best.pt)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0", help="GPU id like '0' or 'cpu'")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for eval")
    ap.add_argument("--iou", type=float, default=0.25, help="IoU threshold for matching")
    ap.add_argument("--save-cm", default=None, help="Optional path to save confusion matrix .npy")
    return ap


def main():
    args = _build_argparser().parse_args()

    val_images, class_names = load_data_yaml(args.data)
    model = YOLO(args.weights)

    out = evaluate_yolo_threshold(
        model=model,
        val_images=val_images,
        class_names=class_names,
        conf_thr=args.conf,
        iou_thr=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    m = out["metrics"]
    print(f"Images: {len(val_images)}")
    print(f"Thresholds: conf={out['conf_thr']}, iou={out['iou_thr']}")
    print(f"TP={out['tp']} FP={out['fp']} FN={out['fn']}")
    print(f"Precision={m['precision']:.4f} Recall={m['recall']:.4f} F1={m['f1']:.4f}")

    if args.save_cm:
        np.save(args.save_cm, out["confusion_matrix"])
        print(f"Saved confusion matrix to: {args.save_cm}")

    # If you want a quick text view:
    cm = out["confusion_matrix"]
    names = out["class_names"] + ["background"]
    print("\nConfusion matrix (rows=pred, cols=true):")
    print("Classes:", names)
    print(cm)


if __name__ == "__main__":
    main()
