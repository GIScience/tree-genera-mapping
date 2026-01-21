import os
import json
import random
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from tree_genera_mapping.dl.detection.tree_dataset import (
    TreeDetectionDataset,
    detection_collate_fn,
    basic_hflip_transform,
)
from tree_genera_mapping.dl.detection.tree_model import get_faster_rcnn_5ch

import rasterio
from tqdm import tqdm
import time, csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ------------------------ Config ------------------------
NUM_CLASSES = 2  # background + tree
BATCH_SIZE = 1
NUM_EPOCHS = 10

# NEW: dataset root with images/{train,val} and labels/{train,val}
DATASET_ROOT = "/Users/ygrinblat/Documents/HeiGIT_Projects/green_spaces/training_ds"
TRAIN_IMAGE_DIR = os.path.join(DATASET_ROOT, "images", "train")
TRAIN_LABEL_DIR = os.path.join(DATASET_ROOT, "labels", "train")
VAL_IMAGE_DIR   = os.path.join(DATASET_ROOT, "images", "val")
VAL_LABEL_DIR   = os.path.join(DATASET_ROOT, "labels", "val")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

SAVE_DIR = "cache/checkpoints"
LOG_FILE = os.path.join(SAVE_DIR, "training_log.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

VIS_SCORE_THRESH = 0.05  # for visualization and COCO dt export filtering

# ------------------------ Utils ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_prediction(image, boxes, labels):
    img_np = image[:3].permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 2, str(int(label)), color="white", fontsize=8, backgroundcolor="red")
    plt.title("Predictions")
    plt.show()

def show_training_examples_grid(dataset, num_samples=6, cols=3):
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        boxes = target["boxes"]
        img_np = image[:3].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        ax = axes[i]
        ax.imshow(img_np)
        ax.set_title(f"Image: {dataset.image_ids[idx]}, Boxes: {len(boxes)}")
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
        ax.axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()

def visualize_predictions_with_confidence(model, dataset, num_samples=6, cols=3, score_thresh=0.05):
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, _ = dataset[idx]
            image = image.to(DEVICE)
            output = model([image])[0]

            img_np = image[:3].permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)

            ax = axes[i]
            ax.imshow(img_np)
            ax.axis("off")

            for box, score in zip(output["boxes"], output["scores"]):
                if float(score) < score_thresh:
                    continue
                x1, y1, x2, y2 = box.cpu().numpy()
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
                ax.add_patch(rect)
                ax.text(
                    x1, y1, f"{float(score):.2f}", color="white", fontsize=8,
                    bbox=dict(facecolor="red", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2")
                )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


# ------------------------ Training ------------------------
def train_one_epoch(model, dataloader, optimizer):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    progress = tqdm(dataloader, desc="Training", leave=True)
    for images, targets in progress:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += float(losses.item())
        progress.set_postfix(loss=float(losses.item()))

    end_time = time.time()
    avg_loss = running_loss / max(1, len(dataloader))
    print(f"🕒 Epoch Time: {end_time - start_time:.2f}s — Avg Loss: {avg_loss:.4f}")
    return avg_loss

# ------------------------ Evaluation ------------------------
def evaluate_model(model, dataloader, device, class_names, score_thresh=0.05, visualize_first=True):
    """
    Returns: (map_50, map_50_95, mean_iou)
    """
    model.eval()
    image_id = 0
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_dt = []
    annotation_id = 1

    # categories ids start at 1 (0 is background)
    for i in range(1, len(class_names)):
        coco_gt["categories"].append({"id": i, "name": class_names[i]})

    with torch.no_grad():
        for images, targets in dataloader:
            image = images[0].to(device)
            height, width = image.shape[1:]
            outputs = model([image])[0]

            gt_boxes = targets[0]["boxes"]
            gt_labels = targets[0]["labels"]

            coco_gt["images"].append({"id": image_id, "height": height, "width": width, "file_name": f"{image_id}.tif"})

            for i in range(len(gt_boxes)):
                xmin, ymin, xmax, ymax = gt_boxes[i].tolist()
                w = xmax - xmin
                h = ymax - ymin
                coco_gt["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(gt_labels[i]),
                        "bbox": [xmin, ymin, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

            # predictions (filter by score)
            for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
                score = float(score)
                if score < score_thresh:
                    continue
                xmin, ymin, xmax, ymax = box.tolist()
                w = xmax - xmin
                h = ymax - ymin
                coco_dt.append({"image_id": image_id, "category_id": int(label), "bbox": [xmin, ymin, w, h], "score": score})

            # visualize one example
            if visualize_first and image_id == 0:
                boxes = outputs["boxes"][:5].cpu()
                labels = outputs["labels"][:5].cpu()
                visualize_prediction(image.cpu(), boxes, labels)

            image_id += 1

    if len(coco_dt) == 0:
        print("⚠️  No predictions above threshold. Skipping COCOeval.")
        return 0.0, 0.0, 0.0

    with tempfile.NamedTemporaryFile("w", delete=False) as gt_file:
        json.dump(coco_gt, gt_file)
        gt_file_path = gt_file.name
    with tempfile.NamedTemporaryFile("w", delete=False) as dt_file:
        json.dump(coco_dt, dt_file)
        dt_file_path = dt_file.name

    coco = COCO(gt_file_path)
    coco_pred = coco.loadRes(dt_file_path)
    coco_eval = COCOeval(coco, coco_pred, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50_95 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])

    ious = coco_eval.eval.get("ious")
    if ious and isinstance(ious, list) and len(ious) > 0 and getattr(ious[0], "size", 0) > 0:
        mean_iou = float(np.nanmean(ious[0]))
    else:
        mean_iou = 0.0

    print(f"mAP@[0.5:0.95]: {map_50_95:.4f}, mAP@0.5: {map_50:.4f}, mean IoU: {mean_iou:.4f}")
    return map_50, map_50_95, mean_iou

# ------------------------ Main ------------------------
def main():
    set_seed(SEED)

    # Train/val come from folders (NO random split here)
    train_dataset = TreeDetectionDataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, transform=basic_hflip_transform)
    val_dataset = TreeDetectionDataset(VAL_IMAGE_DIR, VAL_LABEL_DIR, transform=None)

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    print("🔍 Visualizing a few training examples before training starts...")
    show_training_examples_grid(train_dataset, num_samples=6, cols=3)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=detection_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=detection_collate_fn, num_workers=2)

    model = get_faster_rcnn_5ch(num_classes=NUM_CLASSES, in_channels=5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_map50 = 0.0
    class_names = ["__background__", "tree"]

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        avg_loss = train_one_epoch(model, train_loader, optimizer)
        map_50, map_50_95, mean_iou = evaluate_model(
            model, val_loader, DEVICE, class_names, score_thresh=VIS_SCORE_THRESH, visualize_first=(epoch == 0)
        )

        checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "map_50": map_50,
                "map_50_95": map_50_95,
                "mean_iou": mean_iou,
            },
            checkpoint_path,
        )
        print(f"💾 Saved model checkpoint at: {checkpoint_path}")

        if map_50 > best_map50:
            best_map50 = map_50
            best_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"✅ New best model saved (mAP@0.5 = {best_map50:.4f})")

        log_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not log_exists:
                writer.writerow(["epoch", "mAP@0.5", "mAP@0.5:0.95", "mean_iou", "loss"])
            writer.writerow([epoch + 1, map_50, map_50_95, mean_iou, avg_loss])

        print("🔍 Visualizing predictions on validation set...")
        visualize_predictions_with_confidence(model, val_dataset, num_samples=6, cols=3, score_thresh=VIS_SCORE_THRESH)


if __name__ == "__main__":
    main()
