"""
tree_dataset.py

Dataset + basic augmentation for tree detection training.

Assumes YOLO txt labels:
  <class_id> <x_center> <y_center> <w> <h>   (all relative 0..1)

Returns FasterRCNN-style targets:
  target = {"boxes": FloatTensor[N,4], "labels": LongTensor[N], "image_id": Tensor[1]}

Notes:
- FasterRCNN expects foreground class ids to start at 1 (0 is background),
  so we map YOLO class_id 0 -> 1 by default via label_offset=1.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


# ------------------------ collate ------------------------
def detection_collate_fn(batch):
    """For torchvision detection models."""
    return tuple(zip(*batch))


# ------------------------ transforms ------------------------
def basic_hflip_transform(image: torch.Tensor, target: dict, p: float = 0.5):
    """
    Random horizontal flip.
    image: Tensor[C,H,W]
    target["boxes"]: Tensor[N,4] in pixel coords
    """
    if random.random() > p:
        return image, target

    image = image.flip(-1)  # flip W
    width = image.shape[2]

    boxes = target["boxes"]
    if boxes.numel() > 0:
        # x' = W - x
        boxes = boxes.clone()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target["boxes"] = boxes

    return image, target


# ------------------------ dataset ------------------------
@dataclass(frozen=True)
class DetectionSample:
    image_id: str
    image_path: str
    label_path: str


def list_image_ids(image_dir: str, exts: Sequence[str] = (".tif", ".tiff")) -> List[str]:
    """Return ids without extension, from files in image_dir."""
    ids: List[str] = []
    for f in os.listdir(image_dir):
        fl = f.lower()
        if any(fl.endswith(e) for e in exts):
            ids.append(os.path.splitext(f)[0])
    ids.sort()
    return ids


class TreeDetectionDataset(Dataset):
    """
    image_dir: folder with *.tif
    label_dir: folder with *.txt (YOLO format)
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        image_ids: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[torch.Tensor, dict], Tuple[torch.Tensor, dict]]] = None,
        *,
        scale_mode: str = "uint8_255",
        label_offset: int = 1,
    ):
        self.image_dir = str(image_dir)
        self.label_dir = str(label_dir)
        self.transform = transform
        self.scale_mode = str(scale_mode)
        self.label_offset = int(label_offset)

        if image_ids is None:
            self.image_ids = list_image_ids(self.image_dir)
        else:
            self.image_ids = list(image_ids)

        if len(self.image_ids) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def _read_image(self, image_path: str) -> torch.Tensor:
        with rasterio.open(image_path) as src:
            arr = src.read()  # (C,H,W)
        arr = arr.astype(np.float32)

        # keep the behavior you had: assume 0..255
        if self.scale_mode == "uint8_255":
            arr = arr / 255.0
        elif self.scale_mode == "uint16_65535":
            arr = arr / 65535.0
        elif self.scale_mode == "none":
            pass
        else:
            raise ValueError("scale_mode must be one of {'uint8_255','uint16_65535','none'}")

        return torch.from_numpy(arr)

    def _read_yolo_labels(self, label_path: str, width: int, height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes: List[List[float]] = []
        labels: List[int] = []

        if not os.path.exists(label_path):
            return (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.int64),
            )

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    class_id, x_center, y_center, w, h = map(float, parts)
                except ValueError:
                    continue

                xmin = (x_center - w / 2) * width
                ymin = (y_center - h / 2) * height
                xmax = (x_center + w / 2) * width
                ymax = (y_center + h / 2) * height

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id) + self.label_offset)

        if len(boxes) == 0:
            return (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.int64),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.tif")
        label_path = os.path.join(self.label_dir, f"{image_id}.txt")

        image = self._read_image(image_path)
        _, height, width = image.shape

        boxes, labels = self._read_yolo_labels(label_path, width=width, height=height)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
