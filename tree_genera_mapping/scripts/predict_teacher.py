#!/usr/bin/env python3
"""Generate genus pseudo-labels with a Faster R-CNN/ResNet teacher ensemble.

The detector proposes crown bounding boxes over overlapping windows of each
five-channel RGBIH parent tile. Global non-maximum suppression removes window
duplicates, then the classifier assigns one of the configured tree genera to
each retained crown. Results are written as one georeferenced GeoPackage per
input tile and include ``top1_class``, which ``build_dataset.py`` accepts as a
detection label column.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from shapely.geometry import box
from torchvision.ops import nms
from tqdm import tqdm

from tree_genera_mapping.dl.classification.genus_model import build_resnet_classifier
from tree_genera_mapping.dl.detection.tree_model import build_detector


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate genus pseudo-labels using Faster R-CNN and ResNet"
    )
    parser.add_argument("--tile-dir", required=True, type=Path,
                        help="Directory containing rgbih_*.tif parent tiles")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for per-tile pseudo-label GeoPackages")
    parser.add_argument(
        "--ckpt-paths",
        required=True,
        nargs=2,
        type=Path,
        metavar=("DETECTOR", "CLASSIFIER"),
        help="Faster R-CNN checkpoint followed by ResNet checkpoint",
    )
    parser.add_argument("--tile-id", default=None,
                        help="Optional tile ID; omit to process all RGBIH tiles")
    parser.add_argument("--patch-size", type=int, default=640,
                        help="Detector sliding-window size in pixels")
    parser.add_argument("--image-patch-size", type=int, default=None,
                        help="Classifier crop size; default comes from its checkpoint")
    parser.add_argument("--stride", type=int, default=512,
                        help="Detector sliding-window stride in pixels")
    parser.add_argument("--det-conf", "--conf", dest="det_conf", type=float, default=0.30,
                        help="Minimum detector confidence")
    parser.add_argument("--cls-conf", type=float, default=0.30,
                        help="Minimum top-one classifier probability")
    parser.add_argument("--iou", type=float, default=0.50,
                        help="IoU threshold for global box NMS")
    parser.add_argument("--class-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_float_list(value) -> Optional[list[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _checkpoint_state(checkpoint: dict) -> dict:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_detector(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    in_channels = int(checkpoint_args.get("in_channels", 5))
    model = build_detector(
        detector="fasterrcnn",
        num_classes=int(checkpoint_args.get("num_classes", 2)),
        in_channels=in_channels,
        backbone=checkpoint_args.get("backbone", "resnet50"),
        norm_mean=_parse_float_list(checkpoint_args.get("norm_mean")),
        norm_std=_parse_float_list(checkpoint_args.get("norm_std")),
        pretrained_backbone=False,
    )
    model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
    model.to(device).eval()
    return model, in_channels


def load_classifier(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{path} does not contain the expected classifier key 'model'")

    checkpoint_args = checkpoint.get("args", {})
    experiment = checkpoint_args.get("experiment", "image_only")
    if experiment != "image_only":
        raise ValueError(
            f"Only image_only classifier checkpoints are supported, got {experiment!r}"
        )

    raw_classes = checkpoint.get("classes")
    if not raw_classes:
        raise ValueError(f"{path} does not contain a class mapping")
    classes = {
        int(class_id): str(name).replace("_", " ")
        for class_id, name in raw_classes.items()
    }

    in_channels = int(checkpoint_args.get("in_channels", 5))
    image_size = int(checkpoint_args.get("img_size", 128))
    model = build_resnet_classifier(
        model_name=checkpoint_args.get("backbone", "resnet50"),
        num_classes=len(classes),
        in_channels=in_channels,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    mean = torch.tensor(
        checkpoint.get("norm_mean", [0.5] * in_channels),
        dtype=torch.float32,
        device=device,
    ).view(1, in_channels, 1, 1)
    std = torch.tensor(
        checkpoint.get("norm_std", [0.25] * in_channels),
        dtype=torch.float32,
        device=device,
    ).view(1, in_channels, 1, 1)
    return model, classes, in_channels, image_size, mean, std


def window_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.inference_mode()
def detect_crowns(
    model,
    image: torch.Tensor,
    patch_size: int,
    stride: int,
    confidence: float,
    iou_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, height, width = image.shape
    boxes_all: list[torch.Tensor] = []
    scores_all: list[torch.Tensor] = []

    for y in window_starts(height, patch_size, stride):
        for x in window_starts(width, patch_size, stride):
            patch = image[
                :,
                y:min(y + patch_size, height),
                x:min(x + patch_size, width),
            ].to(device)
            prediction = model([patch])[0]
            selected = prediction["scores"] >= confidence
            boxes = prediction["boxes"][selected].detach().cpu()
            scores = prediction["scores"][selected].detach().cpu()
            if boxes.numel() == 0:
                continue
            boxes[:, [0, 2]] += x
            boxes[:, [1, 3]] += y
            boxes_all.append(boxes)
            scores_all.append(scores)

    if not boxes_all:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty(0)

    boxes = torch.cat(boxes_all)
    scores = torch.cat(scores_all)
    selected = nms(boxes, scores, iou_threshold)
    return boxes[selected], scores[selected]


@torch.inference_mode()
def classify_crowns(
    classifier,
    image: torch.Tensor,
    boxes_px: torch.Tensor,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor]:
    crops: list[torch.Tensor] = []
    valid_indices: list[int] = []
    _, height, width = image.shape

    for index, bbox in enumerate(boxes_px):
        x1, y1, x2, y2 = bbox.tolist()
        x1 = max(0, int(np.floor(x1)))
        y1 = max(0, int(np.floor(y1)))
        x2 = min(width, int(np.ceil(x2)))
        y2 = min(height, int(np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[:, y1:y2, x1:x2].unsqueeze(0)
        crop = F.interpolate(
            crop,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        crops.append(crop.squeeze(0))
        valid_indices.append(index)

    if not crops:
        return [], torch.empty((0, mean.shape[1]))

    probabilities: list[torch.Tensor] = []
    for start in range(0, len(crops), batch_size):
        batch = torch.stack(crops[start:start + batch_size]).to(device)
        logits = classifier((batch - mean) / std)
        probabilities.append(torch.softmax(logits, dim=1).cpu())
    return valid_indices, torch.cat(probabilities)


def pixel_box_to_geometry(transform, bbox: torch.Tensor):
    x1, y1, x2, y2 = bbox.tolist()
    left, top = rasterio.transform.xy(transform, y1, x1, offset="ul")
    right, bottom = rasterio.transform.xy(transform, y2, x2, offset="ul")
    return box(
        min(left, right),
        min(bottom, top),
        max(left, right),
        max(bottom, top),
    )


def process_tile(
    tile_path: Path,
    output_path: Path,
    detector,
    classifier,
    classes: dict[int, str],
    classifier_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    with rasterio.open(tile_path) as source:
        array = source.read().astype(np.float32) / 255.0
        transform = source.transform
        crs = source.crs

    if array.shape[0] != 5:
        raise ValueError(f"{tile_path} has {array.shape[0]} channels; expected 5")
    if crs is None:
        raise ValueError(f"{tile_path} has no CRS")
    image = torch.from_numpy(array)

    boxes_px, detector_scores = detect_crowns(
        model=detector,
        image=image,
        patch_size=args.patch_size,
        stride=args.stride,
        confidence=args.det_conf,
        iou_threshold=args.iou,
        device=device,
    )
    valid_indices, probabilities = classify_crowns(
        classifier=classifier,
        image=image,
        boxes_px=boxes_px,
        image_size=classifier_size,
        mean=mean,
        std=std,
        batch_size=args.class_batch_size,
        device=device,
    )

    records = []
    for probability_index, box_index in enumerate(valid_indices):
        class_probability, class_id_tensor = probabilities[probability_index].max(dim=0)
        class_probability = float(class_probability)
        class_id = int(class_id_tensor)
        if class_probability < args.cls_conf:
            continue

        record = {
            "top1_class": classes[class_id],
            "top1_class_id": class_id,
            "top1_score": class_probability,
            "detector_score": float(detector_scores[box_index]),
            "source_file": tile_path.name,
            "geometry": pixel_box_to_geometry(transform, boxes_px[box_index]),
        }
        for cid, class_name in classes.items():
            record[class_name.replace(" ", "_")] = float(
                probabilities[probability_index, cid]
            )
        records.append(record)

    if not records:
        LOGGER.warning("No accepted predictions for %s", tile_path.name)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    result.to_file(output_path, layer="pseudo_labels", driver="GPKG")
    LOGGER.info("Saved %d predictions to %s", len(result), output_path)


def normalize_tile_id(tile_id: str) -> str:
    value = tile_id.strip().replace("-", "_")
    parts = value.split("_")
    if len(parts) == 2:
        return f"32_{parts[0]}_{parts[1]}"
    if len(parts) == 3:
        return value
    raise ValueError(f"Unsupported tile ID: {tile_id}")


def select_tiles(tile_dir: Path, tile_id: Optional[str]) -> Sequence[Path]:
    if tile_id is None:
        tiles = sorted(tile_dir.glob("rgbih_*.tif"))
    else:
        normalized = normalize_tile_id(tile_id)
        candidate = tile_dir / f"rgbih_{normalized}.tif"
        tiles = [candidate] if candidate.exists() else []
    if not tiles:
        raise FileNotFoundError(f"No matching RGBIH TIFFs found in {tile_dir}")
    return tiles


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("--patch-size and --stride must be positive")
    if not (0.0 <= args.det_conf <= 1.0 and 0.0 <= args.cls_conf <= 1.0):
        raise ValueError("Confidence thresholds must be between 0 and 1")
    if not (0.0 <= args.iou <= 1.0):
        raise ValueError("--iou must be between 0 and 1")

    device = torch.device(args.device)
    detector_path, classifier_path = args.ckpt_paths
    detector, detector_channels = load_detector(detector_path, device)
    classifier, classes, classifier_channels, checkpoint_size, mean, std = load_classifier(
        classifier_path, device
    )
    if detector_channels != 5 or classifier_channels != 5:
        raise ValueError(
            "Teacher inference requires five-channel detector and classifier checkpoints"
        )

    classifier_size = args.image_patch_size or checkpoint_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tile_path in tqdm(select_tiles(args.tile_dir, args.tile_id), desc="Teacher inference"):
        output_path = args.output_dir / f"teacher_{tile_path.stem}.gpkg"
        if output_path.exists() and not args.overwrite:
            LOGGER.info("Skipping existing %s", output_path)
            continue
        process_tile(
            tile_path=tile_path,
            output_path=output_path,
            detector=detector,
            classifier=classifier,
            classes=classes,
            classifier_size=classifier_size,
            mean=mean,
            std=std,
            args=args,
            device=device,
        )


if __name__ == "__main__":
    main()
