#!/usr/bin/env python3
"""
compute_channel_stats.py

Compute per-channel mean/std for TIFF training images and save to JSON.

Expected dataset layout:
IMAGES_DIR/
  train/
    Acer/*.tif
    Quercus/*.tif
    ...
  val/
    ...

Reads only the train split.

Supports:
- GeoTIFF via rasterio
- plain TIFF via tifffile
- HWC or CHW layout
- 3 / 4 / 5 channel images
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import tifffile


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute channel mean/std from training TIFFs")
    ap.add_argument("--images-dir", required=True, help="Dataset root containing train/ and val/")
    ap.add_argument("--in-channels", type=int, choices=[3, 4, 5], default=5)
    ap.add_argument("--out-json", default=None, help="Output JSON path (default: <images-dir>/channel_stats.json)")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap on number of training images (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--percentile-normalize",
        action="store_true",
        help="Apply per-image 2/98 percentile normalization before stats (usually leave OFF)",
    )
    return ap.parse_args()


def read_any_tiff(path: str | Path) -> np.ndarray:
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

    # CHW -> HWC if needed
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
        raise ValueError(f"Image has {arr.shape[-1]} channels but expected {in_channels}")

    if arr.shape[-1] > in_channels:
        arr = arr[..., :in_channels]

    return arr


def scale_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    if np.nanmax(arr) > 1.5:
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


def collect_train_files(images_dir: Path) -> list[Path]:
    train_dir = images_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train dir: {train_dir}")

    files = []
    for ext in ("*.tif", "*.tiff"):
        files.extend(train_dir.rglob(ext))

    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No TIFF files found under {train_dir}")
    return files


def compute_stats(
    files: list[Path],
    in_channels: int,
    use_percentile_normalize: bool,
) -> Tuple[list[float], list[float]]:
    sum_c = np.zeros(in_channels, dtype=np.float64)
    sumsq_c = np.zeros(in_channels, dtype=np.float64)
    count_c = np.zeros(in_channels, dtype=np.float64)

    for fp in files:
        arr = read_any_tiff(fp)
        arr = ensure_hwc_channels(arr, in_channels)
        arr = scale_image(arr)

        if use_percentile_normalize:
            arr = percentile_normalize_hwc(arr)

        flat = arr.reshape(-1, in_channels)
        sum_c += flat.sum(axis=0)
        sumsq_c += (flat ** 2).sum(axis=0)
        count_c += flat.shape[0]

    mean = sum_c / np.maximum(count_c, 1)
    var = (sumsq_c / np.maximum(count_c, 1)) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-12))

    return mean.tolist(), std.tolist()


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    out_json = Path(args.out_json) if args.out_json else images_dir / "channel_stats.json"

    files = collect_train_files(images_dir)

    if args.max_samples > 0 and len(files) > args.max_samples:
        rng = random.Random(args.seed)
        files = rng.sample(files, args.max_samples)

    mean, std = compute_stats(
        files=files,
        in_channels=args.in_channels,
        use_percentile_normalize=args.percentile_normalize,
    )

    stats = {
        "mean": mean,
        "std": std,
        "in_channels": args.in_channels,
        "num_images": len(files),
        "source": str(images_dir / "train"),
        "percentile_normalize": bool(args.percentile_normalize),
    }

    out_json.write_text(json.dumps(stats, indent=2))
    print(f"Saved stats to: {out_json}")
    print(f"mean = {mean}")
    print(f"std  = {std}")


if __name__ == "__main__":
    main()