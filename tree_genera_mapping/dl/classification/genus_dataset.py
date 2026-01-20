"""
genus_dataset.py
Dataset + raster IO + augmentations for genus classification.

Contains:
- read_multiband_tiff(): reads TIFF into CHW float32
- transforms (resize/augment/normalize)
- GenusImageDataset: supports 3/4/5-channel training
- autodetect_tabular_cols(): helper for NDVI/tabular features (optional)

"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tifffile import imread, TiffFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop


# -----------------------------
# Channel stats (training-set)
# -----------------------------
# If you train on 3/4 channels, slice these in NormalizeChannels.
CHANNEL_MEAN_5 = torch.tensor([0.3603, 0.3813, 0.3470, 0.6094, 0.1883], dtype=torch.float32)
CHANNEL_STD_5 = torch.tensor([0.1393, 0.1402, 0.1255, 0.1900, 0.0853], dtype=torch.float32)


# -----------------------------
# IO / scaling helpers
# -----------------------------
def _to_chw(arr: np.ndarray, path: str) -> np.ndarray:
    """Convert common TIFF layouts to CHW float32 numpy."""
    if arr.ndim == 2:
        # multipage TIFF fallback: arr came back 2D (rare)
        with TiffFile(path) as tf:
            pages = [p.asarray() for p in tf.pages]
        arr = np.stack(pages, axis=0)  # CHW

    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 3D array, got shape={arr.shape}")

    # CHW
    if arr.shape[0] in (1, 3, 4, 5) and arr.shape[0] < arr.shape[-1]:
        return arr

    # HWC
    if arr.shape[-1] in (1, 3, 4, 5) and arr.shape[-1] < arr.shape[0]:
        return np.transpose(arr, (2, 0, 1))

    # Otherwise ambiguous; assume CHW
    return arr


def _scale_to_0_1(chw: np.ndarray) -> np.ndarray:
    """
    Deterministic scaling to [0,1] based on dtype/range.
    - uint8   -> /255
    - uint16  -> /65535
    - float   -> assumes already 0..1 or arbitrary, clamps to [0,1]
    """
    if chw.dtype == np.uint8:
        x = chw.astype(np.float32) / 255.0
    elif chw.dtype == np.uint16:
        x = chw.astype(np.float32) / 65535.0
    else:
        x = chw.astype(np.float32)

        # If values look like 0..255 or 0..65535 but are float, normalize heuristically:
        vmax = float(np.nanmax(x)) if np.isfinite(x).any() else 1.0
        if vmax > 300.0:
            x = x / 65535.0
        elif vmax > 2.0:
            x = x / 255.0

    return np.clip(x, 0.0, 1.0)


def _percentile_normalize_per_band(chw01: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """
    Optional: per-image, per-band robust normalization to [0,1] using percentiles.
    Use only if your TIFFs have inconsistent ranges and deterministic scaling is not enough.
    """
    out = np.zeros_like(chw01, dtype=np.float32)
    for c in range(chw01.shape[0]):
        b = chw01[c]
        if not np.isfinite(b).any():
            out[c] = 0.0
            continue
        lo = np.nanpercentile(b, p_lo)
        hi = np.nanpercentile(b, p_hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(b))
            hi = float(np.nanmax(b)) if float(np.nanmax(b)) > lo else lo + 1e-6
        out[c] = np.clip((b - lo) / (hi - lo), 0.0, 1.0)
    return out


def read_multiband_tiff(
    path: str,
    out_size: int,
    in_channels: int = 5,
    band_indices: Optional[Sequence[int]] = None,
    percentile_normalize: bool = False,
) -> np.ndarray:
    """
    Reads TIFF and returns float32 array [C,H,W] in [0,1].

    Args:
        path: TIFF path
        out_size: output spatial size (H=W=out_size)
        in_channels: number of channels to return (3/4/5)
        band_indices: optional explicit band indices (0-based) to select from the source bands.
                     If None, takes first `in_channels`.
        percentile_normalize: if True, apply per-image percentile normalization per band.

    Returns:
        np.ndarray [C,out_size,out_size] float32 in [0,1]
    """
    if in_channels not in (3, 4, 5):
        raise ValueError("in_channels must be 3, 4, or 5")

    arr = imread(path)
    chw = _to_chw(arr, path)

    # Select bands
    if band_indices is None:
        if chw.shape[0] < in_channels:
            raise ValueError(f"{path}: has {chw.shape[0]} bands, need {in_channels}")
        chw = chw[:in_channels]
    else:
        band_indices = list(band_indices)
        if len(band_indices) != in_channels:
            raise ValueError("band_indices length must match in_channels")
        if max(band_indices) >= chw.shape[0]:
            raise ValueError(f"{path}: band_indices out of range for {chw.shape[0]} bands")
        chw = chw[band_indices]

    # Resize each band
    H = W = int(out_size)
    chw = chw.astype(chw.dtype, copy=False)
    bands = [cv2.resize(chw[k], (W, H), interpolation=cv2.INTER_AREA) for k in range(in_channels)]
    chw = np.stack(bands, axis=0)

    # Scale to [0,1] consistently
    chw01 = _scale_to_0_1(chw)

    # Optional per-image percentile normalization
    if percentile_normalize:
        chw01 = _percentile_normalize_per_band(chw01, p_lo=2.0, p_hi=98.0)

    return chw01.astype(np.float32, copy=False)


# -----------------------------
# Transforms
# -----------------------------
class NormalizeChannels(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels not in (3, 4, 5):
            raise ValueError("in_channels must be 3, 4, or 5")
        self.in_channels = int(in_channels)

        mean = CHANNEL_MEAN_5[: self.in_channels].view(self.in_channels, 1, 1)
        std = CHANNEL_STD_5[: self.in_channels].view(self.in_channels, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ResizeCH(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = int(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return x.squeeze(0)


class GeometricAugmentCH(nn.Module):
    def __init__(
        self,
        size: int,
        crop: bool = True,
        hflip: bool = True,
        vflip: bool = True,
        rotation: bool = True,
    ):
        super().__init__()
        self.size = int(size)
        self.crop = crop
        self.hflip = hflip
        self.vflip = vflip
        self.rotation = rotation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape

        if self.crop and random.random() < 0.5:
            crop_size = int(min(h, w) * 0.8)
            i, j, th, tw = RandomCrop.get_params(x, output_size=(crop_size, crop_size))
            x = TF.resized_crop(
                x,
                i, j, th, tw,
                size=(self.size, self.size),
                interpolation=TF.InterpolationMode.BILINEAR,
            )

        if self.hflip and random.random() < 0.5:
            x = TF.hflip(x)
        if self.vflip and random.random() < 0.5:
            x = TF.vflip(x)

        if self.rotation and random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            x = TF.rotate(x, angle)

        return x


class PhotometricAugmentCH(nn.Module):
    """
    Photometric augmentation.
    If channels >= 4: assume first 4 are RGBN (or RGB+NIR-like) and last channel is Height when 5ch.
    If channels == 3: only RGB noise/brightness.
    """
    def __init__(self, in_channels: int, brightness=(0.8, 1.2), noise_rgb=0.05, noise_h=0.02):
        super().__init__()
        self.in_channels = int(in_channels)
        self.brightness = brightness
        self.noise_rgb = float(noise_rgb)
        self.noise_h = float(noise_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 3:
            rgb = x[:3]
            if random.random() < 0.5:
                factor = random.uniform(*self.brightness)
                rgb = torch.clamp(rgb * factor, 0, 1)
            if random.random() < 0.5:
                rgb = torch.clamp(rgb + torch.randn_like(rgb) * self.noise_rgb, 0, 1)
            return rgb

        # 4 or 5 channels
        rgbn = x[:4]
        rest = x[4:]  # empty for 4ch; height for 5ch

        if random.random() < 0.5:
            factor = random.uniform(*self.brightness)
            rgbn = torch.clamp(rgbn * factor, 0, 1)

        if random.random() < 0.5:
            rgbn = torch.clamp(rgbn + torch.randn_like(rgbn) * self.noise_rgb, 0, 1)

        if self.in_channels == 5 and rest.numel() > 0 and random.random() < 0.3:
            rest = torch.clamp(rest + torch.randn_like(rest) * self.noise_h, 0, 1)

        return torch.cat([rgbn, rest], dim=0)


# -----------------------------
# Dataset
# -----------------------------
class GenusImageDataset(Dataset):
    """
    DataFrame must contain:
      - image_path (str)
      - class_id (int) OR class_name (str) + class_to_id mapping

    Returns:
      (x, y) where x is [C,H,W], y is int
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        img_size: int,
        augment: bool,
        in_channels: int = 5,
        class_to_id: Optional[dict] = None,
        band_indices: Optional[Sequence[int]] = None,
        percentile_normalize: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.in_channels = int(in_channels)
        self.band_indices = list(band_indices) if band_indices is not None else None
        self.percentile_normalize = bool(percentile_normalize)

        if "class_id" not in self.df.columns:
            if "class_name" in self.df.columns and class_to_id is not None:
                self.df["class_id"] = self.df["class_name"].map(class_to_id).astype(int)
            else:
                raise ValueError("Need class_id OR (class_name + class_to_id).")

        self.resize = ResizeCH(self.img_size)
        self.geo = GeometricAugmentCH(self.img_size)
        self.photo = PhotometricAugmentCH(in_channels=self.in_channels)
        self.norm = NormalizeChannels(in_channels=self.in_channels)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]

        x = read_multiband_tiff(
            r["image_path"],
            out_size=self.img_size,
            in_channels=self.in_channels,
            band_indices=self.band_indices,
            percentile_normalize=self.percentile_normalize,
        )
        x = torch.from_numpy(x).float()

        # safety resize (read already resizes, but ok)
        x = self.resize(x)

        if self.augment:
            x = self.geo(x)
            x = self.photo(x)

        x = self.norm(x)

        y = int(r["class_id"])
        return x, y


# -----------------------------
# Optional helper (tabular)
# -----------------------------
def autodetect_tabular_cols(ndvi_df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    lower = {c: c.lower() for c in ndvi_df.columns}

    for c in ndvi_df.columns:
        lc = lower[c]
        if lc in ("canopywidt", "canopywidth"):
            cols.append(c)
        if lc in ("canopyheigt", "canopyheight", "canopyheig"):
            cols.append(c)

    for c in ndvi_df.columns:
        if re.fullmatch(r"m(0[1-9]|1[0-2])", c.lower()):
            cols.append(c)

    canopy = [c for c in cols if c.lower().startswith("canopy")]
    months = sorted([c for c in cols if re.fullmatch(r"m(0[1-9]|1[0-2])", c.lower())], key=str.lower)
    return canopy + months
