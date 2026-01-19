"""
tree_delineation.py

Library-only functions for watershed-based tree delineation.

Supported inputs
- Single-band height raster (meters) + canopy mask
- Multi-band workflows where you provide:
    - height_m (meters)
    - optional ndvi (float)
    - canopy mask (bool)

This module:
- builds masks
- detects peaks
- runs watershed
- returns GeoDataFrame polygons
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, median_filter
from shapely.geometry import shape as shp_shape
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


@dataclass(frozen=True)
class MaskParams:
    ndvi_thr: float = 0.2
    height_thr_m: float = 2.0
    min_canopy_area_px: int = 10


@dataclass(frozen=True)
class PeakParams:
    min_distance_px: int = 2
    threshold_abs_m: Optional[float] = None


@dataclass(frozen=True)
class SmoothParams:
    fltr: str = "gaussian"  # "gaussian", "median", "none"
    gaussian_sigma: float = 1.5
    median_size: int = 3


@dataclass(frozen=True)
class SegmentParams:
    use_gradient: bool = False
    fill_holes: bool = True


def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)."""
    return (nir - red) / (nir + red + eps)


def smooth_surface(height_m: np.ndarray, smooth: SmoothParams) -> np.ndarray:
    fltr = smooth.fltr.lower()
    if fltr == "none":
        return height_m
    if fltr == "median":
        return median_filter(height_m, size=int(smooth.median_size))
    return gaussian_filter(height_m, sigma=float(smooth.gaussian_sigma))


def clean_binary_mask(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove small connected components from a boolean mask."""
    mask = mask.astype(bool)
    labeled = label(mask)
    if min_area_px and min_area_px > 1:
        labeled = remove_small_objects(labeled, min_size=int(min_area_px))
    return labeled > 0


def canopy_mask_from_height(height_m: np.ndarray, mask_params: MaskParams) -> np.ndarray:
    """
    Canopy mask from a single-band height raster in meters.
    mask = height >= height_thr_m (or height > 0 if height_thr_m <= 0)
    """
    thr = float(mask_params.height_thr_m)
    base = (height_m > 0) if thr <= 0 else (height_m >= thr)
    return clean_binary_mask(base, mask_params.min_canopy_area_px)


def masks_from_ndvi_and_height(
    ndvi: np.ndarray,
    height_m: np.ndarray,
    mask_params: MaskParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    veg_mask    = ndvi >= ndvi_thr
    height_mask = height_m >= height_thr_m
    canopy_mask = clean(veg & height)
    """
    veg = ndvi >= float(mask_params.ndvi_thr)
    tall = height_m >= float(mask_params.height_thr_m)
    canopy = clean_binary_mask(veg & tall, mask_params.min_canopy_area_px)
    return veg.astype(bool), tall.astype(bool), canopy.astype(bool)


def markers_from_height_peaks(
    height_m: np.ndarray,
    canopy_mask: np.ndarray,
    peak_params: PeakParams,
) -> np.ndarray:
    canopy_mask = canopy_mask.astype(bool)
    h = np.where(canopy_mask, height_m, 0.0)

    coords = peak_local_max(
        h,
        min_distance=int(peak_params.min_distance_px),
        threshold_abs=peak_params.threshold_abs_m,
        labels=canopy_mask.astype(np.uint8),
        exclude_border=False,
    )

    markers = np.zeros_like(height_m, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    return markers


def watershed_segment(
    height_m: np.ndarray,
    canopy_mask: np.ndarray,
    markers: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    *,
    smooth: SmoothParams,
    seg: SegmentParams,
) -> gpd.GeoDataFrame:
    """
    Watershed segmentation returning polygons as GeoDataFrame with:
      - label
      - geometry
      - area
    """
    canopy_mask = canopy_mask.astype(bool)

    if canopy_mask.sum() < 10 or markers.max() == 0:
        return gpd.GeoDataFrame(columns=["label", "geometry", "area"], geometry="geometry", crs=crs)

    h_smoothed = smooth_surface(height_m, smooth)

    if seg.use_gradient:
        grad = sobel(h_smoothed)
        labels_ws = watershed(grad, markers, mask=canopy_mask)
    else:
        dist = distance_transform_edt(canopy_mask)
        labels_ws = watershed(-dist, markers, mask=canopy_mask)

    if seg.fill_holes:
        filled = np.zeros_like(labels_ws, dtype=np.int32)
        max_label = int(labels_ws.max())
        for i in range(1, max_label + 1):
            filled[binary_fill_holes(labels_ws == i)] = i
        labels_ws = filled

    polys = []
    vals = []
    for geom, val in shapes(labels_ws.astype(np.int32), mask=labels_ws > 0, transform=transform):
        polys.append(shp_shape(geom))
        vals.append(int(val))

    gdf = gpd.GeoDataFrame({"label": vals, "geometry": polys}, crs=crs)
    gdf["area"] = gdf.geometry.area if not gdf.empty else []
    return gdf
