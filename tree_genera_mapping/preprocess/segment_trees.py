#!/usr/bin/env python3
"""
segment_trees.py

Watershed-based tree delineation from:
- nDOM rasters (single-band, meters), or
- RGBIH rasters (RGB+NIR+Height(uint8 normalized)), using JSON stats to unnormalize height to meters.

Inputs (expected patterns)
- mode=ndom:
    img_dir contains: ndom1_*.tif  (single-band HeightModel/nDOM, meters)
- mode=rgbih:
    img_dir contains: rgbih_*.tif and rgbih_*.json
    where JSON provides:
      {
        "height_channel": {
          "stats_m": [vmin_m, vmax_m],
          "band_index_1based": 5
        }
      }

Outputs per tile (written to output_dir)
- trees_segmented_<tile_id>.gpkg         (tree crown polygons)
- optional: trees_bbox_<tile_id>.gpkg    (bbox polygons)
- optional masks:
    mask_ndvi_<tile_id>.tif
    mask_height_<tile_id>.tif
    mask_canopy_<tile_id>.tif

Mask logic
- If mode=rgbih:
    NDVI = (NIR - Red) / (NIR + Red)
    veg mask  = NDVI >= ndvi_thr
    height mask = height_m >= height_thr_m
    canopy mask = veg & height
- If mode=ndom:
    canopy mask = ndom_m >= height_thr_m   (or >0 if height_thr_m<=0)

Watershed
- Markers from peak_local_max over height surface within canopy mask.
- Segmentation either:
    distance-based (default): watershed(-distance_transform(mask), markers)
    gradient-based (optional): watershed(sobel(height_smoothed), markers)

Notes
- Assumes CRS is present in rasters. For external peaks input, use --peaks-crs if missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import re, tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union, List

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling

from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, median_filter
from shapely.geometry import shape as shp_shape
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ------------------------- dataclasses -------------------------

@dataclass(frozen=True)
class BandMap:
    """1-indexed bands for RGBIH rasters."""
    r: int = 1
    g: int = 2
    b: int = 3
    nir: int = 4
    h: int = 5


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
    fltr: str = "gaussian"  # gaussian|median|none
    gaussian_sigma: float = 1.5
    median_size: int = 3


@dataclass(frozen=True)
class SegmentParams:
    use_gradient: bool = False
    fill_holes: bool = True


# ------------------------- misc helpers -------------------------

def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (nir - red) / (nir + red + eps)


def smooth_height(height: np.ndarray, smooth: SmoothParams) -> np.ndarray:
    fltr = smooth.fltr.lower()
    if fltr == "none":
        return height
    if fltr == "median":
        return median_filter(height, size=int(smooth.median_size))
    return gaussian_filter(height, sigma=float(smooth.gaussian_sigma))


def write_mask_geotiff(
    out_path: Path,
    mask: np.ndarray,
    ref_profile: dict,
    transform,
    crs,
    encoding: str = "01",  # "01" or "0255"
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if encoding not in ("01", "0255"):
        raise ValueError("encoding must be '01' or '0255'")

    arr = mask.astype(np.uint8)
    if encoding == "0255":
        arr = arr * 255

    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        predictor=2,
        transform=transform,
        crs=crs,
        height=arr.shape[0],
        width=arr.shape[1],
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


def markers_from_height_peaks(
    height_m: np.ndarray,
    mask: np.ndarray,
    peak_params: PeakParams,
) -> np.ndarray:
    mask = mask.astype(bool)
    h = np.where(mask, height_m, 0.0)

    coords = peak_local_max(
        h,
        min_distance=int(peak_params.min_distance_px),
        threshold_abs=peak_params.threshold_abs_m,
        labels=mask.astype(np.uint8),
        exclude_border=False,
    )

    markers = np.zeros_like(height_m, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    return markers


def watershed_segment(
    height_m: np.ndarray,
    mask: np.ndarray,
    markers: np.ndarray,
    transform,
    crs: str,
    smooth: SmoothParams,
    seg: SegmentParams,
) -> gpd.GeoDataFrame:
    if mask.sum() < 10:
        return gpd.GeoDataFrame(columns=["label", "geometry", "area"], geometry="geometry", crs=crs)

    h_smoothed = smooth_height(height_m, smooth)

    if seg.use_gradient:
        grad = sobel(h_smoothed)
        labels_ws = watershed(grad, markers, mask=mask)
    else:
        dist = distance_transform_edt(mask)
        labels_ws = watershed(-dist, markers, mask=mask)

    if seg.fill_holes:
        filled = np.zeros_like(labels_ws, dtype=np.int32)
        max_label = int(labels_ws.max())
        for i in range(1, max_label + 1):
            filled[binary_fill_holes(labels_ws == i)] = i
        labels_ws = filled

    polys: List[object] = []
    vals: List[int] = []
    for geom, val in shapes(labels_ws.astype(np.int32), mask=labels_ws > 0, transform=transform):
        polys.append(shp_shape(geom))
        vals.append(int(val))

    gdf = gpd.GeoDataFrame({"label": vals, "geometry": polys}, crs=crs)
    if not gdf.empty:
        gdf["area"] = gdf.geometry.area
    else:
        gdf["area"] = []
    return gdf


def make_canopy_mask_from_ndom(ndom_m: np.ndarray, mask_params: MaskParams) -> np.ndarray:
    thr = float(mask_params.height_thr_m)
    if thr <= 0:
        base = ndom_m > 0
    else:
        base = ndom_m >= thr

    labeled = label(base)
    if mask_params.min_canopy_area_px and mask_params.min_canopy_area_px > 1:
        labeled = remove_small_objects(labeled, min_size=int(mask_params.min_canopy_area_px))
    return labeled > 0


def make_canopy_mask_from_rgbih(ndvi: np.ndarray, height_m: np.ndarray, mask_params: MaskParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    veg = ndvi >= float(mask_params.ndvi_thr)
    tall = height_m >= float(mask_params.height_thr_m)
    canopy = veg & tall

    labeled = label(canopy)
    if mask_params.min_canopy_area_px and mask_params.min_canopy_area_px > 1:
        labeled = remove_small_objects(labeled, min_size=int(mask_params.min_canopy_area_px))
    canopy = labeled > 0

    return veg, tall, canopy


def parse_tile_id_from_name(path: Path, mode: str) -> str:
    """
    Extract tile_id from file name.
    - rgbih_32_355_6048.tif -> 32_355_6048
    - ndom_32_355_6048_100.tif -> 32_355_6048_100 (or whatever stem has)
    We keep it simple: remove prefix and return remainder of stem.
    """
    stem = path.stem
    mode = mode.lower()

    if mode == "rgbih" and stem.startswith("rgbih_"):
        return stem.replace("rgbih_", "", 1)
    if mode == "ndom" and stem.startswith("ndom1_"):
        return stem.replace("ndom1_", "", 1)
    return stem


def load_rgbih_and_height_m(
    tif_path: Path,
    json_path: Path,
    band_map: BandMap,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, rasterio.Affine, str, dict]:
    """
    Loads RGBIH tile.
    Returns r,g,b,nir,height_m, transform, crs_string, profile
    """
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    hc = meta.get("height_channel", {})
    stats = hc.get("stats_m", None)
    h_band = int(hc.get("band_index_1based", band_map.h))

    if (not isinstance(stats, list)) or len(stats) != 2:
        raise ValueError(f"Invalid stats_m in JSON: {json_path}")

    vmin, vmax = float(stats[0]), float(stats[1])

    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        crs = src.crs.to_string() if src.crs else None
        if crs is None:
            raise ValueError(f"Raster has no CRS: {tif_path}")

        r = src.read(band_map.r).astype(np.float32)
        g = src.read(band_map.g).astype(np.float32)
        b = src.read(band_map.b).astype(np.float32)
        nir = src.read(band_map.nir).astype(np.float32)
        h_u8 = src.read(h_band).astype(np.float32)  # uint8 stored, but read float for math

        transform = src.transform

    # Unnormalize: height_u8 -> meters
    if np.isclose(vmax, vmin):
        height_m = np.zeros_like(h_u8, dtype=np.float32)
    else:
        height_m = (h_u8 / 255.0) * (vmax - vmin) + vmin
        height_m = height_m.astype(np.float32, copy=False)

    return r, g, b, nir, height_m, transform, crs, profile


def load_ndom_m(ndom_path: Path, band: int = 1) -> Tuple[np.ndarray, rasterio.Affine, str, dict]:
    with rasterio.open(ndom_path) as src:
        arr = src.read(band).astype(np.float32)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else None
        profile = src.profile.copy()
    if crs is None:
        raise ValueError(f"Raster has no CRS: {ndom_path}")
    return arr, transform, crs, profile


# ------------------------- batch runner -------------------------

def run_segmentation_batch(
    img_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    mode: str,
    # rgbih only
    band_map: BandMap,
    # options
    mask_params: MaskParams,
    peak_params: PeakParams,
    smooth: SmoothParams,
    seg: SegmentParams,
    # outputs
    write_bbox: bool,
    write_masks: bool,
    mask_encoding: str,  # "01" or "0255"
) -> None:
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = mode.lower()
    if mode not in ("ndom", "rgbih"):
        raise ValueError("--mode must be 'ndom' or 'rgbih'")

    if mode == "rgbih":
        tifs = sorted(img_dir.glob("rgbih_*.tif"))
        if not tifs:
            raise ValueError(f"No rgbih_*.tif found in {img_dir}")
    else:
        tifs = sorted(img_dir.glob("ndom1_*.tif"))
        if not tifs:
            raise ValueError(f"No ndom1_*.tif found in {img_dir}")

    for tif_path in tqdm.tqdm(tifs, desc="Segmenting tiles"):
        tile_id = parse_tile_id_from_name(tif_path, mode=mode)

        # Load raster + derive height (meters) and canopy mask
        if mode == "rgbih":
            json_path = tif_path.with_suffix(".json")
            if not json_path.exists():
                logger.warning("Missing JSON for %s; skipping.", tif_path.name)
                continue

            r, g, b, nir, height_m, transform, crs, profile = load_rgbih_and_height_m(
                tif_path=tif_path,
                json_path=json_path,
                band_map=band_map,
            )
            ndvi = compute_ndvi(nir=nir, red=r)
            veg_mask, height_mask, canopy_mask = make_canopy_mask_from_rgbih(ndvi, height_m, mask_params)

            if write_masks:
                write_mask_geotiff(output_dir / f"mask_ndvi_{tile_id}.tif", veg_mask, profile, transform, crs, encoding=mask_encoding)
                write_mask_geotiff(output_dir / f"mask_height_{tile_id}.tif", height_mask, profile, transform, crs, encoding=mask_encoding)
                write_mask_geotiff(output_dir / f"mask_canopy_{tile_id}.tif", canopy_mask, profile, transform, crs, encoding=mask_encoding)

        else:
            height_m, transform, crs, profile = load_ndom_m(tif_path, band=1)
            canopy_mask = make_canopy_mask_from_ndom(height_m, mask_params)
            veg_mask = None
            height_mask = None
            ndvi = None

            if write_masks:
                # For nDOM mode, only canopy mask is meaningful
                write_mask_geotiff(output_dir / f"mask_canopy_{tile_id}.tif", canopy_mask, profile, transform, crs, encoding=mask_encoding)

        # Markers
        markers = markers_from_height_peaks(height_m=height_m, mask=canopy_mask, peak_params=peak_params)
        if markers.max() == 0:
            logger.info("No peaks found for %s; skipping.", tile_id)
            continue

        # Segment crowns
        gdf = watershed_segment(
            height_m=height_m,
            mask=canopy_mask,
            markers=markers,
            transform=transform,
            crs=crs,
            smooth=smooth,
            seg=seg,
        )
        if gdf.empty:
            logger.info("Empty segmentation for %s; skipping write.", tile_id)
            continue

        out_poly = output_dir / f"trees_segmented_{tile_id}.gpkg"
        gdf.to_file(out_poly, driver="GPKG")
        logger.info("Saved %d polygons -> %s", len(gdf), out_poly)

        if write_bbox:
            gdf_bbox = gdf.copy()
            gdf_bbox["geometry"] = gdf_bbox.geometry.envelope
            out_bbox = output_dir / f"trees_bbox_{tile_id}.gpkg"
            gdf_bbox.to_file(out_bbox, driver="GPKG")
            logger.info("Saved %d bbox polygons -> %s", len(gdf_bbox), out_bbox)


# ------------------------- CLI -------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Tree delineation from nDOM or RGBIH using watershed segmentation.")
    ap.add_argument("--img-dir", required=True, help="Directory with input rasters (merged/).")
    ap.add_argument("--output-dir", required=True, help="Directory for output GPKG/masks.")
    ap.add_argument("--mode", choices=["ndom", "rgbih"], required=True)

    # outputs
    ap.add_argument("--write-bbox", action="store_true", help="Also write bbox polygons GPKG.")
    ap.add_argument("--write-masks", action="store_true", help="Also write NDVI/height/canopy mask rasters.")
    ap.add_argument("--mask-encoding", choices=["01", "0255"], default="01", help="Mask pixel values: 0/1 or 0/255.")

    # rgbih band map
    ap.add_argument("--band-r", type=int, default=1)
    ap.add_argument("--band-g", type=int, default=2)
    ap.add_argument("--band-b", type=int, default=3)
    ap.add_argument("--band-nir", type=int, default=4)
    ap.add_argument("--band-h", type=int, default=5)

    # thresholds
    ap.add_argument("--ndvi-thr", type=float, default=0.2)
    ap.add_argument("--height-thr", type=float, default=2.0)
    ap.add_argument("--min-canopy-area-px", type=int, default=10)

    # peaks
    ap.add_argument("--min-distance-px", type=int, default=2)
    ap.add_argument("--peak-threshold-abs-m", type=float, default=None)

    # smoothing
    ap.add_argument("--smooth-filter", choices=["gaussian", "median", "none"], default="gaussian")
    ap.add_argument("--gaussian-sigma", type=float, default=1.5)
    ap.add_argument("--median-size", type=int, default=3)

    # watershed
    ap.add_argument("--use-gradient", action="store_true", help="Use gradient-based watershed (sobel).")
    ap.add_argument("--no-fill-holes", action="store_true", help="Disable hole filling per segment.")

    return ap


def main():
    args = build_argparser().parse_args()

    band_map = BandMap(
        r=args.band_r,
        g=args.band_g,
        b=args.band_b,
        nir=args.band_nir,
        h=args.band_h,
    )

    mask_params = MaskParams(
        ndvi_thr=args.ndvi_thr,
        height_thr_m=args.height_thr,
        min_canopy_area_px=args.min_canopy_area_px,
    )

    peak_params = PeakParams(
        min_distance_px=args.min_distance_px,
        threshold_abs_m=args.peak_threshold_abs_m,
    )

    smooth = SmoothParams(
        fltr=args.smooth_filter,
        gaussian_sigma=args.gaussian_sigma,
        median_size=args.median_size,
    )

    seg = SegmentParams(
        use_gradient=args.use_gradient,
        fill_holes=not args.no_fill_holes,
    )

    run_segmentation_batch(
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        band_map=band_map,
        mask_params=mask_params,
        peak_params=peak_params,
        smooth=smooth,
        seg=seg,
        write_bbox=args.write_bbox,
        write_masks=args.write_masks,
        mask_encoding=args.mask_encoding,
    )


if __name__ == "__main__":
    main()
