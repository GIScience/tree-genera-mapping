#!/usr/bin/env python3
"""
segment_trees.py

CLI runner that:
- reads nDOM (meters) OR RGBIH (uint8 normalized) + JSON stats
- for RGBIH: unnormalizes height band to meters
- optionally writes masks as GeoTIFF
- calls tree_delineation library for peaks + watershed
- writes polygons (and optional bbox polygons) as GPKG
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import geopandas as gpd
import rasterio
import tqdm

from tree_genera_mapping.preprocess.tree_delineation import (
    MaskParams,
    PeakParams,
    SmoothParams,
    SegmentParams,
    compute_ndvi,
    canopy_mask_from_height,
    masks_from_ndvi_and_height,
    markers_from_height_peaks,
    watershed_segment,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass(frozen=True)
class BandMap:
    r: int = 1
    g: int = 2
    b: int = 3
    nir: int = 4
    h: int = 5


def parse_tile_id_from_name(path: Path, mode: str) -> str:
    stem = path.stem
    mode = mode.lower()
    if mode == "rgbih" and stem.startswith("rgbih_"):
        return stem.replace("rgbih_", "", 1)
    if mode == "ndom" and stem.startswith("ndom1_"):
        return stem.replace("ndom1_", "", 1)
    return stem


def unnormalize_height_u8_to_m(height_u8: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if np.isclose(vmax, vmin):
        return np.zeros_like(height_u8, dtype=np.float32)
    return ((height_u8.astype(np.float32) / 255.0) * (vmax - vmin) + vmin).astype(np.float32, copy=False)


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


def load_rgbih_and_height_m(
    tif_path: Path,
    json_path: Path,
    band_map: BandMap,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, rasterio.Affine, str, dict]:
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
        h_u8 = src.read(h_band)  # keep raw dtype, usually uint8
        transform = src.transform

    height_m = unnormalize_height_u8_to_m(h_u8, vmin, vmax)
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


def run_segmentation_batch(
    img_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    mode: str,
    band_map: BandMap,
    mask_params: MaskParams,
    peak_params: PeakParams,
    smooth: SmoothParams,
    seg: SegmentParams,
    write_pol: bool,
    write_masks: bool,
    mask_encoding: str,
) -> None:
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = mode.lower()
    if mode not in ("ndom", "rgbih"):
        raise ValueError("--mode must be 'ndom' or 'rgbih'")

    tifs = sorted(img_dir.glob("rgbih_*.tif" if mode == "rgbih" else "ndom1_*.tif"))
    if not tifs:
        raise ValueError(f"No input tiles found in {img_dir} for mode={mode}")

    for tif_path in tqdm.tqdm(tifs, desc="Segmenting tiles"):
        tile_id = parse_tile_id_from_name(tif_path, mode=mode)

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
            veg_mask, height_mask, canopy_mask = masks_from_ndvi_and_height(ndvi, height_m, mask_params)

            if write_masks:
                write_mask_geotiff(output_dir / f"mask_ndvi_{tile_id}.tif", veg_mask, profile, transform, crs, encoding=mask_encoding)
                write_mask_geotiff(output_dir / f"mask_height_{tile_id}.tif", height_mask, profile, transform, crs, encoding=mask_encoding)
                write_mask_geotiff(output_dir / f"mask_canopy_{tile_id}.tif", canopy_mask, profile, transform, crs, encoding=mask_encoding)

        else:
            height_m, transform, crs, profile = load_ndom_m(tif_path, band=1)
            canopy_mask = canopy_mask_from_height(height_m, mask_params)

            if write_masks:
                write_mask_geotiff(output_dir / f"mask_canopy_{tile_id}.tif", canopy_mask, profile, transform, crs, encoding=mask_encoding)

        markers = markers_from_height_peaks(height_m, canopy_mask, peak_params)
        if markers.max() == 0:
            logger.info("No peaks found for %s; skipping.", tile_id)
            continue

        gdf = watershed_segment(
            height_m=height_m,
            canopy_mask=canopy_mask,
            markers=markers,
            transform=transform,
            crs=crs,
            smooth=smooth,
            seg=seg,
        )
        if gdf.empty:
            logger.info("Empty segmentation for %s; skipping write.", tile_id)
            continue

        gdf_bbox = gdf.copy()
        gdf_bbox["geometry"] = gdf_bbox.geometry.envelope
        out_bbox = output_dir / f"trees_bbox_{tile_id}.gpkg"
        gdf_bbox.to_file(out_bbox, driver="GPKG")
        logger.info("Saved %d bbox polygons -> %s", len(gdf_bbox), out_bbox)

        if write_pol:
            out_poly = output_dir / f"trees_segmented_{tile_id}.gpkg"
            gdf.to_file(out_poly, driver="GPKG")
            logger.info("Saved %d polygons -> %s", len(gdf), out_poly)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Tree delineation from nDOM or RGBIH using watershed segmentation.")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["ndom", "rgbih"], required=True)

    ap.add_argument("--write-pols", action="store_true")
    ap.add_argument("--write-masks", action="store_true")
    ap.add_argument("--mask-encoding", choices=["01", "0255"], default="01")

    ap.add_argument("--band-r", type=int, default=1)
    ap.add_argument("--band-g", type=int, default=2)
    ap.add_argument("--band-b", type=int, default=3)
    ap.add_argument("--band-nir", type=int, default=4)
    ap.add_argument("--band-h", type=int, default=5)

    ap.add_argument("--ndvi-thr", type=float, default=0.2)
    ap.add_argument("--height-thr", type=float, default=2.0)
    ap.add_argument("--min-canopy-area-px", type=int, default=10)

    ap.add_argument("--min-distance-px", type=int, default=2)
    ap.add_argument("--peak-threshold-abs-m", type=float, default=None)

    ap.add_argument("--smooth-filter", choices=["gaussian", "median", "none"], default="gaussian")
    ap.add_argument("--gaussian-sigma", type=float, default=1.5)
    ap.add_argument("--median-size", type=int, default=3)

    ap.add_argument("--use-gradient", action="store_true")
    ap.add_argument("--no-fill-holes", action="store_true")

    return ap


def main():
    args = build_argparser().parse_args()

    band_map = BandMap(args.band_r, args.band_g, args.band_b, args.band_nir, args.band_h)

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
        write_pol=args.write_pols,
        write_masks=args.write_masks,
        mask_encoding=args.mask_encoding,
    )


if __name__ == "__main__":
    main()
