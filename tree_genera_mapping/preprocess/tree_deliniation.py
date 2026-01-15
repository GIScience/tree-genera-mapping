"""
Watershed-based tree delineation from Canopy Height Models (CHM), optionally using NDVI masking
from a multi-band raster (RGB + NIR + Height).

What this module supports
- Single-band CHM workflow:
    - load CHM (1 band)
    - optional external peaks (GeoDataFrame)
    - or peaks computed from CHM
    - watershed segmentation
- Multi-band workflow (default assumption: 5 bands):
    - compute NDVI from (NIR, Red)
    - vegetation mask from NDVI threshold
    - height mask from height threshold
    - peaks from height within mask (or external peaks)
    - watershed segmentation

Outputs
- Vector polygons (GeoPackage) of delineated tree crowns with label + area.

Notes
- Designed for EPSG:25832 but works with any CRS as long as peaks and raster CRS align.
- Keep this as library code; call it from a job/CLI script if you prefer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
import tqdm
from rasterio.features import shapes
from rasterio.transform import rowcol
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    median_filter,
)
from shapely.geometry import shape as shp_shape
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class BandMap:
    """1-indexed band mapping for multi-band rasters (rasterio uses 1-indexing)."""
    r: int = 1
    g: int = 2
    b: int = 3
    nir: int = 4
    h: int = 5


@dataclass(frozen=True)
class MaskParams:
    """Parameters controlling the canopy mask."""
    ndvi_thr: float = 0.2       # vegetation threshold
    height_thr: float = 2.0     # meters (adjust if your height units differ)
    min_canopy_area_px: int = 10  # remove tiny specks in pixel units


@dataclass(frozen=True)
class PeakParams:
    """Parameters for peak detection."""
    min_distance_px: int = 2
    threshold_abs: Optional[float] = None  # if None, no absolute threshold


@dataclass(frozen=True)
class SmoothParams:
    """Smoothing for height/CHM."""
    fltr: str = "gaussian"  # "gaussian" or "median"
    gaussian_sigma: float = 1.5
    median_size: int = 3


@dataclass(frozen=True)
class SegmentParams:
    """Watershed segmentation parameters."""
    use_gradient: bool = False  # gradient-based watershed vs distance-based
    fill_holes: bool = True


def load_single_band_raster(img_path: Union[str, Path], band: int = 1) -> Tuple[np.ndarray, rasterio.Affine, str]:
    """Load a single band raster as float32."""
    img_path = Path(img_path)
    with rasterio.open(img_path) as src:
        arr = src.read(band).astype("float32")
        transform = src.transform
        crs = src.crs.to_string() if src.crs else None
    if crs is None:
        raise ValueError(f"Raster has no CRS: {img_path}")
    return arr, transform, crs


def load_multiband_raster(
    img_path: Union[str, Path],
    band_map: BandMap = BandMap(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, rasterio.Affine, str]:
    """
    Load RGB, NIR, and Height from a multi-band raster. Returns float32 arrays.
    """
    img_path = Path(img_path)
    with rasterio.open(img_path) as src:
        r = src.read(band_map.r).astype("float32")
        g = src.read(band_map.g).astype("float32")
        b = src.read(band_map.b).astype("float32")
        nir = src.read(band_map.nir).astype("float32")
        h = src.read(band_map.h).astype("float32")
        transform = src.transform
        crs = src.crs.to_string() if src.crs else None
    if crs is None:
        raise ValueError(f"Raster has no CRS: {img_path}")
    return r, g, b, nir, h, transform, crs


def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)."""
    return (nir - red) / (nir + red + eps)


def make_canopy_mask(
    ndvi: np.ndarray,
    height: np.ndarray,
    mask_params: MaskParams,
) -> np.ndarray:
    """
    Create a boolean canopy mask = vegetation & above-height.
    Removes small connected components in pixel units.
    """
    veg = ndvi >= float(mask_params.ndvi_thr)
    tall = height >= float(mask_params.height_thr)
    mask = veg & tall

    # Remove tiny blobs
    labeled = label(mask)
    if mask_params.min_canopy_area_px and mask_params.min_canopy_area_px > 1:
        labeled = remove_small_objects(labeled, min_size=int(mask_params.min_canopy_area_px))
    mask = labeled > 0
    return mask


def smooth_height(height: np.ndarray, smooth: SmoothParams) -> np.ndarray:
    """Apply smoothing to height/CHM."""
    if smooth.fltr.lower() == "median":
        return median_filter(height, size=int(smooth.median_size))
    return gaussian_filter(height, sigma=float(smooth.gaussian_sigma))


def markers_from_external_peaks(
    peaks_gdf: gpd.GeoDataFrame,
    raster_shape: Tuple[int, int],
    transform: rasterio.Affine,
    peaks_crs: str,
    raster_crs: str,
) -> np.ndarray:
    """
    Convert point peaks to a marker array. Reprojects peaks to raster CRS if needed.
    """
    if peaks_gdf.empty:
        return np.zeros(raster_shape, dtype=np.int32)

    gdf = peaks_gdf
    if gdf.crs is None:
        # If missing CRS, assume provided peaks_crs is correct
        gdf = gdf.set_crs(peaks_crs)

    if raster_crs and gdf.crs and gdf.crs.to_string() != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    markers = np.zeros(raster_shape, dtype=np.int32)
    idx_counter = 1
    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        x, y = row.geometry.x, row.geometry.y
        r, c = rowcol(transform, x, y)
        if 0 <= r < raster_shape[0] and 0 <= c < raster_shape[1]:
            markers[r, c] = idx_counter
            idx_counter += 1
    return markers


def markers_from_height_peaks(
    height: np.ndarray,
    mask: np.ndarray,
    peak_params: PeakParams,
) -> np.ndarray:
    """
    Compute markers using local maxima on the height surface within the canopy mask.
    """
    # Ensure mask is boolean
    mask = mask.astype(bool)

    # Optional: suppress outside-mask by zeroing height
    h = np.where(mask, height, 0.0)

    coords = peak_local_max(
        h,
        min_distance=int(peak_params.min_distance_px),
        threshold_abs=peak_params.threshold_abs,
        labels=mask.astype(np.uint8),
        exclude_border=False,
    )

    markers = np.zeros_like(height, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    return markers


def watershed_segment(
    height: np.ndarray,
    mask: np.ndarray,
    markers: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    smooth: SmoothParams = SmoothParams(),
    seg: SegmentParams = SegmentParams(),
) -> gpd.GeoDataFrame:
    """
    Perform watershed segmentation and return polygons as a GeoDataFrame.
    """
    if mask.sum() < 10:
        logger.info("Mask is (nearly) empty; returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(columns=["label", "geometry", "area"], geometry="geometry", crs=crs)

    # Smooth the height to reduce noise
    h_smoothed = smooth_height(height, smooth)

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

    # Vectorize labels into polygons
    polys = []
    vals = []
    for geom, val in shapes(labels_ws.astype(np.int32), mask=labels_ws > 0, transform=transform):
        polys.append(shp_shape(geom))
        vals.append(int(val))

    gdf = gpd.GeoDataFrame({"label": vals, "geometry": polys}, crs=crs)
    if not gdf.empty:
        gdf["area"] = gdf.geometry.area
    else:
        gdf["area"] = []
    return gdf


def extract_tile_suffix(fname: Path) -> Optional[str]:
    """
    Extract suffix like '123_4567' from stem patterns: chm_<digits>_<digits>_<digits>
    Adjust this regex to your naming convention if needed.
    """
    match = re.search(r"chm_(\d+_\d+)_\d+", fname.stem)
    return match.group(1) if match else None


def run_segmentation_batch(
    img_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    peaks_dir: Optional[Union[str, Path]] = None,
    train_tiles: Optional[Union[str, Path]] = None,
    use_gradient: bool = False,
    # Multi-band options
    multiband: bool = False,
    band_map: BandMap = BandMap(),
    mask_params: MaskParams = MaskParams(),
    peak_params: PeakParams = PeakParams(),
    smooth: SmoothParams = SmoothParams(),
    peaks_crs: str = "EPSG:25832",
) -> None:
    """
    Batch segment all rasters in img_dir. Optionally match peaks files by tile naming.

    If multiband=False:
        expects CHM rasters matching pattern 'chm_*.tif' (single band by default)

    If multiband=True:
        expects rasters matching pattern '*.tif' (or change the glob below),
        and height is taken from band_map.h, NDVI from (band_map.nir, band_map.r).
    """
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if multiband:
        img_files = sorted(img_dir.glob("*.tif"))
    else:
        img_files = sorted(img_dir.glob("chm_*.tif"))

    if not img_files:
        raise ValueError(f"No raster images found in {img_dir}")

    # Optional peaks handling
    peaks_files = None
    if peaks_dir is not None:
        peaks_dir = Path(peaks_dir)
        peaks_files = sorted(peaks_dir.glob("*.geojson"))
        if not peaks_files:
            raise ValueError(f"No peaks files found in {peaks_dir}")

    # Optional: filter peaks/images by train tiles selection
    tile_suffixes = None
    if train_tiles is not None:
        train_tiles = Path(train_tiles)
        if train_tiles.exists():
            gdf_train = gpd.read_file(train_tiles)
            if "tile_id" in gdf_train.columns:
                tiles = gdf_train["tile_id"].unique()
                tile_suffixes = {str(tid).replace("32_", "") for tid in tiles}
            elif "dop_kachel" in gdf_train.columns:
                # fallback if your selection uses dop_kachel codes
                tiles = gdf_train["dop_kachel"].astype(str).unique()
                # keep as-is; users may adjust to their naming
                tile_suffixes = set(tiles)

    # Build a lookup for images by stem for matching
    imgs_by_stem: Dict[str, Path] = {f.stem: f for f in img_files}

    # If peaks provided, iterate peaks; else iterate images
    if peaks_files is not None:
        files_iter: Iterable[Path] = peaks_files

        # Optionally filter peaks by tile suffixes
        if tile_suffixes is not None:
            files_iter = [f for f in peaks_files if (extract_tile_suffix(f) in tile_suffixes)]
    else:
        files_iter = img_files

    for item in tqdm.tqdm(list(files_iter), desc="Segmenting tiles"):
        # Decide which image to use
        if peaks_files is not None:
            peak_file = item
            # Your original logic:
            tile_id = peak_file.stem.replace("chm", "").strip("_")[:-2]
            matched_img = next((f for k, f in imgs_by_stem.items() if tile_id in k), None)
            if not matched_img:
                logger.warning("No image found matching peaks file: %s", peak_file.name)
                continue
        else:
            peak_file = None
            matched_img = item
            tile_id = matched_img.stem

        # Load data
        if multiband:
            r, g, b, nir, h, transform, raster_crs = load_multiband_raster(matched_img, band_map=band_map)
            ndvi = compute_ndvi(nir=nir, red=r)
            mask = make_canopy_mask(ndvi=ndvi, height=h, mask_params=mask_params)
            height = h
        else:
            chm, transform, raster_crs = load_single_band_raster(matched_img, band=1)
            # In single-band mode, mask is simply chm > height_thr (optional) or chm > 0
            # If you want height thresholding, use mask_params.height_thr; else set it to 0.
            base = chm > 0
            if mask_params.height_thr is not None and mask_params.height_thr > 0:
                base = chm >= float(mask_params.height_thr)
            labeled = label(base)
            labeled = remove_small_objects(labeled, min_size=int(mask_params.min_canopy_area_px))
            mask = labeled > 0
            height = chm

        # Markers
        external_peaks_gdf = None
        if peak_file is not None:
            try:
                external_peaks_gdf = gpd.read_file(peak_file)
            except Exception as e:
                logger.warning("Failed to read peaks file %s: %s", peak_file, e)
                external_peaks_gdf = None

        if external_peaks_gdf is not None and not external_peaks_gdf.empty:
            markers = markers_from_external_peaks(
                peaks_gdf=external_peaks_gdf,
                raster_shape=height.shape,
                transform=transform,
                peaks_crs=peaks_crs,
                raster_crs=raster_crs,
            )
            # If peaks are outside mask, markers might be empty; fallback to internal peaks:
            if markers.max() == 0:
                logger.info("External peaks produced no markers inside raster; falling back to internal peaks.")
                markers = markers_from_height_peaks(height=height, mask=mask, peak_params=peak_params)
        else:
            markers = markers_from_height_peaks(height=height, mask=mask, peak_params=peak_params)

        if markers.max() == 0:
            logger.info("No peaks/markers found for %s; skipping.", tile_id)
            continue

        # Segment
        gdf_result = watershed_segment(
            height=height,
            mask=mask,
            markers=markers,
            transform=transform,
            crs=raster_crs,
            smooth=smooth,
            seg=SegmentParams(use_gradient=use_gradient, fill_holes=True),
        )

        # Save
        out_path = output_dir / f"trees_segmented_{tile_id}.gpkg"
        gdf_result.to_file(out_path, driver="GPKG")
        logger.info("Saved %d polygons to %s", len(gdf_result), out_path)


def _build_argparser():
    import argparse

    ap = argparse.ArgumentParser(description="Tree delineation from CHM or multi-band RGB+NIR+H using watershed.")
    ap.add_argument("--img-dir", required=True, help="Directory with raster tiles.")
    ap.add_argument("--output-dir", required=True, help="Directory to write segmented polygons (GPKG).")

    ap.add_argument("--peaks-dir", default=None, help="Optional dir with peak GeoJSON files.")
    ap.add_argument("--train-tiles", default=None, help="Optional GeoPackage restricting tiles to a subset.")

    ap.add_argument("--multiband", action="store_true", help="Treat input rasters as multi-band (RGB+NIR+H).")
    ap.add_argument("--use-gradient", action="store_true", help="Use gradient watershed instead of distance transform.")

    # Band map
    ap.add_argument("--band-r", type=int, default=1)
    ap.add_argument("--band-g", type=int, default=2)
    ap.add_argument("--band-b", type=int, default=3)
    ap.add_argument("--band-nir", type=int, default=4)
    ap.add_argument("--band-h", type=int, default=5)

    # Mask params
    ap.add_argument("--ndvi-thr", type=float, default=0.2)
    ap.add_argument("--height-thr", type=float, default=2.0)
    ap.add_argument("--min-canopy-area-px", type=int, default=10)

    # Peak params
    ap.add_argument("--min-distance-px", type=int, default=2)
    ap.add_argument("--peak-threshold-abs", type=float, default=None)

    # Smoothing
    ap.add_argument("--smooth-filter", choices=["gaussian", "median"], default="gaussian")
    ap.add_argument("--gaussian-sigma", type=float, default=1.5)
    ap.add_argument("--median-size", type=int, default=3)

    # CRS for peaks if missing
    ap.add_argument("--peaks-crs", default="EPSG:25832", help="Assumed CRS for peaks if none is set in file.")
    return ap


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    run_segmentation_batch(
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        peaks_dir=args.peaks_dir,
        train_tiles=args.train_tiles,
        use_gradient=args.use_gradient,
        multiband=args.multiband,
        band_map=BandMap(r=args.band_r, g=args.band_g, b=args.band_b, nir=args.band_nir, h=args.band_h),
        mask_params=MaskParams(
            ndvi_thr=args.ndvi_thr,
            height_thr=args.height_thr,
            min_canopy_area_px=args.min_canopy_area_px,
        ),
        peak_params=PeakParams(
            min_distance_px=args.min_distance_px,
            threshold_abs=args.peak_threshold_abs,
        ),
        smooth=SmoothParams(
            fltr=args.smooth_filter,
            gaussian_sigma=args.gaussian_sigma,
            median_size=args.median_size,
        ),
        peaks_crs=args.peaks_crs,
    )
