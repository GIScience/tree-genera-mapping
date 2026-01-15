"""
This module implements a watershed-based segmentation algorithm for tree delineation from Canopy Height Models (CHM).
It uses external peak data to improve segmentation accuracy and can optionally apply gradient-based watershed segmentation.
"""

import geopandas as gpd
import numpy as np
import rasterio
import tqdm
import re
from pathlib import Path
from shapely.geometry import shape
from rasterio.features import shapes
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    median_filter,
    binary_fill_holes
)
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import os


def load_chm(img_path, band=1):
    """Load Canopy Height Model from raster image."""
    with rasterio.open(img_path) as src:
        chm = src.read(band).astype('float32')
        transform = src.transform
        crs = src.crs
    return chm, transform, crs


def generate_markers_from_peaks(peaks_gdf, shape, transform):
    """Create marker array from peak points."""
    from rasterio.transform import rowcol
    markers = np.zeros(shape, dtype=np.int32)
    for idx, row in peaks_gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        r, c = rowcol(transform, x, y)
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            markers[r, c] = idx + 1
    return markers


def compute_local_markers(distance, min_distance, labeled_blobs):
    """Generate markers using local maxima of distance transform."""
    local_max = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=labeled_blobs,
        exclude_border=False
    )
    markers = np.zeros_like(distance, dtype=np.int32)
    for idx, (r, c) in enumerate(local_max, 1):
        markers[r, c] = idx
    return markers


def segment_trees(
    chm,
    transform,
    crs,
    external_peaks: gpd.GeoDataFrame = None,
    fltr='gaussian',
    min_tree_area=10,
    min_distance=2,
    smoothing_sigma=1.5,
    smoothing_size=3,
    use_gradient=False
):
    """Run watershed-based segmentation on a CHM."""
    if np.sum(chm > 0) < 10:
        print("Skipping empty CHM")
        return gpd.GeoDataFrame(columns=["label", "geometry"], geometry="geometry", crs=crs)

    labeled_blobs = label(chm > 0)
    labeled_blobs = remove_small_objects(labeled_blobs, min_size=min_tree_area)

    distance = distance_transform_edt(labeled_blobs > 0)

    if external_peaks is not None and not external_peaks.empty:
        markers = generate_markers_from_peaks(external_peaks, chm.shape, transform)
    else:
        markers = compute_local_markers(distance, min_distance, labeled_blobs)

    # Smooth CHM
    chm_smoothed = median_filter(chm, size=smoothing_size) if fltr == 'median' else gaussian_filter(chm, sigma=smoothing_sigma)

    # Apply watershed
    if use_gradient:
        gradient = sobel(chm_smoothed)
        watershed_labels = watershed(gradient, markers, mask=(labeled_blobs > 0))
    else:
        watershed_labels = watershed(-distance, markers, mask=(labeled_blobs > 0))

    # Fill internal holes
    filled_labels = np.zeros_like(watershed_labels, dtype='int32')
    for i in range(1, watershed_labels.max() + 1):
        filled_labels[binary_fill_holes(watershed_labels == i)] = i

    # Convert labeled regions to polygons
    polygons = []
    values = []
    for geom, val in shapes(filled_labels.astype(np.int32), mask=filled_labels > 0, transform=transform):
        polygons.append(shape(geom))
        values.append(val)

    gdf = gpd.GeoDataFrame({'label': values, 'geometry': polygons}, crs=crs)
    gdf['area'] = gdf.geometry.area
    return gdf

def extract_tile_suffix(fname):

    match = re.search(r'chm_(\d+_\d+)_\d+', fname.stem)
    return match.group(1) if match else None

def run_segmentation_batch(img_dir: Path, peaks_dir: Path, output_dir: Path, train_tiles:Path, use_gradient=False):
    imgs = {f.stem: f for f in img_dir.glob('chm_*.tif')}
    peaks_files = [f for f in peaks_dir.glob('*.geojson')]
    output_dir.mkdir(parents=True, exist_ok=True)
    if not imgs:
        raise ValueError(f"No images found in {img_dir}")
    if not peaks_files:
        raise ValueError(f"No peaks files found in {peaks_dir}")

    if train_tiles.exists():
        gdf_train = gpd.read_file(train_tiles)
        tiles = gdf_train['tile_id'].unique()
        tile_suffixes = {tid.replace('32_', '') for tid in tiles}

        peaks_files = [
            f for f in peaks_files
            if extract_tile_suffix(f) in tile_suffixes
        ]
        # imgs = [
        #   f for f in imgs.values()
        #     if f.stem.replace('rgbih_32_', '') in tile_suffixes
        # ]

    for peak_file in tqdm.tqdm(peaks_files, desc="Processing imgs", total=len(peaks_files)):
        tile_id = peak_file.stem.replace('chm', '').strip('_')[:-2]
        matched_img = next((f for k, f in imgs.items() if tile_id in k), None)

        if not matched_img:
            print(f"[!] No image found for {tile_id}")
            continue

        chm, transform, crs = load_chm(matched_img)

        peaks_gdf = gpd.read_file(peak_file)
        if peaks_gdf.crs != "EPSG:25832":
            peaks_gdf = peaks_gdf.to_crs("EPSG:25832")

        print(f"[✓] Segmenting tile {tile_id}")
        gdf_result = segment_trees(
            chm=chm,
            transform=transform,
            crs=crs,
            external_peaks=peaks_gdf,
            use_gradient=use_gradient
        )

        out_path = output_dir / f"trees_segmented_{tile_id}.gpkg"
        gdf_result.to_file(out_path, driver='GPKG')
        print(f"[✔] Saved to {out_path}")


if __name__ == "__main__":
    IMG_DIR = Path('cache/canopy_height')
    PEAKS_DIR = Path('cache/trees')
    OUTPUT_DIR = Path('cache/trees_deliniation_gradient')
    TRAIN_TILES = Path('data/tiles_sel_train.gpkg')
    # Set use_gradient=True if you want to try edge-based watershed
    run_segmentation_batch(IMG_DIR, PEAKS_DIR, OUTPUT_DIR, TRAIN_TILES, use_gradient=False)
