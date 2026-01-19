"""
generate_training_dataset.py

Generate training data for:
1) Tree detection dataset (tiles + labels) from weak tree bboxes
2) Classification patches (e.g., 128x128) from genus labels

This script is intentionally path-agnostic:
- users must provide input paths via CLI
- outputs go to user-defined output directories
"""
from __future__ import annotations

import logging
from pathlib import Path
from tqdm import tqdm

import geopandas as gpd
import rasterio
from rasterio.windows import Window

from tree_genera_mapping.preprocess.dataset import ImageDataSet
from tree_genera_mapping.preprocess.prepare_genus_labels import generate_training_labels

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from rasterio.merge import merge as merge_tiles
import shutil
import numpy as np
import warnings
# -----------------------------
# LOGGING
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# -----------------------------
# 1) Detection dataset
# -------------------------
def process_ds_obd(raw_img_path:str,
                   output_base_dir:str):
    # Example usage of generating training dataset
    tile_dir = Path(raw_img_path)
    mode = tile_dir.name.split('_')[-1]
    output_dir = Path(f'{output_base_dir}/bboxes_{mode}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read trees and tiles
    gdf_tree = gpd.read_file('cache/tree_bboxes_merged.gpkg')
    gdf_tiles = gpd.read_file('data/tiles.gpkg')

    # Ensure both GeoDataFrames have the same CRS
    if gdf_tiles.crs != gdf_tree.crs:
        gdf_tree = gdf_tree.to_crs(gdf_tiles.crs)
        logger.info(f'Reprojected gdf_tree to {gdf_tiles.crs}')

    # Perform spatial join to find intersecting tiles
    try:
        matching_idxs = gpd.sjoin(gdf_tiles, gdf_tree, how='inner', predicate='intersects').index.unique()
    except:
        tiles = gdf_tree['tile_id'].unique()
        matching_idxs = gdf_tiles[gdf_tiles['tile_id'].isin(tiles)].index.unique()
    finally:
        # Filter tiles that intersect with trees
        gdf_tiles_filtered = gdf_tiles.loc[matching_idxs]
        gdf_tiles_filtered['image_path'] = gdf_tiles_filtered['tile_id'].apply(
            lambda x: tile_dir / 'merged' / f"{mode}_{x}.tif")

        dataset = ImageDataSet(
            img_dir=tile_dir,
            output_dir=output_dir,
            mode=mode,
            size=640,
            overlap=0.0
        )

        for _, row in tqdm(gdf_tiles_filtered.iterrows(), total=len(gdf_tiles_filtered), desc="Processing tiles"):
            tile_path = row['image_path']
            if not tile_path.exists():
                logger.warning(f"Tile {tile_path} does not exist, skipping.")
                continue
            dataset.split_tiff_to_tiles(tile_path, gdf_tree)

# -------------------------
# 2) Classification patches
# -------------------------
def make_classification_patches(raw_img_path:str, output_base_dir: str):
    if not Path('data/tree_labels.gpkg').exists():
        gdf_tree = generate_training_labels()
    else:
        gdf_tree = gpd.read_file('data/tree_labels.gpkg')

    tile_dir = Path(raw_img_path)
    mode = tile_dir.name.split('_')[-1]
    output_dir = Path(f'{output_base_dir}/tree_patches_128_{mode}')
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf_tiles = gpd.read_file('data/tiles.gpkg')

    if gdf_tiles.crs != gdf_tree.crs:
        gdf_tree = gdf_tree.to_crs(gdf_tiles.crs)
        logger.info(f'Reprojected gdf_tree to {gdf_tiles.crs}')

    patch_size = 128  # pixels
    half_size = patch_size // 2

    for idx, tree in tqdm(gdf_tree.iterrows(), total=len(gdf_tree), desc="Generating 128x128 patches"):
        centroid = tree.geometry.centroid
        class_name = str(tree.get('training_class', 'unknown')).replace(' ', '_')
        output_id = tree.get('uuid', idx)

        intersecting_tiles = gdf_tiles[gdf_tiles.intersects(centroid.buffer(0.01))]
        if intersecting_tiles.empty:
            continue

        # Find tile image that contains this centroid
        tile_path = None
        tile_ids = intersecting_tiles['dop_kachel'].apply(lambda x: f"{x[:2]}_{x[2:5]}_{x[-4:]}")  # 324645487
        for tile_id in tile_ids:
            candidate_path = tile_dir / 'merged' / f"{mode}_{tile_id}.tif"
            if candidate_path.exists():
                tile_path = candidate_path
                break

        if tile_path is None:
            continue

        try:
            class_dir = output_dir / str(class_name)
            class_dir.mkdir(parents=True, exist_ok=True)
            patch_path = class_dir / f"{output_id}.tif"
            if patch_path.exists():
                logger.info(f"Patch {patch_path} already exists, skipping.")
                continue
            else:
                with rasterio.open(tile_path) as src:
                    row, col = src.index(centroid.x, centroid.y)
                    window = Window(col - half_size, row - half_size, patch_size, patch_size)

                    if window.col_off < 0 or window.row_off < 0:
                        continue
                    if (window.col_off + window.width > src.width) or (window.row_off + window.height > src.height):
                        continue

                    patch = src.read(window=window)
                    transform = src.window_transform(window)
                    meta = src.meta.copy()
                    meta.update({
                        "height": patch.shape[1],
                        "width": patch.shape[2],
                        "transform": transform
                    })

                    with rasterio.open(patch_path, 'w', **meta) as dst:
                        dst.write(patch)
                    logger.info(f"SAVED to {patch_path}")

        except Exception as e:
            logger.warning(f"Failed to create patch for tree {output_id}: {e}")
# -------------------------
# CLI
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate training datasets (detection tiles/labels + classification patches)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_det = sub.add_parser("det", help="Generate YOLO detection dataset from weak bboxes")
    ap_det.add_argument("--tiles-gpkg", required=True)
    ap_det.add_argument("--weak-bboxes-gpkg", required=True)
    ap_det.add_argument("--images-dir", required=True, help="Directory that contains merged/ with tiles")
    ap_det.add_argument("--output-dir", required=True)
    ap_det.add_argument("--mode", required=True, help="Prefix in filenames, e.g. rgbih")
    ap_det.add_argument("--tile-id-col", default="tile_id")
    ap_det.add_argument("--size", type=int, default=640)
    ap_det.add_argument("--overlap", type=float, default=0.0)

    ap_cls = sub.add_parser("patches", help="Generate classification patches from genus labels")
    ap_cls.add_argument("--tiles-gpkg", required=True)
    ap_cls.add_argument("--genus-labels-gpkg", required=True)
    ap_cls.add_argument("--images-dir", required=True, help="Directory that contains merged/ with tiles")
    ap_cls.add_argument("--output-dir", required=True)
    ap_cls.add_argument("--mode", required=True, help="Prefix in filenames, e.g. rgbih")
    ap_cls.add_argument("--patch-size", type=int, default=128)
    ap_cls.add_argument("--tile-id-col", default="dop_kachel")
    ap_cls.add_argument("--class-col", default="training_class")
    ap_cls.add_argument("--id-col", default="uuid")

    args = ap.parse_args()

    if args.cmd == "det":
        make_detection_dataset(
            tiles_gpkg=args.tiles_gpkg,
            weak_bboxes_gpkg=args.weak_bboxes_gpkg,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            tile_id_col=args.tile_id_col,
            size=args.size,
            overlap=args.overlap,
        )
    elif args.cmd == "patches":
        make_classification_patches(
            tiles_gpkg=args.tiles_gpkg,
            genus_labels_gpkg=args.genus_labels_gpkg,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            patch_size=args.patch_size,
            tile_id_col=args.tile_id_col,
            class_col=args.class_col,
            id_col=args.id_col,
        )


if __name__ == "__main__":
    main()