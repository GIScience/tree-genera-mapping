import os
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
from rasterio.merge import merge as merge_tiles
import logging
import shutil
from tqdm import tqdm
import numpy as np
import warnings

from tree_genera_mapping.preprocess.dataset import ImageDataSet
from tree_genera_mapping.preprocess.prepare_genus_labels import generate_training_labels

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
#####################################
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


def process_tree_patches(raw_img_path:str, output_base_dir: str):
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
