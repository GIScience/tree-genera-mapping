"""
This script generates a dataset of image sub-tiles (640x640) for OBD (Object Detection) Model
AND
IMAGE PATCHES FOR IMAGE CLASSIFICATION
"""
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

# Suppress Albumentations update check
# os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Suppress future warnings from pyogrio
warnings.filterwarnings("ignore", category=FutureWarning, module="pyogrio")
# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class ImageDataSet:
    def __init__(self, img_dir: str, output_dir: str,
                 mode: str, label_col: str = None,
                 size: int = 640, overlap: float = 0.0,
                 train: bool = True
                 ):
        self.img_dir = Path(img_dir)
        self.label_col = label_col
        self.mode = mode
        self.overlap = overlap
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.train = train
        self.class_map = {}
        self.class_dirs = {}

    def get_class_dir(self, class_name: str):
        if class_name not in self.class_dirs:
            class_dir = self.output_dir / 'train' / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            self.class_dirs[class_name] = class_dir
        return self.class_dirs[class_name]

    def __len__(self):
        tif_files = list(self.img_dir.glob(f'**/{self.mode}_*.tif'))
        return len(tif_files)

    def split_tiff_to_tiles(self, image_path, trees_gdf, ensure_full_coverage=True):
        stride = int(self.size * (1 - self.overlap))
        image_name = Path(image_path).stem

        if self.train:
            out_img_dir = self.output_dir / 'images' / 'train'
            out_lbl_dir = self.output_dir / 'labels' / 'train'
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_img_dir = self.output_dir / 'images' / 'test'
        out_img_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(image_path) as src:
            width, height = src.width, src.height
            transform = src.transform
            crs = src.crs
            tile_id = 0

            # --- Compute start indices for rows ---
            if ensure_full_coverage:
                y_starts = list(range(0, max(0, height - self.size), stride))
                if y_starts[-1] + self.size < height:
                    y_starts.append(height - self.size)
            else:
                y_starts = range(0, height - self.size + 1, stride)

            # --- Compute start indices for columns ---
            if ensure_full_coverage:
                x_starts = list(range(0, max(0, width - self.size), stride))
                if x_starts[-1] + self.size < width:
                    x_starts.append(width - self.size)
            else:
                x_starts = range(0, width - self.size + 1, stride)

            # --- Loop over tiles ---
            for i in y_starts:
                for j in x_starts:
                    window = Window(j, i, self.size, self.size)
                    tile_transform = src.window_transform(window)
                    tile_bounds = rasterio.windows.bounds(window, transform)
                    tile_geom = gpd.GeoSeries([box(*tile_bounds).buffer(0.01)], crs=crs)

                    tile_data = src.read(window=window, boundless=True, fill_value=0)
                    tile_meta = src.meta.copy()
                    tile_meta.update({
                        'height': self.size,
                        'width': self.size,
                        'transform': tile_transform,
                        'driver': 'GTiff'
                    })

                    # --- Save image tile ---
                    tile_filename = out_img_dir / f"{image_name}_{tile_id}.tif"
                    with rasterio.open(tile_filename, 'w', **tile_meta) as dst:
                        dst.write(tile_data)

                    # --- Save labels ---
                    if self.train:
                        label_filename = out_lbl_dir / f"{image_name}_{tile_id}.txt"
                        labels = self.extract_labels(trees_gdf, tile_geom, tile_transform)
                        with open(label_filename, 'w') as f:
                            for label in labels:
                                f.write(label + '\n')
                        logger.info(f"Tile {tile_filename.name}: {len(labels)} labels")
                    # ---Next tile----
                    tile_id += 1

    def extract_labels(self, trees_gdf, tile_geom, transform):
        intersecting = trees_gdf[trees_gdf.geometry.within(tile_geom.geometry.iloc[0])]
        labels = []
        for _, row in intersecting.iterrows():
            geom = row.geometry
            if not isinstance(geom, BaseGeometry):
                continue
            bounds = geom.bounds
            try:
                x_min, y_min = ~transform * (bounds[0], bounds[1])
                x_max, y_max = ~transform * (bounds[2], bounds[3])
            except Exception as e:
                logger.warning(f"Transform error: {e}")
                continue

            x_center = ((x_min + x_max) / 2) / self.size
            y_center = ((y_min + y_max) / 2) / self.size
            width = abs(x_max - x_min) / self.size
            height = abs(y_max - y_min) / self.size

            if 0 <= x_center <= 1 and 0 <= y_center <= 1:
                label = 0
                if self.label_col and self.label_col in row:
                    try:
                        label = int(row[self.label_col])
                    except Exception as e:
                        logger.warning(f"Invalid label from column '{self.label_col}': {e}")
                labels.append(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return labels

    def cut_bbox_from_merged_tiles(self, image_paths, geom, output_id, class_name="unknown"):
        class_dir = self.output_dir / 'train' / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        try:
            srcs = [rasterio.open(p) for p in image_paths]
            mosaic, out_transform = merge_tiles(srcs)

            meta = srcs[0].meta.copy()
            meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform
            })

            bounds = geom.bounds
            window = rasterio.windows.from_bounds(*bounds, transform=out_transform)
            window = window.round_offsets().round_shape()
            tile_data = mosaic[:, int(window.row_off):int(window.row_off + window.height),
                        int(window.col_off):int(window.col_off + window.width)]

            tile_transform = rasterio.windows.transform(window, out_transform)
            tile_meta = meta.copy()
            tile_meta.update({
                'height': tile_data.shape[1],
                'width': tile_data.shape[2],
                'transform': tile_transform
            })

            tile_filename = class_dir / f"tree_{output_id}.tif"
            with rasterio.open(tile_filename, 'w', **tile_meta) as dst:
                dst.write(tile_data)

            logger.info(f"Saved bbox tile: {tile_filename.name} to class: {class_name}")
            [src.close() for src in srcs]
        except Exception as e:
            logger.warning(f"Error merging/cutting for bbox {output_id}: {e}")


#####################################
def process_ds_obd():
    # Example usage of generating training dataset
    tile_dir = Path('/Volumes/sd17f001/ygrin/silverways/greenspaces/tiles_rgbih')
    mode = tile_dir.name.split('_')[-1]
    output_dir = Path(f'/Volumes/sd17f001/ygrin/silverways/greenspaces/bboxes_{mode}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read trees and tiles

    gdf_tree = gpd.read_file(
        'cache/tree_bboxes_merged.gpkg')  # '/Users/ygrinblat/Documents/Git_repos/greenspaces/cache/mannheim_trees_bbox_sel.gpkg')
    gdf_tiles = gpd.read_file('/Users/ygrinblat/Documents/Git_repos/greenspaces/data/opendata/tiles_mannheim.gpkg')

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


def process_tree_patches():
    if not Path('data/tree_labels.gpkg').exists():
        from tree_genera_mapping.preprocess.generate_training_labels import generate_training_labels
        gdf_tree = generate_training_labels()
    else:
        gdf_tree = gpd.read_file('data/tree_labels.gpkg')

    tile_dir = Path('/Volumes/sd17f001/ygrin/silverways/greenspaces/tiles_rgbih')
    mode = tile_dir.name.split('_')[-1]
    output_dir = Path(f'/Volumes/sd17f001/ygrin/silverways/greenspaces/tree_patches_128_{mode}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # gdf_tree = gpd.read_file('/Users/ygrinblat/Documents/Git_repos/greenspaces/cache/mannheim_trees_ubbox.gpkg')
    gdf_tiles = gpd.read_file('/Users/ygrinblat/Documents/Git_repos/greenspaces/data/opendata/tiles.gpkg')

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


if __name__ == '__main__':
    # Example usage of generating training dataset
    process_ds_obd()