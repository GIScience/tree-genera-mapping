"""
This module process and generates RGBI/RGBIH Tiles(1x1km)
"""

import os
import logging
import shutil
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling as ResamplingEnums
from rasterio.transform import from_origin
from rasterio.windows import Window
import geopandas as gpd
import numpy as np

from collections import defaultdict
from tqdm import tqdm
import sys

# bring research code into path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
logging.info(PROJECT_ROOT)

from tree_genera_mapping.preprocess.dataset_chm import HeightModel
from tree_genera_mapping.preprocess.utils import img_resample, normalize_hm_to_255
from multiprocessing import Pool, cpu_count

import json

# ----------------------------------------------------------------------
# logger
# ----------------------------------------------------------------------
# Suppress future warnings from pyogrio
warnings.filterwarnings("ignore", category=FutureWarning, module="pyogrio")
# LOGGING
logging.basicConfig( level=logging.INFO,   format='%(asctime)s - %(levelname)s - %(message)s',)
logger = logging.getLogger(__name__)
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

# def img_resample(data, data_transform, crs, scale_factor):
#     """Resample raster data to a new resolution."""
#
#     # Calculate the dimensions of the resampled raster
#     new_width = int(data.shape[1] * scale_factor)
#     new_height = int(data.shape[0] * scale_factor)
#
#     # Initialize an array for the resampled data
#     resampled_data = np.empty((new_height, new_width), dtype=np.float32)
#
#     # Perform the resampling
#     reproject(
#         source=data,
#         destination=resampled_data,
#         src_transform=data_transform,
#         src_crs=crs,
#         dst_transform=data_transform * data_transform.scale(1 / scale_factor, 1 / scale_factor),
#         dst_crs=crs,
#         resampling=ResamplingEnums.bilinear
#     )
#
#     return resampled_data, data_transform * data_transform.scale(1 / scale_factor, 1 / scale_factor)
#
#
# def normalize_hm_to_255(chm_data, glb_min, glb_max):
#     """Normalize Canopy Height Model (CHM) data to a 0-255 range."""
#     chm_min = glb_min  # np.min(chm_data)
#     chm_max = glb_max  # np.max(chm_data)
#
#     # Avoid division by zero
#     if chm_max == chm_min:
#         return np.zeros_like(chm_data, dtype=np.uint8)
#
#     # Normalize to 0-255 range
#     normalized_chm = (255 * (chm_data - chm_min) / (chm_max - chm_min)).astype(np.uint8)
#     return normalized_chm


def safe_get_path(tile_id, file_list):
    matches = [f for f in file_list if str(tile_id) in str(f)]
    return matches[0] if matches else None


# ----------------------------------------------------------------------
# Core dataset class
# ----------------------------------------------------------------------
class TileDataset:
    def __init__(self,
                 tile_id,
                 output_dir,
                 mode='RGBIH',
                 dop_path=None,
                 ndom_path=None,
                 dgm_path=None,
                 dom_path=None,
                 temp_dir=None,
                 ):
        self.tile_id = tile_id
        self.mode = mode.upper()

        self.dop_path = Path(dop_path) if dop_path else None
        self.ndom_path = Path(ndom_path) if ndom_path else None
        self.dgm_path = Path(dgm_path) if dgm_path else None
        self.dom_path = Path(dom_path) if dom_path else None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / 'merged' / f"{self.mode.lower()}_{tile_id}.tif"

        self.temp_dir = Path(temp_dir) if temp_dir else None  # 👈 NEW

    def process(self):
        if self.output_path.exists():
            logger.info(f"Skipping existing: {self.output_path}")
            return self.output_path

        rgbi, transform, meta = self._read_rgbi()
        band_indices = {
            'RGB': [0, 1, 2],
            'RGBI': [0, 1, 2, 3],
            'RGBIH': [0, 1, 2, 3]
        }
        selected = rgbi[band_indices[self.mode]]  # (C,H,W)

        if self.mode == 'RGBIH':
            height = self._get_height(selected.shape[1:], transform, meta)
            if height is None:
                logger.warning(f"No height model for {self.tile_id}")
                return None
            # height: (H, W) -> (1, H, W)
            data = np.vstack([selected, height[None, ...]])
        else:
            data = selected

        # Basic metadata
        meta.update({'count': data.shape[0],
                     'dtype': 'uint8',
                     'nodata': None,
                     # 'photometric': 'RGB' if self.mode in ['RGB', 'RGBI'] else None
                     })
        # Only set photometric for pure RGB / RGBI
        if self.mode in ['RGB', 'RGBI']:
            meta['photometric'] = 'RGB'
        else:
            # Make sure it's not hanging around from the source
            meta.pop('photometric', None)

        # ensure parent dir exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(self.output_path, 'w', **meta) as dst:
            dst.write(data)

        # if (self.output_path.parent/'temp').exists():
        #     shutil.rmtree(self.output_path.parent/'temp')

        logger.info(f"Saved: {self.output_path}")
        return self.output_path

    def _read_rgbi(self):
        # Read DOP 4-band data and build transform from .tfw.

        # tfw = self.dop_path.with_suffix('.tfw')
        # with open(tfw) as f:
        #     lines = [float(x) for x in f.readlines()]

        # transform = from_origin(lines[4], lines[5], lines[0], abs(lines[3]))

        with rasterio.open(self.dop_path) as src:
            data = src.read()
            meta = src.meta.copy()
            # Use the transform directly from the TIFF
            transform = src.transform
            meta.update({'crs': 'EPSG:25832'
                         # 'transform': transform,
                         })

        return data, transform, meta

    def _get_height(self, shape, transform, meta):
        """
        Load / compute CHM (height) aligned to the RGB tile grid.
        shape: (H, W) of the RGB tile
        transform: rasterio.Affine of the RGB tile
        """
        # Prefer precomputed nDSM/CHM if it exists
        if self.ndom_path and self.ndom_path.exists():
            return self._load_height(self.ndom_path, shape, transform)

        # Otherwise, build CHM from DGM + DOM on the fly
        if self.dgm_path and self.dom_path and self.dgm_path.exists() and self.dom_path.exists():
            if self.temp_dir is not None:
                temp_dir = self.temp_dir / "chm"
            else:
                # Use a PER-TILE temp dir to avoid race conditions across jobs
                temp_dir = self.output_path.parent / 'chm_temp' / self.tile_id
            temp_dir.mkdir(exist_ok=True)
            try:
                ndsm = TileDataset.process_hm(self.tile_id,
                                              [self.dgm_path],
                                              [self.dom_path],
                                              temp_dir,
                                              meta['transform'][0],
                                              meta['crs'])
                return self._load_height(ndsm, shape, transform)
            except Exception as e:
                print(f"Failed to process nDSM: {e}")
                return None
            # finally:
            #     # Clean only this tile's temp dir
            #     if temp_dir.exists():
            #         shutil.rmtree(temp_dir, ignore_errors=True)

        # No way to create height
        return None

    def _load_height(self, path, shape, transform):
        """
        Load CHM/nDSM raster and warp it exactly onto the RGB tile grid
        defined by (shape, transform).
        """
        target_h, target_w = shape

        with rasterio.open(path) as src:
            src_data = src.read(1).astype(np.float32)
            src_transform = src.transform
            src_crs = src.crs
            #  if data.shape != shape:
            #     scale = int(src.transform[0] / transform[0])
            #     data, _ = img_resample(data, src.transform, src.crs, scale)
            # return normalize_hm_to_255(data, np.nanmin(data), np.nanmax(data))

        # Allocate destination array
        dst = np.empty((target_h, target_w), dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=src_crs,
            resampling=Resampling.bilinear,
        )
        # If everything is invalid → no height
        if not np.isfinite(dst).any():
            logger.warning(f"CHM {path} reprojected to all-NaN; treating as missing.")
            return None

            # Handle degenerate/all-NaN case safely
        vmin = np.nanmin(dst)
        vmax = np.nanmax(dst)
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            return np.zeros_like(dst, dtype=np.uint8)

        #TODO: drop extreme values to json?
        return normalize_hm_to_255(dst, vmin, vmax)

    @staticmethod
    def process_hm(key, dgm_files, dom_files, output_dir, res, crs):
        """Create CHM via HeightModel"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chm = HeightModel(
            dgm_path=dgm_files[0],
            dom_path=dom_files[0],
            key=key,
            output_dir=output_dir,
            res=res,
            crs=crs
        )
        chm.generate_chm()
        return chm.output_dir / chm.file_name


# ----------------------------------------------------------------------
# Dataset Helpers
# ----------------------------------------------------------------------
def generate_ds_tiles(input_path, dop_folder=None, dgm_folder=None, dom_folder=None, ndsm_folder=None,
                      col='dop_kachel'):
    gdf = gpd.read_file(input_path)
    gdf['tile_id'] = gdf[col].apply(lambda x: f"{x[:2]}_{x[2:5]}_{x[5:9]}")

    if dop_folder:
        dop20_imgs = list(Path(dop_folder).rglob('dop20rgbi_*.tif'))
        gdf['dop20_path'] = gdf['tile_id'].apply(lambda x: safe_get_path(x, dop20_imgs))

    if ndsm_folder:
        ndsm_imgs = list(Path(ndsm_folder).rglob('nDSM_*.tif'))
        gdf['ndsm1_path'] = gdf['tile_id'].apply(lambda x: safe_get_path(x, ndsm_imgs))

    if dgm_folder:
        dgm_imgs = list(Path(dgm_folder).rglob('dgm1_*.xyz'))
        gdf['dgm1_path'] = gdf['tile_id'].apply(lambda x: safe_get_path(x, dgm_imgs))

    if dom_folder:
        dom_imgs = list(Path(dom_folder).rglob('dom1_*.tif'))
        gdf['dom1_path'] = gdf['tile_id'].apply(lambda x: safe_get_path(x, dom_imgs))

    return gdf


def safe_get_path(tile_id, file_list):
    matches = [f for f in file_list if tile_id in f.name]
    return matches[0] if matches else None


def process_tile_row(row, output_dir, mode='RGBIH'):
    tile = TileDataset(
        tile_id=row['tile_id'],
        dop_path=row['dop20_path'],
        ndom_path=row.get('ndsm1_path'),
        dgm_path=row.get('dgm1_path'),
        dom_path=row.get('dom1_path'),
        output_dir=output_dir,
        mode=mode
    )
    return tile.process()


# if __name__ == '__main__':