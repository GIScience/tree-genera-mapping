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

from tree_genera_mapping.preprocess.height_dataset import HeightModel
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

        self.temp_dir = Path(temp_dir) if temp_dir else None

        # ✅ NEW: expose normalization stats for JSON writing
        self.last_height_stats_m = None   # (vmin_m, vmax_m)
        self.last_height_source = None    # "ndom1" or "dom1-dgm1"

    def process(self):
        if self.output_path.exists():
            logger.info(f"Skipping existing: {self.output_path}")
            return self.output_path

        rgbi, transform, meta = self._read_rgbi()

        band_indices = {
            'RGB':  [0, 1, 2],
            'RGBI': [0, 1, 2, 3],
            'RGBIH':[0, 1, 2, 3],
        }
        selected = rgbi[band_indices[self.mode]]  # (C,H,W)

        if self.mode == 'RGBIH':
            height_u8, stats_m, source = self._get_height(selected.shape[1:], transform, meta)
            if height_u8 is None:
                logger.warning(f"No height model for {self.tile_id}")
                return None

            self.last_height_stats_m = stats_m
            self.last_height_source = source

            data = np.vstack([selected, height_u8[None, ...]])
        else:
            data = selected

        meta.update({
            'count': data.shape[0],
            'dtype': 'uint8',
            'nodata': None,
        })

        if self.mode in ['RGB', 'RGBI']:
            meta['photometric'] = 'RGB'
        else:
            meta.pop('photometric', None)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(self.output_path, 'w', **meta) as dst:
            dst.write(data)

        logger.info(f"Saved: {self.output_path}")
        return self.output_path

    def _read_rgbi(self):
        with rasterio.open(self.dop_path) as src:
            data = src.read()
            meta = src.meta.copy()
            transform = src.transform
            meta.update({'crs': 'EPSG:25832'})
        return data, transform, meta

    def _get_height(self, shape, transform, meta):
        """
        Returns:
          height_u8: uint8 (H,W)
          stats_m: (vmin_m, vmax_m) used for normalization
          source: "ndom1" or "dom1-dgm1"
        """
        # Prefer ndom1 if present
        if self.ndom_path and self.ndom_path.exists():
            height_u8, stats_m = self._load_height(self.ndom_path, shape, transform)
            if height_u8 is not None:
                return height_u8, stats_m, "ndom1"

        # Fallback: build CHM from DOM+DGM via HeightModel
        if self.dgm_path and self.dom_path and self.dgm_path.exists() and self.dom_path.exists():
            if self.temp_dir is not None:
                temp_dir = self.temp_dir / "chm" / self.tile_id
            else:
                temp_dir = self.output_path.parent / 'chm_temp' / self.tile_id

            temp_dir.mkdir(parents=True, exist_ok=True)

            try:
                ndsm_path = TileDataset.process_hm(
                    self.tile_id,
                    [self.dgm_path],
                    [self.dom_path],
                    temp_dir,
                    meta['transform'][0],
                    meta['crs']
                )
                height_u8, stats_m = self._load_height(ndsm_path, shape, transform)
                if height_u8 is not None:
                    return height_u8, stats_m, "dom1-dgm1"
            except Exception as e:
                logger.exception(f"Failed to process nDSM: {e}")
                return None, None, None

        return None, None, None

    def _load_height(self, path, shape, transform):
        """
        Warp CHM/nDSM raster onto RGB grid, then normalize using per-tile vmin/vmax.

        Returns:
          height_u8: uint8 (H,W) or None
          stats_m: (vmin_m, vmax_m) or None
        """
        target_h, target_w = shape

        with rasterio.open(path) as src:
            src_data = src.read(1).astype(np.float32)
            src_transform = src.transform
            src_crs = src.crs

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

        if not np.isfinite(dst).any():
            logger.warning(f"CHM {path} reprojected to all-NaN; treating as missing.")
            return None, None

        vmin = float(np.nanmin(dst))
        vmax = float(np.nanmax(dst))

        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            # degenerate tile: store zeros but still return something sensible
            return np.zeros((target_h, target_w), dtype=np.uint8), (0.0, 0.0)

        height_u8 = normalize_hm_to_255(dst, vmin, vmax)
        return height_u8, (vmin, vmax)

    @staticmethod
    def process_hm(key, dgm_files, dom_files, output_dir, res, crs):
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