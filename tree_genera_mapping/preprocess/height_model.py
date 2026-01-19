"""
height_model.py

Create a  Height Model/nDOM (nDSM) from:
  - DOM (DSM raster, GeoTIFF)
  - DGM (DTM raster OR XYZ point cloud)

CHM := DOM - DGM  (meters)

Outputs:
- HeightModel.generate_chm() writes:  ndom1_<key>.tif  (float32, 1 band, nodata=NaN)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import  Optional, Tuple, Union

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling as ResamplingEnums
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject



log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _as_crs(crs: Union[str, CRS, None], fallback: str = "EPSG:25832") -> CRS:
    if crs is None:
        return CRS.from_string(fallback)
    if isinstance(crs, CRS):
        return crs
    return CRS.from_string(crs)


def _read_xyz(path: Path) -> np.ndarray:
    """
    Read XYZ file into Nx3 float array.
    Uses genfromtxt to tolerate malformed rows.
    """
    arr = np.genfromtxt(
        path,
        delimiter=" ",
        usecols=(0, 1, 2),
        invalid_raise=False,
    )

    if arr.size == 0:
        raise ValueError(f"XYZ file is empty: {path}")

    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(f"Unexpected XYZ format (1D, size={arr.size}): {path}")
        arr = arr.reshape((1, 3))

    # Drop rows with NaNs
    arr = arr[~np.isnan(arr).any(axis=1)]
    if arr.size == 0:
        raise ValueError(f"XYZ file has no valid rows after NaN filtering: {path}")

    return arr.astype(np.float64, copy=False)


def _xyz_to_grid_min_z(
    xyz: np.ndarray,
    dom_transform: Affine,
    dom_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Map XYZ points into the DOM grid.
    Returns a DGM grid (H,W) with per-pixel minimum z, NaN where missing.

    IMPORTANT: We do NOT clip out-of-bounds points into the grid.
    We mask them out instead.
    """
    H, W = dom_shape
    xs = xyz[:, 0]
    ys = xyz[:, 1]
    zs = xyz[:, 2].astype(np.float32, copy=False)

    rows, cols = rasterio.transform.rowcol(dom_transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    m = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & np.isfinite(zs)
    if not m.any():
        raise ValueError("No XYZ points fall inside the DOM raster extent.")

    # Per-pixel minimum z for duplicates
    dgm = np.full((H, W), np.inf, dtype=np.float32)
    flat = dgm.reshape(-1)
    lin = rows[m] * W + cols[m]
    np.minimum.at(flat, lin, zs[m])

    dgm = flat.reshape(H, W)
    dgm[~np.isfinite(dgm)] = np.nan
    return dgm


class HeightModel:
    """
    Create HeightModel (nDSM) on the DOM grid, optionally resampling to target resolution.

    Output filename:
      ndom1_<key>.tif
    """
    def __init__(
        self,
        dgm_path: Union[str, Path],
        dom_path: Union[str, Path],
        key: str,  # e.g. "32_355_6048"
        output_dir: Union[str, Path],
        res: float,  # meters (e.g. 0.2 or 1.0)
        crs: str = "EPSG:25832",
    ):
        self.dgm_path = Path(dgm_path)
        self.dom_path = Path(dom_path)
        self.key = key
        self.res = float(res)
        self.crs = crs

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_name = f"ndom1_{key}.tif"

        self.chm: Optional[np.ndarray] = None
        self.transform: Optional[Affine] = None
        self._crs_obj: Optional[CRS] = None

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.file_name

    def generate_chm(self) -> Path:
        """
        Writes CHM to GeoTIFF and returns output path.
        """
        dom_data, dom_transform, dom_crs, dom_nodata = self._load_dom()
        dgm_data, dgm_transform, dgm_crs = self._load_or_build_dgm(dom_transform, dom_data.shape, dom_crs)

        # Align DGM to DOM grid if needed
        if (dgm_crs != dom_crs) or (dgm_data.shape != dom_data.shape) or (dgm_transform != dom_transform):
            dgm_data, dgm_transform = self._reproject(
                data=dgm_data,
                transform=dgm_transform,
                src_crs=dgm_crs,
                dst_crs=dom_crs,
                target_shape=dom_data.shape,
                target_transform=dom_transform,
            )

        # Valid masks
        if dom_nodata is None or (isinstance(dom_nodata, float) and np.isnan(dom_nodata)):
            dom_valid = np.isfinite(dom_data)
        else:
            dom_valid = dom_data != dom_nodata

        dgm_valid = np.isfinite(dgm_data)

        chm = np.full(dom_data.shape, np.nan, dtype=np.float32)
        valid = dom_valid & dgm_valid
        chm[valid] = (dom_data[valid] - dgm_data[valid]).astype(np.float32, copy=False)

        self.transform = dom_transform
        self._crs_obj = dom_crs

        # Optional resample to requested resolution (meters)
        dom_res = float(dom_transform.a)
        if not np.isclose(self.res, dom_res):
            chm, self.transform = self._resample(chm, self.transform, self.res, dom_crs)

        self.chm = chm
        self._save()
        log.info("✅ CHM saved: %s", self.output_path)
        return self.output_path

    # -------------------- internal helpers --------------------

    def _load_dom(self) -> Tuple[np.ndarray, Affine, CRS, Optional[float]]:
        with rasterio.open(self.dom_path) as dom_src:
            dom_data = dom_src.read(1).astype(np.float32, copy=False)
            dom_transform = dom_src.transform
            dom_crs = _as_crs(dom_src.crs, self.crs)
            dom_nodata = dom_src.nodata
        return dom_data, dom_transform, dom_crs, dom_nodata

    def _load_or_build_dgm(
        self,
        dom_transform: Affine,
        dom_shape: Tuple[int, int],
        dom_crs: CRS,
    ) -> Tuple[np.ndarray, Affine, CRS]:
        """
        If DGM is .xyz: rasterize onto DOM grid.
        Else: load as raster.
        """
        if self.dgm_path.suffix.lower() == ".xyz":
            xyz = _read_xyz(self.dgm_path)
            dgm_data = _xyz_to_grid_min_z(xyz, dom_transform, dom_shape)
            return dgm_data, dom_transform, dom_crs

        with rasterio.open(self.dgm_path) as dgm_src:
            dgm_data = dgm_src.read(1).astype(np.float32, copy=False)
            dgm_transform = dgm_src.transform
            dgm_crs = _as_crs(dgm_src.crs, self.crs)
        return dgm_data, dgm_transform, dgm_crs

    def _reproject(
        self,
        data: np.ndarray,
        transform: Affine,
        src_crs: CRS,
        dst_crs: CRS,
        target_shape: Optional[Tuple[int, int]] = None,
        target_transform: Optional[Affine] = None,
    ) -> Tuple[np.ndarray, Affine]:
        """
        Reproject raster data to dst_crs. If target grid is provided, aligns to it.
        """
        if target_shape is not None and target_transform is not None:
            height, width = target_shape
            dst_transform = target_transform
        else:
            bounds = array_bounds(data.shape[0], data.shape[1], transform)
            dst_transform, width, height = calculate_default_transform(
                src_crs, dst_crs, data.shape[1], data.shape[0], *bounds
            )

        out = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=data,
            destination=out,
            src_transform=transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=ResamplingEnums.bilinear,
            init_dest_nodata=np.nan,
        )
        return out, dst_transform

    def _resample(
        self,
        data: np.ndarray,
        transform: Affine,
        target_res: float,
        crs: CRS,
    ) -> Tuple[np.ndarray, Affine]:
        """
        Resample to new resolution in meters. Keeps same CRS.
        """
        src_res = float(transform.a)
        if target_res <= 0:
            raise ValueError("target_res must be > 0")

        scale_factor = src_res / float(target_res)
        new_width = int(round(data.shape[1] * scale_factor))
        new_height = int(round(data.shape[0] * scale_factor))

        dst = np.full((new_height, new_width), np.nan, dtype=np.float32)
        dst_transform = transform * transform.scale(1 / scale_factor, 1 / scale_factor)

        reproject(
            source=data,
            destination=dst,
            src_transform=transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=ResamplingEnums.bilinear,
            init_dest_nodata=np.nan,
        )
        return dst, dst_transform

    def _save(self) -> None:
        if self.chm is None or self.transform is None or self._crs_obj is None:
            raise RuntimeError("CHM not generated yet. Call generate_chm() first.")

        profile = {
            "driver": "GTiff",
            "height": self.chm.shape[0],
            "width": self.chm.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": self._crs_obj,
            "transform": self.transform,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 2,
        }

        with rasterio.open(self.output_path, "w", **profile) as dst:
            dst.write(self.chm.astype(np.float32, copy=False), 1)
