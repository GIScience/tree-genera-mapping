"""
tile_dataset.py

Build merged tiles in modes:
- RGB   : 3 bands (R,G,B) from DOP
- RGBI  : 4 bands (R,G,B,NIR) from DOP
- RGBIH : 5 bands (R,G,B,NIR,Height)

Height handling (RGBIH):
- Prefer ndom_path (ndom1) if it exists.
- Otherwise compute CHM/nDSM = DOM - DGM using HeightModel, write a temp GeoTIFF, and use that.
- Warp CHM onto the RGB tile grid, then normalize to uint8 per tile using vmin/vmax on the warped grid.

Outputs:
- GeoTIFF: <output_dir>/merged/<mode>_<tile_id>.tif

Metadata exposed (for your downloader script to write JSON):
- self.last_height_stats_m  -> (vmin_m, vmax_m) used for normalization
- self.last_height_source   -> "ndom1" or "dom1-dgm1"
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from tree_genera_mapping.preprocess.height_model import HeightModel
from tree_genera_mapping.preprocess.utils import normalize_hm_to_255

# Silence noisy pyogrio FutureWarnings (only if you import geopandas elsewhere)
warnings.filterwarnings("ignore", category=FutureWarning, module="pyogrio")

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TileDataset:
    """
    Create RGB/RGBI/RGBIH tile for a single tile_id.
    """

    def __init__(
        self,
        tile_id: str,
        output_dir: str,
        mode: str = "RGBIH",
        dop_path: Optional[str] = None,
        ndom_path: Optional[str] = None,
        dgm_path: Optional[str] = None,
        dom_path: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ):
        self.tile_id = tile_id
        self.mode = mode.upper()

        self.dop_path = Path(dop_path) if dop_path else None
        self.ndom_path = Path(ndom_path) if ndom_path else None
        self.dgm_path = Path(dgm_path) if dgm_path else None
        self.dom_path = Path(dom_path) if dom_path else None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "merged" / f"{self.mode.lower()}_{tile_id}.tif"

        self.temp_dir = Path(temp_dir) if temp_dir else None

        # Exposed for JSON writing by the caller (download_generate_tiledataset.py)
        self.last_height_stats_m: Optional[Tuple[float, float]] = None
        self.last_height_source: Optional[str] = None  # "ndom1" or "dom1-dgm1"

    def process(self) -> Optional[Path]:
        """
        Write the merged tile and return the output path.
        Returns None if RGBIH height cannot be created.
        """
        if self.output_path.exists():
            logger.info("Skipping existing: %s", self.output_path)
            return self.output_path

        rgbi, transform, meta = self._read_rgbi()

        band_indices = {
            "RGB": [0, 1, 2],
            "RGBI": [0, 1, 2, 3],
            "RGBIH": [0, 1, 2, 3],
        }
        if self.mode not in band_indices:
            raise ValueError(f"Unsupported mode: {self.mode}")

        selected = rgbi[band_indices[self.mode]]  # (C,H,W)

        if self.mode == "RGBIH":
            height_u8, stats_m, source = self._get_height(
                shape=selected.shape[1:],
                transform=transform,
                target_crs=meta.get("crs"),
                res_m=float(transform.a),
            )
            if height_u8 is None:
                logger.warning("No height model for %s", self.tile_id)
                return None

            self.last_height_stats_m = stats_m
            self.last_height_source = source
            data = np.vstack([selected, height_u8[None, ...]])
        else:
            data = selected

        # Write output uint8 GeoTIFF
        meta.update({"count": data.shape[0], "dtype": "uint8", "nodata": None})
        if self.mode in ("RGB", "RGBI"):
            meta["photometric"] = "RGB"
        else:
            meta.pop("photometric", None)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(self.output_path, "w", **meta) as dst:
            dst.write(data)

        logger.info("Saved: %s", self.output_path)
        return self.output_path

    def _read_rgbi(self):
        """
        Read DOP GeoTIFF (RGBI or RGB depending on what you download).
        Returns:
          data: (bands, H, W)
          transform: rasterio Affine
          meta: rasterio profile dict
        """
        if self.dop_path is None:
            raise ValueError("dop_path is required")

        with rasterio.open(self.dop_path) as src:
            data = src.read()
            meta = src.meta.copy()
            transform = src.transform

            # Keep CRS from file; only set fallback if missing
            if meta.get("crs") is None:
                meta["crs"] = "EPSG:25832"

        return data, transform, meta

    def _get_height(self, shape, transform, target_crs, res_m: float):
        """
        Returns:
          height_u8: uint8 (H,W)
          stats_m: (vmin_m, vmax_m) used for normalization on the warped grid
          source: "ndom1" or "dom1-dgm1"
        """
        # 1) Prefer ndom1 if present
        if self.ndom_path and self.ndom_path.exists():
            height_u8, stats_m = self._load_height(self.ndom_path, shape, transform, target_crs)
            if height_u8 is not None:
                return height_u8, stats_m, "ndom1"

        # 2) Fallback: generate ndom1 from dom1-dgm1
        if self.dgm_path and self.dom_path and self.dgm_path.exists() and self.dom_path.exists():
            temp_dir = (self.temp_dir / "chm" / self.tile_id) if self.temp_dir else (self.output_path.parent / "chm_temp" / self.tile_id)
            temp_dir.mkdir(parents=True, exist_ok=True)

            ndsm_path = self._process_hm(
                key=self.tile_id,
                dgm_file=self.dgm_path,
                dom_file=self.dom_path,
                output_dir=temp_dir,
                res=res_m,
                crs=str(target_crs) if target_crs else "EPSG:25832",
            )

            height_u8, stats_m = self._load_height(ndsm_path, shape, transform, target_crs)
            if height_u8 is not None:
                return height_u8, stats_m, "dom1-dgm1"

        return None, None, None

    def _load_height(self, path: Path, shape, transform, target_crs):
        """
        Warp CHM/nDSM raster onto RGB grid, then normalize to uint8 using per-tile vmin/vmax
        computed *after* warping. This makes JSON unnormalization exact.

        Returns:
          height_u8: uint8 (H,W) or None
          stats_m: (vmin_m, vmax_m) or None
        """
        target_h, target_w = shape

        with rasterio.open(path) as src:
            src_data = src.read(1).astype(np.float32, copy=False)
            src_transform = src.transform
            src_crs = src.crs

        dst = np.empty((target_h, target_w), dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=target_crs if target_crs is not None else src_crs,
            resampling=Resampling.bilinear,
        )

        if not np.isfinite(dst).any():
            logger.warning("CHM %s reprojected to all-NaN; treating as missing.", path)
            return None, None

        vmin = float(np.nanmin(dst))
        vmax = float(np.nanmax(dst))
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            return np.zeros((target_h, target_w), dtype=np.uint8), (0.0, 0.0)

        height_u8 = normalize_hm_to_255(dst, vmin, vmax)
        return height_u8, (vmin, vmax)

    @staticmethod
    def _process_hm(key: str, dgm_file: Path, dom_file: Path, output_dir: Path, res: float, crs: str) -> Path:
        """
        Create CHM/nDSM via HeightModel and return produced raster path.
        """
        hm = HeightModel(
            dgm_path=dgm_file,
            dom_path=dom_file,
            key=key,
            output_dir=output_dir,
            res=res,
            crs=crs,
        )
        hm.generate_chm()
        return hm.output_dir / hm.file_name
