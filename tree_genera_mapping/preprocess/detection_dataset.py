"""
dataset.py

Utilities to:
1) Split GeoTIFF tiles into fixed-size sub-tiles (e.g., 640x640)
2) Generate YOLO-format labels (.txt) from vector geometries (bbox polygons recommended)
3) Optionally cut a patch from a mosaic of overlapping rasters

Assumptions:
- Training geometries should be polygons (bbox polygons for YOLO are ideal).
- CRS should be projected (meters) for consistent geometry operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Set

import geopandas as gpd
import rasterio
from rasterio.merge import merge as merge_tiles
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ImageDataSet:
    """
    Splits a parent GeoTIFF into fixed-size chips and writes YOLO label files.

    Output layout:
      <output_dir>/
        images/{train,val,test}/rgbih_32_..._<n>.tif
        labels/{train,val,test}/rgbih_32_..._<n>.txt
    """

    def __init__(
        self,
        img_dir: str | Path,
        output_dir: str | Path,
        mode: str,
        label_col: Optional[str] = None,
        size: int = 640,
        overlap: float = 0.0,
        split: str = "train",  # train|val|test
    ):
        self.img_dir = Path(img_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.label_col = label_col
        self.mode = mode
        self.overlap = overlap
        self.size = size

        self.split = split  # default split if not overridden per call

    def __len__(self) -> int:
        tif_files = list(self.img_dir.glob(f"**/{self.mode}_*.tif"))
        return len(tif_files)

    @staticmethod
    def _validate_split(split: str) -> str:
        split = str(split).strip().lower()
        if split == "valid":
            split = "val"
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train|val|test, got: {split}")
        return split

    @staticmethod
    def _subtile_id_from_chip_stem(chip_stem: str) -> str:
        """
        chip_stem example: rgbih_32_464_5487_91
        returns:          32_464_5487_91   (drop mode)
        """
        parts = chip_stem.split("_")
        if len(parts) < 5:
            raise ValueError(f"Unexpected chip stem: {chip_stem}")
        return "_".join(parts[1:])

    def split_tiff_to_tiles(
        self,
        image_path: str | Path,
        trees_gdf: gpd.GeoDataFrame,
        ensure_full_coverage: bool = True,
        split: Optional[str] = None,
        subtile_whitelist: Optional[Set[str]] = None,
        write_empty_labels: bool = True,
    ) -> None:
        """
        Split one GeoTIFF into sub-tiles and write YOLO label files.

        Parameters
        ----------
        split:
          Overrides self.split for this call.
        subtile_whitelist:
          If provided, keep only chips whose subtile_id (without mode) is in the set.
          Example allowed id: "32_464_5487_91"
        write_empty_labels:
          If False, will skip writing label files for chips with 0 labels.
          (images still written)
        """
        image_path = Path(image_path)
        split = self._validate_split(split or self.split)

        stride = int(self.size * (1 - self.overlap))
        if stride <= 0:
            raise ValueError(f"Invalid stride={stride}. Check size={self.size}, overlap={self.overlap}")

        image_name = image_path.stem  # e.g. rgbih_32_464_5487

        out_img_dir = self.output_dir / "images" / split
        out_lbl_dir = self.output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(image_path) as src:
            width, height = src.width, src.height
            transform = src.transform
            crs = src.crs
            tile_n = 0

            # row starts
            if ensure_full_coverage:
                y_starts = list(range(0, max(0, height - self.size), stride))
                if not y_starts:
                    y_starts = [0]
                if y_starts[-1] + self.size < height:
                    y_starts.append(height - self.size)
            else:
                y_starts = list(range(0, max(0, height - self.size + 1), stride))
                if not y_starts:
                    y_starts = [0]

            # col starts
            if ensure_full_coverage:
                x_starts = list(range(0, max(0, width - self.size), stride))
                if not x_starts:
                    x_starts = [0]
                if x_starts[-1] + self.size < width:
                    x_starts.append(width - self.size)
            else:
                x_starts = list(range(0, max(0, width - self.size + 1), stride))
                if not x_starts:
                    x_starts = [0]

            for i in y_starts:
                for j in x_starts:
                    window = Window(j, i, self.size, self.size)
                    tile_transform = src.window_transform(window)
                    tile_bounds = rasterio.windows.bounds(window, transform)

                    tile_geom = gpd.GeoSeries([box(*tile_bounds).buffer(0.01)], crs=crs)

                    # chip filename
                    chip_stem = f"{image_name}_{tile_n}"  # rgbih_32_..._<n>
                    subtile_id = self._subtile_id_from_chip_stem(chip_stem)

                    # optional filter
                    if subtile_whitelist is not None and subtile_id not in subtile_whitelist:
                        tile_n += 1
                        continue

                    tile_data = src.read(window=window, boundless=True, fill_value=0)
                    tile_meta = src.meta.copy()
                    tile_meta.update(
                        {
                            "height": self.size,
                            "width": self.size,
                            "transform": tile_transform,
                            "driver": "GTiff",
                        }
                    )

                    # write image
                    tile_filename = out_img_dir / f"{chip_stem}.tif"
                    with rasterio.open(tile_filename, "w", **tile_meta) as dst:
                        dst.write(tile_data)

                    # write labels
                    labels = self.extract_labels(trees_gdf, tile_geom, tile_transform)
                    if write_empty_labels or len(labels) > 0:
                        label_filename = out_lbl_dir / f"{chip_stem}.txt"
                        with open(label_filename, "w", encoding="utf-8") as f:
                            for lab in labels:
                                f.write(lab + "\n")

                    logger.info("Chip %s (%s): %d labels", tile_filename.name, split, len(labels))
                    tile_n += 1

    def extract_labels(self, trees_gdf: gpd.GeoDataFrame, tile_geom: gpd.GeoSeries, transform) -> List[str]:
        """
        Convert intersecting geometries to YOLO labels:
          class x_center y_center width height
        Assumes geometries are polygons/bboxes in same CRS as raster.
        """
        if trees_gdf is None:
            return []
        if len(trees_gdf) == 0:
            return []
        if trees_gdf.crs is None:
            raise ValueError("trees_gdf has no CRS")
        if trees_gdf.crs != tile_geom.crs:
            trees_gdf = trees_gdf.to_crs(tile_geom.crs)

        tile_poly = tile_geom.geometry.iloc[0]
        intersecting = trees_gdf[trees_gdf.intersects(tile_poly)]

        labels: List[str] = []
        for _, row in intersecting.iterrows():
            geom = row.geometry
            if not isinstance(geom, BaseGeometry) or geom.is_empty:
                continue

            bounds = geom.bounds
            try:
                x_min, y_min = ~transform * (bounds[0], bounds[1])
                x_max, y_max = ~transform * (bounds[2], bounds[3])
            except Exception as e:
                logger.warning("Transform error: %s", e)
                continue

            x_center = ((x_min + x_max) / 2) / self.size
            y_center = ((y_min + y_max) / 2) / self.size
            width = abs(x_max - x_min) / self.size
            height = abs(y_max - y_min) / self.size

            if 0 <= x_center <= 1 and 0 <= y_center <= 1:
                label = 0
                if self.label_col and (self.label_col in row):
                    try:
                        label = int(row[self.label_col])
                    except Exception as e:
                        logger.warning("Invalid label in '%s': %s", self.label_col, e)

                labels.append(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return labels

    def cut_bbox_from_merged_tiles(
        self,
        image_paths: List[str | Path],
        geom,
        output_id: str,
        class_name: str = "unknown",
        split: str = "train",
    ) -> None:
        """
        Cut a patch around geom.bounds from a mosaic of rasters and save it into:
          <output_dir>/{split}/{class_name}/tree_<output_id>.tif
        """
        split = self._validate_split(split)

        class_dir = self.output_dir / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        srcs = []
        try:
            srcs = [rasterio.open(p) for p in image_paths]
            mosaic, out_transform = merge_tiles(srcs)

            meta = srcs[0].meta.copy()
            meta.update(
                {
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform,
                }
            )

            bounds = geom.bounds
            window = rasterio.windows.from_bounds(*bounds, transform=out_transform)
            window = window.round_offsets().round_shape()

            tile_data = mosaic[
                :,
                int(window.row_off) : int(window.row_off + window.height),
                int(window.col_off) : int(window.col_off + window.width),
            ]

            tile_transform = rasterio.windows.transform(window, out_transform)
            tile_meta = meta.copy()
            tile_meta.update(
                {
                    "height": tile_data.shape[1],
                    "width": tile_data.shape[2],
                    "transform": tile_transform,
                }
            )

            tile_filename = class_dir / f"tree_{output_id}.tif"
            with rasterio.open(tile_filename, "w", **tile_meta) as dst:
                dst.write(tile_data)

            logger.info("Saved bbox tile: %s (split=%s class=%s)", tile_filename.name, split, class_name)

        except Exception as e:
            logger.warning("Error merging/cutting for bbox %s: %s", output_id, e)
        finally:
            for s in srcs:
                try:
                    s.close()
                except Exception:
                    pass