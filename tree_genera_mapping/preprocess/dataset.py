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
from pathlib import Path
from typing import List, Optional
import rasterio
from rasterio.windows import Window
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
from rasterio.merge import merge as merge_tiles
import logging

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',)
logger = logging.getLogger(__name__)
# -----------------------------
class ImageDataSet:
    def __init__(self,
                 img_dir: str,
                 output_dir: str,
                 mode: str,
                 label_col: Optional[str] = None,
                 size: int = 640,
                 overlap: float = 0.0,
                 train: bool = True,
                 ):
        self.img_dir = Path(img_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.label_col = label_col
        self.mode = mode
        self.overlap = overlap
        self.size = size
        self.train = train
        self.class_map = {}
        self.class_dirs = {}

    def __len__(self):
        # counts all matching tif files under img_dir
        tif_files = list(self.img_dir.glob(f'**/{self.mode}_*.tif'))
        return len(tif_files)
    # def get_class_dir(self, class_name: str):
    #     if class_name not in self.class_dirs:
    #         class_dir = self.output_dir / 'train' / class_name
    #         class_dir.mkdir(parents=True, exist_ok=True)
    #         self.class_dirs[class_name] = class_dir
    #     return self.class_dirs[class_name]
    def split_tiff_to_tiles(self,
                            image_path:str | Path,
                            trees_gdf:gpd.GeoDataFrame,
                            ensure_full_coverage:bool = True):
        """
        Split one GeoTIFF into sub-tiles and (optionally) write YOLO label files.
        """
        image_path = Path(image_path)
        stride = int(self.size * (1 - self.overlap))
        image_name = image_path.stem

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
                    if self.train and out_lbl_dir is not None:
                        label_filename = out_lbl_dir / f"{image_name}_{tile_id}.txt"
                        labels = self.extract_labels(trees_gdf, tile_geom, tile_transform)
                        with open(label_filename, 'w') as f:
                            for label in labels:
                                f.write(label + '\n')
                        logger.info(f"Tile {tile_filename.name}: {len(labels)} labels")
                    # ---Next tile----
                    tile_id += 1

    def extract_labels(self,
                       trees_gdf:gpd.GeoDataFrame,
                       tile_geom,
                       transform):
        """
        Convert intersecting geometries to YOLO labels: class x_center y_center width height
        Assumes geometries are polygons/bboxes in same CRS as raster.
        """
        if trees_gdf.crs is None:
            raise ValueError("trees_gdf has no CRS")
        if trees_gdf.crs != tile_geom.crs:
            trees_gdf = trees_gdf.to_crs(tile_geom.crs)

        # IMPORTANT: use intersects, not within (keeps edge objects)
        tile_poly = tile_geom.geometry.iloc[0]
        intersecting = trees_gdf[trees_gdf.intersects(tile_poly)]
        # intersecting = trees_gdf[trees_gdf.geometry.within(tile_poly)] # less ideal

        labels: List[str] = []
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

    def cut_bbox_from_merged_tiles(self,
                                   image_paths: List[str | Path],
                                   geom,
                                   output_id:str,
                                   class_name="unknown"):
        """
        Cut a patch around geom.bounds from a mosaic of rasters and save it into train/<class_name>/.
        """
        class_dir = self.output_dir / 'train' / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        srcs = []
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


