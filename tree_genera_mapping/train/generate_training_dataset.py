#!/usr/bin/env python3
"""
generate_training_dataset.py

Generate training data for two model types:

1) YOLO detection dataset (chips + bbox labels)
   - Input: weak bboxes GeoPackage (polygons or boxes)
   - Input: tiles GeoPackage (tile footprints)
   - Input: merged raster tiles folder with:
       <images-dir>/<mode>_<tile_id>.tif
     e.g. dir/rgbih_32_355_6048.tif

2) Image classification dataset (patches around labeled trees)
   - Input: genus labels GeoPackage (points or polygons) with class column
   - Input: tiles GeoPackage (tile footprints)
   - Input: merged raster tiles folder (same pattern as above)

Notes
- No hardcoded paths. Everything is provided via CLI.
- Assumes your tile filenames use tile_id like: 32_355_6048
- If tiles.gpkg uses dop_kachel like: 323556048 (string),
  we can derive tile_id via helper below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from tree_genera_mapping.preprocess.dataset import ImageDataSet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -----------------------------
# helpers
# -----------------------------

def dopkachel_to_tile_id(dop_kachel: str) -> str:
    """
    Convert BW dop_kachel string like '323556048' -> '32_355_6048'
    """
    s = str(dop_kachel)
    if len(s) < 9:
        raise ValueError(f"dop_kachel looks too short: {dop_kachel}")
    return f"{s[:2]}_{s[2:5]}_{s[-4:]}"


def find_tile_raster(images_dir: Path, mode: str, tile_id: str) -> Path:
    """
    Expected merged tile naming:
      <images_dir>/<mode>_<tile_id>.tif
    """
    return images_dir / f"{mode}_{tile_id}.tif"


def ensure_same_crs(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if a.crs is None or b.crs is None:
        raise ValueError("Both GeoDataFrames must have CRS set.")
    if a.crs != b.crs:
        b = b.to_crs(a.crs)
    return a, b


# -----------------------------
# 1) YOLO detection dataset
# -----------------------------

def make_detection_dataset(
    *,
    tiles_gpkg: str,
    weak_bboxes_gpkg: str,
    images_dir: str,
    output_dir: str,
    mode: str,
    tile_id_col: str = "tile_id",
    size: int = 640,
    overlap: float = 0.0,
) -> None:
    """
    Builds YOLO-ready dataset from weak bboxes.

    - tiles_gpkg: tile polygons
    - weak_bboxes_gpkg: bbox polygons with at least geometry + (optionally tile id)
    - images_dir: directory containing dir/ tiles
    - output_dir: destination
    - mode: 'rgb', 'rgbi', 'rgbih'
    """
    images_dir_p = Path(images_dir)
    output_dir_p = Path(output_dir) / f"yolo_{mode}"
    output_dir_p.mkdir(parents=True, exist_ok=True)

    gdf_tiles = gpd.read_file(tiles_gpkg)
    gdf_tree = gpd.read_file(weak_bboxes_gpkg)

    if gdf_tiles.empty:
        raise ValueError(f"No tiles found in {tiles_gpkg}")
    if gdf_tree.empty:
        raise ValueError(f"No bboxes found in {weak_bboxes_gpkg}")

    gdf_tiles, gdf_tree = ensure_same_crs(gdf_tiles, gdf_tree)

    # Find tiles that intersect with any bbox
    matching_idxs = None
    try:
        matching_idxs = gpd.sjoin(gdf_tiles, gdf_tree, how="inner", predicate="intersects").index.unique()
    except Exception:
        # fallback: if weak bboxes already have tile_id, filter by that
        if tile_id_col in gdf_tree.columns and tile_id_col in gdf_tiles.columns:
            tiles = set(gdf_tree[tile_id_col].astype(str).unique())
            matching_idxs = gdf_tiles[gdf_tiles[tile_id_col].astype(str).isin(tiles)].index.unique()
        else:
            raise RuntimeError(
                "Spatial join failed and cannot fallback by tile id column. "
                f"Check CRS and ensure '{tile_id_col}' exists in both files."
            )

    gdf_tiles_filtered = gdf_tiles.loc[matching_idxs].copy()
    if gdf_tiles_filtered.empty:
        logger.warning("No tiles intersect with bboxes. Nothing to do.")
        return

    # Ensure we have tile_id in tile table. If only dop_kachel exists, derive tile_id.
    if tile_id_col not in gdf_tiles_filtered.columns:
        if "dop_kachel" in gdf_tiles_filtered.columns:
            gdf_tiles_filtered[tile_id_col] = gdf_tiles_filtered["dop_kachel"].astype(str).apply(dopkachel_to_tile_id)
        else:
            raise ValueError(f"Tiles file must contain '{tile_id_col}' or 'dop_kachel'.")

    dataset = ImageDataSet(
        img_dir=images_dir_p,
        output_dir=output_dir_p,
        mode=mode,
        size=size,
        overlap=overlap,
    )

    logger.info("Processing %d tiles for detection dataset...", len(gdf_tiles_filtered))

    for _, row in tqdm(gdf_tiles_filtered.iterrows(), total=len(gdf_tiles_filtered), desc="Detection tiles"):
        tile_id = str(row[tile_id_col])
        tile_path = find_tile_raster(images_dir_p, mode, tile_id)

        if not tile_path.exists():
            logger.warning("Missing tile raster: %s (skipping)", tile_path)
            continue

        # Your ImageDataSet is expected to crop chips + write bbox labels
        dataset.split_tiff_to_tiles(tile_path, gdf_tree)

    logger.info("✅ Detection dataset written to: %s", output_dir_p)


# -----------------------------
# 2) Classification patches
# -----------------------------

def make_classification_patches(
    *,
    tiles_gpkg: str,
    genus_labels_gpkg: str,
    images_dir: str,
    output_dir: str,
    mode: str,
    patch_size: int = 128,
    tile_id_col: str = "tile_id",     # or "dop_kachel" in tiles
    class_col: str = "training_class",
    id_col: str = "uuid",
) -> None:
    """
    Extract patches around labeled trees.

    - genus_labels_gpkg: must contain geometry + class_col (+ id_col optional)
    - tiles_gpkg: tile polygons with tile_id_col or dop_kachel
    """
    images_dir_p = Path(images_dir)
    output_dir_p = Path(output_dir) / f"patches_{mode}_{patch_size}"
    output_dir_p.mkdir(parents=True, exist_ok=True)

    gdf_tiles = gpd.read_file(tiles_gpkg)
    gdf_tree = gpd.read_file(genus_labels_gpkg)

    if gdf_tiles.empty:
        raise ValueError(f"No tiles found in {tiles_gpkg}")
    if gdf_tree.empty:
        raise ValueError(f"No genus labels found in {genus_labels_gpkg}")
    if class_col not in gdf_tree.columns:
        raise ValueError(f"genus_labels_gpkg is missing class column '{class_col}'")

    gdf_tiles, gdf_tree = ensure_same_crs(gdf_tiles, gdf_tree)

    # Ensure we have tile_id usable for filenames
    if tile_id_col not in gdf_tiles.columns:
        if "dop_kachel" in gdf_tiles.columns:
            gdf_tiles[tile_id_col] = gdf_tiles["dop_kachel"].astype(str).apply(dopkachel_to_tile_id)
        else:
            raise ValueError(f"Tiles file must contain '{tile_id_col}' or 'dop_kachel'.")

    # Use centroids for polygons; points stay points
    gdf_tree = gdf_tree.copy()
    gdf_tree["__pt__"] = gdf_tree.geometry.centroid

    half = int(patch_size // 2)

    # Spatial join points to tiles once (fast)
    pts = gpd.GeoDataFrame(gdf_tree[[class_col] + ([id_col] if id_col in gdf_tree.columns else [])].copy(),
                           geometry=gdf_tree["__pt__"], crs=gdf_tree.crs)
    joined = gpd.sjoin(pts, gdf_tiles[[tile_id_col, "geometry"]], how="left", predicate="within")

    missing = joined[tile_id_col].isna().sum()
    if missing > 0:
        logger.info("Points not matched to any tile: %d (they will be skipped)", int(missing))

    for idx, row in tqdm(joined.iterrows(), total=len(joined), desc="Classification patches"):
        tile_id = row.get(tile_id_col, None)
        if tile_id is None or (isinstance(tile_id, float) and tile_id != tile_id):
            continue

        class_name = str(row.get(class_col, "unknown")).strip().replace(" ", "_")
        out_id = row.get(id_col, idx) if id_col in row else idx

        tile_path = find_tile_raster(images_dir_p, mode, str(tile_id))
        if not tile_path.exists():
            continue

        class_dir = output_dir_p / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        patch_path = class_dir / f"{out_id}.tif"
        if patch_path.exists():
            continue

        pt = row.geometry
        if pt is None or pt.is_empty:
            continue

        try:
            with rasterio.open(tile_path) as src:
                r, c = src.index(pt.x, pt.y)

                win = Window(c - half, r - half, patch_size, patch_size)

                # bounds check
                if win.col_off < 0 or win.row_off < 0:
                    continue
                if (win.col_off + win.width > src.width) or (win.row_off + win.height > src.height):
                    continue

                patch = src.read(window=win)
                transform = src.window_transform(win)
                meta = src.meta.copy()
                meta.update(height=patch.shape[1], width=patch.shape[2], transform=transform)

                with rasterio.open(patch_path, "w", **meta) as dst:
                    dst.write(patch)

        except Exception as e:
            logger.warning("Failed patch %s from tile %s: %s", out_id, tile_path.name, e)

    logger.info("✅ Classification patches written to: %s", output_dir_p)


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate training datasets: YOLO detection + classification patches.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Detection
    ap_det = sub.add_parser("det", help="Generate YOLO detection dataset from weak bbox labels")
    ap_det.add_argument("--tiles-gpkg", required=True)
    ap_det.add_argument("--weak-bboxes-gpkg", required=True)
    ap_det.add_argument("--images-dir", required=True, help="Directory containing merged/ tiles")
    ap_det.add_argument("--output-dir", required=True)
    ap_det.add_argument("--mode", required=True, help="Tile prefix in filenames, e.g. rgbih, rgbi, rgb")
    ap_det.add_argument("--tile-id-col", default="tile_id", help="Tile id column in tiles/bboxes gpkg (fallback)")
    ap_det.add_argument("--size", type=int, default=640)
    ap_det.add_argument("--overlap", type=float, default=0.0)

    # Classification
    ap_cls = sub.add_parser("patches", help="Generate classification patches from genus labels")
    ap_cls.add_argument("--tiles-gpkg", required=True)
    ap_cls.add_argument("--genus-labels-gpkg", required=True)
    ap_cls.add_argument("--images-dir", required=True, help="Directory containing merged/ tiles")
    ap_cls.add_argument("--output-dir", required=True)
    ap_cls.add_argument("--mode", required=True, help="Tile prefix in filenames, e.g. rgbih, rgbi, rgb")
    ap_cls.add_argument("--patch-size", type=int, default=128)
    ap_cls.add_argument("--tile-id-col", default="tile_id", help="Column in tiles.gpkg or derived from dop_kachel")
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
