#!/usr/bin/env python3
"""
build_dataset.py

Builds reproducible, leakage-safe ML datasets from geospatial inputs using
STRICT predefined tile splits (no random splitting).

It supports two outputs:

1) YOLO detection dataset
   - chips (subtiles) + YOLO bbox labels
   - output layout:
       <out>/yolo_<mode>/
         images/{train,val,test}/
         labels/{train,val,test}/

2) Classification patches dataset
   - patches around labeled trees for genus classification
   - output layout:
       <out>/patches_<mode>_<patch>/
         {train,val,test}/<class_name>/*.tif

Key design choice (important for geospatial ML):
- Splits are defined at the NON-overlapping 1×1 km parent-tile level.
- Subtiles may overlap (e.g., 20%) WITHIN each split to reduce edge effects,
  but no content overlaps ACROSS splits because a parent tile belongs to only one split.

Requirements:
- You MUST provide predefined tile lists for train/val/test as text files
  (one tile_id per line). The script will error if:
    * a tile appears in multiple splits
    * a tile from lists is not present in the available tiles layer
    * any available tile (optionally) is unassigned (you can relax this)

Notes:
- This script calls your existing ImageDataSet class for YOLO chip writing.
  It assumes ImageDataSet.split_tiff_to_tiles(...) either:
    (A) accepts a `split=` argument, OR
    (B) respects ds.split attribute ("train"/"val"/"test").
  If neither is true, adjust ImageDataSet accordingly (recommended).

Example usage is at the bottom of this file (see "HOW TO RUN").
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from tree_genera_mapping.preprocess.detection_dataset import ImageDataSet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def dopkachel_to_tile_id(dop_kachel: str) -> str:
    """Convert BW dop_kachel string like '323556048' -> '32_355_6048'."""
    s = str(dop_kachel).strip()
    if len(s) < 9:
        raise ValueError(f"dop_kachel looks too short: {dop_kachel}")
    return f"{s[:2]}_{s[2:5]}_{s[-4:]}"


def ensure_same_crs(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if a.crs is None or b.crs is None:
        raise ValueError("Both GeoDataFrames must have CRS set.")
    if a.crs != b.crs:
        b = b.to_crs(a.crs)
    return a, b


def _read_lines(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _ensure_tile_id_column(gdf_tiles: gpd.GeoDataFrame, tile_id_col: str) -> gpd.GeoDataFrame:
    gdf_tiles = gdf_tiles.copy()
    if tile_id_col not in gdf_tiles.columns:
        if "dop_kachel" in gdf_tiles.columns:
            gdf_tiles[tile_id_col] = gdf_tiles["dop_kachel"].astype(str).apply(dopkachel_to_tile_id)
        elif "dopkachel" in gdf_tiles.columns:
            gdf_tiles[tile_id_col] = gdf_tiles["dopkachel"].astype(str).apply(dopkachel_to_tile_id)
        else:
            raise ValueError(f"Tiles layer must contain '{tile_id_col}' or 'dop_kachel'/'dopkachel'.")
    gdf_tiles[tile_id_col] = gdf_tiles[tile_id_col].astype(str)
    return gdf_tiles


def make_tile_split_strict(
    available_tile_ids: Sequence[str],
    *,
    train_list: Sequence[str],
    val_list: Sequence[str],
    test_list: Sequence[str],
    require_complete: bool = False,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Strict split: only predefined lists, no random split.
    - require_complete=False: it's OK if some available tiles are unassigned (ignored).
    - require_complete=True: error if any available tile isn't in any split list.
    """
    all_set = set(map(str, available_tile_ids))
    train_set = set(map(str, train_list))
    val_set = set(map(str, val_list))
    test_set = set(map(str, test_list))

    # Overlap checks (leakage prevention)
    if train_set & val_set:
        raise ValueError(f"train and val overlap (example): {sorted(train_set & val_set)[:10]}")
    if train_set & test_set:
        raise ValueError(f"train and test overlap (example): {sorted(train_set & test_set)[:10]}")
    if val_set & test_set:
        raise ValueError(f"val and test overlap (example): {sorted(val_set & test_set)[:10]}")

    # Unknown IDs in lists
    missing = (train_set | val_set | test_set) - all_set
    if missing:
        raise ValueError(
            "Split lists contain tile_ids not present in tiles layer "
            f"(example): {sorted(list(missing))[:10]}"
        )

    if require_complete:
        unassigned = all_set - (train_set | val_set | test_set)
        if unassigned:
            raise ValueError(
                f"{len(unassigned)} available tiles are not assigned to any split "
                f"(example): {sorted(list(unassigned))[:10]}"
            )

    return train_set, val_set, test_set


def find_tile_raster(images_dir: Path, mode: str, tile_id: str) -> Path:
    """
    Expected naming: <images_dir>/<mode>_<tile_id>.tif
    e.g. rgbih_32_355_6048.tif
    """
    return images_dir / f"{mode}_{tile_id}.tif"


def _call_split_tiff_to_tiles(ds: ImageDataSet, tile_path: Path, gdf_labels: gpd.GeoDataFrame, split: str) -> None:
    """
    Calls ImageDataSet.split_tiff_to_tiles with best-effort compatibility:
    - Prefer split= argument if method supports it.
    - Else set ds.split attribute and call without split.
    """
    fn = ds.split_tiff_to_tiles
    if "split" in fn.__code__.co_varnames:
        fn(tile_path, gdf_labels, split=split)
        return
    # fallback
    if hasattr(ds, "split"):
        setattr(ds, "split", split)
    fn(tile_path, gdf_labels)


def _validate_overlap(overlap: float) -> None:
    if not (0.0 <= overlap < 1.0):
        raise ValueError("--overlap must be in [0.0, 1.0). Example: 0.2 for 20% overlap.")


# -----------------------------------------------------------------------------
# 1) YOLO detection dataset
# -----------------------------------------------------------------------------
def make_detection_dataset(
    *,
    tiles_gpkg: str,
    bboxes_gpkg: str,
    images_dir: str,
    output_dir: str,
    mode: str,
    tile_id_col: str,
    size: int,
    overlap: float,
    train_tiles_txt: str,
    val_tiles_txt: str,
    test_tiles_txt: str,
    include_empty_tiles: bool = True,
    require_complete: bool = False,
) -> None:
    """
    Builds YOLO-ready dataset (chips + bbox labels), using STRICT predefined tile lists.
    """
    _validate_overlap(overlap)

    images_dir_p = Path(images_dir)
    if not images_dir_p.exists():
        raise FileNotFoundError(images_dir_p)

    out_root = Path(output_dir) / f"yolo_{mode}"
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    gdf_tiles = gpd.read_file(tiles_gpkg)
    gdf_boxes = gpd.read_file(bboxes_gpkg)

    if gdf_tiles.empty:
        raise ValueError(f"No tiles found in {tiles_gpkg}")
    if gdf_boxes.empty:
        logger.warning("No bboxes found in %s. You will only get empty/negative tiles.", bboxes_gpkg)

    gdf_tiles, gdf_boxes = ensure_same_crs(gdf_tiles, gdf_boxes)
    gdf_tiles = _ensure_tile_id_column(gdf_tiles, tile_id_col)

    available_tile_ids = gdf_tiles[tile_id_col].astype(str).tolist()
    train_list = _read_lines(train_tiles_txt)
    val_list = _read_lines(val_tiles_txt)
    test_list = _read_lines(test_tiles_txt)

    train_tiles, val_tiles, test_tiles = make_tile_split_strict(
        available_tile_ids,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
        require_complete=require_complete,
    )

    logger.info(
        "Parent-tile split (STRICT): train=%d val=%d test=%d (available=%d)",
        len(train_tiles), len(val_tiles), len(test_tiles), len(set(available_tile_ids))
    )
    logger.info("Subtile params: size=%d overlap=%.2f", size, overlap)
    if include_empty_tiles:
        logger.info("include_empty_tiles=True (negative tiles will be written if ImageDataSet supports it)")

    # Keep only tiles that are in any split list (ignore other tiles)
    allowed = train_tiles | val_tiles | test_tiles
    gdf_tiles = gdf_tiles[gdf_tiles[tile_id_col].astype(str).isin(allowed)].copy()

    # Instantiate one dataset writer per split
    ds_train = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, mode=mode, size=size, overlap=overlap)
    ds_val = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, mode=mode, size=size, overlap=overlap)
    ds_test = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, mode=mode, size=size, overlap=overlap)

    # Process tiles
    for _, row in tqdm(gdf_tiles.iterrows(), total=len(gdf_tiles), desc="YOLO tiles"):
        tile_id = str(row[tile_id_col])
        tile_path = find_tile_raster(images_dir_p, mode, tile_id)

        if not tile_path.exists():
            logger.warning("Missing tile raster: %s (skipping)", tile_path)
            continue

        # Select labels for this tile if possible (fast filter if tile_id exists in labels)
        tile_labels = gdf_boxes
        if tile_id_col in gdf_boxes.columns:
            tile_labels = gdf_boxes[gdf_boxes[tile_id_col].astype(str) == tile_id]
        else:
            # fallback: spatial filter (slower)
            try:
                geom = row.geometry
                tile_labels = gdf_boxes[gdf_boxes.intersects(geom)]
            except Exception:
                pass

        if (tile_labels is None or len(tile_labels) == 0) and not include_empty_tiles:
            continue

        if tile_id in test_tiles:
            _call_split_tiff_to_tiles(ds_test, tile_path, tile_labels, "test")
        elif tile_id in val_tiles:
            _call_split_tiff_to_tiles(ds_val, tile_path, tile_labels, "val")
        else:
            _call_split_tiff_to_tiles(ds_train, tile_path, tile_labels, "train")

    logger.info("✅ YOLO detection dataset written to: %s", out_root)
    logger.info("   images/: %s", out_root / "images")
    logger.info("   labels/: %s", out_root / "labels")


# -----------------------------------------------------------------------------
# 2) Classification patches dataset
# -----------------------------------------------------------------------------
def make_classification_patches(
    *,
    tiles_gpkg: str,
    genus_labels_gpkg: str,
    images_dir: str,
    output_dir: str,
    mode: str,
    patch_size: int,
    tile_id_col: str,
    class_col: str,
    id_col: str,
    train_tiles_txt: str,
    val_tiles_txt: str,
    test_tiles_txt: str,
    require_complete: bool = False,
) -> None:
    """
    Extract classification patches around labeled trees into per-class folders,
    using STRICT predefined tile splits.
    """
    if patch_size % 2 != 0:
        raise ValueError("--patch-size must be even for centered windows.")

    images_dir_p = Path(images_dir)
    if not images_dir_p.exists():
        raise FileNotFoundError(images_dir_p)

    out_root = Path(output_dir) / f"patches_{mode}_{patch_size}"
    for split in ("train", "val", "test"):
        (out_root / split).mkdir(parents=True, exist_ok=True)

    gdf_tiles = gpd.read_file(tiles_gpkg)
    gdf_tree = gpd.read_file(genus_labels_gpkg)

    if gdf_tiles.empty:
        raise ValueError(f"No tiles found in {tiles_gpkg}")
    if gdf_tree.empty:
        raise ValueError(f"No genus labels found in {genus_labels_gpkg}")
    if class_col not in gdf_tree.columns:
        raise ValueError(f"Labels file missing class column '{class_col}'")

    gdf_tiles, gdf_tree = ensure_same_crs(gdf_tiles, gdf_tree)
    gdf_tiles = _ensure_tile_id_column(gdf_tiles, tile_id_col)

    available_tile_ids = gdf_tiles[tile_id_col].astype(str).tolist()
    train_list = _read_lines(train_tiles_txt)
    val_list = _read_lines(val_tiles_txt)
    test_list = _read_lines(test_tiles_txt)

    train_tiles, val_tiles, test_tiles = make_tile_split_strict(
        available_tile_ids,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
        require_complete=require_complete,
    )

    logger.info(
        "Parent-tile split (STRICT): train=%d val=%d test=%d",
        len(train_tiles), len(val_tiles), len(test_tiles)
    )

    # Join each tree instance to its parent tile
    trees = gdf_tree.copy()
    if trees.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        trees["__pt__"] = trees.geometry.centroid
    else:
        trees["__pt__"] = trees.geometry

    pts_cols = [class_col]
    if id_col in trees.columns:
        pts_cols.append(id_col)

    pts = gpd.GeoDataFrame(trees[pts_cols].copy(), geometry=trees["__pt__"], crs=trees.crs)
    joined = gpd.sjoin(pts, gdf_tiles[[tile_id_col, "geometry"]], how="left", predicate="within")

    missing = int(joined[tile_id_col].isna().sum())
    if missing > 0:
        logger.warning("Tree points not matched to any tile: %d (skipping those)", missing)

    half = patch_size // 2

    for idx, row in tqdm(joined.iterrows(), total=len(joined), desc="Genus patches"):
        tile_id = row.get(tile_id_col, None)
        if tile_id is None or (isinstance(tile_id, float) and np.isnan(tile_id)):
            continue
        tile_id = str(tile_id)

        if tile_id in test_tiles:
            split = "test"
        elif tile_id in val_tiles:
            split = "val"
        elif tile_id in train_tiles:
            split = "train"
        else:
            # tile exists but not in lists (only possible if require_complete=False and you pass smaller lists)
            continue

        class_name = str(row.get(class_col, "unknown")).strip().replace(" ", "_")
        out_id = row.get(id_col, idx) if id_col in row else idx

        tile_path = find_tile_raster(images_dir_p, mode, tile_id)
        if not tile_path.exists():
            continue

        class_dir = out_root / split / class_name
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

    logger.info("✅ Classification patches written to: %s", out_root)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build ML datasets with STRICT predefined tile splits (no random split)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_split_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--train-tiles", required=True, help="TXT list of train tile_ids (one per line)")
        p.add_argument("--val-tiles", required=True, help="TXT list of val tile_ids (one per line)")
        p.add_argument("--test-tiles", required=True, help="TXT list of test tile_ids (one per line)")
        p.add_argument(
            "--require-complete",
            action="store_true",
            help="Error if any available tile is not assigned to a split.",
        )

    # Detection (YOLO)
    ap_det = sub.add_parser("det", help="Generate YOLO detection dataset from bbox labels")
    ap_det.add_argument("--tiles-gpkg", required=True, help="GPKG with parent tile polygons")
    ap_det.add_argument("--bboxes-gpkg", required=True, help="GPKG with bbox labels (and ideally tile_id column)")
    ap_det.add_argument("--images-dir", required=True, help="Directory with parent tile rasters: <mode>_<tile_id>.tif")
    ap_det.add_argument("--output-dir", required=True)
    ap_det.add_argument("--mode", required=True, help="Filename prefix, e.g. rgbih (expects rgbih_<tile_id>.tif)")
    ap_det.add_argument("--tile-id-col", default="tile_id")
    ap_det.add_argument("--size", type=int, default=640)
    ap_det.add_argument("--overlap", type=float, default=0.2, help="Fractional overlap (0.2 = 20%)")
    ap_det.add_argument(
        "--include-empty-tiles",
        action="store_true",
        help="Also write negative tiles with no bboxes (recommended for detection).",
    )
    ap_det.set_defaults(include_empty_tiles=True)
    add_split_args(ap_det)

    # Classification patches
    ap_cls = sub.add_parser("patches", help="Generate genus classification patches")
    ap_cls.add_argument("--tiles-gpkg", required=True)
    ap_cls.add_argument("--genus-labels-gpkg", required=True)
    ap_cls.add_argument("--images-dir", required=True)
    ap_cls.add_argument("--output-dir", required=True)
    ap_cls.add_argument("--mode", required=True)
    ap_cls.add_argument("--patch-size", type=int, default=128)
    ap_cls.add_argument("--tile-id-col", default="tile_id")
    ap_cls.add_argument("--class-col", default="top1", help="Genus class column (e.g. top1)")
    ap_cls.add_argument("--id-col", default="uuid")
    add_split_args(ap_cls)

    args = ap.parse_args()

    if args.cmd == "det":
        make_detection_dataset(
            tiles_gpkg=args.tiles_gpkg,
            bboxes_gpkg=args.bboxes_gpkg,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            tile_id_col=args.tile_id_col,
            size=args.size,
            overlap=args.overlap,
            train_tiles_txt=args.train_tiles,
            val_tiles_txt=args.val_tiles,
            test_tiles_txt=args.test_tiles,
            include_empty_tiles=args.include_empty_tiles,
            require_complete=args.require_complete,
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
            train_tiles_txt=args.train_tiles,
            val_tiles_txt=args.val_tiles,
            test_tiles_txt=args.test_tiles,
            require_complete=args.require_complete,
        )


if __name__ == "__main__":
    # Avoid some GDAL multithreading surprises on shared machines (optional)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
