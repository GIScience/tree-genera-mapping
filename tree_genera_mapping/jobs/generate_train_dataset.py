#!/usr/bin/env python3
"""
generate_train_dataset.py

Generate training data for two model types:

1) YOLO detection dataset: chips + bbox labels
   Output layout:
     <out>/yolo_<mode>/images/train, images/val
     <out>/yolo_<mode>/labels/train, labels/val

2) Image classification dataset: patches around labeled trees
   Output layout:
     <out>/patches_<mode>_<patch>/train/<class_name>/*.tif
     <out>/patches_<mode>_<patch>/val/<class_name>/*.tif

Splitting:
- Default split is by tile_id (recommended to avoid spatial leakage).
- If dataset already has split folders, you can supply train/val tile lists explicitly.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from tree_genera_mapping.preprocess.detection_dataset import ImageDataSet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -----------------------------
# helpers
# -----------------------------
def dopkachel_to_tile_id(dop_kachel: str) -> str:
    """Convert BW dop_kachel string like '323556048' -> '32_355_6048'."""
    s = str(dop_kachel)
    if len(s) < 9:
        raise ValueError(f"dop_kachel looks too short: {dop_kachel}")
    return f"{s[:2]}_{s[2:5]}_{s[-4:]}"


def find_tile_raster(images_dir: Path, mode: str, tile_id: str) -> Path:
    """Expected naming: <images_dir>/<mode>_<tile_id>.tif"""
    return images_dir / f"{mode}_{tile_id}.tif"


def ensure_same_crs(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if a.crs is None or b.crs is None:
        raise ValueError("Both GeoDataFrames must have CRS set.")
    if a.crs != b.crs:
        b = b.to_crs(a.crs)
    return a, b


def _read_lines(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    lines = [x.strip() for x in p.read_text().splitlines() if x.strip()]
    return lines or None


def make_tile_split(
    tile_ids: Sequence[str],
    *,
    val_frac: float,
    seed: int,
    train_list: Optional[Sequence[str]] = None,
    val_list: Optional[Sequence[str]] = None,
) -> Tuple[Set[str], Set[str]]:
    """
    Returns train_tiles, val_tiles.

    If train_list/val_list provided -> use them (and validate overlap).
    Else do random split with fixed seed.
    """
    tile_ids = [str(t) for t in tile_ids]
    all_set = set(tile_ids)

    if train_list is not None or val_list is not None:
        train_set = set(map(str, train_list or []))
        val_set = set(map(str, val_list or []))

        overlap = train_set & val_set
        if overlap:
            raise ValueError(f"train_tiles and val_tiles overlap: {sorted(list(overlap))[:10]} ...")

        # If only one provided, fill the rest
        if train_list is None:
            train_set = all_set - val_set
        if val_list is None:
            val_set = all_set - train_set

        # Validate
        missing = (train_set | val_set) - all_set
        if missing:
            raise ValueError(f"Provided tile ids not found in available tiles: {sorted(list(missing))[:10]} ...")

        # Anything not assigned goes to train (safe default)
        unassigned = all_set - (train_set | val_set)
        train_set |= unassigned
        return train_set, val_set

    # random split
    rng = random.Random(seed)
    ids = tile_ids[:]
    rng.shuffle(ids)
    n_val = int(round(len(ids) * float(val_frac)))
    val_set = set(ids[:n_val])
    train_set = set(ids[n_val:])
    return train_set, val_set


def _ensure_tile_id_column(gdf_tiles: gpd.GeoDataFrame, tile_id_col: str) -> gpd.GeoDataFrame:
    gdf_tiles = gdf_tiles.copy()
    if tile_id_col not in gdf_tiles.columns:
        if "dop_kachel" in gdf_tiles.columns:
            gdf_tiles[tile_id_col] = gdf_tiles["dop_kachel"].astype(str).apply(dopkachel_to_tile_id)
        else:
            raise ValueError(f"Tiles file must contain '{tile_id_col}' or 'dop_kachel'.")
    gdf_tiles[tile_id_col] = gdf_tiles[tile_id_col].astype(str)
    return gdf_tiles


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
    val_frac: float = 0.2,
    seed: int = 42,
    train_tiles_txt: Optional[str] = None,
    val_tiles_txt: Optional[str] = None,
) -> None:
    """
    Builds YOLO-ready dataset from weak bboxes with train/val split.
    Output:
      <out>/yolo_<mode>/images/train|val
      <out>/yolo_<mode>/labels/train|val
    """
    images_dir_p = Path(images_dir)
    out_root = Path(output_dir) / f"yolo_{mode}"
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    gdf_tiles = gpd.read_file(tiles_gpkg)
    gdf_tree = gpd.read_file(weak_bboxes_gpkg)

    if gdf_tiles.empty:
        raise ValueError(f"No tiles found in {tiles_gpkg}")
    if gdf_tree.empty:
        raise ValueError(f"No bboxes found in {weak_bboxes_gpkg}")

    gdf_tiles, gdf_tree = ensure_same_crs(gdf_tiles, gdf_tree)
    gdf_tiles = _ensure_tile_id_column(gdf_tiles, tile_id_col)

    # tiles that intersect any bbox
    try:
        matching_idxs = gpd.sjoin(gdf_tiles, gdf_tree, how="inner", predicate="intersects").index.unique()
    except Exception:
        if tile_id_col in gdf_tree.columns:
            tiles = set(gdf_tree[tile_id_col].astype(str).unique())
            matching_idxs = gdf_tiles[gdf_tiles[tile_id_col].astype(str).isin(tiles)].index.unique()
        else:
            raise RuntimeError(
                "Spatial join failed and cannot fallback by tile id column. "
                f"Check CRS and ensure '{tile_id_col}' exists in bboxes gpkg."
            )

    gdf_tiles_filtered = gdf_tiles.loc[matching_idxs].copy()
    if gdf_tiles_filtered.empty:
        logger.warning("No tiles intersect with bboxes. Nothing to do.")
        return

    # split by tile_id
    tile_ids = gdf_tiles_filtered[tile_id_col].astype(str).tolist()
    train_list = _read_lines(train_tiles_txt)
    val_list = _read_lines(val_tiles_txt)
    train_tiles, val_tiles = make_tile_split(tile_ids, val_frac=val_frac, seed=seed, train_list=train_list, val_list=val_list)
    logger.info("Tile split: train=%d val=%d (total=%d)", len(train_tiles), len(val_tiles), len(tile_ids))

    # We still use ImageDataSet to do chip+label writing.
    # Best: if ImageDataSet supports output_dir per call, pass train/val dir.
    # If not, you can:
    #   - instantiate 2 datasets (one for train, one for val)
    #   - each writes into its own out_root/images/train etc.
    ds_train = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, mode=mode, size=size, overlap=overlap)
    ds_val = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, mode=mode, size=size, overlap=overlap)

    # IMPORTANT:
    # This assumes your ImageDataSet writes into:
    #   output_dir/images and output_dir/labels
    # If your implementation is different, adjust the folder names inside ImageDataSet (recommended),
    # OR implement ds.set_split("train"/"val").
    #
    # To keep this script generic, we set an attribute if ImageDataSet supports it.
    if hasattr(ds_train, "split"):
        ds_train.split = "train"
    if hasattr(ds_val, "split"):
        ds_val.split = "val"

    logger.info("Processing %d tiles for detection dataset...", len(gdf_tiles_filtered))

    for _, row in tqdm(gdf_tiles_filtered.iterrows(), total=len(gdf_tiles_filtered), desc="Detection tiles"):
        tile_id = str(row[tile_id_col])
        tile_path = find_tile_raster(images_dir_p, mode, tile_id)
        if not tile_path.exists():
            logger.warning("Missing tile raster: %s (skipping)", tile_path)
            continue

        if tile_id in val_tiles:
            ds_val.split_tiff_to_tiles(tile_path, gdf_tree, split="val") if "split" in ds_val.split_tiff_to_tiles.__code__.co_varnames else ds_val.split_tiff_to_tiles(tile_path, gdf_tree)
        else:
            ds_train.split_tiff_to_tiles(tile_path, gdf_tree, split="train") if "split" in ds_train.split_tiff_to_tiles.__code__.co_varnames else ds_train.split_tiff_to_tiles(tile_path, gdf_tree)

    logger.info("✅ Detection dataset written to: %s", out_root)
    logger.info("Expected YOLO folders: %s", out_root / "images")


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
    tile_id_col: str = "tile_id",
    class_col: str = "training_class",
    id_col: str = "uuid",
    val_frac: float = 0.2,
    seed: int = 42,
    train_tiles_txt: Optional[str] = None,
    val_tiles_txt: Optional[str] = None,
) -> None:
    """
    Extract patches around labeled trees into:
      <out>/patches_<mode>_<patch>/train/<class_name>/*.tif
      <out>/patches_<mode>_<patch>/val/<class_name>/*.tif
    """
    images_dir_p = Path(images_dir)
    out_root = Path(output_dir) / f"patches_{mode}_{patch_size}"
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "val").mkdir(parents=True, exist_ok=True)

    gdf_tiles = gpd.read_file(tiles_gpkg)
    gdf_tree = gpd.read_file(genus_labels_gpkg)

    if gdf_tiles.empty:
        raise ValueError(f"No tiles found in {tiles_gpkg}")
    if gdf_tree.empty:
        raise ValueError(f"No genus labels found in {genus_labels_gpkg}")
    if class_col not in gdf_tree.columns:
        raise ValueError(f"genus_labels_gpkg is missing class column '{class_col}'")

    gdf_tiles, gdf_tree = ensure_same_crs(gdf_tiles, gdf_tree)
    gdf_tiles = _ensure_tile_id_column(gdf_tiles, tile_id_col)

    # Points for join
    gdf_tree = gdf_tree.copy()
    gdf_tree["__pt__"] = gdf_tree.geometry.centroid

    half = int(patch_size // 2)

    pts_cols = [class_col]
    if id_col in gdf_tree.columns:
        pts_cols.append(id_col)

    pts = gpd.GeoDataFrame(gdf_tree[pts_cols].copy(), geometry=gdf_tree["__pt__"], crs=gdf_tree.crs)
    joined = gpd.sjoin(pts, gdf_tiles[[tile_id_col, "geometry"]], how="left", predicate="within")

    missing = joined[tile_id_col].isna().sum()
    if missing > 0:
        logger.info("Points not matched to any tile: %d (they will be skipped)", int(missing))

    # split by tile_id (only among tiles that actually have matched points)
    matched_tile_ids = joined[tile_id_col].dropna().astype(str).unique().tolist()
    train_list = _read_lines(train_tiles_txt)
    val_list = _read_lines(val_tiles_txt)
    train_tiles, val_tiles = make_tile_split(matched_tile_ids, val_frac=val_frac, seed=seed, train_list=train_list, val_list=val_list)
    logger.info("Tile split (classification): train=%d val=%d (matched tiles=%d)", len(train_tiles), len(val_tiles), len(matched_tile_ids))

    # iterate points
    for idx, row in tqdm(joined.iterrows(), total=len(joined), desc="Classification patches"):
        tile_id = row.get(tile_id_col, None)
        if tile_id is None or (isinstance(tile_id, float) and np.isnan(tile_id)):
            continue
        tile_id = str(tile_id)

        split = "val" if tile_id in val_tiles else "train"

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


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate training datasets: YOLO detection + classification patches.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Shared split knobs
    def add_split_args(p: argparse.ArgumentParser):
        p.add_argument("--val-frac", type=float, default=0.2)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--train-tiles", default=None, help="Optional text file listing train tile_ids (one per line)")
        p.add_argument("--val-tiles", default=None, help="Optional text file listing val tile_ids (one per line)")

    # Detection
    ap_det = sub.add_parser("det", help="Generate YOLO detection dataset from weak bbox labels")
    ap_det.add_argument("--tiles-gpkg", required=True)
    ap_det.add_argument("--weak-bboxes-gpkg", required=True)
    ap_det.add_argument("--images-dir", required=True, help="Directory containing tiles")
    ap_det.add_argument("--output-dir", required=True)
    ap_det.add_argument("--mode", required=True, help="Tile prefix in filenames, e.g. rgbih, rgbi, rgb")
    ap_det.add_argument("--tile-id-col", default="tile_id", help="Tile id column in tiles/bboxes gpkg (fallback)")
    ap_det.add_argument("--size", type=int, default=640)
    ap_det.add_argument("--overlap", type=float, default=0.0)
    add_split_args(ap_det)

    # Classification
    ap_cls = sub.add_parser("patches", help="Generate classification patches from genus labels")
    ap_cls.add_argument("--tiles-gpkg", required=True)
    ap_cls.add_argument("--genus-labels-gpkg", required=True)
    ap_cls.add_argument("--images-dir", required=True, help="Directory containing tiles")
    ap_cls.add_argument("--output-dir", required=True)
    ap_cls.add_argument("--mode", required=True, help="Tile prefix in filenames, e.g. rgbih, rgbi, rgb")
    ap_cls.add_argument("--patch-size", type=int, default=128)
    ap_cls.add_argument("--tile-id-col", default="tile_id", help="Column in tiles.gpkg or derived from dop_kachel")
    ap_cls.add_argument("--class-col", default="training_class")
    ap_cls.add_argument("--id-col", default="uuid")
    add_split_args(ap_cls)

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
            val_frac=args.val_frac,
            seed=args.seed,
            train_tiles_txt=args.train_tiles,
            val_tiles_txt=args.val_tiles,
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
            val_frac=args.val_frac,
            seed=args.seed,
            train_tiles_txt=args.train_tiles,
            val_tiles_txt=args.val_tiles,
        )


if __name__ == "__main__":
    main()
