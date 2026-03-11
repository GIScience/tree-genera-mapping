#!/usr/bin/env python3
"""
build_dataset.py

Build YOLO detection chips + labels, and genus classification patches,
with leakage-safe SPLITS at the parent tile level.

Split modes (tile-level):
1) Table mode (recommended):
   --tile-split-table data/tiles_split.txt
   Table must contain columns: tile_id, split (train|val|test).
   Optional columns are allowed (e.g., dop_kachel).

2) Random mode:
   If --tile-split-table is NOT provided -> random split by tile_id using
   --val-frac and --test-frac.

Optional subtile filtering (detection only):
- If you provide --subtile-split-table (CSV/TXT with columns subtile_id,split,...),
  chips will be written ONLY if subtile_id is in the corresponding split list.
  subtile_id format: "32_464_5487_91" (tile_id + "_" + chip_index)

Chip naming:
- Parent tile raster: <images-dir>/<mode>_<tile_id>.tif
  e.g. rgbih_32_388_5279.tif
- Chip output: <mode>_<tile_id>_<n>.tif
  e.g. rgbih_32_388_5279_0.tif

Outputs:
1) Detection YOLO dataset:
   <out>/yolo_<mode>/
     images/{train,val,test}/
     labels/{train,val,test}/

2) Classification patches:
   <out>/patches_<mode>_<patch>/
     {train,val,test}/<class_name>/*.tif
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import cv2
import ast
from rasterio.windows import Window
from tqdm import tqdm


from tree_genera_mapping.preprocess.detection_dataset import ImageDataSet
from tree_genera_mapping.utils.tile_partition import dopkachel_to_tile_id, ensure_tile_id_from_grid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ALLOWED_SPLITS = {"train", "val", "test"}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def strip_georef_from_meta(meta: dict) -> dict:
    """
    Return raster metadata without georeferencing fields.
    Keeps multiband TIFF structure but removes spatial metadata.
    """
    meta = meta.copy()
    for k in ("crs", "transform", "gcps", "rpc"):
        meta.pop(k, None)
    return meta
def _parse_bbox(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (tuple, list)) and len(v) == 4:
        arr = [float(x) for x in v]
    elif isinstance(v, str):
        try:
            arr = list(ast.literal_eval(v))
        except Exception:
            return None
        if len(arr) != 4:
            return None
        arr = [float(x) for x in arr]
    else:
        return None
    if any(np.isnan(x) for x in arr):
        return None
    return tuple(arr)  # (minx, miny, maxx, maxy)
def ensure_same_crs(
    a: gpd.GeoDataFrame, b: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if a.crs is None or b.crs is None:
        raise ValueError("Both GeoDataFrames must have CRS set.")
    if a.crs != b.crs:
        b = b.to_crs(a.crs)
    return a, b


def _sniff_sep(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    return "\t" if "\t" in first else ","


def _validate_overlap(overlap: float) -> None:
    if not (0.0 <= overlap < 1.0):
        raise ValueError("--overlap must be in [0.0, 1.0). Example: 0.2 for 20% overlap.")


def ensure_tile_id_column(
    gdf_tiles: gpd.GeoDataFrame,
    *,
    tile_id_col: str = "tile_id",
    prefer_dop_kachel: bool = True,
) -> gpd.GeoDataFrame:
    """
    Ensure gdf_tiles has a tile_id column.

    Priority:
      1) existing tile_id_col
      2) dop_kachel/dopkachel -> dopkachel_to_tile_id
      3) deterministic grid id -> ensure_tile_id_from_grid
    """
    gdf = gdf_tiles.copy()

    if tile_id_col in gdf.columns:
        gdf[tile_id_col] = gdf[tile_id_col].astype(str)
        return gdf

    if prefer_dop_kachel:
        if "dop_kachel" in gdf.columns:
            gdf[tile_id_col] = gdf["dop_kachel"].astype(str).apply(dopkachel_to_tile_id)
            return gdf
        if "dopkachel" in gdf.columns:
            gdf[tile_id_col] = gdf["dopkachel"].astype(str).apply(dopkachel_to_tile_id)
            return gdf

    # fallback: deterministic grid-based id from geometry (centroid)
    gdf = ensure_tile_id_from_grid(gdf, tile_id_col=tile_id_col, overwrite=False)
    return gdf


def find_tile_raster(images_dir: Path, mode: str, tile_id: str) -> Path:
    """
    Expected naming: <images_dir>/<mode>_<tile_id>.tif
    e.g. rgbih_32_388_5279.tif
    """
    return images_dir / f"{mode}_{tile_id}.tif"


# -----------------------------------------------------------------------------
# split loading
# -----------------------------------------------------------------------------
def load_tile_split_table(path: str, *, tile_id_col: str = "tile_id", split_col: str = "split") -> Dict[str, str]:
    """
    CSV/TSV/TXT with columns: tile_id, split.
    Accepts 'valid' and normalizes to 'val'.
    Returns tile_id -> split.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p, sep=_sniff_sep(p), dtype=str)
    df.columns = [c.strip() for c in df.columns]

    if tile_id_col not in df.columns or split_col not in df.columns:
        raise ValueError(f"Split table must contain columns: {tile_id_col}, {split_col}")

    df[tile_id_col] = df[tile_id_col].astype(str).str.strip()
    df[split_col] = df[split_col].astype(str).str.strip().str.lower().replace({"valid": "val"})

    bad = set(df[split_col].unique()) - ALLOWED_SPLITS
    if bad:
        raise ValueError(f"Invalid split values {bad}. Allowed: {ALLOWED_SPLITS}")

    if df[tile_id_col].duplicated().any():
        dupes = df.loc[df[tile_id_col].duplicated(), tile_id_col].tolist()[:10]
        raise ValueError(f"Duplicate tile_id in tile split table (examples): {dupes}")

    return dict(zip(df[tile_id_col], df[split_col]))


def make_tile_split_random(
    tile_ids: Sequence[str],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[str]]:
    if not (0.0 <= val_frac < 1.0) or not (0.0 <= test_frac < 1.0) or (val_frac + test_frac >= 1.0):
        raise ValueError("--val-frac and --test-frac must be in [0,1) and sum < 1")

    ids = [str(x) for x in tile_ids]
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))

    val = set(ids[:n_val])
    test = set(ids[n_val:n_val + n_test])
    train = set(ids[n_val + n_test:])
    return train, val, test


def build_split_sets(
    *,
    gdf_tiles: gpd.GeoDataFrame,
    tile_id_col: str,
    tile_split_table: Optional[str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[str]]:
    available = gdf_tiles[tile_id_col].astype(str).tolist()
    available_set = set(available)

    if tile_split_table:
        split_map = load_tile_split_table(tile_split_table, tile_id_col=tile_id_col, split_col="split")
        unknown = set(split_map.keys()) - available_set
        if unknown:
            raise ValueError(f"tiles_split contains tile_ids not in tiles gpkg (examples): {sorted(list(unknown))[:10]}")

        train = {tid for tid, sp in split_map.items() if sp == "train"}
        val = {tid for tid, sp in split_map.items() if sp == "val"}
        test = {tid for tid, sp in split_map.items() if sp == "test"}
        return train, val, test

    # random fallback
    return make_tile_split_random(available, val_frac=val_frac, test_frac=test_frac, seed=seed)


def load_subtile_split_table(path: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    CSV/TSV/TXT with columns: subtile_id, split (train|val|test).
    Example subtile_id: 32_464_5487_91
    Returns (train_set, val_set, test_set) of subtile_id strings.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p, sep=_sniff_sep(p), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    # if "subtile_id" not in df.columns or "split" not in df.columns:
    #     raise ValueError("Subtile split table must contain columns: subtile_id, split")

    # df["subtile_id"] = df["subtile_id"].astype(str).str.strip()
    # df["split"] = df["split"].astype(str).str.strip().str.lower().replace({"valid": "val"})

    # bad = set(df["split"].unique()) - ALLOWED_SPLITS
    # if bad:
    #     raise ValueError(f"Invalid split values in subtile table {bad}. Allowed: {ALLOWED_SPLITS}")

    # train = set(df.loc[df["split"] == "train", "subtile_id"])
    # val = set(df.loc[df["split"] == "val", "subtile_id"])
    # test = set(df.loc[df["split"] == "test", "subtile_id"])
    
    # return train, val, test

    # Use the column name you have (subtile_id)
    all_ids = set(df["subtile_id"].astype(str).str.strip())
    
    # Return the same set for all three splits
    # This lets the tile-level logic handle the destination
    return all_ids, all_ids, all_ids

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
    tile_split_table: Optional[str],
    subtile_split_table: Optional[str],
    val_frac: float,
    test_frac: float,
    seed: int,
    include_empty_tiles: bool,
    label_col: Optional[str],
    classes_csv: Optional[str],
    unknown_class: str,
    unknown_map_to: str,
    plain_tiff: bool,
) -> None:
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
        logger.warning("No bboxes found in %s. You may only get negatives.", bboxes_gpkg)

    gdf_tiles, gdf_boxes = ensure_same_crs(gdf_tiles, gdf_boxes)
    gdf_tiles = ensure_tile_id_column(gdf_tiles, tile_id_col=tile_id_col, prefer_dop_kachel=True)

    train_tiles, val_tiles, test_tiles = build_split_sets(
        gdf_tiles=gdf_tiles,
        tile_id_col=tile_id_col,
        tile_split_table=tile_split_table,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    logger.info(
        "Tile split: train=%d val=%d test=%d (available=%d)",
        len(train_tiles), len(val_tiles), len(test_tiles), len(set(gdf_tiles[tile_id_col].tolist()))
    )

    # optional: subtile whitelist sets
    subtile_train = subtile_val = subtile_test = None
    if subtile_split_table:
        st_tr, st_va, st_te = load_subtile_split_table(subtile_split_table)
        subtile_train, subtile_val, subtile_test = st_tr, st_va, st_te
        logger.info(
            "Subtile filter enabled: train=%d val=%d test=%d",
            len(subtile_train), len(subtile_val), len(subtile_test)
        )

    # process only tiles that are in any split
    allowed_tiles = train_tiles | val_tiles | test_tiles
    gdf_tiles = gdf_tiles[gdf_tiles[tile_id_col].astype(str).isin(allowed_tiles)].copy()

    # one writer per split (your new ImageDataSet supports split + whitelist)
    ds_train = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, label_col=label_col, mode=mode,
                            size=size, overlap=overlap, split="train", classes_csv=classes_csv,
                            unknown_class=unknown_class, unknown_map_to=unknown_map_to, plain_tiff=plain_tiff)
    ds_val = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, label_col=label_col,  mode=mode,
                          size=size, overlap=overlap, split="val", classes_csv=classes_csv,
                          unknown_class=unknown_class, unknown_map_to=unknown_map_to, plain_tiff=plain_tiff)
    ds_test = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, label_col=label_col,  mode=mode,
                           size=size, overlap=overlap, split="test", classes_csv=classes_csv,
                           unknown_class=unknown_class,    unknown_map_to=unknown_map_to, plain_tiff=plain_tiff)


    for _, row in tqdm(gdf_tiles.iterrows(), total=len(gdf_tiles), desc="YOLO tiles"):
        tile_id = str(row[tile_id_col])
        tile_path = find_tile_raster(images_dir_p, mode, tile_id)
        if not tile_path.exists():
            logger.warning("Missing tile raster: %s (skipping)", tile_path)
            continue

        # per-tile label selection
        tile_labels = gdf_boxes
        if tile_id_col in gdf_boxes.columns:
            tile_labels = gdf_boxes[gdf_boxes[tile_id_col].astype(str) == tile_id]
        else:
            try:
                tile_labels = gdf_boxes[gdf_boxes.intersects(row.geometry)]
            except Exception:
                pass

        if (tile_labels is None or len(tile_labels) == 0) and not include_empty_tiles:
            continue

        if tile_id in test_tiles:
            ds_test.split_tiff_to_tiles(
                tile_path,
                tile_labels,
                split="test",
                subtile_whitelist=subtile_test,
                write_empty_labels=include_empty_tiles,
            )
        elif tile_id in val_tiles:
            ds_val.split_tiff_to_tiles(
                tile_path,
                tile_labels,
                split="val",
                subtile_whitelist=subtile_val,
                write_empty_labels=include_empty_tiles,
            )
        else:
            ds_train.split_tiff_to_tiles(
                tile_path,
                tile_labels,
                split="train",
                subtile_whitelist=subtile_train,
                write_empty_labels=include_empty_tiles,
            )

    logger.info("✅ YOLO detection dataset written to: %s", out_root)
    logger.info("   images/: %s", out_root / "images")
    logger.info("   labels/: %s", out_root / "labels")


# -----------------------------------------------------------------------------
# 2) Classification patches dataset
# -----------------------------------------------------------------------------
def make_classification_patches(
    *,
    tiles_gpkg: str,
    genus_labels_csv: str,
    images_dir: str,
    output_dir: str,
    mode: str,
    patch_size: int,
    tile_id_col: str,               # column name in tiles gpkg
    labels_tile_col: str,           # column name in labels CSV for tile id
    class_col: str,                 # genus column
    id_col: str,                    # tree_id column
    x_col: str,                     # X coordinate column
    y_col: str,                     # Y coordinate column
    bbox_col: Optional[str],        # bbox column
    crop_mode: str,                 # fixed|bbox
    tile_split_table: Optional[str],
    val_frac: float,
    test_frac: float,
    seed: int,
    plain_tiff: bool,
    split_csv: Optional[str] = None,  # CSV with tree_id,split for tree-level split
) -> None:
    if patch_size <= 0:
        raise ValueError("--patch-size must be > 0")
    if crop_mode not in {"fixed", "bbox"}:
        raise ValueError("--crop-mode must be fixed|bbox")
    if crop_mode == "fixed" and patch_size % 2 != 0:
        raise ValueError("--patch-size must be even for centered fixed windows")

    images_dir_p = Path(images_dir)
    if not images_dir_p.exists():
        raise FileNotFoundError(images_dir_p)

    out_root = Path(output_dir) / f"patches_{mode}_{patch_size}_{crop_mode}"
    for split in ("train", "val", "test"):
        (out_root / split).mkdir(parents=True, exist_ok=True)

    # tiles are still useful for available tile_ids and sanity checks
    gdf_tiles = gpd.read_file(tiles_gpkg)
    if gdf_tiles.empty:
        raise ValueError(f"No tiles found in {tiles_gpkg}")
    gdf_tiles = ensure_tile_id_column(gdf_tiles, tile_id_col=tile_id_col, prefer_dop_kachel=True)
    available_tiles = set(gdf_tiles[tile_id_col].astype(str))

    # labels CSV
    df = pd.read_csv(genus_labels_csv)
    required_cols = [labels_tile_col, class_col, id_col]
    if crop_mode == "fixed":
        required_cols += [x_col, y_col]
    if crop_mode == "bbox":
        if not bbox_col:
            raise ValueError("bbox mode needs --bbox-col")
        required_cols += [bbox_col]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    df[labels_tile_col] = df[labels_tile_col].astype(str).str.strip()
    df[id_col] = df[id_col].astype(str).str.strip()
    df[class_col] = df[class_col].astype(str).str.strip()

    # -------------------------------------------------------------------------
    # split assignment
    # -------------------------------------------------------------------------
    if split_csv:
        df_split = pd.read_csv(split_csv)

        if id_col not in df_split.columns or "split" not in df_split.columns:
            raise ValueError(f"{split_csv} must contain columns: {id_col}, split")

        df_split[id_col] = df_split[id_col].astype(str).str.strip()
        df_split["split"] = (
            df_split["split"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"valid": "val"})
        )

        bad = set(df_split["split"].unique()) - ALLOWED_SPLITS
        if bad:
            raise ValueError(f"Invalid split values in split_csv: {bad}. Allowed: {ALLOWED_SPLITS}")

        if df_split[id_col].duplicated().any():
            dupes = df_split.loc[df_split[id_col].duplicated(), id_col].tolist()[:10]
            raise ValueError(f"Duplicate {id_col} values in split_csv (examples): {dupes}")

        df = df.merge(df_split[[id_col, "split"]], on=id_col, how="inner")

        if df.empty:
            raise ValueError(f"No rows left after merging {genus_labels_csv} with {split_csv} on {id_col}")

        logger.info(
            "Tree-level split enabled from %s: train=%d val=%d test=%d",
            split_csv,
            int((df["split"] == "train").sum()),
            int((df["split"] == "val").sum()),
            int((df["split"] == "test").sum()),
        )

    else:
        train_tiles, val_tiles, test_tiles = build_split_sets(
            gdf_tiles=gdf_tiles,
            tile_id_col=tile_id_col,
            tile_split_table=tile_split_table,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )
        logger.info(
            "Tile split (patches): train=%d val=%d test=%d",
            len(train_tiles), len(val_tiles), len(test_tiles)
        )

    # -------------------------------------------------------------------------
    # patch extraction
    # -------------------------------------------------------------------------
    half = patch_size // 2
    cache_tile_id = None
    cache_src = None

    def _close_cache():
        nonlocal cache_src, cache_tile_id
        if cache_src is not None:
            try:
                cache_src.close()
            except Exception:
                pass
        cache_src = None
        cache_tile_id = None

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Genus patches"):
            tile_id = str(row[labels_tile_col]).strip()

            if tile_id not in available_tiles:
                continue

            if split_csv:
                split = str(row["split"]).strip().lower()
                if split not in ALLOWED_SPLITS:
                    continue
            else:
                if tile_id in test_tiles:
                    split = "test"
                elif tile_id in val_tiles:
                    split = "val"
                elif tile_id in train_tiles:
                    split = "train"
                else:
                    continue

            class_name = str(row[class_col]).strip().replace(" ", "_")
            out_id = str(row[id_col]).strip()

            tile_path = find_tile_raster(images_dir_p, mode, tile_id)
            if not tile_path.exists():
                logger.warning("Missing tile raster: %s", tile_path)
                continue

            class_dir = out_root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # output file should be tree_id.tif
            patch_path = class_dir / f"{out_id}.tif"
            if patch_path.exists():
                continue

            try:
                # reuse raster handle while consecutive rows belong to same tile
                if cache_tile_id != tile_id:
                    _close_cache()
                    cache_src = rasterio.open(tile_path)
                    cache_tile_id = tile_id

                src = cache_src

                if crop_mode == "fixed":
                    x = float(row[x_col])
                    y = float(row[y_col])
                
                    r, c = src.index(x, y)
                    win = Window(c - half, r - half, patch_size, patch_size)
                
                    if (
                        win.col_off < 0 or win.row_off < 0 or
                        win.col_off + win.width > src.width or
                        win.row_off + win.height > src.height
                    ):
                        continue
                
                    patch = src.read(window=win)
                    meta = src.meta.copy()
                    meta.update(
                        driver="GTiff",
                        height=patch.shape[1],
                        width=patch.shape[2],
                        count=patch.shape[0],
                        transform=src.window_transform(win),
                    )
                
                    if plain_tiff:
                        meta = strip_georef_from_meta(meta)
                
                else:  # bbox mode
                    if bbox_col not in df.columns:
                        raise ValueError("bbox mode needs --bbox-col pointing to a bbox column in CSV")
                
                    bb = _parse_bbox(row[bbox_col])
                    if bb is None:
                        continue
                
                    minx, miny, maxx, maxy = bb
                
                    # skip degenerate bboxes
                    if not (maxx > minx and maxy > miny):
                        continue
                
                    win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                    win = win.round_offsets().round_shape()
                
                    if win.width <= 0 or win.height <= 0:
                        continue
                
                    if (
                        win.col_off < 0 or win.row_off < 0 or
                        win.col_off + win.width > src.width or
                        win.row_off + win.height > src.height
                    ):
                        continue
                
                    patch = src.read(window=win)
                    if patch.size == 0:
                        continue
                
                    # resize each band to fixed patch_size for ResNet input
                    patch = np.stack([
                        cv2.resize(b, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                        for b in patch
                    ])
                
                    dst_transform = rasterio.transform.from_bounds(
                        minx, miny, maxx, maxy, patch_size, patch_size
                    )
                
                    meta = src.meta.copy()
                    meta.update(
                        driver="GTiff",
                        height=patch_size,
                        width=patch_size,
                        count=patch.shape[0],
                        transform=dst_transform,
                    )
                
                    if plain_tiff:
                        meta = strip_georef_from_meta(meta)
                
                # write patch to disk
                with rasterio.open(patch_path, "w", **meta) as dst:
                    dst.write(patch)

            except Exception as e:
                logger.warning("Failed patch %s from tile %s: %s", out_id, tile_path.name, e)

    finally:
        _close_cache()

    logger.info("✅ Classification patches written to: %s", out_root)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build datasets: YOLO detection + genus patches (tile-level splits).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_split_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tile-split-table", default=None, help="CSV/TXT/TSV with columns tile_id,split (train|val|test)")
        p.add_argument("--val-frac", type=float, default=0.2, help="Used only if --tile-split-table is not provided")
        p.add_argument("--test-frac", type=float, default=0.1, help="Used only if --tile-split-table is not provided")
        p.add_argument("--seed", type=int, default=42)

    # Detection
    ap_det = sub.add_parser("det", help="Generate YOLO detection dataset from bbox labels")
    ap_det.add_argument(
        "--plain-tiff",
        action="store_true",
        help="Write plain TIFF without georeferencing metadata (default: keep GeoTIFF metadata).",
    )
    ap_det.add_argument("--tiles-gpkg", required=True)
    ap_det.add_argument("--bboxes-gpkg", required=True)
    ap_det.add_argument("--images-dir", required=True, help="Directory with parent tile rasters: <mode>_<tile_id>.tif")
    ap_det.add_argument("--output-dir", required=True)
    ap_det.add_argument("--mode", required=True, help="e.g. rgbih (expects rgbih_<tile_id>.tif)")
    ap_det.add_argument("--tile-id-col", default="tile_id")
    ap_det.add_argument("--label-col", default='genus',
                    help="Column in bboxes_gpkg containing class id (e.g. genus, top1_class)")
    ap_det.add_argument("--classes-csv", default=None, help="CSV mapping (fid,genus)")
    ap_det.add_argument(
        "--unknown-class",
        default="skip",
        choices=["skip", "map"],
        help="What to do if label not found in mapping",
    )
    ap_det.add_argument(
        "--unknown-map-to",
        default="Other Deciduous",
        help="Target class name if unknown-class=map",
    )
    ap_det.add_argument("--size", type=int, default=640)
    ap_det.add_argument("--overlap", type=float, default=0.2)
    ap_det.add_argument("--include-empty-tiles", dest="include_empty_tiles", action="store_true", default=True)
    ap_det.add_argument("--no-empty-tiles", dest="include_empty_tiles", action="store_false")
    ap_det.add_argument(
        "--subtile-split-table",
        default=None,
        help="Optional CSV/TXT/TSV with columns subtile_id,split to keep only specific chips",
    )
    add_split_args(ap_det)

    # Classification patches
    ap_cls = sub.add_parser("cls", help="Generate genus classification patches")
    ap_cls.add_argument(
        "--plain-tiff",
        action="store_true",
        help="Write plain TIFF without georeferencing metadata (default: keep GeoTIFF metadata).",
    )
    ap_cls.add_argument("--tiles-gpkg", required=True)
    ap_cls.add_argument("--genus-labels-csv", required=True)
    ap_cls.add_argument("--images-dir", required=True)
    ap_cls.add_argument("--output-dir", required=True)
    ap_cls.add_argument("--mode", required=True)
    ap_cls.add_argument("--class-col", default="genus")
    ap_cls.add_argument("--tile-id-col", default="tile_id")  # in tiles gpkg    # in CSV
    ap_cls.add_argument("--labels-tile-col", default="tile_id")
    ap_cls.add_argument("--x-col", default="X")
    ap_cls.add_argument("--y-col", default="Y")
    ap_cls.add_argument("--bbox-col", default="bbox")
    ap_cls.add_argument("--patch-size", type=int, default=128)
    ap_cls.add_argument("--crop-mode", default="fixed", choices=["fixed", "bbox"])
    ap_cls.add_argument("--id-col", default="tree_id")
    ap_cls.add_argument("--split-csv", default=None, help="CSV with tree_id,split for tree-level split")
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
            tile_split_table=args.tile_split_table,
            subtile_split_table=args.subtile_split_table,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            include_empty_tiles=args.include_empty_tiles,
            label_col=args.label_col,
            classes_csv=args.classes_csv,
            unknown_class=args.unknown_class,
            unknown_map_to=args.unknown_map_to,
            plain_tiff=args.plain_tiff,
        )
    elif args.cmd == "cls":
        make_classification_patches(
            tiles_gpkg=args.tiles_gpkg,
            genus_labels_csv=args.genus_labels_csv,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            patch_size=args.patch_size,
            tile_id_col=args.tile_id_col,
            labels_tile_col=args.labels_tile_col,
            class_col=args.class_col,
            id_col=args.id_col,
            x_col=args.x_col,
            y_col=args.y_col,
            bbox_col=args.bbox_col,
            crop_mode=args.crop_mode,
            tile_split_table=args.tile_split_table,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            plain_tiff=args.plain_tiff,
            split_csv=args.split_csv,
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()