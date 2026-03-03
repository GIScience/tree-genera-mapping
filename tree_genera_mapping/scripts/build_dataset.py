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
    if "subtile_id" not in df.columns or "split" not in df.columns:
        raise ValueError("Subtile split table must contain columns: subtile_id, split")

    df["subtile_id"] = df["subtile_id"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.lower().replace({"valid": "val"})

    bad = set(df["split"].unique()) - ALLOWED_SPLITS
    if bad:
        raise ValueError(f"Invalid split values in subtile table {bad}. Allowed: {ALLOWED_SPLITS}")

    train = set(df.loc[df["split"] == "train", "subtile_id"])
    val = set(df.loc[df["split"] == "val", "subtile_id"])
    test = set(df.loc[df["split"] == "test", "subtile_id"])
    return train, val, test


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
    ds_train = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, label_col=label_col, mode=mode, size=size, overlap=overlap, split="train", classes_csv=classes_csv,
    unknown_class=unknown_class,
    unknown_map_to=unknown_map_to,)
    ds_val = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, label_col=label_col,  mode=mode, size=size, overlap=overlap, split="val", classes_csv=classes_csv,
    unknown_class=unknown_class,
    unknown_map_to=unknown_map_to,)
    ds_test = ImageDataSet(img_dir=images_dir_p, output_dir=out_root, label_col=label_col,  mode=mode, size=size, overlap=overlap, split="test", classes_csv=classes_csv,
    unknown_class=unknown_class,
    unknown_map_to=unknown_map_to,)

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
    genus_labels_gpkg: str,
    images_dir: str,
    output_dir: str,
    mode: str,
    patch_size: int,
    tile_id_col: str,
    class_col: str,
    id_col: str,
    tile_split_table: Optional[str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> None:
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
    gdf_tiles = ensure_tile_id_column(gdf_tiles, tile_id_col=tile_id_col, prefer_dop_kachel=True)

    train_tiles, val_tiles, test_tiles = build_split_sets(
        gdf_tiles=gdf_tiles,
        tile_id_col=tile_id_col,
        tile_split_table=tile_split_table,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    logger.info("Tile split (patches): train=%d val=%d test=%d", len(train_tiles), len(val_tiles), len(test_tiles))

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

    half = patch_size // 2
    missing = int(joined[tile_id_col].isna().sum())
    if missing > 0:
        logger.warning("Tree points not matched to any tile: %d (skipping)", missing)

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
    ap = argparse.ArgumentParser(description="Build datasets: YOLO detection + genus patches (tile-level splits).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_split_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tile-split-table", default=None, help="CSV/TXT/TSV with columns tile_id,split (train|val|test)")
        p.add_argument("--val-frac", type=float, default=0.2, help="Used only if --tile-split-table is not provided")
        p.add_argument("--test-frac", type=float, default=0.1, help="Used only if --tile-split-table is not provided")
        p.add_argument("--seed", type=int, default=42)

    # Detection
    ap_det = sub.add_parser("det", help="Generate YOLO detection dataset from bbox labels")
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
    ap_cls.add_argument("--tiles-gpkg", required=True)
    ap_cls.add_argument("--genus-labels-gpkg", required=True)
    ap_cls.add_argument("--images-dir", required=True)
    ap_cls.add_argument("--output-dir", required=True)
    ap_cls.add_argument("--mode", required=True)
    ap_cls.add_argument("--patch-size", type=int, default=128)
    ap_cls.add_argument("--tile-id-col", default="tile_id")
    ap_cls.add_argument("--class-col", default="genus")
    ap_cls.add_argument("--id-col", default="tree_id")
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
            
        )
    elif args.cmd == "cls":
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
            tile_split_table=args.tile_split_table,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()