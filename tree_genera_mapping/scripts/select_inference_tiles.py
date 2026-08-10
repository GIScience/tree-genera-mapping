#!/usr/bin/env python3
"""Select raster tiles that overlap the inference region of interest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from tree_genera_mapping.utils.tile_partition import dopkachel_to_tile_id


LOGGER = logging.getLogger("select_inference_tiles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select tiles with a positive-area overlap with an inference ROI"
    )
    parser.add_argument("--tiles-gpkg", required=True, type=Path, help="Input tile grid")
    parser.add_argument("--tiles-layer", default=None, help="Optional layer in the tile GeoPackage")
    parser.add_argument("--roi-gpkg", required=True, type=Path, help="Inference ROI GeoPackage")
    parser.add_argument("--roi-layer", default=None, help="Optional layer in the ROI GeoPackage")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV/TXT with a tile_id column")
    parser.add_argument("--tile-id-col", default="tile_id", help="Tile identifier column")
    parser.add_argument(
        "--dop-kachel-col",
        default="dop_kachel",
        help="Fallback LGL tile column used when --tile-id-col is absent",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output")
    return parser.parse_args()


def read_vector(path: Path, layer: str | None) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)


def select_tiles(
    tiles: gpd.GeoDataFrame,
    roi: gpd.GeoDataFrame,
    *,
    tile_id_col: str,
    dop_kachel_col: str,
) -> pd.DataFrame:
    if tiles.empty:
        raise ValueError("Tile grid is empty")
    if roi.empty:
        raise ValueError("Inference ROI is empty")
    if tiles.crs is None or roi.crs is None:
        raise ValueError("Both the tile grid and ROI must have a CRS")

    tiles = tiles.loc[tiles.geometry.notna() & ~tiles.geometry.is_empty].copy()
    roi = roi.loc[roi.geometry.notna() & ~roi.geometry.is_empty].copy()
    if tiles.empty or roi.empty:
        raise ValueError("Tile grid or ROI contains no usable geometries")

    if roi.crs != tiles.crs:
        roi = roi.to_crs(tiles.crs)

    if tile_id_col not in tiles.columns:
        if dop_kachel_col not in tiles.columns:
            raise ValueError(
                f"Tile grid must contain '{tile_id_col}' or '{dop_kachel_col}'"
            )
        tiles[tile_id_col] = tiles[dop_kachel_col].map(dopkachel_to_tile_id)

    roi_union = roi.geometry.union_all() if hasattr(roi.geometry, "union_all") else roi.unary_union
    candidates = tiles.loc[tiles.geometry.intersects(roi_union)].copy()

    # Exclude tiles that only touch the ROI boundary along an edge or at a point.
    if not candidates.empty:
        overlap_area = candidates.geometry.intersection(roi_union).area
        candidates = candidates.loc[overlap_area > 0].copy()

    raw_ids = candidates[tile_id_col]
    if raw_ids.isna().any():
        raise ValueError(f"Column '{tile_id_col}' contains empty tile identifiers")
    ids = raw_ids.astype(str).str.strip()
    if (ids == "").any():
        raise ValueError(f"Column '{tile_id_col}' contains empty tile identifiers")

    return (
        pd.DataFrame({"tile_id": ids})
        .drop_duplicates()
        .sort_values("tile_id", kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")

    tiles = read_vector(args.tiles_gpkg, args.tiles_layer)
    roi = read_vector(args.roi_gpkg, args.roi_layer)
    selected = select_tiles(
        tiles,
        roi,
        tile_id_col=args.tile_id_col,
        dop_kachel_col=args.dop_kachel_col,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    LOGGER.info("Selected %d of %d tiles: %s", len(selected), len(tiles), args.output)


if __name__ == "__main__":
    main()
