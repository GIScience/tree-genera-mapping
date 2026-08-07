#!/usr/bin/env python3
"""
Derive grid-level and municipality-level indicators from the finalized tree inventory.

Produces the aggregated data record and the municipality table reported in the Data
Descriptor: tree counts, mapped area, tree density, per-class counts, class richness and
normalized Shannon diversity.

Definitions
-----------
normalized Shannon  H / ln(C) with C = number of classes in the scheme (10), so the
                    value is comparable between cells regardless of how many genera are
                    actually present. 1.0 = even split across all ten classes,
                    0.0 = a single genus.
class richness      number of distinct classes present.
mapped area         area of the non-forest ROI inside the unit, not its total area.
municipality        mean of the per-cell normalized Shannon values, matching the
                    "mean normalized Shannon diversity" wording in the data record.

Cells with no trees have no defined diversity. --richness-empty controls whether they
enter the municipality mean as 0.0 or are excluded; the two give different results, so
the choice is explicit rather than implied.

Example
-------
    python -m tree_genera_mapping.scripts.aggregate_indicators \\
        --trees-gpkg cache/trees_bw.gpkg --trees-layer trees \\
        --grid-gpkg data/grid_pop_bw_100.gpkg --population-col Einwohner \\
        --municipalities-shp data/geb01_f.shp \\
        --roi-gpkg cache/bw_nonforest_roi.gpkg \\
        --out-grid cache/bw_tree_indicators_grid.gpkg \\
        --out-municipalities cache/bw_tree_indicators.gpkg \\
        --richness-empty nan
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

LOGGER = logging.getLogger("aggregate_indicators")

CLASS_NAMES = {
    0: "Acer",
    1: "Aesculus",
    2: "Carpinus",
    3: "Coniferous",
    4: "Fagus",
    5: "Other Deciduous",
    6: "Platanus",
    7: "Prunus",
    8: "Quercus",
    9: "Tilia",
}
MUNICIPALITY_TYPE = "AX_KommunalesGebiet"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate tree indicators to grid and municipality")
    p.add_argument("--trees-gpkg", required=True, type=Path, help="Finalized inventory")
    p.add_argument("--trees-layer", default="trees", help="Layer inside --trees-gpkg")
    p.add_argument("--grid-gpkg", required=True, type=Path, help="Analysis grid (e.g. 100 m)")
    p.add_argument("--grid-layer", default=None, help="Layer inside --grid-gpkg")
    p.add_argument("--grid-id-col", default=None,
                   help="Grid cell id column; a sequential id is created if omitted")
    p.add_argument("--population-col", default=None,
                   help="Population attribute on the grid (e.g. Einwohner)")
    p.add_argument("--municipalities-shp", required=True, type=Path,
                   help="Administrative-area layer GEB01_F (geb01_f.shp)")
    p.add_argument("--municipality-type", default=MUNICIPALITY_TYPE,
                   help=f"OBJART_TXT value for municipalities (default: {MUNICIPALITY_TYPE})")
    p.add_argument("--municipality-name-col", default="BEZ_GEM", help="Municipality name column")
    p.add_argument("--type-col", default="OBJART_TXT", help="Object-type attribute")
    p.add_argument("--roi-gpkg", required=True, type=Path,
                   help="Non-forest ROI, used for mapped area")
    p.add_argument("--roi-layer", default=None, help="Layer inside --roi-gpkg")
    p.add_argument("--class-col", default="class_id_det", help="Predicted class id column")
    p.add_argument("--x-col", default="centroid_x", help="Crown centroid x")
    p.add_argument("--y-col", default="centroid_y", help="Crown centroid y")
    p.add_argument("--richness-empty", choices=["zero", "nan"], default="nan",
                   help="How zero-tree cells enter the municipality diversity mean")
    p.add_argument("--out-grid", required=True, type=Path, help="Output grid GeoPackage")
    p.add_argument("--out-municipalities", required=True, type=Path,
                   help="Output municipality GeoPackage")
    p.add_argument("--report", type=Path, default=None, help="JSON summary path")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def read_tree_points(path: Path, layer: str, class_col: str,
                     x_col: str, y_col: str, crs) -> gpd.GeoDataFrame:
    """Read the inventory as points, without loading crown polygons."""
    info = pyogrio.read_info(path, layer=layer)
    fields = list(info["fields"])
    LOGGER.info("Inventory has %s features, fields %s", info["features"], fields)

    if x_col in fields and y_col in fields:
        cols = [class_col, x_col, y_col]
        df = pyogrio.read_dataframe(path, layer=layer, columns=cols, read_geometry=False)
        LOGGER.info("  built %d points from %s/%s", len(df), x_col, y_col)
        geom = gpd.points_from_xy(df[x_col], df[y_col])
        return gpd.GeoDataFrame(df[[class_col]], geometry=geom, crs=crs)

    LOGGER.warning("  %s/%s absent; falling back to polygon centroids", x_col, y_col)
    gdf = pyogrio.read_dataframe(path, layer=layer, columns=[class_col])
    gdf["geometry"] = gdf.geometry.centroid
    return gdf


def normalized_shannon(counts: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy divided by ln(number of classes)."""
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, counts / np.maximum(totals, 1), 0.0)
        logs = np.where(p > 0, np.log(p, where=p > 0), 0.0)
        h = -(p * logs).sum(axis=1)
    return h / np.log(counts.shape[1])


def main() -> None:
    setup_logging()
    args = parse_args()

    for path in (args.out_grid, args.out_municipalities):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite to replace it")

    LOGGER.info("Reading grid: %s", args.grid_gpkg)
    grid = gpd.read_file(args.grid_gpkg, layer=args.grid_layer) if args.grid_layer \
        else gpd.read_file(args.grid_gpkg)
    LOGGER.info("  %d cells, CRS %s", len(grid), grid.crs)
    id_col = args.grid_id_col or "grid_id"
    if id_col not in grid.columns:
        grid[id_col] = np.arange(len(grid), dtype=np.int64)

    trees = read_tree_points(args.trees_gpkg, args.trees_layer,
                             args.class_col, args.x_col, args.y_col, grid.crs)
    if trees.crs is not None and grid.crs is not None and trees.crs != grid.crs:
        trees = trees.to_crs(grid.crs)
    n_trees_total = len(trees)

    LOGGER.info("Joining trees to grid cells")
    joined = gpd.sjoin(trees, grid[[id_col, "geometry"]], how="inner", predicate="within")
    LOGGER.info("  %d of %d trees fell inside a grid cell", len(joined), n_trees_total)

    counts = joined.groupby(id_col).size().rename("num_trees")
    per_class = (joined.groupby([id_col, args.class_col]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=sorted(CLASS_NAMES), fill_value=0))
    per_class.columns = [f"n_{CLASS_NAMES[c]}".replace(" ", "_") for c in per_class.columns]

    cell = grid.merge(counts, on=id_col, how="left").merge(per_class, on=id_col, how="left")
    class_cols = list(per_class.columns)
    cell[["num_trees"] + class_cols] = cell[["num_trees"] + class_cols].fillna(0)
    cell["num_trees"] = cell["num_trees"].astype(np.int64)
    for c in class_cols:
        cell[c] = cell[c].astype(np.int64)

    cell["shannon_norm"] = normalized_shannon(cell[class_cols].to_numpy(dtype=float))
    cell["class_richness"] = (cell[class_cols] > 0).sum(axis=1).astype(np.int16)
    if args.richness_empty == "nan":
        cell.loc[cell["num_trees"] == 0, "shannon_norm"] = np.nan
        cell.loc[cell["num_trees"] == 0, "class_richness"] = 0

    args.out_grid.parent.mkdir(parents=True, exist_ok=True)
    cell.to_file(args.out_grid, layer="tree_indicators_grid", driver="GPKG")
    LOGGER.info("Wrote grid indicators: %s", args.out_grid)

    LOGGER.info("Reading municipalities: %s", args.municipalities_shp)
    muni = gpd.read_file(args.municipalities_shp)
    if args.type_col in muni.columns:
        muni = muni[muni[args.type_col] == args.municipality_type].copy()
    LOGGER.info("  %d municipality features", len(muni))
    muni = muni[[args.municipality_name_col, "geometry"]].dissolve(
        by=args.municipality_name_col).reset_index()
    if muni.crs is not None and grid.crs is not None and muni.crs != grid.crs:
        muni = muni.to_crs(grid.crs)
    muni["area_km2"] = muni.geometry.area / 1e6

    LOGGER.info("Reading ROI for mapped area: %s", args.roi_gpkg)
    roi = gpd.read_file(args.roi_gpkg, layer=args.roi_layer) if args.roi_layer \
        else gpd.read_file(args.roi_gpkg)
    if roi.crs is not None and muni.crs is not None and roi.crs != muni.crs:
        roi = roi.to_crs(muni.crs)
    roi_union = roi.geometry.union_all() if hasattr(roi.geometry, "union_all") \
        else roi.geometry.unary_union
    LOGGER.info("Computing mapped (non-forest) area per municipality")
    muni["area_mapped_km2"] = muni.geometry.intersection(roi_union).area / 1e6

    # Cells are attributed to a municipality by their representative point, so each cell
    # contributes to exactly one municipality and counts are not double-attributed.
    reps = cell.copy()
    reps["geometry"] = reps.geometry.representative_point()
    cell_muni = gpd.sjoin(
        reps[[id_col, "num_trees", "shannon_norm"] + class_cols + ["geometry"]],
        muni[[args.municipality_name_col, "geometry"]],
        how="left", predicate="within",
    )

    agg = {"num_trees": "sum", "shannon_norm": "mean"}
    agg.update({c: "sum" for c in class_cols})
    stats = cell_muni.groupby(args.municipality_name_col).agg(agg).reset_index()
    stats = stats.rename(columns={"shannon_norm": "shannon_norm_mean"})
    stats["class_richness"] = (stats[class_cols] > 0).sum(axis=1).astype(np.int16)

    out = muni.merge(stats, on=args.municipality_name_col, how="left")
    out[["num_trees"] + class_cols] = out[["num_trees"] + class_cols].fillna(0)
    out["num_trees"] = out["num_trees"].astype(np.int64)
    out["tree_density_km2"] = out["num_trees"] / out["area_mapped_km2"].replace(0, np.nan)

    if args.population_col and args.population_col in grid.columns:
        pop = cell_muni.merge(grid[[id_col, args.population_col]], on=id_col, how="left")
        pop = pop.groupby(args.municipality_name_col)[args.population_col].sum().reset_index()
        out = out.merge(pop, on=args.municipality_name_col, how="left")
        out["trees_per_capita"] = out["num_trees"] / out[args.population_col].replace(0, np.nan)

    args.out_municipalities.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(args.out_municipalities, layer="tree_indicators", driver="GPKG")
    LOGGER.info("Wrote municipality indicators: %s", args.out_municipalities)

    summary = {
        "trees_total": int(n_trees_total),
        "trees_assigned_to_grid": int(len(joined)),
        "grid_cells": int(len(cell)),
        "grid_cells_with_trees": int((cell["num_trees"] > 0).sum()),
        "municipalities": int(len(out)),
        "area_km2_total": float(out["area_km2"].sum()),
        "area_mapped_km2_total": float(out["area_mapped_km2"].sum()),
        "trees_in_municipalities": int(out["num_trees"].sum()),
        "density_km2_statewide": float(out["num_trees"].sum() /
                                      max(out["area_mapped_km2"].sum(), 1e-9)),
        "richness_empty": args.richness_empty,
        "class_totals": {CLASS_NAMES[c]: int(cell[f"n_{CLASS_NAMES[c]}".replace(' ', '_')].sum())
                         for c in sorted(CLASS_NAMES)},
    }
    report = args.report or args.out_municipalities.with_suffix(".report.json")
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    LOGGER.info("---- summary ----")
    LOGGER.info("trees total                %d", summary["trees_total"])
    LOGGER.info("trees inside grid          %d", summary["trees_assigned_to_grid"])
    LOGGER.info("trees in municipalities    %d", summary["trees_in_municipalities"])
    LOGGER.info("mapped area                %.2f km2", summary["area_mapped_km2_total"])
    LOGGER.info("statewide density          %.2f trees/km2", summary["density_km2_statewide"])
    LOGGER.info("report                     %s", report)


if __name__ == "__main__":
    main()