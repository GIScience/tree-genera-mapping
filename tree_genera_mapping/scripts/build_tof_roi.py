#!/usr/bin/env python3
"""
Build the non-forest mapping domain (ROI) from Basis-DLM layers.

The ROI is the Baden-Wuerttemberg state boundary with all forest polygons erased. It is
used twice in the workflow: to select which 1 km x 1 km tiles need statewide inference,
and to restrict the released predictions to the mapping domain.

Inputs (ATKIS Basis-DLM, object types selected via OBJART_TXT):
  - geb01_f.shp  AX_Gebiet_Bundesland   -> state boundary
  - veg02_f.shp  AX_Wald                -> forest areas to erase

Expected result for Baden-Wuerttemberg:
  state boundary   35,766.43 km2
  minus AX_Wald    14,059.78 km2
  = ROI            21,706.65 km2

Example
-------
    python -m tree_genera_mapping.scripts.build_nonforest_roi \\
        --boundary-shp data/geb01_f.shp \\
        --forest-shp data/veg02_f.shp \\
        --output cache/bw_nonforest_roi.gpkg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd

LOGGER = logging.getLogger("build_nonforest_roi")

DEFAULT_CRS = "EPSG:25832"
BOUNDARY_TYPE = "AX_Gebiet_Bundesland"
FOREST_TYPE = "AX_Wald"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the non-forest ROI from Basis-DLM layers")
    p.add_argument("--boundary-shp", required=True, type=Path,
                   help="Administrative-area layer GEB01_F (geb01_f.shp)")
    p.add_argument("--forest-shp", required=True, type=Path,
                   help="Vegetation layer VEG02_F (veg02_f.shp)")
    p.add_argument("--output", required=True, type=Path, help="Output GeoPackage")
    p.add_argument("--layer", default="bw_nonforest_roi", help="Output layer name")
    p.add_argument("--boundary-type", default=BOUNDARY_TYPE,
                   help=f"OBJART_TXT value selecting the state boundary (default: {BOUNDARY_TYPE})")
    p.add_argument("--forest-type", default=FOREST_TYPE,
                   help=f"OBJART_TXT value selecting forest areas (default: {FOREST_TYPE})")
    p.add_argument("--type-col", default="OBJART_TXT", help="Object-type attribute")
    p.add_argument("--crs", default=DEFAULT_CRS, help=f"Working CRS (default: {DEFAULT_CRS})")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    return p.parse_args()


def load_by_type(path: Path, type_col: str, type_value: str | None, crs: str, label: str) -> gpd.GeoDataFrame:
    """Read a Basis-DLM layer, optionally filter on OBJART_TXT, reproject and repair."""
    LOGGER.info("Reading %s: %s", label, path)
    gdf = gpd.read_file(path)
    if type_col in gdf.columns:
        present = sorted(gdf[type_col].dropna().unique())
        LOGGER.info("  %s values present: %s", type_col, present)
        if type_value is not None:
            gdf = gdf[gdf[type_col] == type_value].copy()
            LOGGER.info("  filtered to %s = %s -> %d features", type_col, type_value, len(gdf))
    else:
        LOGGER.warning("  %s not found; using all %d features", type_col, len(gdf))

    if gdf.empty:
        raise SystemExit(f"No {label} features left after filtering {path}")

    if gdf.crs is not None and str(gdf.crs) != crs:
        LOGGER.info("  reprojecting %s -> %s", gdf.crs, crs)
        gdf = gdf.to_crs(crs)
    elif gdf.crs is None:
        LOGGER.warning("  %s has no CRS; assuming %s", label, crs)
        gdf = gdf.set_crs(crs)

    invalid = int((~gdf.geometry.is_valid).sum())
    if invalid:
        LOGGER.info("  repairing %d invalid geometries", invalid)
        gdf["geometry"] = gdf.geometry.buffer(0)

    LOGGER.info("  %d features, %.2f km2", len(gdf), gdf.geometry.area.sum() / 1e6)
    return gdf[["geometry"]]


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")

    boundary = load_by_type(args.boundary_shp, args.type_col, args.boundary_type,
                            args.crs, "state boundary")
    forest = load_by_type(args.forest_shp, args.type_col, args.forest_type,
                          args.crs, "forest")

    # Dissolve both sides first: erasing against overlapping polygons is slow and can
    # leave slivers, and the dissolved forest area is the figure to report.
    LOGGER.info("Dissolving state boundary")
    boundary = boundary.dissolve()[["geometry"]]
    area_boundary = boundary.geometry.area.sum() / 1e6

    LOGGER.info("Dissolving forest polygons (this is the slow step)")
    forest = forest.dissolve()[["geometry"]]
    area_forest = forest.geometry.area.sum() / 1e6

    LOGGER.info("Erasing forest from the state boundary")
    roi = gpd.overlay(boundary, forest, how="difference", keep_geom_type=True)
    roi = roi.dissolve()[["geometry"]]          # geometry only, no inherited attributes
    area_roi = roi.geometry.area.sum() / 1e6

    args.output.parent.mkdir(parents=True, exist_ok=True)
    roi.to_file(args.output, layer=args.layer, driver="GPKG")

    LOGGER.info("---- summary ----")
    LOGGER.info("state boundary (%s)   %12.2f km2", args.boundary_type, area_boundary)
    LOGGER.info("forest (%s)                    %12.2f km2", args.forest_type, area_forest)
    LOGGER.info("non-forest ROI                       %12.2f km2", area_roi)
    LOGGER.info("residual (boundary - forest - ROI)   %12.2f km2",
                area_boundary - area_forest - area_roi)
    LOGGER.info("written to %s (layer %s)", args.output, args.layer)


if __name__ == "__main__":
    main()