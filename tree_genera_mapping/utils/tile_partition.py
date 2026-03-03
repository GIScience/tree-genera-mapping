from __future__ import annotations

import geopandas as gpd
import re

def dopkachel_to_tile_id(dop_kachel: str) -> str:
    """
    Convert BW dop_kachel like '323556048' -> '32_355_6048'.

    Expects exactly 9 digits (after cleaning).
    """
    s = str(dop_kachel).strip()

    # If it looks like a float or has decimals, try to coerce cleanly
    # (common when reading from CSV/Excel)
    if re.fullmatch(r"\d+(\.0+)?", s):
        s = s.split(".")[0]

    # Keep digits only (in case of accidental spaces)
    s_digits = re.sub(r"\D", "", s)

    if len(s_digits) != 9:
        raise ValueError(f"dop_kachel must be exactly 9 digits, got '{dop_kachel}' -> '{s_digits}'")

    return f"{s_digits[:2]}_{s_digits[2:5]}_{s_digits[5:9]}"

def ensure_tile_id_from_grid(
    gdf_tiles: gpd.GeoDataFrame,
    *,
    tile_id_col: str = "tile_id",
    target_epsg: int = 25832,
    zone_prefix: str = "32",
    grid_m: int = 1000,
    overwrite: bool = False,
) -> gpd.GeoDataFrame:
    """
    Ensure a deterministic tile_id column based on spatial grid partition.

    tile_id = f"{zone_prefix}_{floor(cx/grid_m)}_{floor(cy/grid_m)}"
    where (cx, cy) is the centroid in target_epsg (meters).

    This guarantees reproducible spatial partitioning suitable for
    leakage-safe dataset splitting.
    """

    if gdf_tiles.empty:
        raise ValueError("Tiles GeoDataFrame is empty.")

    if gdf_tiles.crs is None:
        raise ValueError("Tiles GeoDataFrame has no CRS defined.")

    gdf = gdf_tiles.copy()

    if (tile_id_col in gdf.columns) and not overwrite:
        gdf[tile_id_col] = gdf[tile_id_col].astype(str)
        return gdf

    if gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)

    centroids = gdf.geometry.centroid
    e_km = (centroids.x // grid_m).astype(int)
    n_km = (centroids.y // grid_m).astype(int)

    gdf[tile_id_col] = (
        zone_prefix
        + "_"
        + e_km.astype(str)
        + "_"
        + n_km.astype(str)
    )

    if gdf[tile_id_col].duplicated().any():
        dupes = gdf.loc[gdf[tile_id_col].duplicated(), tile_id_col].head(10).tolist()
        raise ValueError(
            f"Duplicate tile_id detected (examples): {dupes}. "
            f"Check CRS and grid size."
        )

    return gdf