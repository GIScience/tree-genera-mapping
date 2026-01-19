"""
reassign_labels.py

Assign training classes to tree records using:
- Hard-coded coniferous genus mapping
- Genera label mapping CSV (genus -> fid)

Rules:
  - Drop rows with missing genus
  - If genus is coniferous -> training_class = fid("Coniferous")
  - Else if genus in genera_labels.csv -> training_class = fid
  - Else -> training_class = fid("Other Deciduous")

Optional:
  - Generate square bounding boxes around tree point locations using canopy width.

Outputs:
  GeoPackage with columns: uuid, geometry, genus, training_class
"""

from __future__ import annotations

import uuid
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tree_genera_mapping.preprocess.utils import _make_bbox

# -----------------------------
# Coniferous genus mapping
# -----------------------------
CONIFEROUS_MAP = {
    "Abies", "Araucaria", "Calocedrus", "Cedrus", "Chamaecyparis",
    "Cupressus", "Hesperocyparis", "Juniperus", "Larix",
    "Metasequoia", "Picea", "Pinus", "Platycladus",
    "Pseudotsuga", "Sequoia", "Sequoiadendron",
    "Taxodium", "Taxus", "Thuja", "Tsuga", "Wollemia",
}


def _norm_genus(x) -> str:
    """Normalize genus strings for reliable matching."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def assign_training_classes(
    gdf_trees: gpd.GeoDataFrame,
    df_classes: pd.DataFrame,
    genus_col: str = "genus",
    id_col: str = "tree_id",
) -> gpd.GeoDataFrame:
    """
    Assign training_class using genera_labels.csv.
    Keeps stable IDs if `id_col` exists.
    Drops rows with missing genus.
    """
    # Validate inputs
    if genus_col not in gdf_trees.columns:
        raise ValueError(f"Input trees must contain column '{genus_col}'")
    if gdf_trees.geometry is None:
        raise ValueError("Input trees must contain a geometry column")

    if not {"genus", "fid"}.issubset(df_classes.columns):
        raise ValueError("genera_labels.csv must contain columns: 'genus', 'fid'")

    gdf = gdf_trees.copy()
    gdf["genus"] = gdf[genus_col].map(_norm_genus)

    # Drop missing genus
    before = len(gdf)
    gdf = gdf[gdf["genus"] != ""].copy()
    print(f"Dropped {before - len(gdf)} trees with missing genus")

    # Normalize class table + mapping
    dfc = df_classes.copy()
    dfc["genus"] = dfc["genus"].map(_norm_genus)
    class_map = dict(zip(dfc["genus"], dfc["fid"]))

    # Required group classes
    if "Coniferous" not in class_map:
        raise ValueError("genera_labels.csv must contain a 'Coniferous' class")
    if "Other Deciduous" not in class_map:
        raise ValueError("genera_labels.csv must contain an 'Other Deciduous' class")

    coniferous_id = int(class_map["Coniferous"])
    other_deciduous_id = int(class_map["Other Deciduous"])

    def _assign(genus: str) -> int:
        if genus in CONIFEROUS_MAP:
            return coniferous_id
        if genus in class_map:
            return int(class_map[genus])
        return other_deciduous_id

    gdf["training_class"] = gdf["genus"].map(_assign)

    # Keep stable ids if provided
    if id_col in gdf.columns:
        gdf["uuid"] = gdf[id_col]
    else:
        gdf["uuid"] = [uuid.uuid4() for _ in range(len(gdf))]

    return gdf[["uuid", "geometry", "genus", "training_class"]]


def add_bbox_from_canopy_width(
    gdf: gpd.GeoDataFrame,
    canopy_col: str = "canopyWidt",
    min_width: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Create square bounding boxes from a point geometry and canopy width (meters).

    - Assumes geometry is Point
    - canopy_col values are in meters
    - bbox is a square with side length = canopy width
    """
    if canopy_col not in gdf.columns:
        raise ValueError(f"Missing canopy width column '{canopy_col}'")

    # Needs projected CRS in meters
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS. Cannot build metric bboxes safely.")
    if getattr(gdf.crs, "is_geographic", False):
        raise ValueError(
            f"CRS appears geographic ({gdf.crs}). Reproject to a metric CRS (e.g., EPSG:25832) before bbox creation."
        )

    widths = pd.to_numeric(gdf[canopy_col], errors="coerce")
    widths = widths.where(widths >= min_width)


    out = gdf.copy()
    out["bbox"] = [_make_bbox(geom, w) for geom, w in zip(out.geometry, widths)]

    # Drop rows where bbox couldn't be created
    before = len(out)
    out = out[out["bbox"].notna()].copy()
    print(f"Dropped {before - len(out)} trees with missing/invalid {canopy_col} for bbox.")

    # Replace geometry safely
    out["geometry"] = out["bbox"]
    out = out.drop(columns=["bbox"]).set_geometry("geometry")

    return out


def generate_training_labels(
    trees_path: str | None,
    labels_path: str = "conf/genera_labels.csv",
    output_path: str = "cache/tree_labels.gpkg",
    genus_col: str = "genus",
    id_col: str = "tree_id",
    make_bbox: bool = False,
    canopy_col: str = "canopyWidt",
) -> gpd.GeoDataFrame:
    """
    Generate training labels for tree genera.
    - trees_path must be provided by the user
    - output directory is created automatically
    """
    if trees_path is None:
        raise ValueError("trees_path is required. Please provide the path to the downloaded tree dataset.")

    gdf_trees = gpd.read_file(trees_path)
    df_classes = pd.read_csv(labels_path)

    labeled = assign_training_classes(
        gdf_trees=gdf_trees,
        df_classes=df_classes,
        genus_col=genus_col,
        id_col=id_col,
    )

    if make_bbox:
        # need the canopy column present in the source (copy it over)
        if canopy_col not in gdf_trees.columns:
            raise ValueError(f"make_bbox=True but '{canopy_col}' not found in input trees.")
        labeled = labeled.merge(
            gdf_trees[[id_col, canopy_col]],
            left_on="uuid" if id_col in gdf_trees.columns else None,
            right_on=id_col if id_col in gdf_trees.columns else None,
            how="left",
        )
        labeled = add_bbox_from_canopy_width(labeled, canopy_col=canopy_col)

        # keep only expected columns
        labeled = labeled[["uuid", "geometry", "genus", "training_class"]]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_file(out_path, driver="GPKG")
    return labeled


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Assign training classes to tree data.")
    ap.add_argument("--trees", required=True, help="Path to input tree GeoPackage")
    ap.add_argument("--labels", default="conf/genera_labels.csv", help="Path to genera_labels.csv")
    ap.add_argument("--output", default="cache/tree_labels.gpkg", help="Output GPKG path")
    ap.add_argument("--id-col", default="tree_id", help="ID column to preserve if present")
    ap.add_argument("--genus-col", default="genus", help="Genus column name")

    ap.add_argument("--make-bbox", action="store_true", help="Replace point geometry with bbox from canopy width")
    ap.add_argument("--canopy-col", default="canopyWidt", help="Column containing canopy width in meters")

    args = ap.parse_args()

    generate_training_labels(
        trees_path=args.trees,
        labels_path=args.labels,
        output_path=args.output,
        genus_col=args.genus_col,
        id_col=args.id_col,
        make_bbox=args.make_bbox,
        canopy_col=args.canopy_col,
    )
