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

Outputs:
  GeoPackage with columns: uuid, geometry, genus, training_class
"""

from __future__ import annotations

import uuid
from pathlib import Path

import geopandas as gpd
import pandas as pd

CONIFEROUS_MAP = {
    "Abies", "Araucaria", "Calocedrus", "Cedrus", "Chamaecyparis",
    "Cupressus", "Hesperocyparis", "Juniperus", "Larix",
    "Metasequoia", "Picea", "Pinus", "Platycladus",
    "Pseudotsuga", "Sequoia", "Sequoiadendron",
    "Taxodium", "Taxus", "Thuja", "Tsuga", "Wollemia",
}


def _norm_genus(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def assign_training_classes(
    gdf_trees: gpd.GeoDataFrame,
    df_classes: pd.DataFrame,
    genus_col: str = "genus",
    id_col: str = "tree_id",
) -> gpd.GeoDataFrame:
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


def generate_training_labels(
    trees_path: str | None,
    labels_path: str = "conf/genera_labels.csv",
    output_path: str = "cache/tree_labels.gpkg",
) -> gpd.GeoDataFrame:
    if trees_path is None:
        raise ValueError("trees_path is required. Please provide the path to the downloaded tree dataset.")

    gdf_trees = gpd.read_file(trees_path)
    df_classes = pd.read_csv(labels_path)

    labeled = assign_training_classes(gdf_trees, df_classes)

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
    args = ap.parse_args()

    # Pass custom columns if users need them
    # (keeps script usable with different inventories)
    gdf = gpd.read_file(args.trees)
    dfc = pd.read_csv(args.labels)
    labeled = assign_training_classes(gdf, dfc, genus_col=args.genus_col, id_col=args.id_col)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_file(out_path, driver="GPKG")
