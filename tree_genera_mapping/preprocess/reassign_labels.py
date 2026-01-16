"""
reassign_labels.py

Assign training classes to tree records using:
- Hard-coded coniferous genus mapping
- Genera label mapping CSV (genus -> fid)

Rules:
  - If genus is coniferous -> training_class = 3
  - Else if genus in genera_labels.csv -> training_class = fid
  - Else -> training_class = 5  (Other Deciduous)

Outputs:
  GeoPackage with columns: uuid, geometry, genus, training_class
"""

from __future__ import annotations

import uuid
import geopandas as gpd
import pandas as pd
from pathlib import Path

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
    """

    # --- normalize input ---
    gdf = gdf_trees.copy()
    gdf["genus"] = gdf[genus_col].map(_norm_genus)

    # --- drop trees without genus ---
    before = len(gdf)
    gdf = gdf[gdf["genus"] != ""].copy()
    print(f"Dropped {before - len(gdf)} trees with missing genus")

    # --- normalize class table ---
    dfc = df_classes.copy()
    dfc["genus"] = dfc["genus"].map(_norm_genus)

    class_map = dict(zip(dfc["genus"], dfc["fid"]))

    # --- required classes ---
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
    if id_col in gdf.columns:
        gdf["uuid"] = gdf["tree_id"]
    else:
        gdf["uuid"] = [uuid.uuid4() for _ in range(len(gdf))]

    return gdf[["uuid", "geometry", "genus", "training_class"]]


def generate_training_labels(
    trees_path: str | None,
    labels_path: str = "conf/genera_labels.csv",
    output_path: str = "cache/tree_labels.gpkg",
) -> gpd.GeoDataFrame:
    """
    Generate training labels for tree genera:
    1. trees_path must be provided by the user.
    2. Output is written to cache/ by default.
    """
    if trees_path is None:
        raise ValueError(
            "trees_path is required. "
            "Please provide the path to the downloaded tree dataset."
        )
    gdf_trees = gpd.read_file(trees_path)
    df_classes = pd.read_csv(labels_path)

    labeled = assign_training_classes(gdf_trees, df_classes)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_file(output_path, driver="GPKG")
    return labeled

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Assign training classes to tree data.")
    ap.add_argument("--trees", required=True, help="Path to input tree GeoPackage")
    ap.add_argument("--labels", default="conf/genera_labels.csv", help="Path to genera_labels.csv")
    ap.add_argument("--output", default="cache/tree_labels.gpkg", help="Output GPKG path")
    args = ap.parse_args()

    generate_training_labels(
        trees_path=args.trees,
        labels_path=args.labels,
        output_path=args.output,
    )