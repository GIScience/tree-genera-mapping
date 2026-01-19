#!/usr/bin/env python
"""
On-demand RGB/RGBI/RGBIH dataset builder using LGLDownloader + TileDataset.

For each selected tile in a GeoPackage:

  - Check if final merged tile exists: output_tiles/merged/<mode>_<tile_id>.tif
  - If missing:
      * Download dop20rgbi/dom1/dgm1 into a per-tile temp dir via LGLDownloader
      * Unzip into tmp_root/tile_id/unzipped/{dop20rgbi,dom1,dgm1}
      * Build the RGB/RGBI/RGBIH tile with TileDataset
      * Save to output_tiles/merged
      * Delete tmp_root/tile_id

Tile IDs are built from 'dop_kachel' as "32_355_6048".

Usage example (all tiles):

  python job_dataset.py \
      --tiles-gpkg data/urban_tiles.gpkg \
      --tmp-root /mnt/.../tmp_lgl_tiles \
      --output-tiles /mnt/.../tiles_rgbih \
      --mode RGBIH

Usage with index slicing (for SLURM arrays):

  python research_code/jobs/job_dataset.py \
      --tiles-gpkg /mnt/sds-hd/sd17f001/ygrin/silverways/greenspaces/OpenGeoData/tiles.gpkg \
      --tmp-root /mnt/sds-hd/sd17f001/ygrin/silverways/greenspaces/tmp_lgl_tiles \
      --output-tiles /mnt/sds-hd/sd17f001/ygrin/silverways/greenspaces/tiles_rgbih \
      --mode RGBIH \
      --index-start 0 \
      --index-end 100
"""

import argparse
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List

import geopandas as gpd
from tqdm import tqdm
import sys

# bring research code into path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
logging.info(PROJECT_ROOT)

from tree_genera_mapping.acquisition.lgl_downloader import LGLDownloader
from tree_genera_mapping.preprocess.preprocess_tile import TileDataset


# ------------- logging ------------- #

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ------------- helpers ------------- #

def _find_tile_file_for_xy(
    base_dir: Path,
    product_prefix: str,
    x: str,
    y: str,
    ext: str,
) -> Optional[Path]:
    """
    Find the file for a specific (x,y) tile inside an unzipped LGL product directory.

    We *only* require that the filename contains '_<x>_<y>_' or ends with '_<x>_<y><ext>'.
    We do NOT assume a fixed '_2_bw' suffix anymore.
    """
    if not base_dir.exists():
        return None

    # We accept both "..._x_y_..." and "..._x_y.ext"
    needle1 = f"_{x}_{y}_"
    needle2 = f"_{x}_{y}{ext}"

    # Look for all candidate rasters of this product
    # Example pattern: "dop20rgbi_32_*.tif"
    pattern = f"{product_prefix}_32_*{ext}"

    candidates = sorted(base_dir.rglob(pattern))
    if not candidates:
        logging.warning("No %s files at all found in %s", product_prefix, base_dir)
        return None

    matches = [
        c for c in candidates
        if (needle1 in c.name) or c.name.endswith(needle2)
    ]

    if not matches:
        logging.warning("No %s tile matching x=%s, y=%s in %s", product_prefix, x, y, base_dir)
        logging.info("Available candidates were: %s", [c.name for c in candidates])
        return None

    if len(matches) > 1:
        logging.warning(
            "Multiple %s tiles matching x=%s, y=%s in %s, picking first: %s",
            product_prefix, x, y, base_dir, matches[0]
        )

    return matches[0]


def download_tile_to_temp(tile_id: str, tmp_root: Path):
    """
    Download dop20rgbi, dom1, dgm1 for a single tile into a temp folder
    and unzip them.

    Layout:

      tmp_root/tile_id/
        ├── dop20rgbi/  (zips)
        ├── dom1/
        ├── dgm1/
        └── unzipped/
            ├── dop20rgbi/
            ├── dom1/
            └── dgm1/

    Returns
    -------
    dop_path, dom_path, dgm_path, tmp_tile_dir
    """
    prefix, x, y = tile_id.split("_")
    tmp_tile_dir = tmp_root / tile_id
    tmp_tile_dir.mkdir(parents=True, exist_ok=True)

    downloader = LGLDownloader(
        base_folder=str(tmp_tile_dir),
        delay_seconds=0,
        parallel=True,
        max_workers=3,
    )

    logging.info("⬇️ Downloading products for tile %s into %s", tile_id, tmp_tile_dir)
    products = ["dop20rgbi", "dom1", "dgm1"]

    for product in products:
        _ = downloader._download_single(product, x, y)

    # unzip into unzipped/<product>/
    unzipped_dir = tmp_tile_dir / "unzipped"
    unzipped_dir.mkdir(parents=True, exist_ok=True)

    for product in products:
        prod_zip_dir = tmp_tile_dir / product
        prod_unzip_dir = unzipped_dir / product
        prod_unzip_dir.mkdir(parents=True, exist_ok=True)

        if not prod_zip_dir.exists():
            continue

        for zip_fp in prod_zip_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_fp, "r") as zf:
                    zf.extractall(prod_unzip_dir)
            except zipfile.BadZipFile:
                logging.error("Corrupt ZIP for %s: %s", product, zip_fp)

    # --- pick the correct file for this (x,y) tile from the 2×2 block ---
    dop_path = _find_tile_file_for_xy(
        unzipped_dir / "dop20rgbi",
        product_prefix="dop20rgbi",
        x=x,
        y=y,
        ext=".tif",
    )
    dom_path = _find_tile_file_for_xy(
        unzipped_dir / "dom1",
        product_prefix="dom1",
        x=x,
        y=y,
        ext=".tif",
    )
    dgm_path = _find_tile_file_for_xy(
        unzipped_dir / "dgm1",
        product_prefix="dgm1",
        x=x,
        y=y,
        ext=".xyz",
    )

    if dop_path is None or dom_path is None or dgm_path is None:
        logging.warning("⚠️ Missing some inputs for tile %s in temp dir %s", tile_id, tmp_tile_dir)

    return dop_path, dom_path, dgm_path, tmp_tile_dir


def process_tile_on_demand(row,
                           tmp_root: Path,
                           output_tiles: Path,
                           mode: str,
                           keep_tmp: bool = False):
    """
    For a single tile:
    - check if final merged tile exists -> skip
    - otherwise:
      * download inputs into tmp_root
      * unzip under tmp_root/tile_id/unzipped
      * build RGB/RGBI/RGBIH tile via TileDataset
      * delete tmp folder (unless keep_tmp=True)
    """
    tile_id = row["tile_id"]
    merged_dir = output_tiles / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    out_path = merged_dir / f"{mode.lower()}_{tile_id}.tif"
    if out_path.exists():
        logging.info("✅ Merged tile exists, skipping: %s", out_path)
        return

    logging.info("🚀 Processing tile_id=%s", tile_id)

    dop_path, dom_path, dgm_path, tmp_tile_dir = download_tile_to_temp(tile_id, tmp_root)
    if dop_path is None or dom_path is None or dgm_path is None:
        logging.error("❌ Missing dop/dom/dgm for tile %s, skipping", tile_id)
        if not keep_tmp and tmp_tile_dir.exists():
            shutil.rmtree(tmp_tile_dir, ignore_errors=True)
        return

    tile = TileDataset(
        tile_id=tile_id,
        dop_path=str(dop_path),
        ndsm_path=None,
        dgm_path=str(dgm_path),
        dom_path=str(dom_path),
        output_dir=str(output_tiles),
        mode=mode,
        temp_dir=str(tmp_root),
    )
    result = tile.process()
    logging.info("💾 Created tile: %s", result)

    if not keep_tmp and tmp_tile_dir.exists():
        logging.info("🧹 Removing temp dir %s", tmp_tile_dir)
        shutil.rmtree(tmp_tile_dir, ignore_errors=True)


def run_dataset_phase_on_demand(tiles_gpkg: Path,
                                tmp_root: Path,
                                output_tiles: Path,
                                mode: str,
                                keep_tmp: bool = False,
                                index_start: int | None = None,
                                index_end: int | None = None):
    """
    Main driver: on-demand per-tile download & processing.

    If index_start/index_end are None -> process all tiles.
    Otherwise, process tiles in [index_start, index_end) by row index.
    """
    logging.info("🧱 On-demand dataset generation")
    logging.info("   tiles_gpkg   = %s", tiles_gpkg)
    logging.info("   tmp_root     = %s", tmp_root)
    logging.info("   output_tiles = %s", output_tiles)
    logging.info("   mode         = %s", mode)
    logging.info("   keep_tmp     = %s", keep_tmp)
    logging.info("   index_start  = %s", index_start)
    logging.info("   index_end    = %s", index_end)

    gdf = gpd.read_file(tiles_gpkg)
    if "dop_kachel" not in gdf.columns:
        raise ValueError("GeoPackage must contain 'dop_kachel' column")

    # build tile_id "32_355_6048" from dop_kachel string "323556048"
    gdf["tile_id"] = gdf["dop_kachel"].astype(str).apply(
        lambda x: f"{x[:2]}_{x[2:5]}_{x[5:9]}"
    )

    n_tiles = len(gdf)
    logging.info("Total tiles in GeoPackage: %d", n_tiles)

    # --- decide which rows to process ---
    if index_start is None and index_end is None:
        sel = gdf
        logging.info("Processing ALL tiles")
    else:
        # default / clamp
        if index_start is None:
            index_start = 0
        if index_end is None:
            index_end = n_tiles

        # if the whole range is after the last tile -> nothing to do
        if index_start >= n_tiles:
            logging.info(
                "Requested range [%d, %d) is outside available tiles (0..%d). Nothing to do.",
                index_start, index_end, n_tiles - 1
            )
            return

        # clip to dataset bounds
        index_start = max(0, index_start)
        index_end = min(index_end, n_tiles)

        if index_end <= index_start:
            logging.info(
                "Requested range [%d, %d) is empty after clipping. Nothing to do.",
                index_start, index_end
            )
            return

        sel = gdf.iloc[index_start:index_end]
        logging.info(
            "Processing tile indices [%d, %d) → %d tiles",
            index_start, index_end, len(sel)
        )

    tmp_root.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(sel.iterrows(), total=len(sel), desc="Tiles"):
        try:
            process_tile_on_demand(
                row=row,
                tmp_root=tmp_root,
                output_tiles=output_tiles,
                mode=mode,
                keep_tmp=keep_tmp,
            )
            logging.info(f"Tile-{row['tile_id']} is Done!")
        except Exception as e:
            logging.error("❌ Failed to process tile %s: %s", row.get("tile_id", "<unknown>"), e)


# ------------- CLI ------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="On-demand RGB/RGBI/RGBIH dataset builder")

    parser.add_argument(
        "--tiles-gpkg",
        required=True,
        help="Path to the GeoPackage with a 'dop_kachel' column",
    )
    parser.add_argument(
        "--tmp-root",
        required=True,
        help="Temporary folder for per-tile downloads",
    )
    parser.add_argument(
        "--output-tiles",
        required=True,
        help="Output folder; script writes to OUTPUT/merged/",
    )
    parser.add_argument(
        "--mode",
        choices=["RGB", "RGBI", "RGBIH"],
        default="RGBIH",
        help="Dataset variant; RGBIH gives 5 bands (RGBI + H)",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Do NOT delete per-tile tmp directories (for debugging)",
    )
    parser.add_argument(
        "--index-start",
        type=int,
        default=None,
        help="Optional 0-based start index (inclusive) of tiles to process",
    )
    parser.add_argument(
        "--index-end",
        type=int,
        default=None,
        help="Optional 0-based end index (exclusive) of tiles to process",
    )

    return parser.parse_args()


def main():
    configure_logging()
    args = parse_args()

    tiles_gpkg = Path(args.tiles_gpkg)
    tmp_root = Path(args.tmp_root)
    output_tiles = Path(args.output_tiles)

    run_dataset_phase_on_demand(
        tiles_gpkg=tiles_gpkg,
        tmp_root=tmp_root,
        output_tiles=output_tiles,
        mode=args.mode,
        keep_tmp=args.keep_tmp,
        index_start=args.index_start,
        index_end=args.index_end,
    )


if __name__ == "__main__":
    main()