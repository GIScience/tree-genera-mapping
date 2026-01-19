#!/usr/bin/env python3
"""
Build merged RGBIH tiles AND write per-tile JSON height metadata.

For each tile_id (e.g. "32_355_6048") selected from a GeoPackage:

  - If output exists: output/merged/rgbih_<tile_id>.tif
    - skip unless --overwrite

  - Else:
      * Download dop20rgbi + dom1 + dgm1 using LGLDownloader
      * Unzip into tmp_root/<tile_id>/unzipped/{dop20rgbi,dom1,dgm1}
      * Build merged RGBIH tile using TileDataset (creates the TIFF)
      * Compute height stats (min/max meters) from DOM - DGM
      * Write JSON next to TIFF:

        rgbih_<tile_id>.json:
        {
          "tile_id": "...",
          "mode": "RGBIH",
          "height_channel": {
            "band_index_1based": 5,
            "stats_m": [min_height_m, max_height_m],
            "note": "Use these stats to unnormalize the height channel back to meters."
          }
        }

Does NOT modify any existing TIFF unless --overwrite.
"""

import argparse
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from tqdm import tqdm
import sys

# bring project into path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tree_genera_mapping.acquisition.lgl_store import LGLDownloader
from tree_genera_mapping.preprocess.preprocess_tile import TileDataset

NOTE = "Use these stats to unnormalize the height channel back to meters."
# ------------------------- logging -------------------------
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

# ------------------------- io helpers -------------------------
def atomic_write_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)

def _find_single_file(base_dir: Path, patterns: List[str]) -> Optional[Path]:
    """Return first match under base_dir for patterns (rglob)."""
    if not base_dir.exists():
        return None
    for pattern in patterns:
        candidates = sorted(base_dir.rglob(pattern))
        if candidates:
            return candidates[0]
    return None


# ------------------------- tile id selection -------------------------
def load_tile_ids_from_gpkg(tiles_gpkg: Path) -> List[str]:
    """Read gpkg with 'dop_kachel' and return tile_ids like '32_355_6048'."""
    gdf = gpd.read_file(tiles_gpkg)
    if "dop_kachel" not in gdf.columns:
        raise ValueError("GeoPackage must contain 'dop_kachel' column.")

    tile_ids = gdf["dop_kachel"].astype(str).apply(lambda x: f"{x[:2]}_{x[2:5]}_{x[5:9]}").tolist()
    return tile_ids


# ------------------------- download/unzip -------------------------
def download_tile_to_temp(
    tile_id: str,
    tmp_root: Path,
    products: Tuple[str, ...],
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path], Path]:
    """
    Download products for a tile into tmp_root/<tile_id>/ and unzip.

    Returns: dop_path (.tif), ndom_path(.tif) , dom_path (.tif), dgm_path (.xyz), tmp_tile_dir
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

    logging.info("⬇️ Downloading %s for tile %s", list(products), tile_id)
    for product in products:
        _ = downloader._download_single(product, x, y)

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

    # LGL naming stub (as in your existing script)
    stub = f"{prefix}_{x}_{y}_2_bw"

    dop_path = _find_single_file(
        unzipped_dir / "dop20rgbi",
        [f"dop20rgbi_{stub}.tif", "dop20rgbi_*.tif", "*.tif"],
    )
    ndom_path = _find_single_file(
        unzipped_dir / "ndom1",
        [f"ndom1_{stub}.tif", "ndom1_*.tif", "*.tif"],
    )
    dom_path = _find_single_file(
        unzipped_dir / "dom1",
        [f"dom1_{stub}.tif", "dom1_*.tif", "*.tif"],
    )
    dgm_path = _find_single_file(
        unzipped_dir / "dgm1",
        [f"dgm1_{stub}.xyz", "dgm1_*.xyz", "*.xyz"],
    )

    return dop_path, ndom_path, dom_path, dgm_path, tmp_tile_dir


# ------------------------- height stats -------------------------
def height_min_max_from_dom_dgm(dom_path: Path, dgm_xyz_path: Path) -> Tuple[float, float]:
    """
    Compute min/max of height (meters) = DOM - DGM.
    Reads DOM raster and DGM XYZ points, maps points to DOM pixels.
    """
    with rasterio.open(dom_path) as src:
        dom = src.read(1, masked=True).astype(np.float32, copy=False)
        dom_arr = dom.filled(np.nan) if np.ma.isMaskedArray(dom) else dom
        H, W = dom_arr.shape
        transform = src.transform

    vals = np.fromfile(dgm_xyz_path, sep=" ")
    if vals.size % 3 != 0:
        raise ValueError(f"Unexpected XYZ format: {dgm_xyz_path} (count={vals.size})")

    vals = vals.reshape(-1, 3)
    xs = vals[:, 0]
    ys = vals[:, 1]
    zs = vals[:, 2].astype(np.float32, copy=False)

    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    m = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & np.isfinite(zs)
    if not m.any():
        raise ValueError(f"No DGM points mapped into DOM extent for {dom_path.name}")

    dom_vals = dom_arr[rows[m], cols[m]]
    diffs = dom_vals - zs[m]
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError(f"No finite height diffs for {dom_path.name}")

    return float(diffs.min()), float(diffs.max())


# ------------------------- main per-tile -------------------------
def build_rgbih_and_json_for_tile(
    tile_id: str,
    tmp_root: Path,
    output_tiles: Path,
    mode: str,
    products: Tuple[str, ...],
    keep_tmp: bool = False,
    overwrite: bool = False,
):

    merged_dir = output_tiles / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    out_tif  = merged_dir / f"rgbih_{tile_id}.tif"
    out_json = merged_dir / f"rgbih_{tile_id}.json"

    if out_tif.exists() and out_json.exists() and (not overwrite):
        logging.info("✅ Exists (tif+json), skipping: %s", tile_id)
        return

    dop_path, ndom_path, dom_path, dgm_path, tmp_tile_dir = download_tile_to_temp(
        tile_id=tile_id,
        tmp_root=tmp_root,
        products=products,
    )
    mode = mode.upper()

    if mode == "RGB":
        ndom_path = None
        dom_path = None
        dgm_path = None

    elif mode == "RGBI":
        ndom_path = None
        dom_path = None
        dgm_path = None

    elif mode == "RGBIH":
        # same logic we already discussed:
        # if ndom_path is None → build from dom1-dgm1
        ...
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if dop_path is None or dom_path is None or dgm_path is None:
        logging.error("❌ Missing dop/dom/dgm for tile %s. dop=%s dom=%s dgm=%s", tile_id, dop_path, dom_path, dgm_path)
        if not keep_tmp and tmp_tile_dir.exists():
            shutil.rmtree(tmp_tile_dir, ignore_errors=True)
        return

    # Build tile (RGBIH = RGB + NIR + height band)
    tile = TileDataset(
        tile_id=tile_id,
        dop_path=str(dop_path),
        ndom_path=str(ndom_path),
        dgm_path=str(dgm_path),
        dom_path=str(dom_path),
        output_dir=str(output_tiles),
        mode="RGBIH",
        temp_dir=str(tmp_root),
    )
    _ = tile.process()

    # Height min/max (meters)
    mn, mx = height_min_max_from_dom_dgm(dom_path, dgm_path)

    # Write json next to tif (in merged/)
    obj = {
        "tile_id": tile_id,
        "mode": "RGBIH",
        "height_channel": {
            "band_index_1based": 5,
            "stats_m": [mn, mx],
            "note": NOTE,
        },
    }
    atomic_write_json(obj, out_json)
    logging.info("🧾 Wrote JSON: %s", out_json)

    if not keep_tmp and tmp_tile_dir.exists():
        shutil.rmtree(tmp_tile_dir, ignore_errors=True)

def products_for_mode(mode: str) -> Tuple[str, ...]:
    mode = mode.upper()

    if mode == "RGB":
        return ("dop20rgb",)

    if mode == "RGBI":
        return ("dop20rgbi",)

    if mode == "RGBIH":
        # ndom1 is optional at runtime; dom1+dgm1 are needed for fallback + stats
        return ("dop20rgbi", "ndom1", "dom1", "dgm1")

    raise ValueError(f"Unsupported mode: {mode}")
# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build RGBIH tiles + JSON metadata")

    p.add_argument("--tiles-gpkg", required=True, help="GeoPackage containing 'dop_kachel'")
    p.add_argument("--tmp-root", required=True, help="Temp folder for downloads/unzips")
    p.add_argument("--output-tiles", required=True, help="Output folder; writes to OUTPUT/merged/")
    p.add_argument("--mode", default="RGBIH", help="TileDataset mode - RGB, RGBI, RGBIH")
    p.add_argument("--keep-tmp", action="store_true", help="Keep per-tile temp directories")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # Preferred for open-access usage:
    p.add_argument(
        "--tile-ids",
        nargs="*",
        default=None,
        help="Optional explicit tile_ids like: 32_355_6048 32_356_6048. If omitted, uses all tiles from --tiles-gpkg.",
    )
    return p.parse_args()


def main():
    configure_logging()
    args = parse_args()

    mode = args.mode.upper()
    products = products_for_mode(mode)
    logging.info("Mode=%s → downloading products: %s", mode, products)

    tiles_gpkg = Path(args.tiles_gpkg)
    tmp_root = Path(args.tmp_root)
    output_tiles = Path(args.output_tiles)

    tmp_root.mkdir(parents=True, exist_ok=True)
    output_tiles.mkdir(parents=True, exist_ok=True)

    if args.tile_ids and len(args.tile_ids) > 0:
        tile_ids = args.tile_ids
        logging.info("Processing %d explicit tile_ids", len(tile_ids))
    else:
        tile_ids = load_tile_ids_from_gpkg(tiles_gpkg)
        logging.info("Processing ALL tiles from gpkg: %d tiles", len(tile_ids))

    for tile_id in tqdm(tile_ids, desc="Tiles"):
        try:
            build_rgbih_and_json_for_tile(
                tile_id=tile_id,
                tmp_root=tmp_root,
                output_tiles=output_tiles,
                keep_tmp=args.keep_tmp,
                overwrite=args.overwrite,
                mode=mode,
                products=products,
            )
        except Exception as e:
            logging.exception("❌ Failed tile %s: %s", tile_id, e)


if __name__ == "__main__":
    main()
