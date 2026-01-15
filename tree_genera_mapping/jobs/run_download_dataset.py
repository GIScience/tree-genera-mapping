#!/usr/bin/env python3
"""
Fill missing per-tile JSONs next to existing merged RGBIH tiles.

Folder layout (input_dir):
  rgbih_<tile_id>.tif
  rgbih_<tile_id>.json

If JSON missing:
  - download nDOM1 or [DOM1 + DGM1] for that tile using LGLDownloader
  - compute height (meters) = DOM - DGM
  - write JSON with schema:

{
  "tile_id": "...",
  "mode": "RGBIH",
  "height_channel": {
    "band_index_1based": 5,
    "stats": [min_height_m, max_height_m],
    "note": "Use these stats to unnormalize the height channel back to meters."
  }
}

Does NOT modify any existing TIFFs.
"""

import argparse
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import rasterio
from tqdm import tqdm
import sys

# bring research code into path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tree_genera_mapping.acquisition.lgl_downloader import LGLDownloader  # noqa: E402


NOTE = "Use these stats to unnormalize the height channel back to meters."


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def atomic_write_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, out_path)


def extract_tile_id_from_rgbih_tif(tif_path: Path, tif_prefix: str = "rgbih") -> str:
    # expects: rgbih_<tile_id>.tif
    stem = tif_path.stem
    prefix = f"{tif_prefix}_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected tif name (expected '{prefix}...'): {tif_path.name}")
    return stem[len(prefix):]


def _find_tile_file_for_xy(
    base_dir: Path,
    product_prefix: str,
    x: str,
    y: str,
    ext: str,
) -> Optional[Path]:
    """
    Find the file for specific (x,y) tile inside an unzipped product directory.
    Accept filenames containing '_<x>_<y>_' OR ending with '_<x>_<y><ext>'.
    """
    if not base_dir.exists():
        return None

    needle1 = f"_{x}_{y}_"
    needle2 = f"_{x}_{y}{ext}"
    pattern = f"{product_prefix}_32_*{ext}"

    candidates = sorted(base_dir.rglob(pattern))
    if not candidates:
        logging.warning("No %s files found in %s", product_prefix, base_dir)
        return None

    matches = [c for c in candidates if (needle1 in c.name) or c.name.endswith(needle2)]
    if not matches:
        logging.warning("No %s match for x=%s y=%s in %s", product_prefix, x, y, base_dir)
        logging.info("Candidates were: %s", [c.name for c in candidates])
        return None

    if len(matches) > 1:
        logging.warning("Multiple matches for %s x=%s y=%s; picking %s", product_prefix, x, y, matches[0].name)

    return matches[0]


def download_dom_dgm(tile_id: str, tmp_root: Path) -> Tuple[Optional[Path], Optional[Path], Path]:
    """
    Download dom1 + dgm1 into tmp_root/tile_id, unzip, and return selected:
      dom_path (.tif), dgm_path (.xyz), tmp_tile_dir
    """
    _, x, y = tile_id.split("_")
    tmp_tile_dir = tmp_root / tile_id
    tmp_tile_dir.mkdir(parents=True, exist_ok=True)

    downloader = LGLDownloader(
        base_folder=str(tmp_tile_dir),
        delay_seconds=0,
        parallel=True,
        max_workers=3,
    )

    products = ["dom1", "dgm1"]
    logging.info("⬇️ Downloading %s for tile %s", products, tile_id)

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

    dom_path = _find_tile_file_for_xy(unzipped_dir / "dom1", "dom1", x, y, ".tif")
    dgm_path = _find_tile_file_for_xy(unzipped_dir / "dgm1", "dgm1", x, y, ".xyz")

    return dom_path, dgm_path, tmp_tile_dir


def height_min_max_from_dom_dgm(dom_path: Path, dgm_xyz_path: Path) -> Tuple[float, float]:
    """
    Compute min/max of height (meters) = DOM - DGM.

    Strategy:
      - read DOM raster (band 1) to array (masked->nan)
      - read DGM xyz (x y z) using np.fromfile (fast)
      - map xyz coords to DOM pixel indices via rasterio.transform.rowcol (vectorized)
      - compute diffs at those pixels, then min/max over finite diffs
    """
    with rasterio.open(dom_path) as src:
        dom = src.read(1, masked=True)
        dom = dom.astype(np.float32, copy=False)
        dom_arr = dom.filled(np.nan) if np.ma.isMaskedArray(dom) else dom
        H, W = dom_arr.shape
        transform = src.transform

    # Fast parse: whitespace-separated xyz -> 1D array
    vals = np.fromfile(dgm_xyz_path, sep=" ")
    if vals.size % 3 != 0:
        raise ValueError(f"Unexpected XYZ format in {dgm_xyz_path} (count={vals.size}, not multiple of 3)")

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


def list_rgbih_tifs(input_dir: Path, tif_prefix: str = "rgbih") -> List[Path]:
    return sorted(input_dir.glob(f"{tif_prefix}_*.tif"))


def run(args):
    input_dir = Path(args.input_dir)
    tmp_root = Path(args.tmp_root)
    tif_prefix = args.tif_prefix

    tifs = list_rgbih_tifs(input_dir, tif_prefix=tif_prefix)
    if not tifs:
        logging.error("No TIFFs found in %s matching %s_*.tif", input_dir, tif_prefix)
        return

    # only process missing JSONs
    missing = []
    for tif in tifs:
        tile_id = extract_tile_id_from_rgbih_tif(tif, tif_prefix=tif_prefix)
        j = input_dir / f"{tif_prefix}_{tile_id}.json"
        if not j.exists() or args.overwrite:
            missing.append(tif)

    logging.info("Found %d TIFFs total; %d missing JSONs (or overwrite enabled)", len(tifs), len(missing))
    if not missing:
        return

    # slice over missing set (good for SLURM arrays)
    n = len(missing)
    start = 0 if args.index_start is None else max(0, args.index_start)
    end = n if args.index_end is None else min(n, args.index_end)
    if end <= start:
        logging.info("Empty slice [%d, %d). Nothing to do.", start, end)
        return

    subset = missing[start:end]
    logging.info("Processing missing-json TIFF indices [%d, %d) -> %d tiles", start, end, len(subset))

    tmp_root.mkdir(parents=True, exist_ok=True)

    created = skipped = failed = 0

    for tif_path in tqdm(subset, desc="Tiles"):
        tile_id = extract_tile_id_from_rgbih_tif(tif_path, tif_prefix=tif_prefix)
        json_path = input_dir / f"{tif_prefix}_{tile_id}.json"

        # race-safe skip (parallel SLURM)
        if json_path.exists() and not args.overwrite:
            skipped += 1
            continue

        dom_path = dgm_path = None
        tmp_tile_dir = None

        try:
            dom_path, dgm_path, tmp_tile_dir = download_dom_dgm(tile_id, tmp_root)
            if dom_path is None or dgm_path is None:
                raise RuntimeError(f"Missing DOM/DGM after download (dom={dom_path}, dgm={dgm_path})")

            mn, mx = height_min_max_from_dom_dgm(dom_path, dgm_path)

            obj = {
                "tile_id": tile_id,
                "mode": args.mode,
                "height_channel": {
                    "band_index_1based": int(args.height_band_1based),
                    "stats": [mn, mx],
                    "note": NOTE,
                },
            }

            atomic_write_json(obj, json_path)
            created += 1

        except Exception as e:
            logging.exception("❌ Failed tile %s: %s", tile_id, e)
            failed += 1

        finally:
            if (not args.keep_tmp) and tmp_tile_dir and tmp_tile_dir.exists():
                shutil.rmtree(tmp_tile_dir, ignore_errors=True)

    logging.info("Done. created=%d skipped=%d failed=%d", created, skipped, failed)


def parse_args():
    p = argparse.ArgumentParser(description="Download height inputs for tiles missing JSON and write JSON next to existing RGBIH TIFFs.")
    p.add_argument("--input-dir", required=True, help="Folder containing rgbih_<tile_id>.tif and (maybe) rgbih_<tile_id>.json")
    p.add_argument("--tmp-root", required=True, help="Temp folder used for LGL downloads/unzips")
    p.add_argument("--tif-prefix", default="rgbih", help="TIFF prefix (default: rgbih)")
    p.add_argument("--mode", default="RGBIH", choices=["RGB", "RGBI", "RGBIH"], help="Value written to JSON")
    p.add_argument("--height-band-1based", type=int, default=5, help="Value written to JSON (default: 5)")
    p.add_argument("--index-start", type=int, default=None, help="Slice start over *missing-json* tiles list")
    p.add_argument("--index-end", type=int, default=None, help="Slice end over *missing-json* tiles list")
    p.add_argument("--keep-tmp", action="store_true", help="Do not delete tmp per-tile dirs")
    p.add_argument("--overwrite", action="store_true", help="Overwrite JSON if it exists")
    return p.parse_args()


def main():
    configure_logging()
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
