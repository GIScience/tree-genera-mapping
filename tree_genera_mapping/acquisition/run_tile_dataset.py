#!/usr/bin/env python3
"""
Download + build merged tiles (RGB / RGBI / RGBIH) and (for RGBIH) write per-tile JSON
with height normalization stats in meters.

Key behavior
- Mode RGB   : downloads dop20rgb, writes merged/rgb_<tile_id>.tif
- Mode RGBI  : downloads dop20rgbi, writes merged/rgbi_<tile_id>.tif
- Mode RGBIH : downloads dop20rgbi + ndom1 + dom1 + dgm1
    * If ndom1 is missing, TileDataset will compute CHM from dom1+dgm1 internally.
    * Writes merged/rgbih_<tile_id>.tif and merged/rgbih_<tile_id>.json

JSON stats_m
- IMPORTANT: stats_m are the SAME (vmin,vmax) used to normalize the height channel to uint8.
- We get them by asking TileDataset for height stats (requires small additions to TileDataset, see note below).

NOTE (required change in TileDataset):
Your TileDataset currently returns only the uint8 height band; to emit correct stats for unnormalization,
TileDataset must expose (vmin_m, vmax_m) used in normalize_hm_to_255(). Implement the small change:
- _load_height() -> return (height_u8, (vmin, vmax))
- _get_height()  -> return same tuple
- process()      -> store last_height_stats_m and last_height_source on the instance

This script expects TileDataset to have after process():
- tile.last_height_stats_m  : Optional[Tuple[float,float]]
- tile.last_height_source   : Optional[str]   ("ndom1" or "dom1-dgm1")

If you don't want to touch TileDataset, you can fall back to "raw dom-dgm min/max",
but that will NOT perfectly invert your normalization (because normalization uses post-warp vmin/vmax).
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
    return (
        gdf["dop_kachel"]
        .astype(str)
        .apply(lambda x: f"{x[:2]}_{x[2:5]}_{x[5:9]}")
        .tolist()
    )


# ------------------------- mode -> products -------------------------
def products_for_mode(mode: str) -> Tuple[str, ...]:
    mode = mode.upper()
    if mode == "RGB":
        return ("dop20rgb",)
    if mode == "RGBI":
        return ("dop20rgbi",)
    if mode == "RGBIH":
        # ndom1 is preferred; dom1+dgm1 required as fallback and for height derivation
        return ("dop20rgbi", "ndom1", "dom1", "dgm1")
    raise ValueError(f"Unsupported mode: {mode}")


def output_name_for_mode(mode: str, tile_id: str) -> str:
    mode = mode.upper()
    if mode == "RGB":
        return f"rgb_{tile_id}"
    if mode == "RGBI":
        return f"rgbi_{tile_id}"
    if mode == "RGBIH":
        return f"rgbih_{tile_id}"
    raise ValueError(f"Unsupported mode: {mode}")


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

    # Try both dop20rgb and dop20rgbi patterns (depending on mode)
    dop_path = None
    if "dop20rgbi" in products:
        dop_path = _find_single_file(
            unzipped_dir / "dop20rgbi",
            [f"dop20rgbi_{stub}.tif", "dop20rgbi_*.tif", "*.tif"],
        )
    if dop_path is None and "dop20rgb" in products:
        dop_path = _find_single_file(
            unzipped_dir / "dop20rgb",
            [f"dop20rgb_{stub}.tif", "dop20rgb_*.tif", "*.tif"],
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


# ------------------------- main per-tile -------------------------
def build_tile_for_mode(
    tile_id: str,
    tmp_root: Path,
    output_tiles: Path,
    mode: str,
    products: Tuple[str, ...],
    keep_tmp: bool = False,
    overwrite: bool = False,
):
    mode = mode.upper()

    merged_dir = output_tiles / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    base = output_name_for_mode(mode, tile_id)
    out_tif = merged_dir / f"{base}.tif"
    out_json = merged_dir / f"{base}.json"

    # Skip logic:
    # - For RGBIH, require both tif+json to skip.
    # - For RGB/RGBI, require tif only to skip.
    if mode == "RGBIH":
        if out_tif.exists() and out_json.exists() and (not overwrite):
            logging.info("✅ Exists (tif+json), skipping: %s", tile_id)
            return
    else:
        if out_tif.exists() and (not overwrite):
            logging.info("✅ Exists, skipping: %s", tile_id)
            return

    dop_path, ndom_path, dom_path, dgm_path, tmp_tile_dir = download_tile_to_temp(
        tile_id=tile_id,
        tmp_root=tmp_root,
        products=products,
    )

    # Validate minimum requirements by mode
    if dop_path is None:
        logging.error("❌ Missing DOP for tile %s (mode=%s)", tile_id, mode)
        if not keep_tmp and tmp_tile_dir.exists():
            shutil.rmtree(tmp_tile_dir, ignore_errors=True)
        return

    if mode == "RGBIH":
        # Need dom+dgm for fallback generation inside TileDataset if ndom missing
        if dom_path is None or dgm_path is None:
            logging.error(
                "❌ Missing dom/dgm for tile %s (mode=%s). dom=%s dgm=%s",
                tile_id, mode, dom_path, dgm_path
            )
            if not keep_tmp and tmp_tile_dir.exists():
                shutil.rmtree(tmp_tile_dir, ignore_errors=True)
            return
    else:
        # RGB / RGBI don't need dom/dgm/ndom
        ndom_path = None
        dom_path = None
        dgm_path = None

    # Build tile
    tile = TileDataset(
        tile_id=tile_id,
        dop_path=str(dop_path),
        ndom_path=str(ndom_path) if ndom_path is not None else None,
        dgm_path=str(dgm_path) if dgm_path is not None else None,
        dom_path=str(dom_path) if dom_path is not None else None,
        output_dir=str(output_tiles),
        mode=mode,
        temp_dir=str(tmp_root),
    )
    produced_path = tile.process()
    if produced_path is None:
        logging.error("❌ TileDataset returned None for tile %s (mode=%s)", tile_id, mode)
        if not keep_tmp and tmp_tile_dir.exists():
            shutil.rmtree(tmp_tile_dir, ignore_errors=True)
        return

    # For RGBIH: write JSON using TileDataset-provided normalization stats
    if mode == "RGBIH":
        stats = getattr(tile, "last_height_stats_m", None)
        source = getattr(tile, "last_height_source", None)

        if not stats or len(stats) != 2:
            raise RuntimeError(
                f"TileDataset did not expose last_height_stats_m for {tile_id}. "
                f"Implement the small changes in TileDataset to return (vmin,vmax) used for normalization."
            )

        obj = {
            "tile_id": tile_id,
            "mode": mode,
            "height_channel": {
                "band_index_1based": 5,
                "stats_m": [float(stats[0]), float(stats[1])],
                "note": NOTE,
                "source": source or ("ndom1" if ndom_path else "dom1-dgm1"),
            },
        }
        atomic_write_json(obj, out_json)
        logging.info("🧾 Wrote JSON: %s", out_json)

    # Move/rename output tif if TileDataset produced a different filename
    # (Your TileDataset writes: merged/<mode.lower()>_<tile_id>.tif)
    expected = out_tif
    produced = Path(produced_path)
    if produced.resolve() != expected.resolve():
        expected.parent.mkdir(parents=True, exist_ok=True)
        if expected.exists() and overwrite:
            expected.unlink()
        shutil.move(str(produced), str(expected))
        logging.info("📦 Renamed output: %s -> %s", produced.name, expected.name)

    if not keep_tmp and tmp_tile_dir.exists():
        shutil.rmtree(tmp_tile_dir, ignore_errors=True)


# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build merged tiles + (RGBIH) JSON metadata")

    p.add_argument("--tiles-gpkg", required=True, help="GeoPackage containing 'dop_kachel'")
    p.add_argument("--tmp-root", required=True, help="Temp folder for downloads/unzips")
    p.add_argument("--output-tiles", required=True, help="Output folder; writes to OUTPUT/merged/")
    p.add_argument("--mode", default="RGBIH", help="TileDataset mode - RGB, RGBI, RGBIH")
    p.add_argument("--keep-tmp", action="store_true", help="Keep per-tile temp directories")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

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
            build_tile_for_mode(
                tile_id=tile_id,
                tmp_root=tmp_root,
                output_tiles=output_tiles,
                mode=mode,
                products=products,
                keep_tmp=args.keep_tmp,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logging.exception("❌ Failed tile %s: %s", tile_id, e)


if __name__ == "__main__":
    main()
