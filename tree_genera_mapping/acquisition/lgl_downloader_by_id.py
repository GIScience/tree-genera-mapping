import os
import time
import logging
import geopandas as gpd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import re
from typing import Iterable, List, Optional

_TILE_RE = re.compile(r"^\d{3}-\d{4}$")  # e.g., 457-5428
from tree_genera_mapping.acquisition.lgl_store import LGLDownloader
def _parse_requested_tiles(tile_id: Optional[str], tile_ids: Optional[str]) -> List[str]:
    """
    tile_id: single "457-5428"
    tile_ids: comma-separated "457-5428,458-5428"
    """
    requested: List[str] = []

    if tile_id:
        requested.append(tile_id.strip())

    if tile_ids:
        requested.extend([t.strip() for t in tile_ids.split(",") if t.strip()])

    # de-dup while preserving order
    seen = set()
    out = []
    for t in requested:
        if t not in seen:
            out.append(t)
            seen.add(t)

    # validate format
    bad = [t for t in out if not _TILE_RE.match(t)]
    if bad:
        raise ValueError(f"Invalid tile id(s): {bad}. Expected format like '457-5428'.")

    return out


def _extract_tile_ids_from_gpkg(gdf: gpd.GeoDataFrame, col: str = "dop_kachel") -> List[str]:
    if col not in gdf.columns:
        raise ValueError(f"Input file must contain column '{col}'")

    # your existing logic
    raw = gdf[col].astype(str).tolist()
    tile_ids = [f"{tile_id[2:5]}-{tile_id[-4:]}" for tile_id in raw]
    # de-dup while preserving order
    seen = set()
    uniq = []
    for t in tile_ids:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def run_lgl_downloader(
    tiles_gpkg: str,
    output_dir: str = "cache/lgl_store",
    selected_products: Iterable[str] = ("dop20rgbi", "ndom1"),
    parallel: bool = False,
    delay_seconds: int = 2,
    max_workers: int = 4,
    tile_id: Optional[str] = None,
    tile_ids: Optional[str] = None,
):
    # --- validate input ---
    if not os.path.exists(tiles_gpkg):
        raise FileNotFoundError(f"❌ Input file not found: {tiles_gpkg}")

    gdf = gpd.read_file(tiles_gpkg)

    all_tiles = _extract_tile_ids_from_gpkg(gdf, col="dop_kachel")
    requested = _parse_requested_tiles(tile_id=tile_id, tile_ids=tile_ids)

    # if user requested tiles, filter to those; else keep all
    if requested:
        missing = sorted(set(requested) - set(all_tiles))
        if missing:
            raise ValueError(
                f"Requested tile(s) not found in {tiles_gpkg}: {missing}. "
                f"Make sure the tiles file includes them."
            )
        tile_ids_final = [t for t in all_tiles if t in set(requested)]
    else:
        tile_ids_final = all_tiles

    if not tile_ids_final:
        logging.warning("⚠️ No tiles to download. Exiting.")
        return

    downloader = LGLDownloader(
        base_folder=output_dir,
        delay_seconds=delay_seconds,
        parallel=parallel,
        max_workers=max_workers,
    )
    downloader.download_tiles(tile_ids_final, list(selected_products))
    logging.info("✅ All downloads completed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-gpkg", default='dop_kachel' ,required=True, help="GeoPackage with a 'dop_kachel' column")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--delay-seconds", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=6)

    # New, reproducible selection options
    ap.add_argument("--tile-id", default=None, help="Single tile id, e.g. 457-5428")
    ap.add_argument("--tile-ids", default=None, help="Comma-separated tile ids, e.g. 457-5428,458-5428")

    # Optional: product selection
    ap.add_argument("--products", default="dop20rgbi,ndom1", help="Comma-separated products: dop20rgb, dop20rgbi, dom1, dgm1, ndom1")

    args = ap.parse_args()

    products = [p.strip() for p in args.products.split(",") if p.strip()]

    run_lgl_downloader(
        tiles_gpkg=args.tiles_gpkg,
        output_dir=args.output_dir,
        selected_products=products,
        parallel=args.parallel,
        delay_seconds=args.delay_seconds,
        max_workers=args.max_workers,
        tile_id=args.tile_id,
        tile_ids=args.tile_ids,
    )
