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

class LGLDownloader:
    def __init__(self,
                 base_folder="cache/lgl_store",
                 delay_seconds=10,
                 parallel=False,
                 max_workers=4):
        self.base_folder = base_folder
        self.delay_seconds = delay_seconds
        self.parallel = parallel
        self.max_workers = max_workers
        os.makedirs(self.base_folder, exist_ok=True)
        self._init_logger()

    def _init_logger(self):
        log_file = os.path.join(self.base_folder, "lgl_downloader.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        logging.info("🟢 Logger initialized. Saving logs to: %s", log_file)

    def _construct_url(self, product, x, y):
        filename = f"{product}_32_{x}_{y}_2_bw.zip"
        if product == "dgm1":
            url = f"https://opengeodata.lgl-bw.de/data/dgm/{filename}"
        elif product in ("dop20rgbi", "dop20rgb"):
            url = f"https://opengeodata.lgl-bw.de/data/dop20/{filename}"
        elif product == "ndom1":
            url = f"https://opengeodata.lgl-bw.de/data/ndom1/{filename}"
        else:
            url = f"https://opengeodata.lgl-bw.de/data/dom1/{filename}"
        return url, filename

    def _download_single(self, product, x, y):
        def attempt_download(tile_x, tile_y):
            folder = os.path.join(self.base_folder, product)
            os.makedirs(folder, exist_ok=True)

            url, filename = self._construct_url(product, tile_x, tile_y)
            local_path = os.path.join(folder, filename)

            if os.path.exists(local_path):
                logging.info("✅ Skipped (already exists): %s", filename)
                return {"status": "skipped", "file": filename}

            logging.info("⬇️ Downloading: %s", filename)
            try:
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                    logging.info("✅ Saved: %s", local_path)
                    return {"status": "success", "file": filename}
                else:
                    logging.info("❌ Failed (HTTP %d): %s", response.status_code, filename)
                    return {"status": "failed", "file": filename, "code": response.status_code}
            except requests.exceptions.RequestException as e:
                logging.error("⚠️ Network error: %s (%s)", filename, str(e))
                return {"status": "error", "file": filename, "error": str(e)}
            except Exception as e:
                logging.error("⚠️ Unexpected error: %s (%s)", filename, str(e))
                return {"status": "error", "file": filename, "error": str(e)}

        x_int = int(x)
        y_int = int(y)

        # Fallback coordinates: original, up, left, up-left
        fallback_tiles = [
            (x_int, y_int),
            (x_int, y_int - 1),
            (x_int - 1, y_int),
            (x_int - 1, y_int - 1),
        ]

        for i, (xi, yi) in enumerate(fallback_tiles):
            label = "original" if i == 0 else f"fallback {i}"
            logging.info("🔍 Trying %s tile: x=%d, y=%d", label, xi, yi)
            result = attempt_download(str(xi), str(yi))

            if result["status"] in ("success", "skipped"):
                if i > 0:
                    result["fallback_attempted"] = True
                    result["fallback_index"] = i
                return result

        logging.error("🚫 All attempts failed for tile x=%s, y=%s", x, y)
        return {"status": "failed", "file": None, "error": "All fallback attempts failed"}

    def download_tiles(self, tile_ids, products):
        tile_ids = list(tile_ids)
        total = len(tile_ids)
        logging.info(
            "🧠 Starting download of %d tiles (%d products each, mode: %s)",
            total,
            len(products),
            "parallel" if self.parallel else "sequential",
        )

        for idx, tile_id in enumerate(tile_ids, 1):
            try:
                x, y = tile_id.split("-")
            except ValueError:
                logging.warning("❌ Invalid tile format: '%s' — expected 'xxx-yyyy'", tile_id)
                continue

            logging.info("🚀 [%d/%d] Processing tile: %s", idx, total, tile_id)

            if self.parallel:
                with ThreadPoolExecutor(max_workers=min(len(products), self.max_workers)) as executor:
                    futures = {executor.submit(self._download_single, product, x, y): product for product in products}
                    for future in as_completed(futures):
                        result = future.result()
                        logging.info("📦 Finished: %s (%s)", result.get("file"), result.get("status"))
            else:
                for product in products:
                    result = self._download_single(product, x, y)
                    logging.info("📦 Finished: %s (%s)", result.get("file"), result.get("status"))

            logging.info("⏳ Waiting %d seconds before next tile...\n", self.delay_seconds)
            time.sleep(self.delay_seconds)


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
