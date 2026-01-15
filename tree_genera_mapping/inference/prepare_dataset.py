from pathlib import Path
import logging
import tempfile
import zipfile
import shutil
import argparse

from tree_genera_mapping.data_ops.lgl_store import LGLDownloader
from tree_genera_mapping.preprocess.preprocess_tiles import process_tile_row
import re
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
def discover_tiles(product_dir: Path) -> set[str]:
    """
    Scan an extracted product folder and collect tile_ids (like 32_457_5428).
    Works for dop20rgbi, dgm1, dom1.
    """
    if not product_dir.exists():
        return set()

    tile_ids = set()
    pattern = re.compile(r"(\d{2}_\d{3}_\d{4})")  # e.g. 32_457_5428

    for f in product_dir.rglob("*"):
        match = pattern.search(f.name)
        if match:
            tile_ids.add(match.group(1))
    return tile_ids
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
    logger = logging.getLogger(__name__)
    return logger
def try_download_and_unzip(downloader, product, x, y, temp_dir, logger):
    """
    Try downloading product for a tile (x, y).
    Falls back to odd-parent (2x2) zip anchor and unzips if success.
    """
    product_dir = temp_dir / product
    product_dir.mkdir(parents=True, exist_ok=True)

    def unzip_file(zip_path, target_dir):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
            logger.info(f"📂 Unzipped {zip_path} → {target_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to unzip {zip_path}: {e}")

    # normalize to odd anchor for 2x2
    parent_x = x if int(x) % 2 == 1 else str(int(x) - 1)
    parent_y = y if int(y) % 2 == 1 else str(int(y) - 1)

    # fallback candidates (original and odd anchor)
    candidates = [(x, y), (parent_x, parent_y)]

    for xi, yi in candidates:
        result = downloader._download_single(product, xi, yi)
        if result["status"] in ("success", "skipped"):
            zip_path = temp_dir / product / result["file"]
            unzip_file(zip_path, product_dir)
            return zip_path  # success
        else:
            logger.warning(f"⚠️ Failed {product} for tile {xi}-{yi}")

    return None


def resolve_tile_tif(product_dir: Path, tile_id: str) -> Path | None:
    """
    Given the extracted folder of a product and a tile_id (e.g. '32_457_5429'),
    return the path to the matching .tif file.
    """
    if not product_dir or not product_dir.exists():
        return None
    tile_id = tile_id.replace("-", "_")
    # find files like dom1_32_457_5429_1_bw_2022.tif
    matches = [f for f in product_dir.rglob(f"*{tile_id}*") if f.suffix.lower() in [".tif", ".xyz"]]
    if matches:
        return matches[0]
    return None
def run(tile_id: str, final_output_dir: Path, keep_temp: bool = False):
    logger = setup_logging()
    # Check if the tile is already exists
    if Path(final_output_dir/f"merged/rgbih_32_{tile_id[:3]}_{tile_id[-4:]}.tif").exists():
        logger.info(f"✅ Tile already exists, skipping: {final_output_dir/f'merged/rgbih_32_{tile_id[:2]}_{tile_id[-4:]}.tif'}")
        return
    # Create temporary folder for raw downloads
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{tile_id}_"))
    logger.info(f"📂 Temporary folder created: {temp_dir}")


    try:
        # Split tile_id into x/y (format: 457-5428)
        try:
            x, y = tile_id.split("-")
        except ValueError:
            logger.error(f"❌ Invalid tile_id format: {tile_id} (expected 'xxx-yyyy')")
            return

        downloader = LGLDownloader(base_folder=temp_dir, parallel=False)

        # Step 1: Download all products (dop20rgbi, dgm1, dom1)
        products = ["dop20rgbi", "dgm1", "dom1"]
        product_paths = {}
        for product in products:
            zip_path = try_download_and_unzip(downloader, product, x, y, temp_dir, logger)
            if zip_path:
                product_paths[product] = zip_path.parent  # folder with extracted files
            else:
                logger.warning(f"⚠️ Could not retrieve {product} for {tile_id}")

        # Step 2: Process ALL discovered tiles
        all_tile_ids = set()
        for prod, pdir in product_paths.items():
            all_tile_ids |= discover_tiles(pdir)  # union

        for tid in sorted(all_tile_ids):
            row = {
                "tile_id": tid,
                "dop20_path": resolve_tile_tif(product_paths.get("dop20rgbi"), tid),
                "dgm1_path": resolve_tile_tif(product_paths.get("dgm1"), tid),
                "dom1_path": resolve_tile_tif(product_paths.get("dom1"), tid),
            }
            if row["dop20_path"]:
                out_path = process_tile_row(row, output_dir=final_output_dir, mode="RGBIH")
                logger.info(f"✅ Saved final tile: {out_path}")
            else:
                logger.warning(f"⚠️ Skipping {tid}, no dop20rgbi available")
        # if "dop20rgbi" in product_paths:
        #     row = {
        #         "tile_id": f"32_{x}_{y}",
        #         "dop20_path": resolve_tile_tif(product_paths.get("dop20rgbi"), tile_id),
        #         "dgm1_path": resolve_tile_tif(product_paths.get("dgm1"), tile_id),
        #         "dom1_path": resolve_tile_tif(product_paths.get("dom1"), tile_id),
        #     }
        #     out_path = process_tile_row(row, output_dir=final_output_dir, mode="RGBIH")
        #     logger.info(f"✅ Saved final tile: {out_path}")
        # else:
        #     logger.error(f"🚫 Skipping {tile_id}, no dop20rgbi available")

    finally:
        if keep_temp:
            logger.info(f"⏸ Keeping temp folder: {temp_dir}")
        else:
            logger.info(f"🧹 Cleaning up temp folder: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            if Path(final_output_dir/"temp").exists():
                shutil.rmtree(Path(final_output_dir/"temp"), ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess a single LGL tile into RGBIH GeoTIFF")
    parser.add_argument("--tile_id", default='458-5429', type=str, required=False, help="Tile ID (format: xxx_yyyy, e.g. 457-5428)")
    parser.add_argument("--output_dir",default='cache', type=Path, required=False, help="Folder to save final RGBIH tile")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary download folder")

    args = parser.parse_args()

    run(args.tile_id, args.output_dir, args.keep_temp)