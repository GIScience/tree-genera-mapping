import os
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logging.info("🟢 Logger initialized. Saving logs to: %s", log_file)

    def _construct_url(self, product, x, y):
        filename = f"{product}_32_{x}_{y}_2_bw.zip"
        if product == "dgm1":
            url = f"https://opengeodata.lgl-bw.de/data/dgm/{filename}"
        elif product == "dop20rgbi":
            url = f"https://opengeodata.lgl-bw.de/data/dop20/{filename}"
        elif product == "dop20rgb":
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

        # Convert string inputs to integers for arithmetic
        x_int = int(x)
        y_int = int(y)

        # Fallback coordinates: original, up, left, up-left
        fallback_tiles = [
            (x_int, y_int),  # original tile
            (x_int, y_int - 1),  # shift y - 1
            (x_int - 1, y_int),  # shift x - 1
            (x_int - 1, y_int - 1)  # shift both
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

        # All attempts failed
        logging.error("🚫 All attempts failed for tile x=%s, y=%s", x, y)
        return {"status": "failed", "file": None, "error": "All fallback attempts failed"}

    def download_tiles(self, tile_ids, products):
        total = len(tile_ids)
        logging.info("🧠 Starting download of %d tiles (%d products each, mode: %s)",
                     total, len(products),
                     "parallel" if self.parallel else "sequential")

        for idx, tile_id in enumerate(tile_ids, 1):
            try:
                x, y = tile_id.split("-")
            except ValueError:
                logging.warning("❌ Invalid tile format: '%s' — expected format 'xxx-yyyy'", tile_id)
                continue

            logging.info("🚀 [%d/%d] Processing tile: %s", idx, total, tile_id)

            if self.parallel:
                # Parallel download of all products for this tile
                with ThreadPoolExecutor(max_workers=min(len(products), self.max_workers)) as executor:
                    futures = {executor.submit(self._download_single, product, x, y): product for product in products}
                    for future in as_completed(futures):
                        result = future.result()
                        logging.info("📦 Finished: %s (%s)", result.get("file"), result.get("status"))
            else:
                # Sequential download of products for this tile
                for product in products:
                    result = self._download_single(product, x, y)
                    logging.info("📦 Finished: %s (%s)", result.get("file"), result.get("status"))

            logging.info("⏳ Waiting %d seconds before next tile...\n", self.delay_seconds)
            time.sleep(self.delay_seconds)

