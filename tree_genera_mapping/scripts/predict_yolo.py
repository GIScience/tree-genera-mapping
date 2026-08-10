#!/usr/bin/env python3
"""
Run inference on LGL tiles using a YOLO model trained on RGBIH data.
"""
import argparse
import json
import logging
from pathlib import Path

import rasterio
import geopandas as gpd
from shapely.geometry import box
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
    logger = logging.getLogger(__name__)
    return logger
logger = setup_logging()


GLOBAL_HEIGHT_BOUNDS_M = (0.0, 80.0)


def window_starts(length: int, patch_size: int, stride: int) -> list[int]:
    """Return deterministic sliding-window starts with complete edge coverage."""
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def load_height_bounds(tile_path: Path) -> tuple[float, float]:
    """Load height normalization bounds written by ``fetch_tiles.py``.

    The current sidecar is ``<tile>.json`` with ``height_channel.stats_m``. The
    older ``<tile>.height.json`` / ``raw_height_stats`` shape remains supported.
    Released sample chips use the documented global 0--80 m normalization and
    intentionally have no sidecars.
    """
    candidates = [tile_path.with_suffix(".json"), tile_path.with_suffix(".height.json")]
    for json_path in candidates:
        if not json_path.exists():
            continue
        metadata = json.loads(json_path.read_text())
        if "height_channel" in metadata and "stats_m" in metadata["height_channel"]:
            bounds = metadata["height_channel"]["stats_m"]
            if len(bounds) == 2:
                return float(bounds[0]), float(bounds[1])
        if "raw_height_stats" in metadata:
            stats = metadata["raw_height_stats"]
            if "min" in stats and "max" in stats:
                return float(stats["min"]), float(stats["max"])
        raise ValueError(f"Unsupported height metadata format: {json_path}")

    logger.info(
        "No height sidecar for %s; using global %.1f--%.1f m bounds",
        tile_path.name,
        *GLOBAL_HEIGHT_BOUNDS_M,
    )
    return GLOBAL_HEIGHT_BOUNDS_M


def merge_subtile_predictions(gdf, iou_thresh=0.5):
    """
    Merge overlapping detections (from overlapping subtiles).
    Keeps the highest-confidence detection when geometries overlap significantly.
    """
    if gdf.empty:
        return gdf

    # Sort by confidence (highest first)
    gdf = gdf.sort_values("confidence", ascending=False).reset_index(drop=True)

    keep = []
    seen = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if any(geom.intersects(gdf.geometry[j]) and
               (geom.intersection(gdf.geometry[j]).area / geom.union(gdf.geometry[j]).area > iou_thresh)
               for j in seen):
            # skip overlapping duplicate
            continue
        keep.append(row)
        seen.append(idx)

    merged_gdf = gpd.GeoDataFrame(keep, crs=gdf.crs)
    return merged_gdf

def run_inference_on_tile(
    model,
    tile_path,
    out_dir,
    patch_size=640,
    stride=512,
    conf=0.3,
    iou=0.4,
    imgsz=1024,
):
    tile_path = Path(tile_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tile_path) as src:
        transform = src.transform
        crs = src.crs
        img = src.read()  # (C, H, W)
        H, W = img.shape[1:]

    if img.shape[0] != 5:
        raise ValueError(f"{tile_path} has {img.shape[0]} channels; expected 5")
    if crs is None:
        raise ValueError(f"{tile_path} has no CRS; georeferenced output cannot be created")

    height_min_m, height_max_m = load_height_bounds(tile_path)

    records = []
    ys = window_starts(H, patch_size, stride)
    xs = window_starts(W, patch_size, stride)

    for y in ys:
        for x in xs:
            window = img[:, y:y + patch_size, x:x + patch_size]

            # Ultralytics applies its configured ``imgsz`` preprocessing to HWC numpy
            # inputs. Passing a BCHW tensor here bypasses that resize and changes the
            # predictions whenever patch_size != imgsz.
            image_hwc = np.moveaxis(window, 0, -1)
            res = model.predict(image_hwc, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

            boxes = res[0].boxes.xyxy.cpu().numpy()
            confs = res[0].boxes.conf.cpu().numpy()
            clss = res[0].boxes.cls.cpu().numpy()

            for b, c, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = b

                # shift coords back to tile pixel space
                x1_global = x + x1
                x2_global = x + x2
                y1_global = y + y1
                y2_global = y + y2
                # round and clip to valid indices
                x1i = max(0, int(np.floor(x1_global)))
                x2i = min(W, int(np.ceil(x2_global)))
                y1i = max(0, int(np.floor(y1_global)))
                y2i = min(H, int(np.ceil(y2_global)))

                # pixel → map coords
                x1_map, y1_map = transform * (float(x1_global), float(y1_global))
                x2_map, y2_map = transform * (float(x2_global), float(y2_global))

                geom = box(x1_map, y2_map, x2_map, y1_map)
                # === Extra info ===
                # height: max of 5th band inside bbox
                height_patch = img[4, y1i:y2i, x1i:x2i]
                if height_patch.size > 0:
                    max_height_raw = float(height_patch.max())
                    min_height_raw = float(height_patch.min())
                    max_height = height_min_m + (height_max_m - height_min_m) * (max_height_raw / 255.0)
                    min_height = height_min_m + (height_max_m - height_min_m) * (min_height_raw / 255.0)
                else:
                    max_height = np.nan
                    min_height = np.nan

                # bbox diameter (mean of width, height in meters)
                dx = abs(x2_map - x1_map)
                dy = abs(y2_map - y1_map)
                diameter = float(np.sqrt(dx ** 2 + dy ** 2))  # diagonal as "diameter"

                # centroid (map coords)
                cx_map, cy_map = (x1_map + x2_map) / 2, (y1_map + y2_map) / 2

                # append record of detection
                records.append({
                    "class_id": int(cls_id),
                    "class_name": model.names[int(cls_id)],
                    "confidence": float(c),
                    "class_id_det": int(cls_id),
                    "class_name_det": model.names[int(cls_id)],
                    "confidence_det": float(c),
                    "canopy_height_max": max_height,
                    "canopy_height_min": min_height,
                    "canopy_height_m": max_height,
                    "canopy_diameter": diameter,
                    "centroid_x": cx_map,
                    "centroid_y": cy_map,
                    "geometry": geom
                })

    # save gpkg per tile
    if records:
        gdf = gpd.GeoDataFrame(records, crs=crs)
        # gdf = merge_subtile_predictions(gdf, iou_thresh=iou)
        tile_name = tile_path.stem.removeprefix("rgbih_")
        out_file = out_dir / f"tile_{tile_name}.gpkg"
        gdf.to_file(out_file, driver="GPKG")
        logger.info(f"✅ Saved {len(records)} detections to {out_file}")
    else:
        logger.warning(f"⚠️ No detections for {tile_path}")
def run(ckpt_path: str,
        tile_id: str | None,
        tile_dir: str,
        output_dir: str,
        conf: float = 0.25,
        iou: float = 0.4,
        patch_size: int = 640,
        stride: int = 512,
        imgsz: int = 1024,
        ):
    ckpt_path = Path(ckpt_path)
    tile_dir = Path(tile_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not tile_dir.is_dir():
        raise NotADirectoryError(f"Tile directory not found: {tile_dir}")

    model = YOLO(str(ckpt_path))
    model.model.ch = 5  # set number of channels to 5 (RGBIH)

    if tile_id is None:
        img_files = sorted(tile_dir.glob("*.tif"))
    else:
        value = str(tile_id).strip()
        normalized = value.replace("-", "_")
        parts = normalized.split("_")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            normalized = f"32_{normalized}"
        candidates = [
            tile_dir / value,
            tile_dir / f"{value}.tif",
            tile_dir / f"rgbih_{normalized}.tif",
        ]
        img_files = [candidate for candidate in candidates if candidate.is_file()]
        img_files = img_files[:1]

    if not img_files:
        target = f"tile {tile_id!r}" if tile_id is not None else "TIFF files"
        raise FileNotFoundError(f"No {target} found in {tile_dir}")

    for img_file in tqdm(img_files, total=len(img_files), desc="Processing tiles"):
        run_inference_on_tile(
            model=model,
            tile_path=img_file,
            out_dir=output_dir,
            patch_size=patch_size,
            stride=stride,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
        )

    return None

def main():
    parser = argparse.ArgumentParser(description="Run georeferenced YOLO inference on RGBIH tiles")
    parser.add_argument("--tile-id", "--tile_id", dest="tile_id", default=None,
                        help="Optional tile ID or TIFF filename; omit to process all TIFFs")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir",
                        default="cache/predictions", type=Path,
                        help="Folder for per-tile prediction GeoPackages")
    parser.add_argument("--tile-dir", "--tile_dir", dest="tile_dir",
                        default="cache/img_dir", type=Path, help="Directory containing RGBIH TIFFs")
    parser.add_argument("--ckpt-path", "--ckpt_path", dest="ckpt_path",
                        default="cache/weights/yolo11l_tree_genus.pt", type=Path)
    parser.add_argument("--patch-size", "--patch_size", dest="patch_size",
                        default=640, type=int, help="Sliding-window size for inference")
    parser.add_argument("--stride", default=512, type=int, required=False, help="Stride for inference")
    parser.add_argument("--imgsz", default=1024, type=int,
                        help="YOLO inference image size (default: checkpoint training size)")
    parser.add_argument("--conf", default=0.25, type=float, required=False, help="Confidence threshold for inference")
    parser.add_argument("--iou", default=0.4, type=float, required=False, help="IOU threshold for inference")


    args = parser.parse_args()

    run(
        ckpt_path=args.ckpt_path,
        tile_id=args.tile_id,
        tile_dir=args.tile_dir,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        patch_size=args.patch_size,
        stride=args.stride,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
   main()
