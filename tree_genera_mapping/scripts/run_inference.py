#!/usr/bin/env python3
"""
Run inference on LGL tiles using a YOLO model trained on RGBIH data.

"""
import argparse
import logging
from pathlib import Path
import os
import rasterio
import torch
import geopandas as gpd
from shapely.geometry import box
from ultralytics import YOLO
import numpy as np
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
    logger = logging.getLogger(__name__)
    return logger
logger = setup_logging()
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

def run_inference_on_tile(model, tile_path, out_dir, patch_size=640, stride=512, conf=0.3, iou=0.4):
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(tile_path) as src:
        transform = src.transform
        crs = src.crs
        img = src.read()  # (C, H, W)
        H, W = img.shape[1:]

    records = []
    # Ensure full coverage
    ys = list(range(0, H - patch_size, stride)) + [H - patch_size]
    xs = list(range(0, W - patch_size, stride)) + [W - patch_size]

    for y in ys:
        for x in xs:
            window = img[:, y:y + patch_size, x:x + patch_size]

            # to tensor: BCHW
            img_tensor = torch.from_numpy(window).unsqueeze(0).float() / 255.0

            # YOLO inference
            res = model.predict(img_tensor, imgsz=1024, conf=conf, iou=iou, verbose=False)

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
                x1_map, y1_map = rasterio.transform.xy(transform, y1_global, x1_global, offset="ul")
                x2_map, y2_map = rasterio.transform.xy(transform, y2_global, x2_global, offset="lr")

                geom = box(x1_map, y2_map, x2_map, y1_map)
                # === Extra info ===
                # height: max of 5th band inside bbox
                #TODO if there is no height json file, set to normalised values between 0-255
                height_patch = img[4, y1i:y2i, x1i:x2i]
                max_height = float(height_patch.max()) if height_patch.size > 0 else np.nan
                min_height = float(height_patch.min()) if height_patch.size > 0 else np.nan

                json_path = Path(tile_path).with_suffix('.height.json')

                if json_path.exists():
                    with open(json_path) as f:
                        height_meta = json.load(f)
                    raw_min = height_meta['raw_height_stats']['min']
                    raw_max = height_meta['raw_height_stats']['max']
                    if not np.isnan(max_height):
                        max_height = raw_min + (raw_max - raw_min) * (max_height / 255.0)
                        min_height = raw_min + (raw_max - raw_min) * (min_height / 255.0)
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
                    "canopy_height_max": max_height,
                    "canopy_height_min": min_height,
                    "canopy_diameter": diameter,
                    "centroid_x": cx_map,
                    "centroid_y": cy_map,
                    "geometry": geom
                })

    # save gpkg per tile
    if records:
        gdf = gpd.GeoDataFrame(records, crs=crs)
        # gdf = merge_subtile_predictions(gdf, iou_thresh=iou)
        out_file = Path(out_dir) / (Path(tile_path).stem + ".gpkg")
        gdf.to_file(out_file, driver="GPKG")
        logger.info(f"✅ Saved {len(records)} detections to {out_file}")
    else:
        logger.warning(f"⚠️ No detections for {tile_path}")
def run(ckpt_path: str,
        tile_id:str,
        tile_dir:str,
        output_dir:str,
        conf:float = 0.25,
        iou:float = 0.4,
        patch_size:int = 640,
        stride:int = 512,
        ):
    # ------ CONFIG
    config = {
        'ckpt_path': ckpt_path,
        'imgs_path': tile_dir
        }
    # Load model
    model = YOLO(config['ckpt_path'])
    model.model.ch = 5  # set number of channels to 5 (RGBIH)

    # Run inference on the specified tile
    if tile_id is None:
        img_files = list(Path(config['imgs_path']).glob("*.tif"))
        for img_file in tqdm(img_files, total=len(img_files), desc="Processing tiles"):
            run_inference_on_tile(model=model,
                                  tile_path=img_file,
                                  out_dir=output_dir,
                                  patch_size=640,
                                  stride=512,
                                  conf=0.25,
                                  iou=0.4
                              )
    else:
        img_file = Path(config['imgs_path']) / f"rgbih_{tile_id}.tif"
        if img_file.exists():
            run_inference_on_tile(model=model,
                                    tile_path=img_file,
                                    out_dir=output_dir,
                                    patch_size=640,
                                    stride=512,
                                    conf=0.25,
                                    iou=0.4
                                )





    return None

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess a single LGL tile into RGBIH GeoTIFF")
    parser.add_argument("--tile_id", default='457-5428', type=str, required=False,
                        help="Tile ID (format: xxx_yyyy, e.g. 457-5428)")
    parser.add_argument("--output_dir", default='cache/predictions', type=Path, required=False,
                        help="Folder to save final RGBIH tile")
    parser.add_argument("--tile_dir", default='cache/merged', type=str, required=False, help="Tile Directory")
    parser.add_argument("--ckpt_path", default='models/yolov8_rgbih_best.pt', type=str, required=False,)
    parser.add_argument("--patch_size", default=640, type=int, required=False, help="Patch size for inference")
    parser.add_argument("--stride", default=512, type=int, required=False, help="Stride for inference")
    parser.add_argument("--conf", default=0.25, type=float, required=False, help="Confidence threshold for inference")
    parser.add_argument("--iou", default=0.4, type=float, required=False, help="IOU threshold for inference")


    args = parser.parse_args()

    run(args.tile_id, args.tile_dir, args.output_dir, args.ckpt_path, args.conf, args.iou, args.patch_size, args.stride)


if __name__ == "__main__":
   main()