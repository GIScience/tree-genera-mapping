#!/usr/bin/env python3
"""
This module defines a teacher ensemble of pre-trained models to perform inference
Input:
- tile_dir: Directory containing input tiles
- output_dir: Directory to save predictions
- ckpt_paths: List of checkpoint paths for the teacher models
- conf: Confidence threshold for predictions
- iou: IOU threshold for predictions
- patch_size: Size of the patches for inference
- stride: Stride for moving the patch window
"""

import argparse
import logging
from pathlib import Path
import os
import rasterio
import torch
import geopandas as gpd
from shapely.geometry import box

#TODO:
# 1. read models;
# 2. read tile image;
# 3. tile image for first model;
# 4. run first model;
# 5. aggregate results;
# 6. remove duplicates;
# 7. read image patch for predictions;
# 8. run second model on the image patch;
# 9. assign predictions to the aggregated results;
# 10. save results to output directory


# ------------ setup logging ------------
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
    logger = logging.getLogger(__name__)
    return logger
# ------------ Arguments ------------
def parse_args():
    ap = argparse.ArgumentParser(description="Run weak inference using a teacher ensemble of models")
    ap.add_argument("--tile-dir", required=True, help="Directory containing input tiles")
    ap.add_argument("--output-dir", required=True, help="Directory to save predictions")
    ap.add_argument("--ckpt-paths", required=True, nargs='+', help="List of checkpoint paths for the teacher models")
    ap.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for predictions")
    ap.add_argument("--iou", type=float, default=0.4, help="IOU threshold for predictions")
    ap.add_argument("--patch-size", type=int, default=640, help="Size of the subtiles for detection")
    ap.add_argument("--stride", type=int, default=512, help="Stride for moving the subtile window")
    ap.add_argument("--image-patch-size", type=int, default=512, help="Size of the image patch for classification")
    return ap.parse_args()

# ------------- CLI ------------
def run():
    return None
def main():
    logger = setup_logging()
    args = parse_args()
    run()


if __name__ == "__main__":
    main()


