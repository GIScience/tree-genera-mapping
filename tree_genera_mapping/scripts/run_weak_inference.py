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

from pathlib import Path
