import numpy as np
import pandas as pd
from shapely.geometry import box
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling as ResamplingEnums

def img_resample(data, data_transform, crs, scale_factor):
    """Resample raster data to a new resolution."""

    # Calculate the dimensions of the resampled raster
    new_width = int(data.shape[1] * scale_factor)
    new_height = int(data.shape[0] * scale_factor)

    # Initialize an array for the resampled data
    resampled_data = np.empty((new_height, new_width), dtype=np.float32)

    # Perform the resampling
    reproject(
        source=data,
        destination=resampled_data,
        src_transform=data_transform,
        src_crs=crs,
        dst_transform=data_transform * data_transform.scale(1 / scale_factor, 1 / scale_factor),
        dst_crs=crs,
        resampling=ResamplingEnums.bilinear
    )

    return resampled_data, data_transform * data_transform.scale(1 / scale_factor, 1 / scale_factor)

def normalize_hm_to_255(chm_data, glb_min, glb_max):
    """Normalize Canopy Height Model (CHM) data to a 0-255 range."""
    chm_min = glb_min  # np.min(chm_data)
    chm_max = glb_max  # np.max(chm_data)

    # Avoid division by zero
    if chm_max == chm_min:
        return np.zeros_like(chm_data, dtype=np.uint8)

    # Normalize to 0-255 range
    normalized_chm = (255 * (chm_data - chm_min) / (chm_max - chm_min)).astype(np.uint8)
    return normalized_chm

def _make_bbox(geom, w):
    if geom is None or geom.geom_type != "Point":
        return None
    if pd.isna(w) or w <= 0:
        return None
    half = float(w) / 2.0
    return box(geom.x - half, geom.y - half, geom.x + half, geom.y + half)
