import geopandas as gpd
import pandas as pd
from shapely.geometry import box

def _make_bbox(geom, w):
    if geom is None or geom.geom_type != "Point":
        return None
    if pd.isna(w) or w <= 0:
        return None
    half = float(w) / 2.0
    return box(geom.x - half, geom.y - half, geom.x + half, geom.y + half)
