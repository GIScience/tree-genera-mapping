"""This module provides functionality to create a Canopy Height Model (CHM)"""

import shutil, os
from pathlib import Path
from typing import Union, List, Tuple, Dict
from collections import defaultdict
import uuid
import zipfile
import tempfile
import tqdm
from pyproj import CRS as PyprojCRS

import rasterio
from rasterio.crs import CRS
from affine import Affine
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds
from rasterio.enums import Resampling as ResamplingEnums
from scipy.ndimage import label, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

import logging

# LOGGING
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----------------------------------------------------------------------
# Core dataset class
# ----------------------------------------------------------------------
class HeightModel:
    def __init__(self,
                 dgm_path,
                 dom_path,
                 key,
                 output_dir,
                 res,  # [0.2 OR 1] in m
                 crs='EPSG:25832'
                 ):
        self.dgm_path = dgm_path  # Path to the Digital Terrain Model (DTM)
        self.dom_path = dom_path  # Path to the Digital Surface Model (DSM)
        self.key = key  # Tile Identifier for the output file
        self.res = res  # Desired resolution in meters [0.2,1]
        self.crs = crs  # Coordinate Reference System by default is EPSG:25832
        self.output_dir = Path(output_dir)  # Directory to save the output CHM
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_name = f'chm_{key}_{int(res * 100)}.tif'  # Output file name

    def generate_chm(self):
        # 1. Load DSM (DOM)
        with rasterio.open(self.dom_path) as dom_src:
            dom_data = dom_src.read(1)
            dom_transform = dom_src.transform
            dom_crs = dom_src.crs
            dom_nodata = dom_src.nodata
        if dom_crs is None:
            dom_crs = CRS.from_epsg(int(self.crs.split(":")[1]))

        # 2. Load / build DGM on same grid as DOM
        if self.dgm_path.suffix == '.xyz':
            # Robust read of XYZ (ignore malformed rows)
            dgm_xyz_data = np.genfromtxt(
                self.dgm_path,
                delimiter=" ",
                usecols=(0, 1, 2),
                invalid_raise=False,
            )
            # Handle weird files that may produce 1D or contain NaNs
            if dgm_xyz_data.ndim == 1 and dgm_xyz_data.size == 3:
                dgm_xyz_data = dgm_xyz_data.reshape((1, 3))

            # Drop rows with NaNs
            dgm_xyz_data = dgm_xyz_data[~np.isnan(dgm_xyz_data).any(axis=1)]
            if dgm_xyz_data.size == 0:
                raise ValueError(f"DGM xyz file {self.dgm_path} has no valid rows")

            # Read XYZ
            # dgm_xyz_data = np.loadtxt(self.dgm_path, delimiter=' ')

            dgm_x, dgm_y, dgm_z = dgm_xyz_data[:, 0], dgm_xyz_data[:, 1], dgm_xyz_data[:, 2]

            # Use DOM grid (same extent, same resolution, same shape)
            pixel_size_x = dom_transform.a  # > 0
            pixel_size_y = -dom_transform.e  # > 0
            height, width = dom_data.shape

            dgm_data = np.full((height, width), np.nan, dtype=np.float32)

            # Map XYZ coordinates to DOM pixel indices
            # col = (x - x0) / pixel_size_x
            # row = (y0 - y) / pixel_size_y   (note: y0 is top)
            x0 = dom_transform.c
            y0 = dom_transform.f

            col_indices = ((dgm_x - x0) / pixel_size_x).astype(int)
            row_indices = ((y0 - dgm_y) / pixel_size_y).astype(int)

            # Clip to grid bounds
            col_indices = np.clip(col_indices, 0, width - 1)
            row_indices = np.clip(row_indices, 0, height - 1)

            dgm_data[row_indices, col_indices] = dgm_z

            dgm_transform = dom_transform
            dgm_crs = dom_crs if dom_crs is not None else self.crs
            if isinstance(dgm_crs, str):
                dgm_crs = CRS.from_epsg(int(dgm_crs.split(':')[1]))

        else:
            # Load DGM from raster
            with rasterio.open(self.dgm_path) as dgm_src:
                dgm_data = dgm_src.read(1)
                dgm_transform = dgm_src.transform
                dgm_crs = dgm_src.crs
            if dgm_crs is None:
                dgm_crs = dom_crs

        # 3. If CRS or shape mismatch, reproject DGM to DOM grid
        if (dgm_crs != dom_crs) or (dgm_data.shape != dom_data.shape):
            dgm_data, dgm_transform = self._reproject(
                data=dgm_data,
                transform=dgm_transform,
                src_crs=dgm_crs,
                dst_crs=dom_crs,
                target_shape=dom_data.shape,
                target_transform=dom_transform,
            )

        # 4. Compute CHM (DOM - DGM), using DOM nodata mask
        self.chm = np.where(
            (dom_data != dom_nodata) & ~np.isnan(dgm_data),
            dom_data - dgm_data,
            np.nan,
        )

        # Store transform & CRS
        self.transform = dom_transform
        self.crs = dom_crs

        # 5. Resample CHM if needed (your original logic)
        if self.res != self.transform.a:
            self.chm, self.transform = self._resample(self.chm, self.transform, self.res)

        # 6. Save output
        self._save()

    def _reproject(self, data, transform, src_crs, dst_crs, target_shape=None, target_transform=None):
        """Reproject raster data to a different CRS, optionally forcing alignment to target grid."""
        if target_shape is not None and target_transform is not None:
            height, width = target_shape
            dst_transform = target_transform
        else:
            bounds = array_bounds(data.shape[0], data.shape[1], transform)
            dst_transform, width, height = calculate_default_transform(
                src_crs, dst_crs, data.shape[1], data.shape[0], *bounds
            )

        reprojected_data = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=data,
            destination=reprojected_data,
            src_transform=transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            init_dest_nodata=np.nan
        )

        return reprojected_data, dst_transform

    def _resample(self, data, transform, target_res):
        """Resample raster data to a new resolution."""
        # Calculate the scaling factor
        scale_factor = transform.a / target_res

        # Calculate the dimensions of the resampled raster
        new_width = int(data.shape[1] * scale_factor)
        new_height = int(data.shape[0] * scale_factor)

        # Initialize an array for the resampled data
        resampled_data = np.empty((new_height, new_width), dtype=np.float32)

        # Perform the resampling
        reproject(
            source=data,
            destination=resampled_data,
            src_transform=transform,
            src_crs=self.crs,
            dst_transform=transform * transform.scale(1 / scale_factor, 1 / scale_factor),
            dst_crs=self.crs,
            resampling=ResamplingEnums.bilinear
        )

        return resampled_data, transform * transform.scale(1 / scale_factor, 1 / scale_factor)

    def _save(self):
        """Save the CHM to a GeoTIFF file."""
        output_path = os.path.join(self.output_dir, self.file_name)
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=self.chm.shape[0],
                width=self.chm.shape[1],
                count=1,
                dtype=self.chm.dtype,
                crs=self.crs,
                transform=self.transform,
        ) as dst:
            dst.write(self.chm, 1)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def pair_dgm_dom(files: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Pair DGM and DOM files based on items 3 and 4 in the filename.

    Args:
        files (List[str]): List of filenames to pair.

    Returns:
        Tuple[Dict[str, List[str]], List[str]]:
            A dictionary of paired files with the key as the unique identifier
            and a list of associated 'dgm' and 'dom' files as values,
            and a list of unmatched files.
    """
    file_dict = defaultdict(lambda: {'dgm': None, 'dom': None})
    unmatched_files = []

    # Iterate through files to extract the key and categorize as DGM or DOM
    for file in files:
        parts = file.split('_')
        if len(parts) >= 5:
            key = f"{parts[2]}_{parts[3]}"  # Use items 3 and 4 as the pairing key
            if file.startswith('dgm'):
                file_dict[key]['dgm'] = file
            elif file.startswith('dom'):
                file_dict[key]['dom'] = file

    # Create a dictionary with the key and paired files as a list
    pairs = {}
    for key, file_data in file_dict.items():
        if file_data['dgm'] and file_data['dom']:
            pairs[key] = [file_data['dgm'], file_data['dom']]
        else:
            unmatched_files.extend(filter(None, file_data.values()))

    return pairs, unmatched_files


def process_ndsm(input_dir: Path,
                 pairs: Tuple[Dict[str, List[str]], List[str]],
                 output_dir: Path = Path('cache/height')
                 ):
    """
    Process tile to create nDSM [nDSM = DOM - DGM]
    Firstly create tempfolder and unzip all files from these zip files
    Save output raster as f'nDSM_32_{pair.key}_2.tif'
    :param pair:
    :return:
    """

    for key, pair in tqdm.tqdm(pairs.items()):
        temp_dir = tempfile.mkdtemp(dir='cache/temp') if 'cache/temp' else tempfile.mkdtemp()
        try:
            # Unzip files to the temporary directory
            for zip_file in pair:
                zip_path = Path(input_dir) / zip_file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    log.info(f"Extracted {zip_file} to {temp_dir}")
                # Find the unzipped raster files in the temp directory
                if 'dom' in zip_file:
                    dom_files = [f for f in Path(f"{temp_dir}/{zip_file[:-4]}").glob('*dom*.tif')]
                elif 'dgm' in zip_file:
                    dgm_files = [f for f in Path(f"{temp_dir}/{zip_file[:-4]}").glob('*dgm*.xyz')]

            if not dom_files or not dgm_files:
                log.error(f"Missing DOM or DGM file for key {key}")
                return False
            else:
                for dgm_file in tqdm.tqdm(dgm_files):
                    key_2 = dgm_file.name[8:16]
                    # dom_file = [file for file in dom_files if key_2 in str(file.name)][0]
                    # Simplified and improved file selection
                    dom_file = next((file for file in dom_files if key_2 in file.name), None)

                    if not dom_file:
                        log.info(f"Missing DOM for {dgm_file}")

                    # Read the DOM file
                    with rasterio.open(dom_file) as dom_src:
                        dom_data = dom_src.read(1)
                        dom_transform = dom_src.transform
                        dom_crs = dom_src.crs
                        dom_shape = dom_src.shape
                        dom_nodata = dom_src.nodata

                    # Read the DGM file in .xyz format
                    # dgm_df = pd.read_csv(dgm_file, delim_whitespace=True, header=None, names=['x', 'y', 'z'])
                    dgm_data = np.loadtxt(dgm_file, delimiter=' ')
                    dgm_x, dgm_y, dgm_z = dgm_data[:, 0], dgm_data[:, 1], dgm_data[:, 2]

                    # Assuming the .xyz file uses a known CRS (e.g., EPSG:32632)
                    dgm_crs = 'EPSG:25832'

                    # Calculate raster bounds and resolution
                    pixel_size_x = dom_transform.a
                    pixel_size_y = -dom_transform.e

                    min_x, min_y = np.min(dgm_x), np.min(dgm_y)
                    max_x, max_y = np.max(dgm_x), np.max(dgm_y)

                    width = int((max_x - min_x) / pixel_size_x)
                    height = int((max_y - min_y) / pixel_size_y)

                    dgm_raster = np.full((height, width), np.nan, dtype=np.float32)

                    # Convert coordinates to raster indices with boundary checks
                    col_indices = np.clip(((dgm_x - min_x) / pixel_size_x).astype(int), 0, width - 1)
                    row_indices = np.clip(((max_y - dgm_y) / pixel_size_y).astype(int), 0, height - 1)

                    dgm_raster[row_indices, col_indices] = dgm_z

                    # Define the transform for the DGM raster
                    # dgm_transform = rasterio.transform.from_origin(min_x, max_y, pixel_size_x, pixel_size_y)

                    log.info("DGM data successfully converted to raster format")

                    # Now proceed to calculate nDSM
                    ndsm_data = np.where(
                        (dom_data != dom_nodata) & ~np.isnan(dgm_raster),
                        dom_data - dgm_raster,
                        np.nan
                    )

                    # Prepare output path for nDSM
                    output_path = output_dir / f'nDSM_32_{key_2}_2.tif'
                    log.info(f"Saving nDSM to {output_path}")

                    # Write the nDSM raster to a GeoTIFF
                    ndsm_profile = dom_src.meta
                    ndsm_profile.update(
                        dtype=rasterio.float32,
                        count=1,
                        # compress='lzw',
                        nodata=-9999,
                        transform=dom_transform
                    )

                    with rasterio.open(output_path, 'w', **ndsm_profile) as dst:
                        dst.write(ndsm_data.astype(rasterio.float32), 1)

                    log.info(f"Successfully processed nDSM for key {key}")

        except Exception as e:
            log.info(f"An error occurred during processing for key {key}: {e}")

        finally:
            # Always delete the temporary directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
                log.info(f"Deleted temporary directory: {temp_dir}")


def apply_treshold(image: Union[str, Path],
                   output_mask_path: str = None,
                   threshold: float = 2.0,
                   resolution: float = None
                   ):
    # Convert image to a Path if it's a string
    if isinstance(image, str):
        image = Path(image)
    # Open image and perform threshold
    with rasterio.open(image) as img:
        # Read image and meta
        img_meta = img.meta.copy()
        img_meta.update(dtype='float32', count=1)
        values = img.read(1).astype('float32')

        # Replace values below threshold with 0 (or set to a nodata value)
        threshold_values = np.where(values >= threshold, values, 0)

        # If a new resolution is provided, resample the thresholded array
        if resolution is not None:
            # Get the original image bounds
            left, bottom, right, top = img.bounds
            # Create a new transform using the desired resolution
            new_transform = Affine(resolution, 0, left,
                                   0, -resolution, top)
            # Calculate the new width and height from the bounds
            new_width = int(np.ceil((right - left) / resolution))
            new_height = int(np.ceil((top - bottom) / resolution))
            # Update the metadata with the new transform and dimensions
            img_meta.update({
                "transform": new_transform,
                "width": new_width,
                "height": new_height,
            })
            # Prepare an empty array to hold the resampled data
            new_threshold_values = np.empty((new_height, new_width), dtype=np.float32)
            # Resample using rasterio's reproject (using nearest-neighbor resampling)
            reproject(
                source=threshold_values,
                destination=new_threshold_values,
                src_transform=img.transform,
                src_crs=img.crs,
                dst_transform=new_transform,
                dst_crs=img.crs,
                resampling=Resampling.nearest,
            )
            threshold_values = new_threshold_values

        # Save threshold values to new image
        if output_mask_path != None:
            output_mask_path = Path(output_mask_path)
            output_mask_path.mkdir(parents=True, exist_ok=True)
            output_image_path = output_mask_path / image.name

            with rasterio.open(output_image_path, 'w', **img_meta) as dst:
                dst.write(threshold_values.astype(rasterio.float32), 1)
    return threshold_values


def process_ndvi(input_dir: Path,
                 output_dir: Path = Path('cache/ndvi')
                 ):
    zip_files = [zipfolder for zipfolder in input_dir.glob("*dop*.zip")]
    temp_dir = tempfile.mkdtemp(dir='cache/temp') if 'cache/temp' else tempfile.mkdtemp()
    for zip_path in tqdm.tqdm(zip_files):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            log.info(f"Extracted {zip_path} to {temp_dir}")
        tif_files = [f for f in Path(f'{temp_dir}/{zip_path.name[:-4]}').glob('*.tif')]
        for tif_file in tif_files:
            with rasterio.open(tif_file) as img:
                if img.count < 4:
                    log.warning(f"Image {tif_file} does not have enough bands for NDVI calculation.")
                    continue
                if not img.crs:
                    img_profile = img.meta.copy()
                    img_profile.update({'crs': 'EPSG:25832'})
                # Read the red (band 3) and NIR (band 4) bands
                red_band = img.read(3).astype('float32')
                nir_band = img.read(4).astype('float32')

                # Calculate NDVI using the formula
                ndvi = np.where(
                    (nir_band + red_band) == 0,
                    np.nan,
                    (nir_band - red_band) / (nir_band + red_band)
                )

                # Prepare output paths for NDVI and mask
                output_ndvi_path = Path(f'{output_dir}/NDVI_{tif_file.stem}.tif')
                output_ndvi_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the NDVI as a GeoTIFF
                ndvi_profile = img_profile.copy()
                ndvi_profile.update(
                    dtype=rasterio.float32,
                    count=1,
                    compress='lzw',
                    nodata=np.nan
                )

                with rasterio.open(output_ndvi_path, 'w', **ndvi_profile) as dst:
                    dst.write(ndvi.astype(rasterio.float32), 1)

        # Remove temp folder
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            log.info(f"Deleted temporary directory: {temp_dir}")


def tree_canopy(output_dir='cache/canopy_height',
                ndvi_dir='cache/mask_ndvi',
                height_dir='cache/mask_height'
                ):
    output_dir = Path(output_dir)
    ndvis = list(Path(ndvi_dir).glob('*.tif'))
    ndvis.sort()
    heights = list(Path(height_dir).glob('*.tif'))
    skipped = []
    for ndvi_path in tqdm.tqdm(ndvis, total=len(ndvis)):
        key = '_'.join(ndvi_path.name.split('_')[3:5])
        height_path = [h for h in list(heights) if key in h.name]
        if not height_path:
            skipped.append(key)
        else:
            with rasterio.open(height_path[0]) as height_img:
                height = height_img.read(1).astype('float32')
                height = np.where(height >= 2, height, 0)
                height_meta = height_img.meta.copy()

                with rasterio.open(ndvi_path) as ndvi_img:
                    ndvi = ndvi_img.read(1).astype('float32')
                    ndvi_mask = np.where(ndvi >= 0.3, 1, 0)

                    # Multiply height by the NDVI mask
                    masked_height = height * ndvi_mask

                    # Update metadata if needed
                    height_meta.update(dtype='float32', count=1)

                    # Save the result as a GeoTIFF file
                    output_path = output_dir / f'chm_{key}_1.tif'
                    with rasterio.open(output_path, 'w', **height_meta) as dst:
                        dst.write(masked_height.astype('float32'), 1)

    print(skipped)


def main(dom_dgm_dir: str = '/Users/ygrinblat/Documents/HeiGIT_Projects/green_spaces/Data/nDSM1m/DOM',
         dop_dir: str = '/Users/ygrinblat/Documents/HeiGIT_Projects/green_spaces/Data/DOP20',
         chm_dir: str = 'cache/canopy_height'):
    """
        Examples to create Canopy Height Model based on LGL BW Geoportal Data
        1: Create nDSM = DSM - DEM
            a: Select all high objects, H>2m - all values low are none
        2. Create NDVI from DOP20
            a: Make mask for vegetation NDVI>0.3
        4. Clip mask from nDSM
        5. apply Watershed model to assess number of trees.
        6. Results: XY tree, Polygon of a canopy with area, diameter and height of a tree
        Assumption: Good alignment
        """
    # Run the pairing function
    pairs, unmatched_files = pair_dgm_dom([file.name for file in Path(dom_dgm_dir).glob('*.zip')])
    # Run unzip for each pair
    process_ndsm(input_dir=dom_dgm_dir, pairs=pairs)
    # Process NDVI
    process_ndvi(input_dir=Path(dop_dir))
    # Threshold of Crop Height Model and NDVI
    for dir, res in zip(['cache/ndvi', 'cache/height'], [1.0, None]):
        imgs = list(Path(dir).glob('*.tif'))
        for img in tqdm.tqdm(imgs, total=len(imgs)):
            _ = apply_treshold(image=img,
                               output_mask_path=f"cache/mask_{dir.split('/')[-1]}",
                               resolution=res,
                               threshold=2)

    # Clip height mask by ndvi mask
    tree_canopy(output_dir=chm_dir,
                ndvi_dir='cache/mask_ndvi',
                height_dir='cache/mask_height'
                )
    # Apply local peak to detect xy trees and height

    trees_dir = 'cache/trees'
    chms = list(Path(chm_dir).glob('*.tif'))
    # Example usage
    for chm in tqdm.tqdm(chms, total=len(chms)):
        process_trees(chm_img=str(chm),
                      output_geojson_path=f'{trees_dir}/{chm.name[:-4]}.geojson',
                      min_tree_area=4,
                      min_distance=5
                      )


def process_trees(chm_img: str = 'cache/canopy_height.tif',
                  output_geojson_path: str = 'cache/trees/tree_centers.geojson',
                  min_tree_area: int = 4,
                  min_distance: int = 5):
    """
    Tree peak identification using original resolution CHM, optional smoothing,
    and local maxima detection. Saves tree centers and max heights as a GeoJSON file.

    Args:
        chm_img (str): Path to the canopy height model (CHM) image.
        output_geojson_path (str): Path to save the tree centers as a GeoJSON file.
        min_tree_area (int): Minimum area (in pixels) to retain a tree.
        min_distance (int): Minimum distance between local maxima.
        crs (str): Coordinate reference system for GeoJSON output.
    """
    # Ensure the output directory exists
    output_dir = Path(output_geojson_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the CHM image
    with rasterio.open(chm_img) as src:
        chm = src.read(1).astype('float32')
        transform = src.transform
        crs = src.crs

    # Step 1: Detect blobs without smoothing
    labeled_blobs, num_blobs = label(chm > 0)
    print(f"Initially identified {num_blobs} blobs")

    # Step 2: Remove small blobs (trees) with area < min_tree_area
    labeled_blobs = remove_small_objects(labeled_blobs, min_size=min_tree_area)

    # Re-label after cleaning small blobs
    labeled_blobs, num_blobs = label(labeled_blobs > 0)
    print(f"After filtering small objects, {num_blobs} blobs remain")

    # Step 3: Compute the distance transform
    distance = distance_transform_edt(labeled_blobs > 0)

    # Step 4: Detect local maxima using the smoothed CHM
    local_maxima = peak_local_max(distance,
                                  min_distance=min_distance,
                                  labels=labeled_blobs)
    print(f"Detected {len(local_maxima)} local maxima")

    # Prepare tree data for GeoJSON output
    tree_data = []
    for tree_id, (row, col) in enumerate(local_maxima, start=1):
        x, y = rasterio.transform.xy(transform, row, col, offset='center')
        max_height = chm[row, col]
        uuid_treeid = str(uuid.uuid4())

        # Create a GeoDataFrame record
        tree_data.append({
            'TreeID': str(uuid_treeid),
            'MaxHeight': max_height,
            'geometry': Point(x, y)
        })

    # Convert to GeoDataFrame and save as GeoJSON
    gdf = gpd.GeoDataFrame(tree_data, crs=crs)
    if not gdf.empty:
        gdf = gdf.to_crs(epsg=3857)
        gdf.to_file(output_geojson_path, driver='GeoJSON')
        print(f"Saved tree peaks as GeoJSON to {output_geojson_path}")
    else:
        print(f"Empty: {output_geojson_path} is not created")


def height_validation(trees_dir: str, buffer_m: int = 5,
                      trees_true: str = "/Users/ygrinblat/Documents/HeiGIT_Projects/green_spaces/Data/tree-export_detail/tree-export.shp",
                      output_path: str = "cache/trees/closest_single_trees.gpkg"
                      ):
    # read trees into one file
    trees = list(Path(trees_dir).glob("*.geojson"))
    tree_list = []
    for tree_file in tqdm.tqdm(trees, total=len(trees)):
        try:
            gdf = gpd.read_file(str(tree_file))
            if not gdf.empty:
                tree_list.append(gdf)
        except Exception as e:
            print(f"Failed to load {tree_file}: {e}")

    # Combine all loaded GeoDataFrames
    if tree_list:
        gdf_pred = gpd.GeoDataFrame(pd.concat(tree_list, ignore_index=True), crs=tree_list[0].crs)
    else:
        print("No valid tree data loaded.")
    # Select trees close to the True dataset
    gdf_true = gpd.read_file(trees_true)
    gdf_true = gdf_true.to_crs(epsg=3857)
    # Ensure both GeoDataFrames use the same CRS
    if gdf_pred.crs != gdf_true.crs:
        gdf_pred = gdf_pred.to_crs(gdf_true.crs)
    # Prepare columns for the nearest tree data
    gdf_true['nearest_tree_id'] = None
    gdf_true['nearest_tree_height'] = None
    gdf_true['distance_to_tree'] = None

    # Calculate the nearest tree within a 5-meter buffer
    for idx, true_tree in gdf_true.iterrows():
        # Create a 5-meter buffer around the true tree
        buffer = true_tree['geometry'].buffer(buffer_m)

        # Find predicted trees within the buffer
        nearby_trees = gdf_pred[gdf_pred['geometry'].within(buffer)]

        if not nearby_trees.empty:
            # Calculate distances to the true tree
            nearby_trees['distance'] = nearby_trees['geometry'].distance(true_tree['geometry'])
            # Get the nearest tree
            nearest_tree = nearby_trees.loc[nearby_trees['distance'].idxmin()]

            # Update true tree data with nearest tree information
            gdf_true.loc[idx, 'nearest_tree_id'] = nearest_tree['TreeID']
            gdf_true.loc[idx, 'nearest_tree_height'] = nearest_tree['MaxHeight']
            gdf_true.loc[idx, 'distance_to_tree'] = nearest_tree['distance']

    # Save the result as a GeoPackage (GPKG) file
    gdf_true.to_file(output_path, layer='closest_trees', driver='GPKG')

    print(f"Saved closest single trees to {output_path}")


if __name__ == '__main__':
    # Example usage
    import seaborn as sns

    # Load the data
    trees_comparison = 'cache/trees/closest_single_trees.gpkg'
    cols = ['id', 'Corylus', 'ageClass', 'height', 'canopyWidt', 'nearest_tree_id', 'nearest_tree_height',
            'distance_to_tree', 'geometry']

    gdf = gpd.read_file(trees_comparison, columns=cols)
    gdf = gdf.dropna()
    gdf['nearest_tree_height'] = gdf['nearest_tree_height'].astype('float')
    gdf['height'] = gdf['height'].astype('float')

    # Convert to a standard pandas DataFrame for easier plotting
    df = pd.DataFrame(gdf.drop(columns='geometry'))

    # Calculate the height difference for calibration
    df['height_diff'] = df['height'] - df['nearest_tree_height']
    df['height_diff'] = df['height_diff'].dropna()  # Only consider non-null values

    # Create a single figure with multiple subplots
    plt.figure(figsize=(15, 5))

    # Distribution of Tree Heights
    plt.subplot(1, 3, 1)
    sns.histplot(df['height'].dropna(), kde=True, bins=30, color='skyblue')
    plt.title('Distribution of True Tree Heights')
    plt.xlabel('Height (m)')
    plt.ylabel('Frequency')

    # Distribution of Canopy Widths
    plt.subplot(1, 3, 2)
    sns.histplot(df['nearest_tree_height'].dropna(), kde=True, bins=30, color='green')
    plt.title('Distribution of Predicted Tree Heights')
    plt.xlabel('Height (m)')
    plt.ylabel('Frequency')

    # Difference Between Actual and Nearest Tree Heights
    plt.subplot(1, 3, 3)
    sns.histplot(df['height_diff'], kde=True, bins=30, color='orange')
    plt.title('Height Difference for Calibration')
    plt.xlabel('Height Difference (m)')
    plt.ylabel('Frequency')
    plt.axvline(0, color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig('cache/distribution_tree_height.png', format='png')
    plt.show()





