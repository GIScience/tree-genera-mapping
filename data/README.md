# Data

This directory contains the **reference spatial grid** and **partition strategies** used by the project.

## 1. Statewide Tiling for Data Acquisition  

`tiles.gpkg` defines the complete set of spatial grid tiles and their
unique identifiers used throughout the pipeline.

- Acts as the **authoritative tiling scheme**
- Tile IDs are assumed to be stable and consistent
- Used for training, inference, and post-processing

#### Data provenance

The grid tiles correspond to the **LGL UrbanGreen tiling scheme** and are
associated with the mapping of urban green spaces from multispectral
aerial imagery and airborne LiDAR data.

The underlying remote sensing data are provided by the  
**LGL Open GeoData Portal (Baden-Württemberg)**:  
https://www.lgl-bw.de/Produkte/Open-Data/

Using this portal, users can download **1 × 1 km tiles** of:
- very high resolution (VHR) **Multispectral** aerial **imagery** - **20cm spatial resolution**
- **Height Model** derived from an airborne LiDAR - **1m spatial resolution**

These datasets are intended to be aligned with the grid defined in
`tiles.gpkg`.

#### Notes

- The file (~14 MB) is intentionally included in the repository to ensure
  reproducibility.
- Coordinate reference system (CRS): `EPSG:25832`

## 2. Genus Taxonomy
`genera_labels.csv` - Mapping genus names into model labels

## 3. Partition Strategies Files
`tiles_split.txt` and  `tiles_split_city.txt` files contain parent-tile ids and partition status at tile- or city-levels

## 4. Subtiles Files 
`subtile_split.txt` - subtile_id, split mode, size and overlap arguments

## 5. Image samples
5-channel raster stack in `.tif` format for demo purpose in `data/samples` directroy.