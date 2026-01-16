# End-to-End Training and Inference Pipeline
This repository implements a full workflow for urban tree detection and genus mapping, from initial data preparation to scalable statewide inference. The pipeline consists of four main stages:

1. Initial Training Data Preparation
- Download the GreeHill terrestrial LiDAR dataset containing sparse tree locations, canopy size information, and genus reference labels (HEiDATA repository).    
- Select training tiles that spatially intersect with the GreeHill dataset using `data/tiles.gpkg` [1x1km, 20cm spatial resolution]. 
- Acquire multispectral aerial imagery (RGB + NIR) and airborne LiDAR data from the LGL Open GeoData Portal (Baden-Württemberg), or automatically download the data using `acquisition/lgl_downloader.py` with tile IDs from `data/tiles.gpkg`. 
- Preprocess the imagery to generate 5-channel raster sub-tiles (RGB + NIR + normalized height) 640x640px `preprocess/XXXX.py`.
- Select sub-tiles containing vegetation to form the initial training pool. 
- Generate weak tree presence labels using heuristic tree delineation based on NDVI and height thresholding with `preprocess/tree_delineation.py`. 
- Prepare initial tree genus labels from the GreeHill dataset using `preprocess/prepare_genus_labels.py`. 
- Generate image patches for genus classification using `preprocess/generate_genus_patches.py`.
2. Teacher Ensemble Training and Pseudo-Labeling:
   - Train teacher models using the initial weak and reference labels:
     - a tree detection teacher model, 
     - a tree genus classification teacher model.
   - Apply the teacher ensemble to unlabeled sub-tiles to generate pseudo-labels.
   - Perform human-in-the-loop curation on the predictions to improve label quality.
     - visual validation  and manual correction of predictions in QGIS.
   - Treat the curated outputs as hard labels to form the final training dataset.
3. Student Model Training:
   - Train a YOLOv11-L student model using the combined dataset of initial labeled data and curated pseudo-labeled data. 
   - The student model jointly learns tree detection and tree genus classification in a single-stage framework. 
   - Evaluate the trained model on held-out validation tiles.
4. Scalable Statewide Inference:
   - Download multispectral aerial imagery and airborne LiDAR data for all tiles in `data/tiles.gpkg` using the `acquisition/lgl_downloader.py` script. 
   - Perform large-scale inference using the trained student model with tiled processing and GPU-parallel execution. 
     - see the example of the code `jobs/run_inference.py`
   - Merge predictions across tile boundaries and export results as GIS-ready layers for further analysis and visualization.   
   - Export final tree crown detections and genus classifications as GeoPackage files compatible with GIS software `jobs/finalize_results.py`.