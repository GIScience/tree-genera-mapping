# End-to-End Training and Inference Pipeline
This repository provides a workflow for urban tree detection and genus mapping. The pipeline writes all intermediate outputs to `cache/` (ignored by git). The only committed spatial dataset is `data/tiles.gpkg` (1×1 km grid).

## Initial Training Data Preparation

**Inputs (download separately):**

- GreeHill tree inventory (.gpkg) with columns: tree_id, genus, canopyWidt, geometry (Point), CRS in meters (e.g., EPSG:25832).
- LGL imagery + height (download manually or via script).

**Steps:**

1. Download LGL tiles for a selected tile id:
```bash
python lgl_downloader.py \
  --tiles-gpkg data/tiles.gpkg \
  --output-dir cache/lgl_store \
  --tile-id 457-5428 \
  --products dop20rgbi,ndom1
```
Or download multiple tiles by specifying a comma-separated list of tile ids:

```bash
python lgl_downloader.py \
  --tiles-gpkg data/tiles.gpkg \
  --output-dir cache/lgl_store \
  --tile-ids 457-5428,458-5428 \
  --products dop20rgbi,ndom1
```
Download all tiles (for example, dop20rgbi products only): 
```bash 
python lgl_downloader.py \
  --tiles-gpkg data/tiles.gpkg \
  --output-dir cache/lgl_store \
  --products dop20rgbi
```

2. Build 5-channel raster tiles (RGB + NIR + normalized height) into cache/tiles_5ch/:
```bash
python preprocess/build_5ch_tiles.py --input-dir cache/lgl_store --tiles-gpkg data/tiles.gpkg --output-dir cache/tiles_5ch
```

3. Generate weak tree labels from NDVI + height thresholds:
```bash
python preprocess/tree_delineation.py --tiles-dir cache/tiles_5ch --output-gpkg cache/weak_tree_bboxes.gpkg
```

4. Prepare genus labels from GreeHill inventory (and optional canopy bounding boxes):
```bash
python preprocess/prepare_genus_labels.py --trees /path/to/GreeHill_dataset.gpkg --labels conf/genera_labels.csv --output cache/tree_labels_bbox.gpkg --make-bbox
```

5. Generate training datasets:

- detection (YOLO images + txt labels)

- genus patches (classification dataset)
```bash
python preprocess/make_training_data.py det --tiles-gpkg data/tiles.gpkg --weak-bboxes-gpkg cache/weak_tree_bboxes.gpkg --images-dir cache/tiles_5ch --mode rgbih --output-dir cache/datasets/yolo_tree_det
python preprocess/make_training_data.py patches --tiles-gpkg data/tiles.gpkg --genus-labels-gpkg cache/tree_labels_bbox.gpkg --images-dir cache/tiles_5ch --mode rgbih --output-dir cache/datasets/genera_patches --patch-size 128
```

## Teacher Ensemble Training and Pseudo-Labeling:
   - Train teacher models using the initial weak and reference labels:
     - a tree detection teacher model, 
     - a tree genus classification teacher model.
   - Apply the teacher ensemble to unlabeled sub-tiles to generate pseudo-labels.
   - Perform human-in-the-loop curation on the predictions to improve label quality.
     - visual validation  and manual correction of predictions in QGIS.
   - Treat the curated outputs as hard labels to form the final training dataset.
## Student Model Training:
   - Train a YOLOv11-L student model using the combined dataset of initial labeled data and curated pseudo-labeled data. 
   - The student model jointly learns tree detection and tree genus classification in a single-stage framework. 
   - Evaluate the trained model on held-out validation tiles.
## Scalable Statewide Inference:
   - Download multispectral aerial imagery and airborne LiDAR data for all tiles in `data/tiles.gpkg` using the `acquisition/lgl_downloader.py` script. 
   - Perform large-scale inference using the trained student model with tiled processing and GPU-parallel execution. 
     - see the example of the code `jobs/run_inference.py`
   - Merge predictions across tile boundaries and export results as GIS-ready layers for further analysis and visualization.   
   - Export final tree crown detections and genus classifications as GeoPackage files compatible with GIS software `jobs/finalize_results.py`.