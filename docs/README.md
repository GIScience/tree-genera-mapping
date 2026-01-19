# End-to-End Training and Inference Pipeline
This repository provides a workflow for urban tree detection and genus mapping. The pipeline writes all intermediate outputs to `cache/` (ignored by git). The only committed spatial dataset is `data/tiles.gpkg` (1×1 km grid).

## How to run the pre-trained YOLOv11l model 5CH imagery

1. Download LGL products to Generate TileDataset for selected tile ids:
```bash
python acquisition/run_tile_dataset.py  
```

2. Run pre-trained YOLOv11l model to detect and classify tree genus:
```bash
python jobs/run_inference.py --tiles-gpkg data/tiles.gpkg --images-dir cache/tiles_5ch --model-path models/pretrained_yolov11l_tree_genus.pth --output-dir cache/initial_inference
```


## Tree Genera Mapping Teacher-Student Ensemble Training with Human-in-the-Loop Curation:
- Train teacher models using the initial weak and reference labels:
  - a tree detection teacher model, 
  - a tree genus classification teacher model.
- Apply the teacher ensemble to unlabeled sub-tiles to generate pseudo-labels.
- Perform human-in-the-loop curation on the predictions to improve label quality.
  - visual validation  and manual correction of predictions in QGIS.
- Treat the curated outputs as hard labels to form the final training dataset.
### Teacher Ensemble Steps:
1. Generate weak tree labels from NDVI + height thresholds:
```bash
python segment_trees.py \
  --img-dir /path/to/output/merged \
  --output-dir /path/to/seg_out \
  --mode rgbih \
  --write-bbox \
  --write-masks \
  --mask-encoding 0255 \
  --ndvi-thr 0.25 \
  --height-thr 2.0 \
  --min-distance-px 3
  
python preprocess/tree_delineation.py --tiles-dir cache/tiles_5ch --output-gpkg cache/weak_tree_bboxes.gpkg
```

2. GreeHill Tree Genera Labels Preparation: 
- GreeHill tree inventory (.gpkg) with columns: tree_id, genus, canopyWidt, geometry (Point), CRS in meters (e.g., EPSG:25832). Download via `url-link-to-GreeHill-dataset`
- Prepare GreeHill  labels (with optional canopy bounding boxes):
```bash
python preprocess/prepare_genus_labels.py --trees /path/to/GreeHill_dataset.gpkg --labels conf/genera_labels.csv --output cache/tree_labels_bbox.gpkg --make-bbox
```

3. Generate Initial Training Datasets for Teacher Ensemble Models:

3.1. detection (YOLO images + txt labels)
- **NOTE: HUMAN-IN-THE-LOOP CURATION** of tree labels before running generation of training labels. QGIS can be used to visualize and edit the generated weak tree bounding boxes.  
```bash
 python preprocess/make_training_data.py det --tiles-gpkg data/tiles.gpkg --weak-bboxes-gpkg cache/weak_tree_bboxes.gpkg --images-dir cache/tiles_5ch --mode rgbih --output-dir cache/datasets/yolo_tree_det
```

3.2. Genus patches (classification dataset)
- this is an image with 5-channels (RGB + IR + Height) and corresponding genus labels for each tree/patch.
- Size is uniform for all patches (e.g., 128x128px) 
```bash

python preprocess/make_training_data.py patches --tiles-gpkg data/tiles.gpkg --genus-labels-gpkg cache/tree_labels_bbox.gpkg --images-dir cache/tiles_5ch --mode rgbih --output-dir cache/datasets/genera_patches --patch-size 128
```

4. Train Teacher Models:
```bash 
python train/teacher_tree_train.py
```
```bash 
python train/teacher_genus_train.py
```

### Student Model Training:
   1. Download train images and curated pseudo-labels labels via `url-link` -> cache
   2. Train a YOLOv11-L student model
```bash
python train/yolo_train.py 
```
   3. Evaluate the trained model on held-out validation tiles.
```bash
python train/yolo_eval.py
```
## Scalable Statewide Inference:
   1. Download multispectral aerial imagery and airborne LiDAR data for all tiles in `data/tiles.gpkg` using the `acquisition/run_tile_dataset.py` script.

```bash
python acquisition/run_tile_dataset.py --tiles-gpkg data/tiles.gpkg --output-dir cache/tiles_5ch --mode RGBIH
```
   2. Perform large-scale inference using the trained student model with tiled processing and GPU-parallel execution. 
     - see the example of the code `jobs/run_inference.py`
   3. Merge predictions across tile boundaries and export results as GIS-ready layers for further analysis and visualization.   Export final tree crown detections and genus classifications as GeoPackage files compatible with GIS software `jobs/finalize_results.py`.