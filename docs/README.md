# End-to-End Training and Inference Pipeline
This repository provides a workflow for urban tree detection and genus mapping. The pipeline writes all intermediate outputs to `cache/` (ignored by git). The only committed spatial dataset is `data/tiles.gpkg` (1×1 km grid).

## How to run the pre-trained YOLOv11l model 5CH imagery

1. Download LGL products to Generate TileDataset for selected tile ids:
```bash
python scripts/fetch_tiles.py --output-dir cache/merged --tiles-gpkg data/tiles.gpkg --tmp-root cache/ --mode RGBIH --tile_ids 355_6048 355_6049
```

2. Run pre-trained YOLOv11l model to detect and classify tree genus:
```bash
python scripts/predict_yolo.py --tiles-gpkg data/tiles.gpkg --images-dir cache/tiles_5ch --model-path cache/models/pretrained_yolov11l_genus_genus.pth --output-dir cache/predictions
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
1. Detect and Generate tree canopy polygons based on the watershed segmentation of NDVI and Height Model.
```bash
python scripts/segment_trees.py \
  --img-dir merged \
  --output-dir cache/weak_tree_labels \
  --mode rgbih \
  --ndvi-thr 0.2 \
  --height-thr 2.0 
```

2. GreeHill Tree Genera Labels Preparation: 
- GreeHill tree inventory (.gpkg) with columns: tree_id, genus, canopyWidt, geometry (Point), CRS in meters (e.g., EPSG:25832). Download via `url-link-to-GreeHill-dataset`
- Prepare GreeHill  labels (with optional canopy bounding boxes):
```bash
python preprocess/genus_labels.py --trees /path/to/GreeHill_dataset.gpkg --labels conf/genera_labels.csv --output cache/genus_labels.gpkg --make-bbox
```

3. Generate Initial Training Datasets for Teacher Ensemble Models:

3.1. detection (YOLO images + txt labels)
- **NOTE: HUMAN-IN-THE-LOOP CURATION** of tree labels before running generation of training labels. QGIS can be used to visualize and edit the generated weak tree bounding boxes and merged into a single GPKG file.  
```bash
 python scripts/build_datasets.py det \
  --tiles-gpkg data/tiles.gpkg \
  --weak-bboxes-gpkg cache/weak_tree_bboxes.gpkg \
  --images-dir /path/to/tiles \
  --output-dir /path/to/out_dataset \
  --mode rgbih \
  --val-frac 0.2 
```

3.2. Genus patches (classification dataset)
- this is an image with 5-channels (RGB + IR + Height) and corresponding genus labels for each tree/patch.
- Size is uniform for all patches (e.g., 128x128px) 
```bash
python scripts/build_datasets.py patches \
  --tiles-gpkg data/tiles.gpkg \
  --genus-labels-gpkg cache/genus_labels.gpkg \
  --images-dir /path/to/tiles \
  --output-dir /path/to/out_dataset \
  --mode rgbih \
  --patch-size 128 \
  --val-frac 0.2 
```

4. Train Teacher Models:
```bash 
python dl/detection/tree_train.py
```
```bash 
python dl/classification/genus_train.py
```

5. Generate Pseudo-Labels on Unlabeled Tiles:
```bash
python scripts/twostage_teacher_inference.py \
    --tiles-gpkg data/tiles.gpkg \
    
```

6. Human-in-the-Loop Curation of Pseudo-Labels:
    - Visualize and edit the generated pseudo-labels in QGIS to improve label quality.
    - Save the curated labels as `cache/curated_pseudo_labels.gpkg`.
7. Prepare Final Training Dataset with Curated Pseudo-Labels
```bash 
python scripts/build_datasets.py det \
    --tiles-gpkg data/tiles.gpkg \
    --weak-bboxes-gpkg cache/curated_pseudo_labels.gpkg \
    --images-dir /path/to/tiles \
    --output-dir /path/to/final_dataset \
    --mode rgbih \
    --val-frac 0.2
```

### Student Model Training:
   1. Download train images and curated pseudo-labels labels via `url-link` -> cache
   2. Train a YOLOv11-L student model
```bash
python dl/detection/yolo_train.py 
```
   3. Evaluate the trained model on held-out validation tiles.
```bash
python train/yolo_eval.py
```
## Scalable Statewide Inference:
   1. Download multispectral aerial imagery and airborne LiDAR data for all tiles in `data/tiles.gpkg` using the `scripts/fetch_tiles.py` script.

```bash
python scripts/fetch_tiles.py --tiles-gpkg data/tiles.gpkg --output-dir cache/tiles_5ch --mode RGBIH
```
   2. Perform large-scale inference using the trained student model with tiled processing and GPU-parallel execution. 
     - see the example of the code `scripts/predict_yolo.py`
   3. Merge predictions across tile boundaries and export results as GIS-ready layers for further analysis and visualization.   Export final tree crown detections and genus classifications as GeoPackage files compatible with GIS software `scripts/finalize_results.py`.