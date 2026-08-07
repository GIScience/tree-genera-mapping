# Urban Tree Genera Mapping in Baden-Württemberg, Germany

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL3.0-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
> A research pipeline for large-scale urban tree crown detection and tree genus mapping using very-high-resolution multispectral aerial imagery and LiDAR data.

> The multispectral aerial imagery and LiDAR data provided by the **LGL Open GeoData-Portal** https://www.lgl-bw.de/Produkte/Open-Data/ 
> 
![logo.png](docs%2Flogo.png)

##  Overview
Urban Tree Genera Mapping provides an end-to-end, research-oriented workflow to:
- Download and preprocess **LGL Open GeoData** (multi-spectral orthophotos & nDSM).  
- Build **5-channel raster tiles** (RGB + NIR + normalized height). 
- Perform **tree crown delineation and detection**
- Predict **tree genera using deep learning**
- Apply a **teacher–student** learning strategy with human-in-the-loop **curation** 
- Scale inference to **statewide coverage**
- Export results as **GeoPackage** for GIS analysis

The code accompanies an upcoming **open dataset and scientific publication** on regional-scale tree genera mapping in Baden-Württemberg, Germany.

## Method Workflow
![overview_workflow.png](docs%2Foverview_workflow.png)

## Quickstart: 
Clone the repository:
```bash
git clone https://github.com/GIScience/tree-genera-mapping
cd tree-genera-mapping
```
Create and activate a Conda environment:
```bash
conda env create -f environment.yaml
conda activate map-tree-genera
```
Create a kernel to run notebooks scripts
```bash
python -m ipykernel install --user --name map-tree-genera --display-name "Python (tree-genera)"
```
Download the pretrained model (5ch) with weights
```bash
mkdir -p cache/weights
cd cache/weights
wget https://huggingface.co/solo2307/urban-tree-genera/blob/main/yolo11l_tree_genus.pt
wget https://huggingface.co/solo2307/urban-tree-genera/resolve/main/yolo11l_tree.pt
cd ../..
```
## Usage:
### Demo of the pretrained model 
`notebooks/01_demo_inference.ipynb` provides a step-by-step demonstration of the genera predictions over `data/samples` 5-stack images. 

### How to run the Genera Mapping scripts
1. Download LGL products to Generate TileDataset for selected tile ids:
```bash
python tree_genera_mapping/scripts/fetch_tiles.py \
  --tiles-gpkg data/lgl_bw_tiles.gpkg \
  --tile-ids data/tiles_split.txt \
  --tmp-root cache/tmp_dir \
  --output-dir cache/img_dir \
  --mode RGBIH \
  --norm-height global
```

2. Run pre-trained YOLOv11l model to detect and classify tree genus:
```bash
python tree_genera_mapping/scripts/predict_yolo.py \ 
  --tiles-gpkg data/tiles.gpkg \
  --images-dir cache/dataset_dir \
  --model-path cache/yolov11l_tree_genus.pth \
  --output-dir cache/predictions
```

## Train Model
1. Data Preparation

   i. Prepare detection dataset for Genera Mapping
      ```bash
      python -m tree_genera_mapping.scripts.build_dataset det \
          --tiles-gpkg data/lgl_bw_tiles.gpkg \
          --bboxes-gpkg cache/curated_annotations.gpkg \
          --images-dir cache/img_dir \
          --output-dir cache/data \
          --mode rgbih \
          --tile-id-col tile_id \
          --label-col top1_class \
          --classes-csv data/genera_labels.csv  \
          --unknown-class skip \
          --size 640 \
          --overlap 0.2 \
          --tile-split-table data/tiles_split_city.txt  \
          --subtile-split-table data/subtiles_ids.txt   \
          --include-empty-tiles \
          --plain-tiff  
      ```
      **Note**: `yolo_train.py` expects plain TIFF images (Non-GeoTIFF). If your source imagery is stored as GeoTIFF, run the dataset builder with the --plain-tiff flag so that geospatial metadata is removed during chip generation.



   ii. Classification dataset for Genus Prediction
   
   ```bash
     python -m tree_genera_mapping.scripts.build_dataset cls \
      --tiles-gpkg data/lgl_bw_tiles.gpkg \
      --genus-labels-csv /greehill_genera.csv \
      --split-csv data/greehill_genera_split.csv \
      --images-dir cache/img_dir \
      --output-dir cache/patches_dir \
      --mode rgbih \
      --class-col genus \
      --tile-id-col tile_id \
      --labels-tile-col tile_id \
      --id-col tree_id \
      --crop-mode bbox \
      --bbox-col bbox \
      --patch-size 128 
   ```
   
3. Train Ultralytics YOLO model for Genera Detection
```bash
    python -m tree_genera_mapping.dl.detection.yolo_train.py \
        --run-dir cache/models \
        --data conf/data_genera.yaml \
        --model-size l \
        --batch 16 \
        --epochs 200 \
        --img-size 640 
```

More details on the script in `docs/README.md` 



## Model Checkpoints
| Task                              | Model Name | Modification    | URL Link                                                                             |
|-----------------------------------|------------|-----------------|--------------------------------------------------------------------------------------|
| Object Detection  (tree + genus) | YOLO11l    | 5-Channel Input | [yolo11l_tree_genus.pt](https://huggingface.co/solo2307/urban-tree-genera/tree/main) |
| Object Detection (tree)           | YOLO11l    | 5-Channel Input | [yolo11l_tree.pt](https://huggingface.co/solo2307/urban-tree-genera/tree/main)       |


## Dataset 
This repository accompanies:
- **Dataset**: `https://doi.org/10.11588/DATA/MKZPUY` 


If you use this code or workflow, please cite the accompanying paper.
See ```CITATION.cff``` for details.
