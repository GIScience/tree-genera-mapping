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

How to run the pre-trained YOLOv11l model 5CH imagery
1. DEMO with Jupyter NOtebook ...

How to run the Genera Mapping scripts
1. Download LGL products to Generate TileDataset for selected tile ids:
```bash
python tree_genera_mapping/scripts/fetch_tiles.py  \
 --tile-id 32_355_6048
```

2. Run pre-trained YOLOv11l model to detect and classify tree genus:
```bash
python tree_genera_mapping/scripts/predict_yolo.py --tiles-gpkg data/tiles.gpkg --images-dir cache/tiles_5ch --model-path cache/weights/yolov11l_tree_genus.pth --output-dir cache/initial_inference
```

## Train Model
1. Data Preparation

   i. Prepare detection dataset for Genera Mapping
      ```bash
      python -m tree_genera_mapping.scripts.build_dataset det \
          --tiles-gpkg /data/lgl_bw_tiles.gpkg \
          --bboxes-gpkg path/to/pseudo_labels.gpkg \
          --images-dir /path/to/tiles_rgbih/ \
          --output-dir cache/ \
          --mode rgbih \
          --tile-id-col tile_id \
          --label-col top1_class \
          --classes-csv data/genera_labels.csv  \
          --unknown-class skip \
          --size 640 \
          --overlap 0.2 \
          --tile-split-table data/tiles_split.txt  \
          --subtile-split-table data/subtiles_split.txt   \
          --include-empty-tiles \
          --plain-tiff \ # yolo_train.py only reads TIFF (Non-GeoTIFF)
      ```
      **Note**: `yolo_train.py` expects plain TIFF images (Non-GeoTIFF). If your source imagery is stored as GeoTIFF, run the dataset builder with the --plain-tiff flag so that geospatial metadata is removed during chip generation.



   ii. Classification dataset for Genus Prediction

   ```bash
    python -m tree_genera_mapping.scripts.build_dataset cls \
     --tiles-gpkg data/lgl_bw_tiles.gpkg \
     --genus-labels-csv path/to/GreeHill_TreesGenus.csv \
     --class-col genus \
     --images-dir /path/to/tiles_rgbih/ \
     --output-dir cache/genus_patches \
     --mode rgbih \
     --patch-size 128 \
     --labels-tile-col tile_id \
     --tile-split-table data/tiles_split.txt
     --crop-mode fixed \ # bbox or fixed 
     --bbox-col bbox \ # only for bbox crop mode
     --x-col X \ # only for fixed crop mode
     --y-col Y \ # only for fixed crop mode 
   ```
   
3. Run code
```bash
    python -m tree_genera_mapping.dl.detection.yolo_train.py
```

## Model Checkpoints
| Task                              | Model Name | Modification    | URL Link                                                                             |
|-----------------------------------|------------|-----------------|--------------------------------------------------------------------------------------|
| Object Detection  (tree + genus) | YOLO11l    | 5-Channel Input | [yolo11l_tree_genus.pt](https://huggingface.co/solo2307/urban-tree-genera/tree/main) |
| Object Detection (tree)           | YOLO11l    | 5-Channel Input | [yolo11l_tree.pt](https://huggingface.co/solo2307/urban-tree-genera/tree/main)       |


## Dataset & Paper
This repository accompanies:
- **Dataset**: {add}
- **Paper**: {add}

If you use this code or workflow, please cite the accompanying paper.
See ```CITATION.cff``` for details.
