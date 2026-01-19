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

How to run the pre-trained YOLOv11l model 5CH imagery
1. Download LGL products to Generate TileDataset for selected tile ids:
```bash
python jobs/run_tile_dataset.py  \
 --tile-id 32_355_6048
```

2. Run pre-trained YOLOv11l model to detect and classify tree genus:
```bash
python jobs/run_inference.py --tiles-gpkg data/tiles.gpkg --images-dir cache/tiles_5ch --model-path models/pretrained_yolov11l_tree_genus.pth --output-dir cache/initial_inference
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
