# Tree Genera Mapping in Baden-Württemberg, Germany

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL3.0-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
> A research pipeline for large-scale urban tree crown detection and tree genus mapping using very-high-resolution multispectral aerial imagery and LiDAR data.

> UrbanGreen associated with the mapping of urban green spaces from multispectral aerial imagery and LiDAR data provided by the **LGL Open GeoData-Portal** https://www.lgl-bw.de/Produkte/Open-Data/ 
> 
![urban-green-spaces.jpeg](misc%2Furban-green-spaces.jpeg)

##  ✨Overview
UrbanGreenSpaces provides an end-to-end, research-oriented workflow to:
- Download and preprocess **LGL Open GeoData** (multi-spectral orthophotos & nDSM).  
- Build **5-channel raster tiles** (RGB + NIR + normalized height). 
- Perform **tree crown delineation and detection**
- Predict **tree genera using deep learning**
- Apply a **teacher–student** learning strategy with human-in-the-loop **curation** 
- Scale inference to **statewide coverage**
- Export results as **GeoPackage** for GIS analysis

The code accompanies an upcoming **open dataset and scientific publication** on regional-scale tree genera mapping in Baden-Württemberg.

## 🚀 Quickstart: 
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
## 🧠 Method Summary
![greenspace_workflow.png](misc%2Fgreenspace_workflow.png)

The pipeline follows a semi-supervised teacher–student paradigm:

Data acquisition & preprocessing

Co-registration of multispectral imagery and LiDAR

Spatial harmonization and tiling

Height normalization using CHM

Initial tree delineation (heuristic, non-learning)

NDVI and height thresholding

Morphological filtering

Extraction of candidate tree crowns
(NDVI and height are used only at this stage)

Teacher model ensemble (two-stage)

Tree detection model trained on weak delineation labels

Tree genus classifier trained on a labeled subset (GreeHill)

Ensemble inference to generate pseudo-labels

Human-in-the-loop curation

Visual validation and correction of detections

Removal of low-confidence predictions

Creation of a curated pseudo-labeled dataset
(treated as ground truth)

Student model (single-stage YOLOv11)

Joint tree detection and genus classification

Optimized for scalable inference

Scalable statewide inference

GPU-parallel tiled inference

Spatial merging of predictions

Generation of regional tree genera maps

## 📂 Repository Structure
```pgsql
tree-genera-mapping/
│
├── research_code/
│ ├── ops/ # Downloaders, data handling
│ ├── pre_processing/ # Tiling, raster preparation
│ └── jobs/ # CLI entry points (dataset_job.py, inference_job.py)
│
├── environment.yaml # Reproducible conda environment
└── README.md
```

## 🏗️ Data
 Aerial imagery and LiDAR data are provided by [LGL Baden-Württemberg Open GeoData](https://www.lgl-bw.de/Produkte/Open-Data/) under the [Datenlizenz Deutschland – Namensnennung – Version 2.0](https://www.govdata.de/dl-de/by-2-0).

> Samples from the dataset are in the `data/sample` folder.

> Tiles: Processed into 5000 × 5000 px patches (≈ 1 × 1 km²) 

Example tile structure:
```pgsql
32_457_5428_2/
├── rgbih_32_457_5428.tif
└── rgbih_32_457_5428.height.json
```
## 📊 Model Checkpoints
| Task                 | Model Name   | Modification    | URL Link                                                           |
|----------------------|--------------|-----------------|--------------------------------------------------------------------|
| Object Detection     | YOLO11l      | 5-Channel Input |    |
| Object Detection     | YOLO11l_CBAM | 5-Channel Input + CBAM | |
| Object Detection     | FasterRCNN   | 5-Channel Input | |
| Image Classification | ResNet/YOLO  | 5-Channel Input | |

## 📊 Outputs

- Tree crown bounding boxes
- Tree genus predictions (10 genera)
- Canopy height and crown diameter estimates
- GIS-ready outputs (GeoPackage)

## 🔬Dataset & Paper
This repository accompanies:
- **Dataset**: {add}
- **Paper**: {add}

## 📖 Citation
If you use this code or workflow, please cite the accompanying paper:
``` bibtex
@article{grinblat2025urbangreenspaces,
  title   = {Tree Genera Mapping of Baden-Württemberg Using Multispectral Imagery and LiDAR},
  author  = {Grinblat, Yulia, Fulman, Nir},
  journal = {Scientific Data},
  year    = {2026}
}
```
See ```CITATION.cff``` for details.

## 📜 License
This project is licensed under the AGPL-3.0 License.
See the ```LICENSE``` file for details.
