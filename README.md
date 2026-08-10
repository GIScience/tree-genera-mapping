# Urban Tree Genera Mapping in Baden-Württemberg, Germany

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL3.0-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
> A research pipeline for large-scale urban tree crown detection and tree genus mapping using very-high-resolution multispectral aerial imagery and LiDAR data.

> The multispectral aerial imagery and LiDAR-products provided by the **LGL Open GeoData-Portal** https://www.lgl-bw.de/Produkte/Open-Data/ 
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

The same checkpoints are archived with the dataset at https://doi.org/10.11588/DATA/MKZPUY under `weights/`.

## Usage:
### Demo of the pretrained model 
`notebooks/01_demo_inference.ipynb` provides a step-by-step demonstration of the genera predictions over `data/samples` 5-stack images. 

### How to run the Genera Mapping scripts
1. Download LGL products to Generate TileDataset for selected tile ids:
```bash
python tree_genera_mapping/scripts/fetch_tiles.py \
  --tiles-gpkg data/tiles.gpkg \
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
   i. Detection dataset:
      ```bash
      python -m tree_genera_mapping.scripts.build_dataset det \
          --tiles-gpkg data/tiles.gpkg \
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

   ii. Classification dataset (crown-centred genus patches):
   
   ```bash
     python -m tree_genera_mapping.scripts.build_dataset cls \
      --tiles-gpkg data/tiles.gpkg \
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
2. Teacher-Ensemble
   i. Train Faster R-CNN model for Tree Detection
```bash
python -m tree_genera_mapping.dl.detection.tree_train \
       --dataset-root cache/data \
       --save-dir cache/models/frcnn_tree \
       --backbone resnet101 --in-channels 5 --num-classes 2 \
       --epochs 100 --batch-size 8 --lr 5e-4 --weight-decay 0.01 \
       --pretrained-backbone --seed 42
```
   ii. Train Genus Classifier(ResNet)
```bash
python -m tree_genera_mapping.dl.classification.genus_train \
       --images-dir cache/patches_dir \
       --labels-csv data/genera_labels.csv \
       --out-dir cache/models/resnet101_5ch \
       --experiment image_only \
       --backbone resnet101 --in-channels 5 --img-size 128 \
       --epochs 50 --batch-size 32 --lr 5e-4 --weight-decay 0.01 \
       --loss ce --early-stop-monitor val_loss --early-stop-patience 10 --seed 42
```
   iii. *Teacher-ensemble inference to generate pseudo-labels*:
```bash
python -m tree_genera_mapping.scripts.predict_teacher \
       --tile-dir cache/img_dir \
       --ckpt-paths cache/models/frcnn_tree/best_model.pth \
                    cache/models/resnet101_5ch/image_only_best.pt \
       --output-dir cache/pseudo_labels \
       --patch-size 640 \
       --image-patch-size 128 \
       --stride 512 \
       --det-conf 0.30 \
       --cls-conf 0.30 \
       --iou 0.50
 ```

`--ckpt-paths` takes exactly two checkpoints in this order: the Faster R-CNN
crown detector followed by the ResNet genus classifier. The classifier crops
are extracted directly from each in-memory RGBIH tile tensor, so no additional
classifier image directory is required. Omit `--tile-id` to process every
`rgbih_*.tif` file under `--tile-dir`, or pass a single tile such as
`--tile-id 32_413_5320`. The script writes one
`teacher_rgbih_<tile_id>.gpkg` file per tile with `top1_class`, detector and
classifier confidence, and per-class probabilities.

Pseudo-labels are then reviewed and corrected in QGIS before use — see `docs/README.md`.

3. Train Ultralytics YOLO model for Genera Detection
```bash
python -m tree_genera_mapping.dl.detection.yolo_train \
       --run-dir cache/models \
       --run-name y11l_rgbih_genus_1024 \
       --data conf/data_genera.yaml \
       --model-size l --num-bands 5 \
       --img-size 1024 --batch 16 --epochs 160 \
       --optimizer AdamW --lr0 0.0005 --lrf 0.01 --cos-lr \
       --weight-decay 0.01 --warmup-epochs 8 --patience 30
```
4. Evaluation

```bash
python -m tree_genera_mapping.dl.detection.yolo_eval \
       --weights cache/weights/yolo11l_tree_genus.pt \
       --data conf/data_genera.yaml \
       --imgsz 1024 --conf 0.30 --iou 0.5 --save-cm
```

Full pipeline documentation, data conventions and reproducibility notes are in `docs/README.md`.

## Model Checkpoints
| Task                              | Model Name | Modification    | URL Link                                                                             |
|-----------------------------------|------------|-----------------|--------------------------------------------------------------------------------------|
| Object Detection  (tree + genus) | YOLO11l    | 5-Channel Input | [yolo11l_tree_genus.pt](https://huggingface.co/solo2307/urban-tree-genera/tree/main) |
| Object Detection (tree)           | YOLO11l    | 5-Channel Input | [yolo11l_tree.pt](https://huggingface.co/solo2307/urban-tree-genera/tree/main)       |

Both were trained at 1024 × 1024 on five-channel stacks. The genus model covers the ten classes listed in `conf/data_genera.yaml` and `conf/data_tree.yaml`; the one-class model is an auxiliary benchmark and was not used to assign genus labels in the released inventory.

## Dataset 
This repository accompanies:
- **Dataset**: `https://doi.org/10.11588/DATA/MKZPUY` 


If you use this code or workflow, please cite the accompanying paper.
See ```CITATION.cff``` for details.
