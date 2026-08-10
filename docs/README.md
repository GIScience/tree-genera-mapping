# Development and Reproducibility Guide

This document describes the full workflow of the `tree-genera-mapping` pipeline: how the
released dataset was produced, how to verify the deposit, and how to re-run each stage.

Companion resources:

- Dataset: <https://doi.org/10.11588/DATA/MKZPUY>
- Quickstart: see the repository `README.md`

---

## 1. Environment

```bash
conda env create -f environment.yaml     # or environment_gpu.yaml for CUDA
conda activate map-tree-genera
python -m ipykernel install --user --name map-tree-genera --display-name "Python (tree-genera)"
```

All commands are run from the repository root and invoke modules with `python -m`, e.g.

```bash
python -m tree_genera_mapping.scripts.fetch_tiles --help
```

Intermediate outputs are written to `cache/`, which is git-ignored.

---

## 2. Repository layout

```
tree_genera_mapping/
  scripts/         fetch_tiles, segment_crowns, build_dataset,
                   predict_teacher, predict_yolo, compute_channel_stats, 
                   build_tof_roi, select_inference_tiles,
                   compute_agg_indicators, finalize_results
  preprocess/      genus_labels, height_model, detection_dataset, utils
  dl/
    detection/     tree_train, tree_eval        (Faster R-CNN teacher)
                   yolo_train, yolo_eval        (YOLO11l student)
    classification/ genus_train, genus_eval     (ResNet teacher)
    metrics, losses, plots, utils
conf/              data_genera.yaml, data_tree.yaml
data/              tiles.gpkg, genera_labels.csv, greehill_genera_split.csv,
                   tiles_split.txt, subtiles_split.txt, subtiles_ids.txt,
                   tiles_split_city.txt, samples/
```

---

## 3. Data conventions

### 3.1 Raster stacks

Model inputs are 5-channel `uint8` GeoTIFF/TIFF stacks at 0.2 m ground resolution:

| Band | Content             | Source            |
|------|---------------------|-------------------|
| 0 | Red                 | LGL DOP20         |
| 1 | Green               | LGL DOP20         |
| 2 | Blue                | LGL DOP20         |
| 3 | NIR                 | LGL DOP20         |
| 4 | Above-ground height | LGL nDOM (nDSM B) |

The height channel is encoded with fixed global bounds of 0–80 m, so
`height_m = value * 80 / 255`.

The four sample chips are 640 × 640 × 5, `uint8`.

### 3.2 Class scheme

Ten classes, in this order. This ordering is authoritative: it is identical in
`conf/data_genera.yaml`, `data/genera_labels.csv`, and the `names` dictionary stored inside
both released checkpoints.

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | *Acer* | 5 | Other Deciduous |
| 1 | *Aesculus* | 6 | *Platanus* |
| 2 | *Carpinus* | 7 | *Prunus* |
| 3 | Coniferous | 8 | *Quercus* |
| 4 | *Fagus* | 9 | *Tilia* |

### 3.3 Partitioning

A **single spatially blocked partition at the 1 km × 1 km parent-tile level** is used
throughout the whole workflow — teacher stage and student stage alike.

| Partition | Parent tiles | Reference trees |
|-----------|--------------|-----------------|
| train | 111 | 37,696 |
| val | 14 | 2,719 |
| test | 18 | 1,779 |
| **total** | **143** | **42,194** |

Verified: joining `data/greehill_genera_split.csv` to the tile assignment yields zero tiles
containing trees from more than one partition, and the resulting per-tile assignment is
identical to `data/tiles_split.txt`. No tile contributes data to more than one partition at
any stage.

`data/subtiles_split.txt` contains 2,965 unique subtiles (2,482 train / 297 val / 186 test),
each inheriting its parent tile's partition. Of these, 148 are negative subtiles with no
annotated trees.

`data/tiles_split_city.txt` holds the alternative city-to-city assignment (Mannheim +
Karlsruhe for training, Freiburg held out).


### 3.4 The built detection dataset

The dataset the student model was trained on is a standard Ultralytics layout:

```
yolo_rgbih_tile/
  images/{train,val,test}/rgbih_<tile_id>_<subtile>.tif    # 5-channel uint8
  labels/{train,val,test}/rgbih_<tile_id>_<subtile>.txt    # YOLO format
```

Verified contents:

| Split | Subtiles | Empty (negative) | Annotated instances |
|-------|----------|------------------|---------------------|
| train | 2,482 | 103 | 136,208 |
| val | 297 | 37 | 12,836 |
| test | 186 | 8 | 9,042 |
| **total** | **2,965** | **148** | **158,086** |

Negative subtiles are present as empty `.txt` files, not as missing files.

Per-class instance counts (class ids as in §3.2):

| Class | train | val | test |
|-------|-------|-----|------|
| *Acer* | 29,611 | 2,979 | 1,653 |
| *Aesculus* | 7,315 | 1,177 | 675 |
| *Carpinus* | 1,884 | 74 | 150 |
| Coniferous | 11,057 | 383 | 482 |
| *Fagus* | 2,606 | 11 | 21 |
| Other Deciduous | 43,962 | 4,234 | 3,666 |
| *Platanus* | 10,565 | 1,064 | 848 |
| *Prunus* | 2,524 | 199 | 98 |
| *Quercus* | 10,735 | 767 | 629 |
| *Tilia* | 15,949 | 1,948 | 820 |

Because the partition is spatially blocked, rare genera are very thinly represented in the
held-out splits — *Fagus* has 21 test instances and *Carpinus* 150. Per-class precision and
recall for these classes carry wide uncertainty and should be interpreted accordingly.

The folder ships no data YAML. Create one alongside it:

```yaml
# conf/data_genera.yaml
path: cache/yolo_rgbih_tile
train: images/train
val: images/val 
test: images/test
# Number of multispectral image channels
channels: 5
# Classes
nc: 10
names:
    0: Acer
    1: Aesculus
    2: Carpinus
    3: Coniferous
    4: Fagus
    5: Other Deciduous
    6: Platanus
    7: Prunus
    8: Quercus
    9: Tilia
```

---

## 4. Fast verification of the deposit

This is the cheapest end-to-end check and needs no LGL download. It uses the four sample
chips and the released weights.

```python
import tifffile
from ultralytics import YOLO

model = YOLO("weights/yolo11l_tree_genus.pt")
img = tifffile.imread("data/samples/sample_r3_c3.tif")   # (640, 640, 5) uint8
res = model.predict(source=img, imgsz=1024, conf=0.30, verbose=False)
print(len(res[0].boxes), res[0].names)
```

**Verified working.** On `sample_r3_c3.tif` this returns 22 detections with confidences
between 0.326 and 0.926, distributed as Other Deciduous 11, *Tilia* 8, *Acer* 1,
Coniferous 1, *Quercus* 1.
---
## 5. Pipeline

### 5.1 Non-forest mapping domain

Built once from two Basis-DLM layers: the state boundary with all forest erased.

```bash
python -m tree_genera_mapping.scripts.build_tof_roi \
  --boundary-shp data/geb01_f.shp \
  --forest-shp data/veg02_f.shp \
  --output cache/bw_nonforest_roi.gpkg
```

Verified figures for Baden-Württemberg:

```
state boundary   AX_Gebiet_Bundesland    35,766.43 km²   (1 feature)
forest           AX_Wald                 14,059.78 km²   (481,580 polygons)
                                       = 21,706.65 km²
measured ROI                              21,706.69 km²
```

`veg02_f.shp` as distributed contains only `AX_Wald`, so no attribute filter is strictly
needed — but `build_tof_roi.py` filters on `OBJART_TXT` explicitly and logs the values it
found, so a full `VEG02_F` export containing agriculture would be caught rather than
silently erasing most of the state.

The union of `AX_KommunalesGebiet` polygons is 35,721.63 km², i.e. 44.80 km² smaller than
the state polygon (water not assigned to any municipality). That accounts exactly for the
difference between the ROI and the municipality-level mapped-area total of 21,661.89 km².

### 5.2 Tile selection and acquisition

```bash
python -m tree_genera_mapping.scripts.select_inference_tiles \
  --tiles-gpkg data/tiles.gpkg \
  --roi-gpkg cache/bw_nonforest_roi.gpkg \
  --output data/inference_tiles.txt

python -m tree_genera_mapping.scripts.fetch_tiles \
  --tiles-gpkg data/tiles.gpkg \
  --tile-ids data/inference_tiles.txt \
  --tmp-root cache/tmp \
  --output-dir cache/img_dir \
  --mode RGBIH \
  --norm-height local
```

Other `fetch_tiles` flags: `--keep-tmp`, `--overwrite`.

### 5.3 Weak crown labels (teacher input)

```bash
python -m tree_genera_mapping.scripts.segment_crowns \
  --img-dir cache/img_dir \
  --output-dir cache/weak_tree_labels \
  --mode rgbih \
  --ndvi-thr 0.3 \
  --height-thr 2.0 \
  --write-pols
```

NDVI and height masks are combined, local maxima in the height model are taken as crown
peaks, and watershed segmentation splits connected canopy into individual crowns, which
are converted to bounding boxes.

Tuning flags: `--peak-threshold-abs-m`, `--min-distance-px`, `--min-canopy-area-px`,
`--gaussian-sigma`, `--median-size`, `--smooth-filter`, `--use-gradient`,
`--no-fill-holes`, `--write-masks`, `--mask-encoding`, and per-band overrides
`--band-r/-g/-b/-nir/-h`.

The reproducibility command and the code default both use `--ndvi-thr 0.3`, matching
the NDVI ≥ 0.3 threshold in the Data Descriptor.

> **Human-in-the-loop step.** Boxes are reviewed in QGIS: obvious non-trees are removed
> (poles in railway areas, shrubs on bridges, hedges, vineyard structures, rooftop
> vegetation) and mis-sized or merged crowns are corrected. Merge the reviewed layers into
> a single GeoPackage before continuing.

### 5.4 Genus reference labels

`greehill_genera.csv` is distributed through the
[heiDATA dataset](https://doi.org/10.11588/DATA/MKZPUY), not through this Git
repository. Download it and save it as `data/greehill_genera.csv` before running
the following command.

```bash
python -m tree_genera_mapping.preprocess.genus_labels \
  --trees data/greehill_genera.csv \
  --labels data/genera_labels.csv \
  --output cache/genus_labels.gpkg \
  --make-bbox
```

Other flags: `--genus-col`, `--canopy-col`, `--id-col`.

Only the aggregated per-tree attributes (position, crown width, height, genus) are
redistributable. The underlying terrestrial LiDAR point clouds are not part of the deposit
and cannot be republished.

### 5.5 Teacher training datasets

See the `build_dataset det` and `build_dataset cls` invocations in the root
[`README.md`](../README.md). Two points bear repeating: always pass `--tile-split-table`
(detection) or `--split-csv` (classification), because the `--val-frac` fallback performs a
random split that does not reproduce the published partition; and pass `--plain-tiff` when
the source imagery is GeoTIFF, since `yolo_train.py` reads plain TIFF only.

### 5.6 Teacher models

Both teacher components are initialised from ImageNet-pretrained ResNet backbones. Because
the inputs have five channels, the first convolutional layer is rebuilt with five input
channels: the pretrained RGB kernels are copied into channels 0–2, and the NIR and height
channels are initialised from the pretrained red and green kernels respectively
(`extra_channel_init="copy"`). All layers are then fine-tuned.

**Crown detector — Faster R-CNN, one class.** Flags: `--dataset-root` (required),
`--save-dir`, `--backbone`, `--in-channels`, `--num-classes`, `--min-size`, `--max-size`,
`--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--num-workers`, `--device`,
`--norm-mean`, `--norm-std`, `--pretrained-backbone`, `--resume`, `--seed`.
`--num-classes` counts background, so one-class detection uses `2`.

`--pretrained-backbone` is opt-in (`store_true`), so pass it explicitly when training
with ImageNet-pretrained weights. Record the epoch count, learning rate,
`--min-size`/`--max-size`, random seed, and held-out Mannheim subtiles with each training
run so that its detector checkpoint can be reproduced.

**Genus classifier — ResNet-101.** Notes that matter in practice:

- Augmentation is geometric only: random horizontal flip, vertical flip and 90° rotation,
  each at probability 0.5, applied to the training split only.
- `--val-frac` is **inert** whenever the patch directory already contains `train/` and
  `val/` subfolders, which is the normal case. The split then comes from
  `greehill_genera_split.csv` via `build_dataset cls`.
- `--sampler weighted` together with `--class-weights invfreq` applies class rebalancing
  twice — once in the sampler, once in the loss — and degrades performance markedly. Do
  not combine them.
- `--focal-gamma`, `--alpha-mode` and `--alpha` have no effect unless `--loss focal`.
- Selection defaults to `--early-stop-monitor val_loss`, while the reported metric is
  macro F1; consider monitoring macro F1 for consistency.

Evaluation: `genus_eval.py` scores the **val** split only and expects
`val/<class_name>/*.tif`. Its label mapping defaults to `data/genera_labels.csv`; use
`--labels-csv` only to override that file.

### 5.7 Teacher-ensemble inference and curation

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
crown detector followed by the ResNet genus classifier. Faster R-CNN proposes
crown boxes over overlapping windows and global non-maximum suppression removes
duplicates. Each retained box is cropped directly from the same in-memory
five-channel RGBIH tile tensor, resized to 128 × 128, normalized with the
classifier checkpoint statistics, and classified by ResNet-101. No separate
classifier image path is required.

Omit `--tile-id` to process every `rgbih_*.tif` file under `--tile-dir`, or pass
one tile such as `--tile-id 32_413_5320`. The script writes one
`teacher_rgbih_<tile_id>.gpkg` per input tile. Each output contains
`top1_class`, `top1_class_id`, detector and classifier confidence, all per-class
probabilities, and georeferenced crown boxes. A prediction is retained only
when its detector confidence is at least `--det-conf` and its top-one classifier
probability is at least `--cls-conf`.

> **Human-in-the-loop step.** Pseudo-labels are reviewed and corrected in QGIS, then saved
> as `cache/curated_pseudo_labels.gpkg`. Duplicate annotations arising from subtile overlap
> are removed before review. This curated layer trains the student model and is released as
> part of the deposit.

### 5.8 Student model

See the root [`README.md`](../README.md) for the training and evaluation commands. The
hyperparameters there are read from the released checkpoints — see §6.

`yolo_eval.py` evaluates the `val:` entry of the data YAML and does not expose a separate
split argument. To evaluate the test partition, use a separate evaluation YAML whose
`val:` entry points to `images/test`.

### 5.9 Statewide inference

```bash
python -m tree_genera_mapping.scripts.predict_yolo \
  --tile-dir cache/img_dir \
  --ckpt-path cache/weights/yolo11l_tree_genus.pt \
  --output-dir cache/predictions \
  --patch-size 1024 --stride 896 --imgsz 1024 \
  --conf 0.30 --iou 0.4
```

Omitting `--tile-id` processes every TIFF under `--tile-dir`. Hyphenated flags are
preferred; the former underscore spellings remain available as compatibility aliases.
Each input produces `tile_<tile_id>.gpkg` with the column schema consumed by
`finalize_results.py`.

Two behaviours of this script are worth recording. `canopy_diameter` is computed as
`sqrt(dx² + dy²)`, the bounding-box **diagonal** — despite the inline comment saying "mean
of width, height" — so it exceeds the box side by up to a factor of √2. And the call to
`merge_subtile_predictions` is **commented out**, so as shipped the script emits one
detection per sliding window rather than one per tree. With `patch_size` 640 and `stride`
512 that is substantial duplication. Deduplication is performed once, in
`finalize_results.py`, so the predictor intentionally retains all window detections.

### 5.10 Post-processing to the released inventory

```bash
python -m tree_genera_mapping.scripts.finalize_results \
  --pred-dir cache/predictions \
  --domain-gpkg cache/bw_nonforest_roi.gpkg \
  --settlement-shp data/sie01_f.shp \
  --output cache/trees_bw.gpkg --layer trees \
  --dedup-iou 0.5 \
  --acer-threshold 0.50 \
  --resume
```
### 5.11 Aggregated indicators

```bash
python -m tree_genera_mapping.scripts.compute_agg_indicators \
  --trees-gpkg cache/trees_bw.gpkg --trees-layer trees \
  --grid-gpkg data/grid_pop_bw_100.gpkg --population-col Einwohner \
  --municipalities-shp data/geb01_f.shp \
  --roi-gpkg cache/bw_nonforest_roi.gpkg \
  --out-grid cache/bw_tree_indicators_grid.gpkg \
  --out-municipalities cache/bw_tree_indicators.gpkg \
  --richness-empty nan
```

Definitions:

- **normalized Shannon** = H / ln(C) with C = 10, the number of classes in the scheme, so
  values are comparable between cells regardless of how many genera are present.
- **class richness** = number of distinct classes present.
- **mapped area** = area of the non-forest ROI inside the unit, not the unit's total area.
- **tree density** = trees / mapped area, in km⁻². It is area-based and has no population
  dependency.
- **municipality diversity** = mean of the per-cell normalized Shannon values.

Two things are made explicit rather than implied. `--richness-empty {zero,nan}` decides
whether zero-tree cells enter the municipality mean as 0.0 or are excluded; the two give
different results. And cells are attributed to municipalities by **representative point**,
so every cell lands in exactly one municipality — a `within` test on cell polygons drops
cells straddling a boundary along with their trees, while their area still counts in the
denominator, which biases density downward for municipalities with a high
perimeter-to-area ratio.


---

## 6. Released model configurations

Read directly from the deposited checkpoints (`train_args`, Ultralytics 8.3.185):

| | `yolo11l_tree_genus.pt` | `yolo11l_tree.pt` |
|---|---|---|
| classes | 10 | 1 (`tree`) |
| input channels | 5 | 5 |
| image size | 1024 | 1024 |
| epochs configured | 160 | 150 |
| epochs completed | 160 | 135 (early stop, patience 20) |
| batch | 16 | 16 |
| optimizer | AdamW | AdamW |
| lr0 / lrf | 5e-4 / 0.01 | 5e-4 / 0.01 |
| cosine LR | yes | yes |
| warmup epochs | 8 | 5 |
| weight decay | 0.01 | 0.01 |
| patience | 30 | 20 |
| seed | 0 | 0 |
| pretrained | yes | yes |
| augmentation | hsv_h 0.005, fliplr 0.5, flipud 0.3, mosaic 1.0, scale 0.4, erasing 0.15, degrees 0.0 | same but scale 0.5, `auto_augment=randaugment` |


---

## 7. Metrics: which partition is which

The deposited checkpoints carry their own metrics, recorded on the **validation** partition
during training:

| | precision | recall | mAP@0.5 | mAP@0.5–0.95 |
|---|---|---|---|---|
| `yolo11l_tree_genus.pt` | 0.738 | 0.556 | 0.598 | 0.421 |
| `yolo11l_tree.pt` | 0.807 | 0.703 | 0.761 | 0.519 |

The figures reported in the Data Descriptor are computed on the **test** partition and are
consistently lower, because the 18 test tiles are harder than the 14 validation tiles:

| | precision | recall | mAP@0.5 | mAP@0.5–0.95 |
|---|---|---|---|---|
| ten-class genus (test) | 0.465 | 0.385 | 0.382 | 0.210 |
| one-class tree (test) | 0.684 | 0.632 | 0.647 | 0.345 |

Anyone inspecting the checkpoints will see the first table and the paper's the second, so
the distinction must be stated in `weights/README.md` as well as here.

Two consequences:

- Configuration comparisons and final reporting both used the test partition, so no
  independent selection partition was held back. The validation-versus-test gap above
  indicates the reported values are conservative rather than optimistic, but the arrangement
  should be disclosed.
- `yolo_eval` as shipped evaluates the validation split, so it reproduces the first table,
  not the second. Fix this before claiming the deposit reproduces the published numbers.

Determinism: seeds are set (`seed 0` for the YOLO runs, `42` for the classifier), but cuDNN
determinism flags are not, so retraining reproduces results to a tolerance rather than
bit-exactly.

---

## 8. What is reproducible

| Scope | Feasibility |
|---|---|
| Inference on the shipped sample chips with released weights | **Exact** — verified, seconds, no download |
| Evaluation on the released test subtiles with released weights | **Exact** once `yolo_eval` can target the test split |
| Retraining the student from the released curated annotations | **Approximate** — GPU nondeterminism; ~2 h on one GPU |
| Retraining the teachers | **Approximate** — requires regenerating patches |
| Regenerating the statewide inventory | **Not practical for readers** — requires the full LGL download and inference over 36,653 tiles |
| Reproducing the human curation | **Not reproducible** — depends on annotator judgement; the curated layer is released instead |

---

## 9. Open items

1. `training_class` ordering in `greehill_genera.csv` versus the model's class order (§3.2).
2. `conf/data.yaml` referenced by the released genus checkpoint is absent from the repo.
3. Focal loss in the released genus model is undocumented in the manuscript.
4. `yolo_eval` has no split selector.
5. `yolo_rgbih_tile/` ships no data YAML, so the released dataset cannot be used with
    `yolo_train` or `yolo_eval` without the user writing one (§3.4). Include it in the deposit.
