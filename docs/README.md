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
                   predict_teacher, predict_yolo, compute_channel_stats
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

`data/subtiles_split.txt` contains 3,497 subtiles (2,934 train / 344 val / 219 test), each
inheriting its parent tile's partition. This count includes negative subtiles with no
annotated trees; the annotated subset is smaller.

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

Checks worth asserting in CI or a test script:

- both checkpoints load, and their first `Conv2d` reports `in_channels == 5`
- `model.names` matches `conf/data_genera.yaml` exactly
- every sample chip is 640 × 640 × 5 `uint8`
- every path and flag used in `README.md` and this file exists

---

## 5. Pipeline

### 5.1 Fetch LGL rasters and build 5-channel stacks

```bash
python -m tree_genera_mapping.scripts.fetch_tiles \
  --tiles-gpkg data/tiles.gpkg \
  --tile-ids data/subtiles_ids.txt \
  --tmp-root cache/tmp \
  --output-dir cache/img_dir \
  --mode RGBIH \
  --norm-height global
```

Other flags: `--keep-tmp`, `--overwrite`.

The acquisition step retries neighbouring tiles (y−1, x−1, x−1/y−1) on HTTP 404, so the
`dop_kachel` actually downloaded for a given subtile may differ from the one requested.

### 5.2 Weak crown labels (teacher input)

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
peaks, and watershed segmentation splits connected canopy into individual crown polygons,
which are then converted to bounding boxes.

Tuning flags: `--peak-threshold-abs-m`, `--min-distance-px`, `--min-canopy-area-px`,
`--gaussian-sigma`, `--median-size`, `--smooth-filter`, `--use-gradient`, `--no-fill-holes`,
`--write-masks`, `--mask-encoding`, and per-band overrides `--band-r/-g/-b/-nir/-h`.

**[VERIFY]** The code default for `--ndvi-thr` is 0.2, while the manuscript states
NDVI ≥ 0.3. Establish which value was used for the released labels and align code default,
manuscript and this document.

> **Human-in-the-loop step.** The generated boxes are reviewed in QGIS before use: obvious
> non-trees are removed (poles in railway areas, shrubs on bridges, hedges, vineyard
> structures, rooftop vegetation) and mis-sized or merged crowns are corrected. Merge the
> reviewed layers into a single GeoPackage before continuing.

### 5.3 Genus reference labels

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

### 5.4 Teacher training datasets

Detection patches:

```bash
python -m tree_genera_mapping.scripts.build_dataset det \
  --tiles-gpkg data/tiles.gpkg \
  --bboxes-gpkg cache/weak_tree_labels_reviewed.gpkg \
  --images-dir cache/img_dir \
  --output-dir cache/data \
  --mode rgbih \
  --tile-id-col tile_id \
  --label-col top1_class \
  --classes-csv data/genera_labels.csv \
  --unknown-class skip \
  --size 640 --overlap 0.2 \
  --tile-split-table data/tiles_split.txt \
  --subtile-split-table data/subtiles_ids.txt \
  --include-empty-tiles \
  --plain-tiff
```

Classification patches:

```bash
python -m tree_genera_mapping.scripts.build_dataset cls \
  --tiles-gpkg data/tiles.gpkg \
  --genus-labels-csv data/greehill_genera.csv \
  --split-csv data/greehill_genera_split.csv \
  --images-dir cache/img_dir \
  --output-dir cache/patches_dir \
  --mode rgbih \
  --class-col genus \
  --tile-id-col tile_id --labels-tile-col tile_id --id-col tree_id \
  --crop-mode bbox --bbox-col bbox \
  --patch-size 128
```

Always pass `--tile-split-table` (detection) or `--split-csv` (classification). The
`--val-frac` / `--test-frac` fallbacks perform a **random** split and were *not* used for
the released dataset; using them will not reproduce the published partition.

`yolo_train.py` reads plain (non-Geo) TIFF, so pass `--plain-tiff` when the source imagery
is GeoTIFF.

### 5.5 Teacher models

#### Crown detector — Faster R-CNN, one class

Trained on the reviewed weak crown boxes from the Mannheim subtiles. `--dataset-root` expects
the same `images/{split}` + `labels/{split}` layout produced by `build_dataset det`.

```bash
python -m tree_genera_mapping.dl.detection.tree_train \
  --dataset-root /path/to/yolo_rgbih_tile \
  --save-dir cache/models/frcnn_tree \
  --backbone resnet50 \
  --in-channels 5 \
  --num-classes 2 \
  --min-size 640 --max-size 640 \
  --epochs 30 \
  --batch-size 8 \
  --lr 5e-4 \
  --weight-decay 0.01 \
  --num-workers 8 \
  --device cuda \
  --pretrained-backbone \
  --seed 42
```

Full flag set: `--dataset-root` (required), `--save-dir`, `--backbone`, `--in-channels`,
`--num-classes`, `--min-size`, `--max-size`, `--epochs`, `--batch-size`, `--lr`,
`--weight-decay`, `--num-workers`, `--device`, `--norm-mean`, `--norm-std`,
`--pretrained-backbone`, `--resume`, `--seed`.

`--num-classes` counts background, so one-class tree detection uses `2`. The backbone is
built through `resnet_fpn_backbone`, and `conv1` is rebuilt for five input channels with the
pretrained RGB kernels copied into channels 0–2.

Evaluation:

```bash
python -m tree_genera_mapping.dl.detection.tree_eval \
  --ckpt cache/models/frcnn_tree/best.pt \
  --dataset-root /path/to/yolo_rgbih_tile \
  --out-dir cache/eval/frcnn_tree \
  --in-channels 5 --num-classes 2 \
  --score-thresh 0.3 \
  --save-preview --preview-n 12 --preview-cols 4
```

**[VERIFY]** `--pretrained-backbone` is opt-in (`store_true`), so it is **off** unless passed.
Confirm whether the released teacher detector was trained with it, and record the epoch count,
learning rate and `--min-size`/`--max-size` actually used, plus which subset of the 1,568
Mannheim subtiles was held out for validation (Table 2 reports 8,079 reference boxes).

#### Genus classifier — ResNet-101

Trained on 128 × 128 crown-centred patches produced by `build_dataset cls`.

```bash
python -m tree_genera_mapping.dl.classification.genus_train \
  --images-dir cache/patches_dir \
  --labels-csv data/genera_labels.csv \
  --out-dir cache/models/resnet101_5ch \
  --experiment image_only \
  --backbone resnet101 \
  --in-channels 5 \
  --img-size 128 \
  --epochs 50 \
  --batch-size 32 \
  --lr 5e-4 \
  --weight-decay 0.01 \
  --loss ce \
  --sampler none \
  --class-weights none \
  --norm-mode default \
  --num-workers 8 \
  --device cuda \
  --early-stop-monitor val_loss \
  --early-stop-patience 10 \
  --early-stop-min-delta 0.0 \
  --seed 42
```

Full flag set: `--images-dir` and `--out-dir` (required), `--labels-csv`, `--ndvi-csv`,
`--experiment`, `--backbone`, `--in-channels`, `--img-size`, `--epochs`, `--batch-size`,
`--lr`, `--weight-decay`, `--num-workers`, `--device`, `--loss`, `--focal-gamma`,
`--alpha`, `--alpha-mode`, `--sampler`, `--class-weights`, `--norm-mode`,
`--percentile-normalize`, `--norm-stats-json`, `--stats-max-samples`, `--val-frac`,
`--seed`, `--tensorboard`, `--early-stop-monitor`, `--early-stop-patience`,
`--early-stop-min-delta`.

To reproduce the architecture comparison, run the same command twice with
`--backbone resnet50` and `--backbone resnet101` and compare macro F1 on the validation
split.

Notes on the classifier:

- ResNet weights are ImageNet-pretrained (hardcoded `pretrained=True`). Because inputs have
  five channels, `conv1` is rebuilt with five input channels: pretrained RGB kernels are
  copied into channels 0–2, and the NIR and height channels are initialised from the
  pretrained red and green kernels respectively (`extra_channel_init="copy"`). All layers
  are then fine-tuned.
- Augmentation is geometric only: random horizontal flip, vertical flip and 90° rotation,
  each with probability 0.5, applied to the training split only.
- `--val-frac` is **inert** whenever the patch directory already contains `train/` and
  `val/` subfolders, which is the normal case. The split then comes from
  `greehill_genera_split.csv` via `build_dataset cls`.
- `--sampler weighted` together with `--class-weights invfreq` applies class rebalancing
  twice (once in the sampler, once in the loss) and degrades performance markedly. Do not
  combine them.
- `--focal-gamma`, `--alpha-mode` and `--alpha` have no effect unless `--loss focal`.
- Selection uses `--early-stop-monitor val_loss` by default, while the reported metric is
  macro F1. Consider monitoring macro F1 for consistency.

Evaluation:

```bash
python -m tree_genera_mapping.dl.classification.genus_eval \
  --images-dir cache/patches_dir \
  --out-dir cache/models/resnet101_5ch \
  --labels-csv data/genera_labels.csv \
  --backbone resnet101 --in-channels 5 --img-size 128
```

`genus_eval` evaluates the **val** split only; it expects `val/<class_name>/*.tif`.

### 5.6 Teacher ensemble inference (pseudo-labels)

```bash
python -m tree_genera_mapping.scripts.predict_teacher \
  --tile-dir cache/img_dir \
  --ckpt-paths cache/models/frcnn_tree/best.pt cache/models/resnet101_5ch/image_only_best.pt \
  --output-dir cache/pseudo_labels \
  --patch-size 640 --image-patch-size 128 \
  --stride 512 --conf 0.3 --iou 0.5
```

Faster R-CNN proposes crown boxes; each box's patch is resized to 128 × 128 and classified
by ResNet-101. A box and its class are kept together only when both confidences exceed
their thresholds.

> **Human-in-the-loop step.** Pseudo-labels are reviewed and corrected in QGIS, then saved
> as `cache/curated_pseudo_labels.gpkg`. Duplicate annotations arising from subtile overlap
> are removed before review. This curated layer is the training data for the student model
> and is released as part of the deposit.

### 5.7 Student model

Rebuild the detection dataset from the curated layer, then train:

```bash
python -m tree_genera_mapping.dl.detection.yolo_train \
  --run-dir cache/models/yolo \
  --run-name y11l_rgbih_genus_1024 \
  --data conf/data_genera_tile.yaml \
  --model-size l --num-bands 5 \
  --img-size 1024 --batch 16 --epochs 160 \
  --optimizer AdamW --lr0 0.0005 --lrf 0.01 --cos-lr \
  --weight-decay 0.01 --warmup-epochs 8 --patience 30 \
  --hsv-h 0.005 --fliplr 0.5 --flipud 0.3 --mosaic 1.0 --scale 0.4 --erasing 0.15
```

These values are read from the released checkpoints' stored `train_args` — see §6.

Evaluation:

```bash
python -m tree_genera_mapping.dl.detection.yolo_eval \
  --weights weights/yolo11l_tree_genus.pt \
  --data conf/data_genera_tile.yaml \
  --imgsz 1024 --conf 0.30 --iou 0.5 --save-cm
```

**[VERIFY]** `yolo_eval` exposes no split argument, so it evaluates the `val:` entry of the
data YAML. To reproduce the published test-partition figures, either add a `--split`
argument or point `val:` at `images/test`. As shipped, a user cannot reproduce the reported
numbers with this script — see §7.

### 5.8 Statewide inference

```bash
python -m tree_genera_mapping.scripts.predict_yolo \
  --tile_dir cache/img_dir \
  --ckpt_path weights/yolo11l_tree_genus.pt \
  --output_dir cache/predictions \
  --patch_size 1024 --stride 896 \
  --conf 0.30 --iou 0.5
```

Note that `predict_yolo` uses **underscore** flag names while every other script uses
hyphens. Harmonising this would be a welcome cleanup.

A global confidence threshold of 0.30 is applied. Detections assigned to *Acer* with
confidence in [0.30, 0.50) are retained as trees but reassigned to "Other Deciduous";
*Acer* detections at or above 0.50 keep their label. The released layer retains the original
class, the confidence, and the post-processed class so the rule can be reversed or
recalibrated.

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
| training wall time | 6,712 s | 1,932 s |
| checkpoint date | 2025-10-29 | 2025-10-24 |

**[VERIFY]** Two things in this table need reconciling with the manuscript.

1. The genus checkpoint was initialised from
   `.../y11l_rgbih_genus_1024_focal_loss/weights/last.pt`, i.e. the released model descends
   from a **focal-loss** run. The manuscript describes a binary cross-entropy classification
   loss and does not mention focal loss anywhere. Document the actual loss configuration.
2. The genus run's `data` field points to `conf/data.yaml`, which is not in the repository
   (`conf/` contains `data_genera.yaml`, `data_tree.yaml`, `data_visdrone.yaml`). Ship the
   exact YAML that was used, or confirm it is identical to `data_genera.yaml`.

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

1. `--ndvi-thr` default 0.2 versus the manuscript's 0.3.
2. Height-channel encoding of the sample chips (§3.1).
3. `training_class` ordering in `greehill_genera.csv` versus the model's class order (§3.2).
4. `conf/data.yaml` referenced by the released genus checkpoint is absent from the repo.
5. Focal loss in the released genus model is undocumented in the manuscript.
6. `yolo_eval` has no split selector.
7. `predict_yolo` uses underscore flags; all other scripts use hyphens.
8. `genus_eval` defaults to `conf/genus_labels.csv`, which does not exist — the label file is
   `data/genera_labels.csv`.

10. `scripts/finalize_results.py`, referenced in older documentation, does not exist. Either
    add it or document how predictions are merged and exported.
11. The per-class reference counts published in the Data Descriptor's class-specific
    performance table (total 9,663) do not match the deposited test partition (total 9,042;
    see §3.4). Since the dataset is released, this is checkable by any reader. Recompute the
    table from the released test split or state which dataset version produced it.
12. `yolo_rgbih_tile/` ships no data YAML, so the released dataset cannot be used with
    `yolo_train` or `yolo_eval` without the user writing one (§3.4). Include it in the deposit.
