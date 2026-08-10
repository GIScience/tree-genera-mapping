# Data

This directory contains the **reference spatial grid**, the **class scheme** and the
**partition tables** used throughout the project. Everything here is small enough to be
version-controlled so that the published partitions are reproducible; bulk imagery and
model outputs are written to `cache/`, which is git-ignored.

Coordinate reference system throughout: **EPSG:25832** (ETRS89 / UTM zone 32N).

## 1. Statewide tiling grid

`tiles.gpkg` defines the complete set of 1 km × 1 km grid tiles and their
identifiers used at every stage of the pipeline.

- Acts as the **authoritative tiling scheme** for acquisition, training and inference.
- Layer `imagery_tiling`; key columns `tile_id`, `dop_kachel`.
- Also carries per-tile orthophoto acquisition metadata (`befliegung`, `kamera`, `linse`,
  `flughoehe`, `farbtiefe`), which is the provenance record for each downloaded tile.

### Data provenance

The grid follows the **LGL Baden-Württemberg** tiling scheme for its open geodata
products. The underlying remote-sensing data are distributed by the **LGL Open GeoData
Portal**: https://www.lgl-bw.de/Produkte/Open-Data/ under Datenlizenz Deutschland –
Namensnennung – 2.0 (dl-de/by-2-0), attribution `Datenquelle: LGL, www.lgl-bw.de`.

From that portal, 1 km × 1 km tiles can be downloaded of:

- very-high-resolution multispectral aerial imagery (**DOP20**, RGB + NIR, 20 cm)
- **nDOM** normalized digital surface model derived from airborne LiDAR (1 m)
- **DOM1** / **DGM1** surface and terrain models (1 m), used to derive an alternative
  height model

Note that the acquisition step retries neighbouring tiles (y−1, x−1, x−1/y−1) on HTTP 404,
so the `dop_kachel` actually downloaded for a subtile may differ from the one requested.

## 2. Genus taxonomy

`genera_labels.csv` maps genus names to model class ids. **This ordering is
authoritative** — it is identical in `conf/data_genera.yaml` and in the `names` dictionary
stored inside both released model checkpoints. Do not reorder it.

| id | class | id | class |
|----|-------|----|-------|
| 0 | *Acer* | 5 | Other Deciduous |
| 1 | *Aesculus* | 6 | *Platanus* |
| 2 | *Carpinus* | 7 | *Prunus* |
| 3 | Coniferous | 8 | *Quercus* |
| 4 | *Fagus* | 9 | *Tilia* |

`Coniferous` and `Other Deciduous` are aggregate classes used where genus-level
assignment from aerial imagery is not reliable. `preprocess/genus_labels.py` maps an
arbitrary genus name onto this scheme: conifers to `Coniferous`, named genera to their own
class, everything else to `Other Deciduous`.

## 3. Partition tables

A **single spatially blocked partition at the parent-tile level** is used throughout the
workflow — teacher stage and student stage alike — so no tile ever contributes data to
more than one partition.

`tiles_split.txt` — columns `dop_kachel`, `tile_id`, `split`. The pooled tile-based split:

| split | parent tiles | reference trees |
|-------|--------------|-----------------|
| train | 111 | 37,696 |
| val | 14 | 2,719 |
| test | 18 | 1,779 |
| **total** | **143** | **42,194** |

`tiles_split_city.txt` — the city-to-city generalization split, with Freiburg held out of
training entirely. Used only for the domain-shift experiment.

`greehill_genera_split.csv` — columns `tree_id`, `split`. The tree-level split induced by
`tiles_split.txt`; every tree inherits its parent tile's partition. This is what the
classification dataset builder consumes, which is why the genus classifier and the crown
detector share one partition.

The corresponding reference table, `greehill_genera.csv`, is not stored in this Git
repository. Download it from the
[heiDATA dataset](https://doi.org/10.11588/DATA/MKZPUY) and place it at
`data/greehill_genera.csv`. The committed split table is joined to this downloaded file by
`tree_id`.

`subtiles_split.txt` — columns `subtile_id`, `split`, `size`, `overlap`. It contains 2,965
unique subtiles (2,482 train / 297 val / 186 test), each inheriting its parent tile's
partition. Of these, 148 are negative subtiles with no annotated trees.

`subtiles_ids.txt` — the deduplicated plain list of subtile identifiers, for steps that
need the inventory of subtiles without their partition.

Both split tables can be regenerated with:

```bash
python -m tree_genera_mapping.scripts.make_splits \
  --reference-csv data/greehill_genera.csv \
  --tiles-gpkg data/tiles.gpkg \
  --strategy tile --stratify-col city \
  --out-tiles data/tiles_split.txt \
  --out-trees data/greehill_genera_split.csv
```

Always pass a split table to `build_dataset`. The `--val-frac` / `--test-frac` fallbacks
perform a **random** split and will not reproduce the published partition.

## 4. Image samples

`samples/` holds four 5-channel raster stacks in `.tif` format for demonstration and
testing: 640 × 640 pixels, `uint8`, band order RGB, NIR, above-ground height. They allow
the pretrained model to be exercised without any LGL download — see
`notebooks/01_demo_inference.ipynb`.

The included samples use the fixed global 0–80 m height encoding documented in
`docs/README.md` §3.1 and therefore need no sidecars. For locally normalized downloaded
tiles, `fetch_tiles.py` writes `<tile>.json` containing `height_channel.stats_m`, which
`predict_yolo.py` uses to decode the height band back to metres. The legacy
`<tile>.height.json` / `raw_height_stats` format is also supported.

## 5. Ancillary layers (not version-controlled)

These are needed for the non-forest domain, the settlement flag and the aggregated
indicators, and are downloaded separately because of their size:

| file | object type | source | purpose |
|------|-------------|--------|---------|
| `geb01_f.shp` | `AX_Gebiet_Bundesland`, `AX_KommunalesGebiet` | Basis-DLM (LGL) | state boundary; municipality boundaries |
| `veg02_f.shp` | `AX_Wald` | Basis-DLM (LGL) | forest areas erased from the mapping domain |
| `sie01_f.shp` | `AX_Ortslage` | Basis-DLM (LGL) | contiguous settlement areas, for the settlement flag |
| 100 m population grid | — | Zensus 2022, Statistisches Bundesamt | population and the spatial unit for diversity indicators |

Basis-DLM layers are dl-de/by-2-0 with attribution `Datenquelle: LGL, www.lgl-bw.de`; the
census grid is OpenData with attribution to the Statistisches Bundesamt, reference date
15 May 2022.
