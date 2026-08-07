#!/usr/bin/env python3
"""
Post-processing of statewide tree predictions.

Turns the per-tile prediction GeoPackages written by ``predict_yolo.py`` into the
single released inventory layer. Four steps, in this order:

1. **Merge** all per-tile prediction files.
2. **Restrict to the mapping domain** - predictions whose crown centroid falls outside
   the non-forest ROI are dropped.
3. **Flag settlement context** - predictions are classified as inside or outside a
   contiguous settlement area (ATKIS ``AX_Ortslage``). Non-settlement predictions are
   *retained and flagged*, never dropped.
4. **Apply the Acer plausibility rule** - detections assigned to *Acer* with confidence
   below ``--acer-threshold`` are relabelled to "Other Deciduous". The pre-rule label is
   preserved in ``original_pred_class`` for *every* row, so the rule is fully reversible
   and can be recalibrated by the user.

All spatial tests use the crown **centroid**, so a crown straddling a mask boundary is
included or excluded as a whole according to its centroid.

Tiles are streamed one at a time and appended to the output GeoPackage, so peak memory
stays proportional to a single tile rather than to the full statewide layer.

Example
-------
    python -m tree_genera_mapping.scripts.finalize_results \\
        --pred-dir cache/predictions \\
        --domain-gpkg data/bw_nonforest_roi.gpkg \\
        --settlement-shp data/sie01_f.shp \\
        --output cache/trees_bw.gpkg \\
        --layer trees \\
        --acer-threshold 0.50 \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

LOGGER = logging.getLogger("finalize_results")

# Authoritative class scheme - must match conf/data_genera.yaml and the checkpoint's
# `names` mapping. Do not reorder.
CLASS_NAMES = {
    0: "Acer",
    1: "Aesculus",
    2: "Carpinus",
    3: "Coniferous",
    4: "Fagus",
    5: "Other Deciduous",
    6: "Platanus",
    7: "Prunus",
    8: "Quercus",
    9: "Tilia",
}
NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

OUTPUT_COLUMNS = [
    "tile_id",
    "class_id_det",
    "class_name_det",
    "confidence_det",
    "original_pred_class",
    "canopy_height_m",
    "canopy_diameter",
    "centroid_x",
    "centroid_y",
    "settlement_domain",
    "geometry",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge, mask, flag and post-process per-tile tree predictions."
    )
    ap.add_argument("--pred-dir", required=True, type=Path,
                    help="Directory of per-tile prediction GeoPackages")
    ap.add_argument("--pred-glob", default="tile_*.gpkg",
                    help="Glob for prediction files (default: tile_*.gpkg)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output GeoPackage path")
    ap.add_argument("--layer", default="trees", help="Output layer name")

    ap.add_argument("--domain-gpkg", type=Path, default=None,
                    help="Non-forest mapping domain; predictions outside are DROPPED. "
                         "Omit to skip domain masking.")
    ap.add_argument("--domain-layer", default=None, help="Layer name inside --domain-gpkg")
    ap.add_argument("--settlement-shp", type=Path, default=None,
                    help="ATKIS AX_Ortslage settlement polygons (sie01_f.shp); "
                         "predictions are FLAGGED, not dropped. Omit to skip flagging.")
    ap.add_argument("--settlement-name-col", default="NAM",
                    help="Attribute carrying the settlement name (default: NAM)")
    ap.add_argument("--keep-settlement-name", action="store_true",
                    help="Also write the settlement name as a column")

    ap.add_argument("--acer-threshold", type=float, default=0.50,
                    help="Detections labelled --acer-class with confidence below this "
                         "value are relabelled to --acer-fallback (default: 0.50). "
                         "Pass 0 to disable the rule.")
    ap.add_argument("--acer-class", default="Acer",
                    help="Class the plausibility rule applies to (default: Acer)")
    ap.add_argument("--acer-fallback", default="Other Deciduous",
                    help="Class low-confidence detections are moved to")

    ap.add_argument("--dedup-iou", type=float, default=0.5,
                    help="Suppress lower-confidence detections overlapping a kept one above "
                         "this IoU, removing sliding-window duplicates (default: 0.5). "
                         "Pass 0 to disable.")
    ap.add_argument("--dedup-same-class-only", action="store_true",
                    help="Only suppress duplicates carrying the same class. Default is "
                         "class-agnostic, which also removes the same crown detected twice "
                         "with different genus labels.")

    ap.add_argument("--min-confidence", type=float, default=None,
                    help="Optional extra global confidence floor. The inference step "
                         "already applies 0.30; set this only to raise it further.")
    ap.add_argument("--target-crs", default=None,
                    help="Reproject output to this CRS (e.g. EPSG:25832). "
                         "Default: keep the prediction CRS.")

    ap.add_argument("--resume", action="store_true",
                    help="Append to an existing output instead of overwriting, skipping "
                         "tiles already listed in the run report")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N tiles (for testing)")
    ap.add_argument("--report", type=Path, default=None,
                    help="JSON summary path (default: <output>.report.json)")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def load_mask(path: Path, layer: str | None, label: str) -> gpd.GeoDataFrame:
    LOGGER.info("Reading %s mask: %s", label, path)
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    gdf = gdf[~gdf.geometry.isna()].copy()
    if not gdf.geometry.is_valid.all():
        LOGGER.info("  repairing invalid geometries in %s mask", label)
        gdf["geometry"] = gdf.geometry.buffer(0)
    LOGGER.info("  %d polygons, CRS %s", len(gdf), gdf.crs)
    return gdf


def centroid_points(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Crown centroids, preferring the stored centroid_x/centroid_y if present."""
    if {"centroid_x", "centroid_y"}.issubset(gdf.columns):
        pts = gpd.points_from_xy(gdf["centroid_x"], gdf["centroid_y"])
        return gpd.GeoSeries(pts, index=gdf.index, crs=gdf.crs)
    return gdf.geometry.centroid


def within_mask(gdf: gpd.GeoDataFrame, mask: gpd.GeoDataFrame) -> pd.Series:
    """Boolean Series: is each crown centroid inside any mask polygon?"""
    pts = gpd.GeoDataFrame(geometry=centroid_points(gdf), crs=gdf.crs)
    if mask.crs is not None and pts.crs is not None and mask.crs != pts.crs:
        mask = mask.to_crs(pts.crs)
    # Restrict the mask to this tile's envelope so the join stays cheap.
    minx, miny, maxx, maxy = pts.total_bounds
    local = mask.cx[minx:maxx, miny:maxy]
    if local.empty:
        return pd.Series(False, index=gdf.index)
    hit = gpd.sjoin(pts, local[["geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    return hit["index_right"].notna().reindex(gdf.index, fill_value=False)


def settlement_join(
    gdf: gpd.GeoDataFrame, mask: gpd.GeoDataFrame, name_col: str, keep_name: bool
) -> pd.DataFrame:
    """Return a frame with `settlement_domain` and optionally the settlement name."""
    pts = gpd.GeoDataFrame(geometry=centroid_points(gdf), crs=gdf.crs)
    if mask.crs is not None and pts.crs is not None and mask.crs != pts.crs:
        mask = mask.to_crs(pts.crs)
    cols = ["geometry"] + ([name_col] if keep_name and name_col in mask.columns else [])
    minx, miny, maxx, maxy = pts.total_bounds
    local = mask[cols].cx[minx:maxx, miny:maxy]
    out = pd.DataFrame(index=gdf.index)
    if local.empty:
        out["settlement_domain"] = "non_settlement"
        if keep_name:
            out["settlement_name"] = None
        return out
    hit = gpd.sjoin(pts, local, how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")].reindex(gdf.index)
    inside = hit["index_right"].notna()
    out["settlement_domain"] = pd.Series(
        ["settlement" if b else "non_settlement" for b in inside], index=gdf.index
    )
    if keep_name:
        out["settlement_name"] = hit[name_col] if name_col in hit.columns else None
    return out


def deduplicate(
    gdf: gpd.GeoDataFrame, iou_thresh: float, conf_col: str, same_class_only: bool
) -> tuple[gpd.GeoDataFrame, int]:
    """Greedy non-maximum suppression over crown polygons.

    Inference uses a sliding window with stride < patch size, so a tree lying in an
    overlap strip is detected more than once. Detections are visited in descending
    confidence and any lower-confidence detection overlapping a kept one above
    `iou_thresh` is dropped. An STRtree keeps this near-linear rather than O(n^2).
    """
    n = len(gdf)
    if n < 2 or iou_thresh <= 0:
        return gdf, 0

    geoms = gdf.geometry.to_numpy()
    areas = shapely.area(geoms)
    conf = pd.to_numeric(gdf[conf_col], errors="coerce").fillna(0.0).to_numpy()
    classes = gdf["class_id_det"].to_numpy() if same_class_only else None

    order = np.argsort(-conf, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)

    tree = shapely.STRtree(geoms)
    suppressed = np.zeros(n, dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        for j in tree.query(geoms[i]):
            # Only ever suppress a detection that ranks below the current one, so the
            # highest-confidence member of each cluster always survives.
            if j == i or suppressed[j] or rank[j] <= rank[i]:
                continue
            if classes is not None and classes[j] != classes[i]:
                continue
            inter = shapely.area(shapely.intersection(geoms[i], geoms[j]))
            if inter <= 0.0:
                continue
            union = areas[i] + areas[j] - inter
            if union > 0.0 and inter / union > iou_thresh:
                suppressed[j] = True

    return gdf[~suppressed], int(suppressed.sum())


def apply_acer_rule(
    gdf: gpd.GeoDataFrame, source: str, fallback: str, threshold: float
) -> tuple[gpd.GeoDataFrame, int]:
    """Relabel low-confidence `source` detections to `fallback`, preserving the original."""
    # original_pred_class is written for EVERY row, so the rule is always reversible.
    gdf["original_pred_class"] = gdf["class_name_det"]
    if threshold <= 0:
        return gdf, 0
    if fallback not in NAME_TO_ID:
        raise ValueError(f"--acer-fallback {fallback!r} is not one of {sorted(NAME_TO_ID)}")
    hit = (gdf["class_name_det"] == source) & (gdf["confidence_det"] < threshold)
    n = int(hit.sum())
    if n:
        gdf.loc[hit, "class_name_det"] = fallback
        gdf.loc[hit, "class_id_det"] = NAME_TO_ID[fallback]
    return gdf, n


def tile_id_from_path(p: Path) -> str:
    stem = p.stem
    return stem[len("tile_"):] if stem.startswith("tile_") else stem


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    pred_files = sorted(args.pred_dir.glob(args.pred_glob))
    if not pred_files:
        raise SystemExit(f"No files matching {args.pred_glob!r} under {args.pred_dir}")
    if args.limit:
        pred_files = pred_files[: args.limit]
    LOGGER.info("Found %d prediction files", len(pred_files))

    domain = load_mask(args.domain_gpkg, args.domain_layer, "domain") if args.domain_gpkg else None
    settle = load_mask(args.settlement_shp, None, "settlement") if args.settlement_shp else None

    report_path = args.report or args.output.with_suffix(".report.json")
    done: set[str] = set()
    if args.resume and report_path.exists():
        done = set(json.loads(report_path.read_text()).get("tiles_written", []))
        LOGGER.info("Resuming; %d tiles already written", len(done))
    elif args.output.exists():
        LOGGER.warning("Overwriting existing %s", args.output)
        args.output.unlink()

    stats = Counter()
    class_hist: Counter = Counter()
    tiles_written: list[str] = sorted(done)
    first_write = not (args.resume and args.output.exists())

    for path in pred_files:
        tid = tile_id_from_path(path)
        if tid in done:
            continue
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:                                    # noqa: BLE001
            LOGGER.error("%s: unreadable (%s)", path.name, exc)
            stats["tiles_failed"] += 1
            continue

        stats["tiles_read"] += 1
        if gdf.empty:
            stats["tiles_empty"] += 1
            tiles_written.append(tid)
            continue
        stats["detections_in"] += len(gdf)

        if args.target_crs and gdf.crs is not None and str(gdf.crs) != args.target_crs:
            gdf = gdf.to_crs(args.target_crs)

        if args.min_confidence is not None:
            before = len(gdf)
            gdf = gdf[gdf["confidence_det"] >= args.min_confidence]
            stats["dropped_low_confidence"] += before - len(gdf)

        if domain is not None and not gdf.empty:
            keep = within_mask(gdf, domain)
            stats["dropped_outside_domain"] += int((~keep).sum())
            gdf = gdf[keep]

        if gdf.empty:
            stats["tiles_empty_after_mask"] += 1
            tiles_written.append(tid)
            continue

        gdf = gdf.reset_index(drop=True)

        gdf, n_dup = deduplicate(gdf, args.dedup_iou, "confidence_det",
                                 args.dedup_same_class_only)
        stats["duplicates_removed"] += n_dup
        gdf = gdf.reset_index(drop=True)

        gdf["tile_id"] = tid

        if settle is not None:
            flags = settlement_join(gdf, settle, args.settlement_name_col,
                                    args.keep_settlement_name)
            for col in flags.columns:
                gdf[col] = flags[col].values
            stats["settlement"] += int((gdf["settlement_domain"] == "settlement").sum())
            stats["non_settlement"] += int((gdf["settlement_domain"] == "non_settlement").sum())
        else:
            gdf["settlement_domain"] = None

        gdf, n_reassigned = apply_acer_rule(
            gdf, args.acer_class, args.acer_fallback, args.acer_threshold
        )
        stats["acer_reassigned"] += n_reassigned
        class_hist.update(gdf["class_name_det"].tolist())
        stats["detections_out"] += len(gdf)

        cols = [c for c in OUTPUT_COLUMNS if c in gdf.columns]
        if args.keep_settlement_name and "settlement_name" in gdf.columns:
            cols.insert(-1, "settlement_name")
        out = gdf[cols]
        out.to_file(
            args.output,
            layer=args.layer,
            driver="GPKG",
            mode="w" if first_write else "a",
        )
        first_write = False
        tiles_written.append(tid)

        if stats["tiles_read"] % 500 == 0:
            LOGGER.info("  %d tiles, %d trees so far", stats["tiles_read"], stats["detections_out"])

    summary = {
        "output": str(args.output),
        "layer": args.layer,
        "parameters": {
            "pred_dir": str(args.pred_dir),
            "domain_gpkg": str(args.domain_gpkg) if args.domain_gpkg else None,
            "settlement_shp": str(args.settlement_shp) if args.settlement_shp else None,
            "dedup_iou": args.dedup_iou,
            "dedup_same_class_only": args.dedup_same_class_only,
            "acer_class": args.acer_class,
            "acer_fallback": args.acer_fallback,
            "acer_threshold": args.acer_threshold,
            "min_confidence": args.min_confidence,
            "target_crs": args.target_crs,
        },
        "counts": dict(stats),
        "class_distribution": dict(sorted(class_hist.items())),
        "tiles_written": tiles_written,
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    LOGGER.info("---- summary ----")
    LOGGER.info("tiles read              %d", stats["tiles_read"])
    LOGGER.info("detections in           %d", stats["detections_in"])
    LOGGER.info("dropped outside domain  %d", stats["dropped_outside_domain"])
    LOGGER.info("duplicates removed      %d", stats["duplicates_removed"])
    LOGGER.info("settlement / non-       %d / %d", stats["settlement"], stats["non_settlement"])
    LOGGER.info("%s reassigned to %s   %d", args.acer_class, args.acer_fallback,
                stats["acer_reassigned"])
    LOGGER.info("trees written           %d", stats["detections_out"])
    LOGGER.info("report                  %s", report_path)


if __name__ == "__main__":
    main()