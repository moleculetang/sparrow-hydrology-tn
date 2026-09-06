from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds, transform as window_transform


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
RAW_DEFAULT = (
    ROOT
    / "0_reach_topology"
    / "data"
    / "raw"
    / "hydrology"
    / "groundwater"
    / "groundwater_level_china_1km_monthly_2005_2022"
    / "data"
)
BOUNDARY_DEFAULT = ROOT / "0_reach_topology" / "data" / "processed" / "vector" / "prb_boundary.shp"
CATCHMENTS_DEFAULT = RUN / "inputs" / "baseline_snapshot" / "inputs" / "spatial_corrected" / "reach_catchments.shp"
OUT_DEFAULT = ROOT / "0_reach_topology" / "data" / "processed" / "groundwater_level_prb"
MODEL_OUT_DEFAULT = RUN / "inputs" / "processed" / "groundwater_level_monthly_by_reach_2006_2022.csv"
NODATA = -9999.0
DATE_RE = re.compile(r"GWs_((?:19|20)\d{2})-(0[1-9]|1[0-2])\.tif$", re.IGNORECASE)


def discover_months(raw_dir: Path) -> list[tuple[int, int, Path]]:
    files: list[tuple[int, int, Path]] = []
    for path in sorted(raw_dir.rglob("*.tif")):
        match = DATE_RE.fullmatch(path.name)
        if not match:
            raise RuntimeError(f"Unexpected groundwater filename: {path}")
        files.append((int(match.group(1)), int(match.group(2)), path))
    expected = [(year, month) for year in range(2005, 2023) for month in range(1, 13)]
    actual = [(year, month) for year, month, _ in files]
    if actual != expected:
        raise RuntimeError(f"Expected continuous 2005-01 to 2022-12 coverage; found {len(files)} monthly files")
    return files


def intersect_window(src: rasterio.DatasetReader, bounds: tuple[float, float, float, float]) -> Window:
    window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
    return window.intersection(Window(0, 0, src.width, src.height))


def build_masks(
    first_path: Path,
    boundary_path: Path,
    catchments_path: Path,
) -> tuple[Window, object, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[int, tuple[int, int]]]:
    boundary = gpd.read_file(boundary_path)
    catchments = gpd.read_file(catchments_path)
    if "reach_id" not in catchments.columns:
        raise RuntimeError(f"Missing reach_id in {catchments_path}")
    catchments = catchments.loc[catchments.geometry.notna() & ~catchments.geometry.is_empty, ["reach_id", "geometry"]].copy()
    if catchments["reach_id"].duplicated().any() or len(catchments) != 230:
        raise RuntimeError("Expected 230 unique model catchments")

    with rasterio.open(first_path) as src:
        if src.crs is None:
            raise RuntimeError(f"Groundwater raster has no CRS: {first_path}")
        boundary = boundary.to_crs(src.crs)
        catchments = catchments.to_crs(src.crs)
        window = intersect_window(src, tuple(float(v) for v in boundary.total_bounds))
        if window.width <= 0 or window.height <= 0:
            raise RuntimeError("PRB boundary does not overlap groundwater raster")
        transform = window_transform(window, src.transform)
        shape = (int(window.height), int(window.width))

    boundary_mask = rasterize(
        ((geometry, 1) for geometry in boundary.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)
    if not boundary_mask.any():
        raise RuntimeError("PRB boundary produced no groundwater grid cells")

    ordered = catchments.sort_values("reach_id").reset_index(drop=True)
    ordered["label"] = np.arange(1, len(ordered) + 1, dtype=np.int32)
    labels = rasterize(
        ((geometry, int(label)) for geometry, label in zip(ordered.geometry, ordered.label)),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    labels[~boundary_mask] = 0

    # Cell areas vary slightly with latitude in EPSG:4326.  Cosine latitude
    # weights make the per-catchment averages area-weighted without resampling.
    row_numbers = np.arange(shape[0], dtype=float) + 0.5
    latitudes = transform.f + row_numbers * transform.e
    area_weights = np.cos(np.deg2rad(latitudes))[:, None] * np.ones(shape[1], dtype=float)[None, :]

    mapping = ordered.loc[:, ["reach_id", "label"]].copy()
    counts = np.bincount(labels.ravel(), minlength=len(ordered) + 1)[1:]
    mapping["grid_cell_count"] = counts.astype(int)
    mapping["aggregation_method"] = np.where(counts > 0, "cell_center_area_weighted", "catchment_centroid_fallback")

    # Small headwater catchments can have no 1 km cell centre.  Retain them by
    # sampling the nearest source cell to their representative point.
    fallback: dict[int, tuple[int, int]] = {}
    missing = mapping.loc[mapping["grid_cell_count"] == 0, "reach_id"].astype(int).tolist()
    if missing:
        geom_by_reach = ordered.set_index("reach_id").geometry
        with rasterio.open(first_path) as src:
            for reach_id in missing:
                point = geom_by_reach.loc[reach_id].representative_point()
                row, col = src.index(float(point.x), float(point.y))
                local_row = int(row - window.row_off)
                local_col = int(col - window.col_off)
                if not (0 <= local_row < shape[0] and 0 <= local_col < shape[1]):
                    raise RuntimeError(f"Fallback point lies outside PRB groundwater window for reach {reach_id}")
                fallback[reach_id] = (local_row, local_col)

    return window, transform, boundary_mask, labels, area_weights, mapping, fallback


def process(
    raw_dir: Path,
    boundary_path: Path,
    catchments_path: Path,
    output_dir: Path,
    model_output: Path,
) -> dict[str, object]:
    months = discover_months(raw_dir)
    window, transform, boundary_mask, labels, area_weights, mapping, fallback = build_masks(
        months[0][2], boundary_path, catchments_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    stack_path = output_dir / "groundwater_level_monthly_2005_2022_prb.tif"
    table_csv = output_dir / "groundwater_level_monthly_by_reach_2005_2022.csv"
    table_parquet = output_dir / "groundwater_level_monthly_by_reach_2005_2022.parquet"
    mapping_csv = output_dir / "groundwater_grid_to_reach_mapping.csv"
    manifest_path = output_dir / "manifest.json"
    if stack_path.exists() or table_csv.exists() or table_parquet.exists() or model_output.exists():
        raise FileExistsError("Groundwater outputs already exist; remove or rename the output directory before rerunning")

    reach_ids = mapping["reach_id"].to_numpy(dtype=int)
    labels_flat = labels.ravel()
    weight_flat = area_weights.ravel()
    active = labels_flat > 0
    label_active = labels_flat[active]
    weight_active = weight_flat[active]
    n_reaches = len(reach_ids)
    records: list[dict[str, object]] = []

    with rasterio.open(months[0][2]) as first:
        expected_crs = first.crs
        expected_transform = first.transform
        expected_shape = first.shape
        profile = first.profile.copy()
        profile.update(
            driver="GTiff",
            width=int(window.width),
            height=int(window.height),
            count=len(months),
            transform=transform,
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )

    with rasterio.open(stack_path, "w", **profile) as dst:
        dst.update_tags(
            source_root=str(raw_dir),
            source_product="2005-2022 China 1 km monthly groundwater-level rasters",
            variable="groundwater_level_raw",
            unit="not documented in supplied GeoTIFFs; preserve raw values",
            spatial_processing="PRB boundary mask; all_touched=false",
            temporal_coverage="2005-01 to 2022-12",
            catchment_aggregation="cell-center membership; cosine-latitude area weights",
            nodata=NODATA,
        )
        for band_index, (year, month, source_path) in enumerate(months, start=1):
            with rasterio.open(source_path) as src:
                if (
                    src.count != 1
                    or src.crs != expected_crs
                    or src.transform != expected_transform
                    or src.shape != expected_shape
                ):
                    raise RuntimeError(f"Grid mismatch: {source_path}")
                raw = src.read(1, window=window, masked=True).astype("float32")
                values = np.ma.filled(raw, NODATA)
                valid = np.isfinite(values) & (values != NODATA) & boundary_mask
                clipped = np.where(valid, values, NODATA).astype("float32")
                dst.write(clipped, band_index)
                dst.set_band_description(band_index, f"groundwater_level_raw_{year}_{month:02d}")
                dst.update_tags(band_index, year=year, month=month, source_file=str(source_path))

                flat_values = clipped.ravel()[active]
                finite = flat_values != NODATA
                numerators = np.bincount(
                    label_active[finite], weights=flat_values[finite] * weight_active[finite], minlength=n_reaches + 1
                )[1:]
                denominators = np.bincount(label_active[finite], weights=weight_active[finite], minlength=n_reaches + 1)[1:]
                valid_counts = np.bincount(label_active[finite], minlength=n_reaches + 1)[1:].astype(int)
                means = np.divide(numerators, denominators, out=np.full(n_reaches, np.nan), where=denominators > 0)

                for index, reach_id in enumerate(reach_ids):
                    if valid_counts[index] == 0 and int(reach_id) in fallback:
                        row, col = fallback[int(reach_id)]
                        fallback_value = float(clipped[row, col])
                        if fallback_value != NODATA:
                            means[index] = fallback_value
                            valid_counts[index] = 1
                    records.append(
                        {
                            "reach_id": int(reach_id),
                            "year": int(year),
                            "month": int(month),
                            "groundwater_level_raw_mean": float(means[index]) if np.isfinite(means[index]) else np.nan,
                            "valid_cell_count": int(valid_counts[index]),
                            "aggregation_method": mapping.iloc[index]["aggregation_method"],
                        }
                    )
            print(f"processed {year}-{month:02d} ({band_index}/{len(months)})", flush=True)

    table = pd.DataFrame.from_records(records).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    if len(table) != n_reaches * len(months) or table.duplicated(["reach_id", "year", "month"]).any():
        raise RuntimeError("Unexpected groundwater reach-table grain")
    if table["groundwater_level_raw_mean"].isna().any():
        missing_rows = int(table["groundwater_level_raw_mean"].isna().sum())
        raise RuntimeError(f"Groundwater data are missing for {missing_rows} reach-months")
    long_term = table.groupby("reach_id", sort=False)["groundwater_level_raw_mean"].transform("mean")
    table["groundwater_level_raw_mean_2005_2022"] = long_term
    table["groundwater_level_raw_anomaly_2005_2022"] = table["groundwater_level_raw_mean"] - long_term
    table.to_csv(table_csv, index=False, encoding="utf-8-sig")
    try:
        table.to_parquet(table_parquet, index=False)
        canonical_table = table_parquet
    except ImportError:
        canonical_table = table_csv
    model_table = table.loc[table["year"].between(2006, 2022)].copy()
    if len(model_table) != n_reaches * 17 * 12:
        raise RuntimeError("Expected 46,920 model-aligned groundwater reach-months for 2006-2022")
    model_table.to_csv(model_output, index=False, encoding="utf-8-sig")
    mapping.to_csv(mapping_csv, index=False, encoding="utf-8-sig")

    manifest = {
        "source": str(raw_dir),
        "source_files": len(months),
        "source_coverage": "2005-01 to 2022-12",
        "source_grid": "EPSG:4326; nominal 1 km; 0.01-degree raster grid",
        "source_variable": "groundwater_level_raw",
        "unit_caveat": "No unit or formal method metadata were embedded in the supplied GeoTIFFs. Values are preserved without conversion.",
        "prb_raster": str(stack_path),
        "prb_raster_bands": len(months),
        "prb_window_width": int(window.width),
        "prb_window_height": int(window.height),
        "prb_cells": int(boundary_mask.sum()),
        "reach_table": str(canonical_table),
        "model_input": str(model_output),
        "reach_count": int(n_reaches),
        "reach_month_rows": int(len(table)),
        "fallback_reaches": sorted(int(value) for value in fallback),
        "processed_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def finalize_existing(
    raw_dir: Path,
    boundary_path: Path,
    catchments_path: Path,
    output_dir: Path,
    model_output: Path,
) -> dict[str, object]:
    """Finish table/manifest outputs after a completed raster run interrupted at Parquet writing."""
    months = discover_months(raw_dir)
    stack_path = output_dir / "groundwater_level_monthly_2005_2022_prb.tif"
    table_csv = output_dir / "groundwater_level_monthly_by_reach_2005_2022.csv"
    if not stack_path.exists() or not table_csv.exists():
        raise FileNotFoundError("Existing PRB raster stack and reach CSV are both required for finalization")
    with rasterio.open(stack_path) as stack:
        if stack.count != len(months):
            raise RuntimeError(f"Expected {len(months)} stack bands, found {stack.count}")
        width, height = stack.width, stack.height
        tags = stack.tags()

    _, _, boundary_mask, _, _, mapping, fallback = build_masks(months[0][2], boundary_path, catchments_path)
    table = pd.read_csv(table_csv, encoding="utf-8-sig")
    expected_rows = len(mapping) * len(months)
    if len(table) != expected_rows or table.duplicated(["reach_id", "year", "month"]).any():
        raise RuntimeError("Existing groundwater reach CSV has an invalid key grain")
    if table["groundwater_level_raw_mean"].isna().any():
        raise RuntimeError("Existing groundwater reach CSV contains missing reach-month values")
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_table = table.loc[table["year"].between(2006, 2022)].copy()
    if len(model_table) != len(mapping) * 17 * 12:
        raise RuntimeError("Expected 46,920 model-aligned groundwater reach-months for 2006-2022")
    model_table.to_csv(model_output, index=False, encoding="utf-8-sig")
    mapping.to_csv(output_dir / "groundwater_grid_to_reach_mapping.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "source": str(raw_dir),
        "source_files": len(months),
        "source_coverage": "2005-01 to 2022-12",
        "source_grid": "EPSG:4326; nominal 1 km; 0.01-degree raster grid",
        "source_variable": "groundwater_level_raw",
        "unit_caveat": "No unit or formal method metadata were embedded in the supplied GeoTIFFs. Values are preserved without conversion.",
        "prb_raster": str(stack_path),
        "prb_raster_bands": int(len(months)),
        "prb_window_width": int(width),
        "prb_window_height": int(height),
        "prb_cells": int(boundary_mask.sum()),
        "reach_table": str(table_csv),
        "model_input": str(model_output),
        "reach_count": int(len(mapping)),
        "reach_month_rows": int(len(table)),
        "fallback_reaches": sorted(int(value) for value in fallback),
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "stack_tags": tags,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clip monthly groundwater rasters to PRB and aggregate them to model catchments.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--boundary", type=Path, default=BOUNDARY_DEFAULT)
    parser.add_argument("--catchments", type=Path, default=CATCHMENTS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--model-output", type=Path, default=MODEL_OUT_DEFAULT)
    parser.add_argument("--finalize-existing", action="store_true", help="Finish outputs from an already-written PRB raster stack and reach CSV.")
    args = parser.parse_args()
    if args.finalize_existing:
        manifest = finalize_existing(
            args.raw_dir.resolve(),
            args.boundary.resolve(),
            args.catchments.resolve(),
            args.output_dir.resolve(),
            args.model_output.resolve(),
        )
    else:
        manifest = process(
            args.raw_dir.resolve(),
            args.boundary.resolve(),
            args.catchments.resolve(),
            args.output_dir.resolve(),
            args.model_output.resolve(),
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
