from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds, transform as window_transform


NODATA = -9999.0
UNIT = "10^8 m3/month"
SECTOR_CODES = {
    "domestic": "dom",
    "thermal_power_cooling": "ele",
    "manufacturing": "manu",
    "irrigation": "irr",
}


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sector_from_name(name: str) -> str:
    stem = Path(name).stem.lower()
    if "irr" in stem:
        return "irrigation"
    if "manu" in stem:
        return "manufacturing"
    if "ele" in stem:
        return "thermal_power_cooling"
    if "dom" in stem:
        return "domestic"
    return stem


def _sector_from_path(path: Path) -> str:
    text = " ".join([path.name.lower(), path.parent.name.lower()])
    if "灌溉" in text or "irr" in text:
        return "irrigation"
    if "制造" in text or "manu" in text:
        return "manufacturing"
    if "火电" in text or "冷却" in text or "ele" in text:
        return "thermal_power_cooling"
    if "生活" in text or "dom" in text:
        return "domestic"
    return _sector_from_name(path.name)


def _parse_year_month(path: Path, tags: dict[str, str] | None = None) -> tuple[int, int]:
    tags = tags or {}
    year_month = tags.get("YEAR_MONTH") or tags.get("year_month") or ""
    match = re.search(r"((?:19|20)\d{2})[_-](0[1-9]|1[0-2])", year_month)
    if not match:
        match = re.search(r"((?:19|20)\d{2})[_-](0[1-9]|1[0-2])", path.stem)
    if not match:
        raise RuntimeError(f"Cannot parse year/month from {path}")
    return int(match.group(1)), int(match.group(2))


def _discover_monthly_tifs(raw_dir: Path) -> dict[str, list[tuple[int, int, Path]]]:
    grouped: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)
    for path in sorted(raw_dir.rglob("*.tif")):
        sector = _sector_from_path(path)
        with rasterio.open(path) as src:
            year, month = _parse_year_month(path, src.tags())
        grouped[sector].append((year, month, path))

    return {sector: sorted(items) for sector, items in grouped.items()}


def _choose_dataset(nc_path: Path) -> str:
    with rasterio.open(nc_path) as root:
        subdatasets = list(root.subdatasets)
    if not subdatasets:
        return str(nc_path)

    candidates: list[tuple[int, str]] = []
    for dataset in subdatasets:
        try:
            with rasterio.open(dataset) as src:
                score = int(src.count > 1) + int(src.width > 1 and src.height > 1)
                candidates.append((score, dataset))
        except rasterio.errors.RasterioIOError:
            continue
    if not candidates:
        raise RuntimeError(f"No readable raster variables found in {nc_path}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def _intersect_window(src, bounds: tuple[float, float, float, float]) -> Window:
    win = from_bounds(*bounds, transform=src.transform)
    win = win.round_offsets().round_lengths()
    full = Window(0, 0, src.width, src.height)
    return win.intersection(full)


def _band_date(src, band_idx: int) -> tuple[int | None, int | None]:
    desc = src.descriptions[band_idx - 1] if src.descriptions else None
    tags = src.tags(band_idx)
    text = " ".join([desc or "", *[f"{key}={value}" for key, value in tags.items()]])
    match = re.search(r"(19|20)\d{2}[-_/ ]?(0[1-9]|1[0-2])", text)
    if match:
        value = match.group(0).replace("_", "-").replace("/", "-").replace(" ", "-")
        return int(value[:4]), int(value[-2:])

    # HSWUD spans 1965-2022 monthly. If metadata lacks readable dates, derive
    # the timestamp from the band order.
    offset = band_idx - 1
    return 1965 + offset // 12, 1 + offset % 12


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sector",
        "input_file",
        "output_file",
        "bands",
        "width",
        "height",
        "valid_cells_first_band",
        "start_year",
        "start_month",
        "end_year",
        "end_month",
        "source_format",
        "unit",
        "all_touched",
        "updated_utc",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clip_monthly_tif_stack(
    sector: str,
    files: list[tuple[int, int, Path]],
    boundary: gpd.GeoDataFrame,
    output_dir: Path,
    all_touched: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not files:
        raise RuntimeError(f"No TIF files for sector {sector}")

    output_path = output_dir / f"HSWUD_{SECTOR_CODES.get(sector, sector)}_prb.tif"
    first_year, first_month, first_path = files[0]
    last_year, last_month, _ = files[-1]

    with rasterio.open(first_path) as first:
        boundary_ll = boundary.to_crs(first.crs or "EPSG:4326")
        bounds = tuple(float(v) for v in boundary_ll.total_bounds)
        win = _intersect_window(first, bounds)
        transform = window_transform(win, first.transform)
        height = int(win.height)
        width = int(win.width)
        if height <= 0 or width <= 0:
            raise RuntimeError(f"PRB boundary does not overlap {first_path}")

        mask = rasterize(
            ((geom, 1) for geom in boundary_ll.geometry if geom is not None and not geom.is_empty),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        ).astype(bool)
        if not mask.any():
            raise RuntimeError(f"PRB boundary rasterized to zero cells for {first_path}")

        profile = first.profile.copy()
        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            count=len(files),
            transform=transform,
            nodata=NODATA,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )
        expected_crs = first.crs
        expected_transform = first.transform
        expected_shape = (first.height, first.width)

    rows = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.update_tags(
            source_root=str(first_path.parent.parent),
            source_format="monthly_tif",
            sector=sector,
            unit=UNIT,
            clip_boundary="data/processed/vector/prb_boundary.shp",
            all_touched=int(all_touched),
            start=f"{first_year:04d}-{first_month:02d}",
            end=f"{last_year:04d}-{last_month:02d}",
        )
        for band_idx, (year, month, path) in enumerate(files, start=1):
            with rasterio.open(path) as src:
                if src.count != 1:
                    raise RuntimeError(f"Expected a single-band TIF: {path}")
                if src.crs != expected_crs or src.transform != expected_transform or (src.height, src.width) != expected_shape:
                    raise RuntimeError(f"Grid mismatch in {path}")
                arr = src.read(1, window=win, masked=True).astype("float32")
                data = np.ma.filled(arr, NODATA)
                valid = (~np.ma.getmaskarray(arr)) & mask
                if src.nodata is not None:
                    valid &= data != float(src.nodata)
                clipped = np.where(valid, data, NODATA).astype("float32")

            dst.write(clipped, band_idx)
            dst.set_band_description(band_idx, f"{sector}_{year}_{month:02d}")
            dst.update_tags(
                band_idx,
                sector=sector,
                year=year,
                month=month,
                unit=UNIT,
                source_file=str(path),
            )
            valid_values = clipped[clipped != NODATA]
            rows.append(
                {
                    "sector": sector,
                    "year": year,
                    "month": month,
                    "sum_10e8_m3_month": float(valid_values.sum()) if valid_values.size else np.nan,
                    "mean_10e8_m3_month": float(valid_values.mean()) if valid_values.size else np.nan,
                    "valid_cells": int(valid_values.size),
                    "source_file": str(path),
                    "processed_file": output_path.name,
                }
            )

    manifest_row = {
        "sector": sector,
        "input_file": str(first_path.parent),
        "output_file": str(output_path),
        "bands": len(files),
        "width": width,
        "height": height,
        "valid_cells_first_band": int(rows[0]["valid_cells"]) if rows else 0,
        "start_year": first_year,
        "start_month": first_month,
        "end_year": last_year,
        "end_month": last_month,
        "source_format": "monthly_tif",
        "unit": UNIT,
        "all_touched": int(all_touched),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    return manifest_row, pd.DataFrame(rows)


def clip_file(
    nc_path: Path,
    boundary: gpd.GeoDataFrame,
    output_dir: Path,
    all_touched: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    dataset_name = _choose_dataset(nc_path)
    sector = _sector_from_name(nc_path.name)
    output_path = output_dir / f"{nc_path.stem}_prb.tif"

    with rasterio.open(dataset_name) as src:
        boundary_ll = boundary.to_crs(src.crs or "EPSG:4326")
        bounds = tuple(float(v) for v in boundary_ll.total_bounds)
        win = _intersect_window(src, bounds)
        transform = window_transform(win, src.transform)
        height = int(win.height)
        width = int(win.width)
        if height <= 0 or width <= 0:
            raise RuntimeError(f"PRB boundary does not overlap {nc_path}")

        mask = rasterize(
            ((geom, 1) for geom in boundary_ll.geometry if geom is not None and not geom.is_empty),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        ).astype(bool)
        if not mask.any():
            raise RuntimeError(f"PRB boundary rasterized to zero cells for {nc_path}")

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            transform=transform,
            nodata=NODATA,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )

        rows = []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.update_tags(
                source=str(nc_path),
                source_dataset=dataset_name,
                sector=sector,
                unit=UNIT,
                clip_boundary="data/processed/vector/prb_boundary.shp",
                all_touched=int(all_touched),
            )
            for band_idx in range(1, src.count + 1):
                arr = src.read(band_idx, window=win, masked=True).astype("float32")
                data = np.ma.filled(arr, NODATA)
                valid = (~np.ma.getmaskarray(arr)) & mask
                if src.nodata is not None:
                    valid &= data != float(src.nodata)
                clipped = np.where(valid, data, NODATA).astype("float32")
                dst.write(clipped, band_idx)
                year, month = _band_date(src, band_idx)
                dst.set_band_description(band_idx, f"{sector}_{year}_{month:02d}" if year and month else sector)
                dst.update_tags(
                    band_idx,
                    sector=sector,
                    year="" if year is None else year,
                    month="" if month is None else month,
                    unit=UNIT,
                )
                valid_values = clipped[clipped != NODATA]
                rows.append(
                    {
                        "sector": sector,
                        "year": year,
                        "month": month,
                        "sum_10e8_m3_month": float(valid_values.sum()) if valid_values.size else np.nan,
                        "mean_10e8_m3_month": float(valid_values.mean()) if valid_values.size else np.nan,
                        "valid_cells": int(valid_values.size),
                        "source_file": nc_path.name,
                        "processed_file": output_path.name,
                    }
                )

    manifest_row = {
        "sector": sector,
        "input_file": str(nc_path),
        "output_file": str(output_path),
        "bands": src.count,
        "width": width,
        "height": height,
        "valid_cells_first_band": int(rows[0]["valid_cells"]) if rows else 0,
        "start_year": rows[0]["year"] if rows else "",
        "start_month": rows[0]["month"] if rows else "",
        "end_year": rows[-1]["year"] if rows else "",
        "end_month": rows[-1]["month"] if rows else "",
        "source_format": "netcdf",
        "unit": UNIT,
        "all_touched": int(all_touched),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    return manifest_row, pd.DataFrame(rows)


def clip_all(raw_dir: Path, boundary_path: Path, output_dir: Path, summary_csv: Path, all_touched: bool) -> None:
    boundary = gpd.read_file(boundary_path)
    boundary = boundary[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()
    if boundary.empty:
        raise RuntimeError(f"No valid geometries found in {boundary_path}")

    manifest_rows: list[dict[str, object]] = []
    summaries: list[pd.DataFrame] = []
    monthly_tifs = _discover_monthly_tifs(raw_dir)
    if monthly_tifs:
        for sector in sorted(monthly_tifs):
            files = monthly_tifs[sector]
            print(f"[clip] {sector}: {len(files)} monthly TIF files")
            manifest_row, summary = clip_monthly_tif_stack(sector, files, boundary, output_dir, all_touched)
            manifest_rows.append(manifest_row)
            summaries.append(summary)
            print(f"  written: {manifest_row['output_file']}")
    else:
        nc_files = sorted(raw_dir.glob("HSWUD*.nc"))
        if not nc_files:
            raise RuntimeError(f"No HSWUD monthly TIF or NetCDF files found in {raw_dir}")
        for nc_path in nc_files:
            print(f"[clip] {nc_path.name}")
            manifest_row, summary = clip_file(nc_path, boundary, output_dir, all_touched)
            manifest_rows.append(manifest_row)
            summaries.append(summary)
            print(f"  written: {manifest_row['output_file']}")

    summary = pd.concat(summaries, ignore_index=True).sort_values(["sector", "year", "month"])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    _write_manifest(output_dir / "manifest.csv", manifest_rows)
    print(f"written: {summary_csv}")
    print(f"written: {output_dir / 'manifest.csv'}")


def main() -> int:
    root = _root()
    parser = argparse.ArgumentParser(description="Clip HSWUD NetCDF files to the PRB modelling boundary.")
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw" / "hswud")
    parser.add_argument("--boundary", type=Path, default=root / "data" / "processed" / "vector" / "prb_boundary.shp")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed" / "hswud_prb")
    parser.add_argument("--summary-csv", type=Path, default=root / "results" / "tables" / "hswud_prb_monthly_sector_summary.csv")
    parser.add_argument("--all-touched", action="store_true", help="Include every HSWUD cell touched by the PRB boundary.")
    args = parser.parse_args()
    clip_all(
        args.raw_dir.resolve(),
        args.boundary.resolve(),
        args.output_dir.resolve(),
        args.summary_csv.resolve(),
        args.all_touched,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
