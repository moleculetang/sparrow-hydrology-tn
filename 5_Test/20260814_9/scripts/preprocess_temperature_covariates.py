"""Prepare temperature covariates at the 230 PRB reach-catchment grain.

This script deliberately keeps the physical meanings separate:

* Benz et al. (2024) is a static annual-mean groundwater-temperature product
  for 2020 at the groundwater-table depth, plus a 2000--2020 change map.  It
  is not a monthly groundwater-temperature observation series.
* ERA5-Land ``lmlt`` is the modelled lake mix-layer temperature.  It is made
  available as a monthly lake-temperature proxy, never renamed as observed
  stream-water temperature and never used to fill ordinary river reaches.

All derived products are owned by the temporary model database.  Raw Benz
source members are copied from the user-supplied archive without altering it.
Run with D:\\ProgramData\\anaconda3\\envs\\sparrow\\python.exe.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260814_9"
RAW = ROOT / "0_reach_topology" / "data" / "raw"
BENZ_RAW = RAW / "hydrology" / "groundwater_temperature" / "benz_etal_2024_global_groundwater_warming"
BENZ_DATA = BENZ_RAW / "data"
BENZ_META = BENZ_RAW / "metadata"
BENZ_ARCHIVE = BENZ_RAW / "archives" / "doi_10.5683_SP3_GE4VEQ.zip"
UNSORTED_ARCHIVE = ROOT / "0_reach_topology" / "data" / "未整理" / "doi_10.5683_SP3_GE4VEQ.zip"
LMLT_RAW = RAW / "atmosphere" / "meteorology" / "era5_land" / "data" / "lake_mix_layer_temperature_prb_buffer"
SPATIAL = RUN / "inputs" / "spatial"
READY_STATIC = RUN / "inputs" / "model_ready" / "static"
READY_MONTHLY = RUN / "inputs" / "model_ready" / "monthly"
PROVENANCE = RUN / "inputs" / "provenance"
WORK = RUN / "work" / "temperature_covariates"

GDAL_BIN = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin")
GDALINFO = GDAL_BIN / "gdalinfo.exe"
GDALRASTERIZE = GDAL_BIN / "gdal_rasterize.exe"
GDALTRANSLATE = GDAL_BIN / "gdal_translate.exe"

BENZ_MEMBERS = {
    "Data/images/Temp_2020_GWTable.tif": BENZ_DATA / "Temp_2020_GWTable.tif",
    "Data/images/TempChange_2020to2000_GWTable.tif": BENZ_DATA / "TempChange_2020to2000_GWTable.tif",
    "0-Read Me.txt": BENZ_META / "source_readme.txt",
}


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def gdal_info(path: Path) -> dict:
    return json.loads(subprocess.check_output([str(GDALINFO), "-json", str(path)], text=True, encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_benz_source() -> dict[str, object]:
    """Extract only model-relevant Benz source members and verify ZIP CRCs."""
    source_archive = BENZ_ARCHIVE if BENZ_ARCHIVE.exists() else UNSORTED_ARCHIVE
    if not source_archive.exists() or source_archive.stat().st_size == 0:
        raise FileNotFoundError(f"Missing Benz source archive: {source_archive}")
    BENZ_DATA.mkdir(parents=True, exist_ok=True)
    BENZ_META.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archive) as archive:
        names = set(archive.namelist())
        missing = set(BENZ_MEMBERS) - names
        if missing:
            raise RuntimeError(f"Benz archive lacks required members: {sorted(missing)}")
        for member, target in BENZ_MEMBERS.items():
            expected = archive.getinfo(member).file_size
            if target.exists() and target.stat().st_size == expected:
                continue
            # ZipExtFile validates the member CRC when it is fully consumed.
            with archive.open(member) as src, target.open("wb") as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
            if target.stat().st_size != expected:
                raise RuntimeError(f"Incomplete extraction for {member}")
    manifest = {
        "source_archive": str(source_archive),
        "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_sha256": sha256(source_archive),
        "doi": "10.1038/s41561-024-01453-x",
        "citation": "Benz et al. (2024), Global groundwater warming due to climate change, Nature Geoscience.",
        "extracted_members": {member: str(target) for member, target in BENZ_MEMBERS.items()},
    }
    (BENZ_META / "ingest_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def catchments() -> gpd.GeoDataFrame:
    frame = gpd.read_file(SPATIAL / "reach_catchments.shp").loc[:, ["reach_id", "geometry"]]
    if len(frame) != 230 or frame.reach_id.nunique() != 230:
        raise RuntimeError("Expected exactly 230 unique PRB reach catchments")
    if (~frame.geometry.is_valid).any():
        frame.loc[~frame.geometry.is_valid, "geometry"] = frame.loc[~frame.geometry.is_valid, "geometry"].make_valid()
    return frame.sort_values("reach_id").reset_index(drop=True)


def write_wgs84_catchments() -> Path:
    target = WORK / "reach_catchments_wgs84.gpkg"
    if target.exists():
        existing = gpd.read_file(target)
        if len(existing) == 230 and existing.reach_id.nunique() == 230 and existing.geometry.is_valid.all():
            return target
        raise RuntimeError(f"Existing owned processing copy is malformed: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = catchments().to_crs("EPSG:4326")
    if (~frame.geometry.is_valid).any():
        frame.loc[~frame.geometry.is_valid, "geometry"] = frame.loc[~frame.geometry.is_valid, "geometry"].buffer(0)
    if not frame.geometry.is_valid.all():
        raise RuntimeError("Cannot make a valid WGS84 catchment processing copy")
    frame.to_file(target, layer="reach_catchments", driver="GPKG")
    return target


def raster_mask_from_reference(reference: Path, name: str, fallback_crs: str | None = None) -> tuple[Path, tuple[int, int]]:
    info = gdal_info(reference)
    width, height = map(int, info["size"])
    gt = info["geoTransform"]
    min_x, max_y = float(gt[0]), float(gt[3])
    max_x = min_x + width * float(gt[1])
    min_y = max_y + height * float(gt[5])
    crs = info.get("coordinateSystem", {}).get("wkt", fallback_crs)
    if crs is None:
        raise RuntimeError(f"Missing CRS for {reference}")
    mask = WORK / f"catchment_mask_{name}.dat"
    if not mask.exists():
        run([
            str(GDALRASTERIZE), "-l", "reach_catchments", "-a", "reach_id", "-init", "0", "-a_nodata", "0",
            "-te", str(min_x), str(min_y), str(max_x), str(max_y), "-ts", str(width), str(height),
            "-ot", "Int32", "-of", "ENVI", str(write_wgs84_catchments()), str(mask),
        ])
    return mask, (height, width)


def raster_mask_wgs84(lons: np.ndarray, lats: np.ndarray, name: str) -> tuple[Path, tuple[int, int]]:
    if not np.all(np.diff(lons) > 0) or not np.all(np.diff(lats) < 0):
        raise RuntimeError("ERA5-Land grid must have ascending longitude and descending latitude")
    dx = float(np.median(np.diff(lons)))
    dy = float(np.median(np.abs(np.diff(lats))))
    min_x, max_x = float(lons.min() - dx / 2), float(lons.max() + dx / 2)
    min_y, max_y = float(lats.min() - dy / 2), float(lats.max() + dy / 2)
    mask = WORK / f"catchment_mask_{name}.dat"
    if not mask.exists():
        run([
            str(GDALRASTERIZE), "-l", "reach_catchments", "-a", "reach_id", "-init", "0", "-a_nodata", "0",
            "-te", str(min_x), str(min_y), str(max_x), str(max_y), "-ts", str(len(lons)), str(len(lats)),
            "-ot", "Int32", "-of", "ENVI", str(write_wgs84_catchments()), str(mask),
        ])
    return mask, (len(lats), len(lons))


def envi_array(source: Path, target: Path, shape: tuple[int, int]) -> np.memmap:
    if not target.exists():
        run([str(GDALTRANSLATE), "-of", "ENVI", str(source), str(target)])
    gdal_type = gdal_info(source)["bands"][0]["type"]
    dtype_map = {"Float32": np.dtype("<f4"), "Float64": np.dtype("<f8")}
    if gdal_type not in dtype_map:
        raise RuntimeError(f"Unsupported temperature raster type {gdal_type} for {source.name}")
    return np.memmap(target, dtype=dtype_map[gdal_type], mode="r", shape=shape)


def means_from_arrays(values: np.ndarray, mask: np.ndarray, nodata: float | None = None) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    valid = (mask > 0) & np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    ids = mask[valid].astype(np.int64, copy=False)
    totals = np.bincount(ids, weights=values[valid], minlength=231)
    counts = np.bincount(ids, minlength=231)
    total_cells = np.bincount(mask[mask > 0].astype(np.int64, copy=False), minlength=231)
    out = pd.DataFrame({"reach_id": np.arange(1, 231, dtype=np.int64), "n_grid_cells": counts[1:]})
    out["total_grid_cells"] = total_cells[1:]
    out["mean"] = np.divide(totals[1:], counts[1:], out=np.full(230, np.nan), where=counts[1:] > 0)
    out["valid_fraction"] = np.divide(counts[1:], total_cells[1:], out=np.full(230, np.nan), where=total_cells[1:] > 0)
    return out


def equal_area_centroids_wgs84() -> gpd.GeoSeries:
    """Centroids without the geographic-CRS centroid approximation."""
    projected = catchments().to_crs("EPSG:6933")
    return gpd.GeoSeries(projected.geometry.centroid, crs="EPSG:6933").to_crs("EPSG:4326")


def benz_groundwater_temperature() -> tuple[pd.DataFrame, dict[str, object]]:
    reference = BENZ_DATA / "Temp_2020_GWTable.tif"
    change = BENZ_DATA / "TempChange_2020to2000_GWTable.tif"
    mask_path, shape = raster_mask_from_reference(reference, "benz_groundwater_temperature", fallback_crs="EPSG:4326")
    mask = np.memmap(mask_path, dtype=np.int32, mode="r", shape=shape)
    info_ref, info_change = gdal_info(reference), gdal_info(change)
    nodata_ref = info_ref["bands"][0].get("noDataValue")
    nodata_change = info_change["bands"][0].get("noDataValue")
    temp_values = envi_array(reference, WORK / "benz_temp_2020_gwtable.dat", shape)
    delta_values = envi_array(change, WORK / "benz_delta_2000_2020_gwtable.dat", shape)
    temp = means_from_arrays(temp_values, mask, nodata_ref)
    delta = means_from_arrays(delta_values, mask, nodata_change)
    out = temp.loc[:, ["reach_id", "mean", "n_grid_cells", "valid_fraction"]].rename(columns={
        "mean": "groundwater_temperature_2020_c", "n_grid_cells": "groundwater_temperature_2020_n_cells",
        "valid_fraction": "groundwater_temperature_2020_valid_fraction",
    })
    out["groundwater_temperature_change_2000_2020_c"] = delta["mean"].to_numpy()
    out["groundwater_temperature_change_2000_2020_n_cells"] = delta["n_grid_cells"].to_numpy()
    out["groundwater_temperature_spatial_method"] = "catchment_grid_mean"
    # A single small catchment (reach 222) contains no 0.045-degree cell
    # centre.  A nearest valid source cell is a transparent and reproducible
    # fallback, rather than inventing a value or dropping the reach.
    gt = info_ref["geoTransform"]
    xs = float(gt[0]) + float(gt[1]) * (np.arange(shape[1]) + 0.5)
    ys = float(gt[3]) + float(gt[5]) * (np.arange(shape[0]) + 0.5)
    for row_index in out.index[out.groundwater_temperature_2020_c.isna()]:
        point = equal_area_centroids_wgs84().iloc[int(out.at[row_index, "reach_id"]) - 1]
        row = int(np.abs(ys - point.y).argmin())
        col = int(np.abs(xs - point.x).argmin())
        value = float(temp_values[row, col])
        change_value = float(delta_values[row, col])
        if not np.isfinite(value) or value == nodata_ref or not np.isfinite(change_value) or change_value == nodata_change:
            valid = np.isfinite(temp_values) & np.isfinite(delta_values) & (temp_values != nodata_ref) & (delta_values != nodata_change)
            candidates = np.argwhere(valid)
            nearest = candidates[int(((candidates[:, 0] - row) ** 2 + (candidates[:, 1] - col) ** 2).argmin())]
            value, change_value = float(temp_values[nearest[0], nearest[1]]), float(delta_values[nearest[0], nearest[1]])
        out.at[row_index, "groundwater_temperature_2020_c"] = value
        out.at[row_index, "groundwater_temperature_change_2000_2020_c"] = change_value
        out.at[row_index, "groundwater_temperature_2020_n_cells"] = 1
        out.at[row_index, "groundwater_temperature_change_2000_2020_n_cells"] = 1
        out.at[row_index, "groundwater_temperature_2020_valid_fraction"] = 1.0
        out.at[row_index, "groundwater_temperature_spatial_method"] = "nearest_valid_source_grid_cell_fallback"
    # The delivered TIFF has no unit tag.  Its input layer is ERA5 soil
    # temperature and the PRB values are in Kelvin; convert only after a
    # guarded range check so a future Celsius delivery cannot be double-shifted.
    median_temperature = float(np.nanmedian(out["groundwater_temperature_2020_c"]))
    if 200 < median_temperature < 350:
        out["groundwater_temperature_2020_c"] -= 273.15
        source_unit, output_unit = "K", "degC"
    elif -50 < median_temperature < 100:
        source_unit, output_unit = "degC", "degC"
    else:
        raise RuntimeError(f"Unrecognised Benz temperature scale (median={median_temperature})")
    out["groundwater_temperature_temporal_role"] = "annual_mean_2020_static_with_2000_2020_change; not_monthly_observation"
    out["groundwater_temperature_source"] = "Benz_etal_2024_Global_groundwater_warming"
    profile = {
        "reference_file": str(reference), "change_file": str(change), "grid_size": info_ref["size"],
        "reference_nodata": nodata_ref, "change_nodata": nodata_change,
        "groundwater_temperature_c_min": float(out.groundwater_temperature_2020_c.min()),
        "groundwater_temperature_c_max": float(out.groundwater_temperature_2020_c.max()),
        "groundwater_change_c_min": float(out.groundwater_temperature_change_2000_2020_c.min()),
        "groundwater_change_c_max": float(out.groundwater_temperature_change_2000_2020_c.max()),
        "source_temperature_unit": source_unit, "output_temperature_unit": output_unit,
    }
    return out, profile


def lmlt_monthly_by_reach() -> tuple[pd.DataFrame, dict[str, object]]:
    paths = sorted(LMLT_RAW.glob("era5_land_monthly_lmlt_prb_buffer_*.nc"))
    years = [int(re.search(r"(\d{4})$", path.stem).group(1)) for path in paths]
    if years != list(range(2006, 2023)):
        raise RuntimeError(f"Expected complete LMLT years 2006--2022, found: {years}")
    with xr.open_dataset(paths[0]) as sample:
        if list(sample.data_vars) != ["lmlt"]:
            raise RuntimeError(f"Unexpected LMLT variables: {list(sample.data_vars)}")
        lons, lats = sample.longitude.values, sample.latitude.values
    mask_path, shape = raster_mask_wgs84(lons, lats, "era5_land_lmlt_0p1deg")
    mask = np.memmap(mask_path, dtype=np.int32, mode="r", shape=shape)
    centroids = equal_area_centroids_wgs84()
    centroid_indices = [
        (int(np.abs(lats - point.y).argmin()), int(np.abs(lons - point.x).argmin())) for point in centroids
    ]
    records: list[pd.DataFrame] = []
    for path, expected_year in zip(paths, years, strict=True):
        with xr.open_dataset(path) as ds:
            values = ds["lmlt"].values
            times = pd.to_datetime(ds.valid_time.values)
            if values.shape != (12, *shape) or len(times) != 12 or set(times.year) != {expected_year}:
                raise RuntimeError(f"Unexpected LMLT time/grid dimensions in {path.name}")
            for index, timestamp in enumerate(times):
                stats = means_from_arrays(values[index], mask)
                stats["year"] = int(timestamp.year)
                stats["month"] = int(timestamp.month)
                stats["lake_mix_layer_temperature_c"] = stats.pop("mean") - 273.15
                stats["lmlt_spatial_method"] = "catchment_grid_mean"
                # Eight small PRB catchments contain no 0.1-degree cell centre.
                # Use the nearest valid ERA5-Land grid cell only for those
                # catchments and retain the fallback flag for downstream models.
                missing = stats.total_grid_cells.eq(0)
                for row_index in stats.index[missing]:
                    lat_index, lon_index = centroid_indices[int(stats.at[row_index, "reach_id"]) - 1]
                    value = float(values[index, lat_index, lon_index])
                    if not np.isfinite(value):
                        valid_locations = np.argwhere(np.isfinite(values[index]))
                        distance = (valid_locations[:, 0] - lat_index) ** 2 + (valid_locations[:, 1] - lon_index) ** 2
                        nearest = valid_locations[int(distance.argmin())]
                        value = float(values[index, nearest[0], nearest[1]])
                    stats.at[row_index, "lake_mix_layer_temperature_c"] = value - 273.15
                    stats.at[row_index, "n_grid_cells"] = 1
                    stats.at[row_index, "valid_fraction"] = 1.0
                    stats.at[row_index, "lmlt_spatial_method"] = "nearest_valid_era5_grid_cell_fallback"
                stats = stats.rename(columns={
                    "n_grid_cells": "lmlt_valid_grid_cells", "total_grid_cells": "lmlt_total_grid_cells",
                    "valid_fraction": "lmlt_valid_fraction",
                })
                records.append(stats)
    out = pd.concat(records, ignore_index=True).loc[:, [
        "reach_id", "year", "month", "lake_mix_layer_temperature_c", "lmlt_valid_grid_cells",
        "lmlt_total_grid_cells", "lmlt_valid_fraction", "lmlt_spatial_method",
    ]]
    out["surface_temperature_role"] = "ERA5-Land modelled lake_mix_layer_temperature_proxy; not_observed_stream_temperature"
    out["surface_temperature_source"] = "ERA5-Land_monthly_means_lmlt"
    profile = {
        "raw_directory": str(LMLT_RAW), "years": [int(years[0]), int(years[-1])], "n_raw_files": len(paths),
        "raw_grid_shape": [int(shape[0]), int(shape[1])],
        "temperature_c_min": float(out.lake_mix_layer_temperature_c.min()),
        "temperature_c_max": float(out.lake_mix_layer_temperature_c.max()),
        "minimum_valid_fraction": float(out.lmlt_valid_fraction.min()),
    }
    return out, profile


def validate(groundwater: pd.DataFrame, surface: pd.DataFrame) -> dict[str, object]:
    checks = {
        "groundwater_rows_230": len(groundwater) == 230,
        "groundwater_unique_reach": groundwater.reach_id.nunique() == 230,
        "groundwater_complete": groundwater[["groundwater_temperature_2020_c", "groundwater_temperature_change_2000_2020_c"]].notna().all().all(),
        "groundwater_plausible_c": groundwater.groundwater_temperature_2020_c.between(-5, 50).all(),
        "surface_rows_46920": len(surface) == 230 * 17 * 12,
        "surface_unique_reach_month": not surface.duplicated(["reach_id", "year", "month"]).any(),
        "surface_complete": surface.lake_mix_layer_temperature_c.notna().all(),
        "surface_month_valid": surface.month.between(1, 12).all(),
        "surface_plausible_c": surface.lake_mix_layer_temperature_c.between(-10, 45).all(),
        "surface_has_proxy_label": surface.surface_temperature_role.str.contains("not_observed_stream_temperature", regex=False).all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Temperature covariate QA failed: {failed}")
    return {"status": "PASS", "checks": checks, "created_utc": datetime.now(timezone.utc).isoformat()}


def update_data_catalog() -> None:
    """Keep the database's declared complete catalog aligned with new inputs."""
    path = PROVENANCE / "data_catalog.csv"
    columns = ["domain", "dataset", "coverage", "grain", "database_status", "source_path"]
    existing = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame(columns=columns)
    additions = pd.DataFrame([
        {
            "domain": "groundwater thermal state",
            "dataset": "Benz et al. (2024) global groundwater temperature",
            "coverage": "2020 annual mean at groundwater-table depth; 2000–2020 change",
            "grain": "global 0.044915764° grid; PRB reach-static",
            "database_status": "ready: static thermal covariate, explicitly not a monthly groundwater-temperature series",
            "source_path": str(READY_STATIC / "groundwater_temperature_benz_2024_by_reach.parquet"),
        },
        {
            "domain": "surface-water thermal proxy",
            "dataset": "ERA5-Land lake mix-layer temperature (lmlt)",
            "coverage": "2006–2022 monthly",
            "grain": "0.1° grid; PRB reach-month",
            "database_status": "ready as modelled lake-temperature proxy; not observed stream-water temperature",
            "source_path": str(READY_MONTHLY / "lake_mix_layer_temperature_era5_land_by_reach_month_2006_2022.parquet"),
        },
    ])
    combined = pd.concat([existing.loc[~existing.dataset.isin(additions.dataset)], additions], ignore_index=True)
    combined.loc[:, columns].to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    for directory in (READY_STATIC, READY_MONTHLY, PROVENANCE, WORK):
        directory.mkdir(parents=True, exist_ok=True)
    benz_manifest = materialize_benz_source()
    groundwater, groundwater_profile = benz_groundwater_temperature()
    surface, surface_profile = lmlt_monthly_by_reach()
    qa = validate(groundwater, surface)
    groundwater.to_parquet(READY_STATIC / "groundwater_temperature_benz_2024_by_reach.parquet", index=False)
    surface.to_parquet(READY_MONTHLY / "lake_mix_layer_temperature_era5_land_by_reach_month_2006_2022.parquet", index=False)
    update_data_catalog()
    catalog = {
        "benz_2024_groundwater_temperature": {
            **benz_manifest, **groundwater_profile,
            "model_ready_file": str(READY_STATIC / "groundwater_temperature_benz_2024_by_reach.parquet"),
            "model_use": "static groundwater thermal-state and 2000--2020 warming covariates only",
        },
        "era5_land_lake_mix_layer_temperature": {
            **surface_profile,
            "model_ready_file": str(READY_MONTHLY / "lake_mix_layer_temperature_era5_land_by_reach_month_2006_2022.parquet"),
            "model_use": "monthly lake-temperature proxy; do not use as observed river-water temperature",
        },
    }
    (PROVENANCE / "temperature_covariates_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (PROVENANCE / "temperature_covariates_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "groundwater_reaches": len(groundwater), "surface_reach_months": len(surface),
        "groundwater_file": str(READY_STATIC / "groundwater_temperature_benz_2024_by_reach.parquet"),
        "surface_file": str(READY_MONTHLY / "lake_mix_layer_temperature_era5_land_by_reach_month_2006_2022.parquet"),
        "qa": qa["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
