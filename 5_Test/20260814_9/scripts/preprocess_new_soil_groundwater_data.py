"""Prepare newly ingested soil/groundwater datasets at PRB reach-catchment grain.

Outputs are deliberately covariates or validation observations, never treated
as observed TN river loads.  All writes are limited to 20260814_9.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260814_9"
RAW = ROOT / "0_reach_topology" / "data" / "raw"
SPATIAL = RUN / "inputs" / "spatial"
READY = RUN / "inputs" / "model_ready"
WORK = RUN / "work" / "new_soil_groundwater"
GDAL_BIN = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin")
GDALINFO = GDAL_BIN / "gdalinfo.exe"
GDALRASTERIZE = GDAL_BIN / "gdal_rasterize.exe"
GDALTRANSLATE = GDAL_BIN / "gdal_translate.exe"
OGR2OGR = GDAL_BIN / "ogr2ogr.exe"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def gdal_info(path: Path) -> dict:
    return json.loads(subprocess.check_output([str(GDALINFO), "-json", str(path)], text=True, encoding="utf-8"))


def catchments() -> gpd.GeoDataFrame:
    frame = gpd.read_file(SPATIAL / "reach_catchments.shp").loc[:, ["reach_id", "geometry"]]
    if len(frame) != 230 or frame.reach_id.nunique() != 230:
        raise RuntimeError("Expected exactly 230 unique PRB reach catchments")
    invalid = ~frame.geometry.is_valid
    if invalid.any():
        # Repair only this in-memory processing copy; the hydrology baseline is
        # intentionally read-only.  The repair prevents GDAL clip failures.
        frame.loc[invalid, "geometry"] = frame.loc[invalid, "geometry"].make_valid()
    return frame.sort_values("reach_id").reset_index(drop=True)


def write_reprojected_catchments(target: Path, crs: object) -> Path:
    """Create an owned temporary GPKG in the target grid CRS."""
    if target.exists():
        existing = gpd.read_file(target)
        if existing.geometry.is_valid.all() and len(existing) == 230:
            return target
        # Only replace a malformed temporary processing copy under RUN/work.
        for path in (target, target.with_suffix(".gpkg-wal"), target.with_suffix(".gpkg-shm")):
            if path.exists():
                path.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    projected = catchments().to_crs(crs)
    # Coordinate transformation of a few very detailed catchment boundaries
    # can introduce self-intersections; a zero-width buffer repairs polygon
    # topology without changing the source baseline file.
    invalid = ~projected.geometry.is_valid
    if invalid.any():
        projected.loc[invalid, "geometry"] = projected.loc[invalid, "geometry"].buffer(0)
    if not projected.geometry.is_valid.all():
        raise RuntimeError(f"Unable to create valid temporary catchments for {target.name}")
    projected.to_file(target, layer="reach_catchments", driver="GPKG")
    return target


def raster_mask_from_reference(reference: Path, name: str, fallback_crs: str | None = None) -> tuple[Path, tuple[int, int]]:
    """Rasterize catchment IDs onto a reference grid, preserving exact extent."""
    info = gdal_info(reference)
    width, height = map(int, info["size"])
    gt = info["geoTransform"]
    min_x, max_y = float(gt[0]), float(gt[3])
    max_x = min_x + width * float(gt[1])
    min_y = max_y + height * float(gt[5])
    crs = info.get("coordinateSystem", {}).get("wkt", fallback_crs)
    if crs is None:
        raise RuntimeError(f"Reference raster has no CRS and no declared fallback: {reference}")
    vector = write_reprojected_catchments(WORK / f"catchments_{name}.gpkg", crs)
    mask = WORK / f"catchment_mask_{name}.dat"
    if not mask.exists():
        run([
            str(GDALRASTERIZE), "-l", "reach_catchments", "-a", "reach_id", "-init", "0", "-a_nodata", "0",
            "-a_srs", str(crs), "-te", str(min_x), str(min_y), str(max_x), str(max_y), "-ts", str(width), str(height),
            "-ot", "Int32", "-of", "ENVI", str(vector), str(mask),
        ])
    return mask, (height, width)


def raster_mask_wgs84(lons: np.ndarray, lats: np.ndarray, name: str) -> tuple[Path, tuple[int, int]]:
    """Rasterize PRB catchment IDs on an xarray grid whose coordinates are centers."""
    width, height = len(lons), len(lats)
    dx = float(np.median(np.diff(lons)))
    dy = float(np.median(np.abs(np.diff(lats))))
    min_x, max_x = float(lons.min() - dx / 2), float(lons.max() + dx / 2)
    min_y, max_y = float(lats.min() - dy / 2), float(lats.max() + dy / 2)
    vector = write_reprojected_catchments(WORK / f"catchments_{name}.gpkg", "EPSG:4326")
    mask = WORK / f"catchment_mask_{name}.dat"
    if not mask.exists():
        run([
            str(GDALRASTERIZE), "-l", "reach_catchments", "-a", "reach_id", "-init", "0", "-a_nodata", "0",
            "-te", str(min_x), str(min_y), str(max_x), str(max_y), "-ts", str(width), str(height),
            "-ot", "Int32", "-of", "ENVI", str(vector), str(mask),
        ])
    return mask, (height, width)


def means_from_arrays(
    values: np.ndarray, mask: np.ndarray, nodata: float | int | None = None, minimum_valid: float | None = None
) -> pd.DataFrame:
    values = np.asarray(values)
    valid = (mask > 0) & np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    if minimum_valid is not None:
        valid &= values >= minimum_valid
    ids = mask[valid].astype(np.int64, copy=False)
    totals = np.bincount(ids, weights=values[valid].astype(float, copy=False), minlength=231)
    counts = np.bincount(ids, minlength=231)
    out = pd.DataFrame({"reach_id": np.arange(1, 231), "n_cells": counts[1:]} )
    out["mean"] = np.divide(totals[1:], counts[1:], out=np.full(230, np.nan), where=counts[1:] > 0)
    return out


def envi_array(source: Path, target: Path, shape: tuple[int, int], dtype: str) -> np.memmap:
    if not target.exists():
        run([str(GDALTRANSLATE), "-of", "ENVI", str(source), str(target)])
    return np.memmap(target, dtype=np.dtype(dtype), mode="r", shape=shape)


def primary_soil_tn() -> pd.DataFrame:
    root = RAW / "soil" / "china_soil_properties_2010_2018_1km" / "data" / "source_bundle"
    reference = root / "tn05_1km.tif"
    mask_path, shape = raster_mask_from_reference(reference, "china_soil_1km")
    mask = np.memmap(mask_path, dtype=np.int32, mode="r", shape=shape)
    layers = [("0_5", 5), ("5_15", 10), ("15_30", 15), ("30_60", 30), ("60_100", 40), ("100_200", 100)]
    source_stems = {"0_5": "tn05", "5_15": "tn515", "15_30": "tn1530", "30_60": "tn3060", "60_100": "tn60100", "100_200": "tn100200"}
    output = pd.DataFrame({"reach_id": np.arange(1, 231)})
    weighted_100 = np.zeros(230); weight_100 = np.zeros(230)
    weighted_200 = np.zeros(230); weight_200 = np.zeros(230)
    for label, thickness in layers:
        src = root / f"{source_stems[label]}_1km.tif"
        info = gdal_info(src)
        nodata = info["bands"][0].get("noDataValue")
        arr = envi_array(src, WORK / f"{source_stems[label]}.dat", shape, "<i2")
        stats = means_from_arrays(arr, mask, nodata)
        output[f"soil_tn_{label}cm_g_kg"] = stats["mean"].to_numpy() / 100.0
        output[f"soil_tn_{label}cm_n_cells"] = stats["n_cells"].to_numpy()
        use = np.isfinite(stats["mean"].to_numpy())
        if label != "100_200":
            weighted_100[use] += stats.loc[use, "mean"].to_numpy() * thickness
            weight_100[use] += thickness
        weighted_200[use] += stats.loc[use, "mean"].to_numpy() * thickness
        weight_200[use] += thickness
    output["soil_tn_0_100cm_depth_weighted_g_kg"] = np.divide(weighted_100, weight_100, out=np.full(230, np.nan), where=weight_100 > 0) / 100.0
    output["soil_tn_0_200cm_depth_weighted_g_kg"] = np.divide(weighted_200, weight_200, out=np.full(230, np.nan), where=weight_200 > 0) / 100.0
    if output.filter(like="_g_kg").isna().any().any():
        raise RuntimeError("At least one reach lacks a China 1 km soil TN mean")
    return output


def csdl_backup_tn() -> pd.DataFrame:
    root = RAW / "soil" / "csdl_v2_10km" / "data" / "tn"
    source = root / "TN_0-5cm_10km.nc"
    with xr.open_dataset(source) as ds:
        lons, lats = ds.longitude.values, ds.latitude.values
    mask_path, shape = raster_mask_wgs84(lons, lats, "csdl_10km")
    mask = np.memmap(mask_path, dtype=np.int32, mode="r", shape=shape)
    output = pd.DataFrame({"reach_id": np.arange(1, 231)})
    for path in sorted(root.glob("TN_*cm_10km.nc")):
        with xr.open_dataset(path) as ds:
            variable = next(iter(ds.data_vars))
            values = ds[variable].values
        label = path.stem.removeprefix("TN_").removesuffix("cm_10km").replace("-", "_")
        stats = means_from_arrays(values, mask)
        output[f"csdl_v2_tn_{label}cm_native_mean"] = stats["mean"].to_numpy()
        output[f"csdl_v2_tn_{label}cm_n_cells"] = stats["n_cells"].to_numpy()
    return output


def parse_coordinate(value: object) -> float:
    """Parse decimal locations and published coordinate ranges as their midpoint."""
    if pd.isna(value):
        return np.nan
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))
    if not numbers:
        return np.nan
    vals = np.array([float(x) for x in numbers], dtype=float)
    return float(vals.mean())


def groundwater_nitrate_observations() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = RAW / "hydrology" / "groundwater_nitrate_global_1979_2022" / "data" / "global_nitrate_dataset"
    source = root / "measured_nitrate_dataset" / "nitrate_dataset_1979_2022.xlsx"
    frame = pd.read_excel(source)
    frame = frame.rename(columns={"ID": "observation_id", "NO3(mg/L)": "no3_mg_l"})
    frame["latitude"] = frame["latitude"].map(parse_coordinate)
    frame["longitude"] = frame["longitude"].map(parse_coordinate)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["no3_mg_l"] = pd.to_numeric(frame["no3_mg_l"], errors="coerce")
    valid = frame.latitude.between(-90, 90) & frame.longitude.between(-180, 180) & frame.year.between(1979, 2022) & frame.no3_mg_l.ge(0)
    clean = frame.loc[valid].copy()
    points = gpd.GeoDataFrame(clean, geometry=gpd.points_from_xy(clean.longitude, clean.latitude), crs="EPSG:4326").to_crs(catchments().crs)
    joined = gpd.sjoin(points, catchments(), how="inner", predicate="within").drop(columns=["index_right", "geometry"])
    cols = ["observation_id", "latitude", "longitude", "year", "no3_mg_l", "region_or_country", "source", "reach_id"]
    joined = pd.DataFrame(joined.loc[:, cols]).sort_values(["reach_id", "year", "observation_id"])
    profile = pd.DataFrame([{
        "raw_rows": len(frame), "valid_global_rows": len(clean), "prb_rows": len(joined),
        "prb_reaches": joined.reach_id.nunique(), "prb_year_min": joined.year.min() if len(joined) else np.nan,
        "prb_year_max": joined.year.max() if len(joined) else np.nan,
        "prb_no3_mg_l_median": joined.no3_mg_l.median() if len(joined) else np.nan,
    }])
    return joined, profile


def nitrate_decadal_rasters() -> pd.DataFrame:
    root = RAW / "hydrology" / "groundwater_nitrate_global_1979_2022" / "data" / "global_nitrate_dataset" / "decadal_aquiferNO3_avg_datasets"
    reference = root / "aquiferNO3_avg_2010s.tif"
    # The supplied nitrate GeoTIFF has an explicit geographic geotransform but
    # lacks a CRS tag. Its global -180..180 / latitude grid is documented as
    # WGS84, therefore assign EPSG:4326 only to the derived mask.
    mask_path, shape = raster_mask_from_reference(reference, "nitrate_5arcmin", fallback_crs="EPSG:4326")
    mask = np.memmap(mask_path, dtype=np.int32, mode="r", shape=shape)
    output = pd.DataFrame({"reach_id": np.arange(1, 231)})
    for src in sorted(root.glob("aquiferNO3_avg_*.tif")):
        decade = src.stem.rsplit("_", 1)[-1]
        arr = envi_array(src, WORK / f"{src.stem}.dat", shape, "<f4")
        # -2: no data; -1: aquifer has fewer than 20 observations. Neither is a concentration.
        stats = means_from_arrays(arr, mask, -9999, minimum_valid=0)
        output[f"groundwater_no3_{decade}_mg_l_mean"] = stats["mean"].to_numpy()
        output[f"groundwater_no3_{decade}_n_cells"] = stats["n_cells"].to_numpy()
    return output


def glhymps() -> pd.DataFrame:
    source = RAW / "soil" / "glhymps_v2_0" / "data" / "GLHYMPS.shp"
    clip = WORK / "glhymps_prb.gpkg"
    if clip.exists():
        try:
            if gpd.read_file(clip, rows=1).empty:
                raise ValueError("empty temporary clip")
        except Exception:
            for path in (clip, clip.with_suffix(".gpkg-wal"), clip.with_suffix(".gpkg-shm")):
                if path.exists():
                    path.unlink()
    if not clip.exists():
        # Reprojected PRB catchments are in the source CRS; ogr2ogr clips before
        # transferring features and avoids loading 1.8 million global polygons.
        crs = gpd.read_file(source, rows=1).crs
        boundary = write_reprojected_catchments(WORK / "catchments_glhymps.gpkg", crs)
        run([
            str(OGR2OGR), "-f", "GPKG", "-clipsrc", str(boundary), "-select",
            "logK_Ice_x,logK_Ferr_,Porosity_x,K_stdev_x1,GUM_K,Prmfrst", str(clip), str(source),
        ])
    rocks = gpd.read_file(clip)
    if rocks.empty:
        raise RuntimeError("GLHYMPS clip unexpectedly has no PRB polygons")
    cats = catchments().to_crs(rocks.crs)
    pieces = gpd.overlay(cats, rocks, how="intersection", keep_geom_type=True)
    pieces["area_m2"] = pieces.geometry.area
    fields = {"logK_Ferr_": "glhymps_log10_permeability_m2", "Porosity_x": "glhymps_porosity"}
    output = pd.DataFrame({"reach_id": np.arange(1, 231)})
    for source_field, target_field in fields.items():
        values = pd.to_numeric(pieces[source_field], errors="coerce")
        if source_field == "logK_Ferr_":
            values = values / 100.0
            valid = values.between(-30, 5)
        else:
            values = values / 100.0
            valid = values.between(0, 1)
        table = pd.DataFrame({"reach_id": pieces.reach_id, "value": values, "area": pieces.area_m2})
        table = table.loc[valid].copy()
        grouped = table.groupby("reach_id").apply(lambda x: np.average(x.value, weights=x.area), include_groups=False)
        output[target_field] = output.reach_id.map(grouped)
    output["glhymps_permeability_geomean_m2"] = 10.0 ** output["glhymps_log10_permeability_m2"]
    if output.isna().any().any():
        raise RuntimeError("At least one reach lacks a valid GLHYMPS permeability or porosity value")
    return output


def main() -> None:
    for directory in (WORK, READY / "static", READY / "observations", RUN / "inputs" / "provenance"):
        directory.mkdir(parents=True, exist_ok=True)
    soil = primary_soil_tn()
    csdl = csdl_backup_tn()
    gl = glhymps()
    static = soil.merge(csdl, on="reach_id", validate="one_to_one").merge(gl, on="reach_id", validate="one_to_one")
    static.to_parquet(READY / "static" / "soil_tn_glhymps_by_reach.parquet", index=False)
    nitrate_obs, nitrate_profile = groundwater_nitrate_observations()
    nitrate_obs.to_parquet(READY / "observations" / "groundwater_nitrate_observations_prb_1979_2022.parquet", index=False)
    nitrate_grid = nitrate_decadal_rasters()
    nitrate_grid.to_parquet(READY / "static" / "groundwater_nitrate_decadal_by_reach.parquet", index=False)
    nitrate_profile.to_csv(RUN / "inputs" / "provenance" / "groundwater_nitrate_observation_profile.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({
        "soil_static_rows": len(static), "soil_columns": list(static.columns), "nitrate_prb_observations": len(nitrate_obs),
        "nitrate_prb_reaches": int(nitrate_obs.reach_id.nunique()), "nitrate_grid_rows": len(nitrate_grid),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
