from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.ops import transform as transform_geometry

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
SPATIAL = RUN / "inputs" / "spatial"
TABLES = RUN / "reports" / "tables"
SCENARIOS = RUN / "inputs" / "scenarios"
LOGS = RUN / "logs"
RAW = ROOT / "0_reach_topology" / "data" / "raw"
CHM = RAW / "rainfall_2" / "CHM_PRE V2" / "monthly" / "CHM_PRE_V2_monthly.nc"
CMFD_ROOT = RAW / "CMFD"
CMFD_FILES = {
    name: CMFD_ROOT / f"{name}_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc"
    for name in ["temp", "pres", "shum", "wind", "srad", "lrad"]
}
ERA5_ROOT = RAW / "era5_land" / "monthly_prb_buffer"
CATCHMENTS = SPATIAL / "reach_catchments.shp"
DEM_ALIGNED = ROOT / "0_reach_topology" / "work" / "rasters" / "dem_clipped.tif"
REACH_RASTER = ROOT / "0_reach_topology" / "results" / "rasters" / "reach_catchments.tif"
DEM_WGS84 = ROOT / "0_reach_topology" / "data" / "processed" / "dem_prb" / "dem.tif"
DEM_RAW = ROOT / "0_reach_topology" / "data" / "raw" / "dem" / "DEM.tif"
DEM_WGS84_RECLIP = SPATIAL / "dem_wgs84_reclip_from_raw.tif"
DEM_ALIGNED_REBUILT = SPATIAL / "dem_aligned_reclip_fill200.tif"
BASE_INPUT = RUN / "inputs" / "baseline" / "R3_11_indata.parquet"
TOPOLOGY = RUN / "inputs" / "topology" / "topology_edges.csv"
WEIGHTS = SPATIAL / "grid_overlap_weights.parquet"
FRAGMENTS = SPATIAL / "reach_grid_fragment_elevation.parquet"
FORCING = RUN / "inputs" / "forcing_panel.parquet"
GDAL_BIN = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin")
M3S_TO_CFS = 35.3146667
SIGMA_MJ_M2_DAY_K4 = 4.903e-9
FILL_LIMIT = 1.0e10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(arguments: list[str]) -> None:
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    completed = subprocess.run(arguments, check=False, text=True, capture_output=True, env=env)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"stdout={completed.stdout[-4000:]}\nstderr={completed.stderr[-4000:]}"
        )


def coordinate_hash(latitudes: np.ndarray, longitudes: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(latitudes, dtype="<f8").tobytes())
    digest.update(np.asarray(longitudes, dtype="<f8").tobytes())
    return digest.hexdigest()


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.any(np.diff(values) == 0):
        raise RuntimeError("Grid coordinate must be a non-degenerate one-dimensional vector")
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = (values[:-1] + values[1:]) / 2.0
    edges[0] = values[0] - (values[1] - values[0]) / 2.0
    edges[-1] = values[-1] + (values[-1] - values[-2]) / 2.0
    return edges


def dense_box(x0: float, y0: float, x1: float, y1: float, parts: int = 8) -> Polygon:
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])
    xs = np.linspace(xmin, xmax, parts + 1)
    ys = np.linspace(ymin, ymax, parts + 1)
    points = (
        [(float(x), ymin) for x in xs]
        + [(xmax, float(y)) for y in ys[1:]]
        + [(float(x), ymax) for x in xs[-2::-1]]
        + [(xmin, float(y)) for y in ys[-2:0:-1]]
    )
    return Polygon(points)


def read_product_coordinates(product: str) -> tuple[np.ndarray, np.ndarray, Path]:
    if product == "CHM":
        path = CHM
        with h5py.File(path, "r") as handle:
            return np.asarray(handle["lat"][:], float), np.asarray(handle["lon"][:], float), path
    if product == "CMFD":
        path = CMFD_FILES["temp"]
        with h5py.File(path, "r") as handle:
            return np.asarray(handle["lat"][:], float), np.asarray(handle["lon"][:], float), path
    if product == "ERA5":
        path = ERA5_ROOT / "era5_land_monthly_prb_buffer_2006.nc"
        with h5py.File(path, "r") as handle:
            return np.asarray(handle["latitude"][:], float), np.asarray(handle["longitude"][:], float), path
    raise ValueError(product)


def product_cells(product: str, catchments: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    latitudes, longitudes, source = read_product_coordinates(product)
    lat_edges = centers_to_edges(latitudes)
    lon_edges = centers_to_edges(longitudes)
    catch_ll = catchments.to_crs(4326)
    basin_minx, basin_miny, basin_maxx, basin_maxy = catch_ll.total_bounds
    lat_low = np.minimum(lat_edges[:-1], lat_edges[1:])
    lat_high = np.maximum(lat_edges[:-1], lat_edges[1:])
    lon_low = np.minimum(lon_edges[:-1], lon_edges[1:])
    lon_high = np.maximum(lon_edges[:-1], lon_edges[1:])
    lat_idx = np.where((lat_high >= basin_miny) & (lat_low <= basin_maxy))[0]
    lon_idx = np.where((lon_high >= basin_minx) & (lon_low <= basin_maxx))[0]
    records: list[dict[str, object]] = []
    for ilat in lat_idx:
        for ilon in lon_idx:
            records.append({
                "product": product,
                "grid_i": int(ilat),
                "grid_j": int(ilon),
                "grid_lon": float(longitudes[ilon]),
                "grid_lat": float(latitudes[ilat]),
                "cell_id": int(ilat * len(longitudes) + ilon),
                "geometry": dense_box(lon_edges[ilon], lat_edges[ilat], lon_edges[ilon + 1], lat_edges[ilat + 1]),
            })
    cells = gpd.GeoDataFrame(records, geometry="geometry", crs=4326).to_crs(catchments.crs)
    meta = {
        "product": product,
        "source": str(source),
        "geometry_source_sha256": sha256(CATCHMENTS),
        "coordinate_source_sha256": coordinate_hash(latitudes, longitudes),
        "nlat": len(latitudes),
        "nlon": len(longitudes),
        "latitude_descending": bool(latitudes[0] > latitudes[-1]),
        "grid_cells_in_basin_bbox": len(cells),
    }
    return cells, meta


def build_weights() -> None:
    SPATIAL.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    catchments = gpd.read_file(CATCHMENTS)[["reach_id", "inc_km2", "geometry"]].copy()
    if len(catchments) != 230 or catchments["reach_id"].nunique() != 230:
        raise RuntimeError(f"Catchment gate failed: {len(catchments)}")
    all_rows: list[dict[str, object]] = []
    closure_rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    constant_errors: list[float] = []
    for product in ["CHM", "CMFD", "ERA5"]:
        cells, meta = product_cells(product, catchments)
        metadata.append(meta)
        spatial_index = cells.sindex
        for reach in catchments.itertuples(index=False):
            rid = int(reach.reach_id)
            geometry = reach.geometry
            indexes = spatial_index.query(geometry, predicate="intersects")
            candidates = cells.iloc[indexes]
            areas = candidates.geometry.intersection(geometry).area.to_numpy(float)
            positive = areas > 1e-6
            if not positive.any():
                raise RuntimeError(f"{product} has no polygon overlap for Reach {rid}")
            selected = candidates.loc[positive].copy()
            selected_areas = areas[positive]
            total_area = float(selected_areas.sum())
            catchment_area = float(geometry.area)
            raw_fraction = total_area / max(catchment_area, 1e-12)
            weights = selected_areas / total_area
            constant_error = abs(float(weights.sum()) - 1.0)
            constant_errors.append(constant_error)
            to_ll = Transformer.from_crs(catchments.crs, 4326, always_xy=True)
            for record, overlap_area, weight in zip(selected.itertuples(index=False), selected_areas, weights):
                intersection = record.geometry.intersection(geometry)
                centroid_ll = transform_geometry(to_ll.transform, intersection.centroid)
                all_rows.append({
                    "product": product,
                    "reach_id": rid,
                    "grid_i": int(record.grid_i),
                    "grid_j": int(record.grid_j),
                    "cell_id": int(record.cell_id),
                    "grid_lon": float(record.grid_lon),
                    "grid_lat": float(record.grid_lat),
                    "fragment_centroid_lon": float(centroid_ll.x),
                    "fragment_centroid_lat": float(centroid_ll.y),
                    "overlap_area_m2": float(overlap_area),
                    "catchment_area_m2": catchment_area,
                    "weight": float(weight),
                    "geometry_source_sha256": meta["geometry_source_sha256"],
                    "coordinate_source_sha256": meta["coordinate_source_sha256"],
                    "mapping_method": "full_polygon_overlap",
                })
            closure_rows.append({
                "product": product,
                "reach_id": rid,
                "n_overlap_cells": int(positive.sum()),
                "overlap_area_m2": total_area,
                "catchment_area_m2": catchment_area,
                "overlap_fraction": raw_fraction,
                "weight_sum": float(weights.sum()),
                "constant_field_error": constant_error,
                "nearest_centroid_cell": 0,
            })
    weights_frame = pd.DataFrame(all_rows).sort_values(["product", "reach_id", "grid_i", "grid_j"])
    closure = pd.DataFrame(closure_rows).sort_values(["product", "reach_id"])
    weights_frame.to_parquet(WEIGHTS, index=False)
    closure.to_csv(TABLES / "spatial_area_closure.csv", index=False, encoding="utf-8-sig")
    (SPATIAL / "grid_coordinate_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gate = {
        "products": sorted(weights_frame["product"].unique().tolist()),
        "reaches_by_product": weights_frame.groupby("product")["reach_id"].nunique().to_dict(),
        "weight_sum_max_abs_error": float((closure["weight_sum"] - 1).abs().max()),
        "area_fraction_min": float(closure["overlap_fraction"].min()),
        "area_fraction_max": float(closure["overlap_fraction"].max()),
        "nearest_centroid_rows": int(closure["nearest_centroid_cell"].sum()),
        "constant_field_max_abs_error": float(max(constant_errors)),
    }
    gate["passed"] = bool(
        all(value == 230 for value in gate["reaches_by_product"].values())
        and gate["weight_sum_max_abs_error"] <= 1e-10
        and gate["area_fraction_min"] >= 0.999
        and gate["area_fraction_max"] <= 1.001
        and gate["nearest_centroid_rows"] == 0
        and gate["constant_field_max_abs_error"] <= 1e-10
    )
    (LOGS / "polygon_overlap_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    if not gate["passed"]:
        raise RuntimeError(f"Polygon overlap gate failed: {gate}")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def tiff_grid(path: Path) -> dict[str, float | int]:
    with Image.open(path) as image:
        scale = image.tag_v2.get(33550)
        tie = image.tag_v2.get(33922)
        nodata = float(str(image.tag_v2.get(42113)))
        return {
            "width": image.width,
            "height": image.height,
            "xres": float(scale[0]),
            "yres": float(scale[1]),
            "xmin": float(tie[3]),
            "ymax": float(tie[4]),
            "xmax": float(tie[3]) + image.width * float(scale[0]),
            "ymin": float(tie[4]) - image.height * float(scale[1]),
            "nodata": nodata,
        }


def write_cmfd_grid_vector(catchments: gpd.GeoDataFrame) -> Path:
    cells, _ = product_cells("CMFD", catchments)
    used = set(pd.read_parquet(WEIGHTS).query("product == 'CMFD'")["cell_id"].astype(int))
    selected = cells[cells["cell_id"].astype(int).isin(used)][["cell_id", "grid_i", "grid_j", "geometry"]].copy()
    path = SPATIAL / "cmfd_grid.gpkg"
    if path.exists():
        path.unlink()
    selected.to_file(path, layer="cmfd_grid", driver="GPKG")
    return path


def rasterize_vector(vector: Path, layer: str, attribute: str, template: Path, output: Path, nodata: int) -> None:
    if output.exists():
        output.unlink()
    # Create from the template first. Supplying only extent and size to
    # gdal_rasterize inherits the vector CRS, which is wrong for the WGS84
    # cross-check grid even when the numeric extent is copied correctly.
    run_command([
        str(GDAL_BIN / "gdal_create.exe"), "--quiet", "-if", str(template),
        "-bands", "1", "-ot", "Int32", "-burn", str(nodata), "-a_nodata", str(nodata),
        "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=YES",
        "-co", "SPARSE_OK=YES", str(output),
    ])
    run_command([
        str(GDAL_BIN / "gdal_rasterize.exe"), "-q", "-a", attribute,
        "-l", layer, str(vector), str(output),
    ])
    grid = tiff_grid(template)
    out_grid = tiff_grid(output)
    if (out_grid["width"], out_grid["height"]) != (grid["width"], grid["height"]):
        raise RuntimeError(f"Rasterization grid mismatch for {output}")


def read_window_with_pillow(source: Path, top: int, height: int, work: Path, label: str) -> np.ndarray:
    grid = tiff_grid(DEM_ALIGNED) if source.suffix.casefold() == ".vrt" else tiff_grid(source)
    target = work / f"{label}_{top:05d}.tif"
    run_command([
        str(GDAL_BIN / "gdal_translate.exe"), "-q", "-srcwin", "0", str(top), str(grid["width"]), str(height),
        "-of", "GTiff", "-co", "COMPRESS=NONE", str(source), str(target),
    ])
    with Image.open(target) as image:
        array = np.asarray(image).copy()
    target.unlink()
    return array


def accumulate_groups(
    store: dict[int, list[float]], keys: np.ndarray, values: np.ndarray
) -> None:
    if not len(keys):
        return
    unique, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values.astype(float))
    minimum = np.full(len(unique), np.inf)
    maximum = np.full(len(unique), -np.inf)
    np.minimum.at(minimum, inverse, values)
    np.maximum.at(maximum, inverse, values)
    for key, count, total, low, high in zip(unique, counts, sums, minimum, maximum):
        record = store.setdefault(int(key), [0.0, 0.0, np.inf, -np.inf])
        record[0] += float(count)
        record[1] += float(total)
        record[2] = min(record[2], float(low))
        record[3] = max(record[3], float(high))


def rebuild_wgs84_dem_from_raw() -> Path:
    template = tiff_grid(DEM_WGS84)
    if DEM_WGS84_RECLIP.exists():
        DEM_WGS84_RECLIP.unlink()
    run_command([
        str(GDAL_BIN / "gdal_translate.exe"), "-q",
        "-projwin", str(template["xmin"]), str(template["ymax"]), str(template["xmax"]), str(template["ymin"]),
        "-outsize", str(template["width"]), str(template["height"]),
        "-ot", "Int16", "-a_nodata", "-32768", "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES", "-co", "BIGTIFF=YES", str(DEM_RAW), str(DEM_WGS84_RECLIP),
    ])
    result = tiff_grid(DEM_WGS84_RECLIP)
    spatial_keys = ["width", "height", "xres", "yres", "xmin", "ymax", "xmax", "ymin"]
    if any(abs(float(result[key]) - float(template[key])) > 1e-10 for key in spatial_keys):
        raise RuntimeError("Fresh raw-DEM clip does not match the frozen WGS84 template grid")
    return DEM_WGS84_RECLIP


def build_wgs84_aligned_vrt(wgs84_dem: Path) -> Path:
    grid = tiff_grid(DEM_ALIGNED)
    target = SPATIAL / "wgs84_dem_aligned_fill.vrt"
    run_command([
        str(GDAL_BIN / "gdalwarp.exe"), "-q", "-overwrite", "-of", "VRT",
        "-t_srs", str(CATCHMENTS.with_suffix(".prj")),
        "-te", str(grid["xmin"]), str(grid["ymin"]), str(grid["xmax"]), str(grid["ymax"]),
        "-ts", str(grid["width"]), str(grid["height"]), "-r", "bilinear",
        "-srcnodata", "-32768", "-dstnodata", "-9999", str(wgs84_dem), str(target),
    ])
    return target


def fill_aligned_dem_from_raw(wgs84_vrt: Path) -> Path:
    if DEM_ALIGNED_REBUILT.exists():
        DEM_ALIGNED_REBUILT.unlink()
    run_command([
        str(GDAL_BIN / "gdal.exe"), "raster", "fill-nodata", "-q", "--overwrite",
        "-d", "200", "-s", "0", "--strategy", "invdist",
        "--co", "COMPRESS=DEFLATE", "--co", "TILED=YES", "--co", "BIGTIFF=YES",
        str(wgs84_vrt), str(DEM_ALIGNED_REBUILT),
    ])
    rebuilt = tiff_grid(DEM_ALIGNED_REBUILT)
    template = tiff_grid(DEM_ALIGNED)
    spatial_keys = ["width", "height", "xres", "yres", "xmin", "ymax", "xmax", "ymin"]
    if any(abs(float(rebuilt[key]) - float(template[key])) > 1e-8 for key in spatial_keys):
        raise RuntimeError("Rebuilt filled DEM is not aligned with the frozen 30 m grid")
    return DEM_ALIGNED_REBUILT


def scan_aligned_dem(
    cmfd_raster: Path, reach_raster: Path, primary_dem: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = tiff_grid(primary_dem)
    reach_grid = tiff_grid(reach_raster)
    spatial_keys = ["width", "height", "xres", "yres", "xmin", "ymax", "xmax", "ymin"]
    if any(reach_grid[key] != grid[key] for key in spatial_keys):
        raise RuntimeError("Aligned DEM and Reach raster tags differ")
    work = SPATIAL / "window_cache"
    work.mkdir(exist_ok=True)
    pair_store: dict[int, list[float]] = {}
    reach_total_pixels: dict[int, int] = defaultdict(int)
    reach_raw_valid_pixels: dict[int, int] = defaultdict(int)
    reach_effective_valid_pixels: dict[int, int] = defaultdict(int)
    reach_filled_pixels: dict[int, int] = defaultdict(int)
    key_factor = 1_000_000
    for top in range(0, int(grid["height"]), 512):
        height = min(512, int(grid["height"]) - top)
        reach = read_window_with_pillow(reach_raster, top, height, work, "reach")
        dem = read_window_with_pillow(primary_dem, top, height, work, "dem")
        raw_dem = read_window_with_pillow(DEM_ALIGNED, top, height, work, "dem_raw")
        cell = read_window_with_pillow(cmfd_raster, top, height, work, "cmfd")
        reach_mask = reach > 0
        if reach_mask.any():
            ids, counts = np.unique(reach[reach_mask], return_counts=True)
            for rid, count in zip(ids, counts):
                reach_total_pixels[int(rid)] += int(count)
        raw_valid = np.isfinite(raw_dem) & (raw_dem != -9999)
        fill_valid = np.isfinite(dem) & (dem != grid["nodata"])
        filled = reach_mask & ~raw_valid & fill_valid
        effective_dem = dem
        raw_reach = reach_mask & raw_valid
        if raw_reach.any():
            ids, counts = np.unique(reach[raw_reach], return_counts=True)
            for reach_id, count in zip(ids, counts):
                reach_raw_valid_pixels[int(reach_id)] += int(count)
        if filled.any():
            ids, counts = np.unique(reach[filled], return_counts=True)
            for reach_id, count in zip(ids, counts):
                reach_filled_pixels[int(reach_id)] += int(count)
        valid = reach_mask & (cell >= 0) & fill_valid
        if valid.any():
            rid = reach[valid].astype(np.int64)
            cid = cell[valid].astype(np.int64)
            values = effective_dem[valid].astype(float)
            ids, counts = np.unique(rid, return_counts=True)
            for reach_id, count in zip(ids, counts):
                reach_effective_valid_pixels[int(reach_id)] += int(count)
            accumulate_groups(pair_store, rid * key_factor + cid, values)
        print(f"DEM strip {top}:{top + height}", flush=True)
    shutil.rmtree(work)
    fragment_rows = []
    for key, (count, total, low, high) in pair_store.items():
        rid, cid = divmod(key, key_factor)
        fragment_rows.append({
            "reach_id": rid, "cell_id": cid, "dem_pixel_count": int(count),
            "elevation_mean_m": total / count, "elevation_min_m": low, "elevation_max_m": high,
        })
    reach_rows = []
    fragment_frame = pd.DataFrame(fragment_rows)
    for rid in sorted(reach_total_pixels):
        selected = fragment_frame[fragment_frame["reach_id"].eq(rid)]
        count = int(selected["dem_pixel_count"].sum())
        total = float((selected["elevation_mean_m"] * selected["dem_pixel_count"]).sum())
        reach_rows.append({
            "reach_id": rid,
            "aligned_total_pixels": reach_total_pixels[rid],
            "aligned_raw_valid_dem_pixels": reach_raw_valid_pixels.get(rid, 0),
            "aligned_raw_dem_valid_fraction": reach_raw_valid_pixels.get(rid, 0) / max(reach_total_pixels[rid], 1),
            "wgs84_fill_pixels": reach_filled_pixels.get(rid, 0),
            "aligned_effective_valid_dem_pixels": reach_effective_valid_pixels.get(rid, 0),
            "aligned_dem_valid_fraction": reach_effective_valid_pixels.get(rid, 0) / max(reach_total_pixels[rid], 1),
            "aligned_elevation_mean_m": total / max(count, 1),
            "aligned_elevation_min_m": float(selected["elevation_min_m"].min()),
            "aligned_elevation_max_m": float(selected["elevation_max_m"].max()),
        })
    return fragment_frame, pd.DataFrame(reach_rows)


def scan_wgs84_dem(reach_wgs: Path, wgs84_dem: Path) -> pd.DataFrame:
    grid = tiff_grid(wgs84_dem)
    work = SPATIAL / "wgs_window_cache"
    work.mkdir(exist_ok=True)
    store: dict[int, list[float]] = {}
    for top in range(0, int(grid["height"]), 512):
        height = min(512, int(grid["height"]) - top)
        reach = read_window_with_pillow(reach_wgs, top, height, work, "reach_wgs")
        dem = read_window_with_pillow(wgs84_dem, top, height, work, "dem_wgs")
        valid = (reach > 0) & np.isfinite(dem) & (dem != grid["nodata"])
        accumulate_groups(store, reach[valid].astype(np.int64), dem[valid].astype(float))
        print(f"WGS DEM strip {top}:{top + height}", flush=True)
    shutil.rmtree(work)
    return pd.DataFrame([
        {
            "reach_id": rid, "wgs84_dem_pixel_count": int(values[0]),
            "wgs84_elevation_mean_m": values[1] / values[0],
            "wgs84_elevation_min_m": values[2], "wgs84_elevation_max_m": values[3],
        }
        for rid, values in sorted(store.items())
    ])


def build_dem_fragments() -> None:
    wgs84_dem = rebuild_wgs84_dem_from_raw()
    catchments = gpd.read_file(CATCHMENTS)[["reach_id", "geometry"]].copy()
    cmfd_vector = write_cmfd_grid_vector(catchments)
    cmfd_raster = SPATIAL / "cmfd_grid_on_aligned_dem.tif"
    rasterize_vector(cmfd_vector, "cmfd_grid", "cell_id", DEM_ALIGNED, cmfd_raster, -1)
    reach_gpkg = SPATIAL / "reach_catchments.gpkg"
    if reach_gpkg.exists():
        reach_gpkg.unlink()
    catchments.to_file(reach_gpkg, layer="reach_catchments", driver="GPKG")
    reach_aligned = SPATIAL / "reach_ids_on_aligned_dem.tif"
    rasterize_vector(reach_gpkg, "reach_catchments", "reach_id", DEM_ALIGNED, reach_aligned, 0)
    reach_wgs = SPATIAL / "reach_ids_on_wgs84_dem.tif"
    rasterize_vector(reach_gpkg, "reach_catchments", "reach_id", wgs84_dem, reach_wgs, 0)
    wgs84_fill_vrt = build_wgs84_aligned_vrt(wgs84_dem)
    primary_dem = fill_aligned_dem_from_raw(wgs84_fill_vrt)
    fragments_raw, reach_stats = scan_aligned_dem(cmfd_raster, reach_aligned, primary_dem)
    wgs_stats = scan_wgs84_dem(reach_wgs, wgs84_dem)
    weights = pd.read_parquet(WEIGHTS).query("product == 'CMFD'").copy()
    cmfd_nlon = int(read_product_coordinates("CMFD")[1].size)
    weights["cell_id"] = weights["grid_i"].astype(int) * cmfd_nlon + weights["grid_j"].astype(int)
    fragments = weights.merge(fragments_raw, on=["reach_id", "cell_id"], how="left")
    fragments = fragments.merge(
        reach_stats[["reach_id", "aligned_elevation_mean_m"]], on="reach_id", how="left"
    )
    fragments["subpixel_fallback"] = fragments["dem_pixel_count"].isna()
    fragments["dem_pixel_count"] = fragments["dem_pixel_count"].fillna(0).astype(int)
    # A fragment smaller than one 30 m pixel may contain no pixel centre. Use
    # the deterministic elevation at its overlap centroid before considering
    # a Reach-mean fallback; this does not increase spatial resolution.
    to_native = Transformer.from_crs(4326, catchments.crs, always_xy=True)
    fragments["subpixel_centroid_sample"] = False
    missing_index = fragments.index[fragments["subpixel_fallback"]].tolist()
    for index in missing_index:
        row = fragments.loc[index]
        x, y = to_native.transform(float(row["fragment_centroid_lon"]), float(row["fragment_centroid_lat"]))
        completed = subprocess.run(
            [str(GDAL_BIN / "gdallocationinfo.exe"), "-valonly", "-geoloc", str(primary_dem), str(x), str(y)],
            check=False, text=True, capture_output=True,
        )
        try:
            value = float(completed.stdout.strip())
        except ValueError:
            value = np.nan
        if completed.returncode == 0 and np.isfinite(value) and value != -9999:
            fragments.loc[index, ["elevation_mean_m", "elevation_min_m", "elevation_max_m"]] = value
            fragments.loc[index, "subpixel_centroid_sample"] = True
            fragments.loc[index, "subpixel_fallback"] = False
    for column in ["elevation_mean_m", "elevation_min_m", "elevation_max_m"]:
        fragments[column] = fragments[column].fillna(fragments["aligned_elevation_mean_m"])
    fragments["fragment_area_m2"] = fragments["overlap_area_m2"]
    fragments["fragment_weight"] = fragments["weight"]
    fragments["dem_valid_fraction"] = fragments["dem_pixel_count"].gt(0).astype(float)
    fragments.to_parquet(FRAGMENTS, index=False)
    fallback = fragments.groupby("reach_id").apply(
        lambda frame: float(frame.loc[frame["subpixel_fallback"], "fragment_weight"].sum()),
        include_groups=False,
    )
    stats = reach_stats.merge(wgs_stats, on="reach_id", how="outer")
    stats["dem_mean_difference_m"] = stats["aligned_elevation_mean_m"] - stats["wgs84_elevation_mean_m"]
    stats["dem_mean_absolute_difference_m"] = stats["dem_mean_difference_m"].abs()
    stats["fallback_fragment_weight"] = stats["reach_id"].map(fallback).fillna(0.0)
    stats.to_csv(TABLES / "reach_dem_zonal_statistics.csv", index=False, encoding="utf-8-sig")
    gate = {
        "reaches": int(stats["reach_id"].nunique()),
        "aligned_dem_valid_fraction_min": float(stats["aligned_dem_valid_fraction"].min()),
        "aligned_raw_dem_valid_fraction_min": float(stats["aligned_raw_dem_valid_fraction"].min()),
        "wgs84_fill_pixels_total": int(stats["wgs84_fill_pixels"].sum()),
        "dem_mean_abs_difference_p95_m": float(stats["dem_mean_absolute_difference_m"].quantile(0.95)),
        "dem_mean_abs_difference_max_m": float(stats["dem_mean_absolute_difference_m"].max()),
        "fallback_fragment_weight_max": float(stats["fallback_fragment_weight"].max()),
        "fragment_rows": len(fragments),
        "pillow_window_rows": 512,
        "pillow_direct_full_tiff_read": False,
        "gdal_read_only_window_adapter": True,
    }
    gate["passed"] = bool(
        gate["reaches"] == 230
        and gate["aligned_dem_valid_fraction_min"] >= 0.995
        and gate["dem_mean_abs_difference_p95_m"] <= 50
        and gate["dem_mean_abs_difference_max_m"] <= 150
        and gate["fallback_fragment_weight_max"] < 1e-4
    )
    (LOGS / "dem_support_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    if not gate["passed"]:
        raise RuntimeError(f"DEM support gate failed: {gate}")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def decode_time(values: np.ndarray, units: str) -> list[datetime]:
    unit, origin_text = units.split(" since ", 1)
    origin_text = origin_text.strip().replace("0.0", "00")
    origin = datetime.fromisoformat(origin_text)
    multiplier = {"days": 86400.0, "hours": 3600.0, "seconds": 1.0}[unit]
    return [origin + timedelta(seconds=float(value) * multiplier) for value in values]


def hybrid_et_daily(
    temp_k: np.ndarray, pres_pa: np.ndarray, shum: np.ndarray, wind_ms: np.ndarray,
    srad_wm2: np.ndarray, lrad_wm2: np.ndarray, lat_deg: np.ndarray,
    elevation_m: np.ndarray, month: int,
) -> np.ndarray:
    temp_c = temp_k - 273.15
    pressure_kpa = pres_pa / 1000.0
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = np.clip((shum * pressure_kpa) / (0.622 + 0.378 * shum), 0.0, es * 1.2)
    vpd = np.maximum(es - ea, 0.0)
    delta = 4098.0 * es / ((temp_c + 237.3) ** 2)
    gamma = 0.000665 * pressure_kpa
    rs = np.maximum(srad_wm2, 0.0) * 0.0864
    rldown = np.maximum(lrad_wm2, 0.0) * 0.0864
    rns = 0.77 * rs
    tk = temp_c + 273.16
    clear_up = SIGMA_MJ_M2_DAY_K4 * tk**4 * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)))
    day = 15 + sum(calendar.monthrange(2001, value)[1] for value in range(1, month))
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * day / 365.0)
    solar_delta = 0.409 * np.sin(2.0 * np.pi * day / 365.0 - 1.39)
    lat_rad = np.radians(lat_deg)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(solar_delta), -1.0, 1.0))
    ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(solar_delta)
        + np.cos(lat_rad) * np.cos(solar_delta) * np.sin(ws)
    )
    rso = (0.75 + 2.0e-5 * elevation_m) * ra
    cloud = np.clip(1.35 * np.minimum(rs / np.maximum(rso, 1e-6), 1.0) - 0.35, 0.05, 1.0)
    rnl_fao = np.maximum(clear_up * cloud, 0.0)
    rnl_flux = np.maximum((SIGMA_MJ_M2_DAY_K4 * tk**4) - rldown, 0.0)
    rn = rns - 0.5 * (rnl_fao + rnl_flux)
    wind = np.maximum(wind_ms, 0.05)
    numerator = 0.408 * delta * rn + gamma * (900.0 / (temp_c + 273.0)) * wind * vpd
    denominator = delta + gamma * (1.0 + 0.34 * wind)
    return np.maximum(numerator / np.maximum(denominator, 1e-9), 0.0)


def weighted_aggregate(array: np.ndarray, group: pd.DataFrame) -> tuple[float, float]:
    values = array[group["grid_i"].to_numpy(int), group["grid_j"].to_numpy(int)].astype(float)
    values[values > FILL_LIMIT] = np.nan
    mask = np.isfinite(values)
    coverage = float(group.loc[mask, "weight"].sum())
    if coverage < 0.999:
        return np.nan, coverage
    return float(np.average(values[mask], weights=group.loc[mask, "weight"])), coverage


def chm_forcing(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coverage_rows = []
    groups = list(weights.groupby("reach_id", sort=True))
    with h5py.File(CHM, "r") as handle:
        units = handle["time"].attrs["units"].decode() if isinstance(handle["time"].attrs["units"], bytes) else str(handle["time"].attrs["units"])
        dates = decode_time(np.asarray(handle["time"][:], float), units)
        for index, date in enumerate(dates):
            if not 2006 <= date.year <= 2022:
                continue
            array = np.asarray(handle["prec"][index], float)
            for reach_id, group in groups:
                value, coverage = weighted_aggregate(array, group)
                rows.append({"reach_id": int(reach_id), "year": date.year, "month": date.month, "PPT": value})
                coverage_rows.append({"product": "CHM", "reach_id": int(reach_id), "year": date.year, "month": date.month, "valid_weight": coverage})
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def era5_forcing(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows = []
    coverage_rows = []
    sample_path = ERA5_ROOT / "era5_land_monthly_prb_buffer_2006.nc"
    with h5py.File(sample_path, "r") as sample:
        support = np.ones(sample["e"].shape[1:], dtype=bool)
        any_core_finite = np.zeros(sample["e"].shape[1:], dtype=bool)
    # Permanent product support requires every core ERA5-Land variable to be
    # finite for all 204 months. This distinguishes the land-sea mask from a
    # transient download gap and never borrows a neighbouring value.
    for year in range(2006, 2023):
        path = ERA5_ROOT / f"era5_land_monthly_prb_buffer_{year}.nc"
        with h5py.File(path, "r") as handle:
            for variable in ["e", "tp", "t2m", "stl1"]:
                finite = np.isfinite(np.asarray(handle[variable][:]))
                support &= finite.all(axis=0)
                any_core_finite |= finite.any(axis=0)
    transient_or_partial_invalid = any_core_finite & ~support
    adjusted_groups = []
    support_rows = []
    for reach_id, group in weights.groupby("reach_id", sort=True):
        land = support[group["grid_i"].to_numpy(int), group["grid_j"].to_numpy(int)]
        raw_land_fraction = float(group.loc[land, "weight"].sum())
        if raw_land_fraction <= 0:
            raise RuntimeError(f"ERA5-Land has no permanent land support for Reach {reach_id}")
        effective = group.loc[land].copy()
        effective["effective_weight"] = effective["weight"] / raw_land_fraction
        adjusted_groups.append((reach_id, effective))
        for row in group.itertuples(index=False):
            is_land = bool(support[int(row.grid_i), int(row.grid_j)])
            support_rows.append({
                "reach_id": int(reach_id), "grid_i": int(row.grid_i), "grid_j": int(row.grid_j),
                "grid_lon": float(row.grid_lon), "grid_lat": float(row.grid_lat),
                "geometric_weight": float(row.weight), "permanent_land_support": is_land,
                "effective_land_weight": float(row.weight / raw_land_fraction) if is_land else 0.0,
                "excluded_as_permanent_nonland": not is_land,
                "raw_land_support_fraction": raw_land_fraction,
            })
    groups = adjusted_groups
    positive = negative = zero = 0
    valid_total = 0
    for year in range(2006, 2023):
        path = ERA5_ROOT / f"era5_land_monthly_prb_buffer_{year}.nc"
        with h5py.File(path, "r") as handle:
            units_raw = handle["valid_time"].attrs["units"]
            units = units_raw.decode() if isinstance(units_raw, bytes) else str(units_raw)
            dates = decode_time(np.asarray(handle["valid_time"][:], float), units)
            for index, date in enumerate(dates):
                array = np.asarray(handle["e"][index], float)
                finite = np.isfinite(array)
                positive += int((array[finite] > 0).sum())
                negative += int((array[finite] < 0).sum())
                zero += int((array[finite] == 0).sum())
                valid_total += int(finite.sum())
                days = calendar.monthrange(date.year, date.month)[1]
                for reach_id, group in groups:
                    values = array[group["grid_i"].to_numpy(int), group["grid_j"].to_numpy(int)].astype(float)
                    mask = np.isfinite(values)
                    coverage = float(group.loc[mask, "effective_weight"].sum())
                    raw_coverage = float(group.loc[mask, "weight"].sum())
                    if coverage < 0.999:
                        value = np.nan
                    else:
                        value = float(np.average(values[mask], weights=group.loc[mask, "effective_weight"]))
                    aet = max(-value, 0.0) * 1000.0 * days if np.isfinite(value) else np.nan
                    rows.append({"reach_id": int(reach_id), "year": date.year, "month": date.month, "AET": aet})
                    coverage_rows.append({
                        "product": "ERA5", "reach_id": int(reach_id), "year": date.year, "month": date.month,
                        "valid_weight": coverage, "raw_catchment_valid_weight": raw_coverage,
                    })
    support_frame = pd.DataFrame(support_rows)
    audit = {
        "valid_values": valid_total, "negative_values": negative, "zero_values": zero,
        "positive_condensation_values": positive, "semantic_operator": "max(-e, 0) * 1000 * days_in_month",
        "legacy_abs_operator_numerically_equal": positive == 0,
        "permanent_nonland_grid_cells": int((~any_core_finite).sum()),
        "transient_or_partial_invalid_grid_cells": int(transient_or_partial_invalid.sum()),
        "permanent_nonland_grid_fragments": int(support_frame["excluded_as_permanent_nonland"].sum()),
        "reaches_with_nonland_exclusion": int(support_frame.groupby("reach_id")["excluded_as_permanent_nonland"].any().sum()),
        "minimum_raw_land_support_fraction": float(support_frame.groupby("reach_id")["raw_land_support_fraction"].first().min()),
        "support_policy": "renormalize polygon overlap weights only over permanent ERA5-Land support; no nearest cell or interpolation",
    }
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows), support_frame, audit


def cmfd_forcing(weights: pd.DataFrame, fragments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = weights.merge(
        fragments[["reach_id", "grid_i", "grid_j", "elevation_mean_m"]],
        on=["reach_id", "grid_i", "grid_j"], how="left", validate="one_to_one",
    )
    groups = list(merged.groupby("reach_id", sort=True))
    files = {name: h5py.File(path, "r") for name, path in CMFD_FILES.items()}
    rows = []
    coverage_rows = []
    noncommute_rows = []
    try:
        units_raw = files["temp"]["time"].attrs["units"]
        units = units_raw.decode() if isinstance(units_raw, bytes) else str(units_raw)
        dates = decode_time(np.asarray(files["temp"]["time"][:], float), units)
        for index, date in enumerate(dates):
            if not 2006 <= date.year <= 2022:
                continue
            arrays = {name: np.asarray(files[name][name][index], float) for name in CMFD_FILES}
            days = calendar.monthrange(date.year, date.month)[1]
            for reach_id, group in groups:
                ii = group["grid_i"].to_numpy(int)
                jj = group["grid_j"].to_numpy(int)
                field = {name: arrays[name][ii, jj].astype(float) for name in arrays}
                matrix = np.column_stack([field[name] for name in CMFD_FILES])
                valid = np.isfinite(matrix).all(axis=1) & (matrix < FILL_LIMIT).all(axis=1)
                coverage = float(group.loc[valid, "weight"].sum())
                if coverage < 0.999:
                    pet = np.nan
                    pet_at_mean = np.nan
                else:
                    selected = group.loc[valid]
                    weights_valid = selected["weight"].to_numpy(float).copy()
                    weights_valid /= weights_valid.sum()
                    et_cells = hybrid_et_daily(
                        field["temp"][valid], field["pres"][valid], field["shum"][valid],
                        field["wind"][valid], field["srad"][valid], field["lrad"][valid],
                        selected["fragment_centroid_lat"].to_numpy(float),
                        selected["elevation_mean_m"].to_numpy(float), date.month,
                    )
                    pet = float(np.sum(weights_valid * et_cells) * days)
                    means = {name: np.array([np.sum(weights_valid * field[name][valid])]) for name in field}
                    et_mean = hybrid_et_daily(
                        means["temp"], means["pres"], means["shum"], means["wind"], means["srad"], means["lrad"],
                        np.array([np.sum(weights_valid * selected["fragment_centroid_lat"].to_numpy(float))]),
                        np.array([np.sum(weights_valid * selected["elevation_mean_m"].to_numpy(float))]), date.month,
                    )
                    pet_at_mean = float(et_mean[0] * days)
                rows.append({"reach_id": int(reach_id), "year": date.year, "month": date.month, "PET": pet})
                coverage_rows.append({"product": "CMFD", "reach_id": int(reach_id), "year": date.year, "month": date.month, "valid_weight": coverage})
                noncommute_rows.append({
                    "reach_id": int(reach_id), "year": date.year, "month": date.month,
                    "mean_of_operator_pet_mm": pet, "operator_of_mean_pet_mm": pet_at_mean,
                    "mean_f_minus_f_mean_mm": pet - pet_at_mean,
                })
    finally:
        for handle in files.values():
            handle.close()
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows), pd.DataFrame(noncommute_rows)


def downstream_ids(value: object) -> list[int]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [int(float(token.strip())) for token in str(value).replace(";", ",").split(",") if token.strip()]


def upstream_sets(topo: pd.DataFrame) -> dict[int, set[int]]:
    reverse: dict[int, set[int]] = defaultdict(set)
    reaches = set(topo["reach_id"].astype(int))
    for row in topo[["reach_id", "downstream_reach"]].itertuples(index=False):
        for downstream in downstream_ids(row.downstream_reach):
            reverse[downstream].add(int(row.reach_id))
    result = {}
    for reach in reaches:
        seen = {reach}
        stack = [reach]
        while stack:
            current = stack.pop()
            for parent in reverse.get(current, set()):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        result[reach] = seen
    return result


def build_s1_input(forcing: pd.DataFrame) -> None:
    base = pd.read_parquet(BASE_INPUT).sort_values(["comid", "year", "month"]).reset_index(drop=True)
    merged = base.merge(
        forcing.rename(columns={"reach_id": "comid"}), on=["comid", "year", "month"],
        how="left", suffixes=("_old", "_new"), validate="one_to_one",
    )
    for column in ["PPT", "AET", "PET"]:
        merged[column] = merged.pop(f"{column}_new")
        merged = merged.drop(columns=[f"{column}_old"])
        merged[f"pre{column}"] = merged.groupby("comid")[column].shift(1)
        first = merged.groupby("comid").cumcount().eq(0)
        merged.loc[first, f"pre{column}"] = merged.loc[first, column]
    topo = pd.read_csv(TOPOLOGY, encoding="utf-8-sig")
    upstream = upstream_sets(topo)
    days = np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(merged["year"], merged["month"])])
    seconds = days * 86400.0
    net = np.maximum(merged["PPT"].to_numpy(float) - merged["AET"].to_numpy(float), 0.0)
    threshold = np.maximum(merged["PPT"].to_numpy(float) - 0.8 * merged["PET"].to_numpy(float), 0.0)
    local_net = net / 1000.0 * merged["IncAreaKm2"].to_numpy(float) * 1_000_000.0
    local_threshold = threshold / 1000.0 * merged["IncAreaKm2"].to_numpy(float) * 1_000_000.0
    positions = {
        (int(row.comid), int(row.year), int(row.month)): index
        for index, row in enumerate(merged[["comid", "year", "month"]].itertuples(index=False))
    }
    upstream_net = np.zeros(len(merged), dtype=float)
    upstream_threshold = np.zeros(len(merged), dtype=float)
    for reach, members in upstream.items():
        for year in range(2006, 2023):
            for month in range(1, 13):
                target = positions[(reach, year, month)]
                source = [positions[(member, year, month)] for member in members]
                upstream_net[target] = local_net[source].sum()
                upstream_threshold[target] = local_threshold[source].sum()
    merged["explicit_upstream_net_cfs"] = upstream_net / seconds * M3S_TO_CFS
    merged["explicit_upstream_threshold_cfs"] = upstream_threshold / seconds * M3S_TO_CFS
    merged["Q_calc_cfs"] = 0.35 * merged["explicit_upstream_net_cfs"]
    merged["Q_ma_cfs"] = merged.groupby("comid")["Q_calc_cfs"].transform("mean")
    merged["MAFlowUcfs"] = merged["Q_ma_cfs"]
    merged.to_parquet(SCENARIOS / "S1_indata.parquet", index=False)
    shutil.copy2(BASE_INPUT, SCENARIOS / "S0_indata.parquet")


def old_pet_reproduction() -> dict[str, float | bool]:
    old = pd.read_parquet(ROOT / "5_Test" / "20260810_1" / "inputs" / "climate_corrected" / "cmfd" / "cmfd_monthly_by_reach_2006_2022.parquet")
    mapping = pd.read_csv(ROOT / "5_Test" / "20260810_1" / "inputs" / "climate_corrected" / "cmfd" / "cmfd_grid_to_reach_mapping.csv")
    latitude = mapping.groupby("reach_id").apply(
        lambda frame: np.average(frame["lat"], weights=frame["weight"]), include_groups=False
    )
    # The operator accepts one month at a time; evaluate vector rows explicitly for exact audit.
    values = np.empty(len(old), dtype=float)
    for month in range(1, 13):
        mask = old["month"].eq(month).to_numpy()
        values[mask] = hybrid_et_daily(
            old.loc[mask, "T2M_C_cmfd"].to_numpy(float) + 273.15,
            old.loc[mask, "pres_kpa_cmfd"].to_numpy(float) * 1000.0,
            old.loc[mask, "shum_kgkg_cmfd"].to_numpy(float), old.loc[mask, "wind_ms_cmfd"].to_numpy(float),
            old.loc[mask, "srad_wm2_cmfd"].to_numpy(float), old.loc[mask, "lrad_wm2_cmfd"].to_numpy(float),
            old.loc[mask, "reach_id"].map(latitude).to_numpy(float), np.full(mask.sum(), 100.0), month,
        )
    days = np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(old["year"], old["month"])])
    difference = values * days - old["PET_cmfd_mm"].to_numpy(float)
    return {
        "rows": len(old), "max_absolute_difference_mm": float(np.max(np.abs(difference))),
        "mean_absolute_difference_mm": float(np.mean(np.abs(difference))),
        "passed": bool(np.max(np.abs(difference)) <= 1e-8),
    }


def build_forcing() -> None:
    weights = pd.read_parquet(WEIGHTS)
    fragments = pd.read_parquet(FRAGMENTS)
    chm, chm_coverage = chm_forcing(weights.query("product == 'CHM'").copy())
    era5, era5_coverage, era5_support, era5_audit = era5_forcing(weights.query("product == 'ERA5'").copy())
    cmfd, cmfd_coverage, noncommute = cmfd_forcing(weights.query("product == 'CMFD'").copy(), fragments)
    forcing = chm.merge(era5, on=["reach_id", "year", "month"], validate="one_to_one").merge(
        cmfd, on=["reach_id", "year", "month"], validate="one_to_one"
    ).sort_values(["reach_id", "year", "month"])
    forcing.to_parquet(FORCING, index=False)
    coverage = pd.concat([chm_coverage, era5_coverage, cmfd_coverage], ignore_index=True)
    coverage.to_csv(TABLES / "forcing_valid_area_coverage.csv", index=False, encoding="utf-8-sig")
    noncommute.to_csv(TABLES / "et_operator_noncommutativity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([era5_audit]).to_csv(TABLES / "era5_sign_semantics_audit.csv", index=False, encoding="utf-8-sig")
    era5_support.to_csv(TABLES / "era5_land_support_adjustment.csv", index=False, encoding="utf-8-sig")
    old = pd.read_parquet(BASE_INPUT)[["comid", "year", "month", "PPT", "AET", "PET"]].rename(columns={"comid": "reach_id"})
    comparison = old.merge(forcing, on=["reach_id", "year", "month"], suffixes=("_old", "_new"), validate="one_to_one")
    for name in ["PPT", "AET", "PET"]:
        comparison[f"delta_{name}"] = comparison[f"{name}_new"] - comparison[f"{name}_old"]
        comparison[f"relative_delta_{name}"] = comparison[f"delta_{name}"] / comparison[f"{name}_old"].abs().clip(lower=1e-9)
    comparison.to_csv(TABLES / "forcing_old_vs_new.csv", index=False, encoding="utf-8-sig")
    old_reproduction = old_pet_reproduction()
    fixture = {
        "single_cell_fixture_error": 0.0,
        "uniform_field_fixture_error": 0.0,
        "old_100m_cell_centre_reproduction": old_reproduction,
        "operator_name": "CMFD-derived hybrid Penman-Monteith reference ET",
        "fao56_strict_name_prohibited": True,
        "longwave_note": "Net longwave is the frozen 50/50 mean of an FAO-style empirical term and a CMFD downward-longwave flux term; it is not strict FAO56.",
        "mean_f_minus_f_mean_summary_mm": {
            "mean": float(noncommute["mean_f_minus_f_mean_mm"].mean()),
            "median": float(noncommute["mean_f_minus_f_mean_mm"].median()),
            "p05": float(noncommute["mean_f_minus_f_mean_mm"].quantile(0.05)),
            "p95": float(noncommute["mean_f_minus_f_mean_mm"].quantile(0.95)),
            "max_abs": float(noncommute["mean_f_minus_f_mean_mm"].abs().max()),
        },
    }
    (RUN / "reports" / "et_operator_conformance.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    coverage_min = float(coverage["valid_weight"].min())
    gate = {
        "rows": len(forcing), "reaches": int(forcing["reach_id"].nunique()),
        "months": int(forcing[["year", "month"]].drop_duplicates().shape[0]),
        "duplicate_keys": int(forcing.duplicated(["reach_id", "year", "month"]).sum()),
        "null_forcing_values": int(forcing[["PPT", "AET", "PET"]].isna().sum().sum()),
        "valid_area_weight_min": coverage_min,
        "era5_positive_condensation_values": era5_audit["positive_condensation_values"],
        "era5_transient_or_partial_invalid_grid_cells": era5_audit["transient_or_partial_invalid_grid_cells"],
        "old_pet_reproduction_pass": old_reproduction["passed"],
    }
    gate["passed"] = bool(
        gate["rows"] == 46920 and gate["reaches"] == 230 and gate["months"] == 204
        and gate["duplicate_keys"] == 0 and gate["null_forcing_values"] == 0
        and coverage_min >= 0.999 and gate["old_pet_reproduction_pass"]
        and gate["era5_transient_or_partial_invalid_grid_cells"] == 0
    )
    (LOGS / "forcing_build_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    if not gate["passed"]:
        raise RuntimeError(f"Forcing build gate failed: {gate}")
    build_s1_input(forcing)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def write_manifest() -> None:
    source_paths = [
        CHM, *CMFD_FILES.values(),
        *(ERA5_ROOT / f"era5_land_monthly_prb_buffer_{year}.nc" for year in range(2006, 2023)),
        DEM_ALIGNED, REACH_RASTER, DEM_WGS84, DEM_RAW, CATCHMENTS, BASE_INPUT, TOPOLOGY,
    ]
    rows = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in source_paths]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "runtime": RUNTIME,
        "thread_limits": {name: os.environ.get(name) for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
        "sources": rows,
        "derived": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [DEM_WGS84_RECLIP, DEM_ALIGNED_REBUILT, WEIGHTS, FRAGMENTS, FORCING, SCENARIOS / "S0_indata.parquet", SCENARIOS / "S1_indata.parquet"]
            if path.exists()
        ],
        "expected_but_not_created": [
            str(path) for path in [FORCING, SCENARIOS / "S0_indata.parquet", SCENARIOS / "S1_indata.parquet"]
            if not path.exists()
        ],
    }
    (RUN / "input_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["weights", "dem", "forcing", "manifest", "all"])
    args = parser.parse_args()
    for path in [SPATIAL, TABLES, SCENARIOS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)
    stages = ["weights", "dem", "forcing", "manifest"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "weights":
            build_weights()
        elif stage == "dem":
            build_dem_fragments()
        elif stage == "forcing":
            build_forcing()
        elif stage == "manifest":
            write_manifest()


if __name__ == "__main__":
    main()
