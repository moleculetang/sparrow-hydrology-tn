from __future__ import annotations

"""Build traceable reach-scale Legacy-N drivers from downloaded public data.

The script deliberately keeps *gross* agricultural N inputs separate from
net N surplus.  Crop removal and BNF are not inferred where no observed
source exists, so this output cannot silently be used as a SON input.
"""

import json
from pathlib import Path
import re
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from common import RUN, load_config, write_json
from runtime_guard import assert_sparrow_runtime


GDAL = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin")
GDAL_RASTERIZE = GDAL / "gdal_rasterize.exe"
GDAL_TRANSLATE = GDAL / "gdal_translate.exe"
GDAL_INFO = GDAL / "gdalinfo.exe"
RAW = Path(r"E:\SPARROW\0_reach_topology\data\raw")
HANI_ROOT = RAW / "agriculture" / "nitrogen_inputs" / "hani_v1_0" / "data"
HYDE_ROOT = RAW / "land_surface" / "land_use_history" / "hyde_v3_2_1" / "data" / "baseline" / "asc"
SOIL_ROOT = RAW / "soil" / "soilgrids_v2" / "prb_buffer_legacy_n" / "data"
CATCHMENTS = RUN / "inputs" / "baseline_snapshot" / "inputs" / "spatial_corrected" / "reach_catchments.shp"
WORK = RUN / "work" / "historical_n_drivers"
OUT = RUN / "inputs" / "processed"

GRID_NCOLS = 4320
GRID_NROWS = 2160
GRID_TE = (-180.0, -90.0, 180.0, 90.0)
GRID_RES = 1.0 / 12.0
YEARS = np.arange(1860, 2020, dtype=int)
DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm")

HANI_FILES = {
    "fertilizer_crop_nh4_kg_n_year": "nfer_crop_nh4",
    "fertilizer_crop_no3_kg_n_year": "nfer_crop_no3",
    "fertilizer_pasture_nh4_kg_n_year": "nfer_pas_nh4",
    "fertilizer_pasture_no3_kg_n_year": "nfer_pas_no3",
    "manure_crop_kg_n_year": "nmanure_app_crop",
    "manure_pasture_kg_n_year": "nmanure_app_pas",
    "manure_deposition_pasture_kg_n_year": "nmanure_dep_pas",
    "manure_deposition_rangeland_kg_n_year": "nmanure_dep_range",
    "atmospheric_deposition_nhx_kg_n_year": "ndep_nhx",
    "atmospheric_deposition_noy_kg_n_year": "ndep_noy",
}


def cmd(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True, text=True)


def gdal_info(path: Path) -> dict:
    result = subprocess.run([str(GDAL_INFO), "-json", str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def require_tools() -> None:
    missing = [str(path) for path in (GDAL_RASTERIZE, GDAL_TRANSLATE, GDAL_INFO) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"GDAL tools missing: {missing}")


def build_zone_grid() -> dict[str, object]:
    """Create an area-fraction crosswalk from reach catchments to 5' cells.

    A cell-centre assignment drops small headwater catchments.  We rasterize at
    0.005 degrees (16 x 16 subcells per source cell), then turn subcell counts
    into fractions of each HaNi/HYDE cell.  One source-cell total is therefore
    allocated across adjacent catchments without double counting.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    catchments = gpd.read_file(CATCHMENTS).to_crs("EPSG:4326")
    if len(catchments) != 230 or catchments["reach_id"].nunique() != 230:
        raise RuntimeError("Expected 230 unique catchments")
    minx, miny, maxx, maxy = catchments.total_bounds
    # Work in integer source-grid indices.  This prevents a floating-point
    # roundoff in gdal_rasterize -tap from producing an extra fine-grid row.
    col0 = int(np.floor((minx + 180.0) / GRID_RES))
    col1 = int(np.ceil((maxx + 180.0) / GRID_RES))
    row0 = int(np.floor((90.0 - maxy) / GRID_RES))
    row1 = int(np.ceil((90.0 - miny) / GRID_RES))
    coarse_width, coarse_height = col1 - col0, row1 - row0
    left, right = -180.0 + col0 * GRID_RES, -180.0 + col1 * GRID_RES
    top, bottom = 90.0 - row0 * GRID_RES, 90.0 - row1 * GRID_RES
    geographic = WORK / "reach_catchments_wgs84.gpkg"
    if not geographic.exists():
        catchments.to_file(geographic, driver="GPKG")
    zones = WORK / "reach_zone_hani_hyde_fine.tif"
    cmd([
        str(GDAL_RASTERIZE), "-a", "reach_id", "-a_nodata", "0", "-init", "0", "-at", "-ot", "Int32",
        "-te", *(str(value) for value in (left, bottom, right, top)),
        "-ts", str(coarse_width * 16), str(coarse_height * 16),
        "-of", "GTiff", str(geographic), str(zones),
    ])
    info = gdal_info(zones)
    fine_width, fine_height = info["size"]
    if fine_width % 16 or fine_height % 16:
        raise RuntimeError(f"Fine zone grid not aligned to 5' cells: {info['size']}")
    raw = WORK / "reach_zone_hani_hyde_fine.raw"
    cmd([str(GDAL_TRANSLATE), "-q", "-of", "ENVI", "-ot", "Int32", str(zones), str(raw)])
    array = np.fromfile(raw, dtype="<i4").reshape(fine_height, fine_width)
    reaches = np.unique(array[array > 0])
    if len(reaches) != 230 or not np.array_equal(reaches, np.arange(1, 231)):
        raise RuntimeError("Fine-grid crosswalk does not cover every reach")
    coarse_height, coarse_width = fine_height // 16, fine_width // 16
    coarse_index = (np.arange(fine_height)[:, None] // 16) * coarse_width + (np.arange(fine_width)[None, :] // 16)
    weights = np.zeros((230, coarse_height * coarse_width), dtype=np.float32)
    for reach_id in range(1, 231):
        count = np.bincount(coarse_index[array == reach_id].ravel(), minlength=coarse_height * coarse_width)
        weights[reach_id - 1] = count / 256.0
    if not np.all(weights.sum(axis=1) > 0):
        raise RuntimeError("At least one reach has zero source-grid overlap")
    return {
        "left": left, "bottom": bottom, "right": right, "top": top,
        "col0": col0, "col1": col1, "row0": row0, "row1": row1,
        "width": coarse_width, "height": coarse_height, "weights": weights.reshape(230, coarse_height, coarse_width),
    }


def weighted_cell_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.where(np.isfinite(values), values, 0.0)
    return np.einsum("rhw,hw->r", weights, values, optimize=True)


def zonal_mean(values: np.ndarray, zones: np.ndarray, nodata: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values) & (zones > 0)
    if nodata is not None:
        valid &= values != nodata
    counts = np.bincount(zones[valid].ravel(), minlength=231)[1:]
    sums = np.bincount(zones[valid].ravel(), weights=values[valid].ravel(), minlength=231)[1:]
    mean = np.divide(sums, counts, out=np.full(230, np.nan), where=counts > 0)
    return mean, counts


def build_hani(context: dict[str, object]) -> tuple[pd.DataFrame, dict]:
    weights = np.asarray(context["weights"])
    col0, col1 = int(context["col0"]), int(context["col1"])
    row0, row1 = int(context["row0"]), int(context["row1"])
    output = pd.DataFrame({"reach_id": np.tile(np.arange(1, 231), len(YEARS)), "year": np.repeat(YEARS, 230)})
    metadata: dict[str, dict] = {}
    for output_name, stem in HANI_FILES.items():
        path = HANI_ROOT / f"{stem}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path, decode_times=False, engine="h5netcdf") as dataset:
            var = dataset[stem]
            if dict(dataset.sizes).get("lat") != GRID_NROWS or dict(dataset.sizes).get("lon") != GRID_NCOLS:
                raise RuntimeError(f"Unexpected HaNi grid in {path}: {dict(dataset.sizes)}")
            origin = int(re.search(r"(\d{4})-01-01", str(dataset.time.attrs["units"])).group(1))
            available = set(range(origin, origin + int(dataset.sizes["time"])))
            series = np.full((len(YEARS), 230), np.nan, dtype=float)
            for index, year in enumerate(YEARS):
                if int(year) not in available:
                    continue
                # HaNi latitude is south-to-north, whereas the fractional
                # crosswalk is north-to-south.  Slice by exact source indices
                # then flip the data to the crosswalk orientation.
                layer = var.isel(
                    time=int(year) - origin,
                    lat=slice(GRID_NROWS - row1, GRID_NROWS - row0),
                    lon=slice(col0, col1),
                ).values.astype(float)
                layer = np.flipud(layer)
                if layer.shape != weights.shape[1:]:
                    raise RuntimeError(f"HaNi window shape mismatch: {layer.shape} versus {weights.shape[1:]}")
                # HaNi records total grams of N within each cell; aggregation is a sum, then g -> kg.
                series[index, :] = weighted_cell_sum(layer, weights) / 1000.0
        output[output_name] = series.ravel()
        metadata[output_name] = {
            "path": str(path), "units_source": "g N per grid cell per year", "units_output": "kg N/year per reach",
            "first_available_year": min(available), "last_available_year": max(available),
            "missing_before_first_available": int((YEARS < min(available)).sum()),
        }
    output["fertilizer_crop_kg_n_year"] = output[["fertilizer_crop_nh4_kg_n_year", "fertilizer_crop_no3_kg_n_year"]].sum(axis=1, min_count=1)
    output["fertilizer_pasture_kg_n_year"] = output[["fertilizer_pasture_nh4_kg_n_year", "fertilizer_pasture_no3_kg_n_year"]].sum(axis=1, min_count=1)
    output["atmospheric_deposition_total_kg_n_year"] = output[["atmospheric_deposition_nhx_kg_n_year", "atmospheric_deposition_noy_kg_n_year"]].sum(axis=1, min_count=1)
    gross_parts = [
        "fertilizer_crop_kg_n_year", "fertilizer_pasture_kg_n_year", "manure_crop_kg_n_year", "manure_pasture_kg_n_year",
        "manure_deposition_pasture_kg_n_year", "manure_deposition_rangeland_kg_n_year", "atmospheric_deposition_total_kg_n_year",
    ]
    output["gross_agricultural_n_input_kg_n_year"] = output[gross_parts].sum(axis=1, min_count=1)
    output["crop_removal_kg_n_year"] = np.nan
    output["bnf_kg_n_year"] = np.nan
    output["net_agricultural_n_surplus_kg_n_year"] = np.nan
    output["net_surplus_status"] = "blocked_missing_crop_removal_and_bnf"
    return output, metadata


def hyde_year_files() -> list[int]:
    years: set[int] = set()
    for directory in HYDE_ROOT.glob("*AD_lu"):
        match = re.fullmatch(r"(\d+)AD_lu", directory.name)
        if match and (directory / f"cropland{match.group(1)}AD.asc").exists():
            years.add(int(match.group(1)))
    usable = sorted(year for year in years if 1860 <= year <= 2015)
    if not usable or usable[-1] != 2015:
        raise RuntimeError("HYDE 3.2.1 annual end years are incomplete")
    return usable


def read_hyde_window(path: Path, context: dict[str, object]) -> np.ndarray:
    target = WORK / "hyde_window.raw"
    width, height = int(context["width"]), int(context["height"])
    col = int(round((float(context["left"]) + 180.0) / GRID_RES))
    row = int(round((90.0 - float(context["top"])) / GRID_RES))
    cmd([
        str(GDAL_TRANSLATE), "-q", "-srcwin", str(col), str(row), str(width), str(height),
        "-of", "ENVI", "-ot", "Float32", str(path), str(target),
    ])
    return np.fromfile(target, dtype="<f4").reshape(height, width)


def build_hyde(context: dict[str, object]) -> tuple[pd.DataFrame, dict]:
    weights = np.asarray(context["weights"])
    years = hyde_year_files()
    frames: list[pd.DataFrame] = []
    for year in years:
        code = str(year)
        files = {
            "hyde_cropland_km2": HYDE_ROOT / f"{code}AD_lu" / f"cropland{code}AD.asc",
            "hyde_grazing_km2": HYDE_ROOT / f"{code}AD_lu" / f"grazing{code}AD.asc",
            "hyde_pasture_km2": HYDE_ROOT / f"{code}AD_lu" / f"pasture{code}AD.asc",
            "hyde_population_persons": HYDE_ROOT / f"{code}AD_pop" / f"popc_{code}AD.asc",
        }
        values = {name: weighted_cell_sum(read_hyde_window(path, context), weights) for name, path in files.items()}
        frame = pd.DataFrame(values)
        frame.insert(0, "reach_id", np.arange(1, 231))
        frame.insert(1, "source_year", year)
        frames.append(frame)
    snapshots = pd.concat(frames, ignore_index=True).sort_values(["reach_id", "source_year"])
    annual = pd.MultiIndex.from_product([np.arange(1, 231), np.arange(1860, 2020)], names=["reach_id", "year"]).to_frame(index=False)
    annual = annual.merge(snapshots.rename(columns={"source_year": "year"}), on=["reach_id", "year"], how="left")
    columns = [column for column in annual if column.startswith("hyde_")]
    annual[columns] = annual.groupby("reach_id", sort=False)[columns].transform(lambda item: item.interpolate(limit_direction="both"))
    annual["hyde_land_use_status"] = np.where(annual["year"] <= 2015, "observed_or_linear_interpolation_between_HYDE_snapshots", "not_available_after_2015")
    annual.loc[annual["year"] > 2015, columns] = np.nan
    return annual, {"available_snapshot_years": years, "last_observed_year": 2015, "spatial_resolution": "5 arc minutes"}


def build_soil() -> tuple[pd.DataFrame, dict]:
    # SoilGrids is in Mollweide.  Create a zone raster on each layer's own grid, avoiding resampling soil values.
    records: list[pd.DataFrame] = []
    zone_path = WORK / "reach_zone_soilgrids.tif"
    projected_catchments = WORK / "reach_catchments_soilgrids_crs.gpkg"
    for depth in DEPTHS:
        layer_records: dict[str, np.ndarray] = {"reach_id": np.arange(1, 231), "depth": np.repeat(depth, 230)}
        coverage: dict[str, int] = {}
        for prop, divisor, output in (("nitrogen", 100.0, "nitrogen_g_kg"), ("soc", 10.0, "soc_g_kg"), ("bdod", 100.0, "bulk_density_g_cm3")):
            source = SOIL_ROOT / prop / f"{prop}_{depth}_mean_prb_buffer.tif"
            if not source.exists():
                raise FileNotFoundError(source)
            if not projected_catchments.exists():
                source_wkt = gdal_info(source)["coordinateSystem"]["wkt"]
                gpd.read_file(CATCHMENTS).to_crs(source_wkt).to_file(projected_catchments, driver="GPKG")
            info = gdal_info(source)
            width, height = info["size"]
            geotransform = info["geoTransform"]
            left, top, xres, yres = geotransform[0], geotransform[3], geotransform[1], geotransform[5]
            right, bottom = left + width * xres, top + height * yres
            cmd([
                str(GDAL_RASTERIZE), "-q", "-a", "reach_id", "-a_nodata", "0", "-init", "0", "-ot", "Int32",
                "-te", str(left), str(bottom), str(right), str(top), "-ts", str(width), str(height),
                "-of", "GTiff", str(projected_catchments), str(zone_path),
            ])
            zone_raw = WORK / "reach_zone_soilgrids.raw"
            value_raw = WORK / "soil_values.raw"
            cmd([str(GDAL_TRANSLATE), "-q", "-of", "ENVI", "-ot", "Int32", str(zone_path), str(zone_raw)])
            cmd([str(GDAL_TRANSLATE), "-q", "-of", "ENVI", "-ot", "Int16", str(source), str(value_raw)])
            zone_values = np.fromfile(zone_raw, dtype="<i4").reshape(height, width)
            raw_values = np.fromfile(value_raw, dtype="<i2").reshape(height, width).astype(float)
            mean, count = zonal_mean(raw_values, zone_values)
            layer_records[output] = mean / divisor
            coverage[prop] = int((count > 0).sum())
        frame = pd.DataFrame(layer_records)
        if frame[["nitrogen_g_kg", "soc_g_kg", "bulk_density_g_cm3"]].isna().any().any():
            raise RuntimeError(f"Incomplete SoilGrids coverage at {depth}")
        records.append(frame)
    soil = pd.concat(records, ignore_index=True)
    return soil, {"depths": list(DEPTHS), "reaches": 230, "units": {"nitrogen": "g/kg (raw cg/kg / 100)", "soc": "g/kg (raw dg/kg / 10)", "bdod": "g/cm3 (raw cg/cm3 / 100)"}}


def main() -> None:
    runtime = assert_sparrow_runtime()
    require_tools()
    config = load_config()
    if not CATCHMENTS.exists():
        raise FileNotFoundError(CATCHMENTS)
    context = build_zone_grid()
    hani, hani_meta = build_hani(context)
    hyde, hyde_meta = build_hyde(context)
    soil, soil_meta = build_soil()
    OUT.mkdir(parents=True, exist_ok=True)
    hani_path = OUT / "historical_agricultural_n_inputs_hani_1860_2019.parquet"
    hyde_path = OUT / "historical_land_use_population_hyde32_1860_2019.parquet"
    soil_path = OUT / "soilgrids_son_priors_by_reach.parquet"
    hani.to_parquet(hani_path, index=False)
    hyde.to_parquet(hyde_path, index=False)
    soil.to_parquet(soil_path, index=False)
    summary = {
        "run_id": config["run_id"], "runtime": runtime,
        "zone_method": "catchments rasterized at 0.005 degrees, then converted to fractional coverage of each 5-arc-minute source cell",
        "hani": {"rows": int(len(hani)), "reaches": int(hani.reach_id.nunique()), "years": [1860, 2019], "metadata": hani_meta,
                 "net_surplus_status": "blocked: crop removal and BNF have not been supplied"},
        "hyde": {"rows": int(len(hyde)), "metadata": hyde_meta, "post_2015_status": "not extrapolated"},
        "soilgrids": {"rows": int(len(soil)), "metadata": soil_meta},
        "outputs": [str(hani_path), str(hyde_path), str(soil_path)],
    }
    write_json(RUN / "reports" / "historical_n_driver_gate.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
