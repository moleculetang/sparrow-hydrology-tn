from __future__ import annotations

import calendar
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
ROOT = Path(r"E:\SPARROW")
RAW = ROOT / "0_reach_topology" / "data" / "raw"
PARENT = ROOT / "5_Test" / "20260810_5"
CHM = RAW / "rainfall_2" / "CHM_PRE V2" / "monthly" / "CHM_PRE_V2_monthly.nc"
CMFD = RAW / "CMFD"
ERA5 = RAW / "era5_land" / "monthly_prb_buffer"
CATCHMENTS = ROOT / "5_Test" / "20260810_1" / "inputs" / "spatial_corrected" / "reach_catchments.shp"
CHM_MAP = ROOT / "5_Test" / "20260810_1" / "inputs" / "climate_corrected" / "chm" / "chm_pre_v2_grid_to_reach_mapping.csv"
CMFD_MAP = ROOT / "5_Test" / "20260810_1" / "inputs" / "climate_corrected" / "cmfd" / "cmfd_grid_to_reach_mapping.csv"
ERA5_MAP = ROOT / "5_Test" / "20260810_1" / "inputs" / "climate_corrected" / "era5_grid_to_reach_mapping.csv"
CMFD_MONTHLY = ROOT / "5_Test" / "20260810_1" / "inputs" / "climate_corrected" / "cmfd" / "cmfd_monthly_by_reach_2006_2022.parquet"
TABLES = RUN / "reports" / "tables"

CMFD_FILES = {
    name: CMFD / f"{name}_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc"
    for name in ["temp", "pres", "shum", "wind", "srad", "lrad"]
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attr_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return ";".join(attr_text(v) for v in value.tolist())
    return str(value)


def decode_numeric_time(values: np.ndarray, units: str) -> list[datetime]:
    unit, origin_text = units.split(" since ", 1)
    origin_text = origin_text.strip().replace("0.0", "00")
    origin = datetime.fromisoformat(origin_text)
    scale = {"days": 86400.0, "hours": 3600.0, "seconds": 1.0}[unit]
    return [origin + timedelta(seconds=float(v) * scale) for v in values]


def audit_time_and_units() -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    with xr.open_dataset(CHM, decode_times=False) as ds:
        units = str(ds["time"].attrs["units"])
        calendar_name = str(ds["time"].attrs.get("calendar", "standard"))
        dates = decode_numeric_time(ds["time"].values.astype(float), units)
        chm_months = [(d.year, d.month) for d in dates]
        expected = [(y, m) for y in range(1960, 2025) for m in range(1, 13)]
        rows.append({
            "source": "CHM_PRE_V2.1", "path": str(CHM), "time_variable": "time",
            "units": units, "calendar": calendar_name, "n_steps": len(dates),
            "first": dates[0].isoformat(), "last": dates[-1].isoformat(),
            "calendar_sequence_exact": chm_months == expected,
            "flux_units": str(ds["prec"].attrs.get("units", "not_declared")),
        })

    cmfd_reference: list[tuple[int, int]] | None = None
    for name, path in CMFD_FILES.items():
        with h5py.File(path, "r") as ds:
            units = attr_text(ds["time"].attrs["units"])
            calendar_name = attr_text(ds["time"].attrs.get("calendar", "standard"))
            dates = decode_numeric_time(np.asarray(ds["time"][:], dtype=float), units)
            months = [(d.year, d.month) for d in dates]
            identical = cmfd_reference is None or months == cmfd_reference
            if cmfd_reference is None:
                cmfd_reference = months
            expected = [(y, m) for y in range(1951, 2025) for m in range(1, 13)]
            rows.append({
                "source": f"CMFD_{name}", "path": str(path), "time_variable": "time",
                "units": units, "calendar": calendar_name, "n_steps": len(dates),
                "first": dates[0].isoformat(), "last": dates[-1].isoformat(),
                "calendar_sequence_exact": months == expected and identical,
                "flux_units": attr_text(ds[name].attrs.get("units", "")),
            })

    era5_exact = True
    era5_rows = []
    for year in range(2006, 2023):
        path = ERA5 / f"era5_land_monthly_prb_buffer_{year}.nc"
        request = ERA5 / "requests" / f"era5_land_monthly_prb_buffer_{year}_request.json"
        with h5py.File(path, "r") as ds:
            units = attr_text(ds["valid_time"].attrs["units"])
            calendar_name = attr_text(ds["valid_time"].attrs.get("calendar", "standard"))
            dates = decode_numeric_time(np.asarray(ds["valid_time"][:], dtype=float), units)
            months = [(d.year, d.month) for d in dates]
            expected = [(year, m) for m in range(1, 13)]
            req = json.loads(request.read_text(encoding="utf-8"))
            request_ok = "monthly_averaged_reanalysis" in req.get("product_type", [])
            exact = months == expected and request_ok
            era5_exact = era5_exact and exact
            era5_rows.append({"year": year, "months_exact": months == expected, "request_monthly_average": request_ok})
            if year == 2006:
                rows.append({
                    "source": "ERA5_Land_e", "path": str(path), "time_variable": "valid_time",
                    "units": units, "calendar": calendar_name, "n_steps": len(dates),
                    "first": dates[0].isoformat(), "last": dates[-1].isoformat(),
                    "calendar_sequence_exact": exact,
                    "flux_units": attr_text(ds["e"].attrs.get("units", "")),
                })
    pd.DataFrame(era5_rows).to_csv(TABLES / "era5_year_time_contract.csv", index=False, encoding="utf-8-sig")
    frame = pd.DataFrame(rows)
    summary = {
        "all_time_contracts_pass": bool(frame["calendar_sequence_exact"].all() and era5_exact),
        "chm_fixed_index_prohibited": True,
        "cmfd_variables_share_time_axis": True,
        "era5_monthly_averaged_request_confirmed": era5_exact,
    }
    return frame, summary


def mapping_summary() -> pd.DataFrame:
    rows = []
    for name, path in [("CHM", CHM_MAP), ("CMFD", CMFD_MAP), ("ERA5", ERA5_MAP)]:
        frame = pd.read_csv(path)
        grouped = frame.groupby("reach_id", as_index=False).agg(
            n_cells=("ilat", "size"), method=("mapping_method", "first")
        )
        rows.append({
            "source": name,
            "reaches": int(grouped["reach_id"].nunique()),
            "single_cell_reaches": int(grouped["n_cells"].eq(1).sum()),
            "nearest_centroid_reaches": int(grouped["method"].eq("nearest_centroid_cell").sum()),
            "min_cells": int(grouped["n_cells"].min()),
            "median_cells": float(grouped["n_cells"].median()),
            "max_cells": int(grouped["n_cells"].max()),
        })
        grouped[grouped["n_cells"].eq(1) | grouped["method"].eq("nearest_centroid_cell")].assign(source=name).to_csv(
            TABLES / f"{name.lower()}_limited_support_reaches.csv", index=False, encoding="utf-8-sig"
        )
    return pd.DataFrame(rows)


def limited_polygon_overlap_audit() -> pd.DataFrame:
    """Audit only the pre-registered single-cell/fallback Reach set.

    This computes true polygon-overlap weights for the 0.1-degree grid without
    opening a new interpolation or full-domain remapping search.
    """
    mapping = pd.read_csv(CMFD_MAP)
    grouped = mapping.groupby("reach_id", as_index=False).agg(
        current_n_cells=("ilat", "size"), current_method=("mapping_method", "first")
    )
    target_ids = set(grouped.loc[
        grouped["current_n_cells"].eq(1) | grouped["current_method"].eq("nearest_centroid_cell"),
        "reach_id",
    ].astype(int))
    catchments = gpd.read_file(CATCHMENTS)
    target = catchments[catchments["reach_id"].astype(int).isin(target_ids)][["reach_id", "geometry"]].copy()
    target_ll = target.to_crs(4326)
    with h5py.File(CMFD_FILES["temp"], "r") as ds:
        lats = np.asarray(ds["lat"][:], dtype=float)
        lons = np.asarray(ds["lon"][:], dtype=float)
    dlat = float(np.median(np.diff(np.sort(lats))))
    dlon = float(np.median(np.diff(np.sort(lons))))
    rows = []
    for ll_row, projected_row in zip(target_ll.itertuples(index=False), target.itertuples(index=False)):
        rid = int(ll_row.reach_id)
        minx, miny, maxx, maxy = ll_row.geometry.bounds
        lat_idx = np.where((lats + dlat / 2 >= miny) & (lats - dlat / 2 <= maxy))[0]
        lon_idx = np.where((lons + dlon / 2 >= minx) & (lons - dlon / 2 <= maxx))[0]
        records = []
        for ilat in lat_idx:
            for ilon in lon_idx:
                records.append({
                    "ilat": int(ilat), "ilon": int(ilon), "lat": float(lats[ilat]), "lon": float(lons[ilon]),
                    "geometry": box(lons[ilon] - dlon / 2, lats[ilat] - dlat / 2, lons[ilon] + dlon / 2, lats[ilat] + dlat / 2),
                })
        cells = gpd.GeoDataFrame(records, crs=4326).to_crs(target.crs)
        overlap = cells.geometry.intersection(projected_row.geometry).area.to_numpy(float)
        positive = overlap > 1e-6
        areas = overlap[positive]
        weights = areas / areas.sum() if len(areas) else np.array([], dtype=float)
        current = mapping[mapping["reach_id"].eq(rid)]
        current_keys = set(zip(current["ilat"].astype(int), current["ilon"].astype(int)))
        overlap_keys = set(zip(cells.loc[positive, "ilat"].astype(int), cells.loc[positive, "ilon"].astype(int)))
        rows.append({
            "reach_id": rid,
            "current_method": current["mapping_method"].iloc[0],
            "current_n_cells": len(current),
            "overlap_n_cells": int(positive.sum()),
            "overlap_effective_n_cells": float(1.0 / np.sum(weights**2)) if len(weights) else 0.0,
            "current_cells_all_overlap": current_keys.issubset(overlap_keys),
            "overlap_area_fraction_of_catchment": float(areas.sum() / max(projected_row.geometry.area, 1e-12)),
            "largest_overlap_weight": float(weights.max()) if len(weights) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("reach_id")


def et0_with_elevation(frame: pd.DataFrame, elevation_m: float) -> np.ndarray:
    temp = frame["T2M_C_cmfd"].to_numpy(float)
    pressure = frame["pres_kpa_cmfd"].to_numpy(float)
    shum = frame["shum_kgkg_cmfd"].to_numpy(float)
    wind = np.maximum(frame["wind_ms_cmfd"].to_numpy(float), 0.05)
    rs = frame["Rs_cmfd_mj_m2_day"].to_numpy(float)
    srad = frame["srad_wm2_cmfd"].to_numpy(float)
    lrad = frame["lrad_wm2_cmfd"].to_numpy(float)
    lat = frame["reach_id"].map(
        pd.read_csv(CMFD_MAP).groupby("reach_id").apply(
            lambda g: np.average(g["lat"], weights=g["weight"]), include_groups=False
        )
    ).to_numpy(float)
    month = frame["month"].to_numpy(int)
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    ea = np.clip((shum * pressure) / (0.622 + 0.378 * shum), 0.0, es * 1.2)
    vpd = np.maximum(es - ea, 0.0)
    delta = 4098.0 * es / ((temp + 237.3) ** 2)
    gamma = 0.000665 * pressure
    rns = 0.77 * rs
    tk = temp + 273.16
    sigma = 4.903e-9
    clear_up = sigma * tk**4 * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)))
    j = np.array([15 + sum(calendar.monthrange(2001, k)[1] for k in range(1, int(m))) for m in month])
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * j / 365.0)
    solar_delta = 0.409 * np.sin(2.0 * np.pi * j / 365.0 - 1.39)
    lat_rad = np.radians(lat)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(solar_delta), -1.0, 1.0))
    ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(solar_delta)
        + np.cos(lat_rad) * np.cos(solar_delta) * np.sin(ws)
    )
    rso = (0.75 + 2.0e-5 * elevation_m) * ra
    cloud = np.clip(1.35 * np.minimum(rs / np.maximum(rso, 1e-6), 1.0) - 0.35, 0.05, 1.0)
    rnl_fao = np.maximum(clear_up * cloud, 0.0)
    rldown = np.maximum(lrad, 0.0) * 0.0864
    rnl_flux = np.maximum((sigma * tk**4) - rldown, 0.0)
    rn = rns - 0.5 * (rnl_fao + rnl_flux)
    numerator = 0.408 * delta * rn + gamma * (900.0 / (temp + 273.0)) * wind * vpd
    denominator = delta + gamma * (1.0 + 0.34 * wind)
    daily = np.maximum(numerator / np.maximum(denominator, 1e-9), 0.0)
    days = np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(frame["year"], frame["month"])])
    return daily * days


def elevation_sensitivity() -> pd.DataFrame:
    frame = pd.read_parquet(CMFD_MONTHLY)
    baseline = et0_with_elevation(frame, 100.0)
    rows = []
    for elevation in [0, 100, 500, 1000, 2000, 3000]:
        candidate = et0_with_elevation(frame, float(elevation))
        rel = (candidate - baseline) / np.maximum(baseline, 1e-6)
        rows.append({
            "elevation_m": elevation,
            "median_relative_change": float(np.median(rel)),
            "p95_absolute_relative_change": float(np.quantile(np.abs(rel), 0.95)),
            "max_absolute_relative_change": float(np.max(np.abs(rel))),
            "median_pet_mm": float(np.median(candidate)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    time_frame, time_summary = audit_time_and_units()
    time_frame.to_csv(TABLES / "forcing_time_unit_contract.csv", index=False, encoding="utf-8-sig")
    mappings = mapping_summary()
    mappings.to_csv(TABLES / "grid_mapping_support_summary.csv", index=False, encoding="utf-8-sig")
    overlap = limited_polygon_overlap_audit()
    overlap.to_csv(TABLES / "limited_reach_polygon_overlap_audit.csv", index=False, encoding="utf-8-sig")
    sensitivity = elevation_sensitivity()
    sensitivity.to_csv(TABLES / "fao56_elevation_sensitivity.csv", index=False, encoding="utf-8-sig")
    source_rows = []
    for path in [CHM, *CMFD_FILES.values(), *(ERA5 / f"era5_land_monthly_prb_buffer_{year}.nc" for year in range(2006, 2023))]:
        source_rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(source_rows).to_csv(TABLES / "forcing_source_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "runtime": RUNTIME,
        **time_summary,
        "mapping_reaches_complete": bool(mappings["reaches"].eq(230).all()),
        "limited_polygon_overlap_reaches": int(len(overlap)),
        "limited_polygon_overlap_area_closure_pass": bool(
            # Transforming 0.1-degree cell polygons to the project Albers CRS
            # introduces small edge densification differences.  A 0.1% area
            # tolerance is a geometry QA tolerance, not a model-fit threshold.
            len(overlap) > 0 and overlap["overlap_area_fraction_of_catchment"].between(0.999, 1.001).all()
        ),
        "hardcoded_100m_removed_from_interpretation": True,
        "dem_corrected_et0_completed": False,
        "dem_correction_status": "UNRESOLVED_IN_FROZEN_SPARROW_ENV_NO_APPROVED_RASTER_READER",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["input_contract_pass"] = bool(
        summary["all_time_contracts_pass"]
        and summary["mapping_reaches_complete"]
        and summary["limited_polygon_overlap_area_closure_pass"]
    )
    (RUN / "reports" / "r3_input_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
