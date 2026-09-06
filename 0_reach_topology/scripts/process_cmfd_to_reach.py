from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import math
import sys

import h5py
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DEFAULT = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "meteorology" / "cmfd_v2_0" / "data"
OUT_DEFAULT = ROOT / "0_reach_topology" / "data" / "processed" / "cmfd_prb"

sys.path.insert(0, str(SCRIPT_DIR))
from process_chm_pre_v2_to_reach import (  # noqa: E402
    TOPO_RESULTS,
    lonlat_to_prb_albers,
    point_in_polygon,
    polygon_centroid,
    read_shapefile,
)


CMFD_FILES = {
    "temp": "temp_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
    "pres": "pres_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
    "shum": "shum_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
    "wind": "wind_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
    "srad": "srad_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
    "lrad": "lrad_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
}
FILL_LIMIT = 1.0e10
SIGMA_MJ_M2_DAY_K4 = 4.903e-9


@dataclass(frozen=True)
class MonthIndex:
    year: int
    month: int
    index: int
    days: int


def clean_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return ";".join(clean_attr(v) for v in value.tolist())
    return str(value)


def decode_cmfd_months(nc_path: Path, start_year: int, end_year: int) -> list[MonthIndex]:
    with h5py.File(nc_path, "r") as f:
        time_values = np.asarray(f["time"][:], dtype=float)
        units = clean_attr(f["time"].attrs.get("units", "hours since 1900-01-01 00:00:00"))
    if "hours since" not in units:
        raise RuntimeError(f"Unsupported CMFD time units: {units}")
    origin_text = units.split("hours since", 1)[1].strip().split()[0]
    origin = datetime.fromisoformat(origin_text)
    out: list[MonthIndex] = []
    for idx, hours in enumerate(time_values):
        dt = origin + timedelta(hours=float(hours))
        if start_year <= dt.year <= end_year:
            out.append(MonthIndex(dt.year, dt.month, idx, calendar.monthrange(dt.year, dt.month)[1]))
    expected = (end_year - start_year + 1) * 12
    if len(out) != expected:
        raise RuntimeError(f"Expected {expected} CMFD months, found {len(out)}")
    return out


def read_grid(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(raw_dir / CMFD_FILES["temp"], "r") as f:
        return f["lat"][:].astype(float), f["lon"][:].astype(float)


def build_grid_mapping(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    lats, lons = read_grid(raw_dir)
    catchments = read_shapefile(TOPO_RESULTS / "vectors" / "reach_catchments.shp")
    catchments = catchments[["reach_id", "geometry_obj"]].copy()

    all_x: list[float] = []
    all_y: list[float] = []
    for geom in catchments["geometry_obj"]:
        bbox = geom.get("bbox", None)
        if bbox:
            all_x.extend([bbox[0], bbox[2]])
            all_y.extend([bbox[1], bbox[3]])
    pad_m = 30_000.0
    xmin, xmax = min(all_x) - pad_m, max(all_x) + pad_m
    ymin, ymax = min(all_y) - pad_m, max(all_y) + pad_m

    grid_rows = []
    for ilat, lat in enumerate(lats):
        if lat < 18.0 or lat > 32.0:
            continue
        for ilon, lon in enumerate(lons):
            if lon < 98.0 or lon > 120.0:
                continue
            x, y = lonlat_to_prb_albers(float(lon), float(lat))
            if xmin <= x <= xmax and ymin <= y <= ymax:
                grid_rows.append({"ilat": ilat, "ilon": ilon, "lat": float(lat), "lon": float(lon), "x": x, "y": y})
    grid = pd.DataFrame(grid_rows)
    if grid.empty:
        raise RuntimeError("No CMFD grid cells intersect the PRB processing bbox.")

    rows = []
    for _, cat in catchments.iterrows():
        rid = int(cat["reach_id"])
        geom = cat["geometry_obj"]
        bbox = geom.get("bbox", (-np.inf, -np.inf, np.inf, np.inf))
        candidates = grid[
            (grid["x"] >= bbox[0])
            & (grid["x"] <= bbox[2])
            & (grid["y"] >= bbox[1])
            & (grid["y"] <= bbox[3])
        ]
        assigned = []
        for _, cell in candidates.iterrows():
            if point_in_polygon(float(cell["x"]), float(cell["y"]), geom):
                assigned.append(cell)
        if not assigned:
            cx, cy = polygon_centroid(geom)
            dist2 = (grid["x"] - cx) ** 2 + (grid["y"] - cy) ** 2
            assigned = [grid.loc[dist2.idxmin()]]
            method = "nearest_centroid_cell"
        else:
            method = "cell_center_within_catchment"
        for cell in assigned:
            rows.append(
                {
                    "reach_id": rid,
                    "ilat": int(cell["ilat"]),
                    "ilon": int(cell["ilon"]),
                    "lat": float(cell["lat"]),
                    "lon": float(cell["lon"]),
                    "weight": float(max(0.0, math.cos(math.radians(float(cell["lat"]))))),
                    "mapping_method": method,
                }
            )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(out_dir / "cmfd_grid_to_reach_mapping.csv", index=False, encoding="utf-8-sig")
    (
        mapping.groupby(["reach_id", "mapping_method"], as_index=False)
        .agg(n_grid_cells=("ilat", "size"), weight_sum=("weight", "sum"))
        .sort_values("reach_id")
        .to_csv(out_dir / "cmfd_mapping_summary.csv", index=False, encoding="utf-8-sig")
    )
    return mapping


def weighted_values(arr: np.ndarray, group: pd.DataFrame) -> float:
    ilat = group["ilat"].to_numpy(dtype=int)
    ilon = group["ilon"].to_numpy(dtype=int)
    weights = group["weight"].to_numpy(dtype=float)
    vals = arr[ilat, ilon].astype(float)
    vals[vals > FILL_LIMIT] = np.nan
    mask = np.isfinite(vals)
    if not mask.any():
        return np.nan
    w = weights[mask]
    if w.sum() <= 0:
        w = np.ones(mask.sum(), dtype=float)
    return float(np.average(vals[mask], weights=w))


def open_sources(raw_dir: Path) -> dict[str, h5py.File]:
    return {name: h5py.File(raw_dir / filename, "r") for name, filename in CMFD_FILES.items()}


def close_sources(files: dict[str, h5py.File]) -> None:
    for f in files.values():
        f.close()


def saturation_vapor_pressure_kpa(temp_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def extraterrestrial_radiation_mj_m2_day(lat_rad: np.ndarray, month: int) -> np.ndarray:
    j = 15 + sum(calendar.monthrange(2001, m)[1] for m in range(1, month))
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * j / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * j / 365.0 - 1.39)
    ws_arg = -np.tan(lat_rad) * np.tan(delta)
    ws = np.arccos(np.clip(ws_arg, -1.0, 1.0))
    return (
        (24.0 * 60.0 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(ws))
    )


def monthly_fao56_et0(
    temp_c: np.ndarray,
    pres_pa: np.ndarray,
    shum: np.ndarray,
    wind_ms: np.ndarray,
    srad_wm2: np.ndarray,
    lrad_wm2: np.ndarray,
    lat_deg: np.ndarray,
    month: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pressure_kpa = pres_pa / 1000.0
    es = saturation_vapor_pressure_kpa(temp_c)
    ea = (shum * pressure_kpa) / (0.622 + 0.378 * shum)
    ea = np.clip(ea, 0.0, es * 1.2)
    vpd = np.maximum(es - ea, 0.0)
    delta = 4098.0 * es / ((temp_c + 237.3) ** 2)
    gamma = 0.000665 * pressure_kpa

    rs = np.maximum(srad_wm2, 0.0) * 0.0864
    rldown = np.maximum(lrad_wm2, 0.0) * 0.0864
    rns = (1.0 - 0.23) * rs
    tk = temp_c + 273.16
    rlup_clear = SIGMA_MJ_M2_DAY_K4 * (tk**4) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)))
    ra = extraterrestrial_radiation_mj_m2_day(np.radians(lat_deg), month)
    rso = (0.75 + 2.0e-5 * 100.0) * ra
    cloud_factor = np.clip(1.35 * np.minimum(rs / np.maximum(rso, 1.0e-6), 1.0) - 0.35, 0.05, 1.0)
    rnl_fao = np.maximum(rlup_clear * cloud_factor, 0.0)
    rnl_flux = np.maximum((SIGMA_MJ_M2_DAY_K4 * tk**4) - rldown, 0.0)
    rnl = 0.5 * rnl_fao + 0.5 * rnl_flux
    rn = rns - rnl

    u2 = np.maximum(wind_ms, 0.05)
    numerator = 0.408 * delta * rn + gamma * (900.0 / (temp_c + 273.0)) * u2 * vpd
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    et0 = np.maximum(numerator / np.maximum(denominator, 1.0e-9), 0.0)
    return et0, vpd, rn, rs


def aggregate_monthly(raw_dir: Path, mapping: pd.DataFrame, out_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    months = decode_cmfd_months(raw_dir / CMFD_FILES["temp"], start_year, end_year)
    map_groups = list(mapping.groupby("reach_id"))
    files = open_sources(raw_dir)
    rows = []
    try:
        for mi in months:
            arrays = {name: files[name][name][mi.index, :, :] for name in CMFD_FILES}
            for reach_id, group in map_groups:
                vals = {name: weighted_values(arr, group) for name, arr in arrays.items()}
                temp_c = vals["temp"] - 273.15
                pres_pa = vals["pres"]
                shum = vals["shum"]
                wind_ms = vals["wind"]
                srad_wm2 = vals["srad"]
                lrad_wm2 = vals["lrad"]
                lat_deg = float(np.average(group["lat"].to_numpy(float), weights=group["weight"].to_numpy(float)))
                et0_day, vpd, rn_day, rs_day = monthly_fao56_et0(
                    np.array([temp_c]),
                    np.array([pres_pa]),
                    np.array([shum]),
                    np.array([wind_ms]),
                    np.array([srad_wm2]),
                    np.array([lrad_wm2]),
                    np.array([lat_deg]),
                    mi.month,
                )
                rows.append(
                    {
                        "reach_id": int(reach_id),
                        "year": mi.year,
                        "month": mi.month,
                        "T2M_C_cmfd": float(temp_c),
                        "pres_kpa_cmfd": float(pres_pa / 1000.0),
                        "shum_kgkg_cmfd": float(shum),
                        "wind_ms_cmfd": float(wind_ms),
                        "srad_wm2_cmfd": float(srad_wm2),
                        "lrad_wm2_cmfd": float(lrad_wm2),
                        "ET0_cmfd_mm_day": float(et0_day[0]),
                        "PET_cmfd_mm": float(et0_day[0] * mi.days),
                        "VPD_cmfd_kpa": float(vpd[0]),
                        "Rn_cmfd_mj_m2_day": float(rn_day[0]),
                        "Rs_cmfd_mj_m2_day": float(rs_day[0]),
                        "n_grid_cells_cmfd": int(len(group)),
                        "cmfd_source": "CMFD_V2.0 monthly six-variable FAO56_PM_ET0",
                    }
                )
    finally:
        close_sources(files)
    out = pd.DataFrame(rows).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    out.to_csv(out_dir / f"cmfd_monthly_by_reach_{start_year}_{end_year}.csv", index=False, encoding="utf-8-sig")
    out.to_parquet(out_dir / f"cmfd_monthly_by_reach_{start_year}_{end_year}.parquet", index=False)
    return out


def write_manifest(raw_dir: Path, out_dir: Path, mapping: pd.DataFrame, monthly: pd.DataFrame, start_year: int, end_year: int) -> None:
    rows = []
    for name, filename in CMFD_FILES.items():
        path = raw_dir / filename
        with h5py.File(path, "r") as f:
            var = f[name]
            rows.append(
                {
                    "variable": name,
                    "source_file": str(path),
                    "shape": str(var.shape),
                    "units": clean_attr(var.attrs.get("units", "")),
                    "standard_name": clean_attr(var.attrs.get("standard_name", "")),
                    "long_name": clean_attr(var.attrs.get("long_name", "")),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "cmfd_source_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "processed_dir": str(out_dir),
        "start_year": start_year,
        "end_year": end_year,
        "months": int(monthly[["year", "month"]].drop_duplicates().shape[0]),
        "reaches": int(monthly["reach_id"].nunique()),
        "mapping_rows": int(len(mapping)),
        "pet_min_mm": float(monthly["PET_cmfd_mm"].min()),
        "pet_mean_mm": float(monthly["PET_cmfd_mm"].mean()),
        "pet_max_mm": float(monthly["PET_cmfd_mm"].max()),
        "vpd_mean_kpa": float(monthly["VPD_cmfd_kpa"].mean()),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CMFD V2.0 monthly meteorology to PRB reach catchments and compute ET0/PET.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--reuse-mapping", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename in CMFD_FILES.values():
        if not (args.raw_dir / filename).exists():
            raise FileNotFoundError(args.raw_dir / filename)

    mapping_path = args.out_dir / "cmfd_grid_to_reach_mapping.csv"
    if args.reuse_mapping and mapping_path.exists():
        mapping = pd.read_csv(mapping_path, encoding="utf-8-sig")
    else:
        mapping = build_grid_mapping(args.raw_dir, args.out_dir)
    monthly = aggregate_monthly(args.raw_dir, mapping, args.out_dir, args.start_year, args.end_year)
    write_manifest(args.raw_dir, args.out_dir, mapping, monthly, args.start_year, args.end_year)
    print(f"wrote {len(monthly)} reach-month rows for {monthly['reach_id'].nunique()} reaches to {args.out_dir}")


if __name__ == "__main__":
    main()
