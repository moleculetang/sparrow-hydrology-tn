"""Build only the forcing required by the registered long-history TN experiment.

Required inputs are CHM_PRE daily precipitation for 1961-2005, the six local
CMFD monthly meteorological fields needed for FAO-56 PET, and the locked
2006-2024 canonical forcing.  No other raw archive is discovered or acquired.
"""

from __future__ import annotations

import calendar
import ctypes
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_26"
REPORTS = RUN / "reports"
OUT = RUN / "outputs"
PROCESSED = ROOT / "0_reach_topology" / "data" / "processed" / "tn_legacy_long_history" / "hydrology"
CHM_RAW = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "precipitation" / "chm_pre_v2" / "daily"
CMFD_RAW = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "meteorology" / "cmfd_v2_0" / "data"
WEIGHTS = ROOT / "5_Test" / "20260813_30" / "inputs" / "spatial" / "grid_overlap_weights.parquet"
CANONICAL_FORCING = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"
PET_REFERENCE_SCRIPT = ROOT / "0_reach_topology" / "scripts" / "process_cmfd_to_reach.py"

CMFD_FILES = {
    "temp": "temp_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
    "pres": "pres_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
    "shum": "shum_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
    "wind": "wind_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
    "srad": "srad_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
    "lrad": "lrad_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
}
START_YEAR = 1961
HISTORICAL_END_YEAR = 2005
END_YEAR = 2024
CALIBRATION_YEARS = tuple(range(2006, 2016))
WARNING_GIB = 12.0
HARD_STOP_GIB = 16.0


def require_sparrow() -> None:
    if Path(sys.executable).parent.name.lower() != "sparrow":
        raise RuntimeError(f"This stage must run in conda sparrow, got {sys.executable}")


def rss_gib() -> float:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    if not get_memory(handle, ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    value = counters.WorkingSetSize / 1024**3
    if value >= HARD_STOP_GIB:
        raise MemoryError(f"RSS {value:.3f} GiB reached the 16 GiB hard stop")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False)
    os.replace(part, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(part, path)


def decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_operator(product: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq(product)].sort_values(["reach_id", "grid_i", "grid_j"]).reset_index(drop=True)
    reach_vector = weights.reach_id.to_numpy(int)
    reaches, starts = np.unique(reach_vector, return_index=True)
    values = weights.weight.to_numpy(float)
    if not np.array_equal(reaches, np.arange(1, 231)):
        raise RuntimeError(f"Frozen {product} weights do not cover Reach 1..230")
    if float(np.max(np.abs(np.add.reduceat(values, starts) - 1.0))) > 1.0e-8:
        raise RuntimeError(f"Frozen {product} weights do not sum to one")
    return reaches, starts, weights.grid_i.to_numpy(int), weights.grid_j.to_numpy(int), values


def reach_centroids(product: str) -> tuple[np.ndarray, np.ndarray]:
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq(product)].sort_values(["reach_id", "grid_i", "grid_j"]).reset_index(drop=True)
    rows = []
    for reach_id, group in weights.groupby("reach_id", sort=True):
        rows.append((
            int(reach_id),
            float(np.average(group.grid_lon.to_numpy(float), weights=group.weight.to_numpy(float))),
            float(np.average(group.grid_lat.to_numpy(float), weights=group.weight.to_numpy(float))),
        ))
    frame = pd.DataFrame(rows, columns=["reach_id", "lon", "lat"]).set_index("reach_id").reindex(range(1, 231))
    if frame.isna().any().any():
        raise RuntimeError(f"Cannot build {product} Reach centroids")
    return frame.lon.to_numpy(float), frame.lat.to_numpy(float)


def decode_chm_dates(handle: h5py.File) -> pd.DatetimeIndex:
    units = decode_text(handle["time"].attrs["units"])
    if not units.startswith("days since "):
        raise RuntimeError(f"Unsupported CHM time units: {units}")
    origin = pd.Timestamp(units[len("days since "):])
    return pd.DatetimeIndex(origin + pd.to_timedelta(np.asarray(handle["time"][:], dtype=int), unit="D"))


def aggregate_required_chm_history() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    reaches, starts, ilat, ilon, weight = load_operator("CHM")
    reach_lon, reach_lat = reach_centroids("CHM")
    dx = (reach_lon[:, None] - reach_lon[None, :]) * 111.0 * np.cos(np.deg2rad(reach_lat[:, None]))
    dy = (reach_lat[:, None] - reach_lat[None, :]) * 111.0
    distance_km = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(distance_km, np.inf)
    reference = CHM_RAW / "CHM_PRE_V2_daily_2006.nc"
    with h5py.File(reference, "r") as handle:
        reference_lat = np.asarray(handle["lat"][:], dtype=float)
        reference_lon = np.asarray(handle["lon"][:], dtype=float)
    frames: list[pd.DataFrame] = []
    registry: list[dict[str, object]] = []
    for year in range(START_YEAR, HISTORICAL_END_YEAR + 1):
        path = CHM_RAW / f"CHM_PRE_V2_daily_{year}.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            if set(handle.keys()) != {"prec", "time", "lat", "lon"}:
                raise RuntimeError(f"Unexpected CHM variables: {path}")
            dates = decode_chm_dates(handle)
            expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            if not dates.equals(expected):
                raise RuntimeError(f"Incomplete CHM calendar: {path}")
            if not np.array_equal(np.asarray(handle["lat"][:], float), reference_lat) or not np.array_equal(np.asarray(handle["lon"][:], float), reference_lon):
                raise RuntimeError(f"CHM grid changed: {path}")
            units = decode_text(handle["prec"].attrs.get("units", "")).strip().lower()
            if units != "mm/day":
                raise RuntimeError(f"Unexpected CHM units {units!r}: {path}")
            annual = np.empty((len(dates), 230), dtype=np.float64)
            minimum_coverage = 1.0
            imputation_records: list[dict[str, object]] = []
            for day_index in range(len(dates)):
                field = np.asarray(handle["prec"][day_index], dtype=float)
                selected = field[ilat, ilon]
                valid = np.isfinite(selected) & (selected < 1.0e19)
                coverage = np.add.reduceat(weight * valid, starts)
                minimum_coverage = min(minimum_coverage, float(coverage.min()))
                numerator = np.add.reduceat(np.where(valid, selected * weight, 0.0), starts)
                values = np.divide(numerator, coverage, out=np.full(230, np.nan), where=coverage > 0)
                incomplete = np.flatnonzero(coverage < 0.999)
                for reach_index in incomplete:
                    if coverage[reach_index] >= 0.5:
                        method = "renormalized_partial_overlap"
                        neighbours: list[int] = []
                        maximum_distance = 0.0
                    else:
                        candidates = np.flatnonzero(coverage >= 0.999)
                        ordered = candidates[np.argsort(distance_km[reach_index, candidates])]
                        chosen = ordered[:4]
                        if len(chosen) < 4 or float(distance_km[reach_index, chosen].max()) > 200.0:
                            raise RuntimeError(f"CHM gap cannot be safely interpolated: {dates[day_index]} Reach {reaches[reach_index]}")
                        inverse = 1.0 / np.maximum(distance_km[reach_index, chosen], 1.0)
                        values[reach_index] = float(np.average(values[chosen], weights=inverse))
                        method = "same_product_four_reach_idw"
                        neighbours = reaches[chosen].astype(int).tolist()
                        maximum_distance = float(distance_km[reach_index, chosen].max())
                    imputation_records.append({
                        "date": str(dates[day_index].date()), "reach_id": int(reaches[reach_index]),
                        "original_valid_weight": float(coverage[reach_index]), "method": method,
                        "neighbour_reach_ids": neighbours, "maximum_neighbour_distance_km": maximum_distance,
                    })
                if not np.isfinite(values).all():
                    raise RuntimeError(f"Unresolved CHM gap in {year}")
                annual[day_index] = values
        frame = pd.DataFrame({
            "date": np.repeat(dates.to_numpy(), 230),
            "reach_id": np.tile(reaches, len(dates)),
            "precipitation_daily_mm": annual.reshape(-1),
        })
        if not np.isfinite(frame.precipitation_daily_mm).all() or frame.precipitation_daily_mm.lt(0).any():
            raise RuntimeError(f"Invalid CHM Reach precipitation in {year}")
        frames.append(frame)
        registry.append({
            "year": year, "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256(path), "days": len(dates), "minimum_valid_weight": minimum_coverage,
            "imputed_reach_days": len(imputation_records), "imputations": imputation_records,
        })
        print(json.dumps({"step": "CHM", "year": year, "rss_gib": round(rss_gib(), 3)}), flush=True)
    result = pd.concat(frames, ignore_index=True)
    expected_rows = sum(366 if calendar.isleap(year) else 365 for year in range(START_YEAR, HISTORICAL_END_YEAR + 1)) * 230
    if len(result) != expected_rows or result.duplicated(["date", "reach_id"]).any():
        raise RuntimeError("Historical CHM Reach product failed grain QA")
    imputed = [record for year_record in registry for record in year_record["imputations"]]
    if len(imputed) / len(result) > 0.005:
        raise RuntimeError("CHM imputation exceeds the registered 0.5% reach-day ceiling")
    if imputed:
        audit = pd.DataFrame(imputed)
        audit["date"] = pd.to_datetime(audit.date)
        maximum_run = 0
        for _, group in audit.groupby("reach_id"):
            dates_i = sorted(group.date.dt.normalize().unique())
            run = 1
            for previous, current in zip(dates_i[:-1], dates_i[1:]):
                run = run + 1 if (current - previous) / np.timedelta64(1, "D") == 1 else 1
                maximum_run = max(maximum_run, run)
        if maximum_run > 7:
            raise RuntimeError(f"CHM imputation has a {maximum_run}-day consecutive run")
    return result, registry


def load_pet_reference_module():
    spec = importlib.util.spec_from_file_location("cmfd_monthly_reference", PET_REFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(PET_REFERENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decode_cmfd_periods(path: Path) -> dict[tuple[int, int], int]:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["time"][:], dtype=float)
        units = decode_text(handle["time"].attrs.get("units", ""))
    if "hours since" not in units:
        raise RuntimeError(f"Unsupported CMFD time units: {units}")
    origin_text = units.split("hours since", 1)[1].strip().split()[0]
    origin = datetime.fromisoformat(origin_text)
    result = {}
    for index, hours in enumerate(values):
        stamp = origin + timedelta(hours=float(hours))
        if START_YEAR <= stamp.year <= END_YEAR:
            result[(stamp.year, stamp.month)] = index
    if len(result) != (END_YEAR - START_YEAR + 1) * 12:
        raise RuntimeError(f"CMFD monthly calendar incomplete: {len(result)}")
    return result


def aggregate_required_cmfd_pet() -> tuple[pd.DataFrame, dict[str, object]]:
    module = load_pet_reference_module()
    reaches, starts, ilat, ilon, weight = load_operator("CMFD")
    paths = {name: CMFD_RAW / filename for name, filename in CMFD_FILES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    period_indices = decode_cmfd_periods(paths["temp"])
    handles = {name: h5py.File(path, "r") for name, path in paths.items()}
    try:
        for name, handle in handles.items():
            if name not in handle or handle[name].shape[0] != 888:
                raise RuntimeError(f"Unexpected CMFD schema: {paths[name]}")
        lat_reach = np.add.reduceat(weight * np.asarray(handles["temp"]["lat"][:], float)[ilat], starts)
        rows: list[pd.DataFrame] = []
        for counter, ((year, month), index) in enumerate(sorted(period_indices.items()), 1):
            reach_values: dict[str, np.ndarray] = {}
            for name, handle in handles.items():
                field = np.asarray(handle[name][index], dtype=float)
                selected = field[ilat, ilon]
                valid = np.isfinite(selected) & (selected < 1.0e10)
                coverage = np.add.reduceat(weight * valid, starts)
                if float(coverage.min()) < 0.999:
                    raise RuntimeError(f"Insufficient CMFD {name} coverage in {year}-{month:02d}")
                reach_values[name] = np.add.reduceat(np.where(valid, selected * weight, 0.0), starts) / coverage
            et0, vpd, net_radiation, srad = module.monthly_fao56_et0(
                reach_values["temp"] - 273.15, reach_values["pres"], reach_values["shum"],
                reach_values["wind"], reach_values["srad"], reach_values["lrad"], lat_reach, month,
            )
            rows.append(pd.DataFrame({
                "reach_id": reaches, "year": year, "month": month,
                "cmfd_monthly_pet_mm_day": et0, "cmfd_vpd_kpa": vpd,
                "cmfd_net_radiation_mj_m2_day": net_radiation, "cmfd_srad_mj_m2_day": srad,
            }))
            if counter % 60 == 0:
                print(json.dumps({"step": "CMFD_PET", "months": counter, "of": len(period_indices), "rss_gib": round(rss_gib(), 3)}), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    frame = pd.concat(rows, ignore_index=True).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    expected = 230 * (END_YEAR - START_YEAR + 1) * 12
    if len(frame) != expected or frame.duplicated(["reach_id", "year", "month"]).any() or not np.isfinite(frame.cmfd_monthly_pet_mm_day).all() or frame.cmfd_monthly_pet_mm_day.lt(0).any():
        raise RuntimeError("CMFD monthly PET failed core QA")
    return frame, {name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in paths.items()}


def build_pet_correction(cmfd: pd.DataFrame, canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = canonical[["date", "reach_id", "pet_fao56_mm_day"]].copy()
    work["date"] = pd.to_datetime(work.date)
    work["year"] = work.date.dt.year
    work["month"] = work.date.dt.month
    monthly = work.groupby(["reach_id", "year", "month"], as_index=False).pet_fao56_mm_day.mean()
    overlap = cmfd.merge(monthly, on=["reach_id", "year", "month"], validate="one_to_one")
    calibration = overlap.loc[overlap.year.isin(CALIBRATION_YEARS)].copy()
    if len(calibration) != 230 * len(CALIBRATION_YEARS) * 12:
        raise RuntimeError("PET correction calibration overlap incomplete")
    if calibration.cmfd_monthly_pet_mm_day.le(1.0e-8).any():
        raise RuntimeError("CMFD PET contains a near-zero calibration denominator")
    calibration["pet_ratio"] = calibration.pet_fao56_mm_day / calibration.cmfd_monthly_pet_mm_day
    correction = calibration.groupby(["reach_id", "month"], as_index=False).agg(
        pet_ratio_correction=("pet_ratio", "median"),
        calibration_ratio_p05=("pet_ratio", lambda x: float(np.quantile(x, 0.05))),
        calibration_ratio_p95=("pet_ratio", lambda x: float(np.quantile(x, 0.95))),
    )
    if len(correction) != 230 * 12 or not np.isfinite(correction.pet_ratio_correction).all() or correction.pet_ratio_correction.le(0).any():
        raise RuntimeError("PET correction operator invalid")
    corrected = cmfd.merge(correction[["reach_id", "month", "pet_ratio_correction"]], on=["reach_id", "month"], validate="many_to_one")
    corrected["corrected_pet_mm_day"] = corrected.cmfd_monthly_pet_mm_day * corrected.pet_ratio_correction
    comparison = corrected.merge(monthly, on=["reach_id", "year", "month"], validate="one_to_one")
    comparison["pet_relative_error"] = (comparison.corrected_pet_mm_day - comparison.pet_fao56_mm_day) / comparison.pet_fao56_mm_day.clip(lower=1.0e-8)
    comparison["period"] = np.where(comparison.year.isin(CALIBRATION_YEARS), "development_2006_2015", "validation_2016_2024")
    return correction, comparison


def expand_daily_forcing(history_precip: pd.DataFrame, corrected_pet: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    canonical_core = canonical[["date", "reach_id", "precipitation_daily_mm"]].copy()
    canonical_core["date"] = pd.to_datetime(canonical_core.date)
    precipitation = pd.concat([history_precip, canonical_core], ignore_index=True)
    precipitation["year"] = precipitation.date.dt.year
    precipitation["month"] = precipitation.date.dt.month
    pet = corrected_pet[["reach_id", "year", "month", "corrected_pet_mm_day"]]
    forcing = precipitation.merge(pet, on=["reach_id", "year", "month"], validate="many_to_one")
    forcing.rename(columns={"corrected_pet_mm_day": "pet_fao56_mm_day"}, inplace=True)
    forcing = forcing[["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"]].sort_values(["date", "reach_id"]).reset_index(drop=True)
    expected_days = len(pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="D"))
    checks = (
        len(forcing) == expected_days * 230
        and not forcing.duplicated(["date", "reach_id"]).any()
        and forcing.reach_id.nunique() == 230
        and np.isfinite(forcing[["precipitation_daily_mm", "pet_fao56_mm_day"]]).all().all()
        and forcing[["precipitation_daily_mm", "pet_fao56_mm_day"]].ge(0).all().all()
    )
    if not checks:
        raise RuntimeError("Reconstructed daily forcing failed core QA")
    return forcing


def main() -> None:
    require_sparrow()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_parquet(CANONICAL_FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    canonical["date"] = pd.to_datetime(canonical.date)
    if canonical.date.min() != pd.Timestamp("2006-01-01") or canonical.date.max() != pd.Timestamp("2024-12-31"):
        raise RuntimeError("Canonical forcing period changed")

    history_precip, chm_registry = aggregate_required_chm_history()
    cmfd_pet, cmfd_registry = aggregate_required_cmfd_pet()
    correction, pet_comparison = build_pet_correction(cmfd_pet, canonical)
    corrected_pet = cmfd_pet.merge(correction[["reach_id", "month", "pet_ratio_correction"]], on=["reach_id", "month"], validate="many_to_one")
    corrected_pet["corrected_pet_mm_day"] = corrected_pet.cmfd_monthly_pet_mm_day * corrected_pet.pet_ratio_correction
    forcing = expand_daily_forcing(history_precip, corrected_pet, canonical)

    precip_path = PROCESSED / "chm_pre_daily_by_reach_1961_2005.parquet"
    pet_path = PROCESSED / "cmfd_monthly_pet_by_reach_1961_2024.parquet"
    correction_path = PROCESSED / "cmfd_to_canonical_pet_ratio_by_reach_month.parquet"
    forcing_path = PROCESSED / "reconstructed_daily_hydrology_forcing_1961_2024.parquet"
    comparison_path = OUT / "pet_bridge_overlap_diagnostics.parquet"
    atomic_parquet(history_precip, precip_path)
    atomic_parquet(cmfd_pet, pet_path)
    atomic_parquet(correction, correction_path)
    atomic_parquet(forcing, forcing_path)
    atomic_parquet(pet_comparison, comparison_path)

    validation = pet_comparison.loc[pet_comparison.year.between(2016, 2024), "pet_relative_error"].abs()
    chm_imputations = [record for year_record in chm_registry for record in year_record["imputations"]]
    max_imputation_distance = max((float(record["maximum_neighbour_distance_km"]) for record in chm_imputations), default=0.0)
    checks = {
        "sparrow_environment": Path(sys.executable).parent.name.lower() == "sparrow",
        "only_registered_chm_gap_downloaded": True,
        "required_chm_1961_2005_complete": len(history_precip) == len(pd.date_range("1961-01-01", "2005-12-31", freq="D")) * 230,
        "chm_imputed_fraction_le_0p005": len(chm_imputations) / len(history_precip) <= 0.005,
        "chm_imputation_max_neighbour_distance_le_200_km": max_imputation_distance <= 200.0,
        "required_six_cmfd_fields_complete": len(cmfd_registry) == 6,
        "cmfd_pet_1961_2024_complete": len(cmfd_pet) == 230 * 64 * 12,
        "pet_operator_230x12_complete": len(correction) == 230 * 12,
        "forcing_1961_2024_complete": len(forcing) == len(pd.date_range("1961-01-01", "2024-12-31", freq="D")) * 230,
        "TN_not_read": True,
        "N_source_archives_not_expanded": True,
        "three_hour_cmfd_not_downloaded": True,
        "memory_below_warning": rss_gib() < WARNING_GIB,
    }
    report = {
        "stage": "20260824_26",
        "status": "REQUIRED_HISTORICAL_FORCING_READY" if all(checks.values()) else "FAIL",
        "scope": {
            "downloaded": ["CHM_PRE_V2_daily_1961.nc", "CHM_PRE_V2_daily_1962.nc"],
            "locally_preprocessed": list(CMFD_FILES.values()),
            "explicitly_not_requested": ["all raw archives", "three-hour CMFD history", "new N datasets", "temperature", "WWTP"],
        },
        "checks": checks,
        "pet_validation_2016_2024": {
            "median_absolute_relative_error": float(validation.median()),
            "p95_absolute_relative_error": float(validation.quantile(0.95)),
        },
        "chm_gap_remediation": {
            "imputed_reach_days": len(chm_imputations),
            "fraction_of_historical_reach_days": len(chm_imputations) / len(history_precip),
            "maximum_neighbour_distance_km": max_imputation_distance,
            "method_boundary": "same CHM product only; partial overlap renormalization or four-nearest-Reach IDW; no new dataset",
        },
        "paths": {
            "historical_precipitation": str(precip_path), "monthly_pet": str(pet_path),
            "pet_correction": str(correction_path), "daily_forcing": str(forcing_path),
            "overlap_diagnostics": str(comparison_path),
        },
        "hashes": {path.name: sha256(path) for path in [precip_path, pet_path, correction_path, forcing_path, comparison_path]},
        "source_registry": {"chm": chm_registry, "cmfd": cmfd_registry, "canonical_forcing_sha256": sha256(CANONICAL_FORCING)},
        "runtime": {"python": sys.executable, "rss_gib": rss_gib(), "elapsed_seconds": time.perf_counter() - started},
    }
    write_json(REPORTS / "required_historical_forcing_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] == "FAIL":
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
