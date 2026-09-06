"""Build ERA5-Land FAO-56 PET, harmonize it to CMFD, and append 2025."""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_30"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CACHE = RUN / "work" / "era5_daily_month_cache"
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "meteorology" / "era5_land" / "hourly_pet_prb"
ERA_REGISTRY = ROOT / "5_Test" / "20260828_28" / "outputs" / "era5_land_hourly_pet_acquisition_registry.parquet"
ERA_REPORT = ROOT / "5_Test" / "20260828_28" / "reports" / "era5_land_hourly_pet_acquisition.json"
HISTORICAL = ROOT / "5_Test" / "20260828_29" / "outputs" / "daily_hbv_forcing_1961_2024.parquet"
CMFD_REFERENCE = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"
WEIGHTS = ROOT / "5_Test" / "20260813_30" / "inputs" / "spatial" / "grid_overlap_weights.parquet"
CHM_MODULE = ROOT / "5_Test" / "20260828_29" / "scripts" / "build_historical_forcing_1961_2024.py"
SHORT_NAMES = ("t2m", "d2m", "sp", "u10", "v10", "ssrd", "strd")
SIGMA_MJ_M2_DAY_K4 = 4.903e-9
SURFACE_EMISSIVITY = 0.98
ALBEDO = 0.23
WIND_TO_2M = 4.87 / np.log(67.8 * 10.0 - 5.42)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return ";".join(decode_text(item) for item in value.tolist())
    return str(value)


def saturation_vapor_pressure_kpa(temp_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def build_operator() -> tuple[np.ndarray, csr_matrix, tuple[int, int, int, int], pd.DataFrame]:
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq("ERA5")].sort_values(["reach_id", "grid_i", "grid_j"]).reset_index(drop=True)
    reaches = np.sort(weights.reach_id.unique().astype(int))
    if not np.array_equal(reaches, np.arange(1, 231)):
        raise RuntimeError("frozen ERA5 weights do not cover Reach 1..230")
    if float(weights.groupby("reach_id").weight.sum().sub(1.0).abs().max()) > 1e-8:
        raise RuntimeError("frozen ERA5 weights do not sum to one")
    i0, i1 = int(weights.grid_i.min()), int(weights.grid_i.max()) + 1
    j0, j1 = int(weights.grid_j.min()), int(weights.grid_j.max()) + 1
    row = weights.reach_id.to_numpy(int) - 1
    col = (weights.grid_i.to_numpy(int) - i0) * (j1 - j0) + (weights.grid_j.to_numpy(int) - j0)
    operator = csr_matrix((weights.weight.to_numpy(float), (row, col)), shape=(230, (i1 - i0) * (j1 - j0)))
    return reaches, operator, (i0, i1, j0, j1), weights


def decode_time(handle: h5py.File) -> pd.DatetimeIndex:
    name = "valid_time" if "valid_time" in handle else "time"
    values = np.asarray(handle[name][:], dtype=float)
    units = decode_text(handle[name].attrs["units"])
    if units.startswith("seconds since "):
        origin = pd.Timestamp(units[len("seconds since "):])
        return pd.DatetimeIndex(origin + pd.to_timedelta(values, unit="s"))
    if units.startswith("hours since "):
        origin = pd.Timestamp(units[len("hours since "):])
        return pd.DatetimeIndex(origin + pd.to_timedelta(values, unit="h"))
    raise RuntimeError(f"unsupported ERA5 time units: {units}")


def aggregate_variable(handle: h5py.File, name: str, operator: csr_matrix, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, float]:
    i0, i1, j0, j1 = bbox
    data = np.asarray(handle[name][:, i0:i1, j0:j1], dtype=np.float64)
    if data.ndim != 3:
        raise RuntimeError(f"unexpected ERA5 shape for {name}: {data.shape}")
    flat = data.reshape(data.shape[0], -1)
    valid = np.isfinite(flat) & (np.abs(flat) < 1e30)
    coverage = np.asarray(operator @ valid.T, dtype=float).T
    minimum_coverage = float(coverage.min())
    # ERA5-Land carries a stable coastal land mask.  Renormalization is
    # authorized only when at least 99% of the frozen Reach weight is valid.
    if minimum_coverage < 0.99:
        raise RuntimeError(f"insufficient ERA5 Reach support for {name}: {coverage.min()}")
    numerator = np.asarray(operator @ np.where(valid, flat, 0.0).T, dtype=float).T
    return numerator / coverage, minimum_coverage


def process_month(payload: tuple[int, int, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    year, month, path_text = payload
    path = Path(path_text)
    reaches, operator, bbox, weights = build_operator()
    with h5py.File(path, "r") as handle:
        names = set(handle.keys())
        missing = sorted(set(SHORT_NAMES) - names)
        if missing:
            raise RuntimeError(f"ERA5 file missing {missing}: {path}")
        times = decode_time(handle)
        expected = pd.date_range(f"{year}-{month:02d}-01 00:00", periods=calendar.monthrange(year, month)[1] * 24, freq="h")
        if not times.equals(expected):
            raise RuntimeError(f"incomplete ERA5 hourly calendar: {path}")
        latitude = np.asarray(handle["latitude"][:], dtype=float)
        longitude = np.asarray(handle["longitude"][:], dtype=float)
        lat_error = np.abs(latitude[weights.grid_i.to_numpy(int)] - weights.grid_lat.to_numpy(float)).max()
        lon_error = np.abs(longitude[weights.grid_j.to_numpy(int)] - weights.grid_lon.to_numpy(float)).max()
        if max(float(lat_error), float(lon_error)) > 1e-5:
            raise RuntimeError(f"ERA5 grid differs from frozen weights: lat={lat_error}, lon={lon_error}")
        aggregated = {name: aggregate_variable(handle, name, operator, bbox) for name in SHORT_NAMES}
        arrays = {name: value[0] for name, value in aggregated.items()}
        minimum_coverage = {name: value[1] for name, value in aggregated.items()}
    temp_k = arrays["t2m"]
    temp_c = temp_k - 273.15
    dew_c = arrays["d2m"] - 273.15
    pressure_kpa = arrays["sp"] / 1000.0
    wind_2m = np.sqrt(arrays["u10"] ** 2 + arrays["v10"] ** 2) * WIND_TO_2M
    day_codes = pd.Categorical(times.normalize(), categories=times.normalize().unique(), ordered=True).codes
    days = pd.DatetimeIndex(times.normalize().unique())
    rows = []
    for code, date in enumerate(days):
        select = day_codes == code
        if int(select.sum()) != 24:
            raise RuntimeError(f"ERA5 day {date} does not have 24 hours")
        tmean = temp_c[select].mean(axis=0)
        pmean = pressure_kpa[select].mean(axis=0)
        es = saturation_vapor_pressure_kpa(temp_c[select]).mean(axis=0)
        ea = saturation_vapor_pressure_kpa(dew_c[select]).mean(axis=0)
        vpd_raw = es - ea
        vpd = np.maximum(vpd_raw, 0.0)
        u2 = wind_2m[select].mean(axis=0)
        ssrd = arrays["ssrd"][select].sum(axis=0) / 1.0e6
        strd = arrays["strd"][select].sum(axis=0) / 1.0e6
        rlup = (SURFACE_EMISSIVITY * SIGMA_MJ_M2_DAY_K4 * temp_k[select] ** 4 / 24.0).sum(axis=0)
        rn = (1.0 - ALBEDO) * ssrd + strd - rlup
        delta = 4098.0 * saturation_vapor_pressure_kpa(tmean) / (tmean + 237.3) ** 2
        gamma = 0.000665 * pmean
        pet = (0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * vpd) / (delta + gamma * (1.0 + 0.34 * u2))
        specific_humidity = 0.622 * ea / np.maximum(pmean - 0.378 * ea, 1e-9)
        rows.append(pd.DataFrame({
            "reach_id": reaches,
            "date": date,
            "tmean_c": tmean,
            "tmin_3hr_c": temp_c[select].min(axis=0),
            "tmax_3hr_c": temp_c[select].max(axis=0),
            "pressure_kpa": pmean,
            "specific_humidity_kgkg": specific_humidity,
            "wind_2m_m_s": u2,
            "srad_mj_m2_day": ssrd,
            "lrad_down_mj_m2_day": strd,
            "net_radiation_mj_m2_day": rn,
            "vpd_kpa": vpd,
            "pet_fao56_mm_day": pet,
            "cmfd_steps": 24,
            "vpd_unclipped_kpa": vpd_raw,
        }))
    frame = pd.concat(rows, ignore_index=True)
    if not np.isfinite(frame.drop(columns=["date", "reach_id"]).to_numpy(float)).all():
        raise RuntimeError(f"non-finite ERA5 daily values: {path}")
    return frame, {"year": year, "month": month, "path": str(path), "sha256": sha256(path), "days": len(days), "minimum_valid_weight_by_variable": minimum_coverage}


def acquisition_registry() -> pd.DataFrame:
    report = json.loads(ERA_REPORT.read_text(encoding="utf-8"))
    frame = pd.read_parquet(ERA_REGISTRY)
    frame = frame[frame.year.between(2021, 2025)].copy()
    if report.get("status") != "PASS" or len(frame) != 60 or frame.status.eq("failed").any():
        raise RuntimeError("ERA5 acquisition is not complete PASS")
    return frame


def build_era_daily(workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = acquisition_registry()
    frames, records, pending = [], [], []
    for row in registry.itertuples(index=False):
        stamp = f"{int(row.year)}{int(row.month):02d}"
        frame_path = CACHE / f"{stamp}.parquet"
        record_path = CACHE / f"{stamp}.registry.json"
        if frame_path.is_file() and record_path.is_file():
            frames.append(pd.read_parquet(frame_path))
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
        else:
            pending.append((int(row.year), int(row.month), str(row.path)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_month, payload): payload for payload in pending}
        failures = []
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            payload = futures[future]
            try:
                frame, record = future.result()
            except Exception as exc:
                failures.append({"year": payload[0], "month": payload[1], "error": repr(exc)})
                print(f"ERA5 PET {payload[0]}-{payload[1]:02d} FAILED; continuing", flush=True)
                continue
            stamp = f"{payload[0]}{payload[1]:02d}"
            frame_part = CACHE / f"{stamp}.parquet.part"
            frame.to_parquet(frame_part, index=False)
            os.replace(frame_part, CACHE / f"{stamp}.parquet")
            (CACHE / f"{stamp}.registry.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            frames.append(frame)
            records.append(record)
            print(f"ERA5 PET {payload[0]}-{payload[1]:02d} cached ({index}/{len(pending)} new)", flush=True)
    failure_path = REPORTS / "era5_pet_failed_months_latest.json"
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} ERA5 PET months failed; see {failure_path}")
    return pd.concat(frames, ignore_index=True).sort_values(["reach_id", "date"]).reset_index(drop=True), pd.DataFrame(records).sort_values(["year", "month"])


def load_chm_2025() -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("historical_forcing", CHM_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CHM_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.aggregate_chm_year(2025)


def harmonize(era: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cmfd = pd.read_parquet(CMFD_REFERENCE, columns=["date", "reach_id", "pet_fao56_mm_day"])
    cmfd["date"] = pd.to_datetime(cmfd.date)
    era["date"] = pd.to_datetime(era.date)
    overlap = era[era.date.dt.year <= 2024][["date", "reach_id", "pet_fao56_mm_day"]].rename(columns={"pet_fao56_mm_day": "era_pet"})
    overlap = overlap.merge(cmfd.rename(columns={"pet_fao56_mm_day": "cmfd_pet"}), on=["date", "reach_id"], validate="one_to_one")
    overlap["year"] = overlap.date.dt.year
    overlap["calendar_month"] = overlap.date.dt.month
    # Registered short bridge: three complete overlap years estimate the
    # low-dimensional Reach x calendar-month multiplicative correction, and
    # 2024 remains an untouched one-year temporal check.  Earlier ERA5 years
    # are not required to generate the 2025 extension.
    train = overlap[overlap.year.between(2021, 2023)]
    factors = train.groupby(["reach_id", "calendar_month"], as_index=False).agg(era_sum=("era_pet", "sum"), cmfd_sum=("cmfd_pet", "sum"))
    factors["pet_factor"] = factors.cmfd_sum / factors.era_sum.clip(lower=1e-12)
    if len(factors) != 230 * 12 or not np.isfinite(factors.pet_factor).all() or factors.pet_factor.le(0).any():
        raise RuntimeError("invalid ERA5-to-CMFD PET factors")
    era = era.copy()
    era["calendar_month"] = era.date.dt.month
    era = era.merge(factors[["reach_id", "calendar_month", "pet_factor"]], on=["reach_id", "calendar_month"], validate="many_to_one")
    era["pet_fao56_raw_mm_day"] = era.pet_fao56_mm_day
    era["pet_fao56_mm_day"] = era.pet_fao56_raw_mm_day * era.pet_factor
    validation = overlap[overlap.year.eq(2024)].merge(factors[["reach_id", "calendar_month", "pet_factor"]], on=["reach_id", "calendar_month"], validate="many_to_one")
    validation["corrected_pet"] = validation.era_pet * validation.pet_factor
    monthly = validation.groupby(["reach_id", "year", "calendar_month"], as_index=False)[["cmfd_pet", "era_pet", "corrected_pet"]].sum()
    correlation = float(monthly.cmfd_pet.corr(monthly.corrected_pet))
    pooled_bias = float(validation.corrected_pet.sum() / validation.cmfd_pet.sum() - 1.0)
    reach_year = validation.groupby(["reach_id", "year"], as_index=False)[["cmfd_pet", "corrected_pet"]].sum()
    reach_year["bias"] = reach_year.corrected_pet / reach_year.cmfd_pet - 1.0
    median_abs_reach_year_bias = float(reach_year.bias.abs().median())
    raw_rmse = float(np.sqrt(np.mean((validation.era_pet - validation.cmfd_pet) ** 2)))
    corrected_rmse = float(np.sqrt(np.mean((validation.corrected_pet - validation.cmfd_pet) ** 2)))
    gates = {
        "monthly_correlation_ge_0p98": correlation >= 0.98,
        "pooled_annual_bias_abs_le_0p05": abs(pooled_bias) <= 0.05,
        "median_abs_reach_year_bias_le_0p05": median_abs_reach_year_bias <= 0.05,
        "corrected_daily_rmse_lt_raw": corrected_rmse < raw_rmse,
    }
    metrics = {
        "status": "PASS" if all(gates.values()) else "PET_EXTENSION_CONFOUNDED",
        "monthly_pet_correlation": correlation,
        "pooled_pet_bias": pooled_bias,
        "median_abs_reach_year_bias": median_abs_reach_year_bias,
        "raw_daily_rmse_mm": raw_rmse,
        "corrected_daily_rmse_mm": corrected_rmse,
        "factor_min": float(factors.pet_factor.min()),
        "factor_median": float(factors.pet_factor.median()),
        "factor_max": float(factors.pet_factor.max()),
        "gates": gates,
    }
    return era, factors, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        raise RuntimeError("workers are capped at five")
    for directory in (OUT, REPORTS, LOCKS, CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    era, era_registry = build_era_daily(args.workers)
    era, factors, metrics = harmonize(era)
    chm_2025, chm_record = load_chm_2025()
    extension = era[era.date.dt.year.eq(2025)].merge(chm_2025, on=["date", "reach_id"], validate="one_to_one")
    extension["precip_source"] = "CHM_PRE_V2_daily"
    extension["pet_source"] = "ERA5_LAND_HOURLY_CMFD_HARMONIZED"
    extension["pet_bias_corrected"] = True
    extension["forcing_extension_flag"] = True
    historical = pd.read_parquet(HISTORICAL)
    historical["date"] = pd.to_datetime(historical.date)
    historical["pet_fao56_raw_mm_day"] = historical.pet_fao56_mm_day
    historical["pet_factor"] = 1.0
    historical["precip_source"] = "CHM_PRE_V2_daily"
    historical["pet_source"] = "CMFD_V2_0_03HR_FAO56"
    historical["pet_bias_corrected"] = False
    historical["forcing_extension_flag"] = False
    extension = extension[historical.columns]
    sensitivity_path = OUT / "daily_hbv_forcing_2025_era5_sensitivity.parquet"
    extension.to_parquet(sensitivity_path, index=False, compression="zstd")
    combined_path = OUT / "daily_hbv_forcing_1961_2025.parquet"
    if metrics["status"] == "PASS":
        combined = pd.concat([historical, extension], ignore_index=True).sort_values(["reach_id", "date"]).reset_index(drop=True)
        combined.to_parquet(combined_path, index=False, compression="zstd")
    factors.to_parquet(OUT / "era5_to_cmfd_pet_harmonization_factors.parquet", index=False)
    era_registry.to_parquet(OUT / "era5_hourly_source_registry.parquet", index=False)
    pd.DataFrame([chm_record]).to_parquet(OUT / "chm_pre_2025_source_registry.parquet", index=False)
    report = {
        "stage": "20260828_30",
        "status": metrics["status"],
        "metrics": metrics,
        "formal_combined_product_written": combined_path.is_file(),
        "extension_rows": len(extension),
        "extension_date_min": str(extension.date.min().date()),
        "extension_date_max": str(extension.date.max().date()),
        "minimum_era5_valid_reach_weight": float(min(
            min(value.values()) for value in era_registry["minimum_valid_weight_by_variable"]
        )),
        "source_boundary": "2025 PET is harmonized ERA5-Land, not CMFD and not discharge validation.",
        "hashes": {
            "historical_1961_2024": sha256(HISTORICAL),
            "extension_2025": sha256(sensitivity_path),
            "combined_1961_2025": sha256(combined_path) if combined_path.is_file() else None,
        },
    }
    (REPORTS / "era5_pet_harmonization_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (LOCKS / "forcing_1961_2025_lock.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
