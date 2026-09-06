"""Build the locked CHM_PRE + CMFD FAO-56 forcing extension for 2023-2024."""

from __future__ import annotations

import calendar
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CACHE = RUN / "work" / "cmfd_daily_month_cache_2023_2024"
CHM_RAW = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "precipitation" / "chm_pre_v2" / "daily"
CMFD_RAW = ROOT / "5_Test" / "20260825_2" / "inputs" / "cmfd_v2_0_03hr_2023_2024"
CMFD_DOWNLOADS = ROOT / "5_Test" / "20260825_2" / "outputs" / "cmfd_v2_0_03hr_2023_2024_download_registry.parquet"
CMFD_ACQUISITION = ROOT / "5_Test" / "20260825_2" / "reports" / "cmfd_v2_0_03hr_2023_2024_acquisition.json"
OLD_FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
WEIGHTS = ROOT / "5_Test" / "20260813_30" / "inputs" / "spatial" / "grid_overlap_weights.parquet"
PET_SCRIPT = ROOT / "5_Test" / "20260825_2" / "scripts" / "build_daily_cmfd_pet.py"
YEARS = (2023, 2024)


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


def decode_daily_dates(handle: h5py.File) -> pd.DatetimeIndex:
    values = np.asarray(handle["time"][:], dtype=int)
    units = decode_text(handle["time"].attrs["units"])
    prefix = "days since "
    if not units.startswith(prefix):
        raise RuntimeError(f"Unsupported CHM time units: {units}")
    origin = pd.Timestamp(units[len(prefix):])
    return pd.DatetimeIndex(origin + pd.to_timedelta(values, unit="D"))


def build_chm_extension() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq("CHM")].sort_values(["reach_id", "grid_i", "grid_j"]).reset_index(drop=True)
    reach_vector = weights.reach_id.to_numpy(int)
    reaches, starts = np.unique(reach_vector, return_index=True)
    if not np.array_equal(reaches, np.arange(1, 231)):
        raise RuntimeError("Frozen CHM weights do not cover Reach 1..230")
    w = weights.weight.to_numpy(float)
    if float(np.max(np.abs(np.add.reduceat(w, starts) - 1.0))) > 1e-8:
        raise RuntimeError("Frozen CHM weights do not sum to one")
    ilat = weights.grid_i.to_numpy(int)
    ilon = weights.grid_j.to_numpy(int)

    reference = CHM_RAW / "CHM_PRE_V2_daily_2022.nc"
    with h5py.File(reference, "r") as handle:
        reference_lat = np.asarray(handle["lat"][:], dtype=float)
        reference_lon = np.asarray(handle["lon"][:], dtype=float)

    frames: list[pd.DataFrame] = []
    registry: list[dict[str, object]] = []
    for year in YEARS:
        path = CHM_RAW / f"CHM_PRE_V2_daily_{year}.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            if set(handle.keys()) != {"prec", "time", "lat", "lon"}:
                raise RuntimeError(f"Unexpected CHM variables in {path}: {list(handle.keys())}")
            dates = decode_daily_dates(handle)
            expected_days = 366 if calendar.isleap(year) else 365
            expected_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            if len(dates) != expected_days or not dates.equals(expected_dates):
                raise RuntimeError(f"Incomplete CHM calendar: {path}")
            units = decode_text(handle["prec"].attrs.get("units", ""))
            if units.strip().lower() != "mm/day":
                raise RuntimeError(f"Unexpected CHM precipitation units {units!r} in {path}")
            lat = np.asarray(handle["lat"][:], dtype=float)
            lon = np.asarray(handle["lon"][:], dtype=float)
            if not np.array_equal(lat, reference_lat) or not np.array_equal(lon, reference_lon):
                raise RuntimeError(f"CHM grid changed in {path}")
            annual = np.empty((len(dates), 230), dtype=np.float64)
            minimum_coverage = 1.0
            for day_index in range(len(dates)):
                field = np.asarray(handle["prec"][day_index], dtype=float)
                values = field[ilat, ilon]
                valid = np.isfinite(values) & (values < 1.0e19)
                coverage = np.add.reduceat(w * valid, starts)
                minimum_coverage = min(minimum_coverage, float(coverage.min()))
                if float(coverage.min()) < 0.999:
                    raise RuntimeError(f"Insufficient CHM support in {year}: {coverage.min()}")
                numerator = np.add.reduceat(np.where(valid, values * w, 0.0), starts)
                annual[day_index] = numerator / coverage
        frame = pd.DataFrame({
            "reach_id": np.tile(reaches, len(dates)),
            "date": np.repeat(dates.to_numpy(), len(reaches)),
            "precipitation_daily_mm": annual.reshape(-1),
        })
        if not np.isfinite(frame.precipitation_daily_mm).all() or frame.precipitation_daily_mm.lt(0).any():
            raise RuntimeError(f"Invalid Reach precipitation in {year}")
        frames.append(frame)
        registry.append({
            "year": year,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "days": len(dates),
            "minimum_valid_weight": minimum_coverage,
            "units": units,
        })
        print(f"aggregated CHM_PRE {year} to 230 Reach", flush=True)
    return pd.concat(frames, ignore_index=True), registry


def load_pet_module():
    spec = importlib.util.spec_from_file_location("frozen_cmfd_pet", PET_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PET_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RAW = CMFD_RAW
    module.CACHE = CACHE
    module.DOWNLOAD_REGISTRY = CMFD_DOWNLOADS
    return module


def build_cmfd_pet_extension() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    acquisition = json.loads(CMFD_ACQUISITION.read_text(encoding="utf-8"))
    if acquisition.get("status") != "PASS" or acquisition.get("failed_file_count") != 0:
        raise RuntimeError("CMFD 2023-2024 acquisition is not locked PASS")
    module = load_pet_module()
    reaches, operator, bbox, _ = module.build_weight_operator()
    downloads = pd.read_parquet(CMFD_DOWNLOADS)
    if len(downloads) != 144 or downloads.status.eq("failed").any():
        raise RuntimeError("CMFD extension download registry is incomplete")
    source_hashes = dict(zip(downloads.local_path.astype(str), downloads.sha256.astype(str)))
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    registry = []
    periods = pd.period_range("2023-01", "2024-12", freq="M")
    for index, period in enumerate(periods, 1):
        stamp = f"{period.year}{period.month:02d}"
        frame_path = CACHE / f"{stamp}.parquet"
        registry_path = CACHE / f"{stamp}.registry.parquet"
        if frame_path.is_file() and registry_path.is_file():
            frame = pd.read_parquet(frame_path)
            records = pd.read_parquet(registry_path).to_dict("records")
            expected_rows = calendar.monthrange(period.year, period.month)[1] * 230
            if len(frame) != expected_rows or len(records) != 6:
                raise RuntimeError(f"Invalid PET cache for {stamp}")
        else:
            frame, records = module.daily_from_month(
                period.year, period.month, reaches, operator, bbox, source_hashes, False
            )
            frame_part = frame_path.with_suffix(".parquet.part")
            registry_part = registry_path.with_suffix(".parquet.part")
            frame.to_parquet(frame_part, index=False)
            pd.DataFrame(records).to_parquet(registry_part, index=False)
            os.replace(frame_part, frame_path)
            os.replace(registry_part, registry_path)
        frames.append(frame)
        registry.extend(records)
        print(f"CMFD PET month {period} ready ({index}/24)", flush=True)
    daily = pd.concat(frames, ignore_index=True).sort_values(["reach_id", "date"]).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily.date)
    if "vpd_unclipped_kpa" not in daily.columns:
        daily["vpd_unclipped_kpa"] = daily["vpd_kpa"]
    daily["vpd_unclipped_kpa"] = daily["vpd_unclipped_kpa"].fillna(daily["vpd_kpa"])
    daily["vpd_kpa"] = daily["vpd_unclipped_kpa"].clip(lower=0.0)
    delta = 4098.0 * module.saturation_vapor_pressure_kpa(daily.tmean_c.to_numpy(float)) / (daily.tmean_c.to_numpy(float) + 237.3) ** 2
    gamma = 0.000665 * daily.pressure_kpa.to_numpy(float)
    numerator = (
        0.408 * delta * daily.net_radiation_mj_m2_day.to_numpy(float)
        + gamma * (900.0 / (daily.tmean_c.to_numpy(float) + 273.0))
        * daily.wind_2m_m_s.to_numpy(float) * daily.vpd_kpa.to_numpy(float)
    )
    denominator = delta + gamma * (1.0 + 0.34 * daily.wind_2m_m_s.to_numpy(float))
    daily["pet_fao56_mm_day"] = numerator / denominator
    return daily, registry


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    chm, chm_registry = build_chm_extension()
    met, cmfd_registry = build_cmfd_pet_extension()
    extension = met.merge(chm, on=["reach_id", "date"], validate="one_to_one")
    old = pd.read_parquet(OLD_FORCING)
    old["date"] = pd.to_datetime(old.date)
    ordered = list(old.columns)
    missing = sorted(set(ordered) - set(extension.columns))
    extra = sorted(set(extension.columns) - set(ordered))
    if missing or extra:
        raise RuntimeError({"forcing_schema_missing": missing, "forcing_schema_extra": extra})
    extension = extension[ordered].sort_values(["reach_id", "date"]).reset_index(drop=True)
    extension["reach_id"] = extension.reach_id.astype(old.reach_id.dtype)
    combined = pd.concat([old[ordered], extension], ignore_index=True).sort_values(["reach_id", "date"]).reset_index(drop=True)

    expected_extension = 230 * (365 + 366)
    expected_combined = 1_596_200
    checks = {
        "extension_rows_exact": len(extension) == expected_extension,
        "combined_rows_exact": len(combined) == expected_combined,
        "reach_count_exact": combined.reach_id.nunique() == 230,
        "date_range_exact": combined.date.min() == pd.Timestamp("2006-01-01") and combined.date.max() == pd.Timestamp("2024-12-31"),
        "duplicate_reach_dates_zero": not combined.duplicated(["reach_id", "date"]).any(),
        "all_values_finite": bool(np.isfinite(combined.drop(columns=["reach_id", "date"]).to_numpy(float)).all()),
        "precipitation_nonnegative": bool(combined.precipitation_daily_mm.ge(0).all()),
        "pet_nonnegative": bool(combined.pet_fao56_mm_day.ge(0).all()),
        "cmfd_steps_eight": bool(extension.cmfd_steps.eq(8).all()),
        "chm_support_ge_0p999": min(row["minimum_valid_weight"] for row in chm_registry) >= 0.999,
    }
    extension_path = OUT / "daily_hbv_forcing_2023_2024.parquet"
    combined_path = OUT / "daily_hbv_forcing_2006_2024.parquet"
    extension.to_parquet(extension_path, index=False)
    combined.to_parquet(combined_path, index=False)
    pd.DataFrame(chm_registry).to_parquet(OUT / "chm_pre_daily_source_registry_2023_2024.parquet", index=False)
    pd.DataFrame(cmfd_registry).sort_values(["year", "month", "variable"]).to_parquet(OUT / "cmfd_03hr_source_registry_2023_2024.parquet", index=False)
    audit = {
        "stage": "20260827_6/extension_2023_2024",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "precipitation_source": "CHM_PRE V2 daily; no source transition",
        "pet_source": "CMFD V2.0 six-variable 3-hour FAO-56",
        "extension_rows": len(extension),
        "combined_rows": len(combined),
        "extension_precipitation_min_mm_day": float(extension.precipitation_daily_mm.min()),
        "extension_precipitation_p99_mm_day": float(extension.precipitation_daily_mm.quantile(0.99)),
        "extension_precipitation_max_mm_day": float(extension.precipitation_daily_mm.max()),
        "extension_pet_min_mm_day": float(extension.pet_fao56_mm_day.min()),
        "extension_pet_max_mm_day": float(extension.pet_fao56_mm_day.max()),
        "checks": checks,
        "files": {
            "old_forcing_sha256": sha256(OLD_FORCING),
            "extension_forcing_sha256": sha256(extension_path),
            "combined_forcing_sha256": sha256(combined_path),
            "weights_sha256": sha256(WEIGHTS),
        },
        "2025_boundary": "CHM_PRE 2025 is downloaded and preprocessed as precipitation-only data; complete CMFD PET forcing is currently locked only through 2024",
    }
    (REPORTS / "forcing_extension_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)


if __name__ == "__main__":
    main()
