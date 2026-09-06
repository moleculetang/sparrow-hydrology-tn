"""Aggregate official CMFD V2.0 3-hour meteorology and compute daily ET0.

The calculation preserves the official UTC day boundary.  Spatial averaging
uses the frozen 20260813_30 full-polygon overlap weights.  Precipitation comes
only from the already-QA'd CHM_PRE daily product; monthly AET is never split
into daily forcing.
"""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260825_2"
RAW = RUN / "inputs" / "cmfd_v2_0_03hr_2006_2022"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
CACHE = RUN / "work" / "cmfd_daily_month_cache"
WEIGHTS = ROOT / "5_Test" / "20260813_30" / "inputs" / "spatial" / "grid_overlap_weights.parquet"
CHM = ROOT / "5_Test" / "20260823_35" / "outputs" / "daily_reach_forcing_2006_2022.parquet"
MONTHLY_CMFD = ROOT / "0_reach_topology" / "data" / "processed" / "cmfd_prb" / "cmfd_monthly_by_reach_2006_2022.parquet"
DOWNLOAD_REGISTRY = RUN / "outputs" / "cmfd_v2_0_03hr_download_registry.parquet"
VARIABLES = ("temp", "pres", "shum", "wind", "srad", "lrad")
SIGMA_MJ_M2_DAY_K4 = 4.903e-9
SURFACE_EMISSIVITY = 0.98
ALBEDO = 0.23
WIND_HEIGHT_M = 10.0
WIND_TO_2M = 4.87 / np.log(67.8 * WIND_HEIGHT_M - 5.42)
FILL_LIMIT = 1.0e10


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


def decode_time(handle: h5py.File) -> pd.DatetimeIndex:
    values = np.asarray(handle["time"][:], dtype=float)
    units = decode_text(handle["time"].attrs["units"])
    prefix = "hours since "
    if not units.startswith(prefix):
        raise RuntimeError(f"Unsupported CMFD time units: {units}")
    origin_text = units[len(prefix) :].strip().split()[0]
    origin = datetime.fromisoformat(origin_text)
    return pd.DatetimeIndex([origin + timedelta(hours=float(hours)) for hours in values])


def build_weight_operator() -> tuple[np.ndarray, csr_matrix, tuple[int, int, int, int], pd.DataFrame]:
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq("CMFD")].sort_values(["reach_id", "grid_i", "grid_j"]).reset_index(drop=True)
    reaches = np.sort(weights.reach_id.unique().astype(int))
    if not np.array_equal(reaches, np.arange(1, 231)):
        raise RuntimeError("Frozen CMFD weights do not cover Reach 1..230")
    sums = weights.groupby("reach_id").weight.sum().reindex(reaches).to_numpy(float)
    if float(np.max(np.abs(sums - 1.0))) > 1e-8:
        raise RuntimeError("Frozen CMFD full-polygon weights do not sum to one")
    i0, i1 = int(weights.grid_i.min()), int(weights.grid_i.max()) + 1
    j0, j1 = int(weights.grid_j.min()), int(weights.grid_j.max()) + 1
    ncol = (i1 - i0) * (j1 - j0)
    row = weights.reach_id.to_numpy(int) - 1
    col = (weights.grid_i.to_numpy(int) - i0) * (j1 - j0) + (weights.grid_j.to_numpy(int) - j0)
    operator = csr_matrix((weights.weight.to_numpy(float), (row, col)), shape=(230, ncol))
    return reaches, operator, (i0, i1, j0, j1), weights


def aggregate_file(path: Path, variable: str, operator: csr_matrix, bbox: tuple[int, int, int, int]) -> tuple[pd.DatetimeIndex, np.ndarray, float, dict[str, object]]:
    i0, i1, j0, j1 = bbox
    with h5py.File(path, "r") as handle:
        times = decode_time(handle)
        data = np.asarray(handle[variable][:, i0:i1, j0:j1], dtype=np.float64)
        attrs = {
            "units": decode_text(handle[variable].attrs.get("units", "")),
            "standard_name": decode_text(handle[variable].attrs.get("standard_name", "")),
            "shape": list(handle[variable].shape),
        }
        lat = np.asarray(handle["lat"][i0:i1], dtype=float)
        lon = np.asarray(handle["lon"][j0:j1], dtype=float)
    flat = data.reshape(len(times), -1)
    valid = np.isfinite(flat) & (flat < FILL_LIMIT)
    coverage = np.asarray(operator @ valid.T, dtype=float).T
    minimum_coverage = float(coverage.min())
    if minimum_coverage < 0.999:
        raise RuntimeError(f"{path.name}: minimum Reach support {minimum_coverage}")
    numerator = np.asarray(operator @ np.where(valid, flat, 0.0).T, dtype=float).T
    reach_values = numerator / coverage
    attrs["lat_min"] = float(lat.min())
    attrs["lat_max"] = float(lat.max())
    attrs["lon_min"] = float(lon.min())
    attrs["lon_max"] = float(lon.max())
    return times, reach_values, minimum_coverage, attrs


def saturation_vapor_pressure_kpa(temp_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def month_paths(year: int, month: int) -> dict[str, Path]:
    stamp = f"{year}{month:02d}"
    return {
        variable: RAW / variable / f"{variable}_CMFD_V0200_B-01_03hr_010deg_{stamp}.nc"
        for variable in VARIABLES
    }


def daily_from_month(
    year: int,
    month: int,
    reaches: np.ndarray,
    operator: csr_matrix,
    bbox: tuple[int, int, int, int],
    source_hashes: dict[str, str],
    auto_repair_corrupt_payloads: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    arrays: dict[str, np.ndarray] = {}
    reference_times: pd.DatetimeIndex | None = None
    registry: list[dict[str, object]] = []
    for variable, path in month_paths(year, month).items():
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            times, values, minimum_coverage, attrs = aggregate_file(path, variable, operator, bbox)
        except OSError as exc:
            if not auto_repair_corrupt_payloads:
                raise RuntimeError(f"CMFD payload read failed: {path}") from exc
            expected_parent = (RAW / variable).resolve()
            resolved_path = path.resolve(strict=True)
            if resolved_path.parent != expected_parent or resolved_path.name != path.name:
                raise RuntimeError(f"Refusing to quarantine unexpected payload path: {resolved_path}") from exc
            quarantine = path.with_suffix(path.suffix + ".corrupt")
            if quarantine.exists():
                raise RuntimeError(f"A quarantine already exists for repeatedly corrupt payload: {path}") from exc
            os.replace(path, quarantine)
            repair_script = RUN / "scripts" / "download_cmfd_v2_03hr.py"
            repair = subprocess.run(
                [sys.executable, str(repair_script), "--repair", f"{variable}:{year}{month:02d}"],
                cwd=str(RUN),
                check=False,
                capture_output=True,
                text=True,
            )
            if repair.returncode != 0:
                raise RuntimeError(
                    f"Registered repair failed for {path}: {repair.stdout}\n{repair.stderr}"
                ) from exc
            refreshed = pd.read_parquet(DOWNLOAD_REGISTRY)
            repaired_row = refreshed.loc[
                refreshed.variable.eq(variable) & refreshed.year.eq(year) & refreshed.month.eq(month)
            ]
            if len(repaired_row) != 1 or repaired_row.iloc[0].status == "failed":
                raise RuntimeError(f"Repair registry did not validate {path}") from exc
            source_hashes[str(path)] = str(repaired_row.iloc[0].sha256)
            times, values, minimum_coverage, attrs = aggregate_file(path, variable, operator, bbox)
            print(
                f"quarantined, repaired and payload-validated {year}-{month:02d} {variable}",
                flush=True,
            )
        if reference_times is None:
            reference_times = times
        elif not reference_times.equals(times):
            raise RuntimeError(f"Time mismatch for {path}")
        arrays[variable] = values
        registry.append(
            {
                "variable": variable,
                "year": year,
                "month": month,
                "path": str(path),
                "sha256": source_hashes.get(str(path)) or sha256(path),
                "bytes": path.stat().st_size,
                "time_steps": len(times),
                "first_time_utc": str(times.min()),
                "last_time_utc": str(times.max()),
                "minimum_valid_weight": minimum_coverage,
                **attrs,
            }
        )
    assert reference_times is not None
    expected_steps = calendar.monthrange(year, month)[1] * 8
    if len(reference_times) != expected_steps:
        raise RuntimeError(f"{year}-{month:02d}: expected {expected_steps} three-hour steps, found {len(reference_times)}")
    expected_times = pd.date_range(f"{year}-{month:02d}-01 00:00:00", periods=expected_steps, freq="3h")
    if not reference_times.equals(expected_times):
        raise RuntimeError(f"{year}-{month:02d}: three-hour UTC calendar is incomplete")

    temp_k = arrays["temp"]
    temp_c = temp_k - 273.15
    pressure_kpa = arrays["pres"] / 1000.0
    specific_humidity = arrays["shum"]
    es_step = saturation_vapor_pressure_kpa(temp_c)
    ea_step = specific_humidity * pressure_kpa / (0.622 + 0.378 * specific_humidity)
    wind_2m = arrays["wind"] * WIND_TO_2M
    rs_step = arrays["srad"] * (3.0 * 3600.0 / 1.0e6)
    rldown_step = arrays["lrad"] * (3.0 * 3600.0 / 1.0e6)
    rlup_step = SURFACE_EMISSIVITY * SIGMA_MJ_M2_DAY_K4 * temp_k**4 / 8.0
    rn_step = (1.0 - ALBEDO) * rs_step + rldown_step - rlup_step

    day_index = reference_times.normalize()
    unique_days = pd.DatetimeIndex(day_index.unique())
    day_codes = pd.Categorical(day_index, categories=unique_days, ordered=True).codes
    rows: list[pd.DataFrame] = []
    for code, date in enumerate(unique_days):
        select = day_codes == code
        if int(select.sum()) != 8:
            raise RuntimeError(f"{date.date()}: expected 8 CMFD steps")
        tmean = temp_c[select].mean(axis=0)
        tmin = temp_c[select].min(axis=0)
        tmax = temp_c[select].max(axis=0)
        pmean = pressure_kpa[select].mean(axis=0)
        es = es_step[select].mean(axis=0)
        ea = ea_step[select].mean(axis=0)
        vpd_unclipped = es - ea
        # CMFD component fields occasionally imply tiny daily supersaturation.
        # FAO-56 reference ET uses a non-negative vapour-pressure deficit; keep
        # the raw value for QA and floor only the formal aerodynamic driver.
        vpd = np.maximum(vpd_unclipped, 0.0)
        u2 = wind_2m[select].mean(axis=0)
        rn = rn_step[select].sum(axis=0)
        rs = rs_step[select].sum(axis=0)
        delta = 4098.0 * saturation_vapor_pressure_kpa(tmean) / (tmean + 237.3) ** 2
        gamma = 0.000665 * pmean
        numerator = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * vpd
        denominator = delta + gamma * (1.0 + 0.34 * u2)
        et0 = numerator / denominator
        rows.append(
            pd.DataFrame(
                {
                    "reach_id": reaches,
                    "date": date,
                    "tmean_c": tmean,
                    "tmin_3hr_c": tmin,
                    "tmax_3hr_c": tmax,
                    "pressure_kpa": pmean,
                    "specific_humidity_kgkg": specific_humidity[select].mean(axis=0),
                    "wind_2m_m_s": u2,
                    "srad_mj_m2_day": rs,
                    "lrad_down_mj_m2_day": rldown_step[select].sum(axis=0),
                    "net_radiation_mj_m2_day": rn,
                    "vpd_unclipped_kpa": vpd_unclipped,
                    "vpd_kpa": vpd,
                    "pet_fao56_mm_day": et0,
                    "cmfd_steps": int(select.sum()),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="200601")
    parser.add_argument("--end", default="202212")
    parser.add_argument("--qa-only", action="store_true", help="Build selected months but do not require the full registered period")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
        help=(
            "Month-level executor. Process is the formal default because the "
            "HDF5 library serializes concurrent reads within one Python process."
        ),
    )
    parser.add_argument(
        "--auto-repair-corrupt-payloads",
        action="store_true",
        help="Quarantine and re-download only exact registered files that fail HDF5 inflate during the formal full read",
    )
    args = parser.parse_args()
    if args.auto_repair_corrupt_payloads and args.workers != 1:
        raise RuntimeError("Payload auto-repair is permitted only in the deterministic single-worker execution path")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    reaches, operator, bbox, weights = build_weight_operator()
    periods = pd.period_range(pd.Period(args.start, freq="M"), pd.Period(args.end, freq="M"), freq="M")
    source_hashes: dict[str, str] = {}
    if DOWNLOAD_REGISTRY.is_file():
        downloads = pd.read_parquet(DOWNLOAD_REGISTRY)
        if {"local_path", "sha256"}.issubset(downloads.columns):
            source_hashes = dict(zip(downloads.local_path.astype(str), downloads.sha256.astype(str)))
    frames: list[pd.DataFrame] = []
    registry: list[dict[str, object]] = []
    pending: list[pd.Period] = []
    for period in periods:
        stamp = f"{period.year}{period.month:02d}"
        frame_path = CACHE / f"{stamp}.parquet"
        registry_path = CACHE / f"{stamp}.registry.parquet"
        if frame_path.is_file() and registry_path.is_file():
            cached_frame = pd.read_parquet(frame_path)
            cached_registry = pd.read_parquet(registry_path)
            expected_rows = calendar.monthrange(period.year, period.month)[1] * len(reaches)
            if len(cached_frame) != expected_rows or len(cached_registry) != len(VARIABLES):
                raise RuntimeError(f"Invalid completed-month cache for {stamp}")
            frames.append(cached_frame)
            registry.extend(cached_registry.to_dict("records"))
            print(f"loaded verified cache {period}", flush=True)
        else:
            pending.append(period)

    def persist_completed_month(period: pd.Period, frame: pd.DataFrame, records: list[dict[str, object]], index: int) -> None:
        stamp = f"{period.year}{period.month:02d}"
        frame_path = CACHE / f"{stamp}.parquet"
        registry_path = CACHE / f"{stamp}.registry.parquet"
        frame_part = frame_path.with_suffix(".parquet.part")
        registry_part = registry_path.with_suffix(".parquet.part")
        frame.to_parquet(frame_part, index=False)
        pd.DataFrame(records).to_parquet(registry_part, index=False)
        os.replace(frame_part, frame_path)
        os.replace(registry_part, registry_path)
        frames.append(frame)
        registry.extend(records)
        print(
            f"processed and cached {period} ({index}/{len(pending)} new; {len(frames)}/{len(periods)} total)",
            flush=True,
        )

    if args.workers == 1:
        # Submit-at-once executors defer a worker exception until all queued
        # months finish during shutdown.  The formal single-stream path must
        # fail immediately so a damaged source file can be quarantined and
        # the verified month cache can be resumed without wasted work.
        for index, period in enumerate(pending, 1):
            frame, records = daily_from_month(
                period.year,
                period.month,
                reaches,
                operator,
                bbox,
                source_hashes,
                auto_repair_corrupt_payloads=args.auto_repair_corrupt_payloads,
            )
            persist_completed_month(period, frame, records, index)
    else:
        executor_type = (
            concurrent.futures.ProcessPoolExecutor
            if args.executor == "process"
            else concurrent.futures.ThreadPoolExecutor
        )
        with executor_type(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    daily_from_month,
                    period.year,
                    period.month,
                    reaches,
                    operator,
                    bbox,
                    source_hashes,
                    False,
                ): period
                for period in pending
            }
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                period = futures[future]
                frame, records = future.result()
                persist_completed_month(period, frame, records, index)
    cmfd_daily = pd.concat(frames, ignore_index=True).sort_values(["reach_id", "date"]).reset_index(drop=True)
    cmfd_daily["date"] = pd.to_datetime(cmfd_daily.date)
    # Normalize legacy verified month caches produced before the VPD floor was
    # registered.  This makes cache replay scientifically identical to a clean
    # rebuild while preserving the expensive source-payload validation.
    if "vpd_unclipped_kpa" not in cmfd_daily.columns:
        cmfd_daily["vpd_unclipped_kpa"] = cmfd_daily["vpd_kpa"]
    else:
        cmfd_daily["vpd_unclipped_kpa"] = cmfd_daily["vpd_unclipped_kpa"].fillna(cmfd_daily["vpd_kpa"])
    cmfd_daily["vpd_kpa"] = cmfd_daily["vpd_unclipped_kpa"].clip(lower=0.0)
    delta = 4098.0 * saturation_vapor_pressure_kpa(cmfd_daily["tmean_c"].to_numpy(float)) / (
        cmfd_daily["tmean_c"].to_numpy(float) + 237.3
    ) ** 2
    gamma = 0.000665 * cmfd_daily["pressure_kpa"].to_numpy(float)
    numerator = (
        0.408 * delta * cmfd_daily["net_radiation_mj_m2_day"].to_numpy(float)
        + gamma
        * (900.0 / (cmfd_daily["tmean_c"].to_numpy(float) + 273.0))
        * cmfd_daily["wind_2m_m_s"].to_numpy(float)
        * cmfd_daily["vpd_kpa"].to_numpy(float)
    )
    denominator = delta + gamma * (1.0 + 0.34 * cmfd_daily["wind_2m_m_s"].to_numpy(float))
    cmfd_daily["pet_fao56_mm_day"] = numerator / denominator

    chm = pd.read_parquet(CHM, columns=["reach_id", "date", "precipitation_daily_mm"])
    chm["date"] = pd.to_datetime(chm.date)
    forcing = cmfd_daily.merge(chm, on=["reach_id", "date"], how="left", validate="one_to_one")
    forcing = forcing[["reach_id", "date", "precipitation_daily_mm"] + [c for c in cmfd_daily.columns if c not in {"reach_id", "date"}]]
    forcing.to_parquet(OUT / "daily_hbv_forcing_2006_2022.parquet", index=False)
    registry_frame = pd.DataFrame(registry).sort_values(["year", "month", "variable"])
    registry_frame.to_parquet(OUT / "cmfd_v2_0_03hr_source_registry.parquet", index=False)

    forcing["year"] = forcing.date.dt.year
    forcing["month"] = forcing.date.dt.month
    monthly = forcing.groupby(["reach_id", "year", "month"], as_index=False).agg(
        T2M_C_03hr=("tmean_c", "mean"),
        pres_kpa_03hr=("pressure_kpa", "mean"),
        shum_kgkg_03hr=("specific_humidity_kgkg", "mean"),
        wind_2m_03hr=("wind_2m_m_s", "mean"),
        Rs_mj_m2_03hr=("srad_mj_m2_day", "mean"),
        Rn_mj_m2_03hr=("net_radiation_mj_m2_day", "mean"),
        VPD_kpa_03hr=("vpd_kpa", "mean"),
        PET_03hr_mm=("pet_fao56_mm_day", "sum"),
        P_CHM_mm=("precipitation_daily_mm", "sum"),
        days=("date", "size"),
    )
    old_monthly = pd.read_parquet(MONTHLY_CMFD)
    comparison = monthly.merge(old_monthly, on=["reach_id", "year", "month"], how="left", validate="one_to_one")
    comparison["T2M_delta_c"] = comparison.T2M_C_03hr - comparison.T2M_C_cmfd
    comparison["pres_delta_kpa"] = comparison.pres_kpa_03hr - comparison.pres_kpa_cmfd
    comparison["shum_delta"] = comparison.shum_kgkg_03hr - comparison.shum_kgkg_cmfd
    comparison["pet_delta_mm"] = comparison.PET_03hr_mm - comparison.PET_cmfd_mm
    comparison.to_parquet(OUT / "daily_to_monthly_cmfd_consistency.parquet", index=False)

    area = pd.read_parquet(ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet", columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id")
    snow = forcing.merge(area, on="reach_id", validate="many_to_one")
    snow["precip_volume_weight"] = snow.precipitation_daily_mm * snow.catchment_area_km2
    total_p = float(snow.precip_volume_weight.sum())
    cold_mean_fraction = float(snow.loc[snow.tmean_c.le(0), "precip_volume_weight"].sum() / total_p)
    freezing_exposure_fraction = float(snow.loc[snow.tmin_3hr_c.le(0), "precip_volume_weight"].sum() / total_p)

    expected_rows = 230 * sum(calendar.isleap(year) + 365 for year in range(2006, 2023))
    full_period = args.start == "200601" and args.end == "202212" and not args.qa_only
    audit = {
        "stage": "20260825_2",
        "source": "TPDC CMFD V2.0 three-hour + CHM_PRE V2.1 daily",
        "cmfd_doi": "10.11888/Atmos.tpdc.302088",
        "day_boundary": "UTC 00:00",
        "pet_method": "FAO-56 Penman-Monteith with 3-hour es/ea aggregation and direct downward-longwave net radiation",
        "reach_count": int(forcing.reach_id.nunique()),
        "first_date": str(forcing.date.min().date()),
        "last_date": str(forcing.date.max().date()),
        "rows": int(len(forcing)),
        "expected_rows": expected_rows if full_period else None,
        "duplicate_reach_dates": int(forcing.duplicated(["reach_id", "date"]).sum()),
        "missing_precipitation": int(forcing.precipitation_daily_mm.isna().sum()),
        "missing_meteorology": int(forcing[list({"tmean_c", "pressure_kpa", "specific_humidity_kgkg", "wind_2m_m_s", "srad_mj_m2_day", "lrad_down_mj_m2_day"})].isna().sum().sum()),
        "negative_pet_rows": int(forcing.pet_fao56_mm_day.lt(0).sum()),
        "negative_unclipped_vpd_rows": int(forcing.vpd_unclipped_kpa.lt(0).sum()),
        "negative_vpd_rows": int(forcing.vpd_kpa.lt(0).sum()),
        "pet_min_mm_day": float(forcing.pet_fao56_mm_day.min()),
        "pet_median_mm_day": float(forcing.pet_fao56_mm_day.median()),
        "pet_max_mm_day": float(forcing.pet_fao56_mm_day.max()),
        "minimum_cmfd_valid_weight": float(registry_frame.minimum_valid_weight.min()),
        "monthly_temperature_delta_abs_p99_c": float(comparison.T2M_delta_c.abs().quantile(0.99)),
        "monthly_pressure_delta_abs_p99_kpa": float(comparison.pres_delta_kpa.abs().quantile(0.99)),
        "monthly_specific_humidity_delta_abs_p99": float(comparison.shum_delta.abs().quantile(0.99)),
        "cold_mean_precipitation_volume_fraction": cold_mean_fraction,
        "freezing_exposure_precipitation_volume_fraction": freezing_exposure_fraction,
        "rain_only_gate_cold_mean_fraction_lt_0p01": cold_mean_fraction < 0.01,
        "formal_full_period": full_period,
    }
    audit["forcing_gate_pass"] = bool(
        audit["reach_count"] == 230
        and audit["duplicate_reach_dates"] == 0
        and audit["missing_precipitation"] == 0
        and audit["missing_meteorology"] == 0
        and audit["negative_pet_rows"] == 0
        and audit["negative_vpd_rows"] == 0
        and audit["minimum_cmfd_valid_weight"] >= 0.999
        and (not full_period or audit["rows"] == expected_rows)
    )
    (REPORT / "daily_forcing_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if not audit["forcing_gate_pass"]:
        raise RuntimeError(audit)


if __name__ == "__main__":
    main()
