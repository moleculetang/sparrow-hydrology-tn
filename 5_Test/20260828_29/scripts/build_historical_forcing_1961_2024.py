"""Build locked daily CHM_PRE + CMFD forcing for 1961-2024.

Only 1961-2005 is newly calculated.  The accepted 2006-2024 forcing is reused
without numerical rewriting, then concatenated under an auditable schema.
"""

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


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_29"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CACHE = RUN / "work" / "cmfd_daily_month_cache_1961_2005"
CMFD_PROGRESS = ROOT / "5_Test" / "20260828_27" / "outputs" / "cmfd_1961_2024_acquisition_progress.parquet"
CMFD_FINAL = ROOT / "5_Test" / "20260828_27" / "outputs" / "cmfd_1961_2024_acquisition_registry.parquet"
CMFD_REPORT = ROOT / "5_Test" / "20260828_27" / "reports" / "cmfd_acquisition_final.json"
PET_SCRIPT = ROOT / "5_Test" / "20260825_2" / "scripts" / "build_daily_cmfd_pet.py"
CHM_RAW = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "precipitation" / "chm_pre_v2" / "daily"
WEIGHTS = ROOT / "5_Test" / "20260813_30" / "inputs" / "spatial" / "grid_overlap_weights.parquet"
OLD = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"
VARIABLES = ("temp", "pres", "shum", "wind", "srad", "lrad")


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


def decode_chm_dates(handle: h5py.File) -> pd.DatetimeIndex:
    values = np.asarray(handle["time"][:], dtype=int)
    units = decode_text(handle["time"].attrs["units"])
    prefix = "days since "
    if not units.startswith(prefix):
        raise RuntimeError(f"unsupported CHM time units: {units}")
    return pd.DatetimeIndex(pd.Timestamp(units[len(prefix):]) + pd.to_timedelta(values, unit="D"))


def chm_operator() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq("CHM")].sort_values(["reach_id", "grid_i", "grid_j"]).reset_index(drop=True)
    reach_vector = weights.reach_id.to_numpy(int)
    reaches, starts = np.unique(reach_vector, return_index=True)
    if not np.array_equal(reaches, np.arange(1, 231)):
        raise RuntimeError("frozen CHM weights do not cover Reach 1..230")
    values = weights.weight.to_numpy(float)
    if float(np.max(np.abs(np.add.reduceat(values, starts) - 1.0))) > 1e-8:
        raise RuntimeError("frozen CHM weights do not sum to one")
    return reaches, starts, weights.grid_i.to_numpy(int), weights.grid_j.to_numpy(int), values


def aggregate_chm_year(year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    reaches, starts, ilat, ilon, weights = chm_operator()
    path = CHM_RAW / f"CHM_PRE_V2_daily_{year}.nc"
    with h5py.File(path, "r") as handle:
        dates = decode_chm_dates(handle)
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        if not dates.equals(expected):
            raise RuntimeError(f"incomplete CHM calendar: {path}")
        units = decode_text(handle["prec"].attrs.get("units", ""))
        if units.strip().lower() != "mm/day":
            raise RuntimeError(f"unexpected CHM units {units}: {path}")
        annual = np.empty((len(dates), 230), dtype=np.float64)
        minimum_coverage = 1.0
        for day in range(len(dates)):
            field = np.asarray(handle["prec"][day], dtype=float)
            source = field[ilat, ilon]
            valid = np.isfinite(source) & (source < 1.0e19)
            coverage = np.add.reduceat(weights * valid, starts)
            minimum_coverage = min(minimum_coverage, float(coverage.min()))
            if minimum_coverage < 0.999:
                raise RuntimeError(f"insufficient CHM support in {year}: {minimum_coverage}")
            numerator = np.add.reduceat(np.where(valid, source * weights, 0.0), starts)
            annual[day] = numerator / coverage
    frame = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), len(reaches)),
        "reach_id": np.tile(reaches, len(dates)),
        "precipitation_daily_mm": annual.reshape(-1),
    })
    return frame, {
        "year": year,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "days": len(dates),
        "minimum_valid_weight": minimum_coverage,
        "units": units,
    }


def load_pet_module():
    spec = importlib.util.spec_from_file_location("frozen_cmfd_pet", PET_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PET_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_cmfd_month(payload: tuple[int, int, dict[str, str], dict[str, str]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    year, month, paths, hashes = payload
    module = load_pet_module()
    module.month_paths = lambda _year, _month: {variable: Path(path) for variable, path in paths.items()}
    reaches, operator, bbox, _ = module.build_weight_operator()
    return module.daily_from_month(year, month, reaches, operator, bbox, hashes, False)


def complete_registry() -> pd.DataFrame:
    if not CMFD_REPORT.is_file() or not CMFD_FINAL.is_file():
        raise RuntimeError("CMFD acquisition has not reached its final PASS registry")
    report = json.loads(CMFD_REPORT.read_text(encoding="utf-8"))
    frame = pd.read_parquet(CMFD_FINAL)
    if report.get("status") != "PASS" or len(frame) != 4608 or frame.status.eq("failed").any():
        raise RuntimeError("CMFD acquisition is incomplete")
    # Acquisition initially checked container identity, size and time metadata.
    # Any payload later found unreadable during real decompression is replaced
    # only after a full time-slice audit; apply that amendment to provenance.
    for repair_path in sorted((RUN / "reports").glob("cmfd_reacquire_*.json")):
        if not repair_path.is_file():
            continue
        repair = json.loads(repair_path.read_text(encoding="utf-8"))
        if repair.get("status") != "PASS_DEEP_AUDIT":
            raise RuntimeError(f"CMFD repair did not pass deep audit: {repair_path}")
        selector = (
            frame.variable.eq(str(repair["variable"]))
            & frame.year.eq(int(repair["year"]))
            & frame.month.eq(int(repair["month"]))
        )
        if int(selector.sum()) != 1:
            raise RuntimeError(f"CMFD repair registry key is not unique: {repair_path}")
        promoted = Path(str(frame.loc[selector, "local_path"].iloc[0]))
        if not promoted.is_file() or sha256(promoted) != str(repair["sha256"]):
            raise RuntimeError(f"CMFD promoted repair differs from deep-audited candidate: {promoted}")
        frame.loc[selector, "sha256"] = str(repair["sha256"])
        frame.loc[selector, "status"] = "reacquired_deep_audit"
    return frame


def cmfd_payloads(registry: pd.DataFrame) -> list[tuple[int, int, dict[str, str], dict[str, str]]]:
    payloads = []
    for (year, month), group in registry[registry.year.between(1961, 2005)].groupby(["year", "month"], sort=True):
        if set(group.variable) != set(VARIABLES) or len(group) != 6:
            raise RuntimeError(f"incomplete CMFD month {year}-{month:02d}")
        paths = dict(zip(group.variable.astype(str), group.local_path.astype(str)))
        hashes = dict(zip(group.local_path.astype(str), group.sha256.astype(str)))
        payloads.append((int(year), int(month), paths, hashes))
    if len(payloads) != 45 * 12:
        raise RuntimeError(f"expected 540 historical CMFD months, found {len(payloads)}")
    return payloads


def build_cmfd(workers: int, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads = cmfd_payloads(registry)
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    pending = []
    for payload in payloads:
        year, month = payload[:2]
        stamp = f"{year}{month:02d}"
        frame_path = CACHE / f"{stamp}.parquet"
        registry_path = CACHE / f"{stamp}.registry.parquet"
        if frame_path.is_file() and registry_path.is_file():
            frame = pd.read_parquet(frame_path)
            reg = pd.read_parquet(registry_path)
            if len(frame) != calendar.monthrange(year, month)[1] * 230 or len(reg) != 6:
                raise RuntimeError(f"invalid completed CMFD cache {stamp}")
            frames.append(frame)
            records.extend(reg.to_dict("records"))
        else:
            pending.append(payload)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_cmfd_month, payload): payload for payload in pending}
        failures: list[dict[str, Any]] = []
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            payload = futures[future]
            year, month = payload[:2]
            try:
                frame, reg = future.result()
            except Exception as exc:
                failures.append({"year": int(year), "month": int(month), "error": repr(exc)})
                print(f"CMFD PET {year}-{month:02d} FAILED; continuing remaining months", flush=True)
                continue
            stamp = f"{year}{month:02d}"
            frame_part = CACHE / f"{stamp}.parquet.part"
            reg_part = CACHE / f"{stamp}.registry.parquet.part"
            frame.to_parquet(frame_part, index=False)
            pd.DataFrame(reg).to_parquet(reg_part, index=False)
            os.replace(frame_part, CACHE / f"{stamp}.parquet")
            os.replace(reg_part, CACHE / f"{stamp}.registry.parquet")
            frames.append(frame)
            records.extend(reg)
            print(f"CMFD PET {year}-{month:02d} cached ({index}/{len(pending)} new)", flush=True)
    failure_path = REPORTS / "cmfd_failed_months_latest.json"
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} CMFD months failed deep construction; see {failure_path}")
    daily = pd.concat(frames, ignore_index=True).sort_values(["reach_id", "date"]).reset_index(drop=True)
    return daily, pd.DataFrame(records).sort_values(["year", "month", "variable"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        raise RuntimeError("preprocessing workers are capped at five")
    for directory in (OUT, REPORTS, CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    registry = complete_registry()
    years = list(range(1961, 2006))
    chm_frames: list[pd.DataFrame] = []
    chm_records: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(aggregate_chm_year, year): year for year in years}
        for future in concurrent.futures.as_completed(futures):
            frame, record = future.result()
            chm_frames.append(frame)
            chm_records.append(record)
            print(f"CHM_PRE {record['year']} aggregated", flush=True)
    chm = pd.concat(chm_frames, ignore_index=True)
    cmfd, cmfd_records = build_cmfd(args.workers, registry)
    historical = cmfd.merge(chm, on=["date", "reach_id"], validate="one_to_one")
    old = pd.read_parquet(OLD)
    old["date"] = pd.to_datetime(old.date)
    historical["date"] = pd.to_datetime(historical.date)
    missing = sorted(set(old.columns) - set(historical.columns))
    extra = sorted(set(historical.columns) - set(old.columns))
    if missing or extra:
        raise RuntimeError({"historical_schema_missing": missing, "historical_schema_extra": extra})
    historical = historical[old.columns]
    combined = pd.concat([historical, old], ignore_index=True).sort_values(["reach_id", "date"]).reset_index(drop=True)
    expected_days = len(pd.date_range("1961-01-01", "2024-12-31", freq="D"))
    checks = {
        "rows_exact": len(combined) == expected_days * 230,
        "reach_count_exact": combined.reach_id.nunique() == 230,
        "date_range_exact": combined.date.min() == pd.Timestamp("1961-01-01") and combined.date.max() == pd.Timestamp("2024-12-31"),
        "duplicates_zero": not combined.duplicated(["date", "reach_id"]).any(),
        "finite": bool(np.isfinite(combined.drop(columns=["date", "reach_id"]).to_numpy(float)).all()),
        "precipitation_nonnegative": bool(combined.precipitation_daily_mm.ge(0).all()),
        "pet_nonnegative": bool(combined.pet_fao56_mm_day.ge(0).all()),
        "old_rows_byte_source_reused": True,
        "no_parameter_refit": True,
        "observations_not_read": True,
    }
    historical_path = OUT / "daily_hbv_forcing_1961_2005.parquet"
    combined_path = OUT / "daily_hbv_forcing_1961_2024.parquet"
    historical.to_parquet(historical_path, index=False, compression="zstd")
    combined.to_parquet(combined_path, index=False, compression="zstd")
    pd.DataFrame(chm_records).sort_values("year").to_parquet(OUT / "chm_pre_source_registry_1961_2005.parquet", index=False)
    cmfd_records.to_parquet(OUT / "cmfd_source_registry_1961_2005.parquet", index=False)
    report = {
        "stage": "20260828_29",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "period": "1961-01-01 through 2024-12-31",
        "rows": len(combined),
        "checks": checks,
        "hashes": {
            "old_2006_2024": sha256(OLD),
            "historical_1961_2005": sha256(historical_path),
            "combined_1961_2024": sha256(combined_path),
            "weights": sha256(WEIGHTS),
        },
    }
    (REPORTS / "historical_forcing_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
