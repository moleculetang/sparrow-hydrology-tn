"""Acquire CHM_PRE V2 daily 2025, validate it, then aggregate to 230 Reach."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "precipitation" / "chm_pre_v2" / "daily"
TARGET = RAW / "CHM_PRE_V2_daily_2025.nc"
REFERENCE = RAW / "CHM_PRE_V2_daily_2024.nc"
WEIGHTS = ROOT / "5_Test" / "20260813_30" / "inputs" / "spatial" / "grid_overlap_weights.parquet"
FORCING_2006_2024 = OUT / "daily_hbv_forcing_2006_2024.parquet"
METADATA_ID = "e5c335d9-cbb9-48a6-ba35-d67dd614bb8c"
DAILY_FOLDER_ID = "fedce9b3-94f1-4141-ad12-5122b0cca259"
EXPECTED_FILE_ID = "6dd14fc2-76c2-4f22-9306-68c9961f31ed"
API_LIST = f"https://data.tpdc.ac.cn/file/file/getFileDataList?parentId={DAILY_FOLDER_ID}"
DOWNLOAD = "https://data.tpdc.ac.cn/file/file/batchDownloadByFileId?fileId={file_id}"
CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def fetch_registry_entry() -> dict[str, object]:
    response = requests.get(API_LIST, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != "200":
        raise RuntimeError(payload)
    matches = [entry for entry in payload.get("data", []) if entry.get("name") == TARGET.name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {TARGET.name} entry, found {matches}")
    entry = matches[0]
    if entry.get("id") != EXPECTED_FILE_ID or entry.get("metadataId") != METADATA_ID:
        raise RuntimeError(f"CHM_PRE 2025 registry identity changed: {entry}")
    return entry


def download(entry: dict[str, object]) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    if TARGET.is_file() and TARGET.stat().st_size > 0:
        with TARGET.open("rb") as stream:
            if stream.read(8) != b"\x89HDF\r\n\x1a\n":
                raise RuntimeError(f"Existing target is not NetCDF4/HDF5: {TARGET}")
        return "verified_existing"
    part = TARGET.with_suffix(TARGET.suffix + ".part")
    for attempt in range(1, 6):
        try:
            if part.exists():
                part.unlink()
            with requests.post(
                DOWNLOAD.format(file_id=entry["id"]),
                json={"noToken": True},
                headers={"User-Agent": "SPARROW-CHM-PRE-2025-acquisition/1.0"},
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                expected = int(response.headers.get("Content-Length", "-1"))
                if expected <= 0:
                    raise RuntimeError("Missing Content-Length")
                received = 0
                with part.open("wb") as stream:
                    for block in response.iter_content(CHUNK):
                        if block:
                            stream.write(block)
                            received += len(block)
                if received != expected:
                    raise RuntimeError(f"received {received} != Content-Length {expected}")
                with part.open("rb") as stream:
                    if stream.read(8) != b"\x89HDF\r\n\x1a\n":
                        raise RuntimeError("Downloaded payload is not NetCDF4/HDF5")
            os.replace(part, TARGET)
            return "downloaded"
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def decode_dates(handle: h5py.File) -> pd.DatetimeIndex:
    units = decode_text(handle["time"].attrs["units"])
    prefix = "days since "
    if not units.startswith(prefix):
        raise RuntimeError(f"Unsupported CHM time units: {units}")
    origin = pd.Timestamp(units[len(prefix):])
    return pd.DatetimeIndex(origin + pd.to_timedelta(np.asarray(handle["time"][:], dtype=int), unit="D"))


def preprocess() -> tuple[pd.DataFrame, dict[str, object]]:
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq("CHM")].sort_values(["reach_id", "grid_i", "grid_j"]).reset_index(drop=True)
    reach_vector = weights.reach_id.to_numpy(int)
    reaches, starts = np.unique(reach_vector, return_index=True)
    w = weights.weight.to_numpy(float)
    if not np.array_equal(reaches, np.arange(1, 231)) or float(np.max(np.abs(np.add.reduceat(w, starts) - 1.0))) > 1.0e-8:
        raise RuntimeError("Frozen CHM weights are invalid")
    ilat = weights.grid_i.to_numpy(int)
    ilon = weights.grid_j.to_numpy(int)
    with h5py.File(REFERENCE, "r") as reference:
        reference_lat = np.asarray(reference["lat"][:], dtype=float)
        reference_lon = np.asarray(reference["lon"][:], dtype=float)
    with h5py.File(TARGET, "r") as handle:
        if set(handle.keys()) != {"prec", "time", "lat", "lon"}:
            raise RuntimeError(f"Unexpected CHM_PRE variables: {list(handle.keys())}")
        dates = decode_dates(handle)
        expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        if not dates.equals(expected_dates):
            raise RuntimeError("CHM_PRE 2025 daily calendar is incomplete")
        units = decode_text(handle["prec"].attrs.get("units", ""))
        if units.strip().lower() != "mm/day":
            raise RuntimeError(f"Unexpected precipitation units: {units}")
        if not np.array_equal(np.asarray(handle["lat"][:], dtype=float), reference_lat) or not np.array_equal(np.asarray(handle["lon"][:], dtype=float), reference_lon):
            raise RuntimeError("CHM_PRE 2025 grid differs from 2024")
        annual = np.empty((365, 230), dtype=np.float64)
        minimum_coverage = 1.0
        for day_index in range(365):
            field = np.asarray(handle["prec"][day_index], dtype=float)
            values = field[ilat, ilon]
            valid = np.isfinite(values) & (values < 1.0e19)
            coverage = np.add.reduceat(w * valid, starts)
            minimum_coverage = min(minimum_coverage, float(coverage.min()))
            if float(coverage.min()) < 0.999:
                raise RuntimeError(f"Insufficient CHM support: {coverage.min()}")
            annual[day_index] = np.add.reduceat(np.where(valid, values * w, 0.0), starts) / coverage
    frame = pd.DataFrame({
        "reach_id": np.tile(reaches, len(dates)),
        "date": np.repeat(dates.to_numpy(), len(reaches)),
        "precipitation_daily_mm": annual.reshape(-1),
    }).sort_values(["reach_id", "date"]).reset_index(drop=True)
    if len(frame) != 83_950 or frame.duplicated(["reach_id", "date"]).any() or not np.isfinite(frame.precipitation_daily_mm).all() or frame.precipitation_daily_mm.lt(0).any():
        raise RuntimeError("CHM_PRE 2025 Reach product failed core QA")
    return frame, {"days": 365, "minimum_valid_weight": minimum_coverage, "units": units}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    entry = fetch_registry_entry()
    acquisition_status = download(entry)
    frame, qa = preprocess()
    output = OUT / "chm_pre_daily_by_reach_2025.parquet"
    frame.to_parquet(output, index=False)
    historical = pd.read_parquet(FORCING_2006_2024, columns=["reach_id", "date", "precipitation_daily_mm"])
    historical["date"] = pd.to_datetime(historical.date)
    combined = pd.concat([historical, frame], ignore_index=True).sort_values(["reach_id", "date"]).reset_index(drop=True)
    combined_output = OUT / "chm_pre_daily_by_reach_2006_2025_precipitation_only.parquet"
    combined.to_parquet(combined_output, index=False)
    checks = {
        "raw_file_exists": TARGET.is_file(),
        "raw_size_matches_registry": TARGET.stat().st_size == int(entry["size"]),
        "reach_2025_rows_exact": len(frame) == 83_950,
        "combined_rows_exact": len(combined) == 1_680_150,
        "combined_date_range_exact": combined.date.min() == pd.Timestamp("2006-01-01") and combined.date.max() == pd.Timestamp("2025-12-31"),
        "combined_duplicates_zero": not combined.duplicated(["reach_id", "date"]).any(),
        "reach_count_exact": combined.reach_id.nunique() == 230,
        "support_ge_0p999": qa["minimum_valid_weight"] >= 0.999,
    }
    report = {
        "status": "CHM_PRE_2025_ACQUIRED_AND_PREPROCESSED" if all(checks.values()) else "FAIL",
        "source_page": "https://www.tpdc.ac.cn/zh-hans/data/e5c335d9-cbb9-48a6-ba35-d67dd614bb8c",
        "file_id": entry["id"],
        "source_path": entry["path"],
        "acquisition_status": acquisition_status,
        "raw_path": str(TARGET),
        "raw_size_bytes": TARGET.stat().st_size,
        "raw_sha256": sha256(TARGET),
        "reach_output": str(output),
        "reach_output_sha256": sha256(output),
        "precipitation_only_2006_2025": str(combined_output),
        "precipitation_only_2006_2025_sha256": sha256(combined_output),
        "minimum_valid_weight": qa["minimum_valid_weight"],
        "checks": checks,
        "claim_boundary": "2025 is precipitation-only; it is not a complete hydrologic forcing or discharge simulation",
    }
    (REPORTS / "chm_pre_2025_acquisition_preprocessing.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] == "FAIL":
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
