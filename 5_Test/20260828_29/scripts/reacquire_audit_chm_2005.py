"""Reacquire the official CHM_PRE V2 2005 file without overwriting raw data.

The downloaded candidate is retained in the run work directory and is only
audited here.  Promotion into the authoritative raw directory is a separate,
explicit step after all checks pass.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_29"
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "precipitation" / "chm_pre_v2" / "daily"
CURRENT = RAW / "CHM_PRE_V2_daily_2005.nc"
CANDIDATE = RUN / "work" / "chm_pre_2005_official" / "CHM_PRE_V2_daily_2005.nc"
REPORT = RUN / "reports" / "chm_pre_2005_official_reacquisition_audit.json"
WEIGHTS = ROOT / "5_Test" / "20260813_30" / "inputs" / "spatial" / "grid_overlap_weights.parquet"
FILE_ID = "4cfe7335-8104-4f18-b112-c54b7b2b1e8d"
REGISTERED_SIZE = 672_795_741
DOWNLOAD_URL = f"https://data.tpdc.ac.cn/file/file/batchDownloadByFileId?fileId={FILE_ID}"
CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_candidate() -> str:
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    if CANDIDATE.is_file() and CANDIDATE.stat().st_size == REGISTERED_SIZE:
        return "verified_existing_candidate"
    part = CANDIDATE.with_suffix(".nc.part")
    for attempt in range(1, 6):
        try:
            if part.exists():
                part.unlink()
            with requests.post(
                DOWNLOAD_URL,
                json={"noToken": True},
                headers={"User-Agent": "SPARROW-CHM-PRE-2005-audit/1.0"},
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                content_length = int(response.headers.get("Content-Length", "-1"))
                if content_length != REGISTERED_SIZE:
                    raise RuntimeError(
                        f"official Content-Length {content_length} differs from registry {REGISTERED_SIZE}"
                    )
                received = 0
                with part.open("wb") as stream:
                    for block in response.iter_content(CHUNK):
                        if block:
                            stream.write(block)
                            received += len(block)
                if received != REGISTERED_SIZE:
                    raise RuntimeError(f"received {received} of {REGISTERED_SIZE} bytes")
                with part.open("rb") as stream:
                    if stream.read(8) != b"\x89HDF\r\n\x1a\n":
                        raise RuntimeError("candidate is not NetCDF4/HDF5")
            part.replace(CANDIDATE)
            return "downloaded_candidate"
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def audit(path: Path) -> dict[str, object]:
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq("CHM")].sort_values(
        ["reach_id", "grid_i", "grid_j"]
    )
    reach_vector = weights.reach_id.to_numpy(int)
    reaches, starts = np.unique(reach_vector, return_index=True)
    ilat = weights.grid_i.to_numpy(int)
    ilon = weights.grid_j.to_numpy(int)
    values = weights.weight.to_numpy(float)
    affected_dates: list[str] = []
    affected_reaches: set[int] = set()
    minimum_coverage = 1.0
    with h5py.File(path, "r") as handle:
        shape = tuple(int(v) for v in handle["prec"].shape)
        dates = pd.DatetimeIndex(
            pd.Timestamp("1900-01-01")
            + pd.to_timedelta(np.asarray(handle["time"][:], dtype=int), unit="D")
        )
        # Use the recorded origin rather than assuming 1900 when it differs.
        units = handle["time"].attrs["units"]
        if isinstance(units, bytes):
            units = units.decode("utf-8")
        origin = pd.Timestamp(str(units).removeprefix("days since "))
        dates = pd.DatetimeIndex(origin + pd.to_timedelta(np.asarray(handle["time"][:], dtype=int), unit="D"))
        for day, date in enumerate(dates):
            source = np.asarray(handle["prec"][day], dtype=float)[ilat, ilon]
            valid = np.isfinite(source) & (source < 1.0e19)
            coverage = np.add.reduceat(values * valid, starts)
            minimum_coverage = min(minimum_coverage, float(coverage.min()))
            bad = reaches[coverage < 0.999]
            if len(bad):
                affected_dates.append(date.strftime("%Y-%m-%d"))
                affected_reaches.update(int(v) for v in bad)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "precipitation_shape": shape,
        "date_min": dates.min().strftime("%Y-%m-%d"),
        "date_max": dates.max().strftime("%Y-%m-%d"),
        "minimum_reach_weight_coverage": minimum_coverage,
        "affected_date_count": len(affected_dates),
        "affected_dates": affected_dates,
        "affected_reach_count": len(affected_reaches),
        "affected_reaches": sorted(affected_reaches),
    }


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    acquisition = download_candidate()
    current = audit(CURRENT)
    candidate = audit(CANDIDATE)
    checks = {
        "candidate_size_matches_registry": candidate["bytes"] == REGISTERED_SIZE,
        "candidate_calendar_exact": candidate["date_min"] == "2005-01-01"
        and candidate["date_max"] == "2005-12-31",
        "candidate_reach_support_complete": candidate["affected_date_count"] == 0,
    }
    result = {
        "status": "PASS_OFFICIAL_CANDIDATE" if all(checks.values()) else "OFFICIAL_SOURCE_HAS_GAPS",
        "acquisition": acquisition,
        "official_file_id": FILE_ID,
        "official_registered_size": REGISTERED_SIZE,
        "current": current,
        "candidate": candidate,
        "checks": checks,
        "promotion_performed": False,
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
