"""Download the separate 2023-2024 CMFD V2.0 three-hour forcing extension.

This acquisition is deliberately isolated from the registered 2006-2022
hydrology experiment.  It obtains only temp/pres/shum/wind/srad/lrad from the
official public TPDC endpoint, verifies every transfer, and records SHA-256.
It does not download precipitation and does not alter the current model input.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DISCOVERY = ROOT / "reports" / "tpdc_public_api_discovery.json"
RAW = ROOT / "inputs" / "cmfd_v2_0_03hr_2023_2024"
OUT = ROOT / "outputs"
REPORT = ROOT / "reports"
VARS = ("temp", "pres", "shum", "wind", "srad", "lrad")
YEAR_MONTH = re.compile(r"_(202[34])(0[1-9]|1[0-2])\.nc$")
DOWNLOAD_URL = "https://data.tpdc.ac.cn/file/file/batchDownloadByFileId?fileId={file_id}"
CHUNK = 8 * 1024 * 1024
PRINT_LOCK = threading.Lock()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def build_registry() -> pd.DataFrame:
    discovery = json.loads(DISCOVERY.read_text(encoding="utf-8"))
    variables = discovery["file_api"]["three_hour_variables"]
    rows: list[dict[str, object]] = []
    for variable in VARS:
        for entry in variables[variable]["entries"]:
            match = YEAR_MONTH.search(entry["name"])
            if not match:
                continue
            rows.append(
                {
                    "variable": variable,
                    "year": int(match.group(1)),
                    "month": int(match.group(2)),
                    "file_id": entry["id"],
                    "name": entry["name"],
                    "declared_size_bytes": int(entry["size"]),
                    "source_path": entry["path"],
                    "download_url": DOWNLOAD_URL.format(file_id=entry["id"]),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["year", "month", "variable"]).reset_index(drop=True)
    expected = len(VARS) * 2 * 12
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} registered files, found {len(frame)}")
    counts = frame.groupby(["variable", "year", "month"]).size()
    if not counts.eq(1).all():
        raise RuntimeError("CMFD 2023-2024 variable-year-month registry is not unique")
    if set(frame["variable"]) != set(VARS):
        raise RuntimeError("Unexpected variable set; precipitation is forbidden")
    return frame


def download_one(row: dict[str, object], retries: int = 4) -> dict[str, object]:
    variable = str(row["variable"])
    directory = RAW / variable
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / str(row["name"])
    registry_size = int(row["declared_size_bytes"])
    if target.exists() and target.stat().st_size > 0:
        actual_size = target.stat().st_size
        with target.open("rb") as check:
            if check.read(8) != b"\x89HDF\r\n\x1a\n":
                raise RuntimeError(f"Existing file is not NetCDF4/HDF5: {target}")
        return {
            **row,
            "local_path": str(target),
            "actual_size_bytes": actual_size,
            "registry_size_difference_bytes": actual_size - registry_size,
            "sha256": sha256(target),
            "status": "verified_existing",
        }

    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            if part.exists():
                part.unlink()
            with requests.post(
                str(row["download_url"]),
                json={"noToken": True},
                headers={"User-Agent": "SPARROW-CMFD-extension-acquisition/1.0"},
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                content_length = int(response.headers.get("Content-Length", "-1"))
                if content_length <= 0:
                    raise RuntimeError("Missing or invalid Content-Length")
                digest = hashlib.sha256()
                received = 0
                with part.open("wb") as stream:
                    for block in response.iter_content(CHUNK):
                        if not block:
                            continue
                        stream.write(block)
                        digest.update(block)
                        received += len(block)
                if received != content_length:
                    raise RuntimeError(f"received {received} != Content-Length {content_length}")
                with part.open("rb") as check:
                    if check.read(8) != b"\x89HDF\r\n\x1a\n":
                        raise RuntimeError("Downloaded file does not have a NetCDF4/HDF5 signature")
            os.replace(part, target)
            with PRINT_LOCK:
                print(
                    f"downloaded {row['year']}-{int(row['month']):02d} "
                    f"{variable} {received / 1e6:.1f} MB",
                    flush=True,
                )
            return {
                **row,
                "local_path": str(target),
                "actual_size_bytes": received,
                "registry_size_difference_bytes": received - registry_size,
                "sha256": digest.hexdigest(),
                "status": "downloaded",
            }
        except Exception as exc:
            if attempt == retries:
                return {
                    **row,
                    "local_path": str(target),
                    "actual_size_bytes": part.stat().st_size if part.exists() else 0,
                    "sha256": None,
                    "status": "failed",
                    "error": repr(exc),
                }
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    registry = build_registry()
    registry.to_parquet(
        OUT / "cmfd_v2_0_03hr_2023_2024_download_registry_planned.parquet", index=False
    )
    completed: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, row) for row in registry.to_dict("records")]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            completed.append(future.result())
            if index % 12 == 0 or index == len(futures):
                pd.DataFrame(completed).sort_values(["year", "month", "variable"]).to_parquet(
                    OUT / "cmfd_v2_0_03hr_2023_2024_download_registry_progress.parquet",
                    index=False,
                )
                print(f"progress {index}/{len(futures)}", flush=True)

    result = pd.DataFrame(completed).sort_values(["year", "month", "variable"]).reset_index(drop=True)
    result.to_parquet(
        OUT / "cmfd_v2_0_03hr_2023_2024_download_registry.parquet", index=False
    )
    failures = result.loc[result.status.eq("failed")]
    report = {
        "source": "TPDC CMFD V2.0",
        "doi": "10.11888/Atmos.tpdc.302088",
        "dataset_id": "e60dfd96-5fd8-493f-beae-e8e5d24dece4",
        "variables": list(VARS),
        "precipitation_downloaded": False,
        "period": "2023-01 through 2024-12",
        "registered_file_count": int(len(registry)),
        "attempted_file_count": int(len(result)),
        "verified_file_count": int(result.status.ne("failed").sum()),
        "failed_file_count": int(len(failures)),
        "verified_bytes": int(result.loc[result.status.ne("failed"), "actual_size_bytes"].sum()),
        "sha256_recorded_for_every_verified_file": bool(
            result.loc[result.status.ne("failed"), "sha256"].notna().all()
        ),
        "public_no_token_download": True,
        "isolated_from_2006_2022_formal_model": True,
        "status": "PASS" if len(result) == len(registry) and failures.empty else "INCOMPLETE",
    }
    (REPORT / "cmfd_v2_0_03hr_2023_2024_acquisition.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError(failures[["name", "error"]].to_dict("records"))


if __name__ == "__main__":
    main()
