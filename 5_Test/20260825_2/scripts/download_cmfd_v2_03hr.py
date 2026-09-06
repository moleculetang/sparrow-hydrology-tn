"""Download the registered public CMFD V2.0 3-hour forcing files.

The official TPDC endpoint is public for this dataset.  The script selects
only temp/pres/shum/wind/srad/lrad for 2006-2022, verifies the server-declared
size, computes SHA-256 and writes an auditable registry.  It does not access
discharge or TN observations.
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
RAW = ROOT / "inputs" / "cmfd_v2_0_03hr_2006_2022"
OUT = ROOT / "outputs"
REPORT = ROOT / "reports"
VARS = ("temp", "pres", "shum", "wind", "srad", "lrad")
YEAR_MONTH = re.compile(r"_(20(?:0[6-9]|1[0-9]|2[0-2]))(0[1-9]|1[0-2])\.nc$")
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
        entries = variables[variable]["entries"]
        for entry in entries:
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
    expected = len(VARS) * 17 * 12
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} registered files, found {len(frame)}")
    counts = frame.groupby(["variable", "year", "month"]).size()
    if not counts.eq(1).all():
        raise RuntimeError("CMFD variable-year-month registry is not unique")
    return frame


def download_one(row: dict[str, object], retries: int = 4) -> dict[str, object]:
    variable = str(row["variable"])
    directory = RAW / variable
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / str(row["name"])
    registry_size = int(row["declared_size_bytes"])
    if target.exists() and target.stat().st_size > 0:
        actual_size = target.stat().st_size
        if actual_size != registry_size:
            return {
                **row,
                "local_path": str(target),
                "actual_size_bytes": actual_size,
                "registry_size_difference_bytes": actual_size - registry_size,
                "sha256": None,
                "status": "failed",
                "error": "existing_file_size_mismatch_requires_manual_quarantine",
            }
        with target.open("rb") as check:
            if check.read(8) != b"\x89HDF\r\n\x1a\n":
                return {
                    **row,
                    "local_path": str(target),
                    "actual_size_bytes": actual_size,
                    "registry_size_difference_bytes": 0,
                    "sha256": None,
                    "status": "failed",
                    "error": "existing_file_invalid_hdf5_signature_requires_manual_quarantine",
                }
        return {
            **row,
            "local_path": str(target),
            "actual_size_bytes": actual_size,
            "registry_size_difference_bytes": 0,
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
                headers={"User-Agent": "SPARROW-CMFD-input-acquisition/1.0"},
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
                        raise RuntimeError("Downloaded file does not have an HDF5/NetCDF4 signature")
            os.replace(part, target)
            with PRINT_LOCK:
                print(f"downloaded {row['year']}-{int(row['month']):02d} {variable} {received / 1e6:.1f} MB", flush=True)
            return {
                **row,
                "local_path": str(target),
                "actual_size_bytes": received,
                "registry_size_difference_bytes": received - registry_size,
                "sha256": digest.hexdigest(),
                "status": "downloaded",
            }
        except Exception as exc:  # retry only this exact registered file
            if attempt == retries:
                return {**row, "local_path": str(target), "actual_size_bytes": part.stat().st_size if part.exists() else 0, "sha256": None, "status": "failed", "error": repr(exc)}
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="QA-only cap; omit for the registered acquisition")
    parser.add_argument(
        "--repair",
        default=None,
        help="Repair one quarantined registered payload as VARIABLE:YYYYMM and merge it into the full registry",
    )
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    registry = build_registry()
    registry.to_parquet(OUT / "cmfd_v2_0_03hr_download_registry_planned.parquet", index=False)
    if args.repair is not None:
        variable, stamp = args.repair.split(":", 1)
        if variable not in VARS or len(stamp) != 6 or not stamp.isdigit():
            raise RuntimeError("--repair must be VARIABLE:YYYYMM for a registered variable")
        year, month = int(stamp[:4]), int(stamp[4:])
        selected = registry.loc[
            registry.variable.eq(variable) & registry.year.eq(year) & registry.month.eq(month)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Repair target is not unique: {args.repair}")
        result_row = download_one(selected.iloc[0].to_dict())
        if result_row["status"] == "failed":
            raise RuntimeError(result_row)
        full_path = OUT / "cmfd_v2_0_03hr_download_registry.parquet"
        if not full_path.is_file():
            raise RuntimeError("A full download registry must exist before a payload repair")
        full = pd.read_parquet(full_path)
        select_full = full.variable.eq(variable) & full.year.eq(year) & full.month.eq(month)
        if int(select_full.sum()) != 1 or len(full) != len(registry):
            raise RuntimeError("Existing full registry does not match the registered acquisition")
        previous = full.loc[select_full].iloc[0].to_dict()
        for key, value in result_row.items():
            full.loc[select_full, key] = value
        full = full.sort_values(["year", "month", "variable"]).reset_index(drop=True)
        full.to_parquet(full_path, index=False)
        log_path = OUT / "cmfd_v2_0_03hr_payload_repair_log.parquet"
        event = pd.DataFrame(
            [{
                "variable": variable,
                "year": year,
                "month": month,
                "previous_sha256": previous.get("sha256"),
                "replacement_sha256": result_row.get("sha256"),
                "reason": "HDF5 payload read failure during complete payload validation",
                "quarantined_path": str(RAW / variable / (str(result_row["name"]) + ".corrupt")),
                "replacement_status": result_row.get("status"),
            }]
        )
        if log_path.is_file():
            event = pd.concat([pd.read_parquet(log_path), event], ignore_index=True)
        event.to_parquet(log_path, index=False)
        acquisition = {
            "source": "TPDC CMFD V2.0",
            "doi": "10.11888/Atmos.tpdc.302088",
            "dataset_id": "e60dfd96-5fd8-493f-beae-e8e5d24dece4",
            "variables": list(VARS),
            "period": "2006-01 through 2022-12",
            "registered_file_count": int(len(registry)),
            "attempted_file_count": int(len(full)),
            "verified_file_count": int(full.status.ne("failed").sum()),
            "failed_file_count": int(full.status.eq("failed").sum()),
            "verified_bytes": int(full.loc[full.status.ne("failed"), "actual_size_bytes"].sum()),
            "public_no_token_download": True,
            "container_integrity_scope": "declared byte size, HDF5 signature and SHA-256",
            "payload_decompression_validation": "performed by build_daily_cmfd_pet.py while reading every registered three-hour timestep",
            "payload_repair_event_count": int(len(event)),
            "status": "PASS" if len(full) == len(registry) and full.status.ne("failed").all() else "INCOMPLETE",
        }
        (REPORT / "cmfd_v2_0_03hr_acquisition.json").write_text(json.dumps(acquisition, indent=2), encoding="utf-8")
        print(json.dumps(acquisition, indent=2), flush=True)
        return
    records = registry.to_dict("records")
    if args.limit is not None:
        records = records[: args.limit]
    completed: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, row) for row in records]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            completed.append(future.result())
            if index % 12 == 0 or index == len(futures):
                pd.DataFrame(completed).sort_values(["year", "month", "variable"]).to_parquet(
                    OUT / "cmfd_v2_0_03hr_download_registry_progress.parquet", index=False
                )
                print(f"progress {index}/{len(futures)}", flush=True)
    result = pd.DataFrame(completed).sort_values(["year", "month", "variable"]).reset_index(drop=True)
    result.to_parquet(OUT / "cmfd_v2_0_03hr_download_registry.parquet", index=False)
    failures = result.loc[result.status.eq("failed")]
    report = {
        "source": "TPDC CMFD V2.0",
        "doi": "10.11888/Atmos.tpdc.302088",
        "dataset_id": "e60dfd96-5fd8-493f-beae-e8e5d24dece4",
        "variables": list(VARS),
        "period": "2006-01 through 2022-12",
        "registered_file_count": int(len(registry)),
        "attempted_file_count": int(len(result)),
        "verified_file_count": int(result.status.ne("failed").sum()),
        "failed_file_count": int(len(failures)),
        "verified_bytes": int(result.loc[result.status.ne("failed"), "actual_size_bytes"].sum()),
        "public_no_token_download": True,
        "container_integrity_scope": "declared byte size, HDF5 signature and SHA-256",
        "payload_decompression_validation": "performed by build_daily_cmfd_pet.py while reading every registered three-hour timestep",
        "status": "PASS" if len(result) == len(registry) and failures.empty else "INCOMPLETE",
    }
    (REPORT / "cmfd_v2_0_03hr_acquisition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS" and args.limit is None:
        raise RuntimeError(failures[["name", "error"]].to_dict("records"))


if __name__ == "__main__":
    main()
