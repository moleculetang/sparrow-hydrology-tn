"""Register, link and adaptively download CMFD V2.0 PET forcing.

Only temp/pres/shum/wind/srad/lrad are authorized.  Existing verified
2006-2024 payloads are hard-linked into the authoritative raw data tree.
Missing 1961-2005 payloads are downloaded from the public TPDC endpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
import requests


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_27"
DISCOVERY = ROOT / "5_Test" / "20260825_2" / "reports" / "tpdc_public_api_discovery.json"
LEGACY = (
    ROOT / "5_Test" / "20260825_2" / "inputs" / "cmfd_v2_0_03hr_2006_2022",
    ROOT / "5_Test" / "20260825_2" / "inputs" / "cmfd_v2_0_03hr_2023_2024",
)
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "meteorology" / "cmfd_v2_0" / "03hr"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
VARIABLES = ("temp", "pres", "shum", "wind", "srad", "lrad")
PATTERN = re.compile(r"_(19(?:6[1-9]|[7-9][0-9])|20(?:0[0-9]|1[0-9]|2[0-4]))(0[1-9]|1[0-2])\.nc$")
URL = "https://data.tpdc.ac.cn/file/file/batchDownloadByFileId?fileId={file_id}"
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
CHUNK = 8 * 1024 * 1024
PRINT_LOCK = threading.Lock()
SESSION_LOCAL = threading.local()


def session() -> requests.Session:
    value = getattr(SESSION_LOCAL, "value", None)
    if value is None:
        value = requests.Session()
        value.headers.update({"User-Agent": "SPARROW-CMFD-long-acquisition/1.0"})
        adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
        value.mount("https://", adapter)
        SESSION_LOCAL.value = value
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def has_hdf5_signature(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 8:
        return False
    with path.open("rb") as stream:
        return stream.read(8) == HDF5_SIGNATURE


def validate_payload(path: Path, variable: str, expected_size: int) -> dict[str, Any]:
    actual = path.stat().st_size
    if not has_hdf5_signature(path):
        raise RuntimeError(f"invalid HDF5 signature: {path}")
    with h5py.File(path, "r") as handle:
        if variable not in handle or "time" not in handle:
            raise RuntimeError(f"missing {variable} or time dataset: {path}")
        time_steps = int(handle["time"].shape[0])
        if time_steps not in {224, 232, 240, 248}:
            raise RuntimeError(f"unexpected three-hour time count {time_steps}: {path}")
    return {
        "actual_size_bytes": actual,
        "registry_size_difference_bytes": actual - expected_size,
        "time_steps": time_steps,
        "sha256": sha256(path),
    }


def registry() -> pd.DataFrame:
    discovery = json.loads(DISCOVERY.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for variable in VARIABLES:
        entries = discovery["file_api"]["three_hour_variables"][variable]["entries"]
        for entry in entries:
            match = PATTERN.search(str(entry["name"]))
            if not match:
                continue
            year, month = int(match.group(1)), int(match.group(2))
            if not 1961 <= year <= 2024:
                continue
            rows.append(
                {
                    "variable": variable,
                    "year": year,
                    "month": month,
                    "file_id": str(entry["id"]),
                    "name": str(entry["name"]),
                    "declared_size_bytes": int(entry["size"]),
                    "source_path": str(entry["path"]),
                    "download_url": URL.format(file_id=entry["id"]),
                    "local_path": str(RAW / variable / str(entry["name"])),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["year", "month", "variable"]).reset_index(drop=True)
    expected = len(VARIABLES) * 64 * 12
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} files, found {len(frame)}")
    if not frame.groupby(["variable", "year", "month"]).size().eq(1).all():
        raise RuntimeError("registered CMFD keys are not unique")
    return frame


def find_existing(name: str, variable: str) -> Path | None:
    for base in LEGACY:
        candidate = base / variable / name
        if candidate.is_file():
            return candidate
    return None


def link_existing(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    pending: list[int] = []
    for index, row in frame.iterrows():
        target = Path(row.local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = find_existing(str(row["name"]), str(row.variable))
        if target.exists():
            details = validate_payload(target, str(row.variable), int(row.declared_size_bytes))
            records.append({**row.to_dict(), **details, "status": "verified_authoritative"})
        elif source is not None:
            source_details = validate_payload(source, str(row.variable), int(row.declared_size_bytes))
            record = row.to_dict()
            try:
                os.link(source, target)
                if not os.path.samefile(source, target):
                    raise RuntimeError(f"hard-link verification failed: {source} -> {target}")
                status = "hardlinked_existing"
            except OSError as exc:
                # Several archived CMFD payloads carry a Windows filesystem
                # attribute that forbids hard links.  Preserve the one existing
                # payload and make the acquisition registry its authority.
                record["local_path"] = str(source)
                record["hardlink_fallback_reason"] = repr(exc)
                status = "registered_existing_external"
            records.append({**record, **source_details, "status": status, "source_existing_path": str(source)})
        else:
            pending.append(index)
    return frame.loc[pending].reset_index(drop=True), records


def resume_verified_registry(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]] | None:
    path = OUT / "cmfd_1961_2024_acquisition_progress.parquet"
    if not path.is_file():
        return None
    completed = pd.read_parquet(path)
    required = {"variable", "year", "month", "status", "local_path", "actual_size_bytes", "sha256"}
    if not required.issubset(completed.columns):
        return None
    completed = completed.loc[completed.status.ne("failed")].copy()
    if completed.duplicated(["variable", "year", "month"]).any():
        raise RuntimeError("resume registry contains duplicate CMFD keys")
    for row in completed.itertuples(index=False):
        path_value = Path(str(row.local_path))
        if not path_value.is_file() or path_value.stat().st_size != int(row.actual_size_bytes):
            raise RuntimeError(f"resume registry payload changed: {path_value}")
        if not isinstance(row.sha256, str) or len(row.sha256) != 64:
            raise RuntimeError(f"resume registry lacks a valid SHA-256: {path_value}")
    keys = completed[["variable", "year", "month"]].copy()
    merged = frame.merge(keys.assign(_done=True), on=["variable", "year", "month"], how="left")
    pending = merged.loc[merged._done.isna(), frame.columns].reset_index(drop=True)
    return pending, completed.to_dict("records")


def download_one(row: dict[str, Any], retries: int = 6) -> dict[str, Any]:
    target = Path(str(row["local_path"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    expected_size = int(row["declared_size_bytes"])
    started = time.perf_counter()
    total_received = 0
    range_supported: bool | None = None
    transfer_expected_size: int | None = None
    retry_events: list[str] = []
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers: dict[str, str] = {}
        if 0 < offset < expected_size:
            headers["Range"] = f"bytes={offset}-"
        try:
            with session().post(
                str(row["download_url"]),
                json={"noToken": True},
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as response:
                if response.status_code == 429:
                    raise RuntimeError(f"HTTP429 retry_after={response.headers.get('Retry-After')}")
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if offset > 0:
                    range_supported = append
                if not append:
                    offset = 0
                content_length = int(response.headers.get("Content-Length", "-1"))
                if content_length <= 0:
                    raise RuntimeError("missing or invalid HTTP Content-Length")
                if append:
                    content_range = response.headers.get("Content-Range", "")
                    if "/" in content_range:
                        transfer_expected_size = int(content_range.rsplit("/", 1)[1])
                    else:
                        transfer_expected_size = offset + content_length
                else:
                    transfer_expected_size = content_length
                mode = "ab" if append else "wb"
                with part.open(mode) as stream:
                    for block in response.iter_content(CHUNK):
                        if block:
                            stream.write(block)
                            total_received += len(block)
            actual = part.stat().st_size
            if transfer_expected_size is None or actual != transfer_expected_size:
                raise RuntimeError(f"partial size {actual} != HTTP transfer size {transfer_expected_size}")
            os.replace(part, target)
            details = validate_payload(target, str(row["variable"]), expected_size)
            elapsed = max(time.perf_counter() - started, 1e-9)
            with PRINT_LOCK:
                print(f"downloaded {row['year']}-{int(row['month']):02d} {row['variable']} {actual / 1e6:.1f} MB attempts={attempt}", flush=True)
            return {
                **row,
                **details,
                "status": "downloaded",
                "attempts": attempt,
                "elapsed_seconds": elapsed,
                "http_transfer_size_bytes": transfer_expected_size,
                "effective_MB_s": actual / elapsed / 1e6,
                "range_supported": range_supported,
                "retry_events": ";".join(retry_events),
            }
        except Exception as exc:
            retry_events.append(repr(exc))
            if attempt == retries:
                return {
                    **row,
                    "status": "failed",
                    "attempts": attempt,
                    "actual_size_bytes": part.stat().st_size if part.exists() else 0,
                    "elapsed_seconds": time.perf_counter() - started,
                    "range_supported": range_supported,
                    "retry_events": ";".join(retry_events),
                    "error": repr(exc),
                }
            retry_after = 0.0
            if "retry_after=" in retry_events[-1]:
                try:
                    retry_after = float(retry_events[-1].split("retry_after=", 1)[1].split("'", 1)[0])
                except Exception:
                    retry_after = 0.0
            time.sleep(max(retry_after, min(60.0, 2.0**attempt + random.random() * 2.0)))
    raise AssertionError("unreachable")


def write_progress(records: list[dict[str, Any]], pending_total: int, workers: int) -> None:
    result = pd.DataFrame(records).sort_values(["year", "month", "variable"])
    result.to_parquet(OUT / "cmfd_1961_2024_acquisition_progress.parquet", index=False)
    failures = int(result.status.eq("failed").sum()) if "status" in result else 0
    report = {
        "registered_file_count": 4608,
        "completed_record_count": len(result),
        "pending_download_count_at_start": pending_total,
        "failed_file_count": failures,
        "current_workers": workers,
        "verified_bytes": int(result.loc[result.status.ne("failed"), "actual_size_bytes"].fillna(0).sum()),
        "updated_epoch": time.time(),
    }
    (REPORTS / "cmfd_acquisition_progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_batches(pending: pd.DataFrame, records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    queue = pending.to_dict("records")
    workers = args.workers
    previous_rate: float | None = None
    pending_total = len(queue)
    while queue:
        batch_size = min(len(queue), max(24, workers * 2))
        batch, queue = queue[:batch_size], queue[batch_size:]
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            batch_result = list(executor.map(download_one, batch))
        elapsed = max(time.perf_counter() - started, 1e-9)
        records.extend(batch_result)
        bytes_ok = sum(int(item.get("actual_size_bytes", 0)) for item in batch_result if item["status"] != "failed")
        rate = bytes_ok / elapsed / 1e6
        errors = [item for item in batch_result if item["status"] == "failed"]
        retry_pressure = sum(int(item.get("attempts", 1)) - 1 for item in batch_result) / max(len(batch_result), 1)
        rate_drop = previous_rate is not None and rate < previous_rate * 0.80
        if len(errors) / len(batch_result) > 0.02 or retry_pressure > 0.5 or rate_drop:
            workers = max(args.min_workers, workers - 4)
        elif previous_rate is None or rate > previous_rate * 1.05:
            workers = min(args.max_workers, workers + 2)
        previous_rate = rate
        write_progress(records, pending_total, workers)
        print(f"batch completed={len(batch_result)} remaining={len(queue)} rate={rate:.1f} MB/s next_workers={workers} failures={len(errors)}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--min-workers", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--registry-only", action="store_true")
    parser.add_argument("--reverify-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.min_workers <= args.workers <= args.max_workers:
        raise RuntimeError("workers must lie within min/max")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = registry()
    frame.to_parquet(OUT / "cmfd_1961_2024_registry_planned.parquet", index=False)
    resumed = None if args.reverify_existing else resume_verified_registry(frame)
    if resumed is None:
        pending, records = link_existing(frame)
    else:
        pending, records = resumed
    if args.limit is not None:
        pending = pending.head(args.limit).copy()
    preflight = {
        "status": "REGISTERED",
        "registered_files": len(frame),
        "verified_or_linked_existing": len(records),
        "pending_downloads": len(pending),
        "pending_bytes": int(pending.declared_size_bytes.sum()),
        "variables": list(VARIABLES),
        "period": "1961-01 through 2024-12",
        "precipitation_forbidden": True,
        "adaptive_workers": {"initial": args.workers, "minimum": args.min_workers, "maximum": args.max_workers},
    }
    (REPORTS / "cmfd_acquisition_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(json.dumps(preflight, indent=2), flush=True)
    if args.registry_only:
        write_progress(records, len(pending), args.workers)
        return
    records = run_batches(pending, records, args)
    result = pd.DataFrame(records).sort_values(["year", "month", "variable"]).reset_index(drop=True)
    result.to_parquet(OUT / "cmfd_1961_2024_acquisition_registry.parquet", index=False)
    failures = result[result.status.eq("failed")]
    final = {
        **preflight,
        "status": "PASS" if len(result) == len(frame) and failures.empty else "INCOMPLETE",
        "completed_records": len(result),
        "failed_files": len(failures),
        "verified_bytes": int(result.loc[result.status.ne("failed"), "actual_size_bytes"].sum()),
        "sha256_complete": bool(result.loc[result.status.ne("failed"), "sha256"].notna().all()),
    }
    (REPORTS / "cmfd_acquisition_final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)
    if final["status"] != "PASS" and args.limit is None:
        raise RuntimeError(failures[["name", "error"]].to_dict("records"))


if __name__ == "__main__":
    main()
