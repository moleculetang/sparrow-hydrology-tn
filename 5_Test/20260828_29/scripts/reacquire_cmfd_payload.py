"""Reacquire and deeply validate one CMFD payload without overwriting raw data."""

from __future__ import annotations

import argparse
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
RUN = ROOT / "5_Test" / "20260828_29"
REGISTRY = ROOT / "5_Test" / "20260828_27" / "outputs" / "cmfd_1961_2024_acquisition_registry.parquet"
WORK = RUN / "work" / "cmfd_reacquired"
REPORTS = RUN / "reports"
CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def deep_audit(path: Path, variable: str) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        dataset = handle[variable]
        finite = 0
        total = 0
        # Read one time slice at a time so corrupted compressed chunks are
        # detected without allocating the entire field in memory.
        for index in range(dataset.shape[0]):
            values = np.asarray(dataset[index], dtype=np.float32)
            finite += int(np.isfinite(values).sum())
            total += int(values.size)
        return {
            "shape": [int(v) for v in dataset.shape],
            "dtype": str(dataset.dtype),
            "compression": str(dataset.compression),
            "finite_fraction": finite / total,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--month", required=True, type=int)
    parser.add_argument("--allow-official-size-amendment", action="store_true")
    args = parser.parse_args()
    registry = pd.read_parquet(REGISTRY)
    rows = registry.loc[
        registry.variable.eq(args.variable)
        & registry.year.eq(args.year)
        & registry.month.eq(args.month)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"registry match count is {len(rows)}")
    row = rows.iloc[0]
    expected = int(row.declared_size_bytes)
    target = WORK / args.variable / str(row["name"])
    target.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".nc.part")
    status = "verified_existing"
    if not target.is_file() or target.stat().st_size != expected:
        status = "downloaded"
        for attempt in range(1, 6):
            try:
                if part.exists():
                    part.unlink()
                with requests.post(
                    str(row.download_url),
                    json={"noToken": True},
                    headers={"User-Agent": "SPARROW-CMFD-deep-reacquisition/1.0"},
                    stream=True,
                    timeout=(30, 300),
                ) as response:
                    response.raise_for_status()
                    declared = int(response.headers.get("Content-Length", "-1"))
                    if declared != expected and not args.allow_official_size_amendment:
                        raise RuntimeError(f"HTTP size {declared} != registry size {expected}")
                    if declared <= 0:
                        raise RuntimeError(f"Invalid HTTP content length: {declared}")
                    received = 0
                    with part.open("wb") as stream:
                        for block in response.iter_content(CHUNK):
                            if block:
                                stream.write(block)
                                received += len(block)
                    if received != declared:
                        raise RuntimeError(f"received {received} != HTTP-declared {declared}")
                os.replace(part, target)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2**attempt)
    audit = deep_audit(target, args.variable)
    result = {
        "status": "PASS_DEEP_AUDIT",
        "acquisition": status,
        "variable": args.variable,
        "year": args.year,
        "month": args.month,
        "official_file_id": str(row.file_id),
        "source_path": str(row.local_path),
        "candidate_path": str(target),
        "bytes": target.stat().st_size,
        "registry_declared_size_bytes": expected,
        "official_current_size_bytes": target.stat().st_size,
        "official_size_amendment_used": bool(target.stat().st_size != expected),
        "sha256": sha256(target),
        "deep_audit": audit,
        "promotion_performed": False,
    }
    report = REPORTS / f"cmfd_reacquire_{args.variable}_{args.year}{args.month:02d}.json"
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
