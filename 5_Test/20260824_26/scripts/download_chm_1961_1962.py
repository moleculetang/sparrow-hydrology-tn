"""Download the two missing CHM_PRE V2 daily files from the public TPDC endpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h5py
import requests


ROOT = Path(r"E:\SPARROW")
RAW = ROOT / "0_reach_topology/data/raw/atmosphere/precipitation/chm_pre_v2/daily"
REPORT = ROOT / "5_Test/20260824_26/reports/chm_download_audit.json"
ENDPOINT = "https://data.tpdc.ac.cn/file/file/batchDownloadByFileId?fileId={file_id}"
FILES = [
    {"year": 1961, "file_id": "e51b3a83-2b68-4869-a5e1-7ee4b2598196", "name": "CHM_PRE_V2_daily_1961.nc", "size": 672_795_741},
    {"year": 1962, "file_id": "9579c616-ebf0-43ff-8a24-6944fa7378f9", "name": "CHM_PRE_V2_daily_1962.nc", "size": 672_795_741},
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate(path: Path, expected_size: int, year: int) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != expected_size:
        raise RuntimeError(f"size mismatch for {path.name}: {size} != {expected_size}")
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89HDF\r\n\x1a\n":
            raise RuntimeError(f"not an HDF5 file: {path}")
    with h5py.File(path, "r") as handle:
        required = {"prec", "time", "lat", "lon"}
        if not required.issubset(handle.keys()):
            raise RuntimeError(f"missing HDF variables in {path}")
        days = int(handle["time"].shape[0])
        if days not in {365, 366}:
            raise RuntimeError(f"unexpected day count: {days}")
        if int(handle["prec"].shape[0]) != days:
            raise RuntimeError("prec/time dimension mismatch")
    return {"year": year, "path": str(path), "size_bytes": size, "days": days, "sha256": digest(path), "status": "validated"}


def download_one(item: dict[str, object]) -> dict[str, object]:
    target = RAW / str(item["name"])
    partial = target.with_name(target.name + ".part")
    expected = int(item["size"])
    if target.exists():
        return validate(target, expected, int(item["year"])) | {"downloaded_this_run": False}
    if partial.exists():
        # The TPDC endpoint does not advertise byte ranges. A partial file
        # cannot be safely resumed, so only this exact registered .part path is removed.
        partial.unlink()
    url = ENDPOINT.format(file_id=item["file_id"])
    start = time.perf_counter()
    received = 0
    next_report = 64 * 1024 * 1024
    with requests.post(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        remote_size = int(response.headers.get("Content-Length", "0"))
        if remote_size != expected:
            raise RuntimeError(f"remote size changed for {item['name']}: {remote_size} != {expected}")
        with partial.open("wb") as stream:
            for block in response.iter_content(4 * 1024 * 1024):
                if not block:
                    continue
                stream.write(block)
                received += len(block)
                if received >= next_report:
                    elapsed = time.perf_counter() - start
                    print(json.dumps({"file": item["name"], "received_mib": round(received / 2**20, 1), "percent": round(100 * received / expected, 1), "mib_s": round(received / 2**20 / elapsed, 3)}), flush=True)
                    next_report += 64 * 1024 * 1024
            stream.flush()
            os.fsync(stream.fileno())
    if received != expected:
        raise RuntimeError(f"incomplete download for {item['name']}: {received} != {expected}")
    os.replace(partial, target)
    elapsed = time.perf_counter() - start
    result = validate(target, expected, int(item["year"]))
    result.update({"downloaded_this_run": True, "elapsed_seconds": elapsed, "mean_mib_s": received / 2**20 / elapsed, "source_url": url})
    return result


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")
    RAW.mkdir(parents=True, exist_ok=True)
    rows = [download_one(item) for item in FILES]
    payload = {"status": "PASS_CHM_1961_1962_DOWNLOADED_AND_VALIDATED", "source": "TPDC public file endpoint", "zenodo_record": "10.5281/zenodo.14632156", "files": rows}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temp = REPORT.with_name(REPORT.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, REPORT)
    print(json.dumps({"status": payload["status"], "files": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
