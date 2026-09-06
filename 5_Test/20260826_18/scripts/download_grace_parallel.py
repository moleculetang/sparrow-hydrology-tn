"""Safely accelerate the remaining GRACE download with verified byte ranges."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

import requests
import urllib3


URL = "https://download.csr.utexas.edu/outgoing/grace/RL06_mascons/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc"
TARGET = Path(r"E:\SPARROW\0_reach_topology\data\raw\hydrology\terrestrial_water_storage\csr_grace_rl06_mascons_v02\CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc")
WORK = Path(r"E:\SPARROW\5_Test\20260826_18\work\grace_segments")
REPORT = Path(r"E:\SPARROW\5_Test\20260826_18\reports\grace_download_audit.json")
PROXIES = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
EXPECTED_BYTES = 920_716_039
N_SEGMENTS = 4
print_lock = threading.Lock()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_segment(index: int, start: int, end: int) -> Path:
    path = WORK / f"segment_{index:02d}_{start}_{end}.part"
    expected = end - start + 1
    for attempt in range(1, 11):
        existing = path.stat().st_size if path.exists() else 0
        if existing == expected:
            return path
        if existing > expected:
            raise RuntimeError(f"Segment {index} exceeds registered length")
        request_start = start + existing
        try:
            with requests.get(
                URL,
                headers={"Range": f"bytes={request_start}-{end}"},
                proxies=PROXIES,
                verify=False,
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not content_range.startswith(f"bytes {request_start}-{end}/"):
                    raise RuntimeError(f"Unexpected range response: status={response.status_code} content-range={content_range!r}")
                with path.open("ab") as stream:
                    next_report = existing + 64 * 1024 * 1024
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
                        existing += len(chunk)
                        if existing >= next_report:
                            with print_lock:
                                print(f"segment={index} bytes={existing}/{expected}", flush=True)
                            next_report += 64 * 1024 * 1024
            if path.stat().st_size == expected:
                return path
        except Exception as error:
            with print_lock:
                print(f"segment={index} attempt={attempt} resume={existing}/{expected} error={error}", flush=True)
            time.sleep(min(20, 2 * attempt))
    raise RuntimeError(f"Segment {index} incomplete after retries: {path.stat().st_size if path.exists() else 0}/{expected}")


def main() -> None:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    prefix_bytes = TARGET.stat().st_size if TARGET.exists() else 0
    if prefix_bytes == EXPECTED_BYTES:
        digest = sha256(TARGET)
        REPORT.write_text(json.dumps({
            "source_url": URL, "target": str(TARGET), "expected_bytes": EXPECTED_BYTES,
            "actual_bytes": prefix_bytes, "size_pass": True, "sha256": digest,
            "transport_note": "File was already complete before parallel resume."
        }, indent=2), encoding="utf-8")
        print(f"already complete sha256={digest}", flush=True)
        return
    if prefix_bytes <= 0 or prefix_bytes >= EXPECTED_BYTES:
        raise RuntimeError(f"Invalid preserved prefix length: {prefix_bytes}")
    remaining = EXPECTED_BYTES - prefix_bytes
    base, extra = divmod(remaining, N_SEGMENTS)
    ranges = []
    cursor = prefix_bytes
    for index in range(N_SEGMENTS):
        length = base + (1 if index < extra else 0)
        ranges.append((index, cursor, cursor + length - 1))
        cursor += length
    if cursor != EXPECTED_BYTES:
        raise RuntimeError("Registered ranges do not close to expected size")
    print(json.dumps({"preserved_prefix_bytes": prefix_bytes, "ranges": ranges}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_SEGMENTS) as pool:
        futures = [pool.submit(fetch_segment, *item) for item in ranges]
        segment_paths = [future.result() for future in futures]
    if TARGET.stat().st_size != prefix_bytes:
        raise RuntimeError("Preserved prefix changed during segmented download; refusing assembly")
    assembly = TARGET.with_name(TARGET.name + ".assembling")
    with assembly.open("wb") as output, TARGET.open("rb") as prefix:
        shutil.copyfileobj(prefix, output, length=8 * 1024 * 1024)
        for path in segment_paths:
            with path.open("rb") as segment:
                shutil.copyfileobj(segment, output, length=8 * 1024 * 1024)
    actual = assembly.stat().st_size
    if actual != EXPECTED_BYTES:
        raise RuntimeError(f"Assembly size mismatch: {actual}/{EXPECTED_BYTES}")
    digest = sha256(assembly)
    os.replace(assembly, TARGET)
    audit = {
        "source_url": URL,
        "target": str(TARGET),
        "expected_bytes": EXPECTED_BYTES,
        "actual_bytes": TARGET.stat().st_size,
        "size_pass": TARGET.stat().st_size == EXPECTED_BYTES,
        "sha256": digest,
        "preserved_prefix_bytes": prefix_bytes,
        "verified_ranges": ranges,
        "transport_note": "HTTPS through local proxy; every remaining range and final byte count were verified before atomic replacement. Source authority, exact size and SHA256 are recorded.",
    }
    REPORT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for path in segment_paths:
        path.unlink()
    try:
        WORK.rmdir()
    except OSError:
        pass
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
