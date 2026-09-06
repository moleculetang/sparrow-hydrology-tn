"""Reacquire and safely promote the single corrupt ERA5-Land 2023-08 payload.

The accepted source is replaced only after the downloaded candidate passes the
registered structural checks and a full variable-by-variable, hourly-slice
read.  The rejected source is retained in the raw-data archive.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import h5py


ROOT = Path(r"E:\SPARROW")
S28 = ROOT / "5_Test" / "20260828_28"
S30 = ROOT / "5_Test" / "20260828_30"
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "meteorology" / "era5_land" / "hourly_pet_prb"
TARGET = RAW / "era5_land_hourly_pet_202308.nc"
WORK = S30 / "work" / "era5_202308_reacquisition"
CANDIDATE = WORK / "era5_land_hourly_pet_202308.candidate.nc"
ARCHIVE = RAW.parent / "archives" / "corrupt_payloads"
REPORT = S30 / "reports" / "era5_202308_reacquisition.json"
DOWNLOAD_MODULE = S28 / "scripts" / "download_era5_land_hourly_pet.py"
YEAR, MONTH = 2023, 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_download_module():
    spec = importlib.util.spec_from_file_location("era5_download", DOWNLOAD_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {DOWNLOAD_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_deep_read(path: Path, expected_fields: dict[str, str]) -> dict[str, object]:
    slices = 0
    finite_counts: dict[str, int] = {}
    with h5py.File(path, "r") as handle:
        for name in expected_fields:
            variable = handle[name]
            finite = 0
            for step in range(variable.shape[0]):
                values = variable[step, :, :]
                # HDF5 filter errors are raised by the preceding read.  Count
                # finite cells without requiring land-mask NaNs to disappear.
                finite += int((values == values).sum())
                slices += 1
            finite_counts[name] = finite
    return {"hourly_variable_slices_read": slices, "finite_cell_counts": finite_counts}


def download_candidate(module) -> dict[str, object]:
    WORK.mkdir(parents=True, exist_ok=True)
    part = CANDIDATE.with_suffix(".nc.part")
    for path in (CANDIDATE, part):
        if path.exists():
            path.unlink()
    payload = module.request(YEAR, MONTH)
    last_error: Exception | None = None
    started = time.perf_counter()
    for attempt in range(1, 7):
        if part.exists():
            part.unlink()
        try:
            client = module.load_cdsapi().Client(quiet=True, progress=False, retry_max=5, sleep_max=30)
            client.retrieve(module.DATASET, payload, str(part))
            structural = module.validate(part, YEAR, MONTH)
            deep = full_deep_read(part, module.EXPECTED_NATIVE_FIELDS)
            os.replace(part, CANDIDATE)
            return {
                "attempts": attempt,
                "elapsed_seconds": time.perf_counter() - started,
                "request": payload,
                "structural": structural,
                "deep_read": deep,
            }
        except Exception as exc:  # recorded verbatim in the repair report
            last_error = exc
            if attempt == 6:
                break
            time.sleep(min(30 * 2 ** (attempt - 1), 300))
    raise RuntimeError(f"ERA5 2023-08 reacquisition failed: {last_error!r}")


def main() -> None:
    if not TARGET.is_file():
        raise RuntimeError(f"Rejected source is missing: {TARGET}")
    old = {"path": str(TARGET), "bytes": TARGET.stat().st_size, "sha256": sha256(TARGET)}
    module = load_download_module()
    download = download_candidate(module)
    # Verify the promoted bytes one more time before modifying the raw source.
    candidate = {
        "path": str(CANDIDATE),
        "bytes": CANDIDATE.stat().st_size,
        "sha256": sha256(CANDIDATE),
        "structural": module.validate(CANDIDATE, YEAR, MONTH),
        "deep_read": full_deep_read(CANDIDATE, module.EXPECTED_NATIVE_FIELDS),
    }
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    archived = ARCHIVE / f"era5_land_hourly_pet_202308_corrupt_{old['sha256'][:12]}.nc"
    if archived.exists():
        if sha256(archived) != old["sha256"]:
            raise RuntimeError(f"Archive collision with different bytes: {archived}")
        TARGET.unlink()
    else:
        os.replace(TARGET, archived)
    os.replace(CANDIDATE, TARGET)
    promoted = {
        "path": str(TARGET),
        "bytes": TARGET.stat().st_size,
        "sha256": sha256(TARGET),
        "structural": module.validate(TARGET, YEAR, MONTH),
        "deep_read": full_deep_read(TARGET, module.EXPECTED_NATIVE_FIELDS),
    }
    checks = {
        "old_archived": archived.is_file() and sha256(archived) == old["sha256"],
        "candidate_promoted_byte_exact": promoted["sha256"] == candidate["sha256"],
        "full_deep_read_passed": promoted["deep_read"]["hourly_variable_slices_read"] == 31 * 24 * 7,
        "old_and_new_hash_differ": old["sha256"] != promoted["sha256"],
    }
    report = {
        "stage": "20260828_30",
        "status": "PASS_ERA5_202308_REACQUISITION" if all(checks.values()) else "FAIL",
        "year": YEAR,
        "month": MONTH,
        "rejected": old,
        "archived_corrupt_path": str(archived),
        "download": download,
        "candidate": candidate,
        "promoted": promoted,
        "checks": checks,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(REPORT), "checks": checks}, ensure_ascii=False, indent=2))
    if report["status"] != "PASS_ERA5_202308_REACQUISITION":
        raise RuntimeError("ERA5 2023-08 promotion did not satisfy all checks")


if __name__ == "__main__":
    main()
