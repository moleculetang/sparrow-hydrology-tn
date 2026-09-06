"""Download hourly ERA5-Land fields needed for FAO-56 PET.

Requests are split by calendar month for resumability.  The CDS credential is
read only by cdsapi from the user-level .cdsapirc and is never written here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time
from calendar import monthrange
from pathlib import Path
from typing import Any

import h5py
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_28"
VENDOR = ROOT / "0_reach_topology" / "work" / "python_packages"
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "meteorology" / "era5_land" / "hourly_pet_prb"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
REQUESTS = RUN / "work" / "requests"
DATASET = "reanalysis-era5-land"
VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
)
EXPECTED_NATIVE_FIELDS = {
    "t2m": "K",
    "d2m": "K",
    "sp": "Pa",
    "u10": "m s**-1",
    "v10": "m s**-1",
    "ssrd": "J m**-2",
    "strd": "J m**-2",
}
TIMES = tuple(f"{hour:02d}:00" for hour in range(24))
AREA = (28.0, 101.0, 20.5, 117.0)
PRINT_LOCK = threading.Lock()
TRANSIENT_ERROR_MARKERS = (
    "number queued requests for this dataset is temporarily limited",
    "ssleoferror",
    "unexpected_eof_while_reading",
    "read timed out",
    "incompleteread",
    "connection broken",
    "connection reset",
    "max retries exceeded",
)


def load_cdsapi():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import cdsapi  # type: ignore

    return cdsapi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request(year: int, month: int) -> dict[str, Any]:
    return {
        "variable": list(VARIABLES),
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{day:02d}" for day in range(1, monthrange(year, month)[1] + 1)],
        "time": list(TIMES),
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": list(AREA),
    }


def validate(path: Path, year: int, month: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 8:
        raise RuntimeError(f"missing or empty ERA5 payload: {path}")
    with path.open("rb") as stream:
        signature = stream.read(8)
    if signature != b"\x89HDF\r\n\x1a\n":
        raise RuntimeError(f"ERA5 payload is not NetCDF4/HDF5: {path}")
    with h5py.File(path, "r") as handle:
        time_names = [name for name in ("valid_time", "time") if name in handle]
        if not time_names:
            raise RuntimeError(f"ERA5 payload has no time coordinate: {path}")
        steps = int(handle[time_names[0]].shape[0])
        expected = monthrange(year, month)[1] * 24
        if steps != expected:
            raise RuntimeError(f"ERA5 {year}-{month:02d} expected {expected} hours, found {steps}")
        data_names = set(handle.keys()) - {"latitude", "longitude", "valid_time", "time", "number", "expver"}
        expected_names = set(EXPECTED_NATIVE_FIELDS)
        if data_names != expected_names:
            raise RuntimeError(
                f"ERA5 payload variables differ from the registered contract; "
                f"expected={sorted(expected_names)} found={sorted(data_names)}: {path}"
            )
        if handle["latitude"].shape != (76,) or handle["longitude"].shape != (161,):
            raise RuntimeError(
                f"ERA5 payload grid differs from registered PRB domain; "
                f"latitude={handle['latitude'].shape} longitude={handle['longitude'].shape}: {path}"
            )
        for name, expected_units in EXPECTED_NATIVE_FIELDS.items():
            variable = handle[name]
            if variable.shape != (expected, 76, 161):
                raise RuntimeError(
                    f"ERA5 {name} expected shape {(expected, 76, 161)}, found {variable.shape}: {path}"
                )
            units = variable.attrs.get("units", "")
            if isinstance(units, bytes):
                units = units.decode("utf-8")
            if str(units) != expected_units:
                raise RuntimeError(
                    f"ERA5 {name} expected units {expected_units!r}, found {str(units)!r}: {path}"
                )
    return {"bytes": path.stat().st_size, "time_steps": steps, "sha256": sha256(path)}


def is_transient_failure(exc: Exception) -> bool:
    message = repr(exc).lower()
    return any(marker in message for marker in TRANSIENT_ERROR_MARKERS)


def download_one(
    task: tuple[int, int],
    force: bool = False,
    outer_retries: int = 12,
    retry_base_seconds: int = 120,
) -> dict[str, Any]:
    year, month = task
    target = RAW / f"era5_land_hourly_pet_{year}{month:02d}.nc"
    part = target.with_suffix(".nc.part")
    request_path = REQUESTS / f"era5_land_hourly_pet_{year}{month:02d}.json"
    payload = request(year, month)
    request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if target.is_file() and not force:
        return {"year": year, "month": month, "status": "verified_existing", "path": str(target), **validate(target, year, month)}
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, outer_retries + 1):
        if part.exists():
            part.unlink()
        cdsapi = load_cdsapi()
        # Keep the CDS client's inner retry loop bounded.  The outer loop
        # already provides explicit, recorded backoff; a retry_max of 100 can
        # otherwise leave both workers parked for hours during a TLS outage.
        client = cdsapi.Client(quiet=True, progress=False, retry_max=5, sleep_max=30)
        try:
            client.retrieve(DATASET, payload, str(part))
            details = validate(part, year, month)
            os.replace(part, target)
            elapsed = max(time.perf_counter() - started, 1e-9)
            with PRINT_LOCK:
                print(
                    f"ERA5 downloaded {year}-{month:02d} {details['bytes'] / 1e6:.1f} MB "
                    f"{details['bytes'] / elapsed / 1e6:.1f} MB/s attempts={attempt}",
                    flush=True,
                )
            return {
                "year": year,
                "month": month,
                "status": "downloaded",
                "path": str(target),
                "attempts": attempt,
                "elapsed_seconds": elapsed,
                "effective_MB_s": details["bytes"] / elapsed / 1e6,
                **details,
            }
        except Exception as exc:
            last_error = exc
            transient = is_transient_failure(exc)
            if transient and attempt < outer_retries:
                delay = min(retry_base_seconds * (2 ** min(attempt - 1, 3)), 900)
                with PRINT_LOCK:
                    print(
                        f"ERA5 transient failure {year}-{month:02d} attempt={attempt}/{outer_retries}; "
                        f"retrying in {delay}s: {type(exc).__name__}",
                        flush=True,
                    )
                time.sleep(delay)
                continue
            break
    return {
        "year": year,
        "month": month,
        "status": "failed",
        "path": str(target),
        "attempts": attempt,
        "part_bytes": part.stat().st_size if part.exists() else 0,
        "error": repr(last_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--priority-year",
        type=int,
        default=None,
        help="Download this year first while retaining one complete final registry.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outer-retries", type=int, default=12)
    parser.add_argument("--retry-base-seconds", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.start_year > args.end_year
        or not 1 <= args.workers <= 4
        or args.outer_retries < 1
        or args.retry_base_seconds < 1
        or (
            args.priority_year is not None
            and not args.start_year <= args.priority_year <= args.end_year
        )
    ):
        raise RuntimeError("invalid year range or workers; CDS concurrency is capped at four")
    for directory in (RAW, OUT, REPORTS, REQUESTS):
        directory.mkdir(parents=True, exist_ok=True)
    tasks = [(year, month) for year in range(args.start_year, args.end_year + 1) for month in range(1, 13)]
    if args.priority_year is not None:
        tasks = [task for task in tasks if task[0] == args.priority_year] + [
            task for task in tasks if task[0] != args.priority_year
        ]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    planned = pd.DataFrame([{"year": y, "month": m, "target": str(RAW / f"era5_land_hourly_pet_{y}{m:02d}.nc")} for y, m in tasks])
    planned.to_parquet(OUT / "era5_land_hourly_pet_registry_planned.parquet", index=False)
    if args.dry_run:
        for year, month in tasks:
            (REQUESTS / f"era5_land_hourly_pet_{year}{month:02d}.json").write_text(json.dumps(request(year, month), indent=2), encoding="utf-8")
        print(json.dumps({"status": "DRY_RUN", "requests": len(tasks), "variables": list(VARIABLES), "area": AREA}, indent=2))
        return
    completed: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                task,
                args.force,
                args.outer_retries,
                args.retry_base_seconds,
            ): task
            for task in tasks
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            completed.append(future.result())
            if index % 4 == 0 or index == len(futures):
                pd.DataFrame(completed).sort_values(["year", "month"]).to_parquet(OUT / "era5_land_hourly_pet_acquisition_progress.parquet", index=False)
                print(f"ERA5 progress {index}/{len(futures)}", flush=True)
    result = pd.DataFrame(completed).sort_values(["year", "month"]).reset_index(drop=True)
    result.to_parquet(OUT / "era5_land_hourly_pet_acquisition_registry.parquet", index=False)
    failures = result[result.status.eq("failed")]
    report = {
        "status": "PASS" if len(result) == len(tasks) and failures.empty else "INCOMPLETE",
        "dataset": DATASET,
        "period": f"{args.start_year}-01 through {args.end_year}-12",
        "requests": len(tasks),
        "verified": int(result.status.ne("failed").sum()),
        "failed": len(failures),
        "verified_bytes": int(result.loc[result.status.ne("failed"), "bytes"].fillna(0).sum()),
        "variables": list(VARIABLES),
        "precipitation_downloaded": False,
        "credential_serialized": False,
    }
    (REPORTS / "era5_land_hourly_pet_acquisition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS" and args.limit is None:
        raise RuntimeError(failures[["year", "month", "error"]].to_dict("records"))


if __name__ == "__main__":
    main()
