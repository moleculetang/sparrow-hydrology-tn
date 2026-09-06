from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_API_URL = "https://cds.climate.copernicus.eu/api"

DATASET = "reanalysis-era5-land-monthly-means"
PRODUCT_TYPE = "monthly_averaged_reanalysis"
DEFAULT_FILE_PREFIX = "era5_land_monthly_prb_buffer"
VARIABLES = [
    "total_precipitation",
    "total_evaporation",
    "2m_temperature",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
]
MONTHS = [f"{month:02d}" for month in range(1, 13)]
TIME = "00:00"

# CDS area order is [north, west, south, east]. The processed PRB boundary is
# about [102.25E, 21.59N, 115.88E, 26.87N]; this buffered box is rounded outward.
AREA = [28.0, 101.0, 20.5, 117.0]


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_project_vendor_path(root: Path) -> None:
    vendor = root / "work" / "python_packages"
    if vendor.exists():
        sys.path.insert(0, str(vendor))


def _load_cdsapi(root: Path):
    _add_project_vendor_path(root)
    try:
        import cdsapi  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python package 'cdsapi'. Install it in this environment or run:\n"
            f"  python -m pip install --target \"{root / 'work' / 'python_packages'}\" \"cdsapi>=0.7.7\""
        ) from exc
    return cdsapi


def _request_for_year(year: int, variables: list[str]) -> dict[str, Any]:
    return {
        "product_type": [PRODUCT_TYPE],
        "variable": variables,
        "year": [str(year)],
        "month": MONTHS,
        "time": [TIME],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }


def _write_request(path: Path, request: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "year",
        "dataset",
        "file",
        "request_json",
        "status",
        "bytes",
        "updated_utc",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _merge_manifest_rows(existing: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_year: dict[int, dict[str, Any]] = {}
    for row in existing:
        try:
            rows_by_year[int(row["year"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    for row in updates:
        rows_by_year[int(row["year"])] = row
    return [rows_by_year[year] for year in sorted(rows_by_year)]


def download(
    output_dir: Path,
    start_year: int,
    end_year: int,
    force: bool,
    dry_run: bool,
    api_url: str | None,
    api_key: str | None,
    variables: list[str],
    file_prefix: str,
) -> None:
    root = _root()
    output_dir = output_dir.resolve()
    request_dir = output_dir / "requests"
    manifest_path = output_dir / "manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    request_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not dry_run:
        cdsapi = _load_cdsapi(root)
        client_kwargs: dict[str, Any] = {
            "quiet": False,
            "progress": True,
            "retry_max": 500,
            "sleep_max": 120,
        }
        if api_url:
            client_kwargs["url"] = api_url
        if api_key:
            client_kwargs["key"] = api_key
        client = cdsapi.Client(**client_kwargs)

    existing_rows = _read_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        target = output_dir / f"{file_prefix}_{year}.nc"
        request_path = request_dir / f"{file_prefix}_{year}_request.json"
        request = _request_for_year(year, variables)
        _write_request(request_path, request)

        status = "dry_run"
        if target.exists() and target.stat().st_size > 0 and not force:
            status = "exists"
            print(f"[skip] {target}")
        elif dry_run:
            print(f"[dry-run] {DATASET} {year} -> {target}")
        else:
            assert client is not None
            print(f"[download] {DATASET} {year} -> {target}")
            client.retrieve(DATASET, request, str(target))
            status = "downloaded"

        rows.append(
            {
                "year": year,
                "dataset": DATASET,
                "file": str(target.relative_to(root)),
                "request_json": str(request_path.relative_to(root)),
                "status": status,
                "bytes": target.stat().st_size if target.exists() else 0,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_manifest(manifest_path, _merge_manifest_rows(existing_rows, rows))


def main() -> int:
    root = _root()
    parser = argparse.ArgumentParser(description="Download buffered PRB ERA5-Land monthly NetCDF files from CDS.")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "raw" / "atmosphere" / "meteorology" / "era5_land" / "data" / "monthly_prb_buffer",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=VARIABLES,
        help="CDS variable names; defaults to the existing seven-variable climate request.",
    )
    parser.add_argument(
        "--file-prefix",
        default=DEFAULT_FILE_PREFIX,
        help="Prefix for yearly NetCDF and request-record filenames.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing yearly NetCDF files.")
    parser.add_argument("--dry-run", action="store_true", help="Write request JSON files without contacting CDS.")
    parser.add_argument("--api-url", default=os.environ.get("CDSAPI_URL") or DEFAULT_API_URL)
    parser.add_argument("--api-key", default=os.environ.get("CDSAPI_KEY"), help="CDS API key. Defaults to CDSAPI_KEY or .cdsapirc.")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    download(
        args.output_dir,
        args.start_year,
        args.end_year,
        args.force,
        args.dry_run,
        args.api_url,
        args.api_key,
        args.variables,
        args.file_prefix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
