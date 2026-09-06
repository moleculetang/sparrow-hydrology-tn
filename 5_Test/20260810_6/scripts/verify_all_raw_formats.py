from __future__ import annotations

import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

import h5py
import pandas as pd


RAW = Path(r"E:\SPARROW\0_reach_topology\data\raw")
GDALINFO = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin\gdalinfo.exe")
NCDUMP = Path(r"D:\ProgramData\anaconda3\envs\PRB_reach\Library\bin\ncdump.exe")


def run_header(command: list[str]) -> tuple[bool, str]:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode == 0, (result.stderr or result.stdout)[-500:]


def verify_raster(path: Path) -> tuple[Path, bool, str]:
    ok, detail = run_header([str(GDALINFO), "-nomd", "-norat", "-noct", str(path)])
    return path, ok, detail


def verify_netcdf(path: Path) -> tuple[Path, bool, str]:
    ok, detail = run_header([str(NCDUMP), "-h", str(path)])
    return path, ok, detail


def verify_zip(path: Path) -> tuple[Path, bool, str]:
    try:
        with ZipFile(path) as archive:
            bad = archive.testzip()
        return path, bad is None, "" if bad is None else f"CRC failure: {bad}"
    except Exception as exc:
        return path, False, repr(exc)


def verify_rar(path: Path) -> tuple[Path, bool, str]:
    ok, detail = run_header(["tar", "-tf", str(path)])
    return path, ok, detail


def verify_json(path: Path) -> tuple[Path, bool, str]:
    try:
        json.loads(path.read_text(encoding="utf-8-sig"))
        return path, True, ""
    except Exception as exc:
        return path, False, repr(exc)


def verify_csv(path: Path) -> tuple[Path, bool, str]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
        return path, True, ""
    except Exception:
        try:
            with path.open("r", encoding="gb18030", errors="strict", newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
            return path, True, "gb18030"
        except Exception as exc:
            return path, False, repr(exc)


def verify_hdf5(path: Path) -> tuple[Path, bool, str]:
    try:
        with h5py.File(path, "r") as handle:
            _ = list(handle.keys())
        return path, True, ""
    except Exception as exc:
        return path, False, repr(exc)


def verify_shapefiles() -> dict[str, object]:
    shp_paths = sorted(RAW.rglob("*.shp"))
    missing = []
    for shp in shp_paths:
        for extension in (".shx", ".dbf", ".prj"):
            sidecar = shp.with_suffix(extension)
            if not sidecar.exists():
                missing.append(str(sidecar))
    return {"files": len(shp_paths), "missing_required_sidecars": missing}


def run_family(name: str, paths: list[Path], verifier, workers: int) -> dict[str, object]:
    failures = []
    details = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verifier, path) for path in paths]
        for future in as_completed(futures):
            path, ok, detail = future.result()
            if not ok:
                failures.append({"file": str(path), "detail": detail})
            elif detail:
                details.append({"file": str(path), "detail": detail})
    return {"name": name, "files": len(paths), "readable": len(paths) - len(failures), "failures": failures, "notes": details}


def main() -> None:
    families = [
        run_family("rasters", sorted([*RAW.rglob("*.tif"), *RAW.rglob("*.asc")]), verify_raster, 12),
        run_family("netcdf", sorted(RAW.rglob("*.nc")), verify_netcdf, 8),
        run_family("hdf5", sorted(RAW.rglob("*.h5")), verify_hdf5, 4),
        run_family("zip", sorted(RAW.rglob("*.zip")), verify_zip, 4),
        run_family("rar", sorted(RAW.rglob("*.rar")), verify_rar, 4),
        run_family("json", sorted(RAW.rglob("*.json")), verify_json, 4),
        run_family("csv", sorted(RAW.rglob("*.csv")), verify_csv, 4),
    ]
    result = {
        "data_file_count": len([path for path in RAW.rglob("*") if path.is_file() and path.name != "README.md"]),
        "data_byte_count": sum(path.stat().st_size for path in RAW.rglob("*") if path.is_file() and path.name != "README.md"),
        "families": families,
        "shapefiles": verify_shapefiles(),
    }
    print(json.dumps(result, ensure_ascii=False))
    failed = (
        result["data_file_count"] != 6288
        or result["data_byte_count"] != 327593638667
        or any(family["failures"] for family in families)
        or bool(result["shapefiles"]["missing_required_sidecars"])
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
