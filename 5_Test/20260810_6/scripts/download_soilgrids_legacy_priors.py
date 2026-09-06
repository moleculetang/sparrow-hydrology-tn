from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from common import RUN, write_json
from runtime_guard import assert_sparrow_runtime


SOURCE_BDOD = Path(r"E:\SPARROW\5_Test\20260729_38\inputs\soilgrids_prb_buffer\bdod")
DOWNLOAD_ROOT = Path(r"E:\SPARROW\0_reach_topology\data\raw\soil\soilgrids_v2\prb_buffer_legacy_n")
DOWNLOAD_DATA = DOWNLOAD_ROOT / "data"
DOWNLOAD_METADATA = DOWNLOAD_ROOT / "metadata"
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"]
EXPECTED_BOUNDS = [9657439.713561734, 2625614.325485781, 10997044.662708629, 3294858.131649507]
WCS = "https://maps.isric.org/mapserv"
WCS_CRS = "http://www.opengis.net/def/crs/EPSG/0/152160"
LOCAL_CRS = "ESRI:54009"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gdal(name: str) -> Path:
    path = Path(sys.prefix) / "Library" / "bin" / f"{name}.exe"
    if not path.is_file():
        raise FileNotFoundError(f"Missing GDAL command in sparrow environment: {path}")
    return path


def _info(path: Path) -> dict:
    # CRS and raster metadata are sufficient for cache validation.  Requesting
    # statistics rescans every pixel and makes an otherwise small WCS download
    # gate unnecessarily slow.
    result = subprocess.run([str(_gdal("gdalinfo")), "-json", str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _attach_local_crs(path: Path) -> None:
    temporary = path.with_suffix(".crs.tif")
    subprocess.run(
        [str(_gdal("gdal_translate")), "-q", "-a_srs", LOCAL_CRS, "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", str(path), str(temporary)],
        check=True,
        capture_output=True,
        text=True,
    )
    temporary.replace(path)


def _download(session: requests.Session, property_name: str, depth: str, destination: Path) -> str:
    if destination.is_file() and destination.stat().st_size > 1024:
        try:
            existing = _info(destination)
            if existing.get("coordinateSystem", {}).get("wkt"):
                return "existing"
        except Exception:
            pass
        # A prior interrupted run can leave a corrupt or unreferenced TIFF.
        # It is not a valid cache entry and must be replaced from the official WCS.
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    minx, miny, maxx, maxy = EXPECTED_BOUNDS
    params = [
        ("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
        ("COVERAGEID", f"{property_name}_{depth}_mean"), ("FORMAT", "image/tiff"),
        ("SUBSET", f"x({minx},{maxx})"), ("SUBSET", f"y({miny},{maxy})"),
        ("SUBSETTINGCRS", WCS_CRS), ("OUTPUTCRS", WCS_CRS),
    ]
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = session.get(f"{WCS}?map=/map/{property_name}.map", params=params, timeout=(30, 600), stream=True)
            response.raise_for_status()
            if "tiff" not in response.headers.get("content-type", "").lower():
                raise RuntimeError(f"Unexpected content type: {response.headers.get('content-type')}")
            received = 0
            with temporary.open("wb") as handle:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        handle.write(block)
                        received += len(block)
            if received < 1024:
                raise RuntimeError(f"Downloaded only {received} bytes")
            temporary.replace(destination)
            _attach_local_crs(destination)
            return "downloaded"
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed SoilGrids download {property_name}_{depth}_mean: {last_error}")


def main() -> None:
    runtime = assert_sparrow_runtime()
    root = DOWNLOAD_DATA
    if not SOURCE_BDOD.is_dir():
        raise FileNotFoundError(f"Verified read-only bulk-density source is missing: {SOURCE_BDOD}")
    records: list[dict[str, object]] = []
    for depth in DEPTHS:
        source = SOURCE_BDOD / f"bdod_{depth}_mean_prb_buffer.tif"
        if not source.is_file():
            raise FileNotFoundError(f"Missing verified bulk-density layer: {source}")
        destination = root / "bdod" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or _sha256(destination) != _sha256(source):
            shutil.copy2(source, destination)
        records.append({"property": "bdod", "depth": depth, "status": "copied_verified_local", "path": str(destination), "sha256": _sha256(destination)})
    session = requests.Session()
    session.headers.update({"User-Agent": "SPARROW-PRB-LegacyN/20260810_6 (scientific research)"})
    for property_name in ("nitrogen", "soc"):
        for depth in DEPTHS:
            destination = root / property_name / f"{property_name}_{depth}_mean_prb_buffer.tif"
            status = _download(session, property_name, depth, destination)
            info = _info(destination)
            if not info.get("coordinateSystem", {}).get("wkt"):
                raise ValueError(f"No CRS recorded in {destination}")
            records.append({"property": property_name, "depth": depth, "status": status, "path": str(destination), "sha256": _sha256(destination)})
    table = pd.DataFrame.from_records(records).sort_values(["property", "depth"])
    DOWNLOAD_METADATA.mkdir(parents=True, exist_ok=True)
    table.to_csv(DOWNLOAD_METADATA / "soilgrids_legacy_prior_manifest.csv", index=False, encoding="utf-8-sig")
    expected = 3 * len(DEPTHS)
    summary = {
        "runtime": runtime,
        "source": "SoilGrids 2.0 WCS; bdod copied from verified PRB buffer acquisition",
        "depths_cm": DEPTHS,
        "layers": int(len(table)),
        "expected_layers": expected,
        "all_layers_available": bool(len(table) == expected),
        "note": "Raw SON-prior inputs only; no SON calibration or TN fit is performed by this script.",
    }
    write_json(RUN / "reports" / "soilgrids_legacy_prior_gate.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
