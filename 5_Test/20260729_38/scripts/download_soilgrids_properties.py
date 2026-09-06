from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import pandas as pd
import requests


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
INPUTS = RUN / "inputs" / "soilgrids_prb_buffer"
REPORT = RUN / "reports" / "soilgrids_acquisition"
MANIFEST = RUN / "inputs_manifest"
LOGS = RUN / "logs"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_37" / "reports"
    / "cfvo_robustness" / "gate.json"
)
LOCAL_SPATIAL_MANIFEST = (
    ROOT / "0_reach_topology" / "data" / "processed"
    / "soil_prb" / "soil_storage_eff_manifest.json"
)
LOCAL_CATCHMENTS = (
    ROOT / "0_reach_topology" / "results"
    / "vectors" / "reach_catchments.shp"
)
CONTRACT = RUN / "experiment_contract.md"
LITERATURE = RUN / "literature_basis.md"
SOILGRIDS_WCS = "https://maps.isric.org/mapserv"
SOILGRIDS_CRS_URI = "http://www.opengis.net/def/crs/EPSG/0/152160"
SOILGRIDS_LOCAL_CRS = "ESRI:54009"
PROPERTIES = {
    "sand": {
        "raw_unit": "g/kg with d_factor=10",
        "scale_to_mass_percent": 0.1,
    },
    "clay": {
        "raw_unit": "g/kg with d_factor=10",
        "scale_to_mass_percent": 0.1,
    },
    "bdod": {
        "raw_unit": "cg/cm3 with d_factor=100",
        "scale_to_g_cm3": 0.01,
    },
}
DEPTHS = [
    "0-5cm", "5-15cm", "15-30cm",
    "30-60cm", "60-100cm", "100-200cm",
]
EXPECTED_BOUNDS = [
    9657439.713561734,
    2625614.325485781,
    10997044.662708629,
    3294858.131649507,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gdal_bin(name: str) -> Path:
    path = Path(sys.prefix) / "Library" / "bin" / f"{name}.exe"
    if not path.exists():
        raise RuntimeError(f"Missing {name} in sparrow environment")
    return path


def attach_crs(path: Path) -> None:
    temporary = path.with_suffix(".crs.tif")
    command = [
        str(gdal_bin("gdal_translate")), "-q",
        "-a_srs", SOILGRIDS_LOCAL_CRS,
        "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
        str(path), str(temporary),
    ]
    subprocess.run(
        command, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    os.replace(temporary, path)


def gdal_info(path: Path) -> dict:
    result = subprocess.run(
        [str(gdal_bin("gdalinfo")), "-json", "-stats", str(path)],
        check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, encoding="utf-8", errors="replace",
    )
    return json.loads(result.stdout)


def download_one(
    session: requests.Session, property_name: str,
    depth: str, destination: Path,
) -> tuple[str, str]:
    coverage = f"{property_name}_{depth}_mean"
    if destination.exists() and destination.stat().st_size > 1024:
        return "exists", coverage
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(".part")
    minx, miny, maxx, maxy = EXPECTED_BOUNDS
    params = [
        ("SERVICE", "WCS"),
        ("VERSION", "2.0.1"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", coverage),
        ("FORMAT", "image/tiff"),
        ("SUBSET", f"x({minx},{maxx})"),
        ("SUBSET", f"y({miny},{maxy})"),
        ("SUBSETTINGCRS", SOILGRIDS_CRS_URI),
        ("OUTPUTCRS", SOILGRIDS_CRS_URI),
    ]
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = session.get(
                f"{SOILGRIDS_WCS}?map=/map/{property_name}.map",
                params=params, timeout=(30, 600), stream=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "tiff" not in content_type.lower():
                raise RuntimeError(
                    f"Unexpected response: {content_type}"
                )
            received = 0
            next_progress = 10 * 1024 * 1024
            with part.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    received += len(chunk)
                    if received >= next_progress:
                        print(
                            f"[stream] {coverage} "
                            f"{received / 1024 / 1024:.1f} MiB",
                            flush=True,
                        )
                        next_progress += 10 * 1024 * 1024
            if received < 1024:
                raise RuntimeError(
                    f"Unexpected response size: {received} bytes"
                )
            os.replace(part, destination)
            attach_crs(destination)
            return "downloaded", coverage
        except Exception as error:
            last_error = error
            if part.exists():
                part.unlink()
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Download failed for {coverage}: {last_error}"
    )


def main() -> None:
    for directory in (INPUTS, REPORT, MANIFEST, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != (
        "ACQUIRE_MISSING_SAND_CLAY_BULK_DENSITY_AND_DRAINAGE_ATTRIBUTES"
    ):
        raise RuntimeError("Parent gate does not authorize acquisition")
    spatial = json.loads(
        LOCAL_SPATIAL_MANIFEST.read_text(encoding="utf-8")
    )
    if spatial["soilgrids_native_download_bounds"] != EXPECTED_BOUNDS:
        raise RuntimeError("Frozen local SoilGrids bounds changed")
    if Path(spatial["catchment_source"]).resolve() != (
        LOCAL_CATCHMENTS.resolve()
    ):
        raise RuntimeError("Unexpected spatial basis")
    session = requests.Session()
    session.headers.update({
        "User-Agent": "SPARROW-PRB-test-soil-acquisition/1.0"
    })
    rows = []
    total = len(PROPERTIES) * len(DEPTHS)
    completed = 0
    for property_name, metadata in PROPERTIES.items():
        for depth in DEPTHS:
            destination = (
                INPUTS / property_name
                / f"{property_name}_{depth}_mean_prb_buffer.tif"
            )
            status, coverage = download_one(
                session, property_name, depth, destination
            )
            info = gdal_info(destination)
            band = info["bands"][0]
            completed += 1
            row = {
                "property": property_name,
                "depth": depth,
                "coverage_id": coverage,
                "status": status,
                "path": str(destination),
                "bytes": int(destination.stat().st_size),
                "sha256": sha256(destination),
                "width": int(info["size"][0]),
                "height": int(info["size"][1]),
                "crs_wkt_present": bool(
                    info.get("coordinateSystem", {}).get("wkt")
                ),
                "geotransform": json.dumps(info["geoTransform"]),
                "minimum_raw": float(band["minimum"]),
                "maximum_raw": float(band["maximum"]),
                "mean_raw": float(band["mean"]),
                "raw_unit": metadata["raw_unit"],
                "updated_utc": utc_now(),
            }
            row.update({
                key: value for key, value in metadata.items()
                if key != "raw_unit"
            })
            rows.append(row)
            print(
                f"[download] {completed}/{total} "
                f"{property_name} {depth} {status} "
                f"{destination.stat().st_size} bytes",
                flush=True,
            )
    table = pd.DataFrame(rows)
    table_path = REPORT / "soilgrids_layer_manifest.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    reference_size = (
        int(table.iloc[0]["width"]), int(table.iloc[0]["height"])
    )
    reference_transform = str(table.iloc[0]["geotransform"])
    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "local_spatial_basis_frozen": True,
        "eighteen_layers": len(table) == 18,
        "three_properties_six_depths": bool(
            table.groupby("property").size().eq(6).all()
            and set(table["property"]) == set(PROPERTIES)
        ),
        "all_files_nonempty": bool(table["bytes"].gt(1024).all()),
        "all_crs_present": bool(table["crs_wkt_present"].all()),
        "all_grids_aligned": bool(
            table["width"].eq(reference_size[0]).all()
            and table["height"].eq(reference_size[1]).all()
            and table["geotransform"].eq(reference_transform).all()
        ),
        "all_ranges_finite_and_nonnegative": bool(
            table[[
                "minimum_raw", "maximum_raw", "mean_raw"
            ]].notna().all().all()
            and table["minimum_raw"].ge(0).all()
            and table["maximum_raw"].gt(table["minimum_raw"]).all()
        ),
        "no_model_or_station_data_read": True,
        "no_forbidden_period_or_management": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    acquired = all(checks.values())
    gate = {
        "run_id": "20260729_38",
        "phase": "soilgrids_property_acquisition",
        "created_utc": utc_now(),
        "checks": checks,
        "properties": list(PROPERTIES),
        "depths": DEPTHS,
        "layer_count": int(len(table)),
        "download_bounds_mollweide": EXPECTED_BOUNDS,
        "local_catchment_source": str(LOCAL_CATCHMENTS),
        "decision": (
            "SOIL_PROPERTIES_ACQUIRED_AND_FROZEN"
            if acquired else "SOIL_PROPERTY_ACQUISITION_INCOMPLETE"
        ),
        "authorized_next_action": (
            "AUDIT_DIRECT_TEXTURE_AND_MULTIPLE_DRAINAGE_CANDIDATES"
            if acquired else "RETRY_SOIL_PROPERTY_ACQUISITION"
        ),
        "model_run": False,
        "station_discharge_read": False,
        "parameters_calibrated": False,
        "period_2019_2022_read": False,
        "management_fluxes_read": False,
        "runtime_identity": RUNTIME_IDENTITY,
        "series_terminal": False,
    }
    gate_path = REPORT / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sources = [
        PARENT_GATE, LOCAL_SPATIAL_MANIFEST, LOCAL_CATCHMENTS,
        CONTRACT, LITERATURE,
        RUN / "scripts" / "runtime_guard.py",
        RUN / "scripts" / "download_soilgrids_properties.py",
        RUN / "scripts" / "validate_soilgrids_properties.py",
    ]
    provenance = {
        "run_id": "20260729_38",
        "created_utc": utc_now(),
        "sources": [
            {
                "path": str(path),
                "role": "soil_acquisition_source",
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
            for path in sources
        ],
        "products": table.to_dict("records"),
    }
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "decision": gate["decision"],
        "authorized_next_action": gate["authorized_next_action"],
        "layers": len(table),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
    }, ensure_ascii=False, indent=2))
    if not acquired:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
