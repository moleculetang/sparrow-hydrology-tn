"""Archive and integrity-check newly downloaded soil and groundwater datasets.

All transformations use the `sparrow` conda interpreter.  The script moves
only the four explicitly supplied items from data/未整理 into the governed raw
directory, preserving original archives for provenance.  It is intentionally
idempotent so a stopped extraction can be resumed safely.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(r"E:\SPARROW")
UNSORTED = ROOT / "0_reach_topology" / "data" / "未整理"
RAW = ROOT / "0_reach_topology" / "data" / "raw"


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_once(source: Path, target: Path) -> Path:
    """Move a user-supplied source only after verifying its exact destination."""
    if target.exists():
        return target
    if not source.exists():
        raise FileNotFoundError(f"Neither source nor organized target exists: {source}")
    ensure(target.parent)
    shutil.move(str(source), str(target))
    return target


def crc_check(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"CRC check failed in {path.name}: {bad}")


def extract_member(archive_path: Path, member: str, target: Path) -> Path:
    if target.exists():
        return target
    ensure(target.parent)
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return target


def extract_all_once(archive_path: Path, target_dir: Path) -> None:
    marker = target_dir / ".extraction_complete"
    if marker.exists():
        return
    ensure(target_dir)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)
    marker.write_text(f"extracted from {archive_path.name}\n", encoding="utf-8")


def extract_nested_netcdf(outer: Path, member: str, target: Path) -> None:
    if target.exists():
        return
    ensure(target.parent)
    with zipfile.ZipFile(outer) as archive:
        with archive.open(member) as stream:
            payload = BytesIO(stream.read())
    with zipfile.ZipFile(payload) as nested:
        nc = [item for item in nested.infolist() if item.filename.lower().endswith(".nc")]
        if len(nc) != 1:
            raise RuntimeError(f"Expected one NetCDF in {member}, found {len(nc)}")
        with nested.open(nc[0]) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def write_metadata(directory: Path, payload: dict) -> None:
    ensure(directory / "metadata")
    (directory / "metadata" / "ingest_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    # GLHYMPS 2.0: retain outer DOI download and its supplied inner archive.
    gl = RAW / "soil" / "glhymps_v2_0"
    gl_outer = move_once(UNSORTED / "doi_10.5683_SP2_TTJNIU.zip", gl / "archives" / "doi_10.5683_SP2_TTJNIU.zip")
    crc_check(gl_outer)
    gl_inner = extract_member(gl_outer, "GLHYMPS.zip", gl / "archives" / "GLHYMPS.zip")
    readme = extract_member(gl_outer, "Readme_GLHYMPS2_0.txt", gl / "metadata" / "Readme_GLHYMPS2_0.txt")
    crc_check(gl_inner)
    extract_all_once(gl_inner, gl / "data")
    write_metadata(gl, {
        "dataset": "GLHYMPS 2.0",
        "doi": "10.5683/SP2/TTJNIU",
        "content": "global polygon permeability and porosity",
        "permeability_encoding": "logK_Ferr_x100_INT / 100 is log10(permeability_m2)",
        "porosity_encoding": "Porosity_x100 / 100 is volumetric porosity",
        "archives_crc_verified": True,
        "readme": str(readme),
    })

    # The actual downloaded raster names specify 1 km; preserve the complete
    # property bundle, not only TN, because bulk density is needed for N stock.
    china = RAW / "soil" / "china_soil_properties_2010_2018_1km"
    move_once(
        UNSORTED / "中国高分辨率国家土壤信息网格基本属性数据集（2010-2018）",
        china / "data" / "source_bundle",
    )
    tn_files = sorted(
        p for p in (china / "data" / "source_bundle").glob("tn*_1km.tif")
        if not p.name.startswith("tnd")
    )
    expected_tn = {"tn05_1km.tif", "tn515_1km.tif", "tn1530_1km.tif", "tn3060_1km.tif", "tn60100_1km.tif", "tn100200_1km.tif"}
    present_tn = {p.name for p in tn_files}
    if present_tn != expected_tn:
        raise RuntimeError(f"Unexpected TN depth set: {sorted(present_tn)}")
    write_metadata(china, {
        "dataset": "China soil basic properties 2010–2018",
        "actual_resolution": "1 km (from downloaded file names)",
        "tn_depths_cm": ["0-5", "5-15", "15-30", "30-60", "60-100", "100-200"],
        "tn_unit_after_scale": "g kg-1",
        "tn_scale_divisor": 100,
        "tn_files": [p.name for p in tn_files],
    })

    # CSDL backup: outer archive retains all downloaded properties; extract the
    # six TN NetCDF members needed for a reproducible independent TN comparison.
    csdl = RAW / "soil" / "csdl_v2_10km"
    csdl_outer = move_once(UNSORTED / "netCDF.zip", csdl / "archives" / "netCDF.zip")
    crc_check(csdl_outer)
    depth_keys = {"0-5": "0-5cm", "5-15": "5-15cm", "15-30": "15-30cm", "30-60": "30-60cm", "60-100": "60-100cm", "100-200": "100-200cm"}
    for label, source_depth in depth_keys.items():
        extract_nested_netcdf(csdl_outer, f"netCDF/TN/TN_{source_depth}_10km.zip", csdl / "data" / "tn" / f"TN_{label}cm_10km.nc")
    write_metadata(csdl, {
        "dataset": "CSDL v2 backup soil product",
        "actual_resolution": "10 km (from internal file names)",
        "tn_depths_cm": list(depth_keys),
        "role": "independent backup/comparison to primary China 1 km TN product",
        "outer_archive_crc_verified": True,
    })

    # Global nitrate observations and decadal aquifer rasters.
    no3 = RAW / "hydrology" / "groundwater_nitrate_global_1979_2022"
    no3_outer = move_once(UNSORTED / "global_nitrate_dataset.zip", no3 / "archives" / "global_nitrate_dataset.zip")
    crc_check(no3_outer)
    extract_all_once(no3_outer, no3 / "data")
    xlsx = no3 / "data" / "global_nitrate_dataset" / "measured_nitrate_dataset" / "nitrate_dataset_1979_2022.xlsx"
    rasters = sorted((no3 / "data" / "global_nitrate_dataset" / "decadal_aquiferNO3_avg_datasets").glob("aquiferNO3_avg_*.tif"))
    if not xlsx.exists() or len(rasters) != 4:
        raise RuntimeError("Groundwater nitrate archive lacks its expected observations or four decadal rasters")
    write_metadata(no3, {
        "dataset": "Global groundwater nitrate concentration",
        "coverage": "1979–2022 observations; 1980s–2010s aquifer decadal rasters",
        "role": "groundwater nitrate observation/validation layer, not a TN input load",
        "outer_archive_crc_verified": True,
        "observation_workbook": str(xlsx),
        "decadal_rasters": [p.name for p in rasters],
    })
    print("Ingestion and archive CRC checks completed.")


if __name__ == "__main__":
    main()
