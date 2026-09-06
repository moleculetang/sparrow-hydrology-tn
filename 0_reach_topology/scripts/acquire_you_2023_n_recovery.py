"""Acquire and inventory the public You et al. (2023) N-recovery source data.

The immutable archive is downloaded from the DOI-backed Zenodo record into the
central raw-data tree.  Only the published ``Source Data.xlsx`` and repository
``LICENSE`` are extracted, with path-traversal checks.  The workbook is then
inventoried in the central processed-data tree; it is not converted into a
reach forcing because it contains experimental/meta-analytic endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd
import requests


RECORD_API = "https://zenodo.org/api/records/8310785"
EXPECTED_RECORD_ID = 8310785
EXPECTED_ARCHIVE_MD5 = "97b1646929a0d71b576e8318d586fa20"
EXPECTED_ARCHIVE_BYTES = 12_111_161
SOURCE_SUFFIX = PurePosixPath("articles/ncoms23/Source Data.xlsx")
LICENSE_SUFFIX = PurePosixPath("LICENSE")

DEFAULT_RAW_ROOT = Path(
    r"E:\SPARROW\0_reach_topology\data\raw\agriculture\nitrogen_legacy"
) / "you_etal_2023_global_n_recovery"
DEFAULT_PROCESSED_ROOT = Path(
    r"E:\SPARROW\0_reach_topology\data\processed\agricultural_n_legacy_constraints"
) / "you_etal_2023_global_n_recovery"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_stream(session: requests.Session, url: str, target: Path) -> None:
    temp = target.with_suffix(target.suffix + ".part")
    if temp.exists():
        temp.unlink()
    with session.get(url, timeout=(30, 300), stream=True) as response:
        response.raise_for_status()
        with temp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp.replace(target)


def safe_selected_extract(archive: Path, suffix: PurePosixPath, target: Path) -> str:
    with zipfile.ZipFile(archive) as zf:
        candidates: list[str] = []
        for name in zf.namelist():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"Unsafe archive member: {name}")
            if len(member.parts) >= len(suffix.parts) and member.parts[-len(suffix.parts) :] == suffix.parts:
                candidates.append(name)
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one member ending in {suffix}; found {len(candidates)}: {candidates}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(candidates[0]) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        return candidates[0]


def sanitize_sheet_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "sheet"


def inventory_workbook(workbook: Path, processed_root: Path) -> pd.DataFrame:
    excel = pd.ExcelFile(workbook)
    rows: list[dict[str, object]] = []
    for sheet in excel.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet, header=None)
        nonempty = int(frame.notna().sum().sum())
        rows.append(
            {
                "sheet_name": sheet,
                "sheet_slug": sanitize_sheet_name(sheet),
                "n_rows_including_headers": int(frame.shape[0]),
                "n_columns": int(frame.shape[1]),
                "nonempty_cells": nonempty,
                "first_nonempty_row_zero_based": (
                    int(frame.dropna(how="all").index.min()) if nonempty else None
                ),
                "last_nonempty_row_zero_based": (
                    int(frame.dropna(how="all").index.max()) if nonempty else None
                ),
            }
        )
    inventory = pd.DataFrame(rows)
    processed_root.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(processed_root / "source_data_sheet_inventory.parquet", index=False)
    return inventory


def extract_constraint_tables(workbook: Path, processed_root: Path) -> dict[str, object]:
    """Extract only documented endpoint tables, preserving their published values."""

    def normalize_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].astype("string")
        return frame

    primary = pd.read_excel(workbook, sheet_name="Primary database (total)")
    primary = primary.rename(
        columns={
            "replication ": "replication",
            "year": "experimental_year_index",
        }
    )
    primary.insert(0, "source_article_doi", "10.1038/s41467-023-41504-2")
    primary.insert(1, "source_data_doi", "10.5281/zenodo.8310785")
    primary["scientific_role"] = "crop_N_recovery_and_management_endpoint"
    primary["is_reach_forcing"] = False
    primary["has_valid_coordinates"] = (
        pd.to_numeric(primary["lat"], errors="coerce").between(-90, 90)
        & pd.to_numeric(primary["lon"], errors="coerce").between(-180, 180)
    )
    primary = normalize_object_columns(primary)

    meta = pd.read_excel(workbook, sheet_name="Meta-analitical database (total")
    meta.insert(0, "source_article_doi", "10.1038/s41467-023-41504-2")
    meta.insert(1, "source_data_doi", "10.5281/zenodo.8310785")
    meta["scientific_role"] = "management_effect_endpoint"
    meta["is_reach_forcing"] = False
    meta = normalize_object_columns(meta)

    meta_of_meta = pd.read_excel(workbook, sheet_name="meta_of_meta-analytica_data")
    meta_of_meta.insert(0, "source_article_doi", "10.1038/s41467-023-41504-2")
    meta_of_meta.insert(1, "source_data_doi", "10.5281/zenodo.8310785")
    meta_of_meta["scientific_role"] = "published_meta_analysis_endpoint"
    meta_of_meta["is_reach_forcing"] = False
    meta_of_meta = normalize_object_columns(meta_of_meta)

    primary_path = processed_root / "primary_n_recovery_trials.parquet"
    meta_path = processed_root / "meta_analytical_management_effects.parquet"
    meta_of_meta_path = processed_root / "meta_of_meta_management_effects.parquet"
    study_summary_path = processed_root / "study_balanced_crop_recovery_summary.parquet"
    primary.to_parquet(primary_path, index=False)
    meta.to_parquet(meta_path, index=False)
    meta_of_meta.to_parquet(meta_of_meta_path, index=False)

    recovery = primary.loc[
        primary["nue_type"].eq("REN")
        & pd.to_numeric(primary["nuet_mean"], errors="coerce").between(0, 100)
    ].copy()
    recovery["nuet_mean"] = pd.to_numeric(recovery["nuet_mean"], errors="coerce")
    study_level = recovery.groupby(
        ["studyid", "reference", "g_crop_type", "fertilizer_type"],
        dropna=False,
        as_index=False,
    ).agg(
        study_median_treatment_ren_percent=("nuet_mean", "median"),
        source_rows=("id", "size"),
    )

    def q10(values: pd.Series) -> float:
        return float(values.quantile(0.10))

    def q25(values: pd.Series) -> float:
        return float(values.quantile(0.25))

    def q75(values: pd.Series) -> float:
        return float(values.quantile(0.75))

    def q90(values: pd.Series) -> float:
        return float(values.quantile(0.90))

    study_summary = study_level.groupby(
        ["g_crop_type", "fertilizer_type"], dropna=False, as_index=False
    ).agg(
        n_studies=("studyid", "nunique"),
        n_study_treatment_groups=("studyid", "size"),
        ren_p10_percent=("study_median_treatment_ren_percent", q10),
        ren_p25_percent=("study_median_treatment_ren_percent", q25),
        ren_median_percent=("study_median_treatment_ren_percent", "median"),
        ren_p75_percent=("study_median_treatment_ren_percent", q75),
        ren_p90_percent=("study_median_treatment_ren_percent", q90),
    )
    study_summary.insert(0, "source_article_doi", "10.1038/s41467-023-41504-2")
    study_summary.insert(1, "source_data_doi", "10.5281/zenodo.8310785")
    study_summary["weighting"] = "one median per study-crop-fertilizer_type group"
    study_summary["scientific_role"] = "crop_recovery_prior_envelope_only"
    study_summary.to_parquet(study_summary_path, index=False)

    id_numeric = pd.to_numeric(primary["id"], errors="coerce")
    study_numeric = pd.to_numeric(primary["studyid"], errors="coerce")
    lat_numeric = pd.to_numeric(primary["lat"], errors="coerce")
    lon_numeric = pd.to_numeric(primary["lon"], errors="coerce")
    nuet = pd.to_numeric(primary["nuet_mean"], errors="coerce")
    nuec = pd.to_numeric(primary["nuec_mean"], errors="coerce")
    n_dose = pd.to_numeric(primary["n_dose"], errors="coerce")

    qa = {
        "primary_rows": int(len(primary)),
        "unique_ids": int(id_numeric.nunique(dropna=True)),
        "duplicate_id_rows": int(primary.duplicated(subset=["id"], keep=False).sum()),
        "unique_studies": int(study_numeric.nunique(dropna=True)),
        "unique_references": int(primary["reference"].nunique(dropna=True)),
        "valid_coordinate_rows": int(primary["has_valid_coordinates"].sum()),
        "latitude_range": [float(lat_numeric.min()), float(lat_numeric.max())],
        "longitude_range": [float(lon_numeric.min()), float(lon_numeric.max())],
        "crop_counts": {
            str(k): int(v)
            for k, v in primary["g_crop_type"].value_counts(dropna=False).items()
        },
        "fertilizer_type_counts": {
            str(k): int(v)
            for k, v in primary["fertilizer_type"].value_counts(dropna=False).items()
        },
        "nue_type_counts": {
            str(k): int(v)
            for k, v in primary["nue_type"].value_counts(dropna=False).items()
        },
        "treatment_n_recovery_percent_range": [float(nuet.min()), float(nuet.max())],
        "control_n_recovery_percent_range": [float(nuec.min()), float(nuec.max())],
        "n_dose_kg_n_ha_range": [float(n_dose.min()), float(n_dose.max())],
        "meta_analysis_rows": int(len(meta)),
        "meta_of_meta_rows": int(len(meta_of_meta)),
        "study_balanced_summary_rows": int(len(study_summary)),
        "scope_guard": (
            "Apparent/recovery-efficiency and management endpoints constrain crop-recovery "
            "priors only. They do not observe soil retention, immobilization, mineralization, "
            "groundwater export or reach-specific forcing."
        ),
        "processed_paths": [
            str(primary_path),
            str(meta_path),
            str(meta_of_meta_path),
            str(study_summary_path),
        ],
    }
    return qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    archive_dir = raw_root / "archives"
    data_dir = raw_root / "data"
    metadata_dir = raw_root / "metadata"
    for directory in (archive_dir, data_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "SPARROW-agricultural-N-data-audit/1.0"})
    record_response = session.get(RECORD_API, timeout=(30, 120))
    record_response.raise_for_status()
    record = record_response.json()
    if int(record.get("id", -1)) != EXPECTED_RECORD_ID:
        raise RuntimeError(f"Unexpected Zenodo record id: {record.get('id')}")
    files = record.get("files", [])
    if len(files) != 1:
        raise RuntimeError(f"Expected one Zenodo archive, found {len(files)}")
    remote = files[0]
    expected_checksum = str(remote.get("checksum", ""))
    if expected_checksum != f"md5:{EXPECTED_ARCHIVE_MD5}":
        raise RuntimeError(f"Unexpected archive checksum: {expected_checksum}")
    if int(remote.get("size", -1)) != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(f"Unexpected archive size: {remote.get('size')}")

    archive = archive_dir / "gerardhros-phd_luncheng-ncom23.zip"
    if args.force_download or not archive.exists():
        download_stream(session, remote["links"]["self"], archive)
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(f"Archive size mismatch: {archive.stat().st_size}")
    if file_hash(archive, "md5") != EXPECTED_ARCHIVE_MD5:
        raise RuntimeError("Archive MD5 mismatch")

    source_xlsx = data_dir / "Source Data.xlsx"
    license_file = metadata_dir / "LICENSE_GPL-3.0.txt"
    source_member = safe_selected_extract(archive, SOURCE_SUFFIX, source_xlsx)
    license_member = safe_selected_extract(archive, LICENSE_SUFFIX, license_file)

    metadata_snapshot = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "record_api": RECORD_API,
        "record_id": record["id"],
        "concept_record_id": record.get("conceptrecid"),
        "doi": record.get("doi"),
        "concept_doi": record.get("conceptdoi"),
        "title": record.get("metadata", {}).get("title"),
        "description": record.get("metadata", {}).get("description"),
        "access_right": record.get("metadata", {}).get("access_right"),
        "zenodo_license": record.get("metadata", {}).get("license"),
        "repository_license_file": license_member,
        "repository_license_interpretation": "GNU GPL v3 repository license; preserve attribution and license with redistributed derivatives.",
        "archive_member_source_data": source_member,
        "archive_remote": remote,
    }
    (metadata_dir / "zenodo_record_8310785.json").write_text(
        json.dumps(metadata_snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    inventory = inventory_workbook(source_xlsx, processed_root)
    endpoint_qa = extract_constraint_tables(source_xlsx, processed_root)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_files": [
            {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "md5": file_hash(archive, "md5"),
                "sha256": file_hash(archive),
            },
            {
                "path": str(source_xlsx),
                "bytes": source_xlsx.stat().st_size,
                "sha256": file_hash(source_xlsx),
            },
            {
                "path": str(license_file),
                "bytes": license_file.stat().st_size,
                "sha256": file_hash(license_file),
            },
        ],
        "processed_files": [
            str(processed_root / "source_data_sheet_inventory.parquet"),
            *endpoint_qa["processed_paths"],
        ],
        "workbook_sheets": inventory.to_dict(orient="records"),
        "scientific_role": "experimental/meta-analytic N-recovery endpoint constraints; never a reach forcing",
    }
    (metadata_dir / "sha256_manifest_and_qa.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (processed_root / "qa.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "source_workbook_sha256": file_hash(source_xlsx),
                "sheet_count": int(len(inventory)),
                "total_nonempty_cells": int(inventory["nonempty_cells"].sum()),
                "not_reach_forcing": True,
                "parameter_role": "external endpoint prior/validation only",
                "endpoint_table_qa": endpoint_qa,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
