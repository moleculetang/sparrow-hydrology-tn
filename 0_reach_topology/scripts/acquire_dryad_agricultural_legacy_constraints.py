"""Acquire the registered small Dryad agricultural-N constraint datasets.

Only immutable files selected before the Stage 42 result are downloaded.  The
10.48 MB simulation archive in dryad.xd2547dsk is intentionally excluded.
Original bytes, official API metadata, SHA-256 digests and an acquisition
manifest are stored under the central raw data tree; nothing is written to a
5_Test directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(r"E:\SPARROW")
RAW_ROOT = ROOT / "0_reach_topology/data/raw/agriculture/nitrogen_legacy"
API_ROOT = "https://datadryad.org"
CHUNK_BYTES = 1024 * 1024

DATASETS = (
    {
        "id": "global_crop_residue_nutrient_removal_dryad_mgqnk99d1",
        "doi": "10.5061/dryad.mgqnk99d1",
        "version_id": 414282,
        "directory": "global_crop_residue_nutrient_removal_dryad_mgqnk99d1",
        "files": (
            (4513686, "crop_residue_coefficients.csv", 21710, "96e17fdbc1466884ac916513b636ff66738237847c938c6576f4960a05f52f88"),
            (4513685, "Nutrient_removal_with_crop_residue.csv", 3909855, "3aa6714125ec9d58eaebf3b9b08815a665078da668a045dea740e949dc0ae804"),
            (4513689, "README.md", 5382, "58d7955279e0486ab34658f28e537a7b6458c13d5c108b27779f3b741c945f19"),
        ),
        "excluded_files": (),
    },
    {
        "id": "cropland_n_loss_endpoints_dryad_xd2547dsk",
        "doi": "10.5061/dryad.xd2547dsk",
        "version_id": 351790,
        "directory": "cropland_n_loss_endpoints_dryad_xd2547dsk",
        "files": (
            (3946200, "README.md", 6696, "7adbd70894a1cc637d60cc09d0a844588e632bc9250058f35a6b47bc37e08631"),
            (3946192, "Yu_et_al._2025_Data_for_Model_validation_at_site_scales.xlsx", 32009, "4d6f4ccf41a2c89395fd133cd30ccf6f6b661dc52dfe0b455d087aface682a99"),
            (3946194, "Yu_et_al._2025_Data_for_soil_15N_meta.xlsx", 110059, "5250caeb657278770251b636565cb05fd8b9f473a76ddd1204b48df59768d6aa"),
        ),
        "excluded_files": (
            {
                "file_id": 3946193,
                "name": "Yu_et_al._2025_Data_for_Simulation_result.zip",
                "bytes": 10482250,
                "sha256": "ea3532a6b926e0a6580ce55fb8c9b510b8926f834a093d7bd3d955edbefa8f93",
                "reason": "not required by the registered observed-endpoint role",
            },
        ),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(part, path)


def download_file(session: requests.Session, file_id: int, target: Path, signed_url: str | None) -> None:
    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()
    url = signed_url or f"{API_ROOT}/api/v2/files/{file_id}/download"
    with session.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with part.open("wb") as stream:
            for block in response.iter_content(CHUNK_BYTES):
                if block:
                    stream.write(block)
    os.replace(part, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signed-url-manifest", type=Path,
        help="Ephemeral JSON map from Dryad file ID to a browser-validated official S3 URL.",
    )
    args = parser.parse_args()
    signed_urls: dict[str, str] = {}
    if args.signed_url_manifest:
        signed_urls = json.loads(args.signed_url_manifest.read_text(encoding="utf-8"))
    session = requests.Session()
    session.headers.update({"User-Agent": "SPARROW-reproducible-research/1.0"})
    manifest_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        destination = RAW_ROOT / str(dataset["directory"])
        destination.mkdir(parents=True, exist_ok=True)
        doi_encoded = str(dataset["doi"]).replace("/", "%2F")
        dataset_url = f"{API_ROOT}/api/v2/datasets/doi%3A{doi_encoded}"
        metadata = session.get(dataset_url, timeout=60)
        metadata.raise_for_status()
        dataset_metadata = metadata.json()
        files_url = f"{API_ROOT}/api/v2/versions/{dataset['version_id']}/files?per_page=100"
        files_response = session.get(files_url, timeout=60)
        files_response.raise_for_status()
        files_metadata = files_response.json()
        atomic_json(dataset_metadata, destination / "dryad_dataset_metadata.json")
        atomic_json(files_metadata, destination / "dryad_files_metadata.json")
        for file_id, name, expected_bytes, expected_sha256 in dataset["files"]:
            target = destination / name
            status = "existing_verified"
            if not target.exists() or target.stat().st_size != expected_bytes or sha256(target) != expected_sha256:
                download_file(session, file_id, target, signed_urls.get(str(file_id)))
                status = "downloaded"
            actual_bytes = target.stat().st_size
            actual_sha256 = sha256(target)
            if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
                raise RuntimeError(f"Dryad integrity mismatch for {target}")
            manifest_rows.append({
                "dataset_id": dataset["id"], "doi": dataset["doi"],
                "version_id": dataset["version_id"], "file_id": file_id,
                "file": str(target), "bytes": actual_bytes,
                "sha256": actual_sha256, "status": status,
            })
        atomic_json({
            "dataset_id": dataset["id"], "doi": dataset["doi"],
            "version_id": dataset["version_id"],
            "license": dataset_metadata.get("license"),
            "selected_files": [row for row in manifest_rows if row["dataset_id"] == dataset["id"]],
            "excluded_files": dataset["excluded_files"],
            "acquired_utc": datetime.now(timezone.utc).isoformat(),
        }, destination / "acquisition_manifest.json")
    payload = {
        "status": "PASS_DRYAD_REGISTERED_SMALL_DATASETS_ACQUIRED",
        "raw_root": str(RAW_ROOT),
        "files": manifest_rows,
        "selected_bytes": sum(int(row["bytes"]) for row in manifest_rows),
        "excluded_simulation_zip": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(payload, RAW_ROOT / "dryad_registered_constraints_acquisition_manifest.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
