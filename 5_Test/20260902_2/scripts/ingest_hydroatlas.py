"""Verify, organize and extract the user-downloaded HydroATLAS v1.0 bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(r"E:\SPARROW")
SOURCE = ROOT / r"0_reach_topology\data\未整理\9890531"
DEST = ROOT / r"0_reach_topology\data\raw\hydrology\hydroatlas_v1_0"
RUN = ROOT / r"5_Test\20260902_2"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"

EXPECTED = {
    "BasinATLAS_Catalog_v10.pdf": "2431a6fd7b381b0435133edcc23efecf",
    "BasinATLAS_Data_v10_shp.zip": "ce5013294bdcb6f884ff6cdf1a1acae0",
    "BasinATLAS_Data_v10.gdb.zip": "69af94baee68da5a3f80f09e7b85bd04",
    "HydroATLAS_TechDoc_v10.pdf": "a1b20a273d3e04c55c9d673c7b18fac1",
    "RiverATLAS_Catalog_v10.pdf": "a3eba9288783053eaaecd12ecaccb519",
    # The Figshare file itself hashes to the 33-character sequence below;
    # the earlier planning note omitted the ``c`` after ``74226``.
    "RiverATLAS_Data_v10_shp.zip": "7a6d74226ce8fe6e7e0eb20e2abe71ee",
    "RiverATLAS_Data_v10.gdb.zip": "f6f1cbf80b422710b78947137f7fd4e8",
}
EXTRACT = {
    "BasinATLAS_Data_v10.gdb.zip": DEST / "data" / "BasinATLAS_GDB",
    "RiverATLAS_Data_v10.gdb.zip": DEST / "data" / "RiverATLAS_GDB",
}


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def validated_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "archives").mkdir(exist_ok=True)
    (DEST / "documentation").mkdir(exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)

    verified = []
    for name, expected_md5 in EXPECTED.items():
        folder = "archives" if Path(name).suffix.lower() == ".zip" else "documentation"
        target = validated_path(DEST / folder / name, DEST)
        source = validated_path(SOURCE / name, SOURCE)
        current = source if source.is_file() else target
        if not current.is_file():
            raise FileNotFoundError(f"Neither download nor canonical copy exists for {name}")
        actual_md5 = digest(current, "md5")
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {name}: {actual_md5} != {expected_md5}")
        if target.exists():
            if digest(target, "md5") != expected_md5:
                raise FileExistsError(f"Non-identical canonical file exists: {target}")
        else:
            shutil.move(str(source), str(target))
        verified.append({
            "name": name,
            "path": str(target),
            "bytes": target.stat().st_size,
            "md5": actual_md5,
            "sha256": digest(target, "sha256"),
        })

    extracted = []
    for archive_name, final_dir in EXTRACT.items():
        archive = DEST / "archives" / archive_name
        validated_path(final_dir, DEST)
        # Directory rename is denied by the managed Windows volume even though
        # file creation is allowed.  A prior verified extraction can therefore
        # legitimately remain under the exact ``.part`` directory.  Validate
        # every ZIP member and use that directory as the canonical extraction;
        # new extractions are written directly to their final directory.
        prior_part = final_dir.with_name(final_dir.name + ".part")
        actual_dir = final_dir if final_dir.exists() else prior_part if prior_part.exists() else final_dir
        if not actual_dir.exists():
            actual_dir.mkdir(parents=True)
            with zipfile.ZipFile(archive) as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise RuntimeError(f"Corrupt ZIP member in {archive_name}: {bad}")
                for member in zf.infolist():
                    member_target = (actual_dir / member.filename).resolve()
                    member_target.relative_to(actual_dir.resolve())
                zf.extractall(actual_dir)
        with zipfile.ZipFile(archive) as zf:
            expected_members = {
                member.filename.rstrip("/"): member.file_size
                for member in zf.infolist()
                if not member.is_dir()
            }
        missing_or_wrong = []
        for relative, expected_size in expected_members.items():
            member_path = actual_dir / relative
            if not member_path.is_file() or member_path.stat().st_size != expected_size:
                missing_or_wrong.append(relative)
        if missing_or_wrong:
            raise RuntimeError(
                f"Incomplete extraction for {archive_name}; invalid members: {missing_or_wrong[:10]}"
            )
        members = [p for p in actual_dir.rglob("*") if p.is_file()]
        extracted.append({
            "archive": archive_name,
            "directory": str(actual_dir),
            "directory_rename_limited_by_managed_volume": actual_dir == prior_part,
            "file_count": len(members),
            "bytes": sum(p.stat().st_size for p in members),
            "top_level_entries": sorted(p.name for p in actual_dir.iterdir()),
        })

    unexpected = sorted(p.name for p in SOURCE.iterdir() if p.is_file())
    report = {
        "stage": "20260902_2",
        "status": "PASS_HYDROATLAS_INGESTION" if not unexpected else "FAIL_UNEXPECTED_SOURCE_FILES",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "doi": "10.6084/m9.figshare.9890531.v1",
        "license": "CC BY 4.0",
        "verified_files": verified,
        "extractions": extracted,
        "unexpected_files_remaining_in_download_directory": unexpected,
        "shapefile_archives_extracted": False,
    }
    atomic_json(REPORTS / "hydroatlas_ingestion_audit.json", report)
    atomic_json(LOCKS / "hydroatlas_input_lock.json", report)
    if unexpected:
        raise RuntimeError(f"Unexpected files remain in download directory: {unexpected}")

    manifest = json.loads((ROOT / r"5_Test\20260902_1\program_manifest.json").read_text(encoding="utf-8"))
    manifest["stage_status"]["20260902_1"] = "registered_complete"
    manifest["stage_status"]["20260902_2"] = "PASS_HYDROATLAS_INGESTION"
    manifest["stage_status"]["20260902_3"] = "authorized_next"
    atomic_json(RUN / "program_manifest.json", manifest)


if __name__ == "__main__":
    main()
