"""Safely promote one deeply audited CMFD replacement and archive the old payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT = Path(r"E:\SPARROW")
RAW_ROOT = (ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "meteorology" / "cmfd_v2_0" / "03hr").resolve()
WORK_ROOT = (ROOT / "5_Test" / "20260828_29" / "work" / "cmfd_reacquired").resolve()
ARCHIVE = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "meteorology" / "cmfd_v2_0" / "archives" / "corrupt_payloads"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_DEEP_AUDIT" or report.get("promotion_performed"):
        raise RuntimeError("Repair is not an unpromoted PASS_DEEP_AUDIT candidate")
    source = Path(report["source_path"]).resolve()
    candidate = Path(report["candidate_path"]).resolve()
    if not within(source, RAW_ROOT) or not within(candidate, WORK_ROOT):
        raise RuntimeError("Promotion paths are outside their registered roots")
    if not source.is_file() or not candidate.is_file() or source.is_symlink() or candidate.is_symlink():
        raise RuntimeError("Source/candidate must be ordinary existing files")
    candidate_hash = sha256(candidate)
    if candidate_hash != report["sha256"]:
        raise RuntimeError("Candidate hash changed after deep audit")
    old_hash = sha256(source)
    if old_hash == candidate_hash:
        raise RuntimeError("Source already equals candidate; refusing ambiguous promotion")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    archive = ARCHIVE / f"{source.stem}_corrupt_{old_hash[:12]}{source.suffix}"
    if archive.exists():
        raise RuntimeError(f"Archive target already exists: {archive}")
    part = source.with_suffix(source.suffix + ".promote.part")
    if part.exists():
        raise RuntimeError(f"Unexpected promotion part exists: {part}")
    shutil.copy2(candidate, part)
    if sha256(part) != candidate_hash:
        raise RuntimeError("Copied promotion part failed hash verification")
    os.replace(source, archive)
    try:
        os.replace(part, source)
    except Exception:
        if not source.exists() and archive.exists():
            os.replace(archive, source)
        raise
    if sha256(source) != candidate_hash or sha256(archive) != old_hash:
        raise RuntimeError("Post-promotion hash verification failed")
    report.update({
        "promotion_performed": True,
        "promoted_path": str(source),
        "promoted_sha256": candidate_hash,
        "archived_corrupt_path": str(archive),
        "archived_corrupt_sha256": old_hash,
    })
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
