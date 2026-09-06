from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260818_7")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    lock = json.loads((ROOT / "final_lock.json").read_text(encoding="utf-8"))
    audit = pd.read_parquet(ROOT / "outputs" / "requirement_by_requirement_audit.parquet")
    current = {path: sha256(Path(path)) for path in lock["authoritative_output_sha256"]}
    checks = {
        "all_persisted_requirements_pass": bool(audit["pass"].all()),
        "hard_check_count_matches": bool(len(audit) == lock["hard_check_count"]),
        "authoritative_hashes_match": current == lock["authoritative_output_sha256"],
        "no_failed_checks": lock["failed_checks"] == [],
        "no_2022_selection": not lock["selection_uses_2022"],
        "parent_files_not_modified": not lock["parent_files_modified"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    (ROOT / "reports" / "independent_verification.json").write_text(
        json.dumps({"status": status, "checks": checks}, indent=2), encoding="utf-8"
    )
    if status != "PASS":
        raise RuntimeError("STAGE7_INDEPENDENT_VERIFICATION_FAILED")


if __name__ == "__main__":
    main()
