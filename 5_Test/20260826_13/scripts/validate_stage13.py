"""Validate that Stage 13 is complete and internally consistent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


RUN = Path(r"E:\SPARROW\5_Test\20260826_13")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((RUN / "program_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    registry = json.loads((RUN / "reports" / "input_hash_registry.json").read_text(encoding="utf-8"))
    literature = json.loads((RUN / "reports" / "literature_evidence_registry.json").read_text(encoding="utf-8"))
    parent = pd.read_parquet(RUN / "outputs" / "frozen_parent_performance.parquet")

    checks = {
        "stage13_registered": manifest["stage_status"]["20260826_13"] == "PASS_PROGRAM_REGISTERED",
        "stage14_only_next": manifest["authorized_successor"] == "20260826_14",
        "hard_stop_22": "20260826_23" in manifest["hard_stop"],
        "parent_locked": contract["parent"] == "20260825_7 GLOBAL_HBV_R0",
        "target_history_forbidden": contract["spatial_contract"]["target_station_history"].startswith("zero"),
        "TN_forbidden": any(item.startswith("TN observations") for item in contract["forbidden"]),
        "temperature_forbidden": any("temperature" in item for item in contract["forbidden"]),
        "H2_maximum": contract["independent_state_contract"]["maximum_claim"].startswith("H2"),
        "eight_trees": contract["spatial_contract"]["terminal_trees"] == [1, 20, 22, 26, 56, 166, 212, 217],
        "all_hashes_match": all(sha256(Path(row["path"])) == row["sha256"] for row in registry["files"]),
        "literature_count": len(literature["entries"]) == 8,
        "parent_performance_rows": len(parent) == 4,
    }
    result = {"stage": "20260826_13", "checks": checks, "all_checks_pass": all(checks.values())}
    (RUN / "reports" / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["all_checks_pass"]:
        raise RuntimeError(f"Stage 13 validation failed: {checks}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
