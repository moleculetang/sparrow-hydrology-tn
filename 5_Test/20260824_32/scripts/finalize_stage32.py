"""Verify and hash the completed 20260824_25-32 program."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_32"
MANIFEST = ROOT / "5_Test" / "20260824_25" / "program_manifest.json"
VALIDATION = RUN / "reports" / "stage32_validation.json"
LOCK = RUN / "locks" / "tn_mainline_lock.json"
PRODUCTION = ROOT / "0_reach_topology" / "data" / "processed" / "tn_long_history_mainline" / "canonical_tn_reach_monthly_1961_2024.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    if Path(sys.executable).parent.name.lower() != "sparrow":
        raise RuntimeError("conda sparrow required")
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    program = json.loads(MANIFEST.read_text(encoding="utf-8"))
    production = pd.read_parquet(PRODUCTION)
    required = {
        "reach_id", "year", "month", "tn_process_mg_l", "tn_p1_global_mg_l",
        "routed_fert_kg_n", "routed_man_kg_n", "routed_bnf_kg_n", "routed_dep_kg_n",
    }
    stage33 = sorted((ROOT / "5_Test").glob("20260824_33*"))
    checks = {
        "stage32_locked": validation.get("status") == "PASS_STAGE32_TN_MAINLINE_LOCKED",
        "lock_matches_validation": lock.get("status") == validation.get("status"),
        "program_complete": program.get("status") == "completed_tn_mainline_locked_20260824_32",
        "hard_stop_registered": program.get("hard_stop_after") == "20260824_32" and bool(program.get("no_automatic_extension")),
        "no_stage33_created": len(stage33) == 0,
        "production_hash_matches_lock": sha256(PRODUCTION) == lock.get("production_hash"),
        "production_grain_exact": len(production) == 176640 and production.reach_id.nunique() == 230 and production.year.min() == 1961 and production.year.max() == 2024,
        "production_key_unique": not production.duplicated(["reach_id", "year", "month"]).any(),
        "production_schema_complete": required.issubset(production.columns),
        "production_core_values_finite": bool(np.isfinite(production[["tn_process_mg_l", "tn_p1_global_mg_l", "routed_total_kg_n"]].to_numpy(float)).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))
    artifacts = [
        MANIFEST,
        RUN / "experiment_contract.json",
        RUN / "scripts" / "run_stage32.py",
        RUN / "scripts" / "finalize_stage32.py",
        VALIDATION,
        RUN / "reports" / "technical_report.md",
        LOCK,
        RUN / "outputs" / "final_l0_parameters.parquet",
        RUN / "outputs" / "final_p2_station_effects.parquet",
        RUN / "outputs" / "final_fit_and_backreport_predictions.parquet",
        RUN / "outputs" / "final_fit_and_backreport_metrics.parquet",
        PRODUCTION,
    ]
    for stage in range(25, 32):
        stage_root = ROOT / "5_Test" / f"20260824_{stage}"
        candidate = stage_root / "reports" / f"stage{stage}_validation.json"
        if candidate.exists():
            artifacts.append(candidate)
        report = stage_root / "reports" / "technical_report.md"
        if report.exists():
            artifacts.append(report)
    result = {
        "program": "20260824_25_32_LONG_HISTORY_TN_LEGACY_REBUILD",
        "status": "PASS_FINAL_ARTIFACT_AUDIT",
        "checks": checks,
        "artifact_count": len(artifacts),
        "artifacts": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts],
        "authoritative_production": str(PRODUCTION),
        "no_successor": True,
    }
    write_json(RUN / "final_artifact_manifest.json", result)
    print(json.dumps(
        result, ensure_ascii=False, indent=2,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ))


if __name__ == "__main__":
    main()
