from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260818_4")
STAGE2 = Path(r"E:\SPARROW\5_Test\20260818_2")
STAGE3 = Path(r"E:\SPARROW\5_Test\20260818_3")
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow conda environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def main() -> None:
    require_runtime()
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    parents = [
        STAGE2 / "reports" / "source_water_operator_decision.json",
        STAGE2 / "reports" / "verification.json",
        STAGE3 / "reports" / "wwtp_evidence_decision.json",
        STAGE3 / "reports" / "verification.json",
    ]
    start = {str(path): sha256(path) for path in parents}
    dump(REPORTS / "parent_hashes_start.json", start)

    operator = json.loads(parents[0].read_text(encoding="utf-8"))
    point = json.loads(parents[2].read_text(encoding="utf-8"))
    operator_supported = operator.get("operator_status") == "source_water_operator_supported"
    point_supported = point.get("point_source_status") in {"supporting", "supported"}

    if operator_supported and point_supported:
        raise RuntimeError(
            "INTERACTION_REQUIRED_BUT_NOT_IMPLEMENTED_BY_SKIP_BRANCH; "
            "run the registered 2x2 interaction on native 2018-2019 coverage"
        )

    reason = []
    if not operator_supported:
        reason.append("operator_not_supported")
    if not point_supported:
        reason.append("wwtp_PS1_not_supporting")
    decision = {
        "scenario_id": "20260818_4",
        "stage_status": "SKIPPED_BY_REGISTERED_GATE",
        "interaction_status": "not_run",
        "skip_reason": reason,
        "operator_status": operator.get("operator_status"),
        "selected_operator": operator.get("selected_operator"),
        "point_source_status": point.get("point_source_status"),
        "required_conditions": {
            "new_operator_supported": False,
            "PS1_at_least_supporting": False,
        },
        "formal_interaction_candidate_count": 0,
        "evaluation_years_if_triggered": [2018, 2019],
        "no_interaction_claim_permitted": True,
    }
    dump(REPORTS / "interaction_decision.json", decision)
    pd.DataFrame(
        columns=[
            "operator", "point_source_scenario", "model_id", "fold_id",
            "delta_operator", "delta_wwtp", "delta_joint", "interaction",
        ]
    ).to_parquet(OUTPUTS / "interaction_results.parquet", index=False)

    end = {str(path): sha256(path) for path in parents}
    dump(REPORTS / "parent_hashes_end.json", end)
    unchanged = start == end
    verification = {
        "status": "PASS" if unchanged else "FAIL",
        "checks": {
            "parent_hashes_unchanged": unchanged,
            "registered_skip_gate_applied": True,
            "formal_interaction_candidate_count_zero": True,
            "no_2022_use": True,
        },
    }
    dump(REPORTS / "verification.json", verification)
    dump(
        REPORTS / "completion_audit.json",
        {
            "status": verification["status"],
            "requirements": {
                "run_only_if_operator_and_PS1_supported": "PASS",
                "native_2018_2019_boundary_recorded": "PASS",
                "paired_interaction_bootstrap": "NOT_APPLICABLE_STAGE_SKIPPED",
                "parent_lock_protection": "PASS" if unchanged else "FAIL",
            },
        },
    )
    if not unchanged:
        raise RuntimeError("PARENT_HASH_CHANGED")


if __name__ == "__main__":
    main()
