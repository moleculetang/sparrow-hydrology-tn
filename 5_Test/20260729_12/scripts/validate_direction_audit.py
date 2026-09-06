from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
PARENT = ROOT / "5_Test" / "20260729_11"
REPORT = RUN_DIR / "reports" / "reach_direction_adjudication_gate"
MANIFEST = RUN_DIR / "inputs_manifest" / "provenance_manifest.json"
TARGETS = [6, 20, 31, 57, 136, 146]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evidence = pd.read_csv(REPORT / "reach_direction_evidence.csv", encoding="utf-8-sig")
    current = pd.read_parquet(RUN_DIR / "inputs" / "reach_slope_direction_adjudicated.parquet")
    parent = pd.read_parquet(PARENT / "inputs" / "reach_slope_reviewed.parquet")
    static = pd.read_parquet(RUN_DIR / "inputs" / "reach_static_direction_adjudicated.parquet")

    product_hashes_match = all(
        Path(rec["path"]).exists() and sha256(Path(rec["path"])) == rec["sha256"]
        for rec in manifest["products"]
    )
    slope_compare = current[["reach_id", "slope_m_m"]].merge(
        parent[["reach_id", "slope_m_m"]],
        on="reach_id",
        suffixes=("_current", "_parent"),
        validate="one_to_one",
    )
    slopes_identical = bool(
        np.array_equal(
            slope_compare["slope_m_m_current"].to_numpy(),
            slope_compare["slope_m_m_parent"].to_numpy(),
        )
    )
    unresolved = evidence.loc[~evidence["direction_support_pass"].astype(bool), "reach_id"].astype(int).tolist()
    decision_consistent = (
        (gate["passed"] and not unresolved and gate["authorized_next_action"] == "BUILD_Q78_NAT_CONSERVATION_CORE")
        or (
            not gate["passed"]
            and gate["authorized_next_action"] == "SUPPLEMENT_UNRESOLVED_REACH_DIRECTION_EVIDENCE"
            and sorted(unresolved) == sorted(gate["unresolved_reach_ids"])
        )
    )
    checks = {
        "required_files_exist": all(p.exists() for p in [
            REPORT / "gate.json",
            REPORT / "technical_report.md",
            REPORT / "reach_direction_evidence.csv",
            REPORT / "single_edge_reversal_counterfactual.csv",
            REPORT / "same_name_chain_evidence.csv",
            REPORT / "run_manifest.json",
            RUN_DIR / "inputs" / "reach_slope_direction_adjudicated.csv",
            RUN_DIR / "inputs" / "reach_slope_direction_adjudicated.parquet",
            RUN_DIR / "inputs" / "reach_static_direction_adjudicated.parquet",
            MANIFEST,
        ]),
        "target_ids_exact": sorted(evidence["reach_id"].astype(int).tolist()) == TARGETS,
        "target_ids_unique": evidence["reach_id"].is_unique,
        "adjudication_complete": evidence["adjudication_class"].notna().all(),
        "gate_decision_consistent": decision_consistent,
        "parent_and_current_reach_counts_230": len(parent) == len(current) == 230,
        "static_reach_count_230": len(static) == 230,
        "slope_values_bitwise_identical": slopes_identical,
        "no_slope_change_flag": not current[
            "slope_value_changed_by_direction_audit"
        ].astype(bool).any(),
        "all_model_directions_explicit": current["recommended_model_direction"].notna().all(),
        "product_hashes_match_manifest": product_hashes_match,
        "production_topology_declared_unmodified": manifest["production_topology_mutated"] is False,
        "flow_acc_declared_not_truth": manifest["conditioned_flow_accumulation_used_as_truth"] is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "run_id": "20260729_12",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": "conda sparrow",
        "checks": checks,
        "passed_checks": sum(bool(x) for x in checks.values()),
        "total_checks": len(checks),
        "artifact_validation_passed": all(checks.values()),
        "scientific_gate_passed": bool(gate["passed"]),
    }
    (REPORT / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    if not result["artifact_validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
