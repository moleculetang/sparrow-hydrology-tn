from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_10"
REPORT = RUN / "reports"
OUT = RUN / "outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


stage_paths = {
    1: ROOT / "5_Test" / "20260826_1" / "reports" / "stage1_decision.json",
    2: ROOT / "5_Test" / "20260826_2" / "reports" / "stage2_decision.json",
    3: ROOT / "5_Test" / "20260826_3" / "reports" / "stage3_decision.json",
    4: ROOT / "5_Test" / "20260826_4" / "reports" / "stage4_decision.json",
    5: ROOT / "5_Test" / "20260826_5" / "reports" / "stage5_decision.json",
    6: ROOT / "5_Test" / "20260826_6" / "reports" / "stage6_decision.json",
    7: ROOT / "5_Test" / "20260826_7" / "reports" / "stage7_decision.json",
    8: ROOT / "5_Test" / "20260826_8" / "reports" / "stage8_decision.json",
    9: ROOT / "5_Test" / "20260826_9" / "reports" / "ensemble_and_primary_lock.json",
    10: REPORT / "final_program_decision.json",
}
decisions = {stage: json.loads(path.read_text(encoding="utf-8")) for stage, path in stage_paths.items()}
monthly = pd.read_parquet(OUT / "tn_hydrology_response_ensemble_monthly_2006_2022.parquet")
uncertainty = pd.read_parquet(OUT / "tn_hydrology_structural_uncertainty_monthly_2006_2022.parquet")
folds = pd.read_parquet(ROOT / "5_Test" / "20260826_8" / "outputs" / "nested_fold_audit.parquet")
access = json.loads((REPORT / "retrospective_access_audit.json").read_text(encoding="utf-8"))
actual_trees = sorted(folds.heldout_terminal_tree.astype(int).unique().tolist())
checks = {
    "all_stage_decisions_present": len(decisions) == 10,
    "parent_reproduction_zero_delta": decisions[2]["parent_max_abs_delta_m3_s"] == 0.0,
    "all_structure_preflights_passed": decisions[3]["candidate_preflight_passed"] == decisions[3]["candidate_count"],
    "correct_terminal_trees": actual_trees == [1, 20, 22, 26, 56, 166, 212, 217],
    "nested_fold_count": len(folds) == 16,
    "nested_all_folds_have_target_stations": bool((folds.stations > 0).all()),
    "monthly_rows": len(monthly) == 140760,
    "monthly_unique_keys": not monthly.duplicated(["model_id", "year", "month", "reach_id"]).any(),
    "monthly_local_component_closure": float(np.max(np.abs(monthly.local_total_m3_s - monthly[["local_fast_response_m3_s", "local_intermediate_response_m3_s", "local_slow_response_m3_s"]].sum(axis=1)))) <= 1e-10,
    "monthly_routed_component_closure": float(np.max(np.abs(monthly.routed_total_m3_s - monthly[["routed_fast_response_m3_s", "routed_intermediate_response_m3_s", "routed_slow_response_m3_s"]].sum(axis=1)))) <= 1e-10,
    "uncertainty_rows": len(uncertainty) == 46920,
    "uncertainty_unique_keys": not uncertainty.duplicated(["year", "month", "reach_id"]).any(),
    "retrospective_read_once_in_final_script": access["formal_script_retrospective_read_count"] == 1,
    "no_refit_after_retrospective": access["parameters_refit_after_retrospective_read"] is False,
    "TN_not_used": decisions[10]["TN_used_in_hydrology"] is False,
    "temperature_not_used": decisions[10]["temperature_used"] is False,
    "primary_is_HBV": decisions[10]["primary_total_flow_model"] == "HBV3_PARENT_GLOBAL_HBV_R0",
    "components_not_identified": decisions[10]["component_identifiability_status"] == "TOTAL_FLOW_SUPPORTED_FAST_INTERMEDIATE_SLOW_NOT_IDENTIFIED",
    "invalid_first_pass_quarantined": (ROOT / "5_Test" / "20260826_8" / "reports" / "invalid_artifacts_do_not_use.json").is_file(),
}
passed = all(bool(value) for value in checks.values())
manifest = {
    "program": "20260826 multi-structure conserving hydrologic response ensemble",
    "status": "COMPLETE" if passed else "FINAL_VALIDATION_FAILED",
    "stages": {str(stage): decisions[stage].get("status") for stage in range(1, 11)},
    "primary_total_flow": "20260825_7 HBV3/GLOBAL_HBV_R0 routed_total_m3_s",
    "sensitivity_members": ["HBV3_PARENT", "RAVEN_SACSMA3", "MTRS3"],
    "response_component_authorization": "DIAGNOSTIC_STRUCTURAL_SENSITIVITY_ONLY",
    "repair_slots": "20260826_11_to_12_only_if_a_later_integrity_failure_is_found",
    "automatic_extension_beyond_12": "FORBIDDEN",
}
write_json(RUN / "program_manifest.json", manifest)
validation = {"status": "PASS_FINAL_PROGRAM_VALIDATION" if passed else "FINAL_PROGRAM_VALIDATION_FAILED", "checks": checks, "actual_terminal_trees": actual_trees}
write_json(REPORT / "final_validation.json", validation)
integrity_paths = [
    RUN / "experiment_contract.json",
    RUN / "program_manifest.json",
    OUT / "sensitivity_member_spinup_audit.parquet",
    OUT / "tn_hydrology_response_ensemble_monthly_2006_2022.parquet",
    OUT / "tn_hydrology_structural_uncertainty_monthly_2006_2022.parquet",
    OUT / "locked_retrospective_sensitivity_performance.parquet",
    REPORT / "tn_hydrology_bridge_manifest.json",
    REPORT / "retrospective_access_audit.json",
    REPORT / "final_program_decision.json",
    REPORT / "technical_report.md",
    REPORT / "final_validation.json",
]
write_json(REPORT / "integrity.json", {str(path): sha256(path) for path in integrity_paths})
print(json.dumps(validation, ensure_ascii=False, indent=2))
if not passed:
    raise SystemExit(1)
