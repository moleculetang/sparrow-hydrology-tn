from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260824_1")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
LOCKS = ROOT / "locks"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: dict[str, bool] = {}
    required = [
        ROOT / "program_charter.json",
        ROOT / "program_manifest.json",
        ROOT / "experiment_contract.json",
        ROOT / "manifest_history" / "manifest_revision_001.json",
        LOCKS / "input_lock.json",
        LOCKS / "development_tn_2016_2021.parquet",
        LOCKS / "parent_lock_registry.json",
        OUT / "parent_mass_balance_audit.parquet",
        OUT / "parent_readjudication_metrics.parquet",
        OUT / "absolute_spatial_skill.parquet",
        OUT / "small_tree_inference_sensitivity.parquet",
        OUT / "temporal_fold_parent_selection.parquet",
        OUT / "outer_spatial_parent_selection.parquet",
        REPORTS / "hydrologic_provenance_audit.json",
        REPORTS / "parent_readjudication.json",
        REPORTS / "stage_completion_audit.json",
        REPORTS / "current_tn_mainline_audit_and_repair.md",
        REPORTS / "program_status_after_stage1.json",
    ]
    checks["all_required_artifacts_exist"] = all(path.exists() for path in required)

    lock = json.loads((LOCKS / "input_lock.json").read_text(encoding="utf-8"))
    manifest_snapshot = ROOT / "manifest_history" / "manifest_revision_001.json"
    current_manifest_key = str(ROOT / "program_manifest.json")
    checks["manifest_snapshot_matches_registered_run_hash"] = (
        sha256(manifest_snapshot) == lock["files"][current_manifest_key]
    )

    development = pd.read_parquet(LOCKS / "development_tn_2016_2021.parquet")
    checks["development_year_boundary"] = (
        set(development.year.unique()) == {2016, 2017, 2018, 2019, 2020, 2021}
        and not development.year.eq(2022).any()
    )

    mass = pd.read_parquet(OUT / "parent_mass_balance_audit.parquet")
    checks["twelve_parent_members"] = mass.model_id.nunique() == 12 and len(mass) == 12
    checks["h0_exact_reproduction"] = bool(
        mass.h0_reproduction_pass.all()
        and mass.h0_parent_reproduction_relative_error.max() <= 1e-12
    )
    checks["routing_mass_balance"] = bool(
        mass.mass_balance_pass.all()
        and mass[["h0_mass_balance_relative_error", "h1_mass_balance_relative_error"]].to_numpy().max() <= 1e-10
    )

    metrics = pd.read_parquet(OUT / "parent_readjudication_metrics.parquet")
    checks["formal_scopes_complete"] = set(metrics.scope.unique()) == {"temporal", "LOSO", "LOTO"}
    checks["formal_layers_complete"] = set(metrics.layer.unique()) == {"P1", "P2"}
    checks["metrics_finite"] = bool(np.isfinite(metrics[["point_delta", "ci_lower", "ci_upper"]]).all().all())

    skills = pd.read_parquet(OUT / "absolute_spatial_skill.parquet")
    checks["absolute_skill_complete"] = (
        len(skills) == 24
        and skills.model_id.nunique() == 12
        and set(skills.evaluation) == {"LOSO", "LOTO"}
    )
    tree = pd.read_parquet(OUT / "small_tree_inference_sensitivity.parquet")
    checks["small_tree_exact_test_complete"] = bool(
        len(tree) == 12
        and tree.tree_count.eq(7).all()
        and tree.exact_improvement_permutations.eq(128).all()
        and tree.exact_noninferiority_permutations.eq(128).all()
    )

    hydro = json.loads((REPORTS / "hydrologic_provenance_audit.json").read_text(encoding="utf-8"))
    checks["q72_only_hydrology"] = (
        hydro["status"] == "PASS"
        and set(hydro["hydrology_sources"]) == {"Q72_actual_structural_canonical_main"}
    )
    checks["reference_discharge_excluded"] = (
        not hydro["andreadis_reference_discharge_used_in_exposure"]
        and not hydro["andreadis_reference_discharge_used_in_routed_q72"]
    )
    checks["q_rho_locked"] = hydro["q_rho_matches_locked_interface"]
    checks["h1_semantics_corrected"] = hydro["formal_h1_semantics"] == "areal/benthic hydraulic exposure, not residence time"

    decision = json.loads((REPORTS / "parent_readjudication.json").read_text(encoding="utf-8"))
    checks["four_domain_statuses_present"] = all(
        key in decision for key in (
            "temporal_process_status",
            "within_observed_tree_LOSO_status",
            "new_tree_LOTO_status",
            "monitored_station_P2_status",
        )
    )
    checks["stale_mcmc_withdrawn"] = decision["stale_mcmc_role"] == "withdrawn_from_authoritative_evidence"
    checks["tn_2022_not_read"] = not decision["TN_2022_values_read"] and not hydro["TN_2022_values_read"]

    status = "PASS" if all(checks.values()) else "FAIL"
    output = {
        "status": status,
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "failed": [key for key, value in checks.items() if not value],
        "TN_2022_values_read": False,
    }
    (REPORTS / "independent_verification.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
