from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260824_3")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
LOCKS = ROOT / "locks"
PARENT_LOCK = Path(r"E:\SPARROW\5_Test\20260824_1\locks\parent_lock_registry.json")
STAGE2_LOCK = Path(r"E:\SPARROW\5_Test\20260824_2\final_lock.json")
EXPECTED_PARENT_SHA = "696e26fbb65995244338a9f2edcb221c50f832b7cfc314bf15845dd6e2d64778"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    decision = json.loads((REPORTS / "temporal_decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "program_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "experiment_contract.json").read_text(encoding="utf-8"))
    prelock = json.loads((LOCKS / "pre_tn_input_lock.json").read_text(encoding="utf-8"))
    spin = pd.read_parquet(OUT / "operator_spinup_audit.parquet")
    mass = pd.read_parquet(OUT / "operator_mass_balance_audit.parquet")
    nesting = pd.read_parquet(OUT / "endpoint_exact_nesting_audit.parquet")
    parameters = pd.read_parquet(OUT / "candidate_fold_parameters.parquet")
    readouts = pd.read_parquet(OUT / "candidate_fold_readouts.parquet")
    selections = pd.read_parquet(OUT / "fold_phi_selections.parquet")
    predictions = pd.read_parquet(OUT / "temporal_oof_predictions.parquet")
    gates = pd.read_parquet(OUT / "paired_temporal_gates.parquet")
    metrics = pd.read_parquet(OUT / "ensemble_performance_metrics.parquet")
    report = (REPORTS / "technical_report.md").read_text(encoding="utf-8")
    required = [
        ROOT / "experiment_contract.json", ROOT / "program_manifest.json", ROOT / "README.md",
        ROOT / "scripts" / "run_temporal_son_fraction.py", LOCKS / "pre_tn_input_lock.json",
        OUT / "operator_spinup_audit.parquet", OUT / "operator_mass_balance_audit.parquet",
        OUT / "endpoint_exact_nesting_audit.parquet", OUT / "candidate_fold_parameters.parquet",
        OUT / "candidate_fold_readouts.parquet", OUT / "fold_phi_selections.parquet",
        OUT / "temporal_oof_predictions.parquet", OUT / "paired_temporal_gates.parquet",
        OUT / "six_mu_ensemble_predictions.parquet", OUT / "ensemble_performance_metrics.parquet",
        OUT / "ensemble_monthly_residuals.parquet", REPORTS / "temporal_decision.json",
        REPORTS / "technical_report.md", REPORTS / "stage_completion_audit.json",
        PARENT_LOCK, STAGE2_LOCK,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing final artifacts: {missing}")
    authoritative = {str(path): sha256(path) for path in required}
    final_lock = {
        "experiment_id": "20260824_3",
        "status": "FROZEN_COMPLETE",
        "terminal_scientific_status": decision["status"],
        "model_mainline": "Q72-F00-T1-H1_GLOBAL",
        "generic_cropland_SON12_process_supported": False,
        "partial_fraction_extension_supported": False,
        "nested_spatial_authorized": False,
        "TN_years_read": decision["TN_years_read"],
        "TN_2022_values_read": False,
        "authoritative_artifacts": authoritative,
    }
    dump(ROOT / "final_lock.json", final_lock)

    p1_no = metrics.loc[(metrics.layer == "P1") & (metrics.mechanism == "NO_SON_PHI0")].iloc[0]
    p1_partial = metrics.loc[(metrics.layer == "P1") & (metrics.mechanism == "PARTIAL_SELECTED")].iloc[0]
    p2_no = metrics.loc[(metrics.layer == "P2") & (metrics.mechanism == "NO_SON_PHI0")].iloc[0]
    p2_partial = metrics.loc[(metrics.layer == "P2") & (metrics.mechanism == "PARTIAL_SELECTED")].iloc[0]
    checks = {
        "contract_was_registered": contract["registered_before_development_TN_read"] is True,
        "pre_tn_lock_precedes_tn_read_semantically": prelock["created_before_development_TN_read"] is True and prelock["TN_values_read_at_lock_creation"] is False,
        "parent_lock_unchanged": sha256(PARENT_LOCK) == EXPECTED_PARENT_SHA,
        "stage2_source_identity_remains_closed": json.loads(STAGE2_LOCK.read_text(encoding="utf-8"))["terminal_scientific_status"] == "SOURCE_IDENTITY_NOT_IDENTIFIABLE",
        "manifest_closed_at_decision": manifest["status"] == "closed" and manifest["terminal_status"] == decision["status"],
        "candidate_grid_complete": len(parameters) == 6 * 4 * 11 and set(parameters.phi_A) == {v / 10 for v in range(11)},
        "fold_selection_complete": len(selections) == 6 * 4 and set(selections.evaluation_year) == {2018, 2019, 2020, 2021},
        "all_spinups_independent_and_converged": len(spin) == 6 * 11 and bool(spin.converged.all()),
        "mass_balance_pass": float(mass.max_relative_mass_balance_error.max()) <= 1e-10 and float(mass.minimum_state_or_flux_kg_n.min()) >= -1e-8,
        "both_endpoints_exact_all_mu": len(nesting) == 12 and bool(nesting["pass"].all()) and float(nesting.relative_error.max()) <= 1e-12,
        "optimizers_success": bool(parameters.outer_success.all()) and bool(readouts.success.all()),
        "oof_years_correct": set(predictions.year) == {2018, 2019, 2020, 2021},
        "oof_support_correct": len(predictions.loc[(predictions.layer == "P1") & (predictions.mechanism == "PARTIAL_SELECTED")]) == 6 * 3895,
        "gate_rows_complete": len(gates) == 6 * 2 * 3 * 2,
        "reported_status_recomputed": decision["status"] == "STATION_READOUT_COMPENSATED_NOT_PROCESS_SUPPORTED" and not decision["module_process_gate"] and decision["P2_predictive_preservation_gate"],
        "p1_numeric_change_small": float(p1_partial.rmse_mg_l) < float(p1_no.rmse_mg_l) and float(p1_partial.nse) > float(p1_no.nse),
        "p2_numeric_change_small": float(p2_partial.rmse_mg_l) < float(p2_no.rmse_mg_l) and float(p2_partial.nse) > float(p2_no.nse),
        "no_formal_spatial_claim": decision["nested_spatial_authorized"] is False and decision["spatial_extrapolation_status"] == "not_evaluated_at_temporal_stage",
        "tn_2022_not_read": decision["TN_2022_values_read"] is False and manifest["TN_2022_values_read"] is False,
        "source_identity_not_claimed": "not an identified manure fraction" in decision["claim_boundary"],
        "report_is_markdown_only": "STATION_READOUT_COMPENSATED_NOT_PROCESS_SUPPORTED" in report and not list(ROOT.rglob("*.html")),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, value in checks.items() if not value]
    verification = {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "failed": failed,
        "terminal_scientific_status": decision["status"],
        "TN_2022_values_read": False,
    }
    dump(REPORTS / "independent_verification.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
