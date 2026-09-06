from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(r"E:\SPARROW\5_Test\20260824_5")
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
MODELS = tuple(sorted(
    [f"S0_mu_{mu:03d}m" for mu in (12, 36, 60, 96, 144, 240)]
    + [f"S1_tau_012m_mu_{mu:03d}m" for mu in (12, 36, 60, 96, 144, 240)]
))
CANDIDATES = {
    "L1_SHARED_CATCHMENT_MEMORY_REPLACES_T1",
    "L2_Q72_ANTECEDENT_WETNESS_EFFECTIVE_LOSS",
    "L3_PATHWAY_MPR_SLOPE_PERMEABILITY",
}


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"check": name, "pass": bool(passed), "evidence": str(evidence)})

    required = [
        LOCKS / "f2_pre_observed_candidate_input_lock.json",
        REPORTS / "f2_synthetic_identifiability.json",
        REPORTS / "f2_decision.json", REPORTS / "technical_report.md",
        REPORTS / "f2_completion_audit.json",
        OUT / "f2_temporal_oof_predictions.parquet", OUT / "f2_fold_parameters.parquet",
        OUT / "f2_candidate_grid_scores.parquet", OUT / "f2_fold_selections.parquet",
        OUT / "f2_paired_simultaneous_gates.parquet", OUT / "f2_performance_metrics.parquet",
        OUT / "l1_spinup_audit.parquet", OUT / "l1_mass_balance_audit.parquet",
        OUT / "f2_parent_endpoint_nesting_audit.parquet", OUT / "f2_parent_reproduction.parquet",
    ]
    check("all_required_artifacts", all(path.exists() for path in required), [str(p) for p in required if not p.exists()])

    lock = json.loads((LOCKS / "f2_pre_observed_candidate_input_lock.json").read_text(encoding="utf-8"))
    synth = json.loads((REPORTS / "f2_synthetic_identifiability.json").read_text(encoding="utf-8"))
    decision = json.loads((REPORTS / "f2_decision.json").read_text(encoding="utf-8"))
    check("pre_observed_lock", lock["created_before_observed_candidate_fitting"], lock["aggregate_sha256"])
    check("synthetic_gate", synth["status"] == "PASS", synth["status"])
    check("synthetic_candidate_set", {r["candidate"] for r in synth["candidate_reports"]} == CANDIDATES, synth["candidate_reports"])
    check("synthetic_false_rate", all(r["false_upgrade_rate"] <= 0.05 for r in synth["candidate_reports"]), synth["candidate_reports"])
    check("synthetic_power", all(r["power"] >= 0.8 for r in synth["candidate_reports"]), synth["candidate_reports"])

    predictions = pd.read_parquet(OUT / "f2_temporal_oof_predictions.parquet")
    expected_candidates = CANDIDATES | {"GAUSSIAN_PROCESS_PARENT"}
    check("prediction_candidate_set", set(predictions.candidate) == expected_candidates, sorted(predictions.candidate.unique()))
    check("prediction_model_set", set(predictions.model_id) == set(MODELS), sorted(predictions.model_id.unique()))
    check("prediction_layers", set(predictions.layer) == {"P1", "P2"}, sorted(predictions.layer.unique()))
    check("OOF_years", set(predictions.year) == {2018, 2019, 2020, 2021}, sorted(predictions.year.unique()))
    check("no_2022_predictions", not predictions.year.eq(2022).any(), predictions.year.max())
    group_sizes = predictions.groupby(["model_id", "layer", "candidate"], observed=True).size()
    check("3895_keys_each_group", bool(group_sizes.eq(3895).all()), group_sizes.value_counts().to_dict())
    keys = ["station_key", "year", "month", "fold_id"]
    duplicates = predictions.duplicated(["model_id", "layer", "candidate", *keys]).sum()
    check("no_prediction_duplicates", duplicates == 0, duplicates)

    spin = pd.read_parquet(OUT / "l1_spinup_audit.parquet")
    mass = pd.read_parquet(OUT / "l1_mass_balance_audit.parquet")
    nesting = pd.read_parquet(OUT / "f2_parent_endpoint_nesting_audit.parquet")
    reproduction = pd.read_parquet(OUT / "f2_parent_reproduction.parquet")
    check("L1_all_spinups_converged", bool(spin.converged.all()), spin[["model_id", "cycles", "terminal_max_abs_delta_kg_n"]].to_dict("records"))
    check("L1_mass_closure", float(mass.max_relative_mass_balance_error.max()) <= 1e-10, mass.max_relative_mass_balance_error.max())
    check("L1_nonnegative", float(mass.minimum_state_or_flux_kg_n.min()) >= -1e-8, mass.minimum_state_or_flux_kg_n.min())
    check("L2_L3_parent_endpoints", bool(nesting["pass"].all()), nesting.relative_error.max())
    check("parent_reproduction", bool(reproduction["pass"].all()), reproduction.max_relative_difference.max())

    gates = pd.read_parquet(OUT / "f2_paired_simultaneous_gates.parquet")
    check("gate_rows", len(gates) == 12 * 2 * 3 * 3, len(gates))
    check("studentized_maxT_finite", bool(np.isfinite(gates[["bootstrap_standard_error", "max_t_critical_value", "simultaneous_ci95_upper"]]).all().all()), "finite")
    matrix = pd.read_parquet(OUT / "f2_candidate_decision_matrix.parquet")
    check("decision_matrix_candidate_set", set(matrix.candidate) == CANDIDATES, matrix.to_dict("records"))
    check("decision_consistency", decision["supported_candidates"] == matrix.loc[matrix.formal_gate_pass, "candidate"].tolist(), decision["supported_candidates"])
    check("registered_result", decision["status"] == "NO_REGISTERED_F2_UPGRADE_SUPPORTED", decision["status"])
    check("TN_2022_contract", decision["TN_2022_values_read"] is False and lock["TN_2022_values_read"] is False, "false")

    report_text = (REPORTS / "technical_report.md").read_text(encoding="utf-8")
    check("markdown_report", "NO_REGISTERED_F2_UPGRADE_SUPPORTED" in report_text and "2022" in report_text, len(report_text))
    status = "PASS" if all(row["pass"] for row in checks) else "FAIL"
    report = {"status": status, "checks": checks, "TN_2022_values_read": False}
    (REPORTS / "f2_independent_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise RuntimeError("F2 independent verification failed")


if __name__ == "__main__":
    main()
