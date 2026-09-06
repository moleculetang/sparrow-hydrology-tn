from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(r"E:\SPARROW\5_Test\20260824_6")
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
MODELS = tuple(sorted(
    [f"S0_mu_{mu:03d}m" for mu in (12, 36, 60, 96, 144, 240)]
    + [f"S1_tau_012m_mu_{mu:03d}m" for mu in (12, 36, 60, 96, 144, 240)]
))


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    checks = []
    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"check": name, "pass": bool(passed), "evidence": str(evidence)})

    required = [
        LOCKS / "f3_pre_observed_candidate_input_lock.json",
        REPORTS / "f3_synthetic_identifiability.json", REPORTS / "f3_decision.json",
        REPORTS / "technical_report.md", REPORTS / "f3_completion_audit.json",
        OUT / "f3_temporal_oof_predictions.parquet", OUT / "f3_fold_parameters.parquet",
        OUT / "f3_paired_simultaneous_gates.parquet", OUT / "f3_performance_metrics.parquet",
        OUT / "f3_aquatic_numerical_audit.parquet", OUT / "f3_parent_reproduction.parquet",
    ]
    check("required_artifacts", all(p.exists() for p in required), [str(p) for p in required if not p.exists()])
    lock = json.loads((LOCKS / "f3_pre_observed_candidate_input_lock.json").read_text(encoding="utf-8"))
    synth = json.loads((REPORTS / "f3_synthetic_identifiability.json").read_text(encoding="utf-8"))
    decision = json.loads((REPORTS / "f3_decision.json").read_text(encoding="utf-8"))
    reports = {row["candidate"]: row for row in synth["candidate_reports"]}
    check("pre_observed_lock", lock["created_before_observed_candidate_fitting"], lock["aggregate_sha256"])
    check("residence_preflight_pass", reports["TRUE_RESIDENCE_TIME_ATTENUATION"]["pass"], reports["TRUE_RESIDENCE_TIME_ATTENUATION"])
    check("saturation_preflight_block", not reports["H1_CONCENTRATION_LIMITED_KN_GRID"]["pass"] and reports["H1_CONCENTRATION_LIMITED_KN_GRID"]["power"] < 0.8, reports["H1_CONCENTRATION_LIMITED_KN_GRID"])

    pred = pd.read_parquet(OUT / "f3_temporal_oof_predictions.parquet")
    check("candidate_set", set(pred.candidate) == {"GAUSSIAN_PROCESS_PARENT", "TRUE_RESIDENCE_TIME_ATTENUATION"}, sorted(pred.candidate.unique()))
    check("blocked_candidate_not_fit", not pred.candidate.eq("H1_CONCENTRATION_LIMITED_KN_GRID").any(), "no observed fit")
    check("model_set", set(pred.model_id) == set(MODELS), sorted(pred.model_id.unique()))
    check("OOF_years", set(pred.year) == {2018, 2019, 2020, 2021}, sorted(pred.year.unique()))
    check("no_2022", not pred.year.eq(2022).any(), pred.year.max())
    sizes = pred.groupby(["model_id", "layer", "candidate"], observed=True).size()
    check("3895_keys_per_group", bool(sizes.eq(3895).all()), sizes.value_counts().to_dict())
    check("no_duplicate_keys", not pred.duplicated(["model_id", "layer", "candidate", "station_key", "year", "month", "fold_id"]).any(), "unique")

    numerical = pd.read_parquet(OUT / "f3_aquatic_numerical_audit.parquet")
    check("KN0_exact_H1", bool(numerical.endpoint_pass.all()), numerical.relative_error.max())
    check("nonnegative", bool(numerical.nonnegative_pass.all()), numerical.minimum_output_kg_n.min())
    check("no_amplification", bool(numerical.no_amplification_pass.all()), numerical.maximum_output_to_zero_attenuation_ratio.max())
    reproduction = pd.read_parquet(OUT / "f3_parent_reproduction.parquet")
    check("parent_reproduction", bool(reproduction["pass"].all()), reproduction.max_relative_difference.max())
    gates = pd.read_parquet(OUT / "f3_paired_simultaneous_gates.parquet")
    check("gate_rows", len(gates) == 72, len(gates))
    check("maxT_finite", bool(np.isfinite(gates[["bootstrap_standard_error", "max_t_critical_value", "simultaneous_ci95_upper"]]).all().all()), "finite")
    check("decision", decision["status"] == "NO_REGISTERED_F3_UPGRADE_SUPPORTED", decision["status"])
    check("decision_candidate_boundary", decision["preflight_active_candidates"] == ["TRUE_RESIDENCE_TIME_ATTENUATION"] and decision["preflight_blocked_candidates"] == ["H1_CONCENTRATION_LIMITED_KN_GRID"], decision)
    check("TN_2022_contract", decision["TN_2022_values_read"] is False and lock["TN_2022_values_read"] is False, "false")
    report_text = (REPORTS / "technical_report.md").read_text(encoding="utf-8")
    check("markdown_report", "NO_REGISTERED_F3_UPGRADE_SUPPORTED" in report_text and "reference discharge" in report_text, len(report_text))
    status = "PASS" if all(row["pass"] for row in checks) else "FAIL"
    report = {"status": status, "checks": checks, "TN_2022_values_read": False}
    (REPORTS / "f3_independent_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise RuntimeError("F3 independent verification failed")


if __name__ == "__main__":
    main()
