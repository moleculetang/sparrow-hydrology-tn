from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW\5_Test\20260817_6")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
TEST = Path(r"E:\SPARROW\5_Test")


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(sys.prefix)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 required")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def add(rows: list[dict[str, object]], requirement: str, passed: object, evidence: object) -> None:
    rows.append({"requirement": requirement, "pass": bool(passed), "evidence": json.dumps(evidence, ensure_ascii=False, default=str)})


def verify_manifest(path: Path) -> tuple[bool, list[dict[str, object]]]:
    manifest = read_json(path)
    details = []
    passed = True
    for raw, expected in manifest.items():
        source = Path(raw)
        exists = source.exists()
        actual = sha256(source) if exists else None
        ok = exists and actual == expected
        passed = passed and ok
        details.append({"path": raw, "exists": exists, "expected": expected, "actual": actual, "pass": ok})
    return bool(passed), details


def independently_recompute_basin_t95() -> dict[str, object]:
    curves = pd.read_parquet(TEST / "20260817_2" / "outputs" / "basin_source_to_stream_curves.parquet")
    metrics = pd.read_parquet(TEST / "20260817_2" / "outputs" / "source_to_stream_tail_metrics.parquet")
    metrics = metrics.loc[metrics.scope.eq("basin_source_to_stream")].copy()
    checks = []
    for keys, block in curves.groupby(["model_id", "start_month", "weight_scheme"], sort=True):
        block = block.sort_values("lag_month")
        cdf = block.release_mass_fraction.to_numpy(float).cumsum()
        if cdf[-1] + 1e-14 < 0.95:
            lag = elapsed = year = np.nan
            status = "right_censored_200yr"
        else:
            lag = int(np.argmax(cdf >= 0.95))
            elapsed = lag + 1
            year = elapsed / 12.0
            status = "resolved"
        row = metrics.loc[
            metrics.model_id.eq(keys[0])
            & metrics.start_month.eq(keys[1])
            & metrics.weight_scheme.eq(keys[2])
        ].iloc[0]
        same_numeric = (
            (pd.isna(lag) and pd.isna(row.T95_lag_month))
            or (int(row.T95_lag_month) == lag and int(row.T95_elapsed_month) == elapsed and abs(float(row.T95_year) - year) <= 1e-12)
        )
        checks.append(bool(same_numeric and row.T95_status == status))
    return {
        "pass": bool(all(checks) and len(checks) == 16 * 12 * 2),
        "curves": len(checks),
        "definition": "CDF=cumulative release/initial injected mass; elapsed=lag+1; year=elapsed/12",
    }


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for stage in range(1, 6):
        completion = read_json(TEST / f"20260817_{stage}" / "reports" / "completion_audit.json")
        verification = read_json(TEST / f"20260817_{stage}" / "reports" / "verification.json")
        add(rows, f"stage_{stage}_completed_and_verified", completion.get("pass") and verification.get("pass"), {"completion": completion.get("pass"), "verification": verification.get("pass")})

    decision1 = read_json(TEST / "20260817_1" / "reports" / "structural_reconciliation_decision.json")
    registry = pd.read_parquet(TEST / "20260817_1" / "outputs" / "structural_adjudication_registry.parquet")
    m0 = registry.loc[registry.source_structure.eq("M0")]
    persistent = registry.loc[registry.source_structure.isin(["S0", "S1"])]
    algorithmic_required = len(m0) == 7 and not m0.reconciled_admissible.any() and persistent.reconciled_admissible.any()
    add(rows, "all_seven_M0_candidates_rechecked", len(m0) == 7 and set(m0.delivery_mu_month.astype(int)) == {0, 12, 36, 60, 96, 144, 240}, {"count": len(m0), "mu": sorted(m0.delivery_mu_month.astype(int).tolist())})
    add(rows, "source_persistence_required_is_algorithmic_not_predeclared", algorithmic_required and decision1["source_persistence_status"] == "required", {"M0_admissible": int(m0.reconciled_admissible.sum()), "persistent_admissible": int(persistent.reconciled_admissible.sum())})
    s0_ok = registry.loc[registry.source_structure.eq("S0"), "reconciled_admissible"].any()
    s1_12_ok = registry.loc[registry.model_id.str.startswith("S1_tau_012m"), "reconciled_admissible"].any()
    add(rows, "S0_vs_S1_status_generated_from_both_nonempty_sets", s0_ok and s1_12_ok and decision1["source_structure_status"] == "S0_vs_S1_12_nonidentifying", {"S0": s0_ok, "S1_12": s1_12_ok})
    add(rows, "river_and_multievidence_delivery_status_separated", decision1["river_evidence_for_positive_delivery_memory"] == "supportive_but_not_exclusive" and decision1["multievidence_delivery_memory_status"] == "positive_memory_required_under_registered_gates", decision1)
    add(rows, "eta_boundary_confounding_is_not_hard_gate", not registry.eta_boundary_confounding_is_hard_gate.any() and decision1["eta_boundary_confounding_role"] == "diagnostic_only_not_hard_gate", {"hard_gate_true_rows": int(registry.eta_boundary_confounding_is_hard_gate.sum())})

    oof = pd.read_parquet(TEST / "20260817_1" / "outputs" / "analysis_model_oof_predictions.parquet", columns=["model_id", "station_key", "year", "month"])
    unique = oof.drop_duplicates(["model_id", "station_key", "year", "month"]).groupby("model_id").size()
    add(rows, "OOF_years_and_4097_unique_keys", set(oof.year.unique()) == {2018, 2019, 2020, 2021} and (unique == 4097).all(), {"years": sorted(oof.year.unique().tolist()), "min_keys": int(unique.min()), "max_keys": int(unique.max())})

    preflight = read_json(TEST / "20260817_2" / "reports" / "legacy17_core_parent_reproduction_audit.json")
    superposition = read_json(TEST / "20260817_2" / "reports" / "linear_superposition_audit.json")
    add(rows, "new_core_reproduces_frozen_parent", preflight["pass"], {"rtol": preflight["rtol"], "atol_kg_n": preflight["atol_kg_n"]})
    add(rows, "four_linear_superposition_cases", superposition["pass"] and len(superposition["cases"]) == 4 and max(max(x["max_basin_relative_difference"], x["max_terminal_relative_difference"]) for x in superposition["cases"]) <= 1e-12, superposition["cases"])
    weights = pd.read_parquet(TEST / "20260817_2" / "outputs" / "pulse_spatial_weight_registry.parquet")
    local_semantics = pd.read_parquet(TEST / "20260817_2" / "outputs" / "local_source_release_kernels.parquet", columns=["pulse_input_semantics"]).pulse_input_semantics.unique().tolist()
    add(rows, "Pulse_B_is_one_kg_post_ledger_monthly_mass", len(local_semantics) == 1 and "monthly post-ledger" in local_semantics[0] and weights.pulse_b_unit_semantics.str.contains("monthly post-ledger").all(), {"kernel": local_semantics, "registry": weights.pulse_b_unit_semantics.unique().tolist()})
    t95 = independently_recompute_basin_t95()
    add(rows, "CDF_initial_mass_and_elapsed_time_convention_independently_recomputed", t95["pass"], t95)
    pulse_audit = read_json(TEST / "20260817_2" / "reports" / "pulse_mass_balance_audit.json")
    add(rows, "right_censor_and_CDF_denominator_registered", pulse_audit["cdf_denominator"] == "initial injected mass or water; never horizon-released mass" and pulse_audit["right_censor_supported"], pulse_audit)
    n_q = pd.read_parquet(TEST / "20260817_5" / "outputs" / "matched_basin_N_vs_Q_tail_metrics.parquet")
    add(rows, "N_and_Q_use_identical_spatial_weights", n_q.spatial_weight_match.all() and set(n_q.weight_scheme) == {"n_source_weighted", "area_weighted"}, {"rows": len(n_q), "schemes": sorted(n_q.weight_scheme.unique().tolist())})

    age_contract = read_json(TEST / "20260817_3" / "reports" / "age_bin_contract.json")
    expected_logic = ["age_month <= 12", "12 < age_month <= 60", "60 < age_month <= 120", "120 < age_month <= 240", "240 < age_month <= 600", "age_month > 600"]
    add(rows, "age_bins_complete_nonoverlapping_month_cutpoints", [x["logic"] for x in age_contract["bins"]] == expected_logic and age_contract["coverage"].startswith("complete_nonoverlapping"), age_contract["bins"])
    add(rows, "pre1961_is_separate_right_censored_cohort", "right-censored" in age_contract["pre1961"], age_contract["pre1961"])
    no_refit3 = read_json(TEST / "20260817_3" / "reports" / "locked_2022_no_refit_audit.json")
    no_refit4 = read_json(TEST / "20260817_4" / "reports" / "time_boundary_and_no_refit_audit.json")
    locked = pd.read_parquet(TEST / "20260817_4" / "outputs" / "locked_2022_predictions_frozen_readout.parquet", columns=["locked_2022_refit", "readout_fit_support"])
    add(rows, "2022_never_refits_observation_parameters", no_refit3["locked_2022_rows_used_for_fit"] == 0 and no_refit4["locked_2022_refit_calls"] == 0 and not locked.locked_2022_refit.any(), {"stage3": no_refit3, "stage4": no_refit4})
    metric = read_json(TEST / "20260817_4" / "reports" / "performance_metric_contract.json")
    add(rows, "parent_log_metric_inherited_and_duplicate_R2_removed", metric["formula"] == "ln(1 + TN_mg_L)" and metric["offset"] == 1.0 and metric["r_squared_removed_because_1_minus_SSE_over_SST_duplicates_NSE"], metric)
    add(rows, "KGE2012_formula_and_eligibility_locked", metric["kge2012_formula"].startswith("1-sqrt") and metric["kge_predicted_mean_eligibility"] == ">1e-12", {"formula": metric["kge2012_formula"], "eligibility": metric["kge_predicted_mean_eligibility"]})

    history_file = TEST / "20260817_3" / "outputs" / "historical_routed_n_age_1961_2022.parquet"
    metadata = pq.ParquetFile(history_file).metadata
    add(rows, "historical_age_output_shape", metadata.num_rows == 16 * 230 * 744, {"actual_rows": metadata.num_rows, "expected_rows": 16 * 230 * 744})
    terminal = pd.read_parquet(TEST / "20260817_2" / "outputs" / "terminal_tree_source_to_stream_curves.parquet", columns=["terminal_tree_id"])
    add(rows, "topology_has_14_terminal_trees", terminal.terminal_tree_id.nunique() == 14, {"terminal_trees": int(terminal.terminal_tree_id.nunique())})

    manifest_results = []
    for stage in range(1, 6):
        path = TEST / f"20260817_{stage}" / "upstream_manifest.json"
        ok, detail = verify_manifest(path)
        manifest_results.append({"stage": stage, "pass": ok, "files": len(detail)})
        add(rows, f"stage_{stage}_upstream_hashes_unchanged", ok, {"files": len(detail), "failures": [x for x in detail if not x["pass"]]})

    audit = pd.DataFrame(rows)
    audit.to_parquet(OUT / "requirement_by_requirement_audit.parquet", index=False)
    all_pass = bool(audit["pass"].all())
    decision5 = read_json(TEST / "20260817_5" / "reports" / "integrated_scientific_decision.json")
    final = {
        "scenario_id": "20260817_6",
        "status": "complete" if all_pass else "STOP_FINAL_AUDIT_FAILED",
        "pass": all_pass,
        "hard_checks": len(audit),
        "failed_checks": audit.loc[~audit["pass"], "requirement"].tolist(),
        "scientific_decision": decision5,
        "audit_core_imported": False,
        "eta_boundary_confounding_role": "diagnostic_only_not_hard_gate",
        "Pulse_B_unit": "1 kg N in one monthly post-ledger timestep per reach",
        "OOF_evaluation_years": [2018, 2019, 2020, 2021],
        "development_support_period": [2016, 2021],
        "retrospective_locked_year": 2022,
    }
    dump(REPORTS / "final_completion_manifest.json", final)
    dump(REPORTS / "requirement_by_requirement_audit.json", {"pass": all_pass, "checks": rows})
    if not all_pass:
        raise RuntimeError(f"STOP_FINAL_AUDIT_FAILED: {final['failed_checks']}")
    canonical_outputs = [
        TEST / "20260817_1" / "reports" / "structural_reconciliation_decision.json",
        TEST / "20260817_2" / "outputs" / "source_to_stream_tail_metrics.parquet",
        TEST / "20260817_3" / "outputs" / "historical_routed_n_age_1961_2022.parquet",
        TEST / "20260817_4" / "outputs" / "tn_performance_metrics_16models.parquet",
        TEST / "20260817_5" / "reports" / "integrated_scientific_decision.json",
        REPORTS / "final_completion_manifest.json",
    ]
    lock = {
        "scenario_id": "20260817_6",
        "status": "frozen_complete",
        "canonical_output_sha256": {str(path): sha256(path) for path in canonical_outputs},
        "upstream_manifest_audits": manifest_results,
        "scientific_claim_boundary": "broad structural ensemble, no uniquely identified long-tail time constant, no hydrologic water-age claim",
    }
    dump(ROOT / "final_lock.json", lock)
    completion = {"scenario_id": "20260817_6", "pass": True, "hard_checks": len(audit), "failed_checks": 0, "core_imported": False}
    dump(REPORTS / "completion_audit.json", completion)
    print(json.dumps({"completion": completion, "final": final}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
