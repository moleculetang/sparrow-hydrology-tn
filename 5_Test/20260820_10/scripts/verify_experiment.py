from __future__ import annotations

from datetime import datetime
import json

import numpy as np
import pandas as pd

from hleg_shared import (
    OUT,
    REPORTS,
    authoritative_paths,
    dump_json,
    hash_manifest,
    require_runtime,
    sha256,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    require_runtime()
    checks: list[dict[str, object]] = []

    def check(requirement: str, passed: bool, evidence: str, hard: bool = True) -> None:
        checks.append({"requirement": requirement, "pass": bool(passed), "hard": hard, "evidence": evidence})

    start_hashes = read_json(REPORTS / "parent_hashes_start.json")
    end_hashes = hash_manifest(authoritative_paths())
    dump_json(REPORTS / "parent_hashes_end.json", end_hashes)
    check("frozen_parent_files_unchanged", start_hashes == end_hashes, "parent_hashes_start.json vs parent_hashes_end.json")

    stage0 = read_json(REPORTS / "stage0_preflight.json")
    stage1 = read_json(REPORTS / "stage1_mechanism_audit.json")
    hydro = read_json(REPORTS / "q72_hydrologic_state_audit.json")
    temporal = read_json(REPORTS / "temporal_evidence_summary.json")
    spatial = read_json(REPORTS / "nested_spatial_evidence_summary.json")
    decision = read_json(REPORTS / "final_decision.json")
    lock1 = read_json(REPORTS / "development_mechanism_lock.json")
    lock2 = read_json(REPORTS / "full_development_parameter_lock.json")
    access = read_json(REPORTS / "locked_data_access_contract.json")
    a0 = read_json(REPORTS / "a0_independence_contract.json")

    check("stage0_parent_preflight_PASS", stage0["status"] == "PASS" and stage0["all_12_OOF_exactly_reproduced"], "stage0_preflight.json")
    check("stage1_mechanism_audit_PASS", stage1["status"] == "PASS", "stage1_mechanism_audit.json")
    check("Q72_only_state_scale_valid", hydro["h_scale_min"] > 0 and hydro["reach_count"] == 230, "q72_hydrologic_state_audit.json")
    check("historical_H_exactly_zero", hydro["pre2006_H_max_abs"] == 0, "q72_hydrologic_state_audit.json")
    check("Q72_state_recurrence_identity", hydro["g_pre_recurrence_max_abs_mm_excluding_registered_climatology_wrap_and_2006_transition"] <= 1e-12, "q72_hydrologic_state_audit.json")
    check("all_GW_months_have_release_opportunity", sum(hydro["zero_gw_discharge"].values()) == 0, "q72_hydrologic_state_audit.json")
    check("development_TN_excludes_2022", access["development_TN_years"] == [2016, 2017, 2018, 2019, 2020, 2021] and access["TN_2022_rows_materialized"] == 0, "locked_data_access_contract.json")

    parent_reproduction = pd.read_parquet(OUT / "parent_reproduction_audit.parquet")
    engineering = pd.read_parquet(OUT / "candidate_engineering_audit.parquet")
    synthetic = pd.read_parquet(OUT / "synthetic_cohort_semantics_tests.parquet")
    cohort = pd.read_parquet(OUT / "candidate_cohort_diagnostic_summary.parquet")
    check("all_12_parent_routed_and_OOF_reproduced", len(parent_reproduction) == 12 and parent_reproduction.route_pass.all() and parent_reproduction.oof_pass.all(), "parent_reproduction_audit.parquet")
    check("beta0_exact_parent", len(engineering) == 24 and (engineering.beta0_parent_max_abs_kg_n <= 1e-6).all(), "candidate_engineering_audit.parquet")
    check("cohort_mass_nonnegative", engineering.minimum_state_kg_n.min() >= -1e-6, "candidate_engineering_audit.parquet")
    check("synthetic_semantics_100_percent", synthetic["pass"].all(), "synthetic_cohort_semantics_tests.parquet")
    age_nonzero = cohort.loc[cohort.mechanism.eq("HLEG_AGE") & cohort.beta_h.ne(0)]
    check("real_state_semantics_at_least_99_percent", (age_nonzero.semantics_direction_fraction >= 0.99).all(), "candidate_cohort_diagnostic_summary.parquet")
    parent_age = cohort.loc[cohort.mechanism.eq("HLEG_AGE") & cohort.beta_h.eq(0)]
    check("open_tail_below_0_5_percent", parent_age.mean_tail_stock_fraction.max() <= 0.005, "candidate_cohort_diagnostic_summary.parquet")

    oof = pd.read_parquet(OUT / "temporal_oof_predictions.parquet")
    oof_counts = oof.groupby(["model_id", "mechanism", "layer"]).size()
    check("OOF_has_4097_keys_per_model_mechanism_layer", len(oof_counts) == 12 * 3 * 2 and oof_counts.eq(4097).all(), "temporal_oof_predictions.parquet")
    check("OOF_years_are_2018_2021", sorted(map(int, oof.year.unique())) == [2018, 2019, 2020, 2021], "temporal_oof_predictions.parquet")
    check("temporal_summary_never_materialized_2022", temporal["TN_2022_rows_materialized"] == 0, "temporal_evidence_summary.json")
    check("spatial_summary_never_materialized_2022", spatial["TN_2022_rows_materialized"] == 0, "nested_spatial_evidence_summary.json")

    nested = pd.read_parquet(OUT / "nested_spatial_predictions.parquet")
    nested_keys = ["model_id", "mechanism", "spatial_scheme", "station_key", "year", "month", "fold_id"]
    check("nested_predictions_have_unique_test_keys", not nested.duplicated(nested_keys).any(), "nested_spatial_predictions.parquet")
    nested_params = pd.read_parquet(OUT / "nested_spatial_readout_parameters.parquet")
    check("nested_readouts_have_no_eta_boundary_failures", not nested_params.eta_boundary.any(), "nested_spatial_readout_parameters.parquet")
    signflip = pd.read_parquet(OUT / "terminal_tree_signflip_sensitivity.parquet")
    check("eight_tree_256_signflip_saved", signflip.tree_count.eq(8).all() and signflip.enumerations.eq(256).all(), "terminal_tree_signflip_sensitivity.parquet", hard=False)

    check("development_mechanism_locked_before_2022", lock1["TN_2022_rows_materialized_before_lock"] == 0, "development_mechanism_lock.json")
    check("full_parameters_locked_before_2022", lock2["TN_2022_rows_materialized_before_lock"] == 0, "full_development_parameter_lock.json")
    check("registered_decision_retains_parent", lock1["selected_mechanism"] == "PARENT" and decision["status"] == "PARENT_RETAINED", "development_mechanism_lock.json + final_decision.json")
    full_params = pd.read_parquet(OUT / "full_development_readout_parameters.parquet")
    check("full_development_refit_has_12_models_and_2_layers", len(full_params) == 24 and full_params.model_id.nunique() == 12 and set(full_params.layer) == {"P1", "P2"}, "full_development_readout_parameters.parquet")

    retrospective = pd.read_parquet(OUT / "locked_2022_retrospective_predictions.parquet")
    check("locked_2022_rows_complete", len(retrospective) == 1332 * 12 * 2 and retrospective.year.eq(2022).all(), "locked_2022_retrospective_predictions.parquet")
    check("locked_2022_role_is_retrospective", decision["locked_2022_role"] == "locked_2022_retrospective_temporal_check", "final_decision.json")
    lock_time = max(datetime.fromisoformat(lock1["written_at"]).timestamp(), datetime.fromisoformat(lock2["written_at"]).timestamp())
    retrospective_mtime = (OUT / "locked_2022_retrospective_predictions.parquet").stat().st_mtime
    check("lock_timestamps_precede_2022_prediction_output", lock_time <= retrospective_mtime + 1.0, "lock written_at vs retrospective parquet mtime")

    metrics = pd.read_parquet(OUT / "final_performance_metrics.parquet")
    ensemble_pooled = metrics.loc[metrics.model_id.eq("formal_12_member_mean") & metrics.scope.eq("pooled")]
    check("final_metrics_cover_OOF_and_2022_P1_P2", len(ensemble_pooled) == 4 and set(ensemble_pooled.evaluation) == {"development_OOF_2018_2021", "locked_2022_retrospective"} and set(ensemble_pooled.layer) == {"P1", "P2"}, "final_performance_metrics.parquet")
    check("final_primary_metrics_are_finite", np.isfinite(ensemble_pooled[["rmse_log1p", "rmse_raw_mg_l", "raw_nse", "pearson_r2"]].to_numpy(float)).all(), "final_performance_metrics.parquet")
    check("A0_remains_independent", not a0["A0_used_for_HLEG_selection"] and a0["HLEG_primary_forcing"] == "annual_N_divided_by_12", "a0_independence_contract.json")

    report_path = REPORTS / "technical_report.md"
    report = report_path.read_text(encoding="utf-8")
    required_sections = ["## 技术摘要", "## 固定T1", "## 生产层拟合效果", "## 年龄结构", "## 验证与稳健性", "## 结论边界与下一步"]
    check("Markdown_technical_report_complete", all(section in report for section in required_sections) and len(report) > 3000, "technical_report.md")

    frame = pd.DataFrame(checks)
    frame.to_parquet(OUT / "requirement_by_requirement_audit.parquet", index=False)
    hard_failed = frame.loc[frame.hard & ~frame["pass"], "requirement"].tolist()
    status = "PASS" if not hard_failed else "FAIL"
    audit = {
        "scenario_id": "20260820_10",
        "status": status,
        "hard_check_count": int(frame.hard.sum()),
        "advisory_check_count": int((~frame.hard).sum()),
        "failed_hard_checks": hard_failed,
        "parent_files_modified": start_hashes != end_hashes,
        "selected_mechanism": decision["selected_mechanism"],
        "2022_role": decision["locked_2022_role"],
        "technical_report_sha256": sha256(report_path),
    }
    dump_json(REPORTS / "completion_audit.json", audit)
    dump_json(REPORTS / "requirement_by_requirement_audit.json", {"checks": checks, **audit})
    dump_json(REPORTS.parent / "final_lock.json", {
        "scenario_id": "20260820_10",
        "status": "frozen_complete" if status == "PASS" else "verification_failed",
        "selected_mechanism": decision["selected_mechanism"],
        "hard_check_count": audit["hard_check_count"],
        "failed_hard_checks": hard_failed,
        "authoritative_outputs": {
            str(path): sha256(path)
            for path in (
                REPORTS / "development_mechanism_lock.json",
                REPORTS / "full_development_parameter_lock.json",
                REPORTS / "final_decision.json",
                REPORTS / "technical_report.md",
                OUT / "final_performance_metrics.parquet",
                OUT / "requirement_by_requirement_audit.parquet",
            )
        },
    })
    if hard_failed:
        raise RuntimeError(f"verification failed: {hard_failed}")


if __name__ == "__main__":
    main()
