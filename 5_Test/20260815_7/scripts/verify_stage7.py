from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260815_7")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    checks: dict[str, dict[str, object]] = {}
    def check(name: str, passed: bool, evidence: object) -> None:
        checks[name] = {"pass": bool(passed), "evidence": evidence}
    stage_status = {}
    for i in range(1, 7):
        audit = json.loads(Path(rf"E:\SPARROW\5_Test\20260815_{i}\reports\completion_audit.json").read_text(encoding="utf-8"))
        stage_status[str(i)] = bool(audit.get("pass"))
    check("all_parent_stages_pass", all(stage_status.values()), stage_status)
    stage1_contract = json.loads(Path(r"E:\SPARROW\5_Test\20260815_1\experiment_contract.json").read_text(encoding="utf-8"))
    stage2_contract = json.loads(Path(r"E:\SPARROW\5_Test\20260815_2\experiment_contract.json").read_text(encoding="utf-8"))
    stage4_contract = json.loads(Path(r"E:\SPARROW\5_Test\20260815_4\experiment_contract.json").read_text(encoding="utf-8"))
    stage5_registry = json.loads(Path(r"E:\SPARROW\5_Test\20260815_5\parameter_registry.json").read_text(encoding="utf-8"))
    external = json.loads(Path(r"E:\SPARROW\5_Test\20260815_1\reports\external_hydrology_diagnostics.json").read_text(encoding="utf-8"))
    check("historical_Q72_climatology_2006_2015_only", stage1_contract["hydrology_policy"]["climatology_years"] == [2006, 2015] and stage2_contract["historical_hydrology"]["1961_2005"] == "repeat_Q72_2006_2015_monthly_climatology", {"climatology": stage1_contract["hydrology_policy"]["climatology_years"], "history": stage2_contract["historical_hydrology"]})
    check("single_surplus_dynamic_entry_no_gross_double_count", stage2_contract["dynamic_diffuse_input"] == "legacy_eligible_n_surplus_kg_n" and stage2_contract["gross_terms_are_audit_components_not_additional_dynamic_inputs"] and stage2_contract["gross_terms_dynamic_mass_required"] == 0.0, {"entry": stage2_contract["dynamic_diffuse_input"], "forbidden": stage2_contract["forbidden_additional_dynamic_inputs"]})
    check("strict_M0_S0_S1_positive_input_locations", stage4_contract["positive_input_location"] == {"M0": "current_mobile", "S0": "mobile_pool", "S1": "SON_pool"} and not stage4_contract["free_SON_mobile_partition"], stage4_contract["positive_input_location"])
    m0 = pd.read_parquet(Path(r"E:\SPARROW\5_Test\20260815_3\outputs\m0_reach_month_local_n_1961_2022.parquet"), columns=["positive_legacy_eligible_n_surplus_kg_n_month", "direct_current_quick_n_input_kg_n", "quick_bypass_fraction_recomputed", "soil_contact_water_mm", "soil_overflow_to_quick_mm", "gw_recharge_mm", "m0_mobile_state_end_kg_n"])
    bypass_error = float((m0.direct_current_quick_n_input_kg_n - m0.positive_legacy_eligible_n_surplus_kg_n_month * m0.quick_bypass_fraction_recomputed).abs().max())
    contact_error = float((m0.soil_contact_water_mm - m0.soil_overflow_to_quick_mm - m0.gw_recharge_mm).abs().max())
    check("current_N_quick_bypass_and_historical_flush_separated", bypass_error <= 1e-8 and contact_error <= 1e-12 and float(m0.m0_mobile_state_end_kg_n.abs().max()) == 0.0, {"bypass_error": bypass_error, "contact_error": contact_error})
    stage4_decision = json.loads(Path(r"E:\SPARROW\5_Test\20260815_4\reports\source_structure_decision.json").read_text(encoding="utf-8"))
    stage4_cohort = Path(r"E:\SPARROW\5_Test\20260815_4\outputs\selected_source_model_cohort_state_end_2022.parquet")
    check("input_month_cohort_implemented", stage4_contract["cohort_resolution"] == "input_month" and stage4_cohort.is_file() and stage4_cohort.stat().st_size > 0, {"selected": stage4_decision["selected_source_model_id"], "cohort_bytes": stage4_cohort.stat().st_size})
    check("T1_month_units_only", stage5_registry["effective_tn_delivery_mu_month"] == [0, 12, 36, 60, 96, 144, 240] and stage5_registry["model_time_step"] == "month", stage5_registry)
    check("external_hydrology_diagnostic_only", stage1_contract["external_evidence_role"] == "diagnostic_only_not_a_gate_or_parameter_selector" and external["hydrology_external_consistency"] == "non_identifying" and external["groundwater_level_vs_q72_response_state"]["status"] == "non_identifying", {"role": stage1_contract["external_evidence_role"], "status": external["hydrology_external_consistency"]})
    lock = json.loads((REPORTS / "pre_2022_model_lock.json").read_text(encoding="utf-8"))
    science = json.loads((REPORTS / "final_scientific_decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((REPORTS / "final_model_manifest.json").read_text(encoding="utf-8"))
    check("pre2022_lock_complete", lock["selection_complete_before_locked_metrics"] and lock["locked_year"] == 2022 and lock["effective_tn_delivery_mu_month"] in [0,12,36,60,96,144,240], lock)
    locked = pd.read_parquet(OUT / "locked_2022_tn_predictions.parquet")
    check("locked_year_only", set(locked.year) == {2022}, sorted(set(locked.year)))
    check("two_models_same_keys", set(locked.model_id) == {"selected_joint_model", "M0_T0_R0_baseline"} and locked.groupby("model_id").size().nunique() == 1 and not locked.duplicated(["model_id", "station_key", "year", "month"]).any(), locked.groupby("model_id").size().to_dict())
    check("finite_nonnegative_predictions", locked.pred_tn_mg_l.notna().all() and (locked.pred_tn_mg_l >= 0).all() and locked.observed_load_proxy_kg_n.notna().all() and locked.predicted_load_kg_n.notna().all(), None)
    boot = pd.read_parquet(OUT / "locked_2022_bootstrap_distributions.parquet")
    check("locked_bootstrap_contract", len(boot) == 20000 and set(boot.block) == {"station_key", "terminal_tree_id"}, {"rows": len(boot), "blocks": sorted(set(boot.block))})
    age = json.loads((REPORTS / "locked_2022_cohort_age_summary.json").read_text(encoding="utf-8"))
    check("cohort_age_fractions", 0 <= age["fraction_memory_gt10y"] <= age["fraction_memory_gt5y"] <= age["fraction_memory_gt1y"] <= 1 and 0 <= age["fraction_pre1961_equilibrium"] <= 1, age)
    check("age_semantics_and_lower_bound", age["post1961_mean_cohort_age_month"] >= 0 and age["all_history_mean_cohort_age_lower_bound_month"] >= 0 and "not_water_age" in age["interpretation"], age["interpretation"])
    check("source_delivery_routing_answers", science["source_legacy_required"] in {"yes_development_supported", "development_supported_but_locked_not_confirmed"} and science["soil_legacy_time_scale_status"] == "upper_boundary_limited_not_point_identified" and science["independent_effective_tn_delivery_memory_required"] == "no_non_identifying_retain_T0" and science["river_retention_required"] == "not_tested_R1_not_run_without_tau_r", science)
    check("structural_noninferiority_plus_process_evidence_used", stage4_contract["advancement"] == "structural_noninferiority_plus_process_evidence" and stage4_decision["source_structure_status"] in {"S1_structurally_supported_noninferior", "S0_structurally_supported_noninferior", "source_memory_non_identifying_retain_simplest"}, stage4_decision["source_structure_status"])
    routing_decision = json.loads(Path(r"E:\SPARROW\5_Test\20260815_6\reports\routing_decision.json").read_text(encoding="utf-8"))
    check("R1_not_run_without_tau_r", routing_decision["R1_status"] == "not_run_no_reliable_tau_r" and routing_decision["new_fitted_parameter_count"] == 0, routing_decision)
    check("groundwater_nonidentifying", science["groundwater_external_evidence"] == "non_identifying", science["groundwater_external_evidence"])
    check("point_source_not_fabricated", science["point_source_tn_status"] == "missing_not_fabricated", science["point_source_tn_status"])
    check("manifest_locked", manifest["model_status"] == "frozen_after_single_use_2022_evaluation" and manifest["pre_2022_lock_sha256"] == sha256(REPORTS / "pre_2022_model_lock.json"), manifest["model_id"])
    start = json.loads((REPORTS / "parent_hashes_start.json").read_text(encoding="utf-8"))
    end = json.loads((REPORTS / "parent_hashes_end.json").read_text(encoding="utf-8"))
    check("parents_unchanged", start == end, {"files": len(start)})
    check("folder_iteration_complete", all(Path(rf"E:\SPARROW\5_Test\20260815_{i}").is_dir() for i in range(1,8)), list(range(1,8)))
    check("sparrow_runtime", Path(sys.prefix).name.lower() == "sparrow" and all(os.environ.get(k) == "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]), sys.prefix)
    failed = [k for k,v in checks.items() if not v["pass"]]
    default = lambda x: x.item() if hasattr(x, "item") else str(x)
    result = {"scenario_id": "20260815_7", "pass": not failed, "failed": failed, "checks": checks}
    (REPORTS / "completion_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=default), encoding="utf-8")
    requirement_audit = {"objective": "complete_N_Legacy_experiment_tree_with_folder_iteration", "pass": not failed, "requirements": checks, "final_model_id": manifest["model_id"], "scientific_decision": science}
    (REPORTS / "requirement_completion_audit.json").write_text(json.dumps(requirement_audit, ensure_ascii=False, indent=2, default=default), encoding="utf-8")
    print(json.dumps({"scenario_id": "20260815_7", "pass": not failed, "failed": failed, "model_id": manifest["model_id"]}, ensure_ascii=False, indent=2, default=default))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
