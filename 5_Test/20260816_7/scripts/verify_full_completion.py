from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260816_7")
S15_1 = Path(r"E:\SPARROW\5_Test\20260815_1")
S15_2 = Path(r"E:\SPARROW\5_Test\20260815_2")
S16 = Path(r"E:\SPARROW\5_Test")
REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_native(value):
    if isinstance(value, dict):
        return {str(key): as_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    evidence: dict[str, dict] = {}

    def check(name: str, passed: bool, detail, requirement: str) -> None:
        evidence[name] = {
            "pass": bool(passed),
            "requirement": requirement,
            "evidence": as_native(detail),
        }

    registry = pd.read_csv(S16 / "20260816_1" / "outputs" / "candidate_registry_63.csv")
    expected_tau = {12, 36, 60, 96, 144, 240, 480}
    expected_mu = {0, 12, 36, 60, 96, 144, 240}
    structure_counts = registry.source_structure.value_counts().to_dict()
    s1_pairs = set(zip(registry.loc[registry.source_structure.eq("S1"), "soil_tau_month"].astype(int), registry.loc[registry.source_structure.eq("S1"), "delivery_mu_month"].astype(int)))
    check(
        "candidate_registry_exact_63",
        len(registry) == 63 and registry.model_id.nunique() == 63 and structure_counts == {"S1": 49, "M0": 7, "S0": 7} and s1_pairs == {(tau, mu) for tau in expected_tau for mu in expected_mu},
        {"rows": len(registry), "unique_models": registry.model_id.nunique(), "structure_counts": structure_counts, "s1_pair_count": len(s1_pairs)},
        "Exactly 49 S1, 7 M0 and 7 S0 fixed candidates; no grid expansion.",
    )
    expected_operator = np.where(registry.delivery_mu_month.eq(0), "T0_rho_0.85", "T1_rho_mu_over_1_plus_mu")
    check(
        "T0_T1_registry_mutually_exclusive",
        bool((registry.transport_operator.to_numpy() == expected_operator).all() and registry.T0_T1_mutually_exclusive.all()),
        {"mu_zero_rows": int(registry.delivery_mu_month.eq(0).sum()), "all_registry_flags": bool(registry.T0_T1_mutually_exclusive.all())},
        "T0 is used only for mu=0 and T1 replaces rather than serializes T0 for mu>0.",
    )

    annual = pd.read_parquet(S15_2 / "outputs" / "reach_year_n_ledger_1961_2022.parquet")
    monthly = pd.read_parquet(S15_2 / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet")
    climatology = pd.read_parquet(S15_1 / "outputs" / "q72_monthly_climatology_2006_2015.parquet")
    annual_identity = annual.fertilizer_kg_n + annual.manure_kg_n + annual.cropland_bnf_kg_n + annual.atmospheric_deposition_kg_n - annual.crop_removal_kg_n
    annual_identity_error = float(np.max(np.abs(annual_identity - annual.legacy_eligible_n_surplus_kg_n)))
    check(
        "single_surplus_dynamic_entry",
        annual_identity_error <= 1e-9 and float(np.abs(annual.gross_terms_dynamic_entry_kg_n).sum()) == 0.0 and set(annual.diffuse_dynamic_entry_field) == {"legacy_eligible_n_surplus_kg_n"},
        {"surplus_identity_max_abs_kg_n": annual_identity_error, "gross_terms_dynamic_entry_sum_kg_n": float(np.abs(annual.gross_terms_dynamic_entry_kg_n).sum()), "dynamic_entry_fields": sorted(annual.diffuse_dynamic_entry_field.unique())},
        "Gross fertilizer/manure/BNF/deposition/removal terms audit the surplus but never re-enter the dynamics.",
    )
    check(
        "point_source_not_fabricated_or_soil_eligible",
        bool((~annual.point_source_soil_legacy_eligible.astype(bool)).all() and annual.point_source_tn_kg_n_year.isna().all()),
        {"soil_eligible_rows": int(annual.point_source_soil_legacy_eligible.astype(bool).sum()), "non_null_point_tn_rows": int(annual.point_source_tn_kg_n_year.notna().sum()), "statuses": sorted(annual.point_source_tn_status.astype(str).unique())},
        "Unavailable annual point-source TN stays missing and never enters soil legacy.",
    )
    month_to_annual = {
        "fertilizer_kg_n_month": "fertilizer_kg_n",
        "manure_kg_n_month": "manure_kg_n",
        "cropland_bnf_kg_n_month": "cropland_bnf_kg_n",
        "atmospheric_deposition_kg_n_month": "atmospheric_deposition_kg_n",
        "crop_removal_kg_n_month": "crop_removal_kg_n",
        "legacy_eligible_n_surplus_kg_n_month": "legacy_eligible_n_surplus_kg_n",
    }
    annual_indexed = annual.set_index(["reach_id", "year"])
    monthly_sum = monthly.groupby(["reach_id", "year"], sort=True)[list(month_to_annual)].sum()
    closure = {}
    uniformity = {}
    keyed_monthly = monthly.set_index(["reach_id", "year"])
    for monthly_name, annual_name in month_to_annual.items():
        closure[monthly_name] = float(np.max(np.abs(monthly_sum[monthly_name] - annual_indexed.loc[monthly_sum.index, annual_name])))
        annual_values = annual_indexed.loc[keyed_monthly.index, annual_name].to_numpy(float) / 12.0
        uniformity[monthly_name] = float(np.max(np.abs(keyed_monthly[monthly_name].to_numpy(float) - annual_values)))
    check(
        "annual_to_monthly_uniform_and_closed",
        max(closure.values()) <= 1e-5 and max(uniformity.values()) <= 1e-6,
        {"annual_sum_max_abs_errors_kg_n": closure, "uniform_month_max_abs_errors_kg_n": uniformity},
        "Every annual N ledger field is divided by 12 and monthly sums recover the annual ledger.",
    )
    positive_error = float(np.max(np.abs(monthly.positive_legacy_eligible_n_surplus_kg_n_month - monthly.legacy_eligible_n_surplus_kg_n_month.clip(lower=0))))
    negative_error = float(np.max(np.abs(monthly.negative_legacy_eligible_n_surplus_kg_n_month - (-monthly.legacy_eligible_n_surplus_kg_n_month).clip(lower=0))))
    bypass_expected = np.divide(monthly.quick_generated_mm, monthly.positive_input_mm, out=np.zeros(len(monthly), dtype=float), where=monthly.positive_input_mm.to_numpy(float) > 1e-12)
    bypass_expected = np.clip(bypass_expected, 0.0, 1.0)
    bypass_error = float(np.max(np.abs(bypass_expected - monthly.quick_bypass_fraction)))
    contact_error = float(np.max(np.abs(monthly.soil_contact_water_mm - monthly.soil_overflow_to_quick_mm - monthly.gw_recharge_mm)))
    local_q_error = float(np.max(np.abs(monthly.q_local_total_mm - monthly.quick_release_mm - monthly.gw_discharge_mm)))
    check(
        "water_partition_and_surplus_identities",
        max(positive_error, negative_error, bypass_error, contact_error, local_q_error) <= 1e-9,
        {"positive_error": positive_error, "negative_error": negative_error, "quick_bypass_error": bypass_error, "soil_contact_error_mm": contact_error, "local_q_error_mm": local_q_error},
        "Current positive N bypass uses quick_generated/positive_input; historical soil contact water is overflow+recharge; q_local_total is release, not generated water.",
    )
    historical = monthly.loc[monthly.year.le(2005)].copy()
    hydro_fields = ["positive_input_mm", "quick_generated_mm", "soil_overflow_to_quick_mm", "gw_recharge_mm", "gw_discharge_mm", "gw_response_state_end_mm", "source_water_capacity_mm", "catchment_area_km2", "soil_contact_water_mm", "quick_bypass_fraction"]
    comparison = historical.merge(climatology[["reach_id", "month", *hydro_fields]], on=["reach_id", "month"], suffixes=("", "_clim"), validate="many_to_one")
    climatology_errors = {field: float(np.max(np.abs(comparison[field] - comparison[f"{field}_clim"]))) for field in hydro_fields}
    check(
        "historical_hydrology_uses_only_2006_2015_climatology",
        len(climatology) == 230 * 12 and set(climatology.climatology_start_year) == {2006} and set(climatology.climatology_end_year) == {2015} and max(climatology_errors.values()) <= 1e-12,
        {"climatology_rows": len(climatology), "start_years": sorted(climatology.climatology_start_year.unique()), "end_years": sorted(climatology.climatology_end_year.unique()), "historical_rows": len(historical), "field_max_abs_errors": climatology_errors},
        "1961–2005 hydrology repeats the 2006–2015 Q72 climatology, with no 2016–2018 look-ahead.",
    )

    core_source = (S16 / "20260816_1" / "scripts" / "legacy16_core.py").read_text(encoding="utf-8")
    stage3_source = (S16 / "20260816_3" / "scripts" / "run_stage3.py").read_text(encoding="utf-8")
    stage4_source = (S16 / "20260816_4" / "scripts" / "run_stage4.py").read_text(encoding="utf-8")
    stage5_source = (S16 / "20260816_5" / "scripts" / "run_stage5.py").read_text(encoding="utf-8")
    stage7_source = (ROOT / "scripts" / "run_stage7.py").read_text(encoding="utf-8")
    source_tokens = {
        "quick_bypass_separate": "quick_generated = arrays[\"quick_generated_mm\"]" in core_source and "contact = overflow + recharge" in core_source,
        "T1_replaces_T0": "rho_gw = B_RHO if mu_t == 0 else mu_t / (1.0 + mu_t)" in core_source,
        "water_gated_gw_release": "arrays[\"gw_discharge_mm\"][t] > WATER_EPS" in core_source,
        "eta_fit_train_only": "def fit_eta(train:" in stage4_source and "least_squares" in stage4_source,
        "same_aquifer_pairing": "for _, aquifer_group in valid.groupby(\"aquifer_id\")" in stage5_source,
        "prelock_before_2022_read": stage7_source.index('dump_json(REPORTS / "pre_2022_legacy_lock.json"') < stage7_source.index("observations = pd.read_parquet(OBS_PATH)"),
    }
    check(
        "executable_structure_matches_contract",
        all(source_tokens.values()),
        source_tokens,
        "The executable operator separates new-N bypass from historical-soil flushing, replaces T0 with T1, water-gates delivery, and keeps the 2022 read after prelock.",
    )

    hydro_contract = read_json(S16 / "20260816_3" / "experiment_contract.json")
    hydro_decision = read_json(S16 / "20260816_3" / "reports" / "hydrogeo_ttd_decision.json")
    hydro_table = pd.read_csv(S16 / "20260816_3" / "reports" / "delivery_mu_hydrogeo_plausibility.csv")
    hydro_errors = {}
    for method in ("local_gradient", "path_gradient"):
        expected = (hydro_table.kernel_mean_month < hydro_decision["basin_ttd"][method]["p10_month"] / 10.0) | (hydro_table.kernel_mean_month > hydro_decision["basin_ttd"][method]["p90_month"] * 10.0)
        hydro_errors[method] = int((expected != hydro_table[f"{method}_gross_contradiction"]).sum())
    expected_hard = hydro_table.local_gradient_gross_contradiction & hydro_table.path_gradient_gross_contradiction
    check(
        "hydrogeo_10x_rule_documented_and_recomputed",
        hydro_contract["documentation_sync"]["threshold_changed"] is False and sum(hydro_errors.values()) == 0 and int((expected_hard != hydro_table.hydrogeo_hard_contradiction).sum()) == 0 and "gis[\"p10_month\"] / 10.0" in stage3_source and "gis[\"p90_month\"] * 10.0" in stage3_source,
        {"documentation_status": hydro_contract["documentation_sync"]["status"], "per_gradient_mismatches": hydro_errors, "joint_mismatches": int((expected_hard != hydro_table.hydrogeo_hard_contradiction).sum()), "T1_plus_T0_serial_lag_added": hydro_decision["T1_plus_T0_serial_lag_added"]},
        "The already-executed 10x GIS gross-contradiction threshold is now explicit, unchanged, and hard-excludes only when both gradients agree.",
    )

    engineering = read_json(S16 / "20260816_4" / "reports" / "candidate_engineering_and_mass_audit.json")
    engineering_pass = {model_id: bool(item["spinup"]["converged"] and item["minimum_state_or_flux_kg_n"] >= -1e-9 and item["max_relative_mass_balance_error"] <= 1e-12 and item["T0_T1_mutually_exclusive"]) for model_id, item in engineering.items()}
    check(
        "all_63_engineering_and_mass_gates",
        len(engineering_pass) == 63 and all(engineering_pass.values()),
        {"candidates": len(engineering_pass), "failures": sorted(model_id for model_id, passed in engineering_pass.items() if not passed), "maximum_relative_error": max(item["max_relative_mass_balance_error"] for item in engineering.values()), "minimum_state_or_flux_kg_n": min(item["minimum_state_or_flux_kg_n"] for item in engineering.values())},
        "All candidates converge, remain nonnegative, close N mass, and keep T0/T1 mutually exclusive.",
    )
    eta = pd.read_parquet(S16 / "20260816_4" / "outputs" / "candidate_fold_eta_parameters.parquet")
    oof = pd.read_parquet(S16 / "20260816_4" / "outputs" / "candidate_oof_predictions_2018_2021.parquet")
    eta_counts = eta.groupby("model_id").size()
    eta_joined = oof.merge(eta[["model_id", "fold_id", "eta_quick", "eta_gw"]], on=["model_id", "fold_id"], suffixes=("", "_registered"), validate="many_to_one")
    eta_match = max(float(np.max(np.abs(eta_joined.eta_quick - eta_joined.eta_quick_registered))), float(np.max(np.abs(eta_joined.eta_gw - eta_joined.eta_gw_registered))))
    check(
        "eta_refit_per_candidate_fold",
        len(eta) == 63 * 4 and eta_counts.eq(4).all() and eta.success.all() and eta[["eta_quick", "eta_gw"]].min().min() >= 0 and eta[["eta_quick", "eta_gw"]].max().max() <= 1 and eta_match <= 1e-12,
        {"rows": len(eta), "models": eta.model_id.nunique(), "folds": eta.fold_id.nunique(), "all_success": bool(eta.success.all()), "eta_range": [float(eta[["eta_quick", "eta_gw"]].min().min()), float(eta[["eta_quick", "eta_gw"]].max().max())], "oof_eta_max_abs_mismatch": eta_match, "confounded_candidate_folds": int(eta.delivery_efficiency_confounded.sum())},
        "Two basin-wide nuisance efficiencies are refit with the same algorithm in every candidate and fold and are stored for confounding diagnosis.",
    )
    oof_counts = oof.groupby("model_id").size()
    check(
        "OOF_keys_complete_and_2022_excluded",
        len(oof) == 63 * 4097 and oof_counts.eq(4097).all() and not oof.duplicated(["model_id", "station_key", "year", "month"]).any() and set(oof.year.astype(int)) == {2018, 2019, 2020, 2021},
        {"rows": len(oof), "models": oof.model_id.nunique(), "rows_per_model_min": int(oof_counts.min()), "rows_per_model_max": int(oof_counts.max()), "years": sorted(oof.year.astype(int).unique()), "duplicate_keys": int(oof.duplicated(["model_id", "station_key", "year", "month"]).sum())},
        "Every candidate has the same 4,097 development OOF keys and no 2022 observation participates.",
    )
    gw_panel = pd.read_parquet(S16 / "20260816_4" / "outputs" / "groundwater_aquifer_decade_panel.parquet")
    check(
        "groundwater_context_independent_operator",
        not gw_panel.duplicated(["model_id", "aquifer_id", "decade"]).any() and (~gw_panel.eta_gw_included.astype(bool)).all() and gw_panel.model_id.nunique() == 63,
        {"rows": len(gw_panel), "models": gw_panel.model_id.nunique(), "aquifers": gw_panel.aquifer_id.nunique(), "decades": sorted(gw_panel.decade.unique()), "eta_included_rows": int(gw_panel.eta_gw_included.astype(bool).sum()), "same_aquifer_pairing_source_verified": source_tokens["same_aquifer_pairing"]},
        "GW-N context uses aquifer×decade units, same-aquifer temporal pairing, and a formal pattern that excludes river-fitted eta_gw.",
    )

    bootstrap = pd.read_parquet(S16 / "20260816_5" / "outputs" / "candidate_bootstrap_distributions.parquet")
    saved_ci = pd.read_csv(S16 / "20260816_5" / "reports" / "candidate_bootstrap_ci.csv").set_index("model_id")
    bootstrap_counts = bootstrap.groupby(["model_id", "block"]).size()
    ci_differences = []
    for (model_id, block), values in bootstrap.groupby(["model_id", "block"], sort=True):
        low, high = np.percentile(values.delta_log_rmse.to_numpy(float), [2.5, 97.5])
        label = "station" if block == "station_key" else "terminal_tree"
        ci_differences.extend([abs(low - saved_ci.loc[model_id, f"{label}_ci_lower"]), abs(high - saved_ci.loc[model_id, f"{label}_ci_upper"]), abs(values.delta_log_rmse.mean() - saved_ci.loc[model_id, f"{label}_bootstrap_mean"])])
    check(
        "bootstrap_full_distribution_and_CI",
        len(bootstrap) == 63 * 2 * 10000 and bootstrap_counts.eq(10000).all() and max(ci_differences) <= 1e-12 and set(bootstrap.block) == {"station_key", "terminal_tree_id"},
        {"rows": len(bootstrap), "groups": len(bootstrap_counts), "replicates_per_group_min": int(bootstrap_counts.min()), "replicates_per_group_max": int(bootstrap_counts.max()), "maximum_saved_CI_or_mean_difference": max(ci_differences)},
        "Station and terminal-tree block bootstraps each retain all 10,000 seeded replicates and reproduce saved percentile CIs.",
    )
    adjudication = pd.read_csv(S16 / "20260816_5" / "reports" / "progressive_candidate_adjudication.csv")
    decision5 = read_json(S16 / "20260816_5" / "reports" / "legacy_attribution_decision.json")
    recomputed_final = adjudication.engineering_pass.astype(bool) & adjudication.soil_structural_admissible.astype(bool) & adjudication.hydrogeo_pass.astype(bool) & adjudication.river_noninferior.astype(bool)
    admissible_from_rows = set(adjudication.loc[recomputed_final, "model_id"])
    admissible_from_decision = {row["model_id"] for row in decision5["admissible_parameter_pairs"]}
    check(
        "admissible_region_is_exact_gate_intersection",
        bool((recomputed_final == adjudication.final_structural_admissible.astype(bool)).all() and admissible_from_rows == admissible_from_decision and decision5["n_admissible"] == len(admissible_from_rows) and decision5["empty_set_gate_relaxation_applied"] is False and decision5["grid_expanded"] is False),
        {"admissible_models": sorted(admissible_from_rows), "n_admissible": len(admissible_from_rows), "row_flag_mismatches": int((recomputed_final != adjudication.final_structural_admissible.astype(bool)).sum()), "gate_relaxed": decision5["empty_set_gate_relaxation_applied"], "grid_expanded": decision5["grid_expanded"]},
        "The final admissible region is exactly engineering∩soil∩hydrogeo∩river-noninferiority; GW-N stays contextual.",
    )
    decision6 = read_json(S16 / "20260816_6" / "reports" / "regionalization_decision.json")
    check(
        "regionalization_conditionally_not_run",
        decision6["regionalization_authorized"] is False and decision6["regional_delivery_status"] == "hydrogeo_or_tree_signal_nonidentifying_not_run",
        decision6,
        "No spatial mu pattern is fitted when hydrogeologic/tree evidence fails the frozen authorization rule.",
    )

    local = pd.read_parquet(ROOT / "outputs" / "representative_n_legacy_interface_1961_2022.parquet")
    routed = pd.read_parquet(ROOT / "outputs" / "representative_r0_routed_tn_1961_2022.parquet")
    cohort = pd.read_parquet(ROOT / "outputs" / "representative_cohort_state_end_2022.parquet")
    last = local.loc[(local.year == 2022) & (local.month == 12)].set_index("reach_id")
    cohort_end = cohort.groupby(["pool", "reach_id"]).cohort_mass_kg_n.sum()
    pool_state = {"son": "son_state_end_kg_n", "mobile": "mobile_state_end_kg_n", "quick": "quick_state_end_kg_n", "gw": "gw_state_end_kg_n"}
    cohort_errors = {}
    for pool, state in pool_state.items():
        values = cohort_end.loc[pool].reindex(last.index, fill_value=0.0)
        cohort_errors[pool] = float(np.max(np.abs(values.to_numpy(float) - last[state].to_numpy(float))))
    zero_quick = float(local.loc[local.quick_release_mm <= 1e-12, "quick_tn_release_kg_n"].abs().max()) if (local.quick_release_mm <= 1e-12).any() else 0.0
    zero_gw = float(local.loc[local.gw_discharge_mm <= 1e-12, "gw_tn_release_kg_n"].abs().max()) if (local.gw_discharge_mm <= 1e-12).any() else 0.0
    local_release_residual = np.abs(local.local_tn_release_kg_n - local.quick_tn_release_kg_n - local.gw_tn_release_kg_n)
    local_release_error = float(np.max(local_release_residual))
    local_release_relative_error = float(np.max(local_release_residual / np.maximum(np.abs(local.local_tn_release_kg_n) + np.abs(local.quick_tn_release_kg_n) + np.abs(local.gw_tn_release_kg_n), 1.0)))
    check(
        "cohort_state_mass_and_water_gate_closure",
        max(cohort_errors.values()) <= 1e-6 and local.mass_balance_relative_error.max() <= 1e-12 and zero_quick <= 1e-12 and zero_gw <= 1e-12 and local_release_relative_error <= 1e-12,
        {"cohort_vs_state_end_max_abs_kg_n": cohort_errors, "max_monthly_relative_mass_error": float(local.mass_balance_relative_error.max()), "zero_quick_water_release_kg_n": zero_quick, "zero_gw_water_release_kg_n": zero_gw, "local_release_identity_error_kg_n": local_release_error, "local_release_identity_relative_error": local_release_relative_error},
        "Cohorts close to scalar end states, monthly N mass closes, and zero Q72 pathway water causes zero N delivery.",
    )
    age_summary = read_json(ROOT / "reports" / "representative_2022_cohort_age_summary.json")
    terminal_2022 = routed.loc[routed.year.eq(2022) & routed.reach_id.eq(routed.terminal_tree_id)]
    total_mass = terminal_2022.routed_tn_kg_n.sum()
    post_mass = terminal_2022.routed_tn_post1961_kg_n.sum()
    recomputed_age = {
        "fraction_memory_gt1y": float(terminal_2022.routed_tn_gt1y_kg_n.sum() / total_mass),
        "fraction_memory_gt5y": float(terminal_2022.routed_tn_gt5y_kg_n.sum() / total_mass),
        "fraction_memory_gt10y": float(terminal_2022.routed_tn_gt10y_kg_n.sum() / total_mass),
        "post1961_mean_cohort_age_month": float(terminal_2022.routed_tn_post1961_age_moment_month_kg_n.sum() / post_mass),
    }
    age_error = max(abs(recomputed_age[key] - age_summary[key]) for key in recomputed_age)
    check(
        "cohort_age_outputs_directly_recomputed",
        age_error <= 1e-12 and recomputed_age["fraction_memory_gt10y"] <= recomputed_age["fraction_memory_gt5y"] <= recomputed_age["fraction_memory_gt1y"] <= 1.0 and "not hydrologic water age" in age_summary["semantics"],
        {"recomputed": recomputed_age, "maximum_report_difference": age_error, "semantics": age_summary["semantics"]},
        "Age fractions and mean age are direct terminal-output cohort mass summaries, not values inferred from tau or mu and not water ages.",
    )

    ensemble = pd.read_parquet(ROOT / "outputs" / "admissible_ensemble_r0_routed_tn_1961_2022.parquet")
    intervals = pd.read_parquet(ROOT / "outputs" / "admissible_ensemble_structural_quantiles_1961_2022.parquet")
    ensemble_models = set(ensemble.model_id)
    ensemble_counts = ensemble.groupby("model_id").size()
    rep_ensemble = ensemble.loc[ensemble.model_id.eq(decision5["representative_model_id"])].sort_values(["reach_id", "year", "month"])
    rep_routed = routed.sort_values(["reach_id", "year", "month"])
    primary = ["routed_quick_tn_kg_n", "routed_gw_tn_kg_n", "routed_tn_kg_n", "raw_tn_mg_l"]
    representative_errors = {column: float(np.nanmax(np.abs(rep_ensemble[column].to_numpy(float) - rep_routed[column].to_numpy(float)))) for column in primary}
    grouped = ensemble.groupby(["reach_id", "year", "month"], sort=True)
    interval_errors = {}
    for column in primary:
        for probability, label in [(0.0, "min"), (0.05, "p05"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.95, "p95"), (1.0, "max")]:
            actual = grouped[column].quantile(probability).to_numpy(float)
            saved = intervals.sort_values(["reach_id", "year", "month"])[f"{column}_{label}"].to_numpy(float)
            interval_errors[f"{column}_{label}"] = float(np.nanmax(np.abs(actual - saved)))
    check(
        "all_admissible_models_and_structural_intervals_delivered",
        ensemble_models == admissible_from_decision and ensemble_counts.eq(230 * 744).all() and len(intervals) == 230 * 744 and intervals.n_models.eq(len(admissible_from_decision)).all() and max(representative_errors.values()) <= 1e-6 and max(interval_errors.values()) <= 1e-12,
        {"ensemble_models": sorted(ensemble_models), "rows": len(ensemble), "rows_per_model": sorted(ensemble_counts.unique()), "quantile_rows": len(intervals), "representative_primary_max_abs_errors": representative_errors, "structural_interval_max_abs_error": max(interval_errors.values())},
        "All development-admissible pairs form the scenario ensemble; the representative is only one exact member and saved quantiles reproduce the complete ensemble.",
    )

    external_hydrology_path = S15_1 / "reports" / "external_hydrology_diagnostics.json"
    external_hydrology = read_json(external_hydrology_path)
    final = read_json(ROOT / "reports" / "final_scientific_decision.json")
    lock = read_json(ROOT / "reports" / "final_legacy_model_lock.json")
    protected_hash = lock["parent_sha256"].get(str(external_hydrology_path))
    check(
        "external_hydrology_preserved_as_diagnostic_only",
        protected_hash == sha256(external_hydrology_path) and final["watergap_recharge_role"] == "benchmark_only" and final["groundwater_level_role"] == "unresolved_auxiliary_validation" and final["hydrology_external_consistency"] == "non_identifying" and final["external_hydrology_selection_role"] == external_hydrology["decision_role"],
        {"protected_sha256": protected_hash, "actual_sha256": sha256(external_hydrology_path), "watergap_status": final["watergap_recharge_status"], "groundwater_level_status": final["groundwater_level_status"], "overall": final["hydrology_external_consistency"], "selection_role": final["external_hydrology_selection_role"]},
        "WaterGAP is only a recharge benchmark and groundwater level remains non-identifying; both are protected but absent from N selection.",
    )
    locked = pd.read_parquet(ROOT / "outputs" / "locked_2022_predictions.parquet")
    prelock = read_json(ROOT / "reports" / "pre_2022_legacy_lock.json")
    check(
        "locked_2022_temporal_isolation",
        set(locked.year.astype(int)) == {2022} and prelock["selection_complete_before_2022_read"] is True and prelock["admissible_parameter_pairs"] == decision5["admissible_parameter_pairs"] and final["locked_2022_changed_selection"] is False and decision5["locked_2022_used"] is False and source_tokens["prelock_before_2022_read"],
        {"locked_years": sorted(locked.year.astype(int).unique()), "prelock_selection_complete": prelock["selection_complete_before_2022_read"], "development_decision_locked_2022_used": decision5["locked_2022_used"], "changed_selection": final["locked_2022_changed_selection"]},
        "The admissible region and representative are frozen before the one-time 2022 observation read, which cannot alter selection.",
    )

    hash_pairs = []
    stage1_start = S16 / "20260816_1" / "reports" / "frozen_parent_hashes_start.json"
    stage1_end = S16 / "20260816_1" / "reports" / "frozen_parent_hashes_end.json"
    hash_pairs.append(("20260816_1", stage1_start, stage1_end))
    for stage in range(2, 8):
        hash_pairs.append((f"20260816_{stage}", S16 / f"20260816_{stage}" / "reports" / "parent_hashes_start.json", S16 / f"20260816_{stage}" / "reports" / "parent_hashes_end.json"))
    hash_results = {stage: start.exists() and end.exists() and read_json(start) == read_json(end) for stage, start, end in hash_pairs}
    check(
        "all_parent_hash_guards_stable",
        all(hash_results.values()),
        hash_results,
        "Every stage's protected parent manifest is identical before and after execution.",
    )
    canonical_paths = {label: Path(path) for label, path in lock["canonical_outputs"].items()}
    check(
        "final_lock_and_semantic_boundaries_complete",
        all(path.exists() for path in canonical_paths.values()) and final["representative_is_point_identification"] is False and final["future_scenario_uncertainty_rule"] == "all_admissible_pairs_not_representative_only" and final["species_semantics"] == "effective_TN_delivery_memory_not_nitrate_specific_TTD" and final["T1_plus_T0_serial_lag_added"] is False,
        {"canonical_outputs": {label: {"path": str(path), "exists": path.exists()} for label, path in canonical_paths.items()}, "representative_is_point_identification": final["representative_is_point_identification"], "future_scenario_rule": final["future_scenario_uncertainty_rule"], "species_semantics": final["species_semantics"]},
        "The final lock separates representative vs ensemble roles and preserves TN-memory, cohort-age and hydrologic-age semantics.",
    )

    failed = [name for name, item in evidence.items() if not item["pass"]]
    result = {
        "scenario_id": "20260816_7",
        "audit_type": "independent_full_completion_verifier",
        "pass": not failed,
        "confidence": "ready_to_share" if not failed else "needs_revision",
        "check_count": len(evidence),
        "failed_checks": failed,
        "evidence_file": str(REPORTS / "full_requirement_evidence.json"),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "full_requirement_evidence.json").write_text(json.dumps(as_native(evidence), ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "full_completion_audit.json").write_text(json.dumps(as_native(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(as_native(result), ensure_ascii=False, indent=2))
    if failed:
        raise RuntimeError(f"full completion audit failed: {failed}")


if __name__ == "__main__":
    main()
