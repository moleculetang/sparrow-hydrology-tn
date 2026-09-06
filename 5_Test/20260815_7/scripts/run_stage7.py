from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260815_7")
P6 = Path(r"E:\SPARROW\5_Test\20260815_6")
P5 = Path(r"E:\SPARROW\5_Test\20260815_5")
P4 = Path(r"E:\SPARROW\5_Test\20260815_4")
P1 = Path(r"E:\SPARROW\5_Test\20260815_1")
R0_PATH = P6 / "outputs" / "r0_routed_tn_1961_2022.parquet"
M0_CANDIDATES = P4 / "outputs" / "candidate_routed_reach_month_2016_2022.parquet"
OBS_PATH = P1 / "outputs" / "tn_observation_registry_2016_2022.parquet"
SOURCE_DECISION = P4 / "reports" / "source_structure_decision.json"
DELIVERY_DECISION = P5 / "reports" / "delivery_structure_decision.json"
ROUTING_DECISION = P6 / "reports" / "routing_decision.json"
EXTERNAL_HYDROLOGY = P1 / "reports" / "external_hydrology_diagnostics.json"
S4_SCRIPT = P4 / "scripts" / "run_stage4.py"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
MARGIN = 0.01


def load_stage4():
    spec = importlib.util.spec_from_file_location("stage4_final_core", S4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stage4 readout core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


s4 = load_stage4()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    default = lambda x: x.item() if hasattr(x, "item") else str(x)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame.tn_mg_l.to_numpy(float)
    p = frame.pred_tn_mg_l.to_numpy(float)
    ly, lp = np.log1p(y), np.log1p(p)
    oy = frame.observed_load_proxy_kg_n.to_numpy(float)
    op = frame.predicted_load_kg_n.to_numpy(float)
    return {
        "n": len(frame),
        "stations": int(frame.station_key.nunique()),
        "rmse_log1p_concentration": float(np.sqrt(np.mean((lp - ly) ** 2))),
        "rmse_raw_concentration_mg_l": float(np.sqrt(np.mean((p - y) ** 2))),
        "pbias_concentration_percent": float(100.0 * np.sum(p - y) / np.sum(y)),
        "correlation_concentration": float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan,
        "rmse_log1p_load_proxy_kg_n": float(np.sqrt(np.mean((np.log1p(op) - np.log1p(oy)) ** 2))),
        "pbias_load_proxy_percent": float(100.0 * np.sum(op - oy) / np.sum(oy)),
    }


def fit_and_predict(raw: pd.DataFrame, obs: pd.DataFrame, model_id: str) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = ["reach_id", "year", "month", "raw_tn_mg_l", "routed_water_volume_m3", "terminal_tree_id"]
    age_columns = [c for c in ["fraction_memory_gt1y", "fraction_memory_gt5y", "fraction_memory_gt10y", "fraction_pre1961_equilibrium", "post1961_mean_cohort_age_month", "all_history_mean_cohort_age_lower_bound_month", "routed_quick_tn_kg_n", "routed_base_tn_kg_n", "routed_tn_kg_n"] if c in raw.columns]
    joined = obs.merge(raw[columns + age_columns], on=["reach_id", "year", "month"], validate="many_to_one")
    train = joined.loc[joined.year.between(2016, 2021)].copy()
    locked = joined.loc[joined.year.eq(2022)].copy()
    stations = sorted(train.station_key.astype(str).unique())
    global_bias, effects = s4.fit_bias(train, stations)
    effect = locked.station_key.astype(str).map(effects).fillna(0.0).to_numpy(float)
    locked["pred_tn_mg_l"] = np.maximum(np.expm1(np.log1p(locked.raw_tn_mg_l.to_numpy(float)) + global_bias + effect), 0.0)
    locked["observed_load_proxy_kg_n"] = locked.tn_mg_l * locked.routed_water_volume_m3 / 1000.0
    locked["predicted_load_kg_n"] = locked.pred_tn_mg_l * locked.routed_water_volume_m3 / 1000.0
    locked["model_id"] = model_id
    return locked, {"model_id": model_id, "training_years": [2016, 2021], "global_log_bias": global_bias, "station_effect_count": len(effects), "station_log_bias_ridge_lambda": 12.0}


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow" or any(os.environ.get(k) != "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]):
        raise RuntimeError("sparrow runtime and one-thread limits required")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    stage_audits = {i: Path(rf"E:\SPARROW\5_Test\20260815_{i}\reports\completion_audit.json") for i in range(1, 7)}
    for i, path in stage_audits.items():
        if not json.loads(path.read_text(encoding="utf-8")).get("pass"):
            raise RuntimeError(f"stage {i} did not pass")
    source = json.loads(SOURCE_DECISION.read_text(encoding="utf-8"))
    delivery = json.loads(DELIVERY_DECISION.read_text(encoding="utf-8"))
    routing = json.loads(ROUTING_DECISION.read_text(encoding="utf-8"))
    parent_files = [R0_PATH, M0_CANDIDATES, OBS_PATH, SOURCE_DECISION, DELIVERY_DECISION, ROUTING_DECISION, EXTERNAL_HYDROLOGY, S4_SCRIPT, *stage_audits.values()]
    start = {str(p): sha256(p) for p in parent_files}
    dump(REPORTS / "parent_hashes_start.json", start)

    # This lock is written before the TN observation registry is loaded below.
    prelock = {
        "scenario_id": "20260815_7",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_complete_before_locked_metrics": True,
        "source_model_id": source["selected_source_model_id"],
        "delivery_model_id": delivery["selected_delivery_model_id"],
        "effective_tn_delivery_mu_month": delivery["selected_effective_tn_delivery_mu_month"],
        "routing_model_id": routing["routing_model_id"],
        "locked_year": 2022,
        "parent_sha256": start,
    }
    dump(REPORTS / "pre_2022_model_lock.json", prelock)

    selected_raw = pd.read_parquet(R0_PATH).rename(columns={"tn_concentration_proxy_mg_l": "raw_tn_mg_l"})
    candidate_raw = pd.read_parquet(M0_CANDIDATES)
    m0_raw = candidate_raw.loc[candidate_raw.model_id.eq("M0")].copy()
    obs = pd.read_parquet(OBS_PATH)
    selected_locked, selected_readout = fit_and_predict(selected_raw, obs, "selected_joint_model")
    m0_locked, m0_readout = fit_and_predict(m0_raw, obs, "M0_T0_R0_baseline")
    locked = pd.concat([selected_locked, m0_locked], ignore_index=True)
    locked.to_parquet(OUT / "locked_2022_tn_predictions.parquet", index=False)
    dump(REPORTS / "final_readout_parameters.json", {"selected": selected_readout, "M0": m0_readout})
    metric_table = pd.DataFrame([{"model_id": mid, **metrics(group)} for mid, group in locked.groupby("model_id")])
    metric_table.to_csv(REPORTS / "locked_2022_metrics.csv", index=False)
    delta_station = s4.block_bootstrap(selected_locked, m0_locked, "station_key")
    delta_tree = s4.block_bootstrap(selected_locked, m0_locked, "terminal_tree_id")
    boot = pd.concat([
        pd.DataFrame({"block": "station_key", "replicate": np.arange(len(delta_station)), "delta_log_rmse": delta_station}),
        pd.DataFrame({"block": "terminal_tree_id", "replicate": np.arange(len(delta_tree)), "delta_log_rmse": delta_tree}),
    ], ignore_index=True)
    boot.to_parquet(OUT / "locked_2022_bootstrap_distributions.parquet", index=False)
    ci = {}
    for block, values in [("station_key", delta_station), ("terminal_tree_id", delta_tree)]:
        low, high = np.percentile(values, [2.5, 97.5])
        ci[block] = {"lower": float(low), "upper": float(high), "mean": float(values.mean())}
    if all(v["upper"] < 0 for v in ci.values()):
        locked_status = "predictively_superior_to_M0_on_locked_2022"
    elif all(v["upper"] < MARGIN for v in ci.values()):
        locked_status = "structurally_noninferior_to_M0_on_locked_2022"
    else:
        locked_status = "locked_2022_does_not_confirm_noninferiority_to_M0"

    terminal_2022 = selected_raw.loc[selected_raw.year.eq(2022) & selected_raw.reach_id.eq(selected_raw.terminal_tree_id)].copy()
    total_mass = float(terminal_2022.routed_tn_kg_n.sum())
    post_mass = float(terminal_2022.routed_tn_post1961_kg_n.sum())
    age_summary = {
        "scope": "14_terminal_tree_outlets_all_2022_months",
        "fraction_memory_gt1y": float(terminal_2022.routed_tn_gt1y_kg_n.sum() / total_mass),
        "fraction_memory_gt5y": float(terminal_2022.routed_tn_gt5y_kg_n.sum() / total_mass),
        "fraction_memory_gt10y": float(terminal_2022.routed_tn_gt10y_kg_n.sum() / total_mass),
        "fraction_pre1961_equilibrium": float(terminal_2022.routed_tn_pre1961_kg_n.sum() / total_mass),
        "post1961_mean_cohort_age_month": float(terminal_2022.routed_tn_post1961_age_moment_month_kg_n.sum() / post_mass),
        "post1961_mean_cohort_age_year_report_only": float(terminal_2022.routed_tn_post1961_age_moment_month_kg_n.sum() / post_mass / 12.0),
        "all_history_mean_cohort_age_lower_bound_month": float(terminal_2022.routed_tn_age_lower_bound_moment_month_kg_n.sum() / total_mass),
        "all_history_mean_cohort_age_lower_bound_year_report_only": float(terminal_2022.routed_tn_age_lower_bound_moment_month_kg_n.sum() / total_mass / 12.0),
        "interpretation": "N_mass_cohort_age_not_water_age; all-history mean is a lower bound because pre1961 equilibrium input dates are unknown",
    }
    dump(REPORTS / "locked_2022_cohort_age_summary.json", age_summary)

    external = json.loads(EXTERNAL_HYDROLOGY.read_text(encoding="utf-8"))
    source_locked_confirmed = locked_status != "locked_2022_does_not_confirm_noninferiority_to_M0"
    science = {
        "operational_flow_model_id": "H0_hybrid_20260813_54",
        "structural_water_interface_branch_id": "main_20260814_6",
        "source_legacy_required": "yes_development_supported" if source_locked_confirmed else "development_supported_but_locked_not_confirmed",
        "source_model_id": source["selected_source_model_id"],
        "soil_legacy_time_scale_status": "upper_boundary_limited_not_point_identified" if str(source["selected_source_model_id"]).endswith("480m") else "internally_selected",
        "independent_effective_tn_delivery_memory_required": "no_non_identifying_retain_T0",
        "effective_tn_delivery_mu_month": delivery["selected_effective_tn_delivery_mu_month"],
        "river_retention_required": "not_tested_R1_not_run_without_tau_r",
        "routing_model_id": "R0_same_month_mass_conserving",
        "locked_2022_status": locked_status,
        "locked_2022_bootstrap_ci": ci,
        "groundwater_external_evidence": "non_identifying",
        "point_source_tn_status": "missing_not_fabricated",
        "species_semantics": "effective_TN_delivery_memory_not_nitrate_specific_TTD",
        "age_semantics": "input_month_N_mass_cohort_age_not_hydrologic_water_age",
    }
    dump(REPORTS / "final_scientific_decision.json", science)
    manifest = {"model_id": "Q72_main_plus_" + source["selected_source_model_id"] + "_plus_" + delivery["selected_delivery_model_id"] + "_plus_R0", "model_status": "frozen_after_single_use_2022_evaluation", "components": science, "pre_2022_lock_sha256": sha256(REPORTS / "pre_2022_model_lock.json"), "canonical_outputs": {"routed_full_history": str(R0_PATH), "locked_predictions": str(OUT / "locked_2022_tn_predictions.parquet"), "cohort_age_summary": str(REPORTS / "locked_2022_cohort_age_summary.json")}}
    dump(REPORTS / "final_model_manifest.json", manifest)
    end = {str(p): sha256(p) for p in parent_files}
    dump(REPORTS / "parent_hashes_end.json", end)
    if start != end:
        raise RuntimeError("parent changed during stage 7")
    print(json.dumps({"model_id": manifest["model_id"], "locked_2022_status": locked_status, "source_legacy_required": science["source_legacy_required"], "delivery_memory": science["independent_effective_tn_delivery_memory_required"], "post1961_mean_age_year": age_summary["post1961_mean_cohort_age_year_report_only"], "all_history_mean_age_lower_bound_year": age_summary["all_history_mean_cohort_age_lower_bound_year_report_only"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
