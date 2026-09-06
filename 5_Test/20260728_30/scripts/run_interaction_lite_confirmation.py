from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
BASELINE = RUN.parent / "20260728_18"
AUDIT_PARENT = RUN.parent / "20260728_19"
COMPENSATION = RUN.parent / "20260728_20"
OUT = RUN / "reports" / "interaction_lite_confirmation"
EPS = 1.0e-6


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_table(model, frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for site, part in frame.groupby("q_site", sort=False):
        values = model.metric_dict(part["Q_obsv_cfs"].to_numpy(dtype=float), part[column].to_numpy(dtype=float))
        values["q_site"] = str(site)
        values["abs_PBIAS"] = abs(float(values["PBIAS_pct"]))
        values["good"] = bool(values["n"] >= 24 and values["NSE_log"] >= 0.65 and values["KGE_2012"] >= 0.50 and values["abs_PBIAS"] <= 25.0)
        values["severe_pbias"] = bool(values["abs_PBIAS"] > 50.0)
        rows.append(values)
    return pd.DataFrame(rows).set_index("q_site")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_module("audit_parent", AUDIT_PARENT / "scripts" / "run_q72_frozen_module_audit.py")
    audit.RUN = RUN
    compensation = load_module("compensation", COMPENSATION / "scripts" / "run_q72_module_compensation_audit.py")
    model = load_module("q72_model", RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py")
    et = load_module("et_module", RUN / "scripts" / "components" / "run_et_role_experiment.py")
    slow = load_module("slow_module", RUN / "scripts" / "components" / "run_slowflow_redundancy_experiment.py")
    storage = load_module("storage_module", RUN / "scripts" / "components" / "run_storage_timing_experiment.py")
    gate_module = load_module("gate_module", RUN / "scripts" / "components" / "run_hysteresis_gate_experiment.py")
    hp = audit.parse_hyperparams()
    original = audit.capture_lists(model)
    audit.apply_lists(model, storage.base_slow_reduced_lists(slow, original))
    observed = model.load_observed_panel().reset_index(drop=True)
    et_config = {"variant": "et_surplus_water_stress_state", "water_for_sas": "surplus", "water_for_production": "surplus", "wetness": "stress_adjusted", "production_demand_fraction": 0.0, "stress_interaction": "dimensionless_dry_stress", "remove_stress_features": False}
    featured = et.add_hydrologic_features_et_variant(model, observed, hp, et_config)
    featured = storage.recompute_state(featured, hp)
    featured = storage.recompute_sas(featured, hp, "current_clipped", "current_pre_release")
    featured = gate_module.apply_gate_form(featured, "current_overlap")
    featured = storage.recompute_dependent_features(featured, hp, "clipped_delta")
    featured = model.prepare_design(featured)
    featured = et.add_depth_features(featured)
    featured = slow.add_slow_indices(featured)
    stations = sorted(featured["q_site"].astype(str).unique())
    layout = audit.design_layout(model, stations)
    keep = ~layout["module"].eq("interaction").to_numpy()
    train = featured[featured["year"] <= 2018].copy()
    evaluation = featured[featured["year"].between(2019, 2022)].copy()
    mean, std = model.standardize_fit(train)
    x_aug, y_aug = compensation.augmented_design(model, train, stations, mean, std, hp)
    beta_full, *_ = np.linalg.lstsq(x_aug, y_aug, rcond=None)
    beta_ref = model.fit_map_ridge(train, stations, mean, std, hp["fixed_sigma"], hp["production_sigma"], hp["group_sigma"], hp["multistore_sigma"], hp["hysteresis_sigma"], hp["station_sigma"], hp["slope_sigma"], hp["regime_slope_sigma"], 0.0, hp.get("flow_contrast_weight", 1.0))
    beta_delta = float(np.max(np.abs(beta_full - beta_ref)))
    beta_lite, *_ = np.linalg.lstsq(x_aug[:, keep], y_aug, rcond=None)
    x_eval, _ = model.build_matrix(evaluation, stations, mean, std)
    evaluation = evaluation[["q_site", "comid", "year", "month", "Q_obsv_cfs"]].copy()
    evaluation["Q72_lite_cfs"] = np.exp(np.clip(x_eval[:, keep] @ beta_lite, -20, 20))
    baseline = pd.read_csv(BASELINE / "reports" / "main_model" / "reach_class_selected_predictions_long.csv", encoding="utf-8-sig")
    baseline = baseline[baseline["year"].between(2019, 2022)][["q_site", "reach_id", "year", "month", "Q78_mass_cfs", "Q_pred_cfs", "alpha"]].copy()
    baseline = baseline.rename(columns={"reach_id": "comid", "Q_pred_cfs": "baseline_Q_pred_cfs"})
    merged = evaluation.merge(baseline, on=["q_site", "comid", "year", "month"], how="inner", validate="one_to_one")
    merged["candidate_Q_pred_cfs"] = np.exp((1.0 - merged["alpha"].to_numpy(dtype=float)) * np.log(np.maximum(merged["Q72_lite_cfs"].to_numpy(dtype=float), EPS)) + merged["alpha"].to_numpy(dtype=float) * np.log(np.maximum(merged["Q78_mass_cfs"].to_numpy(dtype=float), EPS)))
    base = metric_table(model, merged, "baseline_Q_pred_cfs")
    candidate = metric_table(model, merged, "candidate_Q_pred_cfs")
    common = base.index.intersection(candidate.index)
    base, candidate = base.loc[common], candidate.loc[common]
    delta = candidate[["NSE_log", "KGE_2012", "abs_PBIAS"]] - base[["NSE_log", "KGE_2012", "abs_PBIAS"]]
    new_severe = int((~base["severe_pbias"] & candidate["severe_pbias"]).sum())
    good_delta = int(candidate["good"].sum() - base["good"].sum())
    stone = "石角站"
    stone_ok = bool(stone in common and delta.at[stone, "NSE_log"] >= -0.02 and delta.at[stone, "KGE_2012"] >= -0.03 and delta.at[stone, "abs_PBIAS"] <= 5.0)
    gate = {"run_id": RUN.name, "candidate": "interaction_lite_with_baseline_q78_alpha", "fit_year_max": 2018, "confirmation_years": "2019-2022", "baseline_reproduction_max_abs_beta_delta": beta_delta, "common_station_count": int(len(common)), "median_delta_NSElog": float(delta["NSE_log"].median()), "median_delta_KGE": float(delta["KGE_2012"].median()), "median_delta_absPBIAS": float(delta["abs_PBIAS"].median()), "good_station_delta": good_delta, "new_severe_pbias_count": new_severe, "shijiao_protection_passed": stone_ok}
    gate["passed"] = bool(beta_delta <= 1e-10 and gate["median_delta_NSElog"] >= -0.005 and gate["median_delta_KGE"] >= -0.01 and gate["median_delta_absPBIAS"] <= 1.0 and good_delta >= -1 and new_severe == 0 and stone_ok)
    station = base.add_prefix("baseline_").join(candidate.add_prefix("candidate_")).join(delta.add_prefix("delta_"))
    station.to_csv(OUT / "confirmation_station_metrics.csv", encoding="utf-8-sig")
    merged.to_csv(OUT / "confirmation_predictions_long.csv", index=False, encoding="utf-8-sig")
    (OUT / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Interaction-lite protected confirmation", "", f"- Common stations: {gate['common_station_count']}", f"- Median delta NSElog: {gate['median_delta_NSElog']:+.6f}", f"- Median delta KGE: {gate['median_delta_KGE']:+.6f}", f"- Median delta abs PBIAS: {gate['median_delta_absPBIAS']:+.3f} pp", f"- Good station delta: {good_delta}", f"- New severe PBIAS stations: {new_severe}", f"- Shijiao protection: {stone_ok}", f"- Confirmation: {'PASS' if gate['passed'] else 'FAIL'}"]
    (OUT / "interaction_lite_confirmation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not gate["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
