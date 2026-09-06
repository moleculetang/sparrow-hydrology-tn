from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "reports" / "q72_module_audit"
EPS = 1.0e-12
FOLDS = [
    ("fit_2006_2011_eval_2012_2013", 2011, 2012, 2013),
    ("fit_2006_2013_eval_2014_2015", 2013, 2014, 2015),
    ("fit_2006_2015_eval_2016_2018", 2015, 2016, 2018),
]
MODULES = ["forcing", "storage", "routing", "season", "interaction", "station"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_hyperparams() -> dict[str, float]:
    manifest_path = RUN / "reports" / "workflow" / "run_manifest.csv"
    if not manifest_path.exists():
        manifest_path = RUN.parent / "20260728_18" / "reports" / "workflow" / "run_manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    values: dict[str, float] = {}
    for token in str(manifest.loc[0, "selected_hyperparameters"]).split(";"):
        if "=" in token:
            key, value = token.split("=", 1)
            values[key.strip()] = float(value)
    return values


def capture_lists(model) -> dict[str, list[str]]:
    return {
        name: list(getattr(model, name))
        for name in [
            "FIXED_FEATURES", "PRODUCTION_FEATURES", "MULTISTORE_FEATURES",
            "HYSTERESIS_FEATURES", "RANDOM_SLOPE_FEATURES", "REGIME_SLOPE_FEATURES",
            "REGIME_GATES", "SPATIAL_GROUP_FEATURES", "SPATIAL_GROUP_GATES",
        ]
    }


def apply_lists(model, lists: dict[str, list[str]]) -> None:
    for name, values in lists.items():
        setattr(model, name, values)


def build_featured(model, et, slow, storage, gate, hp: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    original = capture_lists(model)
    apply_lists(model, storage.base_slow_reduced_lists(slow, original))
    observed = model.load_observed_panel()
    observed = observed[observed["year"] <= 2018].copy().reset_index(drop=True)
    et_config = {
        "variant": "et_surplus_water_stress_state",
        "water_for_sas": "surplus", "water_for_production": "surplus",
        "wetness": "stress_adjusted", "production_demand_fraction": 0.0,
        "stress_interaction": "dimensionless_dry_stress", "remove_stress_features": False,
    }
    featured = et.add_hydrologic_features_et_variant(model, observed, hp, et_config)
    featured = storage.recompute_state(featured, hp)
    featured = storage.recompute_sas(featured, hp, "current_clipped", "current_pre_release")
    featured = gate.apply_gate_form(featured, "current_overlap")
    featured = storage.recompute_dependent_features(featured, hp, "clipped_delta")
    featured = model.prepare_design(featured)
    featured = et.add_depth_features(featured)
    featured = slow.add_slow_indices(featured)
    return featured, sorted(featured["q_site"].astype(str).unique())


def fixed_module(feature: str, model) -> str:
    forcing = {
        "log_qcalc", "log_qma", "log_cumarea", "log_basin_net", "log_basin_threshold",
        "aridity", "log_aet", "log_pet", "log_et_deficit", "aet_pet_ratio",
        "aet_ppt_ratio", "pet_ppt_ratio", "log_lag1_aet", "log_lag1_et_deficit",
        "et_deficit_wetness", "antecedent_wetness", "wet_quickflow",
    }
    if feature in forcing:
        return "forcing"
    if feature in model.MULTISTORE_FEATURES or "sas_" in feature:
        return "storage"
    if feature in model.PRODUCTION_FEATURES:
        return "routing"
    if feature in model.HYSTERESIS_FEATURES or feature in {"month_sin", "month_cos"}:
        return "season"
    return "interaction"


def design_layout(model, stations: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = [{"column_index": 0, "column_name": "intercept", "module": "always_on", "term_type": "intercept"}]
    index = 1
    for feature in model.FIXED_FEATURES:
        rows.append({"column_index": index, "column_name": feature, "module": fixed_module(feature, model), "term_type": "fixed"})
        index += 1
    for gate in model.SPATIAL_GROUP_GATES:
        for feature in model.SPATIAL_GROUP_FEATURES:
            rows.append({"column_index": index, "column_name": f"spatial::{gate}::{feature}", "module": "interaction", "term_type": "spatial_group_slope"})
            index += 1
    for station in stations:
        rows.append({"column_index": index, "column_name": f"station_intercept::{station}", "module": "station", "term_type": "station_intercept"})
        index += 1
    for feature in model.RANDOM_SLOPE_FEATURES:
        for station in stations:
            rows.append({"column_index": index, "column_name": f"station_slope::{feature}::{station}", "module": "station", "term_type": "station_slope"})
            index += 1
    for gate in model.REGIME_GATES:
        for feature in model.REGIME_SLOPE_FEATURES:
            for station in stations:
                rows.append({"column_index": index, "column_name": f"regime_station_slope::{gate}::{feature}::{station}", "module": "station", "term_type": "regime_station_slope"})
                index += 1
    return pd.DataFrame(rows)


def fit_beta(model, train: pd.DataFrame, stations: list[str], hp: dict[str, float]) -> tuple[pd.Series, pd.Series, np.ndarray]:
    mean, std = model.standardize_fit(train)
    beta = model.fit_map_ridge(
        train, stations, mean, std,
        fixed_sigma=hp["fixed_sigma"], production_sigma=hp["production_sigma"],
        group_sigma=hp["group_sigma"], multistore_sigma=hp["multistore_sigma"],
        hysteresis_sigma=hp["hysteresis_sigma"], station_sigma=hp["station_sigma"],
        slope_sigma=hp["slope_sigma"], regime_slope_sigma=hp["regime_slope_sigma"],
        anomaly_weight=0.0, flow_contrast_weight=hp.get("flow_contrast_weight", 1.0),
    )
    return mean, std, beta


def station_metrics(model, frame: pd.DataFrame, eta: np.ndarray) -> pd.DataFrame:
    rows = []
    work = frame[["q_site", "Q_obsv_cfs"]].copy()
    work["eta"] = eta
    for site, part in work.groupby("q_site", sort=False):
        values = model.metric_dict(part["Q_obsv_cfs"].to_numpy(dtype=float), np.exp(np.clip(part["eta"].to_numpy(dtype=float), -20, 20)))
        values["q_site"] = site
        values["abs_PBIAS"] = abs(values["PBIAS_pct"])
        rows.append(values)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_module("q72_model", RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py")
    et = load_module("et_module", RUN / "scripts" / "components" / "run_et_role_experiment.py")
    slow = load_module("slow_module", RUN / "scripts" / "components" / "run_slowflow_redundancy_experiment.py")
    storage = load_module("storage_module", RUN / "scripts" / "components" / "run_storage_timing_experiment.py")
    gate = load_module("gate_module", RUN / "scripts" / "components" / "run_hysteresis_gate_experiment.py")
    hp = parse_hyperparams()
    featured, stations = build_featured(model, et, slow, storage, gate, hp)
    layout = design_layout(model, stations)
    probe_train = featured[featured["year"] <= 2011]
    probe_x, _ = model.build_matrix(probe_train.iloc[:1], stations, *model.standardize_fit(probe_train))
    if len(layout) != probe_x.shape[1]:
        raise RuntimeError("Design layout does not match Q72 matrix width")
    if not set(MODULES).issubset(set(layout["module"])) or not set(layout["module"]).issubset(set(MODULES) | {"always_on"}) or layout["column_index"].duplicated().any():
        raise RuntimeError("Module registry is incomplete or non-unique")
    layout.to_csv(OUT / "q72_design_column_registry.csv", index=False, encoding="utf-8-sig")
    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for fold_id, train_end, eval_start, eval_end in FOLDS:
        train = featured[featured["year"] <= train_end].copy()
        evaluation = featured[featured["year"].between(eval_start, eval_end)].copy()
        mean, std, beta = fit_beta(model, train, stations, hp)
        x_eval, _ = model.build_matrix(evaluation, stations, mean, std)
        baseline_eta = x_eval @ beta
        baseline_log_sse = float(np.sum((baseline_eta - evaluation["log_obs"].to_numpy(dtype=float)) ** 2))
        baseline_station = station_metrics(model, evaluation, baseline_eta).set_index("q_site")
        for module in MODULES:
            mask = layout["module"].eq(module).to_numpy()
            beta_drop = beta.copy()
            beta_drop[mask] = 0.0
            eta_drop = x_eval @ beta_drop
            delta = eta_drop - baseline_eta
            ablated_log_sse = float(np.sum((eta_drop - evaluation["log_obs"].to_numpy(dtype=float)) ** 2))
            changed_station = station_metrics(model, evaluation, eta_drop).set_index("q_site")
            common = baseline_station.index.intersection(changed_station.index)
            station_delta = changed_station.loc[common] - baseline_station.loc[common]
            rows.append({
                "fold_id": fold_id, "train_end_year": train_end, "eval_start_year": eval_start, "eval_end_year": eval_end,
                "module": module, "module_column_count": int(mask.sum()), "evaluation_rows": int(len(evaluation)),
                "median_abs_delta_eta": float(np.median(np.abs(delta))), "mean_abs_delta_eta": float(np.mean(np.abs(delta))),
                "baseline_log_sse": baseline_log_sse, "ablated_log_sse": ablated_log_sse,
                "delta_log_sse": ablated_log_sse - baseline_log_sse,
                "relative_log_sse_change_pct": float(100.0 * (ablated_log_sse - baseline_log_sse) / max(baseline_log_sse, EPS)),
                "median_delta_NSElog": float(station_delta["NSE_log"].median()),
                "median_delta_KGE": float(station_delta["KGE_2012"].median()),
                "median_delta_absPBIAS": float(station_delta["abs_PBIAS"].median()),
            })
            sites = evaluation["q_site"].astype(str).to_numpy()
            for site in common:
                station_rows.append({
                    "fold_id": fold_id, "module": module, "q_site": site,
                    "delta_NSElog": float(station_delta.at[site, "NSE_log"]), "delta_KGE": float(station_delta.at[site, "KGE_2012"]),
                    "delta_absPBIAS": float(station_delta.at[site, "abs_PBIAS"]), "mean_abs_delta_eta": float(np.mean(np.abs(delta[sites == str(site)]))),
                })
    by_fold = pd.DataFrame(rows)
    by_station = pd.DataFrame(station_rows)
    summary = by_fold.groupby("module", as_index=False).agg(
        module_column_count=("module_column_count", "first"), folds=("fold_id", "nunique"),
        median_SL=("median_abs_delta_eta", "median"), min_SL=("median_abs_delta_eta", "min"),
        median_relative_log_sse_change_pct=("relative_log_sse_change_pct", "median"), min_relative_log_sse_change_pct=("relative_log_sse_change_pct", "min"),
        median_delta_NSElog=("median_delta_NSElog", "median"), median_delta_KGE=("median_delta_KGE", "median"),
        median_delta_absPBIAS=("median_delta_absPBIAS", "median"),
    )
    summary["practical_frozen_leverage"] = summary["median_SL"] >= 0.01
    summary["low_leverage_simplification_candidate"] = (summary["median_SL"] < 0.01) & (summary["min_relative_log_sse_change_pct"].abs() < 1.0)
    by_fold.to_csv(OUT / "frozen_module_audit_by_fold.csv", index=False, encoding="utf-8-sig")
    by_station.to_csv(OUT / "frozen_module_audit_by_station.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "frozen_module_audit_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "run_id": RUN.name, "parent_run": "20260728_18", "folds": [x[0] for x in FOLDS], "modules": MODULES,
        "used_year_min": int(featured["year"].min()), "used_year_max": int(featured["year"].max()),
        "forbidden_confirmation_years_used": bool(featured["year"].max() > 2018), "registry_column_count": int(len(layout)),
        "registry_unique": bool(not layout["column_index"].duplicated().any()), "all_modules_present": sorted(summary["module"].tolist()) == sorted(MODULES),
        "all_fold_module_rows_present": int(len(by_fold)) == len(FOLDS) * len(MODULES),
    }
    payload["passed"] = bool(payload["used_year_max"] == 2018 and payload["registry_unique"] and payload["all_modules_present"] and payload["all_fold_module_rows_present"])
    (OUT / "gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Q72 frozen module sensitivity audit", "", "This is a pre-2019 frozen-coefficient leverage screen, not a simplified-model fit.", "", "| module | columns | median SL | median relative log-SSE change (%) | low-leverage candidate |", "| --- | ---: | ---: | ---: | --- |"]
    for row in summary.sort_values("median_SL", ascending=False).itertuples(index=False):
        lines.append(f"| {row.module} | {row.module_column_count} | {row.median_SL:.6f} | {row.median_relative_log_sse_change_pct:.3f} | {row.low_leverage_simplification_candidate} |")
    lines.extend(["", f"- Engineering gate: {'PASS' if payload['passed'] else 'FAIL'}", "- 2019-2022 used: False"])
    (OUT / "frozen_module_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not payload["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
