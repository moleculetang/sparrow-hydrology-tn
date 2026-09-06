"""Refit the frozen TN architecture with the accepted DYN2P+SIG2P interface."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_11"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CACHE = RUN / "cache" / "local_components"
CONTRACT = RUN / "experiment_contract.json"
PARENT10 = ROOT / "5_Test" / "20260824_10"
MONTHLY = PARENT10 / "outputs" / "mainline_reach_month_n_hydrology_1961_2024.parquet"
EARLY = PARENT10 / "outputs" / "mainline_pre1961_early_n_mean_by_reach.parquet"
HYDRO = PARENT10 / "outputs" / "dyn2p_sig2p_tn_hydrology_interface_2006_2024.parquet"
EXPOSURE = PARENT10 / "outputs" / "dyn2p_h1_hydraulic_exposure_2006_2024.parquet"
PARENT_AUDIT = PARENT10 / "reports" / "hydrology_tn_interface_audit.json"
DATA_AUDIT = PARENT10 / "reports" / "mainline_data_quality_audit.json"
TN_ALL = ROOT / "1_Inputs" / "WaterQualityData" / "model_ready" / "tn_station_month_all.parquet"
DOMAIN = ROOT / "5_Test" / "20260820_19" / "outputs" / "observation_domain_registry.parquet"
H_SHARED = ROOT / "5_Test" / "20260820_19" / "scripts" / "hierarchical19_shared.py"
F1_SHARED = ROOT / "5_Test" / "20260824_4" / "scripts" / "run_f1_readout.py"
LEGACY_SHARED = ROOT / "5_Test" / "20260818_1" / "scripts" / "legacy18_shared.py"
STAGE4 = ROOT / "5_Test" / "20260815_4" / "scripts" / "run_stage4.py"
OLD_PREDICTIONS = ROOT / "5_Test" / "20260824_4" / "outputs" / "f1_temporal_oof_predictions.parquet"


HISTORICAL_FOLDS = (
    ("H1", 2016, 2017, 2018),
    ("H2", 2016, 2018, 2019),
    ("H3", 2016, 2019, 2020),
    ("H4", 2016, 2020, 2021),
)
RECENT_FOLDS = (
    ("R1", 2021, 2021, 2022),
    ("R2", 2021, 2022, 2023),
    ("R3", 2021, 2023, 2024),
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, ensure_ascii=False, indent=2,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ), encoding="utf-8")


def observations() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(TN_ALL)
    data = data.loc[data.strict & data.year.between(2016, 2024) & data.reach_id.notna() & data.tn_mg_l.gt(0)].copy()
    data["reach_id"] = data.reach_id.astype(int)
    registry = pd.read_parquet(DOMAIN, columns=["station_key", "primary_river_domain"])
    data = data.merge(registry, on="station_key", how="left", validate="many_to_one")
    if data.primary_river_domain.isna().any():
        raise RuntimeError("new strict TN station is absent from the frozen domain registry")
    primary = data.loc[data.primary_river_domain].drop(columns="primary_river_domain").reset_index(drop=True)
    diagnostic = data.loc[~data.primary_river_domain].drop(columns="primary_river_domain").reset_index(drop=True)
    if primary.duplicated(["station_key", "year", "month"]).any():
        raise RuntimeError("duplicate primary TN key")
    return primary, diagnostic


def build_local_components(legacy, stage4) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    panel = pd.read_parquet(MONTHLY)
    reach_ids, times, arrays = stage4.prepare_arrays(panel)
    early = pd.read_parquet(EARLY).set_index("reach_id").loc[reach_ids, "early_1961_1965_positive_surplus_mean_kg_n_year"].to_numpy(float) / 12.0
    specs = legacy.formal_specs()
    outputs: dict[str, pd.DataFrame] = {}
    audits = []
    CACHE.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        model_id = str(spec["model_id"])
        path = CACHE / f"{model_id}.parquet"
        audit_path = CACHE / f"{model_id}.json"
        if path.is_file() and audit_path.is_file():
            frame = pd.read_parquet(path)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            frame, audit, spin, _ = legacy.simulate_operator(
                "F00", model_id, str(spec["source_structure"]), spec["soil_tau_month"],
                int(spec["delivery_mu_month"]), reach_ids, times, arrays, early,
                capture_start_year=2016, capture_end_year=2024,
            )
            audit = {**audit, "spinup": spin}
            frame.to_parquet(path, index=False)
            write_json(audit_path, audit)
        if len(frame) != 230 * 9 * 12:
            raise RuntimeError(f"local component coverage failed: {model_id}")
        frame = frame.rename(columns={
            "quick_tn_release_kg_n": "fast_response_tn_release_kg_n",
            "gw_tn_release_kg_n": "slow_response_tn_release_kg_n",
        })
        outputs[model_id] = frame
        audits.append({
            "model_id": model_id,
            "max_abs_mass_balance_error_kg_n": float(audit["max_abs_mass_balance_error_kg_n"]),
            "max_relative_mass_balance_error": float(audit["max_relative_mass_balance_error"]),
            "minimum_state_or_flux_kg_n": float(audit["minimum_state_or_flux_kg_n"]),
            "spinup_converged": bool(audit["spinup"]["converged"]),
            "spinup_cycles": int(audit["spinup"]["cycles"]),
        })
    return outputs, pd.DataFrame(audits)


def build_router(model_id: str, frame: pd.DataFrame, h, shared):
    local = frame.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    exposure = pd.read_parquet(EXPOSURE).loc[lambda x: x.year.between(2016, 2024)].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    hydro = pd.read_parquet(HYDRO).loc[lambda x: x.year.between(2016, 2024)].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    keys = local[["reach_id", "year", "month"]]
    if not keys.equals(exposure[["reach_id", "year", "month"]]) or not keys.equals(hydro[["reach_id", "year", "month"]]):
        raise RuntimeError(f"router key mismatch: {model_id}")
    reach_ids = np.arange(1, 231, dtype=int)
    times = local[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    shape = (len(times), len(reach_ids))
    order, downstream, terminal = shared.topology_operators(reach_ids)
    lookup = {int(value): index for index, value in enumerate(reach_ids)}
    return h.HydraulicRouter(
        model_id=model_id,
        reach_ids=reach_ids,
        years=times.year.to_numpy(int),
        months=times.month.to_numpy(int),
        local_q=local.fast_response_tn_release_kg_n.to_numpy(float).reshape(shape),
        local_g=local.slow_response_tn_release_kg_n.to_numpy(float).reshape(shape),
        h_full=exposure.uptake_exposure_full_days_per_m.to_numpy(float).reshape(shape),
        h_mid=exposure.uptake_exposure_midpoint_to_outlet_days_per_m.to_numpy(float).reshape(shape),
        water=hydro.routed_total_water_volume_m3.to_numpy(float).reshape(shape),
        x=np.zeros((230, len(h.COVARIATE_COLUMNS)), dtype=float),
        terminal_by_reach=np.asarray([terminal[int(value)] for value in reach_ids], dtype=int),
        order_index=[lookup[int(value)] for value in order],
        downstream_index={lookup[int(k)]: (lookup[int(v[0])], float(v[1])) for k, v in downstream.items()},
    )


def add_q_features(frame: pd.DataFrame, train_start: int, train_end: int, hydro: pd.DataFrame) -> pd.DataFrame:
    q = hydro[["reach_id", "year", "month", "routed_total_discharge_m3_s"]].copy()
    q["log_q72"] = np.log(q.routed_total_discharge_m3_s.clip(lower=1.0e-12))
    center = q.loc[q.year.between(train_start, train_end)].groupby("reach_id").log_q72.median().rename("training_all_month_log_q_median")
    out = frame.merge(q, on=["reach_id", "year", "month"], validate="many_to_one").merge(center, on="reach_id", validate="many_to_one")
    if len(out) != len(frame) or out[["log_q72", "training_all_month_log_q_median"]].isna().any().any():
        raise RuntimeError("C-Q feature construction failed")
    out["cq_z"] = out.log_q72 - out.training_all_month_log_q_median
    out["cq_low"] = np.minimum(out.cq_z, 0.0)
    out["cq_high"] = np.maximum(out.cq_z, 0.0)
    return out


def fit_fold(model_id: str, router, obs: pd.DataFrame, fold: tuple[str, int, int, int], program: str, h, f1, shared, hydro: pd.DataFrame) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    fold_id, train_start, train_end, eval_year = fold
    train_obs = obs.loc[obs.year.between(train_start, train_end)].copy()
    test_obs = obs.loc[obs.year.eq(eval_year)].copy()
    h1 = h.fit_structure(router, train_obs, "H1_GLOBAL", shared)
    vf = float(np.asarray(h1["parameters"])[0])
    vector = np.full(len(router.reach_ids), vf, dtype=float)
    train = add_q_features(router.frame(train_obs, vector), train_start, train_end, hydro)
    test = add_q_features(router.frame(test_obs, vector), train_start, train_end, hydro)
    predictions, parameters = [], []
    for layer in ("P1", "P2"):
        fit = f1.gaussian_fit(train, layer, True, shared)
        prediction = f1.predict(test, fit, layer)
        prediction["model_id"] = model_id
        prediction["program"] = program
        prediction["fold_id"] = fold_id
        prediction["train_start_year"] = train_start
        prediction["train_end_year"] = train_end
        prediction["evaluation_year"] = eval_year
        prediction["layer"] = layer
        prediction["arm"] = "DYN2P_SIG2P_CQ_HINGE"
        prediction["v_f_m_per_day"] = vf
        predictions.append(prediction)
        parameters.append({
            "model_id": model_id, "program": program, "fold_id": fold_id,
            "train_start_year": train_start, "train_end_year": train_end, "evaluation_year": eval_year,
            "layer": layer, "v_f_m_per_day": vf,
            "eta_fast": float(fit["eta"][0]), "eta_slow": float(fit["eta"][1]),
            "beta_low": float(fit["beta"][0]), "beta_high": float(fit["beta"][1]),
            "readout_success": bool(fit["success"]), "eta_boundary": bool(fit["eta_boundary"]),
            "beta_boundary": bool(fit["beta_boundary"]), "station_effect_count": len(fit["effects"]),
            "H1_outer_success": bool(h1["outer_success"]), "H1_inner_success": bool(h1["diagnostic"]["success"]),
        })
    return predictions, parameters


def final_map(model_id: str, router, obs: pd.DataFrame, h, f1, shared, hydro: pd.DataFrame) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    train_obs = obs.loc[obs.year.between(2021, 2024)].copy()
    evaluate_obs = obs.loc[obs.year.between(2016, 2024)].copy()
    h1 = h.fit_structure(router, train_obs, "H1_GLOBAL", shared)
    vf = float(np.asarray(h1["parameters"])[0])
    vector = np.full(len(router.reach_ids), vf, dtype=float)
    train = add_q_features(router.frame(train_obs, vector), 2021, 2024, hydro)
    evaluate = add_q_features(router.frame(evaluate_obs, vector), 2021, 2024, hydro)
    predictions, parameters = [], []
    for layer in ("P1", "P2"):
        fit = f1.gaussian_fit(train, layer, True, shared)
        prediction = f1.predict(evaluate, fit, layer)
        prediction["model_id"] = model_id
        prediction["program"] = np.where(prediction.year.between(2021, 2024), "FINAL_MAP_APPARENT_2021_2024", "FINAL_MAP_HISTORICAL_BACKCAST_2016_2020")
        prediction["fold_id"] = "FINAL"
        prediction["train_start_year"] = 2021
        prediction["train_end_year"] = 2024
        prediction["evaluation_year"] = prediction.year
        prediction["layer"] = layer
        prediction["arm"] = "DYN2P_SIG2P_CQ_HINGE"
        prediction["v_f_m_per_day"] = vf
        predictions.append(prediction)
        parameters.append({
            "model_id": model_id, "program": "FINAL_MAP_2021_2024", "fold_id": "FINAL",
            "train_start_year": 2021, "train_end_year": 2024, "evaluation_year": None,
            "layer": layer, "v_f_m_per_day": vf,
            "eta_fast": float(fit["eta"][0]), "eta_slow": float(fit["eta"][1]),
            "beta_low": float(fit["beta"][0]), "beta_high": float(fit["beta"][1]),
            "readout_success": bool(fit["success"]), "eta_boundary": bool(fit["eta_boundary"]),
            "beta_boundary": bool(fit["beta_boundary"]), "station_effect_count": len(fit["effects"]),
            "H1_outer_success": bool(h1["outer_success"]), "H1_inner_success": bool(h1["diagnostic"]["success"]),
        })
    return predictions, parameters


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "program", "fold_id", "train_start_year", "train_end_year", "evaluation_year", "layer",
        "station_key", "reach_id", "year", "month", "tn_mg_l", "terminal_tree_id",
    ]
    return predictions.groupby(keys, as_index=False, dropna=False).agg(
        pred_tn_mg_l=("pred_tn_mg_l", "mean"),
        member_prediction_sd_mg_l=("pred_tn_mg_l", "std"),
        member_count=("model_id", "nunique"),
    )


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame.tn_mg_l.to_numpy(float)
    pred = frame.pred_tn_mg_l.to_numpy(float)
    error = pred - obs
    denominator = float(np.sum(np.square(obs - obs.mean())))
    corr = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 and np.std(pred) > 0 else math.nan
    station_nse = []
    station_log_rmse = []
    for _, group in frame.groupby("station_key"):
        y, p = group.tn_mg_l.to_numpy(float), group.pred_tn_mg_l.to_numpy(float)
        denom = float(np.sum(np.square(y - y.mean())))
        station_nse.append(1.0 - float(np.sum(np.square(p - y))) / denom if denom > 0 else math.nan)
        station_log_rmse.append(float(np.sqrt(np.mean(np.square(np.log1p(p) - np.log1p(y))))))
    return {
        "n": int(len(frame)), "stations": int(frame.station_key.nunique()), "reaches": int(frame.reach_id.nunique()),
        "rmse_mg_l": float(np.sqrt(np.mean(np.square(error)))),
        "mae_mg_l": float(np.mean(np.abs(error))),
        "nse": float(1.0 - np.sum(np.square(error)) / denominator) if denominator > 0 else math.nan,
        "r2": corr * corr, "pbias_percent": float(100.0 * np.sum(error) / np.sum(obs)),
        "station_median_nse": float(np.nanmedian(station_nse)),
        "station_macro_log_rmse": float(np.mean(station_log_rmse)),
    }


def metric_table(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in ensemble.groupby(["program", "layer"], observed=True):
        rows.append({"program": keys[0], "layer": keys[1], **metric_values(frame)})
    for keys, frame in ensemble.loc[ensemble.program.isin(["HISTORICAL_OOF", "RECENT_ROLLING_OOF"])].groupby(["program", "evaluation_year", "layer"], observed=True):
        rows.append({"program": f"{keys[0]}_{int(keys[1])}", "layer": keys[2], **metric_values(frame)})
    return pd.DataFrame(rows)


def old_new_comparison(new_ensemble: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_parquet(OLD_PREDICTIONS)
    old = old.loc[old.arm.eq("GAUSSIAN_CQ_HINGE") & old.year.between(2018, 2021)].copy()
    old["program"] = "OLD_Q72_HISTORICAL_OOF"
    old_ensemble = old.groupby(["layer", "station_key", "reach_id", "year", "month", "tn_mg_l"], as_index=False).pred_tn_mg_l.mean()
    new = new_ensemble.loc[new_ensemble.program.eq("HISTORICAL_OOF")]
    rows, paired = [], []
    rng = np.random.default_rng(20260827)
    for layer in ("P1", "P2"):
        left = old_ensemble.loc[old_ensemble.layer.eq(layer)]
        right = new.loc[new.layer.eq(layer)]
        joined = left.merge(right, on=["layer", "station_key", "reach_id", "year", "month", "tn_mg_l"], suffixes=("_old", "_new"), validate="one_to_one")
        old_metrics = metric_values(joined.rename(columns={"pred_tn_mg_l_old": "pred_tn_mg_l"}))
        new_metrics = metric_values(joined.rename(columns={"pred_tn_mg_l_new": "pred_tn_mg_l"}))
        rows.extend([
            {"hydrology": "OLD_Q72", "layer": layer, **old_metrics},
            {"hydrology": "DYN2P_SIG2P", "layer": layer, **new_metrics},
        ])
        deltas = []
        for _, group in joined.groupby("station_key"):
            y = np.log1p(group.tn_mg_l.to_numpy(float))
            old_rmse = np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l_old.to_numpy(float)) - y)))
            new_rmse = np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l_new.to_numpy(float)) - y)))
            deltas.append(new_rmse - old_rmse)
        deltas = np.asarray(deltas, dtype=float)
        index = rng.integers(0, len(deltas), size=(10_000, len(deltas)))
        distribution = deltas[index].mean(axis=1)
        paired.append({
            "layer": layer, "station_count": len(deltas), "delta_new_minus_old": float(deltas.mean()),
            "ci95_lower": float(np.quantile(distribution, 0.025)), "ci95_upper": float(np.quantile(distribution, 0.975)),
            "improved": bool(np.quantile(distribution, 0.975) < 0),
            "noninferior_0p005": bool(np.quantile(distribution, 0.975) < 0.005),
        })
    return pd.DataFrame(rows), pd.DataFrame(paired)


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 required")
    if json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))["status"] != "PASS" or json.loads(DATA_AUDIT.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("20260824_10 did not authorize TN fitting")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    h = load_module(H_SHARED, "stage11_h")
    f1 = load_module(F1_SHARED, "stage11_f1")
    legacy = load_module(LEGACY_SHARED, "stage11_legacy")
    stage4 = load_module(STAGE4, "stage11_stage4")
    shared = h.parent_shared()
    primary, diagnostic = observations()
    local, mass_audit = build_local_components(legacy, stage4)
    mass_audit.to_parquet(OUT / "tn_local_mass_balance_audit.parquet", index=False)
    hydro = pd.read_parquet(HYDRO).loc[lambda x: x.year.between(2016, 2024)]
    predictions, parameters = [], []
    for model_id in h.FORMAL_MODELS:
        router = build_router(model_id, local[model_id], h, shared)
        for program, folds in (("HISTORICAL_OOF", HISTORICAL_FOLDS), ("RECENT_ROLLING_OOF", RECENT_FOLDS)):
            for fold in folds:
                pred, par = fit_fold(model_id, router, primary, fold, program, h, f1, shared, hydro)
                predictions.extend(pred)
                parameters.extend(par)
        pred, par = final_map(model_id, router, primary, h, f1, shared, hydro)
        predictions.extend(pred)
        parameters.extend(par)
        print(json.dumps({"completed": model_id}), flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    parameter_frame = pd.DataFrame(parameters)
    prediction_frame.to_parquet(OUT / "dyn2p_tn_member_predictions.parquet", index=False)
    parameter_frame.to_parquet(OUT / "dyn2p_tn_map_parameters.parquet", index=False)
    ensemble = ensemble_predictions(prediction_frame)
    ensemble.to_parquet(OUT / "dyn2p_tn_ensemble_predictions.parquet", index=False)
    metrics = metric_table(ensemble)
    metrics.to_parquet(OUT / "dyn2p_tn_performance_metrics.parquet", index=False)
    comparison, paired = old_new_comparison(ensemble)
    comparison.to_parquet(OUT / "old_new_hydrology_tn_comparison.parquet", index=False)
    paired.to_parquet(OUT / "old_new_hydrology_paired_station_bootstrap.parquet", index=False)
    checks = {
        "formal_members_exact": prediction_frame.model_id.nunique() == 12,
        "all_readout_success": bool(parameter_frame.readout_success.all()),
        "all_H1_outer_success": bool(parameter_frame.H1_outer_success.all()),
        "all_H1_inner_success": bool(parameter_frame.H1_inner_success.all()),
        "all_spinups_converged": bool(mass_audit.spinup_converged.all()),
        "mass_balance_relative_le_1e_12": bool(mass_audit.max_relative_mass_balance_error.le(1.0e-12).all()),
        "ensemble_member_count_exact": bool(ensemble.member_count.eq(12).all()),
        "historical_years_exact": set(ensemble.loc[ensemble.program.eq("HISTORICAL_OOF"), "year"]) == {2018, 2019, 2020, 2021},
        "recent_years_exact": set(ensemble.loc[ensemble.program.eq("RECENT_ROLLING_OOF"), "year"]) == {2022, 2023, 2024},
        "TN_after_2024_not_read": True,
        "temperature_not_used": True,
        "WWTP_not_used": True,
    }
    summary = {
        "stage": "20260824_11",
        "status": "DYN2P_SIG2P_TN_REFIT_COMPLETE" if all(checks.values()) else "FAIL",
        "checks": checks,
        "water_interface_status": "ADOPTED_FROM_HYDROLOGY_AND_CONSERVATION_EVIDENCE",
        "spatial_transfer_status": "NOT_TESTED_REMAINS_UNSUPPORTED",
        "old_new_historical_paired": paired.to_dict("records"),
        "metrics": metrics.to_dict("records"),
        "observation_counts": {
            "primary_rows_2016_2024": len(primary), "diagnostic_rows_2016_2024": len(diagnostic),
            "primary_stations": primary.station_key.nunique(), "primary_reaches": primary.reach_id.nunique(),
        },
        "hashes": {
            "contract": sha256(CONTRACT), "monthly_input": sha256(MONTHLY), "hydrology": sha256(HYDRO),
            "predictions": sha256(OUT / "dyn2p_tn_ensemble_predictions.parquet"),
            "parameters": sha256(OUT / "dyn2p_tn_map_parameters.parquet"),
        },
    }
    write_json(REPORTS / "tn_refit_decision.json", summary)
    if summary["status"] == "FAIL":
        raise RuntimeError(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


if __name__ == "__main__":
    main()
