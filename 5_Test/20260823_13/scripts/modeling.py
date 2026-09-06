from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def load_q72_component(component: Path, input_path: Path, topology_path: Path):
    spec = importlib.util.spec_from_file_location("q72_clean_component", component)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.INPUT_PATH = input_path
    module.TOPOLOGY_PATH = topology_path
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    module.DYNAMIC_BETA_W = 0.0
    module.configure_engineering_repair(True)
    module.configure_full_gaussian_prior_space(False)
    module.ZERO_PRESERVING_FULL_PRIOR_MODE = True
    module.set_et_feature_block_mode("full")
    module.NETWORK_INPUT_SCALE = 1.0
    module.DETERMINISTIC_SPINUP_MODE = True
    return module


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    good = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[good], pred[good]
    if len(obs) < 2:
        return float("nan")
    den = float(np.sum((obs - obs.mean()) ** 2))
    return float(1 - np.sum((obs - pred) ** 2) / den) if den > EPS else float("nan")


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    good = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[good], pred[good]
    if len(obs) < 3 or np.std(obs) <= EPS or np.mean(obs) <= EPS:
        return float("nan")
    r = float(np.corrcoef(obs, pred)[0, 1])
    alpha = float(np.std(pred) / np.std(obs))
    beta = float(np.mean(pred) / np.mean(obs))
    return float(1 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def metric_dict(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    good = np.isfinite(obs) & np.isfinite(pred) & (obs >= 0) & (pred >= 0)
    obs, pred = obs[good], pred[good]
    log_obs, log_pred = np.log1p(obs), np.log1p(pred)
    return {
        "n": int(len(obs)),
        "NSE": nse(obs, pred),
        "NSE_log": nse(log_obs, log_pred),
        "KGE": kge(obs, pred),
        "PBIAS_pct": float(100 * np.sum(pred - obs) / max(np.sum(obs), EPS)),
        "RMSE_log": float(np.sqrt(np.mean((log_pred - log_obs) ** 2))),
        "MAE_log": float(np.mean(np.abs(log_pred - log_obs))),
    }


def station_metrics(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for (site, comid), part in frame.groupby(["q_site", "comid"], sort=False):
        row = {"q_site": site, "comid": int(comid)}
        row.update(metric_dict(part.Q_obsv_cfs.to_numpy(float), part[pred_col].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def station_macro_log_rmse(frame: pd.DataFrame, pred_col: str) -> float:
    metrics = station_metrics(frame, pred_col)
    return float(metrics.RMSE_log.mean())


def physical_fields(module, base: pd.DataFrame, params: dict[str, float], upstream: np.ndarray) -> dict[str, np.ndarray]:
    days = np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(base.year, base.month)], float)
    seconds = days * 86400.0
    raw = module.simulate_production_variant(
        base,
        seconds,
        base.CumAreaKm2.to_numpy(float),
        prod_capacity=float(params["prod_capacity"]),
        runoff_gamma=float(params["runoff_gamma"]),
        quick_rho=float(params["quick_rho"]),
        base_rho=float(params["base_rho"]),
        base_release=float(params["base_release"]),
        highflow_scale=1.0,
        et_state_operator_mode="storage_withdrawal",
    )
    reaches = base.comid.drop_duplicates().to_numpy()
    n_time = int(len(base) / len(reaches))

    def route(values: np.ndarray) -> np.ndarray:
        return (upstream @ np.asarray(values, float).reshape(len(reaches), n_time)).reshape(-1)

    routed_quick = route(raw["routed_quick_cfs"])
    routed_slow = route(raw["routed_base_cfs"])
    return {
        "local_quick_cfs": np.asarray(raw["routed_quick_cfs"], float),
        "local_slow_cfs": np.asarray(raw["routed_base_cfs"], float),
        "routed_quick_cfs": routed_quick,
        "routed_slow_cfs": routed_slow,
        "routed_total_cfs": np.maximum(routed_quick + routed_slow, 0.0),
        "storage_mm": np.asarray(raw["storage_mm"], float),
        "saturation": np.asarray(raw["saturation"], float),
        "mass_balance_error_mm": np.asarray(raw["mass_balance_error_mm"], float),
    }


def physical_objective(panel: pd.DataFrame, prediction: np.ndarray, train_comids: set[int]) -> float:
    work = panel.loc[
        panel.comid.astype(int).isin(train_comids)
        & panel.year.le(2018)
        & panel.Q_obsv_cfs.notna()
        & panel.Q_obsv_cfs.gt(0),
        ["q_site", "comid", "year", "Q_obsv_cfs"],
    ].copy()
    work["pred"] = prediction[work.index]
    full = station_macro_log_rmse(work, "pred")
    annual = []
    for year in [2015, 2016, 2017, 2018]:
        annual.append(station_macro_log_rmse(work[work.year.eq(year)], "pred"))
    return float(0.5 * full + 0.5 * np.mean(annual))


def tune_physical_parameters(module, base: pd.DataFrame, panel: pd.DataFrame, train_comids: set[int]):
    reaches, _, upstream = module._panel_layout(base)
    current = {
        "prod_capacity": 240.0,
        "runoff_gamma": 2.5,
        "quick_rho": 0.25,
        "base_rho": 0.85,
        "base_release": 0.10,
    }
    grids = {
        "prod_capacity": [160.0, 200.0, 240.0, 300.0, 360.0],
        "runoff_gamma": [1.6, 2.0, 2.5, 3.0, 3.5],
        "quick_rho": [0.10, 0.20, 0.30, 0.45, 0.60],
        "base_rho": [0.70, 0.80, 0.85, 0.90, 0.95],
        "base_release": [0.04, 0.07, 0.10, 0.14, 0.20],
    }
    cache: dict[tuple[float, ...], tuple[float, dict[str, np.ndarray]]] = {}
    history: list[dict[str, float | int | str]] = []

    def evaluate(params: dict[str, float]):
        key = tuple(float(params[name]) for name in grids)
        if key not in cache:
            fields = physical_fields(module, base, params, upstream)
            cache[key] = (physical_objective(panel, fields["routed_total_cfs"], train_comids), fields)
        return cache[key]

    start_score, _ = evaluate(current)
    history.append({"round": 0, "parameter": "initial", "candidate": np.nan, "objective": start_score, **current})
    for round_id in [1, 2]:
        for name, candidates in grids.items():
            trials = []
            for candidate in candidates:
                trial = dict(current)
                trial[name] = float(candidate)
                score, _ = evaluate(trial)
                trials.append((score, float(candidate), trial))
                history.append({"round": round_id, "parameter": name, "candidate": float(candidate), "objective": score, **trial})
            _, _, current = min(trials, key=lambda item: item[0])
    score, fields = evaluate(current)
    return current, score, fields, pd.DataFrame(history), upstream


TRANSFER_FEATURES = [
    "log_q72_total",
    "log_q72_quick",
    "log_q72_slow",
    "q72_quick_fraction",
    "antecedent_wetness",
    "production_saturation",
    "production_storage_scaled",
    "log_ppt",
    "log_aet",
    "log_pet",
    "month_sin",
    "month_cos",
    "log_cumarea",
    "log_incarea",
    "log_length",
    "slope_scaled",
    "elevation_scaled",
]


def make_model_frame(base: pd.DataFrame, physical: dict[str, np.ndarray]) -> pd.DataFrame:
    out = base[[
        "comid", "q_site", "station_id", "year", "month", "Q_obsv_cfs",
        "PPT", "AET", "PET", "IncAreaKm2", "CumAreaKm2", "LENGTHKM",
        "SLOPE", "MaxElSmoCm", "antecedent_wetness",
    ]].copy()
    out["q72_routed_quick_cfs"] = physical["routed_quick_cfs"]
    out["q72_routed_slow_cfs"] = physical["routed_slow_cfs"]
    out["q72_routed_total_cfs"] = physical["routed_total_cfs"]
    out["q72_local_quick_cfs"] = physical["local_quick_cfs"]
    out["q72_local_slow_cfs"] = physical["local_slow_cfs"]
    out["production_storage_mm"] = physical["storage_mm"]
    out["production_saturation"] = physical["saturation"]
    out["production_mass_balance_error_mm"] = physical["mass_balance_error_mm"]
    total = out.q72_routed_total_cfs.to_numpy(float)
    out["q72_quick_fraction"] = np.divide(out.q72_routed_quick_cfs, total, out=np.zeros(len(out)), where=total > EPS)
    out["log_q72_total"] = np.log1p(out.q72_routed_total_cfs.clip(lower=0))
    out["log_q72_quick"] = np.log1p(out.q72_routed_quick_cfs.clip(lower=0))
    out["log_q72_slow"] = np.log1p(out.q72_routed_slow_cfs.clip(lower=0))
    out["production_storage_scaled"] = out.production_storage_mm / 240.0
    out["log_ppt"] = np.log1p(out.PPT.clip(lower=0))
    out["log_aet"] = np.log1p(out.AET.clip(lower=0))
    out["log_pet"] = np.log1p(out.PET.clip(lower=0))
    out["month_sin"] = np.sin(2 * np.pi * out.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out.month / 12)
    out["log_cumarea"] = np.log1p(out.CumAreaKm2.clip(lower=1))
    out["log_incarea"] = np.log1p(out.IncAreaKm2.clip(lower=1))
    out["log_length"] = np.log1p(out.LENGTHKM.fillna(0).clip(lower=0))
    out["slope_scaled"] = out.SLOPE.fillna(0).clip(lower=0) * 1000.0
    out["elevation_scaled"] = out.MaxElSmoCm.fillna(0) / 100000.0
    out[TRANSFER_FEATURES] = out[TRANSFER_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def standardization(train: pd.DataFrame):
    mean = train[TRANSFER_FEATURES].mean()
    std = train[TRANSFER_FEATURES].std(ddof=0).replace(0, 1.0)
    return mean, std


def global_design(frame: pd.DataFrame, mean: pd.Series, std: pd.Series) -> np.ndarray:
    z = ((frame[TRANSFER_FEATURES] - mean) / std).fillna(0.0).to_numpy(float)
    return np.column_stack([np.ones(len(frame)), z])


STATION_SLOPES = ["log_q72_total", "q72_quick_fraction", "month_sin", "month_cos"]


def station_design(frame: pd.DataFrame, stations: list[str], mean: pd.Series, std: pd.Series) -> np.ndarray:
    station_index = {str(site): i for i, site in enumerate(stations)}
    n = len(stations)
    block = np.zeros((len(frame), n * (1 + len(STATION_SLOPES))), float)
    z = ((frame[TRANSFER_FEATURES] - mean) / std).fillna(0.0)
    for row_id, site in enumerate(frame.q_site.astype(str)):
        idx = station_index.get(site)
        if idx is None:
            continue
        offset = idx * (1 + len(STATION_SLOPES))
        block[row_id, offset] = 1.0
        for k, feature in enumerate(STATION_SLOPES, start=1):
            block[row_id, offset + k] = float(z.iloc[row_id][feature])
    return block


def fit_gaussian_map(
    train: pd.DataFrame,
    global_sigma: float,
    stations: list[str] | None = None,
    station_sigma: float | None = None,
):
    mean, std = standardization(train)
    x_global = global_design(train, mean, std)
    penalty = [0.0] + [1.0 / max(global_sigma, EPS)] * len(TRANSFER_FEATURES)
    if stations is None:
        x = x_global
    else:
        x_station = station_design(train, stations, mean, std)
        x = np.column_stack([x_global, x_station])
        ss = float(station_sigma)
        for _ in stations:
            penalty.extend([1.0 / max(ss, EPS)] + [1.0 / max(0.5 * ss, EPS)] * len(STATION_SLOPES))
    y = np.log1p(train.Q_obsv_cfs.to_numpy(float))
    precision = np.square(np.asarray(penalty, float))
    lhs = x.T @ x + np.diag(precision)
    rhs = x.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return {"beta": beta, "mean": mean, "std": std, "global_sigma": global_sigma, "station_sigma": station_sigma, "stations": stations}


def predict_gaussian_map(frame: pd.DataFrame, model: dict) -> np.ndarray:
    x_global = global_design(frame, model["mean"], model["std"])
    if model["stations"] is None:
        x = x_global
    else:
        x = np.column_stack([x_global, station_design(frame, model["stations"], model["mean"], model["std"])])
    return np.maximum(np.expm1(np.clip(x @ model["beta"], -20, 20)), 0.0)


def select_global_sigma(train: pd.DataFrame):
    rows = []
    for sigma in [0.25, 0.5, 1.0, 2.0, 4.0]:
        fold_scores = []
        for year in [2015, 2016, 2017, 2018]:
            fit = train[train.year.lt(year)]
            evaluation = train[train.year.eq(year)].copy()
            model = fit_gaussian_map(fit, sigma)
            evaluation["pred"] = predict_gaussian_map(evaluation, model)
            fold_scores.append(station_macro_log_rmse(evaluation, "pred"))
        rows.append({"global_sigma": sigma, "mean_forward_year_station_macro_RMSE_log": float(np.mean(fold_scores))})
    grid = pd.DataFrame(rows).sort_values("mean_forward_year_station_macro_RMSE_log")
    return float(grid.iloc[0].global_sigma), grid


def select_station_sigma(train: pd.DataFrame, global_sigma: float, stations: list[str]):
    rows = []
    for sigma in [0.10, 0.25, 0.50, 1.00]:
        fold_scores = []
        for year in [2015, 2016, 2017, 2018]:
            fit = train[train.year.lt(year)]
            evaluation = train[train.year.eq(year)].copy()
            model = fit_gaussian_map(fit, global_sigma, stations, sigma)
            evaluation["pred"] = predict_gaussian_map(evaluation, model)
            fold_scores.append(station_macro_log_rmse(evaluation, "pred"))
        rows.append({"station_sigma": sigma, "mean_forward_year_station_macro_RMSE_log": float(np.mean(fold_scores))})
    grid = pd.DataFrame(rows).sort_values("mean_forward_year_station_macro_RMSE_log")
    return float(grid.iloc[0].station_sigma), grid


def recursive_assimilation(
    frame: pd.DataFrame,
    base_prediction: np.ndarray,
    eligible_sites: set[str],
    phi: float,
    gain: float,
    update_through: int,
):
    out = np.log1p(np.maximum(np.asarray(base_prediction, float), 0.0))
    state_prior = np.zeros(len(frame), float)
    state_post = np.zeros(len(frame), float)
    result = out.copy()
    ordered = frame.reset_index().sort_values(["q_site", "year", "month"])
    for site, part in ordered.groupby("q_site", sort=False):
        state = 0.0
        allowed = str(site) in eligible_sites
        for row in part.itertuples():
            pos = int(row.index)
            prior = float(phi) * state
            state_prior[pos] = prior
            result[pos] = out[pos] + prior
            if allowed and int(row.year) <= int(update_through) and np.isfinite(row.Q_obsv_cfs) and row.Q_obsv_cfs > 0:
                innovation = math.log1p(float(row.Q_obsv_cfs)) - result[pos]
                state = prior + float(gain) * innovation
            else:
                state = prior
            state_post[pos] = state
    return np.maximum(np.expm1(np.clip(result, -20, 20)), 0.0), state_prior, state_post


def select_assimilation(train: pd.DataFrame, global_sigma: float, station_sigma: float, stations: list[str]):
    rows = []
    for phi in [0.50, 0.80, 0.95, 0.99]:
        for gain in [0.10, 0.25, 0.50, 0.75]:
            fold_scores = []
            for year in [2015, 2016, 2017, 2018]:
                history_and_eval = train[train.year.le(year)].copy().reset_index(drop=True)
                fit = history_and_eval[history_and_eval.year.lt(year)]
                model = fit_gaussian_map(fit, global_sigma, stations, station_sigma)
                base = predict_gaussian_map(history_and_eval, model)
                pred, _, _ = recursive_assimilation(
                    history_and_eval, base, set(map(str, stations)), phi, gain, update_through=year - 1
                )
                evaluation = history_and_eval[history_and_eval.year.eq(year)].copy()
                evaluation["pred"] = pred[history_and_eval.year.eq(year).to_numpy()]
                fold_scores.append(station_macro_log_rmse(evaluation, "pred"))
            rows.append({"phi": phi, "gain": gain, "mean_frozen_forward_year_station_macro_RMSE_log": float(np.mean(fold_scores))})
    grid = pd.DataFrame(rows).sort_values("mean_frozen_forward_year_station_macro_RMSE_log")
    return float(grid.iloc[0].phi), float(grid.iloc[0].gain), grid
