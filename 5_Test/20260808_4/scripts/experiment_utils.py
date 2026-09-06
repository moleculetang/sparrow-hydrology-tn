from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1.0e-6
CFS_PER_M3S = 35.3146667


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_component(path: Path, unique_name: str):
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_map_ridge_quiet(component, train: pd.DataFrame, stations: list[str], mean: pd.Series, std: pd.Series, hp: dict[str, float]) -> np.ndarray:
    x, y = component.build_matrix(train, stations, mean, std)
    x_parts = [x]
    y_parts = [y]
    if hp["anomaly_weight"] > 0:
        sqrt_w = float(np.sqrt(hp["anomaly_weight"]))
        anomaly_x, anomaly_y = [], []
        for _, pos in train.groupby("q_site", sort=False).indices.items():
            pos = np.array(pos, dtype=int)
            local_x, local_y = x[pos, :], y[pos]
            sd = float(np.std(local_y))
            if len(pos) < 12 or sd <= EPS:
                continue
            anomaly_x.append((local_x - local_x.mean(axis=0, keepdims=True)) / sd * sqrt_w)
            anomaly_y.append((local_y - local_y.mean()) / sd * sqrt_w)
        if anomaly_x:
            x_parts.append(np.vstack(anomaly_x))
            y_parts.append(np.concatenate(anomaly_y))
    if hp["flow_contrast_weight"] > 0:
        sqrt_w = float(np.sqrt(hp["flow_contrast_weight"]))
        contrast_x, contrast_y = [], []
        for _, pos in train.groupby("q_site", sort=False).indices.items():
            pos = np.array(pos, dtype=int)
            if len(pos) < 36:
                continue
            local_x, local_y = x[pos, :], y[pos]
            local_q = train.iloc[pos]["Q_obsv_cfs"].to_numpy(float)
            q25, q75, q90 = np.nanquantile(local_q, [0.25, 0.75, 0.90])
            low, high = local_q <= q25, local_q >= q75
            mid, peak = (local_q >= q25) & (local_q <= q75), local_q >= q90
            if low.sum() >= 6 and high.sum() >= 6:
                contrast_x.append((local_x[high].mean(axis=0) - local_x[low].mean(axis=0)) * sqrt_w)
                contrast_y.append((local_y[high].mean() - local_y[low].mean()) * sqrt_w)
            if peak.sum() >= 3 and mid.sum() >= 12:
                contrast_x.append((local_x[peak].mean(axis=0) - local_x[mid].mean(axis=0)) * sqrt_w)
                contrast_y.append((local_y[peak].mean() - local_y[mid].mean()) * sqrt_w)
        if contrast_x:
            x_parts.append(np.vstack(contrast_x))
            y_parts.append(np.asarray(contrast_y, dtype=float))
    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    n_fixed_base = 1 + len(component.FIXED_FEATURES)
    n_group = len(component.SPATIAL_GROUP_GATES) * len(component.SPATIAL_GROUP_FEATURES)
    n_fixed = n_fixed_base + n_group
    n_station = len(stations)
    n_slope = len(component.RANDOM_SLOPE_FEATURES) * n_station
    n_regime = len(component.REGIME_GATES) * len(component.REGIME_SLOPE_FEATURES) * n_station
    penalty = np.zeros(n_fixed + n_station + n_slope + n_regime, dtype=float)
    penalty[1:n_fixed_base] = 1.0 / max(hp["fixed_sigma"], EPS)
    for i, feature in enumerate(component.FIXED_FEATURES, start=1):
        if feature in component.PRODUCTION_FEATURES:
            penalty[i] = 1.0 / max(hp["production_sigma"], EPS)
        if feature in component.MULTISTORE_FEATURES:
            penalty[i] = 1.0 / max(hp["multistore_sigma"], EPS)
        if feature in component.HYSTERESIS_FEATURES:
            penalty[i] = 1.0 / max(hp["hysteresis_sigma"], EPS)
    penalty[n_fixed_base:n_fixed] = 1.0 / max(hp["group_sigma"], EPS)
    penalty[n_fixed:n_fixed + n_station] = 1.0 / max(hp["station_sigma"], EPS)
    penalty[n_fixed + n_station:n_fixed + n_station + n_slope] = 1.0 / max(hp["slope_sigma"], EPS)
    penalty[n_fixed + n_station + n_slope:] = 1.0 / max(hp["regime_slope_sigma"], EPS)
    x_aug = np.vstack([x, np.diag(penalty)])
    y_aug = np.concatenate([y, np.zeros(len(penalty), dtype=float)])
    beta, *_ = np.linalg.lstsq(x_aug, y_aug, rcond=None)
    return beta


STATE_OUTPUT_COLUMNS = [
    "antecedent_wetness", "sas_storage_mm", "sas_young_fraction",
    "sas_young_cfs", "sas_old_release_cfs", "production_storage_mm",
    "production_saturation", "production_quick_cfs", "production_base_cfs",
    "production_overflow_cfs", "routed_quick_cfs", "routed_base_cfs",
    "production_highflow_mass_cfs", "production_month_end_quick_store_mm",
    "production_month_end_base_store_mm", "ms_slow_large_storage_mm",
    "ms_slow_large_routed_quick_cfs", "ms_slow_large_routed_base_cfs",
    "ms_slow_large_month_end_quick_store_mm",
    "ms_slow_large_month_end_base_store_mm",
]


def fit_three_fold_oof(featured: pd.DataFrame, component, config: dict[str, object], scenario: str) -> pd.DataFrame:
    design = component.prepare_design(featured)
    stations = sorted(design["q_site"].astype(str).unique())
    hp = {key: float(value) for key, value in config["fixed_hyperparameters"].items()}
    parts = []
    for fold in config["folds"]:
        train = design[design["year"].le(int(fold["train_end"]))].copy()
        evaluation = design[design["year"].between(int(fold["eval_start"]), int(fold["eval_end"]))].copy()
        mean, std = component.standardize_fit(train)
        beta = fit_map_ridge_quiet(component, train, stations, mean, std, hp)
        prediction = np.exp(np.clip(component.predict_log(evaluation, beta, stations, mean, std), -20, 20))
        keep = ["comid", "q_site", "year", "month", "Q_obsv_cfs", "Q_ma_cfs", "PPT", "AET", "PET"]
        keep += [column for column in STATE_OUTPUT_COLUMNS if column in evaluation.columns]
        part = evaluation[keep].copy()
        part = part.rename(columns={"comid": "reach_id", "q_site": "station_name", "Q_obsv_cfs": "observed_cfs"})
        part["predicted_cfs"] = prediction
        part["scenario"] = scenario
        part["fold_id"] = str(fold["fold_id"])
        part["train_end"] = int(fold["train_end"])
        part["eval_start"] = int(fold["eval_start"])
        part["eval_end"] = int(fold["eval_end"])
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["reach_id"] = out["reach_id"].astype(int)
    if out.duplicated(["station_name", "year", "month"]).any():
        raise RuntimeError(f"Duplicate OOF key in {scenario}")
    return add_flow_class(out)


def add_flow_class(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["flow_class"] = "middle"
    for _, indices in out.groupby(["station_name", "fold_id"], sort=False).groups.items():
        ordered = out.loc[list(indices)].sort_values(["observed_cfs", "year", "month"]).index.to_list()
        count = max(1, int(math.floor(0.25 * len(ordered))))
        out.loc[ordered[:count], "flow_class"] = "low"
        out.loc[ordered[-count:], "flow_class"] = "high"
    return out


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or np.std(obs) <= 0 or np.mean(obs) == 0 or np.std(pred) <= 0:
        return float("nan")
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    obs = frame["observed_cfs"].to_numpy(float)
    pred = frame["predicted_cfs"].to_numpy(float)
    log_obs = np.log(np.clip(obs, EPS, None))
    log_pred = np.log(np.clip(pred, EPS, None))
    raw_sst = float(np.sum((obs - obs.mean()) ** 2))
    log_sst = float(np.sum((log_obs - log_obs.mean()) ** 2))
    return {
        "rows": int(len(frame)), "stations": int(frame["station_name"].nunique()),
        "NSE_raw": float(1.0 - np.sum((pred - obs) ** 2) / raw_sst) if raw_sst > 0 else float("nan"),
        "NSE_log": float(1.0 - np.sum((log_pred - log_obs) ** 2) / log_sst) if log_sst > 0 else float("nan"),
        "KGE": kge(obs, pred),
        "PBIAS_pct": float(100.0 * np.sum(pred - obs) / np.sum(obs)),
        "log_RMSE": float(np.sqrt(np.mean((log_pred - log_obs) ** 2))),
        "median_log_bias": float(np.median(log_pred - log_obs)),
        "absolute_error_cfs": float(np.sum(np.abs(pred - obs))),
        "observed_volume_cfs_sum": float(np.sum(obs)),
    }


def scenario_metric_rows(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows, station_rows = [], []
    for scenario, part in oof.groupby("scenario", sort=True):
        row = {"scenario": scenario}
        for label, subset in [("overall", part), ("lowflow", part[part["flow_class"].eq("low")]), ("highflow", part[part["flow_class"].eq("high")])]:
            row.update({f"{label}_{key}": value for key, value in metrics(subset).items()})
        overall_rows.append(row)
        for station, station_part in part.groupby("station_name", sort=True):
            station_row = {
                "scenario": scenario, "station_name": station,
                "reach_id": int(station_part["reach_id"].median()),
                "Q_ma_cfs": float(station_part["Q_ma_cfs"].median()),
            }
            for label, subset in [("overall", station_part), ("lowflow", station_part[station_part["flow_class"].eq("low")]), ("highflow", station_part[station_part["flow_class"].eq("high")])]:
                station_row.update({f"{label}_{key}": value for key, value in metrics(subset).items()})
            station_rows.append(station_row)
    return pd.DataFrame(overall_rows), pd.DataFrame(station_rows)


def residual_acf_rows(oof: pd.DataFrame, lags: tuple[int, ...] = (1, 3, 6)) -> pd.DataFrame:
    rows = []
    for (scenario, station, fold_id), part in oof.groupby(["scenario", "station_name", "fold_id"], sort=True):
        part = part.sort_values(["year", "month"])
        dates = part["year"].to_numpy(int) * 12 + part["month"].to_numpy(int)
        residual = np.log(np.clip(part["predicted_cfs"].to_numpy(float), EPS, None)) - np.log(np.clip(part["observed_cfs"].to_numpy(float), EPS, None))
        for lag in lags:
            if len(part) <= lag:
                continue
            consecutive = dates[lag:] - dates[:-lag] == lag
            left, right = residual[:-lag][consecutive], residual[lag:][consecutive]
            if len(left) >= 6 and np.std(left) > 0 and np.std(right) > 0:
                rows.append({"scenario": scenario, "station_name": station, "fold_id": fold_id, "lag": lag, "pairs": int(len(left)), "acf": float(np.corrcoef(left, right)[0, 1])})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    summary = detail.groupby(["scenario", "lag"], as_index=False).apply(
        lambda g: pd.Series({"station_fold_count": len(g), "pair_count": int(g["pairs"].sum()), "median_abs_acf": float(g["acf"].abs().median()), "weighted_abs_acf": float(np.average(g["acf"].abs(), weights=g["pairs"]))}),
        include_groups=False,
    ).reset_index(drop=True)
    return summary


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
