from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(r"E:\SPARROW"); RUN = ROOT / "5_Test" / "20260826_7"; OUT = RUN / "outputs"; REPORT = RUN / "reports"
sys.path[:0] = [str(ROOT / "5_Test" / "20260825_3" / "scripts"), str(ROOT / "5_Test" / "20260825_5" / "scripts")]
from hydrology_core import load_topology  # noqa: E402
from run_stage5 import build_support, station_metrics  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DEVELOPMENT_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
ATTRIBUTES = ROOT / "5_Test" / "20260826_6" / "outputs" / "spatial_attributes_raw.parquet"
PARENT = ROOT / "5_Test" / "20260825_7" / "outputs" / "tn_hydrology_bridge_daily_2006_2022.parquet"


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): d.update(b)
    return d.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    result = cumulative[1:].copy(); result[window:] = cumulative[window + 1:] - cumulative[1:-window]
    return result


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)) if len(obs) > 1 else np.nan


def summary(candidate: str, obs: np.ndarray, pred: np.ndarray, stations: pd.DataFrame) -> dict[str, object]:
    valid = np.isfinite(obs) & np.isfinite(pred); pooled = nse(obs[valid], pred[valid]); station = station_metrics(candidate, obs, pred, stations)
    return {"candidate": candidate, "pooled_NSE": pooled, "station_median_NSE": float(station.NSE.median()), "station_mean_NSE": float(station.NSE.mean()), "pooled_log_RMSE": float(np.sqrt(np.mean((np.log1p(pred[valid]) - np.log1p(obs[valid])) ** 2))), "station_median_absolute_PBIAS_pct": float(station.PBIAS_pct.abs().median()), "stations": len(station)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    reach_ids = np.arange(1, 231); order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGES); stations = gauges.loc[gauges.topology_representative].drop_duplicates("station_norm").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True); support = build_support(stations, reach_ids, order, downstream)
    area = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float); contributing_area = support * area[None, :]; station_area = contributing_area.sum(axis=1); weights = contributing_area / station_area[:, None]
    dates = pd.date_range("2010-01-01", "2018-12-31"); forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"]); forcing.date = pd.to_datetime(forcing.date)
    pivot = lambda field: forcing.pivot(index="date", columns="reach_id", values=field).reindex(index=dates, columns=reach_ids).to_numpy(float)
    p, pet = pivot("precipitation_daily_mm") @ weights.T, pivot("pet_fao56_mm_day") @ weights.T
    features = [p, pet]; feature_names = ["precipitation_1d", "pet_1d"]
    for window in [3, 7, 14, 30, 90, 180, 365]: features.append(rolling_sum(p, window)); feature_names.append(f"precipitation_{window}d")
    for window in [7, 30, 90, 365]: features.append(rolling_sum(pet, window)); feature_names.append(f"pet_{window}d")
    day = dates.dayofyear.to_numpy(); features.extend([np.repeat(np.sin(2 * np.pi * day / 365.25)[:, None], len(stations), axis=1), np.repeat(np.cos(2 * np.pi * day / 365.25)[:, None], len(stations), axis=1)]); feature_names.extend(["season_sin", "season_cos"])
    static = pd.read_parquet(ATTRIBUTES).set_index("reach_id").reindex(reach_ids); static_names = [c for c in static.columns if c != "reach_id"]; static_values = static[static_names].to_numpy(float); center = np.nanmean(static_values, axis=0); scale = np.nanstd(static_values, axis=0); scale[scale == 0] = 1; static_station = weights @ ((static_values - center) / scale)
    dynamic = np.stack(features, axis=2); static_repeated = np.broadcast_to(static_station[None, :, :], (len(dates), len(stations), len(static_names))); design = np.concatenate([dynamic, static_repeated], axis=2)
    obs_frame = pd.read_parquet(DEVELOPMENT_Q); obs_frame.date = pd.to_datetime(obs_frame.date); observed = obs_frame.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(float); observed_mm = observed * 86400 / (station_area[None, :] * 1e6) * 1000; target = np.log1p(observed_mm)
    train_time = dates.year <= 2016; evaluation_time = dates.year >= 2017; train_mask = train_time[:, None] & np.isfinite(target); eval_mask = evaluation_time[:, None] & np.isfinite(target)
    x_train = design[train_mask]; y_train = target[train_mask]
    tree_ids = stations.terminal_tree.to_numpy(int); sample_tree = np.broadcast_to(tree_ids[None, :], target.shape)[train_mask]; counts = pd.Series(sample_tree).value_counts().to_dict(); sample_weight = np.asarray([1.0 / counts[int(tree)] for tree in sample_tree]); sample_weight *= len(sample_weight) / sample_weight.sum()
    model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_iter=350, max_leaf_nodes=31, max_depth=10, min_samples_leaf=40, l2_regularization=2.0, random_state=260826, early_stopping=True, validation_fraction=0.12, n_iter_no_change=25)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    prediction_log = model.predict(design[evaluation_time].reshape(-1, design.shape[2])).reshape(int(evaluation_time.sum()), len(stations)); prediction_mm = np.maximum(0, np.expm1(prediction_log)); prediction_q = prediction_mm * station_area[None, :] * 1e6 / 1000 / 86400
    obs_eval = observed[evaluation_time]
    parent = pd.read_parquet(PARENT, columns=["date", "reach_id", "local_q0_m3_s", "local_q1_m3_s", "local_q2_m3_s"]); parent.date = pd.to_datetime(parent.date); parent = parent.loc[parent.date.dt.year.between(2017, 2018)].sort_values(["date", "reach_id"]); local_parent = parent[["local_q0_m3_s", "local_q1_m3_s", "local_q2_m3_s"]].sum(axis=1).to_numpy().reshape(int(evaluation_time.sum()), 230); parent_q = local_parent @ support.T
    performance = pd.DataFrame([summary("REGIONAL_HGB_LAG_CEILING", obs_eval, prediction_q, stations), summary("HBV3_PARENT", obs_eval, parent_q, stations)]); performance.to_parquet(OUT / "development_holdout_performance.parquet", index=False)
    station_ml = station_metrics("REGIONAL_HGB_LAG_CEILING", obs_eval, prediction_q, stations); station_parent = station_metrics("HBV3_PARENT", obs_eval, parent_q, stations); pd.concat([station_ml, station_parent]).to_parquet(OUT / "development_holdout_station_performance.parquet", index=False)
    predictions = pd.DataFrame({"date": np.repeat(dates[evaluation_time].to_numpy(), len(stations)), "station_norm": np.tile(stations.station_norm.to_numpy(), int(evaluation_time.sum())), "reach_id": np.tile(stations.reach_id.to_numpy(int), int(evaluation_time.sum())), "terminal_tree": np.tile(tree_ids, int(evaluation_time.sum())), "q_observed_m3_s": obs_eval.reshape(-1), "q_ml_ceiling_m3_s": prediction_q.reshape(-1), "q_hbv_parent_m3_s": parent_q.reshape(-1)}); predictions.to_parquet(OUT / "development_holdout_predictions.parquet", index=False)
    feature_registry = {"dynamic_features": feature_names, "static_features": static_names, "previous_observed_discharge_used": False, "station_identity_used": False, "terminal_tree_weighted_training": True, "fitted_iterations": int(model.n_iter_), "authorization": "TOTAL_FLOW_INFORMATION_CEILING_ONLY_NOT_TN_INTERFACE"}; write_json(REPORT / "ml_feature_and_authorization_registry.json", feature_registry)
    ml = performance.loc[performance.candidate == "REGIONAL_HGB_LAG_CEILING"].iloc[0]; parent_row = performance.loc[performance.candidate == "HBV3_PARENT"].iloc[0]
    decision = {"stage": "20260826_7", "status": "PASS_ML_TOTAL_FLOW_CEILING_COMPLETED", "ml_pooled_NSE": float(ml.pooled_NSE), "ml_station_median_NSE": float(ml.station_median_NSE), "parent_pooled_NSE": float(parent_row.pooled_NSE), "parent_station_median_NSE": float(parent_row.station_median_NSE), "ML_TN_interface_authorized": False, "retrospective_discharge_used": False, "TN_used": False, "authorized_successor": "20260826_8"}; write_json(REPORT / "stage7_decision.json", decision)
    (REPORT / "technical_report.md").write_text("# 20260826_7 区域机器学习总流量上限\n\n状态：`%s`。模型仅使用上游气象滞后和静态属性，不读取历史流量作为输入，也没有站点ID。它在2017–2018开发期留出段的作用只是判断forcing仍含多少非线性总Q信号。无论性能如何，该模型不守恒、没有预注册水库语义，因此隐藏结构不得进入TN。\n\n%s\n" % (decision["status"], performance.to_markdown(index=False)), encoding="utf-8")
    write_json(REPORT / "integrity.json", {str(p): sha256(p) for p in [RUN / "experiment_contract.json", OUT / "development_holdout_performance.parquet", REPORT / "ml_feature_and_authorization_registry.json", REPORT / "stage7_decision.json", REPORT / "technical_report.md"]}); print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
