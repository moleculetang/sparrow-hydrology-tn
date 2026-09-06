"""Refine and publish the H7 unified TN model trained through 2025."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"E:\SPARROW")
HERE = ROOT / "5_Test/20260904_4/scripts"
RUN = ROOT / "5_Test/20260904_7"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
sys.path.insert(0, str(HERE))
from f25_common import atomic_json, atomic_parquet, json_default, load_trial, observations  # noqa: E402
from unified_fit import FoldDesign, JointObjective, UnifiedTNModel, optimize, predict, s41  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kge(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2 or np.std(y) == 0 or np.mean(y) == 0:
        return math.nan
    correlation = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else math.nan
    if not np.isfinite(correlation):
        return math.nan
    alpha = np.std(p, ddof=0) / np.std(y, ddof=0)
    beta = np.mean(p) / np.mean(y)
    return float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def scalar_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    error = p - y
    denominator = np.sum((y - np.mean(y)) ** 2)
    return {
        "rmse_mg_l": float(np.sqrt(np.mean(error**2))),
        "mae_mg_l": float(np.mean(np.abs(error))),
        "nse": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "r2": float(np.corrcoef(y, p)[0, 1] ** 2) if np.std(y) > 0 and np.std(p) > 0 else math.nan,
        "kge": kge(y, p),
        "pbias_percent": float(100.0 * np.sum(error) / np.sum(y)) if np.sum(y) != 0 else math.nan,
    }


def metrics(frame: pd.DataFrame, layer: str, period: str) -> dict[str, object]:
    block = frame.loc[frame.layer.eq(layer) & frame.pred_tn_mg_l.notna()].copy()
    pooled = scalar_metrics(block.tn_mg_l.to_numpy(float), block.pred_tn_mg_l.to_numpy(float))
    station_rows = []
    log_rmse = []
    temporal_correlations = []
    for _, group in block.groupby("station_key"):
        station_rows.append(scalar_metrics(group.tn_mg_l.to_numpy(float), group.pred_tn_mg_l.to_numpy(float)))
        log_rmse.append(float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2))))
        if len(group) >= 2 and group.tn_mg_l.std(ddof=0) > 0 and group.pred_tn_mg_l.std(ddof=0) > 0:
            temporal_correlations.append(float(np.corrcoef(group.tn_mg_l, group.pred_tn_mg_l)[0, 1]))
    by_station = pd.DataFrame(station_rows)
    station_means = block.groupby("station_key")[["tn_mg_l", "pred_tn_mg_l"]].mean()
    spatial_r = float(np.corrcoef(station_means.tn_mg_l, station_means.pred_tn_mg_l)[0, 1])
    tree_log = []
    for _, group in block.groupby("terminal_tree_id"):
        tree_log.append(float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2))))
    result: dict[str, object] = {
        "period": period, "layer": layer, "rows": len(block),
        "stations": int(block.station_key.nunique()), "reaches": int(block.reach_id.nunique()),
        "trees": int(block.terminal_tree_id.nunique()),
        "station_macro_log_rmse": float(np.mean(log_rmse)),
        "tree_macro_log_rmse": float(np.mean(tree_log)),
        "station_mean_spatial_r": spatial_r,
        "station_median_within_time_r": float(np.median(temporal_correlations)),
        **{f"pooled_{name}": value for name, value in pooled.items()},
    }
    for name in ["rmse_mg_l", "mae_mg_l", "nse", "r2", "kge", "pbias_percent"]:
        result[f"station_median_{name}"] = float(by_station[name].median(skipna=True))
    return result


def reach_monthly_product(model: UnifiedTNModel, result: dict[str, object]) -> pd.DataFrame:
    values = torch.tensor(np.asarray(result["physical"], dtype=float))
    named = dict(zip(model.names(), values))
    model.set_fixed_v_f(float(result["selected_v_f"]))
    with torch.no_grad():
        fast, slow = model.local_fluxes_2006(named)
        fast_layers = torch.stack([fast, torch.exp(named["delta_path"]) * fast])
        slow_layers = torch.stack([slow, torch.exp(-named["delta_path"]) * slow])
        routed = model.route_layers(fast_layers, slow_layers, named["v_f"])
        attenuation = torch.exp(-named["v_f"] * model.route_h)
        raw_load = routed["inlet"][0] * attenuation + (fast + slow) * torch.sqrt(attenuation)
        population_local = fast_layers[1] + slow_layers[1]
        population_load = routed["inlet"][1] * attenuation + population_local * torch.sqrt(attenuation)
        raw_log = torch.log1p(1000.0 * raw_load / torch.clamp(model.route_water_outlet, min=1.0e-12))
        population_log = torch.log1p(1000.0 * population_load / torch.clamp(model.route_water_outlet, min=1.0e-12))
        training = torch.tensor([year in result["training_years"] for year, _ in model.month_keys])
        q = model.route_water_outlet / model.route_seconds[:, None]
        center = torch.median(torch.log(torch.clamp(q[training], min=1.0e-12)), dim=0).values
        anomaly = torch.log(torch.clamp(q, min=1.0e-12)) - center[None, :]
        population_log = population_log + named["beta_low"] * torch.minimum(anomaly, torch.zeros_like(anomaly))
        population_log = population_log + named["beta_high"] * torch.maximum(anomaly, torch.zeros_like(anomaly))
        reach_effect = torch.tensor(np.asarray(result["reach_effect"], dtype=float))
        population_log = population_log + reach_effect[None, :]
        raw_concentration = torch.expm1(raw_log).clamp(min=0.0).numpy()
        population_concentration = torch.expm1(population_log).clamp(min=0.0).numpy()
        fast_np, slow_np = fast.numpy(), slow.numpy()
        q_np = q.numpy()
    years = np.repeat([year for year, _ in model.month_keys], 230)
    months = np.repeat([month for _, month in model.month_keys], 230)
    reaches = np.tile(np.arange(1, 231), len(model.month_keys))
    return pd.DataFrame({
        "year": years, "month": months, "reach_id": reaches,
        "routed_q_m3_s": q_np.reshape(-1),
        "local_fast_tn_kg_n_month": fast_np.reshape(-1),
        "local_slow_tn_kg_n_month": slow_np.reshape(-1),
        "raw_process_tn_mg_l": raw_concentration.reshape(-1),
        "population_transferable_tn_mg_l": population_concentration.reshape(-1),
        "product": "F25_TRAINING_SENSITIVITY",
        "quality_flags": "PET_EXTENSION_CONFOUNDED|SOURCE_2025_CARRYFORWARD_CONFOUNDED|2025_TN_INCOMPLETE_DECEMBER|SENSITIVITY_ONLY",
    })


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    for path in [OUT, REPORTS, LOCKS]:
        path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    obs = observations()
    train = obs.loc[obs.year.between(2021, 2025)].copy()
    trials = [load_trial(variant) for variant in range(5)]
    if not all(bool(trial["finite"]) for trial in trials):
        raise RuntimeError("Not all five F25 starts are finite")
    selected_start = int(min(range(5), key=lambda index: float(trials[index]["objective"])))
    best = trials[selected_start]
    model = UnifiedTNModel("sensitivity")
    design = FoldDesign.build("H7", train)
    objective = JointObjective(model, design, train, [2021, 2022, 2023, 2024, 2025])
    anchor_vf = float(best["physical"][model.names().index("v_f")])
    profile_values = sorted({float(np.clip(anchor_vf + delta, 0.0, 0.5)) for delta in (-0.05, 0.0, 0.05)})
    profile = []
    for value in profile_values:
        model.set_fixed_v_f(value)
        candidate = best["physical"].copy()
        candidate[model.names().index("v_f")] = value
        with torch.no_grad():
            score = float(objective.loss(model.to_raw(candidate), torch.tensor(best["gamma"]), torch.tensor(best["site_raw"])))
        profile.append({"v_f": value, "objective": score})
        print(json.dumps({"stage": "v_f_profile", **profile[-1]}), flush=True)
    selected_vf = float(min(profile, key=lambda row: row["objective"])["v_f"])
    best = optimize(
        objective, best["physical"], best["gamma"], best["site_raw"], selected_vf,
        adam_steps=2, lbfgs_steps=8, adam_lr=0.01,
    )
    topup = 0
    while best["combined_kkt"] > 1.0e-5 and topup < 48:
        print(json.dumps({"stage": "kkt_topup", "completed_steps": topup, "kkt": best["combined_kkt"], "objective": best["objective"]}), flush=True)
        best = optimize(
            objective, best["physical"], best["gamma"], best["site_raw"], selected_vf,
            adam_steps=0, lbfgs_steps=8,
        )
        topup += 8
    best.update({
        "capacity": "H7", "fold_id": "F25", "training_years": [2021, 2022, 2023, 2024, 2025],
        "stations": objective.stations, "design": design, "selected_v_f": selected_vf,
    })
    evaluation = obs.loc[obs.year.between(2016, 2025)].copy()
    layers = predict(model, best, evaluation)
    prediction_frames = []
    base = evaluation[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
    for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
        frame = base.copy()
        frame["pred_tn_mg_l"] = np.where(np.isfinite(layers[layer]), np.maximum(np.expm1(layers[layer]), 0.0), np.nan)
        frame["layer"] = layer
        frame["period"] = np.where(frame.year.between(2021, 2025), "F25_calibration", "2016_2020_backcast")
        frame["product"] = "F25_TRAINING_SENSITIVITY"
        prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metric_rows = []
    for period in ["F25_calibration", "2016_2020_backcast"]:
        subset = predictions.loc[predictions.period.eq(period)]
        for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
            metric_rows.append(metrics(subset, layer, period))
    metrics_frame = pd.DataFrame(metric_rows)
    parameter_row = {
        "product": "F25_TRAINING_SENSITIVITY", "capacity": "H7", "selected_start": selected_start,
        "objective": best["objective"], "projected_kkt_max": best["combined_kkt"],
        "gamma_kkt": best["gamma_kkt"], "selected_v_f": selected_vf, "topup_lbfgs_steps": topup,
        "train_rows": len(train), "train_stations": int(train.station_key.nunique()),
        "train_reaches": int(train.reach_id.nunique()), "all_starts_json": json.dumps([
            {"variant": index, "objective": trial["objective"], "kkt": trial["combined_kkt"]}
            for index, trial in enumerate(trials)
        ]), "v_f_profile_json": json.dumps(profile),
    }
    parameter_row.update(dict(zip(model.names(), map(float, best["physical"]))))
    parameters = pd.DataFrame([parameter_row])
    gamma = pd.DataFrame({"feature": design.fields, "gamma": best["gamma"]})
    gamma["product"] = "F25_TRAINING_SENSITIVITY"
    sites = pd.DataFrame({"station_key": best["stations"], "station_residual_log_unit": best["site_effect"]})
    sites["product"] = "F25_TRAINING_SENSITIVITY"
    offsets = pd.DataFrame({"reach_id": np.arange(1, 231), "transferable_offset_log_unit": best["reach_effect"]})
    offsets["product"] = "F25_TRAINING_SENSITIVITY"
    reach_product = reach_monthly_product(model, best)
    diagnostics = model.carrier_diagnostics(torch.tensor(np.asarray(best["physical"], dtype=float)))
    outputs = {
        "station_predictions": OUT / "f25_station_predictions_2016_2025.parquet",
        "metrics": OUT / "f25_performance_metrics.parquet",
        "parameters": OUT / "f25_parameters.parquet",
        "gamma": OUT / "f25_gamma_coefficients.parquet",
        "sites": OUT / "f25_station_residuals.parquet",
        "offsets": OUT / "f25_reach_transferable_offsets.parquet",
        "reach_monthly": OUT / "f25_reach_monthly_1961_2025.parquet",
    }
    for key, frame in {
        "station_predictions": predictions, "metrics": metrics_frame, "parameters": parameters,
        "gamma": gamma, "sites": sites, "offsets": offsets, "reach_monthly": reach_product,
    }.items():
        atomic_parquet(frame, outputs[key])
    flags = ["PET_EXTENSION_CONFOUNDED", "SOURCE_2025_CARRYFORWARD_CONFOUNDED", "2025_TN_INCOMPLETE_DECEMBER", "SENSITIVITY_ONLY"]
    checks = {
        "five_finite_starts": True,
        "projected_kkt_le_1e_5": bool(best["combined_kkt"] <= 1.0e-5),
        "mass_closure_le_1e_10": bool(diagnostics["mass_balance_relative"] <= 1.0e-10),
        "all_predictions_finite": bool(predictions.loc[predictions.layer.ne("gauged_conditional"), "pred_tn_mg_l"].notna().all()),
        "reach_product_complete_1961_2025": len(reach_product) == 65 * 12 * 230,
        "reach_product_nonnegative_finite": bool(np.isfinite(reach_product.population_transferable_tn_mg_l).all() and (reach_product.population_transferable_tn_mg_l >= 0).all()),
    }
    status = "PASS_F25_TRAINING_SENSITIVITY" if all(checks.values()) else "F25_NUMERIC_REVIEW_REQUIRED"
    report = {
        "stage": "20260904_7", "status": status, "checks": checks, "quality_flags": flags,
        "training": {"years": [2021, 2022, 2023, 2024, 2025], "rows": len(train), "stations": int(train.station_key.nunique()), "2025_rows": int((train.year == 2025).sum())},
        "architecture": "joint process + H7 transferable Reach head + strongly-shrunk centered station residual",
        "hydrology": "20260828_38 forced 1961-2025 sensitivity product",
        "optimization": {"selected_start": selected_start, "all_starts": json.loads(parameter_row["all_starts_json"]), "v_f_profile": profile, "final_objective": best["objective"], "final_kkt": best["combined_kkt"], "topup_steps": topup},
        "carrier_diagnostics": diagnostics,
        "metrics": metrics_frame.to_dict("records"),
        "runtime": {"python": sys.executable, "torch": torch.__version__, "seconds": time.perf_counter() - started, "threads": torch.get_num_threads(), "rss_gib": s41.s28.memory_gib()[0], "peak_gib": s41.s28.memory_gib()[1]},
        "output_hashes": {key: sha256(path) for key, path in outputs.items()},
    }
    atomic_json(report, REPORTS / "f25_training_report.json")
    atomic_json({
        "stage": "20260904_7", "status": status, "product": "F25_TRAINING_SENSITIVITY",
        "quality_flags": flags, "report_sha256": sha256(REPORTS / "f25_training_report.json"),
    }, LOCKS / "f25_training_sensitivity_lock.json")
    lines = [
        "# `20260904_7` 2025进入训练的TN版本", "", f"状态：`{status}`。", "",
        "该版本用2021–2025共五年TN联合拟合过程参数、H7可迁移空间属性头与强收缩站点残差。2025不是只用于评价，而是正式进入本敏感性版本的目标函数。", "",
        "由于2025水文PET延伸未通过正式相关门、2025 N源项沿用2024且缺少12月TN，本产品固定标记为`SENSITIVITY_ONLY`，不替换1961–2024正式产品。", "",
        "| period | layer | rows | station log-RMSE | RMSE | MAE | pooled NSE | pooled R² | pooled KGE | PBIAS | station median NSE | station mean spatial r |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_frame.itertuples(index=False):
        lines.append(f"| {row.period} | {row.layer} | {row.rows} | {row.station_macro_log_rmse:.4f} | {row.pooled_rmse_mg_l:.3f} | {row.pooled_mae_mg_l:.3f} | {row.pooled_nse:.3f} | {row.pooled_r2:.3f} | {row.pooled_kge:.3f} | {row.pooled_pbias_percent:.1f}% | {row.station_median_nse:.3f} | {row.station_mean_spatial_r:.3f} |")
    lines += ["", f"最终KKT：`{best['combined_kkt']:.3e}`；质量闭合相对误差：`{diagnostics['mass_balance_relative']:.3e}`。", ""]
    (REPORTS / "technical_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "metrics": report["metrics"]}, ensure_ascii=False, indent=2, default=json_default), flush=True)
    if status != "PASS_F25_TRAINING_SENSITIVITY":
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
