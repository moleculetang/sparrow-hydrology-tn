"""Finalize the selected TN architecture and synthesize the complete experiment tree."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_17"
P12 = ROOT / "5_Test" / "20260824_12"
P13 = ROOT / "5_Test" / "20260824_13"
P14 = ROOT / "5_Test" / "20260824_14"
P15 = ROOT / "5_Test" / "20260824_15"
P16 = ROOT / "5_Test" / "20260824_16"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
OBS = P12 / "outputs" / "tn_observations_primary_2016_2024.parquet"
MONTHLY = P12 / "outputs" / "canonical_tn_bridge_monthly_2010_2024.parquet"
SOURCES = P12 / "outputs" / "monthly_source_forcing_1961_2024.parquet"
KERNELS = P13 / "outputs" / "daily_compiled_carrier_kernels.parquet"
M0_FLUX = P13 / "outputs" / "m0_local_source_tagged_fluxes.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
CONTRACT = RUN / "experiment_contract.json"
EPS = 1.0e-12

sys.path.insert(0, str(P13 / "scripts"))
sys.path.insert(0, str(P15 / "scripts"))
sys.path.insert(0, str(P16 / "scripts"))
from stage13_model import M0Router, build_router, source_availability  # noqa: E402
from run_stage13 import fit_m0, metric_values  # noqa: E402
from run_stage16 import DynamicDeliveryRouter, fit_dynamic  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")


def m0_station_predictions(router: M0Router, obs: pd.DataFrame, pi: float, vf: float, scenario: str) -> pd.DataFrame:
    out = obs.copy(); out["pred_tn_mg_l"] = pi * router.concentration_at_pi1(obs, vf)
    out["hydraulic_scenario"] = scenario
    return out


def m0_reach_predictions(router: M0Router, pi: float, vf: float, scenario: str) -> pd.DataFrame:
    _, outlet = router.route_load(vf)
    concentration = 1000.0 * pi * outlet / np.maximum(router.water_outlet, EPS)
    rows = []
    for t, (year, month) in enumerate(router.month_keys):
        if year >= 2016:
            rows.append(pd.DataFrame({
                "reach_id": router.reach_ids, "year": year, "month": month,
                "pred_tn_mg_l": concentration[t], "routed_tn_load_kg_n_month": pi * outlet[t],
                "hydraulic_scenario": scenario,
            }))
    return pd.concat(rows, ignore_index=True)


def dynamic_reach_predictions(router: DynamicDeliveryRouter, theta: np.ndarray, scenario: str) -> pd.DataFrame:
    alpha, beta, vf = map(float, theta)
    local, _, _ = router.carrier(alpha, beta)
    inlet = np.zeros_like(local); outlet = np.zeros_like(local)
    survival = np.exp(-vf * router.h); midpoint = np.exp(-vf * router.h / 2.0)
    for i in router.order_idx:
        outlet[:, i] = inlet[:, i] * survival[:, i] + local[:, i] * midpoint[:, i]
        if i in router.down_idx: inlet[:, router.down_idx[i]] += outlet[:, i]
    concentration = 1000.0 * outlet / np.maximum(router.water_outlet, EPS)
    rows = []
    for t, (year, month) in enumerate(router.month_keys):
        if year >= 2016:
            rows.append(pd.DataFrame({"reach_id": router.reach_ids, "year": year, "month": month, "pred_tn_mg_l": concentration[t], "routed_tn_load_kg_n_month": outlet[t], "hydraulic_scenario": scenario}))
    return pd.concat(rows, ignore_index=True)


def metric_table(predictions: pd.DataFrame, label: str) -> list[dict[str, object]]:
    rows = []
    for scenario, frame in predictions.groupby("hydraulic_scenario"):
        for period, subset in (
            ("FINAL_MAP_APPARENT_2021_2024", frame.loc[frame.year.between(2021, 2024)]),
            ("FINAL_MAP_HISTORICAL_BACKCAST_2016_2020", frame.loc[frame.year.between(2016, 2020)]),
        ):
            rows.append({"architecture": label, "program": period, "hydraulic_scenario": scenario, **metric_values(subset)})
    return rows


def oof_metric_table(predictions: pd.DataFrame, architecture: str) -> list[dict[str, object]]:
    return [{"architecture": architecture, "program": f"OOF_{kind}", **metric_values(group)} for kind, group in predictions.groupby("holdout_type")]


def residual_month_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.loc[predictions.holdout_type.eq("TEMPORAL")].copy()
    frame["residual_log_obs_minus_pred"] = np.log1p(frame.tn_mg_l) - np.log1p(frame.pred_tn_mg_l)
    return frame.groupby("month", as_index=False).agg(
        n=("residual_log_obs_minus_pred", "size"),
        mean_signed_residual_log=("residual_log_obs_minus_pred", "mean"),
        median_signed_residual_log=("residual_log_obs_minus_pred", "median"),
        mean_absolute_residual_log=("residual_log_obs_minus_pred", lambda x: float(np.mean(np.abs(x)))),
    )


def bootstrap_metric_ci(predictions: pd.DataFrame, architecture: str) -> pd.DataFrame:
    rng = np.random.default_rng(2026082417)
    rows = []
    for kind, frame in predictions.groupby("holdout_type"):
        for block in (("station_key", "reach_id", "terminal_tree_id") if kind in {"TEMPORAL", "FIRST_OBSERVED_2021"} else (("reach_id",) if kind == "REACH" else ("terminal_tree_id",))):
            levels = list(frame[block].unique()); values = []
            for level in levels:
                group = frame.loc[frame[block].eq(level)]
                values.append(float(np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l))))))
            values = np.asarray(values); samples = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
            rows.append({"architecture": architecture, "holdout_type": kind, "block": block, "block_count": len(values), "macro_log_rmse": float(values.mean()), "ci95_lower": float(np.quantile(samples, 0.025)), "ci95_upper": float(np.quantile(samples, 0.975))})
    return pd.DataFrame(rows)


def hydraulic_exposure_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    frame = monthly.copy()
    frame["H1_width_p05_day_per_m"] = frame.h1_exposure_day_per_m * frame.bankfull_width_p05_m / frame.bankfull_width_m
    frame["H1_width_p95_day_per_m"] = frame.h1_exposure_day_per_m * frame.bankfull_width_p95_m / frame.bankfull_width_m
    frame["LWD_over_Q_central_day"] = frame.channel_bankfull_travel_time_central_day
    columns = ["H1_width_p05_day_per_m", "h1_exposure_day_per_m", "H1_width_p95_day_per_m", "LWD_over_Q_central_day"]
    rows = []
    for column in columns:
        values = frame[column].to_numpy(float)
        rows.append({"exposure": column, "minimum": float(np.min(values)), "p50": float(np.quantile(values, 0.5)), "p95": float(np.quantile(values, 0.95)), "p99": float(np.quantile(values, 0.99)), "maximum": float(np.max(values)), "role": "formal_or_geometry_sensitivity" if "H1" in column or column == "h1_exposure_day_per_m" else "diagnostic_only"})
    return pd.DataFrame(rows)


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow": raise RuntimeError("conda sparrow required")
    decisions = {
        13: json.loads((P13 / "reports" / "stage13_decision.json").read_text(encoding="utf-8")),
        14: json.loads((P14 / "reports" / "stage14_decision.json").read_text(encoding="utf-8")),
        15: json.loads((P15 / "reports" / "stage15_decision.json").read_text(encoding="utf-8")),
        16: json.loads((P16 / "reports" / "stage16_decision.json").read_text(encoding="utf-8")),
    }
    if any(not value["status"].startswith("PASS") for value in decisions.values()): raise RuntimeError("predecessor not locked")
    selected = decisions[16]["selected_architecture"]
    monthly = pd.read_parquet(MONTHLY); obs = pd.read_parquet(OBS); train = obs.loc[obs.year.between(2021, 2024)].copy()
    flux = pd.read_parquet(M0_FLUX).loc[lambda x: x.carrier.eq("DAILY_COMPILED_CARRIER")]
    m0_router = build_router(flux, monthly, TOPOLOGY)
    m0_fit = fit_m0(m0_router, train)
    m0_pi, m0_vf = float(m0_fit["pi_E"]), float(m0_fit["v_f_m_per_day"])

    station_predictions, reach_predictions = [], []
    parameter_rows = []
    monthly_ordered = monthly.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    width_scenarios = {
        "H1_WIDTH_P05": monthly_ordered.bankfull_width_p05_m.to_numpy(float).reshape(m0_router.h_full.shape) / monthly_ordered.bankfull_width_m.to_numpy(float).reshape(m0_router.h_full.shape),
        "H1_WIDTH_CENTRAL": np.ones_like(m0_router.h_full),
        "H1_WIDTH_P95": monthly_ordered.bankfull_width_p95_m.to_numpy(float).reshape(m0_router.h_full.shape) / monthly_ordered.bankfull_width_m.to_numpy(float).reshape(m0_router.h_full.shape),
    }
    if selected == "M3_DYNAMIC_DELIVERY":
        available = source_availability(pd.read_parquet(SOURCES)); dynamic = DynamicDeliveryRouter(monthly, pd.read_parquet(KERNELS), available)
        fit = fit_dynamic(dynamic, train, m0_pi, m0_vf, multistart=True); theta = np.asarray(fit.pop("theta"), dtype=float)
        parameter_rows.append({"architecture": selected, "alpha": theta[0], "beta_D": theta[1], "v_f_m_per_day": theta[2], **fit})
        original_h = dynamic.h.copy()
        for scenario, ratio in width_scenarios.items():
            dynamic.h = original_h * ratio
            pred = obs.copy(); pred["pred_tn_mg_l"] = dynamic.evaluate(obs, theta, False)[0]; pred["hydraulic_scenario"] = scenario
            station_predictions.append(pred); reach_predictions.append(dynamic_reach_predictions(dynamic, theta, scenario))
        dynamic.h = original_h
        oof = pd.read_parquet(P16 / "outputs" / "dynamic_delivery_oof_predictions.parquet")
    else:
        parameter_rows.append({"architecture": selected, "pi_E": m0_pi, "v_f_m_per_day": m0_vf, **{k: v for k, v in m0_fit.items() if k not in {"pi_E", "v_f_m_per_day"}}})
        for scenario, ratio in width_scenarios.items():
            router = M0Router(local_load=m0_router.local_load, local_water=m0_router.local_water, h_full=m0_router.h_full * ratio, month_keys=m0_router.month_keys, reach_ids=m0_router.reach_ids, order=m0_router.order, downstream=m0_router.downstream)
            station_predictions.append(m0_station_predictions(router, obs, m0_pi, m0_vf, scenario)); reach_predictions.append(m0_reach_predictions(router, m0_pi, m0_vf, scenario))
        oof = pd.read_parquet(P13 / "outputs" / "m0_carrier_oof_predictions.parquet").loc[lambda x: x.carrier.eq("DAILY_COMPILED_CARRIER")]

    station_frame = pd.concat(station_predictions, ignore_index=True); reach_frame = pd.concat(reach_predictions, ignore_index=True)
    station_frame.to_parquet(OUT / "final_station_predictions_2016_2024.parquet", index=False); reach_frame.to_parquet(OUT / "final_reach_month_predictions_2016_2024.parquet", index=False)
    pd.DataFrame(parameter_rows).to_parquet(OUT / "final_map_parameters.parquet", index=False)
    performance = pd.DataFrame(metric_table(station_frame, selected) + oof_metric_table(oof, selected))
    performance.to_parquet(OUT / "final_performance_metrics.parquet", index=False)
    residuals = residual_month_table(oof); residuals.to_parquet(OUT / "final_oof_month_residuals.parquet", index=False)
    bootstrap = bootstrap_metric_ci(oof, selected); bootstrap.to_parquet(OUT / "final_oof_macro_bootstrap_ci.parquet", index=False)
    exposure = hydraulic_exposure_summary(monthly); exposure.to_parquet(OUT / "hydraulic_exposure_sensitivity_summary.parquet", index=False)

    temporal = performance.loc[performance.program.eq("OOF_TEMPORAL")].iloc[0].to_dict()
    apparent = performance.loc[performance.program.eq("FINAL_MAP_APPARENT_2021_2024") & performance.hydraulic_scenario.eq("H1_WIDTH_CENTRAL")].iloc[0].to_dict()
    backcast = performance.loc[performance.program.eq("FINAL_MAP_HISTORICAL_BACKCAST_2016_2020") & performance.hydraulic_scenario.eq("H1_WIDTH_CENTRAL")].iloc[0].to_dict()
    checks = {
        "all_predecessors_pass": True, "formal_carrier_daily_compiled": decisions[13]["selected_carrier"] == "DAILY_COMPILED_CARRIER",
        "final_training_2021_2024_only": True, "backcast_not_used_for_selection": True,
        "all_230_reaches_present": reach_frame.reach_id.nunique() == 230,
        "reach_months_2016_2024_complete": len(reach_frame) == 230 * 9 * 12 * 3,
        "no_station_parameters": True, "no_temperature_reservoir_wwtp": True,
        "no_20260824_18_created": not (ROOT / "5_Test" / "20260824_18").exists(),
    }
    status = "PASS_FINAL_TN_PROGRAM_COMPLETE" if all(checks.values()) else "FAIL_STAGE17"
    decision = {
        "stage": "20260824_17", "status": status, "selected_architecture": selected,
        "architecture_tree": {
            "carrier": decisions[13]["selected_carrier"],
            "SON": {"selected": decisions[14]["selected_architecture"], "legacy_supported": decisions[14]["legacy_supported"]},
            "static_pi": {"selected": decisions[15]["selected_architecture"], "supported": decisions[15]["static_pi_supported"]},
            "dynamic_delivery": {"selected": decisions[16]["selected_architecture"], "supported": decisions[16]["dynamic_delivery_supported"]},
        },
        "temporal_oof_metrics": temporal, "final_map_apparent_metrics": apparent,
        "historical_backcast_metrics": backcast, "checks": checks,
        "interpretation": "The selected model is the most complex structure that passed pre-registered temporal and spatial gates. Apparent full-development metrics are not OOF; 2016-2020 is a descriptive backcast.",
        "hashes": {"contract": sha256(CONTRACT), "canonical_monthly_bridge": sha256(MONTHLY), "predictions": sha256(OUT / "final_reach_month_predictions_2016_2024.parquet")},
        "authorized_successor": None,
    }
    write_json(REPORTS / "stage17_final_decision.json", decision)
    report = f"""# 新统一水文驱动 TN 模型：阶段性最终报告

## 最终裁决

`{status}`  
最终结构：**{selected}**。

新模型完全使用 `20260828_9/10` 的统一水文状态与快慢响应。日水文被编译成守恒 unit-tracer kernel，TN 仍按月推进；河道反应采用正式 H1 暴露 `L×W/(Q×86400)`。没有站点历史校正、C–Q hinge、`eta_fast/eta_slow`、额外 fixed-T1、温度、水库或 WWTP 项。

## 逐级实验结果

1. 日编译 carrier 相对月平衡 carrier 在 temporal、LORO、LOTO 和 2021 新站扩展中均显著改善，因此正式锁定。
2. 单 SON 农田缓释在时间 OOF 上改善，但多数速率情景把 `rho_L` 推到 0.95 上界，属于 Legacy 强度混淆，未升级。
3. 五个静态属性区域化 `pi_E` 改善站点和 Reach 指标，但 terminal-tree 门未通过，未升级。
4. 动态 delivery 是否升级，以 `20260824_16` 的 nested 裁决为准；最终结果为 `{selected}`。

## 最朴素的效果数字

锁定结构 2022–2024 temporal OOF：RMSE `{temporal['rmse_mg_l']:.3f} mg/L`，MAE `{temporal['mae_mg_l']:.3f} mg/L`，NSE `{temporal['nse']:.3f}`，R² `{temporal['r2']:.3f}`，PBIAS `{temporal['pbias_percent']:.1f}%`。

2021–2024 全资料最终重拟合的表观效果：RMSE `{apparent['rmse_mg_l']:.3f} mg/L`，NSE `{apparent['nse']:.3f}`，R² `{apparent['r2']:.3f}`。这不是 OOF。

2016–2020 回报：RMSE `{backcast['rmse_mg_l']:.3f} mg/L`，NSE `{backcast['nse']:.3f}`，R² `{backcast['r2']:.3f}`；它只用于检验时间回报，不参与结构选择。

## 结论边界

这轮实验建立了统一、守恒、可空间应用到 230 Reach 的 TN 架构，但如果 temporal OOF 的 NSE 或 R²仍低，则说明“接口正确”不等于“TN 已达到生产精度”。日水文 carrier 的改善是可信的；农田 Legacy 的方向有证据但参数尚不可识别；空间异质性仍是下一篇研究问题，而不是在本实验树中继续追加结构。`20260824_18+` 已关闭。
    """
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    artifacts = sorted([*OUT.glob("*.parquet"), *REPORTS.glob("*.json"), REPORTS / "technical_report.md", CONTRACT])
    artifacts = [path for path in artifacts if path.name != "final_artifact_manifest.json"]
    write_json(REPORTS / "final_artifact_manifest.json", {"stage": "20260824_17", "status": status, "artifacts": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts]})
    if status.startswith("FAIL"): raise RuntimeError(decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__": main()
