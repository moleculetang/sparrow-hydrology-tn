from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
MAIN = RUN / "reports" / "main_model"
EPS = 1.0e-6


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((pred - obs) ** 2) / denom)


def kge_components(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    if len(obs) < 2 or np.std(obs) <= 0 or np.mean(obs) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = float(np.corrcoef(obs, pred)[0, 1]) if np.std(pred) > 0 else 0.0
    beta = float(np.mean(pred) / np.mean(obs))
    cv_obs = float(np.std(obs) / np.mean(obs))
    cv_pred = float(np.std(pred) / np.mean(pred)) if np.mean(pred) != 0 else np.nan
    gamma = float(cv_pred / cv_obs) if cv_obs != 0 else np.nan
    if not np.isfinite(r) or not np.isfinite(beta) or not np.isfinite(gamma):
        return np.nan, r, beta, gamma
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2))
    return kge, r, beta, gamma


def month_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return int((a.year - b.year) * 12 + (a.month - b.month))


def pct_bias(pred_sum: float, obs_sum: float) -> float:
    if obs_sum == 0:
        return np.nan
    return float(100.0 * (pred_sum - obs_sum) / obs_sum)


def safe_ratio(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return np.nan
    return float(num / den)


def metric_block(frame: pd.DataFrame, pred_col: str, prefix: str = "") -> dict[str, float]:
    part = frame[["date", "month", "Q_obsv_cfs", pred_col]].dropna().copy()
    part = part[(part["Q_obsv_cfs"] > 0) & (part[pred_col] > 0)].copy()
    if part.empty:
        return {}
    obs = part["Q_obsv_cfs"].to_numpy(dtype=float)
    pred = part[pred_col].to_numpy(dtype=float)
    log_obs = np.log(obs + EPS)
    log_pred = np.log(pred + EPS)
    err = pred - obs
    log_err = log_pred - log_obs

    kge, r, beta, gamma = kge_components(obs, pred)
    kge_log, r_log, beta_log, gamma_log = kge_components(log_obs, log_pred)
    spearman = float(pd.Series(obs).corr(pd.Series(pred), method="spearman")) if len(obs) > 1 else np.nan

    q10_obs, q25_obs, q50_obs, q75_obs, q90_obs = np.quantile(obs, [0.10, 0.25, 0.50, 0.75, 0.90])
    q10_pred, q25_pred, q50_pred, q75_pred, q90_pred = np.quantile(pred, [0.10, 0.25, 0.50, 0.75, 0.90])
    high = obs >= q75_obs
    very_high = obs >= q90_obs
    low = obs <= q25_obs
    very_low = obs <= q10_obs
    wet = part["month"].isin([4, 5, 6, 7, 8, 9]).to_numpy()
    dry = ~wet

    obs_peak_idx = int(np.argmax(obs))
    pred_peak_idx = int(np.argmax(pred))
    obs_peak_date = pd.Timestamp(part.iloc[obs_peak_idx]["date"])
    pred_peak_date = pd.Timestamp(part.iloc[pred_peak_idx]["date"])
    peak_lag = month_diff(pred_peak_date, obs_peak_date)

    # Monthly residual trend in log space: positive means increasing overprediction through validation.
    x = np.arange(len(log_err), dtype=float)
    trend = float(np.polyfit(x, log_err, 1)[0] * 12.0) if len(log_err) >= 3 else np.nan
    lag1 = float(pd.Series(log_err).autocorr(lag=1)) if len(log_err) >= 4 else np.nan

    good = bool(nse(log_obs, log_pred) >= 0.65 and kge >= 0.50 and abs(pct_bias(np.sum(pred), np.sum(obs))) <= 25.0)
    out = {
        "n_months": int(len(obs)),
        "NSE_raw": nse(obs, pred),
        "NSElog": nse(log_obs, log_pred),
        "KGE_2012": kge,
        "KGE_r": r,
        "KGE_beta_volume_ratio": beta,
        "KGE_gamma_variability_ratio": gamma,
        "KGE_log": kge_log,
        "corr_pearson_raw": r,
        "corr_pearson_log": r_log,
        "corr_spearman_raw": spearman,
        "PBIAS_pct": pct_bias(np.sum(pred), np.sum(obs)),
        "abs_PBIAS_pct": abs(pct_bias(np.sum(pred), np.sum(obs))),
        "RMSE_cfs": float(np.sqrt(np.mean(err**2))),
        "MAE_cfs": float(np.mean(np.abs(err))),
        "MAE_pct_of_mean_obs": float(100.0 * np.mean(np.abs(err)) / np.mean(obs)),
        "RMSElog": float(np.sqrt(np.mean(log_err**2))),
        "MAElog": float(np.mean(np.abs(log_err))),
        "mean_log_error": float(np.mean(log_err)),
        "median_log_error": float(np.median(log_err)),
        "log_error_trend_per_year": trend,
        "log_error_lag1_autocorr": lag1,
        "obs_mean_cfs": float(np.mean(obs)),
        "pred_mean_cfs": float(np.mean(pred)),
        "obs_median_cfs": float(np.median(obs)),
        "pred_median_cfs": float(np.median(pred)),
        "mean_ratio_pred_obs": safe_ratio(float(np.mean(pred)), float(np.mean(obs))),
        "median_ratio_pred_obs": safe_ratio(float(np.median(pred)), float(np.median(obs))),
        "obs_q10_cfs": float(q10_obs),
        "pred_q10_cfs": float(q10_pred),
        "q10_bias_pct": pct_bias(q10_pred, q10_obs),
        "obs_q25_cfs": float(q25_obs),
        "pred_q25_cfs": float(q25_pred),
        "q25_bias_pct": pct_bias(q25_pred, q25_obs),
        "obs_q50_cfs": float(q50_obs),
        "pred_q50_cfs": float(q50_pred),
        "q50_bias_pct": pct_bias(q50_pred, q50_obs),
        "obs_q75_cfs": float(q75_obs),
        "pred_q75_cfs": float(q75_pred),
        "q75_bias_pct": pct_bias(q75_pred, q75_obs),
        "obs_q90_cfs": float(q90_obs),
        "pred_q90_cfs": float(q90_pred),
        "q90_bias_pct": pct_bias(q90_pred, q90_obs),
        "high_flow_volume_bias_pct_obs_ge_q75": pct_bias(float(np.sum(pred[high])), float(np.sum(obs[high]))),
        "very_high_flow_volume_bias_pct_obs_ge_q90": pct_bias(float(np.sum(pred[very_high])), float(np.sum(obs[very_high]))),
        "low_flow_volume_bias_pct_obs_le_q25": pct_bias(float(np.sum(pred[low])), float(np.sum(obs[low]))),
        "very_low_flow_volume_bias_pct_obs_le_q10": pct_bias(float(np.sum(pred[very_low])), float(np.sum(obs[very_low]))),
        "wet_month_volume_bias_pct_Apr_Sep": pct_bias(float(np.sum(pred[wet])), float(np.sum(obs[wet]))),
        "dry_month_volume_bias_pct_Oct_Mar": pct_bias(float(np.sum(pred[dry])), float(np.sum(obs[dry]))),
        "peak_obs_cfs": float(obs[obs_peak_idx]),
        "peak_pred_cfs": float(pred[pred_peak_idx]),
        "peak_magnitude_bias_pct": pct_bias(float(pred[pred_peak_idx]), float(obs[obs_peak_idx])),
        "peak_obs_date": str(obs_peak_date.date()),
        "peak_pred_date": str(pred_peak_date.date()),
        "peak_lag_months_pred_minus_obs": peak_lag,
        "peak_same_month": bool(peak_lag == 0),
        "good": good,
    }
    if prefix:
        return {f"{prefix}{key}": value for key, value in out.items()}
    return out


def classify_problem(row: pd.Series) -> str:
    if bool(row["main_good"]):
        return "good"
    reasons: list[str] = []
    if row["main_NSElog"] < 0.20 and abs(row["main_PBIAS_pct"]) <= 25:
        reasons.append("shape_timing")
    elif row["main_NSElog"] < 0.65:
        reasons.append("shape_skill")
    if abs(row["main_PBIAS_pct"]) > 25:
        reasons.append("systematic_volume_bias")
    if row["main_KGE_r"] < 0.60:
        reasons.append("low_correlation")
    if abs(row["main_KGE_beta_volume_ratio"] - 1.0) > 0.25:
        reasons.append("volume_ratio")
    if abs(row["main_KGE_gamma_variability_ratio"] - 1.0) > 0.35:
        reasons.append("variability_amplitude")
    if abs(row["main_peak_lag_months_pred_minus_obs"]) >= 2:
        reasons.append("peak_timing")
    if abs(row["main_peak_magnitude_bias_pct"]) > 35:
        reasons.append("peak_magnitude")
    if abs(row["main_high_flow_volume_bias_pct_obs_ge_q75"]) > 30:
        reasons.append("high_flow")
    if abs(row["main_low_flow_volume_bias_pct_obs_le_q25"]) > 30:
        reasons.append("low_flow")
    if abs(row["main_wet_month_volume_bias_pct_Apr_Sep"] - row["main_dry_month_volume_bias_pct_Oct_Mar"]) > 30:
        reasons.append("seasonal_bias")
    return "+".join(dict.fromkeys(reasons)) if reasons else "threshold_edge"


def main() -> None:
    pred = pd.read_csv(MAIN / "reach_class_selected_predictions_long.csv", encoding="utf-8-sig")
    pred["date"] = pd.to_datetime({"year": pred["year"].astype(int), "month": pred["month"].astype(int), "day": 1})
    diag = pd.read_csv(MAIN / "station_diagnostic_labels.csv", encoding="utf-8-sig")
    diag = diag[
        [
            "q_site",
            "reach_id",
            "reach_class",
            "reservoir_relation",
            "nearest_upstream_reservoir_name",
            "reservoir_downstream_order",
            "failure_mode",
        ]
    ].drop_duplicates("q_site")

    rows = []
    for (site, rid, rclass), group in pred[pred["split"].eq("validation")].groupby(
        ["q_site", "reach_id", "reach_class"], sort=True
    ):
        row: dict[str, object] = {"q_site": site, "reach_id": int(rid), "reach_class": rclass}
        row.update(metric_block(group, "Q_pred_cfs", "main_"))
        row.update(metric_block(group, "Q72_pred_cfs", "q72_"))
        row.update(metric_block(group, "Q78_mass_cfs", "q78_"))
        row["delta_main_minus_q72_NSElog"] = row.get("main_NSElog", np.nan) - row.get("q72_NSElog", np.nan)
        row["delta_main_minus_q72_KGE"] = row.get("main_KGE_2012", np.nan) - row.get("q72_KGE_2012", np.nan)
        row["delta_main_minus_q72_absPBIAS"] = row.get("main_abs_PBIAS_pct", np.nan) - row.get("q72_abs_PBIAS_pct", np.nan)
        row["delta_main_minus_q78_NSElog"] = row.get("main_NSElog", np.nan) - row.get("q78_NSElog", np.nan)
        rows.append(row)

    out = pd.DataFrame(rows).merge(diag, on=["q_site", "reach_id", "reach_class"], how="left")
    out["diagnostic_problem_signature"] = out.apply(classify_problem, axis=1)
    front = [
        "q_site",
        "reach_id",
        "reach_class",
        "main_good",
        "diagnostic_problem_signature",
        "failure_mode",
        "reservoir_relation",
        "nearest_upstream_reservoir_name",
        "reservoir_downstream_order",
        "main_NSElog",
        "main_KGE_2012",
        "main_KGE_r",
        "main_KGE_beta_volume_ratio",
        "main_KGE_gamma_variability_ratio",
        "main_PBIAS_pct",
        "main_RMSElog",
        "main_MAE_pct_of_mean_obs",
        "main_q10_bias_pct",
        "main_q50_bias_pct",
        "main_q90_bias_pct",
        "main_high_flow_volume_bias_pct_obs_ge_q75",
        "main_low_flow_volume_bias_pct_obs_le_q25",
        "main_wet_month_volume_bias_pct_Apr_Sep",
        "main_dry_month_volume_bias_pct_Oct_Mar",
        "main_peak_magnitude_bias_pct",
        "main_peak_lag_months_pred_minus_obs",
        "main_log_error_trend_per_year",
        "main_log_error_lag1_autocorr",
        "q72_NSElog",
        "q72_KGE_2012",
        "q72_PBIAS_pct",
        "q78_NSElog",
        "q78_KGE_2012",
        "q78_PBIAS_pct",
        "delta_main_minus_q72_NSElog",
        "delta_main_minus_q72_KGE",
        "delta_main_minus_q72_absPBIAS",
        "delta_main_minus_q78_NSElog",
    ]
    remaining = [c for c in out.columns if c not in front]
    out = out[front + remaining].sort_values(["main_good", "main_NSElog"], ascending=[False, False])
    csv_path = MAIN / "station_performance_diagnostics_extended.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    problem_summary = (
        out.groupby("diagnostic_problem_signature", dropna=False)
        .agg(
            station_count=("q_site", "nunique"),
            median_NSElog=("main_NSElog", "median"),
            median_KGE=("main_KGE_2012", "median"),
            median_absPBIAS=("main_abs_PBIAS_pct", "median"),
        )
        .reset_index()
        .sort_values(["station_count", "median_NSElog"], ascending=[False, True])
    )
    problem_summary.to_csv(MAIN / "station_performance_problem_signature_summary.csv", index=False, encoding="utf-8-sig")

    data_dictionary = """# station_performance_diagnostics_extended.csv 字段说明

这个 CSV 是逐站诊断表，核心目的是判断“每个水文站为什么模拟不好”，而不只是给 NSE/KGE/PBIAS。

## 关键字段

- `main_*`: 当前主线模型指标。
- `q72_*`: 纯高技能基底分支指标。
- `q78_*`: 完全质量守恒参照分支指标。
- `delta_main_minus_q72_*`: 主线相对 Q72 的变化。
- `diagnostic_problem_signature`: 根据多指标自动生成的问题标签。
- `failure_mode`: 之前较粗的诊断分类。
- `reservoir_relation`: 是否在水库 reach 或其下游 1-2 级影响范围。

## KGE 分解

- `main_KGE_r`: 相关性，低则说明形状/时序不好。
- `main_KGE_beta_volume_ratio`: 均值比例，偏离 1 表示总体水量偏差。
- `main_KGE_gamma_variability_ratio`: 变异系数比例，偏离 1 表示涨落幅度不对。

## 流量分位数和丰枯水

- `main_q10_bias_pct`, `main_q50_bias_pct`, `main_q90_bias_pct`: 低流量、中位流量、高流量分位点偏差。
- `main_high_flow_volume_bias_pct_obs_ge_q75`: 观测高流量月份的总量偏差。
- `main_low_flow_volume_bias_pct_obs_le_q25`: 观测低流量月份的总量偏差。
- `main_wet_month_volume_bias_pct_Apr_Sep`: 4-9 月偏差。
- `main_dry_month_volume_bias_pct_Oct_Mar`: 10-3 月偏差。

## 峰值与时序

- `main_peak_magnitude_bias_pct`: 最大月流量峰值幅度偏差。
- `main_peak_lag_months_pred_minus_obs`: 预测峰值月份减观测峰值月份。正值表示峰值预测偏晚，负值表示偏早。
- `main_log_error_trend_per_year`: 验证期 log 残差趋势。正值表示后期越来越高估，负值表示后期越来越低估。
- `main_log_error_lag1_autocorr`: log 残差一阶自相关，高则说明误差有持续性。
"""
    (MAIN / "station_performance_diagnostics_extended_dictionary.md").write_text(data_dictionary, encoding="utf-8")
    print(csv_path)
    print(f"stations={out['q_site'].nunique()}, columns={len(out.columns)}")


if __name__ == "__main__":
    main()
