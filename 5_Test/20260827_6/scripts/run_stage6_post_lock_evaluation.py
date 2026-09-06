"""Independent post-lock time and four-station spatial evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_6"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE5 = ROOT / "5_Test" / "20260827_5"
TIME_OBS = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_locked_retrospective_2019_2022.parquet"
SPACE_OBS = ROOT / "5_Test" / "20260823_14" / "outputs" / "supplemental_zero_history_station_month_observations.parquet"
SPACE_REGISTRY = ROOT / "5_Test" / "20260823_14" / "outputs" / "supplemental_zero_history_station_coverage.parquet"
DAILY_PRED = STAGE5 / "outputs" / "blind_reach_daily_2006_2022.parquet"
MONTHLY_PRED = STAGE5 / "outputs" / "blind_reach_monthly_2006_2022.parquet"
MODEL_LOCK = STAGE5 / "outputs" / "full_development_model_lock.pt"
OPERATOR = STAGE5 / "outputs" / "final_signature_operator_by_reach.parquet"
FOUR_STATIONS = ["珠坑", "昭平", "瓦村（二）", "盘江桥（三）"]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    if len(observed) < 2 or np.sum((observed - observed.mean()) ** 2) <= 0:
        return np.nan
    return float(1.0 - np.sum((predicted - observed) ** 2) / np.sum((observed - observed.mean()) ** 2))


def log_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    return float(np.sqrt(np.mean((np.log1p(predicted[valid]) - np.log1p(observed[valid])) ** 2)))


def pbias(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    return float(100.0 * np.sum(predicted[valid] - observed[valid]) / np.sum(observed[valid]))


def station_metrics(frame: pd.DataFrame, station_column: str = "station_norm") -> pd.DataFrame:
    rows = []
    for station, part in frame.groupby(station_column):
        rows.append({
            station_column: station,
            "observations": len(part),
            "NSE": nse(part.observed_m3_s.to_numpy(float), part.predicted_m3_s.to_numpy(float)),
            "log_RMSE": log_rmse(part.observed_m3_s.to_numpy(float), part.predicted_m3_s.to_numpy(float)),
            "PBIAS_pct": pbias(part.observed_m3_s.to_numpy(float), part.predicted_m3_s.to_numpy(float)),
        })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    observed = frame.observed_m3_s.to_numpy(float)
    predicted = frame.predicted_m3_s.to_numpy(float)
    valid_nse = metrics.NSE.dropna()
    return {
        "observations": len(frame),
        "stations": int(metrics.iloc[:, 0].nunique()),
        "pooled_NSE": nse(observed, predicted),
        "pooled_log_RMSE": log_rmse(observed, predicted),
        "pooled_PBIAS_pct": pbias(observed, predicted),
        "station_median_NSE": float(valid_nse.median()),
        "station_mean_NSE": float(valid_nse.mean()),
        "station_P10_NSE": float(valid_nse.quantile(0.10)),
        "station_P25_NSE": float(valid_nse.quantile(0.25)),
        "negative_NSE_count": int(valid_nse.lt(0).sum()),
        "negative_NSE_fraction": float(valid_nse.lt(0).mean()),
        "station_median_PBIAS_pct": float(metrics.PBIAS_pct.median()),
        "station_mean_PBIAS_pct": float(metrics.PBIAS_pct.mean()),
        "station_median_absolute_PBIAS_pct": float(metrics.PBIAS_pct.abs().median()),
        "station_P95_absolute_PBIAS_pct": float(metrics.PBIAS_pct.abs().quantile(0.95)),
        "station_median_log_RMSE": float(metrics.log_RMSE.median()),
    }


def markdown_table(frame: pd.DataFrame, float_digits: int = 3) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{float_digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    lock = json.loads((STAGE5 / "reports" / "blind_prediction_lock.json").read_text(encoding="utf-8"))
    if lock["status"] != "BLIND_230_REACH_EXPORT_LOCKED" or not lock["created_before_test_evaluation"]:
        raise RuntimeError("Blind prediction lock is invalid")
    observed_hashes_before_evaluation = {
        "time_observations_sha256": sha256(TIME_OBS),
        "space_observations_sha256": sha256(SPACE_OBS),
        "space_registry_sha256": sha256(SPACE_REGISTRY),
    }
    hash_checks = {
        "model": sha256(MODEL_LOCK) == lock["model_sha256"],
        "daily": sha256(DAILY_PRED) == lock["daily_prediction_sha256"],
        "monthly": sha256(MONTHLY_PRED) == lock["monthly_prediction_sha256"],
        "operator": sha256(OPERATOR) == lock["signature_operator_sha256"],
    }
    if not all(hash_checks.values()):
        raise RuntimeError(f"Prediction lock hash mismatch: {hash_checks}")

    # Test observations are opened only after all four locked artifacts are verified.
    time_obs = pd.read_parquet(TIME_OBS)
    time_obs.date = pd.to_datetime(time_obs.date)
    stations = pd.read_parquet(STAGE2 / "outputs" / "temporal_station_registry.parquet", columns=["station_norm", "reach_id", "terminal_tree"])
    time_obs = time_obs.loc[
        time_obs.station_norm.isin(stations.station_norm)
        & time_obs.date.dt.year.between(2019, 2022)
    ].copy()
    time_pred = pd.read_parquet(DAILY_PRED, columns=[
        "date", "reach_id", "local_fast_response_m3_s", "local_slow_response_m3_s",
        "routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_total_m3_s",
    ])
    time_pred.date = pd.to_datetime(time_pred.date)
    time_pred = time_pred.loc[time_pred.date.dt.year.between(2019, 2022)].copy()
    time = time_obs.merge(time_pred, on=["date", "reach_id"], validate="many_to_one")
    local_total = time.local_fast_response_m3_s + time.local_slow_response_m3_s
    fraction = time.downstream_fraction_on_reach.fillna(1.0).to_numpy(float)
    time["predicted_m3_s"] = time.routed_total_m3_s - (1.0 - fraction) * local_total
    time["predicted_fast_m3_s"] = time.routed_fast_response_m3_s - (1.0 - fraction) * time.local_fast_response_m3_s
    time["predicted_slow_m3_s"] = time.routed_slow_response_m3_s - (1.0 - fraction) * time.local_slow_response_m3_s
    time.rename(columns={"q_m3_s": "observed_m3_s"}, inplace=True)
    time = time.loc[np.isfinite(time.observed_m3_s) & np.isfinite(time.predicted_m3_s)].copy()
    time.to_parquet(OUT / "time_extrapolation_daily_predictions.parquet", index=False)
    daily_metrics = station_metrics(time).merge(stations, on=["station_norm", "reach_id", "terminal_tree"], how="left") if False else station_metrics(time)
    daily_metrics = daily_metrics.merge(stations, on="station_norm", validate="one_to_one")
    daily_metrics.to_parquet(OUT / "time_extrapolation_daily_station_metrics.parquet", index=False)
    daily_summary = summarize(time, daily_metrics[["station_norm", "observations", "NSE", "log_RMSE", "PBIAS_pct"]])

    time["year"] = time.date.dt.year
    time["month"] = time.date.dt.month
    monthly_time = time.groupby(["station_norm", "reach_id", "terminal_tree", "year", "month"], as_index=False).agg(
        observed_m3_s=("observed_m3_s", "mean"), predicted_m3_s=("predicted_m3_s", "mean"),
        predicted_fast_m3_s=("predicted_fast_m3_s", "mean"), predicted_slow_m3_s=("predicted_slow_m3_s", "mean"),
        valid_days=("observed_m3_s", "size"),
    )
    monthly_time.to_parquet(OUT / "time_extrapolation_monthly_predictions.parquet", index=False)
    monthly_metrics = station_metrics(monthly_time).merge(stations, on="station_norm", validate="one_to_one")
    monthly_metrics.to_parquet(OUT / "time_extrapolation_monthly_station_metrics.parquet", index=False)
    monthly_summary = summarize(monthly_time, monthly_metrics[["station_norm", "observations", "NSE", "log_RMSE", "PBIAS_pct"]])
    tree_rows = []
    for tree, part in monthly_time.groupby("terminal_tree"):
        part_metrics = station_metrics(part)
        tree_rows.append({"terminal_tree": int(tree), **summarize(part, part_metrics)})
    tree_metrics = pd.DataFrame(tree_rows)
    tree_metrics.to_parquet(OUT / "time_extrapolation_monthly_tree_metrics.parquet", index=False)

    space_registry = pd.read_parquet(SPACE_REGISTRY)
    space_registry = space_registry.loc[space_registry.station_norm.isin(FOUR_STATIONS)].copy()
    if set(space_registry.station_norm) != set(FOUR_STATIONS) or space_registry.station_seen_in_training.any() or space_registry.reach_seen_in_training.any():
        raise RuntimeError("Four-station zero-history registry changed")
    space_obs = pd.read_parquet(SPACE_OBS)
    space_obs = space_obs.loc[
        space_obs.station_norm.isin(FOUR_STATIONS) & space_obs.usable & space_obs.year.ge(2010)
    ].copy()
    space_pred = pd.read_parquet(MONTHLY_PRED, columns=[
        "reach_id", "year", "month", "routed_total_m3_s",
        "routed_fast_response_m3_s", "routed_slow_response_m3_s", "is_spinup_period",
    ])
    space = space_obs.merge(
        space_registry[["station_norm", "reach_id"]], on=["station_norm", "reach_id"], validate="many_to_one"
    ).merge(space_pred, on=["reach_id", "year", "month"], validate="many_to_one")
    if space.is_spinup_period.any():
        raise RuntimeError("Spin-up rows entered spatial evaluation")
    space.rename(columns={"q_m3s": "observed_m3_s", "routed_total_m3_s": "predicted_m3_s"}, inplace=True)
    space.to_parquet(OUT / "space_extrapolation_four_station_predictions.parquet", index=False)
    space_metrics = station_metrics(space)
    periods = space.groupby("station_norm").agg(first_year=("year", "min"), last_year=("year", "max"), usable_months=("year", "size")).reset_index()
    component_signature = space.groupby("station_norm").agg(
        predicted_slow_fraction=("routed_slow_response_m3_s", lambda value: float(value.sum())),
        predicted_total=("predicted_m3_s", "sum"),
        observed_q20=("observed_m3_s", lambda value: float(value.quantile(0.2))),
        observed_mean=("observed_m3_s", "mean"),
    ).reset_index()
    component_signature["predicted_slow_fraction"] /= component_signature.predicted_total.clip(lower=1e-12)
    component_signature["observed_monthly_low_flow_ratio"] = component_signature.observed_q20 / component_signature.observed_mean.clip(lower=1e-12)
    space_metrics = space_metrics.merge(periods, on="station_norm").merge(component_signature, on="station_norm")
    space_metrics.to_parquet(OUT / "space_extrapolation_four_station_metrics.parquet", index=False)
    space_summary = summarize(space, space_metrics[["station_norm", "observations", "NSE", "log_RMSE", "PBIAS_pct"]])
    signature_rho = float(component_signature[["predicted_slow_fraction", "observed_monthly_low_flow_ratio"]].corr(method="spearman").iloc[0, 1])

    daily_gates = contract["time_extrapolation"]["daily_gates"]
    monthly_gates = contract["time_extrapolation"]["monthly_gates"]
    spatial_gates = contract["space_extrapolation"]["gates"]
    checks = {
        "lock_hashes_match_before_observation_read": all(hash_checks.values()),
        "time_daily_pooled_NSE": daily_summary["pooled_NSE"] >= daily_gates["pooled_NSE_min"],
        "time_daily_station_median_NSE": daily_summary["station_median_NSE"] >= daily_gates["station_median_NSE_min"],
        "time_monthly_pooled_NSE": monthly_summary["pooled_NSE"] >= monthly_gates["pooled_NSE_min"],
        "time_monthly_station_median_NSE": monthly_summary["station_median_NSE"] >= monthly_gates["station_median_NSE_min"],
        "time_monthly_negative_fraction": monthly_summary["negative_NSE_fraction"] <= monthly_gates["negative_station_fraction_max"],
        "time_monthly_median_abs_PBIAS": monthly_summary["station_median_absolute_PBIAS_pct"] <= monthly_gates["station_median_absolute_PBIAS_max_pct"],
        "space_pooled_NSE": space_summary["pooled_NSE"] >= spatial_gates["pooled_NSE_min"],
        "space_station_median_NSE": space_summary["station_median_NSE"] >= spatial_gates["station_median_NSE_min"],
        "space_negative_station_count": space_summary["negative_NSE_count"] <= spatial_gates["negative_station_count_max"],
        "space_median_abs_PBIAS": space_summary["station_median_absolute_PBIAS_pct"] <= spatial_gates["station_median_absolute_PBIAS_max_pct"],
        "exact_four_spatial_stations": set(space_metrics.station_norm) == set(FOUR_STATIONS),
        "zero_history_registry_verified": bool(~space_registry.station_seen_in_training.any() and ~space_registry.reach_seen_in_training.any()),
        "model_not_mutated": sha256(MODEL_LOCK) == lock["model_sha256"],
        "daily_prediction_not_mutated": sha256(DAILY_PRED) == lock["daily_prediction_sha256"],
        "monthly_prediction_not_mutated": sha256(MONTHLY_PRED) == lock["monthly_prediction_sha256"],
    }
    temporal_gate_names = [name for name in checks if name.startswith("time_")]
    spatial_gate_names = [name for name in checks if name.startswith("space_")]
    temporal_pass = all(checks[name] for name in temporal_gate_names)
    spatial_pass = all(checks[name] for name in spatial_gate_names)
    if temporal_pass and spatial_pass:
        status = "BASELINE_ACCEPTED_TIME_AND_SPACE_TOTAL_FLOW"
    elif temporal_pass:
        status = "TIME_EXTRAPOLATION_PASS_SPACE_EXTRAPOLATION_LIMITED"
    elif spatial_pass:
        status = "SPACE_EXTRAPOLATION_PASS_TIME_EXTRAPOLATION_LIMITED"
    else:
        status = "BASELINE_TOTAL_FLOW_GATES_NOT_MET"
    decision = {
        "stage": "20260827_6",
        "status": status,
        "temporal_gate_pass": temporal_pass,
        "spatial_gate_pass": spatial_pass,
        "time_daily": daily_summary,
        "time_monthly": monthly_summary,
        "space_monthly_four_station": space_summary,
        "four_station_component_supporting_diagnostic": {
            "spearman_predicted_slow_fraction_vs_observed_monthly_low_flow_ratio": signature_rho,
            "station_count": 4,
            "claim": "supporting only; monthly discharge and n=4 do not independently identify fast/slow components"
        },
        "checks": checks,
        "claim_boundary": contract["claim_boundary"],
        "program_closed": True,
        "no_20260827_7_plus_authorized": True,
    }
    write_json(REPORTS / "final_baseline_decision.json", decision)
    write_json(REPORTS / "test_observation_hashes.json", observed_hashes_before_evaluation)
    write_json(REPORTS / "validation.json", {
        "stage": "20260827_6", "hash_checks": hash_checks,
        "program_integrity_pass": all([
            checks["lock_hashes_match_before_observation_read"], checks["exact_four_spatial_stations"],
            checks["zero_history_registry_verified"], checks["model_not_mutated"],
            checks["daily_prediction_not_mutated"], checks["monthly_prediction_not_mutated"],
        ]),
    })

    station_table = markdown_table(space_metrics[["station_norm", "observations", "first_year", "last_year", "NSE", "log_RMSE", "PBIAS_pct"]])
    tree_table = markdown_table(tree_metrics[["terminal_tree", "stations", "pooled_NSE", "station_median_NSE", "station_mean_NSE", "negative_NSE_count"]])
    report = f"""# 20260827 新水文基线最终评价

## 结论

状态：`{status}`。

模型在任何正式测试观测被读取前，已完成2010–2018重拟合、230 Reach导出和SHA-256锁。评价进程首先复核四个锁文件，随后才读取2019–2022流量与四个空间测试站。

## 2019–2022时间外推

日尺度：pooled NSE={daily_summary['pooled_NSE']:.3f}；站点NSE中位={daily_summary['station_median_NSE']:.3f}，平均={daily_summary['station_mean_NSE']:.3f}，P10={daily_summary['station_P10_NSE']:.3f}，P25={daily_summary['station_P25_NSE']:.3f}；负NSE={daily_summary['negative_NSE_count']}/{daily_summary['stations']}；站点绝对PBIAS中位={daily_summary['station_median_absolute_PBIAS_pct']:.2f}%。

月尺度：pooled NSE={monthly_summary['pooled_NSE']:.3f}；站点NSE中位={monthly_summary['station_median_NSE']:.3f}，平均={monthly_summary['station_mean_NSE']:.3f}，P10={monthly_summary['station_P10_NSE']:.3f}，P25={monthly_summary['station_P25_NSE']:.3f}；负NSE={monthly_summary['negative_NSE_count']}/{monthly_summary['stations']}；站点绝对PBIAS中位={monthly_summary['station_median_absolute_PBIAS_pct']:.2f}%，P95={monthly_summary['station_P95_absolute_PBIAS_pct']:.2f}%。

### 按河树

{tree_table}

## 四站空间外推

不向目标站提供任何历史流量、站点参数或站点身份项；直接取锁定Reach预测。总体pooled NSE={space_summary['pooled_NSE']:.3f}；四站NSE中位={space_summary['station_median_NSE']:.3f}，平均={space_summary['station_mean_NSE']:.3f}；负NSE={space_summary['negative_NSE_count']}/4；绝对PBIAS中位={space_summary['station_median_absolute_PBIAS_pct']:.2f}%。

{station_table}

四站预测慢分量比例与观测月低流比例的Spearman={signature_rho:.3f}，但n=4且只有月流量，因此只能作支持性诊断，不能独立验证真实快慢水身份。

## 科学边界

2019–2022和四站在项目历史上曾被查看，所以不能称为“从未接触的独立外部验证”。本轮可以严格声称的是：新基线计算进程在模型与全河网预测锁定前对这两类测试观测零读取、零拟合、零选参，并且锁后评价没有改变模型。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    (RUN / "README.md").write_text("# 20260827_6 post-lock time and four-station spatial evaluation\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
