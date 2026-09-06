"""Evaluate the immutable Stage-7 product after verifying its hashes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_8"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE7 = ROOT / "5_Test" / "20260828_7"
STAGE8REG = ROOT / "5_Test" / "20260828_2"
OLD6 = ROOT / "5_Test" / "20260827_6"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(ROOT / "5_Test" / "20260827_9" / "scripts"),
    str(ROOT / "5_Test" / "20260827_8" / "scripts"),
    str(ROOT / "5_Test" / "20260827_7" / "scripts"),
    str(ROOT / "5_Test" / "20260827_3" / "scripts"),
    str(ROOT / "5_Test" / "20260827_2" / "scripts"),
    str(OLD27 / "scripts"), str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from run_stage9_temporal_development import metrics_by_station, summary  # noqa: E402


DAILY_PRODUCT = STAGE7 / "outputs" / "canonical_reach_daily_2006_2024.parquet"
MONTHLY_PRODUCT = STAGE7 / "outputs" / "canonical_reach_monthly_2006_2024.parquet"
MODEL = STAGE7 / "outputs" / "full_development_state_consistent_model.pt"
TIME_OBS = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_locked_retrospective_2019_2022.parquet"
FOUR_OBS = OLD6 / "outputs" / "space_extrapolation_four_station_predictions.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def cohort_evaluation(
    stations: pd.DataFrame,
    observations: pd.DataFrame,
    daily_product: pd.DataFrame,
    monthly_product: pd.DataFrame,
    label: str,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stations = stations.sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    fractions = observations.groupby("station_norm").downstream_fraction_on_reach.first().reindex(stations.station_norm).to_numpy(float)
    if not np.isfinite(fractions).all() or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise RuntimeError("Invalid station position fractions")
    dates = pd.date_range("2019-01-01", "2022-12-31", freq="D")
    obs = observations.loc[observations.station_norm.isin(stations.station_norm)].pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(float)
    routed_daily = daily_product.pivot(index="date", columns="reach_id", values="routed_total_m3_s").reindex(index=dates, columns=stations.reach_id).to_numpy(float)
    local_daily = daily_product.assign(local_total_m3_s=daily_product.local_fast_response_m3_s + daily_product.local_slow_response_m3_s).pivot(index="date", columns="reach_id", values="local_total_m3_s").reindex(index=dates, columns=stations.reach_id).to_numpy(float)
    reach_daily = routed_daily - (1.0 - fractions[None, :]) * local_daily
    daily_metrics = metrics_by_station(obs, reach_daily, stations, label, "daily")
    daily_summary = summary(obs, reach_daily, daily_metrics)
    periods = pd.period_range("2019-01", "2022-12", freq="M")
    date_periods = dates.to_period("M")
    obs_month = np.stack([np.nanmean(obs[np.asarray(date_periods == period)], axis=0) for period in periods])
    keys = pd.MultiIndex.from_arrays([periods.year, periods.month], names=["year", "month"])
    routed_month = monthly_product.pivot(index=["year", "month"], columns="reach_id", values="routed_total_m3_s").reindex(index=keys, columns=stations.reach_id).to_numpy(float)
    local_month = monthly_product.assign(local_total_m3_s=monthly_product.local_fast_response_m3_s + monthly_product.local_slow_response_m3_s).pivot(index=["year", "month"], columns="reach_id", values="local_total_m3_s").reindex(index=keys, columns=stations.reach_id).to_numpy(float)
    pred_month = routed_month - (1.0 - fractions[None, :]) * local_month
    monthly_metrics = metrics_by_station(obs_month, pred_month, stations, label, "monthly")
    monthly_summary = summary(obs_month, pred_month, monthly_metrics)
    prediction = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), len(stations)),
        "station_norm": np.tile(stations.station_norm.to_numpy(), len(dates)),
        "reach_id": np.tile(stations.reach_id.to_numpy(), len(dates)),
        "observed_m3_s": obs.reshape(-1), "predicted_m3_s": reach_daily.reshape(-1),
    })
    return daily_summary, monthly_summary, daily_metrics, monthly_metrics, prediction


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    lock = json.loads((STAGE7 / "reports" / "state_consistent_product_lock.json").read_text(encoding="utf-8"))
    hash_checks = {
        "model": sha256(MODEL) == lock["hashes"]["model"],
        "daily": sha256(DAILY_PRODUCT) == lock["hashes"]["daily_2006_2024"],
        "monthly": sha256(MONTHLY_PRODUCT) == lock["hashes"]["monthly_2006_2024"],
    }
    if lock["status"] != "STATE_CONSISTENT_230_REACH_PRODUCT_LOCKED" or not all(hash_checks.values()):
        raise RuntimeError({"lock_or_hash_failure": hash_checks})

    # Formal observations are opened only after the immutable hashes above pass.
    observations = pd.read_parquet(TIME_OBS)
    observations.date = pd.to_datetime(observations.date)
    daily_product = pd.read_parquet(DAILY_PRODUCT, columns=[
        "date", "reach_id", "routed_total_m3_s", "routed_fast_response_m3_s", "routed_slow_response_m3_s",
        "local_fast_response_m3_s", "local_slow_response_m3_s"
    ])
    daily_product.date = pd.to_datetime(daily_product.date)
    daily_product = daily_product.loc[daily_product.date.dt.year.between(2019, 2022)]
    monthly_product_all = pd.read_parquet(MONTHLY_PRODUCT, columns=[
        "reach_id", "year", "month", "routed_total_m3_s", "routed_fast_response_m3_s", "routed_slow_response_m3_s",
        "local_fast_response_m3_s", "local_slow_response_m3_s"
    ])
    monthly_product = monthly_product_all.loc[monthly_product_all.year.between(2019, 2022)]
    registry = pd.read_parquet(STAGE8REG / "outputs" / "station_registry_91.parquet")
    if len(registry) != 91:
        raise RuntimeError("Registered 91-station cohort changed")
    main_stations = registry.loc[registry.station_norm.isin(observations.station_norm.unique())].copy()
    if len(main_stations) != 91:
        raise RuntimeError(f"Expected all 91 stations in test observations, got {len(main_stations)}")
    old_time = pd.read_parquet(OLD6 / "outputs" / "time_extrapolation_monthly_predictions.parquet")
    strict_names = sorted(old_time.station_norm.unique())
    strict_stations = main_stations.loc[main_stations.station_norm.isin(strict_names)].copy()
    if len(strict_stations) != 65:
        raise RuntimeError(f"Strict 65-station cohort mismatch: {len(strict_stations)}")

    main_daily, main_monthly, main_daily_metrics, main_monthly_metrics, main_prediction = cohort_evaluation(
        main_stations, observations, daily_product, monthly_product, "STATE_CONSISTENT_91"
    )
    strict_daily, strict_monthly, strict_daily_metrics, strict_monthly_metrics, strict_prediction = cohort_evaluation(
        strict_stations, observations, daily_product, monthly_product, "STATE_CONSISTENT_65_COMPARABLE"
    )
    old = json.loads((OLD6 / "reports" / "final_baseline_decision.json").read_text(encoding="utf-8"))
    old_daily, old_monthly = old["time_daily"], old["time_monthly"]

    four_obs = pd.read_parquet(FOUR_OBS)
    expected_four = {"珠坑", "昭平", "瓦村（二）", "盘江桥（三）"}
    if set(four_obs.station_norm.unique()) != expected_four:
        raise RuntimeError("Four-station identity changed")
    four_new = four_obs.drop(columns=["predicted_m3_s", "routed_fast_response_m3_s", "routed_slow_response_m3_s"], errors="ignore").merge(
        monthly_product_all[["reach_id", "year", "month", "routed_total_m3_s", "routed_fast_response_m3_s", "routed_slow_response_m3_s"]],
        on=["reach_id", "year", "month"], how="left", validate="many_to_one",
    ).rename(columns={"routed_total_m3_s": "predicted_m3_s"})
    four_stations = four_new[["station_norm", "reach_id"]].drop_duplicates().copy()
    four_stations["terminal_tree"] = 0
    periods = pd.MultiIndex.from_frame(four_new[["year", "month"]].drop_duplicates().sort_values(["year", "month"]))
    # Metrics are computed per station directly because the four records have unequal date coverage.
    four_rows = []
    for row in four_stations.itertuples():
        block = four_new.loc[four_new.station_norm.eq(row.station_norm)]
        obs_values = block.observed_m3_s.to_numpy(float)
        pred_values = block.predicted_m3_s.to_numpy(float)
        valid = np.isfinite(obs_values) & np.isfinite(pred_values)
        obs_values, pred_values = obs_values[valid], pred_values[valid]
        denominator = np.sum((obs_values - obs_values.mean()) ** 2)
        four_rows.append({
            "station_norm": row.station_norm, "reach_id": int(row.reach_id), "n": int(len(obs_values)),
            "NSE": float(1.0 - np.sum((pred_values - obs_values) ** 2) / denominator),
            "PBIAS_pct": float(100.0 * np.sum(pred_values - obs_values) / np.sum(obs_values)),
            "log_RMSE": float(np.sqrt(np.mean((np.log1p(pred_values) - np.log1p(obs_values)) ** 2))),
        })
    four_metrics = pd.DataFrame(four_rows)
    valid = np.isfinite(four_new.observed_m3_s) & np.isfinite(four_new.predicted_m3_s)
    four_summary = summary(four_new.loc[valid, "observed_m3_s"].to_numpy(float), four_new.loc[valid, "predicted_m3_s"].to_numpy(float), four_metrics)
    old_four = old["space_monthly_four_station"]

    margin = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))["noninferiority_vs_20260827_6"]
    checks = {
        "lock_hashes_match_before_observation_read": all(hash_checks.values()),
        "strict_time_pooled_log_RMSE_noninferior": strict_monthly["pooled_log_RMSE"] - old_monthly["pooled_log_RMSE"] < margin["pooled_log_RMSE_increase_max"],
        "strict_time_pooled_NSE_noninferior": old_monthly["pooled_NSE"] - strict_monthly["pooled_NSE"] <= margin["pooled_NSE_drop_max"],
        "strict_time_station_median_NSE_noninferior": old_monthly["station_median_NSE"] - strict_monthly["station_median_NSE"] <= margin["station_median_NSE_drop_max"],
        "strict_time_median_abs_PBIAS_noninferior": strict_monthly["station_median_absolute_PBIAS_pct"] - old_monthly["station_median_absolute_PBIAS_pct"] <= margin["station_median_absolute_PBIAS_worsening_max_pct_point"],
        "four_station_pooled_log_RMSE_noninferior": four_summary["pooled_log_RMSE"] - old_four["pooled_log_RMSE"] < margin["pooled_log_RMSE_increase_max"],
        "four_station_pooled_NSE_noninferior": old_four["pooled_NSE"] - four_summary["pooled_NSE"] <= margin["pooled_NSE_drop_max"],
        "four_station_median_NSE_noninferior": old_four["station_median_NSE"] - four_summary["station_median_NSE"] <= margin["station_median_NSE_drop_max"],
        "four_station_median_abs_PBIAS_noninferior": four_summary["station_median_absolute_PBIAS_pct"] - old_four["station_median_absolute_PBIAS_pct"] <= margin["station_median_absolute_PBIAS_worsening_max_pct_point"],
        "model_hash_unchanged_after_evaluation": sha256(MODEL) == lock["hashes"]["model"],
        "daily_hash_unchanged_after_evaluation": sha256(DAILY_PRODUCT) == lock["hashes"]["daily_2006_2024"],
        "monthly_hash_unchanged_after_evaluation": sha256(MONTHLY_PRODUCT) == lock["hashes"]["monthly_2006_2024"],
    }
    accepted = all(checks.values())
    decision = {
        "stage": "20260828_8",
        "status": "CANONICAL_TN_HYDROLOGY_ACCEPTED" if accepted else "FORMAL_TOTAL_FLOW_NONINFERIORITY_FAILED",
        "time_91_daily": main_daily, "time_91_monthly": main_monthly,
        "time_65_comparable_daily": strict_daily, "time_65_comparable_monthly": strict_monthly,
        "old_20260827_6_time_65_daily": old_daily, "old_20260827_6_time_65_monthly": old_monthly,
        "four_station_new": four_summary, "four_station_old_20260827_6": old_four,
        "checks": checks,
        "canonical_daily": str(DAILY_PRODUCT), "canonical_monthly": str(MONTHLY_PRODUCT),
        "claim_boundary": "state-consistent modeled fast/slow hydrology; not tracer-validated true water age or groundwater fraction",
    }
    pd.concat([main_daily_metrics, main_monthly_metrics, strict_daily_metrics, strict_monthly_metrics], ignore_index=True).to_parquet(OUT / "time_test_station_metrics.parquet", index=False)
    main_prediction.to_parquet(OUT / "time_test_daily_predictions_91.parquet", index=False)
    strict_prediction.to_parquet(OUT / "time_test_daily_predictions_65_comparable.parquet", index=False)
    four_metrics.to_parquet(OUT / "space_test_four_station_metrics.parquet", index=False)
    four_new.to_parquet(OUT / "space_test_four_station_predictions.parquet", index=False)
    write_json(REPORTS / "canonical_hydrology_decision.json", decision)
    report = (
        "# 20260828_8 锁后评价与TN水文接口裁决\n\n"
        f"状态：`{decision['status']}`。模型与2006–2024产品在读取观测前已哈希锁定。\n\n"
        f"- 91站2019–2022月总体NSE：{main_monthly['pooled_NSE']:.3f}；逐站中位NSE：{main_monthly['station_median_NSE']:.3f}。\n"
        f"- 65站可比队列月总体NSE：{strict_monthly['pooled_NSE']:.3f}；逐站中位NSE：{strict_monthly['station_median_NSE']:.3f}。\n"
        f"- 4个空间站月总体NSE：{four_summary['pooled_NSE']:.3f}；逐站中位NSE：{four_summary['station_median_NSE']:.3f}。\n\n"
        "快慢分量已与上层/下层库存严格一致，但没有示踪剂独立验证，因此不得解释为真实水龄分布。\n"
    )
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
