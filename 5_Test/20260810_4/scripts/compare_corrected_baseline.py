from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "corrected_baseline_comparison"
OLD_PATH = RUN / "backup_before_correction" / "legacy_baseline" / "outputs" / "q72_three_fold_oof_predictions.parquet"
NEW_PATH = RUN / "outputs" / "q72_three_fold_oof_predictions.parquet"
REGISTRY = RUN / "inputs" / "source_snapshot" / "experiment_reference" / "canonical_signal_registry.csv"
KEYS = ["fold_id", "station_name", "reach_id", "year", "month"]
EPS = 1.0e-12


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    den = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((pred - obs) ** 2) / den) if den > EPS else np.nan


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 4 or obs.std(ddof=0) <= EPS or obs.mean() <= EPS or pred.std(ddof=0) <= EPS:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1]
    if not np.isfinite(r):
        return np.nan
    alpha = pred.std(ddof=0) / obs.std(ddof=0)
    beta = pred.mean() / obs.mean()
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def add_flow_masks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["lowflow"] = False
    frame["highflow"] = False
    for _, idx in frame.groupby(["station_name", "fold_id"], sort=False).groups.items():
        loc = list(idx)
        observed = frame.loc[loc, "observed_m3s"]
        frame.loc[loc, "lowflow"] = observed <= observed.quantile(0.25)
        frame.loc[loc, "highflow"] = observed >= observed.quantile(0.75)
    return frame


def station_metrics(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    frame = add_flow_masks(frame)
    rows: list[dict[str, Any]] = []
    for station, group in frame.groupby("station_name", sort=True):
        if len(group) < 12:
            continue
        obs = group["observed_m3s"].to_numpy(float)
        pred = np.maximum(group["predicted_m3s"].to_numpy(float), 0.0)
        log_obs = np.log1p(np.maximum(obs, 0.0))
        log_pred = np.log1p(pred)
        low = group["lowflow"].to_numpy(bool)
        high = group["highflow"].to_numpy(bool)
        pbias = float(100.0 * (pred.sum() - obs.sum()) / max(obs.sum(), EPS))
        log_nse = nse(log_obs, log_pred)
        metric_kge = kge(obs, pred)
        rows.append({
            "scenario": scenario,
            "station_name": station,
            "reach_id": int(group["reach_id"].iloc[0]),
            "months": int(len(group)),
            "folds": int(group["fold_id"].nunique()),
            "nse": nse(obs, pred),
            "log_nse": log_nse,
            "kge": metric_kge,
            "pbias_pct": pbias,
            "absolute_pbias_pct": abs(pbias),
            "lowflow_log_rmse": float(np.sqrt(np.mean((log_pred[low] - log_obs[low]) ** 2))),
            "lowflow_median_log_bias": float(np.median(log_pred[low] - log_obs[low])),
            "lowflow_absolute_median_log_bias": float(abs(np.median(log_pred[low] - log_obs[low]))),
            "highflow_log_rmse": float(np.sqrt(np.mean((log_pred[high] - log_obs[high]) ** 2))),
            "good": bool(log_nse >= 0.5 and metric_kge >= 0.5 and abs(pbias) <= 25.0),
        })
    return pd.DataFrame(rows)


def summary(metrics: pd.DataFrame) -> dict[str, float | int]:
    return {
        "station_count": int(len(metrics)),
        "median_nse": float(metrics["nse"].median()),
        "median_log_nse": float(metrics["log_nse"].median()),
        "median_kge": float(metrics["kge"].median()),
        "median_absolute_pbias_pct": float(metrics["absolute_pbias_pct"].median()),
        "median_lowflow_log_rmse": float(metrics["lowflow_log_rmse"].median()),
        "median_highflow_log_rmse": float(metrics["highflow_log_rmse"].median()),
        "good_count": int(metrics["good"].sum()),
    }


def normalize_name(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value)).replace("(", "（").replace(")", "）")
    value = re.sub(r"[\s　]", "", value)
    return value[:-1] if value.endswith("站") else value


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(OLD_PATH)
    new = pd.read_parquet(NEW_PATH)
    if old.duplicated(KEYS).any() or new.duplicated(KEYS).any():
        raise ValueError("OOF keys are not unique")
    keep = KEYS + ["observed_m3s", "predicted_m3s"]
    paired = old[keep].merge(new[keep], on=KEYS, how="inner", validate="one_to_one", suffixes=("_s0", "_s1"))
    observed_diff = (paired["observed_m3s_s0"] - paired["observed_m3s_s1"]).abs()
    paired["observed_absolute_difference_m3s"] = observed_diff
    paired.to_parquet(REPORT / "paired_oof_predictions.parquet", index=False)

    metric_parts = []
    fold_summaries = []
    for scenario, suffix in [("S0_before_correction", "s0"), ("S1_corrected", "s1")]:
        base = paired[KEYS + [f"observed_m3s_{suffix}", f"predicted_m3s_{suffix}"]].rename(
            columns={f"observed_m3s_{suffix}": "observed_m3s", f"predicted_m3s_{suffix}": "predicted_m3s"}
        )
        metrics = station_metrics(base, scenario)
        metric_parts.append(metrics)
        for fold_id, fold in base.groupby("fold_id", sort=True):
            fold_summaries.append({"scenario": scenario, "fold_id": fold_id, **summary(station_metrics(fold, scenario))})
    metrics_all = pd.concat(metric_parts, ignore_index=True)
    metrics_all.to_csv(REPORT / "paired_station_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_summaries).to_csv(REPORT / "paired_fold_metrics.csv", index=False, encoding="utf-8-sig")
    old_metrics = metrics_all[metrics_all["scenario"].eq("S0_before_correction")].copy()
    new_metrics = metrics_all[metrics_all["scenario"].eq("S1_corrected")].copy()
    s0 = summary(old_metrics)
    s1 = summary(new_metrics)
    delta = {key: float(s1[key] - s0[key]) for key in [
        "median_nse", "median_log_nse", "median_kge", "median_absolute_pbias_pct",
        "median_lowflow_log_rmse", "median_highflow_log_rmse"
    ]}
    delta["good_count"] = int(s1["good_count"] - s0["good_count"])

    joined_metrics = old_metrics.merge(new_metrics, on=["station_name", "reach_id"], suffixes=("_s0", "_s1"), validate="one_to_one")
    joined_metrics["delta_log_nse"] = joined_metrics["log_nse_s1"] - joined_metrics["log_nse_s0"]
    joined_metrics["delta_kge"] = joined_metrics["kge_s1"] - joined_metrics["kge_s0"]
    joined_metrics["delta_absolute_pbias_pct"] = joined_metrics["absolute_pbias_pct_s1"] - joined_metrics["absolute_pbias_pct_s0"]
    joined_metrics["delta_lowflow_log_rmse"] = joined_metrics["lowflow_log_rmse_s1"] - joined_metrics["lowflow_log_rmse_s0"]
    joined_metrics["delta_highflow_log_rmse"] = joined_metrics["highflow_log_rmse_s1"] - joined_metrics["highflow_log_rmse_s0"]

    affected = {normalize_name(x) for x in [
        "梧州（四）", "高要", "河步（二）", "大湟江口", "武宣（二）", "石角", "飞来峡",
        "都安（二）", "迁江", "天生桥", "天峨", "岔江"
    ]}
    affected_rows = joined_metrics[joined_metrics["station_name"].map(normalize_name).isin(affected)].copy()
    affected_rows.to_csv(REPORT / "affected_reach_station_comparison.csv", index=False, encoding="utf-8-sig")

    registry = pd.read_csv(REGISTRY, encoding="utf-8-sig")
    membership = registry["legacy_canonical_membership"].astype(str).str.casefold().isin({"true", "1"})
    legacy_reaches = set(pd.to_numeric(registry.loc[membership, "reach_id"], errors="coerce").dropna().astype(int))
    legacy = joined_metrics[joined_metrics["reach_id"].isin(legacy_reaches)].copy()
    legacy["lowflow_abs_bias_improved"] = legacy["lowflow_absolute_median_log_bias_s1"] < legacy["lowflow_absolute_median_log_bias_s0"]
    legacy.to_csv(REPORT / "legacy_lowflow_target_comparison.csv", index=False, encoding="utf-8-sig")

    shijiao = joined_metrics[joined_metrics["station_name"].map(normalize_name).eq("石角")]
    shijiao_delta = float(shijiao["delta_log_nse"].iloc[0]) if len(shijiao) == 1 else np.nan
    highflow_relative_change = float(s1["median_highflow_log_rmse"] / max(s0["median_highflow_log_rmse"], EPS) - 1.0)
    guard_checks = {
        "median_log_nse_not_down_more_than_0_02": delta["median_log_nse"] >= -0.02,
        "median_kge_not_down_more_than_0_02": delta["median_kge"] >= -0.02,
        "median_absolute_pbias_not_worse_more_than_2pp": delta["median_absolute_pbias_pct"] <= 2.0,
        "good_station_loss_not_more_than_5": delta["good_count"] >= -5,
        "highflow_log_rmse_not_worse_more_than_5pct": highflow_relative_change <= 0.05,
        "shijiao_log_nse_not_down_more_than_0_05": bool(np.isfinite(shijiao_delta) and shijiao_delta >= -0.05),
    }
    guard_checks = {key: bool(value) for key, value in guard_checks.items()}
    result = {
        "run_id": RUN.name,
        "runtime": RUNTIME,
        "paired_rows": int(len(paired)),
        "paired_stations": int(paired["station_name"].nunique()),
        "s0_rows": int(len(old)),
        "s1_rows": int(len(new)),
        "all_rows_paired": bool(len(paired) == len(old) == len(new)),
        "observations_identical": bool(observed_diff.max() <= 1.0e-10),
        "maximum_observed_difference_m3s": float(observed_diff.max()),
        "s0": s0,
        "s1": s1,
        "delta_s1_minus_s0": delta,
        "fold_summary_rows": fold_summaries,
        "legacy_registry_reach_count": int(len(legacy_reaches)),
        "legacy_targets_evaluable": int(len(legacy)),
        "legacy_lowflow_abs_bias_improved_count": int(legacy["lowflow_abs_bias_improved"].sum()),
        "affected_named_stations_found": int(len(affected_rows)),
        "highflow_log_rmse_relative_change": highflow_relative_change,
        "shijiao_log_nse_delta": shijiao_delta,
        "guard_checks": guard_checks,
        "performance_guard_passed": all(guard_checks.values()),
    }
    result["decision"] = "PERFORMANCE_GUARD_PASS" if result["performance_guard_passed"] else "PERFORMANCE_DEGRADED"
    (REPORT / "performance_guard.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "# 空间矫正前后Q72公平比较", "",
        f"- 共同OOF：{result['paired_rows']}行、{result['paired_stations']}站；实测完全一致：{result['observations_identical']}",
        f"- median log-NSE：{s0['median_log_nse']:.6f} → {s1['median_log_nse']:.6f}（{delta['median_log_nse']:+.6f}）",
        f"- median KGE：{s0['median_kge']:.6f} → {s1['median_kge']:.6f}（{delta['median_kge']:+.6f}）",
        f"- median |PBIAS|：{s0['median_absolute_pbias_pct']:.3f}% → {s1['median_absolute_pbias_pct']:.3f}%（{delta['median_absolute_pbias_pct']:+.3f}个百分点）",
        f"- median低流log-RMSE：{s0['median_lowflow_log_rmse']:.6f} → {s1['median_lowflow_log_rmse']:.6f}（{delta['median_lowflow_log_rmse']:+.6f}）",
        f"- median高流log-RMSE相对变化：{highflow_relative_change:+.2%}",
        f"- good站：{s0['good_count']} → {s1['good_count']}（{delta['good_count']:+d}）",
        f"- legacy低流目标：registry {len(legacy_reaches)}个Reach，本轮可评价{len(legacy)}站，其中{int(legacy['lowflow_abs_bias_improved'].sum())}站绝对低流偏差减小",
        f"- 石角log-NSE变化：{shijiao_delta:+.6f}",
        f"- 性能保护门禁：`{result['decision']}`", "",
        "空间矫正是否成立由坐标、河网、DEM流向和Catchment门禁决定；本表只判定矫正后基线的性能代价，不用于撤销空间证据。",
    ]
    (REPORT / "technical_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_rows_paired"] or not result["observations_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
