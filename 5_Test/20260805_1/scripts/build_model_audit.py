from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
CONFIG = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
MODEL = RUN / "reports" / "q72_baseline"
REPORT = RUN / "reports" / "model_audit"
OUTPUT = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
FOLDS = [
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
]
CFS_PER_M3S = 35.3146667
EPS = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1 - np.sum((predicted - observed) ** 2) / denominator) if denominator > EPS else np.nan


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 4 or observed.std(ddof=0) <= EPS or observed.mean() <= EPS:
        return np.nan
    correlation = np.corrcoef(observed, predicted)[0, 1] if predicted.std(ddof=0) > EPS else np.nan
    if not np.isfinite(correlation):
        return np.nan
    alpha = predicted.std(ddof=0) / observed.std(ddof=0)
    beta = predicted.mean() / observed.mean()
    return float(1 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def station_metrics(frame: pd.DataFrame, scenario: str, minimum_months: int = 12) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station, group in frame.groupby("station_name", sort=True):
        group = group.sort_values(["year", "month"])
        observed = group["observed_m3s"].to_numpy(float)
        predicted = np.maximum(group["predicted_m3s"].to_numpy(float), 0.0)
        if len(group) < minimum_months:
            continue
        log_observed = np.log1p(np.maximum(observed, 0.0))
        log_predicted = np.log1p(predicted)
        low = observed <= np.quantile(observed, 0.25)
        high = observed >= np.quantile(observed, 0.75)
        metric_log_nse = nse(log_observed, log_predicted)
        metric_kge = kge(observed, predicted)
        pbias = float(100 * (predicted.sum() - observed.sum()) / max(observed.sum(), EPS))
        rows.append({
            "scenario": scenario,
            "station_name": station,
            "reach_id": int(group["reach_id"].iloc[0]),
            "months": int(len(group)),
            "folds_with_data": int(group["fold_id"].nunique()),
            "nse": nse(observed, predicted),
            "log_nse": metric_log_nse,
            "kge": metric_kge,
            "pbias_pct": pbias,
            "absolute_pbias_pct": abs(pbias),
            "lowflow_log_error": float(np.median(np.abs(log_predicted[low] - log_observed[low]))),
            "highflow_nrmse": float(np.sqrt(np.mean((predicted[high] - observed[high]) ** 2)) / max(np.mean(observed[high]), EPS)),
            "good": bool(metric_log_nse >= 0.5 and metric_kge >= 0.5 and abs(pbias) <= 25),
            "severe_pbias": bool(abs(pbias) > 50),
        })
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> dict[str, float | int]:
    if metrics.empty:
        return {"station_count": 0}
    return {
        "station_count": int(len(metrics)),
        "median_nse": float(metrics["nse"].median()),
        "median_log_nse": float(metrics["log_nse"].median()),
        "median_kge": float(metrics["kge"].median()),
        "median_absolute_pbias_pct": float(metrics["absolute_pbias_pct"].median()),
        "median_lowflow_log_error": float(metrics["lowflow_log_error"].median()),
        "median_highflow_nrmse": float(metrics["highflow_nrmse"].median()),
        "good_count": int(metrics["good"].sum()),
        "severe_pbias_count": int(metrics["severe_pbias"].sum()),
    }


def load_oof(run: Path) -> pd.DataFrame:
    parts = []
    for fold_id in FOLDS:
        path = run / "reports" / "q72_baseline" / "blocked_folds" / fold_id / "evaluation_predictions.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame = frame.rename(columns={"q_site": "station_name", "comid": "reach_id", "actual": "observed_cfs", "predict": "predicted_cfs"})
        frame["fold_id"] = fold_id
        frame["observed_m3s"] = frame["observed_cfs"] / CFS_PER_M3S
        frame["predicted_m3s"] = frame["predicted_cfs"] / CFS_PER_M3S
        parts.append(frame)
    oof = pd.concat(parts, ignore_index=True)
    return oof.sort_values(["fold_id", "station_name", "year", "month"]).reset_index(drop=True)


def build_comparison() -> dict[str, Any] | None:
    run1 = RUN.parent / "20260805_1"
    run2 = RUN.parent / "20260805_2"
    if not all((run / "reports" / "q72_baseline" / "blocked_folds").exists() for run in [run1, run2]):
        return None
    one = load_oof(run1)
    two = load_oof(run2)
    keys = ["fold_id", "station_name", "reach_id", "year", "month"]
    common = one[keys + ["observed_cfs", "predicted_cfs"]].merge(
        two[keys + ["observed_cfs", "predicted_cfs"]], on=keys, how="inner",
        suffixes=("_20260805_1", "_20260805_2"), validate="one_to_one",
    )
    common["max_actual_difference_cfs"] = (common["observed_cfs_20260805_1"] - common["observed_cfs_20260805_2"]).abs()
    comparison_dir = RUN / "reports" / "baseline_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    common.to_csv(comparison_dir / "identical_common_oof_predictions.csv", index=False, encoding="utf-8-sig")

    metric_frames = []
    summaries: dict[str, dict[str, float | int]] = {}
    for label, pred_col in [("20260805_1_common", "predicted_cfs_20260805_1"), ("20260805_2_common", "predicted_cfs_20260805_2")]:
        frame = common[keys + ["observed_cfs_20260805_1", pred_col]].rename(columns={"observed_cfs_20260805_1": "observed_cfs", pred_col: "predicted_cfs"})
        frame["observed_m3s"] = frame["observed_cfs"] / CFS_PER_M3S
        frame["predicted_m3s"] = frame["predicted_cfs"] / CFS_PER_M3S
        metrics = station_metrics(frame, label)
        metric_frames.append(metrics)
        summaries[label] = summarize(metrics)
    common_metrics = pd.concat(metric_frames, ignore_index=True)
    common_metrics.to_csv(comparison_dir / "identical_common_station_metrics.csv", index=False, encoding="utf-8-sig")

    stations1 = set(one["station_name"].unique())
    added = two[~two["station_name"].isin(stations1)].copy()
    added_metrics = station_metrics(added, "20260805_2_added_only")
    added_metrics.to_csv(comparison_dir / "added_station_oof_metrics.csv", index=False, encoding="utf-8-sig")
    summaries["20260805_1_overall"] = summarize(station_metrics(one, "20260805_1_overall"))
    summaries["20260805_2_overall"] = summarize(station_metrics(two, "20260805_2_overall"))
    summaries["20260805_2_added_only"] = summarize(added_metrics)
    s1, s2 = summaries["20260805_1_common"], summaries["20260805_2_common"]
    result = {
        "common_rows": int(len(common)),
        "common_stations": int(common["station_name"].nunique()),
        "common_actuals_identical": bool(common["max_actual_difference_cfs"].max() < 1e-10),
        "added_oof_rows": int(len(added)),
        "added_oof_stations": int(added["station_name"].nunique()),
        "summaries": summaries,
        "common_delta_20260805_2_minus_20260805_1": {
            "median_log_nse": float(s2["median_log_nse"] - s1["median_log_nse"]),
            "median_kge": float(s2["median_kge"] - s1["median_kge"]),
            "median_absolute_pbias_pct": float(s2["median_absolute_pbias_pct"] - s1["median_absolute_pbias_pct"]),
            "median_lowflow_log_error": float(s2["median_lowflow_log_error"] - s1["median_lowflow_log_error"]),
            "good_count": int(s2["good_count"] - s1["good_count"]),
        },
    }
    (comparison_dir / "comparison_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    for directory in [REPORT, OUTPUT, MANIFEST]:
        directory.mkdir(parents=True, exist_ok=True)
    oof = load_oof(RUN)
    key_columns = ["fold_id", "station_name", "reach_id", "year", "month"]
    oof.to_parquet(OUTPUT / "q72_three_fold_oof_predictions.parquet", index=False)
    oof.to_csv(OUTPUT / "q72_three_fold_oof_predictions.csv", index=False, encoding="utf-8-sig")

    fold_summaries = []
    fold_metrics = []
    for fold_id in FOLDS:
        fold = oof[oof["fold_id"].eq(fold_id)]
        metrics = station_metrics(fold, fold_id)
        fold_metrics.append(metrics)
        fold_summaries.append({"fold_id": fold_id, "oof_rows": int(len(fold)), "stations_with_any_oof": int(fold["station_name"].nunique()), **summarize(metrics)})
    pd.concat(fold_metrics, ignore_index=True).to_csv(REPORT / "station_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_summaries).to_csv(REPORT / "fold_summary_metrics.csv", index=False, encoding="utf-8-sig")
    combined_metrics = station_metrics(oof, "three_fold_oof")
    combined_metrics.to_csv(REPORT / "three_fold_oof_station_metrics.csv", index=False, encoding="utf-8-sig")
    summary = summarize(combined_metrics)
    shijiao = combined_metrics[combined_metrics["station_name"].eq("石角站")].copy()
    shijiao.to_csv(REPORT / "protected_shijiao_metrics.csv", index=False, encoding="utf-8-sig")

    model_hashes = {
        "run_q72_baseline.py": sha256(RUN / "scripts" / "run_q72_baseline.py"),
        "fit_monthly_bayes_seasonal_hysteresis.py": sha256(RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py"),
    }
    parent_hash_match = (
        model_hashes["run_q72_baseline.py"] == CONFIG["parent_run_script_sha256"]
        and model_hashes["fit_monthly_bayes_seasonal_hysteresis.py"] == CONFIG["parent_component_sha256"]
    )
    comparison = build_comparison()

    provenance = []
    for root_name in ["inputs", "scripts"]:
        for path in sorted((RUN / root_name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                provenance.append({"role": root_name, "path": str(path.relative_to(RUN)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    provenance.append({"role": "config", "path": "config.json", "bytes": (RUN / "config.json").stat().st_size, "sha256": sha256(RUN / "config.json")})
    pd.DataFrame(provenance).to_csv(MANIFEST / "provenance_manifest.csv", index=False, encoding="utf-8-sig")
    (MANIFEST / "provenance_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in ["numpy", "pandas", "scipy", "matplotlib", "pyarrow", "geopandas"]},
        "runtime": RUNTIME,
        "deterministic_solver": "numpy.linalg.lstsq",
        "random_seeds": [],
    }
    (MANIFEST / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")

    gate = {
        "run_id": RUN.name,
        "cohort": CONFIG["cohort"],
        "q72_only": True,
        "three_blocked_folds": FOLDS,
        "oof_rows": int(len(oof)),
        "oof_stations_any": int(oof["station_name"].nunique()),
        "oof_keys_unique": not bool(oof.duplicated(key_columns).any()),
        "predictions_positive_finite": bool(np.isfinite(oof["predicted_cfs"]).all() and oof["predicted_cfs"].gt(0).all()),
        "evaluation_years_only_2012_2018": bool(oof["year"].between(2012, 2018).all()),
        "parent_q72_code_hashes_exact": parent_hash_match,
        "model_code_hashes": model_hashes,
        "three_fold_oof": summary,
        "protected_shijiao_oof_present": bool(len(shijiao) == 1),
        "comparison_available": comparison is not None,
    }
    gate["passed"] = bool(gate["oof_keys_unique"] and gate["predictions_positive_finite"] and gate["evaluation_years_only_2012_2018"] and gate["parent_q72_code_hashes_exact"] and gate["protected_shijiao_oof_present"])
    (REPORT / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {RUN.name} Q72新基线模型报告", "", f"- 三折OOF行数：{gate['oof_rows']}", f"- 有OOF站数：{gate['oof_stations_any']}",
        f"- 指标合格站数（至少12个OOF月）：{summary['station_count']}", f"- 中位 log-NSE：{summary['median_log_nse']:.6f}",
        f"- 中位 KGE：{summary['median_kge']:.6f}", f"- 中位 |PBIAS|：{summary['median_absolute_pbias_pct']:.3f}%",
        f"- 中位低流 log 误差：{summary['median_lowflow_log_error']:.6f}", f"- good站：{summary['good_count']}",
        f"- Q72父代码哈希完全一致：{parent_hash_match}", f"- 石角OOF存在：{gate['protected_shijiao_oof_present']}",
        "", "本报告只使用2012–2018三折时间外预测；2019–2022未参与拟合器选择或本报告评分。",
    ]
    (REPORT / "model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
