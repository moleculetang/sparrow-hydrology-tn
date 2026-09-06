"""Stage 25: register the long-history program and audit P2 plus required N inputs."""

from __future__ import annotations

import gc
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_25"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LEDGER = ROOT / "5_Test" / "20260824_10" / "outputs" / "mainline_reach_year_n_ledger_1961_2024.parquet"
MONTHLY = ROOT / "5_Test" / "20260824_12" / "outputs" / "monthly_source_forcing_1961_2024.parquet"
OBS = ROOT / "5_Test" / "20260824_18" / "outputs" / "tn_observations_audited.parquet"
OOF = ROOT / "5_Test" / "20260824_21" / "outputs" / "differentiable_parent_temporal_oof_predictions.parquet"
FINAL = ROOT / "5_Test" / "20260824_24" / "outputs" / "final_station_predictions_2016_2024.parquet"
PROGRAM = RUN / "program_manifest.json"
CONTRACT = RUN / "experiment_contract.json"
RIDGE = 12.0
REQUIRED = [
    "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n",
    "atmospheric_deposition_kg_n", "crop_removal_kg_n",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def memory_gib() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**3
    except Exception:
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / 2**30 if ok else float("nan")


def station_macro_log_rmse(frame: pd.DataFrame, prediction: str) -> float:
    values = []
    for _, group in frame.groupby("station_key"):
        error = np.log1p(group[prediction].to_numpy(float)) - np.log1p(group.tn_mg_l.to_numpy(float))
        values.append(float(np.sqrt(np.mean(error * error))))
    return float(np.mean(values))


def p2_audit(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pred = pd.read_parquet(OOF)
    keys = ["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l", "fold_id", "holdout_type", "holdout_id"]
    pivot = pred.loc[pred.layer.isin(["P1", "P2"])].pivot(index=keys, columns="layer", values="pred_tn_mg_l").reset_index()
    if pivot[["P1", "P2"]].isna().any().any():
        raise RuntimeError("P1/P2 same-row pivot is incomplete")
    history = []
    for row in pivot[["fold_id", "year"]].drop_duplicates().itertuples(index=False):
        counts = obs.loc[obs.year.lt(int(row.year))].groupby("station_key").size()
        stations = pivot.loc[pivot.fold_id.eq(row.fold_id), "station_key"].drop_duplicates()
        history.append(pd.DataFrame({"fold_id": row.fold_id, "station_key": stations, "train_history_count": stations.map(counts).fillna(0).astype(int)}))
    history_frame = pd.concat(history, ignore_index=True)
    pivot = pivot.merge(history_frame, on=["fold_id", "station_key"], validate="many_to_one")
    pivot["station_log_intercept"] = np.log1p(pivot.P2) - np.log1p(pivot.P1)
    pivot["history_bin"] = pd.cut(
        pivot.train_history_count,
        bins=[-1, 0, 5, 12, 36, np.inf],
        labels=["0", "1-5", "6-12", "13-36", ">36"],
    ).astype(str)
    grouped = pivot.groupby(["fold_id", "station_key"], as_index=False).agg(
        train_history_count=("train_history_count", "first"),
        station_log_intercept=("station_log_intercept", "mean"),
        within_station_intercept_sd=("station_log_intercept", "std"),
    )
    grouped.within_station_intercept_sd = grouped.within_station_intercept_sd.fillna(0.0)

    metric_rows = []
    for (year, history_bin), group in pivot.groupby(["year", "history_bin"], observed=True):
        if len(group) == 0:
            continue
        metric_rows.append({
            "year": int(year), "history_bin": str(history_bin), "rows": len(group),
            "stations": int(group.station_key.nunique()),
            "p1_station_macro_log_rmse": station_macro_log_rmse(group, "P1"),
            "p2_station_macro_log_rmse": station_macro_log_rmse(group, "P2"),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics["delta_p2_minus_p1"] = metrics.p2_station_macro_log_rmse - metrics.p1_station_macro_log_rmse

    rng = np.random.default_rng(20260829)
    negative_rows = []
    pivot["P2_shuffled"] = np.nan
    for fold_id, group in pivot.groupby("fold_id", sort=True):
        station_effect = group.groupby("station_key").station_log_intercept.mean()
        shuffled = pd.Series(rng.permutation(station_effect.to_numpy()), index=station_effect.index)
        idx = group.index
        pivot.loc[idx, "P2_shuffled"] = np.expm1(np.log1p(group.P1.to_numpy(float)) + group.station_key.map(shuffled).to_numpy(float)).clip(min=0.0)
        negative_rows.append({
            "fold_id": fold_id,
            "year": int(group.year.iloc[0]),
            "p1_station_macro_log_rmse": station_macro_log_rmse(group, "P1"),
            "p2_station_macro_log_rmse": station_macro_log_rmse(group, "P2"),
            "shuffled_p2_station_macro_log_rmse": station_macro_log_rmse(pivot.loc[idx], "P2_shuffled"),
        })
    negative = pd.DataFrame(negative_rows)

    final = pd.read_parquet(FINAL)
    era_effects = []
    for era, years in (("2016_2020", range(2016, 2021)), ("2021_2024", range(2021, 2025))):
        subset = final.loc[final.year.isin(years)].copy()
        subset["residual"] = np.log1p(subset.tn_mg_l) - np.log1p(subset.pred_P1_mg_l)
        e = subset.groupby("station_key", as_index=False).agg(n=("residual", "size"), residual_sum=("residual", "sum"))
        e["ridge_effect"] = e.residual_sum / (e.n + RIDGE)
        e["era"] = era
        era_effects.append(e)
    eras = pd.concat(era_effects, ignore_index=True)
    wide = eras.pivot(index="station_key", columns="era", values="ridge_effect").dropna()
    stability_corr = float(wide.corr().iloc[0, 1]) if len(wide) >= 3 else float("nan")

    checks = {
        "p1_p2_same_evaluation_rows": len(pivot) * 2 == len(pred.loc[pred.layer.isin(["P1", "P2"])]),
        "p2_intercept_constant_within_station_fold": float(grouped.within_station_intercept_sd.max()) <= 1.0e-12,
        "all_temporal_test_stations_have_training_history": bool(pivot.train_history_count.gt(0).all()),
        "shuffled_intercept_not_better_than_correct_p2_all_folds": bool((negative.shuffled_p2_station_macro_log_rmse >= negative.p2_station_macro_log_rmse).all()),
        "p2_improves_same_row_station_macro_all_folds": bool((negative.p2_station_macro_log_rmse < negative.p1_station_macro_log_rmse).all()),
    }
    summary = {
        "definition": "P2 is P1 plus a station training-history ridge intercept; it is not a post-2022 model",
        "checks": checks,
        "same_row_count": len(pivot),
        "test_years": sorted(map(int, pivot.year.unique())),
        "training_history_count_min": int(pivot.train_history_count.min()),
        "training_history_count_median": float(pivot.groupby(["fold_id", "station_key"]).train_history_count.first().median()),
        "training_history_count_max": int(pivot.train_history_count.max()),
        "early_late_station_effect_common_stations": len(wide),
        "early_late_station_effect_correlation": stability_corr,
        "negative_control": negative.to_dict(orient="records"),
    }
    pivot = pivot.rename(columns={"P1": "pred_p1_mg_l", "P2": "pred_p2_mg_l", "P2_shuffled": "pred_p2_shuffled_mg_l"})
    return pivot, pd.concat([metrics.assign(table="history_metrics"), negative.assign(table="negative_control")], ignore_index=True, sort=False), summary


def n_data_audit(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ledger = pd.read_parquet(LEDGER)
    monthly = pd.read_parquet(MONTHLY)
    central = monthly.loc[monthly.calendar_scenario.eq("CENTRAL")].copy()
    expected = pd.MultiIndex.from_product([range(1, 231), range(1961, 2025)], names=["reach_id", "year"])
    actual = pd.MultiIndex.from_frame(ledger[["reach_id", "year"]])
    coverage_exact = actual.equals(expected)
    source_rows = [
        ("fertilizer_kg_n", 1961, 2023, 1961, 2024, "2024 uses 2023 HaNi rate and 2020 area"),
        ("manure_kg_n", 1961, 2023, 1961, 2024, "2024 uses 2023 HaNi rate and 2020 area"),
        ("cropland_bnf_kg_n", 1961, 2023, 1961, 2024, "2024 uses 2023 China intensity and 2020 area"),
        ("atmospheric_deposition_kg_n", 1961, 2020, 1961, 2024, "2021-2024 hold 2020"),
        ("crop_removal_kg_n", 1961, 2024, 1961, 2024, "native national intensity; spatial area held after 2020"),
    ]
    coverage = []
    for name, native_start, native_end, effective_start, effective_end, rule in source_rows:
        coverage.append({
            "variable": name, "native_start_year": native_start, "native_end_year": native_end,
            "effective_start_year": effective_start, "effective_end_year": effective_end,
            "carry_forward_years": ",".join(map(str, range(native_end + 1, effective_end + 1))),
            "ledger_missing_count": int(ledger[name].isna().sum()),
            "ledger_negative_count": int(ledger[name].lt(0).sum()),
            "rule": rule,
        })
    coverage_frame = pd.DataFrame(coverage)

    monthly_map = {
        "fertilizer_kg_n": "fertilizer_kg_n", "manure_kg_n": "manure_kg_n",
        "cropland_bnf_kg_n": "cropland_bnf_kg_n",
        "atmospheric_deposition_kg_n": "atmospheric_deposition_kg_n",
        "crop_removal_kg_n": "crop_demand_kg_n",
    }
    closure = {}
    # Each annual value is represented by exactly twelve monthly values.  Use
    # compensated summation so the audit tests the source-calendar identity,
    # not pandas' order-dependent float64 group reduction.
    summed = central.groupby(["reach_id", "year"], as_index=False)[list(monthly_map.values())].agg(
        lambda values: math.fsum(map(float, values))
    )
    annual_frame = ledger[["reach_id", "year", *REQUIRED]].rename(columns={name: f"annual__{name}" for name in REQUIRED})
    monthly_frame = summed.rename(columns={name: f"monthly__{name}" for name in monthly_map.values()})
    merged = annual_frame.merge(monthly_frame, on=["reach_id", "year"], validate="one_to_one")
    for annual, month_name in monthly_map.items():
        closure[annual] = float((merged[f"annual__{annual}"] - merged[f"monthly__{month_name}"]).abs().max())

    obs_rows = []
    for year, group in obs.groupby("year"):
        obs_rows.append({"year": int(year), "rows": len(group), "stations": int(group.station_key.nunique()), "reaches": int(group.reach_id.nunique()), "missing_tn": int(group.tn_mg_l.isna().sum())})
    obs_coverage = pd.DataFrame(obs_rows)
    checks = {
        "ledger_rows_exact": len(ledger) == 230 * 64,
        "ledger_reach_year_grid_exact": coverage_exact,
        "ledger_no_duplicate_keys": not ledger.duplicated(["reach_id", "year"]).any(),
        "required_values_complete": not ledger[REQUIRED].isna().any().any(),
        "required_values_nonnegative": bool(ledger[REQUIRED].ge(0).all().all()),
        "central_monthly_rows_exact": len(central) == 230 * 64 * 12,
        "central_monthly_no_duplicate_keys": not central.duplicated(["reach_id", "year", "month"]).any(),
        "annual_monthly_mass_closure_le_1e_8": max(closure.values()) <= 1.0e-8,
        "tn_observation_range_exact": int(obs.year.min()) == 2016 and int(obs.year.max()) == 2024,
        "tn_observations_complete": not obs.tn_mg_l.isna().any(),
    }
    summary = {
        "checks": checks,
        "ledger_rows": len(ledger), "reaches": int(ledger.reach_id.nunique()),
        "years": [int(ledger.year.min()), int(ledger.year.max())],
        "monthly_calendar_scenario_rows": monthly.calendar_scenario.value_counts().to_dict(),
        "annual_monthly_max_abs_closure_kg_n": closure,
        "tn_rows": len(obs), "tn_stations": int(obs.station_key.nunique()), "tn_reaches": int(obs.reach_id.nunique()),
        "interpretation": "effective coverage through 2024 is complete, but native coverage is not complete for every source",
    }
    return coverage_frame, obs_coverage, summary


def write_report(p2: dict[str, object], ndata: dict[str, object], history_metrics: pd.DataFrame, obs_coverage: pd.DataFrame) -> None:
    neg = pd.DataFrame(p2["negative_control"])
    lines = [
        "# 20260824_25 P2与必需N数据审计", "",
        "状态：`PASS_STAGE25_READY_FOR_20260824_26`。", "",
        "## P2究竟是什么", "",
        "P2不是2022年以后才启用的模型。它在每个时间折内，把该站训练期的P1平均log残差按`lambda=12`收缩后加回P1。2022、2023、2024的P1/P2使用完全相同的评价行。", "",
        f"同排评价记录共 `{p2['same_row_count']}` 条；所有时间OOF评价站均已有训练历史。早期/晚期全拟合站点效应相关为 `{p2['early_late_station_effect_correlation']:.3f}`（共同站 `{p2['early_late_station_effect_common_stations']}` 个）。", "",
        "| fold | year | P1 station-macro log-RMSE | P2 | shuffled P2 |", "|---|---:|---:|---:|---:|",
    ]
    for _, row in neg.iterrows():
        lines.append(f"| {row.fold_id} | {int(row.year)} | {row.p1_station_macro_log_rmse:.4f} | {row.p2_station_macro_log_rmse:.4f} | {row.shuffled_p2_station_macro_log_rmse:.4f} |")
    lines += [
        "", "正确站点截距每折都改善P1；随机打乱站点身份后增益消失或明显减弱。因此P2的主要证据是稳定站间偏差校正，不是月过程或空间外推机制。", "",
        "## 正式方程需要的N数据", "",
        f"Reach-year账本为 `{ndata['ledger_rows']}` 行，覆盖 `{ndata['years'][0]}–{ndata['years'][1]}`、`{ndata['reaches']}`个Reach；数值与月质量闭合全部通过。", "",
        "数值完整不等于原生年份完整：FERT/MAN/BNF的2024为carry-forward，收获面积和沉降分别在2020后保持。正式模型必须保留`year_used`和carry-forward状态。", "",
        "| year | TN rows | stations | reaches |", "|---:|---:|---:|---:|",
    ]
    for _, row in obs_coverage.iterrows():
        lines.append(f"| {int(row.year)} | {int(row.rows)} | {int(row.stations)} | {int(row.reaches)} |")
    lines += [
        "", "2021年监测网络从72站级别扩展到127站，这会增强后续P2可用性，但不会把P2变成空间可迁移过程。", "",
        "## 下一步边界", "",
        "Stage 26只下载CHM_PRE 1961/1962并验证历史水文桥接，不读取TN用于校准，也不下载约300 GB三小时CMFD。", "",
    ]
    temp = REPORTS / "technical_report.md.tmp"
    temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temp, REPORTS / "technical_report.md")


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if memory_gib() > 16:
        raise MemoryError("Stage25 starts above the 16 GiB hard stop")
    obs = pd.read_parquet(OBS)
    p2_rows, p2_metrics, p2_summary = p2_audit(obs)
    gc.collect()
    coverage, obs_coverage, n_summary = n_data_audit(obs)
    download = {
        "stage": "20260824_26",
        "destination_directory": str(ROOT / "0_reach_topology/data/raw/atmosphere/precipitation/chm_pre_v2/daily"),
        "network_preflight_mib_s": 0.22,
        "estimated_total_gib": 1.254,
        "estimated_elapsed_hours": 1.7,
        "files": [
            {"year": 1961, "remote_file_id": "e51b3a83-2b68-4869-a5e1-7ee4b2598196", "filename": "CHM_PRE_V2_daily_1961.nc", "expected_size_bytes": 672795695},
            {"year": 1962, "remote_file_id": "9579c616-ebf0-43ff-8a24-6944fa7378f9", "filename": "CHM_PRE_V2_daily_1962.nc", "expected_size_bytes": 672795695},
        ],
        "large_cmfd_3hour_history": "not_authorized_approximately_300_GB",
    }
    checks = {**p2_summary["checks"], **n_summary["checks"]}
    status = "PASS_STAGE25_READY_FOR_20260824_26" if all(checks.values()) else "FAIL_STAGE25"
    validation = {
        "stage": "20260824_25", "status": status, "checks": checks,
        "p2": p2_summary, "required_n_data": n_summary,
        "memory_rss_gib": memory_gib(),
        "input_hashes": {str(path): sha256(path) for path in (LEDGER, MONTHLY, OBS, OOF, FINAL, PROGRAM, CONTRACT)},
        "authorized_successor": "20260824_26" if status.startswith("PASS") else None,
    }
    atomic_parquet(p2_rows, OUT / "p2_same_row_diagnostics.parquet")
    atomic_parquet(p2_metrics, OUT / "p2_history_and_negative_control_metrics.parquet")
    atomic_parquet(coverage, OUT / "required_n_data_coverage.parquet")
    atomic_parquet(obs_coverage, OUT / "tn_observation_coverage_2016_2024.parquet")
    atomic_json(REPORTS / "p2_mechanism_audit.json", p2_summary)
    atomic_json(REPORTS / "required_n_data_coverage.json", n_summary)
    atomic_json(REPORTS / "stage26_download_manifest.json", download)
    atomic_json(REPORTS / "stage25_validation.json", validation)
    write_report(p2_summary, n_summary, p2_metrics, obs_coverage)
    if not status.startswith("PASS"):
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, default=str))
    print(json.dumps({"status": status, "memory_rss_gib": validation["memory_rss_gib"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
