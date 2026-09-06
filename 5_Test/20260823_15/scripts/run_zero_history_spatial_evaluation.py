from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


warnings.simplefilter("ignore", PerformanceWarning)
ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_15"
TRAIN_OBS = TEST / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
COVERAGE = TEST / "20260823_14" / "outputs" / "final_user_locked_station_coverage.parquet"
ALL_MONTHLY = TEST / "20260823_14" / "outputs" / "all_monthly_discharge_after_exclusions.parquet"
SUPPLEMENTAL_COVERAGE = TEST / "20260823_14" / "outputs" / "supplemental_zero_history_station_coverage.parquet"
SUPPLEMENTAL_OBS = TEST / "20260823_14" / "outputs" / "supplemental_zero_history_station_month_observations.parquet"
INPUT = TEST / "20260813_54" / "inputs" / "parent_indata.parquet"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
COMPONENT = TEST / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
MODELING_DIR = TEST / "20260823_13" / "scripts"
TIME_LOCK = RUN / "final_reports" / "four_group_training_lock.json"
OUT = RUN / "final_outputs"
REPORT = RUN / "final_reports"
MIN_TARGET_REACHES = 5
REGISTERED_TARGETS = {
    "岳城": 55,
    "仁化（三）": 97,
    "珠坑": 44,
    "昭平": 84,
    "水口": 122,
    "定安（二）": 133,
    "瓦村（二）": 134,
    "盘江桥（三）": 206,
}

sys.path.insert(0, str(MODELING_DIR))
from modeling import (  # noqa: E402
    fit_gaussian_map,
    load_q72_component,
    make_model_frame,
    metric_dict,
    physical_fields,
    predict_gaussian_map,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_static_fields(base: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(
        INPUT,
        columns=["comid", "year", "month", "station_id", "LENGTHKM", "SLOPE", "MaxElSmoCm"],
    ).sort_values(["comid", "year", "month"]).reset_index(drop=True)
    base = base.sort_values(["comid", "year", "month"]).reset_index(drop=True)
    if not base[["comid", "year", "month"]].equals(frozen[["comid", "year", "month"]]):
        raise RuntimeError("Frozen static-attribute key alignment failed")
    for column in ["station_id", "LENGTHKM", "SLOPE", "MaxElSmoCm"]:
        base[column] = frozen[column].to_numpy()
    return base


def station_metrics(frame: pd.DataFrame, prediction: str, candidate: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (station, reach), part in frame.groupby(["station_norm", "reach_id"], sort=False):
        values = metric_dict(part.Q_obsv_cfs.to_numpy(float), part[prediction].to_numpy(float))
        rows.append({
            "candidate": candidate,
            "station_norm": str(station),
            "reach_id": int(reach),
            **values,
        })
    return pd.DataFrame(rows)


def summary(frame: pd.DataFrame, prediction: str, candidate: str) -> tuple[dict[str, object], pd.DataFrame]:
    station = station_metrics(frame, prediction, candidate)
    pooled = metric_dict(frame.Q_obsv_cfs.to_numpy(float), frame[prediction].to_numpy(float))
    return {
        "candidate": candidate,
        "target_stations": int(station.station_norm.nunique()),
        "target_reaches": int(station.reach_id.nunique()),
        **pooled,
        "station_median_NSE": float(station.NSE.median()),
        "station_mean_NSE": float(station.NSE.mean()),
        "station_mean_RMSE_log": float(station.RMSE_log.mean()),
        "station_median_RMSE_log": float(station.RMSE_log.median()),
    }, station


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    header = "| " + " | ".join(map(str, values.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def paired_station_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    reference: str = "H0_Q72_PROCESS",
    seed: int = 20260823,
) -> dict[str, object]:
    candidate_metrics = station_metrics(frame, candidate, candidate)[
        ["station_norm", "reach_id", "RMSE_log"]
    ].rename(columns={"RMSE_log": "candidate_RMSE_log"})
    reference_metrics = station_metrics(frame, reference, reference)[
        ["station_norm", "reach_id", "RMSE_log"]
    ].rename(columns={"RMSE_log": "reference_RMSE_log"})
    paired = candidate_metrics.merge(reference_metrics, on=["station_norm", "reach_id"], validate="one_to_one")
    delta = paired.candidate_RMSE_log.to_numpy(float) - paired.reference_RMSE_log.to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(10000, len(delta)))
    boot = delta[indices].mean(axis=1)
    return {
        "candidate": candidate,
        "reference": reference,
        "stations": int(len(delta)),
        "point_delta_station_mean_RMSE_log": float(delta.mean()),
        "CI95_lower": float(np.quantile(boot, 0.025)),
        "CI95_upper": float(np.quantile(boot, 0.975)),
        "point_improved_station_count": int((delta < 0).sum()),
        "point_improved_station_fraction": float((delta < 0).mean()),
    }


def flow_regime_bootstrap(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    work = frame.copy()
    quantiles = work.groupby("station_norm").H0_Q72_PROCESS.quantile([0.2, 0.8]).unstack()
    work["q20"] = work.station_norm.map(quantiles[0.2])
    work["q80"] = work.station_norm.map(quantiles[0.8])
    work["regime"] = np.select(
        [work.H0_Q72_PROCESS.le(work.q20), work.H0_Q72_PROCESS.ge(work.q80)],
        ["low", "high"],
        default="middle",
    )
    rows = []
    for index, regime in enumerate(["low", "middle", "high"]):
        row = paired_station_bootstrap(work[work.regime.eq(regime)], candidate, seed=20260824 + index)
        row["regime"] = regime
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    time_lock = json.loads(TIME_LOCK.read_text(encoding="utf-8"))
    training_obs = pd.read_parquet(TRAIN_OBS).copy()
    training_obs["q_site"] = training_obs.station_norm.astype(str)
    training_obs = training_obs[training_obs.year.le(2018)].copy()
    training_stations = set(training_obs.station_norm.astype(str))
    training_reaches = set(training_obs.reach_id.astype(int))

    coverage = pd.read_parquet(COVERAGE)
    target_coverage = coverage[
        ~coverage.reach_id.astype(int).isin(training_reaches)
        & coverage.station_type.eq("ordinary_river_gauge")
        & coverage.snap_distance_m.le(5000)
        & coverage.total_months.ge(12)
    ].copy()
    supplemental_coverage = pd.read_parquet(SUPPLEMENTAL_COVERAGE)
    supplemental_coverage = supplemental_coverage[
        supplemental_coverage.zero_history_candidate
        & supplemental_coverage.station_type.eq("ordinary_river_gauge")
        & supplemental_coverage.usable_months.ge(12)
    ].copy()
    target_coverage = pd.concat([
        target_coverage[["station_norm", "reach_id"]],
        supplemental_coverage[["station_norm", "reach_id"]],
    ], ignore_index=True).drop_duplicates(["station_norm", "reach_id"])
    observed_targets = dict(zip(target_coverage.station_norm.astype(str), target_coverage.reach_id.astype(int)))
    if observed_targets != REGISTERED_TARGETS:
        raise RuntimeError(f"Registered target set changed: {observed_targets}")

    monthly = pd.read_parquet(ALL_MONTHLY)
    supplemental_obs = pd.read_parquet(SUPPLEMENTAL_OBS)
    if set(supplemental_obs.station_norm.astype(str)) != set(supplemental_coverage.station_norm.astype(str)):
        raise RuntimeError("Supplemental coordinate coverage and observation stations differ")
    target_obs = monthly[
        monthly.station_norm.astype(str).isin(REGISTERED_TARGETS)
        & monthly.Q_obsv_cfs.notna()
        & monthly.Q_obsv_cfs.gt(0)
    ].copy()
    target_obs["reach_id"] = target_obs.station_norm.map(REGISTERED_TARGETS).astype(int)
    target_obs["q_site"] = target_obs.station_norm.astype(str)
    target_obs = target_obs.sort_values(["station_norm", "year", "month"]).reset_index(drop=True)
    if set(target_obs.station_norm.astype(str)) & training_stations:
        raise RuntimeError("A zero-history target station appears in the training station set")
    if set(target_obs.reach_id.astype(int)) & training_reaches:
        raise RuntimeError("A zero-history target Reach appears in the training Reach set")
    if target_obs.duplicated(["station_norm", "year", "month"]).any():
        raise RuntimeError("Duplicate target station-month rows")
    if target_coverage.station_norm.duplicated().any() or target_coverage.reach_id.duplicated().any():
        raise RuntimeError("Zero-history contract requires exactly one station per unseen Reach")

    module = load_q72_component(COMPONENT, INPUT, TOPOLOGY)
    forcing = module.load_forcing_panel().sort_values(["comid", "year", "month"]).reset_index(drop=True)
    base = module.add_hydrologic_features(
        forcing,
        rho=0.70,
        wm=480.0,
        et_gamma=0.75,
        sas_rho=0.93,
        young_k=1.5,
        storage_scale=720.0,
        prod_capacity=240.0,
        runoff_gamma=2.5,
        quick_rho=0.25,
        base_rho=0.85,
        base_release=0.10,
    )
    base = add_static_fields(base)
    _, _, upstream = module._panel_layout(base)
    physical = physical_fields(module, base, time_lock["physical_parameters"], upstream)
    all_reach = make_model_frame(base, physical)
    feature_cols = [c for c in all_reach.columns if c not in {"q_site", "station_id", "Q_obsv_cfs"}]

    train = training_obs.merge(
        all_reach[feature_cols].rename(columns={"comid": "reach_id"}),
        on=["reach_id", "year", "month"], how="left", validate="many_to_one",
    )
    target = target_obs.merge(
        all_reach[feature_cols].rename(columns={"comid": "reach_id"}),
        on=["reach_id", "year", "month"], how="left", validate="many_to_one",
    )
    train["comid"] = train.reach_id.astype(int)
    target["comid"] = target.reach_id.astype(int)
    if train[feature_cols[3:]].isna().all(axis=1).any() or target[feature_cols[3:]].isna().all(axis=1).any():
        raise RuntimeError("Hydrologic feature merge failed")

    global_sigma = float(time_lock["global_sigma"])
    station_sigma = float(time_lock["station_sigma"])
    station_list = sorted(training_stations)
    h1_model = fit_gaussian_map(train, global_sigma)
    h2_model = fit_gaussian_map(train, global_sigma, station_list, station_sigma)
    target["H0_Q72_PROCESS"] = target.q72_routed_total_cfs.to_numpy(float)
    target["H1_TRANSFERABLE_GLOBAL_MAP"] = predict_gaussian_map(target, h1_model)
    target["H2_GAUGED_MAP_COLD_START"] = predict_gaussian_map(target, h2_model)
    target["H3_ASSIMILATION_COLD_START"] = target.H2_GAUGED_MAP_COLD_START
    cold_identity_error = float(np.max(np.abs(
        target.H3_ASSIMILATION_COLD_START - target.H2_GAUGED_MAP_COLD_START
    )))
    if cold_identity_error > 1e-12:
        raise RuntimeError(f"Cold-start assimilation identity failed: {cold_identity_error}")

    candidates = [
        "H0_Q72_PROCESS",
        "H1_TRANSFERABLE_GLOBAL_MAP",
        "H2_GAUGED_MAP_COLD_START",
        "H3_ASSIMILATION_COLD_START",
    ]
    summaries: list[dict[str, object]] = []
    station_parts: list[pd.DataFrame] = []
    for candidate in candidates:
        row, station = summary(target, candidate, candidate)
        summaries.append(row)
        station_parts.append(station)
    summary_df = pd.DataFrame(summaries)
    station_df = pd.concat(station_parts, ignore_index=True)
    reference = station_df[station_df.candidate.eq("H0_Q72_PROCESS")][
        ["station_norm", "reach_id", "RMSE_log"]
    ].rename(columns={"RMSE_log": "reference_RMSE_log"})
    paired = station_df.merge(reference, on=["station_norm", "reach_id"], validate="many_to_one")
    paired["delta_RMSE_log_vs_H0"] = paired.RMSE_log - paired.reference_RMSE_log
    comparisons = pd.DataFrame([
        paired_station_bootstrap(target, candidate)
        for candidate in candidates[1:]
    ])
    flow_comparisons = pd.concat([
        flow_regime_bootstrap(target, candidate)
        for candidate in candidates[1:]
    ], ignore_index=True)

    target[[
        "station_norm", "reach_id", "year", "month", "Q_obsv_cfs", *candidates,
    ]].to_parquet(OUT / "zero_history_spatial_predictions.parquet", index=False)
    summary_df.to_parquet(OUT / "zero_history_spatial_summary_metrics.parquet", index=False)
    paired.to_parquet(OUT / "zero_history_spatial_station_metrics.parquet", index=False)
    comparisons.to_parquet(OUT / "zero_history_spatial_paired_bootstrap.parquet", index=False)
    flow_comparisons.to_parquet(OUT / "zero_history_spatial_flow_regime_bootstrap.parquet", index=False)

    sample_sufficient = target.reach_id.nunique() >= MIN_TARGET_REACHES
    h0 = summary_df[summary_df.candidate.eq("H0_Q72_PROCESS")].iloc[0]
    h1 = summary_df[summary_df.candidate.eq("H1_TRANSFERABLE_GLOBAL_MAP")].iloc[0]
    h1_comparison = comparisons[comparisons.candidate.eq("H1_TRANSFERABLE_GLOBAL_MAP")].iloc[0]
    h1_flow = flow_comparisons[flow_comparisons.candidate.eq("H1_TRANSFERABLE_GLOBAL_MAP")]
    performance_gates = {
        "station_macro_RMSE_log_improved_CI95_upper_lt_0": bool(h1_comparison.CI95_upper < 0),
        "pooled_NSE_nonworse_margin_0p01": bool(h1.NSE >= h0.NSE - 0.01),
        "station_median_NSE_nonworse_margin_0p02": bool(h1.station_median_NSE >= h0.station_median_NSE - 0.02),
        "low_flow_noninferior_CI95_upper_lt_0p005": bool(
            h1_flow.loc[h1_flow.regime.eq("low"), "CI95_upper"].iloc[0] < 0.005
        ),
        "high_flow_noninferior_CI95_upper_lt_0p005": bool(
            h1_flow.loc[h1_flow.regime.eq("high"), "CI95_upper"].iloc[0] < 0.005
        ),
        "point_RMSE_log_improved_at_least_75pct_targets": bool(
            h1_comparison.point_improved_station_fraction >= 0.75
        ),
    }
    h1_spatial_pass = bool(sample_sufficient and all(performance_gates.values()))
    decision = {
        "stage": "20260823_15",
        "status": "PASS",
        "evaluation": "ZERO_HISTORY_UNSEEN_STATION_AND_REACH",
        "training_stations": len(training_stations),
        "training_reaches": len(training_reaches),
        "target_stations": int(target.station_norm.nunique()),
        "target_reaches": int(target.reach_id.nunique()),
        "target_rows": int(len(target)),
        "targets": observed_targets,
        "target_station_overlap_with_training": 0,
        "target_reach_overlap_with_training": 0,
        "target_observations_used_for_parameter_selection": 0,
        "target_station_specific_effect": 0,
        "target_assimilation_updates": 0,
        "cold_start_H3_equals_H2_max_abs_cfs": cold_identity_error,
        "minimum_target_reaches_for_promotion": MIN_TARGET_REACHES,
        "sample_sufficient_for_promotion": bool(sample_sufficient),
        "H1_transferable_global_map_performance_gates": performance_gates,
        "H1_transferable_global_map_spatial_gate": h1_spatial_pass,
        "all_reach_promotion": "H1_TRANSFERABLE_GLOBAL_MAP" if h1_spatial_pass else "NOT_AUTHORIZED_SPATIAL_GATE_FAILED",
        "gauged_map_time_product": "LOCKED_FOR_105_TRAINING_STATIONS",
        "recursive_assimilation_product": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "training_registry_sha256": sha256(TRAIN_OBS),
        "target_monthly_registry_sha256": sha256(ALL_MONTHLY),
        "supplemental_coordinate_coverage_sha256": sha256(SUPPLEMENTAL_COVERAGE),
        "time_parameter_lock_sha256": sha256(TIME_LOCK),
    }
    (REPORT / "zero_history_spatial_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = (
        "# 零历史空间检验\n\n"
        "目标站及其Reach均未进入105站训练、过程参数选择、MAP拟合或递归状态更新。"
        "现有站点坐标提供岳城和仁化（三）；解码全国9154站图层后，以精确同名且唯一拓扑候选恢复"
        "珠坑、昭平、水口、定安（二）、瓦村（二）和盘江桥（三）。最终共8站、8条独立未见Reach。\n\n"
        + markdown_table(summary_df[[
            "candidate", "target_stations", "n", "NSE", "KGE", "PBIAS_pct",
            "RMSE_log", "station_median_NSE", "station_mean_RMSE_log",
        ]])
        + "\n\n"
        + markdown_table(paired[[
            "candidate", "station_norm", "reach_id", "n", "NSE", "PBIAS_pct",
            "RMSE_log", "delta_RMSE_log_vs_H0",
        ]])
        + "\n\n"
        + markdown_table(comparisons)
        + "\n\n"
        + markdown_table(flow_comparisons)
        + "\n\n"
        + "H1正式门禁：`" + ("PASS" if h1_spatial_pass else "FAIL") + "`。"
        + "全河网升级状态：`" + str(decision["all_reach_promotion"]) + "`。\n"
    )
    (REPORT / "zero_history_spatial_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(summary_df.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
