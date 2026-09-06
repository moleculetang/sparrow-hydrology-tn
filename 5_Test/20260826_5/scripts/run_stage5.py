from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW"); RUN = ROOT / "5_Test" / "20260826_5"; OUT = RUN / "outputs"; REPORT = RUN / "reports"
sys.path[:0] = [str(ROOT / "5_Test" / "20260826_4" / "scripts"), str(ROOT / "5_Test" / "20260825_3" / "scripts"), str(ROOT / "5_Test" / "20260825_5" / "scripts")]
from hydrology_core import load_topology, route_instantaneous  # noqa: E402
from regional_structures import PARAMETER_NAMES, periodic_spinup, run_model  # noqa: E402
from run_stage5 import build_support, kge, nse, station_metrics  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DEVELOPMENT_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
PARENT = ROOT / "5_Test" / "20260825_7" / "outputs" / "tn_hydrology_bridge_daily_2006_2022.parquet"
LOCKS = ROOT / "5_Test" / "20260826_4" / "reports"


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): d.update(b)
    return d.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def monthly(dates: pd.DatetimeIndex, values: np.ndarray) -> tuple[pd.PeriodIndex, np.ndarray]:
    periods = dates.to_period("M"); unique = periods.unique(); return unique, np.stack([np.nanmean(values[np.asarray(periods == p)], axis=0) for p in unique])


def summarize(candidate: str, scale: str, obs: np.ndarray, pred: np.ndarray, per_station: pd.DataFrame) -> pd.DataFrame:
    valid = np.isfinite(obs) & np.isfinite(pred); o, p = obs[valid], pred[valid]
    pooled = {"candidate": candidate, "temporal_scale": scale, "scope": "pooled", "NSE": nse(o, p), "log_NSE": nse(np.log1p(o), np.log1p(p)), "KGE": kge(o, p), "PBIAS_pct": 100 * (p.sum() - o.sum()) / o.sum(), "absolute_PBIAS_pct": np.nan, "log_RMSE": float(np.sqrt(np.mean((np.log1p(p) - np.log1p(o)) ** 2))), "stations": len(per_station)}
    rows = [pooled]
    for scope, reducer in [("station_median", "median"), ("station_mean", "mean")]:
        rows.append({"candidate": candidate, "temporal_scale": scale, "scope": scope, "NSE": float(getattr(per_station.NSE, reducer)()), "log_NSE": float(getattr(per_station.log_NSE, reducer)()), "KGE": float(getattr(per_station.KGE, reducer)()), "PBIAS_pct": float(getattr(per_station.PBIAS_pct, reducer)()), "absolute_PBIAS_pct": float(getattr(per_station.PBIAS_pct.abs(), reducer)()), "log_RMSE": float(getattr(per_station.log_RMSE, reducer)()), "stations": len(per_station)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    reach_ids = np.arange(1, 231); order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGES); stations = gauges.loc[gauges.topology_representative].drop_duplicates("station_norm").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True); support = build_support(stations, reach_ids, order, downstream)
    area = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)
    forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"]); forcing.date = pd.to_datetime(forcing.date)
    spin_dates = pd.date_range("2006-01-01", "2009-12-31"); dev_dates = pd.date_range("2010-01-01", "2018-12-31")
    pivot = lambda field, dates: forcing.pivot(index="date", columns="reach_id", values=field).reindex(index=dates, columns=reach_ids).to_numpy(float)
    spin_p, spin_pet = pivot("precipitation_daily_mm", spin_dates), pivot("pet_fao56_mm_day", spin_dates); dev_p, dev_pet = pivot("precipitation_daily_mm", dev_dates), pivot("pet_fao56_mm_day", dev_dates)
    obs = pd.read_parquet(DEVELOPMENT_Q); obs.date = pd.to_datetime(obs.date); observed = obs.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dev_dates, columns=stations.station_norm).to_numpy(float)

    model_components: dict[str, np.ndarray] = {}
    parent = pd.read_parquet(PARENT, columns=["date", "reach_id", "local_q0_m3_s", "local_q1_m3_s", "local_q2_m3_s"]); parent.date = pd.to_datetime(parent.date)
    parent = parent.loc[parent.date.dt.year.between(2010, 2018)].sort_values(["date", "reach_id"])
    model_components["HBV3_PARENT"] = parent[["local_q0_m3_s", "local_q1_m3_s", "local_q2_m3_s"]].to_numpy(float).reshape(len(dev_dates), 230, 3)
    spinup_rows = []
    for model in PARAMETER_NAMES:
        lock = json.loads((LOCKS / f"{model.lower()}_global_parameter_lock.json").read_text(encoding="utf-8")); parameters = np.asarray([lock["physical_parameters"][name] for name in PARAMETER_NAMES[model]])
        initial, audit = periodic_spinup(model, spin_p, spin_pet, parameters, tolerance=1e-8, max_cycles=50); sim = run_model(model, dev_p, dev_pet, parameters, initial, collect=True)
        model_components[model] = sim.response_mm * area[None, :, None] * 1000.0 / 86400.0
        spinup_rows.append({"model_id": model, **audit, "development_max_abs_mass_error_mm": sim.max_abs_mass_error_mm})
    pd.DataFrame(spinup_rows).to_parquet(OUT / "development_spinup_audit.parquet", index=False)

    summaries = []; station_frames = []; prediction_rows = []; component_rows = []
    month_periods = dev_dates.to_period("M"); unique_months = month_periods.unique()
    for model, local in model_components.items():
        site_component = np.stack([local[:, :, i] @ support.T for i in range(3)], axis=2); predicted = site_component.sum(axis=2)
        daily_station = station_metrics(model, observed, predicted, stations); daily_station["temporal_scale"] = "daily"
        periods, obs_month = monthly(dev_dates, observed); _, pred_month = monthly(dev_dates, predicted)
        monthly_station = station_metrics(model, obs_month, pred_month, stations); monthly_station["temporal_scale"] = "monthly"
        station_frames.extend([daily_station, monthly_station]); summaries.extend([summarize(model, "daily", observed, predicted, daily_station), summarize(model, "monthly", obs_month, pred_month, monthly_station)])
        routed = route_instantaneous(local * 86400.0, reach_ids, order, downstream) / 86400.0
        for month_index, period in enumerate(unique_months):
            selected = np.asarray(month_periods == period); local_m = local[selected].mean(axis=0); routed_m = routed[selected].mean(axis=0)
            frame = pd.DataFrame({"model_id": model, "year": period.year, "month": period.month, "reach_id": reach_ids, "local_fast_response_m3_s": local_m[:, 0], "local_intermediate_response_m3_s": local_m[:, 1], "local_slow_response_m3_s": local_m[:, 2], "routed_fast_response_m3_s": routed_m[:, 0], "routed_intermediate_response_m3_s": routed_m[:, 1], "routed_slow_response_m3_s": routed_m[:, 2]})
            component_rows.append(frame)
    summary = pd.concat(summaries, ignore_index=True); station_performance = pd.concat(station_frames, ignore_index=True); components = pd.concat(component_rows, ignore_index=True)
    components["routed_total_m3_s"] = components[["routed_fast_response_m3_s", "routed_intermediate_response_m3_s", "routed_slow_response_m3_s"]].sum(axis=1)
    for name in ["fast", "intermediate", "slow"]: components[f"routed_{name}_fraction"] = components[f"routed_{name}_response_m3_s"] / components.routed_total_m3_s.clip(lower=1e-12)
    summary.to_parquet(OUT / "development_performance_summary.parquet", index=False); station_performance.to_parquet(OUT / "development_station_performance.parquet", index=False); components.to_parquet(OUT / "monthly_response_component_ensemble_2010_2018.parquet", index=False)

    pivot_fraction = components.pivot(index=["year", "month", "reach_id"], columns="model_id", values="routed_slow_fraction")
    total_pivot = components.pivot(index=["year", "month", "reach_id"], columns="model_id", values="routed_total_m3_s")
    divergence = pd.DataFrame({"slow_fraction_min": pivot_fraction.min(axis=1), "slow_fraction_max": pivot_fraction.max(axis=1), "slow_fraction_range": pivot_fraction.max(axis=1) - pivot_fraction.min(axis=1), "slow_fraction_std": pivot_fraction.std(axis=1), "total_flow_cv": total_pivot.std(axis=1) / total_pivot.mean(axis=1).clip(lower=1e-12)}).reset_index()
    divergence.to_parquet(OUT / "component_structural_divergence.parquet", index=False)
    correlations = pivot_fraction.corr(method="spearman"); correlations.to_csv(OUT / "slow_fraction_cross_structure_spearman.csv")

    gate = {"daily_pooled": 0.70, "daily_station_median": 0.40, "monthly_pooled": 0.85, "monthly_station_median": 0.60, "monthly_abs_pbias": 25.0}
    eligible = []
    gate_rows = []
    for model in PARAMETER_NAMES:
        value = lambda scale, scope, field: float(summary.loc[(summary.candidate == model) & (summary.temporal_scale == scale) & (summary.scope == scope), field].iloc[0])
        checks = {"daily_pooled_NSE": value("daily", "pooled", "NSE") >= gate["daily_pooled"], "daily_station_median_NSE": value("daily", "station_median", "NSE") >= gate["daily_station_median"], "monthly_pooled_NSE": value("monthly", "pooled", "NSE") >= gate["monthly_pooled"], "monthly_station_median_NSE": value("monthly", "station_median", "NSE") >= gate["monthly_station_median"], "monthly_station_median_absolute_PBIAS": value("monthly", "station_median", "absolute_PBIAS_pct") <= gate["monthly_abs_pbias"]}
        passed = all(checks.values()); gate_rows.append({"model_id": model, **checks, "eligible_for_spatial_mapping": passed})
        if passed: eligible.append(model)
    pd.DataFrame(gate_rows).to_parquet(OUT / "spatial_mapping_eligibility.parquet", index=False)
    decision = {"stage": "20260826_5", "status": "PASS_DEVELOPMENT_COMPARISON_AND_EQUIFINALITY_AUDIT", "eligible_for_spatial_mapping": eligible, "ineligible_global_structures": [m for m in PARAMETER_NAMES if m not in eligible], "median_slow_fraction_range": float(divergence.slow_fraction_range.median()), "p95_slow_fraction_range": float(divergence.slow_fraction_range.quantile(0.95)), "retrospective_discharge_used": False, "TN_used": False, "authorized_successor": "20260826_6"}
    write_json(REPORT / "stage5_decision.json", decision)
    report = "# 20260826_5 开发期性能与结构等效性\n\n状态：`%s`。可进入昂贵空间参数映射的结构：`%s`。\n\n月尺度Reach响应集合的慢响应比例跨结构中位范围为`%.3f`，95分位为`%.3f`。这表示即使总流量可相近，内部快/中/慢分配仍可能明显分歧；它是结构不确定性证据，不是路径真值。\n\n%s\n" % (decision["status"], eligible, decision["median_slow_fraction_range"], decision["p95_slow_fraction_range"], summary.to_markdown(index=False))
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    write_json(REPORT / "integrity.json", {str(p): sha256(p) for p in [RUN / "experiment_contract.json", OUT / "development_performance_summary.parquet", OUT / "monthly_response_component_ensemble_2010_2018.parquet", OUT / "component_structural_divergence.parquet", REPORT / "stage5_decision.json", REPORT / "technical_report.md"]})
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
