from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_33"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
BRIDGE = TEST / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
SUPPORT = TEST / "20260823_28" / "outputs" / "gauge_support_hydrology.parquet"
OBS = TEST / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
LOCK = TEST / "20260823_15" / "final_reports" / "four_group_training_lock.json"
METRICS30 = TEST / "20260823_30" / "outputs" / "total_flow_metrics.parquet"
COMPONENT = TEST / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_matrix(reaches: np.ndarray) -> np.ndarray:
    spec = importlib.util.spec_from_file_location("stage33_topology", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.TOPOLOGY_PATH = TOPOLOGY
    return module._upstream_matrix(reaches)


def metric_dict(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs >= 0) & (pred >= 0)
    obs, pred = obs[valid], pred[valid]
    residual = pred - obs
    log_residual = np.log1p(pred) - np.log1p(obs)
    denominator = np.sum((obs - obs.mean()) ** 2)
    log_denominator = np.sum((np.log1p(obs) - np.log1p(obs).mean()) ** 2)
    return {
        "n": int(len(obs)),
        "NSE": float(1 - np.sum(residual**2) / denominator) if denominator > 0 else math.nan,
        "NSE_log": float(1 - np.sum(log_residual**2) / log_denominator) if log_denominator > 0 else math.nan,
        "PBIAS_pct": float(100 * np.sum(residual) / np.sum(obs)),
        "RMSE_log": float(np.sqrt(np.mean(log_residual**2))),
        "MAE_log": float(np.mean(np.abs(log_residual))),
    }


def process_step(effective, demand, source, fast_store, delayed_store, parameters):
    total_start = source + fast_store + delayed_store
    saturation = np.clip(source / parameters["prod_capacity"], 0.0, 1.5)
    generated = np.minimum(effective, effective * np.power(saturation, parameters["runoff_gamma"]))
    source = np.clip(source + np.clip(effective - generated, 0.0, None), 0.0, None)
    withdrawn = np.minimum(demand, source)
    source = np.clip(source - withdrawn, 0.0, None)
    overflow = np.clip(source - parameters["prod_capacity"], 0.0, None)
    source = np.minimum(source, parameters["prod_capacity"])
    recharge = parameters["base_release"] * source
    source = np.clip(source - recharge, 0.0, None)
    fast_available = np.clip(fast_store + generated + overflow, 0.0, None)
    fast_release = (1.0 - parameters["quick_rho"]) * fast_available
    fast_store = parameters["quick_rho"] * fast_available
    delayed_available = np.clip(delayed_store + recharge, 0.0, None)
    delayed_release = (1.0 - parameters["base_rho"]) * delayed_available
    delayed_store = parameters["base_rho"] * delayed_available
    total_end = source + fast_store + delayed_store
    mass_error = total_start + effective - withdrawn - total_end - fast_release - delayed_release
    return {
        "source": source,
        "fast_store": fast_store,
        "delayed_store": delayed_store,
        "fast_release": fast_release,
        "delayed_release": delayed_release,
        "mass_error": mass_error,
    }


def allocate_correction(delta_volume, fast_release_volume, delayed_release_volume, fast_store_volume, delayed_store_volume):
    adjusted_fast_release = fast_release_volume.copy()
    adjusted_delayed_release = delayed_release_volume.copy()
    adjusted_fast_store = fast_store_volume.copy()
    adjusted_delayed_store = delayed_store_volume.copy()
    positive = delta_volume > 0
    if positive.any():
        extra = delta_volume[positive]
        fs = fast_store_volume[positive]
        ds = delayed_store_volume[positive]
        share = np.divide(fs, fs + ds, out=np.full_like(fs, 0.5), where=(fs + ds) > EPS)
        take_fast = np.minimum(extra * share, fs)
        take_delayed = np.minimum(extra - take_fast, ds)
        remaining = extra - take_fast - take_delayed
        add_fast = np.minimum(remaining, fs - take_fast)
        take_fast += add_fast
        remaining -= add_fast
        take_delayed += np.minimum(remaining, ds - take_delayed)
        adjusted_fast_release[positive] += take_fast
        adjusted_delayed_release[positive] += take_delayed
        adjusted_fast_store[positive] -= take_fast
        adjusted_delayed_store[positive] -= take_delayed
    negative = delta_volume < 0
    if negative.any():
        retained = -delta_volume[negative]
        fr = fast_release_volume[negative]
        dr = delayed_release_volume[negative]
        share = np.divide(fr, fr + dr, out=np.full_like(fr, 0.5), where=(fr + dr) > EPS)
        reduce_fast = np.minimum(retained * share, fr)
        reduce_delayed = np.minimum(retained - reduce_fast, dr)
        remaining = retained - reduce_fast - reduce_delayed
        add_fast = np.minimum(remaining, fr - reduce_fast)
        reduce_fast += add_fast
        remaining -= add_fast
        reduce_delayed += np.minimum(remaining, dr - reduce_delayed)
        adjusted_fast_release[negative] -= reduce_fast
        adjusted_delayed_release[negative] -= reduce_delayed
        adjusted_fast_store[negative] += reduce_fast
        adjusted_delayed_store[negative] += reduce_delayed
    return adjusted_fast_release, adjusted_delayed_release, adjusted_fast_store, adjusted_delayed_store


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    stage30 = pd.read_parquet(METRICS30)
    parent = stage30[stage30.candidate.eq("H0_LOCKED_PARENT") & stage30.period.eq("check_2019_2022")]
    latent = float(parent[parent.layer.eq("LATENT_SUPPORT")].RMSE_log.iloc[0])
    gauge = float(parent[parent.layer.eq("GAUGE_OBSERVATION")].RMSE_log.iloc[0])
    material_gap = latent - gauge >= 0.10
    if not material_gap:
        decision = {"stage": "20260823_33", "status": "NOT_TRIGGERED", "latent_minus_gauge_RMSE_log": latent - gauge}
        (REPORT / "stage33_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        print(json.dumps(decision, indent=2))
        return
    bridge = pd.read_parquet(BRIDGE).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    reaches = bridge.reach_id.drop_duplicates().to_numpy(int)
    nr = len(reaches)
    nt = len(bridge) // nr
    reach_index = {int(value): i for i, value in enumerate(reaches)}
    def rt(column):
        return bridge[column].to_numpy(float).reshape(nr, nt).T
    time = bridge.groupby(["year", "month"], sort=False).head(1)[["year", "month", "month_seconds"]].reset_index(drop=True)
    seconds = time.month_seconds.to_numpy(float)
    area = bridge.groupby("reach_id", sort=False).head(1).catchment_area_km2.to_numpy(float)
    effective = rt("positive_input_mm")
    demand = rt("aet_storage_withdrawn_mm") + rt("aet_unmet_mm")
    source = rt("source_store_start_mm")[0].copy()
    fast_store = rt("quick_store_start_mm")[0].copy()
    delayed_store = rt("slow_store_start_mm")[0].copy()
    parameters = json.loads(LOCK.read_text(encoding="utf-8"))["physical_parameters"]
    upstream = upstream_matrix(reaches)
    support = pd.read_parquet(SUPPORT)[["station_norm", "reach_id", "downstream_fraction_on_reach"]].drop_duplicates("station_norm")
    support["station_norm"] = support.station_norm.astype(str)
    stations = support.station_norm.tolist()
    station_index = {value: i for i, value in enumerate(stations)}
    station_rows = []
    for row in support.itertuples():
        r = reach_index[int(row.reach_id)]
        vector = upstream[r].copy()
        vector[r] -= 1.0 - float(row.downstream_fraction_on_reach)
        station_rows.append(vector)
    support_matrix = np.vstack(station_rows)
    obs = pd.read_parquet(OBS).copy()
    obs["station_norm"] = obs.station_norm.astype(str)
    obs = obs[obs.station_norm.isin(station_index)].copy()
    obs_lookup = {(int(y), int(m)): part for (y, m), part in obs.groupby(["year", "month"], sort=False)}
    reach_records, station_records, solve_records = [], [], []
    ridge = float(contract["network_ridge_precision"])
    for t, timerow in time.iterrows():
        step = process_step(effective[t], demand[t], source, fast_store, delayed_store, parameters)
        fast_release_volume = step["fast_release"] * area * 1000.0
        delayed_release_volume = step["delayed_release"] * area * 1000.0
        fast_store_volume = step["fast_store"] * area * 1000.0
        delayed_store_volume = step["delayed_store"] * area * 1000.0
        local_total = fast_release_volume + delayed_release_volume
        delta = np.zeros(nr, float)
        solver_status = "NO_UPDATE"
        rows_used = 0
        if int(timerow.year) <= 2018 and (int(timerow.year), int(timerow.month)) in obs_lookup:
            month_obs = obs_lookup[(int(timerow.year), int(timerow.month))]
            sidx = np.array([station_index[str(value)] for value in month_obs.station_norm], int)
            a = support_matrix[sidx]
            predicted = a @ local_total
            observed = month_obs.q_m3s.to_numpy(float) * float(timerow.month_seconds)
            relative_target = (observed - predicted) / np.maximum(observed, 1.0)
            scale = np.maximum(local_total + fast_store_volume + delayed_store_volume, area * 1000.0 * 0.01)
            design = a * scale[None, :] / np.maximum(observed[:, None], 1.0)
            augmented_x = np.vstack([design, np.sqrt(ridge) * np.eye(nr)])
            augmented_y = np.concatenate([relative_target, np.zeros(nr)])
            lower = -local_total / scale
            upper = (fast_store_volume + delayed_store_volume) / scale
            fit = lsq_linear(augmented_x, augmented_y, bounds=(lower, upper), method="trf", tol=1e-8, max_iter=200)
            delta = np.clip(fit.x * scale * float(contract["observation_gain"]), -local_total, fast_store_volume + delayed_store_volume)
            solver_status = f"{fit.status}:{fit.message}"
            rows_used = len(month_obs)
        adjusted = allocate_correction(delta, fast_release_volume, delayed_release_volume, fast_store_volume, delayed_store_volume)
        fast_release_volume, delayed_release_volume, fast_store_volume, delayed_store_volume = adjusted
        source = step["source"]
        fast_store = fast_store_volume / (area * 1000.0)
        delayed_store = delayed_store_volume / (area * 1000.0)
        routed_fast = upstream @ fast_release_volume
        routed_delayed = upstream @ delayed_release_volume
        local_closure = (
            step["fast_release"] * area * 1000.0 + step["fast_store"] * area * 1000.0
            - fast_release_volume - fast_store_volume
            + step["delayed_release"] * area * 1000.0 + step["delayed_store"] * area * 1000.0
            - delayed_release_volume - delayed_store_volume
        )
        for i, reach in enumerate(reaches):
            reach_records.append({
                "reach_id": int(reach), "year": int(timerow.year), "month": int(timerow.month),
                "source_store_end_mm": source[i], "fast_store_end_mm": fast_store[i], "delayed_store_end_mm": delayed_store[i],
                "local_fast_m3_s": fast_release_volume[i] / seconds[t], "local_delayed_m3_s": delayed_release_volume[i] / seconds[t],
                "routed_fast_m3_s": routed_fast[i] / seconds[t], "routed_delayed_m3_s": routed_delayed[i] / seconds[t],
                "routed_total_m3_s": (routed_fast[i] + routed_delayed[i]) / seconds[t],
                "analysis_release_correction_m3": delta[i], "analysis_active": int(timerow.year) <= 2018,
                "explicit_store_release_closure_m3": local_closure[i],
            })
        support_fast = support_matrix @ fast_release_volume / seconds[t]
        support_delayed = support_matrix @ delayed_release_volume / seconds[t]
        month_all = obs_lookup.get((int(timerow.year), int(timerow.month)))
        if month_all is not None:
            for row in month_all.itertuples():
                s = station_index[str(row.station_norm)]
                station_records.append({
                    "station_norm": str(row.station_norm), "reach_id": int(row.reach_id), "year": int(row.year), "month": int(row.month),
                    "q_m3s": float(row.q_m3s), "state_analysis_fast_m3_s": support_fast[s],
                    "state_analysis_delayed_m3_s": support_delayed[s], "state_analysis_total_m3_s": support_fast[s] + support_delayed[s],
                    "analysis_active": int(row.year) <= 2018,
                })
        solve_records.append({
            "year": int(timerow.year), "month": int(timerow.month), "observation_rows": rows_used,
            "solver_status": solver_status, "sum_abs_release_correction_m3": float(np.sum(np.abs(delta))),
            "max_abs_release_correction_m3": float(np.max(np.abs(delta))),
        })
        if int(timerow.month) == 12:
            print(f"state analysis year {int(timerow.year)} complete", flush=True)
    reach_frame = pd.DataFrame(reach_records)
    station_frame = pd.DataFrame(station_records)
    solver_frame = pd.DataFrame(solve_records)
    reach_frame.to_parquet(OUT / "diagnostic_state_analysis_reach_hydrology.parquet", index=False)
    station_frame.to_parquet(OUT / "diagnostic_state_analysis_station_predictions.parquet", index=False)
    solver_frame.to_parquet(OUT / "diagnostic_state_analysis_solver_audit.parquet", index=False)
    rows, station_parts = [], []
    for period, selector in [("development_2006_2018", station_frame.year.le(2018)), ("frozen_check_2019_2022", station_frame.year.ge(2019))]:
        part = station_frame[selector]
        pooled = metric_dict(part.q_m3s, part.state_analysis_total_m3_s)
        station_metrics = []
        for station, group in part.groupby("station_norm", sort=False):
            station_metrics.append({"period": period, "station_norm": station, **metric_dict(group.q_m3s, group.state_analysis_total_m3_s)})
        station_metric = pd.DataFrame(station_metrics)
        pooled.update({
            "period": period, "station_count": len(station_metric),
            "station_mean_NSE": float(station_metric.NSE.mean()), "station_median_NSE": float(station_metric.NSE.median()),
            "station_mean_RMSE_log": float(station_metric.RMSE_log.mean()), "station_median_RMSE_log": float(station_metric.RMSE_log.median()),
        })
        rows.append(pooled)
        station_parts.append(station_metric)
    summary = pd.DataFrame(rows)
    summary.to_parquet(OUT / "diagnostic_state_analysis_metrics.parquet", index=False)
    pd.concat(station_parts, ignore_index=True).to_parquet(OUT / "diagnostic_state_analysis_station_metrics.parquet", index=False)
    closure = float(reach_frame.explicit_store_release_closure_m3.abs().max())
    minimum = float(reach_frame[["fast_store_end_mm", "delayed_store_end_mm", "local_fast_m3_s", "local_delayed_m3_s"]].min().min())
    decision = {
        "stage": "20260823_33", "status": "DIAGNOSTIC_STATE_ANALYSIS_COMPLETE",
        "trigger_gap_RMSE_log": latent - gauge, "material_gap_triggered": material_gap,
        "explicit_store_release_closure_max_abs_m3": closure, "minimum_state_or_flux": minimum,
        "post_2018_observation_updates": int(solver_frame[solver_frame.year.ge(2019)].observation_rows.sum()),
        "tn_primary_authorized": False, "role": "KNOWN_GAUGE_STATE_ANALYSIS_UPPER_BOUND_ONLY",
    }
    (REPORT / "stage33_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_33 守恒状态分析诊断\n\n"
        f"状态：`{decision['status']}`。该产品只量化已设站历史条件化上限，不能进入 TN。\n\n"
        "## 结果\n\n```text\n" + summary.to_string(index=False) + "\n```\n\n"
        f"显式储量—释放闭合最大误差：{closure:.3e} m³；2019–2022 更新次数：0。\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "bridge_sha256": sha256(BRIDGE),
        "reach_output_sha256": sha256(OUT / "diagnostic_state_analysis_reach_hydrology.parquet"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
