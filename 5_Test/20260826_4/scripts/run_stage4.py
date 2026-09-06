from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


ROOT = Path(r"E:\SPARROW"); RUN = ROOT / "5_Test" / "20260826_4"; OUT = RUN / "outputs"; REPORT = RUN / "reports"
sys.path[:0] = [str(RUN / "scripts"), str(ROOT / "5_Test" / "20260825_3" / "scripts"), str(ROOT / "5_Test" / "20260825_5" / "scripts")]
from hydrology_core import load_topology  # noqa: E402
from regional_structures import BOUNDS, DEFAULT, PARAMETER_NAMES, periodic_spinup, physical_to_raw, raw_to_physical, run_model  # noqa: E402
from run_stage5 import build_support  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DEVELOPMENT_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
PML = ROOT / "5_Test" / "20260825_2" / "outputs" / "pml_v2_2a_monthly_aet_by_reach_2006_2022.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"


def sha256(path: Path) -> str:
    d = hashlib.sha256();
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): d.update(b)
    return d.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def monthly_sum(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    periods = dates.to_period("M"); return np.stack([values[np.asarray(periods == period)].sum(axis=0) for period in periods.unique()])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    reach_ids = np.arange(1, 231); order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGES); stations = gauges.loc[gauges.topology_representative].drop_duplicates("station_norm").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    support = build_support(stations, reach_ids, order, downstream)
    area = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)
    forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"]); forcing.date = pd.to_datetime(forcing.date)
    spin_dates = pd.date_range("2006-01-01", "2009-12-31"); dev_dates = pd.date_range("2010-01-01", "2018-12-31")
    def matrix(field: str, dates: pd.DatetimeIndex) -> np.ndarray:
        return forcing.pivot(index="date", columns="reach_id", values=field).reindex(index=dates, columns=reach_ids).to_numpy(float)
    spin_p, spin_pet = matrix("precipitation_daily_mm", spin_dates), matrix("pet_fao56_mm_day", spin_dates)
    dev_p, dev_pet = matrix("precipitation_daily_mm", dev_dates), matrix("pet_fao56_mm_day", dev_dates)
    obs_frame = pd.read_parquet(DEVELOPMENT_Q); obs_frame.date = pd.to_datetime(obs_frame.date)
    observed = obs_frame.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dev_dates, columns=stations.station_norm).to_numpy(float)
    obs_log = np.log1p(observed); q20 = np.nanquantile(observed, 0.2, axis=0); q90 = np.nanquantile(observed, 0.9, axis=0)
    tree_ids = stations.terminal_tree.to_numpy(int); unique_trees = np.unique(tree_ids)
    pml = pd.read_parquet(PML); pml = pml.loc[pml.year.between(2010, 2018)]
    pml_matrix = pml.pivot(index=["year", "month"], columns="reach_id", values="pml_aet_mm_month").reindex(columns=reach_ids).to_numpy(float)

    trace_rows: list[dict[str, object]] = []; best_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(PARAMETER_NAMES):
        cache: dict[tuple[float, ...], tuple[float, dict[str, float], dict[str, object]]] = {}
        def evaluate(raw: np.ndarray) -> float:
            key = tuple(np.round(raw, 10));
            if key in cache: return cache[key][0]
            physical = raw_to_physical(model, raw)
            initial, spin = periodic_spinup(model, spin_p, spin_pet, physical, tolerance=1e-8, max_cycles=30)
            sim = run_model(model, dev_p, dev_pet, physical, initial, collect=True)
            local_q = sim.response_mm.sum(axis=2) * area[None, :] * 1000.0 / 86400.0
            predicted = local_q @ support.T
            valid = np.isfinite(observed); log_residual = np.log1p(np.maximum(predicted, 0)) - obs_log
            station_all = np.asarray([np.sqrt(np.nanmean(np.where(valid[:, j], log_residual[:, j] ** 2, np.nan))) for j in range(len(stations))])
            station_bias = np.asarray([abs(np.nansum(predicted[valid[:, j], j]) / np.nansum(observed[valid[:, j], j]) - 1) for j in range(len(stations))])
            station_high = np.asarray([np.sqrt(np.nanmean(log_residual[(valid[:, j]) & (observed[:, j] >= q90[j]), j] ** 2)) for j in range(len(stations))])
            station_low = np.asarray([np.sqrt(np.nanmean(log_residual[(valid[:, j]) & (observed[:, j] <= q20[j]), j] ** 2)) for j in range(len(stations))])
            def tree_mean(values: np.ndarray) -> float: return float(np.mean([np.nanmean(values[tree_ids == tree]) for tree in unique_trees]))
            aet_month = monthly_sum(sim.aet_mm, dev_dates); aet_rmse = float(np.sqrt(np.mean((aet_month - pml_matrix) ** 2)))
            terms = {"all_log_rmse": tree_mean(station_all), "absolute_volume_bias": tree_mean(station_bias), "high_log_rmse": tree_mean(station_high), "low_log_rmse": tree_mean(station_low), "pml_aet_rmse_mm_month": aet_rmse}
            objective = float(np.mean([terms["all_log_rmse"] / 0.35, terms["absolute_volume_bias"] / 0.15, terms["high_log_rmse"] / 0.50, terms["low_log_rmse"] / 0.50, terms["pml_aet_rmse_mm_month"] / 30.0]))
            if not spin["converged"] or sim.max_abs_mass_error_mm > 1e-10: objective += 1000.0
            cache[key] = (objective, terms, {"spinup": spin, "mass_error": sim.max_abs_mass_error_mm})
            trace_rows.append({"model_id": model, "evaluation": len(cache), "objective": objective, **terms})
            return objective
        default_raw = physical_to_raw(model, DEFAULT[model]); default_objective = evaluate(default_raw)
        start = time.time()
        result = differential_evolution(evaluate, [(-1.0, 1.0)] * len(default_raw), seed=260826 + model_index, popsize=4, maxiter=8, polish=False, updating="immediate", workers=1, x0=default_raw, tol=0.01)
        elapsed = time.time() - start
        best_raw = np.asarray(result.x); best_physical = raw_to_physical(model, best_raw); objective, terms, audit = cache[tuple(np.round(best_raw, 10))]
        best_rows.append({"model_id": model, "objective": objective, "default_objective": default_objective, "optimizer_success": bool(result.success), "optimizer_message": str(result.message), "evaluations": int(result.nfev), "elapsed_seconds": elapsed, "parameter_boundary": bool(np.any(np.abs(best_raw) >= 0.99)), "max_abs_mass_error_mm": audit["mass_error"], **terms})
        write_json(REPORT / f"{model.lower()}_global_parameter_lock.json", {"model_id": model, "fit_period": "2010-2018", "retrospective_discharge_used": False, "raw_parameters": dict(zip(PARAMETER_NAMES[model], map(float, best_raw))), "physical_parameters": dict(zip(PARAMETER_NAMES[model], map(float, best_physical))), "objective": objective, "objective_terms": terms, "spinup": audit["spinup"], "optimizer": {"name": "scipy differential_evolution", "success": bool(result.success), "message": str(result.message), "nfev": int(result.nfev)}, "parameter_boundary_confounded": bool(np.any(np.abs(best_raw) >= 0.99))})
    trace = pd.DataFrame(trace_rows); summary = pd.DataFrame(best_rows)
    trace.to_parquet(OUT / "global_fit_trace.parquet", index=False); summary.to_parquet(OUT / "global_fit_summary.parquet", index=False)
    viable = summary.loc[(summary.max_abs_mass_error_mm <= 1e-10) & np.isfinite(summary.objective), "model_id"].tolist()
    decision = {"stage": "20260826_4", "status": "PASS_GLOBAL_COARSE_FIT_COMPLETED" if len(viable) == 4 else "STAGE4_GLOBAL_FIT_FAILED", "viable_models": viable, "candidate_count": 4, "retrospective_discharge_used": False, "TN_used": False, "authorized_successor": "20260826_5" if len(viable) == 4 else None}
    write_json(REPORT / "stage4_decision.json", decision)
    report = "# 20260826_4 全局参数共同粗筛\n\n状态：`%s`。\n\n四个结构均使用相同2010–2018流量、树平衡误差尺度、PML月AET软约束及周期spin-up。此处只建立可比较的全局起点，不做站点专属校正，也未读取2019–2022流量。\n\n%s\n" % (decision["status"], summary.to_markdown(index=False))
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    write_json(REPORT / "integrity.json", {str(p): sha256(p) for p in [RUN / "experiment_contract.json", RUN / "scripts" / "regional_structures.py", OUT / "global_fit_summary.parquet", REPORT / "stage4_decision.json", REPORT / "technical_report.md"]})
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
