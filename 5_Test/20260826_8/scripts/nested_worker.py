from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize


ROOT = Path(r"E:\SPARROW"); RUN = ROOT / "5_Test" / "20260826_8"; OUT = RUN / "outputs"
sys.path[:0] = [str(ROOT / "5_Test" / "20260826_4" / "scripts"), str(ROOT / "5_Test" / "20260826_6" / "scripts"), str(ROOT / "5_Test" / "20260825_3" / "scripts"), str(ROOT / "5_Test" / "20260825_5" / "scripts")]
from hydrology_core import load_topology  # noqa: E402
from regional_structures import DEFAULT, PARAMETER_NAMES, periodic_spinup, physical_to_raw, raw_to_physical, run_model  # noqa: E402
from run_stage5 import build_support  # noqa: E402
from run_stage6 import FEATURE_NAMES, PRIOR_SIGMA, PRIOR_WEIGHT, build_attributes, parameter_map  # noqa: E402

FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DEVELOPMENT_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
PML = ROOT / "5_Test" / "20260825_2" / "outputs" / "pml_v2_2a_monthly_aet_by_reach_2006_2022.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"


def monthly_sum(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    periods = dates.to_period("M"); return np.stack([values[np.asarray(periods == p)].sum(axis=0) for p in periods.unique()])


def fit_fold(task: tuple[str, int]) -> dict[str, object]:
    model, held_tree = task; reach_ids = np.arange(1, 231); order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGES); stations = gauges.loc[gauges.topology_representative].drop_duplicates("station_norm").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True); support = build_support(stations, reach_ids, order, downstream)
    area = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float); x, _, _ = build_attributes(reach_ids, area)
    forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"]); forcing.date = pd.to_datetime(forcing.date); spin_dates = pd.date_range("2006-01-01", "2009-12-31"); dev_dates = pd.date_range("2010-01-01", "2018-12-31")
    pivot = lambda field, dates: forcing.pivot(index="date", columns="reach_id", values=field).reindex(index=dates, columns=reach_ids).to_numpy(float)
    spin_p, spin_pet = pivot("precipitation_daily_mm", spin_dates), pivot("pet_fao56_mm_day", spin_dates); dev_p, dev_pet = pivot("precipitation_daily_mm", dev_dates), pivot("pet_fao56_mm_day", dev_dates)
    obs = pd.read_parquet(DEVELOPMENT_Q); obs.date = pd.to_datetime(obs.date); observed = obs.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dev_dates, columns=stations.station_norm).to_numpy(float); obs_log = np.log1p(observed)
    train_station = stations.terminal_tree.to_numpy(int) != held_tree; target_station = ~train_station; train_support = support[train_station]; train_obs = observed[:, train_station]; train_log = obs_log[:, train_station]; train_tree = stations.loc[train_station, "terminal_tree"].to_numpy(int); unique_train_tree = np.unique(train_tree); q20, q90 = np.nanquantile(train_obs, .2, axis=0), np.nanquantile(train_obs, .9, axis=0)
    pml = pd.read_parquet(PML); pml = pml.loc[pml.year.between(2010, 2018)]; pml_matrix = pml.pivot(index=["year", "month"], columns="reach_id", values="pml_aet_mm_month").reindex(columns=reach_ids).to_numpy(float)

    def data_objective(raw_base: np.ndarray, coefficients: np.ndarray | None) -> tuple[float, dict[str, object]]:
        if coefficients is None: raw_map = np.broadcast_to(np.clip(raw_base, -1, 1)[None, :], (230, len(raw_base)))
        else: raw_map, _, _ = parameter_map(model, raw_base, coefficients, x)
        physical = raw_to_physical(model, raw_map); initial, spin = periodic_spinup(model, spin_p, spin_pet, physical, tolerance=1e-8, max_cycles=35); sim = run_model(model, dev_p, dev_pet, physical, initial, collect=True)
        predicted = (sim.response_mm.sum(axis=2) * area[None, :] * 1000 / 86400) @ train_support.T; valid = np.isfinite(train_obs); residual = np.log1p(predicted) - train_log
        all_e = np.asarray([np.sqrt(np.nanmean(np.where(valid[:, j], residual[:, j] ** 2, np.nan))) for j in range(train_obs.shape[1])]); bias = np.asarray([abs(np.nansum(predicted[valid[:, j], j]) / np.nansum(train_obs[valid[:, j], j]) - 1) for j in range(train_obs.shape[1])]); high = np.asarray([np.sqrt(np.nanmean(residual[valid[:, j] & (train_obs[:, j] >= q90[j]), j] ** 2)) for j in range(train_obs.shape[1])]); low = np.asarray([np.sqrt(np.nanmean(residual[valid[:, j] & (train_obs[:, j] <= q20[j]), j] ** 2)) for j in range(train_obs.shape[1])]); tree_mean = lambda v: float(np.mean([np.nanmean(v[train_tree == tree]) for tree in unique_train_tree])); aet_rmse = float(np.sqrt(np.mean((monthly_sum(sim.aet_mm, dev_dates) - pml_matrix) ** 2)))
        terms = {"all_log_rmse": tree_mean(all_e), "absolute_volume_bias": tree_mean(bias), "high_log_rmse": tree_mean(high), "low_log_rmse": tree_mean(low), "pml_aet_rmse_mm_month": aet_rmse}; value = float(np.mean([terms["all_log_rmse"]/.35, terms["absolute_volume_bias"]/.15, terms["high_log_rmse"]/.5, terms["low_log_rmse"]/.5, terms["pml_aet_rmse_mm_month"]/30]))
        return value, {"spinup": spin, "mass_error": sim.max_abs_mass_error_mm, "terms": terms, "physical": physical, "raw_map": raw_map}

    default_raw = physical_to_raw(model, DEFAULT[model]); global_cache = {}
    def global_eval(raw):
        key = tuple(np.round(raw, 8))
        if key not in global_cache: global_cache[key] = data_objective(raw, None)
        return global_cache[key][0]
    seed = 260826 + int(held_tree) * 17 + (0 if model == "MTRS3" else 10000)
    global_result = differential_evolution(global_eval, [(-1, 1)] * len(default_raw), seed=seed, popsize=3, maxiter=3, polish=False, x0=default_raw, workers=1, updating="immediate")
    base_raw = np.asarray(global_result.x); coefficient_cache = {}
    def map_eval(coef):
        key = tuple(np.round(coef, 8))
        if key not in coefficient_cache:
            data, audit = data_objective(base_raw, coef); coefficient_cache[key] = (data + PRIOR_WEIGHT * float(np.mean((coef / PRIOR_SIGMA) ** 2)), data, audit)
        return coefficient_cache[key][0]
    zero = np.zeros(2 * x.shape[1]); map_result = minimize(map_eval, zero, method="Powell", bounds=[(-.75, .75)] * len(zero), options={"maxfev": 80, "maxiter": 8, "xtol": .025, "ftol": .006}); coef = np.asarray(map_result.x); _, data_value, audit = coefficient_cache[tuple(np.round(coef, 8))]
    raw_map, storage_index, path_index = parameter_map(model, base_raw, coef, x); physical = raw_to_physical(model, raw_map); initial, spin = periodic_spinup(model, spin_p, spin_pet, physical, tolerance=1e-8, max_cycles=40); sim = run_model(model, dev_p, dev_pet, physical, initial, collect=True); local = sim.response_mm * area[None, :, None] * 1000 / 86400
    target_support = support[target_station]; site_components = np.stack([local[:, :, c] @ target_support.T for c in range(3)], axis=2); target = stations.loc[target_station].reset_index(drop=True); target_obs = observed[:, target_station]
    frame = pd.DataFrame({"date": np.repeat(dev_dates.to_numpy(), len(target)), "station_norm": np.tile(target.station_norm.to_numpy(), len(dev_dates)), "reach_id": np.tile(target.reach_id.to_numpy(int), len(dev_dates)), "heldout_terminal_tree": int(held_tree), "q_observed_m3_s": target_obs.reshape(-1), "fast_response_m3_s": site_components[:, :, 0].reshape(-1), "intermediate_response_m3_s": site_components[:, :, 1].reshape(-1), "slow_response_m3_s": site_components[:, :, 2].reshape(-1)})
    frame["q_candidate_m3_s"] = frame[["fast_response_m3_s", "intermediate_response_m3_s", "slow_response_m3_s"]].sum(axis=1); path = OUT / "fold_predictions" / f"{model.lower()}_tree_{held_tree}.parquet"; path.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(path, index=False)
    lock = {"model_id": model, "heldout_terminal_tree": int(held_tree), "target_tree_discharge_used_in_fit": False, "target_tree_q_signatures_used_in_fit": False, "global_raw_parameters": dict(zip(PARAMETER_NAMES[model], map(float, base_raw))), "mapping_coefficients": {"storage_index": dict(zip(FEATURE_NAMES, map(float, coef[:len(FEATURE_NAMES)]))), "path_index": dict(zip(FEATURE_NAMES, map(float, coef[len(FEATURE_NAMES):])))}, "global_optimizer_nfev": int(global_result.nfev), "mapping_optimizer_nfev": int(map_result.nfev), "mapping_optimizer_success": bool(map_result.success), "data_objective": data_value, "spinup": spin, "mass_error": sim.max_abs_mass_error_mm, "parameter_boundary_fraction": float(np.mean(np.abs(raw_map) >= .999)), "prediction_path": str(path)}; lock_path = OUT / "fold_locks" / f"{model.lower()}_tree_{held_tree}.json"; lock_path.parent.mkdir(parents=True, exist_ok=True); lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model_id": model, "heldout_terminal_tree": int(held_tree), "stations": int(target_station.sum()), "global_nfev": int(global_result.nfev), "mapping_nfev": int(map_result.nfev), "mapping_success": bool(map_result.success), "data_objective": data_value, "mass_error": sim.max_abs_mass_error_mm, "parameter_boundary_fraction": lock["parameter_boundary_fraction"], "prediction_path": str(path)}

