from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW"); RUN = ROOT / "5_Test" / "20260826_6"; OUT = RUN / "outputs"; REPORT = RUN / "reports"
sys.path[:0] = [str(ROOT / "5_Test" / "20260826_4" / "scripts"), str(ROOT / "5_Test" / "20260825_3" / "scripts"), str(ROOT / "5_Test" / "20260825_5" / "scripts")]
from hydrology_core import load_topology  # noqa: E402
from regional_structures import PARAMETER_NAMES, periodic_spinup, raw_to_physical, run_model  # noqa: E402
from run_stage5 import build_support  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DEVELOPMENT_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
PML = ROOT / "5_Test" / "20260825_2" / "outputs" / "pml_v2_2a_monthly_aet_by_reach_2006_2022.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
MPR = ROOT / "5_Test" / "20260825_5" / "outputs" / "mpr_attributes_by_reach.parquet"
DEM_STATIC = ROOT / "5_Test" / "20260826_2" / "outputs" / "reach_static_attributes_dem_recomputed.parquet"
LOCKS = ROOT / "5_Test" / "20260826_4" / "reports"

FEATURE_NAMES = ["log_awc", "bulk_density", "log10_permeability", "porosity", "log_slope", "aridity", "log_area"]
PRIOR_SIGMA = 0.25
PRIOR_WEIGHT = 0.10


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): d.update(b)
    return d.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def monthly_sum(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    periods = dates.to_period("M"); return np.stack([values[np.asarray(periods == period)].sum(axis=0) for period in periods.unique()])


def build_attributes(reach_ids: np.ndarray, area: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    old = pd.read_parquet(MPR).set_index("reach_id").reindex(reach_ids); dem = pd.read_parquet(DEM_STATIC).set_index("reach_id").reindex(reach_ids)
    raw = pd.DataFrame({"reach_id": reach_ids, "log_awc": np.log(old.awc_0_200_mm.to_numpy(float)), "bulk_density": old.bulk_density_0_30_g_cm3.to_numpy(float), "log10_permeability": old.glhymps_log10_permeability_m2.to_numpy(float), "porosity": old.glhymps_porosity.to_numpy(float), "log_slope": np.log(dem.slope_for_spatial_mapping.to_numpy(float)), "aridity": old.pet_mm.to_numpy(float) / old.precipitation_mm.to_numpy(float), "log_area": np.log(area)})
    center = raw[FEATURE_NAMES].mean(); scale = raw[FEATURE_NAMES].std(ddof=0).replace(0, 1); standardized = (raw[FEATURE_NAMES] - center) / scale
    registry = {"feature_names": FEATURE_NAMES, "center": center.to_dict(), "scale": scale.to_dict(), "DEM_slope_censored_count": int(dem.slope_below_dem_resolution_censored.sum()), "station_or_tree_identity_used": False}
    return standardized.to_numpy(float), raw, registry


def parameter_map(model: str, base_raw: np.ndarray, coefficients: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nfeature = x.shape[1]; storage_index = np.clip(x @ coefficients[:nfeature], -2, 2); path_index = np.clip(x @ coefficients[nfeature:], -2, 2)
    if model == "MTRS3":
        storage_loading = np.asarray([0.45, 0.15, 0, 0, 0, 0, 0]); path_loading = np.asarray([0, 0.10, -0.45, -0.20, 0.10, 0.30, 0.50])
    else:
        storage_loading = np.asarray([0.35, 0.30, 0.35, 0.30, 0.40, 0, 0, 0, 0, 0, 0, 0]); path_loading = np.asarray([0, 0, 0, 0, 0, 0.20, 0.10, 0.10, 0.20, 0.30, 0.50, -0.30])
    raw_map = base_raw[None, :] + storage_index[:, None] * storage_loading[None, :] + path_index[:, None] * path_loading[None, :]
    return np.clip(raw_map, -1, 1), storage_index, path_index


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    reach_ids = np.arange(1, 231); order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGES); stations = gauges.loc[gauges.topology_representative].drop_duplicates("station_norm").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True); support = build_support(stations, reach_ids, order, downstream)
    area = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)
    x, raw_attributes, feature_registry = build_attributes(reach_ids, area); raw_attributes.to_parquet(OUT / "spatial_attributes_raw.parquet", index=False); write_json(REPORT / "spatial_feature_registry.json", feature_registry)
    forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"]); forcing.date = pd.to_datetime(forcing.date)
    spin_dates = pd.date_range("2006-01-01", "2009-12-31"); dev_dates = pd.date_range("2010-01-01", "2018-12-31")
    pivot = lambda field, dates: forcing.pivot(index="date", columns="reach_id", values=field).reindex(index=dates, columns=reach_ids).to_numpy(float)
    spin_p, spin_pet = pivot("precipitation_daily_mm", spin_dates), pivot("pet_fao56_mm_day", spin_dates); dev_p, dev_pet = pivot("precipitation_daily_mm", dev_dates), pivot("pet_fao56_mm_day", dev_dates)
    obs = pd.read_parquet(DEVELOPMENT_Q); obs.date = pd.to_datetime(obs.date); observed = obs.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dev_dates, columns=stations.station_norm).to_numpy(float); obs_log = np.log1p(observed)
    q20, q90 = np.nanquantile(observed, 0.2, axis=0), np.nanquantile(observed, 0.9, axis=0); tree_ids = stations.terminal_tree.to_numpy(int); unique_trees = np.unique(tree_ids)
    pml = pd.read_parquet(PML); pml = pml.loc[pml.year.between(2010, 2018)]; pml_matrix = pml.pivot(index=["year", "month"], columns="reach_id", values="pml_aet_mm_month").reindex(columns=reach_ids).to_numpy(float)
    trace_rows = []; summary_rows = []; parameter_rows = []
    for model in ["RAVEN_SACSMA3", "MTRS3"]:
        lock = json.loads((LOCKS / f"{model.lower()}_global_parameter_lock.json").read_text(encoding="utf-8")); base_raw = np.asarray([lock["raw_parameters"][name] for name in PARAMETER_NAMES[model]])
        cache = {}
        def evaluate(coef: np.ndarray) -> float:
            key = tuple(np.round(coef, 9))
            if key in cache: return cache[key][0]
            raw_map, storage_index, path_index = parameter_map(model, base_raw, coef, x); physical_map = raw_to_physical(model, raw_map)
            initial, spin = periodic_spinup(model, spin_p, spin_pet, physical_map, tolerance=1e-8, max_cycles=40); sim = run_model(model, dev_p, dev_pet, physical_map, initial, collect=True)
            local = sim.response_mm.sum(axis=2) * area[None, :] * 1000 / 86400; predicted = local @ support.T; valid = np.isfinite(observed); residual = np.log1p(predicted) - obs_log
            all_e = np.asarray([np.sqrt(np.nanmean(np.where(valid[:, j], residual[:, j] ** 2, np.nan))) for j in range(len(stations))]); bias = np.asarray([abs(np.nansum(predicted[valid[:, j], j]) / np.nansum(observed[valid[:, j], j]) - 1) for j in range(len(stations))]); high = np.asarray([np.sqrt(np.nanmean(residual[valid[:, j] & (observed[:, j] >= q90[j]), j] ** 2)) for j in range(len(stations))]); low = np.asarray([np.sqrt(np.nanmean(residual[valid[:, j] & (observed[:, j] <= q20[j]), j] ** 2)) for j in range(len(stations))])
            tree_mean = lambda v: float(np.mean([np.nanmean(v[tree_ids == tree]) for tree in unique_trees])); aet_rmse = float(np.sqrt(np.mean((monthly_sum(sim.aet_mm, dev_dates) - pml_matrix) ** 2)))
            data_terms = {"all_log_rmse": tree_mean(all_e), "absolute_volume_bias": tree_mean(bias), "high_log_rmse": tree_mean(high), "low_log_rmse": tree_mean(low), "pml_aet_rmse_mm_month": aet_rmse}
            data_objective = float(np.mean([data_terms["all_log_rmse"] / .35, data_terms["absolute_volume_bias"] / .15, data_terms["high_log_rmse"] / .5, data_terms["low_log_rmse"] / .5, data_terms["pml_aet_rmse_mm_month"] / 30]))
            prior = PRIOR_WEIGHT * float(np.mean((coef / PRIOR_SIGMA) ** 2)); objective = data_objective + prior
            boundary = float(np.mean(np.abs(raw_map) >= .999)); cache[key] = (objective, data_objective, prior, data_terms, spin, sim.max_abs_mass_error_mm, boundary, raw_map, physical_map, storage_index, path_index)
            trace_rows.append({"model_id": model, "evaluation": len(cache), "objective": objective, "data_objective": data_objective, "prior_penalty": prior, "parameter_boundary_fraction": boundary, **data_terms}); return objective
        zero = np.zeros(2 * x.shape[1]); start = time.time(); result = minimize(evaluate, zero, method="Powell", bounds=[(-0.75, .75)] * len(zero), options={"maxfev": 220, "maxiter": 18, "xtol": .015, "ftol": .003}); elapsed = time.time() - start
        best = np.asarray(result.x); record = cache[tuple(np.round(best, 9))]; objective, data_objective, prior, terms, spin, mass, boundary, raw_map, physical_map, storage_index, path_index = record
        summary_rows.append({"model_id": model, "objective": objective, "data_objective": data_objective, "prior_penalty": prior, "global_data_objective": lock["objective"], "optimizer_success": bool(result.success), "optimizer_message": str(result.message), "evaluations": int(result.nfev), "elapsed_seconds": elapsed, "parameter_boundary_fraction": boundary, "max_abs_mass_error_mm": mass, **terms})
        for i, reach in enumerate(reach_ids):
            row = {"model_id": model, "reach_id": int(reach), "storage_index": storage_index[i], "path_index": path_index[i]}
            row.update({name: physical_map[i, j] for j, name in enumerate(PARAMETER_NAMES[model])}); parameter_rows.append(row)
        write_json(REPORT / f"{model.lower()}_spatial_map_lock.json", {"model_id": model, "fit_period": "2010-2018", "mapping_coefficients": {"storage_index": dict(zip(FEATURE_NAMES, map(float, best[:len(FEATURE_NAMES)]))), "path_index": dict(zip(FEATURE_NAMES, map(float, best[len(FEATURE_NAMES):])))}, "prior_sigma": PRIOR_SIGMA, "prior_weight": PRIOR_WEIGHT, "objective": objective, "data_objective": data_objective, "prior_penalty": prior, "spinup": spin, "parameter_boundary_fraction": boundary, "optimizer": {"name": "Powell", "success": bool(result.success), "message": str(result.message), "nfev": int(result.nfev)}, "station_identity_used": False, "terminal_tree_identity_used": False, "retrospective_discharge_used": False})
    pd.DataFrame(trace_rows).to_parquet(OUT / "spatial_map_fit_trace.parquet", index=False); summary = pd.DataFrame(summary_rows); summary.to_parquet(OUT / "spatial_map_fit_summary.parquet", index=False); pd.DataFrame(parameter_rows).to_parquet(OUT / "spatial_parameter_fields.parquet", index=False)
    passed = summary.loc[(summary.max_abs_mass_error_mm <= 1e-10) & (summary.parameter_boundary_fraction <= .10), "model_id"].tolist(); decision = {"stage": "20260826_6", "status": "PASS_SPATIAL_MAP_FITS_COMPLETED" if len(passed) == 2 else "STAGE6_SPATIAL_MAP_CONFOUNDED", "numerically_valid_models": passed, "station_or_tree_identity_used": False, "retrospective_discharge_used": False, "TN_used": False, "authorized_successor": "20260826_7"}
    write_json(REPORT / "stage6_decision.json", decision); (REPORT / "technical_report.md").write_text("# 20260826_6 静态属性空间参数场\n\n状态：`%s`。\n\n空间化只学习两个低秩潜在指数，并受固定Gaussian prior约束；所有230条Reach共享同一映射。它不是站点外挂。当前只是全开发期拟合，能否推广必须在后续整棵河树完全删除重训中判断。\n\n%s\n" % (decision["status"], summary.to_markdown(index=False)), encoding="utf-8")
    write_json(REPORT / "integrity.json", {str(p): sha256(p) for p in [RUN / "experiment_contract.json", OUT / "spatial_map_fit_summary.parquet", OUT / "spatial_parameter_fields.parquet", REPORT / "stage6_decision.json", REPORT / "technical_report.md"]}); print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
