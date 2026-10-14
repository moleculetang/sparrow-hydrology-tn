"""Fit the global HBV R0 control and evaluate the no-refit R1 routing ablation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260825_4"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
STAGE3_SCRIPTS = ROOT / "5_Test" / "20260825_3" / "scripts"
sys.path.insert(0, str(STAGE3_SCRIPTS))
from hydrology_core import (  # noqa: E402
    HBVParameters,
    PARAMETER_NAMES,
    load_topology,
    parameters_to_raw,
    periodic_spinup,
    raw_to_parameters,
    route_instantaneous,
    route_linear_channels_adaptive,
    simulate_hbv_ordered,
)


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DEVELOPMENT_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGE_AUDIT = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
SIGNATURES = ROOT / "5_Test" / "20260825_2" / "outputs" / "q_derived_response_signatures_development.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
STATIC = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet"
Q72_BRIDGE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
STAGE3_PROGRAM = ROOT / "5_Test" / "20260825_3" / "program_manifest.json"

PRIOR_WEIGHT = 0.002
VOLUME_WEIGHT = 0.1
RAW_BOUNDS = [(-5.5, 5.5)] * 8
SPINUP_TOLERANCE = 1.0e-8
SPINUP_MAX_CYCLES = 500
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260825


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    return float(1.0 - np.sum((predicted - observed) ** 2) / denominator) if denominator > 0.0 else float("nan")


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.std(observed) <= 0.0 or observed.mean() <= 0.0:
        return float("nan")
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if np.std(predicted) > 0 else 0.0
    alpha = float(np.std(predicted) / np.std(observed))
    beta = float(predicted.mean() / observed.mean())
    return float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def performance_metrics(candidate: str, observed: np.ndarray, predicted: np.ndarray, stations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for index, station in enumerate(stations.itertuples()):
        valid = np.isfinite(observed[:, index]) & np.isfinite(predicted[:, index])
        obs, pred = observed[valid, index], predicted[valid, index]
        if len(obs) < 30:
            continue
        rows.append(
            {
                "candidate": candidate,
                "station_norm": station.station_norm,
                "reach_id": int(station.reach_id),
                "terminal_tree": int(station.terminal_tree),
                "n_days": len(obs),
                "NSE": nse(obs, pred),
                "log_NSE": nse(np.log1p(obs), np.log1p(pred)),
                "KGE": kge(obs, pred),
                "PBIAS_pct": float(100.0 * (pred.sum() - obs.sum()) / obs.sum()),
                "RMSE_m3_s": float(np.sqrt(np.mean((pred - obs) ** 2))),
                "log_RMSE": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2))),
            }
        )
    station_frame = pd.DataFrame(rows)
    valid_all = np.isfinite(observed) & np.isfinite(predicted)
    obs_all, pred_all = observed[valid_all], predicted[valid_all]
    summaries = [
        {
            "candidate": candidate,
            "scope": "pooled",
            "NSE": nse(obs_all, pred_all),
            "log_NSE": nse(np.log1p(obs_all), np.log1p(pred_all)),
            "KGE": kge(obs_all, pred_all),
            "PBIAS_pct": float(100.0 * (pred_all.sum() - obs_all.sum()) / obs_all.sum()),
            "log_RMSE": float(np.sqrt(np.mean((np.log1p(pred_all) - np.log1p(obs_all)) ** 2))),
            "stations": len(station_frame),
        }
    ]
    for scope, reducer in (("station_median", "median"), ("station_mean", "mean")):
        summaries.append(
            {
                "candidate": candidate,
                "scope": scope,
                "NSE": float(getattr(station_frame.NSE, reducer)()),
                "log_NSE": float(getattr(station_frame.log_NSE, reducer)()),
                "KGE": float(getattr(station_frame.KGE, reducer)()),
                "PBIAS_pct": float(getattr(station_frame.PBIAS_pct, reducer)()),
                "log_RMSE": float(getattr(station_frame.log_RMSE, reducer)()),
                "stations": len(station_frame),
            }
        )
    return station_frame, pd.DataFrame(summaries)


def build_support(stations: pd.DataFrame, reach_ids: np.ndarray, order: list[int], downstream: dict[int, tuple[int, float]]) -> np.ndarray:
    identity = np.eye(len(reach_ids), dtype=np.float64)[:, :, None]
    accumulation = route_instantaneous(identity, reach_ids, order, downstream)[:, :, 0]
    support = np.empty((len(stations), len(reach_ids)), dtype=np.float64)
    for station_index, row in enumerate(stations.itertuples()):
        target_index = int(row.reach_id) - 1
        support[station_index] = accumulation[:, target_index]
        support[station_index, target_index] = float(row.downstream_fraction_on_reach)
    return support


class GlobalObjective:
    def __init__(
        self,
        spin_p: np.ndarray,
        spin_pet: np.ndarray,
        development_p: np.ndarray,
        development_pet: np.ndarray,
        observed: np.ndarray,
        support: np.ndarray,
        area_km2: np.ndarray,
        station_trees: np.ndarray,
        prior_raw: np.ndarray,
    ) -> None:
        self.spin_p = spin_p
        self.spin_pet = spin_pet
        self.development_p = development_p
        self.development_pet = development_pet
        self.observed = observed
        self.support = support
        self.area_km2 = area_km2
        self.station_trees = station_trees
        self.prior_raw = prior_raw
        self.cache: dict[tuple[float, ...], float] = {}
        self.trace: list[dict[str, object]] = []

    def predict(self, raw: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object], HBVParameters]:
        parameters = raw_to_parameters(raw)
        initial, spinup = periodic_spinup(
            self.spin_p, self.spin_pet, parameters, SPINUP_TOLERANCE, SPINUP_MAX_CYCLES
        )
        if not spinup["converged"]:
            raise RuntimeError("Periodic spin-up did not converge")
        land = simulate_hbv_ordered(self.development_p, self.development_pet, parameters, initial)
        local_depth = land["q0"] + land["q1"] + land["q2"]
        local_volume = local_depth * self.area_km2[None, :] * 1000.0
        site_q = local_volume @ self.support.T / 86400.0
        return site_q, land, spinup, parameters

    def __call__(self, raw: np.ndarray) -> float:
        key = tuple(np.round(np.asarray(raw, dtype=float), 11))
        if key in self.cache:
            return self.cache[key]
        try:
            predicted, land, spinup, parameters = self.predict(np.asarray(raw, dtype=float))
            squared = (np.log1p(predicted) - np.log1p(self.observed)) ** 2
            squared[~np.isfinite(self.observed)] = np.nan
            station_mse = np.nanmean(squared, axis=0)
            station_log_volume_bias = np.empty(predicted.shape[1], dtype=float)
            for station_index in range(predicted.shape[1]):
                valid = np.isfinite(self.observed[:, station_index])
                station_log_volume_bias[station_index] = np.log(
                    (predicted[valid, station_index].sum() + 1.0) / (self.observed[valid, station_index].sum() + 1.0)
                )
            tree_mse = []
            tree_volume = []
            for tree in np.unique(self.station_trees):
                select = self.station_trees == tree
                tree_mse.append(float(np.mean(station_mse[select])))
                tree_volume.append(float(np.mean(station_log_volume_bias[select] ** 2)))
            process_loss = float(np.mean(tree_mse))
            volume_loss = float(np.mean(tree_volume))
            prior_loss = float(np.mean(((raw - self.prior_raw) / 1.5) ** 2))
            objective = process_loss + VOLUME_WEIGHT * volume_loss + PRIOR_WEIGHT * prior_loss
            maximum_mass_error = float(np.max(np.abs(land["mass_error"])))
        except Exception:
            objective = 1.0e6
            process_loss = volume_loss = prior_loss = maximum_mass_error = float("nan")
            spinup = {"cycles": -1}
            parameters = raw_to_parameters(np.clip(raw, -5.5, 5.5))
        row: dict[str, object] = {
            "evaluation": len(self.trace) + 1,
            "objective": objective,
            "process_loss": process_loss,
            "volume_loss": volume_loss,
            "prior_loss": prior_loss,
            "spinup_cycles": spinup.get("cycles", -1),
            "land_mass_max_abs_error_mm": maximum_mass_error,
        }
        row.update({f"raw_{name}": float(value) for name, value in zip(PARAMETER_NAMES, raw)})
        row.update({name: float(getattr(parameters, name)) for name in PARAMETER_NAMES})
        self.trace.append(row)
        self.cache[key] = objective
        if len(self.trace) % 25 == 0:
            print(f"GLOBAL_HBV objective evaluation {len(self.trace)}: {objective:.6f}", flush=True)
        return objective


def paired_bootstrap(differences: pd.DataFrame) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    station_values = differences.delta_log_RMSE.to_numpy(float)
    station_boot = station_values[rng.integers(0, len(station_values), size=(BOOTSTRAP_REPLICATES, len(station_values)))].mean(axis=1)
    tree_values = differences.groupby("terminal_tree").delta_log_RMSE.mean().to_numpy(float)
    tree_boot = tree_values[rng.integers(0, len(tree_values), size=(BOOTSTRAP_REPLICATES, len(tree_values)))].mean(axis=1)
    return {
        "point_mean_delta_log_RMSE": float(station_values.mean()),
        "station_ci95_lower": float(np.quantile(station_boot, 0.025)),
        "station_ci95_upper": float(np.quantile(station_boot, 0.975)),
        "tree_ci95_lower": float(np.quantile(tree_boot, 0.025)),
        "tree_ci95_upper": float(np.quantile(tree_boot, 0.975)),
    }


def model_path_signatures(
    candidate: str,
    dates: pd.DatetimeIndex,
    stations: pd.DataFrame,
    component_volume: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seasons = np.asarray(pd.Series(dates.month).map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}))
    groups: list[tuple[str, str, np.ndarray]] = [("overall", "all", np.ones(len(dates), dtype=bool))]
    groups.extend(("year", str(year), dates.year == year) for year in range(2010, 2019))
    groups.extend(("season", season, seasons == season) for season in ("DJF", "MAM", "JJA", "SON"))
    total = component_volume.sum(axis=2)
    slow = component_volume[:, :, 2]
    for group_type, group_value, select in groups:
        ratio = slow[select].sum(axis=0) / np.maximum(total[select].sum(axis=0), 1.0e-12)
        for station_index, station in enumerate(stations.itertuples()):
            rows.append(
                {
                    "candidate": candidate,
                    "station_norm": station.station_norm,
                    "reach_id": int(station.reach_id),
                    "terminal_tree": int(station.terminal_tree),
                    "group_type": group_type,
                    "group_value": group_value,
                    "model_q2_volume_fraction": float(ratio[station_index]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGE_AUDIT)
    stations = gauges.loc[gauges.topology_representative].drop_duplicates("station_norm").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    if len(stations) != 92 or stations.reach_id.duplicated().any():
        raise RuntimeError("Expected 92 unique representative Gauge-Reach pairs")
    support = build_support(stations, reach_ids, order, downstream)
    area = pd.read_parquet(Q72_BRIDGE, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids)
    area_values = area.catchment_area_km2.to_numpy(float)
    support_area = support @ area_values
    support_area_delta = float(np.max(np.abs(support_area - stations.topology_support_area_km2.to_numpy(float))))
    if support_area_delta > 1.0e-8:
        raise RuntimeError(f"Gauge support operator differs from Stage 2: {support_area_delta}")

    forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    spin = forcing.loc[forcing.date.dt.year.between(2006, 2009)]
    development = forcing.loc[forcing.date.dt.year.between(2010, 2018)]
    spin_dates = pd.date_range("2006-01-01", "2009-12-31", freq="D")
    dates = pd.date_range("2010-01-01", "2018-12-31", freq="D")
    spin_p = spin.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=spin_dates, columns=reach_ids).to_numpy(float)
    spin_pet = spin.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=spin_dates, columns=reach_ids).to_numpy(float)
    development_p = development.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(float)
    development_pet = development.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(float)
    if any(np.isnan(array).any() for array in (spin_p, spin_pet, development_p, development_pet)):
        raise RuntimeError("Forcing matrices are incomplete")

    discharge = pd.read_parquet(DEVELOPMENT_Q)
    discharge["date"] = pd.to_datetime(discharge.date)
    observed = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(float)
    prior = HBVParameters(200.0, 2.0, 0.70, 2.0, 5.0, 1.4426950408889634, 8.048526073147784, 40.00733226320923)
    prior_raw = parameters_to_raw(prior)
    objective = GlobalObjective(
        spin_p,
        spin_pet,
        development_p,
        development_pet,
        observed,
        support,
        area_values,
        stations.terminal_tree.to_numpy(int),
        prior_raw,
    )
    offsets = [
        np.zeros(8),
        np.asarray([0.6, -0.5, 0.3, -0.4, 0.5, -0.3, 0.4, 0.5]),
        np.asarray([-0.6, 0.5, -0.3, 0.4, -0.5, 0.3, -0.4, -0.5]),
    ]
    fit_rows: list[dict[str, object]] = []
    results = []
    for start_index, offset in enumerate(offsets):
        start = np.clip(prior_raw + offset, -5.25, 5.25)
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=RAW_BOUNDS,
            options={"maxiter": 45, "maxfun": 550, "ftol": 1.0e-10, "gtol": 1.0e-5, "maxls": 20},
        )
        results.append(result)
        fit_rows.append(
            {
                "start": start_index,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "objective": float(result.fun),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
            }
        )
        print(f"GLOBAL_HBV start {start_index}: objective={result.fun:.6f}, success={result.success}", flush=True)
    objective_best = min(results, key=lambda result: float(result.fun))
    successful_results = [result for result in results if bool(result.success)]
    successful_best = min(successful_results, key=lambda result: float(result.fun)) if successful_results else None
    # L-BFGS-B can stop at maxfun after reaching the same numerical basin as a
    # formally converged start. Prefer the converged solution when its objective
    # differs from the absolute minimum by no more than the registered numerical
    # equivalence tolerance; do not report an artificial optimizer failure caused
    # by a few last-bit objective differences.
    convergence_equivalence_tolerance = 1.0e-8
    best = (
        successful_best
        if successful_best is not None
        and float(successful_best.fun) <= float(objective_best.fun) + convergence_equivalence_tolerance
        else objective_best
    )
    best_raw = np.asarray(best.x, dtype=float)
    best_parameters = raw_to_parameters(best_raw)
    predictions_r0, land, spinup, _ = objective.predict(best_raw)
    trace = pd.DataFrame(objective.trace)
    trace.to_parquet(OUT / "global_hbv_optimization_trace.parquet", index=False)
    pd.DataFrame(fit_rows).to_parquet(OUT / "global_hbv_multistart_summary.parquet", index=False)
    parameter_lock = {
        "candidate": "GLOBAL_HBV_R0",
        "objective": float(best.fun),
        "raw_parameters": dict(zip(PARAMETER_NAMES, map(float, best_raw))),
        "physical_parameters": {name: float(getattr(best_parameters, name)) for name in PARAMETER_NAMES},
        "k0_day": best_parameters.k0_day,
        "k1_day": best_parameters.k1_day,
        "k2_day": best_parameters.k2_day,
        "spinup": spinup,
        "boundary_confounded": bool(np.any(np.abs(best_raw) >= 5.45)),
        "development_discharge_only": True,
        "convergence_equivalence_tolerance": convergence_equivalence_tolerance,
        "selected_formally_converged_solution": bool(best.success),
    }
    write_json(REPORT / "global_hbv_development_parameter_lock.json", parameter_lock)

    local_components = np.stack([land["q0"], land["q1"], land["q2"]], axis=2) * area_values[None, :, None] * 1000.0
    site_components_r0 = np.stack([local_components[:, :, component] @ support.T for component in range(3)], axis=2)
    r0_check = site_components_r0.sum(axis=2) / 86400.0
    if float(np.max(np.abs(r0_check - predictions_r0))) > 1.0e-10:
        raise RuntimeError("R0 component prediction does not close to fitted total")

    r0_reach = route_instantaneous(local_components, reach_ids, order, downstream)
    geometry = pd.read_parquet(GEOMETRY, columns=["reach_id", "bankfull_width_m", "bankfull_depth_m"]).set_index("reach_id").reindex(reach_ids)
    static = pd.read_parquet(STATIC, columns=["reach_id", "length_km"]).set_index("reach_id").reindex(reach_ids)
    own_q = np.maximum(r0_reach.sum(axis=2) / 86400.0, 1.0e-6)
    tau_dynamic = (
        static.length_km.to_numpy(float)[None, :] * 1000.0
        * geometry.bankfull_width_m.to_numpy(float)[None, :]
        * geometry.bankfull_depth_m.to_numpy(float)[None, :]
        / own_q
        / 86400.0
    )
    print("Routing GLOBAL_HBV_R1 with candidate-own daily hydraulic travel times", flush=True)
    r1 = route_linear_channels_adaptive(local_components, reach_ids, order, downstream, tau_dynamic, cfl_limit=0.9)
    site_components_r1 = np.empty_like(site_components_r0)
    for station_index, row in enumerate(stations.itertuples()):
        reach_index = int(row.reach_id) - 1
        fraction = float(row.downstream_fraction_on_reach)
        site_components_r1[:, station_index, :] = (
            (1.0 - fraction) * r1["upstream_inflow_m3"][:, reach_index, :]
            + fraction * r1["outflow_m3"][:, reach_index, :]
        )
    predictions_r1 = site_components_r1.sum(axis=2) / 86400.0

    station_metrics = []
    summaries = []
    for candidate, prediction in (("GLOBAL_HBV_R0", predictions_r0), ("GLOBAL_HBV_R1", predictions_r1)):
        station_frame, summary = performance_metrics(candidate, observed, prediction, stations)
        station_metrics.append(station_frame)
        summaries.append(summary)
    station_metrics_frame = pd.concat(station_metrics, ignore_index=True)
    summary_frame = pd.concat(summaries, ignore_index=True)

    prediction_rows = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), len(stations)),
            "station_norm": np.tile(stations.station_norm.to_numpy(), len(dates)),
            "reach_id": np.tile(stations.reach_id.to_numpy(int), len(dates)),
            "terminal_tree": np.tile(stations.terminal_tree.to_numpy(int), len(dates)),
            "q_observed_m3_s": observed.reshape(-1),
            "q_global_hbv_r0_m3_s": predictions_r0.reshape(-1),
            "q_global_hbv_r1_m3_s": predictions_r1.reshape(-1),
        }
    )
    for candidate_key, components in (("r0", site_components_r0), ("r1", site_components_r1)):
        for component_index, component_name in enumerate(("q0", "q1", "q2")):
            prediction_rows[f"{candidate_key}_{component_name}_m3_s"] = components[:, :, component_index].reshape(-1) / 86400.0
    prediction_rows.to_parquet(OUT / "global_hbv_development_predictions.parquet", index=False)

    monthly_obs = prediction_rows.assign(year=prediction_rows.date.dt.year, month=prediction_rows.date.dt.month).groupby(
        ["station_norm", "reach_id", "terminal_tree", "year", "month"], as_index=False
    ).agg(
        q_observed_m3_s=("q_observed_m3_s", "mean"),
        observed_days=("q_observed_m3_s", "count"),
        q_global_hbv_r0_m3_s=("q_global_hbv_r0_m3_s", "mean"),
        q_global_hbv_r1_m3_s=("q_global_hbv_r1_m3_s", "mean"),
    )
    q72 = pd.read_parquet(Q72_BRIDGE, columns=["reach_id", "year", "month", "local_total_m3_s", "routed_total_m3_s"])
    q72 = q72.loc[q72.year.between(2010, 2018)]
    q72_station_rows = []
    for row in stations.itertuples():
        selected = q72.loc[q72.reach_id.eq(int(row.reach_id))].copy()
        selected["station_norm"] = row.station_norm
        selected["q_q72_monthly_reference_m3_s"] = selected.routed_total_m3_s - (1.0 - float(row.downstream_fraction_on_reach)) * selected.local_total_m3_s
        q72_station_rows.append(selected[["station_norm", "year", "month", "q_q72_monthly_reference_m3_s"]])
    monthly = monthly_obs.merge(pd.concat(q72_station_rows), on=["station_norm", "year", "month"], how="left", validate="one_to_one")
    monthly = monthly.loc[monthly.observed_days.ge(20)].copy()
    monthly.to_parquet(OUT / "global_hbv_monthly_development_predictions.parquet", index=False)
    monthly_station_metrics = []
    monthly_summaries = []
    monthly_dates = pd.MultiIndex.from_frame(monthly[["year", "month"]]).unique()
    # Re-pivot each candidate on the same station-month support.
    for candidate, column in (
        ("GLOBAL_HBV_R0", "q_global_hbv_r0_m3_s"),
        ("GLOBAL_HBV_R1", "q_global_hbv_r1_m3_s"),
        ("Q72_MONTHLY_REFERENCE", "q_q72_monthly_reference_m3_s"),
    ):
        obs_matrix = monthly.pivot(index=["year", "month"], columns="station_norm", values="q_observed_m3_s").reindex(index=monthly_dates, columns=stations.station_norm).to_numpy(float)
        pred_matrix = monthly.pivot(index=["year", "month"], columns="station_norm", values=column).reindex(index=monthly_dates, columns=stations.station_norm).to_numpy(float)
        sf, sm = performance_metrics(candidate, obs_matrix, pred_matrix, stations)
        sf["temporal_scale"] = "monthly"
        sm["temporal_scale"] = "monthly"
        monthly_station_metrics.append(sf)
        monthly_summaries.append(sm)
    station_metrics_frame["temporal_scale"] = "daily"
    summary_frame["temporal_scale"] = "daily"
    all_station_metrics = pd.concat([station_metrics_frame, *monthly_station_metrics], ignore_index=True)
    all_summaries = pd.concat([summary_frame, *monthly_summaries], ignore_index=True)
    all_station_metrics.to_parquet(OUT / "global_hbv_station_performance.parquet", index=False)
    all_summaries.to_parquet(OUT / "global_hbv_performance_summary.parquet", index=False)

    r0_station = station_metrics_frame.loc[station_metrics_frame.candidate.eq("GLOBAL_HBV_R0"), ["station_norm", "terminal_tree", "log_RMSE", "PBIAS_pct"]]
    r1_station = station_metrics_frame.loc[station_metrics_frame.candidate.eq("GLOBAL_HBV_R1"), ["station_norm", "log_RMSE", "PBIAS_pct"]]
    paired = r0_station.merge(r1_station, on="station_norm", suffixes=("_r0", "_r1"), validate="one_to_one")
    paired["delta_log_RMSE"] = paired.log_RMSE_r1 - paired.log_RMSE_r0
    paired.to_parquet(OUT / "r1_vs_r0_paired_station_metrics.parquet", index=False)
    bootstrap = paired_bootstrap(paired)
    monthly_metric = pd.concat(monthly_station_metrics, ignore_index=True)
    monthly_r0_pbias = float(monthly_metric.loc[monthly_metric.candidate.eq("GLOBAL_HBV_R0"), "PBIAS_pct"].abs().median())
    monthly_r1_pbias = float(monthly_metric.loc[monthly_metric.candidate.eq("GLOBAL_HBV_R1"), "PBIAS_pct"].abs().median())
    r1_selected = bool(
        bootstrap["station_ci95_upper"] < 0.0
        and bootstrap["tree_ci95_upper"] < 0.0
        and monthly_r1_pbias - monthly_r0_pbias <= 1.0
    )
    selected_route = "R1" if r1_selected else "R0"
    route_decision = {
        **bootstrap,
        "monthly_station_median_abs_PBIAS_R0_pct": monthly_r0_pbias,
        "monthly_station_median_abs_PBIAS_R1_pct": monthly_r1_pbias,
        "R1_selected": r1_selected,
        "selected_route_for_stage5": selected_route,
        "R1_land_parameters_refit": False,
        "maximum_cfl": float(r1["maximum_cfl"]),
        "maximum_substeps_per_day": int(np.max(r1["substeps_by_day"])),
        "routing_mass_relative_error": float(np.max(np.abs(r1["mass_error_m3"]))) / max(float(local_components.sum()), 1.0),
    }
    write_json(REPORT / "r0_r1_route_selection.json", route_decision)

    signature_proxy = pd.read_parquet(SIGNATURES)
    signature_proxy = signature_proxy.loc[signature_proxy.signature.eq("bfi_three_method_median"), ["station_norm", "group_type", "group_value", "value"]]
    signature_frames = []
    for candidate, components in (("GLOBAL_HBV_R0", site_components_r0), ("GLOBAL_HBV_R1", site_components_r1)):
        frame = model_path_signatures(candidate, dates, stations, components)
        frame = frame.merge(signature_proxy, on=["station_norm", "group_type", "group_value"], how="left", validate="one_to_one")
        frame = frame.rename(columns={"value": "q_derived_bfi_three_method_median"})
        signature_frames.append(frame)
    signature_audit = pd.concat(signature_frames, ignore_index=True)
    signature_audit.to_parquet(OUT / "global_hbv_q_derived_signature_diagnostic.parquet", index=False)
    signature_summary_rows = []
    for candidate, group in signature_audit.groupby("candidate"):
        valid = group.q_derived_bfi_three_method_median.notna()
        overall = group.loc[valid & group.group_type.eq("overall")]
        signature_summary_rows.append(
            {
                "candidate": candidate,
                "rows": int(valid.sum()),
                "bfi_fraction_RMSE": float(np.sqrt(np.mean((group.loc[valid, "model_q2_volume_fraction"] - group.loc[valid, "q_derived_bfi_three_method_median"]) ** 2))),
                "overall_station_spearman": float(spearmanr(overall.model_q2_volume_fraction, overall.q_derived_bfi_three_method_median).statistic),
                "role": "DIAGNOSTIC_ONLY_NOT_LIKELIHOOD",
            }
        )
    pd.DataFrame(signature_summary_rows).to_parquet(OUT / "global_hbv_q_derived_signature_summary.parquet", index=False)

    daily_selected = all_summaries.loc[(all_summaries.candidate.eq(f"GLOBAL_HBV_{selected_route}")) & all_summaries.temporal_scale.eq("daily") & all_summaries.scope.eq("station_median")].iloc[0]
    routing_mass_relative = route_decision["routing_mass_relative_error"]
    technical_pass = bool(
        np.isfinite(best.fun)
        and float(np.max(np.abs(land["mass_error"]))) <= 1.0e-10
        and routing_mass_relative <= 1.0e-12
        and float(daily_selected.NSE) >= -0.5
    )
    decision = {
        "stage": "20260825_4",
        "status": "PASS_GLOBAL_HBV_CONTROL_AND_ROUTING_ABLATION" if technical_pass else "GLOBAL_HBV_TECHNICAL_GATE_FAILED",
        "selected_route_for_stage5": selected_route if technical_pass else None,
        "global_parameter_objective": float(best.fun),
        "optimizer_reported_success": bool(best.success),
        "parameter_boundary_confounded": parameter_lock["boundary_confounded"],
        "development_selected_route_station_median_NSE": float(daily_selected.NSE),
        "development_selected_route_station_median_log_NSE": float(daily_selected.log_NSE),
        "land_mass_max_abs_error_mm": float(np.max(np.abs(land["mass_error"]))),
        "route_selection": route_decision,
        "support_area_max_abs_delta_km2": support_area_delta,
        "locked_2019_2022_discharge_read": False,
        "signatures_used_in_fit": False,
        "authorized_successor": "20260825_5" if technical_pass else None,
    }
    write_json(REPORT / "stage4_decision.json", decision)
    report = f"""# 20260825_4 Global-HBV与河道路由消融

## 结论

状态：`{decision['status']}`；进入下一阶段的路由为`{selected_route}`。

Global-HBV仅使用2010–2018总流量拟合8个全局过程参数；没有站点参数、Gauge校正、逐日分割比例likelihood或2019–2022流量。最佳目标为`{best.fun:.6f}`，参数边界混淆为`{parameter_lock['boundary_confounded']}`。

所选路由的开发期站点中位NSE为`{daily_selected.NSE:.3f}`，中位log-NSE为`{daily_selected.log_NSE:.3f}`。R1相对R0的站点log-RMSE差为`{bootstrap['point_mean_delta_log_RMSE']:.5f}`，station/tree CI95上界分别为`{bootstrap['station_ci95_upper']:.5f}`和`{bootstrap['tree_ci95_upper']:.5f}`。R1未重新拟合陆面参数。

R1水量相对误差为`{routing_mass_relative:.3e}`，最大CFL为`{route_decision['maximum_cfl']:.3f}`。Andreadis只提供宽和深；动态旅行时间使用候选自身Q，未读取WQD参考流量。

Q2与三种水文曲线分割的关系只作为诊断，未参与拟合。当前开发期性能不能证明未设站空间推广；该问题留给`20260825_5`整棵河树删除重训。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    program = json.loads(STAGE3_PROGRAM.read_text(encoding="utf-8"))
    program["stage_status"]["20260825_4"] = decision["status"]
    program["stage_status"]["20260825_5"] = "authorized_not_started" if technical_pass else "closed"
    (RUN / "program_manifest.json").write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    integrity_paths = [
        RUN / "experiment_contract.json",
        RUN / "scripts" / "run_stage4.py",
        REPORT / "global_hbv_development_parameter_lock.json",
        REPORT / "r0_r1_route_selection.json",
        OUT / "global_hbv_development_predictions.parquet",
        OUT / "global_hbv_station_performance.parquet",
        OUT / "global_hbv_q_derived_signature_diagnostic.parquet",
        REPORT / "stage4_decision.json",
    ]
    write_json(REPORT / "integrity.json", {str(path): sha256(path) for path in integrity_paths})
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    if not technical_pass:
        raise RuntimeError(decision)


if __name__ == "__main__":
    main()
