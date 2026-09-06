"""Run nested complete-terminal-tree evaluation of GLOBAL-HBV and MPR-lite."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260825_5"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
STAGE3_SCRIPTS = ROOT / "5_Test" / "20260825_3" / "scripts"
sys.path.insert(0, str(STAGE3_SCRIPTS))
from hydrology_core import (  # noqa: E402
    HBVParameters,
    load_topology,
    parameters_to_raw,
    route_instantaneous,
    route_linear_channels_adaptive,
)
from regional_hbv_core import (  # noqa: E402
    PARAMETER_NAMES,
    RegionalObjective,
    fit_objective,
)


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DEVELOPMENT_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGE_AUDIT = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
Q72_BRIDGE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
STATIC = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet"
SOIL_HYDRO = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "soil_tn_glhymps_by_reach.parquet"
SOILGRID = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "soilgrids_son_priors_by_reach.parquet"
AWC_GEOJSON = ROOT / "5_Test" / "20260729_25" / "outputs" / "full_profile_awc_reach_zonal.geojson"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
STAGE4_PROGRAM = ROOT / "5_Test" / "20260825_4" / "program_manifest.json"
STAGE4_DECISION = ROOT / "5_Test" / "20260825_4" / "reports" / "stage4_decision.json"

SPINUP_TOLERANCE = 1.0e-8
SPINUP_MAX_CYCLES = 500
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260825
SPATIAL_NONINFERIORITY_MARGIN = 0.01
PBIAS_WORSENING_MARGIN_PP = 2.0
RAW_BOUNDARY_THRESHOLD = 5.5


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
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if np.std(predicted) > 0.0 else 0.0
    alpha = float(np.std(predicted) / np.std(observed))
    beta = float(predicted.mean() / observed.mean())
    return float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def build_support(stations: pd.DataFrame, reach_ids: np.ndarray, order: list[int], downstream: dict[int, tuple[int, float]]) -> np.ndarray:
    identity = np.eye(len(reach_ids), dtype=np.float64)[:, :, None]
    accumulation = route_instantaneous(identity, reach_ids, order, downstream)[:, :, 0]
    support = np.empty((len(stations), len(reach_ids)), dtype=np.float64)
    for station_index, row in enumerate(stations.itertuples()):
        target_index = int(row.reach_id) - 1
        support[station_index] = accumulation[:, target_index]
        support[station_index, target_index] = float(row.downstream_fraction_on_reach)
    return support


def station_metrics(candidate: str, observed: np.ndarray, predicted: np.ndarray, stations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station_index, station in enumerate(stations.itertuples()):
        valid = np.isfinite(observed[:, station_index]) & np.isfinite(predicted[:, station_index])
        obs, pred = observed[valid, station_index], predicted[valid, station_index]
        if len(obs) < 30:
            continue
        rows.append(
            {
                "candidate": candidate,
                "station_norm": station.station_norm,
                "reach_id": int(station.reach_id),
                "terminal_tree": int(station.terminal_tree),
                "n_days": int(len(obs)),
                "NSE": nse(obs, pred),
                "log_NSE": nse(np.log1p(obs), np.log1p(pred)),
                "KGE": kge(obs, pred),
                "PBIAS_pct": float(100.0 * (pred.sum() - obs.sum()) / obs.sum()),
                "RMSE_m3_s": float(np.sqrt(np.mean((pred - obs) ** 2))),
                "log_RMSE": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2))),
            }
        )
    return pd.DataFrame(rows)


def monthly_station_metrics(candidate: str, dates: pd.DatetimeIndex, observed: np.ndarray, predicted: np.ndarray, stations: pd.DataFrame) -> pd.DataFrame:
    month_codes = dates.to_period("M")
    unique_months = month_codes.unique()
    monthly_obs = np.full((len(unique_months), observed.shape[1]), np.nan, dtype=np.float64)
    monthly_pred = np.full_like(monthly_obs, np.nan)
    for month_index, month in enumerate(unique_months):
        selected = np.asarray(month_codes == month)
        for station_index in range(observed.shape[1]):
            valid = selected & np.isfinite(observed[:, station_index]) & np.isfinite(predicted[:, station_index])
            if int(valid.sum()) >= 20:
                monthly_obs[month_index, station_index] = float(observed[valid, station_index].mean())
                monthly_pred[month_index, station_index] = float(predicted[valid, station_index].mean())
    frame = station_metrics(candidate, monthly_obs, monthly_pred, stations)
    frame = frame.rename(columns={"n_days": "n_months"})
    return frame


def bootstrap_delta(paired: pd.DataFrame, seed_offset: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    values = paired.delta_log_RMSE.to_numpy(float)
    station_boot = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1)
    tree_values = paired.groupby("terminal_tree").delta_log_RMSE.mean().to_numpy(float)
    tree_boot = tree_values[rng.integers(0, len(tree_values), size=(BOOTSTRAP_REPLICATES, len(tree_values)))].mean(axis=1)
    return {
        "point_mean_delta_log_RMSE": float(values.mean()),
        "station_ci95_lower": float(np.quantile(station_boot, 0.025)),
        "station_ci95_upper": float(np.quantile(station_boot, 0.975)),
        "tree_ci95_lower": float(np.quantile(tree_boot, 0.025)),
        "tree_ci95_upper": float(np.quantile(tree_boot, 0.975)),
    }


def build_attributes(forcing: pd.DataFrame, terminal: dict[int, int], reach_ids: np.ndarray) -> pd.DataFrame:
    awc_json = json.loads(AWC_GEOJSON.read_text(encoding="utf-8"))
    awc = pd.DataFrame(
        [feature["properties"] for feature in awc_json["features"]]
    ).rename(columns={"mean": "awc_0_200_mm"})[["reach_id", "awc_0_200_mm"]]
    soilgrid = pd.read_parquet(SOILGRID)
    depth_weight = {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0}
    shallow = soilgrid.loc[soilgrid.depth.isin(depth_weight)].copy()
    shallow["weight"] = shallow.depth.map(depth_weight)
    shallow["weighted_bd"] = shallow.bulk_density_g_cm3 * shallow.weight
    shallow_bd = shallow.groupby("reach_id", as_index=False).agg(
        weighted_bd=("weighted_bd", "sum"), weight=("weight", "sum")
    )
    shallow_bd["bulk_density_0_30_g_cm3"] = shallow_bd.weighted_bd / shallow_bd.weight
    hydro = pd.read_parquet(
        SOIL_HYDRO,
        columns=["reach_id", "glhymps_log10_permeability_m2", "glhymps_porosity"],
    )
    static = pd.read_parquet(STATIC, columns=["reach_id", "slope"])
    development = forcing.loc[forcing.date.dt.year.between(2010, 2018)]
    climate = development.groupby("reach_id", as_index=False).agg(
        precipitation_mm=("precipitation_daily_mm", "sum"),
        pet_mm=("pet_fao56_mm_day", "sum"),
    )
    frame = (
        pd.DataFrame({"reach_id": reach_ids})
        .merge(awc, on="reach_id", validate="one_to_one")
        .merge(shallow_bd[["reach_id", "bulk_density_0_30_g_cm3"]], on="reach_id", validate="one_to_one")
        .merge(hydro, on="reach_id", validate="one_to_one")
        .merge(static, on="reach_id", validate="one_to_one")
        .merge(climate, on="reach_id", validate="one_to_one")
    )
    frame["terminal_tree"] = frame.reach_id.map(terminal).astype(int)
    slope = np.maximum(frame.slope.to_numpy(float), 1.0e-6)
    awc_value = np.maximum(frame.awc_0_200_mm.to_numpy(float), 1.0)
    permeability = frame.glhymps_log10_permeability_m2.to_numpy(float)
    porosity = np.clip(frame.glhymps_porosity.to_numpy(float), 1.0e-4, 1.0 - 1.0e-4)
    aridity = np.maximum(frame.pet_mm.to_numpy(float) / np.maximum(frame.precipitation_mm.to_numpy(float), 1.0), 1.0e-6)
    frame["feature_fc_mm"] = np.log(awc_value)
    frame["feature_beta"] = frame.bulk_density_0_30_g_cm3.to_numpy(float)
    frame["feature_lp"] = np.log(aridity)
    frame["feature_perc_mm_day"] = permeability
    frame["feature_uzl_mm"] = np.log(awc_value / slope)
    frame["feature_tau0_day"] = np.log(slope)
    permeability_centered = permeability - float(np.median(permeability))
    log_slope_centered = np.log(slope) - float(np.median(np.log(slope)))
    logit_porosity = np.log(porosity) - np.log1p(-porosity)
    frame["feature_delta_tau10_day"] = permeability_centered * log_slope_centered
    frame["feature_delta_tau21_day"] = permeability_centered + logit_porosity
    feature_columns = [f"feature_{name}" for name in PARAMETER_NAMES]
    if len(frame) != 230 or frame.reach_id.duplicated().any() or not np.isfinite(frame[feature_columns].to_numpy(float)).all():
        raise RuntimeError("Registered MPR attributes are incomplete or invalid")
    return frame


def route_candidate(
    label: str,
    local_components_mm: np.ndarray,
    area_km2: np.ndarray,
    support: np.ndarray,
    reach_ids: np.ndarray,
    order: list[int],
    downstream: dict[int, tuple[int, float]],
    stations: pd.DataFrame,
    observed: np.ndarray,
    training_indices: np.ndarray,
    dates: pd.DatetimeIndex,
    length_km: np.ndarray,
    width_m: np.ndarray,
    depth_m: np.ndarray,
    seed_offset: int,
) -> tuple[np.ndarray, dict[str, object]]:
    local_m3 = local_components_mm * area_km2[None, :, None] * 1000.0
    site_r0 = np.stack([local_m3[:, :, index] @ support.T for index in range(3)], axis=2)
    reach_r0 = route_instantaneous(local_m3, reach_ids, order, downstream)
    own_q = np.maximum(reach_r0.sum(axis=2) / 86400.0, 1.0e-6)
    tau = length_km[None, :] * 1000.0 * width_m[None, :] * depth_m[None, :] / own_q / 86400.0
    r1 = route_linear_channels_adaptive(local_m3, reach_ids, order, downstream, tau, cfl_limit=0.9)
    site_r1 = np.empty_like(site_r0)
    for station_index, station in enumerate(stations.itertuples()):
        reach_index = int(station.reach_id) - 1
        fraction = float(station.downstream_fraction_on_reach)
        site_r1[:, station_index, :] = (
            (1.0 - fraction) * r1["upstream_inflow_m3"][:, reach_index, :]
            + fraction * r1["outflow_m3"][:, reach_index, :]
        )
    train_stations = stations.iloc[training_indices].reset_index(drop=True)
    observed_train = observed[:, training_indices]
    q0_train = site_r0[:, training_indices, :].sum(axis=2) / 86400.0
    q1_train = site_r1[:, training_indices, :].sum(axis=2) / 86400.0
    metric_r0 = station_metrics("R0", observed_train, q0_train, train_stations)
    metric_r1 = station_metrics("R1", observed_train, q1_train, train_stations)
    paired = metric_r0[["station_norm", "terminal_tree", "log_RMSE"]].merge(
        metric_r1[["station_norm", "log_RMSE"]], on="station_norm", suffixes=("_r0", "_r1"), validate="one_to_one"
    )
    paired["delta_log_RMSE"] = paired.log_RMSE_r1 - paired.log_RMSE_r0
    boot = bootstrap_delta(paired, seed_offset)
    monthly_r0 = monthly_station_metrics("R0", dates, observed_train, q0_train, train_stations)
    monthly_r1 = monthly_station_metrics("R1", dates, observed_train, q1_train, train_stations)
    r0_abs_pbias = float(monthly_r0.PBIAS_pct.abs().median())
    r1_abs_pbias = float(monthly_r1.PBIAS_pct.abs().median())
    select_r1 = bool(
        boot["station_ci95_upper"] < 0.0
        and boot["tree_ci95_upper"] < 0.0
        and r1_abs_pbias - r0_abs_pbias <= 1.0
    )
    audit: dict[str, object] = {
        "label": label,
        **boot,
        "monthly_station_median_abs_PBIAS_R0_pct": r0_abs_pbias,
        "monthly_station_median_abs_PBIAS_R1_pct": r1_abs_pbias,
        "selected_route": "R1" if select_r1 else "R0",
        "maximum_cfl": float(r1["maximum_cfl"]),
        "maximum_substeps_per_day": int(np.max(r1["substeps_by_day"])),
        "routing_mass_relative_error": float(np.max(np.abs(r1["mass_error_m3"]))) / max(float(local_m3.sum()), 1.0),
    }
    return (site_r1 if select_r1 else site_r0), audit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    stage4 = json.loads(STAGE4_DECISION.read_text(encoding="utf-8"))
    if stage4["status"] != "PASS_GLOBAL_HBV_CONTROL_AND_ROUTING_ABLATION" or stage4["selected_route_for_stage5"] != "R0":
        raise RuntimeError("Stage 5 requires the locked Stage-4 R0 decision")

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGE_AUDIT)
    stations = gauges.loc[gauges.topology_representative].drop_duplicates("station_norm").sort_values(
        ["terminal_tree", "station_norm"]
    ).reset_index(drop=True)
    if len(stations) != 92 or stations.reach_id.duplicated().any():
        raise RuntimeError("Expected 92 representative unique Gauge-Reach pairs")
    support = build_support(stations, reach_ids, order, downstream)
    area = pd.read_parquet(Q72_BRIDGE, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids)
    area_km2 = area.catchment_area_km2.to_numpy(float)
    support_delta = float(np.max(np.abs(support @ area_km2 - stations.topology_support_area_km2.to_numpy(float))))
    if support_delta > 1.0e-8:
        raise RuntimeError(f"Gauge support differs from Stage 2 by {support_delta}")

    forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    attributes = build_attributes(forcing, terminal, reach_ids)
    attributes.to_parquet(OUT / "mpr_attributes_by_reach.parquet", index=False)
    feature_columns = [f"feature_{name}" for name in PARAMETER_NAMES]
    raw_features = attributes[feature_columns].to_numpy(float)
    reach_terminals = attributes.terminal_tree.to_numpy(int)

    spin_dates = pd.date_range("2006-01-01", "2009-12-31", freq="D")
    dates = pd.date_range("2010-01-01", "2018-12-31", freq="D")
    spin = forcing.loc[forcing.date.dt.year.between(2006, 2009)]
    development = forcing.loc[forcing.date.dt.year.between(2010, 2018)]
    spin_p = spin.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=spin_dates, columns=reach_ids).to_numpy(float)
    spin_pet = spin.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=spin_dates, columns=reach_ids).to_numpy(float)
    development_p = development.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(float)
    development_pet = development.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(float)
    if any(np.isnan(value).any() for value in (spin_p, spin_pet, development_p, development_pet)):
        raise RuntimeError("Forcing matrices are incomplete")
    discharge = pd.read_parquet(DEVELOPMENT_Q)
    discharge["date"] = pd.to_datetime(discharge.date)
    observed = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(float)

    static = pd.read_parquet(STATIC, columns=["reach_id", "length_km"]).set_index("reach_id").reindex(reach_ids)
    geometry = pd.read_parquet(GEOMETRY, columns=["reach_id", "bankfull_width_m", "bankfull_depth_m"]).set_index("reach_id").reindex(reach_ids)
    length_km = static.length_km.to_numpy(float)
    width_m = geometry.bankfull_width_m.to_numpy(float)
    depth_m = geometry.bankfull_depth_m.to_numpy(float)
    if not all(np.isfinite(value).all() and np.all(value > 0.0) for value in (length_km, width_m, depth_m)):
        raise RuntimeError("Allowed reach geometry is incomplete")

    prior = HBVParameters(200.0, 2.0, 0.70, 2.0, 5.0, 1.4426950408889634, 8.048526073147784, 40.00733226320923)
    registered_prior_raw = parameters_to_raw(prior)
    tree_ids = sorted(map(int, stations.terminal_tree.unique()))
    trace_frames: list[pd.DataFrame] = []
    fit_summaries: list[dict[str, object]] = []
    lock_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    daily_metric_frames: list[pd.DataFrame] = []
    monthly_metric_frames: list[pd.DataFrame] = []
    fold_audits: list[dict[str, object]] = []

    for fold_index, heldout_tree in enumerate(tree_ids):
        print(f"STAGE5 fold {fold_index + 1}/{len(tree_ids)} held-out terminal tree {heldout_tree}", flush=True)
        training_indices = np.flatnonzero(stations.terminal_tree.to_numpy(int) != heldout_tree)
        heldout_indices = np.flatnonzero(stations.terminal_tree.to_numpy(int) == heldout_tree)
        training_reaches = reach_terminals != heldout_tree
        feature_mean = raw_features[training_reaches].mean(axis=0)
        feature_sd = raw_features[training_reaches].std(axis=0, ddof=0)
        if np.any(feature_sd <= 1.0e-12):
            raise RuntimeError(f"Degenerate training-only feature scaling in tree {heldout_tree}")
        standardized = (raw_features - feature_mean[None, :]) / feature_sd[None, :]

        global_objective = RegionalObjective(
            f"tree_{heldout_tree}_GLOBAL",
            spin_p,
            spin_pet,
            development_p,
            development_pet,
            observed[:, training_indices],
            support[training_indices],
            area_km2,
            stations.iloc[training_indices].terminal_tree.to_numpy(int),
            standardized,
            registered_prior_raw,
            spinup_tolerance=SPINUP_TOLERANCE,
            spinup_max_cycles=SPINUP_MAX_CYCLES,
        )
        global_result, global_summary = fit_objective(
            global_objective,
            [registered_prior_raw],
            [(-5.5, 5.5)] * 8,
            maxiter=40,
            maxfun=620,
        )
        global_theta = np.asarray(global_result.x, dtype=float)
        mpr_objective = RegionalObjective(
            f"tree_{heldout_tree}_MPR",
            spin_p,
            spin_pet,
            development_p,
            development_pet,
            observed[:, training_indices],
            support[training_indices],
            area_km2,
            stations.iloc[training_indices].terminal_tree.to_numpy(int),
            standardized,
            global_theta,
            spinup_tolerance=SPINUP_TOLERANCE,
            spinup_max_cycles=SPINUP_MAX_CYCLES,
        )
        mpr_start = np.concatenate([global_theta, np.zeros(8, dtype=float)])
        mpr_result, mpr_summary = fit_objective(
            mpr_objective,
            [mpr_start],
            [(-5.5, 5.5)] * 8 + [(-1.5, 1.5)] * 8,
            maxiter=45,
            maxfun=900,
        )
        mpr_theta = np.asarray(mpr_result.x, dtype=float)
        for frame, model in ((pd.DataFrame(global_objective.trace), "FOLD_GLOBAL_HBV"), (pd.DataFrame(mpr_objective.trace), "FOLD_MPR_LITE")):
            frame["heldout_terminal_tree"] = heldout_tree
            frame["model"] = model
            trace_frames.append(frame)
        for row in global_summary + mpr_summary:
            row["heldout_terminal_tree"] = heldout_tree
            fit_summaries.append(row)

        global_prediction = global_objective.predict(global_theta, collect_components=True)
        mpr_prediction = mpr_objective.predict(mpr_theta, collect_components=True)
        assert global_prediction.local_components_mm is not None and mpr_prediction.local_components_mm is not None
        global_site_components, global_route = route_candidate(
            f"tree_{heldout_tree}_GLOBAL",
            global_prediction.local_components_mm,
            area_km2,
            support,
            reach_ids,
            order,
            downstream,
            stations,
            observed,
            training_indices,
            dates,
            length_km,
            width_m,
            depth_m,
            fold_index * 10,
        )
        mpr_site_components, mpr_route = route_candidate(
            f"tree_{heldout_tree}_MPR",
            mpr_prediction.local_components_mm,
            area_km2,
            support,
            reach_ids,
            order,
            downstream,
            stations,
            observed,
            training_indices,
            dates,
            length_km,
            width_m,
            depth_m,
            fold_index * 10 + 1,
        )
        global_route["heldout_terminal_tree"] = heldout_tree
        global_route["model"] = "FOLD_GLOBAL_HBV"
        mpr_route["heldout_terminal_tree"] = heldout_tree
        mpr_route["model"] = "FOLD_MPR_LITE"
        route_rows.extend([global_route, mpr_route])

        held_stations = stations.iloc[heldout_indices].reset_index(drop=True)
        held_observed = observed[:, heldout_indices]
        held_global_components = global_site_components[:, heldout_indices, :] / 86400.0
        held_mpr_components = mpr_site_components[:, heldout_indices, :] / 86400.0
        held_global_q = held_global_components.sum(axis=2)
        held_mpr_q = held_mpr_components.sum(axis=2)
        global_metric = station_metrics("FOLD_GLOBAL_HBV", held_observed, held_global_q, held_stations)
        mpr_metric = station_metrics("FOLD_MPR_LITE", held_observed, held_mpr_q, held_stations)
        global_metric["heldout_terminal_tree"] = heldout_tree
        mpr_metric["heldout_terminal_tree"] = heldout_tree
        daily_metric_frames.extend([global_metric, mpr_metric])
        global_monthly = monthly_station_metrics("FOLD_GLOBAL_HBV", dates, held_observed, held_global_q, held_stations)
        mpr_monthly = monthly_station_metrics("FOLD_MPR_LITE", dates, held_observed, held_mpr_q, held_stations)
        global_monthly["heldout_terminal_tree"] = heldout_tree
        mpr_monthly["heldout_terminal_tree"] = heldout_tree
        monthly_metric_frames.extend([global_monthly, mpr_monthly])
        prediction = pd.DataFrame(
            {
                "date": np.repeat(dates.to_numpy(), len(held_stations)),
                "station_norm": np.tile(held_stations.station_norm.to_numpy(), len(dates)),
                "reach_id": np.tile(held_stations.reach_id.to_numpy(int), len(dates)),
                "heldout_terminal_tree": heldout_tree,
                "q_observed_m3_s": held_observed.reshape(-1),
                "q_fold_global_m3_s": held_global_q.reshape(-1),
                "q_fold_mpr_m3_s": held_mpr_q.reshape(-1),
            }
        )
        for model_key, components in (("global", held_global_components), ("mpr", held_mpr_components)):
            for component_index, component in enumerate(("q0", "q1", "q2")):
                prediction[f"{model_key}_{component}_m3_s"] = components[:, :, component_index].reshape(-1)
        prediction_frames.append(prediction)

        raw_map = mpr_prediction.raw_parameter_map
        physical_map = mpr_prediction.physical_parameter_map
        fold_boundary = bool(np.any(np.abs(raw_map) >= RAW_BOUNDARY_THRESHOLD))
        fold_audits.append(
            {
                "heldout_terminal_tree": heldout_tree,
                "training_station_count": int(len(training_indices)),
                "heldout_station_count": int(len(heldout_indices)),
                "global_objective": float(global_result.fun),
                "mpr_objective": float(mpr_result.fun),
                "global_optimizer_success": bool(global_result.success),
                "mpr_optimizer_success": bool(mpr_result.success),
                "global_mass_max_abs_error_mm": global_prediction.development_mass_max_abs_error_mm,
                "mpr_mass_max_abs_error_mm": mpr_prediction.development_mass_max_abs_error_mm,
                "global_spinup_cycles": int(global_prediction.spinup["cycles"]),
                "mpr_spinup_cycles": int(mpr_prediction.spinup["cycles"]),
                "mpr_raw_map_max_abs": float(np.max(np.abs(raw_map))),
                "mpr_boundary_confounded": fold_boundary,
                "target_tree_discharge_used_in_fit": False,
                "target_tree_signature_used_in_fit": False,
            }
        )
        for parameter_index, parameter in enumerate(PARAMETER_NAMES):
            lock_rows.append(
                {
                    "heldout_terminal_tree": heldout_tree,
                    "parameter": parameter,
                    "global_intercept_raw": float(global_theta[parameter_index]),
                    "mpr_intercept_raw": float(mpr_theta[parameter_index]),
                    "mpr_slope": float(mpr_theta[parameter_index + 8]),
                    "attribute_mean_training_only": float(feature_mean[parameter_index]),
                    "attribute_sd_training_only": float(feature_sd[parameter_index]),
                }
            )
        for reach_index, reach_id in enumerate(reach_ids):
            row: dict[str, object] = {
                "heldout_terminal_tree": heldout_tree,
                "reach_id": int(reach_id),
                "reach_is_in_heldout_tree": bool(reach_terminals[reach_index] == heldout_tree),
            }
            for parameter_index, parameter in enumerate(PARAMETER_NAMES):
                row[f"raw_{parameter}"] = float(raw_map[reach_index, parameter_index])
                row[parameter] = float(physical_map[reach_index, parameter_index])
            map_rows.append(row)

    traces = pd.concat(trace_frames, ignore_index=True)
    traces.to_parquet(OUT / "fold_optimization_trace.parquet", index=False)
    pd.DataFrame(fit_summaries).to_parquet(OUT / "fold_fit_summary.parquet", index=False)
    pd.DataFrame(lock_rows).to_parquet(OUT / "fold_parameter_and_scaling_lock.parquet", index=False)
    pd.DataFrame(map_rows).to_parquet(OUT / "fold_parameter_maps.parquet", index=False)
    pd.DataFrame(route_rows).to_parquet(OUT / "fold_route_selection.parquet", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(OUT / "complete_tree_oof_predictions.parquet", index=False)
    daily_metrics = pd.concat(daily_metric_frames, ignore_index=True)
    daily_metrics["temporal_scale"] = "daily"
    monthly_metrics = pd.concat(monthly_metric_frames, ignore_index=True)
    monthly_metrics["temporal_scale"] = "monthly"
    metrics = pd.concat([daily_metrics, monthly_metrics], ignore_index=True)
    metrics.to_parquet(OUT / "complete_tree_station_performance.parquet", index=False)
    fold_audit_frame = pd.DataFrame(fold_audits)
    fold_audit_frame.to_parquet(OUT / "fold_audit.parquet", index=False)

    global_daily = daily_metrics.loc[daily_metrics.candidate.eq("FOLD_GLOBAL_HBV"), ["station_norm", "terminal_tree", "log_RMSE", "PBIAS_pct"]]
    mpr_daily = daily_metrics.loc[daily_metrics.candidate.eq("FOLD_MPR_LITE"), ["station_norm", "log_RMSE", "PBIAS_pct"]]
    paired = global_daily.merge(mpr_daily, on="station_norm", suffixes=("_global", "_mpr"), validate="one_to_one")
    paired["delta_log_RMSE"] = paired.log_RMSE_mpr - paired.log_RMSE_global
    paired.to_parquet(OUT / "mpr_vs_global_paired_station_metrics.parquet", index=False)
    spatial_bootstrap = bootstrap_delta(paired, 1000)
    global_monthly_abs_pbias = float(monthly_metrics.loc[monthly_metrics.candidate.eq("FOLD_GLOBAL_HBV"), "PBIAS_pct"].abs().median())
    mpr_monthly_abs_pbias = float(monthly_metrics.loc[monthly_metrics.candidate.eq("FOLD_MPR_LITE"), "PBIAS_pct"].abs().median())
    boundary_folds = int(fold_audit_frame.mpr_boundary_confounded.sum())
    mpr_noninferior = bool(
        spatial_bootstrap["station_ci95_upper"] < SPATIAL_NONINFERIORITY_MARGIN
        and spatial_bootstrap["tree_ci95_upper"] < SPATIAL_NONINFERIORITY_MARGIN
        and mpr_monthly_abs_pbias - global_monthly_abs_pbias <= PBIAS_WORSENING_MARGIN_PP
        and boundary_folds < 2
    )
    selected_model = "MPR_LITE" if mpr_noninferior else "GLOBAL_HBV"

    summary_rows = []
    for scale, frame in (("daily", daily_metrics), ("monthly", monthly_metrics)):
        for candidate, group in frame.groupby("candidate"):
            prediction_column = "q_fold_global_m3_s" if candidate == "FOLD_GLOBAL_HBV" else "q_fold_mpr_m3_s"
            obs_all = predictions.q_observed_m3_s.to_numpy(float)
            pred_all = predictions[prediction_column].to_numpy(float)
            valid = np.isfinite(obs_all) & np.isfinite(pred_all)
            if scale == "daily":
                pooled_nse = nse(obs_all[valid], pred_all[valid])
            else:
                pooled_nse = float("nan")
            summary_rows.append(
                {
                    "candidate": candidate,
                    "temporal_scale": scale,
                    "scope": "station_median",
                    "NSE": float(group.NSE.median()),
                    "log_NSE": float(group.log_NSE.median()),
                    "KGE": float(group.KGE.median()),
                    "PBIAS_pct": float(group.PBIAS_pct.median()),
                    "absolute_PBIAS_pct": float(group.PBIAS_pct.abs().median()),
                    "log_RMSE": float(group.log_RMSE.median()),
                    "pooled_NSE": pooled_nse,
                    "stations": int(len(group)),
                }
            )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_parquet(OUT / "complete_tree_performance_summary.parquet", index=False)

    technical_pass = bool(
        len(predictions) == len(dates) * len(stations)
        and predictions[["q_fold_global_m3_s", "q_fold_mpr_m3_s"]].notna().all().all()
        and float(fold_audit_frame[["global_mass_max_abs_error_mm", "mpr_mass_max_abs_error_mm"]].max().max()) <= 1.0e-10
        and pd.DataFrame(route_rows).routing_mass_relative_error.max() <= 1.0e-12
    )
    decision = {
        "stage": "20260825_5",
        "status": "PASS_COMPLETE_TREE_SPATIAL_EVALUATION" if technical_pass else "STAGE5_TECHNICAL_GATE_FAILED",
        "selected_model_for_stage6": selected_model if technical_pass else None,
        "MPR_spatially_noninferior": mpr_noninferior,
        "spatial_noninferiority_margin_log_RMSE": SPATIAL_NONINFERIORITY_MARGIN,
        "spatial_bootstrap": spatial_bootstrap,
        "monthly_station_median_abs_PBIAS_global_pct": global_monthly_abs_pbias,
        "monthly_station_median_abs_PBIAS_mpr_pct": mpr_monthly_abs_pbias,
        "mpr_boundary_confounded_folds": boundary_folds,
        "observed_terminal_tree_folds": len(tree_ids),
        "target_tree_discharge_used_in_fit": False,
        "target_tree_signature_used_in_fit": False,
        "locked_2019_2022_discharge_read": False,
        "support_area_max_abs_delta_km2": support_delta,
        "authorized_successor": "20260825_6" if technical_pass else None,
    }
    write_json(REPORT / "stage5_decision.json", decision)
    write_json(
        REPORT / "complete_tree_spatial_parameter_lock.json",
        {
            "selected_model_for_stage6": selected_model,
            "fold_count": len(tree_ids),
            "parameter_count_global": 8,
            "parameter_count_mpr": 16,
            "attribute_edges": dict(zip(PARAMETER_NAMES, feature_columns)),
            "fold_audit": fold_audits,
            "route_selection": route_rows,
        },
    )
    report = f"""# 20260825_5 MPR-lite整棵河树空间评价

## 结论

状态：`{decision['status']}`；Stage 6候选为`{selected_model}`。MPR相对每折重拟合全局HBV的站点/河树log-RMSE差CI95上界分别为`{spatial_bootstrap['station_ci95_upper']:.5f}`和`{spatial_bootstrap['tree_ci95_upper']:.5f}`，预注册非劣界为`{SPATIAL_NONINFERIORITY_MARGIN:.3f}`。

月尺度站点绝对PBIAS中位数为：全局`{global_monthly_abs_pbias:.2f}%`，MPR`{mpr_monthly_abs_pbias:.2f}%`。MPR原始参数图触及边界的折数为`{boundary_folds}/8`。

每折均删除目标terminal tree全部流量，重新拟合全局HBV、训练树属性标准化和8条MPR属性边；目标树的流量及流量派生签名未进入拟合或路由选择。2019–2022锁定流量未读取。

这一阶段只检验总流量的空间推广。快、中间、慢响应分量是否被流量信息识别，留给`20260825_6`，不得由本阶段性能直接推断。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    program = json.loads(STAGE4_PROGRAM.read_text(encoding="utf-8"))
    program["stage_status"]["20260825_5"] = decision["status"]
    program["stage_status"]["20260825_6"] = "authorized_not_started" if technical_pass else "closed"
    (RUN / "program_manifest.json").write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    integrity_paths = [
        RUN / "experiment_contract.json",
        RUN / "scripts" / "regional_hbv_core.py",
        RUN / "scripts" / "run_stage5.py",
        OUT / "mpr_attributes_by_reach.parquet",
        OUT / "complete_tree_oof_predictions.parquet",
        OUT / "complete_tree_station_performance.parquet",
        OUT / "fold_parameter_and_scaling_lock.parquet",
        REPORT / "complete_tree_spatial_parameter_lock.json",
        REPORT / "stage5_decision.json",
    ]
    write_json(REPORT / "integrity.json", {str(path): sha256(path) for path in integrity_paths})
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    if not technical_pass:
        raise RuntimeError(decision)


if __name__ == "__main__":
    main()
