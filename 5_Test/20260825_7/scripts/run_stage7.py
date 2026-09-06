"""Finalize GLOBAL_HBV_R0, run the locked retrospective check and build the TN bridge."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260825_7"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
STAGE3_SCRIPTS = ROOT / "5_Test" / "20260825_3" / "scripts"
STAGE5_SCRIPTS = ROOT / "5_Test" / "20260825_5" / "scripts"
sys.path[:0] = [str(STAGE3_SCRIPTS), str(STAGE5_SCRIPTS)]
from hydrology_core import (  # noqa: E402
    HBVParameters,
    load_topology,
    parameters_to_raw,
    route_instantaneous,
)
from regional_hbv_core import (  # noqa: E402
    PARAMETER_NAMES,
    RegionalObjective,
    _run_ordered_hbv,
    fit_objective,
    periodic_spinup_regional,
    theta_to_parameter_map,
)
from run_stage5 import build_support, kge, nse, station_metrics  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DEVELOPMENT_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
RETROSPECTIVE_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_locked_retrospective_2019_2022.parquet"
GAUGE_AUDIT = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
Q72_BRIDGE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
STATIC = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
STAGE6_DECISION = ROOT / "5_Test" / "20260825_6" / "reports" / "path_identifiability_decision.json"
STAGE6_PROGRAM = ROOT / "5_Test" / "20260825_6" / "program_manifest.json"

SPINUP_TOLERANCE = 1.0e-8
SPINUP_MAX_CYCLES = 500
ASIA_SHANGHAI = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def now_text() -> str:
    return datetime.now(ASIA_SHANGHAI).isoformat()


def monthly_matrices(
    dates: pd.DatetimeIndex,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[pd.MultiIndex, np.ndarray, np.ndarray]:
    codes = dates.to_period("M")
    months = codes.unique()
    obs_month = np.full((len(months), observed.shape[1]), np.nan, dtype=np.float64)
    pred_month = np.full_like(obs_month, np.nan)
    for month_index, month in enumerate(months):
        selected = np.asarray(codes == month)
        for station_index in range(observed.shape[1]):
            valid = selected & np.isfinite(observed[:, station_index]) & np.isfinite(predicted[:, station_index])
            if int(valid.sum()) >= 20:
                obs_month[month_index, station_index] = float(observed[valid, station_index].mean())
                pred_month[month_index, station_index] = float(predicted[valid, station_index].mean())
    index = pd.MultiIndex.from_arrays(
        [[period.year for period in months], [period.month for period in months]],
        names=["year", "month"],
    )
    return index, obs_month, pred_month


def summarize_performance(
    candidate: str,
    temporal_scale: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    station_frame: pd.DataFrame,
) -> pd.DataFrame:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    obs, pred = observed[valid], predicted[valid]
    rows = [
        {
            "candidate": candidate,
            "temporal_scale": temporal_scale,
            "scope": "pooled",
            "NSE": nse(obs, pred),
            "log_NSE": nse(np.log1p(obs), np.log1p(pred)),
            "KGE": kge(obs, pred),
            "PBIAS_pct": float(100.0 * (pred.sum() - obs.sum()) / obs.sum()),
            "absolute_PBIAS_pct": float("nan"),
            "log_RMSE": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2))),
            "stations": int(len(station_frame)),
        }
    ]
    for scope, reducer in (("station_median", "median"), ("station_mean", "mean")):
        rows.append(
            {
                "candidate": candidate,
                "temporal_scale": temporal_scale,
                "scope": scope,
                "NSE": float(getattr(station_frame.NSE, reducer)()),
                "log_NSE": float(getattr(station_frame.log_NSE, reducer)()),
                "KGE": float(getattr(station_frame.KGE, reducer)()),
                "PBIAS_pct": float(getattr(station_frame.PBIAS_pct, reducer)()),
                "absolute_PBIAS_pct": float(getattr(station_frame.PBIAS_pct.abs(), reducer)()),
                "log_RMSE": float(getattr(station_frame.log_RMSE, reducer)()),
                "stations": int(len(station_frame)),
            }
        )
    return pd.DataFrame(rows)


def bridge_frame(
    dates: pd.DatetimeIndex,
    reach_ids: np.ndarray,
    local_components_m3_s: np.ndarray,
    routed_components_m3_s: np.ndarray,
    length_m: np.ndarray,
    width: np.ndarray,
    depth: np.ndarray,
    width_p05: np.ndarray,
    depth_p05: np.ndarray,
    width_p95: np.ndarray,
    depth_p95: np.ndarray,
) -> pd.DataFrame:
    routed_total = routed_components_m3_s.sum(axis=2)
    q_safe = np.maximum(routed_total, 1.0e-6)
    tau = length_m[None, :] * width[None, :] * depth[None, :] / q_safe / 86400.0
    tau_low = length_m[None, :] * width_p05[None, :] * depth_p05[None, :] / q_safe / 86400.0
    tau_high = length_m[None, :] * width_p95[None, :] * depth_p95[None, :] / q_safe / 86400.0
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), len(reach_ids)),
            "reach_id": np.tile(reach_ids, len(dates)),
            "local_q0_m3_s": local_components_m3_s[:, :, 0].reshape(-1),
            "local_q1_m3_s": local_components_m3_s[:, :, 1].reshape(-1),
            "local_q2_m3_s": local_components_m3_s[:, :, 2].reshape(-1),
            "local_total_m3_s": local_components_m3_s.sum(axis=2).reshape(-1),
            "routed_q0_m3_s": routed_components_m3_s[:, :, 0].reshape(-1),
            "routed_q1_m3_s": routed_components_m3_s[:, :, 1].reshape(-1),
            "routed_q2_m3_s": routed_components_m3_s[:, :, 2].reshape(-1),
            "routed_quick_response_m3_s": routed_components_m3_s[:, :, :2].sum(axis=2).reshape(-1),
            "routed_slow_response_m3_s": routed_components_m3_s[:, :, 2].reshape(-1),
            "routed_total_m3_s": routed_total.reshape(-1),
            "bankfull_travel_time_day": tau.reshape(-1),
            "bankfull_travel_time_p05_geometry_day": tau_low.reshape(-1),
            "bankfull_travel_time_p95_geometry_day": tau_high.reshape(-1),
            "response_components_identified": False,
            "channel_storage_routing_applied": False,
        }
    )
    return frame


def monthly_bridge(
    dates: pd.DatetimeIndex,
    reach_ids: np.ndarray,
    local_components_m3_s: np.ndarray,
    routed_components_m3_s: np.ndarray,
    length_m: np.ndarray,
    width: np.ndarray,
    depth: np.ndarray,
    width_p05: np.ndarray,
    depth_p05: np.ndarray,
    width_p95: np.ndarray,
    depth_p95: np.ndarray,
) -> pd.DataFrame:
    codes = dates.to_period("M")
    rows: list[dict[str, object]] = []
    for period in codes.unique():
        selected = np.asarray(codes == period)
        seconds = int(selected.sum()) * 86400.0
        local_volume = (local_components_m3_s[selected] * 86400.0).sum(axis=0)
        routed_volume = (routed_components_m3_s[selected] * 86400.0).sum(axis=0)
        local_q = local_volume / seconds
        routed_q = routed_volume / seconds
        total = routed_q.sum(axis=1)
        q_safe = np.maximum(total, 1.0e-6)
        tau = length_m * width * depth / q_safe / 86400.0
        tau_low = length_m * width_p05 * depth_p05 / q_safe / 86400.0
        tau_high = length_m * width_p95 * depth_p95 / q_safe / 86400.0
        for reach_index, reach in enumerate(reach_ids):
            rows.append(
                {
                    "year": int(period.year),
                    "month": int(period.month),
                    "reach_id": int(reach),
                    "days_in_month": int(selected.sum()),
                    "local_q0_m3_s": float(local_q[reach_index, 0]),
                    "local_q1_m3_s": float(local_q[reach_index, 1]),
                    "local_q2_m3_s": float(local_q[reach_index, 2]),
                    "local_total_m3_s": float(local_q[reach_index].sum()),
                    "routed_q0_m3_s": float(routed_q[reach_index, 0]),
                    "routed_q1_m3_s": float(routed_q[reach_index, 1]),
                    "routed_q2_m3_s": float(routed_q[reach_index, 2]),
                    "routed_quick_response_m3_s": float(routed_q[reach_index, :2].sum()),
                    "routed_slow_response_m3_s": float(routed_q[reach_index, 2]),
                    "routed_total_m3_s": float(total[reach_index]),
                    "bankfull_travel_time_day": float(tau[reach_index]),
                    "bankfull_travel_time_p05_geometry_day": float(tau_low[reach_index]),
                    "bankfull_travel_time_p95_geometry_day": float(tau_high[reach_index]),
                    "response_components_identified": False,
                    "channel_storage_routing_applied": False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    stage6 = json.loads(STAGE6_DECISION.read_text(encoding="utf-8"))
    if stage6["identifiability_status"] != "TOTAL_FLOW_SUPPORTED_FAST_SLOW_NOT_IDENTIFIED":
        raise RuntimeError("Stage 7 contract expects the locked non-identifiability conclusion")

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGE_AUDIT)
    stations = gauges.loc[gauges.topology_representative].drop_duplicates("station_norm").sort_values(
        ["terminal_tree", "station_norm"]
    ).reset_index(drop=True)
    support = build_support(stations, reach_ids, order, downstream)
    area = pd.read_parquet(Q72_BRIDGE, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)

    forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    spin_dates = pd.date_range("2006-01-01", "2009-12-31", freq="D")
    development_dates = pd.date_range("2010-01-01", "2018-12-31", freq="D")
    full_dates = pd.date_range("2006-01-01", "2022-12-31", freq="D")
    spin = forcing.loc[forcing.date.dt.year.between(2006, 2009)]
    development = forcing.loc[forcing.date.dt.year.between(2010, 2018)]
    spin_p = spin.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=spin_dates, columns=reach_ids).to_numpy(float)
    spin_pet = spin.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=spin_dates, columns=reach_ids).to_numpy(float)
    development_p = development.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=development_dates, columns=reach_ids).to_numpy(float)
    development_pet = development.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=development_dates, columns=reach_ids).to_numpy(float)
    development_q = pd.read_parquet(DEVELOPMENT_Q)
    development_q["date"] = pd.to_datetime(development_q.date)
    observed_development = development_q.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=development_dates, columns=stations.station_norm).to_numpy(float)

    prior = HBVParameters(200.0, 2.0, 0.70, 2.0, 5.0, 1.4426950408889634, 8.048526073147784, 40.00733226320923)
    prior_raw = parameters_to_raw(prior)
    objective = RegionalObjective(
        "FINAL_GLOBAL_HBV",
        spin_p,
        spin_pet,
        development_p,
        development_pet,
        observed_development,
        support,
        area,
        stations.terminal_tree.to_numpy(int),
        np.zeros((230, 8), dtype=np.float64),
        prior_raw,
        spinup_tolerance=SPINUP_TOLERANCE,
        spinup_max_cycles=SPINUP_MAX_CYCLES,
    )
    offsets = [
        np.zeros(8),
        np.asarray([0.6, -0.5, 0.3, -0.4, 0.5, -0.3, 0.4, 0.5]),
        np.asarray([-0.6, 0.5, -0.3, 0.4, -0.5, 0.3, -0.4, -0.5]),
    ]
    starts = [np.clip(prior_raw + offset, -5.25, 5.25) for offset in offsets]
    result, fit_summary = fit_objective(
        objective, starts, [(-5.5, 5.5)] * 8, maxiter=45, maxfun=620
    )
    final_raw = np.asarray(result.x, dtype=float)
    final_prediction = objective.predict(final_raw, collect_components=True)
    _, physical_map = theta_to_parameter_map(final_raw, np.zeros((230, 8), dtype=np.float64))
    physical_parameters = dict(zip(PARAMETER_NAMES, map(float, physical_map[0])))
    pd.DataFrame(objective.trace).to_parquet(OUT / "final_global_hbv_optimization_trace.parquet", index=False)
    pd.DataFrame(fit_summary).to_parquet(OUT / "final_global_hbv_multistart_summary.parquet", index=False)

    mechanism_lock = {
        "written_at": now_text(),
        "structure": "GLOBAL_HBV_R0",
        "land_equations": "Raven 4.12 ordered HBV verified in 20260825_3",
        "routing": "instantaneous conserving topology accumulation",
        "station_specific_parameters": False,
        "MPR": False,
        "R1_channel_storage": False,
        "response_path_status": stage6["identifiability_status"],
        "TN_authorized_hydrology": "routed total flow only; hydraulic exposure diagnostic",
        "temperature_used": False,
    }
    write_json(REPORT / "development_mechanism_lock.json", mechanism_lock)
    parameter_lock = {
        "written_at": now_text(),
        "fit_period": "2010-2018",
        "objective": float(result.fun),
        "optimizer_reported_success": bool(result.success),
        "raw_parameters": dict(zip(PARAMETER_NAMES, map(float, final_raw))),
        "physical_parameters": physical_parameters,
        "spinup": final_prediction.spinup,
        "development_mass_max_abs_error_mm": final_prediction.development_mass_max_abs_error_mm,
        "parameter_boundary_confounded": bool(np.any(np.abs(final_raw) >= 5.45)),
        "retrospective_discharge_used": False,
    }
    write_json(REPORT / "full_development_parameter_lock.json", parameter_lock)
    lock_hashes = {
        "development_mechanism_lock_sha256": sha256(REPORT / "development_mechanism_lock.json"),
        "full_development_parameter_lock_sha256": sha256(REPORT / "full_development_parameter_lock.json"),
    }
    access_audit = {
        "formal_lock_completed_at": now_text(),
        **lock_hashes,
        "known_prelock_orchestrator_schema_access": True,
        "known_prelock_scope": "row count, columns, first five displayed rows; no aggregate metric",
        "formal_script_retrospective_read_count": 0,
        "retrospective_name": "locked post-development retrospective temporal check",
    }
    write_json(REPORT / "retrospective_access_audit.json", access_audit)

    full_p = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=full_dates, columns=reach_ids).to_numpy(float)
    full_pet = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=full_dates, columns=reach_ids).to_numpy(float)
    initial, full_spinup = periodic_spinup_regional(
        spin_p, spin_pet, physical_map, SPINUP_TOLERANCE, SPINUP_MAX_CYCLES
    )
    _, full_components_mm, full_mass_error = _run_ordered_hbv(
        full_p, full_pet, physical_map, initial, True
    )
    assert full_components_mm is not None
    local_components_m3_s = full_components_mm * area[None, :, None] * 1000.0 / 86400.0
    routed_components_m3_s = route_instantaneous(
        local_components_m3_s * 86400.0, reach_ids, order, downstream
    ) / 86400.0

    static = pd.read_parquet(STATIC, columns=["reach_id", "length_km"]).set_index("reach_id").reindex(reach_ids)
    geometry = pd.read_parquet(
        GEOMETRY,
        columns=[
            "reach_id",
            "bankfull_width_m",
            "bankfull_width_p05_m",
            "bankfull_width_p95_m",
            "bankfull_depth_m",
            "bankfull_depth_p05_m",
            "bankfull_depth_p95_m",
        ],
    ).set_index("reach_id").reindex(reach_ids)
    length_m = static.length_km.to_numpy(float) * 1000.0
    width = geometry.bankfull_width_m.to_numpy(float)
    depth = geometry.bankfull_depth_m.to_numpy(float)
    width_p05 = geometry.bankfull_width_p05_m.to_numpy(float)
    depth_p05 = geometry.bankfull_depth_p05_m.to_numpy(float)
    width_p95 = geometry.bankfull_width_p95_m.to_numpy(float)
    depth_p95 = geometry.bankfull_depth_p95_m.to_numpy(float)
    daily_bridge = bridge_frame(
        full_dates, reach_ids, local_components_m3_s, routed_components_m3_s,
        length_m, width, depth, width_p05, depth_p05, width_p95, depth_p95
    )
    daily_bridge.to_parquet(OUT / "tn_hydrology_bridge_daily_2006_2022.parquet", index=False)
    monthly = monthly_bridge(
        full_dates, reach_ids, local_components_m3_s, routed_components_m3_s,
        length_m, width, depth, width_p05, depth_p05, width_p95, depth_p95
    )
    monthly.to_parquet(OUT / "tn_hydrology_bridge_monthly_2006_2022.parquet", index=False)

    # Formal script read occurs only after both locks and their hashes exist.
    retrospective_read_started_at = now_text()
    retrospective = pd.read_parquet(RETROSPECTIVE_Q)
    retrospective_read_completed_at = now_text()
    access_audit.update(
        {
            "formal_script_retrospective_read_count": 1,
            "formal_script_read_started_at": retrospective_read_started_at,
            "formal_script_read_completed_at": retrospective_read_completed_at,
            "locks_existed_before_formal_read": True,
            "parameters_refit_after_formal_read": False,
        }
    )
    write_json(REPORT / "retrospective_access_audit.json", access_audit)

    retro_dates = pd.date_range("2019-01-01", "2022-12-31", freq="D")
    observed_retro = retrospective.assign(date=pd.to_datetime(retrospective.date)).pivot(
        index="date", columns="station_norm", values="q_m3_s"
    ).reindex(index=retro_dates, columns=stations.station_norm).to_numpy(float)
    retro_mask = full_dates.year >= 2019
    site_components = np.stack(
        [local_components_m3_s[:, :, component] @ support.T for component in range(3)], axis=2
    )
    predicted_retro = site_components[retro_mask].sum(axis=2)
    daily_station = station_metrics("GLOBAL_HBV_R0", observed_retro, predicted_retro, stations)
    daily_station["temporal_scale"] = "daily"
    month_index, monthly_obs, monthly_pred = monthly_matrices(
        retro_dates, observed_retro, predicted_retro
    )
    monthly_station = station_metrics("GLOBAL_HBV_R0", monthly_obs, monthly_pred, stations)
    monthly_station["temporal_scale"] = "monthly"

    q72 = pd.read_parquet(
        Q72_BRIDGE,
        columns=["reach_id", "year", "month", "local_total_m3_s", "routed_total_m3_s"],
    )
    q72 = q72.loc[q72.year.between(2019, 2022)]
    q72_matrix = np.full_like(monthly_obs, np.nan)
    q72_key = {(int(row.reach_id), int(row.year), int(row.month)): (float(row.local_total_m3_s), float(row.routed_total_m3_s)) for row in q72.itertuples()}
    for month_position, (year, month_value) in enumerate(month_index):
        for station_index, station in enumerate(stations.itertuples()):
            local_q, routed_q = q72_key[(int(station.reach_id), int(year), int(month_value))]
            q72_matrix[month_position, station_index] = routed_q - (1.0 - float(station.downstream_fraction_on_reach)) * local_q
    q72_station = station_metrics("Q72_MONTHLY_REFERENCE", monthly_obs, q72_matrix, stations)
    q72_station["temporal_scale"] = "monthly"
    station_performance = pd.concat([daily_station, monthly_station, q72_station], ignore_index=True)
    station_performance.to_parquet(OUT / "locked_retrospective_station_performance.parquet", index=False)
    summary = pd.concat(
        [
            summarize_performance("GLOBAL_HBV_R0", "daily", observed_retro, predicted_retro, daily_station),
            summarize_performance("GLOBAL_HBV_R0", "monthly", monthly_obs, monthly_pred, monthly_station),
            summarize_performance("Q72_MONTHLY_REFERENCE", "monthly", monthly_obs, q72_matrix, q72_station),
        ],
        ignore_index=True,
    )
    summary.to_parquet(OUT / "locked_retrospective_performance_summary.parquet", index=False)
    prediction_frame = pd.DataFrame(
        {
            "date": np.repeat(retro_dates.to_numpy(), len(stations)),
            "station_norm": np.tile(stations.station_norm.to_numpy(), len(retro_dates)),
            "reach_id": np.tile(stations.reach_id.to_numpy(int), len(retro_dates)),
            "terminal_tree": np.tile(stations.terminal_tree.to_numpy(int), len(retro_dates)),
            "q_observed_m3_s": observed_retro.reshape(-1),
            "q_global_hbv_r0_m3_s": predicted_retro.reshape(-1),
            "q0_m3_s": (site_components[retro_mask, :, 0]).reshape(-1),
            "q1_m3_s": (site_components[retro_mask, :, 1]).reshape(-1),
            "q2_m3_s": (site_components[retro_mask, :, 2]).reshape(-1),
        }
    )
    prediction_frame.to_parquet(OUT / "locked_retrospective_predictions.parquet", index=False)

    daily_pooled = summary.loc[(summary.candidate.eq("GLOBAL_HBV_R0")) & summary.temporal_scale.eq("daily") & summary.scope.eq("pooled")].iloc[0]
    daily_median = summary.loc[(summary.candidate.eq("GLOBAL_HBV_R0")) & summary.temporal_scale.eq("daily") & summary.scope.eq("station_median")].iloc[0]
    monthly_pooled = summary.loc[(summary.candidate.eq("GLOBAL_HBV_R0")) & summary.temporal_scale.eq("monthly") & summary.scope.eq("pooled")].iloc[0]
    monthly_median = summary.loc[(summary.candidate.eq("GLOBAL_HBV_R0")) & summary.temporal_scale.eq("monthly") & summary.scope.eq("station_median")].iloc[0]
    retrospective_pass = bool(
        daily_pooled.NSE >= 0.75
        and daily_median.NSE >= 0.50
        and monthly_pooled.NSE >= 0.85
        and monthly_median.NSE >= 0.65
        and monthly_median.absolute_PBIAS_pct <= 20.0
    )
    terminal_indices = [int(reach) - 1 for reach in reach_ids if int(reach) not in downstream]
    terminal_volume = float((routed_components_m3_s[:, terminal_indices, :] * 86400.0).sum())
    local_volume = float((local_components_m3_s * 86400.0).sum())
    route_mass_relative_error = abs(terminal_volume - local_volume) / max(local_volume, 1.0)
    daily_monthly_check = (
        daily_bridge.assign(year=daily_bridge.date.dt.year, month=daily_bridge.date.dt.month)
        .groupby(["year", "month", "reach_id"], as_index=False)
        .agg(reconstructed_routed_total_m3_s=("routed_total_m3_s", "mean"))
    )
    closure_check = monthly[["year", "month", "reach_id", "routed_total_m3_s"]].merge(
        daily_monthly_check,
        on=["year", "month", "reach_id"],
        how="left",
        validate="one_to_one",
    )
    monthly_daily_closure = float(
        np.max(
            np.abs(
                closure_check.routed_total_m3_s
                - closure_check.reconstructed_routed_total_m3_s
            )
        )
    )
    technical_pass = bool(
        final_prediction.development_mass_max_abs_error_mm <= 1.0e-10
        and full_mass_error <= 1.0e-10
        and route_mass_relative_error <= 1.0e-12
        and monthly_daily_closure <= 1.0e-10
        and len(daily_bridge) == len(full_dates) * 230
        and len(monthly) == 17 * 12 * 230
    )
    total_flow_status = (
        "GLOBAL_HBV_R0_TOTAL_FLOW_PROMOTED_FOR_TN_HYDROLOGY"
        if retrospective_pass and technical_pass
        else "GLOBAL_HBV_R0_NOT_PROMOTED_RETROSPECTIVE_OR_TECHNICAL_GATE_FAILED"
    )
    decision = {
        "stage": "20260825_7",
        "status": "PASS_FINAL_LOCK_AND_RETROSPECTIVE_COMPLETED" if technical_pass else "STAGE7_TECHNICAL_GATE_FAILED",
        "total_flow_status": total_flow_status,
        "retrospective_total_flow_gate_passed": retrospective_pass,
        "daily_pooled_NSE": float(daily_pooled.NSE),
        "daily_station_median_NSE": float(daily_median.NSE),
        "monthly_pooled_NSE": float(monthly_pooled.NSE),
        "monthly_station_median_NSE": float(monthly_median.NSE),
        "monthly_station_median_absolute_PBIAS_pct": float(monthly_median.absolute_PBIAS_pct),
        "path_identifiability_status": stage6["identifiability_status"],
        "TN_primary_authorized_field": "routed_total_m3_s" if retrospective_pass and technical_pass else None,
        "response_component_role": "DIAGNOSTIC_ONLY_NOT_IDENTIFIED",
        "hydraulic_exposure_role": "DIAGNOSTIC_ONLY_R1_NOT_SELECTED",
        "daily_bridge_rows": int(len(daily_bridge)),
        "monthly_bridge_rows": int(len(monthly)),
        "land_mass_max_abs_error_mm": float(full_mass_error),
        "routing_terminal_mass_relative_error": route_mass_relative_error,
        "monthly_from_daily_max_abs_closure_m3_s": monthly_daily_closure,
        "temperature_used": False,
        "known_prelock_schema_access_disclosed": True,
        "formal_script_retrospective_read_count": 1,
        "program_complete": True,
    }
    write_json(REPORT / "stage7_decision.json", decision)
    write_json(
        REPORT / "tn_hydrology_bridge_contract.json",
        {
            "primary_authorized_field": decision["TN_primary_authorized_field"],
            "daily_file": str(OUT / "tn_hydrology_bridge_daily_2006_2022.parquet"),
            "monthly_file": str(OUT / "tn_hydrology_bridge_monthly_2006_2022.parquet"),
            "monthly_aggregation": "daily volume sum divided by exact month seconds",
            "response_component_fields": [
                "routed_q0_m3_s",
                "routed_q1_m3_s",
                "routed_q2_m3_s",
                "routed_quick_response_m3_s",
                "routed_slow_response_m3_s",
            ],
            "response_component_authorization": "DIAGNOSTIC_ONLY_NOT_IDENTIFIED",
            "hydraulic_exposure_authorization": "DIAGNOSTIC_ONLY_R1_NOT_SELECTED",
            "temperature_used": False,
        },
    )
    report = f"""# 20260825_7 最终水文锁、回顾检查与TN桥

## 最终裁决

总流量状态：`{total_flow_status}`。2019–2022锁定回顾中，日尺度pooled NSE=`{daily_pooled.NSE:.3f}`、站点中位NSE=`{daily_median.NSE:.3f}`；月尺度pooled NSE=`{monthly_pooled.NSE:.3f}`、站点中位NSE=`{monthly_median.NSE:.3f}`，站点绝对PBIAS中位数=`{monthly_median.absolute_PBIAS_pct:.2f}%`。

最终模型是230 Reach统一的Raven方程一致GLOBAL_HBV_R0，无站点Gauge校正、无MPR、无R1河道存储、无水库方程。日桥`{len(daily_bridge):,}`行，月桥`{len(monthly):,}`行；月流量严格由日体积累加。陆面最大质量误差`{full_mass_error:.3e} mm`，终端路由质量相对误差`{route_mass_relative_error:.3e}`。

Stage 6的`TOTAL_FLOW_SUPPORTED_FAST_SLOW_NOT_IDENTIFIED`结论保持不变。因此TN只正式授权`routed_total_m3_s`；Q0/Q1/Q2及quick/slow仅为内部诊断，不能解释为真实快慢路径。bankfull旅行时间为候选自身Q与Andreadis宽深得到的水力暴露诊断，R1并未升级。温度未进入本轮水文或TN反应。

2019–2022只称锁定后回顾检查。正式脚本在机制锁和参数锁写入后读取一次；同时报告已披露脚本运行前的schema/前5行检查，因此不声称首次读取或独立外部验证。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    program = json.loads(STAGE6_PROGRAM.read_text(encoding="utf-8"))
    program["stage_status"]["20260825_7"] = decision["status"]
    program["status"] = "complete"
    (RUN / "program_manifest.json").write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    integrity_paths = [
        RUN / "experiment_contract.json",
        RUN / "scripts" / "run_stage7.py",
        REPORT / "development_mechanism_lock.json",
        REPORT / "full_development_parameter_lock.json",
        REPORT / "retrospective_access_audit.json",
        OUT / "tn_hydrology_bridge_daily_2006_2022.parquet",
        OUT / "tn_hydrology_bridge_monthly_2006_2022.parquet",
        OUT / "locked_retrospective_performance_summary.parquet",
        REPORT / "tn_hydrology_bridge_contract.json",
        REPORT / "stage7_decision.json",
    ]
    write_json(REPORT / "integrity.json", {str(path): sha256(path) for path in integrity_paths})
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    if not technical_pass:
        raise RuntimeError(decision)


if __name__ == "__main__":
    main()
