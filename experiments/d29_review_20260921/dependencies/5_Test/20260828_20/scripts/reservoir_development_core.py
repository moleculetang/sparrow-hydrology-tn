"""Shared development-only reservoir simulation and scoring utilities."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
S13 = ROOT / "5_Test" / "20260828_13"
S14 = ROOT / "5_Test" / "20260828_14"
S15 = ROOT / "5_Test" / "20260828_15"
PARENT = ROOT / "5_Test" / "20260828_9" / "outputs" / "canonical_reach_daily_2006_2024.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
FIREWALL = ROOT / "5_Test" / "20260828_1" / "inputs"
ROLE_AUDIT = ROOT / "5_Test" / "20260828_19" / "outputs" / "registered_91_reservoir_observation_roles.parquet"

for folder in [S15 / "scripts", S14 / "scripts"]:
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from reservoir_network_router import ReservoirRuntime, build_reach_graph, route_network_steps  # noqa: E402
from reservoir_operator import ReservoirRule, ReservoirState  # noqa: E402


SECONDS_PER_DAY = 86400.0


def semicolon_ints(value: object) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(";") if part.strip())


def load_static_context(end_date: str = "2018-12-31") -> dict[str, object]:
    priority = pd.read_parquet(S13 / "outputs" / "priority_reservoir_topology_audit.parquet")
    priority = priority[
        (priority["priority_tier"] == "priority_13_artificial_reservoirs")
        & priority["operator_authorized"].fillna(False)
    ].sort_values("reservoir_entity_id").reset_index(drop=True)
    if len(priority) != 13:
        raise RuntimeError("Priority reservoir registry must contain 13 authorized entities")
    inventory = pd.read_parquet(S13 / "outputs" / "reservoir_entity_inventory.parquet")
    evidence = pd.read_parquet(S13 / "outputs" / "literature_prior_registry.parquet")
    topology = pd.read_csv(TOPOLOGY)
    parent = pd.read_parquet(
        PARENT,
        columns=[
            "date", "reach_id", "local_fast_response_m3_s",
            "local_slow_response_m3_s", "routed_total_m3_s"
        ],
        filters=[("date", "<=", pd.Timestamp(end_date))],
    )
    parent["date"] = pd.to_datetime(parent["date"])
    parent = parent.sort_values(["date", "reach_id"], kind="mergesort").reset_index(drop=True)
    dates = pd.Index(parent["date"].drop_duplicates())
    reach_ids = sorted(parent["reach_id"].unique().astype(int).tolist())
    if reach_ids != list(range(1, 231)) or len(parent) != len(dates) * 230:
        raise RuntimeError("Parent development grid is incomplete")
    local_fast_rate = parent["local_fast_response_m3_s"].to_numpy(float).reshape(len(dates), 230)
    local_slow_rate = parent["local_slow_response_m3_s"].to_numpy(float).reshape(len(dates), 230)
    routed_total_rate = parent["routed_total_m3_s"].to_numpy(float).reshape(len(dates), 230)

    station_roles = pd.read_parquet(ROLE_AUDIT)
    stations = station_roles[station_roles["general_likelihood_eligible"].fillna(False)].copy()
    stations = stations.sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    if len(stations) != 89:
        raise RuntimeError(f"Expected 89 legal development gauges, found {len(stations)}")
    observations = pd.concat(
        [
            pd.read_parquet(FIREWALL / "monthly_observations_2010_2016.parquet"),
            pd.read_parquet(FIREWALL / "monthly_observations_2017_2018.parquet"),
        ],
        ignore_index=True,
    )
    observations = observations[observations["station_norm"].isin(stations["station_norm"])].copy()
    year_column = "year"
    month_column = "month"
    if year_column not in observations or month_column not in observations:
        if "date" not in observations:
            raise RuntimeError("Monthly observations lack year/month and date")
        observations["date"] = pd.to_datetime(observations["date"])
        observations["year"] = observations["date"].dt.year
        observations["month"] = observations["date"].dt.month
    if observations["year"].max() > 2018:
        raise RuntimeError("Held-out observations leaked into development context")
    q_candidates = ["q_m3_s", "q_m3s", "observed_m3_s"]
    q_column = next((column for column in q_candidates if column in observations), None)
    if q_column is None:
        raise RuntimeError(
            f"Monthly observations lack a recognized discharge field: {list(observations.columns)}"
        )
    obs_monthly = observations.pivot_table(
        index=["year", "month"], columns="station_norm", values=q_column, aggfunc="mean"
    ).reindex(columns=stations["station_norm"])
    if obs_monthly.index.duplicated().any():
        raise RuntimeError("Monthly observation keys are duplicated")
    return {
        "priority": priority,
        "inventory": inventory,
        "evidence": evidence,
        "topology": topology,
        "dates": dates,
        "reach_ids": reach_ids,
        "local_fast_rate": local_fast_rate,
        "local_slow_rate": local_slow_rate,
        "routed_total_rate": routed_total_rate,
        "stations": stations,
        "obs_monthly": obs_monthly,
    }


def select_capacities(
    priority: pd.DataFrame,
    inventory: pd.DataFrame,
    evidence: pd.DataFrame,
    annual_inflow: dict[str, float],
) -> tuple[dict[str, float], dict[str, str], float]:
    capacities: dict[str, float] = {}
    sources: dict[str, str] = {}
    eligible = evidence[
        evidence["eligible_for_development_prior"].fillna(False)
        & (evidence["model_parameter"] == "capacity_m3")
        & evidence["value_numeric"].notna()
    ].copy()
    eligible["grade_rank"] = eligible["evidence_grade"].astype(str).str[0].map(
        {"A": 0, "B": 1, "C": 2, "D": 3}
    ).fillna(9)
    for entity, group in eligible.groupby("reservoir_entity_id"):
        row = group.sort_values(["grade_rank", "publication_date"], kind="mergesort").iloc[0]
        capacities[str(entity)] = float(row["value_numeric"]) * 1.0e6
        sources[str(entity)] = f"{row['evidence_record_id']}:{row['source_doi_or_url']}"
    inventory_by_id = inventory.set_index("reservoir_entity_id")
    for row in priority.itertuples(index=False):
        entity = str(row.reservoir_entity_id)
        if entity in capacities:
            continue
        value = inventory_by_id.loc[entity, "capacity_million_m3_grand"]
        if pd.notna(value):
            capacities[entity] = float(value) * 1.0e6
            sources[entity] = "GRanD locked inventory capacity"
    ratios = [
        capacities[e] / annual_inflow[e]
        for e in annual_inflow
        if e in capacities and annual_inflow[e] > 0
    ]
    hierarchical_ratio = float(np.median(ratios))
    for row in priority.itertuples(index=False):
        entity = str(row.reservoir_entity_id)
        if entity not in capacities:
            capacities[entity] = hierarchical_ratio * annual_inflow[entity]
            sources[entity] = "extension-only hierarchical capacity prior"
    return capacities, sources, hierarchical_ratio


def captured_rate(
    row: object,
    routed_total: np.ndarray,
    local_total: np.ndarray,
    reach_index: dict[int, int],
) -> np.ndarray:
    controls = semicolon_ints(row.registered_operator_inflow_reaches)
    if len(controls) > 1:
        result = np.zeros(routed_total.shape[0], dtype=float)
        for reach in controls:
            result += routed_total[:, reach_index[reach]]
        return result
    position = reach_index[controls[0]]
    fraction = float(row.local_capture_fraction)
    return routed_total[:, position] - (1.0 - fraction) * local_total[:, position]


def five_month_wet_window(monthly_climatology: pd.Series) -> tuple[int, ...]:
    values = monthly_climatology.reindex(range(1, 13)).to_numpy(float)
    best_start = max(
        range(12),
        key=lambda start: float(sum(values[(start + offset) % 12] for offset in range(5))),
    )
    return tuple(((best_start + offset) % 12) + 1 for offset in range(5))


def prepare_reservoir_metadata(context: dict[str, object]) -> dict[str, dict[str, object]]:
    priority = context["priority"]
    dates = pd.to_datetime(context["dates"])
    local_total = context["local_fast_rate"] + context["local_slow_rate"]
    routed_total = context["routed_total_rate"]
    reach_index = {reach: index for index, reach in enumerate(context["reach_ids"])}
    fit_mask = np.asarray((dates.year >= 2010) & (dates.year <= 2015))
    metadata: dict[str, dict[str, object]] = {}
    annual_inflow: dict[str, float] = {}
    captured_by_entity: dict[str, np.ndarray] = {}
    for row in priority.itertuples(index=False):
        entity = str(row.reservoir_entity_id)
        capture = captured_rate(row, routed_total, local_total, reach_index)
        captured_by_entity[entity] = capture
        annual_inflow[entity] = float(capture[fit_mask].mean() * SECONDS_PER_DAY * 365.25)
    capacities, capacity_sources, hierarchy_ratio = select_capacities(
        priority, context["inventory"], context["evidence"], annual_inflow
    )
    evidence = context["evidence"]
    for row in priority.itertuples(index=False):
        entity = str(row.reservoir_entity_id)
        capture = captured_by_entity[entity]
        capture_frame = pd.DataFrame({"date": dates[fit_mask], "q": capture[fit_mask]})
        capture_frame["month"] = capture_frame["date"].dt.month
        wet_months = five_month_wet_window(capture_frame.groupby("month")["q"].mean())
        scale = float(row.domain_storage_scale)
        capacity = capacities[entity] * scale
        entity_evidence = evidence[evidence["reservoir_entity_id"] == entity]
        regulation = "generic"
        regulation_rows = entity_evidence[entity_evidence["fact_name"] == "regulation_class"]
        if not regulation_rows.empty:
            regulation = str(regulation_rows.iloc[0]["value_text"])
        dead_fraction_r2 = 0.05
        target_fraction_r2 = 0.60
        if entity == "GRAND_5736":
            dead_fraction_r2 = 4310.0 / 13900.0
            target_fraction_r2 = 10800.0 / 13900.0
        metadata[entity] = {
            "name": row.reservoir_name_zh,
            "control_reaches": semicolon_ints(row.registered_operator_inflow_reaches),
            "outflow_reach": int(row.registered_unique_outflow_reach),
            "active_start": pd.Timestamp(row.operator_start),
            "local_capture_fraction": float(row.local_capture_fraction),
            "capacity_m3": capacity,
            "capacity_source": capacity_sources[entity],
            "annual_inflow_m3": annual_inflow[entity],
            "capacity_annual_inflow_ratio": capacity / annual_inflow[entity],
            "hydrologic_wet_months": wet_months,
            "regulation_class": regulation,
            "dead_fraction_r2": dead_fraction_r2,
            "target_fraction_r2": target_fraction_r2,
            "hierarchical_capacity_ratio": hierarchy_ratio,
        }
    return metadata


def build_runtimes(
    metadata: dict[str, dict[str, object]],
    layer: str,
    tau_day: float,
    inflow_response: float,
    drawdown_fraction: float,
) -> list[ReservoirRuntime]:
    if layer not in {"R1", "R2"}:
        raise ValueError(f"Unsupported reservoir layer {layer}")
    runtimes: list[ReservoirRuntime] = []
    east_river_entities = {"GRAND_5736", "GRAND_5758", "LOCAL_FENGSHUBA"}
    for entity, item in metadata.items():
        capacity = float(item["capacity_m3"])
        mean_step = float(item["annual_inflow_m3"]) / 365.25
        ratio = float(item["capacity_annual_inflow_ratio"])
        ratio_multiplier = float(np.clip(0.5 + 2.0 * ratio, 0.5, 3.0))
        class_multiplier = 1.0
        if layer == "R2":
            regulation = str(item["regulation_class"])
            if "多年" in regulation:
                class_multiplier = 2.0
            elif "不完全年" in regulation:
                class_multiplier = 1.25
        recovery_days = tau_day * ratio_multiplier * class_multiplier
        dead_fraction = float(item["dead_fraction_r2"]) if layer == "R2" else 0.05
        target_fraction = float(item["target_fraction_r2"]) if layer == "R2" else 0.60
        dead = dead_fraction * capacity
        base_target = max(target_fraction * capacity, dead)
        base_rule = ReservoirRule(
            capacity_m3=capacity,
            dead_storage_m3=dead,
            target_storage_m3=base_target,
            baseline_release_m3_step=mean_step,
            climatological_inflow_m3_step=mean_step,
            inflow_response=inflow_response,
            storage_recovery_per_step=1.0 / recovery_days,
            minimum_release_m3_step=0.02 * mean_step,
            maximum_controlled_release_m3_step=max(5.0 * mean_step, capacity / 15.0),
            enabled=True,
        )
        wet_months = tuple(int(value) for value in item["hydrologic_wet_months"])

        def provider(
            step: int,
            timestamp: pd.Timestamp | None,
            rule: ReservoirRule,
            *,
            entity_id: str = entity,
            hydrologic_months: tuple[int, ...] = wet_months,
            cap: float = capacity,
            floor: float = dead,
            base: float = base_target,
        ) -> ReservoirRule:
            if timestamp is None:
                return rule
            months = hydrologic_months
            if layer == "R2" and entity_id in east_river_entities and timestamp >= pd.Timestamp("2015-05-01"):
                months = (4, 5, 6, 7, 8, 9)
            target = base - (drawdown_fraction * cap if timestamp.month in months else 0.0)
            return replace(rule, target_storage_m3=max(floor, min(target, cap)))

        runtimes.append(
            ReservoirRuntime(
                entity_id=entity,
                control_reach_ids=tuple(item["control_reaches"]),
                outflow_reach_id=int(item["outflow_reach"]),
                base_rule=base_rule,
                state=ReservoirState.empty(),
                active_start=pd.Timestamp(item["active_start"]),
                local_capture_fraction=float(item["local_capture_fraction"]),
                rule_provider=provider,
            )
        )
    return runtimes


def simulate_candidate(
    context: dict[str, object],
    metadata: dict[str, dict[str, object]],
    layer: str,
    tau_day: float,
    inflow_response: float,
    drawdown_fraction: float,
    keep_diagnostics: bool = False,
) -> tuple[np.ndarray, tuple[dict[str, object], ...], list[ReservoirRuntime]]:
    graph = build_reach_graph(context["topology"], context["reach_ids"])
    runtimes = build_runtimes(metadata, layer, tau_day, inflow_response, drawdown_fraction)
    routed = route_network_steps(
        context["local_fast_rate"] * SECONDS_PER_DAY,
        context["local_slow_rate"] * SECONDS_PER_DAY,
        graph,
        runtimes,
        dates=context["dates"],
        values_are_volumes_per_step=True,
        keep_reservoir_diagnostics=keep_diagnostics,
    )
    total_rate = (routed.routed_fast + routed.routed_slow + routed.routed_direct) / SECONDS_PER_DAY
    return total_rate, routed.reservoir_diagnostics, runtimes


def station_monthly_prediction(context: dict[str, object], routed_total_rate: np.ndarray) -> pd.DataFrame:
    stations = context["stations"]
    reach_index = {reach: index for index, reach in enumerate(context["reach_ids"])}
    positions = np.array([reach_index[int(value)] for value in stations["reach_id"]], dtype=int)
    fractions = stations["downstream_fraction_on_reach"].to_numpy(float)
    local_total = context["local_fast_rate"] + context["local_slow_rate"]
    prediction = routed_total_rate[:, positions] - (1.0 - fractions[None, :]) * local_total[:, positions]
    if (prediction < -1e-9).any() or not np.isfinite(prediction).all():
        raise RuntimeError("Station prediction is negative or non-finite")
    prediction = np.maximum(prediction, 0.0)
    frame = pd.DataFrame(prediction, index=pd.to_datetime(context["dates"]), columns=stations["station_norm"])
    return frame.groupby([frame.index.year, frame.index.month]).mean().rename_axis(index=["year", "month"])


def metric_summary(obs: pd.DataFrame, pred: pd.DataFrame, years: tuple[int, int]) -> dict[str, float]:
    selector = (obs.index.get_level_values("year") >= years[0]) & (obs.index.get_level_values("year") <= years[1])
    obs_values = obs.loc[selector].to_numpy(float)
    pred_values = pred.reindex(obs.loc[selector].index).to_numpy(float)
    valid = np.isfinite(obs_values) & np.isfinite(pred_values)
    if valid.sum() == 0:
        raise RuntimeError("No valid observations in metric period")
    log_error = np.log1p(pred_values[valid]) - np.log1p(obs_values[valid])
    pooled_log_rmse = float(np.sqrt(np.mean(log_error**2)))
    pooled_nse = float(1.0 - np.sum((pred_values[valid] - obs_values[valid]) ** 2) / np.sum((obs_values[valid] - np.mean(obs_values[valid])) ** 2))
    station_nse: list[float] = []
    station_abs_pbias: list[float] = []
    station_log_rmse: list[float] = []
    for index in range(obs_values.shape[1]):
        ok = valid[:, index]
        if ok.sum() < 12:
            continue
        observed = obs_values[ok, index]
        predicted = pred_values[ok, index]
        denominator = np.sum((observed - observed.mean()) ** 2)
        if denominator > 0:
            station_nse.append(float(1.0 - np.sum((predicted - observed) ** 2) / denominator))
        if observed.sum() > 0:
            station_abs_pbias.append(float(abs(100.0 * (predicted.sum() - observed.sum()) / observed.sum())))
        station_log_rmse.append(float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(observed)) ** 2))))
    return {
        "pooled_log_RMSE": pooled_log_rmse,
        "pooled_NSE": pooled_nse,
        "station_median_NSE": float(np.median(station_nse)),
        "station_mean_NSE": float(np.mean(station_nse)),
        "station_median_absolute_PBIAS_pct": float(np.median(station_abs_pbias)),
        "station_median_log_RMSE": float(np.median(station_log_rmse)),
        "valid_station_months": int(valid.sum()),
        "scored_stations": len(station_log_rmse),
    }
