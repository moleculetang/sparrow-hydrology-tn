"""Audit development-only station coverage for reservoir fitting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test" / "20260828_19"
FIREWALL = ROOT / "5_Test" / "20260828_1" / "inputs"
PRIORITY = ROOT / "5_Test" / "20260828_13" / "outputs" / "priority_reservoir_topology_audit.parquet"
FULL_CANDIDATES = ROOT / "5_Test" / "20260828_13" / "outputs" / "reservoir_discharge_station_coverage.parquet"


def main() -> None:
    outputs = STAGE / "outputs"
    reports = STAGE / "reports"
    outputs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    stations = pd.read_parquet(FIREWALL / "station_registry_91.parquet")
    daily_early = pd.read_parquet(FIREWALL / "daily_observations_2010_2016.parquet")
    daily_late = pd.read_parquet(FIREWALL / "daily_observations_2017_2018.parquet")
    monthly_early = pd.read_parquet(FIREWALL / "monthly_observations_2010_2016.parquet")
    monthly_late = pd.read_parquet(FIREWALL / "monthly_observations_2017_2018.parquet")
    priority = pd.read_parquet(PRIORITY)
    priority = priority[
        priority["priority_tier"] == "priority_13_artificial_reservoirs"
    ].copy()
    full_candidates = pd.read_parquet(FULL_CANDIDATES)

    if len(stations) != 91 or stations["station_norm"].duplicated().any():
        raise RuntimeError("Registered 91-station cohort changed")
    obs = pd.concat([daily_early, daily_late], ignore_index=True)
    station_key = "station_norm"
    if station_key not in obs.columns:
        raise RuntimeError(f"Development observations have no {station_key} field")
    date_column = "date" if "date" in obs.columns else None
    if date_column is None:
        raise RuntimeError("Development observations have no date field")
    obs[date_column] = pd.to_datetime(obs[date_column])
    if obs[date_column].dt.year.max() > 2018:
        raise RuntimeError("Held-out discharge leaked into the observation-role audit")

    counts = (
        obs.groupby(station_key)
        .agg(
            development_daily_rows=(date_column, "size"),
            development_start=(date_column, "min"),
            development_end=(date_column, "max"),
        )
        .reset_index()
    )
    candidates = stations.merge(counts, how="left", on=station_key)
    outflow_lookup = (
        priority[["reservoir_entity_id", "reservoir_name_zh", "registered_unique_outflow_reach"]]
        .rename(columns={"registered_unique_outflow_reach": "reach_id"})
        .assign(reach_id=lambda frame: frame["reach_id"].astype(int))
    )
    candidates = candidates.merge(outflow_lookup, how="left", on="reach_id")
    control_rows: list[dict[str, object]] = []
    for row in priority.itertuples(index=False):
        for value in str(row.registered_operator_inflow_reaches).split(";"):
            value = value.strip()
            if value:
                control_rows.append(
                    {
                        "control_reach_id": int(value),
                        "control_reservoir_entity_id": row.reservoir_entity_id,
                        "control_reservoir_name_zh": row.reservoir_name_zh,
                    }
                )
    control_lookup = pd.DataFrame(control_rows)
    candidates = candidates.merge(
        control_lookup, how="left", left_on="reach_id", right_on="control_reach_id"
    )
    candidates["reservoir_proximity_role"] = candidates["reservoir_entity_id"].notna().map(
        {True: "same_reach_as_registered_reservoir_outflow", False: "general_network_gauge"}
    )
    candidates["formal_role"] = candidates["reservoir_entity_id"].notna().map(
        {
            True: "immediate_downstream_candidate_semantics_not_yet_confirmed",
            False: "general_downstream_flow_likelihood_candidate",
        }
    )
    candidates["release_gate_eligible"] = False
    candidates["release_gate_reason"] = (
        "No station is promoted from Reach coincidence alone; total-release semantics and "
        "distance/local-inflow contamination require an independent record."
    )
    candidates["general_likelihood_eligible"] = candidates[
        "control_reservoir_entity_id"
    ].isna()
    candidates["general_likelihood_reason"] = candidates[
        "general_likelihood_eligible"
    ].map(
        {
            True: "not located on a registered reservoir control Reach",
            False: (
                "excluded because the current edge operator replaces the control-Reach outlet "
                "with reservoir release and cannot represent an upstream within-Reach gauge"
            ),
        }
    )
    candidates.to_parquet(outputs / "registered_91_reservoir_observation_roles.parquet", index=False)

    full_candidates = full_candidates.copy()
    full_candidates["formal_role"] = "unclassified_name_candidate_not_likelihood_eligible"
    full_candidates["release_gate_eligible"] = False
    full_candidates.to_parquet(outputs / "full_registry_reservoir_label_candidates.parquet", index=False)

    same_reach = candidates[candidates["reservoir_entity_id"].notna()].copy()
    control_reach = candidates[candidates["control_reservoir_entity_id"].notna()].copy()
    report = {
        "stage": "20260828_19",
        "status": "OBSERVATION_ROLE_AUDIT_COMPLETE_RELEASE_GATE_NOT_AUTHORIZED",
        "schemas": {
            "station_registry_91": list(stations.columns),
            "daily_observations_2010_2016": list(daily_early.columns),
            "daily_observations_2017_2018": list(daily_late.columns),
            "monthly_observations_2010_2016": list(monthly_early.columns),
            "monthly_observations_2017_2018": list(monthly_late.columns),
            "full_reservoir_name_candidates": list(full_candidates.columns),
        },
        "counts": {
            "registered_stations": len(stations),
            "development_observation_rows": len(obs),
            "development_observation_stations": int(obs[station_key].nunique()),
            "priority_reservoirs": len(priority),
            "registered_stations_on_reservoir_outflow_reaches": len(same_reach),
            "reservoirs_with_same_reach_registered_station": int(same_reach["reservoir_entity_id"].nunique()),
            "registered_stations_on_reservoir_control_reaches": len(control_reach),
            "general_likelihood_eligible_stations": int(candidates["general_likelihood_eligible"].sum()),
            "name_based_full_registry_candidates": len(full_candidates),
            "release_gate_eligible_stations": 0,
        },
        "same_reach_candidates": same_reach[
            [column for column in [
                "station_norm", "reach_id", "reservoir_entity_id", "reservoir_name_zh",
                "development_daily_rows", "development_start", "development_end"
            ] if column in same_reach.columns]
        ].astype(object).where(pd.notna(same_reach), None).to_dict(orient="records"),
        "control_reach_exclusions": control_reach[
            [column for column in [
                "station_norm", "reach_id", "control_reservoir_entity_id",
                "control_reservoir_name_zh", "downstream_fraction_on_reach",
                "development_daily_rows", "development_start", "development_end"
            ] if column in control_reach.columns]
        ].astype(object).where(pd.notna(control_reach), None).to_dict(orient="records"),
        "decision": (
            "The general 91-station development likelihood can be audited separately, but the "
            "registered reservoir-specific release gate is not authorized by station/Reach "
            "coincidence alone. Independent total-release semantics are still required."
        ),
        "held_out_observations_read": False,
        "tn_observations_read": False,
        "parameters_fitted": False,
    }
    (reports / "reservoir_observation_role_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
