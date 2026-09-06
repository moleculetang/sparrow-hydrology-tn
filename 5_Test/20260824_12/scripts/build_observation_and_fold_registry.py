"""Freeze the TN observation domain, station position operator and evaluation folds."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from stage12_common import OUT, REPORTS, ROOT, require_sparrow, sha256, write_json


TN = ROOT / "1_Inputs" / "WaterQualityData" / "model_ready" / "tn_station_month_all.parquet"
DOMAIN = ROOT / "5_Test" / "20260820_19" / "outputs" / "observation_domain_registry.parquet"
REACH_LINES = ROOT / "0_reach_topology" / "results" / "vectors" / "reaches_topology.shp"


def station_positions(stations: pd.DataFrame) -> pd.DataFrame:
    lines = gpd.read_file(REACH_LINES).set_index("reach_id")
    points = gpd.GeoDataFrame(
        stations.copy(),
        geometry=[Point(float(x), float(y)) for x, y in zip(stations.lon, stations.lat)],
        crs="EPSG:4326",
    ).to_crs(lines.crs)
    rows = []
    for row in points.itertuples(index=False):
        reach = int(row.reach_id)
        line = lines.loc[reach].geometry
        fraction = float(line.project(row.geometry) / line.length) if line.length > 0 else np.nan
        distance = float(line.distance(row.geometry))
        rows.append({
            "station_key": str(row.station_key), "reach_id": reach,
            "downstream_fraction_on_reach": fraction,
            "station_to_assigned_reach_distance_m": distance,
        })
    out = pd.DataFrame(rows)
    if out.downstream_fraction_on_reach.isna().any() or not out.downstream_fraction_on_reach.between(0, 1).all():
        raise RuntimeError("invalid station fractions")
    return out


def build_observations() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    domain = pd.read_parquet(DOMAIN)
    domain = domain.loc[domain.primary_river_domain].copy()
    raw = pd.read_parquet(TN)
    raw = raw.loc[
        raw.strict & raw.year.between(2016, 2024) & raw.tn_mg_l.notna() & raw.tn_mg_l.ge(0)
    ].copy()
    raw["reach_id"] = raw.reach_id.astype(int)
    observations = raw.merge(
        domain[["station_key", "reach_id", "terminal_tree_id", "observation_domain", "hydraulic_primary_gate_role"]],
        on=["station_key", "reach_id"], how="inner", validate="many_to_one",
    )
    if observations.duplicated(["station_key", "year", "month"]).any():
        duplicate = observations.loc[observations.duplicated(["station_key", "year", "month"], keep=False)]
        raise RuntimeError(f"duplicate station-month TN: {len(duplicate)}")
    station = observations.groupby(["station_key", "reach_id"], as_index=False).agg(
        station=("station", "first"), lon=("lon", "first"), lat=("lat", "first"),
        terminal_tree_id=("terminal_tree_id", "first"),
        first_tn_year=("year", "min"), last_tn_year=("year", "max"),
        tn_month_count=("tn_mg_l", "size"),
    )
    positions = station_positions(station[["station_key", "reach_id", "lon", "lat"]])
    station = station.merge(positions, on=["station_key", "reach_id"], validate="one_to_one")
    observations = observations.merge(
        station[["station_key", "reach_id", "downstream_fraction_on_reach", "station_to_assigned_reach_distance_m"]],
        on=["station_key", "reach_id"], validate="many_to_one",
    )
    observations["evaluation_period"] = np.where(observations.year <= 2020, "historical_2016_2020", "development_2021_2024")
    observations = observations[[
        "station", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l",
        "lon", "lat", "downstream_fraction_on_reach", "station_to_assigned_reach_distance_m",
        "evaluation_period", "observation_domain", "hydraulic_primary_gate_role",
    ]].sort_values(["station_key", "year", "month"]).reset_index(drop=True)

    pre = observations.loc[observations.year.between(2016, 2020)]
    recent = observations.loc[observations.year.between(2021, 2024)]
    pre_stations, pre_reaches = set(pre.station_key), set(pre.reach_id)
    first_2021 = station.loc[station.first_tn_year.eq(2021)].copy()
    first_2021["new_station_relative_to_2016_2020"] = ~first_2021.station_key.isin(pre_stations)
    first_2021["new_reach_relative_to_2016_2020"] = ~first_2021.reach_id.isin(pre_reaches)
    natural = first_2021.loc[first_2021.new_station_relative_to_2016_2020].copy()

    counts = {
        "observation_rows_2016_2024": int(len(observations)),
        "primary_station_count": int(observations.station_key.nunique()),
        "primary_reach_count": int(observations.reach_id.nunique()),
        "primary_tree_count": int(observations.terminal_tree_id.nunique()),
        "development_rows_2021_2024": int(len(recent)),
        "development_station_count": int(recent.station_key.nunique()),
        "development_reach_count": int(recent.reach_id.nunique()),
        "first_observed_station_2021_count": int(len(natural)),
        "first_observed_2021_new_reach_count": int(natural.loc[natural.new_reach_relative_to_2016_2020, "reach_id"].nunique()),
        "same_reach_multi_station_reach_count": int((station.groupby("reach_id").station_key.nunique() > 1).sum()),
        "maximum_stations_on_one_reach": int(station.groupby("reach_id").station_key.nunique().max()),
    }
    return observations, station, {"counts": counts, "natural_expansion": natural}


def fold_registry(observations: pd.DataFrame) -> pd.DataFrame:
    folds = [("T1", 2021, 2021, 2022), ("T2", 2021, 2022, 2023), ("T3", 2021, 2023, 2024)]
    rows = []
    for fold_id, train_start, train_end, evaluation_year in folds:
        rows.append({
            "fold_id": fold_id, "holdout_type": "TEMPORAL", "holdout_id": "ALL",
            "train_start_year": train_start, "train_end_year": train_end,
            "evaluation_year": evaluation_year,
        })
        evaluation = observations.loc[observations.year.eq(evaluation_year)]
        for reach in sorted(evaluation.reach_id.unique()):
            rows.append({
                "fold_id": f"{fold_id}_LORO_R{int(reach):03d}", "holdout_type": "REACH",
                "holdout_id": str(int(reach)), "train_start_year": train_start,
                "train_end_year": train_end, "evaluation_year": evaluation_year,
            })
        for tree in sorted(evaluation.terminal_tree_id.unique()):
            rows.append({
                "fold_id": f"{fold_id}_LOTO_T{int(tree):03d}", "holdout_type": "TREE",
                "holdout_id": str(int(tree)), "train_start_year": train_start,
                "train_end_year": train_end, "evaluation_year": evaluation_year,
            })
    rows.append({
        "fold_id": "NATURAL_EXPANSION_2021", "holdout_type": "FIRST_OBSERVED_2021",
        "holdout_id": "REGISTERED_2021_NEW_STATIONS", "train_start_year": 2016,
        "train_end_year": 2020, "evaluation_year": 2021,
    })
    return pd.DataFrame(rows)


def main() -> None:
    require_sparrow()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    observations, stations, audit_data = build_observations()
    folds = fold_registry(observations)
    natural = audit_data.pop("natural_expansion")
    observations.to_parquet(OUT / "tn_observations_primary_2016_2024.parquet", index=False)
    stations.to_parquet(OUT / "tn_station_spatial_registry.parquet", index=False)
    folds.to_parquet(OUT / "tn_evaluation_fold_registry.parquet", index=False)
    natural.to_parquet(OUT / "tn_natural_expansion_2021_registry.parquet", index=False)
    audit = {
        "stage": "20260824_12", "status": "PASS_TN_OBSERVATION_AND_FOLD_LOCK",
        **audit_data,
        "fold_counts": folds.holdout_type.value_counts().to_dict(),
        "contracts": {
            "same_reach_stations_held_out_together": True,
            "all_candidate_parameters_refit_inside_each_fold": True,
            "2021_2024_is_development": True,
            "2016_2020_final_use": "historical backcast only",
            "station_identity_in_prediction_equation": False,
        },
        "source_hashes": {"TN": sha256(TN), "domain": sha256(DOMAIN), "reach_lines": sha256(REACH_LINES)},
    }
    write_json(REPORTS / "tn_observation_and_fold_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
