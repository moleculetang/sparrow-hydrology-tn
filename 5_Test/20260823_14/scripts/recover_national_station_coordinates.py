from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260823_14"
AUTHORITATIVE_SCRIPT = RUN / "scripts" / "build_authoritative_registry.py"
NATIONAL_DIR = ROOT / "0_reach_topology" / "data" / "raw" / "vector" / "stations" / "hydrostation"
LINES = ROOT / "0_reach_topology" / "results" / "vectors" / "reaches_topology.shp"
CATCHMENTS = ROOT / "0_reach_topology" / "results" / "vectors" / "reach_catchments.shp"
UNMATCHED = RUN / "reports" / "unmatched_discharge_station_names.csv"
MONTHLY = RUN / "outputs" / "all_monthly_discharge_after_exclusions.parquet"
LOCKED_COVERAGE = RUN / "outputs" / "final_user_locked_station_coverage.parquet"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
MAX_DISTANCE_M = 5000.0


def load_authoritative_module():
    spec = importlib.util.spec_from_file_location("authoritative_registry", AUTHORITATIVE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decode_gbk_mojibake(value: object) -> str:
    return str(value).encode("latin1").decode("gbk", errors="replace")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    module = load_authoritative_module()
    station_key = module.station_key
    classify_station = module.station_type

    unmatched = pd.read_csv(UNMATCHED, encoding="utf-8-sig")
    unmatched_keys = set(unmatched.station_norm.astype(str))
    monthly = pd.read_parquet(MONTHLY)
    locked = pd.read_parquet(LOCKED_COVERAGE)
    training_stations = set(locked.loc[locked.selected_for_model, "station_norm"].astype(str))
    training_reaches = set(locked.loc[locked.selected_for_model, "reach_id"].astype(int))

    shapefiles = list(NATIONAL_DIR.glob("*.shp"))
    if len(shapefiles) != 1:
        raise RuntimeError(f"Expected one national hydrostation shapefile, found {shapefiles}")
    national = gpd.read_file(shapefiles[0])
    columns = list(national.columns)
    if len(columns) < 5:
        raise RuntimeError(f"Unexpected national station schema: {columns}")
    name_col, lon_col, lat_col = columns[0], columns[3], columns[4]
    national["decoded_station_name"] = national[name_col].map(decode_gbk_mojibake)
    national["station_norm"] = national.decoded_station_name.map(station_key)
    national["lon"] = pd.to_numeric(national[lon_col], errors="coerce")
    national["lat"] = pd.to_numeric(national[lat_col], errors="coerce")
    national = national[
        national.station_norm.isin(unmatched_keys)
        & national.lon.between(-180, 180)
        & national.lat.between(-90, 90)
    ].copy()
    national = national.drop_duplicates(["station_norm", "lon", "lat"])

    lines = gpd.read_file(LINES)[["reach_id", "geometry"]].copy()
    catchments = gpd.read_file(CATCHMENTS)[["reach_id", "geometry"]].copy()
    if lines.crs != catchments.crs:
        catchments = catchments.to_crs(lines.crs)
    points = gpd.GeoDataFrame(
        national.drop(columns="geometry"),
        geometry=[Point(xy) for xy in zip(national.lon, national.lat)],
        crs="EPSG:4326",
    ).to_crs(lines.crs)

    rows: list[dict[str, object]] = []
    for point_row in points.itertuples(index=False):
        point = point_row.geometry
        line_distances = lines.geometry.distance(point)
        nearest_index = line_distances.idxmin()
        nearest_line_reach = int(lines.loc[nearest_index, "reach_id"])
        nearest_line_distance = float(line_distances.loc[nearest_index])
        containing = catchments[catchments.geometry.covers(point)]
        containing_reaches = sorted(containing.reach_id.astype(int).tolist())
        if nearest_line_distance <= MAX_DISTANCE_M:
            reach_id = nearest_line_reach
            method = "national_exact_name_nearest_reach_line"
            used = True
        elif containing_reaches:
            reach_id = int(containing_reaches[0])
            method = "national_exact_name_within_catchment"
            used = True
        else:
            catchment_distances = catchments.geometry.distance(point)
            catchment_index = catchment_distances.idxmin()
            reach_id = int(catchments.loc[catchment_index, "reach_id"])
            method = "national_exact_name_too_far"
            used = False
        rows.append({
            "station_norm": str(point_row.station_norm),
            "decoded_station_name": str(point_row.decoded_station_name),
            "lon": float(point_row.lon),
            "lat": float(point_row.lat),
            "reach_id": reach_id,
            "match_method": method,
            "nearest_line_distance_m": nearest_line_distance,
            "containing_reaches": ";".join(map(str, containing_reaches)),
            "used": bool(used),
        })
    candidates = pd.DataFrame(rows)
    candidates.to_parquet(OUT / "national_station_exact_name_candidate_audit.parquet", index=False)

    accepted_rows: list[pd.Series] = []
    ambiguity_rows: list[dict[str, object]] = []
    for station, part in candidates.groupby("station_norm", sort=True):
        valid = part[part.used].drop_duplicates(["lon", "lat", "reach_id"])
        if len(valid) == 1:
            accepted_rows.append(valid.iloc[0])
        else:
            ambiguity_rows.append({
                "station_norm": str(station),
                "national_exact_candidates": int(len(part)),
                "valid_topology_candidates": int(len(valid)),
                "valid_reaches": ";".join(map(str, sorted(valid.reach_id.astype(int).unique()))) if len(valid) else "",
                "status": "AMBIGUOUS" if len(valid) > 1 else "NO_CANDIDATE_WITHIN_TOPOLOGY",
            })
    accepted = pd.DataFrame(accepted_rows).reset_index(drop=True) if accepted_rows else pd.DataFrame(columns=candidates.columns)
    ambiguity = pd.DataFrame(ambiguity_rows)
    ambiguity.to_parquet(OUT / "national_station_coordinate_ambiguity_audit.parquet", index=False)

    station_flow = monthly[monthly.Q_obsv_cfs.notna()].groupby("station_norm", as_index=False).agg(
        usable_months=("Q_obsv_cfs", "size"),
        observed_years=("year", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        mean_Q_cfs=("Q_obsv_cfs", "mean"),
        median_Q_cfs=("Q_obsv_cfs", "median"),
    )
    accepted = accepted.merge(station_flow, on="station_norm", how="left", validate="one_to_one")
    accepted["station_type"] = accepted.station_norm.map(classify_station)
    accepted["station_seen_in_training"] = accepted.station_norm.isin(training_stations)
    accepted["reach_seen_in_training"] = accepted.reach_id.astype(int).isin(training_reaches)
    accepted["zero_history_candidate"] = (
        ~accepted.station_seen_in_training
        & ~accepted.reach_seen_in_training
        & accepted.station_type.eq("ordinary_river_gauge")
        & accepted.usable_months.ge(12)
    )
    accepted.to_parquet(OUT / "national_station_recovered_coverage.parquet", index=False)

    zero = accepted[accepted.zero_history_candidate].copy()
    if len(zero):
        # Preserve the user's one-Reach-one-station rule: use the larger mean
        # observed flow when more than one recovered station maps to a Reach.
        zero = zero.sort_values(
            ["reach_id", "mean_Q_cfs", "usable_months", "station_norm"],
            ascending=[True, False, False, True],
        ).drop_duplicates("reach_id", keep="first")
    zero.to_parquet(OUT / "supplemental_zero_history_station_coverage.parquet", index=False)
    zero_pairs = zero[["station_norm", "reach_id"]].copy()
    zero_obs = monthly[monthly.Q_obsv_cfs.notna() & monthly.Q_obsv_cfs.gt(0)].merge(
        zero_pairs, on="station_norm", how="inner", validate="many_to_one"
    )
    zero_obs.to_parquet(OUT / "supplemental_zero_history_station_month_observations.parquet", index=False)

    report = {
        "stage": "20260823_14",
        "status": "NATIONAL_COORDINATE_RECOVERY_AUDIT_COMPLETE",
        "authoritative_stations_without_existing_coordinate_name_match": int(len(unmatched_keys)),
        "national_exact_name_matched_stations": int(candidates.station_norm.nunique()) if len(candidates) else 0,
        "national_exact_candidate_rows": int(len(candidates)),
        "uniquely_resolved_to_topology": int(len(accepted)),
        "ambiguous_or_outside_topology": int(len(ambiguity)),
        "new_unseen_ordinary_reach_candidates": int(zero.reach_id.nunique()) if len(zero) else 0,
        "new_zero_history_stations": zero.station_norm.astype(str).tolist() if len(zero) else [],
        "new_zero_history_reaches": zero.reach_id.astype(int).tolist() if len(zero) else [],
        "coordinate_contract": "Exact normalized name in decoded national hydrostation layer; exactly one candidate within PRB Reach/catchment topology; no fuzzy name matching.",
    }
    (REPORT / "national_station_coordinate_recovery.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(zero):
        print(zero[[
            "station_norm", "reach_id", "usable_months", "observed_years", "mean_Q_cfs",
            "nearest_line_distance_m", "match_method",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
