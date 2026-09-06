from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from common import RUN, load_config, write_json
from runtime_guard import assert_sparrow_runtime


def norm_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace("\u3000", " ")
    return re.sub(r"\s+", "", text).strip()


def load_coordinates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"站点", "经度", "纬度"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Coordinate file must contain {sorted(required)}; got {frame.columns.tolist()}")
    out = frame.loc[:, ["站点", "经度", "纬度"]].copy()
    out.columns = ["station", "lon", "lat"]
    out["station"] = out["station"].astype(str).str.strip()
    out["station_key"] = out["station"].map(norm_name)
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    duplicate = out[out["station_key"].duplicated(keep=False)].sort_values("station_key")
    if not duplicate.empty:
        conflicting = duplicate.groupby("station_key")[["lon", "lat"]].nunique(dropna=False).max(axis=1)
        if (conflicting > 1).any():
            raise ValueError("Conflicting station coordinates found: " + ", ".join(conflicting[conflicting > 1].index))
        out = out.drop_duplicates("station_key", keep="first")
    return out


def spatial_match(coords: pd.DataFrame, reaches_path: Path, catchments_path: Path) -> pd.DataFrame:
    reaches = gpd.read_file(reaches_path).loc[:, ["reach_id", "geometry"]].copy()
    catchments = gpd.read_file(catchments_path).loc[:, ["reach_id", "geometry"]].copy()
    if reaches.crs is None or catchments.crs is None:
        raise ValueError("Baseline reach/catchment CRS is missing")
    if reaches.crs != catchments.crs:
        catchments = catchments.to_crs(reaches.crs)
    valid_mask = coords["lon"].between(-180, 180) & coords["lat"].between(-90, 90)
    valid = coords[valid_mask].copy()
    points = gpd.GeoDataFrame(valid, geometry=[Point(xy) for xy in zip(valid.lon, valid.lat)], crs="EPSG:4326").to_crs(reaches.crs)
    records: list[dict[str, object]] = []
    for row in points.itertuples(index=False):
        containing = catchments[catchments.geometry.covers(row.geometry)]
        catchment_reach = int(containing.iloc[0].reach_id) if len(containing) == 1 else np.nan
        distances = reaches.geometry.distance(row.geometry)
        nearest_pos = int(distances.to_numpy().argmin())
        nearest_reach = int(reaches.iloc[nearest_pos].reach_id)
        distance_m = float(distances.iloc[nearest_pos])
        records.append(
            {
                "station": row.station,
                "station_key": row.station_key,
                "lon": float(row.lon),
                "lat": float(row.lat),
                "catchment_reach": catchment_reach,
                "nearest_reach": nearest_reach,
                "distance_m": distance_m,
                "catchment_match": bool(pd.notna(catchment_reach) and int(catchment_reach) == nearest_reach),
            }
        )
    valid_result = pd.DataFrame.from_records(records)
    invalid = coords.loc[~valid_mask, ["station", "station_key", "lon", "lat"]].copy()
    if not invalid.empty:
        invalid["catchment_reach"] = np.nan
        invalid["nearest_reach"] = np.nan
        invalid["distance_m"] = np.nan
        invalid["catchment_match"] = False
        valid_result = pd.concat([valid_result, invalid], ignore_index=True, sort=False)
    return valid_result


def collect_tn(csv_root: Path, start_year: int, end_year: int) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for path in sorted(csv_root.glob("*.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {"年份", "月份", "总氮"}
        if not required.issubset(frame.columns):
            continue
        out = frame.loc[:, ["年份", "月份", "总氮"]].copy()
        out.columns = ["year", "month", "tn_mg_l"]
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out["month"] = pd.to_numeric(out["month"], errors="coerce")
        out["tn_mg_l"] = pd.to_numeric(out["tn_mg_l"], errors="coerce")
        out = out[
            out["year"].between(start_year, end_year)
            & out["month"].between(1, 12)
            & np.isfinite(out["tn_mg_l"])
            & (out["tn_mg_l"] > 0)
        ].copy()
        if out.empty:
            continue
        out["year"] = out["year"].astype(int)
        out["month"] = out["month"].astype(int)
        out["station"] = path.stem
        out["station_key"] = norm_name(path.stem)
        records.append(out)
    if not records:
        raise RuntimeError("No positive TN records found in preprocessed station CSV files")
    all_tn = pd.concat(records, ignore_index=True)
    all_tn = (
        all_tn.groupby(["station", "station_key", "year", "month"], as_index=False)["tn_mg_l"]
        .median()
        .sort_values(["station", "year", "month"])
        .reset_index(drop=True)
    )
    return all_tn


def main() -> None:
    runtime = assert_sparrow_runtime()
    config = load_config()
    quality_root = Path(config["water_quality_root"])
    snapshot = RUN / "inputs" / "baseline_snapshot" / "inputs" / "spatial_corrected"
    coords = load_coordinates(quality_root / "站点经纬度.csv")
    matched = spatial_match(coords, snapshot / "reaches_topology.shp", snapshot / "reach_catchments.shp")
    max_distance = float(config["strict_station_max_distance_m"])
    matched["match_class"] = np.where(
        matched["catchment_match"] & (matched["distance_m"] <= max_distance), "strict", "non_strict"
    )
    tn = collect_tn(quality_root / "csv", int(config["tn_start_year"]), int(config["tn_end_year"]))
    observations = tn.merge(
        matched[["station_key", "nearest_reach", "catchment_reach", "distance_m", "catchment_match", "match_class"]],
        on="station_key",
        how="left",
        validate="many_to_one",
    )
    observations = observations.rename(columns={"nearest_reach": "reach_id"})
    station_counts = observations.groupby("station_key").size().rename("tn_months")
    observations = observations.merge(station_counts, on="station_key", how="left", validate="many_to_one")
    is_strict = (
        observations["match_class"].eq("strict")
        & observations["tn_months"].ge(int(config["strict_station_min_months"]))
    )
    strict = observations[is_strict].copy()
    processed = RUN / "inputs" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    matched = matched.merge(station_counts, on="station_key", how="left")
    matched["tn_months"] = matched["tn_months"].fillna(0).astype(int)
    matched.to_csv(processed / "station_reach_match.csv", index=False, encoding="utf-8-sig")
    observations.to_parquet(processed / "tn_observations_all.parquet", index=False)
    strict.to_parquet(processed / "tn_observations_strict.parquet", index=False)
    tn_missing_reach = observations["reach_id"].isna()
    summary = {
        "runtime": runtime,
        "coordinate_table_stations": int(len(coords)),
        "coordinate_stations_with_valid_lon_lat": int(matched["lon"].notna().sum()),
        "coordinate_stations_without_valid_lon_lat": int(matched["lon"].isna().sum()),
        "csv_station_files": int(len(list((quality_root / "csv").glob("*.csv")))),
        "positive_tn_records_2016_2022": int(len(tn)),
        "stations_with_tn": int(tn["station_key"].nunique()),
        "strict_stations": int(strict["station_key"].nunique()),
        "strict_reaches": int(strict["reach_id"].nunique()),
        "strict_records": int(len(strict)),
        "tn_records_without_spatial_match": int(tn_missing_reach.sum()),
        "tn_stations_without_spatial_match": int(observations.loc[tn_missing_reach, "station_key"].nunique()),
        "criteria": {
            "nearest_reach_distance_m_max": max_distance,
            "nearest_reach_equals_catchment": True,
            "minimum_positive_tn_months": int(config["strict_station_min_months"]),
        },
    }
    write_json(RUN / "reports" / "tn_observation_gate.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
