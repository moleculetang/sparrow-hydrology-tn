from __future__ import annotations

import json
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from shapely.ops import transform
from pyproj import Transformer


RUN = Path(__file__).resolve().parents[1]
REPORTS = RUN / "reports"
RAW = REPORTS / "external_raw"
GPKG = RUN / "outputs" / "vectors" / "full_topology_audit.gpkg"
COUNTERFACTUAL = REPORTS / "candidate_edge_counterfactual_summary.csv"

USER_AGENT = "SPARROW-PRB-coordinate-topology-audit/1.0 (research read-only)"
NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
OVERPASS = "https://overpass-api.de/api/interpreter"
TIMEOUT = 90


TERMINAL_PAIRS = [(14, 19), (64, 59), (132, 149), (180, 168), (199, 196)]
CATCHMENT_REVIEW_IDS = [23, 47, 146]


def endpoint(line: LineString, last: bool = True) -> Point:
    return Point(line.coords[-1] if last else line.coords[0])


def interpolate_points(line: LineString, spacing: float = 500.0) -> list[Point]:
    n = max(2, int(math.ceil(line.length / spacing)) + 1)
    return [line.interpolate(float(d)) for d in np.linspace(0.0, line.length, n)]


def request_json(method: str, url: str, **kwargs) -> dict:
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network retry is recorded in raw metadata by final response timestamp
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"External geography request failed after retries: {url}: {last_error}")


def reverse_geocode(case_id: str, point_wgs84: Point) -> dict:
    data = request_json(
        "GET",
        NOMINATIM,
        params={
            "format": "jsonv2",
            "lat": f"{point_wgs84.y:.8f}",
            "lon": f"{point_wgs84.x:.8f}",
            "zoom": 12,
            "addressdetails": 1,
        },
    )
    (RAW / f"{case_id}_nominatim.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data


def overpass_waterways(case_id: str, query: str) -> dict:
    data = request_json("POST", OVERPASS, data={"data": query})
    (RAW / f"{case_id}_overpass.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data


def ways_to_frame(data: dict, to_local: Transformer) -> gpd.GeoDataFrame:
    rows: list[dict[str, object]] = []
    for item in data.get("elements", []):
        geom = item.get("geometry") or []
        coords = [(float(x["lon"]), float(x["lat"])) for x in geom]
        if len(coords) < 2:
            continue
        tags = item.get("tags", {})
        line_wgs = LineString(coords)
        line_local = transform(to_local.transform, line_wgs)
        rows.append(
            {
                "osm_way_id": int(item["id"]),
                "waterway": tags.get("waterway", ""),
                "name": tags.get("name", ""),
                "name_zh": tags.get("name:zh", ""),
                "name_en": tags.get("name:en", ""),
                "intermittent": tags.get("intermittent", ""),
                "geometry": line_local,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(columns=["osm_way_id", "waterway", "name", "name_zh", "name_en", "intermittent", "geometry"], geometry="geometry")
    return gpd.GeoDataFrame(rows, geometry="geometry")


def nearest_distance(points: list[Point], ways: gpd.GeoDataFrame) -> np.ndarray:
    if ways.empty:
        return np.full(len(points), np.nan)
    union = ways.geometry.union_all()
    return np.array([p.distance(union) for p in points], dtype=float)


def local_segment_near_point(line: LineString, point: Point, radius_m: float) -> LineString:
    clipped = line.intersection(point.buffer(radius_m))
    if clipped.is_empty:
        return line
    if clipped.geom_type == "LineString":
        return clipped
    if clipped.geom_type == "MultiLineString":
        return max(clipped.geoms, key=lambda g: g.length)
    return line


def auxiliary_names(ways: gpd.GeoDataFrame, point: Point, radius_m: float = 1200.0) -> str:
    if ways.empty:
        return ""
    near = ways[ways.geometry.distance(point) <= radius_m]
    labels: list[str] = []
    for row in near.sort_values("osm_way_id").itertuples(index=False):
        label = row.name_zh or row.name or row.name_en or f"unnamed_way_{row.osm_way_id}"
        labels.append(f"{row.osm_way_id}:{label}")
    return "|".join(dict.fromkeys(labels))


def locality_text(geo: dict) -> str:
    address = geo.get("address", {})
    keys = ["village", "town", "city_district", "county", "city", "region", "state", "country"]
    return ", ".join(str(address[k]) for k in keys if address.get(k))


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    reaches = gpd.read_file(GPKG, layer="reaches_risk", engine="pyogrio")
    reaches["reach_id"] = pd.to_numeric(reaches.reach_id, errors="raise").astype(int)
    reaches = reaches.set_index("reach_id")
    local_crs = reaches.crs
    to_wgs = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    cf = pd.read_csv(COUNTERFACTUAL, encoding="utf-8-sig").set_index("candidate_source_reach")

    rows: list[dict[str, object]] = []
    for source, receiver in TERMINAL_PAIRS:
        case_id = f"terminal_{source}_to_{receiver}"
        source_line = reaches.loc[source].geometry
        receiver_line = reaches.loc[receiver].geometry
        join_local = endpoint(source_line, True)
        join_wgs = transform(to_wgs.transform, join_local)
        lat, lon = join_wgs.y, join_wgs.x
        query = (
            f"[out:json][timeout:60];"
            f"way(around:8000,{lat:.8f},{lon:.8f})[waterway];"
            "out tags geom;"
        )
        geo = reverse_geocode(case_id, join_wgs)
        ways = ways_to_frame(overpass_waterways(case_id, query), to_local)
        source_near = local_segment_near_point(source_line, join_local, 10000.0)
        receiver_near = local_segment_near_point(receiver_line, join_local, 10000.0)
        source_dist = nearest_distance(interpolate_points(source_near, 400.0), ways)
        receiver_dist = nearest_distance(interpolate_points(receiver_near, 400.0), ways)
        join_distance = float(join_local.distance(ways.geometry.union_all())) if not ways.empty else np.nan
        near_ways = ways[ways.geometry.distance(join_local) <= 500.0] if not ways.empty else ways
        source_med = float(np.nanmedian(source_dist)) if np.isfinite(source_dist).any() else np.nan
        receiver_med = float(np.nanmedian(receiver_dist)) if np.isfinite(receiver_dist).any() else np.nan
        if join_distance <= 250 and source_med <= 500 and receiver_med <= 500 and len(near_ways) >= 2:
            decision = "COORDINATE_NETWORK_STRONGLY_SUPPORTS_JOIN"
        elif join_distance <= 500 and source_med <= 1000 and receiver_med <= 1000:
            decision = "COORDINATE_NETWORK_SUPPORTS_CONTINUITY_JOIN_NEEDS_OFFICIAL_CONFIRMATION"
        else:
            decision = "EXTERNAL_OPEN_NETWORK_INSUFFICIENT_OR_MISMATCHED"
        rows.append(
            {
                "case_id": case_id,
                "case_type": "terminal_geometry_table_contradiction",
                "source_reach": source,
                "receiver_reach": receiver,
                "coordinate_lon": lon,
                "coordinate_lat": lat,
                "coordinate_basis": "local source Reach downstream endpoint; names not used to select location",
                "locality_coordinate_reverse_geocode": locality_text(geo),
                "osm_way_count_in_query": len(ways),
                "osm_way_count_within_500m_of_join": len(near_ways),
                "nearest_osm_waterway_distance_at_join_m": join_distance,
                "source_local_segment_median_distance_to_osm_m": source_med,
                "receiver_local_segment_median_distance_to_osm_m": receiver_med,
                "source_segment_fraction_within_500m_osm": float(np.mean(source_dist <= 500)) if len(source_dist) else np.nan,
                "receiver_segment_fraction_within_500m_osm": float(np.mean(receiver_dist <= 500)) if len(receiver_dist) else np.nan,
                "osm_names_near_coordinate_auxiliary_only": auxiliary_names(ways, join_local),
                "external_coordinate_decision": decision,
                "counterfactual_affected_reach_count": int(cf.loc[source, "downstream_affected_reach_count"]),
                "counterfactual_affected_station_count": int(cf.loc[source, "affected_q72_representative_station_count"]),
                "counterfactual_affected_legacy_count": int(cf.loc[source, "affected_legacy_lowflow_reach_count"]),
                "source_component_area_km2": float(cf.loc[source, "source_component_total_area_km2"]),
                "external_source": "OpenStreetMap Nominatim + Overpass, queried strictly by coordinate",
            }
        )
        time.sleep(1.1)

    for rid in CATCHMENT_REVIEW_IDS:
        case_id = f"catchment_{rid}"
        line = reaches.loc[rid].geometry
        midpoint_local = line.interpolate(0.5, normalized=True)
        midpoint_wgs = transform(to_wgs.transform, midpoint_local)
        line_wgs = transform(to_wgs.transform, line)
        west, south, east, north = line_wgs.bounds
        pad = 0.03
        query = (
            f"[out:json][timeout:90];"
            f"way[waterway]({south-pad:.8f},{west-pad:.8f},{north+pad:.8f},{east+pad:.8f});"
            "out tags geom;"
        )
        geo = reverse_geocode(case_id, midpoint_wgs)
        ways = ways_to_frame(overpass_waterways(case_id, query), to_local)
        distances = nearest_distance(interpolate_points(line, 500.0), ways)
        frac300 = float(np.mean(distances <= 300)) if len(distances) else np.nan
        frac1000 = float(np.mean(distances <= 1000)) if len(distances) else np.nan
        median_dist = float(np.nanmedian(distances)) if np.isfinite(distances).any() else np.nan
        own_fraction = float(reaches.loc[rid, "own_catchment_length_fraction_buffer60m"])
        other_fraction = float(reaches.loc[rid, "largest_other_catchment_fraction"])
        if frac1000 >= 0.80 and own_fraction < 0.90:
            decision = "LOCAL_RIVER_GEOMETRY_EXTERNALLY_SUPPORTED_CATCHMENT_PARTITION_IS_PRIMARY_SUSPECT"
        elif frac1000 >= 0.50:
            decision = "LOCAL_RIVER_PARTLY_SUPPORTED_CATCHMENT_OR_GENERALIZATION_REVIEW"
        else:
            decision = "OPEN_NETWORK_COVERAGE_INSUFFICIENT_OR_LOCAL_REACH_GEOMETRY_REVIEW"
        rows.append(
            {
                "case_id": case_id,
                "case_type": "reach_incremental_catchment_mismatch",
                "source_reach": rid,
                "receiver_reach": reaches.loc[rid, "largest_other_catchment_reach_id"],
                "coordinate_lon": midpoint_wgs.x,
                "coordinate_lat": midpoint_wgs.y,
                "coordinate_basis": "local Reach midpoint and full Reach bounding box; names not used to select location",
                "locality_coordinate_reverse_geocode": locality_text(geo),
                "osm_way_count_in_query": len(ways),
                "osm_way_count_within_500m_of_join": np.nan,
                "nearest_osm_waterway_distance_at_join_m": median_dist,
                "source_local_segment_median_distance_to_osm_m": median_dist,
                "receiver_local_segment_median_distance_to_osm_m": np.nan,
                "source_segment_fraction_within_500m_osm": float(np.mean(distances <= 500)) if len(distances) else np.nan,
                "receiver_segment_fraction_within_500m_osm": np.nan,
                "osm_names_near_coordinate_auxiliary_only": auxiliary_names(ways, midpoint_local, 3000.0),
                "external_coordinate_decision": decision,
                "counterfactual_affected_reach_count": np.nan,
                "counterfactual_affected_station_count": np.nan,
                "counterfactual_affected_legacy_count": np.nan,
                "source_component_area_km2": float(reaches.loc[rid, "tot_km2"]),
                "own_catchment_length_fraction_buffer60m": own_fraction,
                "largest_other_catchment_fraction": other_fraction,
                "largest_other_catchment_reach": reaches.loc[rid, "largest_other_catchment_reach_id"],
                "local_reach_fraction_within_300m_osm": frac300,
                "local_reach_fraction_within_1000m_osm": frac1000,
                "external_source": "OpenStreetMap Nominatim + Overpass, queried strictly by coordinate/bounding box",
            }
        )
        time.sleep(1.1)

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "coordinate_based_external_geography_evidence.csv", index=False, encoding="utf-8-sig")
    meta = {
        "decision_basis": "coordinates and spatial continuity are primary; all river names are auxiliary only",
        "case_count": len(out),
        "terminal_case_count": int((out.case_type == "terminal_geometry_table_contradiction").sum()),
        "catchment_case_count": int((out.case_type == "reach_incremental_catchment_mismatch").sum()),
        "external_sources": [
            "https://nominatim.openstreetmap.org/",
            "https://overpass-api.de/",
        ],
        "license": "OpenStreetMap contributors, ODbL 1.0",
        "important_limit": "OpenStreetMap is an independent open geographic cross-check, not an official hydrological authority; absence is not evidence of absence.",
    }
    (REPORTS / "coordinate_based_external_geography_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out[["case_id", "locality_coordinate_reverse_geocode", "external_coordinate_decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
