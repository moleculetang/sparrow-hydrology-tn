from __future__ import annotations

import calendar
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

sys.path.insert(0, str(Path(r"E:\SPARROW\5_Test\20260820_1\scripts")))
from legacy20_shared import (  # noqa: E402
    ASRIV_PATH, DEM_PATH, FULL_TERMINALS, GEOMETRY_PATH, REACHES_PATH,
    S14_6, S20_1, S20_2, STATIC_PATH, cumulative_path_fractions, dump_json,
    hash_manifest, require_runtime, route_arrays, topology_operators,
)


INTERFACE_PATH = S14_6 / "outputs" / "structural_canonical_main_interface_2006_2022.parquet"
GDALLOCATIONINFO = Path(sys.prefix) / "Library" / "bin" / "gdallocationinfo.exe"
SAMPLE_SPACING_M = 500.0
HYDRAULIC_SCENARIOS = (
    ("central_n0025", "central", 0.025),
    ("central_n0035", "central", 0.035),
    ("central_n0050", "central", 0.050),
    ("geometry_lower95_n0035", "lower95", 0.035),
    ("geometry_upper95_n0035", "upper95", 0.035),
)
MAIN_SCENARIO = "central_n0035"


def _line(geometry):
    if geometry.geom_type == "LineString":
        return geometry
    merged = linemerge(geometry)
    if merged.geom_type != "LineString":
        raise RuntimeError("reach geometry cannot be reduced to one line")
    return merged


def _reverse(line: LineString) -> LineString:
    return LineString(list(line.coords)[::-1])


def dem_value(lon: float, lat: float) -> float:
    result = subprocess.run(
        [str(GDALLOCATIONINFO), "-valonly", "-wgs84", str(DEM_PATH), f"{lon:.10f}", f"{lat:.10f}"],
        capture_output=True, text=True, check=True,
    )
    value = float(result.stdout.strip().splitlines()[-1])
    if not np.isfinite(value) or value <= -32768:
        raise RuntimeError(f"invalid DEM value at {lon},{lat}: {value}")
    return value


def robust_profile_slope(line: LineString, crs, reach_id: int) -> tuple[float, pd.DataFrame]:
    positions = np.linspace(0.025, 0.975, 39)
    points = gpd.GeoSeries([line.interpolate(float(x), normalized=True) for x in positions], crs=crs).to_crs("EPSG:4326")
    elevation = np.asarray([dem_value(point.x, point.y) for point in points], dtype=float)
    distance = positions * line.length
    pair_slopes = []
    for i in range(len(distance)):
        for j in range(i + 1, len(distance)):
            pair_slopes.append((elevation[j] - elevation[i]) / (distance[j] - distance[i]))
    coefficient = float(np.median(pair_slopes))
    slope = -coefficient
    profile = pd.DataFrame({
        "reach_id": reach_id, "profile_fraction": positions, "distance_from_oriented_upstream_m": distance,
        "elevation_m": elevation, "pairwise_median_dz_dx": coefficient,
    })
    return slope, profile


def robust_network_profile_slope(
    reach_id: int,
    geometry: dict[int, LineString],
    crs,
    downstream: dict[int, tuple[int, float]],
    topology: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    upstream_map: dict[int, list[int]] = {}
    for source, (target, _) in downstream.items():
        upstream_map.setdefault(target, []).append(source)
    area = topology.set_index("reach_id").tot_area_km2.to_dict()
    selected_upstream = None
    if reach_id in upstream_map:
        selected_upstream = sorted(upstream_map[reach_id], key=lambda rid: (-float(area[rid]), rid))[0]
    selected_downstream = downstream.get(reach_id, (None, 1.0))[0]
    current = geometry[reach_id]
    pieces: list[tuple[int, LineString, float]] = []
    if selected_upstream is not None:
        up = geometry[selected_upstream]
        pieces.append((selected_upstream, up, -up.length))
    pieces.append((reach_id, current, 0.0))
    if selected_downstream is not None:
        pieces.append((selected_downstream, geometry[selected_downstream], current.length))
    records = []
    for piece_id, line, offset in pieces:
        positions = np.linspace(0.025, 0.975, 39)
        points = gpd.GeoSeries([line.interpolate(float(x), normalized=True) for x in positions], crs=crs).to_crs("EPSG:4326")
        elevations = [dem_value(point.x, point.y) for point in points]
        for fraction, elevation in zip(positions, elevations):
            records.append({
                "reach_id": reach_id, "profile_component_reach_id": piece_id,
                "profile_fraction": fraction, "distance_from_oriented_upstream_m": offset + fraction * line.length,
                "elevation_m": elevation,
            })
    profile = pd.DataFrame(records).sort_values("distance_from_oriented_upstream_m").reset_index(drop=True)
    x = profile.distance_from_oriented_upstream_m.to_numpy(float)
    z = profile.elevation_m.to_numpy(float)
    centered = x - x.mean()
    coefficient = float(np.sum(centered * (z - z.mean())) / np.sum(centered ** 2))
    profile["network_profile_ols_dz_dx"] = coefficient
    profile["network_profile_upstream_reach_id"] = selected_upstream
    profile["network_profile_downstream_reach_id"] = selected_downstream
    return -coefficient, profile


def oriented_reaches_and_slopes() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    reaches = gpd.read_file(REACHES_PATH)[["reach_id", "geometry"]].sort_values("reach_id").reset_index(drop=True)
    reaches["reach_id"] = reaches.reach_id.astype(int)
    if len(reaches) != 230 or reaches.crs is None or not reaches.crs.is_projected:
        raise RuntimeError("expected 230 projected reach lines")
    static = pd.read_parquet(STATIC_PATH).sort_values("reach_id").reset_index(drop=True)
    _, downstream, _ = topology_operators(reaches.reach_id.to_numpy(int))
    topology = pd.read_csv(Path(r"E:\SPARROW\5_Test\20260814_1\inputs\topology\topology_edges.csv"))
    topology["reach_id"] = topology.reach_id.astype(int)
    geometry = {int(row.reach_id): _line(row.geometry) for row in reaches.itertuples()}
    orientation_rows = []
    for rid in sorted(geometry):
        line = geometry[rid]
        orientation = "source_geometry_order_terminal"
        reversed_flag = False
        if rid in downstream:
            down = downstream[rid][0]
            first = Point(line.coords[0]).distance(geometry[down])
            last = Point(line.coords[-1]).distance(geometry[down])
            if first < last:
                line = _reverse(line)
                reversed_flag = True
            orientation = "endpoint_nearest_frozen_downstream_reach"
        geometry[rid] = line
        orientation_rows.append({"reach_id": rid, "orientation_method": orientation, "geometry_reversed": reversed_flag})
    reaches["geometry"] = reaches.reach_id.map(geometry)

    slopes = static[["reach_id", "slope_raw"]].copy()
    slopes["slope_used"] = slopes.slope_raw.astype(float)
    slopes["slope_method"] = "frozen_raw_slope"
    profile_rows: list[pd.DataFrame] = []
    zero_ids = slopes.loc[slopes.slope_raw <= 0, "reach_id"].astype(int).tolist()
    for rid in zero_ids:
        line = geometry[rid]
        estimate, profile = robust_profile_slope(line, reaches.crs, rid)
        # A terminal line has no downstream neighbor to orient its two ends.
        # Only there, a positive DEM trend deterministically reverses the line.
        if estimate <= 0 and rid not in downstream:
            line = _reverse(line)
            geometry[rid] = line
            reaches.loc[reaches.reach_id.eq(rid), "geometry"] = line
            estimate, profile = robust_profile_slope(line, reaches.crs, rid)
            row = next(x for x in orientation_rows if x["reach_id"] == rid)
            row["geometry_reversed"] = not row["geometry_reversed"]
            row["orientation_method"] = "terminal_DEM_profile_high_to_low"
        if estimate <= 0:
            estimate, profile = robust_network_profile_slope(rid, geometry, reaches.crs, downstream, topology)
            method = "topology_oriented_network_39point_DEM_OLS"
        else:
            method = "topology_oriented_local_39point_DEM_TheilSen"
        profile_rows.append(profile)
        if not np.isfinite(estimate) or estimate <= 0:
            raise RuntimeError(f"STOP_NONPOSITIVE_SLOPE_AFTER_PROFILE_REPAIR reach={rid} estimate={estimate}")
        slopes.loc[slopes.reach_id.eq(rid), "slope_used"] = estimate
        slopes.loc[slopes.reach_id.eq(rid), "slope_method"] = method
    if (slopes.slope_used <= 0).any():
        raise RuntimeError("nonpositive slope remains; no floor or absolute-value repair is allowed")
    reaches["geometry"] = reaches.reach_id.map(geometry)
    return reaches, slopes, pd.concat(profile_rows, ignore_index=True).merge(pd.DataFrame(orientation_rows), on="reach_id", how="left")


def build_segments(reaches: gpd.GeoDataFrame, slopes: pd.DataFrame) -> pd.DataFrame:
    records = []
    point_id = 0
    for row in reaches.itertuples(index=False):
        line = _line(row.geometry)
        count = max(1, int(np.ceil(line.length / SAMPLE_SPACING_M)))
        segment_length = line.length / count
        for i in range(count):
            start_fraction, end_fraction = i / count, (i + 1) / count
            point = line.interpolate((i + 0.5) / count, normalized=True)
            records.append({
                "segment_id": point_id, "reach_id": int(row.reach_id), "segment_index": i,
                "segment_count": count, "segment_length_m": segment_length,
                "segment_midpoint_fraction": (i + 0.5) / count,
                "segment_start_fraction": start_fraction, "segment_end_fraction": end_fraction,
                "midpoint_to_outlet_segment_length_m": max(0.0, end_fraction - max(start_fraction, 0.5)) * line.length,
                "geometry": point,
            })
            point_id += 1
    points = gpd.GeoDataFrame(records, crs=reaches.crs)
    bounds = reaches.to_crs("EPSG:4326").total_bounds
    source = gpd.read_file(ASRIV_PATH, bbox=(bounds[0] - 0.2, bounds[1] - 0.2, bounds[2] + 0.2, bounds[3] + 0.2))[
        ["ARCID", "WIDTH", "WIDTH5", "WIDTH95", "DEPTH", "DEPTH5", "DEPTH95", "geometry"]
    ].to_crs(reaches.crs)
    joined = gpd.sjoin_nearest(points, source, how="left", max_distance=50_000.0, distance_col="nearest_wqd_distance_m")
    joined = joined.sort_values(["segment_id", "nearest_wqd_distance_m", "ARCID"]).drop_duplicates("segment_id")
    if len(joined) != len(points) or joined.ARCID.isna().any():
        raise RuntimeError("not every 500 m segment has a deterministic WQD geometry match")
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
    out = out.rename(columns={
        "ARCID": "wqd_arc_id", "WIDTH": "width_central_m", "WIDTH5": "width_lower95_m", "WIDTH95": "width_upper95_m",
        "DEPTH": "depth_central_m", "DEPTH5": "depth_lower95_m", "DEPTH95": "depth_upper95_m",
    }).merge(slopes[["reach_id", "slope_raw", "slope_used", "slope_method"]], on="reach_id", validate="many_to_one")
    out["andreadis_bound_semantics"] = "source_lower_and_upper_95pct_confidence_bounds_not_joint_probability_interval"
    out["wqd_reference_discharge_used"] = False
    return out.sort_values(["reach_id", "segment_index"]).reset_index(drop=True)


def hydraulic_forcing() -> pd.DataFrame:
    frame = pd.read_parquet(INTERFACE_PATH).rename(columns={"comid": "reach_id"}).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    if len(frame) != 46_920 or frame[["reach_id", "year", "month"]].duplicated().any():
        raise RuntimeError("canonical Q72 interface is not 46,920 unique reach-month rows")
    reach_ids = np.sort(frame.reach_id.unique().astype(int))
    times = frame[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    n_t, n_r = len(times), len(reach_ids)
    ordered = frame.set_index(["year", "month", "reach_id"]).loc[
        [(int(y), int(m), int(r)) for y, m in times.itertuples(index=False) for r in reach_ids]
    ].reset_index()
    local = (ordered.q_local_total_mm.to_numpy(float) * ordered.catchment_area_km2.to_numpy(float) * 1000.0).reshape(n_t, n_r)
    routed, terminal = route_arrays(local, reach_ids)
    upstream = routed - local
    if upstream.min() < -1e-6:
        raise RuntimeError("routed Q72 volume is smaller than local volume")
    upstream = np.maximum(upstream, 0.0)
    seconds = ordered.month_seconds.to_numpy(float).reshape(n_t, n_r)
    if not np.allclose(seconds, seconds[:, :1]):
        raise RuntimeError("month_seconds varies across reaches")
    out = pd.DataFrame({
        "reach_id": np.tile(reach_ids, n_t), "year": np.repeat(times.year.to_numpy(int), n_r),
        "month": np.repeat(times.month.to_numpy(int), n_r), "month_seconds": seconds.reshape(-1),
        "q72_local_water_volume_m3": local.reshape(-1), "q72_upstream_water_volume_m3": upstream.reshape(-1),
        "q72_outlet_water_volume_m3": routed.reshape(-1),
    })
    out["q72_local_discharge_m3_s"] = out.q72_local_water_volume_m3 / out.month_seconds
    out["q72_upstream_discharge_m3_s"] = out.q72_upstream_water_volume_m3 / out.month_seconds
    out["q72_outlet_discharge_m3_s"] = out.q72_outlet_water_volume_m3 / out.month_seconds
    out["terminal_tree_id"] = out.reach_id.map(terminal).astype(int)
    out["water_source"] = "Q72_structural_canonical_main_only"
    out["andreadis_discharge_used"] = False
    return out


def manning_depth(q: np.ndarray, width: np.ndarray, slope: np.ndarray, roughness: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    width = np.asarray(width, dtype=float)
    slope = np.asarray(slope, dtype=float)
    h = np.maximum((np.maximum(q, 1e-30) * roughness / (width * np.sqrt(slope))) ** 0.6, 1e-8)
    for _ in range(20):
        area = width * h
        perimeter = width + 2.0 * h
        radius = area / perimeter
        modeled = area * radius ** (2.0 / 3.0) * np.sqrt(slope) / roughness
        dlog = 1.0 / h + (2.0 / 3.0) * (1.0 / h - 2.0 / perimeter)
        update = (modeled - q) / np.maximum(modeled * dlog, 1e-30)
        h = np.maximum(h - update, 1e-10)
    area = width * h
    radius = area / (width + 2.0 * h)
    closure = area * radius ** (2.0 / 3.0) * np.sqrt(slope) / roughness
    if not np.allclose(closure, q, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Manning solve did not close")
    return h


def calculate_exposure(segments: pd.DataFrame, hydro: pd.DataFrame) -> pd.DataFrame:
    reach_ids = np.sort(hydro.reach_id.unique().astype(int))
    rindex = {rid: i for i, rid in enumerate(reach_ids)}
    seg_ridx = segments.reach_id.map(rindex).to_numpy(int)
    x = segments.segment_midpoint_fraction.to_numpy(float)
    length = segments.segment_length_m.to_numpy(float)
    mid_length = segments.midpoint_to_outlet_segment_length_m.to_numpy(float)
    slope = segments.slope_used.to_numpy(float)
    times = hydro[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    n_r = len(reach_ids)
    hordered = hydro.sort_values(["year", "month", "reach_id"])
    local_q = hordered.q72_local_discharge_m3_s.to_numpy(float).reshape(len(times), n_r)
    upstream_q = hordered.q72_upstream_discharge_m3_s.to_numpy(float).reshape(len(times), n_r)
    month_seconds = hordered.month_seconds.to_numpy(float).reshape(len(times), n_r)[:, 0]
    records: list[pd.DataFrame] = []
    for scenario, geometry, roughness in HYDRAULIC_SCENARIOS:
        width = segments[f"width_{geometry}_m"].to_numpy(float)
        bankfull_depth = segments[f"depth_{geometry}_m"].to_numpy(float)
        for ti, (year, month) in enumerate(times.itertuples(index=False)):
            q = upstream_q[ti, seg_ridx] + x * local_q[ti, seg_ridx]
            if (q <= 0).any():
                raise RuntimeError("nonpositive along-reach Q prevents finite same-month exposure")
            depth = manning_depth(q, width, slope, roughness)
            area = width * depth
            velocity = q / area
            tau = length / velocity
            tau_mid = mid_length / velocity
            ab = depth > bankfull_depth
            full = np.bincount(seg_ridx, weights=tau, minlength=n_r)
            mid = np.bincount(seg_ridx, weights=tau_mid, minlength=n_r)
            ab_full = np.bincount(seg_ridx, weights=tau * ab, minlength=n_r)
            ab_mid = np.bincount(seg_ridx, weights=tau_mid * ab, minlength=n_r)
            length_sum = np.bincount(seg_ridx, weights=length, minlength=n_r)
            item = pd.DataFrame({
                "reach_id": reach_ids, "year": int(year), "month": int(month), "hydraulic_scenario": scenario,
                "geometry_scenario": geometry, "manning_n": roughness,
                "travel_time_full_seconds": full, "travel_time_midpoint_to_outlet_seconds": mid,
                "above_bankfull_travel_seconds_full": ab_full, "above_bankfull_travel_seconds_midpoint": ab_mid,
                "above_bankfull_fraction_full": np.divide(ab_full, full, out=np.zeros(n_r), where=full > 0),
                "above_bankfull_fraction_midpoint": np.divide(ab_mid, mid, out=np.zeros(n_r), where=mid > 0),
                "length_weighted_mean_depth_m": np.bincount(seg_ridx, weights=depth * length, minlength=n_r) / length_sum,
                "length_weighted_mean_velocity_m_s": np.bincount(seg_ridx, weights=velocity * length, minlength=n_r) / length_sum,
                "max_depth_to_bankfull_ratio": np.maximum.reduceat(depth / bankfull_depth, np.r_[0, np.cumsum(np.bincount(seg_ridx))[:-1]]),
                "month_seconds": month_seconds[ti],
            })
            item["travel_time_full_days"] = item.travel_time_full_seconds / 86400.0
            item["travel_time_midpoint_to_outlet_days"] = item.travel_time_midpoint_to_outlet_seconds / 86400.0
            item["reach_exceeds_month"] = item.travel_time_full_seconds > item.month_seconds
            item["above_bankfull_interpretation"] = "constant_bankfull_width_exposure_lower_bound"
            records.append(item)
    return pd.concat(records, ignore_index=True).sort_values(["hydraulic_scenario", "year", "month", "reach_id"])


def path_exposure(main: pd.DataFrame, reach_ids: np.ndarray) -> pd.DataFrame:
    order, downstream, terminal = topology_operators(reach_ids)
    path_registry = cumulative_path_fractions(reach_ids)
    # Explicit ordered paths avoid any ambiguity at split nodes.
    path_nodes: dict[tuple[int, int], list[int]] = {}
    for source in map(int, reach_ids):
        current, nodes = source, []
        path_nodes[(source, source)] = []
        while current in downstream:
            current = downstream[current][0]
            nodes = nodes + [current]
            path_nodes[(source, current)] = nodes.copy()
    blocks = []
    for (year, month), frame in main.groupby(["year", "month"], sort=True):
        f = frame.set_index("reach_id")
        rows = []
        for row in path_registry.itertuples(index=False):
            source, target = int(row.source_reach_id), int(row.target_reach_id)
            downstream_nodes = path_nodes[(source, target)]
            full_time = float(f.loc[downstream_nodes, "travel_time_full_seconds"].sum()) if downstream_nodes else 0.0
            full_ab = float(f.loc[downstream_nodes, "above_bankfull_travel_seconds_full"].sum()) if downstream_nodes else 0.0
            total_time = float(f.at[source, "travel_time_midpoint_to_outlet_seconds"]) + full_time
            total_ab = float(f.at[source, "above_bankfull_travel_seconds_midpoint"]) + full_ab
            rows.append({
                "source_reach_id": source, "target_reach_id": target, "year": int(year), "month": int(month),
                "terminal_tree_id": terminal[source], "cumulative_routing_fraction": float(row.cumulative_routing_fraction),
                "path_steps": int(row.path_steps), "path_time_seconds": total_time,
                "path_time_days": total_time / 86400.0, "path_exceeds_month": total_time > float(f.at[target, "month_seconds"]),
                "path_ab_travel_seconds": total_ab,
                "path_ab_fraction": total_ab / total_time if total_time > 0 else 0.0,
                "source_entry_exposure": "actual_geometric_midpoint_to_outlet",
            })
        blocks.append(pd.DataFrame(rows))
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    require_runtime()
    out = S20_2 / "outputs"
    reports = S20_2 / "reports"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    parent_paths = [INTERFACE_PATH, STATIC_PATH, GEOMETRY_PATH, REACHES_PATH, ASRIV_PATH, DEM_PATH, S20_1 / "experiment_contract.json"]
    start = hash_manifest(parent_paths)
    dump_json(reports / "parent_hashes_start.json", start)
    reaches, slopes, profiles = oriented_reaches_and_slopes()
    slopes.to_parquet(out / "reach_slope_registry.parquet", index=False)
    profiles.to_parquet(out / "zero_slope_dem_profiles.parquet", index=False)
    segments = build_segments(reaches, slopes)
    segments.to_parquet(out / "andreadis_500m_channel_segments.parquet", index=False)
    hydro = hydraulic_forcing()
    hydro.to_parquet(out / "q72_routed_monthly_discharge_2006_2022.parquet", index=False)
    exposure = calculate_exposure(segments, hydro)
    exposure.to_parquet(out / "reach_month_hydraulic_exposure.parquet", index=False)
    main_exposure = exposure.loc[exposure.hydraulic_scenario.eq(MAIN_SCENARIO)].copy()
    paths = path_exposure(main_exposure, np.sort(hydro.reach_id.unique().astype(int)))
    paths.to_parquet(out / "source_target_path_exposure_2006_2022.parquet", index=False)

    discharge_reconstructed = hydro.q72_outlet_discharge_m3_s * hydro.month_seconds
    unit_max = float(np.max(np.abs(discharge_reconstructed - hydro.q72_outlet_water_volume_m3)))
    unit_max_relative = float(np.max(np.abs(discharge_reconstructed - hydro.q72_outlet_water_volume_m3) / np.maximum(np.abs(hydro.q72_outlet_water_volume_m3), 1.0)))
    scenario_summary = exposure.groupby("hydraulic_scenario", as_index=False).agg(
        median_full_days=("travel_time_full_days", "median"), p95_full_days=("travel_time_full_days", lambda x: float(np.quantile(x, 0.95))),
        fraction_reach_month_ab=("max_depth_to_bankfull_ratio", lambda x: float(np.mean(x > 1))),
        fraction_reach_month_over_month=("reach_exceeds_month", "mean"), median_velocity_m_s=("length_weighted_mean_velocity_m_s", "median"),
    )
    scenario_summary.to_parquet(out / "hydraulic_scenario_summary.parquet", index=False)
    checks = {
        "segments_cover_230_reaches": segments.reach_id.nunique() == 230,
        "wqd_reference_discharge_never_used": not segments.wqd_reference_discharge_used.any() and not hydro.andreadis_discharge_used.any(),
        "all_slopes_positive_without_floor": (slopes.slope_used > 0).all() and not slopes.slope_method.astype(str).str.contains("floor|absolute", case=False, regex=True).any(),
        "zero_slopes_repaired_by_profile": int((slopes.slope_raw <= 0).sum()) == int(slopes.slope_method.str.contains("DEM_", regex=False).sum()),
        "q72_volume_discharge_closure": unit_max_relative <= 1e-12,
        "five_fixed_hydraulic_scenarios": exposure.hydraulic_scenario.nunique() == 5,
        "main_scenario_unique": len(main_exposure) == 46_920,
        "full_and_mid_exposure_positive": (exposure.travel_time_full_seconds > 0).all() and (exposure.travel_time_midpoint_to_outlet_seconds > 0).all(),
        "mid_exposure_is_actual_not_half": np.max(np.abs(exposure.travel_time_midpoint_to_outlet_seconds - 0.5 * exposure.travel_time_full_seconds)) > 1e-6,
        "path_split_weights_present": paths.cumulative_routing_fraction.between(0, 1, inclusive="right").all(),
        "full_topology_14_trees": paths.terminal_tree_id.nunique() == 14,
        "parent_hashes_unchanged": hash_manifest(parent_paths) == start,
    }
    dump_json(reports / "monthly_volume_to_discharge_unit_audit.json", {"status": "PASS" if checks["q72_volume_discharge_closure"] else "FAIL", "max_abs_volume_reconstruction_error_m3": unit_max, "max_relative_volume_reconstruction_error": unit_max_relative, "relative_tolerance": 1e-12, "formula": "Q=V/(month_seconds)"})
    dump_json(reports / "slope_repair_audit.json", {"status": "PASS" if checks["all_slopes_positive_without_floor"] else "FAIL", "raw_zero_reaches": slopes.loc[slopes.slope_raw <= 0, "reach_id"].astype(int).tolist(), "method": "topology-oriented local profile, extended to deterministic largest-area upstream/current/downstream profile only when local DEM is flat", "silent_floor_forbidden": True})
    dump_json(reports / "hydraulic_contract_audit.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": {k: bool(v) for k, v in checks.items()}, "main_scenario": MAIN_SCENARIO, "geometry_bound_names": ["geometry_lower95", "central", "geometry_upper95"], "above_bankfull_role": "exposure_lower_bound", "path_low_ab_threshold": 0.05})
    dump_json(reports / "parent_hashes_end.json", hash_manifest(parent_paths))
    (S20_2 / "experiment_contract.json").write_text(json.dumps({
        "stage": "20260820_2", "water_source": "Q72 only", "segment_flow_formula": "Q_upstream + x_midpoint*Q_local",
        "full_exposure": "sum(segment_length/segment_velocity)", "local_exposure": "geometric_midpoint_to_outlet segment integration",
        "wqd_reference_discharge_used": False, "manning_n": [0.025, 0.035, 0.050],
        "geometry": ["central", "lower95", "upper95"], "main": MAIN_SCENARIO,
        "low_ab_subset": "pre-aquatic local N mass and cumulative split weighted F_AB <= 0.05",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(f"stage2 failed: {[k for k, v in checks.items() if not v]}")
    print(json.dumps({"status": "PASS", "segments": len(segments), "hydraulic_rows": len(exposure), "path_rows": len(paths)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
