from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TOPO = ROOT / "0_reach_topology"
BASE = ROOT / "5_Test" / "20260805_2"
SIGNAL_RUN = ROOT / "5_Test" / "20260730_12"
REPORTS = RUN / "reports"
FIGURES = RUN / "figures"
VECTORS = RUN / "outputs" / "vectors"

REACH_PATH = TOPO / "results" / "vectors" / "reaches_topology.shp"
CATCH_PATH = TOPO / "results" / "vectors" / "reach_catchments.shp"
NODE_PATH = TOPO / "results" / "vectors" / "nodes.shp"
REACH_SUMMARY_PATH = TOPO / "results" / "tables" / "reach_summary.csv"
EDGE_PATH = TOPO / "results" / "tables" / "topology_edges.csv"
NODE_SUMMARY_PATH = TOPO / "results" / "tables" / "node_summary.csv"
MANUAL_PATH = TOPO / "results" / "tables" / "manual_review.csv"
SOURCE_STREAM_PATH = TOPO / "data" / "processed" / "vector" / "pr_sparrow_streams.shp"
BASIN_PATH = TOPO / "data" / "processed" / "vector" / "prb_boundary.shp"
STATIONS_PATH = TOPO / "data" / "raw" / "vector" / "PRB水文站_全部.shp"
STATION_MATCH_PATH = BASE / "inputs" / "source_metadata" / "station_reach_match.csv"
SAME_REACH_PATH = BASE / "inputs" / "source_metadata" / "same_reach_selection_audit.csv"
SELECTED_STATIONS_PATH = BASE / "inputs" / "source_metadata" / "selected_representative_stations.csv"
SIGNAL_PATH = SIGNAL_RUN / "reports" / "signal_registry" / "canonical_signal_registry.csv"

ENDPOINT_TOL_M = 1.0
CATCH_BUFFER_M = 60.0
STATION_REVIEW_M = 1000.0
STATION_HIGH_M = 2000.0
STATION_SEVERE_M = 5000.0
OWN_CATCH_PASS = 0.95
OWN_CATCH_HIGH = 0.90
AREA_ABS_TOL = 1e-6
AREA_REL_TOL = 1e-10


def norm_name(value: object) -> str:
    text = str(value).strip().replace(" ", "")
    text = text.replace("（", "(").replace("）", ")")
    return re.sub(r"站$", "", text)


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def relaxed_station_name(value: object) -> str:
    text = norm_name(value)
    text = re.sub(r"\(重复\)$", "", text)
    text = re.sub(r"_\d+$", "", text)
    return text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def vector_components(path: Path) -> list[Path]:
    if path.suffix.lower() != ".shp":
        return [path]
    return [path.with_suffix(s) for s in [".shp", ".shx", ".dbf", ".prj", ".cpg"] if path.with_suffix(s).exists()]


def endpoint(line, first: bool) -> Point:
    coords = list(line.coords)
    return Point(coords[0] if first else coords[-1])


def write_layer(gdf: gpd.GeoDataFrame, path: Path, layer: str, first: bool) -> None:
    mode = "w" if first else "a"
    gdf.to_file(path, layer=layer, driver="GPKG", index=False, mode=mode)


def parse_dem_note(text: object) -> tuple[float | None, bool]:
    note = str(text)
    m = re.search(r"DEM endpoint drop\s+(-?[0-9.]+)\s+m", note)
    return (abs(float(m.group(1))) if m else None, "below threshold" in note or "unavailable" in note)


def load_inputs():
    reaches = gpd.read_file(REACH_PATH)
    catches = gpd.read_file(CATCH_PATH).to_crs(reaches.crs)
    nodes = gpd.read_file(NODE_PATH).to_crs(reaches.crs)
    source = gpd.read_file(SOURCE_STREAM_PATH).to_crs(reaches.crs)
    basin = gpd.read_file(BASIN_PATH).to_crs(reaches.crs)
    stations = gpd.read_file(STATIONS_PATH).to_crs(reaches.crs)
    summary = pd.read_csv(REACH_SUMMARY_PATH, encoding="utf-8-sig")
    edges = pd.read_csv(EDGE_PATH, encoding="utf-8-sig")
    node_summary = pd.read_csv(NODE_SUMMARY_PATH, encoding="utf-8-sig")
    manual = pd.read_csv(MANUAL_PATH, encoding="utf-8-sig")
    for frame in [reaches, catches, summary, edges]:
        frame["reach_id"] = pd.to_numeric(frame["reach_id"], errors="raise").astype(int)
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="raise").astype(int)
    node_summary["node_id"] = pd.to_numeric(node_summary["node_id"], errors="raise").astype(int)
    return reaches, catches, nodes, source, basin, stations, summary, edges, node_summary, manual


def graph_audit(reaches: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, summary: pd.DataFrame, edges: pd.DataFrame):
    reach_ids = set(reaches.reach_id)
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(reach_ids))
    invalid_downstream = []
    for row in edges.itertuples(index=False):
        rid = int(row.reach_id)
        if pd.notna(row.downstream_reach):
            down = int(row.downstream_reach)
            if down not in reach_ids:
                invalid_downstream.append((rid, down))
            else:
                graph.add_edge(rid, down)

    node_geom = nodes.set_index("node_id").geometry
    reach_table = reaches.set_index("reach_id")
    edge_table = edges.set_index("reach_id")
    summary_table = summary.set_index("reach_id")
    rows = []
    for rid in sorted(reach_ids):
        rec = reach_table.loc[rid]
        edge = edge_table.loc[rid]
        line = rec.geometry
        fnode, tnode = int(edge.fnode), int(edge.tnode)
        start, end = endpoint(line, True), endpoint(line, False)
        f_dist = float(start.distance(node_geom.loc[fnode])) if fnode in node_geom.index else np.nan
        t_dist = float(end.distance(node_geom.loc[tnode])) if tnode in node_geom.index else np.nan
        down = int(edge.downstream_reach) if pd.notna(edge.downstream_reach) else None
        down_node_match = True
        down_end_distance = 0.0
        if down is not None and down in reach_table.index:
            down_edge = edge_table.loc[down]
            down_line = reach_table.loc[down].geometry
            down_node_match = int(down_edge.fnode) == tnode
            down_end_distance = float(end.distance(endpoint(down_line, True)))
        dem_drop, dem_uncertain = parse_dem_note(summary_table.loc[rid, "qa_notes"])
        rows.append({
            "reach_id": rid,
            "src_id": str(rec.src_id),
            "fnode": fnode,
            "tnode": tnode,
            "downstream_reach": down,
            "graph_in_degree": int(graph.in_degree(rid)),
            "graph_out_degree": int(graph.out_degree(rid)),
            "terminal_field": int(edge.terminal),
            "headwater_field": int(summary_table.loc[rid, "headwater"]),
            "start_to_fnode_m": f_dist,
            "end_to_tnode_m": t_dist,
            "downstream_fnode_matches_tnode": bool(down_node_match),
            "downstream_endpoint_gap_m": down_end_distance,
            "geometry_valid": bool(line.is_valid),
            "geometry_simple": bool(line.is_simple),
            "geometry_empty": bool(line.is_empty),
            "self_loop": bool(down == rid),
            "terminal_consistent_with_graph": bool((down is None) == bool(edge.terminal)),
            "headwater_consistent_with_graph": bool((graph.in_degree(rid) == 0) == bool(summary_table.loc[rid, "headwater"])),
            "dem_endpoint_drop_abs_m": dem_drop,
            "dem_direction_uncertain": bool(dem_uncertain),
        })
    audit = pd.DataFrame(rows)
    meta = {
        "reach_count": len(reach_ids),
        "edge_count": graph.number_of_edges(),
        "cycles": list(nx.simple_cycles(graph)),
        "weak_components": nx.number_weakly_connected_components(graph),
        "terminal_count": sum(1 for n in graph if graph.out_degree(n) == 0),
        "headwater_count": sum(1 for n in graph if graph.in_degree(n) == 0),
        "invalid_downstream": invalid_downstream,
        "dag": nx.is_directed_acyclic_graph(graph),
    }
    return graph, audit, meta


def pairwise_intersections(reaches: gpd.GeoDataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    r = reaches.reset_index(drop=True)
    e = edges.set_index("reach_id")
    rows = []
    sidx = r.sindex
    for i, a in r.iterrows():
        for j in sidx.intersection(a.geometry.bounds):
            if j <= i:
                continue
            b = r.iloc[j]
            if not a.geometry.intersects(b.geometry):
                continue
            inter = a.geometry.intersection(b.geometry)
            if inter.is_empty:
                continue
            rid_a, rid_b = int(a.reach_id), int(b.reach_id)
            nodes_a = {int(e.loc[rid_a, "fnode"]), int(e.loc[rid_a, "tnode"])}
            nodes_b = {int(e.loc[rid_b, "fnode"]), int(e.loc[rid_b, "tnode"])}
            shared = sorted(nodes_a & nodes_b)
            length = float(inter.length)
            points = []
            if inter.geom_type == "Point":
                points = [inter]
            elif inter.geom_type == "MultiPoint":
                points = list(inter.geoms)
            elif inter.geom_type in {"LineString", "MultiLineString"}:
                points = [inter.representative_point()]
            else:
                points = [inter.representative_point()]
            a_start, a_end = endpoint(a.geometry, True), endpoint(a.geometry, False)
            b_start, b_end = endpoint(b.geometry, True), endpoint(b.geometry, False)
            endpoint_distance = min(p.distance(q) for p in points for q in [a_start, a_end, b_start, b_end])
            a_start_distance = min(p.distance(a_start) for p in points)
            a_end_distance = min(p.distance(a_end) for p in points)
            b_start_distance = min(p.distance(b_start) for p in points)
            b_end_distance = min(p.distance(b_end) for p in points)
            expected_graph_adj = bool(e.loc[rid_a, "downstream_reach"] == rid_b or e.loc[rid_b, "downstream_reach"] == rid_a)
            rows.append({
                "reach_id_a": rid_a,
                "src_id_a": str(a.src_id),
                "reach_id_b": rid_b,
                "src_id_b": str(b.src_id),
                "intersection_type": inter.geom_type,
                "intersection_length_m": length,
                "shared_node_ids": "|".join(map(str, shared)),
                "shared_topology_node": bool(shared),
                "direct_graph_adjacency": expected_graph_adj,
                "intersection_near_any_endpoint_m": float(endpoint_distance),
                "reach_a_start_distance_m": float(a_start_distance),
                "reach_a_end_distance_m": float(a_end_distance),
                "reach_b_start_distance_m": float(b_start_distance),
                "reach_b_end_distance_m": float(b_end_distance),
                "reach_a_terminal_outlet_intersection": bool(e.loc[rid_a, "terminal"] == 1 and a_end_distance <= ENDPOINT_TOL_M and not shared),
                "reach_b_terminal_outlet_intersection": bool(e.loc[rid_b, "terminal"] == 1 and b_end_distance <= ENDPOINT_TOL_M and not shared),
                "unmodeled_intersection": not bool(shared),
                "unmodeled_endpoint_intersection": not bool(shared) and endpoint_distance <= ENDPOINT_TOL_M,
                "overlap_fraction_shorter": length / min(a.geometry.length, b.geometry.length) if min(a.geometry.length, b.geometry.length) else np.nan,
                "geometry": points[0],
            })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=reaches.crs)


def catchment_audit(reaches: gpd.GeoDataFrame, catches: gpd.GeoDataFrame, graph: nx.DiGraph) -> pd.DataFrame:
    r = reaches.set_index("reach_id")
    c = catches.set_index("reach_id")
    rows = []
    for rid in sorted(r.index):
        line = r.loc[rid].geometry
        poly = c.loc[rid].geometry
        raw_len = float(line.intersection(poly).length)
        # Catchments originate from a 30 m raster and can contain extremely dense
        # pixel stair-steps.  Simplifying by half a cell before the 60 m tolerance
        # buffer preserves the registered spatial tolerance while avoiding an
        # expensive exact buffer over millions of redundant raster-edge vertices.
        poly_buffer = poly.simplify(15.0, preserve_topology=True).buffer(CATCH_BUFFER_M, quad_segs=1)
        buffered_len = float(line.intersection(poly_buffer).length)
        other = []
        candidate_pos = catches.sindex.query(line, predicate="intersects")
        for oid in catches.iloc[list(candidate_pos)].reach_id.astype(int):
            if oid == rid:
                continue
            olen = float(line.intersection(c.loc[oid].geometry).length)
            if olen > 0:
                other.append((olen, int(oid)))
        other.sort(reverse=True)
        largest_other_len, largest_other_id = other[0] if other else (0.0, None)
        inc = float(r.loc[rid, "inc_km2"])
        tot = float(r.loc[rid, "tot_km2"])
        expected_tot = inc + sum(float(r.loc[u, "tot_km2"]) for u in graph.predecessors(rid))
        area_err = tot - expected_tot
        area_rel = abs(area_err) / max(abs(expected_tot), 1.0)
        down = next(iter(graph.successors(rid)), None)
        down_tot = float(r.loc[down, "tot_km2"]) if down is not None else np.nan
        down_catch_gap = float(poly.distance(c.loc[down].geometry)) if down is not None else np.nan
        rows.append({
            "reach_id": rid,
            "src_id": str(r.loc[rid, "src_id"]),
            "length_km": float(line.length / 1000),
            "incremental_area_km2": inc,
            "catchment_vector_area_km2": float(poly.area / 1e6),
            "catchment_minus_inc_area_km2": float(poly.area / 1e6 - inc),
            "total_area_km2": tot,
            "expected_total_area_km2": expected_tot,
            "total_area_recurrence_error_km2": area_err,
            "total_area_recurrence_relative_error": area_rel,
            "total_area_recurrence_pass": abs(area_err) <= AREA_ABS_TOL or area_rel <= AREA_REL_TOL,
            "downstream_reach": down,
            "downstream_total_area_km2": down_tot,
            "downstream_area_nondecreasing": bool(down is None or down_tot + AREA_ABS_TOL >= tot),
            "downstream_catchment_gap_m": down_catch_gap,
            "own_catchment_length_fraction_raw": raw_len / line.length if line.length else np.nan,
            "own_catchment_length_fraction_buffer60m": buffered_len / line.length if line.length else np.nan,
            "outside_own_catchment_length_km_buffer60m": (line.length - buffered_len) / 1000,
            "largest_other_catchment_reach_id": largest_other_id,
            "largest_other_catchment_length_km": largest_other_len / 1000,
            "largest_other_catchment_fraction": largest_other_len / line.length if line.length else np.nan,
            "start_in_own_catchment_buffer60m": bool(poly_buffer.covers(endpoint(line, True))),
            "end_in_own_catchment_buffer60m": bool(poly_buffer.covers(endpoint(line, False))),
        })
    return pd.DataFrame(rows)


def source_reconciliation(reaches: gpd.GeoDataFrame, source: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = source.copy()
    src["src_id"] = src["NAME"].astype(str)
    input_group = src.assign(length_km=src.geometry.length / 1000).groupby("src_id", as_index=False).agg(
        input_feature_count=("src_id", "size"), input_total_length_km=("length_km", "sum")
    )
    topo_group = reaches.assign(topo_length_km=reaches.geometry.length / 1000).groupby("src_id", as_index=False).agg(
        topology_reach_count=("src_id", "size"), topology_total_length_km=("topo_length_km", "sum")
    )
    merged = input_group.merge(topo_group, on="src_id", how="outer")
    merged["length_difference_km"] = merged["topology_total_length_km"] - merged["input_total_length_km"]
    merged["length_relative_difference"] = merged["length_difference_km"] / merged["input_total_length_km"]
    per_reach = reaches[["reach_id", "src_id"]].merge(merged, on="src_id", how="left")
    return merged.sort_values("length_relative_difference", key=lambda s: s.abs(), ascending=False), per_reach


def station_audit(stations: gpd.GeoDataFrame, reaches: gpd.GeoDataFrame, catches: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = stations.copy()
    st["station"] = st["Station"].map(norm_name)
    st["coord_key"] = st.geometry.x.round(2).astype(str) + "," + st.geometry.y.round(2).astype(str)
    coord_counts = st.groupby("coord_key")["station"].agg(list).to_dict()
    r = reaches.set_index("reach_id")
    c = catches.set_index("reach_id")
    rows = []
    for rec in st.itertuples(index=False):
        point = rec.geometry
        distances = reaches.geometry.distance(point)
        min_idx = distances.idxmin()
        nearest_rid = int(reaches.loc[min_idx, "reach_id"])
        containing = catches.loc[catches.geometry.covers(point), "reach_id"].astype(int).tolist()
        nearest_poly = c.loc[nearest_rid].geometry
        rows.append({
            "station": rec.station,
            "lon": float(getattr(rec, "Lon_dd")),
            "lat": float(getattr(rec, "Lat_dd")),
            "nearest_reach_id": nearest_rid,
            "nearest_reach_name": str(r.loc[nearest_rid, "src_id"]),
            "nearest_reach_distance_m": float(distances.loc[min_idx]),
            "containing_incremental_catchment_count": len(containing),
            "containing_incremental_catchment_ids": "|".join(map(str, containing)),
            "nearest_reach_catchment_contains_station": nearest_rid in containing,
            "distance_to_nearest_reach_catchment_m": 0.0 if nearest_poly.covers(point) else float(point.distance(nearest_poly)),
            "distance_to_nearest_catchment_boundary_m": float(point.distance(nearest_poly.boundary)),
            "same_coordinate_station_count": len(coord_counts[rec.coord_key]),
            "same_coordinate_station_names": "|".join(coord_counts[rec.coord_key]),
            "geometry": point,
        })
    spatial = gpd.GeoDataFrame(rows, geometry="geometry", crs=stations.crs)

    mapping = pd.read_csv(STATION_MATCH_PATH, encoding="utf-8-sig")
    mapping["station"] = mapping["station_norm"].map(norm_name)
    mapping["assigned_reach_id"] = pd.to_numeric(mapping["reach_id"], errors="coerce").astype("Int64")
    mapped = spatial.drop(columns="geometry").merge(
        mapping[[c for c in ["station", "assigned_reach_id", "match_method", "snap_distance_m", "used", "mapping_issue"] if c in mapping.columns]],
        on="station", how="left"
    )
    assigned_rows = []
    for row in mapped.itertuples(index=False):
        assigned = int(row.assigned_reach_id) if pd.notna(row.assigned_reach_id) else None
        point = spatial.loc[spatial.station.eq(row.station), "geometry"].iloc[0]
        if assigned is not None and assigned in r.index:
            assigned_distance = float(point.distance(r.loc[assigned].geometry))
            assigned_contains = bool(c.loc[assigned].geometry.covers(point))
            assigned_catch_distance = 0.0 if assigned_contains else float(point.distance(c.loc[assigned].geometry))
        else:
            assigned_distance, assigned_contains, assigned_catch_distance = np.nan, False, np.nan
        assigned_rows.append({
            **row._asdict(),
            "assigned_reach_distance_m_recomputed": assigned_distance,
            "assigned_reach_is_geometric_nearest": bool(assigned == row.nearest_reach_id) if assigned is not None else False,
            "assigned_reach_catchment_contains_station": assigned_contains,
            "distance_to_assigned_catchment_m": assigned_catch_distance,
        })
    mapped = pd.DataFrame(assigned_rows)
    return spatial, mapped


def classify_reaches(reaches, graph_a, catch_a, source_per, intersections, station_mapped, targets):
    table = reaches.drop(columns="geometry").merge(graph_a, on=["reach_id", "src_id"], how="left")
    table = table.merge(catch_a, on=["reach_id", "src_id"], how="left", suffixes=("", "_catch"))
    table = table.merge(source_per, on=["reach_id", "src_id"], how="left")
    unmodeled = intersections[intersections.unmodeled_intersection].copy()
    endpoint_counts = defaultdict(int)
    crossing_counts = defaultdict(int)
    overlap_counts = defaultdict(int)
    terminal_outlet_counts = defaultdict(int)
    receiving_terminal_counts = defaultdict(int)
    for x in unmodeled.itertuples(index=False):
        for rid in [int(x.reach_id_a), int(x.reach_id_b)]:
            crossing_counts[rid] += 1
            if x.unmodeled_endpoint_intersection:
                endpoint_counts[rid] += 1
            if x.intersection_length_m > 1:
                overlap_counts[rid] += 1
        if x.reach_a_terminal_outlet_intersection:
            terminal_outlet_counts[int(x.reach_id_a)] += 1
            receiving_terminal_counts[int(x.reach_id_b)] += 1
        if x.reach_b_terminal_outlet_intersection:
            terminal_outlet_counts[int(x.reach_id_b)] += 1
            receiving_terminal_counts[int(x.reach_id_a)] += 1
    table["unmodeled_intersection_count"] = table.reach_id.map(crossing_counts).fillna(0).astype(int)
    table["unmodeled_endpoint_intersection_count"] = table.reach_id.map(endpoint_counts).fillna(0).astype(int)
    table["unmodeled_overlap_count"] = table.reach_id.map(overlap_counts).fillna(0).astype(int)
    table["unmodeled_terminal_outlet_count"] = table.reach_id.map(terminal_outlet_counts).fillna(0).astype(int)
    table["receiving_unmodeled_terminal_count"] = table.reach_id.map(receiving_terminal_counts).fillna(0).astype(int)

    selected = pd.read_csv(SELECTED_STATIONS_PATH, encoding="utf-8-sig")
    selected_names = set(selected.station_norm.map(relaxed_station_name))
    mapped = station_mapped[
        station_mapped.assigned_reach_id.notna()
        & station_mapped.station.map(relaxed_station_name).isin(selected_names)
    ].copy()
    mapped["assigned_reach_id"] = mapped.assigned_reach_id.astype(int)
    mapped["station_mapping_hard"] = (
        (mapped.assigned_reach_distance_m_recomputed > STATION_HIGH_M)
        | (~mapped.assigned_reach_is_geometric_nearest)
        | ((~mapped.assigned_reach_catchment_contains_station) & (mapped.distance_to_assigned_catchment_m > STATION_REVIEW_M))
    )
    station_summary = mapped.groupby("assigned_reach_id").agg(
        mapped_station_count=("station", "size"),
        mapped_station_names=("station", lambda s: "|".join(sorted(set(s)))),
        station_mapping_hard_count=("station_mapping_hard", "sum"),
        max_station_to_assigned_reach_m=("assigned_reach_distance_m_recomputed", "max"),
    ).reset_index().rename(columns={"assigned_reach_id": "reach_id"})
    table = table.merge(station_summary, on="reach_id", how="left")
    for c in ["mapped_station_count", "station_mapping_hard_count"]:
        table[c] = table[c].fillna(0).astype(int)
    table["mapped_station_names"] = table.mapped_station_names.fillna("")

    target_ids = set(targets.reach_id.astype(int))
    table["legacy_lowflow_target"] = table.reach_id.isin(target_ids)
    table["hard_graph_contradiction"] = (
        (table.start_to_fnode_m > ENDPOINT_TOL_M)
        | (table.end_to_tnode_m > ENDPOINT_TOL_M)
        | (~table.downstream_fnode_matches_tnode)
        | (table.downstream_endpoint_gap_m > ENDPOINT_TOL_M)
        | table.self_loop
        | (~table.terminal_consistent_with_graph)
        | (~table.headwater_consistent_with_graph)
        | (~table.total_area_recurrence_pass)
        | (~table.downstream_area_nondecreasing)
        | (table.unmodeled_terminal_outlet_count > 0)
    )
    table["geometry_graph_contradiction"] = table.unmodeled_endpoint_intersection_count > 0
    table["catchment_high_risk"] = table.own_catchment_length_fraction_buffer60m < OWN_CATCH_HIGH
    table["catchment_review"] = table.own_catchment_length_fraction_buffer60m < OWN_CATCH_PASS
    table["source_length_high_risk"] = table.length_relative_difference.abs() > 0.01
    table["source_length_review"] = table.length_relative_difference.abs() > 0.001
    table["station_mapping_hard"] = table.station_mapping_hard_count > 0

    def classify(row):
        if row.hard_graph_contradiction:
            return "CONFIRMED_INTERNAL_TOPOLOGY_CONTRADICTION"
        if row.geometry_graph_contradiction or row.catchment_high_risk or row.station_mapping_hard or row.source_length_high_risk:
            return "HIGH_PRIORITY_GEOGRAPHIC_REVIEW"
        if row.unmodeled_intersection_count > 0 or row.catchment_review or row.dem_direction_uncertain or row.source_length_review:
            return "LIKELY_GEOMETRIC_OR_RESOLUTION_ARTIFACT"
        return "NO_HARD_TOPOLOGY_FAILURE"

    table["risk_class"] = table.apply(classify, axis=1)
    weights = {
        "CONFIRMED_INTERNAL_TOPOLOGY_CONTRADICTION": 4,
        "HIGH_PRIORITY_GEOGRAPHIC_REVIEW": 3,
        "LIKELY_GEOMETRIC_OR_RESOLUTION_ARTIFACT": 2,
        "NO_HARD_TOPOLOGY_FAILURE": 1,
    }
    table["risk_rank"] = table.risk_class.map(weights)
    table["risk_reasons"] = table.apply(lambda x: "|".join([
        name for flag, name in [
            (x.hard_graph_contradiction, "graph_geometry_or_area_internal_contradiction"),
            (x.unmodeled_terminal_outlet_count > 0, "terminal_outlet_touches_unconnected_reach"),
            (x.receiving_unmodeled_terminal_count > 0, "receives_unconnected_terminal_outlet"),
            (x.geometry_graph_contradiction, "unmodeled_endpoint_intersection"),
            (x.catchment_high_risk, "reach_outside_own_catchment"),
            (x.station_mapping_hard, "station_mapping_hard_risk"),
            (x.source_length_high_risk, "source_length_not_reconciled"),
            (x.dem_direction_uncertain, "dem_direction_uncertain"),
            (x.unmodeled_intersection_count > 0, "unmodeled_2d_intersection"),
        ] if flag
    ]), axis=1)
    return table.sort_values(["risk_rank", "catchment_high_risk", "station_mapping_hard", "reach_id"], ascending=[False, False, False, True])


def make_figures(reaches, catches, basin, station_spatial, reach_risk, intersections):
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    colors = {
        "CONFIRMED_INTERNAL_TOPOLOGY_CONTRADICTION": "#d97706",
        "HIGH_PRIORITY_GEOGRAPHIC_REVIEW": "#b45309",
        "LIKELY_GEOMETRIC_OR_RESOLUTION_ARTIFACT": "#4f6f8f",
        "NO_HARD_TOPOLOGY_FAILURE": "#b8c2cc",
    }
    mapped = reaches.merge(reach_risk[["reach_id", "risk_class", "legacy_lowflow_target"]], on="reach_id", how="left")
    fig, ax = plt.subplots(figsize=(14, 8.5))
    basin.boundary.plot(ax=ax, color="#303841", linewidth=0.5)
    for cls in colors:
        part = mapped[mapped.risk_class.eq(cls)]
        if len(part):
            part.plot(ax=ax, color=colors[cls], linewidth=2.1 if cls != "NO_HARD_TOPOLOGY_FAILURE" else 0.65, label=f"{cls} ({len(part)})")
    targets = mapped[mapped.legacy_lowflow_target]
    if len(targets):
        targets.geometry.interpolate(0.5, normalized=True).plot(ax=ax, color="none", edgecolor="#111827", marker="o", markersize=24, linewidth=0.7, label="28 legacy low-flow reaches")
    badx = intersections[intersections.unmodeled_intersection]
    if len(badx):
        badx.plot(ax=ax, color="#f3c969", marker="x", markersize=18, label="unmodeled 2D intersections")
    ax.set_title("PRB 230个Reach全网拓扑风险分级", loc="left", fontsize=16, weight="bold")
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_full_network_topology_risk_map.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / "01_full_network_topology_risk_map.svg", bbox_inches="tight")
    plt.close(fig)

    checks = pd.DataFrame({
        "check": ["内部图结构矛盾", "未建模端点相交", "Catchment覆盖<90%", "站点映射硬风险", "DEM方向不确定", "未建模二维相交"],
        "count": [
            int(reach_risk.hard_graph_contradiction.sum()),
            int(reach_risk.geometry_graph_contradiction.sum()),
            int(reach_risk.catchment_high_risk.sum()),
            int(reach_risk.station_mapping_hard.sum()),
            int(reach_risk.dem_direction_uncertain.sum()),
            int((reach_risk.unmodeled_intersection_count > 0).sum()),
        ],
    }).sort_values("count")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(checks.check, checks["count"], color="#4f6f8f", edgecolor="#263746")
    for i, v in enumerate(checks["count"]):
        ax.text(v + 0.4, i, str(v), va="center", fontsize=10)
    ax.set_xlim(0, max(checks["count"].max() * 1.2, 5))
    ax.set_title("全量审计各类风险触发Reach数", loc="left", fontsize=14, weight="bold")
    ax.set_xlabel("Reach数（n=230；同一Reach可触发多项）")
    ax.grid(axis="x", color="#d9dee3", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_topology_check_counts.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / "02_topology_check_counts.svg", bbox_inches="tight")
    plt.close(fig)

    plot_ids = reach_risk[reach_risk.risk_class.isin(["CONFIRMED_INTERNAL_TOPOLOGY_CONTRADICTION", "HIGH_PRIORITY_GEOGRAPHIC_REVIEW"])].head(9).reach_id.tolist()
    if plot_ids:
        ncols = 3
        nrows = math.ceil(len(plot_ids) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, rid in zip(axes, plot_ids):
            line = reaches[reaches.reach_id.eq(rid)]
            poly = catches[catches.reach_id.eq(rid)]
            bounds = line.geometry.iloc[0].buffer(max(10000, line.geometry.iloc[0].length * 0.08)).bounds
            nearby_c = catches.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            nearby_r = reaches.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            nearby_c.plot(ax=ax, facecolor="#edf1f4", edgecolor="#c4ccd3", linewidth=0.4)
            poly.plot(ax=ax, facecolor="#f6d9a6", edgecolor="#b45309", alpha=0.55)
            nearby_r.plot(ax=ax, color="#74889c", linewidth=0.7)
            line.plot(ax=ax, color="#8b2e1e", linewidth=2.4)
            st = station_spatial[station_spatial.nearest_reach_id.eq(rid)]
            if len(st):
                st.plot(ax=ax, color="#111827", markersize=18)
            row = reach_risk[reach_risk.reach_id.eq(rid)].iloc[0]
            ax.set_title(f"Reach {rid} · {row.src_id}\n{row.risk_reasons}", fontsize=9)
            ax.set_axis_off()
        for ax in axes[len(plot_ids):]:
            ax.set_visible(False)
        fig.suptitle("最高风险Reach局部几何—Catchment关系", x=0.01, ha="left", fontsize=15, weight="bold")
        fig.tight_layout()
        fig.savefig(FIGURES / "03_high_risk_local_panels.png", dpi=220, bbox_inches="tight")
        fig.savefig(FIGURES / "03_high_risk_local_panels.svg", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    for d in [REPORTS, FIGURES, VECTORS]:
        d.mkdir(parents=True, exist_ok=True)
    reaches, catches, nodes, source, basin, stations, summary, edges, node_summary, manual = load_inputs()
    if not (len(reaches) == len(catches) == len(summary) == len(edges) == 230):
        raise RuntimeError("Frozen topology layers are not all 230 rows")
    if reaches.reach_id.duplicated().any() or catches.reach_id.duplicated().any():
        raise RuntimeError("Duplicate reach_id detected")

    graph, graph_rows, graph_meta = graph_audit(reaches, nodes, summary, edges)
    intersections = pairwise_intersections(reaches, edges)
    catch_rows = catchment_audit(reaches, catches, graph)
    source_rows, source_per = source_reconciliation(reaches, source)
    station_spatial, station_mapped = station_audit(stations, reaches, catches)
    registry = pd.read_csv(SIGNAL_PATH, encoding="utf-8-sig")
    targets = registry[registry.legacy_canonical_membership.map(bool_value)].copy()
    targets["reach_id"] = pd.to_numeric(targets.reach_id, errors="raise").astype(int)
    risk = classify_reaches(reaches, graph_rows, catch_rows, source_per, intersections, station_mapped, targets)

    graph_rows.to_csv(REPORTS / "reach_graph_endpoint_dem_audit.csv", index=False, encoding="utf-8-sig")
    intersections.drop(columns="geometry").to_csv(REPORTS / "reach_pairwise_intersections.csv", index=False, encoding="utf-8-sig")
    catch_rows.to_csv(REPORTS / "reach_catchment_area_audit.csv", index=False, encoding="utf-8-sig")
    source_rows.to_csv(REPORTS / "source_stream_length_reconciliation.csv", index=False, encoding="utf-8-sig")
    station_mapped.to_csv(REPORTS / "all_station_reach_catchment_audit.csv", index=False, encoding="utf-8-sig")
    risk.to_csv(REPORTS / "full_reach_risk_registry.csv", index=False, encoding="utf-8-sig")

    terminal_ids = [n for n in graph if graph.out_degree(n) == 0]
    terminal_area_sum = float(reaches.set_index("reach_id").loc[terminal_ids, "tot_km2"].sum())
    inc_area_sum = float(reaches.inc_km2.sum())
    target_compare = risk.groupby("legacy_lowflow_target").agg(
        reach_count=("reach_id", "size"),
        internal_contradiction_count=("hard_graph_contradiction", "sum"),
        high_geographic_review_count=("risk_class", lambda s: int((s == "HIGH_PRIORITY_GEOGRAPHIC_REVIEW").sum())),
        catchment_high_risk_count=("catchment_high_risk", "sum"),
        station_mapping_hard_count=("station_mapping_hard", "sum"),
        dem_uncertain_count=("dem_direction_uncertain", "sum"),
    ).reset_index()
    for c in target_compare.columns[2:]:
        target_compare[c.replace("_count", "_rate")] = target_compare[c] / target_compare.reach_count
    target_compare.to_csv(REPORTS / "legacy_lowflow_vs_other_topology_risk.csv", index=False, encoding="utf-8-sig")

    risk_counts = risk.risk_class.value_counts().to_dict()
    positive = risk[risk.reach_id.eq(199)].iloc[0].to_dict()
    terminal = {
        "run_id": RUN.name,
        "graph": graph_meta,
        "area": {
            "incremental_area_sum_km2": inc_area_sum,
            "terminal_total_area_sum_km2": terminal_area_sum,
            "difference_km2": terminal_area_sum - inc_area_sum,
        },
        "risk_class_counts": risk_counts,
        "unmodeled_intersection_pairs": int(intersections.unmodeled_intersection.sum()) if len(intersections) else 0,
        "unmodeled_endpoint_intersection_pairs": int(intersections.unmodeled_endpoint_intersection.sum()) if len(intersections) else 0,
        "low_dem_confidence_reaches": int(graph_rows.dem_direction_uncertain.sum()),
        "station_count": int(len(station_spatial)),
        "mapped_station_count": int(station_mapped.assigned_reach_id.notna().sum()),
        "positive_control_reach_199": {k: positive[k] for k in ["reach_id", "src_id", "risk_class", "risk_reasons", "own_catchment_length_fraction_buffer60m", "largest_other_catchment_reach_id", "largest_other_catchment_fraction", "unmodeled_endpoint_intersection_count"]},
        "positive_control_detected": positive["risk_class"] in {"CONFIRMED_INTERNAL_TOPOLOGY_CONTRADICTION", "HIGH_PRIORITY_GEOGRAPHIC_REVIEW"},
    }
    (REPORTS / "terminal_summary.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    make_figures(reaches, catches, basin, station_spatial, risk, intersections)

    gpkg = VECTORS / "full_topology_audit.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    reach_out = reaches.merge(risk.drop(columns=[c for c in risk.columns if c in reaches.columns and c != "reach_id"]), on="reach_id", how="left")
    write_layer(reach_out, gpkg, "reaches_risk", True)
    write_layer(catches.merge(risk[["reach_id", "risk_class", "risk_reasons", "legacy_lowflow_target"]], on="reach_id", how="left"), gpkg, "catchments_risk", False)
    write_layer(station_spatial, gpkg, "all_stations_nearest", False)
    if len(intersections):
        write_layer(intersections, gpkg, "reach_intersections", False)

    source_paths = [REACH_PATH, CATCH_PATH, NODE_PATH, REACH_SUMMARY_PATH, EDGE_PATH, NODE_SUMMARY_PATH, MANUAL_PATH, SOURCE_STREAM_PATH, BASIN_PATH, STATIONS_PATH, STATION_MATCH_PATH, SAME_REACH_PATH, SELECTED_STATIONS_PATH, SIGNAL_PATH]
    manifest = {
        "run_id": RUN.name,
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
            "platform": platform.platform(),
            "geopandas": gpd.__version__,
            "pandas": pd.__version__,
            "networkx": nx.__version__,
        },
        "thresholds": {
            "endpoint_tolerance_m": ENDPOINT_TOL_M,
            "catchment_buffer_m": CATCH_BUFFER_M,
            "station_review_m": STATION_REVIEW_M,
            "station_high_m": STATION_HIGH_M,
            "station_severe_m": STATION_SEVERE_M,
            "own_catchment_pass": OWN_CATCH_PASS,
            "own_catchment_high": OWN_CATCH_HIGH,
        },
        "sources": [],
    }
    for p in source_paths:
        for f in vector_components(p):
            manifest["sources"].append({"path": str(f), "bytes": f.stat().st_size, "sha256": sha256(f)})
    (RUN / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(terminal, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
