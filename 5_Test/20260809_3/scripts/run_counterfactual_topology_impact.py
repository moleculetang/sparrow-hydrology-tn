from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORTS = RUN / "reports"
VECTORS = RUN / "outputs" / "vectors"
TOPO = ROOT / "0_reach_topology"
BASE = ROOT / "5_Test" / "20260805_2"
SIGNAL_RUN = ROOT / "5_Test" / "20260730_12"

EDGE_PATH = TOPO / "results" / "tables" / "topology_edges.csv"
REACH_PATH = VECTORS / "full_topology_audit.gpkg"
INTERSECTION_PATH = REPORTS / "reach_pairwise_intersections.csv"
SELECTED_STATIONS_PATH = BASE / "inputs" / "source_metadata" / "selected_representative_stations.csv"
SIGNAL_PATH = SIGNAL_RUN / "reports" / "signal_registry" / "canonical_signal_registry.csv"


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_candidate_edges(intersections: pd.DataFrame) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for row in intersections.itertuples(index=False):
        if bool_value(row.reach_a_terminal_outlet_intersection):
            pairs.append((int(row.reach_id_a), int(row.reach_id_b)))
        if bool_value(row.reach_b_terminal_outlet_intersection):
            pairs.append((int(row.reach_id_b), int(row.reach_id_a)))
    return sorted(set(pairs))


def build_graph(edges: pd.DataFrame, reach_ids: set[int]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(reach_ids))
    for row in edges.itertuples(index=False):
        if pd.notna(row.downstream_reach):
            graph.add_edge(int(row.reach_id), int(row.downstream_reach))
    return graph


def accumulated_area(graph: nx.DiGraph, incremental: dict[int, float]) -> dict[int, float]:
    result: dict[int, float] = {}
    for rid in nx.topological_sort(graph):
        result[rid] = float(incremental[rid]) + sum(result[u] for u in graph.predecessors(rid))
    return result


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    reaches = gpd.read_file(REACH_PATH, layer="reaches_risk", engine="pyogrio")
    edges = pd.read_csv(EDGE_PATH, encoding="utf-8-sig")
    intersections = pd.read_csv(INTERSECTION_PATH, encoding="utf-8-sig")
    selected = pd.read_csv(SELECTED_STATIONS_PATH, encoding="utf-8-sig")
    registry = pd.read_csv(SIGNAL_PATH, encoding="utf-8-sig")

    reaches["reach_id"] = pd.to_numeric(reaches.reach_id, errors="raise").astype(int)
    edges["reach_id"] = pd.to_numeric(edges.reach_id, errors="raise").astype(int)
    selected["reach_id"] = pd.to_numeric(selected.reach_id, errors="coerce").astype("Int64")
    registry["reach_id"] = pd.to_numeric(registry.reach_id, errors="coerce").astype("Int64")

    reach_ids = set(reaches.reach_id)
    base = build_graph(edges, reach_ids)
    if not nx.is_directed_acyclic_graph(base):
        raise RuntimeError("Frozen graph is not a DAG")
    candidate_edges = load_candidate_edges(intersections)
    expected = {(14, 19), (64, 59), (132, 149), (180, 168), (199, 196)}
    if set(candidate_edges) != expected:
        raise RuntimeError(f"Candidate edge set changed: {candidate_edges}")

    incremental = reaches.set_index("reach_id").inc_km2.astype(float).to_dict()
    source_name = reaches.set_index("reach_id").src_id.astype(str).to_dict()
    stored_total = reaches.set_index("reach_id").tot_km2.astype(float).to_dict()
    base_total = accumulated_area(base, incremental)
    max_base_error = max(abs(base_total[r] - stored_total[r]) for r in reach_ids)
    if max_base_error > 1e-6:
        raise RuntimeError(f"Baseline area recurrence mismatch: {max_base_error}")

    legacy_ids = set(
        registry.loc[registry.legacy_canonical_membership.map(bool_value), "reach_id"].dropna().astype(int)
    )
    selected = selected[selected.reach_id.notna()].copy()
    selected["reach_id"] = selected.reach_id.astype(int)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for source, receiver in candidate_edges:
        trial = base.copy()
        trial.add_edge(source, receiver)
        if not nx.is_directed_acyclic_graph(trial):
            raise RuntimeError(f"Candidate edge {source}->{receiver} creates a cycle")
        trial_total = accumulated_area(trial, incremental)
        impacted = {receiver, *nx.descendants(trial, receiver)}
        impacted_stations = selected[selected.reach_id.isin(impacted)]
        impacted_legacy = sorted(legacy_ids & impacted)
        source_component = set(nx.ancestors(base, source)) | {source}
        receiver_component_before = set(nx.ancestors(base, receiver)) | {receiver}
        for rid in sorted(impacted):
            stations_here = impacted_stations.loc[impacted_stations.reach_id.eq(rid), "station_norm"].astype(str).tolist()
            detail_rows.append(
                {
                    "candidate_source_reach": source,
                    "candidate_receiver_reach": receiver,
                    "affected_reach_id": rid,
                    "affected_reach_name_local_auxiliary": source_name[rid],
                    "baseline_total_area_km2": base_total[rid],
                    "counterfactual_total_area_km2": trial_total[rid],
                    "total_area_delta_km2": trial_total[rid] - base_total[rid],
                    "total_area_delta_percent": 100.0 * (trial_total[rid] - base_total[rid]) / base_total[rid],
                    "legacy_lowflow_target": rid in legacy_ids,
                    "q72_representative_station_count": len(stations_here),
                    "q72_representative_stations": "|".join(stations_here),
                }
            )
        summary_rows.append(
            {
                "candidate_source_reach": source,
                "candidate_source_name_local_auxiliary": source_name[source],
                "candidate_receiver_reach": receiver,
                "candidate_receiver_name_local_auxiliary": source_name[receiver],
                "source_component_reach_count": len(source_component),
                "source_component_total_area_km2": base_total[source],
                "receiver_upstream_reach_count_before": len(receiver_component_before),
                "downstream_affected_reach_count": len(impacted),
                "affected_q72_representative_station_count": len(impacted_stations),
                "affected_q72_representative_stations": "|".join(impacted_stations.station_norm.astype(str).tolist()),
                "affected_legacy_lowflow_reach_count": len(impacted_legacy),
                "affected_legacy_lowflow_reach_ids": "|".join(map(str, impacted_legacy)),
                "weak_components_before": nx.number_weakly_connected_components(base),
                "weak_components_after_single_edge": nx.number_weakly_connected_components(trial),
                "cycle_created": False,
            }
        )

    revised = base.copy()
    revised.add_edges_from(candidate_edges)
    if not nx.is_directed_acyclic_graph(revised):
        raise RuntimeError("Simultaneous candidate edges create a cycle")
    revised_total = accumulated_area(revised, incremental)
    all_impacted = {rid for rid in reach_ids if abs(revised_total[rid] - base_total[rid]) > 1e-8}
    impacted_selected = selected[selected.reach_id.isin(all_impacted)]
    impacted_legacy = sorted(legacy_ids & all_impacted)

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    summary.to_csv(REPORTS / "candidate_edge_counterfactual_summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(REPORTS / "candidate_edge_downstream_impact.csv", index=False, encoding="utf-8-sig")

    simultaneous = {
        "run_id": RUN.name,
        "decision_basis": "candidate edges are geometry-derived; real-world acceptance requires coordinate-based external corroboration",
        "candidate_edges": [{"source": a, "receiver": b} for a, b in candidate_edges],
        "baseline": {
            "reach_count": base.number_of_nodes(),
            "edge_count": base.number_of_edges(),
            "weak_components": nx.number_weakly_connected_components(base),
            "terminal_count": sum(base.out_degree(n) == 0 for n in base),
            "max_area_recurrence_error_km2": max_base_error,
        },
        "counterfactual": {
            "edge_count": revised.number_of_edges(),
            "weak_components": nx.number_weakly_connected_components(revised),
            "terminal_count": sum(revised.out_degree(n) == 0 for n in revised),
            "cycle_count": len(list(nx.simple_cycles(revised))),
            "affected_reach_count": len(all_impacted),
            "affected_reach_ids": sorted(all_impacted),
            "affected_q72_representative_station_count": int(len(impacted_selected)),
            "affected_q72_representative_stations": impacted_selected.station_norm.astype(str).tolist(),
            "affected_legacy_lowflow_reach_count": len(impacted_legacy),
            "affected_legacy_lowflow_reach_ids": impacted_legacy,
        },
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
    }
    (REPORTS / "simultaneous_candidate_edge_counterfactual.json").write_text(
        json.dumps(simultaneous, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(simultaneous, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
