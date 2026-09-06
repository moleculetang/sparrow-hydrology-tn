from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import Point


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _first_vector_from_node(line, node_point: Point, tolerance: float = 1.0) -> tuple[float, float] | None:
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    d_start = Point(coords[0]).distance(node_point)
    d_end = Point(coords[-1]).distance(node_point)
    if d_start <= d_end:
        origin = coords[0]
        for coord in coords[1:]:
            dx = coord[0] - origin[0]
            dy = coord[1] - origin[1]
            if math.hypot(dx, dy) > tolerance:
                return dx, dy
    else:
        origin = coords[-1]
        for coord in reversed(coords[:-1]):
            dx = coord[0] - origin[0]
            dy = coord[1] - origin[1]
            if math.hypot(dx, dy) > tolerance:
                return dx, dy
    return None


def _angle_between(v1: tuple[float, float] | None, v2: tuple[float, float] | None) -> float | None:
    if v1 is None or v2 is None:
        return None
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0:
        return None
    cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def check(root: Path, config: Path, preview: bool) -> None:
    with config.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    tables = root / cfg["outputs"]["tables_dir"]
    result_vectors = root / cfg["outputs"].get("vectors_dir", cfg["outputs"].get("gis_dir", "results/vectors"))
    work_vectors = root / cfg["outputs"].get("work_dir", "work") / "vectors"
    suffix = "_preview" if preview else ""
    out_csv = tables / f"split_node_anomalies{suffix}.csv"
    out_shp = work_vectors / f"split_node_anomalies{suffix}.shp"

    nodes = pd.read_csv(tables / ("reach_node_preview_nodes.csv" if preview else "node_summary.csv"), encoding="utf-8-sig")
    edges = pd.read_csv(tables / ("reach_node_preview_edges.csv" if preview else "topology_edges.csv"), encoding="utf-8-sig")
    reaches = gpd.read_file(result_vectors / ("reaches_preview.shp" if preview else "reaches_topology.shp"))
    node_points = gpd.read_file(result_vectors / ("nodes_preview.shp" if preview else "nodes.shp"))

    split_nodes = nodes[nodes["out_deg"] > 1].copy()
    rows = []
    for node in split_nodes.itertuples(index=False):
        node_id = int(node.node_id)
        incoming = edges[edges["tnode"].astype(int) == node_id]
        outgoing = edges[edges["fnode"].astype(int) == node_id].copy()
        point_row = node_points[node_points["node_id"].astype(int) == node_id]
        node_point = point_row.geometry.iloc[0] if len(point_row) else Point(float(node.x), float(node.y))
        out_detail = []
        vectors = []
        for edge in outgoing.itertuples(index=False):
            reach = reaches[reaches["reach_id"].astype(int) == int(edge.reach_id)]
            geom = reach.geometry.iloc[0] if len(reach) else None
            length_km = float(reach["length_km"].iloc[0]) if len(reach) and "length_km" in reach.columns else None
            vector = _first_vector_from_node(geom, node_point) if geom is not None else None
            vectors.append(vector)
            out_detail.append(f"{int(edge.reach_id)}:{edge.src_id}:{length_km:.3f}km" if length_km is not None else f"{int(edge.reach_id)}:{edge.src_id}")
        angle = _angle_between(vectors[0], vectors[1]) if len(vectors) == 2 else None
        anomaly = bool(int(node.in_deg) == 0 and int(node.out_deg) > 1)
        rows.append(
            {
                "node_id": node_id,
                "node_type": node.node_type,
                "in_deg": int(node.in_deg),
                "out_deg": int(node.out_deg),
                "incoming_reaches": ",".join(str(int(v)) for v in incoming["reach_id"].tolist()),
                "outgoing_reaches": "; ".join(out_detail),
                "outgoing_angle_deg": angle,
                "source_split_anomaly": int(anomaly),
                "x": float(node.x),
                "y": float(node.y),
                "geometry": node_point,
            }
        )

    if rows:
        out = gpd.GeoDataFrame(rows, geometry="geometry", crs=node_points.crs)
    else:
        out = gpd.GeoDataFrame(
            {
                "node_id": [],
                "node_type": [],
                "in_deg": [],
                "out_deg": [],
                "incoming_reaches": [],
                "outgoing_reaches": [],
                "outgoing_angle_deg": [],
                "source_split_anomaly": [],
                "x": [],
                "y": [],
                "geometry": [],
            },
            geometry="geometry",
            crs=node_points.crs,
        )
    out.drop(columns="geometry").to_csv(out_csv, index=False, encoding="utf-8-sig")
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        path = out_shp.with_suffix(suffix)
        if path.exists():
            path.unlink()
    if len(out):
        out.to_file(out_shp, encoding="UTF-8")

    print(f"split nodes: {len(out)}")
    print(f"source split anomalies: {int(out['source_split_anomaly'].sum()) if len(out) else 0}")
    if len(out):
        print(out.drop(columns="geometry").to_string(index=False))
    print(out_csv)
    print(out_shp)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose topology split nodes, especially source nodes with multiple outgoing reaches.")
    parser.add_argument("--root", type=Path, default=_root())
    parser.add_argument("--config", type=Path, default=_root() / "configs" / "reach_topology.yaml")
    parser.add_argument("--preview", action="store_true", help="Check reach/node preview outputs instead of full topology outputs.")
    args = parser.parse_args()
    check(args.root.resolve(), args.config.resolve(), args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
