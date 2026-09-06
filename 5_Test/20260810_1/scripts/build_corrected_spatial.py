from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import unicodedata
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
ROOT = Path(r"E:\SPARROW")
LEGACY_SPATIAL = RUN / "backup_before_correction" / "legacy_baseline" / "spatial"
LEGACY_TOPOLOGY = RUN / "backup_before_correction" / "legacy_baseline" / "topology" / "topology_edges.csv"
OUT = RUN / "inputs" / "spatial_corrected"
REPORT = RUN / "reports" / "spatial_correction"
REGISTRY = RUN / "inputs" / "registry_corrected"
FLOW_DIR = ROOT / "0_reach_topology" / "work" / "rasters" / "flow_dir.tif"
BASIN = ROOT / "0_reach_topology" / "data" / "processed" / "vector" / "prb_boundary.shp"
WHITEBOX = RUN / "tools" / "whitebox_tools.exe"

CORRECTIONS = [(14, 19), (64, 59), (132, 149), (180, 168), (199, 196)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_command(args: list[str]) -> None:
    completed = subprocess.run(args, cwd=RUN, text=True, encoding="utf-8", errors="replace", capture_output=True)
    log_name = Path(args[0]).stem + "_" + str(abs(hash(tuple(args))) % 1_000_000)
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    (RUN / "logs" / f"{log_name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (RUN / "logs" / f"{log_name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(args)}\n{completed.stderr[-2000:]}")


def parse_downstream(value: object) -> int | None:
    if pd.isna(value) or not str(value).strip():
        return None
    return int(float(str(value).split(",")[0].strip()))


def graph_totals(graph: nx.DiGraph, inc: dict[int, float]) -> dict[int, float]:
    totals = {rid: float(inc[rid]) for rid in graph.nodes}
    for rid in nx.topological_sort(graph):
        for down in graph.successors(rid):
            totals[int(down)] += totals[int(rid)]
    return totals


def copy_station_shapefile() -> Path:
    source = LEGACY_SPATIAL / "PRB水文站_全部.shp"
    for part in source.parent.glob(source.stem + ".*"):
        shutil.copy2(part, OUT / part.name)
    return OUT / source.name


def corrected_graph(reaches: gpd.GeoDataFrame, topo: pd.DataFrame) -> tuple[nx.DiGraph, pd.DataFrame]:
    reach_ids = set(reaches["reach_id"].astype(int))
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(reach_ids))
    downstream = {int(r.reach_id): parse_downstream(r.downstream_reach) for r in topo.itertuples(index=False)}
    for rid, down in downstream.items():
        if down is not None:
            graph.add_edge(rid, down)

    geometry = reaches.set_index("reach_id").geometry
    rows = []
    for source, receiver in CORRECTIONS:
        if downstream.get(source) is not None:
            raise RuntimeError(f"Correction source {source} is not terminal in legacy topology")
        source_end = Point(geometry.loc[source].coords[-1])
        receiver_line = geometry.loc[receiver]
        distance = float(source_end.distance(receiver_line))
        if distance > 0.1:
            raise RuntimeError(f"Correction {source}->{receiver} geometry gap {distance:.6f} m exceeds 0.1 m")
        fraction = float(receiver_line.project(source_end) / receiver_line.length)
        graph.add_edge(source, receiver)
        downstream[source] = receiver
        rows.append({
            "source_reach": source,
            "receiver_reach": receiver,
            "junction_type": "interior_confluence",
            "junction_x": float(source_end.x),
            "junction_y": float(source_end.y),
            "receiver_fraction": fraction,
            "endpoint_to_receiver_m": distance,
            "evidence": "20260809_3 coordinate geometry plus external river-network cross-check",
        })
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Corrected topology contains a directed cycle")
    return graph, pd.DataFrame(rows)


def update_graph_fields(reaches: gpd.GeoDataFrame, topo: pd.DataFrame, graph: nx.DiGraph) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    order = list(nx.topological_sort(graph))
    hydseq = {rid: idx + 1 for idx, rid in enumerate(order)}
    upstream = {rid: sorted(int(x) for x in graph.predecessors(rid)) for rid in graph.nodes}
    downstream = {rid: next(iter(graph.successors(rid)), None) for rid in graph.nodes}

    out = reaches.copy()
    out["hydseq"] = out["reach_id"].map(hydseq).astype(int)
    out["down_rch"] = out["reach_id"].map(downstream).astype("Int64")
    out["up_reaches"] = out["reach_id"].map(lambda rid: ",".join(map(str, upstream[int(rid)])))
    out["terminal"] = out["reach_id"].map(lambda rid: int(graph.out_degree(int(rid)) == 0))
    out["headwater"] = out["reach_id"].map(lambda rid: int(graph.in_degree(int(rid)) == 0))
    for col in ["target", "termflag"]:
        if col in out.columns:
            out[col] = out["terminal"]

    table = topo.set_index("reach_id").reindex(sorted(graph.nodes)).reset_index()
    table["downstream_reach"] = table["reach_id"].map(downstream).astype("Int64")
    table["upstream_reaches"] = table["reach_id"].map(lambda rid: ",".join(map(str, upstream[int(rid)])))
    table["hydseq"] = table["reach_id"].map(hydseq).astype(int)
    table["terminal"] = table["reach_id"].map(lambda rid: int(graph.out_degree(int(rid)) == 0))
    return out, table


def rebuild_catchments(reaches: gpd.GeoDataFrame, graph: nx.DiGraph) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, object]]:
    if not FLOW_DIR.exists() or not WHITEBOX.exists():
        raise FileNotFoundError("Required flow_dir or local Whitebox executable is missing")
    work = RUN / "work" / "spatial_correction"
    work.mkdir(parents=True, exist_ok=True)
    seed_lines = work / "reach_seed_lines.shp"
    seed_raster = work / "reach_seed_cells.tif"
    catchment_raster = OUT / "reach_catchments.tif"
    polygonized = work / "reach_catchments_polygonized.gpkg"

    for path in [seed_raster, catchment_raster, polygonized]:
        if path.exists():
            path.unlink()

    seeds = reaches.sort_values("hydseq")[["reach_id", "geometry"]].copy()
    seeds.to_file(seed_lines, encoding="UTF-8")
    run_command(["gdal_create", "-if", str(FLOW_DIR), "-ot", "Int32", "-a_nodata", "0", "-burn", "0", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", str(seed_raster)])
    run_command(["gdal_rasterize", "-a", "reach_id", "-l", seed_lines.stem, str(seed_lines), str(seed_raster)])
    run_command([str(WHITEBOX), "-r=Watershed", f"--d8_pntr={FLOW_DIR}", f"--pour_pts={seed_raster}", f"--output={catchment_raster}"])
    run_command(["gdal", "raster", "polygonize", str(catchment_raster), str(polygonized), "--output-layer", "reach_catchments", "--attribute-name", "reach_id", "--overwrite", "-f", "GPKG"])

    raw = gpd.read_file(polygonized, layer="reach_catchments")
    raw["reach_id"] = pd.to_numeric(raw["reach_id"], errors="coerce")
    raw = raw[raw["reach_id"].between(1, len(reaches))].copy()
    raw["reach_id"] = raw["reach_id"].astype(int)
    cats = raw.dissolve(by="reach_id", as_index=False)
    basin = gpd.read_file(BASIN).to_crs(cats.crs)
    basin_union = basin.geometry.union_all()
    cats["geometry"] = cats.geometry.intersection(basin_union)
    cats = cats[cats.geometry.notna() & ~cats.geometry.is_empty].copy()
    if cats["reach_id"].nunique() != len(reaches):
        missing = sorted(set(reaches["reach_id"].astype(int)) - set(cats["reach_id"].astype(int)))
        raise RuntimeError(f"Catchment rebuild missing reach ids: {missing}")

    covered = cats.geometry.union_all().area
    basin_area = basin_union.area
    coverage_before_fill = float(covered / basin_area) if basin_area > 0 else 0.0
    filled_component_count = 0
    if coverage_before_fill < 0.995:
        uncovered = basin_union.difference(cats.geometry.union_all())
        parts = list(uncovered.geoms) if hasattr(uncovered, "geoms") else [uncovered]
        parts = [part for part in parts if part is not None and not part.is_empty and part.area > 0]
        for part in sorted(parts, key=lambda geom: geom.area, reverse=True):
            probe = part.representative_point()
            distances = cats.geometry.distance(probe)
            target_index = distances.idxmin()
            cats.at[target_index, "geometry"] = cats.at[target_index, "geometry"].union(part)
            filled_component_count += 1

    cats["inc_km2"] = cats.geometry.area / 1_000_000.0
    inc = dict(zip(cats["reach_id"].astype(int), cats["inc_km2"].astype(float)))
    totals = graph_totals(graph, inc)
    cats["tot_km2"] = cats["reach_id"].map(totals).astype(float)
    cats["qa_flag"] = np.where(cats["inc_km2"] > 0, "ok", "zero_area")
    reaches = reaches.drop(columns=[c for c in ["inc_km2", "tot_km2"] if c in reaches.columns])
    reaches = reaches.merge(cats[["reach_id", "inc_km2", "tot_km2"]], on="reach_id", how="left", validate="one_to_one")

    covered = cats.geometry.union_all().area
    coverage = float(covered / basin_area) if basin_area > 0 else 0.0
    if coverage < 0.995:
        raise RuntimeError(f"Corrected catchment coverage {coverage:.6%} is below 99.5%")
    summary = {
        "coverage_ratio": coverage,
        "coverage_ratio_before_fill": coverage_before_fill,
        "filled_uncovered_component_count": filled_component_count,
        "catchment_count": int(cats["reach_id"].nunique()),
        "zero_area_count": int((cats["inc_km2"] <= 0).sum()),
        "incremental_area_sum_km2": float(cats["inc_km2"].sum()),
        "catchment_raster_sha256": sha256(catchment_raster),
    }
    return reaches, cats, summary


def write_spatial_outputs(reaches: gpd.GeoDataFrame, cats: gpd.GeoDataFrame, topo: pd.DataFrame, junctions: pd.DataFrame) -> None:
    for base in [OUT / "reaches_topology", OUT / "reach_catchments"]:
        for part in OUT.glob(base.name + ".*"):
            part.unlink()
    reaches.to_file(OUT / "reaches_topology.shp", encoding="UTF-8")
    cats.to_file(OUT / "reach_catchments.shp", encoding="UTF-8")
    reaches.to_file(OUT / "corrected_spatial.gpkg", layer="reaches_topology", driver="GPKG")
    cats.to_file(OUT / "corrected_spatial.gpkg", layer="reach_catchments", driver="GPKG", mode="a")
    topo.to_csv(OUT / "topology_edges.csv", index=False, encoding="utf-8-sig")
    junctions.to_csv(OUT / "topology_correction_contract.csv", index=False, encoding="utf-8-sig")


def update_station_registry(station_shp: Path, reaches: gpd.GeoDataFrame, cats: gpd.GeoDataFrame) -> dict[str, object]:
    source = RUN / "inputs" / "source_snapshot" / "registry" / "historical_station_reach_match.csv"
    registry = pd.read_csv(source, encoding="utf-8-sig")
    stations = gpd.read_file(station_shp).to_crs(reaches.crs)
    station_col = next(c for c in ["Station", "STATION", "NAME", "station", "name"] if c in stations.columns)

    def norm(value: object) -> str:
        text = unicodedata.normalize("NFKC", str(value)).replace(" ", "").replace("\u3000", "")
        return text.replace("(", "（").replace(")", "）").strip()

    stations["station_norm"] = stations[station_col].map(norm)
    point_by_name = stations.sort_values(station_col).drop_duplicates("station_norm").set_index("station_norm")
    cat_by_id = cats.set_index("reach_id")
    rows = []
    changed = 0
    for idx, row in registry.iterrows():
        name = norm(row.get("station_norm", row.get("station_name", "")))
        old = int(row["reach_id"])
        new = old
        reason = "no_coordinate_keep_frozen"
        containing_ids: list[int] = []
        old_distance = np.nan
        if name in point_by_name.index:
            point = point_by_name.loc[name].geometry
            containing = cats[cats.geometry.contains(point) | cats.geometry.touches(point)]
            containing_ids = sorted(containing["reach_id"].astype(int).tolist())
            if old in cat_by_id.index:
                old_distance = float(point.distance(cat_by_id.loc[old].geometry))
            if old in containing_ids:
                reason = "coordinate_inside_frozen_assigned_catchment"
            elif len(containing_ids) == 1 and ((old == 199 and containing_ids[0] == 196) or (np.isfinite(old_distance) and old_distance > 5000.0)):
                new = containing_ids[0]
                reason = "coordinate_confirmed_reassignment"
            elif len(containing_ids) >= 1:
                reason = "coordinate_boundary_or_ambiguous_keep_frozen"
            else:
                reason = "coordinate_outside_all_catchments_keep_frozen_review"
        if new != old:
            registry.at[idx, "reach_id"] = new
            changed += 1
        rows.append({"station_norm": name, "old_reach_id": old, "new_reach_id": new, "changed": new != old, "containing_reach_ids": "|".join(map(str, containing_ids)), "distance_to_old_catchment_m": old_distance, "decision": reason})
    registry.to_csv(REGISTRY / "historical_station_reach_match.csv", index=False, encoding="utf-8-sig")
    audit = pd.DataFrame(rows)
    audit.to_csv(REPORT / "station_coordinate_mapping_audit.csv", index=False, encoding="utf-8-sig")
    return {"registry_rows": int(len(registry)), "coordinate_reassignments": int(changed), "confirmed_199_to_196_rows": int(((audit.old_reach_id == 199) & (audit.new_reach_id == 196)).sum())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    reaches = gpd.read_file(LEGACY_SPATIAL / "reaches_topology.shp")
    reaches["reach_id"] = pd.to_numeric(reaches["reach_id"], errors="raise").astype(int)
    topo = pd.read_csv(LEGACY_TOPOLOGY, encoding="utf-8-sig")
    topo["reach_id"] = pd.to_numeric(topo["reach_id"], errors="raise").astype(int)
    graph, junctions = corrected_graph(reaches, topo)
    reaches, topo = update_graph_fields(reaches, topo, graph)
    reaches, cats, catchment_summary = rebuild_catchments(reaches, graph)
    totals = dict(zip(reaches["reach_id"].astype(int), reaches["tot_km2"].astype(float)))
    topo["inc_area_km2"] = topo["reach_id"].map(dict(zip(reaches["reach_id"].astype(int), reaches["inc_km2"].astype(float))))
    topo["tot_area_km2"] = topo["reach_id"].map(totals)
    write_spatial_outputs(reaches, cats, topo, junctions)
    station_shp = copy_station_shapefile()
    station_summary = update_station_registry(station_shp, reaches, cats)

    checks = {
        "reach_count_230": len(reaches) == 230,
        "edge_count_216": graph.number_of_edges() == 216,
        "weak_components_14": nx.number_weakly_connected_components(graph) == 14,
        "terminal_count_14": int((reaches["terminal"] == 1).sum()) == 14,
        "acyclic": nx.is_directed_acyclic_graph(graph),
        "five_coordinate_edges_present": all(graph.has_edge(a, b) for a, b in CORRECTIONS),
        "catchment_coverage": catchment_summary["coverage_ratio"] >= 0.995,
        "catchment_count_230": catchment_summary["catchment_count"] == 230,
        "no_zero_catchments": catchment_summary["zero_area_count"] == 0,
    }
    summary = {
        "runtime": RUNTIME,
        "checks": checks,
        "passed": all(checks.values()),
        "edge_count": graph.number_of_edges(),
        "weak_components": nx.number_weakly_connected_components(graph),
        "terminal_count": int((reaches["terminal"] == 1).sum()),
        "catchments": catchment_summary,
        "station_registry": station_summary,
    }
    (REPORT / "corrected_topology_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not summary["passed"]:
        raise RuntimeError(f"Corrected spatial gates failed: {checks}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
