from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260813_30"
SOURCE = ROOT / "5_Test" / "20260810_1"
PARENT = ROOT / "5_Test" / "20260813_25"
SPATIAL_SOURCE = ROOT / "5_Test" / "20260810_9"

RAW = SOURCE / "work" / "spatial_correction" / "reach_catchments_polygonized.gpkg"
BASIN = ROOT / "0_reach_topology" / "data" / "processed" / "vector" / "prb_boundary.shp"
TOPOLOGY = PARENT / "inputs" / "topology" / "topology_edges.csv"


def copy_shapefile(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in source.parent.glob(source.stem + ".*"):
        shutil.copy2(path, destination_dir / path.name)
    return destination_dir / source.name


def upstream_totals(topo: pd.DataFrame, inc: dict[int, float]) -> dict[int, float]:
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(inc))
    for row in topo.itertuples(index=False):
        if pd.notna(row.downstream_reach):
            graph.add_edge(int(row.reach_id), int(row.downstream_reach))
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("topology is not acyclic")
    result = dict(inc)
    for reach in nx.topological_sort(graph):
        for downstream in graph.successors(reach):
            result[int(downstream)] += result[int(reach)]
    return result


def main() -> None:
    spatial = RUN / "inputs" / "spatial"
    baseline = RUN / "inputs" / "baseline"
    topology_dir = RUN / "inputs" / "topology"
    scenarios = RUN / "inputs" / "scenarios"
    for folder in [spatial, baseline, topology_dir, scenarios, RUN / "reports" / "tables", RUN / "logs"]:
        folder.mkdir(parents=True, exist_ok=True)

    raw = gpd.read_file(RAW, layer="reach_catchments")
    raw["reach_id"] = pd.to_numeric(raw["reach_id"], errors="coerce")
    raw = raw[raw.reach_id.between(1, 230)].copy()
    raw["reach_id"] = raw.reach_id.astype(int)
    catchments = raw.dissolve(by="reach_id", as_index=False).sort_values("reach_id").reset_index(drop=True)
    basin = gpd.read_file(BASIN).to_crs(catchments.crs).geometry.union_all()
    catchments["geometry"] = catchments.geometry.intersection(basin)
    catchments = catchments[~catchments.geometry.is_empty].copy()
    if len(catchments) != 230 or catchments.reach_id.nunique() != 230:
        raise RuntimeError(f"raw D8 Reach count is not 230: {len(catchments)}")

    topo = pd.read_csv(TOPOLOGY, encoding="utf-8-sig")
    inc = dict(zip(catchments.reach_id, catchments.geometry.area / 1_000_000.0))
    totals = upstream_totals(topo, inc)
    catchments["inc_km2"] = catchments.reach_id.map(inc)
    catchments["tot_km2"] = catchments.reach_id.map(totals)
    catchments["qa_flag"] = "whitebox_d8_supported_only"

    target = spatial / "reach_catchments.shp"
    for part in spatial.glob(target.stem + ".*"):
        part.unlink()
    catchments.to_file(target, encoding="UTF-8")
    if "inc_area_km2" in topo.columns:
        topo["inc_area_km2"] = topo.reach_id.map(inc)
    if "tot_area_km2" in topo.columns:
        topo["tot_area_km2"] = topo.reach_id.map(totals)
    topo.to_csv(topology_dir / "topology_edges.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(PARENT / "inputs" / "A1_indata.parquet", baseline / "R3_11_indata.parquet")

    source_script = SPATIAL_SOURCE / "scripts" / "build_full_spatial_inputs.py"
    if not (RUN / "scripts" / "build_full_spatial_inputs.py").exists():
        shutil.copy2(source_script, RUN / "scripts" / "build_full_spatial_inputs.py")
    if not (RUN / "scripts" / "runtime_guard.py").exists():
        shutil.copy2(SPATIAL_SOURCE / "scripts" / "runtime_guard.py", RUN / "scripts" / "runtime_guard.py")
    (RUN / "scripts" / "components").mkdir(parents=True, exist_ok=True)
    if not (RUN / "scripts" / "run_fold_pure_hyperparameters.py").exists():
        shutil.copy2(PARENT / "scripts" / "run_fold_pure_hyperparameters.py", RUN / "scripts" / "run_fold_pure_hyperparameters.py")
    if not (RUN / "scripts" / "components" / "q72_fold_pure_component.py").exists():
        shutil.copy2(PARENT / "scripts" / "components" / "q72_fold_pure_component.py", RUN / "scripts" / "components" / "q72_fold_pure_component.py")

    old = gpd.read_file(SOURCE / "inputs" / "spatial_corrected" / "reach_catchments.shp").set_index("reach_id")
    new = catchments.set_index("reach_id")
    rows = []
    for reach in sorted(new.index):
        rows.append({
            "reach_id": int(reach),
            "old_filled_inc_km2": float(old.loc[reach].geometry.area / 1_000_000.0),
            "new_d8_supported_inc_km2": float(new.loc[reach].geometry.area / 1_000_000.0),
            "removed_zero_label_area_km2": float(old.loc[reach].geometry.difference(new.loc[reach].geometry).area / 1_000_000.0),
            "new_cumulative_area_km2": float(totals[int(reach)]),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(RUN / "reports" / "d8_supported_area_rebuild.csv", index=False, encoding="utf-8-sig")
    payload = {
        "reach_count": int(len(catchments)),
        "d8_supported_area_km2": float(catchments.geometry.area.sum() / 1_000_000.0),
        "basin_area_km2": float(basin.area / 1_000_000.0),
        "unassigned_zero_label_area_km2": float(basin.difference(catchments.geometry.union_all()).area / 1_000_000.0),
        "unassigned_fraction": float(basin.difference(catchments.geometry.union_all()).area / basin.area),
        "overlap_area_km2": float((catchments.geometry.area.sum() - catchments.geometry.union_all().area) / 1_000_000.0),
        "all_reaches_positive": bool((catchments.geometry.area > 0).all()),
        "semantic_contract": "only pixels assigned by Whitebox D8 Watershed to one of the 230 frozen Reach seeds; zero-labelled regions are outside the represented open-loop network",
    }
    payload["passed"] = bool(payload["reach_count"] == 230 and payload["all_reaches_positive"] and abs(payload["overlap_area_km2"]) <= 1e-6)
    (RUN / "logs" / "d8_supported_geometry_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not payload["passed"]:
        raise RuntimeError(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
