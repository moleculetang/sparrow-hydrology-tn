from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260813_30"
SOURCE = ROOT / "5_Test" / "20260810_1"
PARENT = ROOT / "5_Test" / "20260813_25"
RAW_POLYGONS = SOURCE / "work" / "spatial_correction" / "reach_catchments_polygonized.gpkg"
FINAL_CATCHMENTS = SOURCE / "inputs" / "spatial_corrected" / "reach_catchments.shp"
FINAL_REACHES = SOURCE / "inputs" / "spatial_corrected" / "reaches_topology.shp"
STATIONS = SOURCE / "inputs" / "spatial_corrected" / "PRB水文站_全部.shp"
BASIN = ROOT / "0_reach_topology" / "data" / "processed" / "vector" / "prb_boundary.shp"
TOPOLOGY = PARENT / "inputs" / "topology" / "topology_edges.csv"
PARENT_INPUT = PARENT / "inputs" / "A1_indata.parquet"
PARENT_OOF = PARENT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet"
FLOW_DIR = ROOT / "0_reach_topology" / "work" / "rasters" / "flow_dir.tif"
FLOW_ACC = ROOT / "0_reach_topology" / "work" / "rasters" / "flow_acc.tif"
RAW_CATCHMENT_RASTER = SOURCE / "work" / "spatial_correction" / "reach_catchments_polygonized.gpkg"
GDAL_BIN = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin")
EXPECTED_PARENT_SHA = "205b67b9d5d5217ac903a6bcdaccc1bc17f87dd419343ae22272183cebedc9e4"
EXPECTED_CATCHMENT_SHA = "792f8138935ae06ce7730435b406c3f4900516fb2c03e73eff31283eff5476cf"
POSITIVE_CONTROLS = [23, 47, 146, 199]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace(" ", "").replace("\u3000", "")
    return text.replace("(", "（").replace(")", "）").removesuffix("站")


def runtime_gate() -> dict[str, object]:
    expected = Path(r"D:\ProgramData\anaconda3\envs\sparrow")
    actual = Path(sys.prefix)
    result = {
        "sys_prefix": str(actual),
        "expected_prefix": str(expected),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "prefix_match": actual.resolve() == expected.resolve(),
        "environment_name_match": os.environ.get("CONDA_DEFAULT_ENV") == "sparrow",
    }
    if not result["prefix_match"] or not result["environment_name_match"]:
        raise RuntimeError(f"sparrow runtime gate failed: {result}")
    return result


def parent_gate() -> dict[str, object]:
    frame = pd.read_parquet(PARENT_OOF)
    raw_nse = 1.0 - float(np.square(frame.actual - frame.predict).sum() / np.square(frame.actual - frame.actual.mean()).sum())
    log_obs = np.log(frame.actual.to_numpy(float))
    log_pred = np.log(frame.predict.to_numpy(float))
    log_nse = 1.0 - float(np.square(log_obs - log_pred).sum() / np.square(log_obs - log_obs.mean()).sum())
    result = {
        "oof_sha256": sha256(PARENT_OOF),
        "sha_match": sha256(PARENT_OOF) == EXPECTED_PARENT_SHA,
        "rows": int(len(frame)),
        "stations": int(frame.q_site.nunique()),
        "folds": int(frame.fold_id.nunique()),
        "key_duplicates": int(frame[["comid", "q_site", "year", "month", "fold_id"]].duplicated().sum()),
        "raw_nse": raw_nse,
        "log_nse": log_nse,
    }
    if not (result["sha_match"] and result["rows"] == 8738 and result["stations"] == 110 and result["folds"] == 3 and result["key_duplicates"] == 0):
        raise RuntimeError(f"parent reproduction gate failed: {result}")
    return result


def prepare_raw_and_basin(final_crs):
    raw = gpd.read_file(RAW_POLYGONS)
    raw["reach_id"] = pd.to_numeric(raw["reach_id"], errors="coerce")
    raw = raw[raw["reach_id"].between(1, 230)].copy()
    raw["reach_id"] = raw["reach_id"].astype(int)
    cats = raw.dissolve(by="reach_id", as_index=False).to_crs(final_crs)
    basin = gpd.read_file(BASIN).to_crs(final_crs)
    basin_union = basin.geometry.union_all()
    cats["geometry"] = cats.geometry.intersection(basin_union)
    cats = cats[cats.geometry.notna() & ~cats.geometry.is_empty].copy().sort_values("reach_id").reset_index(drop=True)
    return cats, basin_union


def replay_components(raw_cats: gpd.GeoDataFrame, basin_union):
    working = raw_cats.copy()
    uncovered = basin_union.difference(working.geometry.union_all())
    parts = list(uncovered.geoms) if hasattr(uncovered, "geoms") else [uncovered]
    parts = sorted([part for part in parts if part is not None and not part.is_empty and part.area > 0], key=lambda x: x.area, reverse=True)
    records = []
    for rank, part in enumerate(parts, 1):
        probe = part.representative_point()
        distances = working.geometry.distance(probe)
        order = distances.sort_values(kind="stable").index.tolist()
        target_index = order[0]
        second_index = order[1]
        target_id = int(working.at[target_index, "reach_id"])
        second_id = int(working.at[second_index, "reach_id"])
        target_geom = working.at[target_index, "geometry"]
        second_geom = working.at[second_index, "geometry"]
        target_boundary = float(part.boundary.intersection(target_geom.boundary).length)
        second_boundary = float(part.boundary.intersection(second_geom.boundary).length)
        target_geom_distance = float(part.distance(target_geom))
        second_geom_distance = float(part.distance(second_geom))
        records.append({
            "component_rank": rank,
            "component_area_km2": float(part.area / 1_000_000.0),
            "representative_x": float(probe.x),
            "representative_y": float(probe.y),
            "assigned_reach_id": target_id,
            "second_candidate_reach_id": second_id,
            "assigned_probe_distance_m": float(distances.loc[target_index]),
            "second_probe_distance_m": float(distances.loc[second_index]),
            "assigned_geometry_distance_m": target_geom_distance,
            "second_geometry_distance_m": second_geom_distance,
            "assigned_touches_component": bool(target_geom_distance <= 1e-7),
            "second_touches_component": bool(second_geom_distance <= 1e-7),
            "assigned_shared_boundary_m": target_boundary,
            "second_shared_boundary_m": second_boundary,
            "assigned_has_longest_boundary_of_top2": bool(target_boundary + 1e-9 >= second_boundary),
            "assignment_rule": "sequential representative_point nearest current catchment",
        })
        working.at[target_index, "geometry"] = target_geom.union(part)
    return working, gpd.GeoDataFrame(records, geometry=parts, crs=raw_cats.crs)


def compare_replay_to_final(replayed: gpd.GeoDataFrame, final: gpd.GeoDataFrame):
    a = replayed.set_index("reach_id").geometry
    b = final.set_index("reach_id").geometry
    rows = []
    for rid in sorted(set(a.index) | set(b.index)):
        ga = a.loc[rid]
        gb = b.loc[rid]
        rows.append({
            "reach_id": int(rid),
            "replay_area_km2": float(ga.area / 1_000_000.0),
            "final_area_km2": float(gb.area / 1_000_000.0),
            "area_difference_km2": float((ga.area - gb.area) / 1_000_000.0),
            "symmetric_difference_km2": float(ga.symmetric_difference(gb).area / 1_000_000.0),
        })
    return pd.DataFrame(rows)


def graph_and_area_audit(final: gpd.GeoDataFrame, basin_union):
    topo = pd.read_csv(TOPOLOGY, encoding="utf-8-sig")
    graph = nx.DiGraph()
    graph.add_nodes_from(final.reach_id.astype(int))
    for row in topo.itertuples(index=False):
        down = getattr(row, "downstream_reach")
        if pd.notna(down):
            graph.add_edge(int(row.reach_id), int(down))
    inc = dict(zip(final.reach_id.astype(int), final.geometry.area / 1_000_000.0))
    totals = dict(inc)
    for rid in nx.topological_sort(graph):
        for down in graph.successors(rid):
            totals[int(down)] += totals[int(rid)]
    rows = []
    topo_by = topo.set_index("reach_id")
    for rid in sorted(inc):
        source_tot = float(topo_by.loc[rid, "tot_area_km2"] if "tot_area_km2" in topo_by.columns else np.nan)
        rows.append({
            "reach_id": rid,
            "geometry_inc_km2": inc[rid],
            "topology_recomputed_tot_km2": totals[rid],
            "frozen_topology_tot_km2": source_tot,
            "total_difference_km2": totals[rid] - source_tot,
        })
    sum_area = float(final.geometry.area.sum())
    union = final.geometry.union_all()
    return pd.DataFrame(rows), {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "acyclic": nx.is_directed_acyclic_graph(graph),
        "weak_components": nx.number_weakly_connected_components(graph),
        "terminal_count": sum(1 for node in graph if graph.out_degree(node) == 0),
        "basin_area_km2": float(basin_union.area / 1_000_000.0),
        "catchment_sum_area_km2": sum_area / 1_000_000.0,
        "catchment_union_area_km2": float(union.area / 1_000_000.0),
        "overlap_area_km2": float((sum_area - union.area) / 1_000_000.0),
        "uncovered_area_km2": float(basin_union.difference(union).area / 1_000_000.0),
        "outside_basin_area_km2": float(union.difference(basin_union).area / 1_000_000.0),
    }


def line_coverage_audit(reaches: gpd.GeoDataFrame, final: gpd.GeoDataFrame):
    cat = final.set_index("reach_id").geometry
    buffered = final[["reach_id", "geometry"]].copy()
    buffered["geometry"] = buffered.geometry.buffer(60.0)
    spatial_index = buffered.sindex
    rows = []
    for row in reaches.itertuples(index=False):
        rid = int(row.reach_id)
        line = row.geometry
        length = max(float(line.length), 1e-12)
        own_buffer = buffered.loc[buffered.reach_id.eq(rid), "geometry"].iloc[0]
        own = float(line.intersection(own_buffer).length / length)
        candidates = []
        candidate_positions = spatial_index.query(line, predicate="intersects")
        for position in candidate_positions:
            other = int(buffered.iloc[int(position)].reach_id)
            geom = buffered.iloc[int(position)].geometry
            if other == rid:
                continue
            ratio = float(line.intersection(geom).length / length)
            if ratio > 0:
                candidates.append((ratio, int(other)))
        candidates.sort(reverse=True)
        rows.append({
            "reach_id": rid,
            "line_length_km": length / 1000.0,
            "own_catchment_coverage_60m": own,
            "largest_other_coverage_60m": candidates[0][0] if candidates else 0.0,
            "largest_other_reach_id": candidates[0][1] if candidates else pd.NA,
            "positive_control": rid in POSITIVE_CONTROLS,
        })
    return pd.DataFrame(rows)


def raw_support_audit(raw: gpd.GeoDataFrame, reaches: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check that removing zero-labelled fill does not remove modeled entities."""
    raw_by = raw.set_index("reach_id").geometry
    line_rows = []
    for row in reaches.itertuples(index=False):
        rid = int(row.reach_id)
        line = row.geometry
        length = max(float(line.length), 1e-12)
        support = raw_by.loc[rid].buffer(60.0)
        line_rows.append({
            "reach_id": rid,
            "line_length_km": length / 1000.0,
            "own_raw_d8_catchment_coverage_60m": float(line.intersection(support).length / length),
        })

    panel = pd.read_parquet(PARENT_OOF, columns=["comid", "q_site"])[["comid", "q_site"]].drop_duplicates()
    panel["station_norm"] = panel.q_site.map(norm)
    stations = gpd.read_file(STATIONS).to_crs(raw.crs)
    station_col = next(c for c in ["Station", "STATION", "NAME", "station", "name"] if c in stations.columns)
    stations["station_norm"] = stations[station_col].map(norm)
    station_by = stations.sort_values(station_col).drop_duplicates("station_norm").set_index("station_norm")
    station_rows = []
    for ent in panel.itertuples(index=False):
        rid = int(ent.comid)
        name = str(ent.station_norm)
        if name not in station_by.index:
            station_rows.append({"q_site": ent.q_site, "reach_id": rid, "coordinate_found": False, "inside_own_raw_d8_catchment": False, "distance_to_own_raw_d8_catchment_m": np.nan})
            continue
        point = station_by.loc[name].geometry
        own = raw_by.loc[rid]
        station_rows.append({
            "q_site": ent.q_site,
            "reach_id": rid,
            "coordinate_found": True,
            "inside_own_raw_d8_catchment": bool(own.contains(point) or own.touches(point)),
            "distance_to_own_raw_d8_catchment_m": float(point.distance(own)),
        })
    return pd.DataFrame(line_rows), pd.DataFrame(station_rows)


def station_audit(final: gpd.GeoDataFrame, reaches: gpd.GeoDataFrame):
    # The contract is the 110 evaluated P1 entities, not every observed entity
    # available in the wider A1 panel.
    panel = pd.read_parquet(PARENT_OOF, columns=["comid", "q_site"])
    entities = panel[["comid", "q_site"]].drop_duplicates().copy()
    entities["station_norm"] = entities.q_site.map(norm)
    stations = gpd.read_file(STATIONS).to_crs(final.crs)
    station_col = next(c for c in ["Station", "STATION", "NAME", "station", "name"] if c in stations.columns)
    stations["station_norm"] = stations[station_col].map(norm)
    duplicates = stations.groupby("station_norm").size()
    unique_station = stations.sort_values(station_col).drop_duplicates("station_norm").set_index("station_norm")
    cats = final.set_index("reach_id").geometry
    lines = reaches.set_index("reach_id").geometry
    rows = []
    for ent in entities.itertuples(index=False):
        rid = int(ent.comid)
        name = str(ent.station_norm)
        record = {
            "q_site": ent.q_site,
            "station_norm": name,
            "assigned_reach_id": rid,
            "coordinate_record_count": int(duplicates.get(name, 0)),
            "coordinate_found": name in unique_station.index,
        }
        if name not in unique_station.index:
            record.update({"containing_reach_ids": "", "inside_assigned": False, "distance_to_assigned_catchment_m": np.nan, "distance_to_assigned_line_m": np.nan, "nearest_line_reach_id": pd.NA, "nearest_line_distance_m": np.nan, "hard_spatial_conflict": False, "classification": "NO_EXACT_COORDINATE_NAME_MATCH"})
        else:
            point = unique_station.loc[name].geometry
            containing = final[final.geometry.contains(point) | final.geometry.touches(point)].reach_id.astype(int).tolist()
            dist_cat = float(point.distance(cats.loc[rid]))
            line_distances = reaches.geometry.distance(point)
            nearest_idx = line_distances.idxmin()
            nearest_id = int(reaches.loc[nearest_idx, "reach_id"])
            nearest_distance = float(line_distances.loc[nearest_idx])
            assigned_line_distance = float(point.distance(lines.loc[rid]))
            inside = rid in containing
            hard = (not inside) and len(containing) == 1 and dist_cat > 5000.0
            if hard:
                classification = "HARD_OTHER_CATCHMENT_GT5KM"
            elif inside:
                classification = "INSIDE_ASSIGNED_CATCHMENT"
            elif dist_cat <= 2000.0:
                classification = "BOUNDARY_OR_COORDINATE_ROUNDING_LE2KM"
            elif len(containing) == 0:
                classification = "OUTSIDE_ALL_CATCHMENTS_REVIEW"
            else:
                classification = "OTHER_CATCHMENT_2TO5KM_REVIEW"
            record.update({
                "containing_reach_ids": "|".join(map(str, sorted(containing))),
                "inside_assigned": inside,
                "distance_to_assigned_catchment_m": dist_cat,
                "distance_to_assigned_line_m": assigned_line_distance,
                "nearest_line_reach_id": nearest_id,
                "nearest_line_distance_m": nearest_distance,
                "hard_spatial_conflict": hard,
                "classification": classification,
            })
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["classification", "q_site"])


def run_external(args: list[str]) -> None:
    completed = subprocess.run(args, cwd=RUN, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(f"external command failed ({completed.returncode}): {' '.join(args)}\n{completed.stderr[-2000:]}")


def raster_info(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(GDAL_BIN / "gdalinfo.exe"), "-json", str(path)], cwd=RUN,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )
    return json.loads(completed.stdout)


def crop_to_envi(source: Path, target: Path, bounds: tuple[float, float, float, float], dtype: str) -> tuple[np.memmap, dict[str, object]]:
    minx, miny, maxx, maxy = bounds
    run_external([
        str(GDAL_BIN / "gdal_translate.exe"), "-q", "-of", "ENVI", "-ot", dtype,
        "-projwin", str(minx), str(maxy), str(maxx), str(miny), str(source), str(target),
    ])
    info = raster_info(target)
    width, height = map(int, info["size"])
    numpy_dtype = {"Int16": "<i2", "Int32": "<i4", "Float32": "<f4", "Byte": "u1"}[dtype]
    return np.memmap(target, dtype=numpy_dtype, mode="r", shape=(height, width)), info


def rasterize_raw_catchments(reference: Path, target: Path) -> None:
    run_external([str(GDAL_BIN / "gdal_create.exe"), "-q", "-if", str(reference), "-ot", "Int32", "-a_nodata", "0", "-burn", "0", str(target)])
    run_external([
        str(GDAL_BIN / "gdal_rasterize.exe"), "-q", "-a", "reach_id", "-where", "reach_id >= 1 AND reach_id <= 230",
        "-l", "reach_catchments", str(RAW_CATCHMENT_RASTER), str(target),
    ])


def d8_component_audit(components: gpd.GeoDataFrame) -> pd.DataFrame:
    """Trace high-accumulation pixels in the nine material fill components."""
    work = RUN / "inputs" / "d8_component_audit"
    work.mkdir(parents=True, exist_ok=True)
    # Whitebox D8 pointer encoding.
    direction = {1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0), 16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0)}
    records: list[dict[str, object]] = []
    top = components.sort_values("component_area_km2", ascending=False).head(9)
    for row in top.itertuples(index=False):
        rank = int(row.component_rank)
        minx, miny, maxx, maxy = row.geometry.bounds
        bounds = (minx - 3000.0, miny - 3000.0, maxx + 3000.0, maxy + 3000.0)
        prefix = work / f"component_{rank:03d}"
        fd, info = crop_to_envi(FLOW_DIR, prefix.with_name(prefix.name + "_fd.bin"), bounds, "Int16")
        fa, _ = crop_to_envi(FLOW_ACC, prefix.with_name(prefix.name + "_fa.bin"), bounds, "Float32")
        reference = prefix.with_name(prefix.name + "_reference.tif")
        mask_tif = prefix.with_name(prefix.name + "_mask.tif")
        raw_catchment_tif = prefix.with_name(prefix.name + "_raw_catchment.tif")
        run_external([str(GDAL_BIN / "gdal_translate.exe"), "-q", "-of", "GTiff", str(prefix.with_name(prefix.name + "_fd.bin")), str(reference)])
        rasterize_raw_catchments(reference, raw_catchment_tif)
        rc, _ = crop_to_envi(raw_catchment_tif, prefix.with_name(prefix.name + "_rc.bin"), bounds, "Int32")
        run_external([str(GDAL_BIN / "gdal_create.exe"), "-q", "-if", str(reference), "-ot", "Byte", "-a_nodata", "0", "-burn", "0", str(mask_tif)])
        run_external([
            str(GDAL_BIN / "gdal_rasterize.exe"), "-q", "-burn", "1", "-where", f"component_rank={rank}",
            "-l", "fill_components", str(RUN / "reports" / "fill_components.gpkg"), str(mask_tif),
        ])
        mask, _ = crop_to_envi(mask_tif, prefix.with_name(prefix.name + "_mask.bin"), bounds, "Byte")
        valid = (mask == 1) & np.isfinite(fa) & (fa >= 0)
        locations = np.argwhere(valid)
        if not len(locations):
            raise RuntimeError(f"no valid D8 pixels in fill component {rank}")
        values = np.asarray(fa[valid])
        candidate_positions = np.argsort(values)[-min(50, len(values)):][::-1]
        targets: list[int] = []
        traces: list[tuple[int, int, float, int, str]] = []
        height, width = fd.shape
        for pos in candidate_positions:
            rr, cc = map(int, locations[int(pos)])
            start_r, start_c = rr, cc
            status = "MAX_STEPS"
            target = 0
            for _ in range(200000):
                raw_target = int(rc[rr, cc])
                if raw_target > 0:
                    target = raw_target
                    status = "ENTERED_RAW_D8_CATCHMENT"
                    break
                code = int(fd[rr, cc])
                if code not in direction:
                    status = "INVALID_OR_NODATA_D8"
                    break
                dr, dc = direction[code]
                rr += dr
                cc += dc
                if rr < 0 or cc < 0 or rr >= height or cc >= width:
                    status = "LEFT_3KM_CROP_WITHOUT_CATCHMENT"
                    break
            if target > 0:
                targets.append(target)
            traces.append((start_r, start_c, float(fa[start_r, start_c]), target, status))
        target_counts = pd.Series(targets, dtype="int64").value_counts() if targets else pd.Series(dtype="int64")
        modal_target = int(target_counts.index[0]) if len(target_counts) else 0
        modal_support = int(target_counts.iloc[0]) if len(target_counts) else 0
        gt = info["geoTransform"]
        best_r, best_c, best_acc, best_target, best_status = traces[0]
        outlet_x = float(gt[0] + (best_c + 0.5) * gt[1] + (best_r + 0.5) * gt[2])
        outlet_y = float(gt[3] + (best_c + 0.5) * gt[4] + (best_r + 0.5) * gt[5])
        supports = modal_target == int(row.assigned_reach_id) and modal_support >= 25
        records.append({
            "component_rank": rank,
            "component_area_km2": float(row.component_area_km2),
            "assigned_reach_id": int(row.assigned_reach_id),
            "second_candidate_reach_id": int(row.second_candidate_reach_id),
            "assigned_shared_boundary_m": float(row.assigned_shared_boundary_m),
            "second_shared_boundary_m": float(row.second_shared_boundary_m),
            "d8_outlet_x": outlet_x,
            "d8_outlet_y": outlet_y,
            "d8_outlet_accumulation": best_acc,
            "d8_first_trace_target_reach": best_target,
            "d8_first_trace_status": best_status,
            "d8_modal_target_reach": modal_target,
            "d8_modal_target_support_of_50": modal_support,
            "d8_successful_trace_count": len(targets),
            "supports_assignment": bool(supports),
            "classification": "D8_SUPPORTS_ASSIGNMENT" if supports else "D8_DOES_NOT_CONFIRM_ASSIGNMENT",
        })
    return pd.DataFrame(records)


def main() -> None:
    for folder in [RUN / "reports", RUN / "inputs", RUN / "logs"]:
        folder.mkdir(parents=True, exist_ok=True)
    print("STAGE runtime", flush=True)
    runtime = runtime_gate()
    print("STAGE parent", flush=True)
    parent = parent_gate()
    print("STAGE load final geometry", flush=True)
    final = gpd.read_file(FINAL_CATCHMENTS).sort_values("reach_id").reset_index(drop=True)
    reaches = gpd.read_file(FINAL_REACHES).to_crs(final.crs).sort_values("reach_id").reset_index(drop=True)
    catchment_sha = sha256(FINAL_CATCHMENTS)
    if catchment_sha != EXPECTED_CATCHMENT_SHA:
        raise RuntimeError(f"unexpected corrected Catchment SHA: {catchment_sha}")
    print("STAGE prepare raw and basin", flush=True)
    raw_cats, basin_union = prepare_raw_and_basin(final.crs)
    print("STAGE replay fill components", flush=True)
    replayed, components = replay_components(raw_cats, basin_union)
    print("STAGE compare replay", flush=True)
    comparison = compare_replay_to_final(replayed, final)
    print("STAGE topology area", flush=True)
    area_table, topology = graph_and_area_audit(final, basin_union)
    print("STAGE line coverage", flush=True)
    line_table = line_coverage_audit(reaches, final)
    print("STAGE raw support", flush=True)
    raw_line_table, raw_station_table = raw_support_audit(raw_cats, reaches)
    print("STAGE station audit", flush=True)
    station_table = station_audit(final, reaches)
    print("STAGE D8 material fill audit", flush=True)
    d8_table = d8_component_audit(components)

    print("STAGE write reports", flush=True)
    components.drop(columns="geometry").to_csv(RUN / "reports" / "fill_component_audit.csv", index=False, encoding="utf-8-sig")
    components.to_file(RUN / "reports" / "fill_components.gpkg", layer="fill_components", driver="GPKG")
    comparison.to_csv(RUN / "reports" / "replay_vs_frozen_geometry.csv", index=False, encoding="utf-8-sig")
    area_table.to_csv(RUN / "reports" / "topology_area_closure.csv", index=False, encoding="utf-8-sig")
    line_table.to_csv(RUN / "reports" / "reach_line_catchment_coverage.csv", index=False, encoding="utf-8-sig")
    raw_line_table.to_csv(RUN / "reports" / "raw_d8_reach_line_support.csv", index=False, encoding="utf-8-sig")
    raw_station_table.to_csv(RUN / "reports" / "raw_d8_evaluation_station_support.csv", index=False, encoding="utf-8-sig")
    station_table.to_csv(RUN / "reports" / "evaluation_station_spatial_audit.csv", index=False, encoding="utf-8-sig")
    d8_table.to_csv(RUN / "reports" / "material_fill_d8_outlet_audit.csv", index=False, encoding="utf-8-sig")

    component_df = pd.DataFrame(components.drop(columns="geometry"))
    total_fill = float(component_df.component_area_km2.sum())
    component_df = component_df.sort_values("component_area_km2", ascending=False).reset_index(drop=True)
    cumulative = component_df.component_area_km2.cumsum() / max(total_fill, 1e-30)
    count_to_95 = int(np.searchsorted(cumulative.to_numpy(), 0.95) + 1)
    positive = line_table[line_table.positive_control].copy()
    gates = {
        "runtime": runtime,
        "parent": parent,
        "corrected_catchment_sha256": catchment_sha,
        "raw_d8_catchment_count": int(raw_cats.reach_id.nunique()),
        "fill_component_count": int(len(component_df)),
        "fill_total_area_km2": total_fill,
        "fill_area_fraction_of_basin": total_fill / topology["basin_area_km2"],
        "fill_largest_area_km2": float(component_df.component_area_km2.max()),
        "fill_components_for_95pct_area": count_to_95,
        "fill_non_touching_assigned_count": int((~component_df.assigned_touches_component).sum()),
        "fill_non_touching_assigned_area_km2": float(component_df.loc[~component_df.assigned_touches_component, "component_area_km2"].sum()),
        "fill_assigned_not_longest_top2_boundary_count": int((~component_df.assigned_has_longest_boundary_of_top2).sum()),
        "replay_total_symmetric_difference_km2": float(comparison.symmetric_difference_km2.sum()),
        "replay_max_reach_symmetric_difference_km2": float(comparison.symmetric_difference_km2.max()),
        "replay_max_abs_area_difference_km2": float(comparison.area_difference_km2.abs().max()),
        "topology": topology,
        "topology_max_abs_total_difference_km2": float(area_table.total_difference_km2.abs().max()),
        "positive_controls": positive.to_dict(orient="records"),
        "positive_controls_min_own_coverage_60m": float(positive.own_catchment_coverage_60m.min()),
        "raw_d8_min_reach_line_coverage_60m": float(raw_line_table.own_raw_d8_catchment_coverage_60m.min()),
        "raw_d8_reach_lines_below_0_80": int((raw_line_table.own_raw_d8_catchment_coverage_60m < 0.80).sum()),
        "raw_d8_evaluation_stations_outside_own": int((raw_station_table.coordinate_found & ~raw_station_table.inside_own_raw_d8_catchment).sum()),
        "raw_d8_evaluation_stations_gt5km": int((raw_station_table.distance_to_own_raw_d8_catchment_m > 5000.0).sum()),
        "evaluation_stations": int(len(station_table)),
        "evaluation_stations_coordinate_found": int(station_table.coordinate_found.sum()),
        "evaluation_stations_inside_assigned": int(station_table.inside_assigned.sum()),
        "evaluation_stations_hard_conflicts": int(station_table.hard_spatial_conflict.sum()),
        "evaluation_station_class_counts": station_table.classification.value_counts().to_dict(),
        "material_fill_d8": d8_table.to_dict(orient="records"),
        "material_fill_d8_supported_count": int(d8_table.supports_assignment.sum()),
    }

    exact_replay = gates["replay_total_symmetric_difference_km2"] <= 1e-6 and gates["replay_max_abs_area_difference_km2"] <= 1e-8
    closed_geometry = topology["overlap_area_km2"] <= 1e-6 and topology["uncovered_area_km2"] <= 1e-6 and topology["outside_basin_area_km2"] <= 1e-6
    area_closed = gates["topology_max_abs_total_difference_km2"] <= 1e-6
    fills_touch = gates["fill_non_touching_assigned_area_km2"] <= 1e-6
    positive_pass = gates["positive_controls_min_own_coverage_60m"] >= 0.80
    station_pass = gates["evaluation_stations_hard_conflicts"] == 0
    station_count_pass = gates["evaluation_stations"] == 110
    d8_pass = gates["material_fill_d8_supported_count"] == 9
    all_pass = bool(exact_replay and closed_geometry and area_closed and fills_touch and positive_pass and station_pass and station_count_pass and d8_pass)
    gates["decision_gates"] = {
        "exact_fill_replay": exact_replay,
        "no_overlap_gap_or_outside": closed_geometry,
        "topology_area_closure": area_closed,
        "all_material_fills_touch_assigned_catchment": fills_touch,
        "four_positive_controls_own_coverage_ge_0_80": positive_pass,
        "no_hard_evaluation_station_conflict": station_pass,
        "evaluation_station_set_is_exactly_110": station_count_pass,
        "nine_material_fills_confirmed_by_d8": d8_pass,
    }
    gates["all_engineering_gates_pass"] = all_pass
    gates["model_rerun_authorized"] = not all_pass
    gates["model_run_performed"] = False
    if all_pass:
        terminal = "CORRECTED_CATCHMENTS_INDEPENDENTLY_CONFIRMED_NO_CHANGE"
        next_atom = "MONITORED_STATION_OBSERVATION_LINEAGE"
    else:
        terminal = "CATCHMENT_SUPPORT_AMBIGUOUS_REQUIRES_MANUAL_GEOMETRY_REVIEW"
        next_atom = "REPAIR_OR_MANUAL_REVIEW_IN_20260813_30"
    gates["terminal"] = terminal
    gates["next_atom"] = next_atom
    (RUN / "terminal_gate.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(gates, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
