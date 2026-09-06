from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.stats import theilslopes
from shapely.geometry import Point


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent.parent
INPUTS = RUN / "inputs"
MANIFESTS = RUN / "inputs_manifest"
REPORTS = RUN / "reports" / "slope_anomaly_review_gate"
CONFIG = RUN / "config" / "review_contract.json"

PARENT = ROOT / "5_Test" / "20260729_10"
PARENT_GATE = PARENT / "reports" / "reach_slope_quality_gate" / "gate.json"
PARENT_QA = PARENT / "reports" / "reach_slope_quality_gate" / "reach_slope_qa.csv"
PARENT_SLOPE = PARENT / "inputs" / "reach_slope.csv"
PARENT_STATIC = PARENT / "inputs" / "reach_static_with_slope.parquet"
REACH_SHP = (
    ROOT / "0_reach_topology" / "results" / "vectors" / "reaches_topology.shp"
)
NODE_SHP = ROOT / "0_reach_topology" / "results" / "vectors" / "nodes.shp"
TOPOLOGY = (
    ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
)
SUMMARY = ROOT / "0_reach_topology" / "results" / "tables" / "reach_summary.csv"
DEM = ROOT / "0_reach_topology" / "work" / "rasters" / "dem_edge_filled.tif"
FLOW_ACC = ROOT / "0_reach_topology" / "work" / "rasters" / "flow_acc.tif"

WATERBODY_PATTERN = re.compile(r"水库|湖")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str, semantic_state: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "role": role,
        "semantic_state": semantic_state,
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path),
    }


def shapefile_components(path: Path) -> list[Path]:
    return [
        path.with_suffix(suffix)
        for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]
        if path.with_suffix(suffix).exists()
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def local_stat(
    raster: rasterio.io.DatasetReader,
    point: Point,
    radius_cells: int,
    percentile: float,
    mode: str = "percentile",
) -> float:
    row, col = raster.index(point.x, point.y)
    size = 2 * radius_cells + 1
    array = raster.read(
        1,
        window=Window(col - radius_cells, row - radius_cells, size, size),
        boundless=True,
        fill_value=raster.nodata,
    ).astype(float)
    valid = np.isfinite(array)
    if raster.nodata is not None:
        valid &= ~np.isclose(array, float(raster.nodata))
    values = array[valid]
    if len(values) == 0:
        return float("nan")
    if mode == "max":
        return float(np.max(values))
    return float(np.percentile(values, percentile))


def neighborhood_profile(
    line,
    dem: rasterio.io.DatasetReader,
    cfg: dict[str, object],
) -> dict[str, float | int]:
    length_m = float(line.length)
    requested = int(math.ceil(length_m / float(cfg["sample_spacing_m"]))) + 1
    n = max(
        int(cfg["minimum_profile_samples"]),
        min(int(cfg["maximum_profile_samples"]), requested),
    )
    distances = np.linspace(0.0, length_m, n)
    points = [line.interpolate(float(distance)) for distance in distances]
    radius_cells = int(
        math.ceil(float(cfg["neighborhood_radius_m"]) / abs(float(dem.res[0])))
    )
    values = np.array(
        [
            local_stat(
                dem,
                point,
                radius_cells,
                float(cfg["neighborhood_percentile"]),
            )
            for point in points
        ],
        dtype=float,
    )
    valid = np.isfinite(values)
    distances = distances[valid]
    values = values[valid]
    if len(values) < 5:
        return {
            "neighborhood_valid_samples": int(len(values)),
            "neighborhood_window_slope_m_m": np.nan,
            "neighborhood_theil_sen_slope_m_m": np.nan,
            "neighborhood_endpoint_slope_m_m": np.nan,
        }
    smooth = (
        pd.Series(values)
        .rolling(
            int(cfg["rolling_median_window"]),
            center=True,
            min_periods=1,
        )
        .median()
        .to_numpy()
    )
    end_count = max(3, int(math.ceil(len(smooth) * 0.1)))
    end_count = min(end_count, max(1, len(smooth) // 2))
    start_z = float(np.median(smooth[:end_count]))
    end_z = float(np.median(smooth[-end_count:]))
    start_d = float(np.median(distances[:end_count]))
    end_d = float(np.median(distances[-end_count:]))
    window_slope = (start_z - end_z) / max(end_d - start_d, 1.0)
    theil_slope = -float(theilslopes(smooth, distances, 0.95).slope)
    endpoint_slope = (float(values[0]) - float(values[-1])) / max(length_m, 1.0)
    return {
        "neighborhood_valid_samples": int(len(values)),
        "neighborhood_window_slope_m_m": float(window_slope),
        "neighborhood_theil_sen_slope_m_m": float(theil_slope),
        "neighborhood_endpoint_slope_m_m": float(endpoint_slope),
    }


def main() -> None:
    for folder in (INPUTS, MANIFESTS, REPORTS, RUN / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    require(parent_gate["passed"] is False, "Parent slope gate unexpectedly passed")
    require(
        parent_gate["authorized_next_action"] == "REVIEW_REACH_SLOPE_ANOMALIES",
        "Parent gate does not authorize anomaly review",
    )

    reaches = gpd.read_file(REACH_SHP).sort_values("reach_id").reset_index(drop=True)
    nodes = gpd.read_file(NODE_SHP)
    topology = pd.read_csv(TOPOLOGY).sort_values("reach_id")
    summary = pd.read_csv(SUMMARY).sort_values("reach_id")
    parent_qa = pd.read_csv(PARENT_QA, encoding="utf-8-sig")
    parent_slope = pd.read_csv(PARENT_SLOPE, encoding="utf-8-sig")
    expected = int(cfg["expected_reaches"])
    require(len(reaches) == expected, "Reach count mismatch")

    node_map = dict(zip(nodes["node_id"].astype(int), nodes.geometry))
    endpoint_rows = []
    for row in reaches.itertuples():
        coords = list(row.geometry.coords)
        for role, node_id, xy in [
            ("start", int(row.fnode), coords[0]),
            ("end", int(row.tnode), coords[-1]),
        ]:
            point = Point(xy)
            distance = point.distance(node_map[node_id])
            endpoint_rows.append(
                {
                    "reach_id": int(row.reach_id),
                    "endpoint_role": role,
                    "declared_node_id": node_id,
                    "endpoint_x": point.x,
                    "endpoint_y": point.y,
                    "node_x": node_map[node_id].x,
                    "node_y": node_map[node_id].y,
                    "endpoint_to_node_m": distance,
                    "strict_flag": distance > float(cfg["strict_endpoint_flag_m"]),
                }
            )
    endpoints = pd.DataFrame(endpoint_rows)
    strict = endpoints.loc[endpoints["strict_flag"]].copy()
    strict["fraction_of_dem_cell"] = (
        strict["endpoint_to_node_m"] / float(cfg["dem_resolution_m"])
    )
    strict["fraction_of_snap_tolerance"] = (
        strict["endpoint_to_node_m"] / float(cfg["topology_snap_tolerance_m"])
    )
    strict["within_benign_m"] = (
        strict["endpoint_to_node_m"]
        <= float(cfg["maximum_benign_endpoint_difference_m"])
    )
    strict["within_dem_fraction"] = (
        strict["fraction_of_dem_cell"]
        <= float(cfg["maximum_fraction_of_dem_cell"])
    )
    strict["within_snap_fraction"] = (
        strict["fraction_of_snap_tolerance"]
        <= float(cfg["maximum_fraction_of_snap_tolerance"])
    )
    strict["same_declared_node_cluster"] = strict.groupby("declared_node_id")[
        "reach_id"
    ].transform("count") >= 2
    strict["benign_representation_difference"] = strict[
        [
            "within_benign_m",
            "within_dem_fraction",
            "within_snap_fraction",
            "same_declared_node_cluster",
        ]
    ].all(axis=1)
    endpoint_review_pass = bool(
        len(strict) > 0 and strict["benign_representation_difference"].all()
    )

    graph = nx.DiGraph()
    graph.add_nodes_from(topology["reach_id"].astype(int))
    for row in topology.itertuples():
        if pd.notna(row.downstream_reach):
            graph.add_edge(int(row.reach_id), int(row.downstream_reach))
    graph_is_dag = nx.is_directed_acyclic_graph(graph)
    total_area = dict(
        zip(summary["reach_id"].astype(int), summary["tot_area_km2"].astype(float))
    )
    area_rows = []
    for upstream, downstream in graph.edges:
        area_rows.append(
            {
                "upstream_reach_id": upstream,
                "downstream_reach_id": downstream,
                "upstream_total_area_km2": total_area[upstream],
                "downstream_total_area_km2": total_area[downstream],
                "area_change_km2": total_area[downstream] - total_area[upstream],
                "nondecreasing": total_area[downstream] + 1e-9 >= total_area[upstream],
            }
        )
    area_review = pd.DataFrame(area_rows)
    area_monotonic = bool(area_review["nondecreasing"].all())

    imputed = parent_qa.loc[~parent_qa["direct_dem_estimate"].astype(bool)].copy()
    require(
        len(imputed) == int(cfg["expected_imputed_reaches"]),
        "Unexpected number of imputed reaches",
    )
    geometry_map = dict(zip(reaches["reach_id"].astype(int), reaches.geometry))
    downstream_map = {
        int(row.reach_id): (
            None if pd.isna(row.downstream_reach) else int(row.downstream_reach)
        )
        for row in topology.itertuples()
    }
    anomaly_rows = []
    with rasterio.open(DEM) as dem, rasterio.open(FLOW_ACC) as flow:
        require(reaches.crs == dem.crs == flow.crs, "Raster/vector CRS mismatch")
        flow_radius_cells = int(
            math.ceil(
                float(cfg["neighborhood_radius_m"]) / abs(float(flow.res[0]))
            )
        )
        for row in imputed.itertuples():
            rid = int(row.reach_id)
            line = geometry_map[rid]
            profile = neighborhood_profile(line, dem, cfg)
            coords = list(line.coords)
            start_point = Point(coords[0])
            end_point = Point(coords[-1])
            flow_start = local_stat(
                flow, start_point, flow_radius_cells, 100.0, mode="max"
            )
            flow_end = local_stat(
                flow, end_point, flow_radius_cells, 100.0, mode="max"
            )
            flow_support = bool(
                np.isfinite(flow_start)
                and np.isfinite(flow_end)
                and flow_end >= flow_start
            )
            downstream = downstream_map[rid]
            area_support = bool(
                downstream is None
                or total_area[downstream] + 1e-9 >= total_area[rid]
            )
            raw_values = [
                float(row.window_slope_m_m),
                float(row.theil_sen_slope_m_m),
                float(row.endpoint_slope_m_m),
            ]
            waterbody = bool(WATERBODY_PATTERN.search(str(row.src_id)))
            raw_flat = bool(
                max(abs(value) for value in raw_values)
                <= float(cfg["flat_slope_absolute_threshold_m_m"])
            )
            neighborhood_positive = bool(
                profile["neighborhood_window_slope_m_m"] > 0
                or profile["neighborhood_theil_sen_slope_m_m"] > 0
            )
            endpoint_positive = bool(float(row.endpoint_slope_m_m) > 0)
            if waterbody:
                review_class = "waterbody_or_lake_flat_prior"
                support = True
                support_basis = "waterbody geometry; retain explicit neighbor prior"
                confidence = "medium"
            elif raw_flat:
                review_class = "dem_quantized_flat_prior"
                support = area_support and flow_support
                support_basis = "raw DEM flat; topology area and conditioned D8 agree"
                confidence = "medium"
            elif neighborhood_positive:
                review_class = "line_raster_offset_explained"
                support = area_support
                support_basis = "90 m neighborhood profile supports downstream descent"
                confidence = "medium"
            elif endpoint_positive and area_support and flow_support:
                review_class = "endpoint_and_topology_supported_prior"
                support = True
                support_basis = "endpoint drop plus area and conditioned D8 support"
                confidence = "medium"
            elif area_support and flow_support:
                review_class = "topology_supported_but_raw_dem_conflict"
                support = True
                support_basis = "cumulative area and conditioned D8 support; raw DEM conflicts"
                confidence = "low"
            else:
                review_class = "unresolved_direction_conflict"
                support = False
                support_basis = "insufficient consistent evidence"
                confidence = "unresolved"
            anomaly_rows.append(
                {
                    "reach_id": rid,
                    "src_id": row.src_id,
                    "frozen_slope_m_m": float(row.slope_m_m),
                    "frozen_slope_source_status": row.slope_source_status,
                    "raw_window_slope_m_m": float(row.window_slope_m_m),
                    "raw_theil_sen_slope_m_m": float(row.theil_sen_slope_m_m),
                    "raw_endpoint_slope_m_m": float(row.endpoint_slope_m_m),
                    **profile,
                    "flow_acc_start_local_max": flow_start,
                    "flow_acc_end_local_max": flow_end,
                    "conditioned_flowacc_supports_direction": flow_support,
                    "cumulative_area_supports_direction": area_support,
                    "waterbody_or_lake": waterbody,
                    "raw_dem_flat": raw_flat,
                    "review_class": review_class,
                    "direction_support_pass": support,
                    "support_basis": support_basis,
                    "review_confidence": confidence,
                }
            )
    anomaly_review = pd.DataFrame(anomaly_rows).sort_values("reach_id")
    unresolved = anomaly_review.loc[~anomaly_review["direction_support_pass"]]
    imputed_fraction = len(anomaly_review) / expected

    reviewed = parent_slope.merge(
        anomaly_review[
            [
                "reach_id",
                "review_class",
                "direction_support_pass",
                "support_basis",
                "review_confidence",
            ]
        ],
        on="reach_id",
        how="left",
        validate="one_to_one",
    )
    reviewed["review_class"] = reviewed["review_class"].fillna(
        "direct_dem_estimate_not_in_anomaly_set"
    )
    reviewed["direction_support_pass"] = reviewed[
        "direction_support_pass"
    ].fillna(True)
    default_confidence = pd.Series(
        np.where(
            reviewed["slope_source_status"].eq("dem_robust_consensus"),
            "high",
            "medium",
        ),
        index=reviewed.index,
    )
    reviewed["review_confidence"] = reviewed["review_confidence"].fillna(
        default_confidence
    )
    reviewed["support_basis"] = reviewed["support_basis"].fillna(
        "direct positive DEM estimate from 20260729_10"
    )
    reviewed["slope_value_changed_by_review"] = False
    reviewed_path = INPUTS / "reach_slope_reviewed.csv"
    reviewed.to_csv(reviewed_path, index=False, encoding="utf-8-sig")

    endpoint_path = REPORTS / "endpoint_cluster_review.csv"
    area_path = REPORTS / "cumulative_area_edge_review.csv"
    anomaly_path = REPORTS / "imputed_reach_review.csv"
    strict.to_csv(endpoint_path, index=False, encoding="utf-8-sig")
    area_review.to_csv(area_path, index=False, encoding="utf-8-sig")
    anomaly_review.to_csv(anomaly_path, index=False, encoding="utf-8-sig")
    class_summary = (
        anomaly_review.groupby(
            ["review_class", "review_confidence", "direction_support_pass"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "reach_count"})
    )
    class_summary.to_csv(
        REPORTS / "imputed_reach_class_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    checks = {
        "parent_authorization": True,
        "endpoint_differences_are_benign_representation_effects": endpoint_review_pass,
        "graph_is_directed_acyclic": graph_is_dag,
        "cumulative_area_is_nondecreasing_on_every_edge": area_monotonic,
        "all_18_imputed_reaches_classified": (
            len(anomaly_review) == int(cfg["expected_imputed_reaches"])
            and anomaly_review["review_class"].notna().all()
        ),
        "no_unresolved_imputed_reach": len(unresolved) == 0,
        "imputed_fraction_at_or_below_threshold": (
            imputed_fraction <= float(cfg["maximum_imputed_fraction"])
        ),
        "review_did_not_change_slope_values": np.allclose(
            reviewed.sort_values("reach_id")["slope_m_m"],
            parent_slope.sort_values("reach_id")["slope_m_m"],
        ),
        "all_reviewed_slopes_remain_positive_finite": bool(
            np.isfinite(reviewed["slope_m_m"]).all()
            and reviewed["slope_m_m"].gt(0).all()
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = bool(all(checks.values()))
    next_action = (
        "BUILD_Q78_NAT_CONSERVATION_CORE"
        if passed
        else "REPAIR_UNRESOLVED_REACH_DIRECTION"
    )

    source_paths = (
        shapefile_components(REACH_SHP)
        + shapefile_components(NODE_SHP)
        + [
            TOPOLOGY,
            SUMMARY,
            DEM,
            FLOW_ACC,
            PARENT_GATE,
            PARENT_QA,
            PARENT_SLOPE,
            PARENT_STATIC,
            CONFIG,
            RUN / "experiment_contract.md",
        ]
    )
    source_records = [
        file_record(path, "review_source", "reported_or_derived")
        for path in source_paths
    ]
    product_paths = [
        reviewed_path,
        endpoint_path,
        area_path,
        anomaly_path,
        REPORTS / "imputed_reach_class_summary.csv",
    ]
    product_records = [
        file_record(path, "review_product", "derived") for path in product_paths
    ]
    provenance = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": source_records,
        "products": product_records,
        "conditioned_flow_accumulation_limitation": (
            "flow_acc is consistency evidence, not independent direction proof, "
            "because stream burning used the existing topology"
        ),
    }
    provenance_path = MANIFESTS / "provenance_manifest.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(source_records + product_records).to_csv(
        MANIFESTS / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gate = {
        "run_id": RUN.name,
        "phase": "reach_slope_anomaly_review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint_review": {
            "strict_flagged_endpoints": int(len(strict)),
            "affected_declared_nodes": sorted(
                strict["declared_node_id"].astype(int).unique().tolist()
            ),
            "maximum_difference_m": (
                float(strict["endpoint_to_node_m"].max()) if len(strict) else 0.0
            ),
            "maximum_fraction_of_dem_cell": (
                float(strict["fraction_of_dem_cell"].max()) if len(strict) else 0.0
            ),
            "maximum_fraction_of_snap_tolerance": (
                float(strict["fraction_of_snap_tolerance"].max())
                if len(strict)
                else 0.0
            ),
            "interpretation": "sub-pixel node-cluster representation difference",
        },
        "imputed_review": {
            "imputed_reaches": int(len(anomaly_review)),
            "imputed_fraction": imputed_fraction,
            "unresolved_reaches": unresolved["reach_id"].astype(int).tolist(),
            "class_counts": {
                str(key): int(value)
                for key, value in anomaly_review["review_class"]
                .value_counts()
                .to_dict()
                .items()
            },
            "slope_values_changed": False,
        },
        "network_review": {
            "graph_is_dag": graph_is_dag,
            "edges": int(len(area_review)),
            "cumulative_area_reversal_edges": int(
                (~area_review["nondecreasing"]).sum()
            ),
        },
        "checks": checks,
        "q78_nat_full_routing_data_ready": passed,
        "decision": (
            "ENDPOINT_PRECISION_AND_IMPUTED_SLOPES_ADMITTED"
            if passed
            else "UNRESOLVED_DIRECTION_CONFLICT_REMAINS"
        ),
        "authorized_next_action": next_action,
        "authorized_scope": (
            "Build and unit-test the Q78-NAT conservation core only; no calibration, management flux, Q72 refit, or locked-year access."
            if passed
            else "Repair only the listed unresolved reach directions; model construction remains forbidden."
        ),
        "series_terminal": False,
        "passed": passed,
    }
    gate_path = REPORTS / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Reach 坡度异常审查",
        "",
        "## 技术结论",
        "",
        (
            f"异常审查{'通过' if passed else '未通过'}。端点差异被定位为同一声明节点内的亚像元表示差异；"
            f"18 条插补 reach 中仍有 {len(unresolved)} 条未获得足够方向支持。"
        ),
        "",
        "## 端点差异不是拓扑断裂",
        "",
        f"- 严格 0.01 m 阈值下标记 {len(strict)} 个端点，涉及 {strict['declared_node_id'].nunique()} 个声明节点。",
        f"- 最大差异 {strict['endpoint_to_node_m'].max():.6f} m，占30 m DEM像元 {strict['fraction_of_dem_cell'].max():.3%}，占60 m拓扑吸附容差 {strict['fraction_of_snap_tolerance'].max():.3%}。",
        f"- 图为DAG：{graph_is_dag}；所有 {len(area_review)} 条图边累计流域面积不减：{area_monotonic}。",
        "",
        "## 18条插补坡度保持显式而未被伪装成观测",
        "",
        f"- 插补比例：{imputed_fraction:.2%}，预设上限 {float(cfg['maximum_imputed_fraction']):.0%}。",
        f"- 分类：{json.dumps(gate['imputed_review']['class_counts'], ensure_ascii=False)}。",
        f"- 未解决 reach：{gate['imputed_review']['unresolved_reaches']}。",
        "- 90 m邻域低分位DEM只用于解释线—栅格错位；没有改变 `_10` 冻结的任何 slope 数值。",
        "",
        "## 证据定义与限制",
        "",
        "单位分析对象为一条最终有向 reach。端点差异以 reach 几何端点到声明 topology node 的欧氏距离衡量；方向一致性同时检查累计流域面积和现有条件化 D8 累积量。",
        "",
        "D8 累积量使用了现有 topology 的 stream burn，因此只能证明内部一致性，不能作为独立方向真值。水库、湖泊和平坦低地的 DEM 坡度本身不可辨识，继续保留邻接先验和低/中置信度标记。",
        "",
        "## 方法与稳健性检查",
        "",
        "异常 reach 沿最终几何约每1 km采样，并在每个样点90 m邻域取20%高程分位数，随后计算首尾窗口与Theil–Sen趋势。端点表示容差以独立的30 m DEM分辨率和60 m生产拓扑吸附容差预注册。",
        "",
        "本轮不使用汇总图：放行决策依赖逐节点和逐reach精确门禁，`endpoint_cluster_review.csv` 与 `imputed_reach_review.csv` 是更直接的审计证据。",
        "",
        "## 下一步",
        "",
        (
            "允许构建 Q78-NAT 土壤—地下水—河道守恒核心，但仍禁止率定和管理通量。"
            if passed
            else "仅修复未解决方向冲突，继续禁止模型构建。"
        ),
        "",
        "## 仍待回答的问题",
        "",
        "- 插补 slope 对路由时间尺度后验的影响必须在 Q78-NAT 率定前后单独报告。",
        "- 未来若获得河道纵剖面或水面高程，应替换水体和平坦 reach 的邻接先验。",
    ]
    report_path = REPORTS / "technical_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    run_manifest = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": "conda PRB_reach",
        "parent_gate_sha256": sha256(PARENT_GATE),
        "provenance_sha256": sha256(provenance_path),
        "gate_sha256": sha256(gate_path),
        "technical_report_sha256": sha256(report_path),
        "parquet_finalization_pending": True,
    }
    (REPORTS / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "SLOPE_ANOMALY_REVIEW "
        f"passed={passed} unresolved={len(unresolved)} next={next_action}"
    )


if __name__ == "__main__":
    main()
