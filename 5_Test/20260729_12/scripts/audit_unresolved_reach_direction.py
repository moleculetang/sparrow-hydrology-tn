from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
TOPO = ROOT / "0_reach_topology"
PARENT = ROOT / "5_Test" / "20260729_11"
REPORT_DIR = RUN_DIR / "reports" / "reach_direction_adjudication_gate"
INPUT_DIR = RUN_DIR / "inputs"
MANIFEST_DIR = RUN_DIR / "inputs_manifest"

CONFIG_PATH = RUN_DIR / "config" / "direction_review_contract.json"
PARENT_GATE = PARENT / "reports" / "slope_anomaly_review_gate" / "gate.json"
PARENT_SLOPE = PARENT / "inputs" / "reach_slope_reviewed.csv"
PARENT_REVIEW = PARENT / "reports" / "slope_anomaly_review_gate" / "imputed_reach_review.csv"

REACH_SHP = TOPO / "results" / "vectors" / "reaches_topology.shp"
EDGE_CSV = TOPO / "results" / "tables" / "topology_edges.csv"
SUMMARY_CSV = TOPO / "results" / "tables" / "reach_summary.csv"
MANUAL_CSV = TOPO / "results" / "tables" / "manual_review.csv"
TOPO_QA_CSV = TOPO / "results" / "tables" / "topology_qa.csv"
PIPELINE_PY = TOPO / "src" / "reach_topology" / "pipeline.py"
SOURCE_DEM = TOPO / "data" / "processed" / "dem_prb" / "dem.tif"
EDGE_FILLED_DEM = TOPO / "work" / "rasters" / "dem_edge_filled.tif"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path, role: str, semantic_state: str) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": semantic_state,
        "bytes": st.st_size,
        "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def sample_point(ds: rasterio.io.DatasetReader, x: float, y: float, source_crs) -> float:
    if str(source_crs) != str(ds.crs):
        xs, ys = transform(source_crs, ds.crs, [x], [y])
        x, y = xs[0], ys[0]
    value = float(next(ds.sample([(x, y)]))[0])
    if ds.nodata is not None and np.isclose(value, float(ds.nodata)):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def endpoint_drop(ds, geom, source_crs) -> tuple[float, float, float]:
    coords = list(geom.coords)
    z0 = sample_point(ds, coords[0][0], coords[0][1], source_crs)
    z1 = sample_point(ds, coords[-1][0], coords[-1][1], source_crs)
    return z0, z1, z0 - z1


def make_node_points(reaches: gpd.GeoDataFrame) -> dict[int, tuple[float, float]]:
    values: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in reaches.itertuples():
        coords = list(row.geometry.coords)
        values[int(row.fnode)].append(tuple(coords[0]))
        values[int(row.tnode)].append(tuple(coords[-1]))
    return {
        node: (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
        for node, pts in values.items()
    }


def network_metrics(edge_table: pd.DataFrame, total_area: dict[int, float], tol: float) -> dict:
    graph = nx.DiGraph()
    for row in edge_table.itertuples():
        graph.add_edge(int(row.fnode), int(row.tnode), reach_id=int(row.reach_id))

    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for row in edge_table.itertuples():
        incoming[int(row.tnode)].append(int(row.reach_id))
        outgoing[int(row.fnode)].append(int(row.reach_id))

    violations = []
    transitions = []
    for node in sorted(set(incoming) | set(outgoing)):
        for up in incoming.get(node, []):
            for down in outgoing.get(node, []):
                if up == down:
                    continue
                delta = total_area[down] - total_area[up]
                transitions.append((up, down, node, delta))
                if delta < -tol:
                    violations.append((up, down, node, delta))

    return {
        "dag": nx.is_directed_acyclic_graph(graph),
        "weak_components": nx.number_weakly_connected_components(graph),
        "headwater_nodes": sum(graph.in_degree(n) == 0 and graph.out_degree(n) > 0 for n in graph.nodes),
        "terminal_nodes": sum(graph.out_degree(n) == 0 and graph.in_degree(n) > 0 for n in graph.nodes),
        "split_nodes": sum(graph.out_degree(n) > 1 for n in graph.nodes),
        "terminal_confluences": sum(graph.out_degree(n) == 0 and graph.in_degree(n) > 1 for n in graph.nodes),
        "transition_count": len(transitions),
        "area_violation_count": len(violations),
        "area_violations": violations,
    }


def same_name_transition_count(edge_table: pd.DataFrame) -> int:
    incoming: dict[int, list[tuple[int, str]]] = defaultdict(list)
    outgoing: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in edge_table.itertuples():
        incoming[int(row.tnode)].append((int(row.reach_id), str(row.src_id)))
        outgoing[int(row.fnode)].append((int(row.reach_id), str(row.src_id)))
    count = 0
    for node in set(incoming) | set(outgoing):
        count += sum(a[1] == b[1] for a in incoming.get(node, []) for b in outgoing.get(node, []))
    return int(count)


def chain_evidence(
    target_id: int,
    target_name: str,
    edges: pd.DataFrame,
    node_points: dict[int, tuple[float, float]],
    source_ds,
    edge_ds,
    reach_crs,
) -> dict:
    same = edges[edges["src_id"].astype(str) == target_name].copy()
    undirected = nx.Graph()
    for row in same.itertuples():
        undirected.add_edge(int(row.fnode), int(row.tnode), reach_id=int(row.reach_id))
    target = same[same["reach_id"] == target_id].iloc[0]
    component_nodes = nx.node_connected_component(undirected, int(target.fnode))
    component = same[
        same["fnode"].astype(int).isin(component_nodes)
        & same["tnode"].astype(int).isin(component_nodes)
    ].copy()
    directed = nx.DiGraph()
    for row in component.itertuples():
        directed.add_edge(int(row.fnode), int(row.tnode), reach_id=int(row.reach_id))
    heads = [n for n in directed.nodes if directed.in_degree(n) == 0 and directed.out_degree(n) == 1]
    tails = [n for n in directed.nodes if directed.out_degree(n) == 0 and directed.in_degree(n) == 1]
    simple_directed_chain = (
        nx.is_directed_acyclic_graph(directed)
        and len(heads) == 1
        and len(tails) == 1
        and all(directed.in_degree(n) <= 1 and directed.out_degree(n) <= 1 for n in directed.nodes)
        and len(component) == len(component_nodes) - 1
    )
    source_drop = float("nan")
    edge_drop = float("nan")
    head = heads[0] if len(heads) == 1 else None
    tail = tails[0] if len(tails) == 1 else None
    if head is not None and tail is not None:
        hx, hy = node_points[head]
        tx, ty = node_points[tail]
        source_drop = sample_point(source_ds, hx, hy, reach_crs) - sample_point(
            source_ds, tx, ty, reach_crs
        )
        edge_drop = sample_point(edge_ds, hx, hy, reach_crs) - sample_point(
            edge_ds, tx, ty, reach_crs
        )
    return {
        "reach_id": target_id,
        "src_id": target_name,
        "component_reach_ids": ",".join(map(str, sorted(component["reach_id"].astype(int).tolist()))),
        "component_reach_count": int(len(component)),
        "head_node": head,
        "tail_node": tail,
        "simple_directed_chain": bool(simple_directed_chain),
        "source_dem_chain_drop_m": source_drop,
        "edge_filled_dem_chain_drop_m": edge_drop,
    }


def adjudicate(row: pd.Series, cfg: dict) -> tuple[str, bool, str, str]:
    structural_degradation = bool(
        (row["alternative_area_violation_count"] > row["baseline_area_violation_count"])
        or (row["alternative_same_name_transition_count"] < row["baseline_same_name_transition_count"])
        or (not row["alternative_dag"])
    )
    source_local_support = bool(
        np.isfinite(row["source_dem_endpoint_drop_m"])
        and row["source_dem_endpoint_drop_m"] >= cfg["source_dem_endpoint_min_drop_m"]
    )
    chain_dem_support = bool(
        (
            np.isfinite(row["source_dem_chain_drop_m"])
            and row["source_dem_chain_drop_m"] >= cfg["chain_dem_min_drop_m"]
        )
        or (
            np.isfinite(row["edge_filled_dem_chain_drop_m"])
            and row["edge_filled_dem_chain_drop_m"] >= cfg["chain_dem_min_drop_m"]
        )
    )
    chain_support = bool(row["simple_directed_chain"] and chain_dem_support)
    local_negative = bool(
        row["window_slope_m_m"] < -cfg["flat_slope_abs_threshold"]
        or row["theil_sen_slope_m_m"] < -cfg["flat_slope_abs_threshold"]
    )
    local_flat = bool(
        abs(row["window_slope_m_m"]) <= cfg["flat_slope_abs_threshold"]
        and abs(row["theil_sen_slope_m_m"]) <= cfg["flat_slope_abs_threshold"]
    )

    if not structural_degradation:
        return "unresolved", False, "counterfactual reversal does not degrade registered topology tests", "unresolved"
    if bool(row["terminal"]) and local_flat and chain_support:
        return (
            "flat_terminal_or_lowland_prior",
            True,
            "terminal lowland is DEM-flat; named-chain drop and reversal counterfactual support frozen direction",
            "medium",
        )
    if local_flat and chain_support:
        return (
            "topology_direction_valid_flat_lowland",
            True,
            "local DEM is quantized flat; complete named-chain drop and reversal counterfactual support frozen direction",
            "medium",
        )
    if local_negative and chain_support:
        return (
            "topology_direction_valid_geometry_dem_conflict",
            True,
            "local unconditioned DEM conflicts, but complete named-chain drop and reversal counterfactual support frozen direction",
            "medium",
        )
    if source_local_support and row["simple_directed_chain"]:
        return (
            "topology_direction_valid_window_sampling_conflict",
            True,
            "source-DEM endpoints and named-chain topology support direction; robust interior window is locally conflicting",
            "high",
        )
    return "unresolved", False, "insufficient independent support after counterfactual review", "unresolved"


def main() -> None:
    for directory in [REPORT_DIR, INPUT_DIR, MANIFEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent_gate.get("authorized_next_action") != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize this atomic action")
    targets = [int(x) for x in cfg["target_reach_ids"]]
    if sorted(parent_gate["imputed_review"]["unresolved_reaches"]) != sorted(targets):
        raise RuntimeError("Target set differs from parent unresolved set")

    reaches = gpd.read_file(REACH_SHP)
    reaches["reach_id"] = reaches["reach_id"].astype(int)
    edges = pd.read_csv(EDGE_CSV, encoding="utf-8-sig")
    edges["reach_id"] = edges["reach_id"].astype(int)
    summary = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
    summary["reach_id"] = summary["reach_id"].astype(int)
    parent_slope = pd.read_csv(PARENT_SLOPE, encoding="utf-8-sig")
    parent_review = pd.read_csv(PARENT_REVIEW, encoding="utf-8-sig")
    total_area = dict(zip(summary["reach_id"], summary["tot_area_km2"]))
    node_points = make_node_points(reaches)

    baseline = network_metrics(edges, total_area, cfg["area_monotonic_tolerance_km2"])
    baseline_same_name = same_name_transition_count(edges)
    evidence_rows = []
    counterfactual_rows = []
    chain_rows = []

    with rasterio.open(SOURCE_DEM) as source_ds, rasterio.open(EDGE_FILLED_DEM) as edge_ds:
        for rid in targets:
            reach = reaches.loc[reaches["reach_id"] == rid].iloc[0]
            edge = edges.loc[edges["reach_id"] == rid].iloc[0]
            summary_row = summary.loc[summary["reach_id"] == rid].iloc[0]
            review = parent_review.loc[parent_review["reach_id"] == rid].iloc[0]
            slope_row = parent_slope.loc[parent_slope["reach_id"] == rid].iloc[0]
            source_z0, source_z1, source_drop = endpoint_drop(source_ds, reach.geometry, reaches.crs)
            edge_z0, edge_z1, edge_drop = endpoint_drop(edge_ds, reach.geometry, reaches.crs)
            chain = chain_evidence(
                rid, str(edge.src_id), edges, node_points, source_ds, edge_ds, reaches.crs
            )
            chain_rows.append(chain)

            alternative = edges.copy()
            idx = alternative.index[alternative["reach_id"] == rid][0]
            alternative.loc[idx, ["fnode", "tnode"]] = [
                int(edge.tnode),
                int(edge.fnode),
            ]
            alt_metrics = network_metrics(
                alternative, total_area, cfg["area_monotonic_tolerance_km2"]
            )
            alt_same_name = same_name_transition_count(alternative)
            counterfactual_rows.append(
                {
                    "reach_id": rid,
                    "src_id": str(edge.src_id),
                    "baseline_dag": baseline["dag"],
                    "alternative_dag": alt_metrics["dag"],
                    "baseline_area_violation_count": baseline["area_violation_count"],
                    "alternative_area_violation_count": alt_metrics["area_violation_count"],
                    "baseline_same_name_transition_count": baseline_same_name,
                    "alternative_same_name_transition_count": alt_same_name,
                    "baseline_headwater_nodes": baseline["headwater_nodes"],
                    "alternative_headwater_nodes": alt_metrics["headwater_nodes"],
                    "baseline_terminal_nodes": baseline["terminal_nodes"],
                    "alternative_terminal_nodes": alt_metrics["terminal_nodes"],
                    "baseline_split_nodes": baseline["split_nodes"],
                    "alternative_split_nodes": alt_metrics["split_nodes"],
                    "baseline_terminal_confluences": baseline["terminal_confluences"],
                    "alternative_terminal_confluences": alt_metrics["terminal_confluences"],
                }
            )
            evidence_rows.append(
                {
                    "reach_id": rid,
                    "src_id": str(edge.src_id),
                    "fnode": int(edge.fnode),
                    "tnode": int(edge.tnode),
                    "terminal": bool(int(summary_row.terminal)),
                    "qa_status": str(summary_row.qa_status),
                    "qa_notes": str(summary_row.qa_notes),
                    "slope_m_m": float(slope_row.slope_m_m),
                    "window_slope_m_m": float(slope_row.window_slope_m_m),
                    "theil_sen_slope_m_m": float(slope_row.theil_sen_slope_m_m),
                    "endpoint_slope_m_m": float(slope_row.endpoint_slope_m_m),
                    "source_dem_start_m": source_z0,
                    "source_dem_end_m": source_z1,
                    "source_dem_endpoint_drop_m": source_drop,
                    "edge_filled_start_m": edge_z0,
                    "edge_filled_end_m": edge_z1,
                    "edge_filled_endpoint_drop_m": edge_drop,
                    **{k: v for k, v in chain.items() if k not in {"reach_id", "src_id"}},
                    "baseline_dag": baseline["dag"],
                    "alternative_dag": alt_metrics["dag"],
                    "baseline_area_violation_count": baseline["area_violation_count"],
                    "alternative_area_violation_count": alt_metrics["area_violation_count"],
                    "baseline_same_name_transition_count": baseline_same_name,
                    "alternative_same_name_transition_count": alt_same_name,
                }
            )

    evidence = pd.DataFrame(evidence_rows)
    decisions = evidence.apply(lambda row: adjudicate(row, cfg), axis=1)
    evidence[["adjudication_class", "direction_support_pass", "support_basis", "confidence"]] = pd.DataFrame(
        decisions.tolist(), index=evidence.index
    )
    evidence["recommended_model_direction"] = np.where(
        evidence["direction_support_pass"], "retain_fnode_to_tnode", "blocked"
    )
    evidence["slope_value_changed"] = False
    evidence.to_csv(REPORT_DIR / "reach_direction_evidence.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(counterfactual_rows).to_csv(
        REPORT_DIR / "single_edge_reversal_counterfactual.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(chain_rows).to_csv(
        REPORT_DIR / "same_name_chain_evidence.csv", index=False, encoding="utf-8-sig"
    )

    direction_fields = evidence[
        [
            "reach_id",
            "adjudication_class",
            "direction_support_pass",
            "support_basis",
            "confidence",
            "recommended_model_direction",
            "slope_value_changed",
        ]
    ].rename(
        columns={
            "adjudication_class": "direction_adjudication_class",
            "direction_support_pass": "direction_gate_pass",
            "support_basis": "direction_support_basis",
            "confidence": "direction_confidence",
            "slope_value_changed": "slope_value_changed_by_direction_audit",
        }
    )
    resolved_slope = parent_slope.merge(direction_fields, on="reach_id", how="left")
    resolved_slope["direction_adjudication_class"] = resolved_slope[
        "direction_adjudication_class"
    ].fillna("not_in_direction_anomaly_set")
    resolved_slope["direction_gate_pass"] = resolved_slope[
        "direction_gate_pass"
    ].fillna(True).astype(bool)
    resolved_slope["direction_support_basis"] = resolved_slope[
        "direction_support_basis"
    ].fillna("accepted by 20260729_10/11 direct or reviewed slope gate")
    resolved_slope["direction_confidence"] = resolved_slope["direction_confidence"].fillna(
        resolved_slope["review_confidence"]
    )
    resolved_slope["recommended_model_direction"] = resolved_slope[
        "recommended_model_direction"
    ].fillna("retain_fnode_to_tnode")
    resolved_slope["slope_value_changed_by_direction_audit"] = resolved_slope[
        "slope_value_changed_by_direction_audit"
    ].fillna(False)
    resolved_slope.to_csv(
        INPUT_DIR / "reach_slope_direction_adjudicated.csv", index=False, encoding="utf-8-sig"
    )

    unresolved = evidence.loc[~evidence["direction_support_pass"], "reach_id"].astype(int).tolist()
    passed = (
        not unresolved
        and baseline["dag"]
        and baseline["area_violation_count"] == 0
        and not bool(evidence["slope_value_changed"].any())
    )
    gate = {
        "run_id": cfg["run_id"],
        "phase": "unresolved_reach_direction_adjudication",
        "created_utc": utc_now(),
        "parent_authorization": {
            "required": cfg["required_parent_action"],
            "observed": parent_gate["authorized_next_action"],
            "passed": True,
        },
        "target_reach_ids": targets,
        "resolved_reach_ids": evidence.loc[
            evidence["direction_support_pass"], "reach_id"
        ].astype(int).tolist(),
        "unresolved_reach_ids": unresolved,
        "classification_counts": evidence["adjudication_class"].value_counts().to_dict(),
        "frozen_network": {
            "graph_is_dag": baseline["dag"],
            "area_reversal_edges": baseline["area_violation_count"],
            "weak_components": baseline["weak_components"],
            "same_name_transitions": baseline_same_name,
        },
        "checks": {
            "parent_authorization": True,
            "target_set_matches_parent": True,
            "all_six_reaches_adjudicated": len(evidence) == 6,
            "no_unresolved_direction": not unresolved,
            "frozen_graph_is_dag": baseline["dag"],
            "frozen_cumulative_area_is_nondecreasing": baseline["area_violation_count"] == 0,
            "each_retained_direction_has_counterfactual_degradation": bool(
                (
                    (evidence["alternative_area_violation_count"] > evidence["baseline_area_violation_count"])
                    | (
                        evidence["alternative_same_name_transition_count"]
                        < evidence["baseline_same_name_transition_count"]
                    )
                    | (~evidence["alternative_dag"])
                ).all()
            ),
            "slope_values_unchanged": not bool(evidence["slope_value_changed"].any()),
            "production_topology_not_mutated": True,
            "flow_acc_not_used_as_direction_truth": True,
            "2019_2022_not_read": True,
        },
        "q78_nat_full_routing_data_ready": passed,
        "decision": "DIRECTION_GATE_PASSED" if passed else "UNRESOLVED_DIRECTION_EVIDENCE_REMAINS",
        "authorized_next_action": (
            cfg["required_next_action_if_pass"] if passed else cfg["required_next_action_if_fail"]
        ),
        "authorized_scope": (
            "Build only the Q78-NAT conservation core; calibration remains outside this gate."
            if passed
            else "Supplement only unresolved direction evidence; Q78 construction remains forbidden."
        ),
        "series_terminal": False,
        "passed": passed,
    }
    (REPORT_DIR / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# 6 条未决 reach 方向裁定",
        "",
        "## 技术结论",
        "",
        (
            f"方向门禁{'通过' if passed else '未通过'}：6 条中 "
            f"{6 - len(unresolved)} 条获得可复核方向支持，{len(unresolved)} 条仍未解决。"
        ),
        "本轮没有翻转生产拓扑，也没有改变任何坡度值；结论只建立模型专用方向裁定层。",
        "",
        "## 逐 reach 证据",
        "",
        "| reach_id | 名称 | 裁定 | 置信度 | 原始方向DEM端点落差(m) | 同名链端点落差(m) | 单边翻转面积逆转数 |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in evidence.itertuples():
        report_lines.append(
            f"| {row.reach_id} | {row.src_id} | {row.adjudication_class} | "
            f"{row.confidence} | {row.source_dem_endpoint_drop_m:.3f} | "
            f"{row.source_dem_chain_drop_m:.3f} | {row.alternative_area_violation_count} |"
        )
    report_lines += [
        "",
        "## 证据定义与范围",
        "",
        "- 分析单位是一条最终有向 reach；方向定义为 `fnode → tnode`。",
        "- 原始方向 DEM 是生产拓扑最初用于端点定向的 1 arc-second 源栅格。",
        "- 同名链端点落差在完整、无分叉的同名河流连通分量首尾计算，而非只看局地异常段。",
        "- 反事实检验只在内存中翻转一条边，冻结其它边及累计面积，检查结构是否恶化。",
        "- `flow_acc.tif` 由现有拓扑烧河得到，本轮没有把它作为方向真值。",
        "",
        "## 方法与稳健性",
        "",
        f"- 冻结图为 DAG：{baseline['dag']}；累计面积逆转数：{baseline['area_violation_count']}。",
        "- 局地负坡或量化平坦不会被绝对值修复；只有完整同名链与反事实拓扑同时支持时才解除阻断。",
        "- 精确逐 reach 表保存在 `reach_direction_evidence.csv`；本轮不制作汇总图，因为门禁依赖 6 条离散河段的逐项证据，图形会弱化审计精度。",
        "",
        "## 限制",
        "",
        "- 这不是高精度河道纵剖面测量；6 条异常 reach 仍使用 `_10` 冻结的邻接坡度先验。",
        "- 若未来获得测量河床高程、ICESat-2 水面坡度或独立权威河网，应替换局地 DEM 冲突段的先验。",
        "- 本轮证明的是路由方向可用，不证明插补坡度无误；坡度敏感性必须在 Q78-NAT 率定前后单独报告。",
        "",
        "## 下一步",
        "",
        (
            "只允许构建 Q78-NAT 严格守恒核心；暂不率定，也不加入管理通量。"
            if passed
            else f"继续补充 reach {unresolved} 的独立方向证据，禁止构建 Q78。"
        ),
        "",
        "## 仍待回答的问题",
        "",
        "- 邻接插补坡度对月尺度河道路由时间常数的敏感度有多大？",
        "- 低坡河段是否应在 Q78 中采用速度下限或解析线性水库，而不是直接用坡度驱动经验式？",
    ]
    (REPORT_DIR / "technical_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    run_manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "runtime": "conda PRB_reach",
        "atomic_question": "Adjudicate only the six unresolved reach directions.",
        "target_reach_ids": targets,
        "inputs_read": [str(p) for p in [
            PARENT_GATE, PARENT_SLOPE, PARENT_REVIEW, REACH_SHP, EDGE_CSV,
            SUMMARY_CSV, MANUAL_CSV, TOPO_QA_CSV, SOURCE_DEM, EDGE_FILLED_DEM,
        ]],
        "forbidden_period_read": False,
        "production_topology_written": False,
        "chart_omission_reason": "The gate is a six-row categorical adjudication; exact tables are more auditable than an aggregate chart.",
    }
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    source_paths = [
        CONFIG_PATH, RUN_DIR / "experiment_contract.md", PARENT_GATE, PARENT_SLOPE,
        PARENT_REVIEW, EDGE_CSV, SUMMARY_CSV, MANUAL_CSV, TOPO_QA_CSV,
        PIPELINE_PY, SOURCE_DEM, EDGE_FILLED_DEM,
    ]
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        source_paths.append(REACH_SHP.with_suffix(suffix))
    product_paths = [
        INPUT_DIR / "reach_slope_direction_adjudicated.csv",
        REPORT_DIR / "reach_direction_evidence.csv",
        REPORT_DIR / "single_edge_reversal_counterfactual.csv",
        REPORT_DIR / "same_name_chain_evidence.csv",
        REPORT_DIR / "gate.json",
        REPORT_DIR / "technical_report.md",
        REPORT_DIR / "run_manifest.json",
    ]
    manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "sources": [file_record(p, "direction_review_source", "reported_or_derived") for p in source_paths],
        "products": [file_record(p, "direction_review_product", "derived") for p in product_paths],
        "conditioned_flow_accumulation_used_as_truth": False,
        "production_topology_mutated": False,
    }
    (MANIFEST_DIR / "provenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(manifest["sources"] + manifest["products"]).to_csv(
        MANIFEST_DIR / "provenance_manifest.csv", index=False, encoding="utf-8-sig"
    )

    print(json.dumps({
        "passed": passed,
        "resolved": 6 - len(unresolved),
        "unresolved": unresolved,
        "authorized_next_action": gate["authorized_next_action"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
