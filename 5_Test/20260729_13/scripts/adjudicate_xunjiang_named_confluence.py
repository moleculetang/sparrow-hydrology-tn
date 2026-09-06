from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
PARENT = ROOT / "5_Test" / "20260729_12"
TOPO = ROOT / "0_reach_topology"
REPORT = RUN_DIR / "reports" / "xunjiang_direction_gate"
INPUTS = RUN_DIR / "inputs"
MANIFEST_DIR = RUN_DIR / "inputs_manifest"

CONFIG = RUN_DIR / "config" / "xunjiang_review_contract.json"
EXTERNAL_WEB = RUN_DIR / "config" / "external_web_cross_validation.json"
PARENT_GATE = PARENT / "reports" / "reach_direction_adjudication_gate" / "gate.json"
PARENT_SLOPE = PARENT / "inputs" / "reach_slope_direction_adjudicated.csv"
EDGE_CSV = TOPO / "results" / "tables" / "topology_edges.csv"
SUMMARY_CSV = TOPO / "results" / "tables" / "reach_summary.csv"
MANUAL_CSV = TOPO / "results" / "tables" / "manual_review.csv"
TOPO_QA_CSV = TOPO / "results" / "tables" / "topology_qa.csv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path, role: str, state: str) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": state,
        "bytes": st.st_size,
        "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def node_roles(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    nodes = sorted(set(edges["fnode"].astype(int)) | set(edges["tnode"].astype(int)))
    for node in nodes:
        incoming = edges.loc[edges["tnode"].astype(int) == node]
        outgoing = edges.loc[edges["fnode"].astype(int) == node]
        rows.append({
            "node_id": node,
            "incoming_count": len(incoming),
            "outgoing_count": len(outgoing),
            "incoming_reach_ids": ",".join(map(str, incoming["reach_id"].astype(int).tolist())),
            "incoming_names": "|".join(incoming["src_id"].astype(str).tolist()),
            "outgoing_reach_ids": ",".join(map(str, outgoing["reach_id"].astype(int).tolist())),
            "outgoing_names": "|".join(outgoing["src_id"].astype(str).tolist()),
            "outgoing_fraction_sum": float(outgoing["frac"].astype(float).sum()),
            "is_split": len(outgoing) > 1,
            "is_terminal_confluence": len(incoming) > 1 and len(outgoing) == 0,
        })
    table = pd.DataFrame(rows)
    graph = nx.DiGraph()
    for row in edges.itertuples():
        graph.add_edge(int(row.fnode), int(row.tnode), reach_id=int(row.reach_id))
    summary = {
        "dag": nx.is_directed_acyclic_graph(graph),
        "split_nodes": int((table["outgoing_count"] > 1).sum()),
        "terminal_nodes": int(((table["outgoing_count"] == 0) & (table["incoming_count"] > 0)).sum()),
        "terminal_confluences": int(table["is_terminal_confluence"].sum()),
    }
    return table, summary


def area_closure(edges: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    incoming: dict[int, list[int]] = defaultdict(list)
    for row in edges.itertuples():
        incoming[int(row.tnode)].append(int(row.reach_id))
    total = dict(zip(summary["reach_id"].astype(int), summary["tot_area_km2"].astype(float)))
    inc = dict(zip(summary["reach_id"].astype(int), summary["inc_area_km2"].astype(float)))
    rows = []
    for row in edges.itertuples():
        rid = int(row.reach_id)
        ups = incoming.get(int(row.fnode), [])
        expected = inc[rid] + sum(total[u] for u in ups)
        rows.append({
            "reach_id": rid,
            "src_id": str(row.src_id),
            "fnode": int(row.fnode),
            "tnode": int(row.tnode),
            "upstream_reach_ids_recomputed": ",".join(map(str, ups)),
            "incremental_area_km2": inc[rid],
            "frozen_total_area_km2": total[rid],
            "expected_total_from_graph_km2": expected,
            "area_closure_residual_km2": total[rid] - expected,
        })
    return pd.DataFrame(rows)


def main() -> None:
    for directory in [REPORT, INPUTS, MANIFEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    external_web = json.loads(EXTERNAL_WEB.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent_gate["authorized_next_action"] != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize this review")
    if parent_gate["unresolved_reach_ids"] != [cfg["target_reach_id"]]:
        raise RuntimeError("Parent unresolved set is not exactly reach 31")

    edges = pd.read_csv(EDGE_CSV, encoding="utf-8-sig")
    summary = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
    current = pd.read_csv(PARENT_SLOPE, encoding="utf-8-sig")
    for frame in [edges, summary, current]:
        frame["reach_id"] = frame["reach_id"].astype(int)

    baseline_roles, baseline_graph = node_roles(edges)
    baseline_closure = area_closure(edges, summary)
    alternative = edges.copy()
    idx = alternative.index[alternative["reach_id"] == cfg["target_reach_id"]][0]
    fnode = int(alternative.loc[idx, "fnode"])
    tnode = int(alternative.loc[idx, "tnode"])
    alternative.loc[idx, ["fnode", "tnode"]] = [tnode, fnode]
    alternative_roles, alternative_graph = node_roles(alternative)
    alternative_closure = area_closure(alternative, summary)

    node = int(cfg["confluence_node"])
    bnode = baseline_roles.loc[baseline_roles["node_id"] == node].iloc[0]
    anode = alternative_roles.loc[alternative_roles["node_id"] == node].iloc[0]
    baseline_incoming = {
        int(row.reach_id): str(row.src_id)
        for row in edges.loc[edges["tnode"].astype(int) == node].itertuples()
    }
    baseline_outgoing = {
        int(row.reach_id): str(row.src_id)
        for row in edges.loc[edges["fnode"].astype(int) == node].itertuples()
    }
    expected_incoming = {int(k): str(v) for k, v in cfg["expected_incoming"].items()}
    expected_outgoing = {int(k): str(v) for k, v in cfg["expected_outgoing"].items()}
    named_confluence_match = (
        baseline_incoming == expected_incoming and baseline_outgoing == expected_outgoing
    )
    official_web_consistent = (
        "wuzhou.gov.cn" in external_web["url"]
        and any("浔江、桂江相汇成西江" in quote for quote in external_web["quotes"])
        and external_web["decision_role"] == "non_decisive_external_cross_validation"
    )

    focus_ids = [18, 30, 31, 85]
    comparison = baseline_closure.loc[baseline_closure["reach_id"].isin(focus_ids)].merge(
        alternative_closure.loc[alternative_closure["reach_id"].isin(focus_ids)],
        on=["reach_id", "src_id"],
        suffixes=("_baseline", "_alternative"),
        validate="one_to_one",
    )
    comparison.to_csv(
        REPORT / "area_closure_counterfactual.csv", index=False, encoding="utf-8-sig"
    )
    node_table = pd.DataFrame([
        {
            "scenario": "frozen_direction",
            **bnode.to_dict(),
            **{f"graph_{k}": v for k, v in baseline_graph.items()},
        },
        {
            "scenario": "reverse_reach_31",
            **anode.to_dict(),
            **{f"graph_{k}": v for k, v in alternative_graph.items()},
        },
    ])
    node_table.to_csv(
        REPORT / "node_27_counterfactual.csv", index=False, encoding="utf-8-sig"
    )
    named = pd.DataFrame([
        {
            "node_id": node,
            "baseline_incoming_reaches": json.dumps(baseline_incoming, ensure_ascii=False),
            "baseline_outgoing_reaches": json.dumps(baseline_outgoing, ensure_ascii=False),
            "expected_incoming_reaches": json.dumps(expected_incoming, ensure_ascii=False),
            "expected_outgoing_reaches": json.dumps(expected_outgoing, ensure_ascii=False),
            "named_confluence_match": named_confluence_match,
            "interpretation": "浔江 + 桂江(漓江) → 西江",
        }
    ])
    named.to_csv(REPORT / "named_confluence_evidence.csv", index=False, encoding="utf-8-sig")
    external_lines = [
        "# 外部网络交叉验证",
        "",
        f"- 来源类型：{external_web['source_type']}",
        f"- 发布机构：{external_web['publisher']}",
        f"- 标题：{external_web['title']}",
        f"- URL：{external_web['url']}",
        f"- 页面显示发布时间：{external_web['displayed_publication_time']}",
        f"- 文件日期：{external_web['document_date']}",
        f"- 访问时间：{external_web['accessed_at']}",
        "",
        "## 原文摘录",
        "",
    ]
    external_lines.extend([f"> {quote}" for quote in external_web["quotes"]])
    external_lines += [
        "",
        "## 交叉验证结论",
        "",
        external_web["cross_validation_claim"],
        "",
        "该来源仅作外部交叉验证，不替代本地可复算的节点角色和累计面积守恒门禁。",
    ]
    (REPORT / "external_web_cross_validation.md").write_text(
        "\n".join(external_lines) + "\n", encoding="utf-8"
    )

    baseline_max_residual = float(baseline_closure["area_closure_residual_km2"].abs().max())
    alt_focus_max_residual = float(
        alternative_closure.loc[
            alternative_closure["reach_id"].isin([18, 31]), "area_closure_residual_km2"
        ].abs().max()
    )
    checks = {
        "parent_authorization": True,
        "parent_unresolved_set_is_only_31": True,
        "named_confluence_is_xun_plus_gui_to_xi": named_confluence_match,
        "baseline_graph_is_dag": baseline_graph["dag"],
        "alternative_graph_is_dag": alternative_graph["dag"],
        "baseline_has_no_split_node": baseline_graph["split_nodes"] == 0,
        "alternative_creates_split_node": alternative_graph["split_nodes"] > baseline_graph["split_nodes"],
        "node_27_alternative_outgoing_fraction_exceeds_one": (
            float(anode.outgoing_fraction_sum) > 1.0 + cfg["split_fraction_tolerance"]
        ),
        "baseline_area_closes": baseline_max_residual <= cfg["area_closure_tolerance_km2"],
        "alternative_materially_breaks_area_closure": (
            alt_focus_max_residual >= cfg["counterfactual_material_residual_km2"]
        ),
        "baseline_is_two_in_one_out_at_node_27": (
            int(bnode.incoming_count) == 2 and int(bnode.outgoing_count) == 1
        ),
        "alternative_is_one_in_two_out_at_node_27": (
            int(anode.incoming_count) == 1 and int(anode.outgoing_count) == 2
        ),
    }
    checks = {k: bool(v) for k, v in checks.items()}
    passed = all(checks.values())

    updated = current.copy()
    target = updated["reach_id"] == cfg["target_reach_id"]
    if target.sum() != 1:
        raise RuntimeError("Target reach row is not unique")
    if passed:
        updated.loc[target, "direction_adjudication_class"] = (
            "topology_direction_valid_named_confluence"
        )
        updated.loc[target, "direction_gate_pass"] = True
        updated.loc[target, "direction_support_basis"] = (
            "named confluence 浔江+桂江→西江; reversal creates undeclared split, "
            "outgoing fraction sum 2, and material cumulative-area closure failure"
        )
        updated.loc[target, "direction_confidence"] = "high"
        updated.loc[target, "recommended_model_direction"] = "retain_fnode_to_tnode"
    updated["slope_value_changed_by_direction_audit"] = False
    updated.to_csv(INPUTS / "reach_slope_direction_final.csv", index=False, encoding="utf-8-sig")

    gate = {
        "run_id": cfg["run_id"],
        "phase": "xunjiang_named_confluence_direction_adjudication",
        "created_utc": now(),
        "target_reach_id": cfg["target_reach_id"],
        "checks": checks,
        "evidence": {
            "named_confluence": "浔江 + 桂江(漓江) → 西江",
            "baseline_node_27_indegree": int(bnode.incoming_count),
            "baseline_node_27_outdegree": int(bnode.outgoing_count),
            "alternative_node_27_indegree": int(anode.incoming_count),
            "alternative_node_27_outdegree": int(anode.outgoing_count),
            "alternative_node_27_outgoing_fraction_sum": float(anode.outgoing_fraction_sum),
            "baseline_max_area_closure_residual_km2": baseline_max_residual,
            "alternative_focus_max_area_closure_residual_km2": alt_focus_max_residual,
        },
        "external_cross_validation": {
            "source_type": external_web["source_type"],
            "publisher": external_web["publisher"],
            "title": external_web["title"],
            "url": external_web["url"],
            "accessed_at": external_web["accessed_at"],
            "consistent_with_local_direction": official_web_consistent,
            "decision_role": external_web["decision_role"],
        },
        "all_six_direction_issues_resolved": passed,
        "q78_nat_full_routing_data_ready": passed,
        "decision": "XUNJIANG_DIRECTION_RESOLVED" if passed else "XUNJIANG_DIRECTION_UNRESOLVED",
        "authorized_next_action": (
            cfg["required_next_action_if_pass"] if passed else cfg["required_next_action_if_fail"]
        ),
        "authorized_scope": (
            "Build only the Q78-NAT conservation core; do not calibrate and do not add management fluxes."
            if passed
            else "Supplement only Xunjiang external direction evidence; Q78 construction remains forbidden."
        ),
        "production_topology_mutated": False,
        "slope_values_changed": False,
        "flow_acc_used_as_direction_truth": False,
        "period_2019_2022_read": False,
        "series_terminal": False,
        "passed": passed,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# 浔江方向的命名汇流与守恒裁定

## 技术结论

方向门禁{'通过' if passed else '未通过'}。节点 27 的冻结结构精确对应“浔江 + 桂江(漓江) → 西江”；把浔江单边翻转会把二入一出汇流变成一入二出分流，且两条出边 `frac` 均为 1，总和为 {float(anode.outgoing_fraction_sum):.1f}。

## 关键证据

- 原方向节点 27：入度 {int(bnode.incoming_count)}，出度 {int(bnode.outgoing_count)}；
- 翻转反事实：入度 {int(anode.incoming_count)}，出度 {int(anode.outgoing_count)}；
- 冻结河网原本分流节点数：{baseline_graph['split_nodes']}；
- 翻转后分流节点数：{alternative_graph['split_nodes']}；
- 原方向全网最大累计面积闭合残差：{baseline_max_residual:.12g} km²；
- 翻转后 reach 18/31 最大闭合残差：{alt_focus_max_residual:.6f} km²。

因此，翻转并不是另一种同样合理的平坦河段方向；它会在没有分流声明、没有分流比例且没有独立水系证据的情况下复制上游水量，并破坏冻结 catchment 面积账本。

## 官方网络资料与本地拓扑一致

梧州市人民政府办公室公开的《梧州市城市防洪应急预案》明确写道：“浔江、桂江相汇成西江”，并称梧州是“珠江流域西江的起点”。这与 node 27 的本地命名汇流完全一致。网页证据只作为外部交叉验证，未参与阈值设定或替代本地守恒计算。

## 范围与定义

本轮证明的是 Q78 路由图应保留 `reach 31: node 43 → node 27`，不是证明该 11.28 km 河段具有可由 30 m DEM 解析的正水面坡度。它仍保留 `_10` 的邻接插补坡度 `4.1774933459e-05 m/m`。

## 方法与稳健性

反事实只在内存中翻转 reach 31。累计面积闭合使用每条 reach 的 incremental area、冻结 total area 和重新由边表推导的 upstream reach 集合。`flow_acc.tif` 没有参与裁定。

本轮不制作汇总图：证据是一个节点的精确入边、出边、分配比例和面积恒等式，审计表比图更直接。

## 限制

命名汇流和 catchment 账本足以确定模型路由方向，但不能替代实测河床或水面纵坡。未来如获得独立水面高程，应更新坡度置信度，而不是翻转已通过守恒门禁的路由。

## 下一步

{'六条方向问题全部解除，只允许开始构建 Q78-NAT 严格守恒核心；暂不率定、不加入管理通量。' if passed else '继续寻找浔江的权威河网或实测水面高程，Q78 仍禁止构建。'}

## 仍待回答的问题

- 邻接插补的极低坡度对河道线性库时间常数有多敏感？
- Q78-NAT 首个守恒实现是否能在 230 reach × 156 月上达到逐项相对闭合误差 `<1e-8`？
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "run_id": cfg["run_id"],
        "created_utc": now(),
        "runtime": "conda PRB_reach",
        "atomic_question": "Does named confluence and mass accounting resolve only reach 31?",
        "target_reach_id": 31,
        "chart_omission_reason": "A one-node conservation identity is more auditable as exact tables.",
        "production_topology_written": False,
        "station_flow_read": False,
        "period_2019_2022_read": False,
        "external_web_cross_validation": {
            "url": external_web["url"],
            "accessed_at": external_web["accessed_at"],
            "consistent": official_web_consistent,
            "decision_role": external_web["decision_role"],
        },
    }
    (REPORT / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sources = [
        CONFIG, EXTERNAL_WEB, RUN_DIR / "experiment_contract.md", RUN_DIR / "README.md",
        RUN_DIR / "scripts" / "adjudicate_xunjiang_named_confluence.py",
        PARENT_GATE, PARENT_SLOPE, EDGE_CSV, SUMMARY_CSV, MANUAL_CSV, TOPO_QA_CSV,
    ]
    products = [
        INPUTS / "reach_slope_direction_final.csv",
        REPORT / "named_confluence_evidence.csv",
        REPORT / "external_web_cross_validation.md",
        REPORT / "node_27_counterfactual.csv",
        REPORT / "area_closure_counterfactual.csv",
        REPORT / "gate.json",
        REPORT / "technical_report.md",
        REPORT / "run_manifest.json",
    ]
    manifest = {
        "run_id": cfg["run_id"],
        "created_utc": now(),
        "sources": [record(p, "xunjiang_direction_source", "reported_or_derived") for p in sources],
        "products": [record(p, "xunjiang_direction_product", "derived") for p in products],
        "production_topology_mutated": False,
        "flow_acc_used_as_direction_truth": False,
    }
    (MANIFEST_DIR / "provenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(manifest["sources"] + manifest["products"]).to_csv(
        MANIFEST_DIR / "provenance_manifest.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps({
        "passed": passed,
        "authorized_next_action": gate["authorized_next_action"],
        "baseline_max_residual_km2": baseline_max_residual,
        "alternative_focus_max_residual_km2": alt_focus_max_residual,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
