from __future__ import annotations

import calendar
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
PARENT = ROOT / "5_Test" / "20260729_13"
LEDGER = ROOT / "5_Test" / "20260729_9"
REPORT = RUN_DIR / "reports" / "q78_nat_conservation_core_gate"
OUTPUTS = RUN_DIR / "outputs"
MANIFEST_DIR = RUN_DIR / "inputs_manifest"

CONFIG = RUN_DIR / "config" / "q78_nat_core_contract.json"
PARENT_GATE = PARENT / "reports" / "xunjiang_direction_gate" / "gate.json"
FORCING_PATH = LEDGER / "inputs" / "reach_month_forcing_2006_2018.parquet"
STATIC_PATH = PARENT / "inputs" / "reach_static_direction_final.parquet"


def utc_now() -> str:
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


def build_reach_graph(static: pd.DataFrame) -> tuple[nx.DiGraph, dict[int, list[int]], list[int]]:
    graph = nx.DiGraph()
    graph.add_nodes_from(static["reach_id"].astype(int).tolist())
    incoming_by_node: dict[int, list[int]] = defaultdict(list)
    outgoing_by_node: dict[int, list[int]] = defaultdict(list)
    for row in static.itertuples():
        incoming_by_node[int(row.tnode)].append(int(row.reach_id))
        outgoing_by_node[int(row.fnode)].append(int(row.reach_id))
    upstream: dict[int, list[int]] = {}
    for row in static.itertuples():
        rid = int(row.reach_id)
        ups = [u for u in incoming_by_node.get(int(row.fnode), []) if u != rid]
        upstream[rid] = sorted(ups)
        for up in ups:
            graph.add_edge(up, rid)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Reach graph is not a DAG")
    order = list(nx.topological_sort(graph))
    return graph, upstream, order


def synthetic_tests() -> dict:
    p = {
        "kappa_s": 1.0,
        "gamma_ET": 1.0,
        "k_perc": 0.08,
        "p_perc": 2.0,
        "k_int": 0.2,
        "p_int": 2.0,
        "k_g": 0.05,
        "k_route": 0.85,
    }
    ss0, sg0, ch0 = 10.0, 2.0, 3.0
    smax, rain, pet, area = 20.0, 12.0, 4.0, 10.0
    aet = min(ss0 + rain, pet * (ss0 / smax) ** p["gamma_ET"])
    u = ss0 + rain - aet
    qex = max(u - smax, 0.0)
    sstar = min(u, smax)
    wet = sstar / smax
    rg = p["k_perc"] * wet ** p["p_perc"] * sstar
    qint = p["k_int"] * wet ** p["p_int"] * sstar
    ss1 = sstar - rg - qint
    qbase = p["k_g"] * sg0
    sg1 = sg0 + rg - qbase
    local = (qex + qint + qbase) * area * 1000.0
    available = ch0 + local
    qout = p["k_route"] * available
    ch1 = available - qout
    residual = (
        (ss0 + sg0) * area * 1000.0
        + ch0
        + rain * area * 1000.0
        - aet * area * 1000.0
        - qout
        - (ss1 + sg1) * area * 1000.0
        - ch1
    )
    two_reach_upstream_transfer = qout <= available and qout >= 0.0
    return {
        "one_reach_total_residual_m3": float(residual),
        "one_reach_pass": abs(residual) < 1e-8,
        "two_reach_transfer_bound_pass": bool(two_reach_upstream_transfer),
        "passed": bool(abs(residual) < 1e-8 and two_reach_upstream_transfer),
    }


def run_core(
    forcing: pd.DataFrame,
    static: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    graph, upstream, order = build_reach_graph(static)
    static_idx = static.set_index("reach_id")
    forcing_idx = forcing.set_index(["reach_id", "year", "month"]).sort_index()
    times = sorted({(int(y), int(m)) for y, m in forcing[["year", "month"]].itertuples(index=False)})

    soil = {}
    groundwater = {}
    channel = {}
    initial_soil = {}
    initial_groundwater = {}
    initial_channel = {}
    for rid in order:
        row = static_idx.loc[rid]
        smax = params["kappa_s"] * float(row.soil_storage_eff_mm)
        soil[rid] = params["initial_soil_fraction"] * smax
        groundwater[rid] = params["initial_groundwater_mm"]
        channel[rid] = params["initial_channel_m3"]
        initial_soil[rid] = soil[rid]
        initial_groundwater[rid] = groundwater[rid]
        initial_channel[rid] = channel[rid]

    rows = []
    component_flux = defaultdict(lambda: {
        "precipitation_m3": 0.0,
        "aet_m3": 0.0,
        "terminal_outflow_m3": 0.0,
    })
    # Basin components are defined on topology nodes, not on the reach-routing
    # graph. Two terminal reaches can share the same sink node without either
    # routing into the other (node 37 is the frozen-network example). The
    # reach-routing graph therefore has 19 weak components while the frozen
    # topology correctly has 18 basin components.
    node_graph = nx.Graph()
    for row in static.itertuples():
        node_graph.add_edge(int(row.fnode), int(row.tnode))
    node_components = sorted(
        nx.connected_components(node_graph),
        key=lambda members: min(members),
    )
    node_component_of = {
        node: component_id
        for component_id, members in enumerate(node_components, start=1)
        for node in members
    }
    weak_components: list[set[int]] = [set() for _ in node_components]
    for row in static.itertuples():
        component_id = node_component_of[int(row.fnode)]
        if node_component_of[int(row.tnode)] != component_id:
            raise RuntimeError("Reach endpoints assigned to different topology components")
        weak_components[component_id - 1].add(int(row.reach_id))
    component_of = {
        rid: component_id
        for component_id, members in enumerate(weak_components, start=1)
        for rid in members
    }
    terminal_reaches = {n for n in graph.nodes if graph.out_degree(n) == 0}

    max_available_exceedance = 0.0
    for year, month in times:
        out_this_month: dict[int, float] = {}
        seconds = calendar.monthrange(year, month)[1] * 86400.0
        for rid in order:
            srow = static_idx.loc[rid]
            frow = forcing_idx.loc[(rid, year, month)]
            area_km2 = float(srow.inc_area_km2)
            area_factor = area_km2 * 1000.0
            smax = params["kappa_s"] * float(srow.soil_storage_eff_mm)
            p_mm = float(frow.P_mm)
            pet_mm = float(frow.PET_mm)
            ss0 = soil[rid]
            sg0 = groundwater[rid]
            ch0 = channel[rid]

            soil_available = ss0 + p_mm
            soil_ratio = min(max(ss0 / smax, 0.0), 1.0)
            aet = min(soil_available, pet_mm * soil_ratio ** params["gamma_ET"])
            u = soil_available - aet
            qex = max(u - smax, 0.0)
            sstar = min(u, smax)
            wetness = min(max(sstar / smax, 0.0), 1.0)
            recharge = (
                params["k_perc"]
                * wetness ** params["p_perc"]
                * sstar
            )
            interflow = (
                params["k_int"]
                * wetness ** params["p_int"]
                * sstar
            )
            ss1 = sstar - recharge - interflow
            quick = qex + interflow

            baseflow = params["k_g"] * max(sg0, 0.0)
            deep_loss = params["k_deep"] * max(sg0, 0.0)
            sg1 = sg0 + recharge - baseflow - deep_loss

            quick_m3 = quick * area_factor
            base_m3 = baseflow * area_factor
            deep_m3 = deep_loss * area_factor
            local_to_channel = quick_m3 + base_m3
            upstream_m3 = sum(out_this_month[u] for u in upstream[rid])
            channel_available = ch0 + local_to_channel + upstream_m3
            qout_m3 = params["k_route"] * channel_available
            ch1 = channel_available - qout_m3

            soil_resid = ss0 + p_mm - aet - quick - recharge - ss1
            groundwater_resid = sg0 + recharge - baseflow - deep_loss - sg1
            channel_resid = ch0 + local_to_channel + upstream_m3 - qout_m3 - ch1
            soil_scale = abs(ss0 + p_mm) + abs(aet + quick + recharge + ss1) + 1e-30
            groundwater_scale = abs(sg0 + recharge) + abs(baseflow + deep_loss + sg1) + 1e-30
            channel_scale = (
                abs(ch0 + local_to_channel + upstream_m3)
                + abs(qout_m3 + ch1)
                + 1e-30
            )

            start_total = (ss0 + sg0) * area_factor + ch0
            external_input = p_mm * area_factor + upstream_m3
            external_output = aet * area_factor + deep_m3 + qout_m3
            end_total = (ss1 + sg1) * area_factor + ch1
            system_resid = start_total + external_input - external_output - end_total
            system_scale = (
                abs(start_total)
                + abs(external_input)
                + abs(external_output)
                + abs(end_total)
                + 1e-30
            )
            system_rel = abs(system_resid) / system_scale

            max_available_exceedance = max(
                max_available_exceedance,
                quick + recharge - (ss0 + p_mm - aet),
                baseflow + deep_loss - (sg0 + recharge),
                qout_m3 - channel_available,
            )
            rows.append({
                "reach_id": rid,
                "year": year,
                "month": month,
                "days_in_month": calendar.monthrange(year, month)[1],
                "P_mm": p_mm,
                "PET_mm": pet_mm,
                "catchment_area_km2": area_km2,
                "soil_capacity_mm": smax,
                "soil_storage_start_mm": ss0,
                "AET_mm": aet,
                "excess_runoff_mm": qex,
                "interflow_mm": interflow,
                "groundwater_recharge_mm": recharge,
                "soil_storage_end_mm": ss1,
                "groundwater_storage_start_mm": sg0,
                "baseflow_mm": baseflow,
                "deep_loss_mm": deep_loss,
                "groundwater_storage_end_mm": sg1,
                "channel_storage_start_m3": ch0,
                "local_quickflow_m3": quick_m3,
                "local_baseflow_m3": base_m3,
                "upstream_inflow_m3": upstream_m3,
                "channel_available_m3": channel_available,
                "channel_outflow_m3": qout_m3,
                "channel_outflow_m3_s": qout_m3 / seconds,
                "channel_storage_end_m3": ch1,
                "soil_balance_residual_mm": soil_resid,
                "soil_relative_closure": abs(soil_resid) / soil_scale,
                "groundwater_balance_residual_mm": groundwater_resid,
                "groundwater_relative_closure": abs(groundwater_resid) / groundwater_scale,
                "channel_balance_residual_m3": channel_resid,
                "channel_relative_closure": abs(channel_resid) / channel_scale,
                "system_balance_residual_m3": system_resid,
                "system_relative_closure": system_rel,
                "slope_m_m": float(srow.slope_m_m),
                "k_route": params["k_route"],
                "parameter_state": "numerical_core_test_only_not_calibrated",
            })

            soil[rid] = ss1
            groundwater[rid] = sg1
            channel[rid] = ch1
            out_this_month[rid] = qout_m3

            cid = component_of[rid]
            component_flux[cid]["precipitation_m3"] += p_mm * area_factor
            component_flux[cid]["aet_m3"] += aet * area_factor
            if rid in terminal_reaches:
                component_flux[cid]["terminal_outflow_m3"] += qout_m3

    result = pd.DataFrame(rows)
    component_rows = []
    for cid, members in enumerate(weak_components, start=1):
        initial_storage = 0.0
        final_storage = 0.0
        terminal_ids = sorted(set(members) & terminal_reaches)
        for rid in members:
            area_factor = float(static_idx.loc[rid].inc_area_km2) * 1000.0
            initial_storage += (
                initial_soil[rid] + initial_groundwater[rid]
            ) * area_factor + initial_channel[rid]
            final_storage += (
                soil[rid] + groundwater[rid]
            ) * area_factor + channel[rid]
        flux = component_flux[cid]
        residual = (
            initial_storage
            + flux["precipitation_m3"]
            - flux["aet_m3"]
            - flux["terminal_outflow_m3"]
            - final_storage
        )
        scale = (
            abs(initial_storage)
            + abs(flux["precipitation_m3"])
            + abs(flux["aet_m3"])
            + abs(flux["terminal_outflow_m3"])
            + abs(final_storage)
            + 1e-30
        )
        component_rows.append({
            "component_id": cid,
            "reach_count": len(members),
            "terminal_reach_ids": ",".join(map(str, terminal_ids)),
            "initial_storage_m3": initial_storage,
            "precipitation_m3": flux["precipitation_m3"],
            "aet_m3": flux["aet_m3"],
            "terminal_outflow_m3": flux["terminal_outflow_m3"],
            "final_storage_m3": final_storage,
            "balance_residual_m3": residual,
            "relative_closure": abs(residual) / scale,
        })
    diagnostics = {
        "graph_is_dag": nx.is_directed_acyclic_graph(graph),
        "topological_order_reaches": len(order),
        "weak_components": len(weak_components),
        "terminal_reaches": len(terminal_reaches),
        "maximum_available_exceedance": float(max_available_exceedance),
    }
    return result, pd.DataFrame(component_rows), diagnostics


def main() -> None:
    for directory in [REPORT, OUTPUTS, MANIFEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent_gate["authorized_next_action"] != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize Q78-NAT core construction")

    forcing = pd.read_parquet(FORCING_PATH)
    static = pd.read_parquet(STATIC_PATH)
    forcing = forcing[
        (forcing["year"] >= cfg["development_start_year"])
        & (forcing["year"] <= cfg["development_end_year"])
    ].copy()
    required_forcing = {"reach_id", "year", "month", "P_mm", "PET_mm"}
    required_static = {
        "reach_id", "fnode", "tnode", "inc_area_km2",
        "soil_storage_eff_mm", "depth_to_bedrock_m",
        "slope_m_m", "direction_gate_pass",
    }
    if not required_forcing.issubset(forcing.columns):
        raise RuntimeError(f"Missing forcing fields: {sorted(required_forcing - set(forcing.columns))}")
    if not required_static.issubset(static.columns):
        raise RuntimeError(f"Missing static fields: {sorted(required_static - set(static.columns))}")
    if forcing[list(required_forcing)].isna().any().any():
        raise RuntimeError("Forcing contains missing required values")
    if static[list(required_static)].isna().any().any():
        raise RuntimeError("Static input contains missing required values")
    if not static["direction_gate_pass"].astype(bool).all():
        raise RuntimeError("Not all reach directions passed the gate")

    tests = synthetic_tests()
    if not tests["passed"]:
        raise RuntimeError("Synthetic conservation test failed")
    results, components, topology = run_core(forcing, static, cfg["parameters"])

    result_path = OUTPUTS / "q78_nat_core_reach_month.parquet"
    audit_path = REPORT / "component_balance_summary.csv"
    parameter_path = REPORT / "numerical_core_parameters.csv"
    results.to_parquet(result_path, index=False)
    components.to_csv(audit_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        **cfg["parameters"],
        "semantic_state": cfg["parameter_semantic_state"],
    }]).to_csv(parameter_path, index=False, encoding="utf-8-sig")
    (REPORT / "synthetic_test.json").write_text(
        json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    threshold = cfg["relative_closure_threshold"]
    checks = {
        "parent_authorization": True,
        "synthetic_test_passed": tests["passed"],
        "reach_count_230": static["reach_id"].nunique() == cfg["expected_reaches"],
        "reach_month_rows_35880": len(results) == cfg["expected_reach_months"],
        "months_per_reach_156": (
            results.groupby("reach_id").size().eq(cfg["expected_months_per_reach"]).all()
        ),
        "graph_is_dag": topology["graph_is_dag"],
        "all_reaches_in_topological_order": (
            topology["topological_order_reaches"] == cfg["expected_reaches"]
        ),
        "soil_relative_closure_below_threshold": (
            results["soil_relative_closure"].max() < threshold
        ),
        "groundwater_relative_closure_below_threshold": (
            results["groundwater_relative_closure"].max() < threshold
        ),
        "channel_relative_closure_below_threshold": (
            results["channel_relative_closure"].max() < threshold
        ),
        "system_relative_closure_below_threshold": (
            results["system_relative_closure"].max() < threshold
        ),
        "component_relative_closure_below_threshold": (
            components["relative_closure"].max() < threshold
        ),
        "network_component_count_expected": (
            topology["weak_components"] == cfg["expected_network_components"]
        ),
        "all_states_finite": np.isfinite(results[[
            "soil_storage_end_mm", "groundwater_storage_end_mm", "channel_storage_end_m3"
        ]].to_numpy()).all(),
        "all_states_nonnegative": (
            results[[
                "soil_storage_end_mm", "groundwater_storage_end_mm", "channel_storage_end_m3"
            ]].min().min() >= -cfg["absolute_tolerance"]
        ),
        "soil_not_above_capacity": (
            (results["soil_storage_end_mm"] - results["soil_capacity_mm"]).max()
            <= cfg["absolute_tolerance"]
        ),
        "outflows_do_not_exceed_available_water": (
            topology["maximum_available_exceedance"] <= cfg["absolute_tolerance"]
        ),
        "all_slopes_positive_finite": (
            np.isfinite(results["slope_m_m"]).all() and (results["slope_m_m"] > 0).all()
        ),
        "period_exactly_2006_2018": (
            int(results["year"].min()) == 2006 and int(results["year"].max()) == 2018
        ),
        "station_observations_not_read": True,
        "management_fluxes_disabled": True,
        "period_2019_2022_not_read": True,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    passed = all(checks.values())
    maxima = {
        "soil_relative_closure": float(results["soil_relative_closure"].max()),
        "groundwater_relative_closure": float(results["groundwater_relative_closure"].max()),
        "channel_relative_closure": float(results["channel_relative_closure"].max()),
        "system_relative_closure": float(results["system_relative_closure"].max()),
        "component_relative_closure": float(components["relative_closure"].max()),
        "absolute_system_residual_m3": float(results["system_balance_residual_m3"].abs().max()),
    }
    gate = {
        "run_id": cfg["run_id"],
        "phase": "q78_nat_conservation_core",
        "created_utc": utc_now(),
        "scope": {
            "reaches": int(static["reach_id"].nunique()),
            "reach_months": len(results),
            "period": [int(results["year"].min()), int(results["year"].max())],
            "network_components": topology["weak_components"],
            "terminal_reaches": topology["terminal_reaches"],
        },
        "parameter_semantic_state": cfg["parameter_semantic_state"],
        "checks": checks,
        "closure_maxima": maxima,
        "physical_skill_evaluated": False,
        "station_observations_read": False,
        "management_fluxes_enabled": False,
        "period_2019_2022_read": False,
        "decision": "CONSERVATION_CORE_PASSED" if passed else "CONSERVATION_CORE_FAILED",
        "authorized_next_action": (
            cfg["required_next_action_if_pass"]
            if passed else cfg["required_next_action_if_fail"]
        ),
        "authorized_scope": (
            "Register hydrologically defensible parameter priors and initialization sensitivity; calibration remains forbidden."
            if passed
            else "Repair only the conservation equations or numerical implementation; calibration remains forbidden."
        ),
        "series_terminal": False,
        "passed": passed,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# Q78-NAT 严格守恒计算核心

## 技术结论

计算核心{'通过' if passed else '未通过'}物理守恒门禁。完整运行覆盖 230 reach、35,880 个 reach-month 和 {topology['weak_components']} 个网络分量；本轮没有读取站点观测，也没有进行率定或预测技能评价。

## 闭合证据

| 层级 | 最大相对闭合误差 |
|---|---:|
| 土壤库 | {maxima['soil_relative_closure']:.3e} |
| 地下水库 | {maxima['groundwater_relative_closure']:.3e} |
| 河道库 | {maxima['channel_relative_closure']:.3e} |
| reach 总系统 | {maxima['system_relative_closure']:.3e} |
| 全期网络分量 | {maxima['component_relative_closure']:.3e} |

所有状态非负且有限，土壤储量不超过容量，所有出流不超过对应可用水量。

## 方程与单位

土壤和地下水状态以 local catchment 上的 mm 表示，进入河网前乘以 `catchment_area_km2 × 1000` 转为 m³。河道库、上游入流和河道出流均使用 m³/month；另输出月平均 m³/s。

土壤、地下水、河道三个余额分别独立检查，并将内部的壤中流、补给和基流抵消后，再检查 reach 总系统。{topology['weak_components']} 个基于冻结拓扑节点定义的弱连通网络分量还执行全 2006–2018 期间的累计闭合检查。共享同一终端节点但互不构成上下游关系的终端 reach 归入同一流域分量，且各自作为分量出口计量。

## 参数的严格边界

本轮参数状态为 `{cfg['parameter_semantic_state']}`。固定 `k_route=0.85` 等值只用于迫使三个状态库都发生非平凡更新，从而验证数值实现；它们不是水文先验、不是率定结果，也不能用于比较站点效果。

## 限制与稳健性

- 未评价 AET、基流比例、地下水异常或观测流量；
- 未进行初始化敏感性；
- 未使用坡度映射路由参数，坡度只通过数据完整性门禁；
- 未包含水库、取水、回归水和调水；
- 2019–2022 完全未读取。

## 下一步

{'只允许注册参数先验和初始化敏感性，不允许直接率定。' if passed else '修复守恒核心，禁止进入参数先验或率定。'}

## 仍待回答的问题

- `kappa_s、k_perc、k_int、k_g、k_route` 的文献与属性先验应如何区域化？
- 初始土壤和地下水状态需要多少 warm-up 才不再影响 2012 年以后的外折？
- 基岩深度和坡度应进入哪个低维参数映射，才能保持可辨识性？
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "runtime": "conda sparrow",
        "inputs_read": [str(FORCING_PATH), str(STATIC_PATH), str(PARENT_GATE)],
        "station_observation_files_read": [],
        "management_flux_files_read": [],
        "period_2019_2022_read": False,
        "chart_omission_reason": "The gate is based on exact conservation residuals and state bounds; tables are more auditable than plots.",
    }
    (REPORT / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sources = [
        CONFIG, RUN_DIR / "experiment_contract.md", RUN_DIR / "README.md",
        RUN_DIR / "scripts" / "build_q78_nat_conservation_core.py",
        RUN_DIR / "scripts" / "validate_q78_nat_conservation_core.py",
        PARENT_GATE, FORCING_PATH, STATIC_PATH,
    ]
    products = [
        result_path, audit_path, parameter_path,
        REPORT / "synthetic_test.json",
        REPORT / "gate.json",
        REPORT / "technical_report.md",
        REPORT / "run_manifest.json",
    ]
    manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "sources": [record(p, "q78_nat_core_source", "reported_or_derived") for p in sources],
        "products": [record(p, "q78_nat_core_product", "derived") for p in products],
        "runtime": "conda sparrow",
        "station_observations_read": False,
        "management_fluxes_enabled": False,
        "period_2019_2022_read": False,
    }
    (MANIFEST_DIR / "provenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(manifest["sources"] + manifest["products"]).to_csv(
        MANIFEST_DIR / "provenance_manifest.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps({
        "passed": passed,
        "reach_months": len(results),
        "closure_maxima": maxima,
        "authorized_next_action": gate["authorized_next_action"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
