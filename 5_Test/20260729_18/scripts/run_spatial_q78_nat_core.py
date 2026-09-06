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
PARENT = ROOT / "5_Test" / "20260729_17"
LEDGER = ROOT / "5_Test" / "20260729_9"
STATIC_RUN = ROOT / "5_Test" / "20260729_13"
REPORT = RUN_DIR / "reports" / "spatial_q78_nat_core_gate"
OUTPUTS = RUN_DIR / "outputs"
MANIFEST_DIR = RUN_DIR / "inputs_manifest"
CONFIG = RUN_DIR / "config" / "spatial_q78_nat_core_contract.json"
PARENT_GATE = PARENT / "reports" / "attribute_parameter_mapping_gate" / "gate.json"
PARAMETER_MAP = PARENT / "outputs" / "reach_attribute_parameter_map.parquet"
FORCING_PATH = LEDGER / "inputs" / "reach_month_forcing_2006_2018.parquet"
STATIC_PATH = STATIC_RUN / "inputs" / "reach_static_direction_final.parquet"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path, role: str, state: str) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": state,
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def prepare_network(static: pd.DataFrame):
    reach_graph = nx.DiGraph()
    reach_graph.add_nodes_from(static["reach_id"].astype(int))
    incoming_by_node = defaultdict(list)
    for row in static.itertuples():
        incoming_by_node[int(row.tnode)].append(int(row.reach_id))
    upstream = {}
    for row in static.itertuples():
        reach = int(row.reach_id)
        upstream[reach] = [
            item for item in incoming_by_node.get(int(row.fnode), [])
            if item != reach
        ]
        for item in upstream[reach]:
            reach_graph.add_edge(item, reach)
    if not nx.is_directed_acyclic_graph(reach_graph):
        raise RuntimeError("Frozen reach graph is not a DAG")
    order = list(nx.topological_sort(reach_graph))
    terminal = {reach for reach in order if reach_graph.out_degree(reach) == 0}

    node_graph = nx.Graph()
    for row in static.itertuples():
        node_graph.add_edge(int(row.fnode), int(row.tnode))
    node_components = sorted(nx.connected_components(node_graph), key=min)
    node_component_of = {
        node: component
        for component, members in enumerate(node_components, start=1)
        for node in members
    }
    component_of_reach = {
        int(row.reach_id): node_component_of[int(row.fnode)]
        for row in static.itertuples()
    }
    return order, upstream, terminal, node_components, component_of_reach


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent_gate["authorized_next_action"] != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize spatial Q78-NAT")

    forcing = pd.read_parquet(FORCING_PATH)
    static = pd.read_parquet(STATIC_PATH)
    parameter_map = pd.read_parquet(PARAMETER_MAP)
    forcing = forcing[
        (forcing["year"] >= cfg["actual_period"][0])
        & (forcing["year"] <= cfg["actual_period"][1])
    ].copy()
    if len(forcing) != cfg["expected_reach_months"]:
        raise RuntimeError("Forcing scope mismatch")
    if static["reach_id"].nunique() != cfg["expected_reaches"]:
        raise RuntimeError("Static reach scope mismatch")
    if parameter_map["reach_id"].nunique() != cfg["expected_reaches"]:
        raise RuntimeError("Parameter-map scope mismatch")

    order, upstream_ids, terminal, node_components, component_of_reach = prepare_network(static)
    reaches = np.array(order, dtype=int)
    index_of = {reach: index for index, reach in enumerate(reaches)}
    upstream = {
        index_of[reach]: [index_of[item] for item in upstream_ids[reach]]
        for reach in reaches
    }
    static_indexed = static.set_index("reach_id").loc[reaches]
    mapped = parameter_map.set_index("reach_id").loc[reaches]
    forcing_indexed = forcing.set_index(["year", "month", "reach_id"]).sort_index()
    area = static_indexed["inc_area_km2"].to_numpy(float) * 1000.0
    parameters = {
        name: mapped[name].to_numpy(float)
        for name in [
            "kappa_s", "gamma_ET", "k_perc", "p_perc",
            "k_int", "p_int", "k_g", "k_route", "k_deep",
        ]
    }
    soil_capacity = parameters["kappa_s"] * static_indexed[
        "soil_storage_eff_mm"
    ].to_numpy(float)
    soil = cfg["initialization"]["soil_fraction"] * soil_capacity
    groundwater = np.full(
        len(reaches), cfg["initialization"]["groundwater_mm"], dtype=float
    )
    channel = np.full(
        len(reaches), cfg["initialization"]["channel_m3"], dtype=float
    )

    def forcing_block(year: int, month: int):
        block = forcing_indexed.loc[(year, month)].loc[reaches]
        return block["P_mm"].to_numpy(float), block["PET_mm"].to_numpy(float)

    maximum_available_exceedance = 0.0

    def advance(p_mm: np.ndarray, pet_mm: np.ndarray):
        nonlocal soil, groundwater, channel, maximum_available_exceedance
        soil_start = soil.copy()
        groundwater_start = groundwater.copy()
        channel_start = channel.copy()
        soil_available = soil_start + p_mm
        soil_ratio = np.clip(soil_start / soil_capacity, 0.0, 1.0)
        aet = np.minimum(
            soil_available,
            pet_mm * soil_ratio ** parameters["gamma_ET"],
        )
        after_et = soil_available - aet
        excess = np.maximum(after_et - soil_capacity, 0.0)
        temporary_soil = np.minimum(after_et, soil_capacity)
        wetness = np.clip(temporary_soil / soil_capacity, 0.0, 1.0)
        recharge = (
            parameters["k_perc"]
            * wetness ** parameters["p_perc"]
            * temporary_soil
        )
        interflow = (
            parameters["k_int"]
            * wetness ** parameters["p_int"]
            * temporary_soil
        )
        soil_end = temporary_soil - recharge - interflow
        baseflow = parameters["k_g"] * np.maximum(groundwater_start, 0.0)
        deep_loss = parameters["k_deep"] * np.maximum(groundwater_start, 0.0)
        groundwater_end = groundwater_start + recharge - baseflow - deep_loss
        local_quick = (excess + interflow) * area
        local_base = baseflow * area
        upstream_inflow = np.zeros(len(reaches))
        outflow = np.zeros(len(reaches))
        channel_end = np.zeros(len(reaches))
        for position in range(len(reaches)):
            parents = upstream[position]
            if parents:
                upstream_inflow[position] = outflow[parents].sum()
            available = (
                channel_start[position]
                + local_quick[position]
                + local_base[position]
                + upstream_inflow[position]
            )
            outflow[position] = parameters["k_route"][position] * available
            channel_end[position] = available - outflow[position]
        maximum_available_exceedance = max(
            maximum_available_exceedance,
            float(np.max(recharge + interflow - temporary_soil)),
            float(np.max(baseflow + deep_loss - (groundwater_start + recharge))),
        )
        soil, groundwater, channel = soil_end, groundwater_end, channel_end
        return {
            "soil_start": soil_start,
            "groundwater_start": groundwater_start,
            "channel_start": channel_start,
            "aet": aet,
            "excess": excess,
            "recharge": recharge,
            "interflow": interflow,
            "baseflow": baseflow,
            "deep_loss": deep_loss,
            "upstream_inflow": upstream_inflow,
            "outflow": outflow,
            "soil_end": soil_end,
            "groundwater_end": groundwater_end,
            "channel_end": channel_end,
            "p_mm": p_mm,
            "pet_mm": pet_mm,
            "local_quick": local_quick,
            "local_base": local_base,
        }

    spinup_times = [
        (year, month)
        for year in range(cfg["spinup_forcing_period"][0], cfg["spinup_forcing_period"][1] + 1)
        for month in range(1, 13)
    ]
    for _ in range(cfg["spinup_cycles"]):
        for year, month in spinup_times:
            advance(*forcing_block(year, month))

    initial_soil = soil.copy()
    initial_groundwater = groundwater.copy()
    initial_channel = channel.copy()
    component_flux = defaultdict(lambda: {
        "precipitation_m3": 0.0,
        "aet_m3": 0.0,
        "terminal_outflow_m3": 0.0,
    })
    rows = []
    actual_times = [
        (year, month)
        for year in range(cfg["actual_period"][0], cfg["actual_period"][1] + 1)
        for month in range(1, 13)
    ]
    for year, month in actual_times:
        step = advance(*forcing_block(year, month))
        soil_residual = (
            step["soil_start"] + step["p_mm"] - step["aet"]
            - step["excess"] - step["interflow"] - step["recharge"]
            - step["soil_end"]
        )
        groundwater_residual = (
            step["groundwater_start"] + step["recharge"]
            - step["baseflow"] - step["deep_loss"]
            - step["groundwater_end"]
        )
        channel_residual = (
            step["channel_start"] + step["local_quick"] + step["local_base"]
            + step["upstream_inflow"] - step["outflow"] - step["channel_end"]
        )
        start_total = (
            (step["soil_start"] + step["groundwater_start"]) * area
            + step["channel_start"]
        )
        external_input = step["p_mm"] * area + step["upstream_inflow"]
        external_output = (
            step["aet"] * area + step["deep_loss"] * area + step["outflow"]
        )
        end_total = (
            (step["soil_end"] + step["groundwater_end"]) * area
            + step["channel_end"]
        )
        system_residual = start_total + external_input - external_output - end_total
        soil_scale = (
            np.abs(step["soil_start"] + step["p_mm"])
            + np.abs(
                step["aet"] + step["excess"] + step["interflow"]
                + step["recharge"] + step["soil_end"]
            ) + 1e-30
        )
        groundwater_scale = (
            np.abs(step["groundwater_start"] + step["recharge"])
            + np.abs(step["baseflow"] + step["deep_loss"] + step["groundwater_end"])
            + 1e-30
        )
        channel_scale = (
            np.abs(
                step["channel_start"] + step["local_quick"]
                + step["local_base"] + step["upstream_inflow"]
            )
            + np.abs(step["outflow"] + step["channel_end"])
            + 1e-30
        )
        system_scale = (
            np.abs(start_total) + np.abs(external_input)
            + np.abs(external_output) + np.abs(end_total) + 1e-30
        )
        for position, reach in enumerate(reaches):
            rows.append({
                "reach_id": int(reach),
                "year": year,
                "month": month,
                "P_mm": step["p_mm"][position],
                "PET_mm": step["pet_mm"][position],
                "soil_capacity_mm": soil_capacity[position],
                "soil_storage_start_mm": step["soil_start"][position],
                "AET_mm": step["aet"][position],
                "excess_runoff_mm": step["excess"][position],
                "interflow_mm": step["interflow"][position],
                "groundwater_recharge_mm": step["recharge"][position],
                "soil_storage_end_mm": step["soil_end"][position],
                "groundwater_storage_start_mm": step["groundwater_start"][position],
                "baseflow_mm": step["baseflow"][position],
                "groundwater_storage_end_mm": step["groundwater_end"][position],
                "channel_storage_start_m3": step["channel_start"][position],
                "upstream_inflow_m3": step["upstream_inflow"][position],
                "channel_outflow_m3": step["outflow"][position],
                "channel_storage_end_m3": step["channel_end"][position],
                "soil_relative_closure": abs(soil_residual[position]) / soil_scale[position],
                "groundwater_relative_closure": abs(groundwater_residual[position]) / groundwater_scale[position],
                "channel_relative_closure": abs(channel_residual[position]) / channel_scale[position],
                "system_relative_closure": abs(system_residual[position]) / system_scale[position],
                "system_balance_residual_m3": system_residual[position],
                **{name: values[position] for name, values in parameters.items()},
                "parameter_semantic_state": "fixed_attribute_mapping_not_calibrated",
            })
            component = component_of_reach[int(reach)]
            component_flux[component]["precipitation_m3"] += step["p_mm"][position] * area[position]
            component_flux[component]["aet_m3"] += step["aet"][position] * area[position]
            if int(reach) in terminal:
                component_flux[component]["terminal_outflow_m3"] += step["outflow"][position]

    results = pd.DataFrame(rows)
    component_rows = []
    for component, nodes in enumerate(node_components, start=1):
        members = [
            position for position, reach in enumerate(reaches)
            if component_of_reach[int(reach)] == component
        ]
        initial_storage = float(np.sum(
            (initial_soil[members] + initial_groundwater[members]) * area[members]
            + initial_channel[members]
        ))
        final_storage = float(np.sum(
            (soil[members] + groundwater[members]) * area[members]
            + channel[members]
        ))
        flux = component_flux[component]
        residual = (
            initial_storage + flux["precipitation_m3"] - flux["aet_m3"]
            - flux["terminal_outflow_m3"] - final_storage
        )
        scale = (
            abs(initial_storage) + abs(flux["precipitation_m3"])
            + abs(flux["aet_m3"]) + abs(flux["terminal_outflow_m3"])
            + abs(final_storage) + 1e-30
        )
        component_rows.append({
            "component_id": component,
            "reach_count": len(members),
            "balance_residual_m3": residual,
            "relative_closure": abs(residual) / scale,
        })
    components = pd.DataFrame(component_rows)

    output_path = OUTPUTS / "spatial_q78_nat_reach_month.parquet"
    component_path = REPORT / "component_balance_summary.csv"
    results.to_parquet(output_path, index=False)
    components.to_csv(component_path, index=False, encoding="utf-8-sig")
    threshold = cfg["relative_closure_threshold"]
    parameter_match = all(
        np.allclose(
            results.groupby("reach_id")[name].first().loc[parameter_map["reach_id"]].to_numpy(),
            parameter_map.set_index("reach_id")[name].loc[parameter_map["reach_id"]].to_numpy(),
            rtol=0.0,
            atol=0.0,
        )
        for name in parameters
    )
    checks = {
        "parent_authorization": True,
        "reach_count_230": results["reach_id"].nunique() == cfg["expected_reaches"],
        "reach_month_rows_35880": len(results) == cfg["expected_reach_months"],
        "months_per_reach_156": results.groupby("reach_id").size().eq(156).all(),
        "network_components_18": len(components) == cfg["expected_network_components"],
        "soil_closure": results["soil_relative_closure"].max() < threshold,
        "groundwater_closure": results["groundwater_relative_closure"].max() < threshold,
        "channel_closure": results["channel_relative_closure"].max() < threshold,
        "system_closure": results["system_relative_closure"].max() < threshold,
        "component_closure": components["relative_closure"].max() < threshold,
        "states_nonnegative_finite": (
            np.isfinite(results[[
                "soil_storage_end_mm", "groundwater_storage_end_mm",
                "channel_storage_end_m3",
            ]]).all().all()
            and results[[
                "soil_storage_end_mm", "groundwater_storage_end_mm",
                "channel_storage_end_m3",
            ]].min().min() >= -1e-9
        ),
        "soil_not_above_capacity": (
            (results["soil_storage_end_mm"] - results["soil_capacity_mm"]).max() <= 1e-9
        ),
        "outflows_within_available_water": maximum_available_exceedance <= 1e-9,
        "parameters_exactly_match_map": parameter_match,
        "station_observations_not_read": True,
        "management_fluxes_disabled": True,
        "period_2019_2022_not_read": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    next_action = cfg["next_action_if_pass"] if passed else cfg["next_action_if_fail"]
    maxima = {
        "soil_relative_closure": float(results["soil_relative_closure"].max()),
        "groundwater_relative_closure": float(results["groundwater_relative_closure"].max()),
        "channel_relative_closure": float(results["channel_relative_closure"].max()),
        "system_relative_closure": float(results["system_relative_closure"].max()),
        "component_relative_closure": float(components["relative_closure"].max()),
    }
    gate = {
        "run_id": cfg["run_id"],
        "phase": "spatial_q78_nat_core_gate",
        "created_utc": utc_now(),
        "checks": checks,
        "scope": {
            "reaches": 230,
            "reach_months": len(results),
            "period": cfg["actual_period"],
            "network_components": len(components),
            "terminal_reaches": len(terminal),
            "spinup_cycles": cfg["spinup_cycles"],
        },
        "closure_maxima": maxima,
        "decision": "SPATIAL_Q78_NAT_CORE_PASSED" if passed else "SPATIAL_Q78_NAT_CORE_FAILED",
        "authorized_next_action": next_action,
        "authorized_scope": (
            "Assess only internal process plausibility without station-flow calibration."
            if passed else
            "Repair only the spatial-parameter core implementation; calibration remains forbidden."
        ),
        "station_observations_read": False,
        "management_fluxes_enabled": False,
        "period_2019_2022_read": False,
        "series_terminal": False,
        "passed": passed,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 空间参数化 Q78-NAT 守恒核心

## 结论

空间参数核心{'通过' if passed else '未通过'}。运行覆盖 230 reach、35,880 reach-month、18 个冻结拓扑流域分量，并严格使用 `_17` 的参数映射。

| 层级 | 最大相对闭合误差 |
|---|---:|
| 土壤 | {maxima['soil_relative_closure']:.3e} |
| 地下水 | {maxima['groundwater_relative_closure']:.3e} |
| 河道 | {maxima['channel_relative_closure']:.3e} |
| reach总系统 | {maxima['system_relative_closure']:.3e} |
| 流域分量全期 | {maxima['component_relative_closure']:.3e} |

本轮只证明空间参数实现守恒，不证明内部通量比例或站点预测合理。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "runtime": "conda sparrow",
        "inputs_read": [str(PARENT_GATE), str(PARAMETER_MAP), str(FORCING_PATH), str(STATIC_PATH)],
        "station_observation_files_read": [],
        "management_flux_files_read": [],
        "period_2019_2022_read": False,
    }
    (REPORT / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sources = [
        CONFIG, RUN_DIR / "experiment_contract.md", RUN_DIR / "README.md",
        RUN_DIR / "scripts" / "run_spatial_q78_nat_core.py",
        RUN_DIR / "scripts" / "validate_spatial_q78_nat_core.py",
        PARENT_GATE, PARAMETER_MAP, FORCING_PATH, STATIC_PATH,
    ]
    products = [
        output_path, component_path, REPORT / "gate.json",
        REPORT / "technical_report.md", REPORT / "run_manifest.json",
    ]
    manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "sources": [record(path, "spatial_core_source", "reported_or_derived") for path in sources],
        "products": [record(path, "spatial_core_product", "derived") for path in products],
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
        "closure_maxima": maxima,
        "authorized_next_action": next_action,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

