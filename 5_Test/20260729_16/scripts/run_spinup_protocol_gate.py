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
PARENT = ROOT / "5_Test" / "20260729_15"
LEDGER = ROOT / "5_Test" / "20260729_9"
STATIC_RUN = ROOT / "5_Test" / "20260729_13"
REPORT = RUN_DIR / "reports" / "spinup_protocol_gate"
OUTPUTS = RUN_DIR / "outputs"
MANIFEST_DIR = RUN_DIR / "inputs_manifest"
CONFIG = RUN_DIR / "config" / "spinup_protocol_contract.json"
PARENT_GATE = PARENT / "reports" / "parameter_prior_initialization_gate" / "gate.json"
PRIOR_CONFIG = PARENT / "config" / "parameter_prior_initialization_contract.json"
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


def relative_range(values: np.ndarray, axis: int = 0) -> np.ndarray:
    maximum = np.max(values, axis=axis)
    minimum = np.min(values, axis=axis)
    scale = np.maximum(np.max(np.abs(values), axis=axis), 1.0)
    return (maximum - minimum) / scale


def prepare_network(static: pd.DataFrame):
    graph = nx.DiGraph()
    graph.add_nodes_from(static["reach_id"].astype(int))
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
            graph.add_edge(item, reach)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Frozen reach graph is not a DAG")
    order = list(nx.topological_sort(graph))
    terminal = [reach for reach in order if graph.out_degree(reach) == 0]
    return order, upstream, terminal


def run_ensemble(
    forcing: pd.DataFrame,
    static: pd.DataFrame,
    scenario: dict,
    initializations: dict,
    cfg: dict,
    actual_start_cycle: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    order, upstream_ids, terminal_ids = prepare_network(static)
    reaches = np.array(order, dtype=int)
    reach_index = {reach: index for index, reach in enumerate(reaches)}
    upstream = {
        reach_index[reach]: [reach_index[item] for item in upstream_ids[reach]]
        for reach in reaches
    }
    terminal_index = np.array([reach_index[item] for item in terminal_ids], dtype=int)
    static_indexed = static.set_index("reach_id").loc[reaches]
    area_factor = static_indexed["inc_area_km2"].to_numpy(float) * 1000.0
    soil_capacity = (
        scenario["kappa_s"]
        * static_indexed["soil_storage_eff_mm"].to_numpy(float)
    )
    variant_names = list(initializations)
    soil = np.vstack([
        np.full(len(reaches), initializations[name]["initial_soil_fraction"])
        * soil_capacity
        for name in variant_names
    ])
    groundwater = np.vstack([
        np.full(len(reaches), initializations[name]["initial_groundwater_mm"])
        for name in variant_names
    ])
    channel = np.vstack([
        np.full(len(reaches), initializations[name]["initial_channel_m3"])
        for name in variant_names
    ])

    indexed = forcing.set_index(["year", "month", "reach_id"]).sort_index()

    def forcing_arrays(start_year: int, end_year: int):
        keys = [
            (year, month)
            for year in range(start_year, end_year + 1)
            for month in range(1, 13)
        ]
        precipitation = []
        pet = []
        for year, month in keys:
            block = indexed.loc[(year, month)].loc[reaches]
            precipitation.append(block["P_mm"].to_numpy(float))
            pet.append(block["PET_mm"].to_numpy(float))
        return keys, np.vstack(precipitation), np.vstack(pet)

    cycle_keys, cycle_p, cycle_pet = forcing_arrays(*cfg["spinup_forcing_period"])
    actual_keys, actual_p, actual_pet = forcing_arrays(*cfg["actual_diagnostic_period"])
    maximum_system_relative_closure = 0.0
    all_states_nonnegative_finite = True

    def advance(p_mm: np.ndarray, pet_mm: np.ndarray):
        nonlocal soil, groundwater, channel
        nonlocal maximum_system_relative_closure, all_states_nonnegative_finite
        soil_start = soil
        groundwater_start = groundwater
        channel_start = channel
        soil_available = soil_start + p_mm[None, :]
        soil_ratio = np.clip(soil_start / soil_capacity[None, :], 0.0, 1.0)
        aet = np.minimum(
            soil_available,
            pet_mm[None, :] * soil_ratio ** scenario["gamma_ET"],
        )
        after_et = soil_available - aet
        excess = np.maximum(after_et - soil_capacity[None, :], 0.0)
        temporary_soil = np.minimum(after_et, soil_capacity[None, :])
        wetness = np.clip(temporary_soil / soil_capacity[None, :], 0.0, 1.0)
        recharge = (
            scenario["k_perc"]
            * wetness ** scenario["p_perc"]
            * temporary_soil
        )
        interflow = (
            scenario["k_int"]
            * wetness ** scenario["p_int"]
            * temporary_soil
        )
        soil_end = temporary_soil - recharge - interflow
        baseflow = scenario["k_g"] * np.maximum(groundwater_start, 0.0)
        deep_loss = scenario["k_deep"] * np.maximum(groundwater_start, 0.0)
        groundwater_end = (
            groundwater_start + recharge - baseflow - deep_loss
        )
        local_to_channel = (
            excess + interflow + baseflow
        ) * area_factor[None, :]
        outflow = np.zeros_like(channel_start)
        channel_end = np.zeros_like(channel_start)
        upstream_inflow = np.zeros_like(channel_start)
        for reach_position in range(len(reaches)):
            parents = upstream[reach_position]
            if parents:
                upstream_inflow[:, reach_position] = outflow[:, parents].sum(axis=1)
            available = (
                channel_start[:, reach_position]
                + local_to_channel[:, reach_position]
                + upstream_inflow[:, reach_position]
            )
            outflow[:, reach_position] = scenario["k_route"] * available
            channel_end[:, reach_position] = available - outflow[:, reach_position]

        start_total = (
            (soil_start + groundwater_start) * area_factor[None, :]
            + channel_start
        )
        external_input = p_mm[None, :] * area_factor[None, :] + upstream_inflow
        external_output = (
            aet * area_factor[None, :]
            + deep_loss * area_factor[None, :]
            + outflow
        )
        end_total = (
            (soil_end + groundwater_end) * area_factor[None, :]
            + channel_end
        )
        residual = start_total + external_input - external_output - end_total
        scale = (
            np.abs(start_total)
            + np.abs(external_input)
            + np.abs(external_output)
            + np.abs(end_total)
            + 1e-30
        )
        maximum_system_relative_closure = max(
            maximum_system_relative_closure,
            float(np.max(np.abs(residual) / scale)),
        )
        all_states_nonnegative_finite &= bool(
            np.isfinite(soil_end).all()
            and np.isfinite(groundwater_end).all()
            and np.isfinite(channel_end).all()
            and min(
                float(soil_end.min()),
                float(groundwater_end.min()),
                float(channel_end.min()),
            ) >= -1e-9
        )
        soil, groundwater, channel = soil_end, groundwater_end, channel_end
        return outflow

    cycle_rows = []
    for cycle in range(1, cfg["maximum_cycles"] + 1):
        for time_index in range(len(cycle_keys)):
            advance(cycle_p[time_index], cycle_pet[time_index])
        reach_total_storage = (
            (soil + groundwater) * area_factor[None, :] + channel
        )
        cycle_rows.append({
            "cycle": cycle,
            "network_total_storage_relative_spread": float(
                relative_range(reach_total_storage.sum(axis=1), axis=0)
            ),
            "maximum_reach_total_storage_relative_spread": float(
                relative_range(reach_total_storage, axis=0).max()
            ),
        })
        if actual_start_cycle is not None and cycle == actual_start_cycle:
            break

    actual_rows = []
    if actual_start_cycle is not None:
        for time_index, (year, month) in enumerate(actual_keys):
            outflow = advance(actual_p[time_index], actual_pet[time_index])
            reach_total_storage = (
                (soil + groundwater) * area_factor[None, :] + channel
            )
            actual_rows.append({
                "year": year,
                "month": month,
                "terminal_outflow_relative_spread": float(
                    relative_range(outflow[:, terminal_index].sum(axis=1), axis=0)
                ),
                "network_total_storage_relative_spread": float(
                    relative_range(reach_total_storage.sum(axis=1), axis=0)
                ),
                "maximum_reach_total_storage_relative_spread": float(
                    relative_range(reach_total_storage, axis=0).max()
                ),
            })
    diagnostics = {
        "variants": variant_names,
        "reaches": len(reaches),
        "terminal_reaches": len(terminal_ids),
        "maximum_system_relative_closure": maximum_system_relative_closure,
        "all_states_nonnegative_finite": all_states_nonnegative_finite,
    }
    return pd.DataFrame(cycle_rows), pd.DataFrame(actual_rows), diagnostics


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    priors = json.loads(PRIOR_CONFIG.read_text(encoding="utf-8"))
    if parent_gate["authorized_next_action"] != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize spin-up design")
    forcing = pd.read_parquet(FORCING_PATH)
    static = pd.read_parquet(STATIC_PATH)
    forcing = forcing[
        (forcing["year"] >= cfg["actual_diagnostic_period"][0])
        & (forcing["year"] <= cfg["actual_diagnostic_period"][1])
    ].copy()
    if len(forcing) != 35880 or static["reach_id"].nunique() != 230:
        raise RuntimeError("Frozen scope mismatch")

    threshold = cfg["relative_spread_threshold"]
    cycle_frames = []
    scenario_rows = []
    first_pass_diagnostics = {}
    for scenario_name, scenario in priors["parameter_scenarios"].items():
        cycles, _, diagnostics = run_ensemble(
            forcing,
            static,
            scenario,
            priors["initialization_variants"],
            cfg,
            actual_start_cycle=None,
        )
        cycles["parameter_scenario"] = scenario_name
        cycle_frames.append(cycles)
        first_pass_diagnostics[scenario_name] = diagnostics
        converged_cycles = cycles[
            (cycles["network_total_storage_relative_spread"] < threshold)
            & (cycles["maximum_reach_total_storage_relative_spread"] < threshold)
        ]
        first_cycle = (
            int(converged_cycles.iloc[0]["cycle"])
            if not converged_cycles.empty
            else None
        )
        scenario_rows.append({
            "parameter_scenario": scenario_name,
            "first_converged_cycle": first_cycle,
            "converged_within_maximum_cycles": first_cycle is not None,
        })
    cycles_all = pd.concat(cycle_frames, ignore_index=True)
    scenario_summary = pd.DataFrame(scenario_rows)
    required_common_cycles = (
        int(scenario_summary["first_converged_cycle"].max())
        if scenario_summary["first_converged_cycle"].notna().all()
        else None
    )
    actual_frames = []
    completed_rows = []
    for row in scenario_summary.itertuples(index=False):
        scenario_name = row.parameter_scenario
        if required_common_cycles is None:
            completed_rows.append({
                **row._asdict(),
                "max_actual_terminal_outflow_spread": float("inf"),
                "max_actual_network_storage_spread": float("inf"),
                "max_actual_reach_storage_spread": float("inf"),
                **first_pass_diagnostics[scenario_name],
            })
            continue
        _, actual, diagnostics = run_ensemble(
            forcing,
            static,
            priors["parameter_scenarios"][scenario_name],
            priors["initialization_variants"],
            cfg,
            actual_start_cycle=required_common_cycles,
        )
        actual["parameter_scenario"] = scenario_name
        actual_frames.append(actual)
        completed_rows.append({
            **row._asdict(),
            "max_actual_terminal_outflow_spread": float(
                actual["terminal_outflow_relative_spread"].max()
            ),
            "max_actual_network_storage_spread": float(
                actual["network_total_storage_relative_spread"].max()
            ),
            "max_actual_reach_storage_spread": float(
                actual["maximum_reach_total_storage_relative_spread"].max()
            ),
            "maximum_system_relative_closure": max(
                diagnostics["maximum_system_relative_closure"],
                first_pass_diagnostics[scenario_name][
                    "maximum_system_relative_closure"
                ],
            ),
            "all_states_nonnegative_finite": (
                diagnostics["all_states_nonnegative_finite"]
                and first_pass_diagnostics[scenario_name][
                    "all_states_nonnegative_finite"
                ]
            ),
            "variants": diagnostics["variants"],
            "reaches": diagnostics["reaches"],
            "terminal_reaches": diagnostics["terminal_reaches"],
        })
    scenario_summary = pd.DataFrame(completed_rows)
    actual_all = (
        pd.concat(actual_frames, ignore_index=True)
        if actual_frames
        else pd.DataFrame()
    )

    cycle_path = REPORT / "spinup_cycle_convergence.csv"
    actual_path = OUTPUTS / "post_spinup_actual_spread.parquet"
    summary_path = REPORT / "spinup_scenario_summary.csv"
    cycles_all.to_csv(cycle_path, index=False, encoding="utf-8-sig")
    actual_all.to_parquet(actual_path, index=False)
    scenario_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    checks = {
        "parent_authorization": True,
        "three_registered_parameter_scenarios": len(scenario_summary) == 3,
        "five_initialization_variants_per_scenario": all(
            len(item) == 5 for item in scenario_summary["variants"]
        ),
        "all_scenarios_converged_within_12_cycles": (
            scenario_summary["converged_within_maximum_cycles"].all()
        ),
        "post_spinup_terminal_outflow_spread_below_1pct": (
            scenario_summary["max_actual_terminal_outflow_spread"].max() < threshold
        ),
        "post_spinup_network_storage_spread_below_1pct": (
            scenario_summary["max_actual_network_storage_spread"].max() < threshold
        ),
        "post_spinup_reach_storage_spread_below_1pct": (
            scenario_summary["max_actual_reach_storage_spread"].max() < threshold
        ),
        "all_states_nonnegative_finite": (
            scenario_summary["all_states_nonnegative_finite"].all()
        ),
        "strict_reach_system_closure": (
            scenario_summary["maximum_system_relative_closure"].max()
            < cfg["relative_closure_threshold"]
        ),
        "forcing_scope_2006_2018_only": (
            int(forcing["year"].min()) == 2006
            and int(forcing["year"].max()) == 2018
        ),
        "station_observations_not_read": True,
        "management_fluxes_disabled": True,
        "period_2019_2022_not_read": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    next_action = (
        cfg["next_action_if_pass"] if passed else cfg["next_action_if_fail"]
    )
    records = json.loads(
        scenario_summary.to_json(orient="records", double_precision=15)
    )
    gate = {
        "run_id": cfg["run_id"],
        "phase": "spinup_protocol_gate",
        "created_utc": utc_now(),
        "checks": checks,
        "protocol": {
            "forcing_cycle": cfg["spinup_forcing_period"],
            "cycle_years": cfg["cycle_years"],
            "maximum_cycles": cfg["maximum_cycles"],
            "required_common_cycles": required_common_cycles,
            "required_spinup_years": (
                required_common_cycles * cfg["cycle_years"]
                if required_common_cycles is not None
                else None
            ),
            "threshold": threshold,
        },
        "scenario_summary": records,
        "decision": "SPINUP_PROTOCOL_PASSED" if passed else "SPINUP_PROTOCOL_FAILED",
        "authorized_next_action": next_action,
        "authorized_scope": (
            "Build only a low-dimensional attribute-to-parameter mapping; calibration remains forbidden."
            if passed
            else "Repair only the spin-up protocol; parameter calibration and prior narrowing remain forbidden."
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
    table = "\n".join(
        f"| {row.parameter_scenario} | "
        f"{'未收敛' if pd.isna(row.first_converged_cycle) else int(row.first_converged_cycle)} | "
        f"{row.max_actual_terminal_outflow_spread:.3e} | "
        f"{row.max_actual_network_storage_spread:.3e} | "
        f"{row.max_actual_reach_storage_spread:.3e} |"
        for row in scenario_summary.itertuples()
    )
    report = f"""# Q78-NAT 显式 spin-up 协议

## 结论

循环 spin-up 门禁{'通过' if passed else '未通过'}。使用 2006–2011 forcing 的最小共同循环数为 `{required_common_cycles}`，对应 `{required_common_cycles * 6 if required_common_cycles else '未确定'}` 个合成年。

| 参数情景 | 首次收敛循环 | 实际期最大末端出流极差 | 实际期最大全网储量极差 | 实际期最大单reach储量极差 |
|---|---:|---:|---:|---:|
{table}

## 使用规则

- spin-up 必须在每个时间折内仅使用该折训练期可获得 forcing；
- 循环 forcing 只建立初始周期状态，不产生独立观测信息；
- 不允许用站点流量选择循环数；
- 本轮未读取管理通量或 2019–2022。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "runtime": "conda sparrow",
        "inputs_read": [str(PARENT_GATE), str(PRIOR_CONFIG), str(FORCING_PATH), str(STATIC_PATH)],
        "station_observation_files_read": [],
        "management_flux_files_read": [],
        "period_2019_2022_read": False,
    }
    (REPORT / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sources = [
        CONFIG,
        RUN_DIR / "experiment_contract.md",
        RUN_DIR / "README.md",
        RUN_DIR / "scripts" / "run_spinup_protocol_gate.py",
        RUN_DIR / "scripts" / "validate_spinup_protocol_gate.py",
        PARENT_GATE,
        PRIOR_CONFIG,
        FORCING_PATH,
        STATIC_PATH,
    ]
    products = [
        cycle_path,
        actual_path,
        summary_path,
        REPORT / "gate.json",
        REPORT / "technical_report.md",
        REPORT / "run_manifest.json",
    ]
    manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "sources": [record(path, "spinup_source", "reported_or_derived") for path in sources],
        "products": [record(path, "spinup_product", "derived") for path in products],
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
        "required_common_cycles": required_common_cycles,
        "authorized_next_action": next_action,
        "scenario_summary": records,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
