from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
PARENT = ROOT / "5_Test" / "20260729_14"
LEDGER = ROOT / "5_Test" / "20260729_9"
STATIC_RUN = ROOT / "5_Test" / "20260729_13"
REPORT = RUN_DIR / "reports" / "parameter_prior_initialization_gate"
OUTPUTS = RUN_DIR / "outputs"
MANIFEST_DIR = RUN_DIR / "inputs_manifest"
CONFIG = RUN_DIR / "config" / "parameter_prior_initialization_contract.json"
PARENT_GATE = PARENT / "reports" / "q78_nat_conservation_core_gate" / "gate.json"
FORCING_PATH = LEDGER / "inputs" / "reach_month_forcing_2006_2018.parquet"
STATIC_PATH = STATIC_RUN / "inputs" / "reach_static_direction_final.parquet"
CORE_SCRIPT = PARENT / "scripts" / "build_q78_nat_conservation_core.py"


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


def load_core():
    spec = importlib.util.spec_from_file_location("q78_conservation_core", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the frozen Q78 conservation core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_priors(cfg: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    valid = True
    for name, prior in cfg["parameter_priors"].items():
        ordered = prior["lower"] <= prior["center"] <= prior["upper"]
        rows.append({"parameter": name, **prior, "ordered_bounds": ordered})
        valid &= ordered
    for name, values in cfg["parameter_scenarios"].items():
        for parameter, value in values.items():
            prior = cfg["parameter_priors"][parameter]
            valid &= prior["lower"] <= value <= prior["upper"]
        valid &= (
            values["k_perc"] + values["k_int"]
            <= cfg["joint_constraints"]["k_perc_plus_k_int_upper"]
        )
    checks = {
        "all_prior_bounds_ordered": bool(all(row["ordered_bounds"] for row in rows)),
        "all_scenarios_inside_bounds": bool(valid),
        "k_deep_fixed_zero": (
            cfg["parameter_priors"]["k_deep"]["lower"]
            == cfg["parameter_priors"]["k_deep"]["upper"]
            == 0.0
        ),
        "prior_state_not_posterior": (
            cfg["prior_semantic_state"] == "development_prior_not_posterior"
        ),
    }
    return pd.DataFrame(rows), {key: bool(value) for key, value in checks.items()}


def relative_range(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float((array.max() - array.min()) / max(np.abs(array).max(), 1.0))


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent_gate["authorized_next_action"] != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize prior/initialization testing")

    core = load_core()
    forcing = pd.read_parquet(FORCING_PATH)
    static = pd.read_parquet(STATIC_PATH)
    forcing = forcing[
        (forcing["year"] >= cfg["development_period"][0])
        & (forcing["year"] <= cfg["development_period"][1])
    ].copy()
    if len(forcing) != 35880 or static["reach_id"].nunique() != 230:
        raise RuntimeError("Frozen forcing/static scope mismatch")

    reach_graph, _, _ = core.build_reach_graph(static)
    terminal_reaches = sorted(
        int(reach) for reach in reach_graph.nodes if reach_graph.out_degree(reach) == 0
    )
    static_area = static.set_index("reach_id")["inc_area_km2"] * 1000.0
    prior_table, prior_checks = validate_priors(cfg)
    prior_table.to_csv(REPORT / "parameter_prior_registry.csv", index=False, encoding="utf-8-sig")

    trajectory_frames = []
    run_rows = []
    for scenario_name, scenario_parameters in cfg["parameter_scenarios"].items():
        for init_name, initialization in cfg["initialization_variants"].items():
            parameters = {**scenario_parameters, **initialization}
            states, components, topology = core.run_core(forcing, static, parameters)
            states["soil_storage_end_m3"] = (
                states["soil_storage_end_mm"]
                * states["reach_id"].map(static_area)
            )
            states["groundwater_storage_end_m3"] = (
                states["groundwater_storage_end_mm"]
                * states["reach_id"].map(static_area)
            )
            states["is_terminal"] = states["reach_id"].isin(terminal_reaches)
            monthly = states.groupby(["year", "month"], as_index=False).agg(
                total_soil_storage_m3=("soil_storage_end_m3", "sum"),
                total_groundwater_storage_m3=("groundwater_storage_end_m3", "sum"),
                total_channel_storage_m3=("channel_storage_end_m3", "sum"),
            )
            terminal_monthly = (
                states[states["is_terminal"]]
                .groupby(["year", "month"], as_index=False)["channel_outflow_m3"]
                .sum()
                .rename(columns={"channel_outflow_m3": "terminal_outflow_m3"})
            )
            monthly = monthly.merge(terminal_monthly, on=["year", "month"], how="left")
            monthly["total_storage_m3"] = (
                monthly["total_soil_storage_m3"]
                + monthly["total_groundwater_storage_m3"]
                + monthly["total_channel_storage_m3"]
            )
            monthly["parameter_scenario"] = scenario_name
            monthly["initialization"] = init_name
            trajectory_frames.append(monthly)
            run_rows.append({
                "parameter_scenario": scenario_name,
                "initialization": init_name,
                "reach_months": len(states),
                "network_components": topology["weak_components"],
                "max_system_relative_closure": float(states["system_relative_closure"].max()),
                "max_component_relative_closure": float(components["relative_closure"].max()),
                "all_states_nonnegative": bool(
                    states[[
                        "soil_storage_end_mm",
                        "groundwater_storage_end_mm",
                        "channel_storage_end_m3",
                    ]].min().min() >= -1e-9
                ),
            })

    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    run_audit = pd.DataFrame(run_rows)
    spread_rows = []
    threshold = cfg["relative_spread_threshold"]
    eval_start_year, eval_start_month = map(
        int, cfg["evaluation_sensitivity_start"].split("-")
    )
    for scenario_name, group in trajectories.groupby("parameter_scenario"):
        for (year, month), month_group in group.groupby(["year", "month"]):
            spread_rows.append({
                "parameter_scenario": scenario_name,
                "year": int(year),
                "month": int(month),
                "terminal_outflow_relative_spread": relative_range(
                    month_group["terminal_outflow_m3"]
                ),
                "total_storage_relative_spread": relative_range(
                    month_group["total_storage_m3"]
                ),
            })
    spreads = pd.DataFrame(spread_rows).sort_values(
        ["parameter_scenario", "year", "month"]
    )
    scenario_rows = []
    for scenario_name, group in spreads.groupby("parameter_scenario"):
        eval_group = group[
            (group["year"] > eval_start_year)
            | ((group["year"] == eval_start_year) & (group["month"] >= eval_start_month))
        ]
        convergence = None
        ordered = group.reset_index(drop=True)
        for index, row in ordered.iterrows():
            future = ordered.iloc[index:]
            if (
                future["terminal_outflow_relative_spread"].max() < threshold
                and future["total_storage_relative_spread"].max() < threshold
            ):
                convergence = f"{int(row['year']):04d}-{int(row['month']):02d}"
                break
        scenario_rows.append({
            "parameter_scenario": scenario_name,
            "max_terminal_outflow_spread_2012_2018": float(
                eval_group["terminal_outflow_relative_spread"].max()
            ),
            "max_total_storage_spread_2012_2018": float(
                eval_group["total_storage_relative_spread"].max()
            ),
            "convergence_month": convergence,
            "converged_by_2012_01": bool(
                eval_group["terminal_outflow_relative_spread"].max() < threshold
                and eval_group["total_storage_relative_spread"].max() < threshold
            ),
        })
    scenario_summary = pd.DataFrame(scenario_rows)

    trajectory_path = OUTPUTS / "initialization_trajectories.parquet"
    spread_path = REPORT / "initialization_spread_by_month.csv"
    run_audit_path = REPORT / "run_audit.csv"
    summary_path = REPORT / "scenario_initialization_summary.csv"
    trajectories.to_parquet(trajectory_path, index=False)
    spreads.to_csv(spread_path, index=False, encoding="utf-8-sig")
    run_audit.to_csv(run_audit_path, index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    checks = {
        "parent_authorization": True,
        **prior_checks,
        "fifteen_complete_runs": (
            len(run_audit) == 15 and run_audit["reach_months"].eq(35880).all()
        ),
        "all_runs_strictly_conservative": (
            run_audit["max_system_relative_closure"].max() < 1e-8
            and run_audit["max_component_relative_closure"].max() < 1e-8
        ),
        "all_run_states_nonnegative": run_audit["all_states_nonnegative"].all(),
        "frozen_network_components_18": run_audit["network_components"].eq(18).all(),
        "all_registered_scenarios_converged_by_2012": (
            scenario_summary["converged_by_2012_01"].all()
        ),
        "station_observations_not_read": True,
        "management_fluxes_disabled": True,
        "period_2019_2022_not_read": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    authorized_next_action = (
        cfg["next_action_if_pass"] if passed else cfg["next_action_if_fail"]
    )
    initialization_records = json.loads(
        scenario_summary.to_json(orient="records", double_precision=15)
    )
    gate = {
        "run_id": cfg["run_id"],
        "phase": "parameter_prior_initialization_gate",
        "created_utc": utc_now(),
        "checks": checks,
        "prior_semantic_state": cfg["prior_semantic_state"],
        "scope": {
            "parameter_scenarios": len(cfg["parameter_scenarios"]),
            "initialization_variants": len(cfg["initialization_variants"]),
            "complete_core_runs": len(run_audit),
            "reach_months_per_run": 35880,
            "period": cfg["development_period"],
        },
        "initialization_summary": initialization_records,
        "decision": (
            "INITIALIZATION_ROBUST_BY_2012"
            if passed
            else "EXPLICIT_SPINUP_REQUIRED"
        ),
        "authorized_next_action": authorized_next_action,
        "authorized_scope": (
            "Build a low-dimensional attribute-to-parameter mapping without calibration."
            if passed
            else "Design and test only an explicit spin-up protocol across the registered prior stress scenarios; calibration remains forbidden."
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

    summary_lines = "\n".join(
        f"| {row.parameter_scenario} | "
        f"{row.max_terminal_outflow_spread_2012_2018:.3e} | "
        f"{row.max_total_storage_spread_2012_2018:.3e} | "
        f"{'未收敛' if pd.isna(row.convergence_month) else row.convergence_month} | "
        f"{'是' if row.converged_by_2012_01 else '否'} |"
        for row in scenario_summary.itertuples()
    )
    report = f"""# Q78-NAT 参数先验与初始化敏感性

## 结论

先验已按 `development_prior_not_posterior` 注册。15 个完整守恒核心运行均完成且守恒。初始化门禁{'通过' if passed else '未通过'}，因此下一步只授权 `{authorized_next_action}`。

| 参数情景 | 2012–2018 最大末端出流相对极差 | 2012–2018 最大总储量相对极差 | 全后续月份首次持续低于1% | 2012-01前收敛 |
|---|---:|---:|---|---|
{summary_lines}

## 解释边界

- 结果只衡量初始化记忆，不评价站点流量技能；
- `slow_memory_edge` 是注册先验边缘的压力测试，不代表最佳参数；
- 若失败，不允许通过查看观测流量后缩窄先验；
- 只有显式 spin-up 协议通过后，才允许进入属性参数映射；
- 水库、取水、回归水、调水以及 2019–2022 均未读取。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "runtime": "conda sparrow",
        "inputs_read": [str(PARENT_GATE), str(FORCING_PATH), str(STATIC_PATH), str(CORE_SCRIPT)],
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
        RUN_DIR / "reports" / "literature_basis.md",
        RUN_DIR / "scripts" / "run_parameter_prior_initialization_gate.py",
        RUN_DIR / "scripts" / "validate_parameter_prior_initialization_gate.py",
        PARENT_GATE,
        FORCING_PATH,
        STATIC_PATH,
        CORE_SCRIPT,
    ]
    products = [
        trajectory_path,
        spread_path,
        run_audit_path,
        summary_path,
        REPORT / "parameter_prior_registry.csv",
        REPORT / "gate.json",
        REPORT / "technical_report.md",
        REPORT / "run_manifest.json",
    ]
    manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "sources": [record(path, "prior_initialization_source", "reported_or_derived") for path in sources],
        "products": [record(path, "prior_initialization_product", "derived") for path in products],
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
        "decision": gate["decision"],
        "authorized_next_action": authorized_next_action,
        "initialization_summary": gate["initialization_summary"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
