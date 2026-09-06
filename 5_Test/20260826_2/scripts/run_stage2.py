from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_2"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
STAGE3_SCRIPTS = ROOT / "5_Test" / "20260825_3" / "scripts"
STAGE5_SCRIPTS = ROOT / "5_Test" / "20260825_5" / "scripts"
sys.path[:0] = [str(STAGE3_SCRIPTS), str(STAGE5_SCRIPTS)]

from hydrology_core import load_topology, route_instantaneous  # noqa: E402
from regional_hbv_core import PARAMETER_NAMES, _run_ordered_hbv, periodic_spinup_regional  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
Q72_BRIDGE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
PARENT_LOCK = ROOT / "5_Test" / "20260825_7" / "reports" / "full_development_parameter_lock.json"
PARENT_BRIDGE = ROOT / "5_Test" / "20260825_7" / "outputs" / "tn_hydrology_bridge_daily_2006_2022.parquet"
PARENT_STATIC = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet"
PARENT_STATIC_SNAPSHOT = RUN / "inputs" / "parent_static_snapshot.csv"
REPAIRED_STATIC_CSV = OUT / "reach_static_attributes_dem_recomputed.csv"
REPAIRED_STATIC = OUT / "reach_static_attributes_dem_recomputed.parquet"
REPAIR_DECISION = REPORT / "dem_static_attribute_repair_decision.json"

SPINUP_TOLERANCE = 1.0e-8
SPINUP_MAX_CYCLES = 500
REPRODUCTION_TOLERANCE = 1.0e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if not REPAIRED_STATIC_CSV.is_file() or not REPAIR_DECISION.is_file():
        raise RuntimeError("Run repair_static_attributes.py before run_stage2.py")
    repaired_frame = pd.read_csv(REPAIRED_STATIC_CSV)
    repaired_frame.to_parquet(REPAIRED_STATIC, index=False)
    for stem in ["dem_static_attribute_repair_audit", "dem_static_attribute_repair_targets_resolved"]:
        csv_path = OUT / f"{stem}.csv"
        pd.read_csv(csv_path).to_parquet(OUT / f"{stem}.parquet", index=False)
    repair = json.loads(REPAIR_DECISION.read_text(encoding="utf-8"))
    if repair["mapping_attribute_status"] != "PASS_DEM_RECOMPUTED_WITH_EXPLICIT_CENSORING":
        raise RuntimeError("DEM repair did not pass")

    evaluation_contract = {
        "registered_before_candidate_fitting": True,
        "spinup": {"forcing": "2006-2009 repeated", "terminal_max_abs_delta_mm": 1e-8, "max_cycles": 500},
        "development": "2010-2018 daily discharge; candidate fitting and internal temporal folds only",
        "retrospective": "2019-2022 locked post-development check; never used to fit or select parameters",
        "space": "entire observed terminal tree removed before parameter fitting and spatial mapping evaluation",
        "total_flow_metrics": ["daily and monthly pooled NSE", "station median and mean NSE", "log NSE", "KGE", "PBIAS", "log RMSE", "high-flow and low-flow diagnostics"],
        "process_soft_targets": ["PML V2.2a monthly AET", "Q-derived FDC, recession, autocorrelation, low-flow and BFI signatures"],
        "process_target_boundary": "Q-derived signatures are not independent component observations and cannot alone identify paths",
        "numerical_gates": {"land_mass_max_abs_error_mm": 1e-10, "routing_mass_relative_error": 1e-12, "restart_max_abs_delta": 1e-10},
        "component_names": ["fast_response", "intermediate_response", "slow_response"],
        "component_claim": "model response components only; not observed surface water, groundwater, new water, old water or water age",
        "TN_used": False,
    }
    write_json(REPORT / "common_evaluation_contract.json", evaluation_contract)

    lock = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    physical = np.asarray([lock["physical_parameters"][name] for name in PARAMETER_NAMES], dtype=np.float64)
    physical_map = np.tile(physical[None, :], (230, 1))
    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    full_dates = pd.date_range("2006-01-01", "2022-12-31", freq="D")
    spin_dates = pd.date_range("2006-01-01", "2009-12-31", freq="D")
    precipitation = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=full_dates, columns=reach_ids).to_numpy(float)
    pet = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=full_dates, columns=reach_ids).to_numpy(float)
    spin_mask = full_dates.isin(spin_dates)
    initial, spinup = periodic_spinup_regional(
        precipitation[spin_mask], pet[spin_mask], physical_map, SPINUP_TOLERANCE, SPINUP_MAX_CYCLES
    )
    _, components_mm, mass_error = _run_ordered_hbv(precipitation, pet, physical_map, initial, True)
    if components_mm is None:
        raise RuntimeError("Parent replay did not return components")
    area = (
        pd.read_parquet(Q72_BRIDGE, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id")
        .set_index("reach_id")
        .reindex(reach_ids)
        .catchment_area_km2.to_numpy(float)
    )
    local = components_mm * area[None, :, None] * 1000.0 / 86400.0
    routed = route_instantaneous(local * 86400.0, reach_ids, order, downstream) / 86400.0
    parent = pd.read_parquet(
        PARENT_BRIDGE,
        columns=["date", "reach_id", "local_q0_m3_s", "local_q1_m3_s", "local_q2_m3_s", "routed_q0_m3_s", "routed_q1_m3_s", "routed_q2_m3_s", "routed_total_m3_s"],
    ).sort_values(["date", "reach_id"]).reset_index(drop=True)
    expected_rows = len(full_dates) * len(reach_ids)
    if len(parent) != expected_rows:
        raise RuntimeError("Frozen bridge row count changed")
    replay_columns = {
        "local_q0_m3_s": local[:, :, 0].reshape(-1),
        "local_q1_m3_s": local[:, :, 1].reshape(-1),
        "local_q2_m3_s": local[:, :, 2].reshape(-1),
        "routed_q0_m3_s": routed[:, :, 0].reshape(-1),
        "routed_q1_m3_s": routed[:, :, 1].reshape(-1),
        "routed_q2_m3_s": routed[:, :, 2].reshape(-1),
        "routed_total_m3_s": routed.sum(axis=2).reshape(-1),
    }
    differences = {
        name: float(np.max(np.abs(values - parent[name].to_numpy(float))))
        for name, values in replay_columns.items()
    }
    parent_reproduced = bool(max(differences.values()) <= REPRODUCTION_TOLERANCE)
    terminal_indices = [int(reach) - 1 for reach in reach_ids if int(reach) not in downstream]
    routed_terminal_volume = float((routed[:, terminal_indices, :] * 86400.0).sum())
    local_volume = float((local * 86400.0).sum())
    route_error = abs(routed_terminal_volume - local_volume) / max(local_volume, 1.0)
    reproduction = {
        "parent": "20260825_7 GLOBAL_HBV_R0",
        "frozen_parameter_hash": sha256(PARENT_LOCK),
        "frozen_bridge_hash": sha256(PARENT_BRIDGE),
        "spinup": spinup,
        "full_period_land_mass_max_abs_error_mm": float(mass_error),
        "routing_terminal_mass_relative_error": route_error,
        "max_abs_delta_by_field_m3_s": differences,
        "tolerance_m3_s": REPRODUCTION_TOLERANCE,
        "exact_reproduction_passed": parent_reproduced,
    }
    write_json(REPORT / "parent_exact_reproduction.json", reproduction)

    technical_pass = bool(
        parent_reproduced
        and mass_error <= evaluation_contract["numerical_gates"]["land_mass_max_abs_error_mm"]
        and route_error <= evaluation_contract["numerical_gates"]["routing_mass_relative_error"]
    )
    decision = {
        "stage": "20260826_2",
        "status": "PASS_STATIC_REPAIR_CONTRACT_AND_PARENT_REPRODUCTION" if technical_pass else "STAGE2_TECHNICAL_GATE_FAILED",
        "static_attribute_status": repair["mapping_attribute_status"],
        "parent_exact_reproduction_passed": parent_reproduced,
        "parent_max_abs_delta_m3_s": max(differences.values()),
        "candidate_structures_fitted": 0,
        "authorized_successor": "20260826_3" if technical_pass else None,
    }
    write_json(REPORT / "stage2_decision.json", decision)
    report = f"""# 20260826_2 静态属性修复与父模型复现

## 结论

状态：`{decision['status']}`。

230条Reach均从权威PRB DEM重新采样纵剖面。原先含`-32768`的高程和统一坡度下限不再直接进入空间参数映射；低于DEM垂向分辨率的坡度被显式标为censored，并采用各Reach长度对应的分辨率下限，而不是伪装成精确坡度。

冻结的`GLOBAL_HBV_R0`使用已锁参数独立重放，所有本地/路由Q0–Q2及总流量最大绝对差为`{max(differences.values()):.3e} m3/s`，陆面质量误差为`{mass_error:.3e} mm`，河网终端质量相对误差为`{route_error:.3e}`。

本阶段没有拟合任何新结构。下一阶段才允许实现候选方程与合成/数值预检。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    integrity = {}
    for path in [RUN / "experiment_contract.json", REPAIRED_STATIC, REPORT / "common_evaluation_contract.json", REPORT / "parent_exact_reproduction.json", REPORT / "stage2_decision.json", REPORT / "technical_report.md"]:
        integrity[str(path)] = sha256(path)
    write_json(REPORT / "integrity.json", integrity)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
