from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_27"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
COMPONENT = TEST / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
INPUT = TEST / "20260813_54" / "inputs" / "parent_indata.parquet"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
LOCK = TEST / "20260823_15" / "final_reports" / "four_group_training_lock.json"
PARENT = TEST / "20260823_26" / "outputs" / "monthly_q72_reach_hydrology.parquet"
OLD_TN = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
CFS_PER_M3S = 35.3146667
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_component():
    spec = importlib.util.spec_from_file_location("stage27_q72_component", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.INPUT_PATH = INPUT
    module.TOPOLOGY_PATH = TOPOLOGY
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    module.DYNAMIC_BETA_W = 0.0
    module.configure_engineering_repair(True)
    module.configure_full_gaussian_prior_space(False)
    module.ZERO_PRESERVING_FULL_PRIOR_MODE = True
    module.set_et_feature_block_mode("full")
    module.NETWORK_INPUT_SCALE = 1.0
    module.DETERMINISTIC_SPINUP_MODE = True
    return module


def prepare_base(module) -> pd.DataFrame:
    forcing = module.load_forcing_panel().sort_values(["comid", "year", "month"]).reset_index(drop=True)
    return module.add_hydrologic_features(
        forcing,
        rho=0.70,
        wm=480.0,
        et_gamma=0.75,
        sas_rho=0.93,
        young_k=1.5,
        storage_scale=720.0,
        prod_capacity=240.0,
        runoff_gamma=2.5,
        quick_rho=0.25,
        base_rho=0.85,
        base_release=0.10,
    ).sort_values(["comid", "year", "month"]).reset_index(drop=True)


def spinup_state(module, frame: pd.DataFrame, positions: list[int], p: dict[str, float]):
    state = np.array([0.5 * p["prod_capacity"], 0.0, 0.0], dtype=float)
    last_delta = np.inf
    for cycle in range(1, int(module.SPINUP_MAX_CYCLES) + 1):
        previous = state.copy()
        soil, quick, delayed = state.tolist()
        for pos in positions:
            effective = float(frame.at[pos, "sas_effective_mm"])
            saturation = float(np.clip(soil / max(p["prod_capacity"], EPS), 0.0, 1.5))
            generated = min(effective, effective * saturation ** p["runoff_gamma"])
            soil = max(soil + max(effective - generated, 0.0), 0.0)
            demand = float(frame.at[pos, "aet_storage_demand_mm"])
            soil -= min(demand, soil)
            overflow = max(soil - p["prod_capacity"], 0.0)
            soil = min(soil, p["prod_capacity"])
            recharge = p["base_release"] * soil
            soil = max(soil - recharge, 0.0)
            _, _, quick = module.retention_step(quick, generated + overflow, p["quick_rho"])
            _, _, delayed = module.retention_step(delayed, recharge, p["base_rho"])
        state = np.array([soil, quick, delayed], dtype=float)
        last_delta = float(np.max(np.abs(state - previous)) / max(p["prod_capacity"], EPS))
        if last_delta < float(module.SPINUP_TOLERANCE):
            return state, cycle, last_delta
    raise RuntimeError("Stage27 production spin-up failed to converge")


def instrument(module, base: pd.DataFrame, p: dict[str, float]):
    n = len(base)
    names = [
        "source_store_start_mm", "quick_store_start_mm", "slow_store_start_mm",
        "positive_input_mm", "source_positive_input_to_store_mm", "quick_generated_mm",
        "aet_storage_withdrawn_mm", "actual_et_mm", "aet_unmet_mm",
        "source_store_pre_recharge_mm", "soil_overflow_to_quick_mm", "slow_path_recharge_mm",
        "quick_input_mm", "quick_available_mm", "slow_available_mm",
        "fast_path_release_mm", "slow_path_discharge_mm",
        "source_store_end_mm", "quick_store_end_mm", "slow_store_end_mm",
        "local_mass_balance_error_mm", "full_water_balance_error_mm",
    ]
    arrays = {name: np.zeros(n, dtype=float) for name in names}
    spin_rows: list[dict[str, float | int | bool]] = []
    for reach_id, idx in base.groupby("comid", sort=False).groups.items():
        positions = [int(i) for i in idx]
        spin_positions = [i for i in positions if int(base.at[i, "year"]) <= int(module.SPINUP_END_YEAR)]
        state, cycles, delta = spinup_state(module, base, spin_positions, p)
        soil, quick, delayed = state.tolist()
        spin_rows.append({
            "reach_id": int(reach_id), "cycles": int(cycles),
            "terminal_normalized_max_abs_delta": float(delta),
            "source_store_end_mm": soil, "quick_store_end_mm": quick,
            "slow_store_end_mm": delayed, "converged": True,
        })
        for pos in positions:
            start_total = soil + quick + delayed
            arrays["source_store_start_mm"][pos] = soil
            arrays["quick_store_start_mm"][pos] = quick
            arrays["slow_store_start_mm"][pos] = delayed
            effective = float(base.at[pos, "sas_effective_mm"])
            demand = float(base.at[pos, "aet_storage_demand_mm"])
            ppt = float(base.at[pos, "PPT"])
            aet = float(base.at[pos, "aet_mm"])
            saturation = float(np.clip(soil / max(p["prod_capacity"], EPS), 0.0, 1.5))
            generated = min(effective, effective * saturation ** p["runoff_gamma"])
            infiltrated = max(effective - generated, 0.0)
            soil = max(soil + infiltrated, 0.0)
            withdrawn = min(demand, soil)
            soil = max(soil - withdrawn, 0.0)
            unmet = demand - withdrawn
            overflow = max(soil - p["prod_capacity"], 0.0)
            soil = min(soil, p["prod_capacity"])
            pre_recharge = soil
            recharge = p["base_release"] * soil
            soil = max(soil - recharge, 0.0)
            quick_input = generated + overflow
            quick_available, fast_release, quick = module.retention_step(quick, quick_input, p["quick_rho"])
            slow_available, slow_discharge, delayed = module.retention_step(delayed, recharge, p["base_rho"])
            end_total = soil + quick + delayed
            actual_et = min(ppt, aet) + withdrawn
            local_error = start_total + effective - withdrawn - end_total - fast_release - slow_discharge
            full_error = start_total + ppt - actual_et - end_total - fast_release - slow_discharge
            values = {
                "positive_input_mm": effective,
                "source_positive_input_to_store_mm": infiltrated,
                "quick_generated_mm": generated,
                "aet_storage_withdrawn_mm": withdrawn,
                "actual_et_mm": actual_et,
                "aet_unmet_mm": unmet,
                "source_store_pre_recharge_mm": pre_recharge,
                "soil_overflow_to_quick_mm": overflow,
                "slow_path_recharge_mm": recharge,
                "quick_input_mm": quick_input,
                "quick_available_mm": quick_available,
                "slow_available_mm": slow_available,
                "fast_path_release_mm": fast_release,
                "slow_path_discharge_mm": slow_discharge,
                "source_store_end_mm": soil,
                "quick_store_end_mm": quick,
                "slow_store_end_mm": delayed,
                "local_mass_balance_error_mm": local_error,
                "full_water_balance_error_mm": full_error,
            }
            for name, value in values.items():
                arrays[name][pos] = value
    return arrays, pd.DataFrame(spin_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    p = {name: float(value) for name, value in lock["physical_parameters"].items()}
    module = load_component()
    base = prepare_base(module)
    reaches, n_time, upstream = module._panel_layout(base)
    arrays, spin = instrument(module, base, p)
    days = np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(base.year, base.month)], float)
    seconds = days * 86400.0
    area = base.IncAreaKm2.to_numpy(float)
    local_fast_volume = arrays["fast_path_release_mm"] * area * 1000.0
    local_slow_volume = arrays["slow_path_discharge_mm"] * area * 1000.0
    routed_fast_volume = (upstream @ local_fast_volume.reshape(len(reaches), n_time)).reshape(-1)
    routed_slow_volume = (upstream @ local_slow_volume.reshape(len(reaches), n_time)).reshape(-1)
    local_total_volume = local_fast_volume + local_slow_volume
    routed_total_volume = routed_fast_volume + routed_slow_volume
    out = pd.DataFrame({
        "reach_id": base.comid.astype(int), "year": base.year.astype(int), "month": base.month.astype(int),
        "month_seconds": seconds, "catchment_area_km2": area,
        "precipitation_mm": base.PPT.to_numpy(float), "prescribed_aet_mm": base.aet_mm.to_numpy(float),
    })
    for name, values in arrays.items():
        out[name] = values
    out["source_water_capacity_mm"] = p["prod_capacity"]
    out["soil_contact_water_mm"] = out.soil_overflow_to_quick_mm + out.slow_path_recharge_mm
    out["quick_bypass_fraction"] = np.divide(
        out.quick_generated_mm, out.positive_input_mm,
        out=np.zeros(len(out), dtype=float), where=out.positive_input_mm.to_numpy(float) > EPS,
    )
    out["q_local_total_mm"] = out.fast_path_release_mm + out.slow_path_discharge_mm
    out["local_fast_volume_m3"] = local_fast_volume
    out["local_delayed_volume_m3"] = local_slow_volume
    out["local_total_volume_m3"] = local_total_volume
    out["routed_fast_volume_m3"] = routed_fast_volume
    out["routed_delayed_volume_m3"] = routed_slow_volume
    out["routed_total_volume_m3"] = routed_total_volume
    for prefix in ["local", "routed"]:
        out[f"{prefix}_fast_m3_s"] = out[f"{prefix}_fast_volume_m3"] / seconds
        out[f"{prefix}_delayed_m3_s"] = out[f"{prefix}_delayed_volume_m3"] / seconds
        out[f"{prefix}_total_m3_s"] = out[f"{prefix}_total_volume_m3"] / seconds
    out["routed_fast_fraction"] = np.divide(
        routed_fast_volume, routed_total_volume,
        out=np.zeros(len(out), dtype=float), where=routed_total_volume > EPS,
    )
    out["gw_recharge_mm"] = out.slow_path_recharge_mm
    out["gw_discharge_mm"] = out.slow_path_discharge_mm
    out["gw_response_state_end_mm"] = out.slow_store_end_mm
    out["quick_release_mm"] = out.fast_path_release_mm
    out["parameter_version"] = "20260823_15_LOCKED_PHYSICAL_PARAMETERS"
    out["conditioning_mode"] = "OPEN_LOOP_LOCKED_PARENT_REPRODUCTION"
    out["hydrology_source"] = "20260823_26_Q72_PROCESS_INSTRUMENTED"
    out.to_parquet(OUT / "q72_full_state_tn_bridge_2006_2022.parquet", index=False)
    spin.to_parquet(OUT / "spinup_audit.parquet", index=False)

    parent = pd.read_parquet(PARENT)
    check = out.merge(parent, on=["reach_id", "year", "month"], validate="one_to_one")
    differences = {
        "local_fast_cfs": check.local_fast_m3_s.to_numpy(float) * CFS_PER_M3S - check.q72_local_quick_cfs.to_numpy(float),
        "local_delayed_cfs": check.local_delayed_m3_s.to_numpy(float) * CFS_PER_M3S - check.q72_local_slow_cfs.to_numpy(float),
        "routed_fast_cfs": check.routed_fast_m3_s.to_numpy(float) * CFS_PER_M3S - check.q72_routed_quick_cfs.to_numpy(float),
        "routed_delayed_cfs": check.routed_delayed_m3_s.to_numpy(float) * CFS_PER_M3S - check.q72_routed_slow_cfs.to_numpy(float),
        "routed_total_cfs": check.routed_total_m3_s.to_numpy(float) * CFS_PER_M3S - check.q72_routed_total_cfs.to_numpy(float),
        "source_store_mm": check.source_store_end_mm.to_numpy(float) - check.production_storage_mm.to_numpy(float),
        "saturation": check.source_store_end_mm.to_numpy(float) / p["prod_capacity"] - check.production_saturation.to_numpy(float),
    }
    reproduction = {f"{name}_max_abs": float(np.max(np.abs(value))) for name, value in differences.items()}

    old = pd.read_parquet(OLD_TN)
    old = old.loc[old.year.between(2006, 2022), [
        "reach_id", "year", "month", "q_local_total_mm", "quick_release_mm", "gw_discharge_mm", "catchment_area_km2"
    ]].copy()
    comparison = out.merge(old, on=["reach_id", "year", "month"], suffixes=("_new", "_old"), validate="one_to_one")
    comparison["old_local_total_m3_s"] = (
        comparison.q_local_total_mm_old * comparison.catchment_area_km2_old * 1000.0 / comparison.month_seconds
    )
    comparison["new_old_local_total_ratio"] = np.divide(
        comparison.local_total_m3_s, comparison.old_local_total_m3_s,
        out=np.full(len(comparison), np.nan), where=comparison.old_local_total_m3_s.to_numpy(float) > EPS,
    )
    comparison["new_fast_fraction"] = np.divide(
        comparison.fast_path_release_mm, comparison.q_local_total_mm_new,
        out=np.zeros(len(comparison)), where=comparison.q_local_total_mm_new.to_numpy(float) > EPS,
    )
    comparison["old_fast_fraction"] = np.divide(
        comparison.quick_release_mm_old, comparison.q_local_total_mm_old,
        out=np.zeros(len(comparison)), where=comparison.q_local_total_mm_old.to_numpy(float) > EPS,
    )
    comparison[[
        "reach_id", "year", "month", "local_total_m3_s", "old_local_total_m3_s",
        "new_old_local_total_ratio", "new_fast_fraction", "old_fast_fraction"
    ]].to_parquet(OUT / "old_tn_hydrology_mismatch.parquet", index=False)

    local_fast_routing_error = routed_fast_volume - (upstream @ local_fast_volume.reshape(len(reaches), n_time)).reshape(-1)
    local_slow_routing_error = routed_slow_volume - (upstream @ local_slow_volume.reshape(len(reaches), n_time)).reshape(-1)
    min_state_flux = float(out[[
        "source_store_start_mm", "quick_store_start_mm", "slow_store_start_mm",
        "quick_generated_mm", "slow_path_recharge_mm", "fast_path_release_mm",
        "slow_path_discharge_mm", "source_store_end_mm", "quick_store_end_mm", "slow_store_end_mm",
    ]].min().min())
    gates = contract["hard_gates"]
    audit = {
        "stage": "20260823_27",
        "reach_count": int(out.reach_id.nunique()),
        "row_count": int(len(out)),
        "month_count": int(out[["year", "month"]].drop_duplicates().shape[0]),
        "spinup_all_converged": bool(spin.converged.all()),
        "spinup_max_cycles": int(spin.cycles.max()),
        "spinup_max_terminal_delta": float(spin.terminal_normalized_max_abs_delta.max()),
        "local_mass_balance_max_abs_mm": float(out.local_mass_balance_error_mm.abs().max()),
        "full_water_balance_max_abs_mm": float(out.full_water_balance_error_mm.abs().max()),
        "minimum_state_or_flux_mm": min_state_flux,
        "routing_fast_max_abs_m3": float(np.max(np.abs(local_fast_routing_error))),
        "routing_delayed_max_abs_m3": float(np.max(np.abs(local_slow_routing_error))),
        "routing_fast_delayed_total_max_abs_m3": float(np.max(np.abs(routed_total_volume-routed_fast_volume-routed_slow_volume))),
        "parent_reproduction": reproduction,
        "old_tn_new_over_old_local_total_median": float(comparison.new_old_local_total_ratio.median()),
        "old_tn_fast_fraction_median": float(comparison.old_fast_fraction.median()),
        "new_fast_fraction_median": float(comparison.new_fast_fraction.median()),
    }
    audit["hard_gate_pass"] = bool(
        audit["reach_count"] == int(gates["reach_count"])
        and audit["row_count"] == int(gates["row_count"])
        and audit["spinup_all_converged"]
        and audit["local_mass_balance_max_abs_mm"] <= float(gates["local_mass_balance_max_abs_mm"])
        and audit["full_water_balance_max_abs_mm"] <= float(gates["local_mass_balance_max_abs_mm"])
        and audit["minimum_state_or_flux_mm"] >= float(gates["minimum_state_or_flux_mm"])
        and max(value for key, value in reproduction.items() if key.endswith("cfs_max_abs")) <= float(gates["parent_flow_max_abs_cfs"])
    )
    if not audit["hard_gate_pass"]:
        raise RuntimeError(f"Stage27 hard gate failed: {audit}")
    (REPORT / "stage27_bridge_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    interface_schema = {
        "canonical_path_names": {"fast": "fast-response pathway", "delayed": "delayed-storage pathway"},
        "compatibility_aliases": {"gw_recharge_mm": "slow_path_recharge_mm", "gw_discharge_mm": "slow_path_discharge_mm"},
        "claim_boundary": "The bridge reproduces a conceptual fast/delayed partition; it does not independently observe groundwater or water age.",
        "columns": {name: str(dtype) for name, dtype in out.dtypes.items()},
    }
    (REPORT / "tn_hydrology_interface_schema.json").write_text(json.dumps(interface_schema, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# 20260823_27 Q72—TN完整状态桥接\n\n## 结论\n\n`PASS`。最终 `_26` Q72 已被无参数改动地重建为完整TN水文接口。\n\n- Reach：{audit['reach_count']}；行：{audit['row_count']}；月份：{audit['month_count']}。\n- 父流量最大复现误差：{max(value for key, value in reproduction.items() if key.endswith('cfs_max_abs')):.3e} cfs。\n- 完整水账最大误差：{audit['full_water_balance_max_abs_mm']:.3e} mm。\n- spin-up最多循环：{audit['spinup_max_cycles']}，最大终态差：{audit['spinup_max_terminal_delta']:.3e}。\n\n## 关键纠正\n\n旧TN接口并非最终Q72：新/旧局地产流中位比为{audit['old_tn_new_over_old_local_total_median']:.3f}；旧/新快流比例中位数分别为{audit['old_tn_fast_fraction_median']:.3f}和{audit['new_fast_fraction_median']:.3f}。因此后续TN必须使用本轮统一接口，不能继续沿用`20260814_6`的水文状态。\n\n## 科学边界\n\n这里证明的是状态递推、完整水账和父流量复现。快流/延迟流是否受观测支持，要由 `_28–34` 的逐日分割、时间和空间检验决定。\n"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    integrity = {
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "parameter_lock_sha256": sha256(LOCK),
        "parent_product_sha256": sha256(PARENT),
        "bridge_product_sha256": sha256(OUT / "q72_full_state_tn_bridge_2006_2022.parquet"),
        "audit_sha256": sha256(REPORT / "stage27_bridge_audit.json"),
    }
    (REPORT / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
