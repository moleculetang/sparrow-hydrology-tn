from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "scripts" / "components" / "q72_prior_semantics_component.py"
INPUT = ROOT / "inputs" / "state_forcing_2006_2022_no_locked_observations.parquet"
TOPOLOGY = ROOT / "inputs" / "topology" / "topology_edges.csv"
EPS = 1.0e-12


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    prod_capacity: float
    runoff_gamma: float
    quick_rho: float
    base_release: float
    base_rho: float
    highflow_scale: float


BRANCHES = {
    "main": BranchSpec("main", 240.0, 2.5, 0.25, 0.10, 0.85, 1.00),
    "flash": BranchSpec("flash", 132.0, 1.7, 0.10, 0.06, 0.76, 1.20),
    "slow": BranchSpec("slow", 432.0, 3.1, 0.48, 0.07, 0.94, 0.85),
    "buffer": BranchSpec("buffer", 348.0, 3.3, 0.66, 0.12, 0.96, 0.65),
    "wet": BranchSpec("wet", 288.0, 2.2, 0.34, 0.08, 0.90, 1.10),
}


def load_component(name: str = "interface"):
    spec = importlib.util.spec_from_file_location(f"q72_{name}", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.INPUT_PATH = INPUT
    module.TOPOLOGY_PATH = TOPOLOGY
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    module.configure_engineering_repair(True)
    module.configure_full_gaussian_prior_space(True)
    module.ZERO_PRESERVING_FULL_PRIOR_MODE = True
    module.DETERMINISTIC_SPINUP_MODE = True
    module.ET_STATE_OPERATOR_MODE = "baseline_clip"
    return module


def prepare_forcing(module) -> pd.DataFrame:
    out = module.load_forcing_panel().sort_values(["comid", "year", "month"]).reset_index(drop=True)
    out["sas_effective_mm"] = np.maximum(
        out["PPT"].fillna(0.0).to_numpy(float) - out["AET"].fillna(0.0).to_numpy(float),
        0.0,
    )
    out["aet_storage_demand_mm"] = np.maximum(
        out["AET"].fillna(0.0).to_numpy(float) - out["PPT"].fillna(0.0).to_numpy(float),
        0.0,
    )
    return out


def _days_seconds(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [pd.Period(f"{int(y)}-{int(m):02d}").days_in_month * 86400.0 for y, m in zip(frame.year, frame.month)],
        dtype=float,
    )


def simulate_local_interface(module, frame: pd.DataFrame, spec: BranchSpec) -> pd.DataFrame:
    out = frame.sort_values(["comid", "year", "month"]).reset_index(drop=True).copy()
    seconds = _days_seconds(out)
    rows: list[dict[str, float | int | str]] = []

    for _, idx in out.groupby("comid", sort=False).groups.items():
        positions = [int(i) for i in idx]
        spin_positions = module._spinup_positions(out, idx)
        source_store, quick_store, base_store = module._periodic_production_state(
            out,
            spin_positions,
            spec.prod_capacity,
            spec.runoff_gamma,
            spec.quick_rho,
            spec.base_rho,
            spec.base_release,
            spec.highflow_scale,
            "baseline_clip",
        )
        for pos in positions:
            source_start = float(source_store)
            quick_start = float(quick_store)
            gw_start = float(base_store)
            positive_input = float(out.at[pos, "sas_effective_mm"])
            saturation_start = float(np.clip(source_start / spec.prod_capacity, 0.0, 1.5))
            quick_generated = min(
                positive_input,
                positive_input * (saturation_start ** spec.runoff_gamma) * spec.highflow_scale,
            )
            source_input = max(positive_input - quick_generated, 0.0)
            source_available = max(source_start + source_input, 0.0)
            overflow = max(source_available - spec.prod_capacity, 0.0)
            source_pre_recharge = min(source_available, spec.prod_capacity)
            gw_recharge = spec.base_release * source_pre_recharge
            source_store = max(source_pre_recharge - gw_recharge, 0.0)
            quick_input = quick_generated + overflow
            quick_available = quick_start + quick_input
            quick_release = (1.0 - spec.quick_rho) * quick_available
            quick_store = spec.quick_rho * quick_available
            gw_available = gw_start + gw_recharge
            gw_discharge = (1.0 - spec.base_rho) * gw_available
            base_store = spec.base_rho * gw_available
            mass_error = (
                source_start + quick_start + gw_start + positive_input
                - source_store - quick_store - base_store - quick_release - gw_discharge
            )
            rows.append({
                "branch_id": spec.branch_id,
                "comid": int(out.at[pos, "comid"]),
                "year": int(out.at[pos, "year"]),
                "month": int(out.at[pos, "month"]),
                "parameter_version": "q72_20260813_54_fixed_branch",
                "state_timing": "start=pre_positive_input;end=post_release",
                "source_store_start_mm": source_start,
                "source_store_pre_recharge_mm": source_pre_recharge,
                "source_store_end_mm": float(source_store),
                "source_saturation_start": saturation_start,
                "source_saturation_end": float(source_store / spec.prod_capacity),
                "positive_input_mm": positive_input,
                "source_positive_input_to_store_mm": source_input,
                "quick_generated_mm": quick_generated,
                "soil_overflow_to_quick_mm": overflow,
                "quick_input_mm": quick_input,
                "quick_routing_store_start_mm": quick_start,
                "quick_release_mm": quick_release,
                "quick_routing_store_end_mm": float(quick_store),
                "gw_recharge_mm": gw_recharge,
                "gw_response_state_start_mm": gw_start,
                "gw_discharge_mm": gw_discharge,
                "gw_response_state_end_mm": float(base_store),
                "q_local_total_mm": quick_release + gw_discharge,
                "et_storage_withdrawn_mm": 0.0,
                "mass_balance_error_mm": mass_error,
                "source_water_capacity_mm": spec.prod_capacity,
                "catchment_area_km2": float(out.at[pos, "IncAreaKm2"]),
                "prod_capacity_mm": spec.prod_capacity,
                "runoff_gamma": spec.runoff_gamma,
                "quick_rho": spec.quick_rho,
                "k_p": spec.base_release,
                "rho_b": spec.base_rho,
                "highflow_scale": spec.highflow_scale,
                "month_seconds": float(seconds[pos]),
                "inc_area_km2": float(out.at[pos, "IncAreaKm2"]),
            })
    return pd.DataFrame(rows)


def crosscheck_component(module, forcing: pd.DataFrame, states: pd.DataFrame, spec: BranchSpec) -> dict[str, float]:
    seconds = _days_seconds(forcing)
    cumarea = forcing["CumAreaKm2"].fillna(forcing["CumAreaKm2"].median()).clip(lower=1).to_numpy(float)
    legacy = module.simulate_production_variant(
        forcing,
        seconds,
        cumarea,
        spec.prod_capacity,
        spec.runoff_gamma,
        spec.quick_rho,
        spec.base_rho,
        spec.base_release,
        highflow_scale=spec.highflow_scale,
        et_state_operator_mode="baseline_clip",
    )
    area = states["inc_area_km2"].to_numpy(float)
    factor = area * 1_000_000.0 / 1000.0 / states["month_seconds"].to_numpy(float) * 35.3146667
    checks = {
        "source_store_end_mm": np.max(np.abs(states["source_store_end_mm"].to_numpy(float) - legacy["storage_mm"])),
        "source_saturation_end": np.max(np.abs(states["source_saturation_end"].to_numpy(float) - legacy["saturation"])),
        "quick_generated_cfs": np.max(np.abs(states["quick_generated_mm"].to_numpy(float) * factor - legacy["quick_cfs"])),
        "gw_recharge_cfs": np.max(np.abs(states["gw_recharge_mm"].to_numpy(float) * factor - legacy["base_cfs"])),
        "overflow_cfs": np.max(np.abs(states["soil_overflow_to_quick_mm"].to_numpy(float) * factor - legacy["overflow_cfs"])),
        "quick_release_cfs": np.max(np.abs(states["quick_release_mm"].to_numpy(float) * factor - legacy["routed_quick_cfs"])),
        "gw_discharge_cfs": np.max(np.abs(states["gw_discharge_mm"].to_numpy(float) * factor - legacy["routed_base_cfs"])),
        "quick_store_mm": np.max(np.abs(states["quick_routing_store_end_mm"].to_numpy(float) - legacy["quick_routing_storage_mm"])),
        "gw_store_mm": np.max(np.abs(states["gw_response_state_end_mm"].to_numpy(float) - legacy["base_routing_storage_mm"])),
    }
    return {key: float(value) for key, value in checks.items()}
