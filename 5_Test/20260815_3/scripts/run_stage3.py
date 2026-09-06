from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260815_3")
PARENT = Path(r"E:\SPARROW\5_Test\20260815_2")
INPUT = PARENT / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
EARLY = PARENT / "outputs" / "pre1961_early_n_mean_by_reach.parquet"
PARENT_AUDIT = PARENT / "reports" / "completion_audit.json"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
Q_RHO = 0.25
B_RHO = 0.85
WATER_EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow conda environment required, got {sys.prefix}")
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def source_partition(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    positive = out.positive_legacy_eligible_n_surplus_kg_n_month.to_numpy(float)
    positive_water = out.positive_input_mm.to_numpy(float)
    quick_generated = out.quick_generated_mm.to_numpy(float)
    bypass = np.divide(
        quick_generated,
        positive_water,
        out=np.zeros(len(out), dtype=float),
        where=positive_water > WATER_EPS,
    )
    bypass = np.clip(bypass, 0.0, 1.0)
    contact = out.soil_contact_water_mm.to_numpy(float)
    capacity = out.source_water_capacity_mm.to_numpy(float)
    flush = np.where(contact > WATER_EPS, 1.0 - np.exp(-contact / capacity), 0.0)
    direct = positive * bypass
    remaining = positive - direct
    mobilized = remaining * flush
    overflow = out.soil_overflow_to_quick_mm.to_numpy(float)
    recharge = out.gw_recharge_mm.to_numpy(float)
    overflow_share = np.divide(
        overflow, contact, out=np.zeros(len(out), dtype=float), where=contact > WATER_EPS
    )
    recharge_share = np.divide(
        recharge, contact, out=np.zeros(len(out), dtype=float), where=contact > WATER_EPS
    )
    out["quick_bypass_fraction_recomputed"] = bypass
    out["source_flush_fraction"] = flush
    out["direct_current_quick_n_input_kg_n"] = direct
    out["m0_current_mobile_available_kg_n"] = remaining
    out["m0_current_mobile_released_kg_n"] = mobilized
    out["soil_current_n_to_quick_input_kg_n"] = mobilized * overflow_share
    out["soil_current_n_to_gw_input_kg_n"] = mobilized * recharge_share
    out["quick_path_n_input_kg_n"] = direct + out.soil_current_n_to_quick_input_kg_n
    out["gw_path_n_input_kg_n"] = out.soil_current_n_to_gw_input_kg_n
    out["m0_same_month_unmobilized_sink_kg_n"] = remaining - mobilized
    out["m0_unresolved_negative_surplus_kg_n"] = out.negative_legacy_eligible_n_surplus_kg_n_month
    out["m0_mobile_state_end_kg_n"] = 0.0
    return out


def route_one_month(
    q_start: np.ndarray,
    b_start: np.ndarray,
    q_input: np.ndarray,
    b_input: np.ndarray,
    q_water_release: np.ndarray,
    b_water_release: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_available = q_start + q_input
    b_available = b_start + b_input
    q_release = np.where(q_water_release > WATER_EPS, (1.0 - Q_RHO) * q_available, 0.0)
    b_release = np.where(b_water_release > WATER_EPS, (1.0 - B_RHO) * b_available, 0.0)
    return q_release, q_available - q_release, b_release, b_available - b_release


def spinup_states(monthly: pd.DataFrame, early: pd.DataFrame, reach_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    climate = monthly.loc[monthly.year.eq(1961)].copy()
    early_month = early.set_index("reach_id").loc[reach_ids, "early_1961_1965_positive_surplus_mean_kg_n_year"].to_numpy(float) / 12.0
    q_state = np.zeros(len(reach_ids), dtype=float)
    b_state = np.zeros(len(reach_ids), dtype=float)
    last_delta = np.inf
    cycles = 0
    for cycle in range(1, 501):
        start_q = q_state.copy()
        start_b = b_state.copy()
        for month in range(1, 13):
            block = climate.loc[climate.month.eq(month)].set_index("reach_id").loc[reach_ids]
            temp = block.copy()
            temp["positive_legacy_eligible_n_surplus_kg_n_month"] = early_month
            temp["negative_legacy_eligible_n_surplus_kg_n_month"] = 0.0
            part = source_partition(temp)
            _, q_state, _, b_state = route_one_month(
                q_state,
                b_state,
                part.quick_path_n_input_kg_n.to_numpy(float),
                part.gw_path_n_input_kg_n.to_numpy(float),
                part.quick_release_mm.to_numpy(float),
                part.gw_discharge_mm.to_numpy(float),
            )
        last_delta = float(max(np.max(np.abs(q_state - start_q)), np.max(np.abs(b_state - start_b))))
        cycles = cycle
        if last_delta <= 1e-10:
            break
    return q_state, b_state, {"cycles": cycles, "terminal_max_abs_delta_kg_n": last_delta, "converged": last_delta <= 1e-10}


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))
    if not parent.get("pass"):
        raise RuntimeError("20260815_2 did not pass")
    start_hashes = {str(p): sha256(p) for p in [INPUT, EARLY, PARENT_AUDIT]}
    dump(REPORTS / "parent_hashes_start.json", start_hashes)

    monthly = pd.read_parquet(INPUT).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    early = pd.read_parquet(EARLY)
    reach_ids = np.array(sorted(monthly.reach_id.unique()), dtype=int)
    q_state, b_state, spinup = spinup_states(monthly, early, reach_ids)
    dump(REPORTS / "spinup_audit.json", spinup)
    rows: list[pd.DataFrame] = []
    for (year, month), block in monthly.groupby(["year", "month"], sort=True):
        part = source_partition(block.set_index("reach_id").loc[reach_ids].reset_index())
        q_start = q_state.copy()
        b_start = b_state.copy()
        q_release, q_state, b_release, b_state = route_one_month(
            q_start,
            b_start,
            part.quick_path_n_input_kg_n.to_numpy(float),
            part.gw_path_n_input_kg_n.to_numpy(float),
            part.quick_release_mm.to_numpy(float),
            part.gw_discharge_mm.to_numpy(float),
        )
        part["quick_n_state_start_kg_n"] = q_start
        part["quick_n_release_kg_n"] = q_release
        part["quick_n_state_end_kg_n"] = q_state
        part["base_n_state_start_kg_n"] = b_start
        part["base_n_release_kg_n"] = b_release
        part["base_n_state_end_kg_n"] = b_state
        part["local_diffuse_tn_release_kg_n"] = q_release + b_release
        part["m0_system_mass_balance_error_kg_n"] = (
            part.positive_legacy_eligible_n_surplus_kg_n_month
            + part.quick_n_state_start_kg_n
            + part.base_n_state_start_kg_n
            - part.quick_n_release_kg_n
            - part.base_n_release_kg_n
            - part.quick_n_state_end_kg_n
            - part.base_n_state_end_kg_n
            - part.m0_same_month_unmobilized_sink_kg_n
        )
        water_volume_factor = part.q_local_total_mm * part.catchment_area_km2
        part["local_diffuse_tn_concentration_proxy_mg_l"] = np.divide(
            part.local_diffuse_tn_release_kg_n,
            water_volume_factor,
            out=np.full(len(part), np.nan),
            where=water_volume_factor.to_numpy(float) > WATER_EPS,
        )
        part["source_model_id"] = "M0"
        part["transport_model_id"] = "T0_q72_fixed_response"
        rows.append(part)
    result = pd.concat(rows, ignore_index=True).sort_values(["reach_id", "year", "month"])
    result.to_parquet(OUT / "m0_reach_month_local_n_1961_2022.parquet", index=False)

    max_balance = float(result.m0_system_mass_balance_error_kg_n.abs().max())
    diagnostics = {
        "scenario_id": "20260815_3",
        "rows": len(result),
        "reaches": int(result.reach_id.nunique()),
        "months": int(len(result[["year", "month"]].drop_duplicates())),
        "max_abs_mass_balance_error_kg_n": max_balance,
        "max_abs_mobile_end_kg_n": float(result.m0_mobile_state_end_kg_n.abs().max()),
        "quick_bypass_max": float(result.quick_bypass_fraction_recomputed.max()),
        "quick_bypass_min": float(result.quick_bypass_fraction_recomputed.min()),
        "negative_surplus_treatment": "reported_unresolved_no_negative_river_load_in_M0",
        "point_source_tn_status": "missing_not_fabricated",
        "spinup": spinup,
    }
    dump(REPORTS / "mass_balance_audit.json", diagnostics)
    end_hashes = {str(p): sha256(p) for p in [INPUT, EARLY, PARENT_AUDIT]}
    dump(REPORTS / "parent_hashes_end.json", end_hashes)
    if start_hashes != end_hashes:
        raise RuntimeError("parent changed during stage 3")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
