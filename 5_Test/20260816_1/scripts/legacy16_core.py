from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
S14_6 = TEST / "20260814_6"
S14_9 = TEST / "20260814_9"
S15_1 = TEST / "20260815_1"
S15_2 = TEST / "20260815_2"
S15_4 = TEST / "20260815_4"
S15_5 = TEST / "20260815_5"
S15_7 = TEST / "20260815_7"
S16_1 = TEST / "20260816_1"

MONTHLY_PATH = S15_2 / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
EARLY_PATH = S15_2 / "outputs" / "pre1961_early_n_mean_by_reach.parquet"
OBS_PATH = S15_1 / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLD_PATH = S15_1 / "outputs" / "tn_fold_registry.parquet"
TOPOLOGY_PATH = TEST / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
STAGE4_SCRIPT = S15_4 / "scripts" / "run_stage4.py"
STAGE5_SCRIPT = S15_5 / "scripts" / "run_stage5.py"

TAUS = (12, 36, 60, 96, 144, 240, 480)
MUS = (0, 12, 36, 60, 96, 144, 240)
Q_RHO = 0.25
B_RHO = 0.85
WATER_EPS = 1e-12


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow conda environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_manifest(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths}


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def all_candidate_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for structure in ("M0", "S0"):
        for mu in MUS:
            specs.append({
                "model_id": f"{structure}_mu_{mu:03d}m",
                "source_structure": structure,
                "soil_tau_month": None,
                "delivery_mu_month": mu,
            })
    for tau in TAUS:
        for mu in MUS:
            specs.append({
                "model_id": f"S1_tau_{tau:03d}m_mu_{mu:03d}m",
                "source_structure": "S1",
                "soil_tau_month": tau,
                "delivery_mu_month": mu,
            })
    if len(specs) != 63 or len({str(x["model_id"]) for x in specs}) != 63:
        raise RuntimeError("candidate registry must contain exactly 63 unique models")
    return specs


def candidate_registry_frame() -> pd.DataFrame:
    frame = pd.DataFrame(all_candidate_specs())
    frame["transport_operator"] = np.where(
        frame.delivery_mu_month.eq(0), "T0_rho_0.85", "T1_rho_mu_over_1_plus_mu"
    )
    frame["T0_T1_mutually_exclusive"] = True
    return frame


def numeric_comparison(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    columns: list[str],
    rtol: float = 1e-12,
    atol: float = 1e-9,
) -> dict[str, object]:
    left = actual.sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    key_equal = left[keys].equals(right[keys])
    details: dict[str, object] = {
        "actual_rows": len(left),
        "expected_rows": len(right),
        "key_equal": bool(key_equal),
        "columns": {},
    }
    passed = len(left) == len(right) and key_equal
    if len(left) != len(right) or not key_equal:
        details["pass"] = False
        return details
    for column in columns:
        a = left[column].to_numpy(float)
        b = right[column].to_numpy(float)
        finite_equal = np.array_equal(np.isfinite(a), np.isfinite(b))
        max_abs = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
        close = bool(finite_equal and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
        details["columns"][column] = {"max_abs_difference": max_abs, "pass": close}
        passed = passed and close
    details["pass"] = bool(passed)
    return details


def prepare_model_arrays() -> tuple[np.ndarray, list[tuple[int, int]], dict[str, np.ndarray], np.ndarray]:
    stage4 = load_module(STAGE4_SCRIPT, "legacy16_shared_stage4")
    monthly = pd.read_parquet(MONTHLY_PATH)
    early = pd.read_parquet(EARLY_PATH)
    reach_ids, times, arrays = stage4.prepare_arrays(monthly)
    early_positive = (
        early.set_index("reach_id")
        .loc[reach_ids, "early_1961_1965_positive_surplus_mean_kg_n_year"]
        .to_numpy(float)
        / 12.0
    )
    return reach_ids, times, arrays, early_positive


def _withdraw_total(pool: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    removed = np.minimum(pool, demand)
    pool -= removed
    return removed, demand - removed


def _water_partitions(arrays: dict[str, np.ndarray], t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive_input = arrays["positive_input_mm"][t]
    quick_generated = arrays["quick_generated_mm"][t]
    bypass = np.clip(
        np.divide(quick_generated, positive_input, out=np.zeros_like(positive_input), where=positive_input > WATER_EPS),
        0.0,
        1.0,
    )
    overflow = arrays["soil_overflow_to_quick_mm"][t]
    recharge = arrays["gw_recharge_mm"][t]
    contact = overflow + recharge
    capacity = arrays["source_water_capacity_mm"][t]
    flush = np.where(contact > WATER_EPS, 1.0 - np.exp(-contact / capacity), 0.0)
    quick_share = np.divide(overflow, contact, out=np.zeros_like(contact), where=contact > WATER_EPS)
    gw_share = np.divide(recharge, contact, out=np.zeros_like(contact), where=contact > WATER_EPS)
    return bypass, flush, quick_share, gw_share


def spinup_candidate_totals(
    structure: str,
    tau_s: int | None,
    mu_t: int,
    early_positive: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    state = {name: np.zeros(len(early_positive), dtype=float) for name in ("son", "mobile", "quick", "gw")}
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_gw = B_RHO if mu_t == 0 else mu_t / (1.0 + mu_t)
    delta = np.inf
    for cycle in range(1, 5001):
        before = np.concatenate([value.copy() for value in state.values()])
        for t in range(12):
            bypass, flush, quick_share, gw_share = _water_partitions(arrays, t)
            direct = early_positive * bypass
            remaining = early_positive - direct
            if structure == "M0":
                source_release = remaining * flush
            elif structure == "S0":
                state["mobile"] += remaining
                source_release = state["mobile"] * flush
                state["mobile"] -= source_release
            else:
                state["son"] += remaining
                mineralized = state["son"] * (1.0 - rho_s)
                state["son"] -= mineralized
                state["mobile"] += mineralized
                source_release = state["mobile"] * flush
                state["mobile"] -= source_release
            state["quick"] += direct + source_release * quick_share
            state["gw"] += source_release * gw_share
            quick_release = np.where(
                arrays["quick_release_mm"][t] > WATER_EPS,
                (1.0 - Q_RHO) * state["quick"],
                0.0,
            )
            gw_release = np.where(
                arrays["gw_discharge_mm"][t] > WATER_EPS,
                (1.0 - rho_gw) * state["gw"],
                0.0,
            )
            state["quick"] -= quick_release
            state["gw"] -= gw_release
        after = np.concatenate(list(state.values()))
        delta = float(np.max(np.abs(after - before)))
        if delta <= 1e-9:
            break
    return state, {
        "cycles": cycle,
        "terminal_max_abs_delta_kg_n": delta,
        "converged": bool(delta <= 1e-9),
        "T0_T1_mutually_exclusive": True,
        "gw_release_rho": rho_gw,
    }


def simulate_candidate_totals(
    model_id: str,
    structure: str,
    tau_s: int | None,
    mu_t: int,
    reach_ids: np.ndarray,
    times: list[tuple[int, int]],
    arrays: dict[str, np.ndarray],
    early_positive: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    state, spinup = spinup_candidate_totals(structure, tau_s, mu_t, early_positive, arrays)
    n_t, n_r = len(times), len(reach_ids)
    names = (
        "son_state_end_kg_n", "mobile_state_end_kg_n", "quick_state_end_kg_n", "gw_state_end_kg_n",
        "quick_tn_release_kg_n", "gw_tn_release_kg_n", "local_tn_release_kg_n",
        "negative_removed_kg_n", "negative_unmet_kg_n", "same_month_unmobilized_sink_kg_n",
        "mass_balance_error_kg_n", "mass_balance_relative_error",
    )
    values = {name: np.zeros((n_t, n_r), dtype=float) for name in names}
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_gw = B_RHO if mu_t == 0 else mu_t / (1.0 + mu_t)
    for t in range(n_t):
        start_total = sum(value.astype(np.longdouble) for value in state.values())
        positive = arrays["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        negative = arrays["negative_legacy_eligible_n_surplus_kg_n_month"][t]
        bypass, flush, quick_share, gw_share = _water_partitions(arrays, t)
        direct = positive * bypass
        remaining = positive - direct
        removed_total = np.zeros(n_r, dtype=float)
        unmet = negative.copy()
        sink = np.zeros(n_r, dtype=float)
        if structure == "M0":
            source_release = remaining * flush
            sink = remaining - source_release
        elif structure == "S0":
            state["mobile"] += remaining
            removed, unmet = _withdraw_total(state["mobile"], negative)
            removed_total += removed
            source_release = state["mobile"] * flush
            state["mobile"] -= source_release
        else:
            removed, left = _withdraw_total(state["mobile"], negative)
            removed_total += removed
            removed_son, unmet = _withdraw_total(state["son"], left)
            removed_total += removed_son
            state["son"] += remaining
            mineralized = state["son"] * (1.0 - rho_s)
            state["son"] -= mineralized
            state["mobile"] += mineralized
            source_release = state["mobile"] * flush
            state["mobile"] -= source_release
        state["quick"] += direct + source_release * quick_share
        state["gw"] += source_release * gw_share
        quick_release = np.where(
            arrays["quick_release_mm"][t] > WATER_EPS,
            (1.0 - Q_RHO) * state["quick"],
            0.0,
        )
        gw_release = np.where(
            arrays["gw_discharge_mm"][t] > WATER_EPS,
            (1.0 - rho_gw) * state["gw"],
            0.0,
        )
        state["quick"] -= quick_release
        state["gw"] -= gw_release
        release = quick_release + gw_release
        end_total = sum(value.astype(np.longdouble) for value in state.values())
        balance = (
            positive.astype(np.longdouble) + start_total - release.astype(np.longdouble)
            - end_total - removed_total.astype(np.longdouble) - sink.astype(np.longdouble)
        )
        scale = (
            np.abs(positive.astype(np.longdouble)) + np.abs(start_total) + np.abs(release.astype(np.longdouble))
            + np.abs(end_total) + np.abs(removed_total.astype(np.longdouble)) + np.abs(sink.astype(np.longdouble))
        )
        values["son_state_end_kg_n"][t] = state["son"]
        values["mobile_state_end_kg_n"][t] = state["mobile"]
        values["quick_state_end_kg_n"][t] = state["quick"]
        values["gw_state_end_kg_n"][t] = state["gw"]
        values["quick_tn_release_kg_n"][t] = quick_release
        values["gw_tn_release_kg_n"][t] = gw_release
        values["local_tn_release_kg_n"][t] = release
        values["negative_removed_kg_n"][t] = removed_total
        values["negative_unmet_kg_n"][t] = unmet
        values["same_month_unmobilized_sink_kg_n"][t] = sink
        values["mass_balance_error_kg_n"][t] = np.asarray(balance, dtype=float)
        values["mass_balance_relative_error"][t] = np.asarray(np.abs(balance) / np.maximum(scale, 1.0), dtype=float)
    frame = pd.DataFrame({
        "reach_id": np.tile(reach_ids, n_t),
        "year": np.repeat([year for year, _ in times], n_r),
        "month": np.repeat([month for _, month in times], n_r),
    })
    for name, array in values.items():
        frame[name] = array.reshape(-1)
    for name in ("q_local_total_mm", "catchment_area_km2", "quick_release_mm", "gw_discharge_mm"):
        frame[name] = arrays[name].reshape(-1)
    frame["model_id"] = model_id
    frame["source_structure"] = structure
    frame["soil_legacy_tau_month"] = np.nan if tau_s is None else tau_s
    frame["effective_tn_delivery_mu_month"] = mu_t
    audit = {
        "model_id": model_id,
        "spinup": spinup,
        "max_abs_mass_balance_error_kg_n": float(np.max(np.abs(values["mass_balance_error_kg_n"]))),
        "max_relative_mass_balance_error": float(np.max(values["mass_balance_relative_error"])),
        "minimum_state_or_flux_kg_n": float(min(
            *(value.min() for value in state.values()),
            values["quick_tn_release_kg_n"].min(),
            values["gw_tn_release_kg_n"].min(),
        )),
        "T0_T1_mutually_exclusive": True,
    }
    return frame, audit
