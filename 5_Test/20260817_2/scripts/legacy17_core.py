from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


TEST = Path(r"E:\SPARROW\5_Test")
PARENT_CORE = TEST / "20260816_1" / "scripts" / "legacy16_core.py"
MONTHLY = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
TOPOLOGY = TEST / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
Q_RHO = 0.25
B_RHO = 0.85
WATER_EPS = 1e-12


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_core() -> ModuleType:
    return load_module(PARENT_CORE, "legacy16_frozen_for_17")


def model_spec(model_id: str) -> tuple[str, int | None, int]:
    if model_id.startswith("M0_mu_"):
        return "M0", None, int(model_id.split("_mu_")[1][:-1])
    if model_id.startswith("S0_mu_"):
        return "S0", None, int(model_id.split("_mu_")[1][:-1])
    if model_id.startswith("S1_tau_"):
        left, right = model_id.split("_mu_")
        return "S1", int(left.split("_tau_")[1][:-1]), int(right[:-1])
    raise ValueError(model_id)


def topology_operators(reach_ids: np.ndarray) -> tuple[list[int], dict[int, tuple[int, float]], dict[int, int]]:
    frame = pd.read_csv(TOPOLOGY)
    valid = set(map(int, reach_ids))
    downstream: dict[int, tuple[int, float]] = {}
    for row in frame.itertuples():
        rid = int(row.reach_id)
        if rid in valid and pd.notna(row.downstream_reach) and int(row.downstream_reach) in valid:
            downstream[rid] = (int(row.downstream_reach), float(row.frac))
    indegree = {rid: 0 for rid in valid}
    for _, (down, _) in downstream.items():
        indegree[down] += 1
    queue = sorted([rid for rid, deg in indegree.items() if deg == 0])
    order: list[int] = []
    while queue:
        rid = queue.pop(0)
        order.append(rid)
        if rid in downstream:
            down = downstream[rid][0]
            indegree[down] -= 1
            if indegree[down] == 0:
                queue.append(down)
                queue.sort()
    if len(order) != len(valid):
        raise RuntimeError("topology cycle")
    terminal: dict[int, int] = {}
    for rid in valid:
        current = rid
        seen = set()
        while current in downstream:
            if current in seen:
                raise RuntimeError("topology cycle")
            seen.add(current)
            current = downstream[current][0]
        terminal[rid] = current
    return order, downstream, terminal


def route_matrix(local: np.ndarray, reach_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Route [time, reach] local mass with the frozen same-month R0 operator."""
    order, downstream, terminal = topology_operators(reach_ids)
    index = {int(rid): i for i, rid in enumerate(reach_ids)}
    routed = np.asarray(local, dtype=float).copy()
    for rid in order:
        if rid in downstream:
            down, frac = downstream[rid]
            routed[:, index[down]] += routed[:, index[rid]] * frac
    terminals = np.array(sorted(set(terminal.values())), dtype=int)
    return routed, terminals, terminal


def climate_arrays() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    monthly = pd.read_parquet(MONTHLY)
    climate = monthly.loc[monthly.year.eq(1961)].sort_values(["month", "reach_id"])
    reach_ids = np.sort(climate.reach_id.unique().astype(int))
    fields = [
        "positive_input_mm", "quick_generated_mm", "quick_release_mm",
        "soil_overflow_to_quick_mm", "gw_recharge_mm", "gw_discharge_mm",
        "source_water_capacity_mm", "catchment_area_km2",
    ]
    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        pivot = climate.pivot(index="month", columns="reach_id", values=field).loc[range(1, 13), reach_ids]
        arrays[field] = pivot.to_numpy(float)
    return reach_ids, arrays


def water_partitions(arrays: dict[str, np.ndarray], month_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive_input = arrays["positive_input_mm"][month_index]
    quick_generated = arrays["quick_generated_mm"][month_index]
    bypass = np.clip(np.divide(quick_generated, positive_input, out=np.zeros_like(positive_input), where=positive_input > WATER_EPS), 0.0, 1.0)
    overflow = arrays["soil_overflow_to_quick_mm"][month_index]
    recharge = arrays["gw_recharge_mm"][month_index]
    contact = overflow + recharge
    capacity = arrays["source_water_capacity_mm"][month_index]
    flush = np.where(contact > WATER_EPS, 1.0 - np.exp(-contact / capacity), 0.0)
    quick_share = np.divide(overflow, contact, out=np.zeros_like(contact), where=contact > WATER_EPS)
    gw_share = np.divide(recharge, contact, out=np.zeros_like(contact), where=contact > WATER_EPS)
    return bypass, flush, quick_share, gw_share


def simulate_n_pulse(
    structure: str,
    tau_s: int | None,
    mu_t: int,
    start_month: int,
    injected_kg_n: np.ndarray,
    horizon: int = 2400,
) -> dict[str, np.ndarray]:
    """Monthly post-ledger N-mass pulse under a fixed climatological water sequence."""
    reach_ids, arrays = climate_arrays()
    n_r = len(reach_ids)
    injected = np.asarray(injected_kg_n, dtype=float)
    if injected.shape != (n_r,) or np.any(injected < 0):
        raise ValueError("injected_kg_n must be a nonnegative post-ledger monthly mass vector")
    states = {name: np.zeros(n_r, dtype=float) for name in ("son", "mobile", "quick", "gw")}
    quick_out = np.zeros((horizon, n_r), dtype=float)
    gw_out = np.zeros((horizon, n_r), dtype=float)
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_gw = B_RHO if mu_t == 0 else mu_t / (1.0 + mu_t)
    for t in range(horizon):
        mi = (start_month - 1 + t) % 12
        pulse = injected if t == 0 else np.zeros(n_r)
        bypass, flush, quick_share, gw_share = water_partitions(arrays, mi)
        direct = pulse * bypass
        remaining = pulse - direct
        if structure == "M0":
            source_release = remaining * flush
        elif structure == "S0":
            states["mobile"] += remaining
            source_release = states["mobile"] * flush
            states["mobile"] -= source_release
        elif structure == "S1":
            states["son"] += remaining
            mineralized = states["son"] * (1.0 - rho_s)
            states["son"] -= mineralized
            states["mobile"] += mineralized
            source_release = states["mobile"] * flush
            states["mobile"] -= source_release
        else:
            raise ValueError(structure)
        states["quick"] += direct + source_release * quick_share
        states["gw"] += source_release * gw_share
        q = np.where(arrays["quick_release_mm"][mi] > WATER_EPS, (1.0 - Q_RHO) * states["quick"], 0.0)
        g = np.where(arrays["gw_discharge_mm"][mi] > WATER_EPS, (1.0 - rho_gw) * states["gw"], 0.0)
        states["quick"] -= q
        states["gw"] -= g
        quick_out[t] = q
        gw_out[t] = g
    retained = sum(states.values())
    return {"reach_ids": reach_ids, "quick_release": quick_out, "gw_release": gw_out, "local_release": quick_out + gw_out, "retained": retained}


def simulate_history_replay(model_id: str) -> pd.DataFrame:
    """Independent monthly recurrence initialized by the frozen parent's periodic spinup."""
    parent = parent_core()
    reach_ids, times, arrays, early_positive = parent.prepare_model_arrays()
    structure, tau_s, mu_t = model_spec(model_id)
    initial, _ = parent.spinup_candidate_totals(structure, tau_s, mu_t, early_positive, arrays)
    state = {name: value.copy() for name, value in initial.items()}
    n_t, n_r = len(times), len(reach_ids)
    names = [
        "son_state_end_kg_n", "mobile_state_end_kg_n", "quick_state_end_kg_n", "gw_state_end_kg_n",
        "quick_tn_release_kg_n", "gw_tn_release_kg_n", "local_tn_release_kg_n",
        "negative_removed_kg_n", "negative_unmet_kg_n", "mass_balance_error_kg_n",
    ]
    values = {name: np.zeros((n_t, n_r), dtype=float) for name in names}
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_gw = B_RHO if mu_t == 0 else mu_t / (1.0 + mu_t)
    for t in range(n_t):
        starts = sum(v.astype(np.longdouble) for v in state.values())
        positive = arrays["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        negative = arrays["negative_legacy_eligible_n_surplus_kg_n_month"][t]
        bypass = np.clip(np.divide(arrays["quick_generated_mm"][t], arrays["positive_input_mm"][t], out=np.zeros(n_r), where=arrays["positive_input_mm"][t] > WATER_EPS), 0.0, 1.0)
        contact = arrays["soil_overflow_to_quick_mm"][t] + arrays["gw_recharge_mm"][t]
        flush = np.where(contact > WATER_EPS, 1.0 - np.exp(-contact / arrays["source_water_capacity_mm"][t]), 0.0)
        quick_share = np.divide(arrays["soil_overflow_to_quick_mm"][t], contact, out=np.zeros(n_r), where=contact > WATER_EPS)
        gw_share = np.divide(arrays["gw_recharge_mm"][t], contact, out=np.zeros(n_r), where=contact > WATER_EPS)
        direct = positive * bypass
        remaining = positive - direct
        removed = np.zeros(n_r)
        unmet = negative.copy()
        sink = np.zeros(n_r)
        if structure == "M0":
            source_release = remaining * flush
            sink = remaining - source_release
        elif structure == "S0":
            state["mobile"] += remaining
            take = np.minimum(state["mobile"], negative)
            state["mobile"] -= take
            removed += take
            unmet -= take
            source_release = state["mobile"] * flush
            state["mobile"] -= source_release
        else:
            take = np.minimum(state["mobile"], negative)
            state["mobile"] -= take
            removed += take
            left = negative - take
            take_son = np.minimum(state["son"], left)
            state["son"] -= take_son
            removed += take_son
            unmet = left - take_son
            state["son"] += remaining
            mineralized = state["son"] * (1.0 - rho_s)
            state["son"] -= mineralized
            state["mobile"] += mineralized
            source_release = state["mobile"] * flush
            state["mobile"] -= source_release
        state["quick"] += direct + source_release * quick_share
        state["gw"] += source_release * gw_share
        quick = np.where(arrays["quick_release_mm"][t] > WATER_EPS, (1.0 - Q_RHO) * state["quick"], 0.0)
        gw = np.where(arrays["gw_discharge_mm"][t] > WATER_EPS, (1.0 - rho_gw) * state["gw"], 0.0)
        state["quick"] -= quick
        state["gw"] -= gw
        release = quick + gw
        ends = sum(v.astype(np.longdouble) for v in state.values())
        balance = positive.astype(np.longdouble) + starts - release.astype(np.longdouble) - ends - removed.astype(np.longdouble) - sink.astype(np.longdouble)
        values["son_state_end_kg_n"][t] = state["son"]
        values["mobile_state_end_kg_n"][t] = state["mobile"]
        values["quick_state_end_kg_n"][t] = state["quick"]
        values["gw_state_end_kg_n"][t] = state["gw"]
        values["quick_tn_release_kg_n"][t] = quick
        values["gw_tn_release_kg_n"][t] = gw
        values["local_tn_release_kg_n"][t] = release
        values["negative_removed_kg_n"][t] = removed
        values["negative_unmet_kg_n"][t] = unmet
        values["mass_balance_error_kg_n"][t] = np.asarray(balance, dtype=float)
    frame = pd.DataFrame({
        "reach_id": np.tile(reach_ids, n_t),
        "year": np.repeat([x[0] for x in times], n_r),
        "month": np.repeat([x[1] for x in times], n_r),
    })
    for name, array in values.items():
        frame[name] = array.reshape(-1)
    frame["model_id"] = model_id
    return frame


def simulate_delivery_pulse(mu_t: int, start_month: int, gated: bool, horizon: int = 2400) -> dict[str, np.ndarray]:
    reach_ids, arrays = climate_arrays()
    state = np.ones(len(reach_ids), dtype=float)
    out = np.zeros((horizon, len(reach_ids)), dtype=float)
    rho = mu_t / (1.0 + mu_t)
    for t in range(horizon):
        mi = (start_month - 1 + t) % 12
        allowed = arrays["gw_discharge_mm"][mi] > WATER_EPS if gated else np.ones(len(reach_ids), dtype=bool)
        release = np.where(allowed, (1.0 - rho) * state, 0.0)
        state -= release
        out[t] = release
    return {"reach_ids": reach_ids, "release": out, "retained": state}


def q72_periodic_start_states(arrays: dict[str, np.ndarray]) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_r = arrays["positive_input_mm"].shape[1]
    source = np.zeros(n_r)
    quick = np.zeros(n_r)
    gw = np.zeros(n_r)
    for _ in range(5000):
        before = np.concatenate([source.copy(), quick.copy(), gw.copy()])
        starts: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for mi in range(12):
            starts[mi + 1] = (source.copy(), quick.copy(), gw.copy())
            source, quick, gw, _, _ = q72_step(source, quick, gw, arrays["positive_input_mm"][mi])
        if np.max(np.abs(np.concatenate([source, quick, gw]) - before)) <= 1e-12:
            return starts
    raise RuntimeError("Q72 climatological spinup failed")


def q72_step(source: np.ndarray, quick: np.ndarray, gw: np.ndarray, positive: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    saturation = np.clip(source / 240.0, 0.0, 1.5)
    quick_generated = np.minimum(positive, positive * saturation ** 2.5)
    source_available = np.maximum(source + np.maximum(positive - quick_generated, 0.0), 0.0)
    overflow = np.maximum(source_available - 240.0, 0.0)
    pre = np.minimum(source_available, 240.0)
    recharge = 0.10 * pre
    source_end = np.maximum(pre - recharge, 0.0)
    quick_available = quick + quick_generated + overflow
    quick_release = 0.75 * quick_available
    quick_end = 0.25 * quick_available
    gw_available = gw + recharge
    gw_release = 0.15 * gw_available
    gw_end = 0.85 * gw_available
    return source_end, quick_end, gw_end, quick_release, gw_release


def simulate_q72_hydraulic_pulse(start_month: int, horizon: int = 2400) -> dict[str, np.ndarray]:
    reach_ids, arrays = climate_arrays()
    starts = q72_periodic_start_states(arrays)
    b_source, b_quick, b_gw = [x.copy() for x in starts[start_month]]
    p_source, p_quick, p_gw = [x.copy() for x in starts[start_month]]
    response = np.zeros((horizon, len(reach_ids)), dtype=float)
    for t in range(horizon):
        mi = (start_month - 1 + t) % 12
        baseline = arrays["positive_input_mm"][mi]
        b_source, b_quick, b_gw, bq, bg = q72_step(b_source, b_quick, b_gw, baseline)
        perturbed = baseline + (1.0 if t == 0 else 0.0)
        p_source, p_quick, p_gw, pq, pg = q72_step(p_source, p_quick, p_gw, perturbed)
        response[t] = (pq + pg) - (bq + bg)
    retained = (p_source + p_quick + p_gw) - (b_source + b_quick + b_gw)
    if response.min() < -1e-10:
        raise RuntimeError(f"negative hydraulic incremental response: {response.min()}")
    response = np.maximum(response, 0.0)
    return {"reach_ids": reach_ids, "release": response, "retained": retained}


def tail_metrics(release: np.ndarray, injected: float, horizon: int = 2400) -> dict[str, object]:
    values = np.asarray(release, dtype=float)
    cdf = np.cumsum(values) / injected if injected > 0 else np.full_like(values, np.nan)
    result: dict[str, object] = {"injected": float(injected), "released_at_horizon": float(values.sum()), "cdf_at_horizon": float(cdf[-1]) if injected > 0 else np.nan}
    for probability in (0.50, 0.90, 0.95, 0.99):
        label = f"T{int(probability * 100)}"
        if injected <= 0:
            lag = elapsed = year = np.nan
            status = "not_applicable_zero_injection"
        elif cdf[-1] + 1e-14 < probability:
            lag = elapsed = year = np.nan
            status = "right_censored_200yr"
        else:
            lag = int(np.argmax(cdf >= probability))
            elapsed = lag + 1
            year = elapsed / 12.0
            status = "resolved"
        result[f"{label}_lag_month"] = lag
        result[f"{label}_elapsed_month"] = elapsed
        result[f"{label}_year"] = year
        result[f"{label}_status"] = status
        result[f"{label}_lower_bound_year"] = 200.0 if status == "right_censored_200yr" else np.nan
    for year in (1, 5, 10, 20, 30, 50):
        elapsed = 12 * year
        released = values[:elapsed].sum() / injected if injected > 0 else np.nan
        result[f"tail_mass_fraction_not_yet_released_after_{year}yr"] = float(1.0 - released) if injected > 0 else np.nan
    result["cdf_denominator_semantics"] = "initial_injected_mass_or_water"
    return result
