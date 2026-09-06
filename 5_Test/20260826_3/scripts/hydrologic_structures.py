from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Simulation:
    response_mm: np.ndarray
    aet_mm: np.ndarray
    states_mm: np.ndarray
    mass_error_mm: np.ndarray
    final_state_mm: np.ndarray


def _release(storage: float, tau_day: float) -> tuple[float, float]:
    fraction = 1.0 - np.exp(-1.0 / max(tau_day, 1.0e-6))
    flux = storage * fraction
    return storage - flux, flux


def _evap(storage: float, pet: float, reference: float) -> tuple[float, float]:
    demand = pet * min(1.0, storage / max(reference, 1.0e-9))
    actual = min(storage, max(0.0, demand))
    return storage - actual, actual


def _finalize(
    responses: list[list[float]], aet: list[float], states: list[np.ndarray], errors: list[float], final: np.ndarray
) -> Simulation:
    result = Simulation(
        np.asarray(responses, dtype=np.float64),
        np.asarray(aet, dtype=np.float64),
        np.asarray(states, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        np.asarray(final, dtype=np.float64),
    )
    if np.any(result.response_mm < -1.0e-12) or np.any(result.aet_mm < -1.0e-12) or np.any(result.states_mm < -1.0e-12):
        raise RuntimeError("Negative water state or flux")
    return result


def run_sacsma3(p: np.ndarray, pet: np.ndarray, par: np.ndarray, initial: np.ndarray | None = None) -> Simulation:
    """Mass-conserving SAC-SMA response subset with upper tension/free and two lower free stores.

    The accounting follows SAC-SMA storage roles and Sacramento-style deficit-enhanced
    percolation. It is deliberately called a response subset, not a bitwise reproduction
    of every operational NWS SAC-SMA option.
    """
    uztwm, uzfwm, lztwm, lzfsm, lzfpm, zperc, rexp, pfree, tau_uz, tau_lzfs, tau_lzfp, pctim = par
    state = np.zeros(5, dtype=float) if initial is None else np.asarray(initial, dtype=float).copy()
    responses: list[list[float]] = []
    aets: list[float] = []
    states: list[np.ndarray] = []
    errors: list[float] = []
    capacities = np.asarray([uztwm, uzfwm, lztwm, lzfsm, lzfpm])
    for rain, demand in zip(p, pet):
        before = float(state.sum())
        uztw, uzfw, lztw, lzfs, lzfp = state
        fast = max(0.0, float(rain)) * pctim
        water = max(0.0, float(rain)) - fast
        fill = min(water, uztwm - uztw); uztw += fill; water -= fill
        fill = min(water, uzfwm - uzfw); uzfw += fill; water -= fill
        fast += water
        uztw, aet1 = _evap(uztw, float(demand), 0.5 * uztwm)
        remaining_pet = max(0.0, float(demand) - aet1)
        lztw, aet2 = _evap(lztw, remaining_pet, 0.5 * lztwm)
        deficit = 1.0 - (lztw + lzfs + lzfp) / max(lztwm + lzfsm + lzfpm, 1.0e-9)
        percolation = min(uzfw, zperc * (1.0 + rexp * max(deficit, 0.0) ** 2))
        uzfw -= percolation
        tension_fill = min(percolation * (1.0 - pfree), lztwm - lztw)
        lztw += tension_fill
        free = percolation - tension_fill
        supplemental = min(free * (1.0 - pfree), lzfsm - lzfs); lzfs += supplemental
        primary = min(free - supplemental, lzfpm - lzfp); lzfp += primary
        fast += max(0.0, free - supplemental - primary)
        uzfw, intermediate = _release(uzfw, tau_uz)
        lzfs, slow_s = _release(lzfs, tau_lzfs)
        lzfp, slow_p = _release(lzfp, tau_lzfp)
        state = np.minimum(np.asarray([uztw, uzfw, lztw, lzfs, lzfp]), capacities)
        slow = slow_s + slow_p
        after = float(state.sum())
        errors.append(before + float(rain) - aet1 - aet2 - fast - intermediate - slow - after)
        responses.append([fast, intermediate, slow]); aets.append(aet1 + aet2); states.append(state.copy())
    return _finalize(responses, aets, states, errors, state)


def run_topmodel3(p: np.ndarray, pet: np.ndarray, par: np.ndarray, initial: np.ndarray | None = None) -> Simulation:
    """TOPMODEL-style saturation area plus intermediate and nonlinear slow reservoirs."""
    root_cap, m_deficit, tau_inter, slow_scale, slow_shape, perc_fraction, lateral_fraction = par
    state = np.zeros(3, dtype=float) if initial is None else np.asarray(initial, dtype=float).copy()
    responses: list[list[float]] = []; aets: list[float] = []; states: list[np.ndarray] = []; errors: list[float] = []
    for rain, demand in zip(p, pet):
        before = float(state.sum())
        root, inter, groundwater = state
        deficit = max(0.0, root_cap - root)
        saturated_fraction = float(np.exp(-deficit / max(m_deficit, 1.0e-9)))
        fast = float(rain) * min(1.0, saturated_fraction)
        infiltration = float(rain) - fast
        lateral_input = infiltration * lateral_fraction * max(0.05, saturated_fraction)
        inter += lateral_input
        root += infiltration - lateral_input
        overflow = max(0.0, root - root_cap); root -= overflow; inter += overflow
        root, actual_et = _evap(root, float(demand), 0.6 * root_cap)
        recharge = min(root, root * perc_fraction); root -= recharge; groundwater += recharge
        inter, intermediate = _release(inter, tau_inter)
        nonlinear_fraction = 1.0 - np.exp(-slow_scale * np.exp(slow_shape * groundwater / max(root_cap, 1.0e-9)))
        slow = min(groundwater, groundwater * min(1.0, nonlinear_fraction)); groundwater -= slow
        state = np.asarray([root, inter, groundwater])
        after = float(state.sum())
        errors.append(before + float(rain) - actual_et - fast - intermediate - slow - after)
        responses.append([fast, intermediate, slow]); aets.append(actual_et); states.append(state.copy())
    return _finalize(responses, aets, states, errors, state)


def run_hype3l(p: np.ndarray, pet: np.ndarray, par: np.ndarray, initial: np.ndarray | None = None) -> Simulation:
    """HYPE-style hydrology-only three-soil-layer response model."""
    c1, c2, c3, beta_surface, k_inter, k_base, perc12, perc23 = par
    state = np.zeros(3, dtype=float) if initial is None else np.asarray(initial, dtype=float).copy()
    responses: list[list[float]] = []; aets: list[float] = []; states: list[np.ndarray] = []; errors: list[float] = []
    for rain, demand in zip(p, pet):
        before = float(state.sum())
        s1, s2, s3 = state
        s1 += float(rain)
        overflow = max(0.0, s1 - c1); s1 -= overflow
        saturation = min(s1, beta_surface * (s1 / max(c1, 1.0e-9)) ** 2 * s1); s1 -= saturation
        fast = overflow + saturation
        s1, aet1 = _evap(s1, 0.75 * float(demand), 0.5 * c1)
        s2, aet2 = _evap(s2, max(0.0, float(demand) - aet1), 0.6 * c2)
        transfer12 = min(s1, perc12 * s1); s1 -= transfer12; s2 += transfer12
        overflow2 = max(0.0, s2 - c2); s2 -= overflow2
        transfer23 = min(s2, perc23 * s2); s2 -= transfer23; s3 += transfer23 + overflow2
        intermediate = min(s2, k_inter * s2); s2 -= intermediate
        overflow3 = max(0.0, s3 - c3); s3 -= overflow3
        slow = min(s3, k_base * s3); s3 -= slow
        slow += overflow3
        state = np.asarray([s1, s2, s3])
        after = float(state.sum())
        errors.append(before + float(rain) - aet1 - aet2 - fast - intermediate - slow - after)
        responses.append([fast, intermediate, slow]); aets.append(aet1 + aet2); states.append(state.copy())
    return _finalize(responses, aets, states, errors, state)


def run_mtrs3(p: np.ndarray, pet: np.ndarray, par: np.ndarray, initial: np.ndarray | None = None) -> Simulation:
    """Mass-conserving multi-timescale response stores with explicit 1-7, 8-90, 91-730 d scales."""
    soil_capacity, runoff_beta, f_fast, f_inter, tau_fast, tau_inter, tau_slow = par
    if not (1.0 <= tau_fast <= 7.0 and 8.0 <= tau_inter <= 90.0 and 91.0 <= tau_slow <= 730.0):
        raise ValueError("MTRS3 response times violate registered bands")
    if f_fast < 0 or f_inter < 0 or f_fast + f_inter > 1:
        raise ValueError("Invalid response fractions")
    state = np.zeros(4, dtype=float) if initial is None else np.asarray(initial, dtype=float).copy()
    responses: list[list[float]] = []; aets: list[float] = []; states: list[np.ndarray] = []; errors: list[float] = []
    for rain, demand in zip(p, pet):
        before = float(state.sum())
        soil, fast_store, inter_store, slow_store = state
        wetness = np.clip(soil / max(soil_capacity, 1.0e-9), 0.0, 1.0)
        effective = float(rain) * wetness ** runoff_beta
        soil += float(rain) - effective
        overflow = max(0.0, soil - soil_capacity); soil -= overflow; effective += overflow
        soil, actual_et = _evap(soil, float(demand), 0.6 * soil_capacity)
        fast_store += effective * f_fast
        inter_store += effective * f_inter
        slow_store += effective * (1.0 - f_fast - f_inter)
        fast_store, fast = _release(fast_store, tau_fast)
        inter_store, intermediate = _release(inter_store, tau_inter)
        slow_store, slow = _release(slow_store, tau_slow)
        state = np.asarray([soil, fast_store, inter_store, slow_store])
        after = float(state.sum())
        errors.append(before + float(rain) - actual_et - fast - intermediate - slow - after)
        responses.append([fast, intermediate, slow]); aets.append(actual_et); states.append(state.copy())
    return _finalize(responses, aets, states, errors, state)


RUNNERS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None], Simulation]] = {
    "RAVEN_SACSMA3": run_sacsma3,
    "RAVEN_TOPMODEL3": run_topmodel3,
    "HYPE3L_HYDROLOGY": run_hype3l,
    "MTRS3": run_mtrs3,
}


DEFAULT_PARAMETERS = {
    "RAVEN_SACSMA3": np.asarray([60, 45, 140, 90, 260, 2.0, 4.0, 0.35, 8.0, 45.0, 180.0, 0.02], float),
    "RAVEN_TOPMODEL3": np.asarray([300, 80, 25, 0.004, 1.5, 0.015, 0.25], float),
    "HYPE3L_HYDROLOGY": np.asarray([120, 220, 500, 0.08, 0.025, 0.004, 0.05, 0.018], float),
    "MTRS3": np.asarray([350, 1.8, 0.30, 0.30, 3.0, 30.0, 180.0], float),
}


def periodic_spinup(
    runner: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None], Simulation],
    p: np.ndarray,
    pet: np.ndarray,
    parameters: np.ndarray,
    tolerance: float = 1.0e-8,
    max_cycles: int = 500,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    state = None
    delta = np.inf
    max_error = 0.0
    for cycle in range(1, max_cycles + 1):
        before = None if state is None else state.copy()
        simulation = runner(p, pet, parameters, state)
        state = simulation.final_state_mm
        max_error = max(max_error, float(np.max(np.abs(simulation.mass_error_mm))))
        if before is not None:
            delta = float(np.max(np.abs(state - before)))
            if delta <= tolerance:
                return state, {"cycles": cycle, "terminal_max_abs_delta_mm": delta, "max_abs_mass_error_mm": max_error, "converged": True}
    return np.asarray(state), {"cycles": max_cycles, "terminal_max_abs_delta_mm": float(delta), "max_abs_mass_error_mm": max_error, "converged": False}
