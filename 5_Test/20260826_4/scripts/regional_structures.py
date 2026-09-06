from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegionalSimulation:
    response_mm: np.ndarray
    aet_mm: np.ndarray
    final_state_mm: np.ndarray
    max_abs_mass_error_mm: float


PARAMETER_NAMES = {
    "RAVEN_SACSMA3": ["uztwm", "uzfwm", "lztwm", "lzfsm", "lzfpm", "zperc", "rexp", "pfree", "tau_uz", "tau_lzfs", "tau_lzfp", "pctim"],
    "RAVEN_TOPMODEL3": ["root_cap", "m_deficit", "tau_inter", "slow_scale", "slow_shape", "perc_fraction", "lateral_fraction"],
    "HYPE3L_HYDROLOGY": ["c1", "c2", "c3", "beta_surface", "k_inter", "k_base", "perc12", "perc23"],
    "MTRS3": ["soil_capacity", "runoff_beta", "fast_logit", "inter_logit", "tau_fast", "tau_inter", "tau_slow"],
}

BOUNDS = {
    "RAVEN_SACSMA3": np.asarray([[20, 200], [10, 150], [40, 300], [20, 200], [100, 700], [0.1, 10], [0.5, 10], [0.05, 0.8], [2, 20], [20, 120], [100, 500], [0, 0.08]], float),
    "RAVEN_TOPMODEL3": np.asarray([[100, 800], [20, 250], [8, 90], [0.0005, 0.03], [0.2, 4], [0.002, 0.08], [0.05, 0.6]], float),
    "HYPE3L_HYDROLOGY": np.asarray([[50, 250], [100, 500], [200, 1000], [0.01, 0.3], [0.005, 0.1], [0.001, 0.02], [0.01, 0.15], [0.002, 0.08]], float),
    "MTRS3": np.asarray([[100, 800], [0.5, 4], [-3, 3], [-3, 3], [1, 7], [8, 90], [91, 730]], float),
}

DEFAULT = {
    "RAVEN_SACSMA3": np.asarray([60, 45, 140, 90, 260, 2, 4, 0.35, 8, 45, 180, 0.02], float),
    "RAVEN_TOPMODEL3": np.asarray([300, 80, 25, 0.004, 1.5, 0.015, 0.25], float),
    "HYPE3L_HYDROLOGY": np.asarray([120, 220, 500, 0.08, 0.025, 0.004, 0.05, 0.018], float),
    "MTRS3": np.asarray([350, 1.8, 0.0, 0.0, 3, 30, 180], float),
}


def raw_to_physical(model: str, raw: np.ndarray) -> np.ndarray:
    bounds = BOUNDS[model]
    clipped = np.clip(np.asarray(raw, float), -1.0, 1.0)
    return bounds[:, 0] + (clipped + 1.0) * 0.5 * (bounds[:, 1] - bounds[:, 0])


def physical_to_raw(model: str, physical: np.ndarray) -> np.ndarray:
    bounds = BOUNDS[model]
    return 2.0 * (np.asarray(physical) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1.0


def mtrs_fractions(parameters: np.ndarray) -> tuple[float, float, float]:
    if parameters.ndim == 1:
        logits = np.asarray([parameters[2], parameters[3], 0.0])
        weights = np.exp(logits - logits.max()); weights /= weights.sum()
        return float(weights[0]), float(weights[1]), float(weights[2])
    logits = np.column_stack([parameters[:, 2], parameters[:, 3], np.zeros(len(parameters))])
    weights = np.exp(logits - logits.max(axis=1, keepdims=True)); weights /= weights.sum(axis=1, keepdims=True)
    return weights[:, 0], weights[:, 1], weights[:, 2]


def run_model(model: str, p: np.ndarray, pet: np.ndarray, par: np.ndarray, initial: np.ndarray | None = None, collect: bool = True) -> RegionalSimulation:
    if p.shape != pet.shape or p.ndim != 2:
        raise ValueError("Forcing must be day x reach matrices")
    nday, nreach = p.shape
    state_count = {"RAVEN_SACSMA3": 5, "RAVEN_TOPMODEL3": 3, "HYPE3L_HYDROLOGY": 3, "MTRS3": 4}[model]
    state = np.zeros((nreach, state_count), dtype=np.float64) if initial is None else np.asarray(initial, dtype=np.float64).copy()
    response = np.zeros((nday, nreach, 3), dtype=np.float64) if collect else np.empty((0, nreach, 3))
    aet = np.zeros((nday, nreach), dtype=np.float64) if collect else np.empty((0, nreach))
    max_error = 0.0
    values = par.T if np.asarray(par).ndim == 2 else par
    for t in range(nday):
        rain = np.maximum(0.0, p[t]); demand = np.maximum(0.0, pet[t]); before = state.sum(axis=1)
        if model == "RAVEN_SACSMA3":
            uztwm, uzfwm, lztwm, lzfsm, lzfpm, zperc, rexp, pfree, tau_uz, tau_lzfs, tau_lzfp, pctim = values
            uztw, uzfw, lztw, lzfs, lzfp = state.T
            fast = rain * pctim; water = rain - fast
            fill = np.minimum(water, uztwm - uztw); uztw += fill; water -= fill
            fill = np.minimum(water, uzfwm - uzfw); uzfw += fill; water -= fill; fast += water
            aet1 = np.minimum(uztw, demand * np.minimum(1.0, uztw / (0.5 * uztwm))); uztw -= aet1
            rem = np.maximum(0.0, demand - aet1)
            aet2 = np.minimum(lztw, rem * np.minimum(1.0, lztw / (0.5 * lztwm))); lztw -= aet2
            deficit = 1.0 - (lztw + lzfs + lzfp) / (lztwm + lzfsm + lzfpm)
            perc = np.minimum(uzfw, zperc * (1.0 + rexp * np.maximum(deficit, 0.0) ** 2)); uzfw -= perc
            tension = np.minimum(perc * (1.0 - pfree), lztwm - lztw); lztw += tension; free = perc - tension
            supplemental = np.minimum(free * (1.0 - pfree), lzfsm - lzfs); lzfs += supplemental
            primary = np.minimum(free - supplemental, lzfpm - lzfp); lzfp += primary; fast += np.maximum(0.0, free - supplemental - primary)
            f = 1 - np.exp(-1 / tau_uz); intermediate = uzfw * f; uzfw -= intermediate
            fs = 1 - np.exp(-1 / tau_lzfs); fp = 1 - np.exp(-1 / tau_lzfp)
            slow = lzfs * fs + lzfp * fp; lzfs *= 1 - fs; lzfp *= 1 - fp
            state = np.column_stack([uztw, uzfw, lztw, lzfs, lzfp]); actual_et = aet1 + aet2
        elif model == "RAVEN_TOPMODEL3":
            root_cap, m_deficit, tau_inter, slow_scale, slow_shape, perc_fraction, lateral_fraction = values
            root, inter, groundwater = state.T; deficit = np.maximum(0.0, root_cap - root)
            sat = np.exp(-deficit / m_deficit); fast = rain * np.minimum(1.0, sat); infiltration = rain - fast
            lateral = infiltration * lateral_fraction * np.maximum(0.05, sat); inter += lateral; root += infiltration - lateral
            overflow = np.maximum(0.0, root - root_cap); root -= overflow; inter += overflow
            actual_et = np.minimum(root, demand * np.minimum(1.0, root / (0.6 * root_cap))); root -= actual_et
            recharge = np.minimum(root, root * perc_fraction); root -= recharge; groundwater += recharge
            f = 1 - np.exp(-1 / tau_inter); intermediate = inter * f; inter -= intermediate
            slow_fraction = 1 - np.exp(-slow_scale * np.exp(np.minimum(40.0, slow_shape * groundwater / root_cap)))
            slow = np.minimum(groundwater, groundwater * np.minimum(1.0, slow_fraction)); groundwater -= slow
            state = np.column_stack([root, inter, groundwater])
        elif model == "HYPE3L_HYDROLOGY":
            c1, c2, c3, beta_surface, k_inter, k_base, perc12, perc23 = values
            s1, s2, s3 = state.T; s1 += rain; overflow = np.maximum(0.0, s1 - c1); s1 -= overflow
            saturation = np.minimum(s1, beta_surface * (s1 / c1) ** 2 * s1); s1 -= saturation; fast = overflow + saturation
            aet1 = np.minimum(s1, 0.75 * demand * np.minimum(1.0, s1 / (0.5 * c1))); s1 -= aet1
            rem = np.maximum(0.0, demand - aet1); aet2 = np.minimum(s2, rem * np.minimum(1.0, s2 / (0.6 * c2))); s2 -= aet2
            transfer = np.minimum(s1, perc12 * s1); s1 -= transfer; s2 += transfer
            overflow2 = np.maximum(0.0, s2 - c2); s2 -= overflow2
            transfer = np.minimum(s2, perc23 * s2); s2 -= transfer; s3 += transfer + overflow2
            intermediate = np.minimum(s2, k_inter * s2); s2 -= intermediate
            overflow3 = np.maximum(0.0, s3 - c3); s3 -= overflow3; slow = np.minimum(s3, k_base * s3); s3 -= slow; slow += overflow3
            state = np.column_stack([s1, s2, s3]); actual_et = aet1 + aet2
        elif model == "MTRS3":
            soil_capacity, runoff_beta, _, _, tau_fast, tau_inter, tau_slow = values
            ff, fi, fs = mtrs_fractions(par); soil, sf, si, ss = state.T
            wetness = np.clip(soil / soil_capacity, 0, 1); effective = rain * wetness ** runoff_beta; soil += rain - effective
            overflow = np.maximum(0.0, soil - soil_capacity); soil -= overflow; effective += overflow
            actual_et = np.minimum(soil, demand * np.minimum(1.0, soil / (0.6 * soil_capacity))); soil -= actual_et
            sf += effective * ff; si += effective * fi; ss += effective * fs
            kf = 1 - np.exp(-1 / tau_fast); ki = 1 - np.exp(-1 / tau_inter); ks = 1 - np.exp(-1 / tau_slow)
            fast = sf * kf; intermediate = si * ki; slow = ss * ks; sf -= fast; si -= intermediate; ss -= slow
            state = np.column_stack([soil, sf, si, ss])
        else:
            raise KeyError(model)
        after = state.sum(axis=1); error = before + rain - actual_et - fast - intermediate - slow - after
        max_error = max(max_error, float(np.max(np.abs(error))))
        if collect:
            response[t, :, 0] = fast; response[t, :, 1] = intermediate; response[t, :, 2] = slow; aet[t] = actual_et
    return RegionalSimulation(response, aet, state, max_error)


def periodic_spinup(model: str, p: np.ndarray, pet: np.ndarray, parameters: np.ndarray, tolerance: float = 1e-8, max_cycles: int = 50):
    state = None; delta = np.inf; max_error = 0.0
    for cycle in range(1, max_cycles + 1):
        before = None if state is None else state.copy()
        result = run_model(model, p, pet, parameters, state, collect=False); state = result.final_state_mm
        max_error = max(max_error, result.max_abs_mass_error_mm)
        if before is not None:
            delta = float(np.max(np.abs(state - before)))
            if delta <= tolerance:
                return state, {"cycles": cycle, "terminal_max_abs_delta_mm": delta, "max_abs_mass_error_mm": max_error, "converged": True}
    return state, {"cycles": max_cycles, "terminal_max_abs_delta_mm": delta, "max_abs_mass_error_mm": max_error, "converged": False}
