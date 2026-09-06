"""Raven-equation-conformant rain-only HBV and conserving routing core.

The land model follows the registered Raven ORDERED_SERIES process order:
PRECIP_RAVEN -> INF_HBV -> flush to FAST_RESERVOIR -> SOILEVAP_HBV ->
PERC_CONSTANT -> BASE_THRESH_STOR -> BASE_LINEAR (fast) -> BASE_LINEAR
(slow).  All arrays and state recursions use float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86400.0
PARAMETER_NAMES = (
    "fc_mm",
    "beta",
    "lp",
    "perc_mm_day",
    "uzl_mm",
    "tau0_day",
    "delta_tau10_day",
    "delta_tau21_day",
)
PARAMETER_BOUNDS = np.asarray(
    [
        [50.0, 1500.0],
        [0.25, 6.0],
        [0.20, 1.0],
        [0.01, 12.0],
        [0.0, 150.0],
        [0.20, 8.0],
        [0.20, 45.0],
        [2.0, 1200.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HBVParameters:
    fc_mm: float
    beta: float
    lp: float
    perc_mm_day: float
    uzl_mm: float
    tau0_day: float
    delta_tau10_day: float
    delta_tau21_day: float

    def as_array(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in PARAMETER_NAMES], dtype=np.float64)

    def validate(self) -> None:
        values = self.as_array()
        if not np.isfinite(values).all():
            raise ValueError("HBV parameters must be finite")
        if np.any(values < PARAMETER_BOUNDS[:, 0]) or np.any(values > PARAMETER_BOUNDS[:, 1]):
            raise ValueError(f"HBV parameters outside registered bounds: {values}")
        if not (self.lp * self.fc_mm > 0):
            raise ValueError("LP*FC must be positive")

    @property
    def tau1_day(self) -> float:
        return self.tau0_day + self.delta_tau10_day

    @property
    def tau2_day(self) -> float:
        return self.tau1_day + self.delta_tau21_day

    @staticmethod
    def recession_fraction(tau_day: float) -> float:
        return float(1.0 - np.exp(-1.0 / tau_day))

    @property
    def k0_day(self) -> float:
        return self.recession_fraction(self.tau0_day)

    @property
    def k1_day(self) -> float:
        return self.recession_fraction(self.tau1_day)

    @property
    def k2_day(self) -> float:
        return self.recession_fraction(self.tau2_day)


def sigmoid(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    return np.where(raw >= 0, 1.0 / (1.0 + np.exp(-raw)), np.exp(raw) / (1.0 + np.exp(raw)))


def raw_to_parameters(raw: Iterable[float]) -> HBVParameters:
    raw_array = np.asarray(tuple(raw), dtype=np.float64)
    if raw_array.shape != (len(PARAMETER_NAMES),):
        raise ValueError("Expected exactly eight raw HBV parameters")
    unit = sigmoid(raw_array)
    values = PARAMETER_BOUNDS[:, 0] + unit * (PARAMETER_BOUNDS[:, 1] - PARAMETER_BOUNDS[:, 0])
    parameters = HBVParameters(**dict(zip(PARAMETER_NAMES, map(float, values))))
    parameters.validate()
    return parameters


def parameters_to_raw(parameters: HBVParameters) -> np.ndarray:
    parameters.validate()
    values = parameters.as_array()
    unit = (values - PARAMETER_BOUNDS[:, 0]) / (PARAMETER_BOUNDS[:, 1] - PARAMETER_BOUNDS[:, 0])
    if np.any(unit <= 0.0) or np.any(unit >= 1.0):
        raise ValueError("Exact raw inverse is undefined at a registered physical boundary")
    return np.log(unit) - np.log1p(-unit)


def _as_time_reach(values: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite time-by-Reach array")
    if np.any(array < 0.0):
        raise ValueError(f"{name} cannot be negative")
    return array


def simulate_hbv_ordered(
    precipitation_mm_day: np.ndarray | Iterable[float],
    pet_mm_day: np.ndarray | Iterable[float],
    parameters: HBVParameters,
    initial_state_mm: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Simulate the exact registered Raven ordered-series process sequence."""

    parameters.validate()
    precipitation = _as_time_reach(precipitation_mm_day, "precipitation")
    pet = _as_time_reach(pet_mm_day, "PET")
    if precipitation.shape != pet.shape:
        raise ValueError("Precipitation and PET shapes differ")
    n_time, n_reach = precipitation.shape
    if initial_state_mm is None:
        state = np.zeros((n_reach, 3), dtype=np.float64)
    else:
        state = np.asarray(initial_state_mm, dtype=np.float64).copy()
        if state.shape == (3,) and n_reach == 1:
            state = state[None, :]
        if state.shape != (n_reach, 3) or not np.isfinite(state).all() or np.any(state < 0.0):
            raise ValueError("Initial state must be non-negative Reach-by-3 storage")
    storage = np.empty((n_time, n_reach, 3), dtype=np.float64)
    flux_names = ("infiltration", "excess", "aet", "percolation", "q0", "q1", "q2")
    fluxes = {name: np.empty((n_time, n_reach), dtype=np.float64) for name in flux_names}
    mass_error = np.empty((n_time, n_reach), dtype=np.float64)
    k0, k1, k2 = parameters.k0_day, parameters.k1_day, parameters.k2_day
    for time_index in range(n_time):
        p = precipitation[time_index]
        e = pet[time_index]
        pre_total = state.sum(axis=1).copy()
        sm, fast, slow = state[:, 0], state[:, 1], state[:, 2]

        saturation = np.clip(sm / parameters.fc_mm, 0.0, 1.0)
        initial_excess = np.power(saturation, parameters.beta) * p
        requested_infiltration = p - initial_excess
        infiltration = np.minimum(requested_infiltration, np.maximum(parameters.fc_mm - sm, 0.0))
        excess = p - infiltration
        sm += infiltration
        fast += excess

        aet = np.minimum(e * np.minimum(sm / (parameters.lp * parameters.fc_mm), 1.0), sm)
        sm -= aet

        percolation = np.minimum(parameters.perc_mm_day, fast)
        fast -= percolation
        slow += percolation

        q0 = np.minimum(k0 * np.maximum(fast - parameters.uzl_mm, 0.0), fast)
        fast -= q0
        q1 = np.minimum(k1 * fast, fast)
        fast -= q1
        q2 = np.minimum(k2 * slow, slow)
        slow -= q2

        state[:, 0], state[:, 1], state[:, 2] = sm, fast, slow
        storage[time_index] = state
        for name, value in (
            ("infiltration", infiltration),
            ("excess", excess),
            ("aet", aet),
            ("percolation", percolation),
            ("q0", q0),
            ("q1", q1),
            ("q2", q2),
        ):
            fluxes[name][time_index] = value
        mass_error[time_index] = pre_total + p - aet - q0 - q1 - q2 - state.sum(axis=1)
    return {"storage": storage, "mass_error": mass_error, **fluxes}


def periodic_spinup(
    precipitation_cycle: np.ndarray,
    pet_cycle: np.ndarray,
    parameters: HBVParameters,
    tolerance_mm: float = 1.0e-8,
    max_cycles: int = 500,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    precipitation = _as_time_reach(precipitation_cycle, "spin-up precipitation")
    pet = _as_time_reach(pet_cycle, "spin-up PET")
    state = np.zeros((precipitation.shape[1], 3), dtype=np.float64)
    terminal_delta = float("inf")
    maximum_mass_error = 0.0
    for cycle in range(1, max_cycles + 1):
        previous = state.copy()
        result = simulate_hbv_ordered(precipitation, pet, parameters, state)
        state = result["storage"][-1].copy()
        terminal_delta = float(np.max(np.abs(state - previous)))
        maximum_mass_error = max(maximum_mass_error, float(np.max(np.abs(result["mass_error"]))))
        if terminal_delta <= tolerance_mm:
            return state, {
                "cycles": cycle,
                "terminal_max_abs_delta_mm": terminal_delta,
                "max_abs_mass_error_mm": maximum_mass_error,
                "converged": True,
            }
    return state, {
        "cycles": max_cycles,
        "terminal_max_abs_delta_mm": terminal_delta,
        "max_abs_mass_error_mm": maximum_mass_error,
        "converged": False,
    }


def load_topology(path: Path, reach_ids: np.ndarray) -> tuple[list[int], dict[int, tuple[int, float]], dict[int, int]]:
    topology = pd.read_csv(path)
    topology["reach_id"] = topology.reach_id.astype(int)
    nodes = list(map(int, reach_ids))
    node_set = set(nodes)
    downstream: dict[int, tuple[int, float]] = {}
    for row in topology.itertuples():
        reach = int(row.reach_id)
        if reach not in node_set or pd.isna(row.downstream_reach):
            continue
        target = int(row.downstream_reach)
        if target not in node_set:
            raise ValueError("Topology points outside registered domain")
        fraction = float(row.frac)
        if not (0.0 < fraction <= 1.0):
            raise ValueError("Routing fractions must be in (0,1]")
        downstream[reach] = (target, fraction)
    indegree = {reach: 0 for reach in nodes}
    for target, _ in downstream.values():
        indegree[target] += 1
    queue = sorted(reach for reach, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while queue:
        reach = queue.pop(0)
        order.append(reach)
        if reach in downstream:
            target = downstream[reach][0]
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    if len(order) != len(nodes):
        raise ValueError("Topology must be a complete DAG")
    terminal: dict[int, int] = {}
    for reach in nodes:
        target = reach
        seen: set[int] = set()
        while target in downstream:
            if target in seen:
                raise ValueError("Topology cycle")
            seen.add(target)
            target = downstream[target][0]
        terminal[reach] = target
    return order, downstream, terminal


def route_instantaneous(
    local_components: np.ndarray,
    reach_ids: np.ndarray,
    order: list[int],
    downstream: dict[int, tuple[int, float]],
) -> np.ndarray:
    local = np.asarray(local_components, dtype=np.float64)
    if local.ndim != 3 or local.shape[1] != len(reach_ids) or np.any(local < 0.0):
        raise ValueError("Local components must be non-negative time-by-Reach-by-component")
    routed = local.copy()
    index = {int(reach): i for i, reach in enumerate(reach_ids)}
    for reach in order:
        if reach in downstream:
            target, fraction = downstream[reach]
            routed[:, index[target], :] += fraction * routed[:, index[reach], :]
    return routed


def route_linear_channels_adaptive(
    local_components_m3: np.ndarray,
    reach_ids: np.ndarray,
    order: list[int],
    downstream: dict[int, tuple[int, float]],
    travel_time_days: np.ndarray,
    initial_storage_m3: np.ndarray | None = None,
    cfl_limit: float = 0.9,
    max_substeps: int = 10000,
) -> dict[str, np.ndarray | float | int]:
    """Route passive component volumes through shared linear channel stores."""

    local = np.asarray(local_components_m3, dtype=np.float64)
    tau = np.asarray(travel_time_days, dtype=np.float64)
    if local.ndim != 3 or local.shape[1] != len(reach_ids) or np.any(local < 0.0):
        raise ValueError("Local routed input has invalid dimensions or negative values")
    if tau.ndim == 1:
        if tau.shape != (len(reach_ids),):
            raise ValueError("Static Reach travel time has the wrong shape")
        tau_by_day = np.broadcast_to(tau[None, :], (local.shape[0], len(reach_ids)))
    elif tau.ndim == 2:
        if tau.shape != (local.shape[0], len(reach_ids)):
            raise ValueError("Dynamic Reach travel time must be time-by-Reach")
        tau_by_day = tau
    else:
        raise ValueError("Travel time must be Reach or time-by-Reach")
    if not np.isfinite(tau_by_day).all() or np.any(tau_by_day <= 0.0):
        raise ValueError("Every Reach travel time must be finite and positive")
    if not (0.0 < cfl_limit <= 0.9):
        raise ValueError("Registered CFL limit must be in (0,0.9]")
    n_time, n_reach, n_component = local.shape
    if initial_storage_m3 is None:
        storage = np.zeros((n_reach, n_component), dtype=np.float64)
    else:
        storage = np.asarray(initial_storage_m3, dtype=np.float64).copy()
        if storage.shape != (n_reach, n_component) or np.any(storage < 0.0):
            raise ValueError("Invalid initial channel component storage")
    index = {int(reach): i for i, reach in enumerate(reach_ids)}
    terminal_indices = [index[reach] for reach in reach_ids if int(reach) not in downstream]
    outflow = np.zeros_like(local)
    upstream_inflow = np.zeros_like(local)
    storage_end = np.zeros_like(local)
    terminal_outflow = np.zeros((n_time, n_component), dtype=np.float64)
    mass_error = np.zeros((n_time, n_component), dtype=np.float64)
    substeps_by_day = np.empty(n_time, dtype=np.int64)
    maximum_cfl = 0.0
    for time_index in range(n_time):
        storage_start = storage.sum(axis=0).copy()
        tau_day = tau_by_day[time_index]
        n_substeps = int(np.ceil(np.max(1.0 / tau_day) / cfl_limit))
        n_substeps = max(1, n_substeps)
        if n_substeps > max_substeps:
            raise RuntimeError(f"Required routing substeps {n_substeps} exceed registered maximum")
        substeps_by_day[time_index] = n_substeps
        substep_days = 1.0 / n_substeps
        release_fraction = substep_days / tau_day
        maximum_cfl = max(maximum_cfl, float(release_fraction.max()))
        local_substep = local[time_index] / n_substeps
        terminal_day = np.zeros(n_component, dtype=np.float64)
        for _ in range(n_substeps):
            storage += local_substep
            for reach in order:
                reach_index = index[reach]
                released = release_fraction[reach_index] * storage[reach_index]
                storage[reach_index] -= released
                outflow[time_index, reach_index] += released
                if reach in downstream:
                    target, fraction = downstream[reach]
                    transferred = fraction * released
                    storage[index[target]] += transferred
                    upstream_inflow[time_index, index[target]] += transferred
                else:
                    terminal_day += released
        terminal_outflow[time_index] = terminal_day
        storage_end[time_index] = storage
        mass_error[time_index] = storage_start + local[time_index].sum(axis=0) - storage.sum(axis=0) - terminal_day
    return {
        "outflow_m3": outflow,
        "upstream_inflow_m3": upstream_inflow,
        "storage_end_m3": storage_end,
        "terminal_outflow_m3": terminal_outflow,
        "mass_error_m3": mass_error,
        "substeps_by_day": substeps_by_day,
        "maximum_cfl": maximum_cfl,
    }
