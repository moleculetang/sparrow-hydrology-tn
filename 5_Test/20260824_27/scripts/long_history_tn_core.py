"""Source-tagged, mass-conserving long-history agricultural N state core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-12
SOURCE_TAGS = ("FERT", "MAN", "BNF", "DEP")
SOURCE_INDEX = {name: index for index, name in enumerate(SOURCE_TAGS)}
LEGACY_TAGS = (SOURCE_INDEX["MAN"], SOURCE_INDEX["BNF"])
DIRECT_TAGS = (SOURCE_INDEX["FERT"], SOURCE_INDEX["DEP"])


@dataclass(frozen=True)
class Candidate:
    name: str
    k_year_minus_1: float | None

    @property
    def monthly_release_probability(self) -> float:
        if self.k_year_minus_1 is None:
            return 0.0
        return float(1.0 - np.exp(-self.k_year_minus_1 / 12.0))


@dataclass
class State:
    legacy_kg_n: np.ndarray
    mineral_kg_n: np.ndarray
    lower_dissolved_kg_n: np.ndarray

    def copy(self) -> "State":
        return State(self.legacy_kg_n.copy(), self.mineral_kg_n.copy(), self.lower_dissolved_kg_n.copy())

    @property
    def total_kg_n(self) -> np.ndarray:
        return self.legacy_kg_n + self.mineral_kg_n + self.lower_dissolved_kg_n


@dataclass
class Drivers:
    source_kg_n: np.ndarray  # time, reach, source tag
    crop_demand_kg_n: np.ndarray  # time, reach
    fast_water_mm: np.ndarray
    percolation_water_mm: np.ndarray
    slow_water_mm: np.ndarray
    upper_mixing_store_mm: np.ndarray
    lower_water_store_mm: np.ndarray

    def validate(self) -> None:
        if self.source_kg_n.ndim != 3 or self.source_kg_n.shape[2] != len(SOURCE_TAGS):
            raise ValueError("source_kg_n must have shape time x reach x four source tags")
        shape = self.source_kg_n.shape[:2]
        for name in (
            "crop_demand_kg_n", "fast_water_mm", "percolation_water_mm", "slow_water_mm",
            "upper_mixing_store_mm", "lower_water_store_mm",
        ):
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} does not match {shape}")
            if not np.isfinite(value).all() or np.min(value) < -1.0e-12:
                raise ValueError(f"{name} is not finite and nonnegative")
        if not np.isfinite(self.source_kg_n).all() or np.min(self.source_kg_n) < -1.0e-12:
            raise ValueError("source input is not finite and nonnegative")


@dataclass
class Simulation:
    terminal_state: State
    arrays: dict[str, np.ndarray]
    maximum_absolute_mass_error_kg_n: float
    maximum_relative_mass_error: float
    minimum_state_or_flux_kg_n: float
    initial_state_mass_kg_n: float
    total_input_kg_n: float
    total_crop_uptake_kg_n: float
    total_fast_export_kg_n: float
    total_slow_export_kg_n: float
    total_unmet_crop_demand_kg_n: float


def zero_state(n_reaches: int) -> State:
    shape = (n_reaches, len(SOURCE_TAGS))
    return State(np.zeros(shape), np.zeros(shape), np.zeros(shape))


def _allocate_inputs(source: np.ndarray, candidate: Candidate) -> tuple[np.ndarray, np.ndarray]:
    legacy = np.zeros_like(source)
    direct = np.zeros_like(source)
    if candidate.k_year_minus_1 is None:
        direct[:] = source
    else:
        direct[:, DIRECT_TAGS] = source[:, DIRECT_TAGS]
        legacy[:, LEGACY_TAGS] = source[:, LEGACY_TAGS]
    return legacy, direct


def simulate(
    drivers: Drivers,
    candidate: Candidate,
    initial_state: State | None = None,
    contact_alpha: float = 1.0,
    contact_beta: float = 1.0,
    record: bool = True,
) -> Simulation:
    """Run one candidate.

    `contact_alpha` is a positive dimensionless multiplier and `contact_beta`
    is a positive exponent. They affect only contact between the shared mobile
    mineral-N stock and the actual fast plus percolation water volumes.
    """
    drivers.validate()
    if not np.isfinite(contact_alpha) or contact_alpha <= 0:
        raise ValueError("contact_alpha must be positive")
    if not np.isfinite(contact_beta) or contact_beta <= 0:
        raise ValueError("contact_beta must be positive")
    nt, nr, ns = drivers.source_kg_n.shape
    state = zero_state(nr) if initial_state is None else initial_state.copy()
    for value in (state.legacy_kg_n, state.mineral_kg_n, state.lower_dissolved_kg_n):
        if value.shape != (nr, ns) or not np.isfinite(value).all() or np.min(value) < -1.0e-12:
            raise ValueError("invalid initial state")
    initial_mass_by_tag = state.total_kg_n.copy()
    cumulative_input = np.zeros((nr, ns))
    cumulative_crop = np.zeros((nr, ns))
    cumulative_fast = np.zeros((nr, ns))
    cumulative_slow = np.zeros((nr, ns))
    names = (
        "legacy_end_kg_n", "mineral_end_kg_n", "lower_dissolved_end_kg_n",
        "mineralized_kg_n", "crop_uptake_kg_n", "fast_export_kg_n",
        "percolated_to_lower_kg_n", "slow_export_kg_n",
        "mobilized_kg_n", "contact_probability", "slow_release_probability",
        "unmet_crop_demand_kg_n",
    )
    arrays: dict[str, np.ndarray] = {}
    if record:
        for name in names[:9]:
            arrays[name] = np.zeros((nt, nr, ns))
        arrays["contact_probability"] = np.zeros((nt, nr))
        arrays["slow_release_probability"] = np.zeros((nt, nr))
        arrays["unmet_crop_demand_kg_n"] = np.zeros((nt, nr))
    max_abs_error = 0.0
    max_relative_error = 0.0
    minimum_value = float("inf")
    total_unmet = 0.0
    q_legacy = candidate.monthly_release_probability

    for t in range(nt):
        source = drivers.source_kg_n[t]
        legacy_input, direct_input = _allocate_inputs(source, candidate)
        legacy_pre = state.legacy_kg_n + legacy_input
        mineralized = q_legacy * legacy_pre
        legacy_end = legacy_pre - mineralized
        mineral_pre = state.mineral_kg_n + direct_input + mineralized

        available_total = mineral_pre.sum(axis=1)
        demand = drivers.crop_demand_kg_n[t]
        crop_fraction = np.divide(
            np.minimum(available_total, demand), available_total,
            out=np.zeros(nr), where=available_total > EPS,
        )
        crop_uptake = mineral_pre * crop_fraction[:, None]
        unmet_crop = np.maximum(demand - crop_uptake.sum(axis=1), 0.0)
        mineral_after_crop = mineral_pre - crop_uptake

        fast_water = drivers.fast_water_mm[t]
        percolation_water = drivers.percolation_water_mm[t]
        contact_water = fast_water + percolation_water
        contact_ratio = np.divide(
            contact_water, drivers.upper_mixing_store_mm[t],
            out=np.full(nr, 1.0e6), where=drivers.upper_mixing_store_mm[t] > EPS,
        )
        contact_exposure = np.minimum(contact_alpha * np.power(np.clip(contact_ratio, 0.0, 1.0e6), contact_beta), 700.0)
        contact_probability = 1.0 - np.exp(-contact_exposure)
        contact_probability[contact_water <= EPS] = 0.0
        mobilized = mineral_after_crop * contact_probability[:, None]
        fast_fraction = np.divide(fast_water, contact_water, out=np.zeros(nr), where=contact_water > EPS)
        fast_export = mobilized * fast_fraction[:, None]
        percolated = mobilized - fast_export
        mineral_end = mineral_after_crop - mobilized

        lower_pre = state.lower_dissolved_kg_n + percolated
        slow_water = drivers.slow_water_mm[t]
        slow_exposure = np.divide(
            slow_water, drivers.lower_water_store_mm[t],
            out=np.full(nr, 700.0), where=drivers.lower_water_store_mm[t] > EPS,
        )
        slow_probability = 1.0 - np.exp(-np.minimum(slow_exposure, 700.0))
        slow_probability[slow_water <= EPS] = 0.0
        slow_export = lower_pre * slow_probability[:, None]
        lower_end = lower_pre - slow_export

        state = State(legacy_end, mineral_end, lower_end)
        cumulative_input += source
        cumulative_crop += crop_uptake
        cumulative_fast += fast_export
        cumulative_slow += slow_export
        closure = initial_mass_by_tag + cumulative_input - cumulative_crop - cumulative_fast - cumulative_slow - state.total_kg_n
        scale = np.maximum(initial_mass_by_tag + cumulative_input, 1.0)
        max_abs_error = max(max_abs_error, float(np.max(np.abs(closure))))
        max_relative_error = max(max_relative_error, float(np.max(np.abs(closure) / scale)))
        minimum_value = min(minimum_value, float(np.min(np.concatenate([
            legacy_end.ravel(), mineralized.ravel(), mineral_end.ravel(), crop_uptake.ravel(),
            mobilized.ravel(), fast_export.ravel(), percolated.ravel(), lower_end.ravel(), slow_export.ravel(),
        ]))))
        total_unmet += float(unmet_crop.sum())
        if record:
            arrays["legacy_end_kg_n"][t] = legacy_end
            arrays["mineral_end_kg_n"][t] = mineral_end
            arrays["lower_dissolved_end_kg_n"][t] = lower_end
            arrays["mineralized_kg_n"][t] = mineralized
            arrays["crop_uptake_kg_n"][t] = crop_uptake
            arrays["fast_export_kg_n"][t] = fast_export
            arrays["percolated_to_lower_kg_n"][t] = percolated
            arrays["slow_export_kg_n"][t] = slow_export
            arrays["mobilized_kg_n"][t] = mobilized
            arrays["contact_probability"][t] = contact_probability
            arrays["slow_release_probability"][t] = slow_probability
            arrays["unmet_crop_demand_kg_n"][t] = unmet_crop

    return Simulation(
        terminal_state=state,
        arrays=arrays,
        maximum_absolute_mass_error_kg_n=max_abs_error,
        maximum_relative_mass_error=max_relative_error,
        minimum_state_or_flux_kg_n=minimum_value,
        initial_state_mass_kg_n=float(initial_mass_by_tag.sum()),
        total_input_kg_n=float(drivers.source_kg_n.sum()),
        total_crop_uptake_kg_n=float(cumulative_crop.sum()),
        total_fast_export_kg_n=float(cumulative_fast.sum()),
        total_slow_export_kg_n=float(cumulative_slow.sum()),
        total_unmet_crop_demand_kg_n=total_unmet,
    )


def independent_periodic_spinup(
    drivers: Drivers,
    candidate: Candidate,
    tolerance_kg_n: float = 1.0e-6,
    max_cycles: int = 2000,
    contact_alpha: float = 1.0,
    contact_beta: float = 1.0,
) -> tuple[State, dict[str, float | int | bool]]:
    """Cycle the same early forcing independently for one candidate."""
    state = zero_state(drivers.source_kg_n.shape[1])
    delta = float("inf")
    cycle_mass_error = 0.0
    for cycle in range(1, max_cycles + 1):
        start = state.copy()
        result = simulate(drivers, candidate, start, contact_alpha, contact_beta, record=False)
        state = result.terminal_state
        delta = float(max(
            np.max(np.abs(state.legacy_kg_n - start.legacy_kg_n)),
            np.max(np.abs(state.mineral_kg_n - start.mineral_kg_n)),
            np.max(np.abs(state.lower_dissolved_kg_n - start.lower_dissolved_kg_n)),
        ))
        cycle_mass_error = max(cycle_mass_error, result.maximum_absolute_mass_error_kg_n)
        if delta <= tolerance_kg_n:
            return state, {
                "cycles": cycle, "terminal_max_abs_delta_kg_n": delta,
                "mass_balance_error_kg_n": cycle_mass_error, "converged": True,
            }
    return state, {
        "cycles": max_cycles, "terminal_max_abs_delta_kg_n": delta,
        "mass_balance_error_kg_n": cycle_mass_error, "converged": False,
    }
