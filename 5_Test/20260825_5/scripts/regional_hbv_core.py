"""Low-capacity regional HBV utilities for the registered Stage-5 experiment.

The land equations are identical to the Raven-conformant ordered HBV sequence
verified in 20260825_3.  The only extension is that each of the eight raw HBV
parameters may have one attribute slope.  A sigmoid maps every Reach-specific
raw value into the already registered physical bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize


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


def sigmoid(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    return np.where(raw >= 0.0, 1.0 / (1.0 + np.exp(-raw)), np.exp(raw) / (1.0 + np.exp(raw)))


def theta_to_parameter_map(theta: Iterable[float], standardized_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tuple(theta), dtype=np.float64)
    features = np.asarray(standardized_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 8 or not np.isfinite(features).all():
        raise ValueError("Expected a finite Reach-by-8 feature matrix")
    if values.shape == (8,):
        intercept = values
        slopes = np.zeros(8, dtype=np.float64)
    elif values.shape == (16,):
        intercept, slopes = values[:8], values[8:]
    else:
        raise ValueError("Theta must contain eight intercepts or eight intercepts plus eight slopes")
    raw_by_reach = intercept[None, :] + features * slopes[None, :]
    unit = sigmoid(raw_by_reach)
    physical = PARAMETER_BOUNDS[:, 0][None, :] + unit * (
        PARAMETER_BOUNDS[:, 1] - PARAMETER_BOUNDS[:, 0]
    )[None, :]
    return raw_by_reach, physical


def _run_ordered_hbv(
    precipitation: np.ndarray,
    pet: np.ndarray,
    physical_parameters: np.ndarray,
    initial_state: np.ndarray,
    collect_components: bool,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    precipitation = np.asarray(precipitation, dtype=np.float64)
    pet = np.asarray(pet, dtype=np.float64)
    physical = np.asarray(physical_parameters, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    if precipitation.shape != pet.shape or precipitation.ndim != 2:
        raise ValueError("Precipitation and PET must be equal time-by-Reach matrices")
    if physical.shape != (precipitation.shape[1], 8):
        raise ValueError("Physical parameter map must be Reach-by-8")
    if state.shape != (precipitation.shape[1], 3):
        raise ValueError("Initial state must be Reach-by-3")
    if not all(np.isfinite(x).all() for x in (precipitation, pet, physical, state)):
        raise ValueError("HBV inputs and state must be finite")
    if np.any(precipitation < 0.0) or np.any(pet < 0.0) or np.any(state < 0.0):
        raise ValueError("HBV forcing and state must be non-negative")

    fc, beta, lp, perc, uzl, tau0, dt10, dt21 = physical.T
    tau1 = tau0 + dt10
    tau2 = tau1 + dt21
    k0 = 1.0 - np.exp(-1.0 / tau0)
    k1 = 1.0 - np.exp(-1.0 / tau1)
    k2 = 1.0 - np.exp(-1.0 / tau2)
    components = (
        np.empty((precipitation.shape[0], precipitation.shape[1], 3), dtype=np.float64)
        if collect_components
        else None
    )
    maximum_mass_error = 0.0
    for time_index in range(precipitation.shape[0]):
        p = precipitation[time_index]
        e = pet[time_index]
        pre_total = state.sum(axis=1).copy()
        sm, fast, slow = state[:, 0], state[:, 1], state[:, 2]

        saturation = np.clip(sm / fc, 0.0, 1.0)
        initial_excess = np.power(saturation, beta) * p
        requested_infiltration = p - initial_excess
        infiltration = np.minimum(requested_infiltration, np.maximum(fc - sm, 0.0))
        excess = p - infiltration
        sm += infiltration
        fast += excess

        aet = np.minimum(e * np.minimum(sm / (lp * fc), 1.0), sm)
        sm -= aet
        percolation = np.minimum(perc, fast)
        fast -= percolation
        slow += percolation
        q0 = np.minimum(k0 * np.maximum(fast - uzl, 0.0), fast)
        fast -= q0
        q1 = np.minimum(k1 * fast, fast)
        fast -= q1
        q2 = np.minimum(k2 * slow, slow)
        slow -= q2

        state[:, 0], state[:, 1], state[:, 2] = sm, fast, slow
        error = pre_total + p - aet - q0 - q1 - q2 - state.sum(axis=1)
        maximum_mass_error = max(maximum_mass_error, float(np.max(np.abs(error))))
        if components is not None:
            components[time_index, :, 0] = q0
            components[time_index, :, 1] = q1
            components[time_index, :, 2] = q2
    return state, components, maximum_mass_error


def periodic_spinup_regional(
    precipitation_cycle: np.ndarray,
    pet_cycle: np.ndarray,
    physical_parameters: np.ndarray,
    tolerance_mm: float,
    max_cycles: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    state = np.zeros((precipitation_cycle.shape[1], 3), dtype=np.float64)
    terminal_delta = float("inf")
    maximum_mass_error = 0.0
    for cycle in range(1, max_cycles + 1):
        previous = state.copy()
        state, _, mass_error = _run_ordered_hbv(
            precipitation_cycle, pet_cycle, physical_parameters, state, False
        )
        terminal_delta = float(np.max(np.abs(state - previous)))
        maximum_mass_error = max(maximum_mass_error, mass_error)
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


@dataclass
class Prediction:
    site_q_m3_s: np.ndarray
    local_components_mm: np.ndarray | None
    raw_parameter_map: np.ndarray
    physical_parameter_map: np.ndarray
    spinup: dict[str, float | int | bool]
    development_mass_max_abs_error_mm: float


class RegionalObjective:
    def __init__(
        self,
        label: str,
        spin_p: np.ndarray,
        spin_pet: np.ndarray,
        development_p: np.ndarray,
        development_pet: np.ndarray,
        observed: np.ndarray,
        support: np.ndarray,
        area_km2: np.ndarray,
        station_trees: np.ndarray,
        standardized_features: np.ndarray,
        prior_intercept: np.ndarray,
        prior_weight: float = 0.002,
        volume_weight: float = 0.1,
        spinup_tolerance: float = 1.0e-8,
        spinup_max_cycles: int = 500,
    ) -> None:
        self.label = label
        self.spin_p = spin_p
        self.spin_pet = spin_pet
        self.development_p = development_p
        self.development_pet = development_pet
        self.observed = observed
        self.support = support
        self.area_km2 = area_km2
        self.station_trees = station_trees
        self.features = standardized_features
        self.prior_intercept = np.asarray(prior_intercept, dtype=np.float64)
        self.prior_weight = prior_weight
        self.volume_weight = volume_weight
        self.spinup_tolerance = spinup_tolerance
        self.spinup_max_cycles = spinup_max_cycles
        self.cache: dict[tuple[float, ...], float] = {}
        self.trace: list[dict[str, object]] = []

    def predict(self, theta: np.ndarray, collect_components: bool = False) -> Prediction:
        raw_map, physical = theta_to_parameter_map(theta, self.features)
        initial, spinup = periodic_spinup_regional(
            self.spin_p,
            self.spin_pet,
            physical,
            self.spinup_tolerance,
            self.spinup_max_cycles,
        )
        if not bool(spinup["converged"]):
            raise RuntimeError("Regional periodic spin-up did not converge")
        _, components, mass_error = _run_ordered_hbv(
            self.development_p,
            self.development_pet,
            physical,
            initial,
            True,
        )
        assert components is not None
        local_total_m3 = components.sum(axis=2) * self.area_km2[None, :] * 1000.0
        site_q = local_total_m3 @ self.support.T / SECONDS_PER_DAY
        return Prediction(
            site_q_m3_s=site_q,
            local_components_mm=components if collect_components else None,
            raw_parameter_map=raw_map,
            physical_parameter_map=physical,
            spinup=spinup,
            development_mass_max_abs_error_mm=mass_error,
        )

    def __call__(self, theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype=np.float64)
        key = tuple(np.round(values, 11))
        if key in self.cache:
            return self.cache[key]
        try:
            prediction = self.predict(values, collect_components=False)
            squared = (np.log1p(prediction.site_q_m3_s) - np.log1p(self.observed)) ** 2
            squared[~np.isfinite(self.observed)] = np.nan
            station_mse = np.nanmean(squared, axis=0)
            station_volume = np.empty(prediction.site_q_m3_s.shape[1], dtype=np.float64)
            for station_index in range(prediction.site_q_m3_s.shape[1]):
                valid = np.isfinite(self.observed[:, station_index])
                station_volume[station_index] = np.log(
                    (prediction.site_q_m3_s[valid, station_index].sum() + 1.0)
                    / (self.observed[valid, station_index].sum() + 1.0)
                )
            process_by_tree = []
            volume_by_tree = []
            for tree in np.unique(self.station_trees):
                select = self.station_trees == tree
                process_by_tree.append(float(np.mean(station_mse[select])))
                volume_by_tree.append(float(np.mean(station_volume[select] ** 2)))
            process_loss = float(np.mean(process_by_tree))
            volume_loss = float(np.mean(volume_by_tree))
            intercept = values[:8]
            slopes = values[8:] if len(values) == 16 else np.zeros(8)
            prior_loss = float(
                np.mean(((intercept - self.prior_intercept) / 1.5) ** 2)
                + np.mean((slopes / 0.75) ** 2)
            )
            raw_excess = np.maximum(np.abs(prediction.raw_parameter_map) - 5.5, 0.0)
            range_loss = float(np.mean(raw_excess**2))
            objective = process_loss + self.volume_weight * volume_loss + self.prior_weight * prior_loss + 0.01 * range_loss
            spinup_cycles = int(prediction.spinup["cycles"])
            mass_error = prediction.development_mass_max_abs_error_mm
            raw_map_max = float(np.max(np.abs(prediction.raw_parameter_map)))
        except Exception:
            objective = 1.0e6
            process_loss = volume_loss = prior_loss = range_loss = mass_error = raw_map_max = float("nan")
            spinup_cycles = -1
        row: dict[str, object] = {
            "label": self.label,
            "evaluation": len(self.trace) + 1,
            "objective": objective,
            "process_loss": process_loss,
            "volume_loss": volume_loss,
            "prior_loss": prior_loss,
            "raw_range_loss": range_loss,
            "spinup_cycles": spinup_cycles,
            "development_mass_max_abs_error_mm": mass_error,
            "raw_map_max_abs": raw_map_max,
        }
        for index, name in enumerate(PARAMETER_NAMES):
            row[f"intercept_{name}"] = float(values[index])
            row[f"slope_{name}"] = float(values[index + 8]) if len(values) == 16 else 0.0
        self.trace.append(row)
        self.cache[key] = objective
        if len(self.trace) % 50 == 0:
            print(f"{self.label} evaluation {len(self.trace)} objective={objective:.6f}", flush=True)
        return objective


def fit_objective(
    objective: RegionalObjective,
    starts: list[np.ndarray],
    bounds: list[tuple[float, float]],
    maxiter: int,
    maxfun: int,
) -> tuple[object, list[dict[str, object]]]:
    results = []
    summary: list[dict[str, object]] = []
    for start_index, start in enumerate(starts):
        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "maxfun": maxfun, "ftol": 1.0e-10, "gtol": 1.0e-5, "maxls": 20},
        )
        results.append(result)
        summary.append(
            {
                "label": objective.label,
                "start": start_index,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "objective": float(result.fun),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
            }
        )
        print(f"{objective.label} start {start_index} objective={result.fun:.6f} success={result.success}", flush=True)
    absolute_best = min(results, key=lambda result: float(result.fun))
    successful = [result for result in results if bool(result.success)]
    successful_best = min(successful, key=lambda result: float(result.fun)) if successful else None
    best = (
        successful_best
        if successful_best is not None and float(successful_best.fun) <= float(absolute_best.fun) + 1.0e-8
        else absolute_best
    )
    return best, summary
