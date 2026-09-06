"""Float64 differentiable implementation of the frozen ordered HBV parent.

The process order matches 20260825_3/scripts/hydrology_core.py.  This module
contains no station, loss, period-selection or TN logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
PARAMETER_BOUNDS = (
    (50.0, 1500.0),
    (0.25, 6.0),
    (0.20, 1.0),
    (0.01, 12.0),
    (0.0, 150.0),
    (0.20, 8.0),
    (0.20, 45.0),
    (2.0, 1200.0),
)


@dataclass
class HBVResult:
    final_state_mm: torch.Tensor
    components_mm_day: torch.Tensor | None
    storage_mm: torch.Tensor | None
    mass_error_mm: torch.Tensor | None
    diagnostic_fluxes_mm_day: dict[str, torch.Tensor] | None
    max_abs_mass_error_mm: torch.Tensor


def bounds_tensor(reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(PARAMETER_BOUNDS, dtype=reference.dtype, device=reference.device)


def raw_to_physical(raw: torch.Tensor) -> torch.Tensor:
    """Map (..., 8) unconstrained raw parameters to registered HBV bounds."""

    if raw.shape[-1] != 8:
        raise ValueError("raw parameter tensor must end in eight HBV parameters")
    bounds = bounds_tensor(raw)
    return bounds[:, 0] + torch.sigmoid(raw) * (bounds[:, 1] - bounds[:, 0])


def physical_to_raw(physical: torch.Tensor) -> torch.Tensor:
    if physical.shape[-1] != 8:
        raise ValueError("physical parameter tensor must end in eight HBV parameters")
    bounds = bounds_tensor(physical)
    unit = (physical - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    if bool(torch.any((unit <= 0.0) | (unit >= 1.0)).detach().cpu()):
        raise ValueError("physical parameter lies on or outside a registered bound")
    return torch.log(unit) - torch.log1p(-unit)


def expand_parameters(physical: torch.Tensor, n_reach: int) -> torch.Tensor:
    if physical.shape == (8,):
        return physical.unsqueeze(0).expand(n_reach, -1)
    if physical.shape != (n_reach, 8):
        raise ValueError("physical parameters must be 8 or Reach-by-8")
    return physical


def simulate_ordered_hbv(
    precipitation_mm_day: torch.Tensor,
    pet_mm_day: torch.Tensor,
    physical_parameters: torch.Tensor,
    initial_state_mm: torch.Tensor | None = None,
    *,
    collect_components: bool = True,
    collect_storage: bool = False,
    collect_mass_error: bool = False,
    collect_diagnostic_fluxes: bool = False,
) -> HBVResult:
    """Run the exact registered process order using differentiable operations."""

    if precipitation_mm_day.ndim != 2 or pet_mm_day.shape != precipitation_mm_day.shape:
        raise ValueError("forcing must be equal time-by-Reach tensors")
    if precipitation_mm_day.dtype != torch.float64 or pet_mm_day.dtype != torch.float64:
        raise ValueError("registered implementation requires float64 forcing")
    n_time, n_reach = precipitation_mm_day.shape
    physical = expand_parameters(physical_parameters, n_reach)
    if physical.dtype != torch.float64 or physical.device != precipitation_mm_day.device:
        raise ValueError("parameters must be float64 and on the forcing device")
    if initial_state_mm is None:
        state = torch.zeros((n_reach, 3), dtype=torch.float64, device=precipitation_mm_day.device)
    else:
        if initial_state_mm.shape != (n_reach, 3):
            raise ValueError("initial state must be Reach-by-3")
        state = initial_state_mm

    fc, beta, lp, perc, uzl, tau0, dt10, dt21 = physical.unbind(dim=1)
    tau1 = tau0 + dt10
    tau2 = tau1 + dt21
    k0 = 1.0 - torch.exp(-1.0 / tau0)
    k1 = 1.0 - torch.exp(-1.0 / tau1)
    k2 = 1.0 - torch.exp(-1.0 / tau2)

    component_rows: list[torch.Tensor] = []
    storage_rows: list[torch.Tensor] = []
    error_rows: list[torch.Tensor] = []
    diagnostic_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("infiltration", "excess", "aet", "percolation")
    }
    maximum_error = torch.zeros((), dtype=torch.float64, device=precipitation_mm_day.device)

    for time_index in range(n_time):
        p = precipitation_mm_day[time_index]
        e = pet_mm_day[time_index]
        pre_total = state.sum(dim=1)
        sm, fast, slow = state.unbind(dim=1)

        saturation = torch.clamp(sm / fc, 0.0, 1.0)
        initial_excess = torch.pow(saturation, beta) * p
        requested_infiltration = p - initial_excess
        infiltration = torch.minimum(requested_infiltration, torch.clamp(fc - sm, min=0.0))
        excess = p - infiltration
        sm_after_infiltration = sm + infiltration
        fast_after_excess = fast + excess

        aet = torch.minimum(e * torch.minimum(sm_after_infiltration / (lp * fc), torch.ones_like(sm)), sm_after_infiltration)
        sm_after_aet = sm_after_infiltration - aet

        percolation = torch.minimum(perc, fast_after_excess)
        fast_after_perc = fast_after_excess - percolation
        slow_after_perc = slow + percolation

        q0 = torch.minimum(k0 * torch.clamp(fast_after_perc - uzl, min=0.0), fast_after_perc)
        fast_after_q0 = fast_after_perc - q0
        q1 = torch.minimum(k1 * fast_after_q0, fast_after_q0)
        fast_after_q1 = fast_after_q0 - q1
        q2 = torch.minimum(k2 * slow_after_perc, slow_after_perc)
        slow_after_q2 = slow_after_perc - q2

        state = torch.stack((sm_after_aet, fast_after_q1, slow_after_q2), dim=1)
        error = pre_total + p - aet - q0 - q1 - q2 - state.sum(dim=1)
        maximum_error = torch.maximum(maximum_error, torch.max(torch.abs(error)))
        if collect_components:
            component_rows.append(torch.stack((q0, q1, q2), dim=1))
        if collect_storage:
            storage_rows.append(state)
        if collect_mass_error:
            error_rows.append(error)
        if collect_diagnostic_fluxes:
            for name, value in (
                ("infiltration", infiltration),
                ("excess", excess),
                ("aet", aet),
                ("percolation", percolation),
            ):
                diagnostic_rows[name].append(value)

    components = torch.stack(component_rows, dim=0) if collect_components else None
    storage = torch.stack(storage_rows, dim=0) if collect_storage else None
    mass_error = torch.stack(error_rows, dim=0) if collect_mass_error else None
    diagnostic_fluxes = (
        {name: torch.stack(rows, dim=0) for name, rows in diagnostic_rows.items()}
        if collect_diagnostic_fluxes
        else None
    )
    return HBVResult(state, components, storage, mass_error, diagnostic_fluxes, maximum_error)


def periodic_spinup(
    precipitation_cycle: torch.Tensor,
    pet_cycle: torch.Tensor,
    physical_parameters: torch.Tensor,
    tolerance_mm: float = 1.0e-8,
    max_cycles: int = 500,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    state = torch.zeros(
        (precipitation_cycle.shape[1], 3),
        dtype=torch.float64,
        device=precipitation_cycle.device,
    )
    terminal_delta = float("inf")
    maximum_mass_error = 0.0
    with torch.no_grad():
        for cycle in range(1, max_cycles + 1):
            previous = state
            result = simulate_ordered_hbv(
                precipitation_cycle,
                pet_cycle,
                physical_parameters,
                state,
                collect_components=False,
            )
            state = result.final_state_mm
            terminal_delta = float(torch.max(torch.abs(state - previous)).cpu())
            maximum_mass_error = max(maximum_mass_error, float(result.max_abs_mass_error_mm.cpu()))
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


def route_instantaneous(
    local_components: torch.Tensor,
    reach_ids: list[int],
    order: list[int],
    downstream: dict[int, tuple[int, float]],
) -> torch.Tensor:
    """Differentiable topology accumulation with the frozen route order."""

    if local_components.ndim != 3 or local_components.shape[1] != len(reach_ids):
        raise ValueError("local components must be time-by-Reach-by-component")
    index = {reach: position for position, reach in enumerate(reach_ids)}
    accumulated = [local_components[:, position, :] for position in range(len(reach_ids))]
    for reach in order:
        if reach in downstream:
            target, fraction = downstream[reach]
            accumulated[index[target]] = accumulated[index[target]] + fraction * accumulated[index[reach]]
    return torch.stack(accumulated, dim=1)
