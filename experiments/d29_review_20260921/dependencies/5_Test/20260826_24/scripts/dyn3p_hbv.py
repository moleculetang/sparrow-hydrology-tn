"""Low-capacity dynamic conserving HBV and component-preserving routing.

The learnable layer can only redistribute water that is already present in a
registered HBV flux group.  It cannot add a residual discharge term.  Channel
routing carries fast/intermediate/slow mass tags through identical hydraulic
stores, so the tags sum exactly to routed total flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DYN3PResult:
    final_state_mm: torch.Tensor
    components_mm_day: torch.Tensor
    storage_mm: torch.Tensor | None
    aet_mm_day: torch.Tensor | None
    maximum_mass_error_mm: torch.Tensor
    gate_strength: torch.Tensor


@dataclass
class RouteResult:
    outflow_m3_day: torch.Tensor
    storage_m3: torch.Tensor
    travel_time_day: torch.Tensor
    maximum_mass_error_m3: torch.Tensor
    routing_strength: torch.Tensor


class DynamicFluxGate(nn.Module):
    """One shared width-8 residual gate for three conserving flux groups."""

    def __init__(self, static_size: int = 7, hidden_size: int = 8, seed: int = 260826) -> None:
        super().__init__()
        if static_size != 7 or hidden_size != 8:
            raise ValueError("The registered gate is fixed at 7 static features and width 8")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.fc1 = nn.Linear(9 + static_size, hidden_size, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_size, 8, dtype=torch.float64)
        with torch.no_grad():
            self.fc1.weight.copy_(0.08 * torch.randn(self.fc1.weight.shape, generator=generator, dtype=torch.float64))
            self.fc1.bias.zero_()
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
        # sigmoid(-3.8918) ~= 0.02; the registered null is supplied explicitly.
        self.raw_gate_strength = nn.Parameter(torch.tensor(-3.8918202981106265, dtype=torch.float64))

    def strength(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.raw_gate_strength)

    def residual_logits(self, dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if dynamic.shape[-1] != 9 or static.shape[-1] != 7:
            raise ValueError("Unexpected dynamic or static gate feature count")
        values = torch.cat((dynamic, static), dim=-1)
        return torch.tanh(self.fc2(torch.tanh(self.fc1(values))))


def _reweight(
    parent_fluxes: torch.Tensor,
    residual_logits: torch.Tensor,
    strength: torch.Tensor,
    force_parent: bool,
) -> torch.Tensor:
    if force_parent:
        return parent_fluxes
    total = parent_fluxes.sum(dim=-1, keepdim=True)
    parent_share = torch.where(total > 0.0, parent_fluxes / total.clamp_min(1.0e-30), torch.zeros_like(parent_fluxes))
    proposal = torch.softmax(torch.log(parent_share + 1.0e-12) + residual_logits, dim=-1)
    candidate_share = (1.0 - strength) * parent_share + strength * proposal
    return torch.where(total > 0.0, total * candidate_share, parent_fluxes)


def _dynamic_features(
    p: torch.Tensor,
    pet: torch.Tensor,
    api3: torch.Tensor,
    api30: torch.Tensor,
    sm: torch.Tensor,
    upper: torch.Tensor,
    lower: torch.Tensor,
    fc: torch.Tensor,
    sin_doy: torch.Tensor,
    cos_doy: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    raw = torch.stack(
        (
            torch.log1p(p),
            torch.log1p(api3),
            torch.log1p(api30),
            torch.log1p(pet),
            torch.clamp(sm / fc, 0.0, 2.0),
            torch.clamp(upper / fc, 0.0, 2.0),
            torch.clamp(lower / fc, 0.0, 4.0),
            sin_doy.expand_as(p),
            cos_doy.expand_as(p),
        ),
        dim=-1,
    )
    return (raw - center) / scale


def simulate_dyn3p_hbv(
    precipitation_mm_day: torch.Tensor,
    pet_mm_day: torch.Tensor,
    api3_mm: torch.Tensor,
    api30_mm: torch.Tensor,
    sin_doy: torch.Tensor,
    cos_doy: torch.Tensor,
    physical_parameters: torch.Tensor,
    initial_state_mm: torch.Tensor,
    static_features: torch.Tensor,
    dynamic_center: torch.Tensor,
    dynamic_scale: torch.Tensor,
    gate: DynamicFluxGate | None,
    *,
    force_parent: bool = False,
    collect_storage: bool = False,
    collect_aet: bool = False,
) -> DYN3PResult:
    """Run the registered HBV order with optional conserving flux reallocation."""

    if precipitation_mm_day.dtype != torch.float64:
        raise ValueError("Registered implementation requires float64")
    n_time, n_reach = precipitation_mm_day.shape
    if physical_parameters.shape == (8,):
        physical = physical_parameters.unsqueeze(0).expand(n_reach, -1)
    elif physical_parameters.shape == (n_reach, 8):
        physical = physical_parameters
    else:
        raise ValueError("physical_parameters must be 8 or Reach-by-8")
    if initial_state_mm.shape != (n_reach, 3) or static_features.shape != (n_reach, 7):
        raise ValueError("Invalid initial state or static feature shape")
    if dynamic_center.shape != (9,) or dynamic_scale.shape != (9,) or bool(torch.any(dynamic_scale <= 0.0)):
        raise ValueError("Invalid dynamic scaling")

    fc, beta, lp, perc, uzl, tau0, dt10, dt21 = physical.unbind(dim=1)
    tau1 = tau0 + dt10
    tau2 = tau1 + dt21
    k0 = 1.0 - torch.exp(-1.0 / tau0)
    k1 = 1.0 - torch.exp(-1.0 / tau1)
    k2 = 1.0 - torch.exp(-1.0 / tau2)
    state = initial_state_mm
    strength = torch.zeros((), dtype=torch.float64, device=state.device) if gate is None else gate.strength()
    component_rows: list[torch.Tensor] = []
    storage_rows: list[torch.Tensor] = []
    aet_rows: list[torch.Tensor] = []
    maximum_error = torch.zeros((), dtype=torch.float64, device=state.device)

    for time_index in range(n_time):
        p = precipitation_mm_day[time_index]
        pet = pet_mm_day[time_index]
        sm, upper, lower = state.unbind(dim=1)
        pre_total = state.sum(dim=1)
        if gate is None:
            logits = torch.zeros((n_reach, 8), dtype=torch.float64, device=state.device)
        else:
            dynamic = _dynamic_features(
                p,
                pet,
                api3_mm[time_index],
                api30_mm[time_index],
                sm,
                upper,
                lower,
                fc,
                sin_doy[time_index],
                cos_doy[time_index],
                dynamic_center,
                dynamic_scale,
            )
            logits = gate.residual_logits(dynamic, static_features)

        saturation = torch.clamp(sm / fc, 0.0, 1.0)
        initial_excess = torch.pow(saturation, beta) * p
        parent_infiltration = torch.minimum(p - initial_excess, torch.clamp(fc - sm, min=0.0))
        parent_excess = p - parent_infiltration
        rain_fluxes = _reweight(
            torch.stack((parent_infiltration, parent_excess), dim=1), logits[:, :2], strength, force_parent or gate is None
        )
        infiltration = torch.minimum(rain_fluxes[:, 0], torch.clamp(fc - sm, min=0.0))
        excess = p - infiltration
        sm_after_infiltration = sm + infiltration
        upper_available = upper + excess

        aet = torch.minimum(
            pet * torch.minimum(sm_after_infiltration / (lp * fc), torch.ones_like(sm)), sm_after_infiltration
        )
        sm_after_aet = sm_after_infiltration - aet

        parent_perc = torch.minimum(perc, upper_available)
        after_perc = upper_available - parent_perc
        parent_q0 = torch.minimum(k0 * torch.clamp(after_perc - uzl, min=0.0), after_perc)
        after_q0 = after_perc - parent_q0
        parent_q1 = torch.minimum(k1 * after_q0, after_q0)
        parent_upper_carry = after_q0 - parent_q1
        upper_fluxes = _reweight(
            torch.stack((parent_q0, parent_q1, parent_perc, parent_upper_carry), dim=1),
            logits[:, 2:6],
            strength,
            force_parent or gate is None,
        )
        q0, q1, percolation, upper_carry = upper_fluxes.unbind(dim=1)

        lower_available = lower + percolation
        parent_q2 = torch.minimum(k2 * lower_available, lower_available)
        parent_lower_carry = lower_available - parent_q2
        lower_fluxes = _reweight(
            torch.stack((parent_q2, parent_lower_carry), dim=1),
            logits[:, 6:8],
            strength,
            force_parent or gate is None,
        )
        q2, lower_carry = lower_fluxes.unbind(dim=1)

        state = torch.stack((sm_after_aet, upper_carry, lower_carry), dim=1)
        error = pre_total + p - aet - q0 - q1 - q2 - state.sum(dim=1)
        maximum_error = torch.maximum(maximum_error, torch.max(torch.abs(error)))
        component_rows.append(torch.stack((q0, q1, q2), dim=1))
        if collect_storage:
            storage_rows.append(state)
        if collect_aet:
            aet_rows.append(aet)

    return DYN3PResult(
        final_state_mm=state,
        components_mm_day=torch.stack(component_rows),
        storage_mm=torch.stack(storage_rows) if collect_storage else None,
        aet_mm_day=torch.stack(aet_rows) if collect_aet else None,
        maximum_mass_error_mm=maximum_error,
        gate_strength=strength,
    )


def route_component_stores(
    local_components_m3_day: torch.Tensor,
    reach_ids: list[int],
    order: list[int],
    downstream: dict[int, tuple[int, float]],
    length_m: torch.Tensor,
    width_m: torch.Tensor,
    depth_m: torch.Tensor,
    q_floor_m3_s: torch.Tensor,
    raw_channel_strength: torch.Tensor | None,
    *,
    force_instantaneous: bool = False,
) -> RouteResult:
    """Route component tags through identical linear stores at each Reach."""

    n_time, n_reach, n_component = local_components_m3_day.shape
    if n_component not in (2, 3) or n_reach != len(reach_ids):
        raise ValueError("Expected time-by-Reach-by-two/three components")
    index = {reach: position for position, reach in enumerate(reach_ids)}
    if force_instantaneous or raw_channel_strength is None:
        strength = torch.zeros((), dtype=torch.float64, device=local_components_m3_day.device)
    else:
        strength = torch.sigmoid(raw_channel_strength)

    # Own-model instantaneous topology accumulation defines hydraulic Q; it is
    # not the routed output and never uses the Andreadis reference discharge.
    instantaneous = [local_components_m3_day[:, position, :] for position in range(n_reach)]
    for reach in order:
        if reach in downstream:
            target, fraction = downstream[reach]
            instantaneous[index[target]] = instantaneous[index[target]] + fraction * instantaneous[index[reach]]
    instantaneous_tensor = torch.stack(instantaneous, dim=1)
    q_pre = instantaneous_tensor.sum(dim=2) / 86400.0
    geometry_tau = length_m[None, :] * width_m[None, :] * depth_m[None, :] / (
        86400.0 * torch.maximum(q_pre, q_floor_m3_s[None, :])
    )
    if force_instantaneous or raw_channel_strength is None:
        travel_time = torch.zeros_like(geometry_tau)
    else:
        travel_time = torch.clamp(strength * geometry_tau, min=0.02, max=30.0)

    storage = [
        torch.zeros((n_component,), dtype=torch.float64, device=local_components_m3_day.device)
        for _ in range(n_reach)
    ]
    out_rows: list[torch.Tensor] = []
    storage_rows: list[torch.Tensor] = []
    device = local_components_m3_day.device
    maximum_error = torch.zeros((), dtype=torch.float64, device=device)
    for time_index in range(n_time):
        storage_start = torch.stack(storage).sum(dim=0)
        storage = [value + local_components_m3_day[time_index, position] for position, value in enumerate(storage)]
        outflow_reach = [torch.zeros((n_component,), dtype=torch.float64, device=device) for _ in range(n_reach)]
        terminal_out = torch.zeros((n_component,), dtype=torch.float64, device=device)
        for reach in order:
            position = index[reach]
            if force_instantaneous or raw_channel_strength is None:
                release_fraction = torch.ones((), dtype=torch.float64, device=device)
            else:
                release_fraction = 1.0 - torch.exp(-1.0 / travel_time[time_index, position])
            released = release_fraction * storage[position]
            storage[position] = storage[position] - released
            outflow_reach[position] = released
            if reach in downstream:
                target, fraction = downstream[reach]
                target_position = index[target]
                storage[target_position] = storage[target_position] + fraction * released
            else:
                terminal_out = terminal_out + released
        outflow = torch.stack(outflow_reach)
        storage_tensor = torch.stack(storage)
        error = storage_start + local_components_m3_day[time_index].sum(dim=0) - terminal_out - storage_tensor.sum(dim=0)
        maximum_error = torch.maximum(maximum_error, torch.max(torch.abs(error)))
        out_rows.append(outflow)
        storage_rows.append(storage_tensor)

    return RouteResult(
        outflow_m3_day=torch.stack(out_rows),
        storage_m3=torch.stack(storage_rows),
        travel_time_day=travel_time,
        maximum_mass_error_m3=maximum_error,
        routing_strength=strength,
    )
