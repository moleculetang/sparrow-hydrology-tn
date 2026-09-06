"""Nested two-response version of the conserving dynamic HBV core."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


CORE = Path(r"E:\SPARROW\5_Test\20260826_24\scripts")
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from dyn3p_hbv import DYN3PResult, _dynamic_features, _reweight  # noqa: E402


class DynamicFluxGate2P(nn.Module):
    """Width-8 gate with two operational responses: upper-store fast and lower-store slow."""

    def __init__(self, static_size: int = 7, hidden_size: int = 8, seed: int = 260826) -> None:
        super().__init__()
        if static_size != 7 or hidden_size != 8:
            raise ValueError("Registered 2P gate is fixed at 7 static features and width 8")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.fc1 = nn.Linear(9 + static_size, hidden_size, dtype=torch.float64)
        # rain(2), upper fast/percolation/carry(3), lower slow/carry(2)
        self.fc2 = nn.Linear(hidden_size, 7, dtype=torch.float64)
        with torch.no_grad():
            self.fc1.weight.copy_(0.08 * torch.randn(self.fc1.weight.shape, generator=generator, dtype=torch.float64))
            self.fc1.bias.zero_()
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
        self.raw_gate_strength = nn.Parameter(torch.tensor(-3.8918202981106265, dtype=torch.float64))

    def strength(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.raw_gate_strength)

    def residual_logits(self, dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        values = torch.cat((dynamic, static), dim=-1)
        return torch.tanh(self.fc2(torch.tanh(self.fc1(values))))


def simulate_dyn2p_hbv(
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
    gate: DynamicFluxGate2P | None,
    *,
    force_parent: bool = False,
    collect_storage: bool = False,
    collect_aet: bool = False,
) -> DYN3PResult:
    n_time, n_reach = precipitation_mm_day.shape
    if physical_parameters.shape == (8,):
        physical = physical_parameters.unsqueeze(0).expand(n_reach, -1)
    elif physical_parameters.shape == (n_reach, 8):
        physical = physical_parameters
    else:
        raise ValueError("physical_parameters must be 8 or Reach-by-8")
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
            logits = torch.zeros((n_reach, 7), dtype=torch.float64, device=state.device)
        else:
            dynamic = _dynamic_features(
                p, pet, api3_mm[time_index], api30_mm[time_index], sm, upper, lower, fc,
                sin_doy[time_index], cos_doy[time_index], dynamic_center, dynamic_scale,
            )
            logits = gate.residual_logits(dynamic, static_features)

        saturation = torch.clamp(sm / fc, 0.0, 1.0)
        initial_excess = torch.pow(saturation, beta) * p
        parent_infiltration = torch.minimum(p - initial_excess, torch.clamp(fc - sm, min=0.0))
        parent_excess = p - parent_infiltration
        rain = _reweight(
            torch.stack((parent_infiltration, parent_excess), dim=1), logits[:, :2], strength,
            force_parent or gate is None,
        )
        infiltration = torch.minimum(rain[:, 0], torch.clamp(fc - sm, min=0.0))
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
            torch.stack((parent_q0 + parent_q1, parent_perc, parent_upper_carry), dim=1),
            logits[:, 2:5], strength, force_parent or gate is None,
        )
        fast, percolation, upper_carry = upper_fluxes.unbind(dim=1)

        lower_available = lower + percolation
        parent_slow = torch.minimum(k2 * lower_available, lower_available)
        parent_lower_carry = lower_available - parent_slow
        lower_fluxes = _reweight(
            torch.stack((parent_slow, parent_lower_carry), dim=1), logits[:, 5:7], strength,
            force_parent or gate is None,
        )
        slow, lower_carry = lower_fluxes.unbind(dim=1)
        state = torch.stack((sm_after_aet, upper_carry, lower_carry), dim=1)
        error = pre_total + p - aet - fast - slow - state.sum(dim=1)
        maximum_error = torch.maximum(maximum_error, torch.max(torch.abs(error)))
        component_rows.append(torch.stack((fast, slow), dim=1))
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
