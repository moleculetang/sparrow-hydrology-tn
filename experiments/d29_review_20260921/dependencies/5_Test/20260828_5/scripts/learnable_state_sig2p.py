"""Differentiable global-strength version of the registered SIG2P-S/P state operator."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(r"E:\SPARROW")
SOURCE = ROOT / "5_Test" / "20260827_7" / "scripts"
OLD27 = ROOT / "5_Test" / "20260826_27" / "scripts"
sys.path[:0] = [str(SOURCE), str(OLD27)]

from state_consistent_sig2p import StateConsistent2PResult  # noqa: E402
from dyn2p_hbv import DynamicFluxGate2P  # noqa: E402
from dyn3p_hbv import _dynamic_features, _reweight  # noqa: E402


def _partition(
    fast: torch.Tensor,
    percolation: torch.Tensor,
    score: torch.Tensor,
    lambda_s: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(lambda_s) and float(lambda_s) == 0.0:
        return fast, percolation
    movable = fast + percolation
    active = movable > 1.0e-12
    share = torch.clamp(percolation / movable.clamp_min(1.0e-12), 1.0e-8, 1.0 - 1.0e-8)
    logit_share = torch.log(share) - torch.log1p(-share)
    strength = lambda_s if torch.is_tensor(lambda_s) else torch.as_tensor(lambda_s, dtype=fast.dtype, device=fast.device)
    adjusted_share = torch.sigmoid(logit_share + strength * score)
    return (
        torch.where(active, movable * (1.0 - adjusted_share), fast),
        torch.where(active, movable * adjusted_share, percolation),
    )


def simulate_learnable_sig2p(
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
    regionalized_slow_score: torch.Tensor,
    lambda_s: float | torch.Tensor,
    *,
    force_parent: bool = False,
    collect_storage: bool = False,
    collect_aet: bool = False,
    collect_internal_fluxes: bool = False,
) -> StateConsistent2PResult:
    n_time, n_reach = precipitation_mm_day.shape
    if regionalized_slow_score.shape != (n_reach,):
        raise ValueError("regionalized_slow_score must have shape (n_reach,)")
    detached_lambda = float(lambda_s.detach()) if torch.is_tensor(lambda_s) else float(lambda_s)
    if not 0.0 <= detached_lambda <= 1.0:
        raise ValueError("lambda_s must be in [0,1]")
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
    gate_strength = torch.zeros((), dtype=torch.float64, device=state.device) if gate is None else gate.strength()
    component_rows, storage_rows, aet_rows = [], [], []
    percolation_rows, raw_percolation_rows, raw_fast_rows = [], [], []
    maximum_error = torch.zeros((), dtype=torch.float64, device=state.device)

    for time_index in range(n_time):
        p, pet = precipitation_mm_day[time_index], pet_mm_day[time_index]
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
        rain = _reweight(torch.stack((parent_infiltration, parent_excess), dim=1), logits[:, :2], gate_strength, force_parent or gate is None)
        infiltration = torch.minimum(rain[:, 0], torch.clamp(fc - sm, min=0.0))
        excess = p - infiltration
        sm_after_infiltration = sm + infiltration
        upper_available = upper + excess
        aet = torch.minimum(pet * torch.minimum(sm_after_infiltration / (lp * fc), torch.ones_like(sm)), sm_after_infiltration)
        sm_after_aet = sm_after_infiltration - aet

        parent_perc = torch.minimum(perc, upper_available)
        after_perc = upper_available - parent_perc
        parent_q0 = torch.minimum(k0 * torch.clamp(after_perc - uzl, min=0.0), after_perc)
        after_q0 = after_perc - parent_q0
        parent_q1 = torch.minimum(k1 * after_q0, after_q0)
        parent_upper_carry = after_q0 - parent_q1
        upper_fluxes = _reweight(
            torch.stack((parent_q0 + parent_q1, parent_perc, parent_upper_carry), dim=1),
            logits[:, 2:5], gate_strength, force_parent or gate is None,
        )
        raw_fast, raw_percolation, upper_carry = upper_fluxes.unbind(dim=1)
        fast, percolation = _partition(raw_fast, raw_percolation, regionalized_slow_score, lambda_s)

        lower_available = lower + percolation
        parent_slow = torch.minimum(k2 * lower_available, lower_available)
        parent_lower_carry = lower_available - parent_slow
        lower_fluxes = _reweight(
            torch.stack((parent_slow, parent_lower_carry), dim=1), logits[:, 5:7],
            gate_strength, force_parent or gate is None,
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
        if collect_internal_fluxes:
            percolation_rows.append(percolation)
            raw_percolation_rows.append(raw_percolation)
            raw_fast_rows.append(raw_fast)

    return StateConsistent2PResult(
        final_state_mm=state,
        components_mm_day=torch.stack(component_rows),
        storage_mm=torch.stack(storage_rows) if collect_storage else None,
        aet_mm_day=torch.stack(aet_rows) if collect_aet else None,
        percolation_to_lower_mm_day=torch.stack(percolation_rows) if collect_internal_fluxes else None,
        unadjusted_percolation_mm_day=torch.stack(raw_percolation_rows) if collect_internal_fluxes else None,
        unadjusted_fast_mm_day=torch.stack(raw_fast_rows) if collect_internal_fluxes else None,
        maximum_mass_error_mm=maximum_error,
        gate_strength=gate_strength,
        state_operator_lambda=detached_lambda,
    )
