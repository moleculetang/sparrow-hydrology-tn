"""State-consistent regionalized fast/percolation operator for DYN2P.

The regionalized score acts inside the upper-store flux partition. Adjusted
percolation then enters the lower store before the original lower slow/carry
equation is evaluated. This creates a physically corresponding storage path
for every corrected slow-response flux.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import torch


ROOT = Path(r"E:\SPARROW")
OLD27 = ROOT / "5_Test" / "20260826_27" / "scripts"
if str(OLD27) not in sys.path:
    sys.path.insert(0, str(OLD27))

from dyn2p_hbv import DynamicFluxGate2P  # noqa: E402
from dyn3p_hbv import _dynamic_features, _reweight  # noqa: E402


@dataclass
class StateConsistent2PResult:
    final_state_mm: torch.Tensor
    components_mm_day: torch.Tensor
    storage_mm: torch.Tensor | None
    aet_mm_day: torch.Tensor | None
    percolation_to_lower_mm_day: torch.Tensor | None
    unadjusted_percolation_mm_day: torch.Tensor | None
    unadjusted_fast_mm_day: torch.Tensor | None
    maximum_mass_error_mm: torch.Tensor
    gate_strength: torch.Tensor
    state_operator_lambda: float


def _state_partition(
    fast: torch.Tensor,
    percolation: torch.Tensor,
    regionalized_score: torch.Tensor,
    state_operator_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservatively reallocate only fast+percolation.

    A literal lambda=0 branch is intentional: it makes the registered parent
    identity exact rather than merely close after logit clipping.
    """
    if float(state_operator_lambda) == 0.0:
        return fast, percolation
    movable = fast + percolation
    active = movable > 1.0e-12
    share = torch.clamp(percolation / movable.clamp_min(1.0e-12), 1.0e-8, 1.0 - 1.0e-8)
    logit_share = torch.log(share) - torch.log1p(-share)
    adjusted_share = torch.sigmoid(logit_share + float(state_operator_lambda) * regionalized_score)
    adjusted_fast = torch.where(active, movable * (1.0 - adjusted_share), fast)
    adjusted_percolation = torch.where(active, movable * adjusted_share, percolation)
    return adjusted_fast, adjusted_percolation


def simulate_state_consistent_dyn2p(
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
    state_operator_lambda: float,
    *,
    force_parent: bool = False,
    collect_storage: bool = False,
    collect_aet: bool = False,
    collect_internal_fluxes: bool = False,
) -> StateConsistent2PResult:
    n_time, n_reach = precipitation_mm_day.shape
    if regionalized_slow_score.shape != (n_reach,):
        raise ValueError("regionalized_slow_score must have shape (n_reach,)")
    if not 0.0 <= float(state_operator_lambda) <= 1.0:
        raise ValueError("state_operator_lambda must be in [0, 1]")
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
    percolation_rows: list[torch.Tensor] = []
    unadjusted_percolation_rows: list[torch.Tensor] = []
    unadjusted_fast_rows: list[torch.Tensor] = []
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
            pet * torch.minimum(sm_after_infiltration / (lp * fc), torch.ones_like(sm)),
            sm_after_infiltration,
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
        fast_unadjusted, percolation_unadjusted, upper_carry = upper_fluxes.unbind(dim=1)
        fast, percolation = _state_partition(
            fast_unadjusted, percolation_unadjusted, regionalized_slow_score, state_operator_lambda
        )

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
        if collect_internal_fluxes:
            percolation_rows.append(percolation)
            unadjusted_percolation_rows.append(percolation_unadjusted)
            unadjusted_fast_rows.append(fast_unadjusted)

    return StateConsistent2PResult(
        final_state_mm=state,
        components_mm_day=torch.stack(component_rows),
        storage_mm=torch.stack(storage_rows) if collect_storage else None,
        aet_mm_day=torch.stack(aet_rows) if collect_aet else None,
        percolation_to_lower_mm_day=torch.stack(percolation_rows) if collect_internal_fluxes else None,
        unadjusted_percolation_mm_day=torch.stack(unadjusted_percolation_rows) if collect_internal_fluxes else None,
        unadjusted_fast_mm_day=torch.stack(unadjusted_fast_rows) if collect_internal_fluxes else None,
        maximum_mass_error_mm=maximum_error,
        gate_strength=strength,
        state_operator_lambda=float(state_operator_lambda),
    )
