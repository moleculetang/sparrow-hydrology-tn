from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


torch.set_default_dtype(torch.float64)
EPS = 1e-12


@dataclass
class HydroParameters:
    prod_capacity: torch.Tensor
    runoff_gamma: torch.Tensor
    quick_rho: torch.Tensor
    base_release: torch.Tensor
    base_rho: torch.Tensor


def _logit(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    value = np.clip(value, 1e-9, 1 - 1e-9)
    return np.log(value / (1 - value))


def physical_to_raw(values: dict[str, float]) -> torch.Tensor:
    c = (values["prod_capacity"] - 50.0) / 950.0
    gamma = (values["runoff_gamma"] - 0.5) / 4.5
    rho_q = (values["quick_rho"] - 0.01) / 0.89
    release = (values["base_release"] - 0.005) / 0.495
    remaining = 0.999 - values["quick_rho"] - 0.05
    rho_s = (values["base_rho"] - values["quick_rho"] - 0.05) / max(remaining, 1e-9)
    return torch.tensor([_logit(c), _logit(gamma), _logit(rho_q), _logit(release), _logit(rho_s)], dtype=torch.float64)


def raw_to_physical(raw: torch.Tensor) -> HydroParameters:
    unit = torch.sigmoid(raw)
    capacity = 50.0 + 950.0 * unit[..., 0]
    gamma = 0.5 + 4.5 * unit[..., 1]
    rho_q = 0.01 + 0.89 * unit[..., 2]
    release = 0.005 + 0.495 * unit[..., 3]
    rho_s = rho_q + 0.05 + (0.999 - rho_q - 0.05) * unit[..., 4]
    return HydroParameters(capacity, gamma, rho_q, release, rho_s)


def parameter_dict(parameters: HydroParameters) -> dict[str, float]:
    return {
        "prod_capacity": float(parameters.prod_capacity.detach().cpu()),
        "runoff_gamma": float(parameters.runoff_gamma.detach().cpu()),
        "quick_rho": float(parameters.quick_rho.detach().cpu()),
        "base_release": float(parameters.base_release.detach().cpu()),
        "base_rho": float(parameters.base_rho.detach().cpu()),
    }


def simulate(
    effective: torch.Tensor,
    aet_demand: torch.Tensor,
    parameters: HydroParameters,
    initial_source: torch.Tensor,
    initial_quick: torch.Tensor,
    initial_delayed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run a Reach-parallel, time-sequential, mass-conserving three-store model.

    Inputs use shape [time, reach]. Parameters may be scalar or Reach vectors.
    """
    source = initial_source
    quick_store = initial_quick
    delayed_store = initial_delayed
    names = [
        "source_start", "quick_start", "delayed_start", "quick_generated", "overflow",
        "delayed_recharge", "quick_release", "delayed_discharge", "source_end",
        "quick_end", "delayed_end", "mass_error",
    ]
    records: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for t in range(effective.shape[0]):
        source_start, quick_start, delayed_start = source, quick_store, delayed_store
        total_start = source + quick_store + delayed_store
        saturation = torch.clamp(source / parameters.prod_capacity, min=0.0, max=1.5)
        generated = torch.minimum(effective[t], effective[t] * torch.pow(saturation, parameters.runoff_gamma))
        source = torch.clamp(source + torch.clamp(effective[t] - generated, min=0.0), min=0.0)
        withdrawn = torch.minimum(aet_demand[t], source)
        source = torch.clamp(source - withdrawn, min=0.0)
        overflow = torch.clamp(source - parameters.prod_capacity, min=0.0)
        source = torch.minimum(source, parameters.prod_capacity)
        recharge = parameters.base_release * source
        source = torch.clamp(source - recharge, min=0.0)
        quick_available = torch.clamp(quick_store + generated + overflow, min=0.0)
        quick_release = (1.0 - parameters.quick_rho) * quick_available
        quick_store = parameters.quick_rho * quick_available
        delayed_available = torch.clamp(delayed_store + recharge, min=0.0)
        delayed_discharge = (1.0 - parameters.base_rho) * delayed_available
        delayed_store = parameters.base_rho * delayed_available
        total_end = source + quick_store + delayed_store
        mass_error = total_start + effective[t] - withdrawn - total_end - quick_release - delayed_discharge
        values = {
            "source_start": source_start, "quick_start": quick_start, "delayed_start": delayed_start,
            "quick_generated": generated, "overflow": overflow, "delayed_recharge": recharge,
            "quick_release": quick_release, "delayed_discharge": delayed_discharge,
            "source_end": source, "quick_end": quick_store, "delayed_end": delayed_store,
            "mass_error": mass_error,
        }
        for name, value in values.items():
            records[name].append(value)
    return {name: torch.stack(values, dim=0) for name, values in records.items()}


def route_volumes(local_depth: torch.Tensor, area_km2: torch.Tensor, upstream: torch.Tensor) -> torch.Tensor:
    local_volume = local_depth * area_km2[None, :] * 1000.0
    return torch.matmul(local_volume, upstream.T)
