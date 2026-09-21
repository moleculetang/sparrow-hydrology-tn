"""Numba-backed exact reverse mode for the registered channel recurrence.

Hydraulic release fractions are evaluated from candidate-own instantaneous
flow.  Their dependence on local flow is intentionally stop-gradient; the
route remains fully differentiable with respect to component volumes through
the conserving recurrence and with respect to the global routing strength.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import njit


@njit(cache=True)
def _forward_route(local: np.ndarray, k: np.ndarray, order: np.ndarray, down: np.ndarray, frac: np.ndarray):
    n_time, n_reach, n_component = local.shape
    storage = np.zeros((n_reach, n_component), dtype=np.float64)
    out = np.zeros_like(local)
    upstream = np.zeros_like(local)
    available = np.zeros_like(local)
    storage_end = np.zeros_like(local)
    for t in range(n_time):
        storage += local[t]
        for j in range(n_reach):
            r = order[j]
            available[t, r] = storage[r]
            released = k[t, r] * storage[r]
            storage[r] -= released
            out[t, r] = released
            d = down[r]
            if d >= 0:
                moved = frac[r] * released
                storage[d] += moved
                upstream[t, d] += moved
        storage_end[t] = storage
    return out, upstream, available, storage_end


@njit(cache=True)
def _backward_route(
    grad_out: np.ndarray,
    grad_upstream: np.ndarray,
    available: np.ndarray,
    k: np.ndarray,
    order: np.ndarray,
    down: np.ndarray,
    frac: np.ndarray,
):
    n_time, n_reach, n_component = grad_out.shape
    grad_storage = np.zeros((n_reach, n_component), dtype=np.float64)
    grad_local = np.zeros_like(grad_out)
    grad_k = np.zeros((n_time, n_reach), dtype=np.float64)
    for t in range(n_time - 1, -1, -1):
        for j in range(n_reach - 1, -1, -1):
            r = order[j]
            d = down[r]
            grad_released = grad_out[t, r].copy()
            if d >= 0:
                grad_released += frac[r] * (grad_storage[d] + grad_upstream[t, d])
            grad_available = (1.0 - k[t, r]) * grad_storage[r] + k[t, r] * grad_released
            grad_k[t, r] = np.sum(available[t, r] * (grad_released - grad_storage[r]))
            grad_storage[r] = grad_available
        grad_local[t] = grad_storage
    return grad_local, grad_k


class _RouteFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local: torch.Tensor,
        raw_strength: torch.Tensor,
        order: torch.Tensor,
        down: torch.Tensor,
        frac: torch.Tensor,
        geometry_numerator: torch.Tensor,
        q_floor: torch.Tensor,
    ):
        local_np = np.ascontiguousarray(local.detach().cpu().numpy())
        order_np = np.ascontiguousarray(order.detach().cpu().numpy(), dtype=np.int64)
        down_np = np.ascontiguousarray(down.detach().cpu().numpy(), dtype=np.int64)
        frac_np = np.ascontiguousarray(frac.detach().cpu().numpy(), dtype=np.float64)
        instantaneous = local_np.copy()
        for r in order_np:
            d = down_np[r]
            if d >= 0:
                instantaneous[:, d, :] += frac_np[r] * instantaneous[:, r, :]
        q_pre = instantaneous.sum(axis=2) / 86400.0
        strength = 1.0 / (1.0 + np.exp(-float(raw_strength.detach().cpu())))
        geom = np.asarray(geometry_numerator.detach().cpu().numpy(), dtype=np.float64)[None, :] / np.maximum(
            q_pre, np.asarray(q_floor.detach().cpu().numpy(), dtype=np.float64)[None, :]
        )
        unbounded_tau = strength * geom
        tau = np.clip(unbounded_tau, 0.02, 30.0)
        k = 1.0 - np.exp(-1.0 / tau)
        out, upstream, available, storage_end = _forward_route(local_np, k, order_np, down_np, frac_np)
        ctx.save_for_backward(raw_strength.detach(), geometry_numerator.detach(), q_floor.detach())
        ctx.order_np = order_np
        ctx.down_np = down_np
        ctx.frac_np = frac_np
        ctx.available = available
        ctx.k = k
        ctx.geom = geom
        ctx.unbounded_tau = unbounded_tau
        ctx.strength = strength
        device = local.device
        return (
            torch.from_numpy(out).to(device=device),
            torch.from_numpy(upstream).to(device=device),
            torch.from_numpy(storage_end).to(device=device),
            torch.from_numpy(tau).to(device=device),
        )

    @staticmethod
    def backward(ctx, grad_out, grad_upstream, grad_storage_end, grad_tau):
        del grad_storage_end, grad_tau
        grad_out_np = np.ascontiguousarray(grad_out.detach().cpu().numpy())
        grad_up_np = np.ascontiguousarray(grad_upstream.detach().cpu().numpy())
        grad_local, grad_k = _backward_route(
            grad_out_np,
            grad_up_np,
            ctx.available,
            ctx.k,
            ctx.order_np,
            ctx.down_np,
            ctx.frac_np,
        )
        tau = np.clip(ctx.unbounded_tau, 0.02, 30.0)
        dk_dtau = -np.exp(-1.0 / tau) / (tau * tau)
        active = (ctx.unbounded_tau > 0.02) & (ctx.unbounded_tau < 30.0)
        dtau_dstrength = np.where(active, ctx.geom, 0.0)
        dstrength_draw = ctx.strength * (1.0 - ctx.strength)
        grad_raw = float(np.sum(grad_k * dk_dtau * dtau_dstrength) * dstrength_draw)
        raw_strength, _, _ = ctx.saved_tensors
        return (
            torch.from_numpy(grad_local).to(device=grad_out.device),
            torch.as_tensor(grad_raw, dtype=raw_strength.dtype, device=raw_strength.device),
            None,
            None,
            None,
            None,
            None,
        )


def route_autograd(
    local_components_m3_day: torch.Tensor,
    raw_strength: torch.Tensor,
    order_index: torch.Tensor,
    downstream_index: torch.Tensor,
    downstream_fraction: torch.Tensor,
    geometry_numerator: torch.Tensor,
    q_floor_m3_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _RouteFunction.apply(
        local_components_m3_day,
        raw_strength,
        order_index,
        downstream_index,
        downstream_fraction,
        geometry_numerator,
        q_floor_m3_s,
    )
