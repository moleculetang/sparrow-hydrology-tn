"""TorchScript recurrence for component-preserving channel stores."""

from __future__ import annotations

from typing import List

import torch


@torch.jit.script
def route_kernel(
    local_components_m3_day: torch.Tensor,
    release_fraction: torch.Tensor,
    order_index: torch.Tensor,
    downstream_index: torch.Tensor,
    downstream_fraction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_time = local_components_m3_day.size(0)
    n_reach = local_components_m3_day.size(1)
    n_component = local_components_m3_day.size(2)
    storage: List[torch.Tensor] = []
    for _ in range(n_reach):
        storage.append(torch.zeros((n_component,), dtype=local_components_m3_day.dtype, device=local_components_m3_day.device))
    out_rows: List[torch.Tensor] = []
    storage_rows: List[torch.Tensor] = []
    for time_index in range(n_time):
        for position in range(n_reach):
            storage[position] = storage[position] + local_components_m3_day[time_index, position]
        reach_out: List[torch.Tensor] = []
        for _ in range(n_reach):
            reach_out.append(torch.zeros((n_component,), dtype=local_components_m3_day.dtype, device=local_components_m3_day.device))
        for order_position in range(n_reach):
            position = int(order_index[order_position])
            released = release_fraction[time_index, position] * storage[position]
            storage[position] = storage[position] - released
            reach_out[position] = released
            target = int(downstream_index[position])
            if target >= 0:
                storage[target] = storage[target] + downstream_fraction[position] * released
        out_rows.append(torch.stack(reach_out))
        storage_rows.append(torch.stack(storage))
    return torch.stack(out_rows), torch.stack(storage_rows)
