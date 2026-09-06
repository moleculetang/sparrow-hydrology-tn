"""Registered low-capacity static attribute-to-HBV-parameter network."""

from __future__ import annotations

import torch
from torch import nn


class StaticParameterNetwork(nn.Module):
    def __init__(self, input_features: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = nn.Linear(input_features, 16, dtype=torch.float64)
        self.output = nn.Linear(16, 8, dtype=torch.float64)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        # Required nesting: every seed and feature family starts exactly at the parent.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def offsets(self, attributes: torch.Tensor) -> torch.Tensor:
        return 1.5 * torch.tanh(self.output(torch.tanh(self.hidden(attributes))))

    def raw_parameter_map(self, attributes: torch.Tensor, parent_raw: torch.Tensor) -> torch.Tensor:
        return parent_raw.unsqueeze(0) + self.offsets(attributes)
