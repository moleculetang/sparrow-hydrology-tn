"""One-pool Active/Fresh agricultural N extension of L0-v2."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from l0_v2_core import TorchL0V2


ROOT = Path(r"E:\SPARROW")
FORCING = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_v2/activelegacy_v2_monthly_forcing_1961_2024.parquet"


class TorchActiveLegacyV2(TorchL0V2):
    def __init__(self, instant_control: bool = False) -> None:
        super().__init__(regionalized=False)
        self.instant_control = bool(instant_control)
        forcing = pd.read_parquet(FORCING).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        if len(forcing) != 768 * 230 or forcing.forcing_id.nunique() != 1:
            raise RuntimeError("ActiveLegacy-v2 forcing grain or ID changed")
        shape = (768, 230)
        self.mineral_external = torch.tensor((
            forcing.fertilizer_kg_n + forcing.cropland_bnf_kg_n
            + forcing.atmospheric_deposition_kg_n + forcing.manure_mineral_kg_n
        ).to_numpy(np.float64).reshape(shape))
        self.organic_manure = torch.tensor(forcing.manure_organic_kg_n.to_numpy(np.float64).reshape(shape))
        self.product_demand = torch.tensor(forcing.potential_product_n_demand_kg_n.to_numpy(np.float64).reshape(shape))
        self.return_demand = torch.tensor(forcing.potential_residue_return_n_demand_kg_n.to_numpy(np.float64).reshape(shape))
        self.removed_demand = torch.tensor(forcing.potential_residue_removed_n_demand_kg_n.to_numpy(np.float64).reshape(shape))
        self.input = torch.tensor((
            forcing.fertilizer_kg_n + forcing.manure_kg_n + forcing.cropland_bnf_kg_n
            + forcing.atmospheric_deposition_kg_n
        ).to_numpy(np.float64).reshape(shape))
        if not self.instant_control:
            self.lower["log_k_active"] = math.log(0.02)
            self.upper["log_k_active"] = math.log(0.20)

    def names(self) -> list[str]:
        names = super().names()
        return names if self.instant_control else names + ["log_k_active"]

    def initial(self, variant: int) -> np.ndarray:
        base = super().initial(variant)
        return base if self.instant_control else np.concatenate([base, [math.log(0.13 if variant == 0 else 0.05)]])

    def local_fluxes(self, values: dict[str, torch.Tensor], diagnostics: bool = False):
        coefficient = self._daily_monthly_coefficients(values)
        q_active = torch.ones(()) if self.instant_control else 1.0 - torch.exp(-torch.exp(values["log_k_active"]) / 12.0)
        active = torch.zeros(230)
        mineral = torch.zeros(230)
        lower = torch.zeros(230)
        fast_rows = []
        slow_rows = []
        active_rows = []
        mineralized_rows = []
        other_rows = []
        mineral_rows = []
        lower_rows = []
        product_rows = []
        removed_rows = []
        returned_rows = []
        if diagnostics:
            cumulative_input = torch.zeros(())
            cumulative_product = torch.zeros(())
            cumulative_removed = torch.zeros(())
            cumulative_fast = torch.zeros(())
            cumulative_slow = torch.zeros(())
            cumulative_other = torch.zeros(())
        for index in range(768):
            mineralized = active * q_active
            active_after = active - mineralized
            mineral_pre = mineral + self.mineral_external[index] + mineralized
            total_demand = self.product_demand[index] + self.return_demand[index] + self.removed_demand[index]
            available = mineral_pre
            crop = torch.minimum(available, total_demand)
            demand_scale = torch.where(total_demand > 0.0, crop / torch.clamp(total_demand, min=1.0e-12), torch.zeros_like(crop))
            product = self.product_demand[index] * demand_scale
            returned = self.return_demand[index] * demand_scale
            removed = self.removed_demand[index] * demand_scale
            if torch.max(torch.abs(crop - product - returned - removed)).detach() > 1.0e-8:
                raise RuntimeError("Crop product/residue partition does not close")
            after_crop = mineral_pre - crop
            fast = after_crop * coefficient["fast"][index]
            slow = lower * (1.0 - coefficient["lower_carry"][index]) + after_crop * coefficient["slow_from_upper"][index]
            lower = lower * coefficient["lower_carry"][index] + after_crop * coefficient["lower_from_upper"][index]
            other = after_crop * coefficient["other"][index]
            mineral = after_crop * coefficient["upper_carry"][index]
            active = active_after + self.organic_manure[index] + returned
            if diagnostics:
                cumulative_input = cumulative_input + torch.sum(self.input[index])
                cumulative_product = cumulative_product + torch.sum(product)
                cumulative_removed = cumulative_removed + torch.sum(removed)
                cumulative_fast = cumulative_fast + torch.sum(fast)
                cumulative_slow = cumulative_slow + torch.sum(slow)
                cumulative_other = cumulative_other + torch.sum(other)
            if index >= self.formal_start:
                fast_rows.append(fast)
                slow_rows.append(slow)
                if diagnostics:
                    active_rows.append(active)
                    mineralized_rows.append(mineralized)
                    other_rows.append(other)
                    mineral_rows.append(mineral)
                    lower_rows.append(lower)
                    product_rows.append(product)
                    removed_rows.append(removed)
                    returned_rows.append(returned)
        result = (torch.stack(fast_rows), torch.stack(slow_rows))
        if not diagnostics:
            return result
        outputs_state = (
            cumulative_product + cumulative_removed + cumulative_fast + cumulative_slow
            + cumulative_other + torch.sum(active) + torch.sum(mineral) + torch.sum(lower)
        )
        return result + ({
            "active_end": torch.stack(active_rows), "mineralized": torch.stack(mineralized_rows),
            "other_loss": torch.stack(other_rows), "mineral_end": torch.stack(mineral_rows),
            "lower_end": torch.stack(lower_rows), "product": torch.stack(product_rows),
            "removed_residue": torch.stack(removed_rows), "returned_residue": torch.stack(returned_rows),
            "coefficients": coefficient, "q_active_month": q_active,
            "cumulative_input_all": cumulative_input,
            "cumulative_outputs_and_terminal_state_all": outputs_state,
            "land_mass_closure_relative": torch.abs(cumulative_input - outputs_state) / torch.clamp(cumulative_input, min=1.0),
        },)

    def structural_diagnostics(self, physical: torch.Tensor) -> dict[str, float]:
        values = dict(zip(self.names(), physical))
        fast, slow, state = self.local_fluxes(values, diagnostics=True)
        local = fast + slow
        _, outlet, removed = self._route(local, values["v_f"])
        terminal = [index for index in self.order_idx if index not in self.down_idx]
        channel_input = torch.sum(local)
        channel_removed = torch.sum(removed)
        terminal_output = torch.sum(outlet[:, terminal])
        return {
            "operator_closure_max_abs": float(torch.max(torch.abs(state["coefficients"]["closure"] - 1.0)).detach()),
            "initial_active_1961_kg_n": 0.0,
            "initial_mineral_1961_kg_n": 0.0,
            "initial_lower_1961_kg_n": 0.0,
            "land_mass_closure_relative": float(state["land_mass_closure_relative"].detach()),
            "channel_closure_relative": float((
                torch.abs(channel_input - channel_removed - terminal_output) / torch.clamp(channel_input, min=1.0)
            ).detach()),
            "terminal_active_2024_kg_n": float(torch.sum(state["active_end"][-1]).detach()),
            "terminal_mineral_2024_kg_n": float(torch.sum(state["mineral_end"][-1]).detach()),
            "terminal_lower_2024_kg_n": float(torch.sum(state["lower_end"][-1]).detach()),
            "formal_active_mineralization_kg_n": float(torch.sum(state["mineralized"]).detach()),
            "monthly_active_release_probability": float(state["q_active_month"].detach()),
            "k_active_year_minus_1": float("inf") if self.instant_control else float(torch.exp(values["log_k_active"]).detach()),
        }
