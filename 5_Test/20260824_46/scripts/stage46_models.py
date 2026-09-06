"""Six preregistered low-capacity extensions of the Stage45 canonical L0-v2 parent."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
STAGE41 = ROOT / "5_Test/20260824_41/scripts"
sys.path.insert(0, str(STAGE41))
import l0_v2_core as l0  # noqa: E402


CANDIDATES = [
    "REG3_CONTACT", "REG3_OBSERVATION", "DYN_HYDRO_DELIVERY",
    "SOURCE2", "MINERAL_LIFETIME", "AQ_SIZE",
]
REG_FEATURE_INDICES = [2, 3, 4]


def hydrologic_state_score() -> np.ndarray:
    frame = pd.read_parquet(l0.h39.LONG_OUTPUT).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    frame["routed_fast_fraction"] = np.divide(
        frame.routed_fast_flow_mean_m3_s.to_numpy(float),
        frame.routed_total_flow_mean_m3_s.to_numpy(float),
        out=np.zeros(len(frame), dtype=float),
        where=frame.routed_total_flow_mean_m3_s.to_numpy(float) > l0.EPS,
    )
    features = {
        "upper": "upper_storage_start_mm",
        "lower": "lower_storage_start_mm",
        "excess": "upper_excess_input_water_mm_month",
        "fast_fraction": "routed_fast_fraction",
    }
    z_columns = []
    for short, column in features.items():
        if short == "fast_fraction":
            x = np.clip(frame[column].to_numpy(float), 1.0e-6, 1.0 - 1.0e-6)
            raw = np.log(x / (1.0 - x))
        else:
            raw = np.log1p(frame[column].to_numpy(float))
        raw_column = f"raw_{short}"
        frame[raw_column] = raw
        climatology = (
            frame.loc[frame.year.between(2010, 2020)]
            .groupby(["reach_id", "month"])[raw_column].mean().rename("climatology")
        )
        frame = frame.merge(climatology, on=["reach_id", "month"], validate="many_to_one")
        anomaly_column = f"anomaly_{short}"
        frame[anomaly_column] = frame[raw_column] - frame.climatology
        sd = (
            frame.loc[frame.year.between(2010, 2020)]
            .groupby("reach_id")[anomaly_column].std(ddof=0).replace(0.0, 1.0).rename("sd")
        )
        frame = frame.merge(sd, on="reach_id", validate="many_to_one")
        z_column = f"z_{short}"
        frame[z_column] = frame[anomaly_column] / frame.sd
        z_columns.append(z_column)
        frame = frame.drop(columns=["climatology", "sd"])
    score = np.tanh(frame[z_columns].mean(axis=1).to_numpy(float).reshape(768, 230) / 2.0)
    if not np.isfinite(score).all() or np.max(np.abs(score)) > 1.0 + 1.0e-12:
        raise RuntimeError("Invalid frozen hydrologic-state score")
    return score


class Stage46Model(l0.TorchL0V2):
    def __init__(self, candidate: str) -> None:
        if candidate not in CANDIDATES:
            raise ValueError(candidate)
        super().__init__(regionalized=False)
        self.candidate = candidate
        self.reg_features = self.features[:, REG_FEATURE_INDICES]
        if candidate == "DYN_HYDRO_DELIVERY":
            self.state_score = torch.from_numpy(hydrologic_state_score())
        if candidate == "SOURCE2":
            source = (
                pd.read_parquet(l0.SOURCE).loc[lambda x: x.calendar_scenario.eq("CENTRAL")]
                .sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
            )
            agricultural = source[["fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n"]].sum(axis=1)
            atmospheric = source.atmospheric_deposition_kg_n
            self.source_agricultural = torch.tensor(agricultural.to_numpy(np.float64).reshape(768, 230))
            self.source_atmospheric = torch.tensor(atmospheric.to_numpy(np.float64).reshape(768, 230))
        if candidate == "AQ_SIZE":
            monthly = pd.read_parquet(l0.h39.LONG_OUTPUT)
            median_q = (
                monthly.loc[monthly.year.between(2010, 2020)]
                .groupby("reach_id").routed_total_flow_mean_m3_s.median().reindex(range(1, 231)).to_numpy(float)
            )
            logq = np.log1p(np.maximum(median_q, 0.0))
            self.aq_size_score = torch.tensor((logq - logq.mean()) / max(logq.std(ddof=0), 1.0e-12))
        extras = self.extra_names()
        bounds = {
            "gamma_reg_1": (-0.5, 0.5), "gamma_reg_2": (-0.5, 0.5), "gamma_reg_3": (-0.5, 0.5),
            "beta_D": (-1.0, 1.0), "delta_source": (-1.0, 1.0),
            "log_tau_mineral_days": (math.log(182.625), math.log(3652.5)),
            "beta_aq_size": (-0.75, 0.75),
        }
        for name in extras:
            self.lower[name], self.upper[name] = bounds[name]

    def extra_names(self) -> list[str]:
        if self.candidate in {"REG3_CONTACT", "REG3_OBSERVATION"}:
            return ["gamma_reg_1", "gamma_reg_2", "gamma_reg_3"]
        return {
            "DYN_HYDRO_DELIVERY": ["beta_D"],
            "SOURCE2": ["delta_source"],
            "MINERAL_LIFETIME": ["log_tau_mineral_days"],
            "AQ_SIZE": ["beta_aq_size"],
        }[self.candidate]

    def names(self) -> list[str]:
        return ["log_alpha_contact", "beta_contact", "v_f", "delta_path", "beta_low", "beta_high", "log_sigma"] + self.extra_names()

    def initial(self, variant: int) -> np.ndarray:
        base = super().initial(variant)
        extra = np.zeros(len(self.extra_names()), dtype=float)
        if self.candidate == "MINERAL_LIFETIME":
            extra[0] = math.log(365.25)
        return np.concatenate([base, extra])

    def _daily_monthly_coefficients(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_alpha = values["log_alpha_contact"]
        if self.candidate == "REG3_CONTACT":
            gamma = torch.stack([values[f"gamma_reg_{index}"] for index in range(1, 4)])
            log_alpha = log_alpha + self.reg_features @ gamma
        if self.candidate == "DYN_HYDRO_DELIVERY":
            log_alpha = log_alpha + values["beta_D"] * self.state_score
        if log_alpha.ndim == 0:
            log_alpha_operator = log_alpha.expand(230)[None, None, :]
            log_alpha_report = log_alpha.expand(230)
        elif log_alpha.ndim == 1:
            log_alpha_operator = log_alpha[None, None, :]
            log_alpha_report = log_alpha
        else:
            log_alpha_operator = log_alpha[:, None, :]
            log_alpha_report = log_alpha
        alpha = torch.exp(log_alpha_operator)
        probability = 1.0 - torch.exp(-torch.clamp(alpha * torch.pow(self.contact_ratio, values["beta_contact"]), max=700.0))
        probability = torch.where(self.valid_day, probability, torch.zeros_like(probability))
        tau = torch.exp(values["log_tau_mineral_days"]) if self.candidate == "MINERAL_LIFETIME" else torch.tensor(365.25)
        q_loss = 1.0 - torch.exp(-1.0 / tau)
        loss_probability = torch.where(self.valid_day, torch.ones_like(probability) * q_loss, torch.zeros_like(probability))
        retention = (1.0 - probability) * (1.0 - loss_probability)
        before = torch.cumprod(torch.cat([torch.ones_like(retention[:, :1]), retention[:, :-1]], dim=1), dim=1)
        mobilized_day = before * probability
        fast = torch.sum(mobilized_day * self.fast_fraction, dim=1)
        percolated_day = mobilized_day * (1.0 - self.fast_fraction)
        other = torch.sum(before * (1.0 - probability) * loss_probability, dim=1)
        upper_carry = torch.prod(retention, dim=1)
        lower_daily_carry = 1.0 - self.lower_release
        tail_carry = torch.flip(torch.cumprod(torch.flip(lower_daily_carry, dims=[1]), dim=1), dims=[1])
        slow_from_upper = torch.sum(percolated_day * (1.0 - tail_carry), dim=1)
        lower_from_upper = torch.sum(percolated_day * tail_carry, dim=1)
        lower_carry = torch.prod(lower_daily_carry, dim=1)
        closure = fast + slow_from_upper + lower_from_upper + other + upper_carry
        return {
            "fast": fast, "slow_from_upper": slow_from_upper, "lower_from_upper": lower_from_upper,
            "other": other, "upper_carry": upper_carry, "lower_carry": lower_carry,
            "probability": probability, "closure": closure, "log_alpha_reach": log_alpha_report,
        }

    def local_fluxes(self, values: dict[str, torch.Tensor], diagnostics: bool = False):
        if self.candidate != "SOURCE2":
            return super().local_fluxes(values, diagnostics=diagnostics)
        effective_input = (
            torch.exp(values["delta_source"]) * self.source_agricultural
            + torch.exp(-values["delta_source"]) * self.source_atmospheric
        )
        coefficient = self._daily_monthly_coefficients(values)
        mineral = torch.zeros(230)
        lower = torch.zeros(230)
        fast_rows, slow_rows, other_rows, mineral_rows, lower_rows, crop_rows = [], [], [], [], [], []
        cumulative = {name: torch.zeros(()) for name in ["input", "crop", "fast", "slow", "other"]}
        for index in range(768):
            pre = mineral + effective_input[index]
            crop = torch.minimum(pre, self.crop[index])
            after_crop = pre - crop
            fast = after_crop * coefficient["fast"][index]
            slow = lower * (1.0 - coefficient["lower_carry"][index]) + after_crop * coefficient["slow_from_upper"][index]
            lower = lower * coefficient["lower_carry"][index] + after_crop * coefficient["lower_from_upper"][index]
            other = after_crop * coefficient["other"][index]
            mineral = after_crop * coefficient["upper_carry"][index]
            if diagnostics:
                for name, value in {"input": effective_input[index], "crop": crop, "fast": fast, "slow": slow, "other": other}.items():
                    cumulative[name] = cumulative[name] + torch.sum(value)
            if index >= self.formal_start:
                fast_rows.append(fast); slow_rows.append(slow)
                if diagnostics:
                    other_rows.append(other); mineral_rows.append(mineral); lower_rows.append(lower); crop_rows.append(crop)
        result = (torch.stack(fast_rows), torch.stack(slow_rows))
        if not diagnostics:
            return result
        return result + ({
            "other_loss": torch.stack(other_rows), "mineral_end": torch.stack(mineral_rows),
            "lower_end": torch.stack(lower_rows), "crop": torch.stack(crop_rows), "coefficients": coefficient,
            "cumulative_input_all": cumulative["input"], "cumulative_crop_all": cumulative["crop"],
            "cumulative_fast_all": cumulative["fast"], "cumulative_slow_all": cumulative["slow"],
            "cumulative_other_all": cumulative["other"], "terminal_mineral_all": torch.sum(mineral),
            "terminal_lower_all": torch.sum(lower),
        },)

    def _reach_vf(self, v_f: torch.Tensor, values: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if self.candidate != "AQ_SIZE":
            return v_f.expand(230)
        if values is None:
            raise RuntimeError("AQ_SIZE requires parameter dictionary")
        return v_f * torch.exp(values["beta_aq_size"] * self.aq_size_score)

    def _route_with_values(self, local: torch.Tensor, values: dict[str, torch.Tensor]):
        rate = self._reach_vf(values["v_f"], values)
        inlet = [torch.zeros(self.shape[0]) for _ in range(230)]
        outlets = [torch.zeros(self.shape[0]) for _ in range(230)]
        removed = [torch.zeros(self.shape[0]) for _ in range(230)]
        for index in self.order_idx:
            attenuation_up = torch.exp(-rate[index] * self.h[:, index])
            attenuation_local = torch.exp(-rate[index] * self.h[:, index] / 2.0)
            outlet = inlet[index] * attenuation_up + local[:, index] * attenuation_local
            outlets[index] = outlet
            removed[index] = inlet[index] + local[:, index] - outlet
            if index in self.down_idx:
                inlet[self.down_idx[index]] = inlet[self.down_idx[index]] + outlet
        return torch.stack(inlet, dim=1), torch.stack(outlets, dim=1), torch.stack(removed, dim=1)

    def _station_with_values(self, obs: pd.DataFrame, local: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        inlet, _, _ = self._route_with_values(local, values)
        tidx, ridx, fraction = self.obs_indices(obs)
        rate = self._reach_vf(values["v_f"], values)[ridx]
        exposure = self.h[tidx, ridx]
        load = inlet[tidx, ridx] * torch.exp(-rate * exposure * fraction)
        load = load + fraction * local[tidx, ridx] * torch.exp(-rate * exposure * fraction / 2.0)
        return torch.log1p(1000.0 * load / torch.clamp(self.station_water(obs), min=l0.EPS))

    def evaluate(self, obs: pd.DataFrame, physical: torch.Tensor, train_start: int, train_end: int):
        values = dict(zip(self.names(), physical))
        fast, slow = self.local_fluxes(values)
        raw = self._station_with_values(obs, fast + slow, values)
        calibrated = torch.exp(values["delta_path"]) * fast + torch.exp(-values["delta_path"]) * slow
        population = self._station_with_values(obs, calibrated, values)
        low, high = self.q_features(obs, train_start, train_end)
        population = population + values["beta_low"] * low + values["beta_high"] * high
        if self.candidate == "REG3_OBSERVATION":
            gamma = torch.stack([values[f"gamma_reg_{index}"] for index in range(1, 4)])
            _, ridx, _ = self.obs_indices(obs)
            population = population + (self.reg_features @ gamma)[ridx]
        return raw, population

    def structural_diagnostics(self, physical: torch.Tensor) -> dict[str, float]:
        values = dict(zip(self.names(), physical))
        fast, slow, state = self.local_fluxes(values, diagnostics=True)
        coefficient = state["coefficients"]
        local = fast + slow
        _, outlet, removed = self._route_with_values(local, values)
        terminal = [index for index in self.order_idx if index not in self.down_idx]
        land_input = state["cumulative_input_all"]
        land_outputs = (
            state["cumulative_crop_all"] + state["cumulative_fast_all"] + state["cumulative_slow_all"]
            + state["cumulative_other_all"] + state["terminal_mineral_all"] + state["terminal_lower_all"]
        )
        total_local = torch.sum(local); total_removed = torch.sum(removed); total_terminal = torch.sum(outlet[:, terminal])
        modifier = torch.zeros(1)
        if self.candidate in {"REG3_CONTACT", "DYN_HYDRO_DELIVERY"}:
            modifier = coefficient["log_alpha_reach"] - values["log_alpha_contact"]
        elif self.candidate == "REG3_OBSERVATION":
            gamma = torch.stack([values[f"gamma_reg_{index}"] for index in range(1, 4)])
            modifier = self.reg_features @ gamma
        elif self.candidate == "AQ_SIZE":
            modifier = values["beta_aq_size"] * self.aq_size_score
        return {
            "operator_closure_max_abs": float(torch.max(torch.abs(coefficient["closure"] - 1.0)).detach()),
            "daily_probability_min": float(torch.min(coefficient["probability"]).detach()),
            "daily_probability_max": float(torch.max(coefficient["probability"]).detach()),
            "zero_contact_probability_max": float(torch.max(torch.where(self.contact_ratio <= l0.EPS, coefficient["probability"], torch.zeros_like(coefficient["probability"]))).detach()),
            "initial_mineral_1961_kg_n": 0.0, "initial_lower_1961_kg_n": 0.0,
            "land_input_all_kg_n": float(land_input.detach()),
            "land_outputs_and_terminal_state_all_kg_n": float(land_outputs.detach()),
            "land_mass_closure_relative": float((torch.abs(land_input - land_outputs) / torch.clamp(land_input, min=1.0)).detach()),
            "channel_input_kg_n": float(total_local.detach()), "channel_removed_kg_n": float(total_removed.detach()),
            "terminal_output_kg_n": float(total_terminal.detach()),
            "channel_closure_relative": float((torch.abs(total_local - total_removed - total_terminal) / torch.clamp(total_local, min=1.0)).detach()),
            "maximum_abs_registered_modifier": float(torch.max(torch.abs(modifier)).detach()),
        }

    def extra_prior(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.candidate in {"REG3_CONTACT", "REG3_OBSERVATION"}:
            return 0.5 * sum((values[f"gamma_reg_{index}"] / 0.25) ** 2 for index in range(1, 4))
        if self.candidate == "DYN_HYDRO_DELIVERY":
            return 0.5 * (values["beta_D"] / 0.35) ** 2
        if self.candidate == "SOURCE2":
            return 0.5 * (values["delta_source"] / 0.5) ** 2
        if self.candidate == "MINERAL_LIFETIME":
            return 0.5 * ((values["log_tau_mineral_days"] - math.log(365.25)) / math.log(2.0)) ** 2
        if self.candidate == "AQ_SIZE":
            return 0.5 * (values["beta_aq_size"] / 0.35) ** 2
        raise ValueError(self.candidate)
