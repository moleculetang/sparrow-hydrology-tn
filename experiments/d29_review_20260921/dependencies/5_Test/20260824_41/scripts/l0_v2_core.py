"""Differentiable L0-v2 TN core driven by frozen daily canonical hydrology."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
STAGE28 = ROOT / "5_Test/20260824_28/scripts"
STAGE39 = ROOT / "5_Test/20260824_39/scripts"
SOURCE = ROOT / "5_Test/20260824_12/outputs/monthly_source_forcing_1961_2024.parquet"
STATIC = ROOT / "5_Test/20260824_12/outputs/canonical_tn_reach_static_registry.parquet"
COVARIATES = ROOT / "5_Test/20260824_15/outputs/static_delivery_covariates_standardized.parquet"
CANONICAL_DAILY = ROOT / "5_Test/20260828_9/outputs/canonical_reach_daily_2006_2024.parquet"

sys.path.insert(0, str(STAGE28))
sys.path.insert(0, str(STAGE39))
import run_stage28 as s28  # noqa: E402
import build_canonical_daily_to_monthly_tn_interface as h39  # noqa: E402


EPS = 1.0e-12
REACH_IDS = np.arange(1, 231, dtype=np.int64)
FEATURE_COLUMNS = [
    "z_cropland_fraction_clcd_multiyear",
    "z_agricultural_n_intensity_kg_n_km2_year",
    "z_soc_0_30cm_depth_weighted",
    "z_clay_0_20cm",
    "z_dem_slope",
]


def _splice_daily_hydrology() -> dict[str, object]:
    frozen = h39.load_frozen_hydrology()
    memory: dict[str, float] = {}
    historical, _ = h39.simulate_historical(frozen, memory)
    canonical_initial, _ = h39.canonical_initial_state(frozen)
    canonical_frame = pd.read_parquet(CANONICAL_DAILY)
    canonical = h39.reshape_canonical_daily(canonical_frame, frozen, canonical_initial)

    def add_starts(arrays: dict[str, object]) -> dict[str, object]:
        lower_end = np.asarray(arrays["lower_end"], dtype=np.float64)
        upper_end = np.asarray(arrays["upper_end"], dtype=np.float64)
        arrays["lower_start"] = np.concatenate([
            np.asarray(arrays["initial_lower"], dtype=np.float64)[None, :], lower_end[:-1]
        ], axis=0)
        arrays["upper_start"] = np.concatenate([
            np.asarray(arrays["initial_upper"], dtype=np.float64)[None, :], upper_end[:-1]
        ], axis=0)
        return arrays

    historical = add_starts(historical)
    canonical = add_starts(canonical)
    h_dates = pd.DatetimeIndex(historical["dates"])
    c_dates = pd.DatetimeIndex(canonical["dates"])
    hi = np.asarray(h_dates.year <= 2009)
    ci = np.asarray(c_dates.year >= 2010)
    dates = h_dates[hi].append(c_dates[ci])
    expected = pd.date_range("1961-01-01", "2024-12-31", freq="D")
    if not dates.equals(expected):
        raise RuntimeError("Daily hydrology splice does not cover 1961-2024 exactly")
    keys = ["fast", "percolation", "slow", "upper_start", "upper_end", "lower_start", "lower_end"]
    out: dict[str, object] = {"dates": dates}
    for key in keys:
        value = np.concatenate([
            np.asarray(historical[key], dtype=np.float64)[hi],
            np.asarray(canonical[key], dtype=np.float64)[ci],
        ], axis=0)
        if value.shape != (len(expected), 230) or not np.isfinite(value).all() or np.min(value) < -1.0e-10:
            raise RuntimeError(f"Invalid spliced daily hydrology field {key}")
        out[key] = np.maximum(value, 0.0)
    return out


def _pad_daily(arrays: dict[str, object]) -> dict[str, np.ndarray]:
    dates = pd.DatetimeIndex(arrays["dates"])
    periods = dates.to_period("M")
    unique = periods.unique()
    if len(unique) != 768:
        raise RuntimeError("Expected 768 months in 1961-2024")
    shape = (len(unique), 31, 230)
    padded = {name: np.zeros(shape, dtype=np.float64) for name in (
        "fast", "percolation", "slow", "upper_start", "upper_end", "lower_start", "lower_end"
    )}
    valid = np.zeros((len(unique), 31, 1), dtype=bool)
    for month_index, period in enumerate(unique):
        ii = np.flatnonzero(periods == period)
        valid[month_index, : len(ii), 0] = True
        for name in padded:
            padded[name][month_index, : len(ii)] = np.asarray(arrays[name])[ii]
    padded["valid"] = valid
    return padded


class TorchL0V2:
    """L0-v2 with a fixed one-year mineral e-fold lifetime."""

    def __init__(self, regionalized: bool = False) -> None:
        source = (
            pd.read_parquet(SOURCE)
            .loc[lambda x: x.calendar_scenario.eq("CENTRAL")]
            .sort_values(["year", "month", "reach_id"])
            .reset_index(drop=True)
        )
        if len(source) != 768 * 230:
            raise RuntimeError("Monthly source forcing grain changed")
        self.month_keys = [
            (int(year), int(month))
            for year, month in source[["year", "month"]].drop_duplicates().itertuples(index=False)
        ]
        self.tlookup_all = {key: index for index, key in enumerate(self.month_keys)}
        self.formal_start = self.tlookup_all[(2010, 1)]
        self.formal_keys = self.month_keys[self.formal_start :]
        self.tlookup = {key: index for index, key in enumerate(self.formal_keys)}
        self.rlookup = {reach: reach - 1 for reach in range(1, 231)}
        self.shape = (len(self.formal_keys), 230)
        self.regionalized = bool(regionalized)

        source_total = source[[
            "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n"
        ]].sum(axis=1).to_numpy(np.float64).reshape(768, 230)
        crop = source.crop_demand_kg_n.to_numpy(np.float64).reshape(768, 230)
        self.input = torch.from_numpy(source_total.copy())
        self.crop = torch.from_numpy(crop.copy())
        self.days_in_month = source.days_in_month.to_numpy(np.float64).reshape(768, 230)[:, 0]

        padded = _pad_daily(_splice_daily_hydrology())
        valid = padded["valid"]
        fast = padded["fast"]
        percolation = padded["percolation"]
        slow = padded["slow"]
        upper_available = fast + percolation + padded["upper_end"]
        contact_water = fast + percolation
        contact_ratio = np.divide(
            contact_water, upper_available, out=np.zeros_like(contact_water), where=upper_available > EPS
        )
        contact_ratio[(contact_water <= EPS) | ~valid] = 0.0
        fast_fraction = np.divide(
            fast, contact_water, out=np.zeros_like(fast), where=contact_water > EPS
        )
        lower_available = padded["lower_start"] + percolation
        lower_release = np.divide(
            slow, lower_available, out=np.zeros_like(slow), where=lower_available > EPS
        )
        lower_release[(slow <= EPS) | ~valid] = 0.0
        if np.max(contact_ratio) > 1.0 + 1.0e-10 or np.max(lower_release) > 1.0 + 1.0e-10:
            raise RuntimeError("Daily hydraulic probability exceeds one")
        self.contact_ratio = torch.from_numpy(contact_ratio.copy())
        self.fast_fraction = torch.from_numpy(fast_fraction.copy())
        self.lower_release = torch.from_numpy(lower_release.copy())
        self.valid_day = torch.from_numpy(valid.copy())

        static = pd.read_parquet(STATIC).sort_values("reach_id")
        area = static.catchment_area_km2.to_numpy(np.float64)
        depth = static.bankfull_depth_m.to_numpy(np.float64)
        monthly_h = pd.read_parquet(h39.LONG_OUTPUT).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        if len(monthly_h) != 768 * 230:
            raise RuntimeError("Monthly hydraulic interface grain changed")
        formal_h = monthly_h.loc[monthly_h.year.ge(2010)].reset_index(drop=True)
        local_volume = (
            (formal_h.local_fast_water_mm_month + formal_h.local_slow_water_mm_month).to_numpy(np.float64)
            * np.tile(area * 1000.0, self.shape[0])
        ).reshape(self.shape)
        self.local_water = torch.from_numpy(local_volume.copy())
        exposure = (
            formal_h.channel_bankfull_travel_time_central_day.to_numpy(np.float64)
            / np.maximum(np.tile(depth, self.shape[0]), EPS)
        ).reshape(self.shape)
        self.h = torch.from_numpy(exposure.copy())
        self.seconds = torch.from_numpy((self.days_in_month[self.formal_start :] * 86400.0).copy())

        order, downstream = s28.topology_operators()
        self.order_idx = [reach - 1 for reach in order]
        self.down_idx = {reach - 1: target - 1 for reach, target in downstream.items()}
        water_inlet = np.zeros(self.shape, dtype=np.float64)
        for index in self.order_idx:
            outlet = water_inlet[:, index] + local_volume[:, index]
            if index in self.down_idx:
                water_inlet[:, self.down_idx[index]] += outlet
        self.water_inlet = torch.from_numpy(water_inlet.copy())

        covariates = pd.read_parquet(COVARIATES).sort_values("reach_id")
        self.features = torch.from_numpy(covariates[FEATURE_COLUMNS].to_numpy(np.float64).copy())
        self.lower = {
            "log_alpha_contact": -9.21, "beta_contact": 0.25, "v_f": 0.0,
            "delta_path": -2.0, "beta_low": -1.0, "beta_high": -1.0, "log_sigma": -4.0,
        }
        self.upper = {
            "log_alpha_contact": 4.605170186, "beta_contact": 2.0, "v_f": 0.5,
            "delta_path": 2.0, "beta_low": 1.0, "beta_high": 1.0, "log_sigma": 1.0,
        }
        for index in range(5):
            self.lower[f"gamma_{index + 1}"] = -0.5
            self.upper[f"gamma_{index + 1}"] = 0.5
        self._obs_index_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._q_feature_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def names(self) -> list[str]:
        base = ["log_alpha_contact", "beta_contact", "v_f", "delta_path", "beta_low", "beta_high", "log_sigma"]
        return base + ([f"gamma_{index + 1}" for index in range(5)] if self.regionalized else [])

    def to_physical(self, raw: torch.Tensor) -> torch.Tensor:
        names = self.names()
        low = torch.tensor([self.lower[name] for name in names])
        high = torch.tensor([self.upper[name] for name in names])
        return low + (high - low) * torch.sigmoid(raw)

    def to_raw(self, physical: np.ndarray) -> torch.Tensor:
        names = self.names()
        low = np.asarray([self.lower[name] for name in names])
        high = np.asarray([self.upper[name] for name in names])
        fraction = np.clip((physical - low) / (high - low), 1.0e-8, 1.0 - 1.0e-8)
        return torch.tensor(np.log(fraction / (1.0 - fraction)))

    def initial(self, variant: int) -> np.ndarray:
        base = np.array((
            [-1.0, 1.0, 0.12, 0.0, 0.1, 0.2, math.log(0.35)],
            [-4.0, 0.7, 0.22, 0.0, 0.0, 0.0, math.log(0.35)],
        )[variant])
        if self.regionalized:
            base = np.concatenate([base, np.zeros(5)])
        return base

    def _daily_monthly_coefficients(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_alpha = values["log_alpha_contact"]
        if self.regionalized:
            gamma = torch.stack([values[f"gamma_{index + 1}"] for index in range(5)])
            log_alpha = log_alpha + self.features @ gamma
        log_alpha_reach = log_alpha if log_alpha.ndim else log_alpha.expand(230)
        alpha = torch.exp(log_alpha_reach)[None, None, :]
        beta = values["beta_contact"]
        probability = 1.0 - torch.exp(-torch.clamp(alpha * torch.pow(self.contact_ratio, beta), max=700.0))
        probability = torch.where(self.valid_day, probability, torch.zeros_like(probability))
        q_loss = 1.0 - math.exp(-1.0 / 365.25)
        loss_probability = torch.where(
            self.valid_day, torch.full_like(probability, q_loss), torch.zeros_like(probability)
        )
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
            "fast": fast, "slow_from_upper": slow_from_upper,
            "lower_from_upper": lower_from_upper, "other": other,
            "upper_carry": upper_carry, "lower_carry": lower_carry,
            "probability": probability, "closure": closure,
            "log_alpha_reach": log_alpha_reach,
        }

    def local_fluxes(self, values: dict[str, torch.Tensor], diagnostics: bool = False):
        coefficient = self._daily_monthly_coefficients(values)
        mineral = torch.zeros(230)
        lower = torch.zeros(230)
        fast_rows = []
        slow_rows = []
        other_rows = []
        mineral_rows = []
        lower_rows = []
        crop_rows = []
        if diagnostics:
            cumulative_input = torch.zeros(())
            cumulative_crop = torch.zeros(())
            cumulative_fast = torch.zeros(())
            cumulative_slow = torch.zeros(())
            cumulative_other = torch.zeros(())
        for index in range(768):
            pre = mineral + self.input[index]
            crop = torch.minimum(pre, self.crop[index])
            after_crop = pre - crop
            fast = after_crop * coefficient["fast"][index]
            slow = lower * (1.0 - coefficient["lower_carry"][index]) + after_crop * coefficient["slow_from_upper"][index]
            lower = lower * coefficient["lower_carry"][index] + after_crop * coefficient["lower_from_upper"][index]
            other = after_crop * coefficient["other"][index]
            mineral = after_crop * coefficient["upper_carry"][index]
            if diagnostics:
                cumulative_input = cumulative_input + torch.sum(self.input[index])
                cumulative_crop = cumulative_crop + torch.sum(crop)
                cumulative_fast = cumulative_fast + torch.sum(fast)
                cumulative_slow = cumulative_slow + torch.sum(slow)
                cumulative_other = cumulative_other + torch.sum(other)
            if index >= self.formal_start:
                fast_rows.append(fast)
                slow_rows.append(slow)
                if diagnostics:
                    other_rows.append(other)
                    mineral_rows.append(mineral)
                    lower_rows.append(lower)
                    crop_rows.append(crop)
        result = (torch.stack(fast_rows), torch.stack(slow_rows))
        if not diagnostics:
            return result
        return result + ({
            "other_loss": torch.stack(other_rows), "mineral_end": torch.stack(mineral_rows),
            "lower_end": torch.stack(lower_rows), "crop": torch.stack(crop_rows),
            "coefficients": coefficient,
            "cumulative_input_all": cumulative_input,
            "cumulative_crop_all": cumulative_crop,
            "cumulative_fast_all": cumulative_fast,
            "cumulative_slow_all": cumulative_slow,
            "cumulative_other_all": cumulative_other,
            "terminal_mineral_all": torch.sum(mineral),
            "terminal_lower_all": torch.sum(lower),
        },)

    def obs_indices(self, obs: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = id(obs)
        if cache_key in self._obs_index_cache:
            return self._obs_index_cache[cache_key]
        ridx = obs.reach_id.astype(int).map(self.rlookup).to_numpy(int)
        tidx = np.fromiter(
            (self.tlookup[(int(year), int(month))] for year, month in obs[["year", "month"]].itertuples(index=False)),
            dtype=int, count=len(obs),
        )
        result = (
            torch.tensor(tidx), torch.tensor(ridx),
            torch.tensor(obs.downstream_fraction_on_reach.to_numpy(float)),
        )
        self._obs_index_cache[cache_key] = result
        return result

    def station_water(self, obs: pd.DataFrame) -> torch.Tensor:
        tidx, ridx, fraction = self.obs_indices(obs)
        return self.water_inlet[tidx, ridx] + fraction * self.local_water[tidx, ridx]

    def q_features(self, obs: pd.DataFrame, train_start: int, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (id(obs), int(train_start), int(train_end))
        if cache_key in self._q_feature_cache:
            return self._q_feature_cache[cache_key]
        _, ridx, fraction = self.obs_indices(obs)
        indices = torch.tensor([
            index for index, (year, _) in enumerate(self.formal_keys) if train_start <= year <= train_end
        ])
        centers = []
        for reach, frac in zip(ridx.tolist(), fraction.tolist()):
            volume = self.water_inlet[indices, reach] + frac * self.local_water[indices, reach]
            q = volume / self.seconds[indices]
            centers.append(torch.median(torch.log(torch.clamp(q, min=EPS))))
        q_obs = self.station_water(obs) / self.seconds[self.obs_indices(obs)[0]]
        z = torch.log(torch.clamp(q_obs, min=EPS)) - torch.stack(centers)
        result = (torch.minimum(z, torch.zeros_like(z)), torch.maximum(z, torch.zeros_like(z)))
        self._q_feature_cache[cache_key] = result
        return result

    def _route(self, local: torch.Tensor, v_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inlet = [torch.zeros(self.shape[0]) for _ in range(230)]
        outlets = [torch.zeros(self.shape[0]) for _ in range(230)]
        removed = [torch.zeros(self.shape[0]) for _ in range(230)]
        for index in self.order_idx:
            attenuation_up = torch.exp(-v_f * self.h[:, index])
            attenuation_local = torch.exp(-v_f * self.h[:, index] / 2.0)
            outlet = inlet[index] * attenuation_up + local[:, index] * attenuation_local
            outlets[index] = outlet
            removed[index] = inlet[index] + local[:, index] - outlet
            if index in self.down_idx:
                inlet[self.down_idx[index]] = inlet[self.down_idx[index]] + outlet
        return torch.stack(inlet, dim=1), torch.stack(outlets, dim=1), torch.stack(removed, dim=1)

    def _station_log_concentration(self, obs: pd.DataFrame, local: torch.Tensor, v_f: torch.Tensor) -> torch.Tensor:
        inlet, _, _ = self._route(local, v_f)
        tidx, ridx, fraction = self.obs_indices(obs)
        exposure = self.h[tidx, ridx]
        load = inlet[tidx, ridx] * torch.exp(-v_f * exposure * fraction)
        load = load + fraction * local[tidx, ridx] * torch.exp(-v_f * exposure * fraction / 2.0)
        water = self.station_water(obs)
        return torch.log1p(1000.0 * load / torch.clamp(water, min=EPS))

    def evaluate(self, obs: pd.DataFrame, physical: torch.Tensor, train_start: int, train_end: int):
        values = dict(zip(self.names(), physical))
        fast, slow = self.local_fluxes(values)
        raw = self._station_log_concentration(obs, fast + slow, values["v_f"])
        calibrated_local = torch.exp(values["delta_path"]) * fast + torch.exp(-values["delta_path"]) * slow
        population = self._station_log_concentration(obs, calibrated_local, values["v_f"])
        low, high = self.q_features(obs, train_start, train_end)
        population = population + values["beta_low"] * low + values["beta_high"] * high
        return raw, population

    def structural_diagnostics(self, physical: torch.Tensor) -> dict[str, float]:
        values = dict(zip(self.names(), physical))
        fast, slow, state = self.local_fluxes(values, diagnostics=True)
        coefficient = state["coefficients"]
        local = fast + slow
        _, outlet, removed = self._route(local, values["v_f"])
        terminal = [index for index in self.order_idx if index not in self.down_idx]
        formal_input = torch.sum(self.input[self.formal_start :])
        initial_2010 = (
            torch.sum(self.input[: self.formal_start])
            - torch.sum(self.crop[: self.formal_start])
        )
        # The exact cumulative land balance is recomputed from recorded states
        # below, not inferred from the descriptive initial_2010 quantity.
        total_local = torch.sum(local)
        total_removed = torch.sum(removed)
        total_terminal = torch.sum(outlet[:, terminal])
        land_input = state["cumulative_input_all"]
        land_outputs_and_state = (
            state["cumulative_crop_all"] + state["cumulative_fast_all"]
            + state["cumulative_slow_all"] + state["cumulative_other_all"]
            + state["terminal_mineral_all"] + state["terminal_lower_all"]
        )
        return {
            "operator_closure_max_abs": float(torch.max(torch.abs(coefficient["closure"] - 1.0)).detach()),
            "daily_probability_min": float(torch.min(coefficient["probability"]).detach()),
            "daily_probability_max": float(torch.max(coefficient["probability"]).detach()),
            "zero_contact_probability_max": float(torch.max(torch.where(
                self.contact_ratio <= EPS, coefficient["probability"],
                torch.zeros_like(coefficient["probability"]),
            )).detach()),
            "initial_mineral_1961_kg_n": 0.0,
            "initial_lower_1961_kg_n": 0.0,
            "land_input_all_kg_n": float(land_input.detach()),
            "land_outputs_and_terminal_state_all_kg_n": float(land_outputs_and_state.detach()),
            "land_mass_closure_relative": float((
                torch.abs(land_input - land_outputs_and_state) / torch.clamp(land_input, min=1.0)
            ).detach()),
            "formal_other_loss_kg_n": float(torch.sum(state["other_loss"]).detach()),
            "formal_mineral_end_kg_n": float(torch.sum(state["mineral_end"][-1]).detach()),
            "formal_lower_end_kg_n": float(torch.sum(state["lower_end"][-1]).detach()),
            "formal_fast_export_kg_n": float(torch.sum(fast).detach()),
            "formal_slow_export_kg_n": float(torch.sum(slow).detach()),
            "channel_input_kg_n": float(total_local.detach()),
            "channel_removed_kg_n": float(total_removed.detach()),
            "terminal_output_kg_n": float(total_terminal.detach()),
            "channel_closure_relative": float((torch.abs(total_local - total_removed - total_terminal) / torch.clamp(total_local, min=1.0)).detach()),
            "descriptive_formal_input_kg_n": float(formal_input.detach()),
            "descriptive_pre2010_input_minus_crop_kg_n": float(initial_2010.detach()),
            "maximum_abs_log_alpha_modifier": float(torch.max(torch.abs(coefficient["log_alpha_reach"] - values["log_alpha_contact"])).detach()),
        }
