"""Differentiable R2-consistent conservative TN carrier.

The expensive daily topology is reduced to month-specific linear segments.
Only the thirteen mixed reservoir TN stocks are stepped daily.  All matrices
remain differentiable with respect to the registered channel removal rate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
STAGE41 = ROOT / "5_Test/20260824_41/scripts"
STAGE46 = ROOT / "5_Test/20260824_46/scripts"
sys.path.insert(0, str(STAGE41))
sys.path.insert(0, str(STAGE46))
import l0_v2_core as l0  # noqa: E402
from stage46_models import Stage46Model  # noqa: E402


REACH_DAILY = ROOT / "5_Test/20260828_24/outputs/tn_hydrology_reach_daily_2006_2024.parquet"
REACH_MONTHLY = ROOT / "5_Test/20260828_24/outputs/tn_hydrology_reach_monthly_2006_2024.parquet"
RESERVOIR_DAILY = ROOT / "5_Test/20260828_24/outputs/tn_hydrology_reservoir_daily_2006_2024.parquet"
RESERVOIR_STATIC = ROOT / "5_Test/20260828_24/outputs/tn_hydrology_reservoir_static_metadata.parquet"
SECONDS_PER_DAY = 86400.0
EPS = 1.0e-12


def _month_index(dates: pd.Series) -> pd.PeriodIndex:
    return pd.to_datetime(dates).dt.to_period("M")


class ReservoirTNModel(Stage46Model):
    """MINERAL_LIFETIME parent using the locked R2 water and reservoir carrier."""

    def __init__(self, reservoir_enabled: bool = True) -> None:
        super().__init__("MINERAL_LIFETIME")
        self.reservoir_enabled = bool(reservoir_enabled)
        self.fixed_v_f: float | None = None
        self._segment_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
        self._prepare_r2_interface()

    def set_fixed_v_f(self, value: float | None) -> None:
        """Cache the linear Reach operators for block-coordinate fitting."""
        self.fixed_v_f = None if value is None else float(value)
        self._segment_cache = None
        if self.fixed_v_f is None:
            return
        rate = torch.tensor(self.fixed_v_f, dtype=torch.get_default_dtype())
        with torch.no_grad():
            self._segment_cache = [
                self._segment_matrices(torch.exp(-rate * self.route_h[month]), self.reservoir_enabled)
                for month in range(self.route_month_count)
            ]

    def _prepare_r2_interface(self) -> None:
        daily = pd.read_parquet(REACH_DAILY).sort_values(["date", "reach_id"]).reset_index(drop=True)
        daily["date"] = pd.to_datetime(daily["date"])
        expected_dates = pd.date_range("2006-01-01", "2024-12-31", freq="D")
        if len(daily) != len(expected_dates) * 230:
            raise RuntimeError("R2 formal daily Reach grain changed")
        if not pd.DatetimeIndex(daily.date.drop_duplicates()).equals(expected_dates):
            raise RuntimeError("R2 formal daily dates changed")

        month_periods = _month_index(daily["date"])
        unique_months = month_periods.drop_duplicates()
        self.route_month_count = 19 * 12
        self.route_formal_offset = 4 * 12
        self.route_source_start = self.tlookup_all[(2006, 1)]
        if len(unique_months) != self.route_month_count:
            raise RuntimeError("R2 formal month count changed")
        fast_weight = np.zeros((self.route_month_count, 31, 230), dtype=np.float64)
        slow_weight = np.zeros_like(fast_weight)
        valid = np.zeros((self.route_month_count, 31, 1), dtype=bool)
        date_rows: list[pd.DatetimeIndex] = []
        for m, period in enumerate(unique_months):
            block = daily.loc[month_periods.eq(period)].sort_values(["date", "reach_id"])
            dates = pd.DatetimeIndex(block.date.drop_duplicates())
            date_rows.append(dates)
            nday = len(dates)
            valid[m, :nday, 0] = True
            fast = block.local_fast_response_m3_s.to_numpy(np.float64).reshape(nday, 230)
            slow = block.local_slow_response_m3_s.to_numpy(np.float64).reshape(nday, 230)
            fast_sum = fast.sum(axis=0)
            slow_sum = slow.sum(axis=0)
            fast_weight[m, :nday] = np.divide(fast, fast_sum, out=np.zeros_like(fast), where=fast_sum > EPS)
            slow_weight[m, :nday] = np.divide(slow, slow_sum, out=np.zeros_like(slow), where=slow_sum > EPS)
        self.fast_day_weight = torch.from_numpy(fast_weight)
        self.slow_day_weight = torch.from_numpy(slow_weight)
        self.formal_valid_day = torch.from_numpy(valid)
        self.formal_dates = date_rows
        # The previous TN parent deliberately used climatological hydrology
        # through 2009.  R2 now provides actual 2006-2009 states, so replace
        # the land-to-water probabilities over the complete R2 period.  This
        # prevents an impossible fast-N flux in a month with zero R2 fast water.
        static_for_area = pd.read_parquet(l0.STATIC).sort_values("reach_id")
        area = static_for_area.catchment_area_km2.to_numpy(np.float64)
        factor = 86.4 / area[None, :]
        nday_total = len(expected_dates)
        fast_mm = daily.local_fast_response_m3_s.to_numpy(np.float64).reshape(nday_total, 230) * factor
        slow_mm = daily.local_slow_response_m3_s.to_numpy(np.float64).reshape(nday_total, 230) * factor
        percolation_mm = daily.percolation_to_lower_mm_day.to_numpy(np.float64).reshape(nday_total, 230)
        upper_end_mm = daily.upper_response_storage_mm.to_numpy(np.float64).reshape(nday_total, 230)
        lower_end_mm = daily.lower_slow_storage_mm.to_numpy(np.float64).reshape(nday_total, 230)
        lower_start_mm = np.concatenate([lower_end_mm[:1], lower_end_mm[:-1]], axis=0)
        contact_water = fast_mm + percolation_mm
        upper_available = contact_water + upper_end_mm
        contact_ratio_daily = np.divide(
            contact_water, upper_available, out=np.zeros_like(contact_water), where=upper_available > EPS
        )
        fast_fraction_daily = np.divide(
            fast_mm, contact_water, out=np.zeros_like(fast_mm), where=contact_water > EPS
        )
        lower_available = lower_start_mm + percolation_mm
        lower_release_daily = np.divide(
            slow_mm, lower_available, out=np.zeros_like(slow_mm), where=lower_available > EPS
        )
        if max(contact_ratio_daily.max(), lower_release_daily.max()) > 1.0 + 1.0e-9:
            raise RuntimeError("R2 daily land hydraulic probability exceeds one")
        contact_padded = np.zeros_like(fast_weight)
        fast_fraction_padded = np.zeros_like(fast_weight)
        lower_release_padded = np.zeros_like(fast_weight)
        cursor = 0
        for m, dates in enumerate(date_rows):
            count = len(dates)
            contact_padded[m, :count] = contact_ratio_daily[cursor : cursor + count]
            fast_fraction_padded[m, :count] = fast_fraction_daily[cursor : cursor + count]
            lower_release_padded[m, :count] = lower_release_daily[cursor : cursor + count]
            cursor += count
        if cursor != nday_total:
            raise RuntimeError("R2 land-hydrology cursor mismatch")
        self.contact_ratio[self.route_source_start :] = torch.from_numpy(contact_padded)
        self.fast_fraction[self.route_source_start :] = torch.from_numpy(fast_fraction_padded)
        self.lower_release[self.route_source_start :] = torch.from_numpy(lower_release_padded)

        metadata_raw = pd.read_parquet(RESERVOIR_STATIC).sort_values("reservoir_entity_id").reset_index(drop=True)
        reservoir_raw = pd.read_parquet(RESERVOIR_DAILY)
        reservoir_raw["date"] = pd.to_datetime(reservoir_raw["date"])

        monthly = pd.read_parquet(REACH_MONTHLY).sort_values(["month", "reach_id"]).reset_index(drop=True)
        monthly["month"] = pd.to_datetime(monthly["month"])
        if len(monthly) != self.route_month_count * 230:
            raise RuntimeError("R2 formal monthly Reach grain changed")
        static = static_for_area
        width = static.bankfull_width_m.to_numpy(np.float64)
        length = static.length_km.to_numpy(np.float64) * 1000.0
        route_shape = (self.route_month_count, 230)
        q = monthly.routed_total_m3_s.to_numpy(np.float64).reshape(route_shape)
        exposure = np.divide(
            np.tile(length * width, (self.route_month_count, 1)),
            q * SECONDS_PER_DAY,
            out=np.full(route_shape, 1.0e6, dtype=np.float64),
            where=q > EPS,
        )
        self.route_h = torch.from_numpy(exposure)
        self.h = self.route_h[self.route_formal_offset :]
        days = monthly.month.dt.days_in_month.to_numpy(np.float64).reshape(route_shape)[:, 0]
        self.route_seconds = torch.from_numpy(days * SECONDS_PER_DAY)
        self.seconds = self.route_seconds[self.route_formal_offset :]
        outlet_water = q * (days[:, None] * SECONDS_PER_DAY)
        area = static.catchment_area_km2.to_numpy(np.float64)
        local_q = (
            monthly.local_fast_response_m3_s.to_numpy(np.float64)
            + monthly.local_slow_response_m3_s.to_numpy(np.float64)
        ).reshape(route_shape)
        local_water = local_q * (days[:, None] * SECONDS_PER_DAY)
        inlet_water = outlet_water - local_water
        # The R2 product overwrites a single-arm control Reach with its
        # post-reservoir release.  Stations on those Reaches are upstream of
        # the edge-control dam, so reconstruct their pre-dam water from the
        # exact daily captured+bypass diagnostic.  Multi-arm controls retain
        # their pre-dam routed value in the R2 Reach product already.
        for row in metadata_raw.itertuples(index=False):
            controls = [int(value) for value in row.control_reaches]
            if len(controls) != 1:
                continue
            reach = controls[0] - 1
            block = reservoir_raw.loc[
                reservoir_raw.reservoir_entity_id.eq(row.reservoir_entity_id),
                ["date", "captured_inflow_m3", "bypass_inflow_m3"],
            ].copy()
            block["month"] = block.date.dt.to_period("M").dt.to_timestamp()
            pre_dam = (
                block.assign(pre_dam_m3=block.captured_inflow_m3 + block.bypass_inflow_m3)
                .groupby("month", sort=True).pre_dam_m3.sum().to_numpy(np.float64)
            )
            if len(pre_dam) != self.route_month_count:
                raise RuntimeError(f"Incomplete pre-dam water series for {row.reservoir_entity_id}")
            outlet_water[:, reach] = pre_dam
            inlet_water[:, reach] = pre_dam - local_water[:, reach]
        if np.min(inlet_water) < -1.0e-3:
            raise RuntimeError(f"R2 inlet water became negative: {np.min(inlet_water)}")
        self.route_local_water = torch.from_numpy(local_water)
        self.route_water_inlet = torch.from_numpy(np.maximum(inlet_water, 0.0))
        self.route_water_outlet = torch.from_numpy(outlet_water)
        self.local_water = self.route_local_water[self.route_formal_offset :]
        self.water_inlet = self.route_water_inlet[self.route_formal_offset :]
        self.water_outlet = self.route_water_outlet[self.route_formal_offset :]

        metadata = metadata_raw.copy()
        topo_position = {index + 1: position for position, index in enumerate(self.order_idx)}
        metadata["operator_position"] = metadata.control_reaches.map(
            lambda reaches: max(topo_position[int(reach)] for reach in reaches)
        )
        metadata = metadata.sort_values(["operator_position", "reservoir_entity_id"]).reset_index(drop=True)
        self.reservoir_ids = metadata.reservoir_entity_id.astype(str).tolist()
        self.reservoir_lookup = {entity: i for i, entity in enumerate(self.reservoir_ids)}
        self.control_to_reservoir: dict[int, int] = {}
        self.reservoir_controls: list[list[int]] = []
        self.reservoir_outflow: list[int] = []
        self.capture_fraction: list[float] = []
        for r, row in metadata.iterrows():
            controls = [int(value) - 1 for value in row.control_reaches]
            self.reservoir_controls.append(controls)
            self.reservoir_outflow.append(int(row.outflow_reach) - 1)
            self.capture_fraction.append(float(row.local_capture_fraction))
            for control in controls:
                if control in self.control_to_reservoir:
                    raise RuntimeError("Duplicate R2 reservoir control Reach")
                self.control_to_reservoir[control] = r

        reservoir = reservoir_raw.copy()
        reservoir["reservoir_order"] = reservoir.reservoir_entity_id.map(self.reservoir_lookup)
        reservoir = reservoir.sort_values(["date", "reservoir_order"]).reset_index(drop=True)
        if len(reservoir) != len(expected_dates) * len(metadata):
            raise RuntimeError("R2 formal daily reservoir grain changed")
        denominator = reservoir.storage_m3.to_numpy(np.float64) + reservoir.total_release_m3.to_numpy(np.float64)
        release = np.divide(
            reservoir.total_release_m3.to_numpy(np.float64), denominator,
            out=np.zeros(len(reservoir), dtype=np.float64), where=denominator > EPS,
        ).reshape(len(expected_dates), len(metadata))
        release_padded = np.zeros((self.route_month_count, 31, len(metadata)), dtype=np.float64)
        cursor = 0
        for m, dates in enumerate(date_rows):
            release_padded[m, : len(dates)] = release[cursor : cursor + len(dates)]
            cursor += len(dates)
        if cursor != len(expected_dates):
            raise RuntimeError("R2 release-fraction cursor mismatch")
        self.release_fraction = torch.from_numpy(release_padded)
        self.initial_water_storage_2006 = torch.from_numpy(
            reservoir.loc[reservoir.date.eq(pd.Timestamp("2006-01-01"))]
            .sort_values("reservoir_order").storage_m3.to_numpy(np.float64).copy()
        )

    def local_fluxes_2006(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the registered land-state flux beginning with R2 in 2006."""
        coefficient = self._daily_monthly_coefficients(values)
        mineral = torch.zeros(230)
        lower = torch.zeros(230)
        fast_rows, slow_rows = [], []
        for index in range(768):
            pre = mineral + self.input[index]
            crop = torch.minimum(pre, self.crop[index])
            after_crop = pre - crop
            fast = after_crop * coefficient["fast"][index]
            slow = lower * (1.0 - coefficient["lower_carry"][index]) + after_crop * coefficient["slow_from_upper"][index]
            lower = lower * coefficient["lower_carry"][index] + after_crop * coefficient["lower_from_upper"][index]
            mineral = after_crop * coefficient["upper_carry"][index]
            if index >= self.route_source_start:
                fast_rows.append(fast)
                slow_rows.append(slow)
        fast_result, slow_result = torch.stack(fast_rows), torch.stack(slow_rows)
        if fast_result.shape != (self.route_month_count, 230):
            raise RuntimeError("2006-2024 land-N flux shape changed")
        return fast_result, slow_result

    def station_water(self, obs: pd.DataFrame) -> torch.Tensor:
        tidx, ridx, fraction = self.obs_indices(obs)
        return self.water_inlet[tidx, ridx] + fraction * self.local_water[tidx, ridx]

    def q_features(self, obs: pd.DataFrame, train_start: int, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (id(obs), int(train_start), int(train_end), "R2")
        if cache_key in self._q_feature_cache:
            return self._q_feature_cache[cache_key]
        _, ridx, fraction = self.obs_indices(obs)
        indices = torch.tensor([i for i, (year, _) in enumerate(self.formal_keys) if train_start <= year <= train_end])
        centers = []
        for reach, frac in zip(ridx.tolist(), fraction.tolist()):
            volume = self.water_inlet[indices, reach] + frac * self.local_water[indices, reach]
            centers.append(torch.median(torch.log(torch.clamp(volume / self.seconds[indices], min=EPS))))
        q_obs = self.station_water(obs) / self.seconds[self.obs_indices(obs)[0]]
        z = torch.log(torch.clamp(q_obs, min=EPS)) - torch.stack(centers)
        result = (torch.minimum(z, torch.zeros_like(z)), torch.maximum(z, torch.zeros_like(z)))
        self._q_feature_cache[cache_key] = result
        return result

    def _segment_matrices(self, attenuation: torch.Tensor, enabled: bool):
        """Map local and reservoir-release mass to Reach inlets/outlets/captures."""
        n_res = len(self.reservoir_ids)
        dimension = 230 + n_res
        zero = torch.zeros(dimension, dtype=attenuation.dtype, device=attenuation.device)
        inlet = [zero for _ in range(230)]
        inlet_rows = [zero for _ in range(230)]
        outlet_rows = [zero for _ in range(230)]
        captures = [zero for _ in range(n_res)]
        seen: list[set[int]] = [set() for _ in range(n_res)]
        # Do not use sqrt(exp(-v_f H)): at zero-flow H is very large and the
        # backward pass can evaluate 0 * infinity.  The equivalent direct
        # exponential has a finite limiting derivative.
        half = torch.exp(-0.5 * (-torch.log(torch.clamp(attenuation, min=1.0e-300))))
        for reach in self.order_idx:
            inlet_rows[reach] = inlet[reach]
            local_basis = torch.nn.functional.one_hot(
                torch.tensor(reach, device=attenuation.device), num_classes=dimension
            ).to(dtype=attenuation.dtype) * half[reach]
            outlet = inlet[reach] * attenuation[reach] + local_basis
            outlet_rows[reach] = outlet
            res = self.control_to_reservoir.get(reach) if enabled else None
            if res is not None:
                controls = self.reservoir_controls[res]
                fraction = self.capture_fraction[res]
                if len(controls) > 1:
                    captures[res] = captures[res] + outlet
                    seen[res].add(reach)
                    if seen[res] != set(controls):
                        continue
                    target = self.reservoir_outflow[res]
                    release_basis = torch.nn.functional.one_hot(
                        torch.tensor(230 + res, device=attenuation.device), num_classes=dimension
                    ).to(dtype=attenuation.dtype)
                    inlet[target] = inlet[target] + release_basis
                    continue
                captured = outlet - (1.0 - fraction) * local_basis
                bypass = (1.0 - fraction) * local_basis
                captures[res] = captures[res] + captured
                target = self.reservoir_outflow[res]
                release_basis = torch.nn.functional.one_hot(
                    torch.tensor(230 + res, device=attenuation.device), num_classes=dimension
                ).to(dtype=attenuation.dtype)
                inlet[target] = inlet[target] + bypass + release_basis
                continue
            if reach in self.down_idx:
                target = self.down_idx[reach]
                inlet[target] = inlet[target] + outlet
        return torch.stack(inlet_rows), torch.stack(outlet_rows), torch.stack(captures)

    def route_layers(
        self,
        fast_layers: torch.Tensor,
        slow_layers: torch.Tensor,
        v_f: torch.Tensor,
        *,
        enabled: bool | None = None,
        initial_tn_storage: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        route_shape = (self.route_month_count, 230)
        if fast_layers.ndim != 3 or fast_layers.shape != slow_layers.shape or fast_layers.shape[1:] != route_shape:
            raise ValueError("route_layers expects layer x month x Reach fast/slow arrays")
        enabled = self.reservoir_enabled if enabled is None else bool(enabled)
        layers = fast_layers.shape[0]
        n_res = len(self.reservoir_ids)
        storage = (
            torch.zeros((layers, n_res), dtype=fast_layers.dtype, device=fast_layers.device)
            if initial_tn_storage is None else initial_tn_storage.clone()
        )
        monthly_inlet = []
        monthly_outlet = []
        total_removed = torch.zeros(layers, dtype=fast_layers.dtype, device=fast_layers.device)
        total_terminal = torch.zeros(layers, dtype=fast_layers.dtype, device=fast_layers.device)
        terminals = [reach for reach in self.order_idx if reach not in self.down_idx]
        for month in range(self.route_month_count):
            effective_v_f = (
                torch.tensor(self.fixed_v_f, dtype=v_f.dtype, device=v_f.device)
                if self.fixed_v_f is not None else v_f
            )
            attenuation = torch.exp(-effective_v_f * self.route_h[month])
            if enabled == self.reservoir_enabled and self._segment_cache is not None:
                inlet_matrix, outlet_matrix, capture_matrix = self._segment_cache[month]
            else:
                inlet_matrix, outlet_matrix, capture_matrix = self._segment_matrices(attenuation, enabled)
            inlet_local, inlet_release = inlet_matrix[:, :230], inlet_matrix[:, 230:]
            outlet_local, outlet_release = outlet_matrix[:, :230], outlet_matrix[:, 230:]
            capture_local, capture_release = capture_matrix[:, :230], capture_matrix[:, 230:]
            month_inlet = torch.zeros((layers, 230), dtype=fast_layers.dtype, device=fast_layers.device)
            month_outlet = torch.zeros_like(month_inlet)
            for day in range(31):
                if not bool(self.formal_valid_day[month, day, 0]):
                    continue
                local = (
                    fast_layers[:, month] * self.fast_day_weight[month, day]
                    + slow_layers[:, month] * self.slow_day_weight[month, day]
                )
                if enabled:
                    base_capture = local @ capture_local.T
                    releases: list[torch.Tensor] = []
                    for res in range(n_res):
                        prior_release = (
                            torch.stack(releases, dim=1) if releases else torch.zeros((layers, 0), dtype=local.dtype, device=local.device)
                        )
                        captured = base_capture[:, res]
                        if res:
                            captured = captured + torch.sum(prior_release * capture_release[res, :res], dim=1)
                        pre = storage[:, res] + captured
                        release = pre * self.release_fraction[month, day, res]
                        storage_next = pre - release
                        storage = torch.cat([storage[:, :res], storage_next[:, None], storage[:, res + 1 :]], dim=1)
                        releases.append(release)
                    release_vector = torch.stack(releases, dim=1)
                else:
                    release_vector = torch.zeros((layers, n_res), dtype=local.dtype, device=local.device)
                inlet_day = local @ inlet_local.T + release_vector @ inlet_release.T
                outlet_day = local @ outlet_local.T + release_vector @ outlet_release.T
                month_inlet = month_inlet + inlet_day
                month_outlet = month_outlet + outlet_day
                half_attenuation = torch.exp(-0.5 * effective_v_f * self.route_h[month])
                total_removed = total_removed + torch.sum(
                    inlet_day * (1.0 - attenuation) + local * (1.0 - half_attenuation),
                    dim=1,
                )
                total_terminal = total_terminal + torch.sum(outlet_day[:, terminals], dim=1)
            monthly_inlet.append(month_inlet)
            monthly_outlet.append(month_outlet)
        return {
            "inlet": torch.stack(monthly_inlet, dim=1),
            "outlet": torch.stack(monthly_outlet, dim=1),
            "final_reservoir_storage": storage,
            "channel_removed": total_removed,
            "terminal_output": total_terminal,
        }

    def _station_from_routed(
        self, obs: pd.DataFrame, local: torch.Tensor, inlet: torch.Tensor, v_f: torch.Tensor
    ) -> torch.Tensor:
        tidx, ridx, fraction = self.obs_indices(obs)
        exposure = self.h[tidx, ridx]
        load = inlet[tidx, ridx] * torch.exp(-v_f * exposure * fraction)
        load = load + fraction * local[tidx, ridx] * torch.exp(-v_f * exposure * fraction / 2.0)
        return torch.log1p(1000.0 * load / torch.clamp(self.station_water(obs), min=EPS))

    def evaluate(self, obs: pd.DataFrame, physical: torch.Tensor, train_start: int, train_end: int):
        values = dict(zip(self.names(), physical))
        fast_all, slow_all = self.local_fluxes_2006(values)
        fast_layers = torch.stack([fast_all, torch.exp(values["delta_path"]) * fast_all])
        slow_layers = torch.stack([slow_all, torch.exp(-values["delta_path"]) * slow_all])
        route_v_f = (
            torch.tensor(self.fixed_v_f, dtype=physical.dtype, device=physical.device)
            if self.fixed_v_f is not None else values["v_f"]
        )
        routed = self.route_layers(fast_layers, slow_layers, route_v_f)
        fast = fast_all[self.route_formal_offset :]
        slow = slow_all[self.route_formal_offset :]
        raw_local = fast + slow
        population_local = fast_layers[1, self.route_formal_offset :] + slow_layers[1, self.route_formal_offset :]
        raw = self._station_from_routed(
            obs, raw_local, routed["inlet"][0, self.route_formal_offset :], route_v_f
        )
        population = self._station_from_routed(
            obs, population_local, routed["inlet"][1, self.route_formal_offset :], route_v_f
        )
        low, high = self.q_features(obs, train_start, train_end)
        population = population + values["beta_low"] * low + values["beta_high"] * high
        return raw, population

    def carrier_diagnostics(self, physical: torch.Tensor, enabled: bool = True) -> dict[str, float]:
        values = dict(zip(self.names(), physical))
        fast, slow = self.local_fluxes_2006(values)
        routed = self.route_layers(fast[None], slow[None], values["v_f"], enabled=enabled)
        total_input = torch.sum(fast + slow)
        stock = torch.sum(routed["final_reservoir_storage"])
        balance = total_input - routed["channel_removed"][0] - routed["terminal_output"][0] - stock
        return {
            "total_local_input_kg_n": float(total_input.detach()),
            "channel_removed_kg_n": float(routed["channel_removed"][0].detach()),
            "terminal_output_kg_n": float(routed["terminal_output"][0].detach()),
            "final_reservoir_stock_kg_n": float(stock.detach()),
            "mass_balance_error_kg_n": float(balance.detach()),
            "mass_balance_relative": float((torch.abs(balance) / torch.clamp(total_input, min=1.0)).detach()),
            "reservoir_stock_min_kg_n": float(torch.min(routed["final_reservoir_storage"]).detach()),
        }


def parent_physical(fold_id: str = "T3") -> np.ndarray:
    table = pd.read_parquet(
        ROOT / "5_Test/20260824_46/outputs/by_candidate/mineral_lifetime_parameters.parquet"
    )
    row = table.loc[table.fold_id.eq(fold_id)].iloc[0]
    model = ReservoirTNModel(reservoir_enabled=True)
    return np.asarray([row[name] for name in model.names()], dtype=np.float64)
