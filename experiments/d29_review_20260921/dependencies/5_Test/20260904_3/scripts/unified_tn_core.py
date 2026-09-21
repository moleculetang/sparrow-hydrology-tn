"""Unified MINERAL_LIFETIME TN core for the Stage35/38 hydrology products.

This ports the already validated 20260831_2 conservative reservoir carrier to
the complete 1961 hydrologic history and optionally to the 2025 sensitivity
extension.  No reservoir reaction parameter is introduced.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
OLD_CORE_DIR = ROOT / "5_Test/20260831_2/scripts"
STAGE41_DIR = ROOT / "5_Test/20260824_41/scripts"
for item in (OLD_CORE_DIR, STAGE41_DIR):
    sys.path.insert(0, str(item))

import l0_v2_core as l0  # noqa: E402
from reservoir_tn_core import ReservoirTNModel  # noqa: E402


FORMAL_HYDRO = ROOT / "5_Test/20260828_35/outputs"
SENSITIVITY_HYDRO = ROOT / "5_Test/20260828_38/outputs"
FORMAL_SOURCE = ROOT / "5_Test/20260824_12/outputs/monthly_source_forcing_1961_2024.parquet"
SENSITIVITY_SOURCE = ROOT / "5_Test/20260904_2/outputs/monthly_source_forcing_1961_2025_sensitivity.parquet"
SECONDS_PER_DAY = 86400.0
EPS = 1.0e-12


class UnifiedTNModel(ReservoirTNModel):
    """One TN process implementation for either the formal or sensitivity hydrology."""

    def __init__(self, product: str = "formal", reservoir_enabled: bool = True) -> None:
        if product not in {"formal", "sensitivity"}:
            raise ValueError(product)
        self.product = product
        self.hydro_root = FORMAL_HYDRO if product == "formal" else SENSITIVITY_HYDRO
        self.source_path = FORMAL_SOURCE if product == "formal" else SENSITIVITY_SOURCE
        self.end_year = 2024 if product == "formal" else 2025
        self.training_years_override: tuple[int, ...] | None = None
        super().__init__(reservoir_enabled=reservoir_enabled)
        gc.collect()

    def set_training_years(self, years: list[int] | tuple[int, ...] | np.ndarray | None) -> None:
        self.training_years_override = None if years is None else tuple(sorted({int(value) for value in years}))
        self._q_feature_cache.clear()

    def _prepare_r2_interface(self) -> None:
        source = (
            pd.read_parquet(self.source_path)
            .loc[lambda frame: frame.calendar_scenario.eq("CENTRAL")]
            .sort_values(["year", "month", "reach_id"])
            .reset_index(drop=True)
        )
        expected_months = (self.end_year - 1961 + 1) * 12
        if len(source) != expected_months * 230:
            raise RuntimeError(f"Source grain changed: {len(source)}")
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
        source_total = source[[
            "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n",
        ]].sum(axis=1).to_numpy(np.float64).reshape(expected_months, 230)
        self.input = torch.from_numpy(source_total.copy())
        self.crop = torch.from_numpy(source.crop_demand_kg_n.to_numpy(np.float64).reshape(expected_months, 230).copy())
        self.days_in_month = source.days_in_month.to_numpy(np.float64).reshape(expected_months, 230)[:, 0]

        daily_path = self.hydro_root / "tn_hydrology_reach_daily.parquet"
        monthly_path = self.hydro_root / "tn_hydrology_reach_monthly.parquet"
        reservoir_daily_path = self.hydro_root / "tn_hydrology_reservoir_daily.parquet"
        reservoir_static_path = self.hydro_root / "tn_hydrology_reservoir_static_metadata.parquet"
        daily_columns = [
            "date", "reach_id", "local_fast_response_m3_s", "local_slow_response_m3_s",
            "percolation_to_lower_mm_day", "upper_response_storage_mm", "lower_slow_storage_mm",
        ]
        daily = pd.read_parquet(daily_path, columns=daily_columns).sort_values(["date", "reach_id"]).reset_index(drop=True)
        daily["date"] = pd.to_datetime(daily.date)
        expected_dates = pd.date_range("1961-01-01", f"{self.end_year}-12-31", freq="D")
        if len(daily) != len(expected_dates) * 230:
            raise RuntimeError("Stage35/38 daily Reach grain changed")
        if not pd.DatetimeIndex(daily.date.drop_duplicates()).equals(expected_dates):
            raise RuntimeError("Stage35/38 daily dates changed")

        monthly_columns = [
            "month", "reach_id", "routed_total_m3_s", "local_fast_response_m3_s", "local_slow_response_m3_s",
            "channel_bankfull_hydraulic_exposure_central_day", "bankfull_depth_m", "catchment_area_km2",
        ]
        monthly = pd.read_parquet(monthly_path, columns=monthly_columns).sort_values(["month", "reach_id"]).reset_index(drop=True)
        monthly["month"] = pd.to_datetime(monthly.month)
        if len(monthly) != expected_months * 230:
            raise RuntimeError("Stage35/38 monthly Reach grain changed")

        month_periods = daily.date.dt.to_period("M")
        unique_months = month_periods.drop_duplicates()
        if len(unique_months) != expected_months:
            raise RuntimeError("Hydrologic month count changed")
        self.route_month_count = expected_months
        self.route_formal_offset = self.formal_start
        self.route_source_start = 0
        padded_shape = (expected_months, 31, 230)
        fast_weight = np.zeros(padded_shape, dtype=np.float64)
        slow_weight = np.zeros_like(fast_weight)
        contact_padded = np.zeros_like(fast_weight)
        fast_fraction_padded = np.zeros_like(fast_weight)
        lower_release_padded = np.zeros_like(fast_weight)
        valid = np.zeros((expected_months, 31, 1), dtype=bool)
        date_rows: list[pd.DatetimeIndex] = []

        area = monthly.loc[monthly.month.eq(monthly.month.min())].sort_values("reach_id").catchment_area_km2.to_numpy(np.float64)
        conversion = 86.4 / area[None, :]
        day_count = len(expected_dates)
        fast_q = daily.local_fast_response_m3_s.to_numpy(np.float64).reshape(day_count, 230)
        slow_q = daily.local_slow_response_m3_s.to_numpy(np.float64).reshape(day_count, 230)
        fast_mm = fast_q * conversion
        slow_mm = slow_q * conversion
        percolation = daily.percolation_to_lower_mm_day.to_numpy(np.float64).reshape(day_count, 230)
        upper_end = daily.upper_response_storage_mm.to_numpy(np.float64).reshape(day_count, 230)
        lower_end = daily.lower_slow_storage_mm.to_numpy(np.float64).reshape(day_count, 230)
        lower_start = np.concatenate([lower_end[:1], lower_end[:-1]], axis=0)
        contact_water = fast_mm + percolation
        upper_available = contact_water + upper_end
        contact = np.divide(contact_water, upper_available, out=np.zeros_like(contact_water), where=upper_available > EPS)
        fast_fraction = np.divide(fast_mm, contact_water, out=np.zeros_like(fast_mm), where=contact_water > EPS)
        lower_available = lower_start + percolation
        lower_release = np.divide(slow_mm, lower_available, out=np.zeros_like(slow_mm), where=lower_available > EPS)
        if min(contact.min(), fast_fraction.min(), lower_release.min()) < -1.0e-10:
            raise RuntimeError("Negative daily hydrologic transfer probability")
        if max(contact.max(), fast_fraction.max(), lower_release.max()) > 1.0 + 1.0e-8:
            raise RuntimeError("Daily hydrologic transfer probability exceeds one")

        cursor = 0
        for month_index, period in enumerate(unique_months):
            block = daily.loc[month_periods.eq(period)].sort_values(["date", "reach_id"])
            dates = pd.DatetimeIndex(block.date.drop_duplicates())
            date_rows.append(dates)
            count = len(dates)
            valid[month_index, :count, 0] = True
            fq = fast_q[cursor : cursor + count]
            sq = slow_q[cursor : cursor + count]
            fsum, ssum = fq.sum(axis=0), sq.sum(axis=0)
            fast_weight[month_index, :count] = np.divide(fq, fsum, out=np.zeros_like(fq), where=fsum > EPS)
            slow_weight[month_index, :count] = np.divide(sq, ssum, out=np.zeros_like(sq), where=ssum > EPS)
            contact_padded[month_index, :count] = contact[cursor : cursor + count]
            fast_fraction_padded[month_index, :count] = fast_fraction[cursor : cursor + count]
            lower_release_padded[month_index, :count] = lower_release[cursor : cursor + count]
            cursor += count
        if cursor != day_count:
            raise RuntimeError("Daily hydrology padding cursor mismatch")
        self.fast_day_weight = torch.from_numpy(fast_weight)
        self.slow_day_weight = torch.from_numpy(slow_weight)
        self.formal_valid_day = torch.from_numpy(valid)
        self.formal_dates = date_rows
        self.contact_ratio = torch.from_numpy(contact_padded)
        self.fast_fraction = torch.from_numpy(fast_fraction_padded)
        self.lower_release = torch.from_numpy(lower_release_padded)
        self.valid_day = torch.from_numpy(valid)

        route_shape = (expected_months, 230)
        depth = monthly.bankfull_depth_m.to_numpy(np.float64).reshape(route_shape)
        official_exposure = monthly.channel_bankfull_hydraulic_exposure_central_day.to_numpy(np.float64).reshape(route_shape)
        reaction_exposure = np.divide(
            official_exposure, depth,
            out=np.full(route_shape, 1.0e6, dtype=np.float64), where=depth > EPS,
        )
        # The hydrology interface intentionally stores null exposure at zero
        # flow.  For the attenuation operator use its mathematical dry-reach
        # limit rather than inventing a positive discharge floor.
        reaction_exposure = np.where(np.isfinite(reaction_exposure), reaction_exposure, 1.0e6)
        self.route_h = torch.from_numpy(reaction_exposure)
        self.h = self.route_h[self.route_formal_offset :]
        days = monthly.month.dt.days_in_month.to_numpy(np.float64).reshape(route_shape)[:, 0]
        self.route_seconds = torch.from_numpy(days * SECONDS_PER_DAY)
        self.seconds = self.route_seconds[self.route_formal_offset :]
        q = monthly.routed_total_m3_s.to_numpy(np.float64).reshape(route_shape)
        outlet_water = q * days[:, None] * SECONDS_PER_DAY
        local_q = (monthly.local_fast_response_m3_s + monthly.local_slow_response_m3_s).to_numpy(np.float64).reshape(route_shape)
        local_water = local_q * days[:, None] * SECONDS_PER_DAY
        inlet_water = outlet_water - local_water

        metadata_raw = pd.read_parquet(reservoir_static_path).sort_values("reservoir_entity_id").reset_index(drop=True)
        reservoir_raw = pd.read_parquet(reservoir_daily_path)
        reservoir_raw["date"] = pd.to_datetime(reservoir_raw.date)
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
            pre_dam = block.assign(pre_dam=block.captured_inflow_m3 + block.bypass_inflow_m3).groupby("month", sort=True).pre_dam.sum().to_numpy(np.float64)
            if len(pre_dam) != expected_months:
                raise RuntimeError(f"Incomplete pre-dam water for {row.reservoir_entity_id}")
            outlet_water[:, reach] = pre_dam
            inlet_water[:, reach] = pre_dam - local_water[:, reach]
        if np.min(inlet_water) < -1.0e-3:
            raise RuntimeError(f"Negative water inlet: {np.min(inlet_water)}")
        self.route_local_water = torch.from_numpy(local_water)
        self.route_water_inlet = torch.from_numpy(np.maximum(inlet_water, 0.0))
        self.route_water_outlet = torch.from_numpy(outlet_water)
        self.local_water = self.route_local_water[self.route_formal_offset :]
        self.water_inlet = self.route_water_inlet[self.route_formal_offset :]
        self.water_outlet = self.route_water_outlet[self.route_formal_offset :]

        topo_position = {index + 1: position for position, index in enumerate(self.order_idx)}
        metadata = metadata_raw.copy()
        metadata["operator_position"] = metadata.control_reaches.map(lambda reaches: max(topo_position[int(reach)] for reach in reaches))
        metadata = metadata.sort_values(["operator_position", "reservoir_entity_id"]).reset_index(drop=True)
        self.reservoir_ids = metadata.reservoir_entity_id.astype(str).tolist()
        self.reservoir_lookup = {entity: index for index, entity in enumerate(self.reservoir_ids)}
        self.control_to_reservoir = {}
        self.reservoir_controls = []
        self.reservoir_outflow = []
        self.capture_fraction = []
        for res, row in metadata.iterrows():
            controls = [int(value) - 1 for value in row.control_reaches]
            self.reservoir_controls.append(controls)
            self.reservoir_outflow.append(int(row.outflow_reach) - 1)
            self.capture_fraction.append(float(row.local_capture_fraction))
            for control in controls:
                if control in self.control_to_reservoir:
                    raise RuntimeError("Duplicate reservoir control Reach")
                self.control_to_reservoir[control] = res

        reservoir = reservoir_raw.copy()
        reservoir["reservoir_order"] = reservoir.reservoir_entity_id.map(self.reservoir_lookup)
        reservoir = reservoir.sort_values(["date", "reservoir_order"]).reset_index(drop=True)
        if len(reservoir) != day_count * len(metadata):
            raise RuntimeError("Reservoir daily grain changed")
        denominator = reservoir.storage_m3.to_numpy(np.float64) + reservoir.total_release_m3.to_numpy(np.float64)
        release = np.divide(
            reservoir.total_release_m3.to_numpy(np.float64), denominator,
            out=np.zeros(len(reservoir), dtype=np.float64), where=denominator > EPS,
        ).reshape(day_count, len(metadata))
        release_padded = np.zeros((expected_months, 31, len(metadata)), dtype=np.float64)
        cursor = 0
        for month_index, dates in enumerate(date_rows):
            count = len(dates)
            release_padded[month_index, :count] = release[cursor : cursor + count]
            cursor += count
        self.release_fraction = torch.from_numpy(release_padded)
        self.initial_water_storage_2006 = torch.zeros(len(metadata), dtype=torch.float64)

    def local_fluxes_2006(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return land-to-water TN fluxes for the complete 1961 product period."""
        coefficient = self._daily_monthly_coefficients(values)
        mineral = torch.zeros(230, dtype=self.input.dtype)
        lower = torch.zeros(230, dtype=self.input.dtype)
        fast_rows: list[torch.Tensor] = []
        slow_rows: list[torch.Tensor] = []
        for index in range(self.route_month_count):
            pre = mineral + self.input[index]
            crop = torch.minimum(pre, self.crop[index])
            after_crop = pre - crop
            fast = after_crop * coefficient["fast"][index]
            slow = lower * (1.0 - coefficient["lower_carry"][index]) + after_crop * coefficient["slow_from_upper"][index]
            lower = lower * coefficient["lower_carry"][index] + after_crop * coefficient["lower_from_upper"][index]
            mineral = after_crop * coefficient["upper_carry"][index]
            fast_rows.append(fast)
            slow_rows.append(slow)
        return torch.stack(fast_rows), torch.stack(slow_rows)

    def q_features(self, obs: pd.DataFrame, train_start: int, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        years = self.training_years_override or tuple(range(int(train_start), int(train_end) + 1))
        cache_key = (id(obs), years, self.product)
        if cache_key in self._q_feature_cache:
            return self._q_feature_cache[cache_key]
        _, ridx, fraction = self.obs_indices(obs)
        indices = torch.tensor([index for index, (year, _) in enumerate(self.formal_keys) if year in years])
        if len(indices) == 0:
            raise RuntimeError("No hydrologic months for training-year baseline")
        centers = []
        for reach, frac in zip(ridx.tolist(), fraction.tolist()):
            volume = self.water_inlet[indices, reach] + frac * self.local_water[indices, reach]
            centers.append(torch.median(torch.log(torch.clamp(volume / self.seconds[indices], min=EPS))))
        q_obs = self.station_water(obs) / self.seconds[self.obs_indices(obs)[0]]
        z = torch.log(torch.clamp(q_obs, min=EPS)) - torch.stack(centers)
        result = (torch.minimum(z, torch.zeros_like(z)), torch.maximum(z, torch.zeros_like(z)))
        self._q_feature_cache[cache_key] = result
        return result
