"""Conservative hydrology-driven monthly TN core for stage 13."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1.0e-12
SOURCE_COLUMNS = {
    "FERT": "fertilizer_kg_n",
    "MAN": "manure_kg_n",
    "BNF": "cropland_bnf_kg_n",
    "DEP": "atmospheric_deposition_kg_n",
}
KERNEL_COLUMNS = [
    "k_zu_zu", "k_zu_zl", "k_zu_x",
    "k_zl_zu", "k_zl_zl", "k_zl_x",
    "k_fast_zu", "k_fast_zl", "k_fast_x",
    "k_slow_zu", "k_slow_zl", "k_slow_x",
]


def _safe_partition(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > EPS)


def monthly_kernels(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compile a 4-output x 3-input conservative kernel from monthly balances.

    Inputs are upper start N, lower start N and the month's delivered local N.
    Monthly input is placed in the upper carrier before the monthly partition.
    """
    frame = monthly.sort_values(["year", "month", "reach_id"]).reset_index(drop=True).copy()
    du = frame.upper_response_storage_start_mm.to_numpy(float) + frame.effective_excess_to_upper_mm.to_numpy(float)
    a = _safe_partition(frame.upper_response_storage_end_mm.to_numpy(float), du)
    b = _safe_partition(frame.local_fast_response_mm.to_numpy(float), du)
    c = _safe_partition(frame.percolation_to_lower_mm.to_numpy(float), du)
    dry_u = du <= EPS
    a[dry_u], b[dry_u], c[dry_u] = 1.0, 0.0, 0.0
    upper_sum = a + b + c
    a, b, c = a / upper_sum, b / upper_sum, c / upper_sum

    dl = frame.lower_slow_storage_start_mm.to_numpy(float) + frame.percolation_to_lower_mm.to_numpy(float)
    e = _safe_partition(frame.lower_slow_storage_end_mm.to_numpy(float), dl)
    s = _safe_partition(frame.local_slow_response_mm.to_numpy(float), dl)
    dry_l = dl <= EPS
    e[dry_l], s[dry_l] = 1.0, 0.0
    lower_sum = e + s
    e, s = e / lower_sum, s / lower_sum

    # Row order: upper end, lower end, fast output, slow output.
    # Column order: upper start, lower start, monthly input.
    values = np.column_stack([
        a, np.zeros(len(frame)), a,
        e * c, e, e * c,
        b, np.zeros(len(frame)), b,
        s * c, s, s * c,
    ])
    out = frame[["reach_id", "year", "month"]].copy()
    out["carrier"] = "MONTHLY_BALANCE_CARRIER"
    out[KERNEL_COLUMNS] = values
    return out


def daily_compiled_kernels(daily: pd.DataFrame) -> pd.DataFrame:
    """Compile exact within-month unit-tracer kernels from daily water balances."""
    d = daily.sort_values(["date", "reach_id"]).reset_index(drop=True).copy()
    rows: list[pd.DataFrame] = []
    for (year, month), block in d.groupby(["year", "month"], sort=True):
        block = block.sort_values(["date", "reach_id"])
        reaches = np.sort(block.reach_id.unique().astype(int))
        n = len(reaches)
        if n != 230 or block.date.nunique() * n != len(block):
            raise RuntimeError(f"daily coverage failure: {year}-{month:02d}")
        # Basis columns: initial upper, initial lower, total monthly input.
        zu = np.zeros((n, 3), dtype=float)
        zl = np.zeros((n, 3), dtype=float)
        yf = np.zeros((n, 3), dtype=float)
        ys = np.zeros((n, 3), dtype=float)
        zu[:, 0] = 1.0
        zl[:, 1] = 1.0
        days = int(block.date.nunique())
        for _, day in block.groupby("date", sort=True):
            day = day.sort_values("reach_id")
            du = day.upper_response_storage_start_mm.to_numpy(float) + day.effective_excess_to_upper_mm_day.to_numpy(float)
            a = _safe_partition(day.upper_response_storage_end_mm.to_numpy(float), du)
            b = _safe_partition(day.local_fast_response_mm_day.to_numpy(float), du)
            c = _safe_partition(day.percolation_to_lower_mm_day.to_numpy(float), du)
            dry_u = du <= EPS
            a[dry_u], b[dry_u], c[dry_u] = 1.0, 0.0, 0.0
            us = a + b + c
            a, b, c = a / us, b / us, c / us

            dl = day.lower_slow_storage_start_mm.to_numpy(float) + day.percolation_to_lower_mm_day.to_numpy(float)
            e = _safe_partition(day.lower_slow_storage_end_mm.to_numpy(float), dl)
            s = _safe_partition(day.local_slow_response_mm_day.to_numpy(float), dl)
            dry_l = dl <= EPS
            e[dry_l], s[dry_l] = 1.0, 0.0
            ls = e + s
            e, s = e / ls, s / ls

            x = np.zeros_like(zu)
            x[:, 2] = 1.0 / days
            upper_available = zu + x
            nperc = c[:, None] * upper_available
            yf += b[:, None] * upper_available
            zu = a[:, None] * upper_available
            lower_available = zl + nperc
            ys += s[:, None] * lower_available
            zl = e[:, None] * lower_available

        values = np.column_stack([
            zu[:, 0], zu[:, 1], zu[:, 2],
            zl[:, 0], zl[:, 1], zl[:, 2],
            yf[:, 0], yf[:, 1], yf[:, 2],
            ys[:, 0], ys[:, 1], ys[:, 2],
        ])
        out = pd.DataFrame({"reach_id": reaches, "year": int(year), "month": int(month)})
        out["carrier"] = "DAILY_COMPILED_CARRIER"
        out[KERNEL_COLUMNS] = values
        rows.append(out)
    return pd.concat(rows, ignore_index=True).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)


def kernel_array(kernels: pd.DataFrame) -> np.ndarray:
    values = kernels[KERNEL_COLUMNS].to_numpy(float)
    return values.reshape((-1, 4, 3))


def kernel_audit(kernels: pd.DataFrame) -> dict[str, float | bool | str]:
    k = kernel_array(kernels)
    column_sums = k.sum(axis=1)
    return {
        "carrier": str(kernels.carrier.iloc[0]),
        "rows": int(len(kernels)),
        "minimum_coefficient": float(k.min()),
        "maximum_coefficient": float(k.max()),
        "maximum_column_sum_error": float(np.max(np.abs(column_sums - 1.0))),
        "all_nonnegative": bool(k.min() >= -1.0e-12),
        "all_columns_close": bool(np.max(np.abs(column_sums - 1.0)) <= 1.0e-12),
    }


def source_availability(source: pd.DataFrame) -> pd.DataFrame:
    """Apply crop demand proportionally while preserving positive source tags."""
    frame = source.loc[source.calendar_scenario.eq("CENTRAL") & source.year.ge(2010)].copy()
    frame = frame.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    positive = sum(frame[column].to_numpy(float) for column in SOURCE_COLUMNS.values())
    demand = frame.crop_demand_kg_n.to_numpy(float)
    survival = np.maximum(positive - demand, 0.0) / np.maximum(positive, EPS)
    for tag, column in SOURCE_COLUMNS.items():
        frame[f"available_{tag.lower()}_kg_n"] = frame[column].to_numpy(float) * survival
    frame["positive_source_total_kg_n"] = positive
    frame["crop_demand_satisfied_kg_n"] = np.minimum(positive, demand)
    frame["unmet_crop_demand_kg_n"] = np.maximum(demand - positive, 0.0)
    frame["available_total_kg_n"] = positive * survival
    return frame


def simulate_source_tagged(kernels: pd.DataFrame, available: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run source-tagged N through a selected conservative carrier at pi_E=1."""
    keys = ["reach_id", "year", "month"]
    ker = kernels.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    src = available.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    if not np.array_equal(ker[keys].to_numpy(), src[keys].to_numpy()):
        raise RuntimeError("kernel/source key mismatch")
    reaches = np.sort(ker.reach_id.unique().astype(int))
    n = len(reaches)
    tags = tuple(SOURCE_COLUMNS)
    zu = np.zeros((n, len(tags)), dtype=float)
    zl = np.zeros((n, len(tags)), dtype=float)
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    cumulative_input = np.zeros_like(zu)
    cumulative_output = np.zeros_like(zu)
    for (year, month), idx in ker.groupby(["year", "month"], sort=True).groups.items():
        positions = np.asarray(sorted(idx), dtype=int)
        block = ker.loc[positions].sort_values("reach_id")
        sblock = src.loc[positions].sort_values("reach_id")
        if not np.array_equal(block.reach_id.to_numpy(int), reaches):
            raise RuntimeError("reach order failure")
        k = kernel_array(block)
        x = np.column_stack([sblock[f"available_{tag.lower()}_kg_n"].to_numpy(float) for tag in tags])
        input_vector = np.stack([zu, zl, x], axis=1)  # reach, input state, tag
        result = np.einsum("roi,rit->rot", k, input_vector)
        zu, zl, fast, slow = result[:, 0, :], result[:, 1, :], result[:, 2, :], result[:, 3, :]
        cumulative_input += x
        cumulative_output += fast + slow
        closure = cumulative_input - cumulative_output - zu - zl
        for j, tag in enumerate(tags):
            outputs.append(pd.DataFrame({
                "carrier": block.carrier.iloc[0], "reach_id": reaches,
                "year": int(year), "month": int(month), "source_tag": tag,
                "input_available_kg_n": x[:, j], "upper_end_kg_n": zu[:, j],
                "lower_end_kg_n": zl[:, j], "local_fast_release_kg_n": fast[:, j],
                "local_slow_release_kg_n": slow[:, j],
            }))
            audits.append({
                "carrier": block.carrier.iloc[0], "year": int(year), "month": int(month),
                "source_tag": tag, "max_abs_cumulative_mass_error_kg_n": float(np.max(np.abs(closure[:, j]))),
                "max_relative_cumulative_mass_error": float(np.max(
                    np.abs(closure[:, j]) / np.maximum(cumulative_input[:, j], 1.0)
                )),
                "minimum_state_or_flux_kg_n": float(np.min(np.column_stack([zu[:, j], zl[:, j], fast[:, j], slow[:, j]]))),
            })
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audits)


def topology_operators(path: Path, reach_ids: np.ndarray) -> tuple[list[int], dict[int, int], dict[int, int]]:
    table = pd.read_csv(path)
    downstream = {
        int(row.reach_id): int(row.downstream_reach)
        for row in table.itertuples(index=False) if pd.notna(row.downstream_reach)
    }
    indegree = {int(r): 0 for r in reach_ids}
    for down in downstream.values():
        indegree[down] += 1
    queue = sorted([r for r, degree in indegree.items() if degree == 0])
    order: list[int] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        if current in downstream:
            down = downstream[current]
            indegree[down] -= 1
            if indegree[down] == 0:
                queue.append(down)
                queue.sort()
    if len(order) != len(reach_ids):
        raise RuntimeError("topology is not a DAG")
    terminal = {}
    for reach in reach_ids:
        cursor = int(reach)
        while cursor in downstream:
            cursor = downstream[cursor]
        terminal[int(reach)] = cursor
    return order, downstream, terminal


@dataclass
class M0Router:
    local_load: np.ndarray
    local_water: np.ndarray
    h_full: np.ndarray
    month_keys: list[tuple[int, int]]
    reach_ids: np.ndarray
    order: list[int]
    downstream: dict[int, int]

    def __post_init__(self) -> None:
        self.rlookup = {int(r): i for i, r in enumerate(self.reach_ids)}
        self.tlookup = {key: i for i, key in enumerate(self.month_keys)}
        self.order_idx = [self.rlookup[r] for r in self.order]
        self.down_idx = {self.rlookup[r]: self.rlookup[d] for r, d in self.downstream.items()}
        # Numerical optimization may visit thousands of nearby vf values across
        # nested folds. Keep only a small LRU cache; an unbounded cache would
        # retain full 180 x 230 routing arrays for every evaluation.
        self._cache: OrderedDict[float, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.water_inlet, self.water_outlet = self._route_water()

    def _route_water(self) -> tuple[np.ndarray, np.ndarray]:
        inlet = np.zeros_like(self.local_water)
        outlet = np.zeros_like(self.local_water)
        for i in self.order_idx:
            outlet[:, i] = inlet[:, i] + self.local_water[:, i]
            if i in self.down_idx:
                inlet[:, self.down_idx[i]] += outlet[:, i]
        return inlet, outlet

    def route_load(self, vf: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(vf), 10)
        if key in self._cache:
            inlet, outlet = self._cache.pop(key)
            self._cache[key] = (inlet, outlet)
            return inlet, outlet
        survival = np.exp(-float(vf) * self.h_full)
        midpoint = np.exp(-float(vf) * self.h_full / 2.0)
        inlet = np.zeros_like(self.local_load)
        outlet = np.zeros_like(self.local_load)
        for i in self.order_idx:
            outlet[:, i] = inlet[:, i] * survival[:, i] + self.local_load[:, i] * midpoint[:, i]
            if i in self.down_idx:
                inlet[:, self.down_idx[i]] += outlet[:, i]
        self._cache[key] = (inlet, outlet)
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return inlet, outlet

    def concentration_at_pi1(self, observations: pd.DataFrame, vf: float) -> np.ndarray:
        inlet, _ = self.route_load(vf)
        ridx = observations.reach_id.astype(int).map(self.rlookup).to_numpy(int)
        tidx = np.fromiter(
            (self.tlookup[(int(y), int(m))] for y, m in observations[["year", "month"]].itertuples(index=False)),
            dtype=int, count=len(observations),
        )
        frac = observations.downstream_fraction_on_reach.to_numpy(float)
        h = self.h_full[tidx, ridx]
        local = self.local_load[tidx, ridx]
        station_load = inlet[tidx, ridx] * np.exp(-float(vf) * h * frac)
        station_load += frac * local * np.exp(-float(vf) * h * frac / 2.0)
        station_water = self.water_inlet[tidx, ridx] + frac * self.local_water[tidx, ridx]
        return 1000.0 * station_load / np.maximum(station_water, EPS)

    def outlet_closure_vf0(self) -> float:
        _, outlet = self.route_load(0.0)
        terminals = [self.rlookup[r] for r in self.reach_ids if int(r) not in self.downstream]
        expected = self.local_load.sum(axis=1)
        return float(np.max(np.abs(outlet[:, terminals].sum(axis=1) - expected) / np.maximum(expected, 1.0)))


def build_router(local_flux: pd.DataFrame, monthly: pd.DataFrame, topology_path: Path) -> M0Router:
    total = local_flux.groupby(["year", "month", "reach_id"], as_index=False).agg(
        local_fast_release_kg_n=("local_fast_release_kg_n", "sum"),
        local_slow_release_kg_n=("local_slow_release_kg_n", "sum"),
    ).sort_values(["year", "month", "reach_id"])
    hydro = monthly.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    if not np.array_equal(
        total[["year", "month", "reach_id"]].to_numpy(),
        hydro[["year", "month", "reach_id"]].to_numpy(),
    ):
        raise RuntimeError("local flux/hydrology mismatch")
    reach_ids = np.sort(hydro.reach_id.unique().astype(int))
    months = hydro[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    shape = (len(months), len(reach_ids))
    order, downstream, _ = topology_operators(topology_path, reach_ids)
    return M0Router(
        local_load=(total.local_fast_release_kg_n.to_numpy(float) + total.local_slow_release_kg_n.to_numpy(float)).reshape(shape),
        local_water=(hydro.local_fast_response_volume_m3.to_numpy(float) + hydro.local_slow_response_volume_m3.to_numpy(float)).reshape(shape),
        h_full=hydro.h1_exposure_day_per_m.to_numpy(float).reshape(shape),
        month_keys=[(int(y), int(m)) for y, m in months.itertuples(index=False)],
        reach_ids=reach_ids, order=order, downstream=downstream,
    )
