"""Run the corrected, parameter-equivalent M3 river-channel baseline."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_19"
P18 = ROOT / "5_Test" / "20260824_18"
P13 = ROOT / "5_Test" / "20260824_13"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CANON_DAILY = ROOT / "5_Test" / "20260828_9" / "outputs" / "canonical_reach_daily_2006_2024.parquet"
CANON_MONTHLY = ROOT / "5_Test" / "20260828_9" / "outputs" / "canonical_reach_monthly_2006_2024.parquet"
FORCING = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"
SOURCE = ROOT / "5_Test" / "20260824_12" / "outputs" / "monthly_source_forcing_1961_2024.parquet"
OLD_MONTHLY = ROOT / "5_Test" / "20260824_12" / "outputs" / "canonical_tn_bridge_monthly_2010_2024.parquet"
OLD_OOF = ROOT / "5_Test" / "20260824_16" / "outputs" / "dynamic_delivery_oof_predictions.parquet"
OBS_AUDIT = P18 / "outputs" / "tn_observations_audited.parquet"
CURRENT_OBS = ROOT / "5_Test" / "20260824_12" / "outputs" / "tn_observations_primary_2016_2024.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
CONTRACT = RUN / "experiment_contract.json"
EPS = 1.0e-12
PRIOR_SD = 0.35
MARGIN = 0.005

sys.path.insert(0, str(P13 / "scripts"))
from stage13_model import daily_compiled_kernels, kernel_array, topology_operators  # noqa: E402


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def current_memory_gib() -> tuple[float, float]:
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return math.nan, math.nan
    return counters.WorkingSetSize / 2**30, counters.PeakWorkingSetSize / 2**30


def check_memory(label: str) -> dict[str, object]:
    current, peak = current_memory_gib()
    if current > 16.0:
        raise MemoryError(f"RSS hard stop at {label}: {current:.3f} GiB")
    return {"label": label, "rss_gib": current, "peak_rss_gib": peak, "warning": bool(current > 12.0)}


def build_daily_bridge() -> tuple[pd.DataFrame, dict[str, float]]:
    columns = [
        "date", "reach_id", "local_fast_response_m3_s", "local_slow_response_m3_s",
        "routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_total_m3_s",
        "percolation_to_lower_mm_day", "soil_storage_mm", "upper_response_storage_mm",
        "lower_slow_storage_mm", "actual_aet_mm_day", "state_consistent_fast_fraction",
    ]
    hydro = pd.read_parquet(CANON_DAILY, columns=columns).sort_values(["reach_id", "date"]).reset_index(drop=True)
    hydro["date"] = pd.to_datetime(hydro.date)
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    static = pd.read_parquet(OLD_MONTHLY).sort_values(["reach_id", "year", "month"]).drop_duplicates("reach_id")
    static_columns = [
        "reach_id", "catchment_area_km2", "terminal_tree_id", "length_km",
        "bankfull_width_m", "bankfull_width_p05_m", "bankfull_width_p95_m",
        "bankfull_depth_m", "bankfull_depth_p05_m", "bankfull_depth_p95_m",
        "bankfull_geometry_nearest_distance_mean_m", "bankfull_geometry_samples_within_5km_fraction",
    ]
    static = static[static_columns]
    hydro = hydro.merge(forcing, on=["date", "reach_id"], validate="one_to_one").merge(
        static[["reach_id", "catchment_area_km2"]], on="reach_id", validate="many_to_one"
    )
    for state in ("soil_storage_mm", "upper_response_storage_mm", "lower_slow_storage_mm"):
        hydro[state.replace("_mm", "_start_mm")] = hydro.groupby("reach_id", sort=False)[state].shift(1)
        hydro.rename(columns={state: state.replace("_mm", "_end_mm")}, inplace=True)
    # January 2006 lacks a preceding canonical end state.  It is excluded from
    # the N compiler but remains part of the hydrologic model's own spin-up.
    hydro = hydro.loc[hydro.date.ge("2006-02-01")].copy()
    starts = ["soil_storage_start_mm", "upper_response_storage_start_mm", "lower_slow_storage_start_mm"]
    if hydro[starts].isna().any().any():
        raise RuntimeError("corrected daily bridge has missing start state")
    area = hydro.catchment_area_km2.to_numpy(float)
    hydro["local_fast_response_mm_day"] = hydro.local_fast_response_m3_s * 86400.0 / (area * 1000.0)
    hydro["local_slow_response_mm_day"] = hydro.local_slow_response_m3_s * 86400.0 / (area * 1000.0)
    hydro["soil_infiltration_mm_day"] = hydro.soil_storage_end_mm - hydro.soil_storage_start_mm + hydro.actual_aet_mm_day
    hydro["effective_excess_to_upper_mm_day"] = hydro.precipitation_daily_mm - hydro.soil_infiltration_mm_day
    for column in ("soil_infiltration_mm_day", "effective_excess_to_upper_mm_day"):
        # Arrow-backed pandas columns may expose a read-only NumPy view.
        # This bridge deliberately normalizes only numerical round-off, so
        # take an explicit writable copy before applying the in-place mask.
        values = hydro[column].to_numpy(dtype=float, copy=True)
        values[np.abs(values) <= 5.0e-12] = 0.0
        hydro[column] = values
    upper = hydro.upper_response_storage_start_mm + hydro.effective_excess_to_upper_mm_day - hydro.local_fast_response_mm_day - hydro.percolation_to_lower_mm_day - hydro.upper_response_storage_end_mm
    lower = hydro.lower_slow_storage_start_mm + hydro.percolation_to_lower_mm_day - hydro.local_slow_response_mm_day - hydro.lower_slow_storage_end_mm
    total = hydro.soil_storage_start_mm + hydro.upper_response_storage_start_mm + hydro.lower_slow_storage_start_mm + hydro.precipitation_daily_mm - hydro.actual_aet_mm_day - hydro.local_fast_response_mm_day - hydro.local_slow_response_mm_day - hydro.soil_storage_end_mm - hydro.upper_response_storage_end_mm - hydro.lower_slow_storage_end_mm
    metrics = {
        "upper_balance_max_abs_mm": float(upper.abs().max()),
        "lower_balance_max_abs_mm": float(lower.abs().max()),
        "total_balance_max_abs_mm": float(total.abs().max()),
        "minimum_infiltration_mm_day": float(hydro.soil_infiltration_mm_day.min()),
        "minimum_excess_mm_day": float(hydro.effective_excess_to_upper_mm_day.min()),
    }
    if max(metrics["upper_balance_max_abs_mm"], metrics["lower_balance_max_abs_mm"], metrics["total_balance_max_abs_mm"]) > 1.0e-9:
        raise RuntimeError(f"daily balance failure: {metrics}")
    hydro["year"] = hydro.date.dt.year.astype(np.int16)
    hydro["month"] = hydro.date.dt.month.astype(np.int8)
    return hydro.sort_values(["date", "reach_id"]).reset_index(drop=True), metrics


def build_monthly_bridge(daily: pd.DataFrame) -> pd.DataFrame:
    keys = ["reach_id", "year", "month"]
    sum_fields = [
        "precipitation_daily_mm", "pet_fao56_mm_day", "actual_aet_mm_day", "soil_infiltration_mm_day",
        "effective_excess_to_upper_mm_day", "percolation_to_lower_mm_day", "local_fast_response_mm_day", "local_slow_response_mm_day",
    ]
    sums = daily.groupby(keys, as_index=False)[sum_fields].sum()
    sums = sums.rename(columns={name: name.replace("_daily", "").replace("_mm_day", "_mm") for name in sum_fields})
    first = daily.groupby(keys, as_index=False)[["soil_storage_start_mm", "upper_response_storage_start_mm", "lower_slow_storage_start_mm"]].first()
    last = daily.groupby(keys, as_index=False)[["soil_storage_end_mm", "upper_response_storage_end_mm", "lower_slow_storage_end_mm"]].last()
    days = daily.groupby(keys, as_index=False).agg(days_in_month=("date", "nunique"))
    out = sums.merge(first, on=keys, validate="one_to_one").merge(last, on=keys, validate="one_to_one").merge(days, on=keys, validate="one_to_one")
    canonical = pd.read_parquet(CANON_MONTHLY)
    qfields = [
        *keys, "local_fast_response_m3_s", "local_slow_response_m3_s", "routed_fast_response_m3_s",
        "routed_slow_response_m3_s", "routed_total_m3_s", "state_consistent_fast_fraction",
        "channel_bankfull_travel_time_central_day", "channel_bankfull_travel_time_geometry_p05_day",
        "channel_bankfull_travel_time_geometry_p95_day",
    ]
    old = pd.read_parquet(OLD_MONTHLY).sort_values(["reach_id", "year", "month"]).drop_duplicates("reach_id")
    static_columns = [
        "reach_id", "catchment_area_km2", "terminal_tree_id", "length_km", "bankfull_width_m",
        "bankfull_width_p05_m", "bankfull_width_p95_m", "bankfull_depth_m", "bankfull_depth_p05_m",
        "bankfull_depth_p95_m", "bankfull_geometry_nearest_distance_mean_m",
        "bankfull_geometry_samples_within_5km_fraction",
    ]
    out = out.merge(canonical[qfields], on=keys, validate="one_to_one").merge(old[static_columns], on="reach_id", validate="many_to_one")
    out["month_seconds"] = out.days_in_month.astype(float) * 86400.0
    out["local_fast_response_volume_m3"] = out.local_fast_response_m3_s * out.month_seconds
    out["local_slow_response_volume_m3"] = out.local_slow_response_m3_s * out.month_seconds
    out["routed_fast_response_volume_m3"] = out.routed_fast_response_m3_s * out.month_seconds
    out["routed_slow_response_volume_m3"] = out.routed_slow_response_m3_s * out.month_seconds
    out["routed_total_water_volume_m3"] = out.routed_total_m3_s * out.month_seconds
    out["h1_exposure_day_per_m"] = out.length_km * 1000.0 * out.bankfull_width_m / np.maximum(out.routed_total_m3_s * 86400.0, EPS)
    return out.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)


def source_availability(start: tuple[int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    source = pd.read_parquet(SOURCE)
    source = source.loc[source.calendar_scenario.eq("CENTRAL")].copy()
    source = source.loc[(source.year > start[0]) | ((source.year == start[0]) & (source.month >= start[1]))]
    source = source.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    positive = source.fertilizer_kg_n + source.manure_kg_n + source.cropland_bnf_kg_n + source.atmospheric_deposition_kg_n
    demand = source.crop_demand_kg_n
    source["positive_source_total_kg_n"] = positive
    source["crop_demand_satisfied_kg_n"] = np.minimum(positive, demand)
    source["unmet_crop_demand_kg_n"] = np.maximum(demand - positive, 0.0)
    source["available_total_kg_n"] = np.maximum(positive - demand, 0.0)
    audit = {
        "positive_source_kg_n": float(positive.sum()),
        "crop_demand_satisfied_kg_n": float(source.crop_demand_satisfied_kg_n.sum()),
        "unmet_crop_demand_kg_n": float(source.unmet_crop_demand_kg_n.sum()),
        "available_kg_n": float(source.available_total_kg_n.sum()),
        "closure_error_kg_n": float((positive - source.crop_demand_satisfied_kg_n - source.available_total_kg_n).abs().max()),
    }
    return source, audit


def hydrologic_anomalies(monthly: pd.DataFrame) -> pd.DataFrame:
    features = {
        "upper": "upper_response_storage_start_mm", "lower": "lower_slow_storage_start_mm",
        "excess": "effective_excess_to_upper_mm", "fast_fraction": "state_consistent_fast_fraction",
    }
    h = monthly[["reach_id", "year", "month", *features.values()]].copy()
    for short, column in features.items():
        if short == "fast_fraction":
            values = np.log(np.clip(h[column], 1.0e-6, 1.0 - 1.0e-6) / np.clip(1.0 - h[column], 1.0e-6, 1.0))
        else:
            values = np.log1p(h[column].to_numpy(float))
        h[f"raw_{short}"] = values
        clim = h.loc[h.year.between(2010, 2020)].groupby(["reach_id", "month"])[f"raw_{short}"].mean().rename("clim")
        h = h.merge(clim, on=["reach_id", "month"], validate="many_to_one")
        h[f"anom_{short}"] = h[f"raw_{short}"] - h.clim
        sd = h.loc[h.year.between(2010, 2020)].groupby("reach_id")[f"anom_{short}"].std(ddof=0).replace(0, 1.0).rename("sd")
        h = h.merge(sd, on="reach_id", validate="many_to_one")
        h[f"z_anom_{short}"] = h[f"anom_{short}"] / h.sd
        h = h.drop(columns=["clim", "sd"])
    return h[["reach_id", "year", "month", *[f"z_anom_{name}" for name in features]]]


def observation_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=float)
    for column in ("station_key", "reach_id", "terminal_tree_id"):
        codes = pd.Categorical(frame[column]).codes
        counts = np.bincount(codes)
        weights += 1.0 / (3.0 * len(counts) * counts[codes])
    return weights


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class DynamicDeliveryRouter:
    def __init__(self, monthly: pd.DataFrame, kernels: pd.DataFrame, source: pd.DataFrame):
        hydro = monthly.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        src = source.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        keys = ["year", "month", "reach_id"]
        if not np.array_equal(hydro[keys].to_numpy(), src[keys].to_numpy()):
            raise RuntimeError("source/hydrology mismatch")
        self.reach_ids = np.sort(hydro.reach_id.unique().astype(int))
        times = hydro[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
        self.month_keys = [(int(y), int(m)) for y, m in times.itertuples(index=False)]
        self.shape = (len(times), len(self.reach_ids))
        self.rlookup = {int(r): i for i, r in enumerate(self.reach_ids)}
        self.tlookup = {key: i for i, key in enumerate(self.month_keys)}
        self.kernels = kernel_array(kernels.sort_values(["year", "month", "reach_id"]))
        self.input = src.available_total_kg_n.to_numpy(float).reshape(self.shape)
        anomaly = hydrologic_anomalies(hydro).sort_values(["year", "month", "reach_id"])
        columns = ["z_anom_upper", "z_anom_lower", "z_anom_excess", "z_anom_fast_fraction"]
        self.state_score = np.tanh(anomaly[columns].mean(axis=1).to_numpy(float).reshape(self.shape) / 2.0)
        self.local_water = (hydro.local_fast_response_volume_m3 + hydro.local_slow_response_volume_m3).to_numpy(float).reshape(self.shape)
        self.h = hydro.h1_exposure_day_per_m.to_numpy(float).reshape(self.shape)
        order, downstream, _ = topology_operators(TOPOLOGY, self.reach_ids)
        self.order_idx = [self.rlookup[r] for r in order]
        self.down_idx = {self.rlookup[r]: self.rlookup[d] for r, d in downstream.items()}
        self.water_inlet = np.zeros_like(self.local_water)
        for i in self.order_idx:
            outlet = self.water_inlet[:, i] + self.local_water[:, i]
            if i in self.down_idx:
                self.water_inlet[:, self.down_idx[i]] += outlet

    def carrier(self, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pi = sigmoid(alpha + beta * self.state_score)
        dpi = pi * (1.0 - pi)
        x = self.input * pi
        dx = np.stack([self.input * dpi, self.input * dpi * self.state_score], axis=2)
        n = len(self.reach_ids)
        zu, zl = np.zeros(n), np.zeros(n)
        dzu, dzl = np.zeros((n, 2)), np.zeros((n, 2))
        local, dlocal = np.zeros(self.shape), np.zeros((*self.shape, 2))
        stock = np.zeros((*self.shape, 2))
        for t in range(self.shape[0]):
            k = self.kernels[t * n:(t + 1) * n]
            result = np.einsum("roi,ri->ro", k, np.column_stack([zu, zl, x[t]]))
            derivatives = np.stack([dzu, dzl, dx[t]], axis=2)
            dresult = np.einsum("roi,rpi->rop", k, derivatives)
            zu, zl = result[:, 0], result[:, 1]
            dzu, dzl = dresult[:, 0], dresult[:, 1]
            local[t] = result[:, 2] + result[:, 3]
            dlocal[t] = dresult[:, 2] + dresult[:, 3]
            stock[t, :, 0], stock[t, :, 1] = zu, zl
        return local, dlocal, pi, stock[:, :, 0], stock[:, :, 1]

    def evaluate(self, obs: pd.DataFrame, theta: np.ndarray, derivatives: bool = True) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        alpha, beta, vf = map(float, theta)
        local, dlocal, pi, _, _ = self.carrier(alpha, beta)
        inlet, outlet = np.zeros_like(local), np.zeros_like(local)
        dinlet, doutlet = np.zeros((*local.shape, 3)), np.zeros((*local.shape, 3))
        survival, midpoint = np.exp(-vf * self.h), np.exp(-vf * self.h / 2.0)
        for i in self.order_idx:
            outlet[:, i] = inlet[:, i] * survival[:, i] + local[:, i] * midpoint[:, i]
            doutlet[:, i, :2] = dinlet[:, i, :2] * survival[:, i, None] + dlocal[:, i] * midpoint[:, i, None]
            doutlet[:, i, 2] = dinlet[:, i, 2] * survival[:, i] - inlet[:, i] * self.h[:, i] * survival[:, i] - 0.5 * local[:, i] * self.h[:, i] * midpoint[:, i]
            if i in self.down_idx:
                down = self.down_idx[i]
                inlet[:, down] += outlet[:, i]
                dinlet[:, down] += doutlet[:, i]
        ridx = obs.reach_id.astype(int).map(self.rlookup).to_numpy(int)
        tidx = np.fromiter((self.tlookup[(int(y), int(m))] for y, m in obs[["year", "month"]].itertuples(index=False)), dtype=int, count=len(obs))
        frac = obs.downstream_fraction_on_reach.to_numpy(float)
        h, sf, sm = self.h[tidx, ridx], np.exp(-vf * self.h[tidx, ridx] * frac), np.exp(-vf * self.h[tidx, ridx] * frac / 2.0)
        local_obs = local[tidx, ridx]
        load = inlet[tidx, ridx] * sf + frac * local_obs * sm
        water = self.water_inlet[tidx, ridx] + frac * self.local_water[tidx, ridx]
        pred = 1000.0 * load / np.maximum(water, EPS)
        if not derivatives:
            return pred, None, pi
        dload = np.zeros((len(obs), 3))
        dload[:, :2] = dinlet[tidx, ridx, :2] * sf[:, None] + frac[:, None] * dlocal[tidx, ridx] * sm[:, None]
        dload[:, 2] = dinlet[tidx, ridx, 2] * sf - inlet[tidx, ridx] * h * frac * sf - 0.5 * frac * local_obs * h * frac * sm
        return pred, 1000.0 * dload / np.maximum(water[:, None], EPS), pi


def fit_dynamic(router: DynamicDeliveryRouter, train: pd.DataFrame, multistart: bool) -> dict[str, object]:
    y = np.log1p(train.tn_mg_l.to_numpy(float))
    weights = observation_weights(train)
    nblocks = train.station_key.nunique() + train.reach_id.nunique() + train.terminal_tree_id.nunique()
    def value_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        pred, dpred, _ = router.evaluate(train, theta, True)
        assert dpred is not None
        error = np.log1p(np.maximum(pred, 0.0)) - y
        loss = float(np.sum(weights * error * error) + 0.5 * (theta[1] / PRIOR_SD) ** 2 / nblocks)
        grad = 2.0 * np.sum((weights * error)[:, None] * dpred / (1.0 + pred[:, None]), axis=0)
        grad[1] += theta[1] / (PRIOR_SD**2 * nblocks)
        return loss, grad
    starts = [np.array([-4.0, 0.0, 0.05]), np.array([-2.0, 0.25, 0.15]), np.array([-6.0, -0.25, 0.0])] if multistart else [np.array([-4.0, 0.0, 0.05])]
    fits = [minimize(value_grad, start, jac=True, method="L-BFGS-B", bounds=((-9.21, 9.21), (-1.0, 1.0), (0.0, 0.5)), options={"ftol": 1.0e-10, "gtol": 1.0e-7, "maxiter": 200, "maxls": 30}) for start in starts]
    valid = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    best = min(valid if valid else fits, key=lambda fit: float(fit.fun))
    _, _, pi = router.evaluate(train, best.x, False)
    return {
        "theta": np.asarray(best.x, dtype=float), "objective": float(best.fun), "success": bool(best.success),
        "message": str(best.message), "iterations": int(best.nit), "function_evaluations": int(best.nfev),
        "pi_min": float(pi.min()), "pi_median": float(np.median(pi)), "pi_max": float(pi.max()),
        "beta_boundary": bool(abs(best.x[1]) >= 1.0 - 1.0e-6),
        "vf_boundary": bool(best.x[2] <= 1.0e-7 or best.x[2] >= 0.5 - 1.0e-6),
    }


def build_observations() -> pd.DataFrame:
    obs = pd.read_parquet(OBS_AUDIT).loc[lambda x: x.formal_river_channel].copy()
    positions = pd.read_parquet(CURRENT_OBS)[["station_key", "reach_id", "downstream_fraction_on_reach", "station_to_assigned_reach_distance_m"]].drop_duplicates(["station_key", "reach_id"])
    obs = obs.merge(positions, on=["station_key", "reach_id"], how="left", validate="many_to_one")
    if obs.downstream_fraction_on_reach.isna().any():
        raise RuntimeError("river observation position missing")
    return obs.sort_values(["station_key", "year", "month"]).reset_index(drop=True)


def build_folds(obs: pd.DataFrame, scope: str) -> pd.DataFrame:
    temporal = [("T1", 2021, 2021, 2022), ("T2", 2021, 2022, 2023), ("T3", 2021, 2023, 2024)]
    rows: list[dict[str, object]] = []
    for fold_id, start, end, year in temporal:
        rows.append({"fold_id": fold_id, "holdout_type": "TEMPORAL", "holdout_id": "ALL", "train_start_year": start, "train_end_year": end, "evaluation_year": year})
        if scope == "full":
            evaluation = obs.loc[obs.year.eq(year)]
            for reach in sorted(evaluation.reach_id.unique()):
                rows.append({"fold_id": f"{fold_id}_LORO_R{int(reach):03d}", "holdout_type": "REACH", "holdout_id": str(int(reach)), "train_start_year": start, "train_end_year": end, "evaluation_year": year})
            for tree in sorted(evaluation.terminal_tree_id.unique()):
                rows.append({"fold_id": f"{fold_id}_LOTO_T{int(tree):03d}", "holdout_type": "TREE", "holdout_id": str(int(tree)), "train_start_year": start, "train_end_year": end, "evaluation_year": year})
    if scope == "full":
        rows.append({"fold_id": "NATURAL_EXPANSION_2021", "holdout_type": "FIRST_OBSERVED_2021", "holdout_id": "NEW_2021_STATIONS", "train_start_year": 2016, "train_end_year": 2020, "evaluation_year": 2021})
    return pd.DataFrame(rows)


def fold_frames(obs: pd.DataFrame, fold: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = obs.loc[obs.year.between(int(fold.train_start_year), int(fold.train_end_year))].copy()
    test = obs.loc[obs.year.eq(int(fold.evaluation_year))].copy()
    kind = str(fold.holdout_type)
    if kind == "REACH":
        holdout = int(fold.holdout_id); train = train.loc[train.reach_id.ne(holdout)]; test = test.loc[test.reach_id.eq(holdout)]
    elif kind == "TREE":
        holdout = int(fold.holdout_id); train = train.loc[train.terminal_tree_id.ne(holdout)]; test = test.loc[test.terminal_tree_id.eq(holdout)]
    elif kind == "FIRST_OBSERVED_2021":
        prior = set(obs.loc[obs.year.between(2016, 2020), "station_key"])
        new = set(obs.loc[obs.year.eq(2021), "station_key"]) - prior
        train = train.loc[~train.station_key.isin(new)]; test = test.loc[test.station_key.isin(new)]
    if train.empty or test.empty:
        raise RuntimeError(f"empty fold: {fold.fold_id}")
    return train, test


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y, p = frame.tn_mg_l.to_numpy(float), frame.pred_tn_mg_l.to_numpy(float)
    residual = p - y
    denominator = np.sum(np.square(y - y.mean()))
    station_rmse = []
    for _, block in frame.groupby("station_key"):
        station_rmse.append(np.sqrt(np.mean(np.square(np.log1p(block.pred_tn_mg_l) - np.log1p(block.tn_mg_l)))))
    return {
        "n": len(frame), "stations": frame.station_key.nunique(), "reaches": frame.reach_id.nunique(), "trees": frame.terminal_tree_id.nunique(),
        "rmse_mg_l": float(np.sqrt(np.mean(np.square(residual)))), "mae_mg_l": float(np.mean(np.abs(residual))),
        "nse": float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0 else math.nan,
        "r2": float(np.corrcoef(y, p)[0, 1] ** 2) if np.std(y) > 0 and np.std(p) > 0 else math.nan,
        "pbias_percent": float(100.0 * residual.sum() / y.sum()), "station_macro_log_rmse": float(np.mean(station_rmse)),
    }


def run_folds(router: DynamicDeliveryRouter, obs: pd.DataFrame, folds: pd.DataFrame, replicate: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    predictions, parameters, memory = [], [], []
    for index, fold in folds.iterrows():
        train, test = fold_frames(obs, fold)
        fit = fit_dynamic(router, train, multistart=str(fold.holdout_type) == "TEMPORAL")
        theta = np.asarray(fit.pop("theta"), dtype=float)
        pred, _, _ = router.evaluate(test, theta, False)
        output = test.copy()
        output["pred_tn_mg_l"] = pred
        output["fold_id"] = str(fold.fold_id); output["holdout_type"] = str(fold.holdout_type); output["holdout_id"] = str(fold.holdout_id); output["evaluation_year"] = int(fold.evaluation_year); output["replicate"] = replicate
        predictions.append(output)
        parameters.append({"fold_id": str(fold.fold_id), "holdout_type": str(fold.holdout_type), "holdout_id": str(fold.holdout_id), "train_start_year": int(fold.train_start_year), "train_end_year": int(fold.train_end_year), "evaluation_year": int(fold.evaluation_year), "alpha": float(theta[0]), "beta_D": float(theta[1]), "v_f_m_per_day": float(theta[2]), "train_rows": len(train), "test_rows": len(test), "replicate": replicate, **fit})
        if index % 20 == 0 or index == len(folds) - 1:
            memory.append(check_memory(f"{replicate}:{fold.fold_id}"))
            print(json.dumps({"replicate": replicate, "completed": str(fold.fold_id), "index": int(index + 1), "folds": len(folds), **memory[-1]}), flush=True)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(parameters), memory


def compare_replicates(a: pd.DataFrame, b: pd.DataFrame, pa: pd.DataFrame, pb: pd.DataFrame) -> dict[str, object]:
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    joined = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    pkeys = ["fold_id", "holdout_type", "holdout_id", "evaluation_year"]
    pj = pa.merge(pb, on=pkeys, suffixes=("_a", "_b"), validate="one_to_one")
    return {
        "prediction_rows_exact": len(joined) == len(a) == len(b),
        "prediction_max_abs_difference": float(np.max(np.abs(joined.pred_tn_mg_l_a - joined.pred_tn_mg_l_b))),
        "alpha_max_abs_difference": float(np.max(np.abs(pj.alpha_a - pj.alpha_b))),
        "beta_max_abs_difference": float(np.max(np.abs(pj.beta_D_a - pj.beta_D_b))),
        "vf_max_abs_difference": float(np.max(np.abs(pj.v_f_m_per_day_a - pj.v_f_m_per_day_b))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["temporal", "full"], default="temporal")
    args = parser.parse_args()
    require_runtime()
    parent = json.loads((P18 / "reports" / "stage18_final_validation.json").read_text(encoding="utf-8"))
    if parent["status"] != "PASS_STAGE18_READY_FOR_20260824_19":
        raise RuntimeError("stage18 parent not locked")
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    memory: list[dict[str, object]] = [check_memory("start")]
    daily, daily_audit = build_daily_bridge(); memory.append(check_memory("daily_bridge"))
    monthly = build_monthly_bridge(daily); memory.append(check_memory("monthly_bridge"))
    kernels = daily_compiled_kernels(daily); memory.append(check_memory("kernels"))
    source, source_audit = source_availability((2006, 2)); memory.append(check_memory("source"))
    obs = build_observations(); folds = build_folds(obs, args.scope)
    router_a = DynamicDeliveryRouter(monthly, kernels, source); memory.append(check_memory("router_a"))
    pred_a, par_a, mem_a = run_folds(router_a, obs, folds, "A"); memory.extend(mem_a)
    # A fresh router ensures no model-state cache is reused for the determinism replicate.
    router_b = DynamicDeliveryRouter(monthly.copy(), kernels.copy(), source.copy()); memory.append(check_memory("router_b"))
    pred_b, par_b, mem_b = run_folds(router_b, obs, folds, "B"); memory.extend(mem_b)
    determinism = compare_replicates(pred_a, pred_b, par_a, par_b)
    temporal = pred_a.loc[pred_a.holdout_type.eq("TEMPORAL")].copy()
    metric_rows = [{"scope": "corrected_river_temporal", "year": "ALL", **metrics(temporal)}]
    for year, block in temporal.groupby("year"):
        metric_rows.append({"scope": "corrected_river_temporal", "year": str(int(year)), **metrics(block)})
    old = pd.read_parquet(OLD_OOF).loc[lambda x: x.holdout_type.eq("TEMPORAL") & x.observation_domain.eq("river_channel")]
    old_keys = ["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    joined = old.merge(temporal, on=old_keys, suffixes=("_old", "_new"), validate="one_to_one")
    old_frame = joined[old_keys].copy(); old_frame["pred_tn_mg_l"] = joined.pred_tn_mg_l_old
    new_frame = joined[old_keys].copy(); new_frame["pred_tn_mg_l"] = joined.pred_tn_mg_l_new
    metric_rows.append({"scope": "old_stage16_common_river", "year": "ALL", **metrics(old_frame)})
    metric_rows.append({"scope": "corrected_common_river", "year": "ALL", **metrics(new_frame)})
    metric_frame = pd.DataFrame(metric_rows)
    if args.scope == "full":
        atomic_parquet(daily, OUT / "corrected_tn_bridge_daily_2006_2024.parquet")
        atomic_parquet(monthly, OUT / "corrected_tn_bridge_monthly_2006_2024.parquet")
        atomic_parquet(kernels, OUT / "corrected_daily_carrier_kernels_2006_2024.parquet")
        atomic_parquet(source, OUT / "corrected_source_availability_2006_2024.parquet")
    suffix = "full" if args.scope == "full" else "temporal"
    atomic_parquet(pred_a, OUT / f"corrected_m3_{suffix}_oof_predictions.parquet")
    atomic_parquet(par_a, OUT / f"corrected_m3_{suffix}_fold_parameters.parquet")
    atomic_parquet(metric_frame, OUT / f"corrected_m3_{suffix}_metrics.parquet")
    current, peak = current_memory_gib()
    validation = {
        "stage": "20260824_19", "scope": args.scope,
        "status": "PASS_STAGE19_TEMPORAL_PREFLIGHT" if args.scope == "temporal" else "PASS_STAGE19_READY_FOR_20260824_20",
        "daily_balance": daily_audit, "source_mass": source_audit, "determinism": determinism,
        "memory": {"samples": memory, "final_rss_gib": current, "peak_rss_gib": peak, "hard_stop_gib": 16},
        "counts": {"daily_rows": len(daily), "monthly_rows": len(monthly), "kernel_rows": len(kernels), "source_rows": len(source), "river_observations": len(obs), "folds": len(folds), "prediction_rows": len(pred_a)},
        "checks": {
            "hydrology_starts_2006": int(monthly.year.min()) == 2006,
            "ten_year_pre_evaluation_warmup": int(2016 - monthly.year.min()) == 10,
            "river_domain_only": bool(obs.formal_river_channel.all()),
            "all_fits_success": bool(par_a.success.all() and par_b.success.all()),
            "prediction_deterministic": determinism["prediction_max_abs_difference"] <= 1.0e-12,
            "parameter_deterministic": max(determinism["alpha_max_abs_difference"], determinism["beta_max_abs_difference"], determinism["vf_max_abs_difference"]) <= 1.0e-10,
            "memory_below_hard_stop": peak < 16.0,
        },
        "input_hashes": {str(path): sha256(path) for path in [CANON_DAILY, CANON_MONTHLY, FORCING, SOURCE, OBS_AUDIT, CONTRACT]},
        "authorized_successor": "20260824_20" if args.scope == "full" else None,
    }
    if not all(validation["checks"].values()):
        validation["status"] = "FAIL_STAGE19"
    write_json(REPORTS / f"stage19_{suffix}_validation.json", validation)
    row = metric_frame.loc[(metric_frame.scope == "corrected_river_temporal") & (metric_frame.year == "ALL")].iloc[0]
    report = f"""# 20260824_19 修正M3基线

运行范围：`{args.scope}`；状态：`{validation['status']}`。

- TN carrier从2006-02开始，2016首个评价年前有接近10年warm-up。
- 主评价只包含river_channel，共{obs.station_key.nunique()}站、{obs.reach_id.nunique()}个Reach。
- 2022–2024 temporal OOF：RMSE `{row.rmse_mg_l:.3f} mg/L`，NSE `{row.nse:.3f}`，R² `{row.r2:.3f}`，station-macro log-RMSE `{row.station_macro_log_rmse:.4f}`。
- 两次新router重跑预测最大差：`{determinism['prediction_max_abs_difference']:.3e}`；参数最大差：`{max(determinism['alpha_max_abs_difference'], determinism['beta_max_abs_difference'], determinism['vf_max_abs_difference']):.3e}`。
- 进程峰值RSS：`{peak:.3f} GiB`，低于16 GiB硬门禁。

本阶段没有增加fast/slow eta、C-Q、TN lag、Legacy或站点效应。完整nested结果通过后才授权`20260824_20`。
"""
    (REPORTS / f"technical_report_{suffix}.md").write_text(report, encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
