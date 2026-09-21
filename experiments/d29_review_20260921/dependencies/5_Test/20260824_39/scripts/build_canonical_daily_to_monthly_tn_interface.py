"""Compile frozen daily hydrology into exact monthly TN transfer operators.

This stage never reads TN observations and never refits hydrologic parameters.
It produces two central, model-ready products:

* exact 20260828_9 canonical hydrology for 2006-2024; and
* the frozen 20260824_26 reconstruction for 1961-2009 spliced to exact
  20260828_9 canonical hydrology for 2010-2024.

The lower-store release operator is composed from the daily fraction

    g_d = slow_d / (lower_start_d + percolation_d)

with an explicit zero-water gate.  No monthly flux is ever divided by a
month-end storage.  The upper/lower coupled coefficients are exact hydraulic
water-tracing diagnostics induced by the realised daily water fractions.  They
are not a candidate-independent upper-layer TN operator when contact parameters
vary.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_39"
REPORTS = RUN / "reports"
PROCESSED = ROOT / "0_reach_topology" / "data" / "processed" / "tn_legacy_long_history" / "hydrology"

CANONICAL_DAILY = ROOT / "5_Test" / "20260828_9" / "outputs" / "canonical_reach_daily_2006_2024.parquet"
CANONICAL_MONTHLY = ROOT / "5_Test" / "20260828_9" / "outputs" / "canonical_reach_monthly_2006_2024.parquet"
CANONICAL_MODEL = ROOT / "5_Test" / "20260828_9" / "outputs" / "parent_preserving_state_consistent_model.pt"
CANONICAL_LOCK = ROOT / "5_Test" / "20260828_9" / "reports" / "parent_preserving_product_lock.json"
CANONICAL_MANIFEST = ROOT / "5_Test" / "20260828_10" / "reports" / "canonical_tn_hydrology_manifest.json"
CANONICAL_FORCING = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"

HISTORICAL_FORCING = PROCESSED / "reconstructed_daily_hydrology_forcing_1961_2024.parquet"
HISTORICAL_MONTHLY = PROCESSED / "long_history_hydrology_monthly_1961_2024.parquet"
HISTORICAL_LOCK = ROOT / "5_Test" / "20260824_26" / "reports" / "historical_hydrology_bridge_audit.json"

CANONICAL_OUTPUT = PROCESSED / "canonical_daily_to_monthly_tn_transfer_2006_2024.parquet"
LONG_OUTPUT = PROCESSED / "long_history_daily_to_monthly_tn_transfer_1961_2024.parquet"
CENTRAL_METADATA_PATH = PROCESSED / "daily_to_monthly_tn_transfer_metadata.json"
QA_PATH = REPORTS / "canonical_daily_to_monthly_tn_interface_qa.json"
CONTRACT_PATH = REPORTS / "canonical_daily_to_monthly_tn_interface_contract.json"
REPORT_PATH = REPORTS / "daily_to_monthly_tn_interface.md"

STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE7OP = ROOT / "5_Test" / "20260827_7"
STAGE8 = ROOT / "5_Test" / "20260828_2"
STAGE5 = ROOT / "5_Test" / "20260828_5"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(ROOT / "5_Test" / "20260828_7" / "scripts"),
    str(STAGE5 / "scripts"),
    str(STAGE8 / "scripts"),
    str(ROOT / "5_Test" / "20260827_9" / "scripts"),
    str(ROOT / "5_Test" / "20260827_8" / "scripts"),
    str(STAGE7OP / "scripts"),
    str(STAGE3 / "scripts"),
    str(STAGE2 / "scripts"),
    str(OLD27 / "scripts"),
    str(OLD26 / "scripts"),
    str(OLD25 / "scripts"),
    str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from evaluate_component_development import periodic_spinup  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from learnable_state_sig2p import simulate_learnable_sig2p  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate  # noqa: E402
from run_stage3_diagnostics import route_instantaneous_np  # noqa: E402
from run_stage25 import Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


WARNING_GIB = 8.0
HARD_STOP_GIB = 12.0
EPS = 1.0e-12
REACH_IDS = np.arange(1, 231, dtype=np.int64)
FLOAT_TOL = 1.0e-10


@dataclass
class FrozenHydrology:
    saved: dict[str, Any]
    model: AlphaTwoPathCandidate
    physical: torch.Tensor
    static: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor
    score: torch.Tensor
    area_km2: np.ndarray
    order: list[int]
    downstream: dict[int, int]


def require_environment() -> None:
    if Path(sys.executable).parent.name.lower() != "sparrow":
        raise RuntimeError(f"This stage requires conda sparrow, got {sys.executable}")
    if torch.get_default_dtype() != torch.float64:
        raise RuntimeError("torch default dtype must be float64")


def rss_gib() -> float:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    if not get_memory(handle, ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return counters.WorkingSetSize / 1024**3


def memory_checkpoint(label: str, record: dict[str, float]) -> float:
    value = rss_gib()
    record[label] = value
    if value >= HARD_STOP_GIB:
        raise MemoryError(f"RSS {value:.3f} GiB reached the {HARD_STOP_GIB:g} GiB hard stop at {label}")
    if value >= WARNING_GIB:
        warnings.warn(f"RSS {value:.3f} GiB exceeded the {WARNING_GIB:g} GiB warning at {label}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False)
    os.replace(part, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(part, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(value, encoding="utf-8")
    os.replace(part, path)


def load_frozen_hydrology() -> FrozenHydrology:
    lock = json.loads(CANONICAL_LOCK.read_text(encoding="utf-8"))
    manifest = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    expected = lock.get("hashes", {})
    if lock.get("status") != "PARENT_PRESERVING_STATE_PRODUCT_LOCKED":
        raise RuntimeError("Canonical hydrology lock is not active")
    if manifest.get("status") != "CANONICAL_TN_HYDROLOGY_INTERFACE_RELEASED":
        raise RuntimeError("20260828_10 has not released the canonical TN hydrology interface")
    if manifest.get("structural_QA") != expected:
        raise RuntimeError("20260828_9 lock and 20260828_10 release manifest disagree")
    checks = {
        "model": sha256(CANONICAL_MODEL) == expected.get("state_consistent_model"),
        "daily": sha256(CANONICAL_DAILY) == expected.get("daily_2006_2024"),
        "monthly": sha256(CANONICAL_MONTHLY) == expected.get("monthly_2006_2024"),
        "forcing": sha256(CANONICAL_FORCING) == expected.get("forcing"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Canonical input hash mismatch: {checks}")

    saved = torch.load(CANONICAL_MODEL, map_location="cpu", weights_only=False)
    model = AlphaTwoPathCandidate(int(saved["seed"]))
    model.load_state_dict(saved["model_state"])
    model.eval()
    physical = raw_to_physical(saved["raw_parameters"].to(torch.float64).detach())
    static = torch.from_numpy(
        pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy()
    )
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center = torch.tensor(scaling["center"], dtype=torch.float64)
    scale = torch.tensor(scaling["scale"], dtype=torch.float64)
    score_path = Path(saved["regionalized_score_path"])
    if sha256(score_path) != expected.get("regionalized_score"):
        raise RuntimeError("Regionalized slow-score hash mismatch")
    score = torch.from_numpy(
        pd.read_parquet(score_path).sort_values("reach_id").regionalized_slow_score.to_numpy(np.float64).copy()
    )
    area_km2 = (
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id")
        .set_index("reach_id")
        .reindex(REACH_IDS)
        .catchment_area_km2.to_numpy(np.float64)
    )
    if not np.isfinite(area_km2).all() or np.any(area_km2 <= 0):
        raise RuntimeError("Catchment area is incomplete or nonpositive")
    order, downstream, _ = load_topology(TOPOLOGY, REACH_IDS)
    return FrozenHydrology(saved, model, physical, static, center, scale, score, area_km2, list(order), downstream)


def read_forcing(path: Path, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    forcing = pd.read_parquet(
        path,
        columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"],
    )
    forcing["date"] = pd.to_datetime(forcing["date"])
    if forcing.duplicated(["date", "reach_id"]).any():
        raise RuntimeError(f"Duplicate forcing date-reach rows: {path}")
    p = (
        forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm")
        .reindex(index=dates, columns=REACH_IDS)
        .to_numpy(np.float64)
    )
    pet = (
        forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day")
        .reindex(index=dates, columns=REACH_IDS)
        .to_numpy(np.float64)
    )
    if not np.isfinite(p).all() or not np.isfinite(pet).all() or np.any(p < 0) or np.any(pet < 0):
        raise RuntimeError(f"Forcing is incomplete, non-finite or negative: {path}")
    return p, pet


def canonical_initial_state(frozen: FrozenHydrology) -> tuple[np.ndarray, dict[str, Any]]:
    dates = pd.date_range("2006-01-01", "2009-12-31", freq="D")
    p_np, pet_np = read_forcing(CANONICAL_FORCING, dates)
    initial, audit = periodic_spinup(
        torch.from_numpy(p_np.copy()),
        torch.from_numpy(pet_np.copy()),
        frozen.physical,
        frozen.static,
        frozen.center,
        frozen.scale,
        frozen.model.gate,
        frozen.score,
        float(frozen.saved["lambda_S"]),
    )
    if not bool(audit["converged"]):
        raise RuntimeError(f"Canonical spin-up did not converge: {audit}")
    return initial.numpy(), audit


def reshape_canonical_daily(
    frame: pd.DataFrame,
    frozen: FrozenHydrology,
    initial: np.ndarray,
) -> dict[str, Any]:
    frame = frame.sort_values(["date", "reach_id"]).reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    dates = pd.DatetimeIndex(frame.date.drop_duplicates())
    expected = len(dates) * len(REACH_IDS)
    if len(frame) != expected or frame.duplicated(["date", "reach_id"]).any():
        raise RuntimeError("Canonical daily product is not complete date-by-reach grain")
    if not np.array_equal(frame.reach_id.to_numpy().reshape(len(dates), -1)[0], REACH_IDS):
        raise RuntimeError("Canonical reach order is not 1..230")
    shape = (len(dates), len(REACH_IDS))
    factor = 86.4 / frozen.area_km2[None, :]
    local_fast_q = frame.local_fast_response_m3_s.to_numpy(np.float64).reshape(shape)
    local_slow_q = frame.local_slow_response_m3_s.to_numpy(np.float64).reshape(shape)
    return {
        "dates": dates,
        "initial_upper": initial[:, 1].astype(np.float64, copy=False),
        "initial_lower": initial[:, 2].astype(np.float64, copy=False),
        "fast": local_fast_q * factor,
        "percolation": frame.percolation_to_lower_mm_day.to_numpy(np.float64).reshape(shape),
        "slow": local_slow_q * factor,
        "upper_end": frame.upper_response_storage_mm.to_numpy(np.float64).reshape(shape),
        "lower_end": frame.lower_slow_storage_mm.to_numpy(np.float64).reshape(shape),
        "routed_fast": frame.routed_fast_response_m3_s.to_numpy(np.float64).reshape(shape),
        "routed_slow": frame.routed_slow_response_m3_s.to_numpy(np.float64).reshape(shape),
        "routed_total": frame.routed_total_m3_s.to_numpy(np.float64).reshape(shape),
    }


def simulate_historical(
    frozen: FrozenHydrology,
    memory: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    dates = pd.date_range("1961-01-01", "2024-12-31", freq="D")
    p_np, pet_np = read_forcing(HISTORICAL_FORCING, dates)
    memory_checkpoint("historical_forcing_loaded", memory)
    p = torch.from_numpy(p_np.copy())
    pet = torch.from_numpy(pet_np.copy())
    spin_mask = np.asarray(dates.year <= 1990)
    initial, spin = periodic_spinup(
        p[spin_mask],
        pet[spin_mask],
        frozen.physical,
        frozen.static,
        frozen.center,
        frozen.scale,
        frozen.model.gate,
        frozen.score,
        float(frozen.saved["lambda_S"]),
    )
    if not bool(spin["converged"]):
        raise RuntimeError(f"Historical spin-up did not converge: {spin}")
    api3 = torch.from_numpy(antecedent_mean(p_np, 3))
    api30 = torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    with torch.no_grad():
        result = simulate_learnable_sig2p(
            p,
            pet,
            api3,
            api30,
            sin_doy,
            cos_doy,
            frozen.physical,
            initial,
            frozen.static,
            frozen.center,
            frozen.scale,
            frozen.model.gate,
            frozen.score,
            float(frozen.saved["lambda_S"]),
            collect_storage=True,
            collect_aet=False,
            collect_internal_fluxes=True,
        )
    if result.storage_mm is None or result.percolation_to_lower_mm_day is None:
        raise RuntimeError("Frozen historical simulation omitted required states")
    components = result.components_mm_day.numpy()
    storage = result.storage_mm.numpy()
    percolation = result.percolation_to_lower_mm_day.numpy()
    local_q = components * frozen.area_km2[None, :, None] * 1000.0 / 86400.0
    routed = route_instantaneous_np(local_q, frozen.order, frozen.downstream)
    arrays = {
        "dates": dates,
        "initial_upper": initial[:, 1].numpy(),
        "initial_lower": initial[:, 2].numpy(),
        "fast": components[:, :, 0],
        "percolation": percolation,
        "slow": components[:, :, 1],
        "upper_end": storage[:, :, 1],
        "lower_end": storage[:, :, 2],
        "routed_fast": routed[:, :, 0],
        "routed_slow": routed[:, :, 1],
        "routed_total": routed.sum(axis=2),
    }
    simulation = {
        "spinup": spin,
        "maximum_mass_error_mm": float(result.maximum_mass_error_mm),
    }
    memory_checkpoint("historical_simulation_complete", memory)
    return arrays, simulation


def _safe_fraction(numerator: np.ndarray, denominator: np.ndarray, zero_value: float) -> np.ndarray:
    result = np.full_like(denominator, zero_value, dtype=np.float64)
    positive = denominator > EPS
    result[positive] = numerator[positive] / denominator[positive]
    return result


def compile_monthly(arrays: dict[str, Any], provenance: str) -> tuple[pd.DataFrame, dict[str, float]]:
    dates: pd.DatetimeIndex = arrays["dates"]
    initial_upper = np.asarray(arrays["initial_upper"], dtype=np.float64)
    initial_lower = np.asarray(arrays["initial_lower"], dtype=np.float64)
    fast = np.asarray(arrays["fast"], dtype=np.float64)
    percolation = np.asarray(arrays["percolation"], dtype=np.float64)
    slow = np.asarray(arrays["slow"], dtype=np.float64)
    upper_end = np.asarray(arrays["upper_end"], dtype=np.float64)
    lower_end = np.asarray(arrays["lower_end"], dtype=np.float64)
    routed_fast = np.asarray(arrays["routed_fast"], dtype=np.float64)
    routed_slow = np.asarray(arrays["routed_slow"], dtype=np.float64)
    routed_total = np.asarray(arrays["routed_total"], dtype=np.float64)
    expected_shape = (len(dates), len(REACH_IDS))
    for name, value in [
        ("fast", fast),
        ("percolation", percolation),
        ("slow", slow),
        ("upper_end", upper_end),
        ("lower_end", lower_end),
        ("routed_fast", routed_fast),
        ("routed_slow", routed_slow),
        ("routed_total", routed_total),
    ]:
        if value.shape != expected_shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid {name}: shape={value.shape}, dtype={value.dtype}")
    if any(np.min(value) < -FLOAT_TOL for value in [fast, percolation, slow, upper_end, lower_end, routed_fast, routed_slow, routed_total]):
        raise RuntimeError("Hydrologic flux or state is materially negative")

    upper_start = np.concatenate([initial_upper[None, :], upper_end[:-1]], axis=0)
    lower_start = np.concatenate([initial_lower[None, :], lower_end[:-1]], axis=0)
    upper_available = fast + percolation + upper_end
    upper_excess = upper_available - upper_start
    lower_available = lower_start + percolation
    upper_zero = upper_available <= EPS
    lower_zero = lower_available <= EPS
    f = _safe_fraction(fast, upper_available, 0.0)
    p = _safe_fraction(percolation, upper_available, 0.0)
    c = _safe_fraction(upper_end, upper_available, 1.0)
    g = _safe_fraction(slow, lower_available, 0.0)
    lower_carry = np.where(lower_zero, 1.0, 1.0 - g)

    diagnostics = {
        "daily_upper_balance_max_abs_mm": float(np.max(np.abs(upper_available - fast - percolation - upper_end))),
        "daily_lower_balance_max_abs_mm": float(np.max(np.abs(lower_available - slow - lower_end))),
        "daily_upper_fraction_closure_max_abs": float(np.max(np.abs(f + p + c - 1.0))),
        "daily_lower_fraction_closure_max_abs": float(np.max(np.abs(g + lower_carry - 1.0))),
        "daily_upper_excess_min_mm": float(np.min(upper_excess)),
        "daily_zero_upper_gate_max_flux_mm": float(np.max(np.where(upper_zero, fast + percolation + upper_end, 0.0))),
        "daily_zero_lower_gate_max_flux_mm": float(np.max(np.where(lower_zero, slow + lower_end, 0.0))),
        "daily_routed_component_closure_max_abs_m3_s": float(np.max(np.abs(routed_fast + routed_slow - routed_total))),
    }
    if diagnostics["daily_lower_balance_max_abs_mm"] > FLOAT_TOL:
        raise RuntimeError(f"Lower daily balance failed: {diagnostics}")
    if diagnostics["daily_upper_fraction_closure_max_abs"] > FLOAT_TOL:
        raise RuntimeError(f"Upper daily fraction closure failed: {diagnostics}")
    if diagnostics["daily_lower_fraction_closure_max_abs"] > FLOAT_TOL:
        raise RuntimeError(f"Lower daily fraction closure failed: {diagnostics}")
    if diagnostics["daily_upper_excess_min_mm"] < -FLOAT_TOL:
        raise RuntimeError(f"Derived upper input is negative: {diagnostics}")
    upper_excess = np.maximum(upper_excess, 0.0)

    periods = dates.to_period("M")
    rows: list[pd.DataFrame] = []
    monthly_water_closure = 0.0
    monthly_operator_closure = 0.0
    lower_product_difference = 0.0
    for period in periods.unique():
        ii = np.flatnonzero(periods == period)
        first, last = int(ii[0]), int(ii[-1])
        n_days = len(ii)
        f_m, p_m, c_m, g_m = f[ii], p[ii], c[ii], g[ii]
        x_m = upper_excess[ii]
        perc_m = percolation[ii]

        # Exact coupled operator for a unit of upper or lower stock present at
        # the beginning of the month.
        uu = np.ones(len(REACH_IDS), dtype=np.float64)
        lu = np.zeros(len(REACH_IDS), dtype=np.float64)
        fu = np.zeros(len(REACH_IDS), dtype=np.float64)
        su = np.zeros(len(REACH_IDS), dtype=np.float64)
        ll = np.ones(len(REACH_IDS), dtype=np.float64)
        sl = np.zeros(len(REACH_IDS), dtype=np.float64)

        # Exact fate of new upper input under the realised within-month water
        # input timing, and exact fate of an exogenous lower input distributed
        # in proportion to daily percolation.
        ux = np.zeros(len(REACH_IDS), dtype=np.float64)
        lx = np.zeros(len(REACH_IDS), dtype=np.float64)
        fx = np.zeros(len(REACH_IDS), dtype=np.float64)
        sx = np.zeros(len(REACH_IDS), dtype=np.float64)
        lp = np.zeros(len(REACH_IDS), dtype=np.float64)
        sp = np.zeros(len(REACH_IDS), dtype=np.float64)

        for day in range(n_days):
            # Initial-upper basis.
            fu += f_m[day] * uu
            lower_from_u = lu + p_m[day] * uu
            su += g_m[day] * lower_from_u
            lu = (1.0 - g_m[day]) * lower_from_u
            uu = c_m[day] * uu

            # Initial-lower basis.
            sl += g_m[day] * ll
            ll = (1.0 - g_m[day]) * ll

            # Realised upper-input timing basis.
            upper_input_available = ux + x_m[day]
            fx += f_m[day] * upper_input_available
            lower_input_available = lx + p_m[day] * upper_input_available
            sx += g_m[day] * lower_input_available
            lx = (1.0 - g_m[day]) * lower_input_available
            ux = c_m[day] * upper_input_available

            # Lower input proportional to actual daily percolation.
            lower_perc_available = lp + perc_m[day]
            sp += g_m[day] * lower_perc_available
            lp = (1.0 - g_m[day]) * lower_perc_available

        x_total = x_m.sum(axis=0)
        perc_total = perc_m.sum(axis=0)
        x_positive = x_total > EPS
        perc_positive = perc_total > EPS
        x_fast_fraction = np.zeros(len(REACH_IDS), dtype=np.float64)
        x_slow_fraction = np.zeros(len(REACH_IDS), dtype=np.float64)
        x_upper_fraction = np.zeros(len(REACH_IDS), dtype=np.float64)
        x_lower_fraction = np.zeros(len(REACH_IDS), dtype=np.float64)
        x_fast_fraction[x_positive] = fx[x_positive] / x_total[x_positive]
        x_slow_fraction[x_positive] = sx[x_positive] / x_total[x_positive]
        x_upper_fraction[x_positive] = ux[x_positive] / x_total[x_positive]
        x_lower_fraction[x_positive] = lx[x_positive] / x_total[x_positive]
        perc_slow_fraction = np.zeros(len(REACH_IDS), dtype=np.float64)
        perc_carry_fraction = np.zeros(len(REACH_IDS), dtype=np.float64)
        perc_slow_fraction[perc_positive] = sp[perc_positive] / perc_total[perc_positive]
        perc_carry_fraction[perc_positive] = lp[perc_positive] / perc_total[perc_positive]

        fast_sum = fast[ii].sum(axis=0)
        slow_sum = slow[ii].sum(axis=0)
        perc_sum = percolation[ii].sum(axis=0)
        upper_start_month = upper_start[first]
        lower_start_month = lower_start[first]
        upper_end_month = upper_end[last]
        lower_end_month = lower_end[last]
        total_input = upper_start_month + lower_start_month + x_total
        total_output = fast_sum + slow_sum + upper_end_month + lower_end_month
        monthly_water_closure = max(monthly_water_closure, float(np.max(np.abs(total_input - total_output))))
        monthly_operator_closure = max(
            monthly_operator_closure,
            float(np.max(np.abs(fu + su + uu + lu - 1.0))),
            float(np.max(np.abs(sl + ll - 1.0))),
            float(np.max(np.where(x_positive, np.abs(x_fast_fraction + x_slow_fraction + x_upper_fraction + x_lower_fraction - 1.0), 0.0))),
            float(np.max(np.where(perc_positive, np.abs(perc_slow_fraction + perc_carry_fraction - 1.0), 0.0))),
        )
        daily_product_release = 1.0 - np.prod(1.0 - g_m, axis=0)
        lower_product_difference = max(lower_product_difference, float(np.max(np.abs(sl - daily_product_release))))

        routed_fast_volume = routed_fast[ii].sum(axis=0) * 86400.0
        routed_slow_volume = routed_slow[ii].sum(axis=0) * 86400.0
        routed_total_volume = routed_total[ii].sum(axis=0) * 86400.0
        frame = pd.DataFrame(
            {
                "reach_id": REACH_IDS,
                "year": int(period.year),
                "month": int(period.month),
                "days_in_month": n_days,
                "hydrology_provenance": provenance,
                "upper_storage_start_mm": upper_start_month,
                "upper_storage_mean_mm": 0.5 * (upper_start[ii] + upper_end[ii]).mean(axis=0),
                "upper_storage_daily_start_mean_mm": upper_start[ii].mean(axis=0),
                "upper_storage_daily_end_mean_mm": upper_end[ii].mean(axis=0),
                "upper_storage_end_mm": upper_end_month,
                "lower_storage_start_mm": lower_start_month,
                "lower_storage_mean_mm": 0.5 * (lower_start[ii] + lower_end[ii]).mean(axis=0),
                "lower_storage_daily_start_mean_mm": lower_start[ii].mean(axis=0),
                "lower_storage_daily_end_mean_mm": lower_end[ii].mean(axis=0),
                "lower_storage_end_mm": lower_end_month,
                "upper_excess_input_water_mm_month": x_total,
                "local_fast_water_mm_month": fast_sum,
                "percolation_water_mm_month": perc_sum,
                "local_slow_water_mm_month": slow_sum,
                "routed_fast_flow_mean_m3_s": routed_fast[ii].mean(axis=0),
                "routed_slow_flow_mean_m3_s": routed_slow[ii].mean(axis=0),
                "routed_total_flow_mean_m3_s": routed_total[ii].mean(axis=0),
                "routed_fast_water_volume_m3": routed_fast_volume,
                "routed_slow_water_volume_m3": routed_slow_volume,
                "routed_total_water_volume_m3": routed_total_volume,
                "exact_lower_release_fraction": sl,
                "lower_start_carry_fraction": ll,
                "daily_lower_release_fraction_mean": g_m.mean(axis=0),
                "daily_lower_release_fraction_min": g_m.min(axis=0),
                "daily_lower_release_fraction_max": g_m.max(axis=0),
                "hydraulic_upper_start_to_fast_fraction": fu,
                "hydraulic_upper_start_to_slow_fraction": su,
                "hydraulic_upper_start_to_upper_end_fraction": uu,
                "hydraulic_upper_start_to_lower_end_fraction": lu,
                "upper_input_positive": x_positive,
                "upper_water_active": (~upper_zero[ii]).any(axis=0),
                "hydraulic_upper_input_to_fast_fraction_if_daily_excess": x_fast_fraction,
                "hydraulic_upper_input_to_slow_fraction_if_daily_excess": x_slow_fraction,
                "hydraulic_upper_input_to_upper_end_fraction_if_daily_excess": x_upper_fraction,
                "hydraulic_upper_input_to_lower_end_fraction_if_daily_excess": x_lower_fraction,
                "percolation_input_positive": perc_positive,
                "lower_water_active": (~lower_zero[ii]).any(axis=0),
                "routed_water_active": (routed_total[ii] > EPS).any(axis=0),
                "hydraulic_percolation_weighted_to_slow_same_month_fraction": perc_slow_fraction,
                "hydraulic_percolation_weighted_to_lower_end_fraction": perc_carry_fraction,
                "upper_zero_available_days": upper_zero[ii].sum(axis=0).astype(np.int16),
                "lower_zero_available_days": lower_zero[ii].sum(axis=0).astype(np.int16),
                "local_fast_zero_days": (fast[ii] <= EPS).sum(axis=0).astype(np.int16),
                "percolation_zero_days": (percolation[ii] <= EPS).sum(axis=0).astype(np.int16),
                "local_slow_zero_days": (slow[ii] <= EPS).sum(axis=0).astype(np.int16),
                "routed_total_zero_days": (routed_total[ii] <= EPS).sum(axis=0).astype(np.int16),
                "lower_release_active_days": (g_m > EPS).sum(axis=0).astype(np.int16),
                "upper_zero_water_gate_compliant": np.where(
                    upper_zero[ii], fast[ii] + percolation[ii] + upper_end[ii], 0.0
                ).max(axis=0)
                <= FLOAT_TOL,
                "lower_zero_water_gate_compliant": np.where(
                    lower_zero[ii], slow[ii] + lower_end[ii], 0.0
                ).max(axis=0)
                <= FLOAT_TOL,
                "monthly_water_balance_error_mm": total_input - total_output,
                "upper_initial_operator_closure_error": fu + su + uu + lu - 1.0,
                "lower_initial_operator_closure_error": sl + ll - 1.0,
            }
        )
        rows.append(frame)

    output = pd.concat(rows, ignore_index=True).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    diagnostics.update(
        {
            "monthly_water_balance_max_abs_mm": monthly_water_closure,
            "monthly_operator_closure_max_abs": monthly_operator_closure,
            "lower_release_product_max_abs_difference": lower_product_difference,
        }
    )
    return output, diagnostics


def attach_channel_hydraulics(output: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys = ["reach_id", "year", "month"]
    columns = [
        "channel_bankfull_travel_time_central_day",
        "channel_bankfull_travel_time_geometry_p05_day",
        "channel_bankfull_travel_time_geometry_p95_day",
    ]
    missing = [column for column in columns if column not in reference.columns]
    if missing:
        raise RuntimeError(f"Hydraulic reference is missing columns: {missing}")
    right = reference[keys + columns].copy()
    if right.duplicated(keys).any():
        raise RuntimeError("Hydraulic reference contains duplicate reach-month rows")
    merged = output.merge(right, on=keys, how="left", validate="one_to_one")
    if merged[columns].isna().any().any():
        raise RuntimeError("Hydraulic travel-time join lost reach-month coverage")
    return merged


def compare_to_reference(output: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    keys = ["reach_id", "year", "month"]
    columns = [
        "routed_fast_response_m3_s",
        "routed_slow_response_m3_s",
        "routed_total_m3_s",
        "percolation_to_lower_mm_day",
        "upper_response_storage_mm",
        "lower_slow_storage_mm",
    ]
    right = reference[keys + columns].copy()
    merged = output.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(output):
        raise RuntimeError("Reference comparison did not cover all output rows")
    days = merged.days_in_month.to_numpy(np.float64)
    return {
        "routed_fast_mean_max_abs_m3_s": float(np.max(np.abs(merged.routed_fast_flow_mean_m3_s - merged.routed_fast_response_m3_s))),
        "routed_slow_mean_max_abs_m3_s": float(np.max(np.abs(merged.routed_slow_flow_mean_m3_s - merged.routed_slow_response_m3_s))),
        "routed_total_mean_max_abs_m3_s": float(np.max(np.abs(merged.routed_total_flow_mean_m3_s - merged.routed_total_m3_s))),
        "percolation_month_sum_vs_mean_max_abs_mm": float(np.max(np.abs(merged.percolation_water_mm_month - merged.percolation_to_lower_mm_day * days))),
        "upper_month_end_max_abs_mm": float(np.max(np.abs(merged.upper_storage_end_mm - merged.upper_response_storage_mm))),
        "lower_month_end_max_abs_mm": float(np.max(np.abs(merged.lower_storage_end_mm - merged.lower_slow_storage_mm))),
    }


def validate_grain(frame: pd.DataFrame, start_year: int, end_year: int) -> dict[str, Any]:
    expected_rows = len(REACH_IDS) * (end_year - start_year + 1) * 12
    float_columns = frame.select_dtypes(include=["floating"]).columns
    return {
        "rows": int(len(frame)),
        "expected_rows": int(expected_rows),
        "row_count_exact": len(frame) == expected_rows,
        "reach_count": int(frame.reach_id.nunique()),
        "reach_count_exact": frame.reach_id.nunique() == len(REACH_IDS),
        "year_min": int(frame.year.min()),
        "year_max": int(frame.year.max()),
        "month_domain_exact": sorted(frame.month.unique().tolist()) == list(range(1, 13)),
        "duplicate_keys": int(frame.duplicated(["reach_id", "year", "month"]).sum()),
        "float64_only": all(frame[column].dtype == np.dtype("float64") for column in float_columns),
        "numeric_finite": bool(np.isfinite(frame.select_dtypes(include=["number"]).to_numpy(np.float64)).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Build and validate only the exact 2006-2024 canonical product.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    require_environment()
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    memory: dict[str, float] = {}
    memory_checkpoint("start", memory)

    frozen = load_frozen_hydrology()
    canonical_initial, canonical_spin = canonical_initial_state(frozen)
    canonical_daily = pd.read_parquet(CANONICAL_DAILY)
    memory_checkpoint("canonical_daily_loaded", memory)
    canonical_arrays = reshape_canonical_daily(canonical_daily, frozen, canonical_initial)
    del canonical_daily
    canonical_interface, canonical_diagnostics = compile_monthly(
        canonical_arrays, "exact_20260828_9_canonical_daily"
    )
    canonical_reference = pd.read_parquet(CANONICAL_MONTHLY)
    canonical_interface = attach_channel_hydraulics(canonical_interface, canonical_reference)
    canonical_comparison = compare_to_reference(canonical_interface, canonical_reference)
    canonical_grain = validate_grain(canonical_interface, 2006, 2024)
    atomic_parquet(canonical_interface, CANONICAL_OUTPUT)
    memory_checkpoint("canonical_interface_written", memory)

    long_interface: pd.DataFrame | None = None
    historical_diagnostics: dict[str, float] | None = None
    historical_comparison: dict[str, float] | None = None
    historical_simulation: dict[str, Any] | None = None
    long_grain: dict[str, Any] | None = None
    if not args.canonical_only:
        historical_lock = json.loads(HISTORICAL_LOCK.read_text(encoding="utf-8"))
        if historical_lock.get("status") != "PASS_HISTORICAL_HYDROLOGY_BRIDGE":
            raise RuntimeError("Frozen historical bridge is not in PASS state")
        if sha256(HISTORICAL_FORCING) != historical_lock["hashes"]["forcing"]:
            raise RuntimeError("Historical forcing hash does not match the frozen bridge")
        if sha256(HISTORICAL_MONTHLY) != historical_lock["hashes"]["production"]:
            raise RuntimeError("Historical monthly production hash does not match the frozen bridge")
        historical_arrays, historical_simulation = simulate_historical(frozen, memory)
        historical_all, historical_diagnostics = compile_monthly(
            historical_arrays, "frozen_20260824_26_daily_reconstruction"
        )
        del historical_arrays
        historical_reference = pd.read_parquet(HISTORICAL_MONTHLY)
        historical_part = historical_all.loc[historical_all.year <= 2009].copy()
        historical_comparison = compare_to_reference(
            historical_part,
            historical_reference.loc[historical_reference.year <= 2009],
        )
        canonical_part = canonical_interface.loc[canonical_interface.year >= 2010].copy()
        long_interface = (
            pd.concat([historical_part, canonical_part], ignore_index=True)
            .sort_values(["reach_id", "year", "month"])
            .reset_index(drop=True)
        )
        # Travel-time values in historical_part were not attached yet.  Use the
        # frozen bridge for 1961-2009, and keep exact canonical values thereafter.
        travel_columns = [column for column in canonical_interface.columns if column.startswith("channel_bankfull_travel_time_")]
        long_interface.drop(columns=travel_columns, inplace=True, errors="ignore")
        long_interface = attach_channel_hydraulics(long_interface, historical_reference)
        long_grain = validate_grain(long_interface, 1961, 2024)
        atomic_parquet(long_interface, LONG_OUTPUT)
        memory_checkpoint("long_interface_written", memory)

    canonical_gate = {
        "grain": all(
            [
                canonical_grain["row_count_exact"],
                canonical_grain["reach_count_exact"],
                canonical_grain["month_domain_exact"],
                canonical_grain["duplicate_keys"] == 0,
                canonical_grain["float64_only"],
                canonical_grain["numeric_finite"],
            ]
        ),
        "daily_lower_balance": canonical_diagnostics["daily_lower_balance_max_abs_mm"] <= FLOAT_TOL,
        "daily_fraction_closure": max(
            canonical_diagnostics["daily_upper_fraction_closure_max_abs"],
            canonical_diagnostics["daily_lower_fraction_closure_max_abs"],
        )
        <= FLOAT_TOL,
        "daily_zero_water_gates": max(
            canonical_diagnostics["daily_zero_upper_gate_max_flux_mm"],
            canonical_diagnostics["daily_zero_lower_gate_max_flux_mm"],
        )
        <= FLOAT_TOL,
        "daily_nonnegative_derived_upper_input": canonical_diagnostics["daily_upper_excess_min_mm"] >= -FLOAT_TOL,
        "daily_routed_component_closure": canonical_diagnostics["daily_routed_component_closure_max_abs_m3_s"] <= FLOAT_TOL,
        "monthly_water_balance": canonical_diagnostics["monthly_water_balance_max_abs_mm"] <= FLOAT_TOL,
        "monthly_operator_closure": canonical_diagnostics["monthly_operator_closure_max_abs"] <= FLOAT_TOL,
        "lower_release_exact_product": canonical_diagnostics["lower_release_product_max_abs_difference"] <= FLOAT_TOL,
        "official_monthly_reproduction": max(canonical_comparison.values()) <= FLOAT_TOL,
    }
    long_gate: dict[str, bool] | None = None
    if long_interface is not None and historical_diagnostics is not None and historical_comparison is not None and long_grain is not None:
        long_gate = {
            "grain": all(
                [
                    long_grain["row_count_exact"],
                    long_grain["reach_count_exact"],
                    long_grain["month_domain_exact"],
                    long_grain["duplicate_keys"] == 0,
                    long_grain["float64_only"],
                    long_grain["numeric_finite"],
                ]
            ),
            "daily_lower_balance": historical_diagnostics["daily_lower_balance_max_abs_mm"] <= FLOAT_TOL,
            "daily_fraction_closure": max(
                historical_diagnostics["daily_upper_fraction_closure_max_abs"],
                historical_diagnostics["daily_lower_fraction_closure_max_abs"],
            )
            <= FLOAT_TOL,
            "daily_zero_water_gates": max(
                historical_diagnostics["daily_zero_upper_gate_max_flux_mm"],
                historical_diagnostics["daily_zero_lower_gate_max_flux_mm"],
            )
            <= FLOAT_TOL,
            "daily_nonnegative_derived_upper_input": historical_diagnostics["daily_upper_excess_min_mm"] >= -FLOAT_TOL,
            "daily_routed_component_closure": historical_diagnostics["daily_routed_component_closure_max_abs_m3_s"] <= FLOAT_TOL,
            "monthly_water_balance": historical_diagnostics["monthly_water_balance_max_abs_mm"] <= FLOAT_TOL,
            "monthly_operator_closure": historical_diagnostics["monthly_operator_closure_max_abs"] <= FLOAT_TOL,
            "lower_release_exact_product": historical_diagnostics["lower_release_product_max_abs_difference"] <= FLOAT_TOL,
            "frozen_historical_monthly_reproduction": max(historical_comparison.values()) <= FLOAT_TOL,
            "historical_model_mass_balance": bool(historical_simulation and historical_simulation["maximum_mass_error_mm"] <= 1.0e-9),
        }
    all_pass = all(canonical_gate.values()) and (long_gate is None or all(long_gate.values()))
    max_rss = max(memory.values())
    memory_gate = max_rss < HARD_STOP_GIB
    all_pass = all_pass and memory_gate

    contract = {
        "contract_id": "CANONICAL_DAILY_TO_MONTHLY_TN_TRANSFER_V1",
        "status": "LOCKED",
        "hydrology_parameters_fitted_by_tn": False,
        "tn_observations_read": False,
        "dtype": "float64",
        "environment": "conda sparrow",
        "memory": {"warning_gib": WARNING_GIB, "hard_stop_gib": HARD_STOP_GIB},
        "monthly_store_semantics": {
            "start": "pre-update storage at the beginning of the first day",
            "mean": "mean of the daily trapezoid 0.5*(pre-update + post-update storage)",
            "end": "post-update storage at the end of the last day",
        },
        "lower_operator": {
            "daily_fraction": "g_d = slow_d / (lower_start_d + percolation_d) when available water > 1e-12 mm; otherwise g_d=0",
            "exact_monthly_release": "1 - product_d(1-g_d)",
            "zero_water_gate": "available<=1e-12 => g=0 and slow=end=0 within numerical tolerance",
        },
        "upper_operator": {
            "daily_available": "fast_d + percolation_d + upper_end_d",
            "daily_fractions": "fast/available, percolation/available, upper_end/available",
            "zero_water_gate": "available<=1e-12 => fast=percolation=end=0 and hypothetical carry fraction=1",
            "role": "hydraulic diagnostic and sufficient frozen driver only; not a candidate-independent TN transfer operator",
        },
        "formal_tn_forward_requirement": {
            "daily_recompilation_required": True,
            "reason": "upper-layer N mobilization still depends on candidate alpha/beta contact equations",
            "rule": "each candidate must use frozen daily stores/flows and its own alpha/beta to compile N transport; monthly hydraulic fractions cannot replace that forward pass unless alpha/beta are separately frozen",
            "monthly_table_sufficient_by_itself": False,
            "daily_sources": {
                "exact_2006_2024": str(CANONICAL_DAILY),
                "historical_1961_2009": "deterministic reconstruction by this script from the frozen 20260824_26 forcing and the unchanged 20260828_9 model",
            },
            "lower_exception": "exact_lower_release_fraction is candidate-independent only for the registered well-mixed lower dissolved state; the percolation-weighted coefficient additionally assumes N input timing proportional to daily water percolation",
        },
        "forbidden_calculation": "month-end storage divided into a monthly flux",
        "long_history_splice": "1961-2009 frozen daily reconstruction; 2010-2024 exact 20260828_9 canonical daily hydrology",
        "channel_hydraulics": "Andreadis bankfull width/depth and topology reach length are used only with canonical simulated routed flow; reference discharge is not used",
        "station_position_boundary": "routed flow and volume columns are reach-outlet values; a partial-reach station must use upstream routed flow plus station_fraction times local flow for both the concentration denominator and any C-Q covariate",
        "outputs": {"canonical": str(CANONICAL_OUTPUT), "long_history": None if args.canonical_only else str(LONG_OUTPUT)},
    }
    atomic_json(CONTRACT_PATH, contract)
    qa = {
        "stage": "20260824_39",
        "status": "PASS_CANONICAL_DAILY_TO_MONTHLY_TN_INTERFACE" if all_pass else "FAIL_CANONICAL_DAILY_TO_MONTHLY_TN_INTERFACE",
        "checks": {
            "canonical": canonical_gate,
            "long_history": long_gate,
            "no_hydrology_parameter_refit": True,
            "TN_not_read": True,
            "month_end_storage_division_forbidden": True,
            "monthly_table_not_used_as_complete_upper_TN_forward": True,
            "canonical_release_manifest_verified": True,
            "andreadis_reference_discharge_not_used": True,
            "rss_below_warning": max_rss < WARNING_GIB,
            "rss_below_hard_stop": memory_gate,
        },
        "canonical": {
            "grain": canonical_grain,
            "diagnostics": canonical_diagnostics,
            "reference_reproduction": canonical_comparison,
            "spinup": canonical_spin,
        },
        "long_history": None
        if long_interface is None
        else {
            "grain": long_grain,
            "diagnostics": historical_diagnostics,
            "reference_reproduction_1961_2009": historical_comparison,
            "simulation": historical_simulation,
        },
        "paths": {"contract": str(CONTRACT_PATH), "canonical": str(CANONICAL_OUTPUT), "long_history": None if args.canonical_only else str(LONG_OUTPUT)},
        "hashes": {
            "builder_script": sha256(Path(__file__).resolve()),
            "interface_contract": sha256(CONTRACT_PATH),
            "canonical_daily": sha256(CANONICAL_DAILY),
            "canonical_monthly": sha256(CANONICAL_MONTHLY),
            "canonical_model": sha256(CANONICAL_MODEL),
            "canonical_release_manifest": sha256(CANONICAL_MANIFEST),
            "canonical_forcing": sha256(CANONICAL_FORCING),
            "historical_forcing": None if args.canonical_only else sha256(HISTORICAL_FORCING),
            "historical_monthly": None if args.canonical_only else sha256(HISTORICAL_MONTHLY),
            "canonical_output": sha256(CANONICAL_OUTPUT),
            "long_output": None if args.canonical_only else sha256(LONG_OUTPUT),
        },
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "rss_gib_by_checkpoint": memory,
            "maximum_recorded_rss_gib": max_rss,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    atomic_json(QA_PATH, qa)
    central_metadata = {
        "dataset_id": "TN_HYDROLOGY_DAILY_TO_MONTHLY_TRANSFER_V1",
        "status": qa["status"],
        "created_by": str(Path(__file__).resolve()),
        "contract": contract,
        "qa_report": str(QA_PATH),
        "source_hashes": {
            key: value
            for key, value in qa["hashes"].items()
            if key not in {"canonical_output", "long_output"}
        },
        "output_hashes": {
            "canonical": qa["hashes"]["canonical_output"],
            "long_history": qa["hashes"]["long_output"],
        },
        "grain": "one row per reach_id x year x month",
        "use_boundary": {
            "monthly_product": "evaluation summaries, state initialization diagnostics, and fixed lower-store release when its mixing assumption is retained",
            "upper_hydraulic_fraction_columns": "diagnostic water-tracing coefficients only",
            "formal_candidate_forward": "must recompile daily N transport from frozen daily hydrology using the candidate alpha/beta equations",
        },
        "units": {
            "stores_and_local_fluxes": "mm or mm month-1 as named",
            "routed_flow": "m3 s-1",
            "routed_volume": "m3 month-1",
            "travel_time": "day",
            "operator_coefficients": "dimensionless",
        },
    }
    atomic_json(CENTRAL_METADATA_PATH, central_metadata)
    report = f"""# Canonical 日水文到 TN 月转移接口

## 结论

状态：`{qa['status']}`。

本接口没有读取 TN、没有重拟合水文参数，也没有用月末库存除以整月流量。它从每日上层/下层水库状态与快流、下渗、慢流通量构造精确的月水力追踪摘要与下层完全混合释放算子；上层 TN 仍须按候选逐日重编。

## 正式产品

- 2006–2024 exact canonical：`{CANONICAL_OUTPUT}`
- 1961–2024 长历史：`{'未在 canonical-only 模式生成' if args.canonical_only else LONG_OUTPUT}`

长历史口径固定为 1961–2009 冻结重建、2010–2024 exact `20260828_9` canonical；2010 接口不重置 TN 状态。

## 状态与通量口径

- `*_storage_start_mm`：月首第一日更新前状态；
- `*_storage_mean_mm`：每日更新前后状态梯形均值的月平均；
- `*_storage_end_mm`：月末最后一日更新后状态；
- 快流、下渗和慢流均保存月累计水深，routed 水保存月均流量和月体积；
- `exact_lower_release_fraction = 1 - product(1-g_d)`，其中 `g_d=slow/(lower_start+percolation)`；
- 零可用水时 `g_d=0`，不允许 `0/0`；
- `hydraulic_upper_*` 联合系数只描述冻结水体在完全混合假设下的水力追踪，是诊断/充分驱动，不是与候选无关的 TN 上层转移算子。

## 正式 TN forward 边界

月表本身不足以重编所有候选的上层 N 输送。每个候选仍必须读取冻结的逐日状态/通量，并用该候选自己的 `alpha/beta` contact 方程逐日计算 N mobilization；除非未来另行冻结 `alpha/beta`，不得用 `hydraulic_upper_*` 系数替代 forward pass。`exact_lower_release_fraction` 只在已注册的下层溶解态完全混合假设下可独立使用；percolation-weighted 系数还要求 N 输入时序与逐日水下渗成比例。

## QA 摘要

- canonical 月算子闭合最大绝对误差：`{canonical_diagnostics['monthly_operator_closure_max_abs']:.3e}`；
- canonical 月水量闭合最大绝对误差：`{canonical_diagnostics['monthly_water_balance_max_abs_mm']:.3e} mm`；
- canonical lower product 复核误差：`{canonical_diagnostics['lower_release_product_max_abs_difference']:.3e}`；
- 官方 canonical 月产品复现最大差：`{max(canonical_comparison.values()):.3e}`；
- 记录的最大 RSS：`{max_rss:.3f} GiB`（warning={WARNING_GIB:g}, hard stop={HARD_STOP_GIB:g}）。
"""
    atomic_text(REPORT_PATH, report)
    print(json.dumps({"status": qa["status"], "qa": str(QA_PATH), "outputs": contract["outputs"]}, ensure_ascii=False, indent=2))
    if not all_pass:
        raise RuntimeError(qa["status"])


if __name__ == "__main__":
    main()
