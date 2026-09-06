"""Run and audit the frozen 20260828_9 hydrology model on 1961-2024 forcing."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import ctypes
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_26"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
PROCESSED = ROOT / "0_reach_topology" / "data" / "processed" / "tn_legacy_long_history" / "hydrology"
FORCING = PROCESSED / "reconstructed_daily_hydrology_forcing_1961_2024.parquet"
CANONICAL_MONTHLY = ROOT / "5_Test" / "20260828_9" / "outputs" / "canonical_reach_monthly_2006_2024.parquet"
CANONICAL_MODEL = ROOT / "5_Test" / "20260828_9" / "outputs" / "parent_preserving_state_consistent_model.pt"
CANONICAL_LOCK = ROOT / "5_Test" / "20260828_9" / "reports" / "parent_preserving_product_lock.json"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
CHANNEL = ROOT / "5_Test" / "20260826_24" / "outputs" / "registered_channel_attributes.parquet"

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
    str(ROOT / "5_Test" / "20260828_7" / "scripts"), str(STAGE5 / "scripts"),
    str(STAGE8 / "scripts"), str(ROOT / "5_Test" / "20260827_9" / "scripts"),
    str(ROOT / "5_Test" / "20260827_8" / "scripts"), str(STAGE7OP / "scripts"),
    str(STAGE3 / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts"),
    str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from learnable_state_sig2p import simulate_learnable_sig2p  # noqa: E402
from evaluate_component_development import periodic_spinup  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate  # noqa: E402
from run_stage3_diagnostics import route_instantaneous_np  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


WARNING_GIB = 12.0
HARD_STOP_GIB = 16.0
SEAM_STATE_COLUMNS = ["soil_storage_mm", "upper_response_storage_mm", "lower_slow_storage_mm"]


def require_sparrow() -> None:
    if Path(sys.executable).parent.name.lower() != "sparrow":
        raise RuntimeError(f"This stage must run in conda sparrow, got {sys.executable}")


def rss_gib() -> float:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
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
    value = counters.WorkingSetSize / 1024**3
    if value >= HARD_STOP_GIB:
        raise MemoryError(f"RSS {value:.3f} GiB reached the 16 GiB hard stop")
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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(part, path)


def build_monthly_from_arrays(
    dates: pd.DatetimeIndex,
    reach_ids: np.ndarray,
    local: np.ndarray,
    routed: np.ndarray,
    percolation: np.ndarray,
    storage: np.ndarray,
    aet: np.ndarray,
) -> pd.DataFrame:
    periods = dates.to_period("M")
    unique_periods = periods.unique()
    frames: list[pd.DataFrame] = []
    for period in unique_periods:
        index = np.flatnonzero(periods == period)
        last = index[-1]
        local_mean = local[index].mean(axis=0)
        routed_mean = routed[index].mean(axis=0)
        frame = pd.DataFrame({
            "reach_id": reach_ids,
            "year": int(period.year),
            "month": int(period.month),
            "local_fast_response_m3_s": local_mean[:, 0],
            "local_slow_response_m3_s": local_mean[:, 1],
            "routed_fast_response_m3_s": routed_mean[:, 0],
            "routed_slow_response_m3_s": routed_mean[:, 1],
            "routed_total_m3_s": routed_mean.sum(axis=1),
            "percolation_to_lower_mm_day": percolation[index].mean(axis=0),
            "actual_aet_mm_day": aet[index].mean(axis=0),
            "soil_storage_mm": storage[last, :, 0],
            "upper_response_storage_mm": storage[last, :, 1],
            "lower_slow_storage_mm": storage[last, :, 2],
            "is_spinup_period": False,
        })
        frame["state_consistent_fast_fraction"] = frame.routed_fast_response_m3_s / frame.routed_total_m3_s.clip(lower=1.0e-12)
        frames.append(frame)
    monthly = pd.concat(frames, ignore_index=True)
    geometry = pd.read_parquet(GEOMETRY, columns=[
        "reach_id", "bankfull_width_m", "bankfull_width_p05_m", "bankfull_width_p95_m",
        "bankfull_depth_m", "bankfull_depth_p05_m", "bankfull_depth_p95_m",
    ]).merge(pd.read_parquet(CHANNEL, columns=["reach_id", "reach_length_m"]), on="reach_id", validate="one_to_one")
    monthly = monthly.merge(geometry, on="reach_id", validate="many_to_one")
    positive = monthly.routed_total_m3_s.to_numpy(float) > 1.0e-12
    for label, width, depth in [
        ("central", "bankfull_width_m", "bankfull_depth_m"),
        ("geometry_p05", "bankfull_width_p05_m", "bankfull_depth_p05_m"),
        ("geometry_p95", "bankfull_width_p95_m", "bankfull_depth_p95_m"),
    ]:
        value = np.full(len(monthly), np.nan)
        value[positive] = (
            monthly.loc[positive, "reach_length_m"] * monthly.loc[positive, width]
            * monthly.loc[positive, depth] / monthly.loc[positive, "routed_total_m3_s"] / 86400.0
        )
        monthly[f"channel_bankfull_travel_time_{label}_day"] = value
    monthly.drop(columns=[column for column in geometry.columns if column != "reach_id"], inplace=True)
    return monthly.sort_values(["reach_id", "year", "month"]).reset_index(drop=True)


def compare_overlap(reconstructed: pd.DataFrame, canonical: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    columns = [
        "routed_total_m3_s", "state_consistent_fast_fraction", "lower_slow_storage_mm",
        "routed_fast_response_m3_s", "routed_slow_response_m3_s",
    ]
    left = reconstructed.loc[reconstructed.year.between(2016, 2024), ["reach_id", "year", "month", *columns]].copy()
    right = canonical.loc[canonical.year.between(2016, 2024), ["reach_id", "year", "month", *columns]].copy()
    merged = left.merge(right, on=["reach_id", "year", "month"], suffixes=("_reconstructed", "_canonical"), validate="one_to_one")
    pred = merged.routed_total_m3_s_reconstructed.to_numpy(float)
    obs = merged.routed_total_m3_s_canonical.to_numpy(float)
    merged["q_relative_error"] = (pred - obs) / np.maximum(obs, 1.0e-12)
    merged["fast_fraction_absolute_error"] = np.abs(
        merged.state_consistent_fast_fraction_reconstructed - merged.state_consistent_fast_fraction_canonical
    )
    metrics = {
        "routed_q_log_pearson": float(np.corrcoef(np.log1p(pred), np.log1p(obs))[0, 1]),
        "routed_q_raw_pearson": float(np.corrcoef(pred, obs)[0, 1]),
        "monthly_water_pbias_fraction": float(np.sum(pred - obs) / np.sum(obs)),
        "median_absolute_relative_q_difference": float(np.median(np.abs(merged.q_relative_error))),
        "fast_fraction_median_absolute_difference": float(np.median(merged.fast_fraction_absolute_error)),
        "lower_storage_spearman": float(spearmanr(
            merged.lower_slow_storage_mm_reconstructed, merged.lower_slow_storage_mm_canonical
        ).statistic),
    }
    return merged, metrics


def seam_state_p95(reconstructed: pd.DataFrame, canonical: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    left = reconstructed.loc[(reconstructed.year == 2010) & (reconstructed.month == 1), ["reach_id", *SEAM_STATE_COLUMNS]]
    right = canonical.loc[(canonical.year == 2010) & (canonical.month == 1), ["reach_id", *SEAM_STATE_COLUMNS]]
    seam = left.merge(right, on="reach_id", suffixes=("_jan2010_reconstructed", "_jan2010_canonical"), validate="one_to_one")
    relative_columns = []
    for column in SEAM_STATE_COLUMNS:
        rel = f"{column}_seam_relative_difference"
        seam[rel] = np.abs(seam[f"{column}_jan2010_reconstructed"] - seam[f"{column}_jan2010_canonical"]) / np.maximum(
            seam[f"{column}_jan2010_canonical"], 1.0
        )
        relative_columns.append(rel)
    return seam, float(np.quantile(seam[relative_columns].to_numpy(float).reshape(-1), 0.95))


def main() -> None:
    require_sparrow()
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    lock = json.loads(CANONICAL_LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "PARENT_PRESERVING_STATE_PRODUCT_LOCKED" or sha256(CANONICAL_MODEL) != lock["hashes"]["state_consistent_model"]:
        raise RuntimeError("20260828_9 canonical hydrology lock changed")
    saved = torch.load(CANONICAL_MODEL, map_location="cpu", weights_only=False)
    seed = int(saved["seed"])
    lambda_s = float(saved["lambda_S"])
    model = AlphaTwoPathCandidate(seed)
    model.load_state_dict(saved["model_state"])
    model.eval()
    physical = raw_to_physical(saved["raw_parameters"].to(torch.float64).detach())

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    area_np = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64)
    area = torch.from_numpy(area_np.copy())
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])
    score_path = Path(saved["regionalized_score_path"])
    score = torch.from_numpy(pd.read_parquet(score_path).sort_values("reach_id").regionalized_slow_score.to_numpy(np.float64).copy())

    dates = pd.date_range("1961-01-01", "2024-12-31", freq="D")
    forcing = pd.read_parquet(FORCING)
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    del forcing
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("Historical reconstructed forcing incomplete")
    p, pet = torch.from_numpy(p_np.copy()), torch.from_numpy(pet_np.copy())
    api3, api30 = torch.from_numpy(antecedent_mean(p_np, 3)), torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    spin_mask = np.asarray(dates.year <= 1990)
    initial, spin = periodic_spinup(p[spin_mask], pet[spin_mask], physical, static, center, scale, model.gate, score, lambda_s)
    print(json.dumps({"step": "spinup", "audit": spin, "rss_gib": round(rss_gib(), 3)}), flush=True)
    with torch.no_grad():
        result = simulate_learnable_sig2p(
            p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
            static, center, scale, model.gate, score, lambda_s,
            collect_storage=True, collect_aet=True, collect_internal_fluxes=True,
        )
    if result.storage_mm is None or result.aet_mm_day is None or result.percolation_to_lower_mm_day is None:
        raise RuntimeError("Frozen model did not return required TN-interface states")
    components = result.components_mm_day.numpy()
    storage = result.storage_mm.numpy()
    percolation = result.percolation_to_lower_mm_day.numpy()
    aet = result.aet_mm_day.numpy()
    local = components * area_np[None, :, None] * 1000.0 / 86400.0
    routed = route_instantaneous_np(local, list(order), downstream)
    previous_lower = np.concatenate([initial[:, 2].numpy()[None, :], storage[:-1, :, 2]], axis=0)
    lower_balance = float(np.max(np.abs(previous_lower + percolation - components[:, :, 1] - storage[:, :, 2])))
    component_closure = float(np.max(np.abs(routed.sum(axis=2) - routed[:, :, 0] - routed[:, :, 1])))
    print(json.dumps({"step": "simulation_complete", "rss_gib": round(rss_gib(), 3)}), flush=True)

    reconstructed = build_monthly_from_arrays(dates, reach_ids, local, routed, percolation, storage, aet)
    canonical = pd.read_parquet(CANONICAL_MONTHLY).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    overlap, metrics = compare_overlap(reconstructed, canonical)
    seam, seam_p95 = seam_state_p95(reconstructed, canonical)
    metrics.update({
        "seam_state_p95_relative_difference": seam_p95,
        "model_maximum_mass_error_mm": float(result.maximum_mass_error_mm),
        "lower_store_balance_error_mm": lower_balance,
        "routed_component_closure_m3_s": component_closure,
    })
    checks = {
        "canonical_lock_verified": True,
        "no_hydrology_parameter_refit": True,
        "TN_not_read": True,
        "spinup_converged": bool(spin["converged"]),
        "routed_q_correlation_ge_0p98": metrics["routed_q_log_pearson"] >= 0.98,
        "monthly_water_abs_pbias_le_0p05": abs(metrics["monthly_water_pbias_fraction"]) <= 0.05,
        "median_absolute_relative_q_difference_le_0p10": metrics["median_absolute_relative_q_difference"] <= 0.10,
        "fast_fraction_median_absolute_difference_le_0p05": metrics["fast_fraction_median_absolute_difference"] <= 0.05,
        "lower_storage_spearman_ge_0p90": metrics["lower_storage_spearman"] >= 0.90,
        "seam_state_p95_le_0p25": seam_p95 <= 0.25,
        "water_mass_balance_le_1e_9": float(result.maximum_mass_error_mm) <= 1.0e-9 and lower_balance <= 1.0e-9,
        "component_closure_le_1e_9": component_closure <= 1.0e-9,
        "memory_below_warning": rss_gib() < WARNING_GIB,
    }

    reconstructed_path = OUT / "reconstructed_reach_monthly_1961_2024.parquet"
    overlap_path = OUT / "historical_bridge_overlap_2016_2024.parquet"
    seam_path = OUT / "historical_canonical_seam_audit.parquet"
    atomic_parquet(reconstructed, reconstructed_path)
    atomic_parquet(overlap, overlap_path)
    atomic_parquet(seam, seam_path)
    status = "PASS_HISTORICAL_HYDROLOGY_BRIDGE" if all(checks.values()) else "HISTORICAL_HYDROLOGY_BRIDGE_FAILED"
    production_path = PROCESSED / "long_history_hydrology_monthly_1961_2024.parquet"
    if status.startswith("PASS"):
        historical = reconstructed.loc[reconstructed.year <= 2009].copy()
        historical["hydrology_provenance"] = "frozen_model_reconstruction"
        exact = canonical.loc[canonical.year >= 2010].copy()
        exact["hydrology_provenance"] = "exact_20260828_9_canonical"
        production = pd.concat([historical, exact], ignore_index=True).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
        if len(production) != 230 * 64 * 12 or production.duplicated(["reach_id", "year", "month"]).any():
            raise RuntimeError("Production hydrology splice failed grain QA")
        atomic_parquet(production, production_path)

    audit = {
        "stage": "20260824_26", "status": status, "checks": checks, "metrics": metrics,
        "forcing_boundary": "1961-2009 state-continuous frozen reconstruction; 2010-2024 production values are exact canonical rows after its spin-up interval",
        "state_boundary": "TN state must continue across 2009/2010; it must not be reset at the hydrology provenance seam",
        "handoff_definition": "same-time reconstructed versus canonical states at 2010-01; consecutive month-end states are not treated as a discontinuity metric",
        "paths": {
            "reconstructed_monthly": str(reconstructed_path), "overlap": str(overlap_path),
            "seam": str(seam_path), "production": str(production_path) if status.startswith("PASS") else None,
        },
        "hashes": {
            "forcing": sha256(FORCING), "canonical_monthly": sha256(CANONICAL_MONTHLY),
            "canonical_model": sha256(CANONICAL_MODEL), "reconstructed_monthly": sha256(reconstructed_path),
            "overlap": sha256(overlap_path), "seam": sha256(seam_path),
            "production": sha256(production_path) if status.startswith("PASS") else None,
        },
        "runtime": {"python": sys.executable, "torch": torch.__version__, "threads": torch.get_num_threads(), "rss_gib": rss_gib(), "elapsed_seconds": time.perf_counter() - started},
        "authorized_successor": "20260824_27" if status.startswith("PASS") else None,
    }
    write_json(REPORTS / "historical_hydrology_bridge_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if not status.startswith("PASS"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
