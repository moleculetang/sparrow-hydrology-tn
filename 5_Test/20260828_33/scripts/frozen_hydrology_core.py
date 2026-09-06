"""Shared frozen DYN2P/Q72 utilities for the 1961-2025 extension."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE5 = ROOT / "5_Test" / "20260828_5"
STAGE8 = ROOT / "5_Test" / "20260828_2"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(ROOT / "5_Test" / "20260828_7" / "scripts"), str(STAGE5 / "scripts"),
    str(STAGE8 / "scripts"), str(ROOT / "5_Test" / "20260827_9" / "scripts"),
    str(ROOT / "5_Test" / "20260827_8" / "scripts"), str(ROOT / "5_Test" / "20260827_7" / "scripts"),
    str(STAGE3 / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts"),
    str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from learnable_state_sig2p import simulate_learnable_sig2p  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate, periodic_antecedent  # noqa: E402
from run_stage3_diagnostics import route_instantaneous_np  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


CHECKPOINT = ROOT / "5_Test" / "20260828_9" / "outputs" / "parent_preserving_state_consistent_model.pt"
SCORE = ROOT / "5_Test" / "20260828_2" / "outputs" / "regionalized_slow_score.parquet"
LAMBDA_S = 0.1805437376850875
REACH_IDS = np.arange(1, 231, dtype=int)


def load_frozen_context() -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(saved["seed"]) != 260827 or abs(float(saved["lambda_S"]) - LAMBDA_S) > 1e-15:
        raise RuntimeError("Frozen checkpoint identity changed")
    model = AlphaTwoPathCandidate(int(saved["seed"]))
    model.load_state_dict(saved["model_state"])
    model.eval()
    raw = saved["raw_parameters"].to(torch.float64).detach()
    physical = raw_to_physical(raw)
    saved_physical = saved["physical_parameters"].to(torch.float64)
    if not torch.allclose(physical, saved_physical, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Checkpoint physical parameters are inconsistent")
    order, downstream, _ = load_topology(TOPOLOGY, REACH_IDS)
    area = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"])
    area = area.drop_duplicates("reach_id").set_index("reach_id").reindex(REACH_IDS).catchment_area_km2.to_numpy(np.float64)
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    score = pd.read_parquet(SCORE).sort_values("reach_id").regionalized_slow_score.to_numpy(np.float64)
    return {
        "saved": saved,
        "model": model,
        "physical": physical,
        "area_km2": area,
        "static": static,
        "center": torch.tensor(scaling["center"], dtype=torch.float64),
        "scale": torch.tensor(scaling["scale"], dtype=torch.float64),
        "score": torch.from_numpy(score.copy()),
        "order": list(order),
        "downstream": downstream,
    }


def load_forcing(path: Path, start: str, end: str) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    dates = pd.date_range(start, end, freq="D")
    frame = pd.read_parquet(path, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    frame["date"] = pd.to_datetime(frame.date)
    frame = frame[frame.date.between(dates.min(), dates.max())]
    p = frame.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=REACH_IDS).to_numpy(np.float64)
    pet = frame.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=REACH_IDS).to_numpy(np.float64)
    if not np.isfinite(p).all() or not np.isfinite(pet).all() or (p < 0).any() or (pet < 0).any():
        raise RuntimeError(f"Incomplete or invalid forcing: {path} {start}..{end}")
    return dates, p, pet


def periodic_spinup_actual_dates(
    p: np.ndarray,
    pet: np.ndarray,
    dates: pd.DatetimeIndex,
    context: dict[str, Any],
    tolerance: float = 1e-8,
    max_cycles: int = 500,
) -> tuple[torch.Tensor, dict[str, Any]]:
    state = torch.zeros((230, 3), dtype=torch.float64)
    p_t, pet_t = torch.from_numpy(p.copy()), torch.from_numpy(pet.copy())
    api3 = torch.from_numpy(periodic_antecedent(p, 3))
    api30 = torch.from_numpy(periodic_antecedent(p, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    maximum_mass_error = 0.0
    with torch.no_grad():
        for cycle in range(1, max_cycles + 1):
            previous = state.clone()
            result = simulate_learnable_sig2p(
                p_t, pet_t, api3, api30, sin_doy, cos_doy,
                context["physical"], state, context["static"], context["center"], context["scale"],
                context["model"].gate, context["score"], LAMBDA_S,
            )
            state = result.final_state_mm
            delta = float(torch.max(torch.abs(state - previous)))
            maximum_mass_error = max(maximum_mass_error, float(result.maximum_mass_error_mm))
            if delta <= tolerance:
                return state, {
                    "converged": True, "cycles": cycle,
                    "terminal_max_abs_delta_mm": delta,
                    "maximum_mass_error_mm": maximum_mass_error,
                    "date_semantics": "actual 1961-1970 day-of-year including leap years",
                }
    return state, {
        "converged": False, "cycles": max_cycles,
        "terminal_max_abs_delta_mm": delta,
        "maximum_mass_error_mm": maximum_mass_error,
        "date_semantics": "actual 1961-1970 day-of-year including leap years",
    }


def antecedent_with_periodic_prefix(values: np.ndarray, window: int, cycle: np.ndarray) -> np.ndarray:
    prefix = cycle[-window:]
    extended = np.concatenate((prefix, values), axis=0)
    return antecedent_mean(extended, window)[window:]


def simulate_process(
    dates: pd.DatetimeIndex,
    p: np.ndarray,
    pet: np.ndarray,
    initial: torch.Tensor,
    context: dict[str, Any],
    *,
    periodic_prefix: np.ndarray | None,
) -> tuple[pd.DataFrame, dict[str, float], np.ndarray, np.ndarray]:
    if periodic_prefix is None:
        api3_np, api30_np = antecedent_mean(p, 3), antecedent_mean(p, 30)
    else:
        api3_np = antecedent_with_periodic_prefix(p, 3, periodic_prefix)
        api30_np = antecedent_with_periodic_prefix(p, 30, periodic_prefix)
    doy = dates.dayofyear.to_numpy(float)
    with torch.no_grad():
        result = simulate_learnable_sig2p(
            torch.from_numpy(p.copy()), torch.from_numpy(pet.copy()),
            torch.from_numpy(api3_np), torch.from_numpy(api30_np),
            torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25)),
            torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25)),
            context["physical"], initial, context["static"], context["center"], context["scale"],
            context["model"].gate, context["score"], LAMBDA_S,
            collect_storage=True, collect_aet=True, collect_internal_fluxes=True,
        )
    components = result.components_mm_day.numpy()
    storage = result.storage_mm.numpy()
    percolation = result.percolation_to_lower_mm_day.numpy()
    local = components * context["area_km2"][None, :, None] * 1000.0 / 86400.0
    routed = route_instantaneous_np(local, context["order"], context["downstream"])
    frame = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), 230),
        "reach_id": np.tile(REACH_IDS, len(dates)),
        "is_spinup_period": False,
        "local_fast_response_m3_s": local[:, :, 0].reshape(-1),
        "local_slow_response_m3_s": local[:, :, 1].reshape(-1),
        "routed_fast_response_m3_s": routed[:, :, 0].reshape(-1),
        "routed_slow_response_m3_s": routed[:, :, 1].reshape(-1),
        "routed_total_m3_s": routed.sum(axis=2).reshape(-1),
        "percolation_to_lower_mm_day": percolation.reshape(-1),
        "soil_storage_mm": storage[:, :, 0].reshape(-1),
        "upper_response_storage_mm": storage[:, :, 1].reshape(-1),
        "lower_slow_storage_mm": storage[:, :, 2].reshape(-1),
        "actual_aet_mm_day": result.aet_mm_day.numpy().reshape(-1),
    })
    frame["state_consistent_fast_fraction"] = np.divide(
        frame.routed_fast_response_m3_s, frame.routed_total_m3_s,
        out=np.zeros(len(frame), dtype=float), where=frame.routed_total_m3_s.to_numpy(float) > 0,
    )
    lower_previous = np.concatenate((initial[:, 2].numpy()[None, :], storage[:-1, :, 2]), axis=0)
    audit = {
        "maximum_mass_error_mm": float(result.maximum_mass_error_mm),
        "maximum_lower_store_balance_error_mm": float(np.max(np.abs(lower_previous + percolation - components[:, :, 1] - storage[:, :, 2]))),
        "maximum_routing_component_error_m3_s": float(np.max(np.abs(routed.sum(axis=2) - routed[:, :, 0] - routed[:, :, 1]))),
    }
    return frame, audit, local[:, :, 0], local[:, :, 1]
