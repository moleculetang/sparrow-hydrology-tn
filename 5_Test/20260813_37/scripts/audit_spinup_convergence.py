from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs" / "parent_indata.parquet"
RHO = 0.70
WM = 480.0
SAS_RHO = 0.93
YOUNG_K = 1.5
PROD_CAPACITY = 240.0
RUNOFF_GAMMA = 2.5
QUICK_RHO = 0.25
BASE_RHO = 0.85
BASE_RELEASE = 0.10
STORAGE_SCALE = 720.0
VARIANTS = {
    "production": (PROD_CAPACITY, RUNOFF_GAMMA, QUICK_RHO, BASE_RHO, BASE_RELEASE, 1.0),
    "flash_headwater": (0.55 * PROD_CAPACITY, max(1.3, RUNOFF_GAMMA - 0.8), 0.10, 0.76, 0.06, 1.20),
    "slow_large": (1.80 * PROD_CAPACITY, RUNOFF_GAMMA + 0.6, 0.48, 0.94, 0.07, 0.85),
    "buffer_reservoir": (1.45 * PROD_CAPACITY, RUNOFF_GAMMA + 0.8, 0.66, 0.96, 0.12, 0.65),
    "wet_large": (1.20 * PROD_CAPACITY, max(1.5, RUNOFF_GAMMA - 0.3), 0.34, 0.90, 0.08, 1.10),
}


def initial(mode: str) -> dict[str, float]:
    state = {"wetness": 0.0, "sas": 0.0}
    for name, (capacity, *_rest) in VARIANTS.items():
        if mode == "default":
            soil, quick, base = 0.5 * capacity, 0.0, 0.0
        elif mode == "low":
            soil, quick, base = 0.0, 0.0, 0.0
        elif mode == "high":
            soil, quick, base = capacity, capacity, capacity
        else:
            raise ValueError(mode)
        state[f"{name}.soil"] = soil
        state[f"{name}.quick"] = quick
        state[f"{name}.base"] = base
    if mode == "low":
        state["wetness"] = -3.0
    if mode == "high":
        state["wetness"] = 3.0
        state["sas"] = STORAGE_SCALE
    return state


def step(state: dict[str, float], ppt: float, aet: float) -> tuple[dict[str, float], dict[str, float]]:
    new = dict(state)
    flux: dict[str, float] = {}
    effective = max(ppt - aet, 0.0)
    wetness = float(np.clip(RHO * state["wetness"] + (ppt - aet) / WM, -3.0, 3.0))
    new["wetness"] = wetness
    young = float(np.clip(1.0 / (1.0 + np.exp(-YOUNG_K * wetness)), 0.05, 0.95))
    sas_available = state["sas"] + (1.0 - young) * effective
    flux["sas_young_mm"] = young * effective
    flux["sas_old_release_mm"] = (1.0 - SAS_RHO) * sas_available
    new["sas"] = SAS_RHO * sas_available
    for name, (capacity, gamma, quick_rho, base_rho, base_release, scale) in VARIANTS.items():
        soil0 = state[f"{name}.soil"]
        quick0 = state[f"{name}.quick"]
        base0 = state[f"{name}.base"]
        saturation = float(np.clip(soil0 / capacity, 0.0, 1.5))
        quick_generated = min(effective, effective * saturation**gamma * scale)
        soil = max(soil0 + effective - quick_generated, 0.0)
        overflow = max(soil - capacity, 0.0)
        soil = min(soil, capacity)
        quick_input = quick_generated + overflow
        slow = base_release * soil
        soil -= slow
        quick_available = quick0 + quick_input
        base_available = base0 + slow
        quick_release = (1.0 - quick_rho) * quick_available
        base_out = (1.0 - base_rho) * base_available
        new[f"{name}.soil"] = soil
        new[f"{name}.quick"] = quick_rho * quick_available
        new[f"{name}.base"] = base_rho * base_available
        flux[f"{name}.quick_release_mm"] = quick_release
        flux[f"{name}.base_release_mm"] = base_out
        flux[f"{name}.overflow_mm"] = overflow
    return new, flux


def scale_for(key: str) -> float:
    if key == "wetness":
        return 3.0
    if key == "sas":
        return STORAGE_SCALE
    name = key.split(".")[0]
    return VARIANTS[name][0]


def state_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(a[key] - b[key]) / max(scale_for(key), 1e-12) for key in a)


def run_sequence(frame: pd.DataFrame, state: dict[str, float], capture: bool = False):
    records = []
    current = dict(state)
    for row in frame.itertuples(index=False):
        current, flux = step(current, max(float(row.PPT), 0.0), max(float(row.AET), 0.0))
        if capture:
            records.append({"year": int(row.year), "month": int(row.month), **current, **flux})
    return current, pd.DataFrame(records)


def converged_cycle(cycle: pd.DataFrame) -> tuple[dict[str, float], int, float]:
    state = initial("default")
    distance = np.inf
    for iteration in range(1, 1001):
        next_state, _ = run_sequence(cycle, state)
        distance = state_distance(next_state, state)
        state = next_state
        if distance < 1e-6:
            return state, iteration, distance
    raise RuntimeError(f"Cycle spin-up failed to converge: {distance}")


def relative_rows(reference: pd.DataFrame, candidate: pd.DataFrame, scenario: str, comid: int):
    rows = []
    state_cols = ["wetness", "sas"] + [f"{name}.{part}" for name in VARIANTS for part in ("soil", "quick", "base")]
    flux_cols = [c for c in reference.columns if c.endswith("_mm") and c not in state_cols]
    for year in (2008, 2010, 2011):
        r = reference[(reference.year == year) & (reference.month == 12)].iloc[0]
        c = candidate[(candidate.year == year) & (candidate.month == 12)].iloc[0]
        for column in state_cols:
            denom = scale_for(column)
            rows.append({"comid": comid, "scenario": scenario, "period": f"{year}-12", "kind": "state", "variable": column, "relative_difference": abs(float(c[column]) - float(r[column])) / max(denom, 1e-12)})
    r12 = reference[reference.year == 2012].reset_index(drop=True)
    c12 = candidate[candidate.year == 2012].reset_index(drop=True)
    for column in flux_cols:
        denom = np.maximum(np.abs(r12[column].to_numpy(float)), 1.0)
        for month, value in enumerate(np.abs(c12[column].to_numpy(float) - r12[column].to_numpy(float)) / denom, start=1):
            rows.append({"comid": comid, "scenario": scenario, "period": f"2012-{month:02d}", "kind": "flux", "variable": column, "relative_difference": float(value)})
    return rows


def main() -> None:
    panel = pd.read_parquet(INPUT, columns=["comid", "year", "month", "PPT", "AET"])
    panel = panel.sort_values(["comid", "year", "month"]).reset_index(drop=True)
    all_diffs = []
    convergence = []
    for comid, frame in panel.groupby("comid", sort=True):
        cycle = frame[frame.year.between(2006, 2011)].copy()
        future = frame[frame.year == 2012].copy()
        periodic_start, iterations, last_distance = converged_cycle(cycle)
        periodic_end, periodic_hist = run_sequence(pd.concat([cycle, future]), periodic_start, capture=True)
        convergence.append({"comid": int(comid), "iterations": iterations, "last_scaled_max_difference": last_distance})
        for mode in ("default", "low", "high"):
            _end, history = run_sequence(pd.concat([cycle, future]), initial(mode), capture=True)
            all_diffs.extend(relative_rows(periodic_hist, history, mode, int(comid)))
    differences = pd.DataFrame(all_diffs)
    differences.to_csv(ROOT / "spinup_state_flux_differences.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(convergence).to_csv(ROOT / "spinup_cycle_convergence.csv", index=False, encoding="utf-8-sig")
    summary = differences.groupby(["scenario", "period", "kind"], as_index=False).relative_difference.agg(
        p50="median", p95=lambda x: x.quantile(0.95), p99=lambda x: x.quantile(0.99), maximum="max"
    )
    summary.to_csv(ROOT / "spinup_convergence_summary.csv", index=False, encoding="utf-8-sig")
    default_end = summary[(summary.scenario == "default") & (summary.period == "2011-12") & (summary.kind == "state")].iloc[0]
    default_flux = differences[(differences.scenario == "default") & (differences.kind == "flux")]
    flux_p99 = float(default_flux.relative_difference.quantile(0.99))
    passed = bool(float(default_end.p99) < 1e-4 and flux_p99 < 1e-4)
    payload = {
        "parent": "20260813_36/A1",
        "reaches": int(panel.comid.nunique()),
        "forcing_only": True,
        "periodic_convergence_max_iterations": int(pd.DataFrame(convergence).iterations.max()),
        "periodic_convergence_max_final_distance": float(pd.DataFrame(convergence).last_scaled_max_difference.max()),
        "default_2011_state_p99_relative_difference": float(default_end.p99),
        "default_2012_flux_p99_relative_difference": flux_p99,
        "threshold": 1e-4,
        "pass_no_change": passed,
        "terminal": "INITIAL_STATE_SPINUP_CONVERGENCE_CONFIRMED_NO_CHANGE" if passed else "DETERMINISTIC_SPINUP_REPAIR_REQUIRED",
    }
    (ROOT / "terminal_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
