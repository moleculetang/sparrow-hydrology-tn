from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260820_10"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
CACHE = HERE / "cache"
SCRIPTS = HERE / "scripts"

PARENT_SHARED_PATH = TEST / "20260818_1" / "scripts" / "legacy18_shared.py"
PARENT_CORE_PATH = TEST / "20260816_1" / "scripts" / "legacy16_core.py"
MONTHLY_PATH = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
OBS_PATH = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLD_PATH = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
TOPOLOGY_PATH = TEST / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
FROZEN_ROUTED_PATH = TEST / "20260816_4" / "outputs" / "candidate_routed_paths_2016_2021.parquet"
FROZEN_OOF_PATH = TEST / "20260816_4" / "outputs" / "candidate_oof_predictions_2018_2021.parquet"
PARENT_FINAL_LOCK = TEST / "20260818_7" / "final_lock.json"

BETA_GRID = np.asarray([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
FORMAL_MUS = (12, 36, 60, 96, 144, 240)
WATER_EPS = 1e-12
AGE_EXACT_MAX = 2399
TAIL_START = 2400
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_REPLICATES = 10000
NONINFERIOR_MARGIN = 0.005


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow conda environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_shared() -> ModuleType:
    return load_module(PARENT_SHARED_PATH, "hleg20_parent_shared")


def parent_core() -> ModuleType:
    return load_module(PARENT_CORE_PATH, "hleg20_parent_core")


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_manifest(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths}


def authoritative_paths() -> list[Path]:
    return [
        PARENT_SHARED_PATH,
        PARENT_CORE_PATH,
        MONTHLY_PATH,
        OBS_PATH,
        FOLD_PATH,
        TOPOLOGY_PATH,
        FROZEN_ROUTED_PATH,
        FROZEN_OOF_PATH,
        PARENT_FINAL_LOCK,
    ]


def formal_specs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mu in FORMAL_MUS:
        rows.extend([
            {"model_id": f"S0_mu_{mu:03d}m", "source_structure": "S0", "soil_tau_month": None, "delivery_mu_month": mu},
            {"model_id": f"S1_tau_012m_mu_{mu:03d}m", "source_structure": "S1", "soil_tau_month": 12, "delivery_mu_month": mu},
        ])
    return sorted(rows, key=lambda row: str(row["model_id"]))


def development_observations() -> pd.DataFrame:
    frame = pd.read_parquet(OBS_PATH, filters=[("year", "<=", 2021)])
    if frame.empty or frame.year.min() != 2016 or frame.year.max() != 2021 or (frame.year == 2022).any():
        raise RuntimeError("development TN boundary violation")
    return frame


def locked_2022_observations() -> pd.DataFrame:
    required = [REPORTS / "development_mechanism_lock.json", REPORTS / "full_development_parameter_lock.json"]
    if not all(path.exists() for path in required):
        raise RuntimeError("STOP_2022_TN_LOCKS_NOT_WRITTEN")
    frame = pd.read_parquet(OBS_PATH, filters=[("year", "==", 2022)])
    if frame.empty or set(frame.year.unique()) != {2022}:
        raise RuntimeError("locked 2022 TN filter violation")
    return frame


def route_batch(local: np.ndarray, reach_ids: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Route arrays with reach on the final axis; preceding axes are arbitrary."""
    shared = parent_shared()
    order, downstream, terminal = shared.topology_operators(reach_ids)
    index = {int(rid): i for i, rid in enumerate(reach_ids)}
    routed = np.asarray(local, dtype=float).copy()
    for rid in order:
        if rid in downstream:
            down, frac = downstream[rid]
            routed[..., index[down]] += frac * routed[..., index[rid]]
    return routed, terminal


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    observed = frame.tn_mg_l.to_numpy(float)
    predicted = frame.pred_tn_mg_l.to_numpy(float)
    lo = np.log1p(observed)
    lp = np.log1p(predicted)
    error = predicted - observed
    denom = float(np.sum((observed - observed.mean()) ** 2))
    log_denom = float(np.sum((lo - lo.mean()) ** 2))
    pearson = float(np.corrcoef(observed, predicted)[0, 1]) if len(frame) > 1 else np.nan
    spearman = float(pd.Series(observed).corr(pd.Series(predicted), method="spearman")) if len(frame) > 1 else np.nan
    mean_o, mean_p = float(observed.mean()), float(predicted.mean())
    sd_o, sd_p = float(observed.std(ddof=1)), float(predicted.std(ddof=1))
    beta = mean_p / mean_o if mean_o > 1e-12 else np.nan
    cv_o = sd_o / mean_o if mean_o > 1e-12 else np.nan
    cv_p = sd_p / mean_p if mean_p > 1e-12 else np.nan
    gamma = cv_p / cv_o if np.isfinite(cv_o) and cv_o > 0 else np.nan
    kge = 1.0 - np.sqrt((pearson - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2) if np.all(np.isfinite([pearson, beta, gamma])) else np.nan
    return {
        "n": int(len(frame)),
        "rmse_log1p": float(np.sqrt(np.mean((lp - lo) ** 2))),
        "rmse_raw_mg_l": float(np.sqrt(np.mean(error ** 2))),
        "mae_mg_l": float(np.mean(np.abs(error))),
        "median_ae_mg_l": float(np.median(np.abs(error))),
        "pbias_percent": float(100.0 * np.sum(error) / np.sum(observed)) if np.sum(observed) else np.nan,
        "pearson_r": pearson,
        "pearson_r2": pearson ** 2 if np.isfinite(pearson) else np.nan,
        "spearman_rho": spearman,
        "raw_nse": float(1.0 - np.sum(error ** 2) / denom) if denom > 0 else np.nan,
        "log_nse": float(1.0 - np.sum((lp - lo) ** 2) / log_denom) if log_denom > 0 else np.nan,
        "kge2012": float(kge),
    }


def station_macro_rmse(frame: pd.DataFrame) -> float:
    values = [
        float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2)))
        for _, group in frame.groupby("station_key")
    ]
    return float(np.mean(values))


def tree_macro_rmse(frame: pd.DataFrame) -> float:
    values = [
        float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2)))
        for _, group in frame.groupby("terminal_tree_id")
    ]
    return float(np.mean(values))


def paired_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    block_column: str,
    seed_offset: int = 0,
) -> dict[str, float | bool | int]:
    keys = ["station_key", "year", "month", "fold_id"]
    reference_keep = list(dict.fromkeys(keys + ["tn_mg_l", "pred_tn_mg_l", block_column]))
    joined = reference[reference_keep].merge(
        candidate[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_reference", "_candidate"), validate="one_to_one"
    )
    if joined.empty:
        return {"n": 0, "point_delta": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "noninferior": False, "predictively_improved": False, "clear_failure": False}
    deltas = []
    for _, group in joined.groupby(block_column):
        obs = np.log1p(group.tn_mg_l.to_numpy(float))
        rr = float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_reference.to_numpy(float)) - obs) ** 2)))
        cr = float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate.to_numpy(float)) - obs) ** 2)))
        deltas.append(cr - rr)
    block_delta = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    sampled = rng.integers(0, len(block_delta), size=(BOOTSTRAP_REPLICATES, len(block_delta)))
    distribution = block_delta[sampled].mean(axis=1)
    lo, hi = np.quantile(distribution, [0.025, 0.975])
    point = station_macro_rmse(candidate) - station_macro_rmse(reference) if block_column == "station_key" else tree_macro_rmse(candidate) - tree_macro_rmse(reference)
    return {
        "n": int(len(joined)),
        "blocks": int(len(block_delta)),
        "point_delta": float(point),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "point_improved": bool(point < 0),
        "point_nonworse": bool(point <= 0),
        "noninferior": bool(hi < NONINFERIOR_MARGIN),
        "predictively_improved": bool(hi < 0),
        "clear_failure": bool(point > 0.01 and lo > 0),
    }


def periodic_initial_cohorts(
    inferred_monthly_input: np.ndarray,
    exact_total_state: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Post-December periodic cohort state, scaled reachwise to the exact scalar parent state."""
    n_r = inferred_monthly_input.shape[1]
    ages = np.arange(TAIL_START, dtype=float)
    month_index = (11 - np.arange(TAIL_START)) % 12
    exact = inferred_monthly_input[month_index].T * np.power(rho, ages + 1.0)[None, :]
    ratio12 = rho ** 12
    tail_m = np.zeros(n_r, dtype=float)
    tail_j = np.zeros(n_r, dtype=float)
    tail_k = np.zeros(n_r, dtype=float)
    for residue in range(12):
        a0 = TAIL_START + ((residue - TAIL_START) % 12)
        month = (11 - a0) % 12
        base = inferred_monthly_input[month] * (rho ** (a0 + 1))
        one = 1.0 - ratio12
        tail_m += base / one
        tail_j += base * (a0 / one + 12.0 * ratio12 / one ** 2)
        tail_k += base * (
            a0 ** 2 / one
            + 24.0 * a0 * ratio12 / one ** 2
            + 144.0 * ratio12 * (1.0 + ratio12) / one ** 3
        )
    unscaled = exact.sum(axis=1) + tail_m
    scale = np.divide(exact_total_state, unscaled, out=np.ones_like(unscaled), where=unscaled > 0)
    exact *= scale[:, None]
    tail_m *= scale
    tail_j *= scale
    tail_k *= scale
    audit = {
        "initial_total_max_abs_kg_n": float(np.max(np.abs(exact.sum(axis=1) + tail_m - exact_total_state))),
        "periodic_scale_min": float(scale.min()),
        "periodic_scale_max": float(scale.max()),
        "periodic_scale_max_abs_from_one": float(np.max(np.abs(scale - 1.0))),
    }
    return exact, tail_m, tail_j, tail_k, audit


def cohort_state_end_2005(
    gw_input_1961_2005: np.ndarray,
    initial_exact: np.ndarray,
    initial_tail_m: np.ndarray,
    initial_tail_j: np.ndarray,
    initial_tail_k: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_hist, n_r = gw_input_1961_2005.shape
    if n_hist != 540:
        raise RuntimeError(f"expected 540 history months, got {n_hist}")
    exact = np.zeros((n_r, TAIL_START), dtype=float)
    hist_age = np.arange(n_hist)
    exact[:, :n_hist] = gw_input_1961_2005[::-1].T * np.power(rho, hist_age + 1.0)[None, :]
    survival = rho ** n_hist
    retained = TAIL_START - n_hist
    exact[:, n_hist:] = initial_exact[:, :retained] * survival
    moved = initial_exact[:, retained:] * survival
    moved_age = np.arange(retained, TAIL_START, dtype=float) + n_hist
    tail_m = initial_tail_m * survival + moved.sum(axis=1)
    tail_j = (initial_tail_j + n_hist * initial_tail_m) * survival + moved @ moved_age
    tail_k = (
        initial_tail_k + 2.0 * n_hist * initial_tail_j + (n_hist ** 2) * initial_tail_m
    ) * survival + moved @ (moved_age ** 2)
    return exact, tail_m, tail_j, tail_k


def simulate_bulk(
    gw_input: np.ndarray,
    initial_state: np.ndarray,
    h_anom: np.ndarray,
    gw_gate: np.ndarray,
    mu: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    rho = mu / (1.0 + mu)
    p0 = 1.0 - rho
    logit0 = np.log(p0 / (1.0 - p0))
    n_t, n_r = gw_input.shape
    state = np.repeat(initial_state[None, :], len(BETA_GRID), axis=0)
    release = np.zeros((n_t, len(BETA_GRID), n_r), dtype=float)
    max_balance = 0.0
    for t in range(n_t):
        pre = state + gw_input[t][None, :]
        p = expit(logit0 + BETA_GRID[:, None] * h_anom[t][None, :])
        rel = p * pre * gw_gate[t][None, :]
        state = pre - rel
        release[t] = rel
        max_balance = max(max_balance, float(np.max(np.abs(pre - rel - state))))
    diagnostics = pd.DataFrame({
        "beta_h": BETA_GRID,
        "bulk_age_composition_identity": True,
        "max_abs_mass_balance_error_kg_n": max_balance,
        "minimum_state_kg_n": state.min(axis=1),
    })
    return release, diagnostics, {"max_abs_mass_balance_error_kg_n": max_balance, "minimum_state_kg_n": float(state.min())}


def simulate_age(
    gw_input: np.ndarray,
    initial_exact: np.ndarray,
    initial_tail_m: np.ndarray,
    initial_tail_j: np.ndarray,
    initial_tail_k: np.ndarray,
    h_anom: np.ndarray,
    gw_gate: np.ndarray,
    mu: int,
    times: list[tuple[int, int]],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    rho = mu / (1.0 + mu)
    p0 = 1.0 - rho
    logit0 = np.log(p0 / (1.0 - p0))
    ages = np.arange(TAIL_START, dtype=float)
    age_score = 2.0 * np.power(rho, ages + 1.0) - 1.0
    state = np.repeat(initial_exact[None, :, :], len(BETA_GRID), axis=0)
    tail_m = np.repeat(initial_tail_m[None, :], len(BETA_GRID), axis=0)
    tail_j = np.repeat(initial_tail_j[None, :], len(BETA_GRID), axis=0)
    tail_k = np.repeat(initial_tail_k[None, :], len(BETA_GRID), axis=0)
    n_t, n_r = gw_input.shape
    release_total = np.zeros((n_t, len(BETA_GRID), n_r), dtype=float)
    summary_rows: list[dict[str, object]] = []
    accum: dict[int, dict[str, list[np.ndarray] | int]] = {
        i: {"si_high": [], "si_low": [], "si_all": [], "tail": [], "pre_age": [], "release_age": [], "valid": 0, "correct": 0}
        for i in range(len(BETA_GRID))
    }
    max_balance = 0.0
    min_state = np.inf
    for t, (year, _) in enumerate(times):
        boundary = state[:, :, -1].copy()
        tail_k += 2.0 * tail_j + tail_m
        tail_j += tail_m
        tail_m += boundary
        tail_j += TAIL_START * boundary
        tail_k += (TAIL_START ** 2) * boundary
        state[:, :, 1:] = state[:, :, :-1]
        state[:, :, 0] = gw_input[t][None, :]

        pre_m = state.sum(axis=2) + tail_m
        pre_j = np.einsum("bra,a->br", state, ages, optimize=True) + tail_j
        pre_k = np.einsum("bra,a->br", state, ages ** 2, optimize=True) + tail_k
        interaction = BETA_GRID[:, None] * h_anom[t][None, :]
        hazard = expit(logit0 + interaction[:, :, None] * age_score[None, None, :])
        gate = gw_gate[t][None, :, None]
        rel_exact = hazard * state * gate
        hazard_tail = expit(logit0 - interaction) * gw_gate[t][None, :]
        rel_tail_m = hazard_tail * tail_m
        rel_tail_j = hazard_tail * tail_j
        rel_tail_k = hazard_tail * tail_k
        rel_m = rel_exact.sum(axis=2) + rel_tail_m
        rel_j = np.einsum("bra,a->br", rel_exact, ages, optimize=True) + rel_tail_j
        release_total[t] = rel_m

        state -= rel_exact
        tail_m -= rel_tail_m
        tail_j -= rel_tail_j
        tail_k -= rel_tail_k
        balance = pre_m - rel_m - (state.sum(axis=2) + tail_m)
        max_balance = max(max_balance, float(np.max(np.abs(balance))))
        min_state = min(min_state, float(state.min()), float(tail_m.min()))

        if 2016 <= year <= 2021:
            pre_mean = np.divide(pre_j, pre_m, out=np.zeros_like(pre_j), where=pre_m > 1e-9)
            rel_mean = np.divide(rel_j, rel_m, out=np.zeros_like(rel_j), where=rel_m > 1e-12)
            variance = np.divide(pre_k, pre_m, out=np.zeros_like(pre_k), where=pre_m > 1e-9) - pre_mean ** 2
            si = rel_mean - pre_mean
            tail_fraction = np.divide(tail_m + rel_tail_m, pre_m, out=np.zeros_like(pre_m), where=pre_m > 0)
            valid = (pre_m > 1e-9) & (variance > 1e-12) & (np.abs(interaction) > 1e-6) & (rel_m > 1e-12)
            expected = -np.sign(interaction)
            correct = valid & (np.sign(si) == expected)
            for b in range(len(BETA_GRID)):
                store = accum[b]
                store["si_all"].append(si[b].copy())
                store["tail"].append(tail_fraction[b].copy())
                store["pre_age"].append(pre_mean[b].copy())
                store["release_age"].append(rel_mean[b].copy())
                if np.any(h_anom[t] >= 1.0):
                    store["si_high"].append(si[b, h_anom[t] >= 1.0].copy())
                if np.any(h_anom[t] <= -1.0):
                    store["si_low"].append(si[b, h_anom[t] <= -1.0].copy())
                store["valid"] = int(store["valid"]) + int(valid[b].sum())
                store["correct"] = int(store["correct"]) + int(correct[b].sum())

    for b, beta in enumerate(BETA_GRID):
        store = accum[b]
        concat = lambda key: np.concatenate(store[key]) if store[key] else np.asarray([], dtype=float)
        si_high, si_low, si_all = concat("si_high"), concat("si_low"), concat("si_all")
        valid_n = int(store["valid"])
        summary_rows.append({
            "beta_h": float(beta),
            "valid_semantics_rows": valid_n,
            "correct_semantics_rows": int(store["correct"]),
            "semantics_direction_fraction": float(store["correct"] / valid_n) if valid_n else np.nan,
            "median_si_all_month": float(np.median(si_all)) if len(si_all) else np.nan,
            "median_si_high_H_month": float(np.median(si_high)) if len(si_high) else np.nan,
            "median_si_low_H_month": float(np.median(si_low)) if len(si_low) else np.nan,
            "delta_si_high_minus_low_month": float(np.median(si_high) - np.median(si_low)) if len(si_high) and len(si_low) else np.nan,
            "mean_pre_stock_age_month": float(np.mean(concat("pre_age"))),
            "mean_release_age_month": float(np.mean(concat("release_age"))),
            "mean_tail_stock_fraction": float(np.mean(concat("tail"))),
            "max_abs_mass_balance_error_kg_n": max_balance,
            "minimum_state_kg_n": min_state,
        })
    return release_total, pd.DataFrame(summary_rows), {"max_abs_mass_balance_error_kg_n": max_balance, "minimum_state_kg_n": min_state}


def synthetic_semantics_tests() -> pd.DataFrame:
    ages = np.asarray([1.0, 100.0])
    masses = np.asarray([1.0, 1.0])
    rows = []
    for mu in FORMAL_MUS:
        rho = mu / (1.0 + mu)
        p0 = 1.0 - rho
        logit0 = np.log(p0 / (1.0 - p0))
        score = 2.0 * np.power(rho, ages + 1.0) - 1.0
        pre_age = float(np.sum(ages * masses) / masses.sum())
        for beta in (-0.5, 0.5):
            for h in (-1.0, 1.0):
                hazard = expit(logit0 + beta * h * score)
                rel = masses * hazard
                age = float(np.sum(ages * rel) / rel.sum())
                si = age - pre_age
                rows.append({"mu_month": mu, "beta_h": beta, "H": h, "si_month": si,
                             "expected_sign": int(-np.sign(beta * h)), "pass": bool(np.sign(si) == -np.sign(beta * h))})
                bulk_rel = masses * expit(logit0 + beta * h)
                bulk_age = float(np.sum(ages * bulk_rel) / bulk_rel.sum())
                rows.append({"mu_month": mu, "beta_h": beta, "H": h, "si_month": bulk_age - pre_age,
                             "expected_sign": 0, "pass": bool(abs(bulk_age - pre_age) <= 1e-12), "mechanism": "HLEG_BULK"})
    frame = pd.DataFrame(rows)
    frame["mechanism"] = frame.mechanism.fillna("HLEG_AGE")
    return frame


def candidate_path_frame(
    model_id: str,
    mechanism: str,
    beta_values: np.ndarray,
    quick_local: np.ndarray,
    gw_local: np.ndarray,
    water_local: np.ndarray,
    reach_ids: np.ndarray,
    times: list[tuple[int, int]],
    observations: pd.DataFrame,
) -> pd.DataFrame:
    rq, terminal = route_batch(quick_local, reach_ids)
    rg, _ = route_batch(gw_local, reach_ids)
    rw, _ = route_batch(water_local, reach_ids)
    index = {int(rid): i for i, rid in enumerate(reach_ids)}
    time_index = {tuple(value): i for i, value in enumerate(times)}
    base = observations[["station_key", "reach_id", "year", "month", "tn_mg_l"]].copy()
    ti = np.asarray([time_index[(int(y), int(m))] for y, m in zip(base.year, base.month)], dtype=int)
    ri = np.asarray([index[int(rid)] for rid in base.reach_id], dtype=int)
    rows = []
    if gw_local.ndim == 2:
        gw_routed = rg[ti, ri]
        frame = base.copy()
        frame["routed_quick_tn_kg_n"] = rq[ti, ri]
        frame["routed_gw_tn_kg_n"] = gw_routed
        frame["routed_water_volume_m3"] = rw[ti, ri]
        frame["terminal_tree_id"] = frame.reach_id.map(terminal).astype(int)
        frame["model_id"] = model_id
        frame["mechanism"] = mechanism
        frame["beta_h"] = float(beta_values[0])
        rows.append(frame)
    else:
        for b, beta in enumerate(beta_values):
            frame = base.copy()
            frame["routed_quick_tn_kg_n"] = rq[ti, ri]
            frame["routed_gw_tn_kg_n"] = rg[ti, b, ri]
            frame["routed_water_volume_m3"] = rw[ti, ri]
            frame["terminal_tree_id"] = frame.reach_id.map(terminal).astype(int)
            frame["model_id"] = model_id
            frame["mechanism"] = mechanism
            frame["beta_h"] = float(beta)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def predict_with_readout(
    shared: ModuleType,
    train: pd.DataFrame,
    test: pd.DataFrame,
    layer: str,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    eta, effects, diagnostic = shared.fit_readout(train, layer)
    predicted = shared.predict_layer(test, layer, eta, effects)
    return predicted, diagnostic, effects


def select_beta(
    shared: ModuleType,
    candidates: pd.DataFrame,
    training_year_start: int,
    training_year_end: int,
    layer: str = "P1",
) -> tuple[float, pd.DataFrame, np.ndarray, dict[str, float]]:
    score_rows = []
    fitted: dict[float, tuple[np.ndarray, dict[str, float], dict[str, object]]] = {}
    for beta, group in candidates.groupby("beta_h", sort=True):
        train = group.loc[group.year.between(training_year_start, training_year_end)].copy()
        eta, effects, diagnostic = shared.fit_readout(train, layer)
        prediction = shared.predict_layer(train, layer, eta, effects)
        score = station_macro_rmse(prediction)
        score_rows.append({"beta_h": float(beta), "training_station_macro_rmse_log1p": score, **diagnostic})
        fitted[float(beta)] = (eta, effects, diagnostic)
    scores = pd.DataFrame(score_rows).sort_values(["training_station_macro_rmse_log1p", "beta_h"], key=lambda column: np.abs(column) if column.name == "beta_h" else column)
    selected = float(scores.iloc[0].beta_h)
    eta, effects, _ = fitted[selected]
    return selected, scores, eta, effects
