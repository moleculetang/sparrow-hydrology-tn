from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260820_12"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
CACHE = HERE / "cache"
SCRIPTS = HERE / "scripts"
PROGRAM = TEST / "20260820_11"

PARENT_SHARED_PATH = TEST / "20260818_1" / "scripts" / "legacy18_shared.py"
PARENT_LOCAL_CACHE = TEST / "20260820_10" / "cache" / "parent_local"
PARENT_OBSERVATION_PATHS = TEST / "20260820_10" / "outputs" / "candidate_development_observation_paths.parquet"
PARENT_OOF = TEST / "20260820_10" / "outputs" / "temporal_oof_predictions.parquet"
PARENT_FINAL_LOCK = TEST / "20260820_10" / "final_lock.json"
OBS_PATH = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLD_PATH = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
MONTHLY_PATH = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"

FORMAL_MUS = (12, 36, 60, 96, 144, 240)
REGISTERED_MONTHS = (2, 3, 7, 10, 11, 12)
AMAX = float(np.log(5.0) / 2.0)
PHASE_RATIO_MIN = 1.2
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260820
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
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_shared() -> ModuleType:
    return load_module(PARENT_SHARED_PATH, "a0_parent_shared")


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


def authoritative_inputs() -> list[Path]:
    return [
        PARENT_SHARED_PATH, PARENT_FINAL_LOCK, OBS_PATH, FOLD_PATH, MONTHLY_PATH,
        PROGRAM / "program_manifest.json", PROGRAM / "experiment_contract.json",
        HERE / "experiment_contract.json",
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
    if frame.empty or frame.year.min() != 2016 or frame.year.max() != 2021 or frame.year.eq(2022).any():
        raise RuntimeError("development TN boundary violation")
    return frame


def locked_2022_observations() -> pd.DataFrame:
    locks = [REPORTS / "development_mechanism_lock.json", REPORTS / "full_development_parameter_lock.json"]
    if not all(path.exists() for path in locks):
        raise RuntimeError("STOP_2022_TN_LOCKS_NOT_WRITTEN")
    frame = pd.read_parquet(OBS_PATH, filters=[("year", "==", 2022)])
    if frame.empty or set(frame.year.unique()) != {2022}:
        raise RuntimeError("locked 2022 filter violation")
    return frame


def harmonic_weights(a: float, b: float) -> np.ndarray:
    months = np.arange(1, 13, dtype=float)
    u = a * np.sin(2.0 * np.pi * months / 12.0) + b * np.cos(2.0 * np.pi * months / 12.0)
    values = np.exp(u - np.max(u))
    return values / values.sum()


def weight_diagnostics(a: float, b: float) -> dict[str, object]:
    w = harmonic_weights(a, b)
    amplitude = float(np.hypot(a, b))
    ratio = float(w.max() / w.min())
    return {
        "a": float(a), "b": float(b), "amplitude": amplitude, "weight_ratio": ratio,
        "peak_month": int(np.argmax(w) + 1), "w_max": float(w.max()), "w_min": float(w.min()),
        "cv_w": float(np.std(w, ddof=0) / np.mean(w)), "top_month_annual_mass_share": float(w.max()),
        "phase_evaluable": bool(ratio > PHASE_RATIO_MIN),
        "amplitude_boundary": bool(amplitude >= 0.99 * AMAX),
        **{f"w_month_{month:02d}": float(w[month - 1]) for month in range(1, 13)},
    }


def seasonal_positive_arrays(
    times: list[tuple[int, int]], positive: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    years = np.asarray([year for year, _ in times], dtype=int)
    months = np.asarray([month for _, month in times], dtype=int)
    out = np.zeros_like(positive)
    for year in np.unique(years):
        idx = np.flatnonzero(years == year)
        total = positive[idx].sum(axis=0)
        for t in idx:
            out[t] = total * weights[months[t] - 1]
        # Float64 products can leave one-ULP annual closure error at the
        # largest Reach loads.  Return that deterministic residual to the
        # final month; repeat once because NumPy's pairwise summation order
        # can otherwise retain the same last bit.
        out[idx[-1]] += total - out[idx].sum(axis=0)
        out[idx[-1]] += total - out[idx].sum(axis=0)
        target = idx[int(np.argmax(weights))]
        for _ in range(16):
            residual = total - out[idx].sum(axis=0)
            mask = residual != 0.0
            if not np.any(mask):
                break
            direction = np.where(residual[mask] > 0.0, np.inf, -np.inf)
            out[target, mask] = np.nextafter(out[target, mask], direction)
    return out


def spinup_seasonal(
    shared: ModuleType, structure: str, tau_s: int | None, mu_t: int,
    early_by_month: np.ndarray, arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    n_r = early_by_month.shape[1]
    state = {name: np.zeros(n_r, dtype=float) for name in ("son", "mobile", "quick", "gw")}
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_gw = mu_t / (1.0 + mu_t)
    final_balance = np.zeros(n_r, dtype=float)
    for cycle in range(1, shared.SPINUP_MAX_CYCLES + 1):
        before = np.concatenate([value.copy() for value in state.values()])
        for t in range(12):
            start = sum(value.astype(np.longdouble) for value in state.values())
            positive = early_by_month[t]
            bypass, _, flush, quick_share, gw_share = shared.operator_water_partitions(arrays, t, "F00")
            direct = positive * bypass
            pool_input = positive - direct
            if structure == "S0":
                state["mobile"] += pool_input
            else:
                state["son"] += pool_input
                mineralized = state["son"] * (1.0 - rho_s)
                state["son"] -= mineralized
                state["mobile"] += mineralized
            source_release = state["mobile"] * flush
            state["mobile"] -= source_release
            state["quick"] += direct + source_release * quick_share
            state["gw"] += source_release * gw_share
            quick_release = np.where(arrays["quick_release_mm"][t] > shared.WATER_EPS, (1.0 - shared.Q_RHO) * state["quick"], 0.0)
            gw_release = np.where(arrays["gw_discharge_mm"][t] > shared.WATER_EPS, (1.0 - rho_gw) * state["gw"], 0.0)
            state["quick"] -= quick_release
            state["gw"] -= gw_release
            end = sum(value.astype(np.longdouble) for value in state.values())
            final_balance = np.asarray(positive.astype(np.longdouble) + start - quick_release.astype(np.longdouble) - gw_release.astype(np.longdouble) - end, dtype=float)
        after = np.concatenate(list(state.values()))
        delta = float(np.max(np.abs(after - before)))
        if delta <= shared.SPINUP_TOL_KG:
            break
    scale = max(float(sum(np.sum(np.abs(value)) for value in state.values())), 1.0)
    return state, {
        "cycles": int(cycle), "converged": bool(delta <= shared.SPINUP_TOL_KG),
        "terminal_max_abs_delta_kg_n": delta,
        "son_end_total_kg_n": float(state["son"].sum()),
        "mobile_end_total_kg_n": float(state["mobile"].sum()),
        "quick_end_total_kg_n": float(state["quick"].sum()),
        "gw_end_total_kg_n": float(state["gw"].sum()),
        "minimum_state_kg_n": float(min(value.min() for value in state.values())),
        "final_cycle_mass_balance_error_kg_n": float(np.max(np.abs(final_balance))),
        "final_cycle_mass_balance_relative_error": float(np.max(np.abs(final_balance)) / scale),
    }


def simulate_seasonal(
    spec: dict[str, object], weights: np.ndarray, capture_start: int = 2016, capture_end: int = 2022,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    shared = parent_shared()
    reach_ids, times, arrays, early_uniform = shared.prepare_arrays()
    if np.any(arrays["negative_legacy_eligible_n_surplus_kg_n_month"] > 0):
        raise RuntimeError("STOP_UNREGISTERED_NEGATIVE_REACH_YEAR")
    local_arrays = dict(arrays)
    local_arrays["positive_legacy_eligible_n_surplus_kg_n_month"] = seasonal_positive_arrays(
        times, arrays["positive_legacy_eligible_n_surplus_kg_n_month"], weights
    )
    early_by_month = (early_uniform * 12.0)[None, :] * weights[:, None]
    state, spinup = spinup_seasonal(
        shared, str(spec["source_structure"]), spec["soil_tau_month"], int(spec["delivery_mu_month"]),
        early_by_month, local_arrays,
    )
    if not spinup["converged"]:
        raise RuntimeError(f"spinup failed {spec['model_id']}")
    tau_s = spec["soil_tau_month"]
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_gw = int(spec["delivery_mu_month"]) / (1.0 + int(spec["delivery_mu_month"]))
    rows: list[pd.DataFrame] = []
    max_abs, max_rel, minimum = 0.0, 0.0, np.inf
    for t, (year, month) in enumerate(times):
        start = sum(value.astype(np.longdouble) for value in state.values())
        positive = local_arrays["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        bypass, _, flush, quick_share, gw_share = shared.operator_water_partitions(local_arrays, t, "F00")
        direct = positive * bypass
        pool_input = positive - direct
        if str(spec["source_structure"]) == "S0":
            state["mobile"] += pool_input
        else:
            state["son"] += pool_input
            mineralized = state["son"] * (1.0 - rho_s)
            state["son"] -= mineralized
            state["mobile"] += mineralized
        source_release = state["mobile"] * flush
        state["mobile"] -= source_release
        state["quick"] += direct + source_release * quick_share
        state["gw"] += source_release * gw_share
        quick_release = np.where(local_arrays["quick_release_mm"][t] > shared.WATER_EPS, (1.0 - shared.Q_RHO) * state["quick"], 0.0)
        gw_release = np.where(local_arrays["gw_discharge_mm"][t] > shared.WATER_EPS, (1.0 - rho_gw) * state["gw"], 0.0)
        state["quick"] -= quick_release
        state["gw"] -= gw_release
        end = sum(value.astype(np.longdouble) for value in state.values())
        balance = positive.astype(np.longdouble) + start - quick_release.astype(np.longdouble) - gw_release.astype(np.longdouble) - end
        scale = np.abs(positive.astype(np.longdouble)) + np.abs(start) + np.abs(quick_release.astype(np.longdouble)) + np.abs(gw_release.astype(np.longdouble)) + np.abs(end)
        rel = np.asarray(np.abs(balance) / np.maximum(scale, 1.0), dtype=float)
        max_abs = max(max_abs, float(np.max(np.abs(balance))))
        max_rel = max(max_rel, float(np.max(rel)))
        minimum = min(minimum, *(float(value.min()) for value in state.values()), float(quick_release.min()), float(gw_release.min()))
        if capture_start <= year <= capture_end:
            rows.append(pd.DataFrame({
                "reach_id": reach_ids, "year": year, "month": month,
                "quick_tn_release_kg_n": quick_release, "gw_tn_release_kg_n": gw_release,
                "q_local_total_mm": local_arrays["q_local_total_mm"][t],
                "catchment_area_km2": local_arrays["catchment_area_km2"][t],
            }))
    return pd.concat(rows, ignore_index=True), {
        "model_id": str(spec["model_id"]), "max_abs_mass_balance_error_kg_n": max_abs,
        "max_relative_mass_balance_error": max_rel, "minimum_state_or_flux_kg_n": minimum,
    }, spinup


def route_frame(frame: pd.DataFrame) -> pd.DataFrame:
    shared = parent_shared()
    reach_ids = np.sort(frame.reach_id.unique().astype(int))
    work = frame.copy()
    work["quick_release_mm"] = 0.0
    work["gw_discharge_mm"] = 0.0
    return shared.route_candidate(work, reach_ids)


@dataclass
class ObservationBasis:
    base: pd.DataFrame
    quick: np.ndarray
    gw: np.ndarray
    parent_quick: np.ndarray
    parent_gw: np.ndarray

    def frame(self, weights: np.ndarray | None = None, indices: np.ndarray | None = None, parent: bool = False) -> pd.DataFrame:
        idx = np.arange(len(self.base)) if indices is None else np.asarray(indices, dtype=int)
        out = self.base.iloc[idx].copy()
        if parent:
            out["routed_quick_tn_kg_n"] = self.parent_quick[idx]
            out["routed_gw_tn_kg_n"] = self.parent_gw[idx]
        else:
            if weights is None:
                raise ValueError("weights required")
            out["routed_quick_tn_kg_n"] = self.quick[idx] @ weights
            out["routed_gw_tn_kg_n"] = self.gw[idx] @ weights
        return out


def observation_basis(model_id: str, observations: pd.DataFrame) -> ObservationBasis:
    basis = pd.read_parquet(CACHE / "basis_routed" / f"{model_id}.parquet")
    keys = ["reach_id", "year", "month"]
    base = observations.merge(
        basis.loc[basis.basis_month.eq(1), keys + ["routed_water_volume_m3", "terminal_tree_id"]],
        on=keys, how="inner", validate="many_to_one",
    ).sort_values(["station_key", "year", "month"]).reset_index(drop=True)
    quick = np.zeros((len(base), 12), dtype=float)
    gw = np.zeros((len(base), 12), dtype=float)
    for month in range(1, 13):
        part = base[keys].merge(
            basis.loc[basis.basis_month.eq(month), keys + ["routed_quick_tn_kg_n", "routed_gw_tn_kg_n"]],
            on=keys, how="left", validate="many_to_one",
        )
        quick[:, month - 1] = part.routed_quick_tn_kg_n.to_numpy(float)
        gw[:, month - 1] = part.routed_gw_tn_kg_n.to_numpy(float)
    direct = pd.read_parquet(CACHE / "direct_parent_routed" / f"{model_id}.parquet")
    direct_obs = base[keys].merge(direct[keys + ["routed_quick_tn_kg_n", "routed_gw_tn_kg_n"]], on=keys, how="left", validate="many_to_one")
    return ObservationBasis(base, quick, gw, direct_obs.routed_quick_tn_kg_n.to_numpy(float), direct_obs.routed_gw_tn_kg_n.to_numpy(float))


def station_macro_rmse(frame: pd.DataFrame) -> float:
    return float(np.mean([
        np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2))
        for _, group in frame.groupby("station_key")
    ]))


def tree_macro_rmse(frame: pd.DataFrame) -> float:
    return float(np.mean([
        np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2))
        for _, group in frame.groupby("terminal_tree_id")
    ]))


def fit_uniform(shared: ModuleType, basis: ObservationBasis, indices: np.ndarray, layer: str) -> dict[str, object]:
    train = basis.frame(indices=indices, parent=True)
    eta, effects, diagnostic = shared.fit_readout(train, layer)
    return {"a": 0.0, "b": 0.0, "weights": np.full(12, 1.0 / 12.0), "eta": eta, "effects": effects, "diagnostic": diagnostic,
            "objective": station_macro_rmse(shared.predict_layer(train, layer, eta, effects)), "outer_success": True, "outer_nit": 0, "outer_nfev": 1, "start_id": "uniform_parent"}


def fit_harmonic(shared: ModuleType, basis: ObservationBasis, indices: np.ndarray, layer: str) -> dict[str, object]:
    cache: dict[tuple[float, float], tuple[float, np.ndarray, dict[str, float], dict[str, object]]] = {}

    def evaluate(x: np.ndarray) -> tuple[float, np.ndarray, dict[str, float], dict[str, object]]:
        key = (round(float(x[0]), 12), round(float(x[1]), 12))
        if key not in cache:
            weights = harmonic_weights(*key)
            train = basis.frame(weights=weights, indices=indices)
            eta, effects, diagnostic = shared.fit_readout(train, layer)
            prediction = shared.predict_layer(train, layer, eta, effects)
            cache[key] = (station_macro_rmse(prediction), eta, effects, diagnostic)
        return cache[key]

    starts = [
        ("zero", np.array([0.0, 0.0])),
        ("half_positive_a", np.array([0.5 * AMAX, 0.0])),
        ("half_negative_a", np.array([-0.5 * AMAX, 0.0])),
        ("half_positive_b", np.array([0.0, 0.5 * AMAX])),
        ("half_negative_b", np.array([0.0, -0.5 * AMAX])),
    ]
    candidates: list[dict[str, object]] = []
    constraint = {"type": "ineq", "fun": lambda x: AMAX ** 2 - float(x[0] ** 2 + x[1] ** 2)}
    for start_id, x0 in starts:
        result = minimize(
            lambda x: evaluate(np.asarray(x, dtype=float))[0], x0=x0,
            method="SLSQP", bounds=[(-AMAX, AMAX), (-AMAX, AMAX)], constraints=[constraint],
            options={"ftol": 1e-10, "maxiter": 200, "disp": False},
        )
        x = np.asarray(result.x, dtype=float)
        if np.hypot(*x) > AMAX:
            x *= AMAX / np.hypot(*x)
        objective, eta, effects, diagnostic = evaluate(x)
        candidates.append({"x": x, "objective": objective, "eta": eta, "effects": effects, "diagnostic": diagnostic,
                           "outer_success": bool(result.success), "outer_status": int(result.status), "outer_message": str(result.message),
                           "outer_nit": int(result.nit), "outer_nfev": int(result.nfev), "start_id": start_id})
    best_value = min(float(row["objective"]) for row in candidates)
    tied = [row for row in candidates if float(row["objective"]) <= best_value + 1e-8]
    tied.sort(key=lambda row: (float(np.hypot(*row["x"])), float(row["x"][0]), float(row["x"][1])))
    best = tied[0]
    a, b = map(float, best.pop("x"))
    return {"a": a, "b": b, "weights": harmonic_weights(a, b), **best, "objective_evaluations": len(cache)}


def predict_fit(shared: ModuleType, basis: ObservationBasis, fit: dict[str, object], indices: np.ndarray, layer: str, parent: bool) -> pd.DataFrame:
    frame = basis.frame(weights=None if parent else np.asarray(fit["weights"]), indices=indices, parent=parent)
    return shared.predict_layer(frame, layer, np.asarray(fit["eta"]), fit["effects"])


def parameter_row(model_id: str, fold_id: str, layer: str, mechanism: str, fit: dict[str, object]) -> dict[str, object]:
    diagnostic = fit["diagnostic"]
    row = {
        "model_id": model_id, "fold_id": fold_id, "layer": layer, "mechanism": mechanism,
        **weight_diagnostics(float(fit["a"]), float(fit["b"])),
        "eta_quick": float(fit["eta"][0]), "eta_gw": float(fit["eta"][1]),
        "eta_boundary": bool(np.any(np.asarray(fit["eta"]) <= 0.01) or np.any(np.asarray(fit["eta"]) >= 0.99)),
        "training_station_macro_rmse_log1p": float(fit["objective"]),
        "outer_success": bool(fit["outer_success"]), "outer_nit": int(fit["outer_nit"]), "outer_nfev": int(fit["outer_nfev"]),
        "outer_start_id": str(fit["start_id"]), "inner_success": bool(diagnostic["success"]), "inner_nfev": int(diagnostic["nfev"]),
    }
    return row


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    return parent_shared().metric_values(frame)


def paired_bootstrap(reference: pd.DataFrame, candidate: pd.DataFrame, block: str, seed_offset: int = 0) -> dict[str, object]:
    keys = ["station_key", "year", "month", "fold_id"]
    reference_columns = list(dict.fromkeys(keys + ["tn_mg_l", "pred_tn_mg_l", block]))
    joined = reference[reference_columns].merge(
        candidate[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_parent", "_candidate"), validate="one_to_one"
    )
    deltas = []
    for _, group in joined.groupby(block):
        obs = np.log1p(group.tn_mg_l.to_numpy(float))
        pr = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_parent.to_numpy(float)) - obs) ** 2))
        cr = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate.to_numpy(float)) - obs) ** 2))
        deltas.append(float(cr - pr))
    values = np.asarray(deltas)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    dist = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1)
    lo, hi = np.quantile(dist, [0.025, 0.975])
    point = station_macro_rmse(candidate) - station_macro_rmse(reference) if block == "station_key" else tree_macro_rmse(candidate) - tree_macro_rmse(reference)
    return {"blocks": int(len(values)), "point_delta": float(point), "ci_lower": float(lo), "ci_upper": float(hi),
            "noninferior": bool(hi < NONINFERIOR_MARGIN), "predictively_improved": bool(hi < 0)}


def season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"
