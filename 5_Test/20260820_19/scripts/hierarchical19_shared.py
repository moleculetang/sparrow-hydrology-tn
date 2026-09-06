from __future__ import annotations

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
from scipy.optimize import minimize, minimize_scalar


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260820_19"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
CACHE = HERE / "cache"
OBS = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLDS = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
LOCAL = TEST / "20260820_10" / "cache" / "parent_local"
PARENT_ROUTED = TEST / "20260820_12" / "cache" / "direct_parent_routed"
EXPOSURE = TEST / "20260820_18" / "outputs" / "reach_month_aquatic_uptake_exposure_2006_2021.parquet"
DOMAIN = OUT / "observation_domain_registry.parquet"
COVARIATES = OUT / "reach_spatial_hydraulic_covariates.parquet"
PARENT_SHARED = TEST / "20260818_1" / "scripts" / "legacy18_shared.py"

FORMAL_MUS = (12, 36, 60, 96, 144, 240)
FORMAL_MODELS = tuple(sorted(
    [f"S0_mu_{mu:03d}m" for mu in FORMAL_MUS]
    + [f"S1_tau_012m_mu_{mu:03d}m" for mu in FORMAL_MUS]
))
STRUCTURES = ("H0_PARENT", "H1_GLOBAL", "H2_COVARIATE", "H3_TREE_PARTIAL_POOL")
COVARIATE_COLUMNS = (
    "z_log_cumulative_area",
    "z_q72_groundwater_fraction_median",
    "z_q72_monthly_flow_cv",
    "q72_reservoir_influenced",
)
VF_MAX = 0.5
PRIOR_SD = 0.35
TREE_PRIOR_SD = 0.35
DELTA_PRIOR_SD = 0.35
STATION_LAMBDA = 12.0
ETA_LAMBDA = 1.0
TIE_TOL = 1e-8


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_shared() -> ModuleType:
    return load_module(PARENT_SHARED, "hierarchical19_parent_shared")


def development_observations() -> pd.DataFrame:
    obs = pd.read_parquet(OBS, filters=[("year", "<=", 2021)])
    if obs.year.eq(2022).any() or obs.year.min() != 2016 or obs.year.max() != 2021:
        raise RuntimeError("STOP_DEVELOPMENT_TN_BOUNDARY")
    registry = pd.read_parquet(DOMAIN, columns=["station_key", "primary_river_domain"])
    obs = obs.merge(registry, on="station_key", validate="many_to_one")
    obs = obs.loc[obs.primary_river_domain].drop(columns="primary_river_domain")
    return obs.reset_index(drop=True)


def fold_registry() -> pd.DataFrame:
    folds = pd.read_parquet(FOLDS)[
        ["fold_id", "train_start_year", "train_end_year", "evaluation_year"]
    ].drop_duplicates().sort_values("evaluation_year")
    if set(folds.evaluation_year) != {2018, 2019, 2020, 2021} or len(folds) != 4:
        raise RuntimeError("STOP_OOF_FOLD_CONTRACT")
    return folds


@dataclass
class HydraulicRouter:
    model_id: str
    reach_ids: np.ndarray
    years: np.ndarray
    months: np.ndarray
    local_q: np.ndarray
    local_g: np.ndarray
    h_full: np.ndarray
    h_mid: np.ndarray
    water: np.ndarray
    x: np.ndarray
    terminal_by_reach: np.ndarray
    order_index: list[int]
    downstream_index: dict[int, tuple[int, float]]

    def __post_init__(self) -> None:
        self.reach_lookup = {int(v): i for i, v in enumerate(self.reach_ids)}
        self.time_lookup = {
            (int(y), int(m)): i for i, (y, m) in enumerate(zip(self.years, self.months))
        }
        self._route_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def route(self, vf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(vf, dtype=float)
        if vector.shape != (len(self.reach_ids),):
            raise ValueError("vf vector shape mismatch")
        if np.any(vector < 0) or np.any(vector > VF_MAX + 1e-12):
            raise ValueError("effective vf outside registered bounds")
        key = np.round(vector, 10).tobytes()
        if key in self._route_cache:
            return self._route_cache[key]
        sf = np.exp(-self.h_full * vector[None, :])
        sm = np.exp(-self.h_mid * vector[None, :])
        uq = np.zeros_like(self.local_q)
        ug = np.zeros_like(self.local_g)
        oq = np.zeros_like(self.local_q)
        og = np.zeros_like(self.local_g)
        for i in self.order_index:
            oq[:, i] = uq[:, i] * sf[:, i] + self.local_q[:, i] * sm[:, i]
            og[:, i] = ug[:, i] * sf[:, i] + self.local_g[:, i] * sm[:, i]
            if i in self.downstream_index:
                down, fraction = self.downstream_index[i]
                uq[:, down] += fraction * oq[:, i]
                ug[:, down] += fraction * og[:, i]
        self._route_cache[key] = (oq, og)
        return oq, og

    def frame(self, observations: pd.DataFrame, vf: np.ndarray) -> pd.DataFrame:
        oq, og = self.route(vf)
        ridx = observations.reach_id.astype(int).map(self.reach_lookup).to_numpy(int)
        tidx = np.fromiter(
            (self.time_lookup[(int(y), int(m))] for y, m in observations[["year", "month"]].itertuples(index=False)),
            dtype=int,
            count=len(observations),
        )
        out = observations.copy()
        out["routed_quick_tn_kg_n"] = oq[tidx, ridx]
        out["routed_gw_tn_kg_n"] = og[tidx, ridx]
        out["routed_water_volume_m3"] = self.water[tidx, ridx]
        out["terminal_tree_id"] = self.terminal_by_reach[ridx]
        for j, column in enumerate(COVARIATE_COLUMNS):
            out[column] = self.x[ridx, j]
        return out


def build_router(model_id: str, shared: ModuleType) -> HydraulicRouter:
    local = pd.read_parquet(LOCAL / f"{model_id}.parquet")
    local = local.loc[local.year.between(2016, 2021)].sort_values(["year", "month", "reach_id"])
    exp = pd.read_parquet(EXPOSURE)
    exp = exp.loc[exp.year.between(2016, 2021)].sort_values(["year", "month", "reach_id"])
    parent = pd.read_parquet(PARENT_ROUTED / f"{model_id}.parquet")
    parent = parent.loc[parent.year.between(2016, 2021)].sort_values(["year", "month", "reach_id"])
    cov = pd.read_parquet(COVARIATES).sort_values("reach_id")
    reach_ids = np.sort(local.reach_id.unique().astype(int))
    times = local[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    keys = local[["reach_id", "year", "month"]].reset_index(drop=True)
    if not keys.equals(exp[["reach_id", "year", "month"]].reset_index(drop=True)):
        raise RuntimeError(f"exposure key mismatch: {model_id}")
    if not keys.equals(parent[["reach_id", "year", "month"]].reset_index(drop=True)):
        raise RuntimeError(f"parent key mismatch: {model_id}")
    if not np.array_equal(reach_ids, cov.reach_id.to_numpy(int)):
        raise RuntimeError("covariate reach mismatch")
    order, downstream, terminal = shared.topology_operators(reach_ids)
    lookup = {int(v): i for i, v in enumerate(reach_ids)}
    shape = (len(times), len(reach_ids))
    return HydraulicRouter(
        model_id=model_id,
        reach_ids=reach_ids,
        years=times.year.to_numpy(int),
        months=times.month.to_numpy(int),
        local_q=local.quick_tn_release_kg_n.to_numpy(float).reshape(shape),
        local_g=local.gw_tn_release_kg_n.to_numpy(float).reshape(shape),
        h_full=exp.uptake_exposure_full_days_per_m.to_numpy(float).reshape(shape),
        h_mid=exp.uptake_exposure_midpoint_to_outlet_days_per_m.to_numpy(float).reshape(shape),
        water=parent.routed_water_volume_m3.to_numpy(float).reshape(shape),
        x=cov[list(COVARIATE_COLUMNS)].to_numpy(float),
        terminal_by_reach=np.array([terminal[int(v)] for v in reach_ids], dtype=int),
        order_index=[lookup[int(v)] for v in order],
        downstream_index={lookup[int(k)]: (lookup[int(v[0])], float(v[1])) for k, v in downstream.items()},
    )


def station_macro_rmse(frame: pd.DataFrame) -> float:
    return float(np.mean([
        np.sqrt(np.mean((np.log1p(g.pred_tn_mg_l) - np.log1p(g.tn_mg_l)) ** 2))
        for _, g in frame.groupby("station_key")
    ]))


def tree_macro_rmse(frame: pd.DataFrame) -> float:
    return float(np.mean([
        np.sqrt(np.mean((np.log1p(g.pred_tn_mg_l) - np.log1p(g.tn_mg_l)) ** 2))
        for _, g in frame.groupby("terminal_tree_id")
    ]))


def macro_objective(frame: pd.DataFrame) -> float:
    return 0.5 * (station_macro_rmse(frame) + tree_macro_rmse(frame))


def vf_from_parameters(
    router: HydraulicRouter,
    structure: str,
    parameters: np.ndarray,
    tree_levels: tuple[int, ...],
) -> np.ndarray:
    if structure == "H0_PARENT":
        return np.zeros(len(router.reach_ids), dtype=float)
    v0 = float(parameters[0])
    if structure == "H1_GLOBAL":
        linear = np.zeros(len(router.reach_ids), dtype=float)
    else:
        linear = router.x @ np.asarray(parameters[1:5], dtype=float)
        if structure == "H3_TREE_PARTIAL_POOL":
            effects = {tree: float(parameters[5 + i]) for i, tree in enumerate(tree_levels)}
            linear += np.array([effects.get(int(tree), 0.0) for tree in router.terminal_by_reach])
    return v0 * np.exp(linear)


def _prior_penalty(structure: str, parameters: np.ndarray, n_blocks: int) -> float:
    if structure in {"H0_PARENT", "H1_GLOBAL"}:
        return 0.0
    gamma = np.asarray(parameters[1:5], dtype=float)
    value = np.sum((gamma / PRIOR_SD) ** 2)
    if structure == "H3_TREE_PARTIAL_POOL":
        value += np.sum((np.asarray(parameters[5:], dtype=float) / TREE_PRIOR_SD) ** 2)
    return float(0.5 * value / max(n_blocks, 1))


def fit_structure(
    router: HydraulicRouter,
    train_obs: pd.DataFrame,
    structure: str,
    shared: ModuleType,
    warm: dict[str, object] | None = None,
) -> dict[str, object]:
    if "terminal_tree_id" not in train_obs:
        reach_to_tree = dict(zip(router.reach_ids.astype(int), router.terminal_by_reach.astype(int)))
        train_obs = train_obs.copy()
        train_obs["terminal_tree_id"] = train_obs.reach_id.astype(int).map(reach_to_tree).astype(int)
    trees = tuple(sorted(train_obs.terminal_tree_id.astype(int).unique()))
    n_blocks = train_obs.station_key.nunique() + len(trees)
    reach_index = train_obs.reach_id.astype(int).map(router.reach_lookup).to_numpy(int)
    time_index = np.fromiter(
        (router.time_lookup[(int(y), int(m))] for y, m in train_obs[["year", "month"]].itertuples(index=False)),
        dtype=int, count=len(train_obs),
    )
    observed_log = np.log1p(train_obs.tn_mg_l.to_numpy(float))
    station_code = pd.Categorical(train_obs.station_key.astype(str)).codes
    tree_code = pd.Categorical(train_obs.terminal_tree_id.astype(int)).codes
    n_station = int(station_code.max()) + 1
    n_tree = int(tree_code.max()) + 1
    water_obs = router.water[time_index, reach_index]
    cache: dict[tuple[float, ...], dict[str, object]] = {}

    def evaluate(raw: np.ndarray) -> dict[str, object]:
        x = np.asarray(raw, dtype=float)
        key = tuple(np.round(x, 9))
        if key not in cache:
            vf = vf_from_parameters(router, structure, x, trees)
            if np.any(~np.isfinite(vf)) or np.max(vf) > VF_MAX + 1e-12:
                overflow = max(float(np.nanmax(vf)) - VF_MAX, 0.0) if np.isfinite(vf).any() else 1.0
                cache[key] = {"objective": 1000.0 + 1000.0 * overflow, "valid": False}
            else:
                routed_q, routed_g = router.route(vf)
                quick_obs = routed_q[time_index, reach_index]
                gw_obs = routed_g[time_index, reach_index]

                def eta_residual(eta_value: np.ndarray) -> np.ndarray:
                    concentration = np.divide(
                        (eta_value[0] * quick_obs + eta_value[1] * gw_obs) * 1000.0,
                        water_obs, out=np.zeros_like(water_obs), where=water_obs > 1e-12,
                    )
                    return np.concatenate([
                        np.log1p(np.maximum(concentration, 0.0)) - observed_log,
                        np.sqrt(ETA_LAMBDA) * (eta_value - 1.0),
                    ])

                from scipy.optimize import least_squares
                eta_opt = least_squares(
                    eta_residual, np.array([0.8, 0.8]), bounds=(np.zeros(2), np.ones(2)),
                    xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=500,
                )
                eta = eta_opt.x
                data_residual = eta_residual(eta)[:len(observed_log)]
                station_rmse = np.mean([
                    np.sqrt(np.mean(data_residual[station_code == i] ** 2)) for i in range(n_station)
                ])
                tree_rmse = np.mean([
                    np.sqrt(np.mean(data_residual[tree_code == i] ** 2)) for i in range(n_tree)
                ])
                data_score = float(0.5 * (station_rmse + tree_rmse))
                penalty = _prior_penalty(structure, x, n_blocks)
                cache[key] = {
                    "objective": data_score + penalty,
                    "data_objective": data_score,
                    "prior_penalty": penalty,
                    "parameters": x,
                    "vf": vf,
                    "eta": eta,
                    "effects": {},
                    "diagnostic": {
                        "success": bool(eta_opt.success), "nfev": int(eta_opt.nfev),
                        "eta_boundary": bool(np.any(eta <= 0.01) or np.any(eta >= 0.99)),
                    },
                    "valid": True,
                }
        return cache[key]

    if structure == "H0_PARENT":
        result = evaluate(np.empty(0, dtype=float))
        result.update({"outer_success": True, "outer_nfev": 1, "start_id": "exact_parent"})
        return result
    if structure == "H1_GLOBAL":
        grid = np.array([0.0, 0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5])
        values = [evaluate(np.array([v])) for v in grid]
        k = int(np.argmin([row["objective"] for row in values]))
        if 0 < k < len(grid) - 1:
            opt = minimize_scalar(
                lambda v: evaluate(np.array([v]))["objective"],
                bounds=(float(grid[k - 1]), float(grid[k + 1])),
                method="bounded",
                options={"xatol": 1e-6, "maxiter": 60},
            )
            evaluate(np.array([opt.x]))
            success, nfev = bool(opt.success), int(opt.nfev)
        else:
            success, nfev = True, 0
        best = min((v for v in cache.values() if v.get("valid")), key=lambda row: (row["objective"], row["parameters"][0]))
        best.update({"outer_success": success, "outer_nfev": nfev + len(grid), "start_id": "grid_bounded"})
        return best

    size = 5 + (len(trees) if structure == "H3_TREE_PARTIAL_POOL" else 0)
    vstart = 0.02 if warm is None else float(np.asarray(warm["parameters"])[0])
    base = np.zeros(size, dtype=float)
    base[0] = np.clip(vstart, 1e-6, VF_MAX)
    if structure == "H3_TREE_PARTIAL_POOL" and warm is not None and len(np.asarray(warm["parameters"])) >= 5:
        base[1:5] = np.asarray(warm["parameters"], dtype=float)[1:5]
    starts = [("zero", base.copy())]
    direction = np.resize(np.array([1.0, -1.0]), size - 1)
    for label, sign in (("symmetric_plus", 1.0), ("symmetric_minus", -1.0)):
        value = base.copy()
        value[1:] = sign * 0.08 * direction
        starts.append((label, value))
    bounds = [(0.0, VF_MAX)] + [(-1.4, 1.4)] * (size - 1)
    candidates: list[dict[str, object]] = []
    for start_id, x0 in starts:
        opt = minimize(
            lambda z: evaluate(z)["objective"],
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-10, "gtol": 1e-7, "maxiter": 100, "maxfun": 2500},
        )
        row = evaluate(opt.x)
        if row.get("valid"):
            candidates.append({**row, "outer_success": bool(opt.success), "outer_nfev": int(opt.nfev), "start_id": start_id})
    if not candidates:
        raise RuntimeError(f"no valid optimizer result: {structure}")
    candidates.sort(key=lambda row: (float(row["objective"]), float(np.linalg.norm(row["parameters"]))))
    return candidates[0]


def select_structure(fits: dict[str, dict[str, object]]) -> str:
    best = min(float(v["objective"]) for v in fits.values())
    tied = [name for name in STRUCTURES if float(fits[name]["objective"]) <= best + TIE_TOL]
    return tied[0]


def fit_all_structures(router: HydraulicRouter, train_obs: pd.DataFrame, shared: ModuleType) -> dict[str, dict[str, object]]:
    h0 = fit_structure(router, train_obs, "H0_PARENT", shared)
    h1 = fit_structure(router, train_obs, "H1_GLOBAL", shared)
    h2 = fit_structure(router, train_obs, "H2_COVARIATE", shared, h1)
    h3 = fit_structure(router, train_obs, "H3_TREE_PARTIAL_POOL", shared, h2)
    return {"H0_PARENT": h0, "H1_GLOBAL": h1, "H2_COVARIATE": h2, "H3_TREE_PARTIAL_POOL": h3}


def _effect_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    trees = sorted(frame.terminal_tree_id.astype(int).unique())
    stations = sorted(frame.station_key.astype(str).unique())
    tree_lookup = {v: i for i, v in enumerate(trees)}
    station_lookup = {v: i for i, v in enumerate(stations)}
    z = frame[list(COVARIATE_COLUMNS)].to_numpy(float)
    tree = np.zeros((len(frame), len(trees)), dtype=float)
    station = np.zeros((len(frame), len(stations)), dtype=float)
    tree[np.arange(len(frame)), frame.terminal_tree_id.astype(int).map(tree_lookup).to_numpy(int)] = 1.0
    station[np.arange(len(frame)), frame.station_key.astype(str).map(station_lookup).to_numpy(int)] = 1.0
    return np.column_stack([z, tree, station]), list(COVARIATE_COLUMNS), trees, stations


def fit_p2r(train: pd.DataFrame, shared: ModuleType) -> dict[str, object]:
    design, _, trees, stations = _effect_design(train)
    n_cov, n_tree = len(COVARIATE_COLUMNS), len(trees)
    penalty_diag = np.concatenate([
        np.full(n_cov, 1.0 / DELTA_PRIOR_SD ** 2),
        np.full(n_tree, 1.0 / TREE_PRIOR_SD ** 2),
        np.full(len(stations), STATION_LAMBDA),
    ])
    observed = np.log1p(train.tn_mg_l.to_numpy(float))

    def solve_effects(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.log1p(np.maximum(shared.concentration(train, eta), 0.0))
        target = observed - raw
        lhs = design.T @ design + np.diag(penalty_diag)
        coefficient = np.linalg.solve(lhs, design.T @ target)
        return coefficient, raw + design @ coefficient - observed

    def residual(eta: np.ndarray) -> np.ndarray:
        coefficient, data = solve_effects(eta)
        return np.concatenate([data, np.sqrt(penalty_diag) * coefficient, np.sqrt(ETA_LAMBDA) * (eta - 1.0)])

    from scipy.optimize import least_squares
    opt = least_squares(
        residual, np.array([0.8, 0.8]), bounds=(np.zeros(2), np.ones(2)),
        xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=1000,
    )
    coefficient, _ = solve_effects(opt.x)
    return {
        "eta": opt.x,
        "delta": coefficient[:n_cov],
        "tree_effects": {int(v): float(coefficient[n_cov + i]) for i, v in enumerate(trees)},
        "station_effects": {str(v): float(coefficient[n_cov + n_tree + i]) for i, v in enumerate(stations)},
        "diagnostic": {
            "success": bool(opt.success), "nfev": int(opt.nfev),
            "eta_boundary": bool(np.any(opt.x <= 0.01) or np.any(opt.x >= 0.99)),
        },
    }


def predict_p2r(frame: pd.DataFrame, fit: dict[str, object], shared: ModuleType) -> pd.DataFrame:
    out = frame.copy()
    eta = np.asarray(fit["eta"], dtype=float)
    raw = shared.concentration(out, eta)
    effect = out[list(COVARIATE_COLUMNS)].to_numpy(float) @ np.asarray(fit["delta"], dtype=float)
    effect += out.terminal_tree_id.astype(int).map(fit["tree_effects"]).fillna(0.0).to_numpy(float)
    effect += out.station_key.astype(str).map(fit["station_effects"]).fillna(0.0).to_numpy(float)
    out["raw_eta_scaled_tn_mg_l"] = raw
    out["pred_tn_mg_l"] = np.maximum(np.expm1(np.log1p(np.maximum(raw, 0.0)) + effect), 0.0)
    out["layer"] = "P2R"
    out["eta_quick"] = float(eta[0])
    out["eta_gw"] = float(eta[1])
    out["station_effect"] = out.station_key.astype(str).map(fit["station_effects"]).fillna(0.0).to_numpy(float)
    out["regionalized_effect"] = effect - out.station_effect
    return out


def fit_selected_readout(train: pd.DataFrame, layer: str, shared: ModuleType) -> dict[str, object]:
    if layer == "P2R":
        return fit_p2r(train, shared)
    eta, effects, diagnostic = shared.fit_readout(train, layer)
    return {"eta": eta, "effects": effects, "diagnostic": diagnostic}


def predict_selected(test: pd.DataFrame, layer: str, fit: dict[str, object], shared: ModuleType) -> pd.DataFrame:
    if layer == "P2R":
        return predict_p2r(test, fit, shared)
    return shared.predict_layer(test, layer, np.asarray(fit["eta"]), fit["effects"])


def structure_parameter_row(
    model_id: str, fold_id: str, structure: str, fit: dict[str, object], selected: bool,
    evaluation: str = "temporal", holdout_id: str = "",
) -> dict[str, object]:
    params = np.asarray(fit["parameters"], dtype=float)
    vf = np.asarray(fit["vf"], dtype=float)
    row: dict[str, object] = {
        "model_id": model_id, "fold_id": fold_id, "evaluation": evaluation,
        "holdout_id": holdout_id, "structure": structure, "selected": selected,
        "training_objective": float(fit["objective"]),
        "training_data_objective": float(fit["data_objective"]),
        "prior_penalty": float(fit["prior_penalty"]),
        "v0_m_per_day": float(params[0]) if len(params) else 0.0,
        "vf_min_m_per_day": float(vf.min()), "vf_median_m_per_day": float(np.median(vf)),
        "vf_max_m_per_day": float(vf.max()),
        "vf_boundary": bool(vf.max() >= 0.49),
        "eta_quick": float(fit["eta"][0]), "eta_gw": float(fit["eta"][1]),
        "eta_boundary": bool(fit["diagnostic"]["eta_boundary"]),
        "outer_success": bool(fit["outer_success"]), "outer_nfev": int(fit["outer_nfev"]),
        "start_id": str(fit["start_id"]),
        "parameter_json": json.dumps(params.tolist()),
    }
    return row


def paired_bootstrap(reference: pd.DataFrame, candidate: pd.DataFrame, block: str, seed: int) -> dict[str, object]:
    keys = ["station_key", "year", "month", "fold_id"]
    left_columns = list(dict.fromkeys(keys + ["tn_mg_l", "pred_tn_mg_l", block]))
    left = reference[left_columns]
    joined = left.merge(candidate[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_ref", "_cand"), validate="one_to_one")
    deltas = []
    for _, group in joined.groupby(block):
        obs = np.log1p(group.tn_mg_l.to_numpy(float))
        a = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_ref) - obs) ** 2))
        b = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_cand) - obs) ** 2))
        deltas.append(float(b - a))
    values = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    dist = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    lo, hi = np.quantile(dist, [0.025, 0.975])
    return {
        "blocks": len(values), "point_delta": float(values.mean()),
        "ci_lower": float(lo), "ci_upper": float(hi),
        "noninferior": bool(hi < 0.005), "predictively_improved": bool(hi < 0.0),
    }


def exact_nesting_audit(router: HydraulicRouter) -> pd.DataFrame:
    rows = []
    trees = tuple(sorted(set(map(int, router.terminal_by_reach))))
    definitions = {
        "H0_vs_H1_v0_zero": (np.zeros(len(router.reach_ids)), vf_from_parameters(router, "H1_GLOBAL", np.array([0.0]), trees)),
        "H1_vs_H2_gamma_zero": (vf_from_parameters(router, "H1_GLOBAL", np.array([0.05]), trees), vf_from_parameters(router, "H2_COVARIATE", np.array([0.05, 0, 0, 0, 0]), trees)),
        "H2_vs_H3_u_zero": (vf_from_parameters(router, "H2_COVARIATE", np.array([0.05, 0.1, -0.1, 0.05, -0.05]), trees), vf_from_parameters(router, "H3_TREE_PARTIAL_POOL", np.r_[0.05, 0.1, -0.1, 0.05, -0.05, np.zeros(len(trees))], trees)),
    }
    for name, (a, b) in definitions.items():
        aq, ag = router.route(a)
        bq, bg = router.route(b)
        scale = max(float(np.max(np.abs(aq))), float(np.max(np.abs(ag))), 1.0)
        error = max(float(np.max(np.abs(aq - bq))), float(np.max(np.abs(ag - bg)))) / scale
        rows.append({"test": name, "relative_error": error, "pass": bool(error <= 1e-12)})
    return pd.DataFrame(rows)
