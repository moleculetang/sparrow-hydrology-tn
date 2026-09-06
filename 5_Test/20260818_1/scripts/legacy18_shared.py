from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
S16_1 = TEST / "20260816_1"
S16_4 = TEST / "20260816_4"
S17_1 = TEST / "20260817_1"
S17_4 = TEST / "20260817_4"
S17_6 = TEST / "20260817_6"
S17_9 = TEST / "20260817_9"

PARENT_CORE_PATH = S16_1 / "scripts" / "legacy16_core.py"
OBS_PATH = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLD_PATH = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
TOPOLOGY_PATH = TEST / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
MONTHLY_PATH = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"

FORMAL_MUS = (12, 36, 60, 96, 144, 240)
OPERATORS = ("F00", "F10", "F01", "F11")
Q_RHO = 0.25
B_RHO = 0.85
WATER_EPS = 1e-12
ETA_LAMBDA = 1.0
STATION_LAMBDA = 12.0
ETA_BOUNDARY_TOL = 0.01
SPINUP_TOL_KG = 1e-9
SPINUP_MAX_CYCLES = 5000
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_REPLICATES = 10000
NONINFERIOR_MARGIN = 0.005


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow conda environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


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


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_core() -> ModuleType:
    return load_module(PARENT_CORE_PATH, "legacy18_frozen_parent_core")


def formal_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for mu in FORMAL_MUS:
        specs.append({
            "model_id": f"S0_mu_{mu:03d}m",
            "source_structure": "S0",
            "soil_tau_month": None,
            "delivery_mu_month": mu,
        })
        specs.append({
            "model_id": f"S1_tau_012m_mu_{mu:03d}m",
            "source_structure": "S1",
            "soil_tau_month": 12,
            "delivery_mu_month": mu,
        })
    return sorted(specs, key=lambda x: str(x["model_id"]))


def topology_operators(reach_ids: np.ndarray) -> tuple[list[int], dict[int, tuple[int, float]], dict[int, int]]:
    topo = pd.read_csv(TOPOLOGY_PATH)
    topo["reach_id"] = topo.reach_id.astype(int)
    downstream: dict[int, tuple[int, float]] = {}
    for row in topo.itertuples():
        if pd.notna(row.downstream_reach):
            downstream[int(row.reach_id)] = (int(row.downstream_reach), float(row.frac))
    nodes = list(map(int, reach_ids))
    indegree = {rid: 0 for rid in nodes}
    for rid, (down, _) in downstream.items():
        if rid in indegree and down in indegree:
            indegree[down] += 1
    queue = deque(sorted(rid for rid, degree in indegree.items() if degree == 0))
    order: list[int] = []
    while queue:
        rid = queue.popleft()
        order.append(rid)
        if rid in downstream:
            down = downstream[rid][0]
            indegree[down] -= 1
            if indegree[down] == 0:
                queue.append(down)
    if len(order) != len(nodes):
        raise RuntimeError("topology is cyclic or incomplete")
    terminal: dict[int, int] = {}
    for rid in nodes:
        current = rid
        seen: set[int] = set()
        while current in downstream:
            if current in seen:
                raise RuntimeError("topology cycle")
            seen.add(current)
            current = downstream[current][0]
        terminal[rid] = current
    return order, downstream, terminal


def route_arrays(local: np.ndarray, reach_ids: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    order, downstream, terminal = topology_operators(reach_ids)
    index = {int(rid): i for i, rid in enumerate(reach_ids)}
    routed = np.asarray(local, dtype=float).copy()
    for rid in order:
        if rid in downstream:
            down, frac = downstream[rid]
            routed[:, index[down]] += frac * routed[:, index[rid]]
    return routed, terminal


def prepare_arrays() -> tuple[np.ndarray, list[tuple[int, int]], dict[str, np.ndarray], np.ndarray]:
    core = parent_core()
    return core.prepare_model_arrays()


def route_candidate(frame: pd.DataFrame, reach_ids: np.ndarray) -> pd.DataFrame:
    times = frame[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    n_t = len(times)
    n_r = len(reach_ids)
    ordered = frame.sort_values(["year", "month", "reach_id"])
    q = ordered.quick_tn_release_kg_n.to_numpy(float).reshape(n_t, n_r)
    g = ordered.gw_tn_release_kg_n.to_numpy(float).reshape(n_t, n_r)
    water = (
        ordered.q_local_total_mm.to_numpy(float)
        * ordered.catchment_area_km2.to_numpy(float)
        * 1000.0
    ).reshape(n_t, n_r)
    rq, terminal = route_arrays(q, reach_ids)
    rg, _ = route_arrays(g, reach_ids)
    rw, _ = route_arrays(water, reach_ids)
    out = pd.DataFrame({
        "reach_id": np.tile(reach_ids, n_t),
        "year": np.repeat(times.year.to_numpy(int), n_r),
        "month": np.repeat(times.month.to_numpy(int), n_r),
        "routed_quick_tn_kg_n": rq.reshape(-1),
        "routed_gw_tn_kg_n": rg.reshape(-1),
        "routed_water_volume_m3": rw.reshape(-1),
    })
    out["terminal_tree_id"] = out.reach_id.map(terminal).astype(int)
    return out


def hydrology_registry(year_start: int = 2016, year_end: int = 2021) -> pd.DataFrame:
    monthly = pd.read_parquet(MONTHLY_PATH)
    monthly = monthly.loc[monthly.year.between(year_start, year_end)].copy()
    reach_ids = np.sort(monthly.reach_id.unique().astype(int))
    times = monthly[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    ordered = monthly.sort_values(["year", "month", "reach_id"])
    n_t, n_r = len(times), len(reach_ids)
    area = ordered.catchment_area_km2.to_numpy(float).reshape(n_t, n_r)
    qlocal = ordered.quick_release_mm.to_numpy(float).reshape(n_t, n_r) * area * 1000.0
    glocal = ordered.gw_discharge_mm.to_numpy(float).reshape(n_t, n_r) * area * 1000.0
    qr, terminal = route_arrays(qlocal, reach_ids)
    gr, _ = route_arrays(glocal, reach_ids)
    total = qr + gr
    frac = np.divide(qr, total, out=np.zeros_like(qr), where=total > WATER_EPS)
    out = pd.DataFrame({
        "reach_id": np.tile(reach_ids, n_t),
        "year": np.repeat(times.year.to_numpy(int), n_r),
        "month": np.repeat(times.month.to_numpy(int), n_r),
        "routed_quick_water_m3": qr.reshape(-1),
        "routed_gw_water_m3": gr.reshape(-1),
        "routed_total_water_m3": total.reshape(-1),
        "routed_quick_fraction": frac.reshape(-1),
    })
    out["terminal_tree_id"] = out.reach_id.map(terminal).astype(int)
    return out


def flow_regime_registry(observations: pd.DataFrame, folds: pd.DataFrame, hydro: pd.DataFrame) -> pd.DataFrame:
    station_reaches = observations[["station_key", "reach_id"]].drop_duplicates()
    definitions = folds[["fold_id", "train_start_year", "train_end_year", "evaluation_year"]].drop_duplicates()
    rows: list[pd.DataFrame] = []
    for fold in definitions.itertuples():
        train_h = hydro.loc[hydro.year.between(fold.train_start_year, fold.train_end_year)].merge(
            station_reaches, on="reach_id", how="inner", validate="many_to_many"
        )
        thresholds = train_h.groupby("station_key", as_index=False).agg(
            flow_q20_m3=("routed_total_water_m3", lambda x: float(np.quantile(x, 0.2))),
            flow_q80_m3=("routed_total_water_m3", lambda x: float(np.quantile(x, 0.8))),
            quick_fraction_q80=("routed_quick_fraction", lambda x: float(np.quantile(x, 0.8))),
            threshold_months=("month", "size"),
        )
        evaluate = hydro.loc[hydro.year.eq(fold.evaluation_year)].merge(
            station_reaches, on="reach_id", how="inner", validate="many_to_many"
        ).merge(thresholds, on="station_key", how="left", validate="many_to_one")
        evaluate["fold_id"] = fold.fold_id
        evaluate["flow_regime"] = np.where(
            evaluate.routed_total_water_m3 <= evaluate.flow_q20_m3,
            "low",
            np.where(evaluate.routed_total_water_m3 >= evaluate.flow_q80_m3, "high", "middle"),
        )
        evaluate["high_quick_fraction"] = evaluate.routed_quick_fraction >= evaluate.quick_fraction_q80
        evaluate["wet_season"] = evaluate.month.between(4, 9)
        rows.append(evaluate)
    return pd.concat(rows, ignore_index=True)


def station_effects(log_residual: np.ndarray, station_index: np.ndarray, n_stations: int) -> np.ndarray:
    sums = np.bincount(station_index, weights=log_residual, minlength=n_stations)
    counts = np.bincount(station_index, minlength=n_stations).astype(float)
    return -sums / (counts + STATION_LAMBDA)


def concentration(
    frame: pd.DataFrame,
    eta: np.ndarray,
    point_column: str | None = None,
) -> np.ndarray:
    q = frame.routed_quick_tn_kg_n.to_numpy(float)
    g = frame.routed_gw_tn_kg_n.to_numpy(float)
    point = np.zeros(len(frame), dtype=float) if point_column is None else frame[point_column].fillna(0.0).to_numpy(float)
    water = frame.routed_water_volume_m3.to_numpy(float)
    return np.divide(
        (eta[0] * q + eta[1] * g + point) * 1000.0,
        water,
        out=np.zeros_like(water),
        where=water > WATER_EPS,
    )


def fit_readout(
    train: pd.DataFrame,
    layer: str,
    point_column: str | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, object]]:
    if layer not in {"P1", "P2"}:
        raise ValueError(layer)
    observed_log = np.log1p(train.tn_mg_l.to_numpy(float))
    levels = sorted(train.station_key.astype(str).unique())
    lookup = {station: i for i, station in enumerate(levels)}
    station_index = train.station_key.astype(str).map(lookup).to_numpy(int)

    def components(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.log1p(np.maximum(concentration(train, eta, point_column), 0.0)) - observed_log
        effects = (
            station_effects(raw, station_index, len(levels))
            if layer == "P2"
            else np.zeros(len(levels), dtype=float)
        )
        return raw + effects[station_index], effects

    def residual(eta: np.ndarray) -> np.ndarray:
        data_residual, effects = components(eta)
        pieces = [data_residual, np.sqrt(ETA_LAMBDA) * (eta - 1.0)]
        if layer == "P2":
            pieces.append(np.sqrt(STATION_LAMBDA) * effects)
        return np.concatenate(pieces)

    result = least_squares(
        residual,
        x0=np.array([0.8, 0.8]),
        bounds=(np.zeros(2), np.ones(2)),
        method="trf",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )
    _, effects = components(result.x)
    effect_map = {station: float(effects[i]) for i, station in enumerate(levels)}
    diagnostic = {
        "layer": layer,
        "eta_quick": float(result.x[0]),
        "eta_gw": float(result.x[1]),
        "success": bool(result.success),
        "status": int(result.status),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "station_effect_count": len(levels) if layer == "P2" else 0,
        "eta_boundary": bool(np.any(result.x <= ETA_BOUNDARY_TOL) or np.any(result.x >= 1.0 - ETA_BOUNDARY_TOL)),
    }
    return result.x, effect_map, diagnostic


def joined_observations(routed: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    return observations.merge(
        routed,
        on=["reach_id", "year", "month"],
        how="inner",
        validate="many_to_one",
    )


def predict_layer(
    test: pd.DataFrame,
    layer: str,
    eta: np.ndarray | None = None,
    effect_map: dict[str, float] | None = None,
    point_column: str | None = None,
) -> pd.DataFrame:
    out = test.copy()
    if layer == "P0":
        eta_use = np.array([1.0, 1.0])
        effects = np.zeros(len(out), dtype=float)
    else:
        if eta is None:
            raise ValueError("eta required")
        eta_use = eta
        effects = (
            out.station_key.astype(str).map(effect_map or {}).fillna(0.0).to_numpy(float)
            if layer == "P2"
            else np.zeros(len(out), dtype=float)
        )
    raw = concentration(out, eta_use, point_column)
    out["raw_eta_scaled_tn_mg_l"] = raw
    out["pred_tn_mg_l"] = np.maximum(np.expm1(np.log1p(np.maximum(raw, 0.0)) + effects), 0.0)
    out["layer"] = layer
    out["eta_quick"] = float(eta_use[0])
    out["eta_gw"] = float(eta_use[1])
    out["station_effect"] = effects
    return out


def temporal_oof(
    routed: pd.DataFrame,
    observations: pd.DataFrame,
    folds: pd.DataFrame,
    layers: tuple[str, ...] = ("P0", "P1", "P2"),
    point_column: str | None = None,
    evaluation_years: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined = joined_observations(routed, observations)
    definitions = folds[["fold_id", "train_start_year", "train_end_year", "evaluation_year"]].drop_duplicates().sort_values("evaluation_year")
    if evaluation_years is not None:
        definitions = definitions.loc[definitions.evaluation_year.isin(evaluation_years)]
    predictions: list[pd.DataFrame] = []
    params: list[dict[str, object]] = []
    effects_rows: list[dict[str, object]] = []
    for fold in definitions.itertuples():
        train = joined.loc[joined.year.between(fold.train_start_year, fold.train_end_year)].copy()
        test = joined.loc[joined.year.eq(fold.evaluation_year)].copy()
        for layer in layers:
            if layer == "P0":
                pred = predict_layer(test, layer, point_column=point_column)
                diagnostic = {"layer": layer, "eta_quick": 1.0, "eta_gw": 1.0, "success": True, "eta_boundary": True}
            else:
                eta, effect_map, diagnostic = fit_readout(train, layer, point_column)
                pred = predict_layer(test, layer, eta, effect_map, point_column)
                for station, value in effect_map.items():
                    effects_rows.append({"fold_id": fold.fold_id, "layer": layer, "station_key": station, "station_effect": value})
            pred["fold_id"] = fold.fold_id
            predictions.append(pred)
            params.append({"fold_id": fold.fold_id, "evaluation_year": int(fold.evaluation_year), **diagnostic})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(params), pd.DataFrame(effects_rows)


def spatial_holdout(
    routed: pd.DataFrame,
    observations: pd.DataFrame,
    holdout_column: str,
    point_column: str | None = None,
    year_start: int = 2016,
    year_end: int = 2021,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = joined_observations(routed, observations.loc[observations.year.between(year_start, year_end)])
    predictions: list[pd.DataFrame] = []
    params: list[dict[str, object]] = []
    for holdout in sorted(joined[holdout_column].dropna().unique(), key=str):
        test = joined.loc[joined[holdout_column].eq(holdout)].copy()
        train = joined.loc[~joined[holdout_column].eq(holdout)].copy()
        station_means = train.assign(y=np.log1p(train.tn_mg_l)).groupby("station_key").y.mean()
        baseline_log = float(station_means.mean())
        for layer in ("P0", "P1"):
            if layer == "P0":
                pred = predict_layer(test, layer, point_column=point_column)
                diagnostic = {"layer": layer, "eta_quick": 1.0, "eta_gw": 1.0, "success": True, "eta_boundary": True}
            else:
                eta, _, diagnostic = fit_readout(train, layer, point_column)
                pred = predict_layer(test, layer, eta, {}, point_column)
            pred["holdout_column"] = holdout_column
            pred["holdout_id"] = str(holdout)
            pred["baseline_log_station_equal"] = baseline_log
            predictions.append(pred)
            params.append({"holdout_column": holdout_column, "holdout_id": str(holdout), **diagnostic})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(params)


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame.tn_mg_l.to_numpy(float)
    predicted = frame.pred_tn_mg_l.to_numpy(float)
    lo = np.log1p(observed)
    lp = np.log1p(predicted)
    error = predicted - observed
    denom = float(np.sum((observed - observed.mean()) ** 2))
    log_denom = float(np.sum((lo - lo.mean()) ** 2))
    return {
        "n": int(len(frame)),
        "rmse_log1p": float(np.sqrt(np.mean((lp - lo) ** 2))),
        "rmse_raw_mg_l": float(np.sqrt(np.mean(error ** 2))),
        "mae_mg_l": float(np.mean(np.abs(error))),
        "pbias_percent": float(100.0 * np.sum(error) / np.sum(observed)) if np.sum(observed) else np.nan,
        "pearson_r": float(np.corrcoef(observed, predicted)[0, 1]) if len(frame) > 1 else np.nan,
        "raw_nse": float(1.0 - np.sum(error ** 2) / denom) if denom > 0 else np.nan,
        "log_nse": float(1.0 - np.sum((lp - lo) ** 2) / log_denom) if log_denom > 0 else np.nan,
    }


def station_macro_rmse(frame: pd.DataFrame) -> float:
    values = []
    for _, group in frame.groupby("station_key"):
        values.append(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2)))
    return float(np.mean(values))


def tree_macro_rmse(frame: pd.DataFrame) -> float:
    values = []
    for _, group in frame.groupby("terminal_tree_id"):
        values.append(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2)))
    return float(np.mean(values))


def spatial_skill(frame: pd.DataFrame) -> float:
    work = frame.copy()
    work["p1_sq"] = (np.log1p(work.pred_tn_mg_l) - np.log1p(work.tn_mg_l)) ** 2
    work["base_sq"] = (work.baseline_log_station_equal - np.log1p(work.tn_mg_l)) ** 2
    station = work.groupby("station_key", as_index=False)[["p1_sq", "base_sq"]].mean()
    p1 = float(station.p1_sq.mean())
    base = float(station.base_sq.mean())
    return float(1.0 - p1 / base) if base > 0 else np.nan


def spatial_skill_bootstrap(frame: pd.DataFrame, block_column: str, seed_offset: int = 0) -> tuple[float, float, float]:
    work = frame.copy()
    work["p1_sq"] = (np.log1p(work.pred_tn_mg_l) - np.log1p(work.tn_mg_l)) ** 2
    work["base_sq"] = (work.baseline_log_station_equal - np.log1p(work.tn_mg_l)) ** 2
    station = work.groupby([block_column, "station_key"], as_index=False)[["p1_sq", "base_sq"]].mean()
    groups = {key: g[["p1_sq", "base_sq"]].to_numpy(float) for key, g in station.groupby(block_column)}
    keys = list(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    values = np.zeros(BOOTSTRAP_REPLICATES, dtype=float)
    for i in range(BOOTSTRAP_REPLICATES):
        chosen = rng.integers(0, len(keys), len(keys))
        sampled = np.concatenate([groups[keys[j]] for j in chosen], axis=0)
        values[i] = 1.0 - sampled[:, 0].mean() / sampled[:, 1].mean()
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), float(np.mean(values))


def paired_rmse_bootstrap(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    block_column: str,
    seed_offset: int = 0,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    keys = ["station_key", "year", "month", "fold_id"]
    keep = list(dict.fromkeys(keys + ["tn_mg_l", "pred_tn_mg_l", block_column]))
    joined = parent[keep].merge(
        candidate[keys + ["pred_tn_mg_l"]],
        on=keys,
        suffixes=("_parent", "_candidate"),
        validate="one_to_one",
    )
    block_delta: list[float] = []
    for _, group in joined.groupby(block_column):
        observed_log = np.log1p(group.tn_mg_l.to_numpy(float))
        parent_rmse = float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_parent.to_numpy(float)) - observed_log) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate.to_numpy(float)) - observed_log) ** 2)))
        block_delta.append(candidate_rmse - parent_rmse)
    block_delta_array = np.asarray(block_delta, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    sampled_index = rng.integers(
        0,
        len(block_delta_array),
        size=(BOOTSTRAP_REPLICATES, len(block_delta_array)),
    )
    delta = block_delta_array[sampled_index].mean(axis=1)
    lo, hi = np.quantile(delta, [0.025, 0.975])
    point = (
        station_macro_rmse(candidate) - station_macro_rmse(parent)
        if block_column == "station_key"
        else tree_macro_rmse(candidate) - tree_macro_rmse(parent)
    )
    summary: dict[str, float | bool] = {
        "point_delta": float(point),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "point_improved": bool(point < 0),
        "point_nonworse": bool(point <= 0),
        "noninferior": bool(hi < NONINFERIOR_MARGIN),
        "predictively_improved": bool(hi < 0),
        "clear_failure": bool(point > 0.01 and lo > 0),
    }
    return delta, summary


def _withdraw(pool: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    removed = np.minimum(pool, demand)
    pool -= removed
    return removed, demand - removed


def operator_water_partitions(
    arrays: dict[str, np.ndarray],
    t: int,
    operator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if operator not in OPERATORS:
        raise ValueError(operator)
    positive_water = arrays["positive_input_mm"][t]
    quick_generated = arrays["quick_generated_mm"][t]
    fresh_bypass_enabled = operator in {"F00", "F01"}
    quick_contact_enabled = operator in {"F01", "F11"}
    bypass = (
        np.clip(
            np.divide(
                quick_generated,
                positive_water,
                out=np.zeros_like(positive_water),
                where=positive_water > WATER_EPS,
            ),
            0.0,
            1.0,
        )
        if fresh_bypass_enabled
        else np.zeros_like(positive_water)
    )
    overflow = arrays["soil_overflow_to_quick_mm"][t]
    recharge = arrays["gw_recharge_mm"][t]
    quick_contact = quick_generated if quick_contact_enabled else np.zeros_like(quick_generated)
    contact = overflow + recharge + quick_contact
    capacity = arrays["source_water_capacity_mm"][t]
    flush = np.where(
        contact > WATER_EPS,
        1.0 - np.exp(-contact / capacity),
        0.0,
    )
    quick_share = np.divide(
        overflow + quick_contact,
        contact,
        out=np.zeros_like(contact),
        where=contact > WATER_EPS,
    )
    gw_share = np.divide(
        recharge,
        contact,
        out=np.zeros_like(contact),
        where=contact > WATER_EPS,
    )
    return bypass, contact, flush, quick_share, gw_share


def spinup_operator(
    operator: str,
    structure: str,
    tau_s: int | None,
    mu_t: int,
    early_positive: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object], pd.DataFrame]:
    if structure not in {"S0", "S1"}:
        raise ValueError(structure)
    state = {name: np.zeros(len(early_positive), dtype=float) for name in ("son", "mobile", "quick", "gw")}
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_gw = B_RHO if mu_t == 0 else mu_t / (1.0 + mu_t)
    delta = np.inf
    final_balance = np.zeros(len(early_positive), dtype=float)
    for cycle in range(1, SPINUP_MAX_CYCLES + 1):
        before_cycle = np.concatenate([value.copy() for value in state.values()])
        for t in range(12):
            start_total = sum(value.astype(np.longdouble) for value in state.values())
            bypass, _, flush, quick_share, gw_share = operator_water_partitions(arrays, t, operator)
            direct = early_positive * bypass
            remaining = early_positive - direct
            if structure == "S0":
                state["mobile"] += remaining
            else:
                state["son"] += remaining
                mineralized = state["son"] * (1.0 - rho_s)
                state["son"] -= mineralized
                state["mobile"] += mineralized
            source_release = state["mobile"] * flush
            state["mobile"] -= source_release
            state["quick"] += direct + source_release * quick_share
            state["gw"] += source_release * gw_share
            quick_release = np.where(
                arrays["quick_release_mm"][t] > WATER_EPS,
                (1.0 - Q_RHO) * state["quick"],
                0.0,
            )
            gw_release = np.where(
                arrays["gw_discharge_mm"][t] > WATER_EPS,
                (1.0 - rho_gw) * state["gw"],
                0.0,
            )
            state["quick"] -= quick_release
            state["gw"] -= gw_release
            end_total = sum(value.astype(np.longdouble) for value in state.values())
            final_balance = np.asarray(
                early_positive.astype(np.longdouble)
                + start_total
                - quick_release.astype(np.longdouble)
                - gw_release.astype(np.longdouble)
                - end_total,
                dtype=float,
            )
        after_cycle = np.concatenate(list(state.values()))
        delta = float(np.max(np.abs(after_cycle - before_cycle)))
        if delta <= SPINUP_TOL_KG:
            break
    scale = max(float(sum(np.sum(np.abs(x)) for x in state.values())), 1.0)
    audit = {
        "operator": operator,
        "source_structure": structure,
        "soil_tau_month": tau_s,
        "delivery_mu_month": mu_t,
        "cycles": cycle,
        "converged": bool(delta <= SPINUP_TOL_KG),
        "terminal_max_abs_delta_kg_n": delta,
        "son_end_total_kg_n": float(state["son"].sum()),
        "mobile_end_total_kg_n": float(state["mobile"].sum()),
        "quick_end_total_kg_n": float(state["quick"].sum()),
        "gw_end_total_kg_n": float(state["gw"].sum()),
        "minimum_state_kg_n": float(min(x.min() for x in state.values())),
        "final_cycle_mass_balance_error_kg_n": float(np.max(np.abs(final_balance))),
        "final_cycle_mass_balance_relative_error": float(np.max(np.abs(final_balance)) / scale),
    }
    final_states = pd.DataFrame({
        "reach_index": np.arange(len(early_positive), dtype=int),
        "son_end_kg_n": state["son"],
        "mobile_end_kg_n": state["mobile"],
        "quick_end_kg_n": state["quick"],
        "gw_end_kg_n": state["gw"],
    })
    return state, audit, final_states


def simulate_operator(
    operator: str,
    model_id: str,
    structure: str,
    tau_s: int | None,
    mu_t: int,
    reach_ids: np.ndarray,
    times: list[tuple[int, int]],
    arrays: dict[str, np.ndarray],
    early_positive: np.ndarray,
    capture_start_year: int = 2016,
    capture_end_year: int = 2021,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], pd.DataFrame]:
    state, spinup, spinup_states = spinup_operator(operator, structure, tau_s, mu_t, early_positive, arrays)
    if not spinup["converged"]:
        raise RuntimeError(f"spinup did not converge: {operator} {model_id}")
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_gw = B_RHO if mu_t == 0 else mu_t / (1.0 + mu_t)
    records: list[pd.DataFrame] = []
    max_abs_balance = 0.0
    max_rel_balance = 0.0
    min_state_flux = np.inf
    for t, (year, month) in enumerate(times):
        start_total = sum(value.astype(np.longdouble) for value in state.values())
        positive = arrays["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        negative = arrays["negative_legacy_eligible_n_surplus_kg_n_month"][t]
        bypass, contact, flush, quick_share, gw_share = operator_water_partitions(arrays, t, operator)
        direct = positive * bypass
        pool_input = positive - direct
        removed_total = np.zeros(len(reach_ids), dtype=float)
        if structure == "S0":
            removed, unmet = _withdraw(state["mobile"], negative.copy())
            removed_total += removed
            state["mobile"] += pool_input
            mineralized = np.zeros(len(reach_ids), dtype=float)
        else:
            removed, left = _withdraw(state["mobile"], negative.copy())
            removed_total += removed
            removed_son, unmet = _withdraw(state["son"], left)
            removed_total += removed_son
            state["son"] += pool_input
            mineralized = state["son"] * (1.0 - rho_s)
            state["son"] -= mineralized
            state["mobile"] += mineralized
        mobile_pre_flush = state["mobile"].copy()
        source_release = mobile_pre_flush * flush
        state["mobile"] -= source_release
        quick_input = direct + source_release * quick_share
        gw_input = source_release * gw_share
        state["quick"] += quick_input
        state["gw"] += gw_input
        quick_release = np.where(
            arrays["quick_release_mm"][t] > WATER_EPS,
            (1.0 - Q_RHO) * state["quick"],
            0.0,
        )
        gw_release = np.where(
            arrays["gw_discharge_mm"][t] > WATER_EPS,
            (1.0 - rho_gw) * state["gw"],
            0.0,
        )
        state["quick"] -= quick_release
        state["gw"] -= gw_release
        end_total = sum(value.astype(np.longdouble) for value in state.values())
        balance = (
            positive.astype(np.longdouble)
            + start_total
            - quick_release.astype(np.longdouble)
            - gw_release.astype(np.longdouble)
            - end_total
            - removed_total.astype(np.longdouble)
        )
        scale = (
            np.abs(positive.astype(np.longdouble))
            + np.abs(start_total)
            + np.abs(quick_release.astype(np.longdouble))
            + np.abs(gw_release.astype(np.longdouble))
            + np.abs(end_total)
            + np.abs(removed_total.astype(np.longdouble))
        )
        rel = np.asarray(np.abs(balance) / np.maximum(scale, 1.0), dtype=float)
        max_abs_balance = max(max_abs_balance, float(np.max(np.abs(balance))))
        max_rel_balance = max(max_rel_balance, float(np.max(rel)))
        min_state_flux = min(
            min_state_flux,
            *(float(x.min()) for x in state.values()),
            float(quick_release.min()),
            float(gw_release.min()),
        )
        if capture_start_year <= year <= capture_end_year:
            records.append(pd.DataFrame({
                "reach_id": reach_ids,
                "year": year,
                "month": month,
                "son_state_end_kg_n": state["son"],
                "mobile_state_pre_flush_kg_n": mobile_pre_flush,
                "mobile_state_end_kg_n": state["mobile"],
                "quick_state_end_kg_n": state["quick"],
                "gw_state_end_kg_n": state["gw"],
                "direct_current_quick_n_kg_n": direct,
                "source_pool_input_kg_n": pool_input,
                "mineralized_n_kg_n": mineralized,
                "source_release_kg_n": source_release,
                "quick_path_n_input_kg_n": quick_input,
                "gw_path_n_input_kg_n": gw_input,
                "quick_tn_release_kg_n": quick_release,
                "gw_tn_release_kg_n": gw_release,
                "local_tn_release_kg_n": quick_release + gw_release,
                "negative_removed_kg_n": removed_total,
                "negative_unmet_kg_n": unmet,
                "contact_water_mm": contact,
                "source_flush_fraction": flush,
                "mass_balance_error_kg_n": np.asarray(balance, dtype=float),
                "mass_balance_relative_error": rel,
                "q_local_total_mm": arrays["q_local_total_mm"][t],
                "catchment_area_km2": arrays["catchment_area_km2"][t],
                "quick_release_mm": arrays["quick_release_mm"][t],
                "gw_discharge_mm": arrays["gw_discharge_mm"][t],
            }))
    frame = pd.concat(records, ignore_index=True)
    frame["operator"] = operator
    frame["model_id"] = model_id
    frame["candidate_id"] = operator + "__" + model_id
    frame["source_structure"] = structure
    frame["soil_legacy_tau_month"] = np.nan if tau_s is None else tau_s
    frame["effective_tn_delivery_mu_month"] = mu_t
    audit = {
        "operator": operator,
        "model_id": model_id,
        "candidate_id": operator + "__" + model_id,
        "max_abs_mass_balance_error_kg_n": max_abs_balance,
        "max_relative_mass_balance_error": max_rel_balance,
        "minimum_state_or_flux_kg_n": min_state_flux,
    }
    spinup_states["reach_id"] = reach_ids
    spinup_states["operator"] = operator
    spinup_states["model_id"] = model_id
    spinup_states["candidate_id"] = operator + "__" + model_id
    spinup["model_id"] = model_id
    spinup["candidate_id"] = operator + "__" + model_id
    return frame, audit, spinup, spinup_states
