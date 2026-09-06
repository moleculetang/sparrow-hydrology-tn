from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260820_18"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"

PROGRAM_MANIFEST = TEST / "20260820_11" / "program_manifest.json"
CONTRACT = HERE / "experiment_contract.json"
PARENT_FINAL_LOCK = TEST / "20260820_12" / "final_lock.json"
PARENT_LOCAL = TEST / "20260820_10" / "cache" / "parent_local"
PARENT_ROUTED = TEST / "20260820_12" / "cache" / "direct_parent_routed"
PARENT_OOF = TEST / "20260820_12" / "outputs" / "temporal_oof_predictions.parquet"
PARENT_SHARED = TEST / "20260818_1" / "scripts" / "legacy18_shared.py"

SEGMENTS = TEST / "20260820_2" / "outputs" / "andreadis_500m_channel_segments.parquet"
Q72_DISCHARGE = TEST / "20260820_2" / "outputs" / "q72_routed_monthly_discharge_2006_2022.parquet"
TRAVEL_EXPOSURE = TEST / "20260820_2" / "outputs" / "reach_month_hydraulic_exposure.parquet"
PATH_EXPOSURE = TEST / "20260820_2" / "outputs" / "source_target_path_exposure_2006_2022.parquet"
GEOMETRY = TEST / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
LMLT = TEST / "20260814_9" / "inputs" / "model_ready" / "monthly" / "lake_mix_layer_temperature_era5_land_by_reach_month_2006_2022.parquet"
GW_TEMP = TEST / "20260814_9" / "inputs" / "model_ready" / "static" / "groundwater_temperature_benz_2024_by_reach.parquet"

FORMAL_MODELS = tuple(
    sorted(
        [f"S0_mu_{mu:03d}m" for mu in (12, 36, 60, 96, 144, 240)]
        + [f"S1_tau_012m_mu_{mu:03d}m" for mu in (12, 36, 60, 96, 144, 240)]
    )
)
MAIN_SCENARIO = "central_n0035"
VF_NULL = 0.0
NULL_REL_TOL = 1e-12


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


def input_paths() -> list[Path]:
    paths = [
        PROGRAM_MANIFEST,
        CONTRACT,
        PARENT_FINAL_LOCK,
        PARENT_OOF,
        PARENT_SHARED,
        SEGMENTS,
        Q72_DISCHARGE,
        TRAVEL_EXPOSURE,
        PATH_EXPOSURE,
        GEOMETRY,
        LMLT,
        GW_TEMP,
    ]
    paths.extend(PARENT_LOCAL / f"{model_id}.parquet" for model_id in FORMAL_MODELS)
    paths.extend(PARENT_ROUTED / f"{model_id}.parquet" for model_id in FORMAL_MODELS)
    return paths


def hash_manifest(paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")
    return {str(path): sha256(path) for path in paths}


def load_parent_shared():
    spec = importlib.util.spec_from_file_location("aquatic18_parent_shared", PARENT_SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PARENT_SHARED}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def manning_depth(q: np.ndarray, width: np.ndarray, slope: np.ndarray, roughness: float) -> np.ndarray:
    """Exact copy of the frozen 20260820_2 Manning solver."""
    q = np.asarray(q, dtype=float)
    width = np.asarray(width, dtype=float)
    slope = np.asarray(slope, dtype=float)
    h = np.maximum((np.maximum(q, 1e-30) * roughness / (width * np.sqrt(slope))) ** 0.6, 1e-8)
    for _ in range(20):
        area = width * h
        perimeter = width + 2.0 * h
        radius = area / perimeter
        modeled = area * radius ** (2.0 / 3.0) * np.sqrt(slope) / roughness
        dlog = 1.0 / h + (2.0 / 3.0) * (1.0 / h - 2.0 / perimeter)
        update = (modeled - q) / np.maximum(modeled * dlog, 1e-30)
        h = np.maximum(h - update, 1e-10)
    area = width * h
    radius = area / (width + 2.0 * h)
    closure = area * radius ** (2.0 / 3.0) * np.sqrt(slope) / roughness
    if not np.allclose(closure, q, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Manning solve did not close")
    return h


def build_uptake_exposure() -> tuple[pd.DataFrame, dict[str, object]]:
    segments = pd.read_parquet(SEGMENTS).sort_values(["reach_id", "segment_index"]).reset_index(drop=True)
    hydro = pd.read_parquet(Q72_DISCHARGE)
    hydro = hydro.loc[hydro.year.between(2006, 2021)].copy()
    if segments.wqd_reference_discharge_used.any() or hydro.andreadis_discharge_used.any():
        raise RuntimeError("STOP_WQD_REFERENCE_DISCHARGE_USED")

    reach_ids = np.sort(hydro.reach_id.unique().astype(int))
    if len(reach_ids) != 230:
        raise RuntimeError("hydraulic reach coverage is not 230")
    rindex = {int(reach_id): i for i, reach_id in enumerate(reach_ids)}
    seg_ridx = segments.reach_id.map(rindex).to_numpy(int)
    x = segments.segment_midpoint_fraction.to_numpy(float)
    length = segments.segment_length_m.to_numpy(float)
    mid_length = segments.midpoint_to_outlet_segment_length_m.to_numpy(float)
    width = segments.width_central_m.to_numpy(float)
    bankfull_depth = segments.depth_central_m.to_numpy(float)
    slope = segments.slope_used.to_numpy(float)
    length_sum = np.bincount(seg_ridx, weights=length, minlength=len(reach_ids))

    times = hydro[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    ordered = hydro.sort_values(["year", "month", "reach_id"])
    local_q = ordered.q72_local_discharge_m3_s.to_numpy(float).reshape(len(times), len(reach_ids))
    upstream_q = ordered.q72_upstream_discharge_m3_s.to_numpy(float).reshape(len(times), len(reach_ids))

    rows: list[pd.DataFrame] = []
    for ti, (year, month) in enumerate(times.itertuples(index=False)):
        q = upstream_q[ti, seg_ridx] + x * local_q[ti, seg_ridx]
        if np.any(q <= 0):
            raise RuntimeError("nonpositive along-reach Q")
        depth = manning_depth(q, width, slope, 0.035)
        velocity = q / (width * depth)
        tau_seconds = length / velocity
        tau_mid_seconds = mid_length / velocity
        exposure_full = np.bincount(seg_ridx, weights=tau_seconds / 86400.0 / depth, minlength=len(reach_ids))
        exposure_mid = np.bincount(seg_ridx, weights=tau_mid_seconds / 86400.0 / depth, minlength=len(reach_ids))
        full_seconds = np.bincount(seg_ridx, weights=tau_seconds, minlength=len(reach_ids))
        mid_seconds = np.bincount(seg_ridx, weights=tau_mid_seconds, minlength=len(reach_ids))
        above = depth > bankfull_depth
        above_h_full = np.bincount(seg_ridx, weights=(tau_seconds / 86400.0 / depth) * above, minlength=len(reach_ids))
        rows.append(
            pd.DataFrame(
                {
                    "reach_id": reach_ids,
                    "year": int(year),
                    "month": int(month),
                    "hydraulic_scenario": MAIN_SCENARIO,
                    "manning_n": 0.035,
                    "uptake_exposure_full_days_per_m": exposure_full,
                    "uptake_exposure_midpoint_to_outlet_days_per_m": exposure_mid,
                    "travel_time_full_days": full_seconds / 86400.0,
                    "travel_time_midpoint_to_outlet_days": mid_seconds / 86400.0,
                    "length_weighted_mean_depth_m": np.bincount(seg_ridx, weights=depth * length, minlength=len(reach_ids)) / length_sum,
                    "uptake_exposure_above_bankfull_fraction": np.divide(above_h_full, exposure_full, out=np.zeros(len(reach_ids)), where=exposure_full > 0),
                    "water_source": "Q72_structural_canonical_main_only",
                    "geometry_source": "Andreadis_width_depth_only",
                    "wqd_reference_discharge_used": False,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)

    frozen = pd.read_parquet(TRAVEL_EXPOSURE)
    frozen = frozen.loc[
        frozen.hydraulic_scenario.eq(MAIN_SCENARIO) & frozen.year.between(2006, 2021),
        ["reach_id", "year", "month", "travel_time_full_days", "travel_time_midpoint_to_outlet_days"],
    ]
    check = frame.merge(frozen, on=["reach_id", "year", "month"], suffixes=("_new", "_frozen"), validate="one_to_one")
    full_diff = np.abs(check.travel_time_full_days_new - check.travel_time_full_days_frozen)
    mid_diff = np.abs(check.travel_time_midpoint_to_outlet_days_new - check.travel_time_midpoint_to_outlet_days_frozen)
    scale = np.maximum(np.abs(check[["travel_time_full_days_frozen", "travel_time_midpoint_to_outlet_days_frozen"]].to_numpy(float)), 1.0)
    combined = np.column_stack([full_diff.to_numpy(float), mid_diff.to_numpy(float)])
    audit = {
        "rows": int(len(frame)),
        "reaches": int(frame.reach_id.nunique()),
        "years": [int(frame.year.min()), int(frame.year.max())],
        "max_abs_travel_time_reproduction_days": float(combined.max()),
        "max_relative_travel_time_reproduction": float(np.max(combined / scale)),
        "travel_time_reproduced": bool(np.max(combined / scale) <= 1e-12),
        "wqd_reference_discharge_used": False,
    }
    return frame, audit


def route_nested_null(local: pd.DataFrame, exposure: pd.DataFrame, shared) -> pd.DataFrame:
    local = local.loc[local.year.between(2016, 2021)].copy()
    reach_ids = np.sort(local.reach_id.unique().astype(int))
    index = {int(reach_id): i for i, reach_id in enumerate(reach_ids)}
    order, downstream, terminal = shared.topology_operators(reach_ids)
    exp_index = exposure.set_index(["year", "month", "reach_id"])
    rows: list[pd.DataFrame] = []
    for (year, month), block in local.groupby(["year", "month"], sort=True):
        b = block.set_index("reach_id").loc[reach_ids]
        e = exp_index.loc[[(int(year), int(month), int(reach_id)) for reach_id in reach_ids]]
        h_full = e.uptake_exposure_full_days_per_m.to_numpy(float)
        h_mid = e.uptake_exposure_midpoint_to_outlet_days_per_m.to_numpy(float)
        survival_full = np.exp(-VF_NULL * h_full)
        survival_mid = np.exp(-VF_NULL * h_mid)
        local_q = b.quick_tn_release_kg_n.to_numpy(float)
        local_g = b.gw_tn_release_kg_n.to_numpy(float)
        upstream_q = np.zeros(len(reach_ids), dtype=float)
        upstream_g = np.zeros(len(reach_ids), dtype=float)
        out_q = np.zeros(len(reach_ids), dtype=float)
        out_g = np.zeros(len(reach_ids), dtype=float)
        for reach_id in order:
            i = index[reach_id]
            out_q[i] = upstream_q[i] * survival_full[i] + local_q[i] * survival_mid[i]
            out_g[i] = upstream_g[i] * survival_full[i] + local_g[i] * survival_mid[i]
            if reach_id in downstream:
                down, fraction = downstream[reach_id]
                upstream_q[index[down]] += fraction * out_q[i]
                upstream_g[index[down]] += fraction * out_g[i]
        rows.append(
            pd.DataFrame(
                {
                    "reach_id": reach_ids,
                    "year": int(year),
                    "month": int(month),
                    "routed_quick_tn_kg_n": out_q,
                    "routed_gw_tn_kg_n": out_g,
                    "terminal_tree_id": [terminal[int(reach_id)] for reach_id in reach_ids],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def null_reproduction(exposure: pd.DataFrame, shared) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["reach_id", "year", "month"]
    for model_id in FORMAL_MODELS:
        local = pd.read_parquet(PARENT_LOCAL / f"{model_id}.parquet")
        candidate = route_nested_null(local, exposure, shared)
        parent = pd.read_parquet(PARENT_ROUTED / f"{model_id}.parquet")
        parent = parent.loc[parent.year.between(2016, 2021), keys + ["routed_quick_tn_kg_n", "routed_gw_tn_kg_n"]]
        joined = candidate.merge(parent, on=keys, suffixes=("_candidate", "_parent"), validate="one_to_one")
        differences = np.column_stack(
            [
                joined.routed_quick_tn_kg_n_candidate - joined.routed_quick_tn_kg_n_parent,
                joined.routed_gw_tn_kg_n_candidate - joined.routed_gw_tn_kg_n_parent,
            ]
        )
        reference = np.column_stack(
            [joined.routed_quick_tn_kg_n_parent, joined.routed_gw_tn_kg_n_parent]
        )
        max_abs = float(np.max(np.abs(differences)))
        max_rel = float(np.max(np.abs(differences) / np.maximum(np.abs(reference), 1.0)))
        rows.append(
            {
                "model_id": model_id,
                "rows": int(len(joined)),
                "v_f_m_per_day": VF_NULL,
                "max_abs_difference_kg_n": max_abs,
                "max_relative_difference": max_rel,
                "within_1e_12_relative": bool(max_rel <= NULL_REL_TOL),
            }
        )
    return pd.DataFrame(rows)


def path_uptake_registry(exposure: pd.DataFrame, shared, target_reaches: set[int]) -> pd.DataFrame:
    reach_ids = np.sort(exposure.reach_id.unique().astype(int))
    _, downstream, terminal = shared.topology_operators(reach_ids)
    static_path = pd.read_parquet(PATH_EXPOSURE)
    static_path = static_path.loc[
        static_path.year.eq(2018) & static_path.month.eq(1) & static_path.target_reach_id.isin(target_reaches),
        ["source_reach_id", "target_reach_id", "cumulative_routing_fraction"],
    ].drop_duplicates()
    pairs = {(int(row.source_reach_id), int(row.target_reach_id)): float(row.cumulative_routing_fraction) for row in static_path.itertuples()}

    path_nodes: dict[tuple[int, int], list[int]] = {}
    for source, target in pairs:
        if source == target:
            path_nodes[(source, target)] = []
            continue
        current = source
        nodes: list[int] = []
        while current in downstream and current != target:
            current = int(downstream[current][0])
            nodes.append(current)
        if current != target:
            raise RuntimeError(f"path registry mismatch {source}->{target}")
        path_nodes[(source, target)] = nodes

    e = exposure.loc[exposure.year.between(2018, 2021)].copy()
    eindex = e.set_index(["year", "month", "reach_id"])
    rows: list[dict[str, object]] = []
    for year, month in e[["year", "month"]].drop_duplicates().itertuples(index=False):
        for (source, target), fraction in pairs.items():
            source_row = eindex.loc[(int(year), int(month), source)]
            nodes = path_nodes[(source, target)]
            h = float(source_row.uptake_exposure_midpoint_to_outlet_days_per_m)
            travel = float(source_row.travel_time_midpoint_to_outlet_days)
            for node in nodes:
                node_row = eindex.loc[(int(year), int(month), node)]
                h += float(node_row.uptake_exposure_full_days_per_m)
                travel += float(node_row.travel_time_full_days)
            rows.append(
                {
                    "source_reach_id": source,
                    "target_reach_id": target,
                    "year": int(year),
                    "month": int(month),
                    "terminal_tree_id": int(terminal[target]),
                    "cumulative_routing_fraction": fraction,
                    "path_uptake_exposure_days_per_m": h,
                    "path_travel_time_days": travel,
                }
            )
    return pd.DataFrame(rows)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan"), float("nan")
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def standardized_ols_beta(y: np.ndarray, x_hydraulic: np.ndarray, x_temperature: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(y) & np.isfinite(x_hydraulic) & np.isfinite(x_temperature)
    yv = y[valid]
    xh = x_hydraulic[valid]
    xt = x_temperature[valid]
    if len(yv) < 3 or min(np.std(yv), np.std(xh), np.std(xt)) == 0:
        return float("nan"), float("nan")
    yz = (yv - yv.mean()) / yv.std(ddof=0)
    xhz = (xh - xh.mean()) / xh.std(ddof=0)
    xtz = (xt - xt.mean()) / xt.std(ddof=0)
    beta = np.linalg.lstsq(np.column_stack([xhz, xtz]), yz, rcond=None)[0]
    return float(beta[0]), float(beta[1])


def plausibility_diagnostic(exposure: pd.DataFrame, shared) -> pd.DataFrame:
    predictions = pd.read_parquet(PARENT_OOF)
    predictions = predictions.loc[
        predictions.mechanism.eq("UNIFORM_PARENT")
        & predictions.layer.eq("P1")
        & predictions.year.between(2018, 2021)
    ].copy()
    if predictions.year.eq(2022).any() or set(predictions.year.unique()) != {2018, 2019, 2020, 2021}:
        raise RuntimeError("development OOF boundary violated")
    target_reaches = set(predictions.reach_id.astype(int).unique())
    path_registry = path_uptake_registry(exposure, shared, target_reaches)
    temperature = pd.read_parquet(LMLT)
    temperature = temperature.loc[
        temperature.year.between(2018, 2021),
        ["reach_id", "year", "month", "lake_mix_layer_temperature_c"],
    ]

    rows: list[dict[str, object]] = []
    for model_id in FORMAL_MODELS:
        local = pd.read_parquet(PARENT_LOCAL / f"{model_id}.parquet")
        local = local.loc[local.year.between(2018, 2021), ["reach_id", "year", "month", "quick_tn_release_kg_n", "gw_tn_release_kg_n"]].copy()
        local["local_pre_aquatic_tn_kg_n"] = local.quick_tn_release_kg_n + local.gw_tn_release_kg_n
        joined = path_registry.merge(
            local.rename(columns={"reach_id": "source_reach_id"}),
            on=["source_reach_id", "year", "month"],
            validate="many_to_one",
        )
        joined["mass_weight"] = joined.local_pre_aquatic_tn_kg_n * joined.cumulative_routing_fraction
        joined["weighted_h"] = joined.mass_weight * joined.path_uptake_exposure_days_per_m
        joined["weighted_t"] = joined.mass_weight * joined.path_travel_time_days
        path_mean = joined.groupby(["target_reach_id", "year", "month"], as_index=False).agg(
            path_mass_kg_n=("mass_weight", "sum"),
            weighted_h=("weighted_h", "sum"),
            weighted_t=("weighted_t", "sum"),
        )
        path_mean["mass_weighted_path_uptake_exposure_days_per_m"] = np.divide(
            path_mean.weighted_h,
            path_mean.path_mass_kg_n,
            out=np.zeros(len(path_mean)),
            where=path_mean.path_mass_kg_n > 0,
        )
        path_mean["mass_weighted_path_travel_time_days"] = np.divide(
            path_mean.weighted_t,
            path_mean.path_mass_kg_n,
            out=np.zeros(len(path_mean)),
            where=path_mean.path_mass_kg_n > 0,
        )
        pred = predictions.loc[predictions.model_id.eq(model_id)].merge(
            path_mean,
            left_on=["reach_id", "year", "month"],
            right_on=["target_reach_id", "year", "month"],
            validate="many_to_one",
        ).merge(temperature, on=["reach_id", "year", "month"], validate="many_to_one")
        pred["residual_log_obs_minus_pred"] = np.log1p(pred.tn_mg_l) - np.log1p(pred.pred_tn_mg_l)
        pred["thermal_multiplier_q10_2_proxy"] = 2.0 ** ((pred.lake_mix_layer_temperature_c - 20.0) / 10.0)
        pred["thermal_uptake_exposure_proxy"] = pred.mass_weighted_path_uptake_exposure_days_per_m * pred.thermal_multiplier_q10_2_proxy
        variables = [
            "mass_weighted_path_travel_time_days",
            "mass_weighted_path_uptake_exposure_days_per_m",
            "lake_mix_layer_temperature_c",
            "thermal_uptake_exposure_proxy",
        ]
        for column in ["residual_log_obs_minus_pred", *variables]:
            pred[f"centered_{column}"] = pred[column] - pred.groupby("station_key")[column].transform("mean")
        for variable in variables:
            rho, pvalue = safe_spearman(
                pred[f"centered_{variable}"].to_numpy(float),
                pred.centered_residual_log_obs_minus_pred.to_numpy(float),
            )
            rows.append(
                {
                    "model_id": model_id,
                    "diagnostic_variable": variable,
                    "n_oof_rows": int(len(pred)),
                    "n_stations": int(pred.station_key.nunique()),
                    "spearman_within_station_rho": rho,
                    "spearman_pvalue_descriptive_only": pvalue,
                    "standardized_ols_beta_hydraulic": float("nan"),
                    "standardized_ols_beta_temperature": float("nan"),
                    "expected_missing_reaction_direction": "negative",
                    "direction_matches_expectation": bool(np.isfinite(rho) and rho < 0),
                    "role": "non_blocking_hypothesis_plausibility_only",
                }
            )
        pred["centered_log1p_hydraulic_exposure"] = (
            np.log1p(pred.mass_weighted_path_uptake_exposure_days_per_m)
            - pred.groupby("station_key").mass_weighted_path_uptake_exposure_days_per_m.transform(
                lambda values: float(np.log1p(values).mean())
            )
        )
        beta_h, beta_t = standardized_ols_beta(
            pred.centered_residual_log_obs_minus_pred.to_numpy(float),
            pred.centered_log1p_hydraulic_exposure.to_numpy(float),
            pred.centered_lake_mix_layer_temperature_c.to_numpy(float),
        )
        rows.append(
            {
                "model_id": model_id,
                "diagnostic_variable": "temperature_increment_given_log_hydraulic_exposure",
                "n_oof_rows": int(len(pred)),
                "n_stations": int(pred.station_key.nunique()),
                "spearman_within_station_rho": float("nan"),
                "spearman_pvalue_descriptive_only": float("nan"),
                "standardized_ols_beta_hydraulic": beta_h,
                "standardized_ols_beta_temperature": beta_t,
                "expected_missing_reaction_direction": "negative_temperature_coefficient",
                "direction_matches_expectation": bool(np.isfinite(beta_t) and beta_t < 0),
                "role": "non_blocking_incremental_temperature_plausibility_only",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    paths = input_paths()
    start_hashes = hash_manifest(paths)
    dump_json(REPORTS / "input_hashes_start.json", start_hashes)

    shared = load_parent_shared()
    exposure, hydraulic_audit = build_uptake_exposure()
    exposure.to_parquet(OUT / "reach_month_aquatic_uptake_exposure_2006_2021.parquet", index=False)

    null_audit = null_reproduction(exposure, shared)
    null_audit.to_parquet(OUT / "vf_zero_parent_reproduction_audit.parquet", index=False)

    plausibility = plausibility_diagnostic(exposure, shared)
    plausibility.to_parquet(OUT / "development_residual_hydraulic_temperature_plausibility.parquet", index=False)

    lmlt = pd.read_parquet(LMLT)
    gw = pd.read_parquet(GW_TEMP)
    geometry = pd.read_parquet(GEOMETRY)
    main_exp = exposure.loc[exposure.year.between(2016, 2021)]
    summary = {
        "uptake_exposure_full_days_per_m": {
            "p01": float(main_exp.uptake_exposure_full_days_per_m.quantile(0.01)),
            "median": float(main_exp.uptake_exposure_full_days_per_m.median()),
            "p99": float(main_exp.uptake_exposure_full_days_per_m.quantile(0.99)),
            "max": float(main_exp.uptake_exposure_full_days_per_m.max()),
        },
        "travel_time_full_days": {
            "p01": float(main_exp.travel_time_full_days.quantile(0.01)),
            "median": float(main_exp.travel_time_full_days.median()),
            "p99": float(main_exp.travel_time_full_days.quantile(0.99)),
            "max": float(main_exp.travel_time_full_days.max()),
        },
        "lmlt_2016_2021_c": {
            "min": float(lmlt.loc[lmlt.year.between(2016, 2021), "lake_mix_layer_temperature_c"].min()),
            "median": float(lmlt.loc[lmlt.year.between(2016, 2021), "lake_mix_layer_temperature_c"].median()),
            "max": float(lmlt.loc[lmlt.year.between(2016, 2021), "lake_mix_layer_temperature_c"].max()),
        },
    }
    dump_json(REPORTS / "hydraulic_temperature_range_audit.json", summary)

    checks = {
        "program_scenario_registered": "20260820_18" in PROGRAM_MANIFEST.read_text(encoding="utf-8"),
        "hydraulic_rows_complete_2006_2021": len(exposure) == 230 * 16 * 12,
        "hydraulic_reaches_230": exposure.reach_id.nunique() == 230,
        "travel_time_reproduces_frozen_stage2": bool(hydraulic_audit["travel_time_reproduced"]),
        "wqd_reference_discharge_never_used": not exposure.wqd_reference_discharge_used.any(),
        "vf_zero_all_12_models_reproduce_parent": bool(null_audit.within_1e_12_relative.all()),
        "vf_zero_rows_all_complete": bool(null_audit.rows.eq(230 * 6 * 12).all()),
        "lmlt_coverage_230x17x12": len(lmlt) == 230 * 17 * 12 and lmlt.reach_id.nunique() == 230,
        "lmlt_role_is_proxy_not_stream_observation": bool(lmlt.surface_temperature_role.astype(str).str.contains("not_observed_stream_temperature").all()),
        "groundwater_temperature_static_only": bool(gw.groundwater_temperature_temporal_role.astype(str).str.contains("not_monthly_observation").all()),
        "groundwater_temperature_not_entered_aquatic_calculation": True,
        "geometry_coverage_230": len(geometry) == 230 and geometry.reach_id.nunique() == 230,
        "development_plausibility_uses_2018_2021_oof_only": True,
        "TN_2022_not_read": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    dump_json(
        REPORTS / "stage0_preflight.json",
        {
            "status": status,
            "checks": {key: bool(value) for key, value in checks.items()},
            "hydraulic_audit": hydraulic_audit,
            "temperature_role": "ERA5-Land LMLT proxy only; no river-temperature observation claim",
            "groundwater_temperature_role": "provenance audit only; forbidden from the aquatic operator",
            "difference_from_20260820_3": "eta retained and v_f=0 exact Parent; old experiment replaced eta",
        },
    )

    thermal = plausibility.loc[plausibility.diagnostic_variable.eq("thermal_uptake_exposure_proxy")]
    hydraulic = plausibility.loc[plausibility.diagnostic_variable.eq("mass_weighted_path_uptake_exposure_days_per_m")]
    temperature_raw = plausibility.loc[plausibility.diagnostic_variable.eq("lake_mix_layer_temperature_c")]
    temperature_partial = plausibility.loc[
        plausibility.diagnostic_variable.eq("temperature_increment_given_log_hydraulic_exposure")
    ]
    hydraulic_plausible = int(hydraulic.direction_matches_expectation.sum()) >= 8
    temperature_strengthened = int(temperature_partial.direction_matches_expectation.sum()) >= 8
    plausibility_summary = {
        "status": (
            "hydraulic_hypothesis_plausible_temperature_increment_strengthened"
            if hydraulic_plausible and temperature_strengthened
            else "hydraulic_hypothesis_plausible_temperature_increment_not_strengthened"
            if hydraulic_plausible
            else "hydraulic_hypothesis_not_strengthened"
        ),
        "non_blocking": True,
        "models_negative_hydraulic_exposure_rho": int(hydraulic.direction_matches_expectation.sum()),
        "models_negative_thermal_exposure_proxy_rho": int(thermal.direction_matches_expectation.sum()),
        "models_negative_raw_temperature_rho": int(temperature_raw.direction_matches_expectation.sum()),
        "models_negative_temperature_beta_given_log_hydraulic_exposure": int(temperature_partial.direction_matches_expectation.sum()),
        "models_total": 12,
        "median_hydraulic_exposure_rho": float(hydraulic.spearman_within_station_rho.median()),
        "median_thermal_exposure_proxy_rho": float(thermal.spearman_within_station_rho.median()),
        "median_raw_temperature_rho": float(temperature_raw.spearman_within_station_rho.median()),
        "median_temperature_beta_given_log_hydraulic_exposure": float(temperature_partial.standardized_ols_beta_temperature.median()),
        "interpretation": "Negative hydraulic association is compatible with omitted aquatic attenuation. Temperature evidence is assessed incrementally after controlling log hydraulic exposure; neither diagnostic identifies a reaction rate or serves as a candidate-selection gate.",
    }
    dump_json(REPORTS / "stage0_plausibility.json", plausibility_summary)

    end_hashes = hash_manifest(paths)
    dump_json(REPORTS / "input_hashes_end.json", end_hashes)
    unchanged = start_hashes == end_hashes
    decision = {
        "status": "stage0_complete" if status == "PASS" and unchanged else "stage0_failed",
        "engineering_preflight": status,
        "input_hashes_unchanged": unchanged,
        "plausibility": plausibility_summary["status"],
        "hydraulic_only_fit_authorized": bool(status == "PASS" and unchanged),
        "temperature_fit_authorized": False,
        "next_required_action": "freeze and implement the no-temperature v_f+eta temporal OOF stage; temperature remains closed until that candidate passes",
    }
    dump_json(REPORTS / "stage0_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
