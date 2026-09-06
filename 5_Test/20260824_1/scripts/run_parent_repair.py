from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260824_1"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
OLD = TEST / "20260820_19"
OLD_OUT = OLD / "outputs"
OLD_SHARED = OLD / "scripts" / "hierarchical19_shared.py"
OBS = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLDS = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
HYDROLOGY = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
Q72_INTERFACE = TEST / "20260814_6" / "outputs" / "structural_canonical_main_interface_2006_2022.parquet"
Q72_LOCK = TEST / "20260814_6" / "structural_interface_lock.json"
Q72_ROUTED = TEST / "20260820_2" / "outputs" / "q72_routed_monthly_discharge_2006_2022.parquet"
GEOMETRY = TEST / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
EXPOSURE = TEST / "20260820_18" / "outputs" / "reach_month_aquatic_uptake_exposure_2006_2021.parquet"
PARENT_ROUTED = TEST / "20260820_12" / "cache" / "direct_parent_routed"
LEGACY_SHARED = TEST / "20260818_1" / "scripts" / "legacy18_shared.py"

FORMAL_MUS = (12, 36, 60, 96, 144, 240)
FORMAL_MODELS = tuple(sorted(
    [f"S0_mu_{mu:03d}m" for mu in FORMAL_MUS]
    + [f"S1_tau_012m_mu_{mu:03d}m" for mu in FORMAL_MUS]
))
BOOTSTRAP_REPLICATES = 10_000
NONINFERIOR_MARGIN = 0.005
TIE_TOL = 1e-8
SEED = 2026082401


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def input_lock() -> dict[str, object]:
    paths = [
        HERE / "program_charter.json",
        HERE / "program_manifest.json",
        HERE / "experiment_contract.json",
        Path(__file__),
        OLD / "final_lock.json",
        OLD / "experiment_contract.json",
        OLD_SHARED,
        OLD_OUT / "temporal_oof_predictions.parquet",
        OLD_OUT / "temporal_h1_readout_predictions.parquet",
        OLD_OUT / "temporal_fold_parameters.parquet",
        OLD_OUT / "nested_spatial_predictions.parquet",
        OLD_OUT / "nested_spatial_parameters.parquet",
        OLD_OUT / "nested_spatial_readout_parameters.parquet",
        OLD_OUT / "observation_domain_registry.parquet",
        OLD_OUT / "reach_spatial_hydraulic_covariates.parquet",
        OBS,
        FOLDS,
        HYDROLOGY,
        Q72_INTERFACE,
        Q72_LOCK,
        Q72_ROUTED,
        GEOMETRY,
        EXPOSURE,
        LEGACY_SHARED,
    ]
    paths.extend(sorted(PARENT_ROUTED.glob("*.parquet")))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"STOP_MISSING_INPUTS: {missing}")
    hashes = {str(path): sha256(path) for path in paths}
    return {
        "lock_id": "20260824_1_parent_repair_inputs",
        "TN_2022_values_read": False,
        "files": hashes,
        "aggregate_sha256": hashlib.sha256(
            "\n".join(f"{key}|{value}" for key, value in sorted(hashes.items())).encode("utf-8")
        ).hexdigest(),
    }


def development_observations() -> pd.DataFrame:
    obs = pd.read_parquet(OBS, filters=[("year", "<=", 2021)])
    if obs.empty or obs.year.min() != 2016 or obs.year.max() != 2021 or obs.year.eq(2022).any():
        raise RuntimeError("STOP_DEVELOPMENT_TN_BOUNDARY")
    domain = pd.read_parquet(
        OLD_OUT / "observation_domain_registry.parquet",
        columns=["station_key", "primary_river_domain"],
    )
    obs = obs.merge(domain, on="station_key", validate="many_to_one")
    obs = obs.loc[obs.primary_river_domain].drop(columns="primary_river_domain")
    return obs.reset_index(drop=True)


def fold_registry() -> pd.DataFrame:
    folds = pd.read_parquet(FOLDS)[
        ["fold_id", "train_start_year", "train_end_year", "evaluation_year"]
    ].drop_duplicates().sort_values("evaluation_year")
    if folds.evaluation_year.tolist() != [2018, 2019, 2020, 2021]:
        raise RuntimeError("STOP_OOF_FOLD_CONTRACT")
    return folds


def macro_rmse(frame: pd.DataFrame, block: str) -> float:
    values = []
    for _, group in frame.groupby(block):
        observed = np.log1p(group.tn_mg_l.to_numpy(float))
        predicted = np.log1p(group.pred_tn_mg_l.to_numpy(float))
        values.append(float(np.sqrt(np.mean((predicted - observed) ** 2))))
    return float(np.mean(values))


def block_deltas(reference: pd.DataFrame, candidate: pd.DataFrame, block: str) -> np.ndarray:
    keys = ["station_key", "year", "month", "fold_id"]
    left = reference[list(dict.fromkeys(keys + ["tn_mg_l", "pred_tn_mg_l", block]))]
    joined = left.merge(
        candidate[keys + ["pred_tn_mg_l"]],
        on=keys,
        suffixes=("_ref", "_cand"),
        validate="one_to_one",
    )
    values = []
    for _, group in joined.groupby(block):
        observed = np.log1p(group.tn_mg_l.to_numpy(float))
        reference_rmse = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_ref) - observed) ** 2))
        candidate_rmse = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_cand) - observed) ** 2))
        values.append(float(candidate_rmse - reference_rmse))
    return np.asarray(values, dtype=float)


def paired_summary(reference: pd.DataFrame, candidate: pd.DataFrame, block: str, seed: int) -> tuple[dict[str, object], np.ndarray]:
    values = block_deltas(reference, candidate, block)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    distribution = values[indices].mean(axis=1)
    lo, hi = np.quantile(distribution, [0.025, 0.975])
    return {
        "blocks": int(len(values)),
        "point_delta": float(values.mean()),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "noninferior": bool(hi < NONINFERIOR_MARGIN),
        "predictively_improved": bool(hi < 0),
    }, distribution


def exact_sign_flip(values: np.ndarray, margin: float = 0.0) -> dict[str, object]:
    n = len(values)
    if n > 20:
        return {"available": False, "reason": "more_than_20_blocks"}
    signs = np.array([
        [1.0 if (mask >> j) & 1 else -1.0 for j in range(n)]
        for mask in range(2 ** n)
    ])
    centered = values - margin
    observed = float(centered.mean())
    randomized = (signs * centered[None, :]).mean(axis=1)
    p_lower = float((np.sum(randomized <= observed) + 1) / (len(randomized) + 1))
    return {
        "available": True,
        "permutations": int(len(randomized)),
        "observed_mean_minus_margin": observed,
        "one_sided_p_lower": p_lower,
        "supports_lower_than_margin_at_0_05": bool(p_lower < 0.05),
    }


def wild_cluster_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    centered = values - values.mean()
    signs = rng.choice(np.array([-1.0, 1.0]), size=(BOOTSTRAP_REPLICATES, len(values)))
    distribution = values.mean() + (signs * centered[None, :]).mean(axis=1)
    lo, hi = np.quantile(distribution, [0.025, 0.975])
    return float(lo), float(hi)


def attach_station_equal_baseline(
    predictions: pd.DataFrame,
    selections: pd.DataFrame,
    obs: pd.DataFrame,
    folds: pd.DataFrame,
    reach_to_tree: dict[int, int],
) -> pd.DataFrame:
    obs = obs.copy()
    obs["terminal_tree_id"] = obs.reach_id.astype(int).map(reach_to_tree).astype(int)
    baseline_rows = []
    for row in selections.itertuples(index=False):
        fold = folds.loc[folds.fold_id.astype(str).eq(str(row.fold_id))].iloc[0]
        training = obs.loc[obs.year.between(int(fold.train_start_year), int(fold.train_end_year))].copy()
        if row.evaluation == "LOSO":
            training = training.loc[~training.station_key.astype(str).eq(str(row.holdout_id))]
        else:
            training = training.loc[~training.terminal_tree_id.eq(int(row.holdout_id))]
        station_mean = training.assign(y=np.log1p(training.tn_mg_l)).groupby("station_key").y.mean()
        baseline_rows.append({
            "model_id": row.model_id,
            "fold_id": str(row.fold_id),
            "evaluation": row.evaluation,
            "holdout_id": str(row.holdout_id),
            "baseline_log_station_equal": float(station_mean.mean()),
        })
    baseline = pd.DataFrame(baseline_rows).drop_duplicates()
    return predictions.merge(
        baseline,
        on=["model_id", "fold_id", "evaluation", "holdout_id"],
        validate="many_to_one",
    )


def spatial_skill_summary(frame: pd.DataFrame, block: str, seed: int) -> dict[str, object]:
    work = frame.copy()
    work["model_sq"] = (np.log1p(work.pred_tn_mg_l) - np.log1p(work.tn_mg_l)) ** 2
    work["baseline_sq"] = (work.baseline_log_station_equal - np.log1p(work.tn_mg_l)) ** 2
    station = work.groupby([block, "station_key"], as_index=False)[["model_sq", "baseline_sq"]].mean()
    groups = {key: value[["model_sq", "baseline_sq"]].to_numpy(float) for key, value in station.groupby(block)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    values = np.zeros(BOOTSTRAP_REPLICATES, dtype=float)
    for i in range(BOOTSTRAP_REPLICATES):
        chosen = rng.integers(0, len(keys), len(keys))
        sample = np.concatenate([groups[keys[j]] for j in chosen], axis=0)
        values[i] = 1.0 - sample[:, 0].mean() / sample[:, 1].mean()
    point = 1.0 - station.model_sq.mean() / station.baseline_sq.mean()
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "skill_log": float(point),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "supported": bool(lo > 0),
        "blocks": int(len(keys)),
    }


def select_h0_h1(parameters: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    subset = parameters.loc[parameters.structure.isin(["H0_PARENT", "H1_GLOBAL"])].copy()
    if not subset.outer_success.all():
        failed = subset.loc[~subset.outer_success, group_columns + ["structure"]].to_dict("records")
        raise RuntimeError(f"STOP_SELECTED_OPTIMIZER_FAILURE: {failed[:10]}")
    rows = []
    for keys, group in subset.groupby(group_columns, sort=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        h0 = group.loc[group.structure.eq("H0_PARENT")].iloc[0]
        h1 = group.loc[group.structure.eq("H1_GLOBAL")].iloc[0]
        selected = "H1_GLOBAL" if h1.training_objective < h0.training_objective - TIE_TOL else "H0_PARENT"
        rows.append({
            **dict(zip(group_columns, key_tuple)),
            "selected_structure": selected,
            "h0_training_objective": float(h0.training_objective),
            "h1_training_objective": float(h1.training_objective),
            "training_delta_h1_minus_h0": float(h1.training_objective - h0.training_objective),
            "h1_v0_m_per_day": float(h1.v0_m_per_day),
            "h1_eta_quick": float(h1.eta_quick),
            "h1_eta_gw": float(h1.eta_gw),
            "h1_eta_boundary": bool(h1.eta_boundary),
            "h1_vf_boundary": bool(h1.vf_boundary),
        })
    return pd.DataFrame(rows)


def choose_nested_predictions(predictions: pd.DataFrame, selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model_id", "fold_id", "evaluation", "holdout_id"]
    work = predictions.loc[predictions.layer.isin(["P1", "P2"])].copy()
    work["holdout_id"] = work.holdout_id.astype(str)
    selections = selections.copy()
    selections["fold_id"] = selections.fold_id.astype(str)
    selections["holdout_id"] = selections.holdout_id.astype(str)
    work = work.merge(selections[keys + ["selected_structure"]], on=keys, validate="many_to_one")
    parent = work.loc[work.mechanism.eq("H0_PARENT")].copy()
    h1 = work.loc[work.mechanism.eq("NESTED_SELECTED")].copy()
    h1["mechanism"] = "H1_GLOBAL"
    selected_parts = []
    for structure, source in (("H0_PARENT", parent), ("H1_GLOBAL", h1)):
        picked = source.merge(
            selections.loc[selections.selected_structure.eq(structure), keys],
            on=keys,
            validate="many_to_one",
        )
        selected_parts.append(picked)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["mechanism"] = "OUTER_TRAINING_SELECTED"
    return parent, selected


def temporal_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    p1 = pd.read_parquet(OLD_OUT / "temporal_oof_predictions.parquet")
    p1 = p1.loc[p1.layer.eq("P1") & p1.mechanism.isin(["H0_PARENT", "H1_GLOBAL"])].copy()
    p2 = pd.read_parquet(OLD_OUT / "temporal_h1_readout_predictions.parquet")
    p2 = p2.loc[p2.layer.eq("P2") & p2.mechanism.isin(["H0_PARENT", "H1_GLOBAL"])].copy()
    predictions = pd.concat([p1, p2], ignore_index=True)
    if set(predictions.year.unique()) != {2018, 2019, 2020, 2021}:
        raise RuntimeError("STOP_TEMPORAL_OOF_BOUNDARY")
    rows = []
    distributions = []
    seed = SEED
    for model_id in FORMAL_MODELS:
        for layer in ("P1", "P2"):
            subset = predictions.loc[predictions.model_id.eq(model_id) & predictions.layer.eq(layer)]
            reference = subset.loc[subset.mechanism.eq("H0_PARENT")]
            candidate = subset.loc[subset.mechanism.eq("H1_GLOBAL")]
            for label, block in (("station", "station_key"), ("tree", "terminal_tree_id")):
                seed += 1
                summary, dist = paired_summary(reference, candidate, block, seed)
                rows.append({"model_id": model_id, "scope": "temporal", "layer": layer, "block": label, **summary})
                distributions.append(pd.DataFrame({
                    "model_id": model_id,
                    "scope": "temporal",
                    "layer": layer,
                    "block": label,
                    "replicate": np.arange(len(dist), dtype=int),
                    "delta_rmse_log1p": dist,
                }))
    return pd.DataFrame(rows), pd.concat(distributions, ignore_index=True)


def nested_evidence(obs: pd.DataFrame, folds: pd.DataFrame, h19: ModuleType) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameters = pd.read_parquet(OLD_OUT / "nested_spatial_parameters.parquet")
    parameters["holdout_id"] = parameters.holdout_id.astype(str)
    selections = select_h0_h1(parameters, ["model_id", "fold_id", "evaluation", "holdout_id"])
    predictions = pd.read_parquet(OLD_OUT / "nested_spatial_predictions.parquet")
    predictions["holdout_id"] = predictions.holdout_id.astype(str)
    if set(predictions.year.unique()) != {2018, 2019, 2020, 2021}:
        raise RuntimeError("STOP_NESTED_OOF_BOUNDARY")
    parent, selected = choose_nested_predictions(predictions, selections)
    router = h19.build_router(FORMAL_MODELS[0], h19.parent_shared())
    reach_to_tree = dict(zip(router.reach_ids.astype(int), router.terminal_by_reach.astype(int)))
    selected = attach_station_equal_baseline(selected, selections, obs, folds, reach_to_tree)
    rows = []
    skill_rows = []
    tree_rows = []
    distributions = []
    seed = SEED + 10_000
    for model_id in FORMAL_MODELS:
        for evaluation, block in (("LOSO", "station_key"), ("LOTO", "terminal_tree_id")):
            for layer in ("P1", "P2"):
                ref = parent.loc[
                    parent.model_id.eq(model_id) & parent.evaluation.eq(evaluation) & parent.layer.eq(layer)
                ]
                cand = selected.loc[
                    selected.model_id.eq(model_id) & selected.evaluation.eq(evaluation) & selected.layer.eq(layer)
                ]
                seed += 1
                summary, dist = paired_summary(ref, cand, block, seed)
                rows.append({"model_id": model_id, "scope": evaluation, "layer": layer, "block": evaluation, **summary})
                distributions.append(pd.DataFrame({
                    "model_id": model_id,
                    "scope": evaluation,
                    "layer": layer,
                    "block": evaluation,
                    "replicate": np.arange(len(dist), dtype=int),
                    "delta_rmse_log1p": dist,
                }))
                if layer == "P1":
                    skill_rows.append({
                        "model_id": model_id,
                        "evaluation": evaluation,
                        **spatial_skill_summary(cand, block, seed + 100_000),
                    })
                    if evaluation == "LOTO":
                        values = block_deltas(ref, cand, block)
                        wild_lo, wild_hi = wild_cluster_interval(values, seed + 200_000)
                        leave_one = np.array([np.delete(values, i).mean() for i in range(len(values))])
                        tree_rows.append({
                            "model_id": model_id,
                            "tree_count": int(len(values)),
                            "ordinary_noninferior": bool(summary["noninferior"]),
                            "ordinary_improved": bool(summary["predictively_improved"]),
                            "wild_ci_lower": wild_lo,
                            "wild_ci_upper": wild_hi,
                            "wild_noninferior": bool(wild_hi < NONINFERIOR_MARGIN),
                            "wild_improved": bool(wild_hi < 0),
                            "leave_one_tree_min_delta": float(leave_one.min()),
                            "leave_one_tree_max_delta": float(leave_one.max()),
                            "leave_one_tree_all_nonworse": bool((leave_one <= 0).all()),
                            **{f"exact_improvement_{key}": value for key, value in exact_sign_flip(values, 0.0).items()},
                            **{f"exact_noninferiority_{key}": value for key, value in exact_sign_flip(values, NONINFERIOR_MARGIN).items()},
                        })
    return (
        selections,
        pd.concat([parent.assign(comparison_role="H0_PARENT"), selected.assign(comparison_role="OUTER_TRAINING_SELECTED")], ignore_index=True),
        pd.DataFrame(rows),
        pd.DataFrame(skill_rows),
        pd.DataFrame(tree_rows),
        pd.concat(distributions, ignore_index=True),
    )


def route_with_closure(router: object, vf: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    sf = np.exp(-router.h_full * vf[None, :])
    sm = np.exp(-router.h_mid * vf[None, :])
    max_error = 0.0
    outputs = []
    for local in (router.local_q, router.local_g):
        upstream = np.zeros_like(local)
        out = np.zeros_like(local)
        removed = np.zeros_like(local)
        external = np.zeros_like(local)
        for i in router.order_index:
            incoming = upstream[:, i] + local[:, i]
            out[:, i] = upstream[:, i] * sf[:, i] + local[:, i] * sm[:, i]
            removed[:, i] = incoming - out[:, i]
            if i in router.downstream_index:
                down, fraction = router.downstream_index[i]
                upstream[:, down] += fraction * out[:, i]
                external[:, i] = (1.0 - fraction) * out[:, i]
            else:
                external[:, i] = out[:, i]
        balance = local.sum(axis=1) - removed.sum(axis=1) - external.sum(axis=1)
        scale = np.maximum(np.abs(local).sum(axis=1), 1.0)
        max_error = max(max_error, float(np.max(np.abs(balance) / scale)))
        outputs.append(out)
    return outputs[0], outputs[1], max_error


def hydrology_and_mass_audits(h19: ModuleType) -> tuple[dict[str, object], pd.DataFrame]:
    interface = pd.read_parquet(Q72_INTERFACE)
    quick_rhos = sorted(interface.quick_rho.dropna().unique().astype(float).tolist())
    hydro = pd.read_parquet(HYDROLOGY, filters=[("year", ">=", 2006), ("year", "<=", 2021)])
    qclosure = np.abs(hydro.q_local_total_mm - hydro.quick_release_mm - hydro.gw_discharge_mm)
    exposure = pd.read_parquet(EXPOSURE)
    routed = pd.read_parquet(Q72_ROUTED, filters=[("year", "<=", 2021)])
    geometry = pd.read_parquet(GEOMETRY)
    old_legacy = load_module(LEGACY_SHARED, "legacy18_for_qrho_audit")
    exposure_ratio = np.divide(
        exposure.travel_time_full_days.to_numpy(float),
        exposure.length_weighted_mean_depth_m.to_numpy(float),
        out=np.zeros(len(exposure), dtype=float),
        where=exposure.length_weighted_mean_depth_m.to_numpy(float) > 0,
    )
    exposure_scale = np.maximum(np.abs(exposure.uptake_exposure_full_days_per_m.to_numpy(float)), 1e-12)
    depth_identity_rel = float(np.max(np.abs(exposure_ratio - exposure.uptake_exposure_full_days_per_m) / exposure_scale))
    rows = []
    for model_id in FORMAL_MODELS:
        router = h19.build_router(model_id, h19.parent_shared())
        zeros = np.zeros(len(router.reach_ids), dtype=float)
        q0, g0, closure0 = route_with_closure(router, zeros)
        parent = pd.read_parquet(PARENT_ROUTED / f"{model_id}.parquet")
        parent = parent.loc[parent.year.between(2016, 2021)].sort_values(["year", "month", "reach_id"])
        qref = parent.routed_quick_tn_kg_n.to_numpy(float).reshape(q0.shape)
        gref = parent.routed_gw_tn_kg_n.to_numpy(float).reshape(g0.shape)
        scale = max(float(np.max(np.abs(qref))), float(np.max(np.abs(gref))), 1.0)
        reproduction = max(float(np.max(np.abs(q0 - qref))), float(np.max(np.abs(g0 - gref)))) / scale
        temporal_parameters = pd.read_parquet(OLD_OUT / "temporal_fold_parameters.parquet")
        h1 = temporal_parameters.loc[
            temporal_parameters.model_id.eq(model_id) & temporal_parameters.structure.eq("H1_GLOBAL")
        ]
        vf = np.full(len(router.reach_ids), float(h1.v0_m_per_day.median()))
        _, _, closure1 = route_with_closure(router, vf)
        rows.append({
            "model_id": model_id,
            "h0_parent_reproduction_relative_error": reproduction,
            "h0_mass_balance_relative_error": closure0,
            "h1_mass_balance_relative_error": closure1,
            "h0_reproduction_pass": bool(reproduction <= 1e-12),
            "mass_balance_pass": bool(max(closure0, closure1) <= 1e-10),
        })
    audit = {
        "status": "PASS",
        "formal_water_flux_provenance": "Q72 structural_canonical main only",
        "hydrology_sources": hydro.hydrology_source.value_counts().to_dict(),
        "q_local_quick_plus_gw_max_abs_error_mm": float(qclosure.max()),
        "q72_quick_rho_values": quick_rhos,
        "tn_legacy_code_q_rho": float(old_legacy.Q_RHO),
        "q_rho_matches_locked_interface": bool(quick_rhos == [float(old_legacy.Q_RHO)]),
        "andreadis_reference_discharge_used_in_exposure": bool(exposure.wqd_reference_discharge_used.any()),
        "andreadis_reference_discharge_used_in_routed_q72": bool(routed.andreadis_discharge_used.any()),
        "geometry_missing_reaches": int(230 - geometry.reach_id.nunique()),
        "formal_h1_equation": "H = travel_time / depth = L * W / (86400 * Q)",
        "travel_time_over_depth_identity_max_relative_error": depth_identity_rel,
        "formal_h1_semantics": "areal/benthic hydraulic exposure, not residence time",
        "depth_slope_manning_formal_role": "none; diagnostic only",
        "temperature_used": False,
        "reservoir_specific_storage_used": False,
        "TN_2022_values_read": False,
    }
    if (
        set(audit["hydrology_sources"]) != {"Q72_actual_structural_canonical_main"}
        or not audit["q_rho_matches_locked_interface"]
        or audit["andreadis_reference_discharge_used_in_exposure"]
        or audit["andreadis_reference_discharge_used_in_routed_q72"]
        or audit["geometry_missing_reaches"] != 0
    ):
        audit["status"] = "FAIL"
    return audit, pd.DataFrame(rows)


def count_gate(metrics: pd.DataFrame, scope: str, layer: str, column: str) -> int:
    return int(metrics.loc[metrics.scope.eq(scope) & metrics.layer.eq(layer), column].sum())


def decision(
    metrics: pd.DataFrame,
    skills: pd.DataFrame,
    tree: pd.DataFrame,
    temporal_parameters: pd.DataFrame,
) -> dict[str, object]:
    temporal_p1 = {
        f"{block}_{metric}": int(metrics.loc[
            metrics.scope.eq("temporal") & metrics.layer.eq("P1") & metrics.block.eq(block), metric
        ].sum())
        for block in ("station", "tree") for metric in ("noninferior", "predictively_improved")
    }
    temporal_p2 = {
        f"{block}_{metric}": int(metrics.loc[
            metrics.scope.eq("temporal") & metrics.layer.eq("P2") & metrics.block.eq(block), metric
        ].sum())
        for block in ("station", "tree") for metric in ("noninferior", "predictively_improved")
    }
    nested = {
        f"{scope}_{metric}": int(metrics.loc[
            metrics.scope.eq(scope) & metrics.layer.eq("P1"), metric
        ].sum())
        for scope in ("LOSO", "LOTO") for metric in ("noninferior", "predictively_improved")
    }
    absolute = {
        f"{scope}_supported": int(skills.loc[skills.evaluation.eq(scope), "supported"].sum())
        for scope in ("LOSO", "LOTO")
    }
    h1 = temporal_parameters.loc[temporal_parameters.structure.eq("H1_GLOBAL")]
    boundary_models = int((h1.groupby("model_id").vf_boundary.sum() >= 2).sum())
    eta_boundary_models = int((h1.groupby("model_id").eta_boundary.sum() >= 2).sum())
    temporal_supported = bool(
        temporal_p1["station_noninferior"] >= 10
        and temporal_p1["tree_noninferior"] >= 10
        and max(temporal_p1["station_predictively_improved"], temporal_p1["tree_predictively_improved"]) >= 8
        and boundary_models <= 2
        and eta_boundary_models <= 2
    )
    loso_supported = bool(
        nested["LOSO_noninferior"] >= 10
        and nested["LOSO_predictively_improved"] >= 8
        and absolute["LOSO_supported"] >= 10
    )
    ordinary_loto = int(tree.ordinary_noninferior.sum())
    exact_loto = int(tree.exact_noninferiority_supports_lower_than_margin_at_0_05.sum())
    wild_loto = int(tree.wild_noninferior.sum())
    small_tree_sensitive = bool(len({ordinary_loto, exact_loto, wild_loto}) > 1)
    loto_supported = bool(
        nested["LOTO_noninferior"] >= 10
        and nested["LOTO_predictively_improved"] >= 8
        and absolute["LOTO_supported"] >= 10
        and exact_loto >= 10
        and wild_loto >= 10
        and not small_tree_sensitive
    )
    p2_supported = bool(
        temporal_p2["station_noninferior"] >= 10
        and temporal_p2["tree_noninferior"] >= 10
        and max(temporal_p2["station_predictively_improved"], temporal_p2["tree_predictively_improved"]) >= 8
    )
    selected = "H1_GLOBAL" if temporal_supported else "H0_PARENT"
    return {
        "status": "PASS",
        "selected_full_development_process_parent": selected,
        "temporal_process_status": "supported" if temporal_supported else "not_supported",
        "within_observed_tree_LOSO_status": "supported" if loso_supported else "not_supported",
        "new_tree_LOTO_status": (
            "small_tree_count_inference_sensitive" if small_tree_sensitive
            else "supported" if loto_supported else "not_supported"
        ),
        "monitored_station_P2_status": "supported" if p2_supported else "not_supported",
        "temporal_P1_counts": temporal_p1,
        "temporal_P2_counts": temporal_p2,
        "nested_P1_counts": nested,
        "absolute_spatial_skill_supported_counts": absolute,
        "small_tree_inference": {
            "ordinary_noninferior_models": ordinary_loto,
            "exact_sign_flip_noninferior_models": exact_loto,
            "wild_cluster_noninferior_models": wild_loto,
            "sensitive": small_tree_sensitive,
        },
        "boundary_confounded_models": boundary_models,
        "eta_boundary_confounded_models": eta_boundary_models,
        "claim_boundary": (
            "H1 can be used as the temporal parent but new-tree spatial transfer remains unresolved."
            if temporal_supported and not loto_supported
            else "Domain-specific claims follow the four registered status fields."
        ),
        "tree_163_role": "diagnostic_only_open_lake_center",
        "stale_mcmc_role": "withdrawn_from_authoritative_evidence",
        "temperature_used": False,
        "TN_2022_values_read": False,
    }


def write_report(result: dict[str, object], hydro: dict[str, object], mass: pd.DataFrame) -> None:
    title = "# 20260824 Q72–TN父主线修复与无偏重新裁决"
    lines = [
        title,
        "",
        "## 技术摘要",
        "",
        f"修复后的全开发过程Parent为`{result['selected_full_development_process_parent']}`。本轮不再用一个笼统的“空间可迁移”状态，而是分别锁定时间过程、同一已观测tree内LOSO、新terminal tree LOTO和监测站P2。",
        "",
        f"- 时间过程：`{result['temporal_process_status']}`；",
        f"- 同tree LOSO：`{result['within_observed_tree_LOSO_status']}`；",
        f"- 新tree LOTO：`{result['new_tree_LOTO_status']}`；",
        f"- 监测站P2：`{result['monitored_station_P2_status']}`。",
        "",
        "## 水文与H1语义已纠正",
        "",
        "正式水量通量、quick/GW分配、浓度分母和河道流量全部来自Q72 `structural_canonical main`。Andreadis只提供静态河宽，参考流量没有进入模型。",
        "",
        "当前H1严格为：",
        "",
        "$$H=\\tau/h=LW/(86400Q).$$",
        "",
        "它是areal/benthic hydraulic exposure，不是河道停留时间。深度、坡度和Manning系数不进入正式预测量。",
        "",
        "## 数值与血缘审计",
        "",
        f"- Q72 quick_rho：`{hydro['q72_quick_rho_values']}`；TN代码值：`{hydro['tn_legacy_code_q_rho']}`；",
        f"- q_local=quick+GW最大绝对误差：`{hydro['q_local_quick_plus_gw_max_abs_error_mm']:.3e} mm`；",
        f"- H0全12成员最大复现误差：`{mass.h0_parent_reproduction_relative_error.max():.3e}`；",
        f"- 河道路由最大相对质量误差：`{mass[['h0_mass_balance_relative_error','h1_mass_balance_relative_error']].to_numpy().max():.3e}`；",
        f"- small-tree inference sensitive：`{result['small_tree_inference']['sensitive']}`；",
        "- 旧iid-SSE MCMC与macro-RMSE锁不一致，已退出权威证据链。",
        "",
        "## 科学边界",
        "",
        result["claim_boundary"],
        "本轮没有加入温度、真实停留时间、水库storage、新Legacy结构或WWTP。2022 TN没有读取。",
    ]
    (REPORTS / "current_tn_mainline_audit_and_repair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)
    lock = input_lock()
    dump_json(LOCKS / "input_lock.json", lock)
    obs = development_observations()
    folds = fold_registry()
    obs.to_parquet(LOCKS / "development_tn_2016_2021.parquet", index=False)
    h19 = load_module(OLD_SHARED, "hierarchical19_readonly")

    hydro, mass = hydrology_and_mass_audits(h19)
    mass.to_parquet(OUT / "parent_mass_balance_audit.parquet", index=False)
    dump_json(REPORTS / "hydrologic_provenance_audit.json", hydro)
    if hydro["status"] != "PASS" or not mass.h0_reproduction_pass.all() or not mass.mass_balance_pass.all():
        raise RuntimeError("STOP_PARENT_HYDROLOGY_OR_MASS_AUDIT")

    temporal_metrics, temporal_dist = temporal_evidence()
    nested = nested_evidence(obs, folds, h19)
    selections, nested_predictions, nested_metrics, skills, tree, nested_dist = nested
    metrics = pd.concat([temporal_metrics, nested_metrics], ignore_index=True)
    distributions = pd.concat([temporal_dist, nested_dist], ignore_index=True)

    temporal_params = pd.read_parquet(OLD_OUT / "temporal_fold_parameters.parquet")
    temporal_selection = select_h0_h1(temporal_params, ["model_id", "fold_id"])
    temporal_selection.to_parquet(OUT / "temporal_fold_parent_selection.parquet", index=False)
    selections.to_parquet(OUT / "outer_spatial_parent_selection.parquet", index=False)
    nested_predictions.to_parquet(OUT / "unbiased_nested_parent_predictions.parquet", index=False)
    metrics.to_parquet(OUT / "parent_readjudication_metrics.parquet", index=False)
    distributions.to_parquet(OUT / "paired_bootstrap_distributions.parquet", index=False)
    skills.to_parquet(OUT / "absolute_spatial_skill.parquet", index=False)
    tree.to_parquet(OUT / "small_tree_inference_sensitivity.parquet", index=False)

    result = decision(metrics, skills, tree, temporal_params)
    result["input_lock_sha256"] = sha256(LOCKS / "input_lock.json")
    result["hydrology_audit_sha256"] = sha256(REPORTS / "hydrologic_provenance_audit.json")
    dump_json(REPORTS / "parent_readjudication.json", result)
    parent_lock = {
        "lock_id": "20260824_repaired_parent",
        "selected_full_development_process_parent": result["selected_full_development_process_parent"],
        "domain_status": {
            key: result[key]
            for key in (
                "temporal_process_status",
                "within_observed_tree_LOSO_status",
                "new_tree_LOTO_status",
                "monitored_station_P2_status",
            )
        },
        "temporal_fold_registry": str(OUT / "temporal_fold_parent_selection.parquet"),
        "outer_spatial_registry": str(OUT / "outer_spatial_parent_selection.parquet"),
        "water_flux_source": "Q72 structural_canonical main",
        "formal_h1_semantics": "areal/benthic hydraulic exposure",
        "temperature_used": False,
        "WWTP_used": False,
        "development_years": [2016, 2021],
        "OOF_years": [2018, 2019, 2020, 2021],
        "TN_2022_values_read": False,
    }
    dump_json(LOCKS / "parent_lock_registry.json", parent_lock)
    write_report(result, hydro, mass)
    completion = {
        "status": "PASS",
        "contract_sha256": sha256(HERE / "experiment_contract.json"),
        "input_lock_sha256": sha256(LOCKS / "input_lock.json"),
        "parent_lock_sha256": sha256(LOCKS / "parent_lock_registry.json"),
        "all_h0_reproductions_pass": bool(mass.h0_reproduction_pass.all()),
        "all_mass_balance_checks_pass": bool(mass.mass_balance_pass.all()),
        "hydrologic_provenance_pass": hydro["status"] == "PASS",
        "TN_2022_values_read": False,
    }
    dump_json(REPORTS / "stage_completion_audit.json", completion)
    print(json.dumps({"decision": result, "completion": completion}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
