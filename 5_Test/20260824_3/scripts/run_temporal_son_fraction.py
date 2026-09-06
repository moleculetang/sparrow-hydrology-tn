from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260824_3"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
CACHE = HERE / "cache" / "local"
CONTRACT = HERE / "experiment_contract.json"
MANIFEST = HERE / "program_manifest.json"
PARENT_LOCK = TEST / "20260824_1" / "locks" / "parent_lock_registry.json"
STAGE2_LOCK = TEST / "20260824_2" / "final_lock.json"
H_SHARED_PATH = TEST / "20260820_19" / "scripts" / "hierarchical19_shared.py"
LEGACY_SHARED_PATH = TEST / "20260818_1" / "scripts" / "legacy18_shared.py"
PARENT_LOCAL = TEST / "20260820_10" / "cache" / "parent_local"
SOURCE_LEDGER = TEST / "20260815_2" / "outputs" / "reach_year_n_ledger_1961_2022.parquet"
MONTHLY_INPUT = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
OBSERVATIONS = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLD_REGISTRY = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
EXPOSURE = TEST / "20260820_18" / "outputs" / "reach_month_aquatic_uptake_exposure_2006_2021.parquet"

MUS = (12, 36, 60, 96, 144, 240)
PHI_GRID = tuple(float(v) for v in np.round(np.linspace(0.0, 1.0, 11), 1))
TAU_SON = 12
RHO_SON = TAU_SON / (1.0 + TAU_SON)
SPIN_TOL = 1e-8
SPIN_MAX = 20_000
WATER_EPS = 1e-12
MASS_REL_TOL = 1e-10
NESTING_REL_TOL = 1e-12
TIE_TOL = 1e-8
SEED = 2026082403


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_pre_tn_input_lock() -> dict[str, object]:
    paths = [
        CONTRACT, MANIFEST, Path(__file__), PARENT_LOCK, STAGE2_LOCK,
        H_SHARED_PATH, LEGACY_SHARED_PATH, SOURCE_LEDGER, MONTHLY_INPUT,
        OBSERVATIONS, FOLD_REGISTRY, EXPOSURE,
    ]
    paths.extend(PARENT_LOCAL / f"S0_mu_{mu:03d}m.parquet" for mu in MUS)
    paths.extend(PARENT_LOCAL / f"S1_tau_012m_mu_{mu:03d}m.parquet" for mu in MUS)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"STOP_MISSING_INPUTS: {missing}")
    parent = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    stage2 = json.loads(STAGE2_LOCK.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if parent["selected_full_development_process_parent"] != "H1_GLOBAL":
        raise RuntimeError("STOP_PARENT_NOT_H1_GLOBAL")
    if stage2["terminal_scientific_status"] != "SOURCE_IDENTITY_NOT_IDENTIFIABLE":
        raise RuntimeError("STOP_STAGE2_TERMINAL_CHANGED")
    if not contract["registered_before_development_TN_read"]:
        raise RuntimeError("STOP_NOT_REGISTERED")
    hashes = {str(path): sha256(path) for path in paths}
    lock = {
        "lock_id": "20260824_3_pre_development_TN_input_lock",
        "created_before_development_TN_read": True,
        "TN_values_read_at_lock_creation": False,
        "TN_2022_values_read": False,
        "files": hashes,
        "aggregate_sha256": hashlib.sha256(
            "\n".join(f"{key}|{value}" for key, value in sorted(hashes.items())).encode("utf-8")
        ).hexdigest(),
    }
    dump_json(LOCKS / "pre_tn_input_lock.json", lock)
    return lock


def withdraw(pool: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    removed = np.minimum(pool, demand)
    pool -= removed
    return removed, demand - removed


def spinup_fraction(
    shared: ModuleType,
    phi: float,
    mu: int,
    early_positive: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    state = {name: np.zeros(len(early_positive), dtype=float) for name in ("son", "mobile", "quick", "gw")}
    rho_gw = mu / (1.0 + mu)
    final_balance = np.zeros(len(early_positive), dtype=float)
    delta = np.inf
    for cycle in range(1, SPIN_MAX + 1):
        before = np.concatenate([value.copy() for value in state.values()])
        for t in range(12):
            start = sum(value.astype(np.longdouble) for value in state.values())
            bypass, _, flush, qshare, gshare = shared.operator_water_partitions(arrays, t, "F00")
            direct = early_positive * bypass
            pool_input = early_positive - direct
            son_input = phi * pool_input
            mobile_input = (1.0 - phi) * pool_input
            state["son"] += son_input
            mineralized = (1.0 - RHO_SON) * state["son"]
            state["son"] -= mineralized
            state["mobile"] += mobile_input + mineralized
            mobilized = state["mobile"] * flush
            state["mobile"] -= mobilized
            state["quick"] += direct + mobilized * qshare
            state["gw"] += mobilized * gshare
            qrel = np.where(arrays["quick_release_mm"][t] > WATER_EPS, (1.0 - shared.Q_RHO) * state["quick"], 0.0)
            grel = np.where(arrays["gw_discharge_mm"][t] > WATER_EPS, (1.0 - rho_gw) * state["gw"], 0.0)
            state["quick"] -= qrel
            state["gw"] -= grel
            end = sum(value.astype(np.longdouble) for value in state.values())
            final_balance = np.asarray(
                direct.astype(np.longdouble) + mobile_input.astype(np.longdouble) + son_input.astype(np.longdouble)
                + start - qrel.astype(np.longdouble) - grel.astype(np.longdouble) - end,
                dtype=float,
            )
        after = np.concatenate(list(state.values()))
        delta = float(np.max(np.abs(after - before)))
        if delta <= SPIN_TOL:
            break
    return state, {
        "phi_A": phi, "mu_month": mu, "cycles": cycle,
        "converged": bool(delta <= SPIN_TOL),
        "terminal_max_abs_delta_kg_n": delta,
        "final_cycle_max_abs_mass_balance_error_kg_n": float(np.max(np.abs(final_balance))),
        "son_end_kg_n": float(state["son"].sum()),
        "mobile_end_kg_n": float(state["mobile"].sum()),
        "quick_end_kg_n": float(state["quick"].sum()),
        "gw_end_kg_n": float(state["gw"].sum()),
    }


def simulate_fraction(
    shared: ModuleType,
    phi: float,
    mu: int,
    reach_ids: np.ndarray,
    times: list[tuple[int, int]],
    arrays: dict[str, np.ndarray],
    early_positive: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    state, spin = spinup_fraction(shared, phi, mu, early_positive, arrays)
    if not spin["converged"]:
        raise RuntimeError(f"STOP_SPINUP: phi={phi}, mu={mu}")
    rho_gw = mu / (1.0 + mu)
    rows: list[pd.DataFrame] = []
    max_abs = 0.0
    max_rel = 0.0
    minimum = np.inf
    for t, (year, month) in enumerate(times):
        if year > 2021:
            break
        start = sum(value.astype(np.longdouble) for value in state.values())
        positive = arrays["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        negative = arrays["negative_legacy_eligible_n_surplus_kg_n_month"][t].copy()
        removed_mobile, left = withdraw(state["mobile"], negative)
        removed_son, unmet = withdraw(state["son"], left)
        removed = removed_mobile + removed_son
        bypass, _, flush, qshare, gshare = shared.operator_water_partitions(arrays, t, "F00")
        direct = positive * bypass
        pool_input = positive - direct
        son_input = phi * pool_input
        mobile_input = (1.0 - phi) * pool_input
        state["son"] += son_input
        mineralized = (1.0 - RHO_SON) * state["son"]
        state["son"] -= mineralized
        state["mobile"] += mobile_input + mineralized
        mobilized = state["mobile"] * flush
        state["mobile"] -= mobilized
        qin = direct + mobilized * qshare
        gin = mobilized * gshare
        state["quick"] += qin
        state["gw"] += gin
        qrel = np.where(arrays["quick_release_mm"][t] > WATER_EPS, (1.0 - shared.Q_RHO) * state["quick"], 0.0)
        grel = np.where(arrays["gw_discharge_mm"][t] > WATER_EPS, (1.0 - rho_gw) * state["gw"], 0.0)
        state["quick"] -= qrel
        state["gw"] -= grel
        end = sum(value.astype(np.longdouble) for value in state.values())
        inputs = direct.astype(np.longdouble) + mobile_input.astype(np.longdouble) + son_input.astype(np.longdouble)
        balance = inputs + start - qrel.astype(np.longdouble) - grel.astype(np.longdouble) - removed.astype(np.longdouble) - end
        scale = np.abs(inputs) + np.abs(start) + np.abs(qrel) + np.abs(grel) + np.abs(removed) + np.abs(end)
        relative = np.asarray(np.abs(balance) / np.maximum(scale, 1.0), dtype=float)
        max_abs = max(max_abs, float(np.max(np.abs(balance))))
        max_rel = max(max_rel, float(np.max(relative)))
        minimum = min(minimum, *(float(v.min()) for v in state.values()), float(qrel.min()), float(grel.min()))
        if 2016 <= year <= 2021:
            rows.append(pd.DataFrame({
                "reach_id": reach_ids, "year": year, "month": month,
                "quick_tn_release_kg_n": qrel, "gw_tn_release_kg_n": grel,
            }))
    return pd.concat(rows, ignore_index=True), spin, {
        "phi_A": phi, "mu_month": mu,
        "max_abs_mass_balance_error_kg_n": max_abs,
        "max_relative_mass_balance_error": max_rel,
        "minimum_state_or_flux_kg_n": minimum,
        "unmet_negative_surplus_end_kg_n": float(unmet.sum()),
    }


def prepare_local_candidates() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shared = load_module(LEGACY_SHARED_PATH, "son_fraction_legacy_shared_prepare")
    reach_ids, times, arrays, early = shared.prepare_arrays()
    spin_rows = []
    mass_rows = []
    nesting_rows = []
    CACHE.mkdir(parents=True, exist_ok=True)
    for mu in MUS:
        endpoints: dict[float, pd.DataFrame] = {}
        for phi in PHI_GRID:
            path = CACHE / f"phi_{phi:.1f}_mu_{mu:03d}m.parquet"
            frame, spin, mass = simulate_fraction(shared, phi, mu, reach_ids, times, arrays, early)
            frame.to_parquet(path, index=False)
            spin_rows.append(spin)
            mass_rows.append(mass)
            if phi in (0.0, 1.0):
                endpoints[phi] = frame
        for phi, model_id in ((0.0, f"S0_mu_{mu:03d}m"), (1.0, f"S1_tau_012m_mu_{mu:03d}m")):
            reference = pd.read_parquet(PARENT_LOCAL / f"{model_id}.parquet")
            reference = reference.loc[reference.year.between(2016, 2021), [
                "reach_id", "year", "month", "quick_tn_release_kg_n", "gw_tn_release_kg_n"
            ]].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
            candidate = endpoints[phi].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
            if not candidate[["reach_id", "year", "month"]].equals(reference[["reach_id", "year", "month"]]):
                raise RuntimeError("STOP_ENDPOINT_KEY_MISMATCH")
            qerr = float(np.max(np.abs(candidate.quick_tn_release_kg_n - reference.quick_tn_release_kg_n)))
            gerr = float(np.max(np.abs(candidate.gw_tn_release_kg_n - reference.gw_tn_release_kg_n)))
            scale = max(float(reference[["quick_tn_release_kg_n", "gw_tn_release_kg_n"]].abs().to_numpy().max()), 1.0)
            rel = max(qerr, gerr) / scale
            nesting_rows.append({
                "phi_A": phi, "mu_month": mu, "reference_model_id": model_id,
                "max_abs_quick_error_kg_n": qerr, "max_abs_gw_error_kg_n": gerr,
                "relative_error": rel, "pass": bool(rel <= NESTING_REL_TOL),
            })
    spin = pd.DataFrame(spin_rows)
    mass = pd.DataFrame(mass_rows)
    nesting = pd.DataFrame(nesting_rows)
    spin.to_parquet(OUT / "operator_spinup_audit.parquet", index=False)
    mass.to_parquet(OUT / "operator_mass_balance_audit.parquet", index=False)
    nesting.to_parquet(OUT / "endpoint_exact_nesting_audit.parquet", index=False)
    if not spin.converged.all() or mass.max_relative_mass_balance_error.max() > MASS_REL_TOL or mass.minimum_state_or_flux_kg_n.min() < -1e-8:
        raise RuntimeError("STOP_OPERATOR_NUMERICAL_CONTRACT")
    if not nesting["pass"].all():
        raise RuntimeError("STOP_ENDPOINT_EXACT_NESTING")
    return spin, mass, nesting


def router_for_phi(h: ModuleType, base: object, phi: float, mu: int) -> object:
    local = pd.read_parquet(CACHE / f"phi_{phi:.1f}_mu_{mu:03d}m.parquet").sort_values(["year", "month", "reach_id"])
    shape = base.local_q.shape
    return h.HydraulicRouter(
        model_id=f"phi_{phi:.1f}_mu_{mu:03d}m",
        reach_ids=base.reach_ids.copy(), years=base.years.copy(), months=base.months.copy(),
        local_q=local.quick_tn_release_kg_n.to_numpy(float).reshape(shape),
        local_g=local.gw_tn_release_kg_n.to_numpy(float).reshape(shape),
        h_full=base.h_full.copy(), h_mid=base.h_mid.copy(), water=base.water.copy(), x=base.x.copy(),
        terminal_by_reach=base.terminal_by_reach.copy(), order_index=list(base.order_index),
        downstream_index=dict(base.downstream_index),
    )


def parameter_row(mu: int, fold_id: str, phi: float, fit: dict[str, object], selected: bool, endpoint_selected: bool) -> dict[str, object]:
    vf = np.asarray(fit["vf"], dtype=float)
    return {
        "mu_month": mu, "fold_id": fold_id, "phi_A": phi,
        "selected_partial": selected, "selected_endpoint": endpoint_selected,
        "training_objective": float(fit["objective"]),
        "training_data_objective": float(fit["data_objective"]),
        "v_f_m_per_day": float(np.asarray(fit["parameters"])[0]),
        "vf_boundary": bool(vf.max() >= 0.49),
        "eta_quick": float(fit["eta"][0]), "eta_gw": float(fit["eta"][1]),
        "eta_boundary": bool(fit["diagnostic"]["eta_boundary"]),
        "outer_success": bool(fit["outer_success"]), "outer_nfev": int(fit["outer_nfev"]),
    }


def fit_mu(mu: int) -> tuple[str, str, str, str]:
    h = load_module(H_SHARED_PATH, f"son_fraction_h_{mu}")
    shared = h.parent_shared()
    obs = h.development_observations()
    folds = h.fold_registry()
    base = h.build_router(f"S0_mu_{mu:03d}m", shared)
    routers = {phi: router_for_phi(h, base, phi, mu) for phi in PHI_GRID}
    pred_rows = []
    param_rows = []
    readout_rows = []
    selection_rows = []
    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)
        train = obs.loc[obs.year.between(int(fold.train_start_year), int(fold.train_end_year))].copy()
        test = obs.loc[obs.year.eq(int(fold.evaluation_year))].copy()
        fits = {phi: h.fit_structure(routers[phi], train, "H1_GLOBAL", shared) for phi in PHI_GRID}
        best = min(float(fit["objective"]) for fit in fits.values())
        selected_phi = min(phi for phi, fit in fits.items() if float(fit["objective"]) <= best + TIE_TOL)
        endpoint_best = min(float(fits[phi]["objective"]) for phi in (0.0, 1.0))
        endpoint_phi = min(phi for phi in (0.0, 1.0) if float(fits[phi]["objective"]) <= endpoint_best + TIE_TOL)
        selection_rows.append({
            "mu_month": mu, "fold_id": fold_id, "evaluation_year": int(fold.evaluation_year),
            "selected_phi_A": selected_phi, "selected_endpoint_phi_A": endpoint_phi,
            "selected_training_objective": float(fits[selected_phi]["objective"]),
            "endpoint_training_objective": float(fits[endpoint_phi]["objective"]),
        })
        for phi, fit in fits.items():
            param_rows.append(parameter_row(mu, fold_id, phi, fit, phi == selected_phi, phi == endpoint_phi and phi in (0.0, 1.0)))
        mechanisms = {
            "NO_SON_PHI0": 0.0,
            "FULL_SON_PHI1": 1.0,
            "ENDPOINT_SELECTED": endpoint_phi,
            "PARTIAL_SELECTED": selected_phi,
        }
        for mechanism, phi in mechanisms.items():
            fit = fits[phi]
            router = routers[phi]
            vf = np.asarray(fit["vf"], dtype=float)
            test_frame = router.frame(test, vf)
            p1 = shared.predict_layer(test_frame, "P1", np.asarray(fit["eta"]), fit["effects"])
            p1["mu_month"] = mu; p1["fold_id"] = fold_id; p1["mechanism"] = mechanism; p1["phi_A"] = phi
            pred_rows.append(p1)
            train_frame = router.frame(train, vf)
            readout = h.fit_selected_readout(train_frame, "P2", shared)
            p2 = h.predict_selected(test_frame, "P2", readout, shared)
            p2["mu_month"] = mu; p2["fold_id"] = fold_id; p2["mechanism"] = mechanism; p2["phi_A"] = phi
            pred_rows.append(p2)
            readout_rows.append({
                "mu_month": mu, "fold_id": fold_id, "mechanism": mechanism, "phi_A": phi,
                "eta_quick": float(readout["eta"][0]), "eta_gw": float(readout["eta"][1]),
                "eta_boundary": bool(readout["diagnostic"]["eta_boundary"]),
                "success": bool(readout["diagnostic"]["success"]),
                "station_effect_count": len(readout["effects"]),
            })
    mu_cache = HERE / "cache" / "temporal"
    mu_cache.mkdir(parents=True, exist_ok=True)
    pred_path = mu_cache / f"mu_{mu:03d}m_predictions.parquet"
    param_path = mu_cache / f"mu_{mu:03d}m_parameters.parquet"
    readout_path = mu_cache / f"mu_{mu:03d}m_readouts.parquet"
    selection_path = mu_cache / f"mu_{mu:03d}m_selections.parquet"
    pd.concat(pred_rows, ignore_index=True).to_parquet(pred_path, index=False)
    pd.DataFrame(param_rows).to_parquet(param_path, index=False)
    pd.DataFrame(readout_rows).to_parquet(readout_path, index=False)
    pd.DataFrame(selection_rows).to_parquet(selection_path, index=False)
    return str(pred_path), str(param_path), str(readout_path), str(selection_path)


def overall_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame.tn_mg_l.to_numpy(float)
    pred = frame.pred_tn_mg_l.to_numpy(float)
    residual = pred - obs
    denominator = float(np.sum((obs - obs.mean()) ** 2))
    nse = 1.0 - float(np.sum(residual ** 2)) / denominator if denominator > 0 else np.nan
    r = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 and np.std(obs) > 0 and np.std(pred) > 0 else np.nan
    return {
        "n": len(frame), "rmse_mg_l": float(np.sqrt(np.mean(residual ** 2))),
        "mae_mg_l": float(np.mean(np.abs(residual))),
        "pbias_percent": float(100.0 * np.sum(residual) / np.sum(obs)),
        "nse": nse, "pearson_r": r, "pearson_r2": r * r,
        "rmse_log1p": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2))),
    }


def station_distribution_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    rows = []
    for station, group in frame.groupby("station_key"):
        values = overall_metrics(group)
        values["station_key"] = station
        rows.append(values)
    station = pd.DataFrame(rows)
    finite_nse = station.nse.replace([np.inf, -np.inf], np.nan).dropna()
    finite_r2 = station.pearson_r2.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "stations": int(len(station)), "stations_with_valid_nse": int(len(finite_nse)),
        "median_station_nse": float(finite_nse.median()),
        "median_station_pearson_r2": float(finite_r2.median()),
        "median_station_rmse_mg_l": float(station.rmse_mg_l.median()),
        "mean_station_rmse_log1p": float(station.rmse_log1p.mean()),
    }


def assemble_results(paths: list[tuple[str, str, str, str]], h: ModuleType) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.concat([pd.read_parquet(row[0]) for row in paths], ignore_index=True)
    parameters = pd.concat([pd.read_parquet(row[1]) for row in paths], ignore_index=True)
    readouts = pd.concat([pd.read_parquet(row[2]) for row in paths], ignore_index=True)
    selections = pd.concat([pd.read_parquet(row[3]) for row in paths], ignore_index=True)
    predictions.to_parquet(OUT / "temporal_oof_predictions.parquet", index=False)
    parameters.to_parquet(OUT / "candidate_fold_parameters.parquet", index=False)
    readouts.to_parquet(OUT / "candidate_fold_readouts.parquet", index=False)
    selections.to_parquet(OUT / "fold_phi_selections.parquet", index=False)
    gate_rows = []
    seed = SEED
    comparisons = (("PARTIAL_SELECTED", "NO_SON_PHI0"), ("PARTIAL_SELECTED", "ENDPOINT_SELECTED"), ("FULL_SON_PHI1", "NO_SON_PHI0"))
    for mu in MUS:
        for layer in ("P1", "P2"):
            subset = predictions.loc[predictions.mu_month.eq(mu) & predictions.layer.eq(layer)]
            for candidate_name, reference_name in comparisons:
                candidate = subset.loc[subset.mechanism.eq(candidate_name)]
                reference = subset.loc[subset.mechanism.eq(reference_name)]
                for block, block_col in (("station", "station_key"), ("tree", "terminal_tree_id")):
                    seed += 1
                    gate_rows.append({
                        "mu_month": mu, "layer": layer, "candidate": candidate_name,
                        "reference": reference_name, "block": block,
                        **h.paired_bootstrap(reference, candidate, block_col, seed),
                    })
    gates = pd.DataFrame(gate_rows)
    gates.to_parquet(OUT / "paired_temporal_gates.parquet", index=False)
    return predictions, parameters, readouts, selections, gates


def ensemble_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["station_key", "year", "month", "fold_id", "layer", "mechanism"]
    first = ["tn_mg_l", "terminal_tree_id"]
    ensemble = predictions.groupby(keys, as_index=False).agg(
        pred_tn_mg_l=("pred_tn_mg_l", "mean"),
        tn_mg_l=("tn_mg_l", "first"),
        terminal_tree_id=("terminal_tree_id", "first"),
    )
    ensemble.to_parquet(OUT / "six_mu_ensemble_predictions.parquet", index=False)
    metric_rows = []
    monthly_rows = []
    for (layer, mechanism), group in ensemble.groupby(["layer", "mechanism"]):
        metric_rows.append({"layer": layer, "mechanism": mechanism, **overall_metrics(group), **station_distribution_metrics(group)})
        work = group.assign(residual_log=np.log1p(group.tn_mg_l) - np.log1p(group.pred_tn_mg_l))
        for month, month_group in work.groupby("month"):
            monthly_rows.append({
                "layer": layer, "mechanism": mechanism, "month": int(month),
                "signed_bias_log_obs_minus_pred": float(month_group.residual_log.mean()),
                "rmse_log1p": float(np.sqrt(np.mean(month_group.residual_log ** 2))),
                "n": len(month_group),
            })
    metrics = pd.DataFrame(metric_rows)
    monthly = pd.DataFrame(monthly_rows)
    metrics.to_parquet(OUT / "ensemble_performance_metrics.parquet", index=False)
    monthly.to_parquet(OUT / "ensemble_monthly_residuals.parquet", index=False)
    return ensemble, metrics, monthly


def count_gate(gates: pd.DataFrame, layer: str, reference: str, field: str, block: str) -> int:
    part = gates.loc[
        gates.layer.eq(layer) & gates.candidate.eq("PARTIAL_SELECTED")
        & gates.reference.eq(reference) & gates.block.eq(block)
    ]
    return int(part[field].sum())


def decide(
    spin: pd.DataFrame,
    mass: pd.DataFrame,
    nesting: pd.DataFrame,
    parameters: pd.DataFrame,
    readouts: pd.DataFrame,
    selections: pd.DataFrame,
    gates: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict[str, object]:
    counts: dict[str, int] = {}
    for layer, reference, label in (
        ("P1", "NO_SON_PHI0", "p1_vs_no_son"),
        ("P1", "ENDPOINT_SELECTED", "p1_vs_endpoint"),
        ("P2", "NO_SON_PHI0", "p2_vs_no_son"),
    ):
        for block in ("station", "tree"):
            counts[f"{label}_{block}_noninferior"] = count_gate(gates, layer, reference, "noninferior", block)
            counts[f"{label}_{block}_improved"] = count_gate(gates, layer, reference, "predictively_improved", block)
    module_process = all(
        counts[f"p1_vs_no_son_{block}_noninferior"] >= 5
        and counts[f"p1_vs_no_son_{block}_improved"] >= 4
        for block in ("station", "tree")
    )
    fractional_extension = all(
        counts[f"p1_vs_endpoint_{block}_noninferior"] >= 5
        and counts[f"p1_vs_endpoint_{block}_improved"] >= 4
        for block in ("station", "tree")
    )
    p2_preserved = all(counts[f"p2_vs_no_son_{block}_noninferior"] >= 5 for block in ("station", "tree"))
    stability = selections.groupby("mu_month").selected_phi_A.apply(lambda x: int((x > 0).sum()) >= 3)
    stable_models = int(stability.sum())
    selected_params = parameters.loc[parameters.selected_partial]
    boundary_models = int((selected_params.groupby("mu_month").apply(
        lambda g: int((g.eta_boundary | g.vf_boundary).sum()) >= 2, include_groups=False
    )).sum())
    numeric = bool(
        spin.converged.all() and mass.max_relative_mass_balance_error.max() <= MASS_REL_TOL
        and mass.minimum_state_or_flux_kg_n.min() >= -1e-8 and nesting["pass"].all()
        and parameters.outer_success.all() and readouts.success.all() and boundary_models <= 2
    )
    phi_stable = stable_models >= 5
    if not numeric:
        status = "NUMERICALLY_CONFOUNDED"
    elif module_process and fractional_extension and p2_preserved and phi_stable:
        status = "PARTIAL_CROPLAND_SLOW_RELEASE_TEMPORALLY_SUPPORTED_PENDING_SPATIAL"
    elif module_process and p2_preserved and phi_stable:
        status = "SLOW_RELEASE_SUPPORTED_EXISTING_ENDPOINT_SUFFICIENT"
    elif p2_preserved and not module_process:
        status = "STATION_READOUT_COMPENSATED_NOT_PROCESS_SUPPORTED"
    else:
        status = "CROPLAND_SLOW_RELEASE_NOT_SUPPORTED"
    selected_distribution = selections.selected_phi_A.value_counts().sort_index().to_dict()
    metric_records = metrics.loc[metrics.mechanism.isin(["NO_SON_PHI0", "ENDPOINT_SELECTED", "PARTIAL_SELECTED"])].to_dict("records")
    return {
        "status": status,
        "question_answer": "yes" if status.startswith("PARTIAL_CROPLAND") or status.startswith("SLOW_RELEASE_SUPPORTED") else "no_registered_support",
        "module_process_gate": module_process,
        "fractional_extension_gate": fractional_extension,
        "P2_predictive_preservation_gate": p2_preserved,
        "phi_stability_gate": phi_stable,
        "stable_nonzero_phi_models": stable_models,
        "selected_phi_fold_distribution": {str(k): int(v) for k, v in selected_distribution.items()},
        "ensemble_boundary_confounded_models": boundary_models,
        "numeric_contract_pass": numeric,
        "model_gate_counts": counts,
        "ensemble_metrics": metric_records,
        "spatial_extrapolation_status": "not_evaluated_at_temporal_stage",
        "nested_spatial_authorized": status == "PARTIAL_CROPLAND_SLOW_RELEASE_TEMPORALLY_SUPPORTED_PENDING_SPATIAL",
        "TN_values_read": True,
        "TN_years_read": [2016, 2017, 2018, 2019, 2020, 2021],
        "TN_2022_values_read": False,
        "claim_boundary": "phi_A is an effective generic cropland source-zone slow-release fraction, not an identified manure fraction, soil SON inventory or universal mineralization constant.",
    }


def write_report(decision: dict[str, object], metrics: pd.DataFrame, selections: pd.DataFrame) -> None:
    p1 = metrics.loc[metrics.layer.eq("P1")].set_index("mechanism")
    p2 = metrics.loc[metrics.layer.eq("P2")].set_index("mechanism")
    lines = [
        "# 通用农田氮缓释比例实验结果",
        "",
        f"正式裁决：`{decision['status']}`。",
        "",
        "## 模型定义",
        "",
        "本轮没有重复增加已有S1-12。它在S0与S1-12之间加入一个全流域比例：",
        "",
        r"\[N_{SON,in}=\phi_A N_{post-bypass},\qquad N_{mobile,in}=(1-\phi_A)N_{post-bypass}.\]",
        "",
        "`phi_A=0`精确复现S0，`phi_A=1`精确复现S1-12；Q72、F00、T1、H1_GLOBAL和12个月SON时间尺度均冻结。",
        "",
        "## 六个mu成员OOF集成效果",
        "",
        "| Layer | Structure | RMSE (mg/L) | NSE | Pearson R2 | log-RMSE | median station NSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for layer, table in (("P1", p1), ("P2", p2)):
        for mechanism in ("NO_SON_PHI0", "ENDPOINT_SELECTED", "PARTIAL_SELECTED"):
            row = table.loc[mechanism]
            lines.append(
                f"| {layer} | {mechanism} | {row.rmse_mg_l:.4f} | {row.nse:.4f} | {row.pearson_r2:.4f} | {row.rmse_log1p:.4f} | {row.median_station_nse:.4f} |"
            )
    lines.extend([
        "",
        "## 参数与门禁",
        "",
        f"- 非零phi稳定模型：{decision['stable_nonzero_phi_models']}/6；",
        f"- fold选择分布：`{json.dumps(decision['selected_phi_fold_distribution'], ensure_ascii=False)}`；",
        f"- P1缓释过程门：`{decision['module_process_gate']}`；",
        f"- 部分比例相对现有端点新增价值门：`{decision['fractional_extension_gate']}`；",
        f"- P2预测保持门：`{decision['P2_predictive_preservation_gate']}`；",
        f"- 数值合同：`{decision['numeric_contract_pass']}`；",
        "",
        "## 科学边界",
        "",
        "即使缓释结构获得支持，也只能说明一个通用、有效的农田源区缓释算子改善了时间外TN预测。河流TN不能据此识别粪肥比例、真实土壤SON库存或普适矿化常数。只有时间门通过后，才允许做nested LOSO/LOTO；在此之前不得升级空间主线，也不得读取2022 TN。",
    ])
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require_runtime()
    for path in (OUT, REPORTS, LOCKS, CACHE):
        path.mkdir(parents=True, exist_ok=True)
    input_lock = write_pre_tn_input_lock()
    spin, mass, nesting = prepare_local_candidates()
    # Development TN is first read inside the workers, after the registered lock above exists.
    workers = min(4, len(MUS), os.cpu_count() or 1)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fit_mu, mu): mu for mu in MUS}
        for future in as_completed(futures):
            mu = futures[future]
            results.append((mu, future.result()))
            print(json.dumps({"temporal_mu_complete": mu, "count": len(results)}), flush=True)
    paths = [row[1] for row in sorted(results)]
    h = load_module(H_SHARED_PATH, "son_fraction_h_assemble")
    predictions, parameters, readouts, selections, gates = assemble_results(paths, h)
    ensemble, metrics, monthly = ensemble_tables(predictions)
    decision = decide(spin, mass, nesting, parameters, readouts, selections, gates, metrics)
    decision["pre_tn_input_lock_sha256"] = sha256(LOCKS / "pre_tn_input_lock.json")
    dump_json(REPORTS / "temporal_decision.json", decision)
    write_report(decision, metrics, selections)
    completion = {
        "status": "PASS" if decision["numeric_contract_pass"] else "FAIL",
        "terminal_or_conditional_status": decision["status"],
        "artifacts": {
            str(path): sha256(path) for path in sorted([
                LOCKS / "pre_tn_input_lock.json",
                OUT / "operator_spinup_audit.parquet", OUT / "operator_mass_balance_audit.parquet",
                OUT / "endpoint_exact_nesting_audit.parquet", OUT / "temporal_oof_predictions.parquet",
                OUT / "candidate_fold_parameters.parquet", OUT / "candidate_fold_readouts.parquet",
                OUT / "fold_phi_selections.parquet", OUT / "paired_temporal_gates.parquet",
                OUT / "six_mu_ensemble_predictions.parquet", OUT / "ensemble_performance_metrics.parquet",
                OUT / "ensemble_monthly_residuals.parquet", REPORTS / "temporal_decision.json",
                REPORTS / "technical_report.md",
            ])
        },
        "TN_2022_values_read": False,
        "nested_spatial_authorized": decision["nested_spatial_authorized"],
    }
    dump_json(REPORTS / "stage_completion_audit.json", completion)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
