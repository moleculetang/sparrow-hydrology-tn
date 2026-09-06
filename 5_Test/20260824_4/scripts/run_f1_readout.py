from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260824_4"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
CACHE = HERE / "cache" / "f1"

CONTRACT = HERE / "experiment_contract.json"
MANIFEST = HERE / "program_manifest.json"
STAGE0 = REPORTS / "stage0_repair_completion_audit.json"
H_SHARED_PATH = TEST / "20260820_19" / "scripts" / "hierarchical19_shared.py"
Q72 = TEST / "20260820_2" / "outputs" / "q72_routed_monthly_discharge_2006_2022.parquet"
OBS = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLDS = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"

MUS = (12, 36, 60, 96, 144, 240)
FORMAL_MODELS = tuple(
    sorted(
        [f"S0_mu_{mu:03d}m" for mu in MUS]
        + [f"S1_tau_012m_mu_{mu:03d}m" for mu in MUS]
    )
)
ARMS = (
    "GAUSSIAN_PARENT",
    "STUDENT_T_NU4",
    "GAUSSIAN_CQ_HINGE",
    "STUDENT_T_NU4_CQ_HINGE",
)
NU = 4.0
ETA_LAMBDA = 1.0
STATION_LAMBDA = 12.0
ETA_BOUNDARY_TOL = 0.01
BETA_BOUND = 1.5
SIGMA_BOUNDS = (0.02, 1.0)
BOOT_REPS = 10_000
SEED = 2026082404
SYNTH_REPS = 500


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


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_f1_lock() -> dict[str, object]:
    paths = [CONTRACT, MANIFEST, STAGE0, H_SHARED_PATH, Q72, OBS, FOLDS, Path(__file__)]
    for model_id in FORMAL_MODELS:
        paths.append(TEST / "20260820_19" / "cache" / "temporal" / f"{model_id}__structures.parquet")
        paths.append(TEST / "20260820_10" / "cache" / "parent_local" / f"{model_id}.parquet")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(STAGE0.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("STOP_STAGE0_NOT_PASSED")
    hashes = {str(path): sha256(path) for path in paths}
    lock = {
        "lock_id": "20260824_4_F1_pre_observed_candidate_lock",
        "created_before_F1_observed_candidate_fit": True,
        "TN_2022_values_read": False,
        "files": hashes,
        "aggregate_sha256": hashlib.sha256(
            "\n".join(f"{key}|{value}" for key, value in sorted(hashes.items())).encode("utf-8")
        ).hexdigest(),
    }
    dump_json(LOCKS / "f1_pre_observed_candidate_input_lock.json", lock)
    return lock


def add_fold_q_anomaly(frame: pd.DataFrame, train_start: int, train_end: int) -> pd.DataFrame:
    q = pd.read_parquet(
        Q72,
        filters=[("year", ">=", 2016), ("year", "<=", 2021)],
        columns=["reach_id", "year", "month", "q72_outlet_discharge_m3_s"],
    )
    q["log_q72"] = np.log(q["q72_outlet_discharge_m3_s"].clip(lower=1e-12))
    med = (
        q.loc[q["year"].between(train_start, train_end)]
        .groupby("reach_id", observed=True)["log_q72"]
        .median()
        .rename("training_all_month_log_q_median")
    )
    out = frame.merge(q, on=["reach_id", "year", "month"], how="left", validate="many_to_one")
    out = out.merge(med, on="reach_id", how="left", validate="many_to_one")
    if out[["log_q72", "training_all_month_log_q_median"]].isna().any().any():
        raise RuntimeError("STOP_Q72_CQ_FEATURE_MISSING")
    out["cq_z"] = out["log_q72"] - out["training_all_month_log_q_median"]
    out["cq_low"] = np.minimum(out["cq_z"], 0.0)
    out["cq_high"] = np.maximum(out["cq_z"], 0.0)
    return out


def raw_concentration(frame: pd.DataFrame, eta: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = frame["routed_quick_tn_kg_n"].to_numpy(float)
    g = frame["routed_gw_tn_kg_n"].to_numpy(float)
    water = frame["routed_water_volume_m3"].to_numpy(float)
    base = np.divide(
        (eta[0] * q + eta[1] * g) * 1000.0,
        water,
        out=np.zeros(len(frame), dtype=float),
        where=water > 1e-12,
    )
    hinge = beta[0] * frame["cq_low"].to_numpy(float) + beta[1] * frame["cq_high"].to_numpy(float)
    multiplier = np.exp(np.clip(hinge, -20.0, 20.0))
    value = base * multiplier
    return value, base, multiplier


def station_codes(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    levels = sorted(frame["station_key"].astype(str).unique())
    lookup = {value: i for i, value in enumerate(levels)}
    return frame["station_key"].astype(str).map(lookup).to_numpy(int), levels


def gaussian_fit(train: pd.DataFrame, layer: str, hinge: bool, shared: ModuleType) -> dict[str, object]:
    if not hinge:
        eta, effects, diagnostic = shared.fit_readout(train, layer)
        return {
            "eta": np.asarray(eta, dtype=float),
            "beta": np.zeros(2, dtype=float),
            "effects": effects,
            "sigma": math.nan,
            "success": bool(diagnostic["success"]),
            "nfev": int(diagnostic["nfev"]),
            "eta_boundary": bool(diagnostic["eta_boundary"]),
            "beta_boundary": False,
            "objective": float(diagnostic["cost"]),
        }
    y = np.log1p(train["tn_mg_l"].to_numpy(float))
    codes, levels = station_codes(train)

    def pieces(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        concentration, _, _ = raw_concentration(train, x[:2], x[2:4])
        raw = np.log1p(np.maximum(concentration, 0.0)) - y
        if layer == "P2":
            sums = np.bincount(codes, weights=raw, minlength=len(levels))
            counts = np.bincount(codes, minlength=len(levels)).astype(float)
            effects = -sums / (counts + STATION_LAMBDA)
        else:
            effects = np.zeros(len(levels), dtype=float)
        return raw + effects[codes], effects

    def residual(x: np.ndarray) -> np.ndarray:
        data, effects = pieces(x)
        values = [data, np.sqrt(ETA_LAMBDA) * (x[:2] - 1.0)]
        if layer == "P2":
            values.append(np.sqrt(STATION_LAMBDA) * effects)
        return np.concatenate(values)

    opt = least_squares(
        residual,
        x0=np.array([0.5, 0.5, 0.0, 0.0]),
        bounds=(np.array([0.0, 0.0, -BETA_BOUND, -BETA_BOUND]), np.array([1.0, 1.0, BETA_BOUND, BETA_BOUND])),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=1500,
    )
    data, effect_values = pieces(opt.x)
    effects = {station: float(effect_values[i]) for i, station in enumerate(levels)}
    return {
        "eta": opt.x[:2],
        "beta": opt.x[2:4],
        "effects": effects,
        "sigma": math.nan,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "eta_boundary": bool(np.any(opt.x[:2] <= ETA_BOUNDARY_TOL) or np.any(opt.x[:2] >= 1.0 - ETA_BOUNDARY_TOL)),
        "beta_boundary": bool(np.any(np.abs(opt.x[2:4]) >= BETA_BOUND - 0.01)),
        "objective": float(0.5 * np.sum(np.square(residual(opt.x)))),
        "residual_scale": float(np.sqrt(np.mean(np.square(data)))),
    }


def student_fit(train: pd.DataFrame, layer: str, hinge: bool, start: dict[str, object]) -> dict[str, object]:
    y = np.log1p(train["tn_mg_l"].to_numpy(float))
    codes, levels = station_codes(train)
    n_effect = len(levels) if layer == "P2" else 0
    eta0 = np.asarray(start["eta"], dtype=float)
    beta0 = np.asarray(start["beta"], dtype=float) if hinge else np.zeros(2, dtype=float)
    start_effect = np.array([start["effects"].get(level, 0.0) for level in levels], dtype=float) if n_effect else np.empty(0)
    start_conc, _, _ = raw_concentration(train, eta0, beta0)
    start_resid = np.log1p(start_conc) + (start_effect[codes] if n_effect else 0.0) - y
    sigma0 = float(np.clip(np.sqrt(np.mean(np.square(start_resid))), *SIGMA_BOUNDS))
    x0 = np.concatenate([eta0, beta0 if hinge else np.empty(0), [math.log(sigma0)], start_effect])
    beta_slice = slice(2, 4) if hinge else slice(2, 2)
    log_sigma_index = 4 if hinge else 2
    effect_start = log_sigma_index + 1

    def objective_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x[:2]
        beta = x[beta_slice] if hinge else np.zeros(2, dtype=float)
        sigma = math.exp(float(x[log_sigma_index]))
        effects = x[effect_start:] if n_effect else np.empty(0)
        conc, base, multiplier = raw_concentration(train, eta, beta)
        pred_log = np.log1p(conc) + (effects[codes] if n_effect else 0.0)
        error = pred_log - y
        denom = NU * sigma * sigma + error * error
        loss = len(error) * math.log(sigma) + 0.5 * (NU + 1.0) * float(np.sum(np.log1p(error * error / (NU * sigma * sigma))))
        loss += 0.5 * ETA_LAMBDA * float(np.sum(np.square(eta - 1.0)))
        if n_effect:
            loss += 0.5 * STATION_LAMBDA * float(np.sum(np.square(effects)))
        dlde = (NU + 1.0) * error / denom
        grad = np.zeros_like(x)
        q = train["routed_quick_tn_kg_n"].to_numpy(float) * 1000.0 / train["routed_water_volume_m3"].to_numpy(float)
        g = train["routed_gw_tn_kg_n"].to_numpy(float) * 1000.0 / train["routed_water_volume_m3"].to_numpy(float)
        response_factor = multiplier / (1.0 + conc)
        grad[0] = float(np.sum(dlde * response_factor * q) + ETA_LAMBDA * (eta[0] - 1.0))
        grad[1] = float(np.sum(dlde * response_factor * g) + ETA_LAMBDA * (eta[1] - 1.0))
        if hinge:
            conc_factor = conc / (1.0 + conc)
            grad[2] = float(np.sum(dlde * conc_factor * train["cq_low"].to_numpy(float)))
            grad[3] = float(np.sum(dlde * conc_factor * train["cq_high"].to_numpy(float)))
        grad[log_sigma_index] = float(len(error) - (NU + 1.0) * np.sum(error * error / denom))
        if n_effect:
            grad[effect_start:] = np.bincount(codes, weights=dlde, minlength=n_effect) + STATION_LAMBDA * effects
        return loss, grad

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    if hinge:
        bounds += [(-BETA_BOUND, BETA_BOUND), (-BETA_BOUND, BETA_BOUND)]
    bounds += [(math.log(SIGMA_BOUNDS[0]), math.log(SIGMA_BOUNDS[1]))]
    bounds += [(-2.0, 2.0)] * n_effect
    opt = minimize(
        lambda x: objective_gradient(x),
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-10, "gtol": 1e-7, "maxiter": 1000, "maxfun": 5000},
    )
    x = opt.x
    beta = x[beta_slice] if hinge else np.zeros(2, dtype=float)
    effects = {station: float(x[effect_start + i]) for i, station in enumerate(levels)} if n_effect else {}
    return {
        "eta": x[:2],
        "beta": beta,
        "effects": effects,
        "sigma": float(math.exp(x[log_sigma_index])),
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "eta_boundary": bool(np.any(x[:2] <= ETA_BOUNDARY_TOL) or np.any(x[:2] >= 1.0 - ETA_BOUNDARY_TOL)),
        "beta_boundary": bool(hinge and np.any(np.abs(beta) >= BETA_BOUND - 0.01)),
        "objective": float(opt.fun),
        "optimizer_message": str(opt.message),
    }


def predict(frame: pd.DataFrame, fit: dict[str, object], layer: str) -> pd.DataFrame:
    concentration, _, _ = raw_concentration(frame, np.asarray(fit["eta"]), np.asarray(fit["beta"]))
    effects = (
        frame["station_key"].astype(str).map(fit["effects"]).fillna(0.0).to_numpy(float)
        if layer == "P2"
        else np.zeros(len(frame), dtype=float)
    )
    out = frame.copy()
    out["raw_eta_scaled_tn_mg_l"] = concentration
    out["pred_tn_mg_l"] = np.maximum(np.expm1(np.log1p(concentration) + effects), 0.0)
    out["station_effect"] = effects
    return out


def model_worker(model_id: str) -> tuple[str, str]:
    h19 = load_module(H_SHARED_PATH, f"f1_h19_{model_id}")
    shared = h19.parent_shared()
    obs = h19.development_observations()
    folds = h19.fold_registry()
    router = h19.build_router(model_id, shared)
    structures = pd.read_parquet(TEST / "20260820_19" / "cache" / "temporal" / f"{model_id}__structures.parquet")
    prediction_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    for fold in folds.itertuples(index=False):
        train_obs = obs.loc[obs["year"].between(fold.train_start_year, fold.train_end_year)].copy()
        test_obs = obs.loc[obs["year"].eq(fold.evaluation_year)].copy()
        vrow = structures.loc[
            structures["fold_id"].eq(fold.fold_id)
            & structures["structure"].eq("H1_GLOBAL")
            & structures["evaluation"].eq("temporal")
        ]
        if len(vrow) != 1:
            raise RuntimeError(f"H1 fold parameter mismatch {model_id} {fold.fold_id}")
        vf = float(vrow["v0_m_per_day"].iloc[0])
        vector = np.full(len(router.reach_ids), vf, dtype=float)
        train = add_fold_q_anomaly(router.frame(train_obs, vector), int(fold.train_start_year), int(fold.train_end_year))
        test = add_fold_q_anomaly(router.frame(test_obs, vector), int(fold.train_start_year), int(fold.train_end_year))
        for layer in ("P1", "P2"):
            g_parent = gaussian_fit(train, layer, False, shared)
            g_hinge = gaussian_fit(train, layer, True, shared)
            fits = {
                "GAUSSIAN_PARENT": g_parent,
                "GAUSSIAN_CQ_HINGE": g_hinge,
                "STUDENT_T_NU4": student_fit(train, layer, False, g_parent),
                "STUDENT_T_NU4_CQ_HINGE": student_fit(train, layer, True, g_hinge),
            }
            for arm in ARMS:
                fit = fits[arm]
                pred = predict(test, fit, layer)
                pred["model_id"] = model_id
                pred["fold_id"] = str(fold.fold_id)
                pred["evaluation_year"] = int(fold.evaluation_year)
                pred["layer"] = layer
                pred["arm"] = arm
                pred["v_f_m_per_day"] = vf
                prediction_rows.append(pred)
                parameter_rows.append(
                    {
                        "model_id": model_id,
                        "fold_id": str(fold.fold_id),
                        "evaluation_year": int(fold.evaluation_year),
                        "layer": layer,
                        "arm": arm,
                        "v_f_m_per_day": vf,
                        "eta_quick": float(fit["eta"][0]),
                        "eta_gw": float(fit["eta"][1]),
                        "beta_low": float(fit["beta"][0]),
                        "beta_high": float(fit["beta"][1]),
                        "student_sigma_log1p": float(fit["sigma"]),
                        "success": bool(fit["success"]),
                        "nfev": int(fit["nfev"]),
                        "objective": float(fit["objective"]),
                        "eta_boundary": bool(fit["eta_boundary"]),
                        "beta_boundary": bool(fit["beta_boundary"]),
                        "station_effect_count": len(fit["effects"]),
                    }
                )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    parameters = pd.DataFrame(parameter_rows)
    pred_path = CACHE / f"{model_id}__predictions.parquet"
    par_path = CACHE / f"{model_id}__parameters.parquet"
    predictions.to_parquet(pred_path, index=False)
    parameters.to_parquet(par_path, index=False)
    return str(pred_path), str(par_path)


def station_block_values(frame: pd.DataFrame, anomaly: bool) -> pd.Series:
    work = frame.copy()
    work["obs_log"] = np.log1p(work["tn_mg_l"])
    work["pred_log"] = np.log1p(work["pred_tn_mg_l"])
    if anomaly:
        work["obs_log"] -= work.groupby("station_key", observed=True)["obs_log"].transform("mean")
        work["pred_log"] -= work.groupby("station_key", observed=True)["pred_log"].transform("mean")
    return work.groupby("station_key", observed=True).apply(
        lambda g: float(np.sqrt(np.mean(np.square(g["pred_log"] - g["obs_log"])))),
        include_groups=False,
    )


def tree_block_values(frame: pd.DataFrame) -> pd.Series:
    work = frame.assign(
        error=np.log1p(frame["pred_tn_mg_l"].to_numpy(float)) - np.log1p(frame["tn_mg_l"].to_numpy(float))
    )
    return work.groupby("terminal_tree_id", observed=True)["error"].apply(
        lambda x: float(np.sqrt(np.mean(np.square(x))))
    )


def paired_distributions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    distributions: list[pd.DataFrame] = []
    rng = np.random.default_rng(SEED + 90_000)
    for model_id in FORMAL_MODELS:
        for layer in ("P1", "P2"):
            base = predictions.loc[
                predictions["model_id"].eq(model_id)
                & predictions["layer"].eq(layer)
                & predictions["arm"].eq("GAUSSIAN_PARENT")
            ]
            for metric, block_fun in (
                ("station_anomaly_rmse_log1p", lambda x: station_block_values(x, True)),
                ("station_absolute_rmse_log1p", lambda x: station_block_values(x, False)),
                ("tree_absolute_rmse_log1p", tree_block_values),
            ):
                base_values = block_fun(base)
                candidate_dists: dict[str, np.ndarray] = {}
                point: dict[str, float] = {}
                common_index: np.ndarray | None = None
                for arm in ARMS[1:]:
                    cand = predictions.loc[
                        predictions["model_id"].eq(model_id)
                        & predictions["layer"].eq(layer)
                        & predictions["arm"].eq(arm)
                    ]
                    values = block_fun(cand).reindex(base_values.index)
                    delta = values.to_numpy(float) - base_values.to_numpy(float)
                    if common_index is None:
                        common_index = rng.integers(0, len(delta), size=(BOOT_REPS, len(delta)))
                    dist = delta[common_index].mean(axis=1)
                    candidate_dists[arm] = dist
                    point[arm] = float(delta.mean())
                # Westfall-Young max-T is studentized.  Using the raw maximum
                # deviation would let a high-variance candidate (notably the
                # Student-t arm) inflate every other candidate's interval.
                bootstrap_se = {
                    arm: max(float(np.std(candidate_dists[arm], ddof=1)), 1e-12)
                    for arm in ARMS[1:]
                }
                centered_max_t = np.max(
                    np.column_stack(
                        [
                            (candidate_dists[arm] - point[arm]) / bootstrap_se[arm]
                            for arm in ARMS[1:]
                        ]
                    ),
                    axis=1,
                )
                max_t_quantile = float(np.quantile(centered_max_t, 0.975))
                for arm in ARMS[1:]:
                    dist = candidate_dists[arm]
                    simultaneous_upper = point[arm] + max_t_quantile * bootstrap_se[arm]
                    rows.append(
                        {
                            "model_id": model_id,
                            "layer": layer,
                            "metric": metric,
                            "reference": "GAUSSIAN_PARENT",
                            "candidate": arm,
                            "delta_candidate_minus_parent": point[arm],
                            "paired_ci95_lower": float(np.quantile(dist, 0.025)),
                            "paired_ci95_upper": float(np.quantile(dist, 0.975)),
                            "bootstrap_standard_error": bootstrap_se[arm],
                            "max_t_critical_value": max_t_quantile,
                            "simultaneous_ci95_upper": simultaneous_upper,
                            "point_improved": point[arm] < 0,
                            "simultaneous_improved": simultaneous_upper < 0,
                            "simultaneous_noninferior_0p005": simultaneous_upper < 0.005,
                        }
                    )
                    distributions.append(
                        pd.DataFrame(
                            {
                                "model_id": model_id,
                                "layer": layer,
                                "metric": metric,
                                "candidate": arm,
                                "replicate": np.arange(BOOT_REPS, dtype=int),
                                "delta_rmse_log1p": dist,
                            }
                        )
                    )
    return pd.DataFrame(rows), pd.concat(distributions, ignore_index=True)


def overall_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, layer, arm), frame in predictions.groupby(["model_id", "layer", "arm"], observed=True):
        obs = frame["tn_mg_l"].to_numpy(float)
        pred = frame["pred_tn_mg_l"].to_numpy(float)
        denominator = float(np.sum(np.square(obs - np.mean(obs))))
        corr = float(np.corrcoef(obs, pred)[0, 1]) if np.std(pred) > 0 else math.nan
        rows.append(
            {
                "model_id": model_id,
                "layer": layer,
                "arm": arm,
                "n": len(frame),
                "rmse_mg_l": float(np.sqrt(np.mean(np.square(pred - obs)))),
                "mae_mg_l": float(np.mean(np.abs(pred - obs))),
                "nse_mg_l": float(1.0 - np.sum(np.square(pred - obs)) / denominator),
                "pearson_r2_mg_l": corr * corr,
                "pbias_percent": float(100.0 * np.sum(pred - obs) / np.sum(obs)),
                "station_macro_anomaly_rmse_log1p": float(station_block_values(frame, True).mean()),
                "station_macro_absolute_rmse_log1p": float(station_block_values(frame, False).mean()),
                "tree_macro_absolute_rmse_log1p": float(tree_block_values(frame).mean()),
            }
        )
    return pd.DataFrame(rows)


def synthetic_design() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    h19 = load_module(H_SHARED_PATH, "f1_synthetic_h19")
    shared = h19.parent_shared()
    obs = h19.development_observations()
    router = h19.build_router("S0_mu_096m", shared)
    structures = pd.read_parquet(TEST / "20260820_19" / "cache" / "temporal" / "S0_mu_096m__structures.parquet")
    vf = float(structures.loc[structures["fold_id"].eq("F4") & structures["structure"].eq("H1_GLOBAL") & structures["evaluation"].eq("temporal"), "v0_m_per_day"].iloc[0])
    frame = add_fold_q_anomaly(router.frame(obs, np.full(len(router.reach_ids), vf)), 2016, 2020)
    train = frame.loc[frame["year"].between(2016, 2020)].copy()
    test = frame.loc[frame["year"].eq(2021)].copy()
    fit = gaussian_fit(train, "P1", False, shared)
    base_train = predict(train, fit, "P1")
    base_test = predict(test, fit, "P1")
    quick = train["routed_quick_tn_kg_n"].to_numpy(float)
    gw = train["routed_gw_tn_kg_n"].to_numpy(float)
    fraction = quick / np.maximum(quick + gw, 1e-30)
    design = np.column_stack([fraction, train["cq_low"], train["cq_high"]])
    design = (design - design.mean(axis=0)) / np.maximum(design.std(axis=0), 1e-12)
    singular = np.linalg.svd(design, compute_uv=False)
    correlation = np.corrcoef(design, rowvar=False)
    audit = {
        "representative_model": "S0_mu_096m",
        "representative_fold": "F4",
        "condition_number": float(singular.max() / singular.min()),
        "maximum_absolute_offdiagonal_correlation": float(np.max(np.abs(correlation - np.eye(correlation.shape[0])))),
        "train_rows": len(train),
        "test_rows": len(test),
    }
    return base_train, base_test, audit


def synthetic_one(
    rep: int,
    alternative: bool,
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    shared: ModuleType,
) -> dict[str, object]:
    # Synthetic recovery is intentionally P1-only: it asks whether the global
    # F1 mean-layer addition is distinguishable before station intercepts can
    # absorb it.
    rng = np.random.default_rng(SEED + rep + (1_000_000 if alternative else 0))
    train = train_base.copy()
    test = test_base.copy()
    beta = np.array([-0.35, 0.35]) if alternative else np.zeros(2)
    for frame in (train, test):
        mean_log = np.log1p(frame["pred_tn_mg_l"].to_numpy(float))
        mean_log += beta[0] * frame["cq_low"].to_numpy(float) + beta[1] * frame["cq_high"].to_numpy(float)
        noise = rng.standard_t(df=NU, size=len(frame)) * 0.14 if alternative else rng.normal(0.0, 0.14, len(frame))
        frame["tn_mg_l"] = np.maximum(np.expm1(mean_log + noise), 0.0)
    # Reuse the model component columns.  Fit the two Gaussian arms; the
    # Student-t material alternative is represented by the t noise, while the
    # upgrade decision is based on the registered hinge mean effect.
    parent_fit = gaussian_fit(train, "P1", False, shared)
    hinge_fit = gaussian_fit(train, "P1", True, shared)
    parent_pred = predict(test, parent_fit, "P1")
    hinge_pred = predict(test, hinge_fit, "P1")
    parent_values = station_block_values(parent_pred, True)
    hinge_values = station_block_values(hinge_pred, True).reindex(parent_values.index)
    delta = hinge_values.to_numpy(float) - parent_values.to_numpy(float)
    # One-sided paired t upper bound is used only inside the known-truth
    # preflight; production uses the registered paired max-T bootstrap.
    se = float(np.std(delta, ddof=1) / np.sqrt(len(delta)))
    upper = float(np.mean(delta) + 1.96 * se)
    return {
        "replicate": rep,
        "scenario": "material_alternative" if alternative else "null",
        "delta_station_macro_anomaly_rmse": float(np.mean(delta)),
        "ci95_upper": upper,
        "upgrade_called": upper < 0,
        "beta_low_hat": float(hinge_fit["beta"][0]),
        "beta_high_hat": float(hinge_fit["beta"][1]),
        "optimizer_success": bool(parent_fit["success"] and hinge_fit["success"]),
    }


def synthetic_preflight() -> tuple[pd.DataFrame, dict[str, object]]:
    train, test, design = synthetic_design()
    shared = load_module(H_SHARED_PATH, "f1_synthetic_fit_shared").parent_shared()
    rows = []
    # Keeping this serial avoids nested process pools on Windows and makes the
    # known-truth run deterministic.
    for alternative in (False, True):
        for rep in range(SYNTH_REPS):
            rows.append(synthetic_one(rep, alternative, train, test, shared))
    frame = pd.DataFrame(rows)
    null = frame.loc[frame["scenario"].eq("null")]
    alt = frame.loc[frame["scenario"].eq("material_alternative")]
    false_rate = float(null["upgrade_called"].mean())
    power = float(alt["upgrade_called"].mean())
    report = {
        "status": "PASS"
        if false_rate <= 0.05
        and power >= 0.80
        and design["condition_number"] < 30
        and design["maximum_absolute_offdiagonal_correlation"] < 0.9
        and frame["optimizer_success"].all()
        else "FAIL",
        "null_replicates": int(len(null)),
        "material_alternative_replicates": int(len(alt)),
        "false_upgrade_rate": false_rate,
        "power": power,
        "design": design,
        "exact_parent_nesting": True,
        "TN_2022_values_read": False,
    }
    return frame, report


def decision(gates: pd.DataFrame, parameters: pd.DataFrame, synthetic: dict[str, object]) -> dict[str, object]:
    candidate_rows = []
    for arm in ARMS[1:]:
        p1_anom = gates.loc[
            gates["candidate"].eq(arm)
            & gates["layer"].eq("P1")
            & gates["metric"].eq("station_anomaly_rmse_log1p")
        ]
        p1_abs = gates.loc[
            gates["candidate"].eq(arm)
            & gates["layer"].eq("P1")
            & gates["metric"].eq("station_absolute_rmse_log1p")
        ]
        p2_abs = gates.loc[
            gates["candidate"].eq(arm)
            & gates["layer"].eq("P2")
            & gates["metric"].eq("station_absolute_rmse_log1p")
        ]
        anomaly_point = int(p1_anom["point_improved"].sum())
        anomaly_sim = int(p1_anom["simultaneous_improved"].sum())
        abs_noninf = int(p1_abs["simultaneous_noninferior_0p005"].sum())
        p2_noninf = int(p2_abs["simultaneous_noninferior_0p005"].sum())
        par = parameters.loc[parameters["arm"].eq(arm)]
        confounded = bool((~par["success"]).any() or par["eta_boundary"].groupby(par["model_id"]).sum().ge(2).any() or par["beta_boundary"].groupby(par["model_id"]).sum().ge(2).any())
        mechanistic = anomaly_sim >= 10 and anomaly_point >= 10 and abs_noninf >= 10 and p2_noninf >= 10 and not confounded
        p2_only = p2_noninf >= 10 and anomaly_sim < 10 and not confounded
        candidate_rows.append(
            {
                "candidate": arm,
                "P1_anomaly_simultaneously_improved_models": anomaly_sim,
                "P1_anomaly_point_improved_models": anomaly_point,
                "P1_absolute_noninferior_models": abs_noninf,
                "P2_absolute_noninferior_models": p2_noninf,
                "confounded": confounded,
                "mechanistic_gate_pass": mechanistic,
                "prediction_only_gate_pass": p2_only,
            }
        )
    table = pd.DataFrame(candidate_rows)
    table.to_parquet(OUT / "f1_candidate_decision_matrix.parquet", index=False)
    if synthetic["status"] != "PASS":
        status = "IDENTIFIABILITY_BLOCKED"
    elif table["mechanistic_gate_pass"].any():
        status = "F1_MEAN_LAYER_TEMPORAL_UPGRADE_SUPPORTED"
    elif table["prediction_only_gate_pass"].any():
        status = "MONITORED_STATION_PREDICTION_UPGRADE_ONLY"
    else:
        status = "NO_F1_UPGRADE_SUPPORTED"
    return {
        "status": status,
        "candidate_matrix": candidate_rows,
        "synthetic_gate": synthetic["status"],
        "P2_is_process_evidence": False,
        "TN_2022_values_read": False,
    }


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = write_f1_lock()

    synthetic_path = OUT / "f1_synthetic_recovery.parquet"
    synthetic_report_path = REPORTS / "f1_synthetic_identifiability.json"
    if synthetic_path.exists() and synthetic_report_path.exists():
        synthetic_rows = pd.read_parquet(synthetic_path)
        synthetic_report = json.loads(synthetic_report_path.read_text(encoding="utf-8"))
    else:
        synthetic_rows, synthetic_report = synthetic_preflight()
        synthetic_rows.to_parquet(synthetic_path, index=False)
        dump_json(synthetic_report_path, synthetic_report)
    if synthetic_report["status"] != "PASS":
        decision_report = {
            "status": "IDENTIFIABILITY_BLOCKED",
            "reason": "F1 synthetic false-upgrade/power/design gate failed before observed candidate fitting",
            "synthetic": synthetic_report,
            "observed_candidate_fit_performed": False,
            "TN_2022_values_read": False,
        }
        dump_json(REPORTS / "f1_decision.json", decision_report)
        print(json.dumps(decision_report, ensure_ascii=False, indent=2))
        return

    paths = [
        (
            str(CACHE / f"{model_id}__predictions.parquet"),
            str(CACHE / f"{model_id}__parameters.parquet"),
        )
        for model_id in FORMAL_MODELS
    ]
    if not all(Path(pred).exists() and Path(par).exists() for pred, par in paths):
        paths = []
        workers = min(6, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(model_worker, model_id): model_id for model_id in FORMAL_MODELS}
            for future in as_completed(futures):
                paths.append(future.result())
    predictions = pd.concat([pd.read_parquet(path[0]) for path in paths], ignore_index=True)
    parameters = pd.concat([pd.read_parquet(path[1]) for path in paths], ignore_index=True)
    predictions.to_parquet(OUT / "f1_temporal_oof_predictions.parquet", index=False)
    parameters.to_parquet(OUT / "f1_fold_parameters.parquet", index=False)
    metrics = overall_metrics(predictions)
    metrics.to_parquet(OUT / "f1_performance_metrics.parquet", index=False)
    gates, distributions = paired_distributions(predictions)
    gates.to_parquet(OUT / "f1_paired_simultaneous_gates.parquet", index=False)
    distributions.to_parquet(OUT / "f1_paired_bootstrap_distributions.parquet", index=False)
    result = decision(gates, parameters, synthetic_report)
    result["input_lock_sha256"] = sha256(LOCKS / "f1_pre_observed_candidate_input_lock.json")
    result["observed_candidate_fit_performed"] = True
    dump_json(REPORTS / "f1_decision.json", result)
    print(json.dumps({"lock": lock["aggregate_sha256"], "decision": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
