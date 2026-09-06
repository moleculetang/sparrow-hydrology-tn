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


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260824_6"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
CACHE = HERE / "cache"

CONTRACT = HERE / "experiment_contract.json"
CONTINUATION = HERE / "program_continuation.json"
PARENT_MANIFEST = TEST / "20260824_4" / "program_manifest.json"
STAGE0 = TEST / "20260824_4" / "reports" / "stage0_repair_completion_audit.json"
F2_DECISION = TEST / "20260824_5" / "reports" / "f2_decision.json"
H_SHARED_PATH = TEST / "20260820_19" / "scripts" / "hierarchical19_shared.py"
F2_SHARED_PATH = TEST / "20260824_5" / "scripts" / "run_f2_terrestrial.py"
EXPOSURE = TEST / "20260820_18" / "outputs" / "reach_month_aquatic_uptake_exposure_2006_2021.parquet"
OBS = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLDS = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
DOMAIN = TEST / "20260820_19" / "outputs" / "observation_domain_registry.parquet"

MUS = (12, 36, 60, 96, 144, 240)
FORMAL_MODELS = tuple(sorted(
    [f"S0_mu_{mu:03d}m" for mu in MUS]
    + [f"S1_tau_012m_mu_{mu:03d}m" for mu in MUS]
))
CANDIDATES = (
    "TRUE_RESIDENCE_TIME_ATTENUATION",
    "H1_CONCENTRATION_LIMITED_KN_GRID",
)
KN_GRID = (0.0, 0.5, 1.0, 2.0)
TIE_TOL = 1e-8
WATER_EPS = 1e-12
SYNTH_REPS = 500
SEED = 2026082406


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


def write_input_lock() -> dict[str, object]:
    paths = [
        CONTRACT, CONTINUATION, PARENT_MANIFEST, STAGE0, F2_DECISION,
        Path(__file__), H_SHARED_PATH, F2_SHARED_PATH, EXPOSURE, OBS, FOLDS, DOMAIN,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    hashes = {str(path): sha256(path) for path in paths}
    lock = {
        "experiment_id": "20260824_6",
        "created_before_observed_candidate_fitting": True,
        "files": hashes,
        "aggregate_sha256": hashlib.sha256(
            "\n".join(f"{k}|{v}" for k, v in sorted(hashes.items())).encode("utf-8")
        ).hexdigest(),
        "development_TN_years_allowed": [2016, 2017, 2018, 2019, 2020, 2021],
        "TN_2022_values_read": False,
    }
    dump_json(LOCKS / "f3_pre_observed_candidate_input_lock.json", lock)
    return lock


class SaturationRouter:
    def __init__(self, base: object, kn_mg_l: float):
        self.model_id = f"{base.model_id}__KN_{kn_mg_l:g}"
        self.kn_mg_l = float(kn_mg_l)
        self.reach_ids = base.reach_ids.copy()
        self.years = base.years.copy()
        self.months = base.months.copy()
        self.local_q = base.local_q.copy()
        self.local_g = base.local_g.copy()
        self.h_full = base.h_full.copy()
        self.h_mid = base.h_mid.copy()
        self.water = base.water.copy()
        self.x = base.x.copy()
        self.terminal_by_reach = base.terminal_by_reach.copy()
        self.order_index = list(base.order_index)
        self.downstream_index = dict(base.downstream_index)
        self.reach_lookup = dict(base.reach_lookup)
        self.time_lookup = dict(base.time_lookup)
        self._route_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def route(self, vf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(vf, dtype=float)
        if vector.shape != (len(self.reach_ids),):
            raise ValueError("vf vector shape mismatch")
        if np.any(vector < 0) or np.any(vector > 0.5 + 1e-12):
            raise ValueError("effective vf outside registered bounds")
        key = np.round(vector, 10).tobytes()
        if key in self._route_cache:
            return self._route_cache[key]
        uq = np.zeros_like(self.local_q)
        ug = np.zeros_like(self.local_g)
        oq = np.zeros_like(self.local_q)
        og = np.zeros_like(self.local_g)
        for i in self.order_index:
            incoming_total = uq[:, i] + ug[:, i] + self.local_q[:, i] + self.local_g[:, i]
            concentration = np.divide(
                incoming_total * 1000.0,
                self.water[:, i],
                out=np.zeros_like(incoming_total),
                where=self.water[:, i] > WATER_EPS,
            )
            if self.kn_mg_l == 0.0:
                factor = np.ones_like(concentration)
            else:
                factor = np.divide(
                    concentration, concentration + self.kn_mg_l,
                    out=np.zeros_like(concentration),
                    where=(concentration + self.kn_mg_l) > 0,
                )
            sf = np.exp(-self.h_full[:, i] * vector[i] * factor)
            sm = np.exp(-self.h_mid[:, i] * vector[i] * factor)
            oq[:, i] = uq[:, i] * sf + self.local_q[:, i] * sm
            og[:, i] = ug[:, i] * sf + self.local_g[:, i] * sm
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
            dtype=int, count=len(observations),
        )
        out = observations.copy()
        out["routed_quick_tn_kg_n"] = oq[tidx, ridx]
        out["routed_gw_tn_kg_n"] = og[tidx, ridx]
        out["routed_water_volume_m3"] = self.water[tidx, ridx]
        out["terminal_tree_id"] = self.terminal_by_reach[ridx]
        # H1 fitting only needs the common spatial columns to be present.
        return out


def residence_router(h: ModuleType, base: object) -> object:
    exposure = pd.read_parquet(EXPOSURE).loc[lambda d: d.year.between(2016, 2021)].sort_values(
        ["year", "month", "reach_id"]
    )
    keys = exposure[["reach_id", "year", "month"]].reset_index(drop=True)
    expected = pd.DataFrame({
        "reach_id": np.tile(base.reach_ids, len(base.years)),
        "year": np.repeat(base.years, len(base.reach_ids)),
        "month": np.repeat(base.months, len(base.reach_ids)),
    })
    if not keys.equals(expected):
        raise RuntimeError("STOP_RESIDENCE_EXPOSURE_KEYS")
    shape = base.local_q.shape
    return h.HydraulicRouter(
        model_id=f"{base.model_id}__RESIDENCE",
        reach_ids=base.reach_ids.copy(), years=base.years.copy(), months=base.months.copy(),
        local_q=base.local_q.copy(), local_g=base.local_g.copy(),
        h_full=exposure.travel_time_full_days.to_numpy(float).reshape(shape),
        h_mid=exposure.travel_time_midpoint_to_outlet_days.to_numpy(float).reshape(shape),
        water=base.water.copy(), x=base.x.copy(), terminal_by_reach=base.terminal_by_reach.copy(),
        order_index=list(base.order_index), downstream_index=dict(base.downstream_index),
    )


def endpoint_and_range_audit() -> pd.DataFrame:
    h = load_module(H_SHARED_PATH, "f3_endpoint_h")
    shared = h.parent_shared()
    rows = []
    for model_id in FORMAL_MODELS:
        base = h.build_router(model_id, shared)
        endpoint = SaturationRouter(base, 0.0)
        for vf in (0.0, 0.05, 0.13, 0.30):
            vector = np.full(len(base.reach_ids), vf)
            bq, bg = base.route(vector)
            eq, eg = endpoint.route(vector)
            scale = max(float(np.max(np.abs(bq))), float(np.max(np.abs(bg))), 1.0)
            rows.append({
                "model_id": model_id, "v_f_m_per_day": vf,
                "max_abs_quick_error_kg_n": float(np.max(np.abs(eq - bq))),
                "max_abs_gw_error_kg_n": float(np.max(np.abs(eg - bg))),
                "relative_error": max(float(np.max(np.abs(eq - bq))), float(np.max(np.abs(eg - bg)))) / scale,
                "minimum_output_kg_n": float(min(eq.min(), eg.min())),
                "maximum_output_to_zero_attenuation_ratio": float(
                    max(np.max(eq / np.maximum(base.route(np.zeros(len(base.reach_ids)))[0], 1e-30)),
                        np.max(eg / np.maximum(base.route(np.zeros(len(base.reach_ids)))[1], 1e-30)))
                ),
            })
    frame = pd.DataFrame(rows)
    frame["endpoint_pass"] = frame.relative_error <= 1e-10
    frame["nonnegative_pass"] = frame.minimum_output_kg_n >= -1e-8
    frame["no_amplification_pass"] = frame.maximum_output_to_zero_attenuation_ratio <= 1.0 + 1e-10
    frame.to_parquet(OUT / "f3_aquatic_numerical_audit.parquet", index=False)
    if not frame[["endpoint_pass", "nonnegative_pass", "no_amplification_pass"]].all().all():
        raise RuntimeError("STOP_F3_AQUATIC_NUMERICAL_AUDIT")
    return frame


def route_log(router: object, coefficient: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vector = np.full(len(router.reach_ids), coefficient, dtype=float)
    rq, rg = router.route(vector)
    concentration = np.divide(
        (0.8 * rq + 0.8 * rg) * 1000.0,
        router.water, out=np.zeros_like(router.water), where=router.water > WATER_EPS,
    )
    return np.log1p(np.maximum(concentration, 0.0)), rq, rg


def synthetic_design() -> tuple[pd.DataFrame, dict[str, object]]:
    h = load_module(H_SHARED_PATH, "f3_synth_h")
    shared = h.parent_shared()
    base = h.build_router("S0_mu_096m", shared)
    residence = residence_router(h, base)
    saturation = SaturationRouter(base, 1.0)
    base_log, rq, rg = route_log(base, 0.13)
    candidate_logs = {
        CANDIDATES[0]: route_log(residence, 0.02)[0],
        CANDIDATES[1]: route_log(saturation, 0.13)[0],
    }
    keys = pd.read_parquet(
        OBS, columns=["station_key", "reach_id", "year", "month"],
        filters=[("year", "<=", 2021)],
    )
    domain = pd.read_parquet(DOMAIN, columns=["station_key", "primary_river_domain"])
    keys = keys.merge(domain, on="station_key", validate="many_to_one")
    keys = keys.loc[keys.primary_river_domain & keys.year.between(2016, 2021)].drop(columns="primary_river_domain")
    ridx = keys.reach_id.astype(int).map(base.reach_lookup).to_numpy(int)
    tidx = np.fromiter(
        (base.time_lookup[(int(y), int(m))] for y, m in keys[["year", "month"]].itertuples(index=False)),
        dtype=int, count=len(keys),
    )
    frame = keys.copy()
    frame["base_log"] = base_log[tidx, ridx]
    for candidate, values in candidate_logs.items():
        frame[candidate] = values[tidx, ridx] - frame.base_log.to_numpy(float)
    quick_fraction = np.divide(rq, rq + rg, out=np.zeros_like(rq), where=(rq + rg) > 1e-30)
    frame["quick_fraction"] = quick_fraction[tidx, ridx]
    audit = {}
    for candidate in CANDIDATES:
        design = frame[["quick_fraction", candidate]].to_numpy(float)
        design = (design - design.mean(axis=0)) / np.maximum(design.std(axis=0), 1e-12)
        singular = np.linalg.svd(design, compute_uv=False)
        corr = np.corrcoef(design, rowvar=False)
        audit[candidate] = {
            "condition_number": float(singular.max() / singular.min()),
            "maximum_absolute_offdiagonal_correlation": float(abs(corr[0, 1])),
            "fingerprint_sd_log1p": float(frame[candidate].std(ddof=0)),
        }
    return frame, audit


def synthetic_recovery() -> tuple[pd.DataFrame, dict[str, object]]:
    frame, design = synthetic_design()
    train = frame.loc[frame.year.between(2016, 2020)].copy()
    test = frame.loc[frame.year.eq(2021)].copy()
    rows = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        x_train = train[candidate].to_numpy(float)
        x_test = test[candidate].to_numpy(float)
        centered = x_train - x_train.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= 1e-12:
            raise RuntimeError(f"STOP_SYNTH_ZERO_FINGERPRINT:{candidate}")
        for scenario in ("null", "material_alternative"):
            theta_true = 0.0 if scenario == "null" else 1.0
            for rep in range(SYNTH_REPS):
                rng = np.random.default_rng(SEED + candidate_index * 2_000_000 + rep + (1_000_000 if theta_true else 0))
                y_train = train.base_log.to_numpy(float) + theta_true * x_train + rng.normal(0.0, 0.14, len(train))
                y_residual = y_train - train.base_log.to_numpy(float)
                theta_hat = float(np.clip(
                    np.dot(centered, y_residual - y_residual.mean()) / denominator, 0.0, 1.0
                ))
                intercept = float(np.mean(y_residual - theta_hat * x_train))
                y_test = test.base_log.to_numpy(float) + theta_true * x_test + rng.normal(0.0, 0.14, len(test))
                parent_pred = test.base_log.to_numpy(float) + intercept
                candidate_pred = parent_pred + theta_hat * x_test
                work = test[["station_key"]].copy()
                work["obs"] = y_test; work["parent"] = parent_pred; work["candidate"] = candidate_pred
                block = []
                for _, group in work.groupby("station_key", observed=True):
                    obs = group.obs.to_numpy(float) - group.obs.mean()
                    pa = group.parent.to_numpy(float) - group.parent.mean()
                    ca = group.candidate.to_numpy(float) - group.candidate.mean()
                    block.append(float(
                        np.sqrt(np.mean(np.square(ca - obs))) - np.sqrt(np.mean(np.square(pa - obs)))
                    ))
                delta = np.asarray(block)
                upper = float(delta.mean() + 1.96 * np.std(delta, ddof=1) / np.sqrt(len(delta)))
                rows.append({
                    "candidate": candidate, "scenario": scenario, "replicate": rep,
                    "theta_true": theta_true, "theta_hat": theta_hat,
                    "delta_station_anomaly_rmse": float(delta.mean()),
                    "ci95_upper": upper, "upgrade_called": bool(upper < 0),
                })
    recovery = pd.DataFrame(rows)
    candidate_reports = []
    for candidate in CANDIDATES:
        part = recovery.loc[recovery.candidate.eq(candidate)]
        false_rate = float(part.loc[part.scenario.eq("null"), "upgrade_called"].mean())
        power = float(part.loc[part.scenario.eq("material_alternative"), "upgrade_called"].mean())
        d = design[candidate]
        passed = bool(
            false_rate <= 0.05 and power >= 0.80
            and d["condition_number"] < 30
            and d["maximum_absolute_offdiagonal_correlation"] < 0.9
        )
        candidate_reports.append({
            "candidate": candidate, "false_upgrade_rate": false_rate,
            "power": power, "design": d, "pass": passed,
        })
    report = {
        "status": "PASS" if all(row["pass"] for row in candidate_reports) else "FAIL",
        "null_replicates_per_candidate": SYNTH_REPS,
        "material_alternative_replicates_per_candidate": SYNTH_REPS,
        "candidate_reports": candidate_reports,
        "residence_parent_bridge_synthetic_only": True,
        "saturation_exact_parent_endpoint": "K_N=0",
        "TN_2022_values_read": False,
    }
    recovery.to_parquet(OUT / "f3_synthetic_recovery.parquet", index=False)
    dump_json(REPORTS / "f3_synthetic_identifiability.json", report)
    return recovery, report


def parameter_row(
    model_id: str, fold_id: str, candidate: str, layer: str,
    fit: dict[str, object], kn: float = math.nan, selected: bool = True,
) -> dict[str, object]:
    parameters = np.asarray(fit.get("parameters", []), dtype=float)
    vf = np.asarray(fit.get("vf", []), dtype=float)
    eta = np.asarray(fit["eta"], dtype=float)
    diagnostic = fit["diagnostic"]
    coefficient = float(parameters[0]) if len(parameters) else math.nan
    return {
        "model_id": model_id, "fold_id": fold_id, "candidate": candidate,
        "layer": layer, "selected": selected, "K_N_mg_L": kn,
        "aquatic_coefficient": coefficient,
        "coefficient_semantics": "k_day" if candidate == CANDIDATES[0] else "v_f_m_per_day",
        "training_objective": float(fit.get("objective", math.nan)),
        "training_data_objective": float(fit.get("data_objective", math.nan)),
        "eta_quick": float(eta[0]), "eta_gw": float(eta[1]),
        "eta_boundary": bool(diagnostic["eta_boundary"]),
        "coefficient_boundary": bool(len(vf) and vf.max() >= 0.49),
        "optimizer_success": bool(diagnostic["success"] and fit.get("outer_success", True)),
        "outer_nfev": int(fit.get("outer_nfev", 0)),
    }


def fit_model(model_id: str, active_candidates: tuple[str, ...]) -> tuple[str, str, str, str]:
    h = load_module(H_SHARED_PATH, f"f3_h_{model_id}")
    shared = h.parent_shared()
    observations = h.development_observations()
    folds = h.fold_registry()
    base = h.build_router(model_id, shared)
    residence = residence_router(h, base) if CANDIDATES[0] in active_candidates else None
    saturation = (
        {kn: SaturationRouter(base, kn) for kn in KN_GRID}
        if CANDIDATES[1] in active_candidates else {}
    )
    pred_rows = []; par_rows = []; grid_rows = []; selection_rows = []
    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)
        train_obs = observations.loc[observations.year.between(
            int(fold.train_start_year), int(fold.train_end_year)
        )].copy()
        test_obs = observations.loc[observations.year.eq(int(fold.evaluation_year))].copy()
        parent_fit = h.fit_structure(base, train_obs, "H1_GLOBAL", shared)
        selected = {"GAUSSIAN_PROCESS_PARENT": (base, parent_fit, math.nan)}
        if CANDIDATES[0] in active_candidates:
            if residence is None:
                raise RuntimeError("STOP_RESIDENCE_ROUTER_MISSING")
            residence_fit = h.fit_structure(residence, train_obs, "H1_GLOBAL", shared)
            selection_rows.append({
                "model_id": model_id, "fold_id": fold_id, "evaluation_year": int(fold.evaluation_year),
                "candidate": CANDIDATES[0], "K_N_mg_L": math.nan,
                "training_objective": float(residence_fit["objective"]),
                "parent_training_objective": float(parent_fit["objective"]),
            })
            selected[CANDIDATES[0]] = (residence, residence_fit, math.nan)
        if CANDIDATES[1] in active_candidates:
            kn_fits = {
                kn: h.fit_structure(router, train_obs, "H1_GLOBAL", shared)
                for kn, router in saturation.items()
            }
            best = min(float(fit["objective"]) for fit in kn_fits.values())
            selected_kn = min(kn for kn in KN_GRID if float(kn_fits[kn]["objective"]) <= best + TIE_TOL)
            for kn, fit in kn_fits.items():
                grid_rows.append(parameter_row(
                    model_id, fold_id, CANDIDATES[1], "P1", fit, kn, selected=kn == selected_kn
                ))
            selection_rows.append({
                "model_id": model_id, "fold_id": fold_id, "evaluation_year": int(fold.evaluation_year),
                "candidate": CANDIDATES[1], "K_N_mg_L": selected_kn,
                "training_objective": float(kn_fits[selected_kn]["objective"]),
                "parent_training_objective": float(parent_fit["objective"]),
            })
            selected[CANDIDATES[1]] = (saturation[selected_kn], kn_fits[selected_kn], selected_kn)
        for candidate, (router, fit, kn) in selected.items():
            vf = np.asarray(fit["vf"], dtype=float)
            train_frame = router.frame(train_obs, vf)
            test_frame = router.frame(test_obs, vf)
            p1 = shared.predict_layer(test_frame, "P1", np.asarray(fit["eta"]), fit["effects"])
            p1["candidate"] = candidate; p1["model_id"] = model_id; p1["fold_id"] = fold_id
            p1["evaluation_year"] = int(fold.evaluation_year); p1["K_N_mg_L"] = kn
            pred_rows.append(p1)
            par_rows.append(parameter_row(model_id, fold_id, candidate, "P1", fit, kn))
            p2_fit = h.fit_selected_readout(train_frame, "P2", shared)
            p2 = h.predict_selected(test_frame, "P2", p2_fit, shared)
            p2["candidate"] = candidate; p2["model_id"] = model_id; p2["fold_id"] = fold_id
            p2["evaluation_year"] = int(fold.evaluation_year); p2["K_N_mg_L"] = kn
            pred_rows.append(p2)
            proxy = {
                "eta": p2_fit["eta"], "diagnostic": p2_fit["diagnostic"],
                "parameters": fit["parameters"], "vf": fit["vf"],
                "objective": math.nan, "data_objective": math.nan,
                "outer_success": True, "outer_nfev": 0,
            }
            par_rows.append(parameter_row(model_id, fold_id, candidate, "P2", proxy, kn))
    model_cache = CACHE / "formal"; model_cache.mkdir(parents=True, exist_ok=True)
    paths = (
        model_cache / f"{model_id}__predictions.parquet",
        model_cache / f"{model_id}__parameters.parquet",
        model_cache / f"{model_id}__grid_scores.parquet",
        model_cache / f"{model_id}__selections.parquet",
    )
    pd.concat(pred_rows, ignore_index=True).to_parquet(paths[0], index=False)
    pd.DataFrame(par_rows).to_parquet(paths[1], index=False)
    pd.DataFrame(grid_rows, columns=[
        "model_id", "fold_id", "candidate", "layer", "selected", "K_N_mg_L",
        "aquatic_coefficient", "coefficient_semantics", "training_objective",
        "training_data_objective", "eta_quick", "eta_gw", "eta_boundary",
        "coefficient_boundary", "optimizer_success", "outer_nfev",
    ]).to_parquet(paths[2], index=False)
    pd.DataFrame(selection_rows).to_parquet(paths[3], index=False)
    return tuple(map(str, paths))


def run_formal_oof(active_candidates: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_cache = CACHE / "formal"
    paths = [(
        model_cache / f"{model_id}__predictions.parquet",
        model_cache / f"{model_id}__parameters.parquet",
        model_cache / f"{model_id}__grid_scores.parquet",
        model_cache / f"{model_id}__selections.parquet",
    ) for model_id in FORMAL_MODELS]
    if not all(all(path.exists() for path in group) for group in paths):
        completed = []
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
            futures = {pool.submit(fit_model, model_id, active_candidates): model_id for model_id in FORMAL_MODELS}
            for future in as_completed(futures):
                completed.append(future.result())
                print(json.dumps({
                    "f3_model_complete": futures[future], "completed": len(completed), "total": len(FORMAL_MODELS)
                }), flush=True)
        paths = [tuple(Path(value) for value in group) for group in completed]
    predictions = pd.concat([pd.read_parquet(group[0]) for group in paths], ignore_index=True)
    parameters = pd.concat([pd.read_parquet(group[1]) for group in paths], ignore_index=True)
    grid_scores = pd.concat([pd.read_parquet(group[2]) for group in paths], ignore_index=True)
    selections = pd.concat([pd.read_parquet(group[3]) for group in paths], ignore_index=True)
    predictions.to_parquet(OUT / "f3_temporal_oof_predictions.parquet", index=False)
    parameters.to_parquet(OUT / "f3_fold_parameters.parquet", index=False)
    grid_scores.to_parquet(OUT / "f3_candidate_grid_scores.parquet", index=False)
    selections.to_parquet(OUT / "f3_fold_selections.parquet", index=False)
    return predictions, parameters, grid_scores, selections


def evaluate_formal(predictions: pd.DataFrame, active_candidates: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    helper = load_module(F2_SHARED_PATH, "f3_eval_helper")
    helper.OUT = OUT
    helper.REPORTS = REPORTS
    helper.CANDIDATES = active_candidates
    helper.FORMAL_MODELS = FORMAL_MODELS
    helper.SEED = SEED
    gates, distributions = helper.paired_simultaneous_gates(predictions)
    metrics = helper.overall_metrics(predictions)
    reproduction = helper.parent_reproduction(predictions)
    # Rename helper-generated F2 filenames to this experiment's authoritative names.
    renames = {
        "f2_paired_simultaneous_gates.parquet": "f3_paired_simultaneous_gates.parquet",
        "f2_paired_bootstrap_distributions.parquet": "f3_paired_bootstrap_distributions.parquet",
        "f2_performance_metrics.parquet": "f3_performance_metrics.parquet",
        "f2_parent_reproduction.parquet": "f3_parent_reproduction.parquet",
    }
    for source, target in renames.items():
        (OUT / source).replace(OUT / target)
    return gates, distributions, metrics, reproduction


def stability_table(selections: pd.DataFrame, parameters: pd.DataFrame, active_candidates: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for model_id in FORMAL_MODELS:
        for candidate in active_candidates:
            sel = selections.loc[selections.model_id.eq(model_id) & selections.candidate.eq(candidate)]
            par = parameters.loc[parameters.model_id.eq(model_id) & parameters.candidate.eq(candidate)]
            fit_confounded = bool(par.groupby("layer", observed=True).apply(
                lambda g: int((g.eta_boundary | g.coefficient_boundary | ~g.optimizer_success).sum()) >= 2,
                include_groups=False,
            ).any())
            if candidate == CANDIDATES[0]:
                coefficient = par.loc[par.layer.eq("P1"), "aquatic_coefficient"].to_numpy(float)
                stable = bool(np.all(coefficient > 0))
                boundary = int(np.sum(coefficient >= 0.49)) >= 2
                endpoint = bool(np.all(coefficient <= 1e-10))
                direction = "positive_k_day" if stable else "zero_or_unstable"
            else:
                values = sel.K_N_mg_L.to_numpy(float)
                counts = pd.Series(values).value_counts()
                modal = float(counts.index[0])
                stable = bool(int(counts.iloc[0]) >= 3 and modal > 0)
                boundary = int(np.sum(values == max(KN_GRID))) >= 2
                endpoint = bool(np.all(values == 0))
                direction = f"K_N={modal:g}_mg_L" if stable else "zero_or_unstable"
            rows.append({
                "model_id": model_id, "candidate": candidate,
                "parameter_stable": stable, "parent_endpoint_all_folds": endpoint,
                "candidate_boundary_confounded": boundary,
                "fit_or_readout_confounded": fit_confounded,
                "direction_label": direction,
            })
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "f3_parameter_stability.parquet", index=False)
    return frame


def decide(
    gates: pd.DataFrame, stability: pd.DataFrame, synthetic: dict[str, object],
    numerical: pd.DataFrame, active_candidates: tuple[str, ...],
) -> dict[str, object]:
    candidate_rows = []
    synth_map = {row["candidate"]: row for row in synthetic["candidate_reports"]}
    for candidate in CANDIDATES:
        if candidate not in active_candidates:
            candidate_rows.append({
                "candidate": candidate,
                "preflight_status": "IDENTIFIABILITY_BLOCKED",
                "P1_anomaly_point_improved": 0,
                "P1_anomaly_simultaneously_improved": 0,
                "P1_station_absolute_noninferior": 0,
                "P1_tree_absolute_noninferior": 0,
                "P2_station_absolute_noninferior": 0,
                "P2_tree_absolute_noninferior": 0,
                "parameter_stable_models": 0,
                "candidate_boundary_confounded_models": 0,
                "fit_or_readout_confounded_models": 0,
                "formal_gate_pass": False,
                "synthetic_power": synth_map[candidate]["power"],
            })
            continue
        def count(layer: str, metric: str, field: str) -> int:
            return int(gates.loc[
                gates.candidate.eq(candidate) & gates.layer.eq(layer) & gates.metric.eq(metric), field
            ].sum())
        counts = {
            "P1_anomaly_point_improved": count("P1", "station_anomaly_rmse_log1p", "point_improved"),
            "P1_anomaly_simultaneously_improved": count("P1", "station_anomaly_rmse_log1p", "simultaneous_improved"),
            "P1_station_absolute_noninferior": count("P1", "station_absolute_rmse_log1p", "simultaneous_noninferior_0p005"),
            "P1_tree_absolute_noninferior": count("P1", "tree_absolute_rmse_log1p", "simultaneous_noninferior_0p005"),
            "P2_station_absolute_noninferior": count("P2", "station_absolute_rmse_log1p", "simultaneous_noninferior_0p005"),
            "P2_tree_absolute_noninferior": count("P2", "tree_absolute_rmse_log1p", "simultaneous_noninferior_0p005"),
        }
        part = stability.loc[stability.candidate.eq(candidate)]
        stable_models = int(part.parameter_stable.sum())
        boundary_models = int(part.candidate_boundary_confounded.sum())
        fit_confounded_models = int(part.fit_or_readout_confounded.sum())
        formal_pass = bool(
            counts["P1_anomaly_point_improved"] >= 10
            and counts["P1_anomaly_simultaneously_improved"] >= 10
            and counts["P1_station_absolute_noninferior"] >= 10
            and counts["P1_tree_absolute_noninferior"] >= 10
            and counts["P2_station_absolute_noninferior"] >= 10
            and counts["P2_tree_absolute_noninferior"] >= 10
            and stable_models >= 10 and boundary_models <= 2 and fit_confounded_models <= 2
        )
        candidate_rows.append({
            "candidate": candidate, "preflight_status": "PASS", **counts,
            "parameter_stable_models": stable_models,
            "candidate_boundary_confounded_models": boundary_models,
            "fit_or_readout_confounded_models": fit_confounded_models,
            "formal_gate_pass": formal_pass,
        })
    matrix = pd.DataFrame(candidate_rows)
    matrix.to_parquet(OUT / "f3_candidate_decision_matrix.parquet", index=False)
    numeric_pass = bool(numerical[["endpoint_pass", "nonnegative_pass", "no_amplification_pass"]].all().all())
    if not active_candidates:
        status = "IDENTIFIABILITY_BLOCKED"
    elif not numeric_pass:
        status = "IMPLEMENTATION_BLOCKED"
    elif matrix.formal_gate_pass.any():
        status = "F3_AQUATIC_TEMPORAL_UPGRADE_SUPPORTED_PENDING_SPATIAL"
    elif any(row["P2_station_absolute_noninferior"] >= 10 for row in candidate_rows):
        status = "F3_PROCESS_NOT_SUPPORTED_PREDICTION_PRESERVED"
    else:
        status = "NO_REGISTERED_F3_UPGRADE_SUPPORTED"
    return {
        "status": status,
        "supported_candidates": matrix.loc[matrix.formal_gate_pass, "candidate"].tolist(),
        "candidate_matrix": candidate_rows,
        "synthetic_gate": synthetic["status"],
        "preflight_active_candidates": list(active_candidates),
        "preflight_blocked_candidates": [c for c in CANDIDATES if c not in active_candidates],
        "numeric_contract_pass": numeric_pass,
        "F1_hinge_was_not_in_parent_comparison": True,
        "temperature_used": False, "reservoir_module_used": False,
        "andreadis_reference_discharge_used": False,
        "TN_years_read": [2016, 2017, 2018, 2019, 2020, 2021],
        "TN_2022_values_read": False,
    }


def write_report(result: dict[str, object], metrics: pd.DataFrame, selections: pd.DataFrame, synthetic: dict[str, object]) -> None:
    mean = metrics.groupby(["layer", "candidate"], as_index=False).agg(
        rmse=("rmse_mg_l", "mean"), nse=("nse_mg_l", "mean"), r2=("pearson_r2_mg_l", "mean"),
        anomaly=("station_macro_anomaly_rmse_log1p", "mean"),
        absolute=("station_macro_absolute_rmse_log1p", "mean"),
        median_nse=("median_station_nse", "mean"), median_r2=("median_station_pearson_r2", "mean"),
    )
    lines = [
        "# F3 河道反应结构挑战", "", f"正式裁决：`{result['status']}`。", "",
        "## 结构", "",
        "`TRUE_RESIDENCE_TIME_ATTENUATION`只使用Q72流量与Andreadis宽/深/长计算的旅行时间；WQD reference discharge没有进入。`H1_CONCENTRATION_LIMITED_KN_GRID`用进入河段的quick+GW总N和Q72水量计算浓度，在H1指数中加入 `C/(C+K_N)`，并在`K_N=0`时精确恢复H1。温度与水库均关闭。", "",
        "## 合成可识别性", "",
        f"状态：`{synthetic['status']}`。", "",
        "| Candidate | false upgrade | power | condition number | max |r| |", "|---|---:|---:|---:|---:|",
    ]
    for row in synthetic["candidate_reports"]:
        lines.append(f"| {row['candidate']} | {row['false_upgrade_rate']:.3f} | {row['power']:.3f} | {row['design']['condition_number']:.3f} | {row['design']['maximum_absolute_offdiagonal_correlation']:.3f} |")
    lines.extend(["", "## 2018–2021 OOF（12成员均值）", "",
        "| Layer | Candidate | RMSE mg/L | NSE | R² | anomaly log-RMSE | absolute log-RMSE | median station NSE | median station R² |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for layer in ("P1", "P2"):
        table = mean.loc[mean.layer.eq(layer)].set_index("candidate")
        available = set(table.index)
        for candidate in ("GAUSSIAN_PROCESS_PARENT", *CANDIDATES):
            if candidate not in available:
                continue
            row = table.loc[candidate]
            lines.append(f"| {layer} | {candidate} | {row.rmse:.4f} | {row.nse:.4f} | {row.r2:.4f} | {row.anomaly:.4f} | {row.absolute:.4f} | {row.median_nse:.4f} | {row.median_r2:.4f} |")
    lines.extend(["", "## 门禁", "",
        "| Candidate | P1 anomaly improved | P1 station NI | P1 tree NI | P2 station NI | P2 tree NI | stable | boundary/fit confounded | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in result["candidate_matrix"]:
        lines.append(f"| {row['candidate']} | {row['P1_anomaly_simultaneously_improved']}/12 | {row['P1_station_absolute_noninferior']}/12 | {row['P1_tree_absolute_noninferior']}/12 | {row['P2_station_absolute_noninferior']}/12 | {row['P2_tree_absolute_noninferior']}/12 | {row['parameter_stable_models']}/12 | {row['candidate_boundary_confounded_models'] + row['fit_or_readout_confounded_models']} | {row['formal_gate_pass']} |")
    distribution = selections.groupby(["candidate", "K_N_mg_L"], dropna=False).size().reset_index(name="fold_count").fillna("NA").to_dict("records")
    lines.extend(["", "## K_N选择", "", "```json", json.dumps(distribution, ensure_ascii=False, indent=2, default=str), "```", "",
        "## 结论边界", "",
        "本轮若通过，只能支持注册的低维河道算子；不能把TN拟合解释为实测反应速率。若基础水力候选未通过，不允许增加温度来补偿。2022 TN未读取。",
    ])
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require_runtime()
    for path in (OUT, REPORTS, LOCKS, CACHE): path.mkdir(parents=True, exist_ok=True)
    if json.loads(STAGE0.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("STOP_PARENT_STAGE0_NOT_PASS")
    if json.loads(F2_DECISION.read_text(encoding="utf-8")).get("TN_2022_values_read") is not False:
        raise RuntimeError("STOP_2022_BOUNDARY")
    lock = write_input_lock()
    numerical = endpoint_and_range_audit()
    synth_path = REPORTS / "f3_synthetic_identifiability.json"
    if synth_path.exists() and (OUT / "f3_synthetic_recovery.parquet").exists():
        synthetic = json.loads(synth_path.read_text(encoding="utf-8"))
    else:
        _, synthetic = synthetic_recovery()
    active_candidates = tuple(row["candidate"] for row in synthetic["candidate_reports"] if row["pass"])
    if not active_candidates:
        result = {"status": "IDENTIFIABILITY_BLOCKED", "observed_candidate_fit_performed": False, "synthetic": synthetic, "TN_2022_values_read": False}
        dump_json(REPORTS / "f3_decision.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    predictions, parameters, grid, selections = run_formal_oof(active_candidates)
    gates, distributions, metrics, reproduction = evaluate_formal(predictions, active_candidates)
    stability = stability_table(selections, parameters, active_candidates)
    result = decide(gates, stability, synthetic, numerical, active_candidates)
    result["input_lock_sha256"] = sha256(LOCKS / "f3_pre_observed_candidate_input_lock.json")
    result["observed_candidate_fit_performed"] = True
    result["parent_reproduction_pass"] = bool(reproduction["pass"].all())
    dump_json(REPORTS / "f3_decision.json", result)
    write_report(result, metrics, selections, synthetic)
    continuation = json.loads(CONTINUATION.read_text(encoding="utf-8"))
    continuation["F3_status"] = result["status"]; continuation["status"] = "completed"
    dump_json(CONTINUATION, continuation)
    completion_files = [
        LOCKS / "f3_pre_observed_candidate_input_lock.json", REPORTS / "f3_synthetic_identifiability.json",
        REPORTS / "f3_decision.json", REPORTS / "technical_report.md",
        OUT / "f3_temporal_oof_predictions.parquet", OUT / "f3_fold_parameters.parquet",
        OUT / "f3_paired_simultaneous_gates.parquet", OUT / "f3_performance_metrics.parquet",
        OUT / "f3_aquatic_numerical_audit.parquet", OUT / "f3_parent_reproduction.parquet",
    ]
    completion = {"status": "PASS", "decision": result["status"], "files": {str(p): sha256(p) for p in completion_files}, "TN_2022_values_read": False}
    dump_json(REPORTS / "f3_completion_audit.json", completion)
    print(json.dumps({"input_lock": lock["aggregate_sha256"], "decision": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
