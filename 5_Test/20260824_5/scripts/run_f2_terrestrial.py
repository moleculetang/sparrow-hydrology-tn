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
HERE = TEST / "20260824_5"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
CACHE = HERE / "cache"

CONTRACT = HERE / "experiment_contract.json"
CONTINUATION = HERE / "program_continuation.json"
PARENT_MANIFEST = TEST / "20260824_4" / "program_manifest.json"
STAGE0 = TEST / "20260824_4" / "reports" / "stage0_repair_completion_audit.json"
LEGACY_SHARED_PATH = TEST / "20260818_1" / "scripts" / "legacy18_shared.py"
H_SHARED_PATH = TEST / "20260820_19" / "scripts" / "hierarchical19_shared.py"
MONTHLY = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
OBS = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLDS = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
DOMAIN = TEST / "20260820_19" / "outputs" / "observation_domain_registry.parquet"
PARENT_LOCAL = TEST / "20260820_10" / "cache" / "parent_local"
PARENT_TEMPORAL = TEST / "20260820_19" / "cache" / "temporal"
SEGMENTS = TEST / "20260820_2" / "outputs" / "andreadis_500m_channel_segments.parquet"
GLHYMPS = TEST / "20260814_9" / "inputs" / "model_ready" / "static" / "soil_tn_glhymps_by_reach.parquet"

MUS = (12, 36, 60, 96, 144, 240)
FORMAL_MODELS = tuple(sorted(
    [f"S0_mu_{mu:03d}m" for mu in MUS]
    + [f"S1_tau_012m_mu_{mu:03d}m" for mu in MUS]
))
CANDIDATES = (
    "L1_SHARED_CATCHMENT_MEMORY_REPLACES_T1",
    "L2_Q72_ANTECEDENT_WETNESS_EFFECTIVE_LOSS",
    "L3_PATHWAY_MPR_SLOPE_PERMEABILITY",
)
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
GAMMA_GRID = (-0.5, 0.0, 0.5)
Q_RHO = 0.25
WATER_EPS = 1e-12
SPIN_TOL = 1e-9
SPIN_MAX = 5000
MASS_REL_TOL = 1e-10
NESTING_REL_TOL = 1e-10
TIE_TOL = 1e-8
BOOT_REPS = 10_000
SYNTH_REPS = 500
SEED = 2026082405


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


def formal_spec(model_id: str) -> tuple[str, int | None, int]:
    mu = int(model_id.split("mu_")[1][:3])
    if model_id.startswith("S0_"):
        return "S0", None, mu
    return "S1", 12, mu


def write_input_lock() -> dict[str, object]:
    paths = [
        CONTRACT, CONTINUATION, PARENT_MANIFEST, STAGE0,
        Path(__file__), LEGACY_SHARED_PATH, H_SHARED_PATH, MONTHLY,
        OBS, FOLDS, DOMAIN, SEGMENTS, GLHYMPS,
    ]
    paths.extend(PARENT_LOCAL / f"{model_id}.parquet" for model_id in FORMAL_MODELS)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    hashes = {str(path): sha256(path) for path in paths}
    lock = {
        "experiment_id": "20260824_5",
        "created_before_observed_candidate_fitting": True,
        "files": hashes,
        "aggregate_sha256": hashlib.sha256(
            "\n".join(f"{key}|{value}" for key, value in sorted(hashes.items())).encode("utf-8")
        ).hexdigest(),
        "development_TN_years_allowed": [2016, 2017, 2018, 2019, 2020, 2021],
        "OOF_evaluation_years": [2018, 2019, 2020, 2021],
        "TN_2022_values_read": False,
    }
    dump_json(LOCKS / "f2_pre_observed_candidate_input_lock.json", lock)
    return lock


def withdraw(pool: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    removed = np.minimum(pool, demand)
    pool -= removed
    return removed, demand - removed


def l1_partition(
    q_input: np.ndarray,
    g_input: np.ndarray,
    quick_water: np.ndarray,
    gw_water: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_n = q_input + g_input
    total_water = quick_water + gw_water
    fallback = np.divide(
        quick_water, total_water, out=np.zeros_like(total_water), where=total_water > WATER_EPS
    )
    qshare = np.divide(q_input, total_n, out=fallback.copy(), where=total_n > 1e-30)
    qshare = np.clip(qshare, 0.0, 1.0)
    return qshare, 1.0 - qshare


def spinup_l1(
    legacy: ModuleType,
    structure: str,
    tau_s: int | None,
    mu: int,
    early_positive: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    state = {
        name: np.zeros(len(early_positive), dtype=float)
        for name in ("son", "mobile", "shared", "quick", "gw_gate")
    }
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_shared = mu / (1.0 + mu)
    final_balance = np.zeros(len(early_positive), dtype=float)
    delta = np.inf
    for cycle in range(1, SPIN_MAX + 1):
        before = np.concatenate([value.copy() for value in state.values()])
        for t in range(12):
            start = sum(value.astype(np.longdouble) for value in state.values())
            bypass, _, flush, q_contact_share, g_contact_share = legacy.operator_water_partitions(arrays, t, "F00")
            direct = early_positive * bypass
            pool_input = early_positive - direct
            if structure == "S0":
                state["mobile"] += pool_input
            else:
                state["son"] += pool_input
                mineralized = state["son"] * (1.0 - rho_s)
                state["son"] -= mineralized
                state["mobile"] += mineralized
            mobilized = state["mobile"] * flush
            state["mobile"] -= mobilized
            qin = direct + mobilized * q_contact_share
            gin = mobilized * g_contact_share
            state["shared"] += qin + gin
            shared_release = (1.0 - rho_shared) * state["shared"]
            state["shared"] -= shared_release
            qshare, gshare = l1_partition(
                qin, gin, arrays["quick_release_mm"][t], arrays["gw_discharge_mm"][t]
            )
            state["quick"] += shared_release * qshare
            state["gw_gate"] += shared_release * gshare
            qrel = np.where(
                arrays["quick_release_mm"][t] > WATER_EPS,
                (1.0 - Q_RHO) * state["quick"], 0.0,
            )
            grel = np.where(
                arrays["gw_discharge_mm"][t] > WATER_EPS,
                state["gw_gate"], 0.0,
            )
            state["quick"] -= qrel
            state["gw_gate"] -= grel
            end = sum(value.astype(np.longdouble) for value in state.values())
            final_balance = np.asarray(
                early_positive.astype(np.longdouble) + start
                - qrel.astype(np.longdouble) - grel.astype(np.longdouble) - end,
                dtype=float,
            )
        after = np.concatenate(list(state.values()))
        delta = float(np.max(np.abs(after - before)))
        if delta <= SPIN_TOL:
            break
    return state, {
        "source_structure": structure,
        "soil_tau_month": tau_s,
        "delivery_mu_month": mu,
        "cycles": cycle,
        "converged": bool(delta <= SPIN_TOL),
        "terminal_max_abs_delta_kg_n": delta,
        "son_end_kg_n": float(state["son"].sum()),
        "mobile_end_kg_n": float(state["mobile"].sum()),
        "shared_end_kg_n": float(state["shared"].sum()),
        "quick_end_kg_n": float(state["quick"].sum()),
        "gw_gate_end_kg_n": float(state["gw_gate"].sum()),
        "final_cycle_max_abs_mass_balance_error_kg_n": float(np.max(np.abs(final_balance))),
    }


def simulate_l1(model_id: str) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    legacy = load_module(LEGACY_SHARED_PATH, f"f2_l1_legacy_{model_id}")
    reach_ids, times, arrays, early_positive = legacy.prepare_arrays()
    structure, tau_s, mu = formal_spec(model_id)
    state, spin = spinup_l1(legacy, structure, tau_s, mu, early_positive, arrays)
    if not spin["converged"]:
        raise RuntimeError(f"STOP_L1_SPINUP:{model_id}")
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_shared = mu / (1.0 + mu)
    rows: list[pd.DataFrame] = []
    max_abs = 0.0
    max_rel = 0.0
    minimum = np.inf
    total_input = total_release = total_negative = 0.0
    for t, (year, month) in enumerate(times):
        if year > 2021:
            break
        start = sum(value.astype(np.longdouble) for value in state.values())
        positive = arrays["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        negative = arrays["negative_legacy_eligible_n_surplus_kg_n_month"][t].copy()
        removed_mobile, left = withdraw(state["mobile"], negative)
        removed_son, unmet = withdraw(state["son"], left)
        removed = removed_mobile + removed_son
        bypass, _, flush, q_contact_share, g_contact_share = legacy.operator_water_partitions(arrays, t, "F00")
        direct = positive * bypass
        pool_input = positive - direct
        if structure == "S0":
            state["mobile"] += pool_input
            mineralized = np.zeros(len(reach_ids), dtype=float)
        else:
            state["son"] += pool_input
            mineralized = state["son"] * (1.0 - rho_s)
            state["son"] -= mineralized
            state["mobile"] += mineralized
        mobilized = state["mobile"] * flush
        state["mobile"] -= mobilized
        qin = direct + mobilized * q_contact_share
        gin = mobilized * g_contact_share
        state["shared"] += qin + gin
        shared_release = (1.0 - rho_shared) * state["shared"]
        state["shared"] -= shared_release
        qshare, gshare = l1_partition(
            qin, gin, arrays["quick_release_mm"][t], arrays["gw_discharge_mm"][t]
        )
        qalloc = shared_release * qshare
        galloc = shared_release * gshare
        state["quick"] += qalloc
        state["gw_gate"] += galloc
        qrel = np.where(
            arrays["quick_release_mm"][t] > WATER_EPS,
            (1.0 - Q_RHO) * state["quick"], 0.0,
        )
        grel = np.where(
            arrays["gw_discharge_mm"][t] > WATER_EPS,
            state["gw_gate"], 0.0,
        )
        state["quick"] -= qrel
        state["gw_gate"] -= grel
        end = sum(value.astype(np.longdouble) for value in state.values())
        balance = (
            positive.astype(np.longdouble) + start
            - removed.astype(np.longdouble) - qrel.astype(np.longdouble)
            - grel.astype(np.longdouble) - end
        )
        scale = (
            np.abs(positive.astype(np.longdouble)) + np.abs(start)
            + np.abs(removed.astype(np.longdouble)) + np.abs(qrel.astype(np.longdouble))
            + np.abs(grel.astype(np.longdouble)) + np.abs(end)
        )
        relative = np.asarray(np.abs(balance) / np.maximum(scale, 1.0), dtype=float)
        max_abs = max(max_abs, float(np.max(np.abs(balance))))
        max_rel = max(max_rel, float(np.max(relative)))
        minimum = min(
            minimum, *(float(v.min()) for v in state.values()),
            float(qrel.min()), float(grel.min()), float(qalloc.min()), float(galloc.min()),
        )
        total_input += float(positive.sum())
        total_release += float(qrel.sum() + grel.sum())
        total_negative += float(removed.sum())
        if 2016 <= year <= 2021:
            rows.append(pd.DataFrame({
                "reach_id": reach_ids, "year": year, "month": month,
                "quick_tn_release_kg_n": qrel, "gw_tn_release_kg_n": grel,
                "shared_state_end_kg_n": state["shared"],
                "quick_state_end_kg_n": state["quick"],
                "gw_gate_state_end_kg_n": state["gw_gate"],
                "shared_release_kg_n": shared_release,
                "quick_allocation_fraction": qshare,
                "mass_balance_error_kg_n": np.asarray(balance, dtype=float),
            }))
    frame = pd.concat(rows, ignore_index=True)
    frame["model_id"] = model_id
    frame["candidate"] = CANDIDATES[0]
    audit = {
        "model_id": model_id,
        "max_abs_mass_balance_error_kg_n": max_abs,
        "max_relative_mass_balance_error": max_rel,
        "minimum_state_or_flux_kg_n": minimum,
        "total_positive_input_kg_n": total_input,
        "total_release_kg_n": total_release,
        "total_negative_removed_kg_n": total_negative,
        "unmet_negative_end_kg_n": float(unmet.sum()),
    }
    spin["model_id"] = model_id
    return frame, spin, audit


def prepare_l1_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    local_dir = CACHE / "l1_local"
    local_dir.mkdir(parents=True, exist_ok=True)
    spin_rows: list[dict[str, object]] = []
    mass_rows: list[dict[str, object]] = []
    for model_id in FORMAL_MODELS:
        path = local_dir / f"{model_id}.parquet"
        if path.exists():
            # Audits are deliberately recomputed from the simulator when the
            # consolidated audit files do not yet exist.
            if (OUT / "l1_spinup_audit.parquet").exists() and (OUT / "l1_mass_balance_audit.parquet").exists():
                continue
        frame, spin, mass = simulate_l1(model_id)
        frame.to_parquet(path, index=False)
        spin_rows.append(spin)
        mass_rows.append(mass)
    if spin_rows:
        spin = pd.DataFrame(spin_rows)
        mass = pd.DataFrame(mass_rows)
        spin.to_parquet(OUT / "l1_spinup_audit.parquet", index=False)
        mass.to_parquet(OUT / "l1_mass_balance_audit.parquet", index=False)
    else:
        spin = pd.read_parquet(OUT / "l1_spinup_audit.parquet")
        mass = pd.read_parquet(OUT / "l1_mass_balance_audit.parquet")
    if not spin.converged.all() or mass.max_relative_mass_balance_error.max() > MASS_REL_TOL:
        raise RuntimeError("STOP_L1_NUMERICAL_CONTRACT")
    if mass.minimum_state_or_flux_kg_n.min() < -1e-8:
        raise RuntimeError("STOP_L1_NEGATIVE_STATE")
    return spin, mass


def prepare_wetness() -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_parquet(
        MONTHLY,
        columns=["reach_id", "year", "month", "gw_response_state_end_mm"],
        filters=[("year", "<=", 2021)],
    ).sort_values(["reach_id", "year", "month"])
    monthly["antecedent_gw_state_mm"] = monthly.groupby("reach_id", observed=True)[
        "gw_response_state_end_mm"
    ].shift(1)
    monthly = monthly.dropna(subset=["antecedent_gw_state_mm"])
    folds = pd.read_parquet(FOLDS)[
        ["fold_id", "train_start_year", "train_end_year", "evaluation_year"]
    ].drop_duplicates().sort_values("evaluation_year")
    rows = []
    audits = []
    for fold in folds.itertuples(index=False):
        climatology = monthly.loc[monthly.year.between(1961, int(fold.train_end_year))].groupby(
            ["reach_id", "month"], as_index=False
        ).agg(
            antecedent_mean_mm=("antecedent_gw_state_mm", "mean"),
            antecedent_sd_mm=("antecedent_gw_state_mm", lambda x: float(np.std(x, ddof=0))),
            climatology_years=("year", "nunique"),
        )
        current = monthly.loc[monthly.year.between(2016, 2021)].merge(
            climatology, on=["reach_id", "month"], validate="many_to_one"
        )
        sd = current.antecedent_sd_mm.to_numpy(float)
        current["wetness_anomaly"] = np.divide(
            current.antecedent_gw_state_mm.to_numpy(float) - current.antecedent_mean_mm.to_numpy(float),
            sd,
            out=np.zeros(len(current), dtype=float),
            where=sd > 1e-12,
        )
        current["positive_wetness_anomaly"] = np.maximum(current.wetness_anomaly, 0.0)
        current["fold_id"] = str(fold.fold_id)
        rows.append(current)
        audits.append({
            "fold_id": str(fold.fold_id),
            "training_end_year": int(fold.train_end_year),
            "minimum_climatology_years": int(current.climatology_years.min()),
            "zero_sd_rows": int((sd <= 1e-12).sum()),
            "anomaly_min": float(current.wetness_anomaly.min()),
            "anomaly_p01": float(current.wetness_anomaly.quantile(0.01)),
            "anomaly_p50": float(current.wetness_anomaly.quantile(0.50)),
            "anomaly_p99": float(current.wetness_anomaly.quantile(0.99)),
            "anomaly_max": float(current.wetness_anomaly.max()),
        })
    registry = pd.concat(rows, ignore_index=True)
    audit = pd.DataFrame(audits)
    registry.to_parquet(OUT / "q72_antecedent_wetness_registry.parquet", index=False)
    audit.to_parquet(OUT / "q72_antecedent_wetness_audit.parquet", index=False)
    return registry, audit


def prepare_mpr_covariates() -> pd.DataFrame:
    segments = pd.read_parquet(SEGMENTS, columns=["reach_id", "segment_length_m", "slope_used"])
    if (segments.slope_used <= 0).any():
        raise RuntimeError("STOP_NONPOSITIVE_SLOPE")
    segments["weighted_log_slope"] = segments.segment_length_m * np.log(segments.slope_used)
    slope = segments.groupby("reach_id", as_index=False).agg(
        length_sum_m=("segment_length_m", "sum"),
        weighted_log_slope_sum=("weighted_log_slope", "sum"),
    )
    slope["slope_geomean"] = np.exp(slope.weighted_log_slope_sum / slope.length_sum_m)
    glh = pd.read_parquet(GLHYMPS, columns=["reach_id", "glhymps_log10_permeability_m2"])
    cov = slope.merge(glh, on="reach_id", validate="one_to_one").sort_values("reach_id")
    for source, target in (
        ("slope_geomean", "z_log_slope"),
        ("glhymps_log10_permeability_m2", "z_log10_permeability"),
    ):
        values = np.log(cov[source].to_numpy(float)) if source == "slope_geomean" else cov[source].to_numpy(float)
        cov[target] = (values - values.mean()) / values.std(ddof=0)
    cov["quick_multiplier_gamma_neg0p5"] = np.exp(-0.5 * cov.z_log_slope)
    cov["quick_multiplier_gamma_pos0p5"] = np.exp(0.5 * cov.z_log_slope)
    cov["gw_multiplier_gamma_neg0p5"] = np.exp(-0.5 * cov.z_log10_permeability)
    cov["gw_multiplier_gamma_pos0p5"] = np.exp(0.5 * cov.z_log10_permeability)
    cov.to_parquet(OUT / "f2_pathway_mpr_covariates.parquet", index=False)
    return cov


def clone_router(h: ModuleType, base: object, model_id: str, local_q: np.ndarray, local_g: np.ndarray) -> object:
    return h.HydraulicRouter(
        model_id=model_id,
        reach_ids=base.reach_ids.copy(), years=base.years.copy(), months=base.months.copy(),
        local_q=np.asarray(local_q, dtype=float), local_g=np.asarray(local_g, dtype=float),
        h_full=base.h_full.copy(), h_mid=base.h_mid.copy(), water=base.water.copy(), x=base.x.copy(),
        terminal_by_reach=base.terminal_by_reach.copy(), order_index=list(base.order_index),
        downstream_index=dict(base.downstream_index),
    )


def l1_router(h: ModuleType, base: object, model_id: str) -> object:
    local = pd.read_parquet(CACHE / "l1_local" / f"{model_id}.parquet").sort_values(
        ["year", "month", "reach_id"]
    )
    shape = base.local_q.shape
    if len(local) != shape[0] * shape[1]:
        raise RuntimeError(f"STOP_L1_ROUTER_SHAPE:{model_id}")
    return clone_router(
        h, base, f"{model_id}__L1",
        local.quick_tn_release_kg_n.to_numpy(float).reshape(shape),
        local.gw_tn_release_kg_n.to_numpy(float).reshape(shape),
    )


def l2_router(
    h: ModuleType,
    base: object,
    model_id: str,
    fold_id: str,
    lambda_value: float,
    wetness: pd.DataFrame,
) -> object:
    part = wetness.loc[wetness.fold_id.eq(fold_id)].sort_values(["year", "month", "reach_id"])
    if len(part) != base.local_q.size:
        raise RuntimeError(f"STOP_L2_WETNESS_SHAPE:{fold_id}")
    survival = np.exp(-float(lambda_value) * part.positive_wetness_anomaly.to_numpy(float)).reshape(base.local_q.shape)
    return clone_router(
        h, base, f"{model_id}__L2__{fold_id}__lambda_{lambda_value:g}",
        base.local_q * survival, base.local_g * survival,
    )


def l3_router(
    h: ModuleType,
    base: object,
    model_id: str,
    gamma_q: float,
    gamma_g: float,
    covariates: pd.DataFrame,
) -> object:
    cov = covariates.set_index("reach_id").reindex(base.reach_ids.astype(int))
    if cov.isna().any().any():
        raise RuntimeError("STOP_L3_COVARIATE_KEY")
    mq = np.exp(float(gamma_q) * cov.z_log_slope.to_numpy(float))
    mg = np.exp(float(gamma_g) * cov.z_log10_permeability.to_numpy(float))
    if abs(float(np.mean(np.log(mq)))) > 1e-12 or abs(float(np.mean(np.log(mg)))) > 1e-12:
        raise RuntimeError("STOP_L3_GEOMETRIC_NORMALIZATION")
    return clone_router(
        h, base, f"{model_id}__L3__gq_{gamma_q:g}__gg_{gamma_g:g}",
        base.local_q * mq[None, :], base.local_g * mg[None, :],
    )


def endpoint_audits(wetness: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    h = load_module(H_SHARED_PATH, "f2_endpoint_h")
    shared = h.parent_shared()
    rows = []
    for model_id in FORMAL_MODELS:
        base = h.build_router(model_id, shared)
        for fold_id in sorted(wetness.fold_id.unique()):
            endpoint = l2_router(h, base, model_id, str(fold_id), 0.0, wetness)
            qerr = float(np.max(np.abs(endpoint.local_q - base.local_q)))
            gerr = float(np.max(np.abs(endpoint.local_g - base.local_g)))
            scale = max(float(np.max(np.abs(base.local_q))), float(np.max(np.abs(base.local_g))), 1.0)
            rows.append({
                "model_id": model_id, "candidate": CANDIDATES[1], "fold_id": str(fold_id),
                "max_abs_quick_error_kg_n": qerr, "max_abs_gw_error_kg_n": gerr,
                "relative_error": max(qerr, gerr) / scale,
            })
        endpoint = l3_router(h, base, model_id, 0.0, 0.0, covariates)
        qerr = float(np.max(np.abs(endpoint.local_q - base.local_q)))
        gerr = float(np.max(np.abs(endpoint.local_g - base.local_g)))
        scale = max(float(np.max(np.abs(base.local_q))), float(np.max(np.abs(base.local_g))), 1.0)
        rows.append({
            "model_id": model_id, "candidate": CANDIDATES[2], "fold_id": "all",
            "max_abs_quick_error_kg_n": qerr, "max_abs_gw_error_kg_n": gerr,
            "relative_error": max(qerr, gerr) / scale,
        })
    frame = pd.DataFrame(rows)
    frame["pass"] = frame.relative_error <= NESTING_REL_TOL
    frame.to_parquet(OUT / "f2_parent_endpoint_nesting_audit.parquet", index=False)
    if not frame["pass"].all():
        raise RuntimeError("STOP_F2_ENDPOINT_NESTING")
    return frame


def route_log_concentration(router: object, vf_value: float = 0.13) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vf = np.full(len(router.reach_ids), vf_value, dtype=float)
    rq, rg = router.route(vf)
    concentration = np.divide(
        (0.8 * rq + 0.8 * rg) * 1000.0,
        router.water,
        out=np.zeros_like(router.water),
        where=router.water > WATER_EPS,
    )
    return np.log1p(np.maximum(concentration, 0.0)), rq, rg


def synthetic_design_frames(
    wetness: pd.DataFrame, covariates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    h = load_module(H_SHARED_PATH, "f2_synth_h")
    shared = h.parent_shared()
    model_id = "S0_mu_096m"
    base = h.build_router(model_id, shared)
    l1 = l1_router(h, base, model_id)
    l2 = l2_router(h, base, model_id, "F4", 0.5, wetness)
    l3 = l3_router(h, base, model_id, 0.5, -0.5, covariates)
    base_log, rq, rg = route_log_concentration(base)
    logs = {
        CANDIDATES[0]: route_log_concentration(l1)[0],
        CANDIDATES[1]: route_log_concentration(l2)[0],
        CANDIDATES[2]: route_log_concentration(l3)[0],
    }
    keys = pd.read_parquet(
        OBS,
        columns=["station_key", "reach_id", "year", "month"],
        filters=[("year", "<=", 2021)],
    )
    domain = pd.read_parquet(DOMAIN, columns=["station_key", "primary_river_domain"])
    keys = keys.merge(domain, on="station_key", validate="many_to_one")
    keys = keys.loc[keys.primary_river_domain & keys.year.between(2016, 2021)].drop(columns="primary_river_domain")
    reach_lookup = base.reach_lookup
    time_lookup = base.time_lookup
    ridx = keys.reach_id.astype(int).map(reach_lookup).to_numpy(int)
    tidx = np.fromiter(
        (time_lookup[(int(y), int(m))] for y, m in keys[["year", "month"]].itertuples(index=False)),
        dtype=int, count=len(keys),
    )
    frame = keys.copy()
    frame["base_log"] = base_log[tidx, ridx]
    for candidate, value in logs.items():
        frame[candidate] = value[tidx, ridx] - frame.base_log.to_numpy(float)
    quick_fraction = np.divide(rq, rq + rg, out=np.zeros_like(rq), where=(rq + rg) > 1e-30)
    frame["quick_fraction"] = quick_fraction[tidx, ridx]
    design_audit: dict[str, object] = {}
    for candidate in CANDIDATES:
        design = frame[["quick_fraction", candidate]].to_numpy(float)
        design = (design - design.mean(axis=0)) / np.maximum(design.std(axis=0), 1e-12)
        singular = np.linalg.svd(design, compute_uv=False)
        corr = np.corrcoef(design, rowvar=False)
        design_audit[candidate] = {
            "condition_number": float(singular.max() / singular.min()),
            "maximum_absolute_offdiagonal_correlation": float(abs(corr[0, 1])),
            "fingerprint_sd_log1p": float(frame[candidate].std(ddof=0)),
            "fingerprint_p01": float(frame[candidate].quantile(0.01)),
            "fingerprint_p99": float(frame[candidate].quantile(0.99)),
        }
    return frame, design_audit


def synthetic_recovery(
    wetness: pd.DataFrame, covariates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame, design = synthetic_design_frames(wetness, covariates)
    train = frame.loc[frame.year.between(2016, 2020)].copy()
    test = frame.loc[frame.year.eq(2021)].copy()
    rows: list[dict[str, object]] = []
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
                # The intercept is a global nuisance analogous to scale; the
                # registered bridge coefficient is bounded to [0, 1].
                x_mean = x_train.mean()
                y_center = y_train - train.base_log.to_numpy(float)
                theta_hat = float(np.clip(np.dot(x_train - x_mean, y_center - y_center.mean()) / denominator, 0.0, 1.0))
                intercept = float(np.mean(y_center - theta_hat * x_train))
                y_test = test.base_log.to_numpy(float) + theta_true * x_test + rng.normal(0.0, 0.14, len(test))
                parent_pred = test.base_log.to_numpy(float) + intercept
                candidate_pred = parent_pred + theta_hat * x_test
                block_rows = []
                work = test[["station_key"]].copy()
                work["obs"] = y_test
                work["parent"] = parent_pred
                work["candidate"] = candidate_pred
                for _, group in work.groupby("station_key", observed=True):
                    obs = group.obs.to_numpy(float) - group.obs.mean()
                    pa = group.parent.to_numpy(float) - group.parent.mean()
                    ca = group.candidate.to_numpy(float) - group.candidate.mean()
                    block_rows.append(float(np.sqrt(np.mean((ca - obs) ** 2)) - np.sqrt(np.mean((pa - obs) ** 2))))
                delta = np.asarray(block_rows, dtype=float)
                se = float(np.std(delta, ddof=1) / np.sqrt(len(delta)))
                upper = float(delta.mean() + 1.96 * se)
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
        passed = (
            false_rate <= 0.05 and power >= 0.80
            and d["condition_number"] < 30
            and d["maximum_absolute_offdiagonal_correlation"] < 0.9
        )
        candidate_reports.append({
            "candidate": candidate, "false_upgrade_rate": false_rate, "power": power,
            "design": d, "pass": bool(passed),
        })
    report = {
        "status": "PASS" if all(row["pass"] for row in candidate_reports) else "FAIL",
        "null_replicates_per_candidate": SYNTH_REPS,
        "material_alternative_replicates_per_candidate": SYNTH_REPS,
        "candidate_reports": candidate_reports,
        "L1_parent_bridge_used_only_for_synthetic_recovery": True,
        "L2_exact_parent_endpoint": "lambda=0",
        "L3_exact_parent_endpoint": "gamma_q=gamma_g=0",
        "TN_2022_values_read": False,
    }
    recovery.to_parquet(OUT / "f2_synthetic_recovery.parquet", index=False)
    dump_json(REPORTS / "f2_synthetic_identifiability.json", report)
    return recovery, report


def fit_parameter_row(
    model_id: str,
    fold_id: str,
    candidate: str,
    fit: dict[str, object],
    lambda_value: float = math.nan,
    gamma_q: float = math.nan,
    gamma_g: float = math.nan,
    selected: bool = True,
    layer: str = "P1",
) -> dict[str, object]:
    params = np.asarray(fit.get("parameters", []), dtype=float)
    vf = np.asarray(fit.get("vf", []), dtype=float)
    diagnostic = fit["diagnostic"]
    eta = np.asarray(fit["eta"], dtype=float)
    return {
        "model_id": model_id, "fold_id": fold_id, "candidate": candidate,
        "layer": layer, "selected": selected,
        "lambda_wetness": lambda_value, "gamma_quick_slope": gamma_q,
        "gamma_gw_permeability": gamma_g,
        "training_objective": float(fit.get("objective", math.nan)),
        "training_data_objective": float(fit.get("data_objective", math.nan)),
        "v_f_m_per_day": float(params[0]) if len(params) else math.nan,
        "vf_boundary": bool(len(vf) and vf.max() >= 0.49),
        "eta_quick": float(eta[0]), "eta_gw": float(eta[1]),
        "eta_boundary": bool(diagnostic["eta_boundary"]),
        "optimizer_success": bool(diagnostic["success"] and fit.get("outer_success", True)),
        "outer_nfev": int(fit.get("outer_nfev", 0)),
    }


def fit_model(model_id: str) -> tuple[str, str, str, str]:
    h = load_module(H_SHARED_PATH, f"f2_h_{model_id}")
    shared = h.parent_shared()
    observations = h.development_observations()
    folds = h.fold_registry()
    wetness = pd.read_parquet(OUT / "q72_antecedent_wetness_registry.parquet")
    covariates = pd.read_parquet(OUT / "f2_pathway_mpr_covariates.parquet")
    base = h.build_router(model_id, shared)
    l1 = l1_router(h, base, model_id)
    l3_routers = {
        (gq, gg): l3_router(h, base, model_id, gq, gg, covariates)
        for gq in GAMMA_GRID for gg in GAMMA_GRID
    }
    prediction_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)
        train_obs = observations.loc[observations.year.between(
            int(fold.train_start_year), int(fold.train_end_year)
        )].copy()
        test_obs = observations.loc[observations.year.eq(int(fold.evaluation_year))].copy()

        parent_fit = h.fit_structure(base, train_obs, "H1_GLOBAL", shared)
        l1_fit = h.fit_structure(l1, train_obs, "H1_GLOBAL", shared)

        l2_routers = {
            value: l2_router(h, base, model_id, fold_id, value, wetness)
            for value in LAMBDA_GRID
        }
        l2_fits = {
            value: h.fit_structure(router, train_obs, "H1_GLOBAL", shared)
            for value, router in l2_routers.items()
        }
        l2_best = min(float(fit["objective"]) for fit in l2_fits.values())
        selected_lambda = min(
            value for value in LAMBDA_GRID
            if float(l2_fits[value]["objective"]) <= l2_best + TIE_TOL
        )

        l3_fits = {
            pair: h.fit_structure(router, train_obs, "H1_GLOBAL", shared)
            for pair, router in l3_routers.items()
        }
        l3_best = min(float(fit["objective"]) for fit in l3_fits.values())
        l3_ties = [
            pair for pair, fit in l3_fits.items()
            if float(fit["objective"]) <= l3_best + TIE_TOL
        ]
        selected_gq, selected_gg = min(
            l3_ties, key=lambda pair: (abs(pair[0]) + abs(pair[1]), abs(pair[0]), abs(pair[1]), pair)
        )

        for value, fit in l2_fits.items():
            grid_rows.append(fit_parameter_row(
                model_id, fold_id, CANDIDATES[1], fit,
                lambda_value=value, selected=value == selected_lambda,
            ))
        for (gq, gg), fit in l3_fits.items():
            grid_rows.append(fit_parameter_row(
                model_id, fold_id, CANDIDATES[2], fit,
                gamma_q=gq, gamma_g=gg,
                selected=(gq, gg) == (selected_gq, selected_gg),
            ))
        selection_rows.extend([
            {
                "model_id": model_id, "fold_id": fold_id,
                "evaluation_year": int(fold.evaluation_year),
                "candidate": CANDIDATES[0], "lambda_wetness": math.nan,
                "gamma_quick_slope": math.nan, "gamma_gw_permeability": math.nan,
                "training_objective": float(l1_fit["objective"]),
                "parent_training_objective": float(parent_fit["objective"]),
            },
            {
                "model_id": model_id, "fold_id": fold_id,
                "evaluation_year": int(fold.evaluation_year),
                "candidate": CANDIDATES[1], "lambda_wetness": selected_lambda,
                "gamma_quick_slope": math.nan, "gamma_gw_permeability": math.nan,
                "training_objective": float(l2_fits[selected_lambda]["objective"]),
                "parent_training_objective": float(parent_fit["objective"]),
            },
            {
                "model_id": model_id, "fold_id": fold_id,
                "evaluation_year": int(fold.evaluation_year),
                "candidate": CANDIDATES[2], "lambda_wetness": math.nan,
                "gamma_quick_slope": selected_gq, "gamma_gw_permeability": selected_gg,
                "training_objective": float(l3_fits[(selected_gq, selected_gg)]["objective"]),
                "parent_training_objective": float(parent_fit["objective"]),
            },
        ])

        selected = {
            "GAUSSIAN_PROCESS_PARENT": (base, parent_fit, math.nan, math.nan, math.nan),
            CANDIDATES[0]: (l1, l1_fit, math.nan, math.nan, math.nan),
            CANDIDATES[1]: (
                l2_routers[selected_lambda], l2_fits[selected_lambda], selected_lambda, math.nan, math.nan
            ),
            CANDIDATES[2]: (
                l3_routers[(selected_gq, selected_gg)], l3_fits[(selected_gq, selected_gg)],
                math.nan, selected_gq, selected_gg,
            ),
        }
        for candidate, (router, fit, lambda_value, gamma_q, gamma_g) in selected.items():
            vf = np.asarray(fit["vf"], dtype=float)
            train_frame = router.frame(train_obs, vf)
            test_frame = router.frame(test_obs, vf)
            p1 = shared.predict_layer(test_frame, "P1", np.asarray(fit["eta"]), fit["effects"])
            p1["candidate"] = candidate
            p1["model_id"] = model_id
            p1["fold_id"] = fold_id
            p1["evaluation_year"] = int(fold.evaluation_year)
            p1["lambda_wetness"] = lambda_value
            p1["gamma_quick_slope"] = gamma_q
            p1["gamma_gw_permeability"] = gamma_g
            prediction_rows.append(p1)
            parameter_rows.append(fit_parameter_row(
                model_id, fold_id, candidate, fit,
                lambda_value, gamma_q, gamma_g, layer="P1",
            ))

            p2_fit = h.fit_selected_readout(train_frame, "P2", shared)
            p2 = h.predict_selected(test_frame, "P2", p2_fit, shared)
            p2["candidate"] = candidate
            p2["model_id"] = model_id
            p2["fold_id"] = fold_id
            p2["evaluation_year"] = int(fold.evaluation_year)
            p2["lambda_wetness"] = lambda_value
            p2["gamma_quick_slope"] = gamma_q
            p2["gamma_gw_permeability"] = gamma_g
            prediction_rows.append(p2)
            p2_proxy = {
                "eta": p2_fit["eta"], "diagnostic": p2_fit["diagnostic"],
                "parameters": fit["parameters"], "vf": fit["vf"],
                "objective": math.nan, "data_objective": math.nan,
                "outer_success": True, "outer_nfev": 0,
            }
            parameter_rows.append(fit_parameter_row(
                model_id, fold_id, candidate, p2_proxy,
                lambda_value, gamma_q, gamma_g, layer="P2",
            ))

    model_cache = CACHE / "formal"
    model_cache.mkdir(parents=True, exist_ok=True)
    pred_path = model_cache / f"{model_id}__predictions.parquet"
    par_path = model_cache / f"{model_id}__parameters.parquet"
    grid_path = model_cache / f"{model_id}__grid_scores.parquet"
    sel_path = model_cache / f"{model_id}__selections.parquet"
    pd.concat(prediction_rows, ignore_index=True).to_parquet(pred_path, index=False)
    pd.DataFrame(parameter_rows).to_parquet(par_path, index=False)
    pd.DataFrame(grid_rows).to_parquet(grid_path, index=False)
    pd.DataFrame(selection_rows).to_parquet(sel_path, index=False)
    return str(pred_path), str(par_path), str(grid_path), str(sel_path)


def run_formal_oof() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_cache = CACHE / "formal"
    paths = [(
        model_cache / f"{model_id}__predictions.parquet",
        model_cache / f"{model_id}__parameters.parquet",
        model_cache / f"{model_id}__grid_scores.parquet",
        model_cache / f"{model_id}__selections.parquet",
    ) for model_id in FORMAL_MODELS]
    if not all(all(path.exists() for path in group) for group in paths):
        workers = min(4, os.cpu_count() or 1)
        completed = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fit_model, model_id): model_id for model_id in FORMAL_MODELS}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(json.dumps({
                    "f2_model_complete": futures[future], "completed": len(completed), "total": len(FORMAL_MODELS)
                }), flush=True)
        paths = [tuple(Path(value) for value in group) for group in completed]
    predictions = pd.concat([pd.read_parquet(group[0]) for group in paths], ignore_index=True)
    parameters = pd.concat([pd.read_parquet(group[1]) for group in paths], ignore_index=True)
    grid_scores = pd.concat([pd.read_parquet(group[2]) for group in paths], ignore_index=True)
    selections = pd.concat([pd.read_parquet(group[3]) for group in paths], ignore_index=True)
    predictions.to_parquet(OUT / "f2_temporal_oof_predictions.parquet", index=False)
    parameters.to_parquet(OUT / "f2_fold_parameters.parquet", index=False)
    grid_scores.to_parquet(OUT / "f2_candidate_grid_scores.parquet", index=False)
    selections.to_parquet(OUT / "f2_fold_selections.parquet", index=False)
    return predictions, parameters, grid_scores, selections


def station_block_values(frame: pd.DataFrame, anomaly: bool) -> pd.Series:
    work = frame.copy()
    work["obs_log"] = np.log1p(work.tn_mg_l.to_numpy(float))
    work["pred_log"] = np.log1p(work.pred_tn_mg_l.to_numpy(float))
    if anomaly:
        work["obs_log"] -= work.groupby("station_key", observed=True).obs_log.transform("mean")
        work["pred_log"] -= work.groupby("station_key", observed=True).pred_log.transform("mean")
    return work.groupby("station_key", observed=True).apply(
        lambda g: float(np.sqrt(np.mean(np.square(g.pred_log - g.obs_log)))),
        include_groups=False,
    )


def tree_block_values(frame: pd.DataFrame) -> pd.Series:
    work = frame.assign(
        error=np.log1p(frame.pred_tn_mg_l.to_numpy(float)) - np.log1p(frame.tn_mg_l.to_numpy(float))
    )
    return work.groupby("terminal_tree_id", observed=True).error.apply(
        lambda x: float(np.sqrt(np.mean(np.square(x))))
    )


def paired_simultaneous_gates(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    distributions: list[pd.DataFrame] = []
    rng = np.random.default_rng(SEED + 90_000)
    for model_id in FORMAL_MODELS:
        for layer in ("P1", "P2"):
            parent = predictions.loc[
                predictions.model_id.eq(model_id)
                & predictions.layer.eq(layer)
                & predictions.candidate.eq("GAUSSIAN_PROCESS_PARENT")
            ]
            metric_funs = (
                ("station_anomaly_rmse_log1p", lambda x: station_block_values(x, True)),
                ("station_absolute_rmse_log1p", lambda x: station_block_values(x, False)),
                ("tree_absolute_rmse_log1p", tree_block_values),
            )
            for metric, block_fun in metric_funs:
                parent_values = block_fun(parent)
                common_index = rng.integers(0, len(parent_values), size=(BOOT_REPS, len(parent_values)))
                candidate_dist: dict[str, np.ndarray] = {}
                point: dict[str, float] = {}
                for candidate in CANDIDATES:
                    cand = predictions.loc[
                        predictions.model_id.eq(model_id)
                        & predictions.layer.eq(layer)
                        & predictions.candidate.eq(candidate)
                    ]
                    values = block_fun(cand).reindex(parent_values.index)
                    if values.isna().any():
                        raise RuntimeError("STOP_PAIRED_BLOCK_KEY")
                    delta = values.to_numpy(float) - parent_values.to_numpy(float)
                    candidate_dist[candidate] = delta[common_index].mean(axis=1)
                    point[candidate] = float(delta.mean())
                bootstrap_se = {
                    candidate: max(float(np.std(candidate_dist[candidate], ddof=1)), 1e-12)
                    for candidate in CANDIDATES
                }
                centered_max_t = np.max(np.column_stack([
                    (candidate_dist[candidate] - point[candidate]) / bootstrap_se[candidate]
                    for candidate in CANDIDATES
                ]), axis=1)
                critical = float(np.quantile(centered_max_t, 0.975))
                for candidate in CANDIDATES:
                    dist = candidate_dist[candidate]
                    simultaneous_upper = point[candidate] + critical * bootstrap_se[candidate]
                    rows.append({
                        "model_id": model_id, "layer": layer, "metric": metric,
                        "reference": "GAUSSIAN_PROCESS_PARENT", "candidate": candidate,
                        "delta_candidate_minus_parent": point[candidate],
                        "paired_ci95_lower": float(np.quantile(dist, 0.025)),
                        "paired_ci95_upper": float(np.quantile(dist, 0.975)),
                        "bootstrap_standard_error": bootstrap_se[candidate],
                        "max_t_critical_value": critical,
                        "simultaneous_ci95_upper": simultaneous_upper,
                        "point_improved": bool(point[candidate] < 0),
                        "simultaneous_improved": bool(simultaneous_upper < 0),
                        "simultaneous_noninferior_0p005": bool(simultaneous_upper < 0.005),
                    })
                    distributions.append(pd.DataFrame({
                        "model_id": model_id, "layer": layer, "metric": metric,
                        "candidate": candidate, "replicate": np.arange(BOOT_REPS, dtype=int),
                        "delta_rmse_log1p": dist,
                    }))
    gates = pd.DataFrame(rows)
    dist = pd.concat(distributions, ignore_index=True)
    gates.to_parquet(OUT / "f2_paired_simultaneous_gates.parquet", index=False)
    dist.to_parquet(OUT / "f2_paired_bootstrap_distributions.parquet", index=False)
    return gates, dist


def overall_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, layer, candidate), frame in predictions.groupby(
        ["model_id", "layer", "candidate"], observed=True
    ):
        obs = frame.tn_mg_l.to_numpy(float)
        pred = frame.pred_tn_mg_l.to_numpy(float)
        denominator = float(np.sum(np.square(obs - obs.mean())))
        corr = float(np.corrcoef(obs, pred)[0, 1]) if np.std(pred) > 0 else math.nan
        station_rows = []
        for _, g in frame.groupby("station_key", observed=True):
            go = g.tn_mg_l.to_numpy(float)
            gp = g.pred_tn_mg_l.to_numpy(float)
            denom = float(np.sum(np.square(go - go.mean())))
            r = float(np.corrcoef(go, gp)[0, 1]) if np.std(go) > 0 and np.std(gp) > 0 else math.nan
            station_rows.append({
                "nse": 1.0 - float(np.sum(np.square(gp - go))) / denom if denom > 0 else math.nan,
                "r2": r * r,
                "rmse": float(np.sqrt(np.mean(np.square(gp - go)))),
            })
        station = pd.DataFrame(station_rows)
        rows.append({
            "model_id": model_id, "layer": layer, "candidate": candidate, "n": len(frame),
            "rmse_mg_l": float(np.sqrt(np.mean(np.square(pred - obs)))),
            "mae_mg_l": float(np.mean(np.abs(pred - obs))),
            "nse_mg_l": 1.0 - float(np.sum(np.square(pred - obs))) / denominator,
            "pearson_r2_mg_l": corr * corr,
            "pbias_percent": float(100.0 * np.sum(pred - obs) / np.sum(obs)),
            "station_macro_anomaly_rmse_log1p": float(station_block_values(frame, True).mean()),
            "station_macro_absolute_rmse_log1p": float(station_block_values(frame, False).mean()),
            "tree_macro_absolute_rmse_log1p": float(tree_block_values(frame).mean()),
            "median_station_nse": float(station.nse.replace([np.inf, -np.inf], np.nan).median()),
            "median_station_pearson_r2": float(station.r2.median()),
            "median_station_rmse_mg_l": float(station.rmse.median()),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(OUT / "f2_performance_metrics.parquet", index=False)
    return metrics


def parameter_stability(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, candidate), group in selections.groupby(["model_id", "candidate"], observed=True):
        if candidate == CANDIDATES[0]:
            stable = True
            endpoint = False
            boundary = False
            direction = "equal_complexity_fixed"
        elif candidate == CANDIDATES[1]:
            values = group.lambda_wetness.to_numpy(float)
            stable = int(np.sum(values > 0)) >= 3
            endpoint = bool(np.all(values == 0))
            boundary = int(np.sum(values == max(LAMBDA_GRID))) >= 2
            direction = "positive_effective_loss" if stable else "zero_or_unstable"
        else:
            pairs = list(zip(group.gamma_quick_slope.to_numpy(float), group.gamma_gw_permeability.to_numpy(float)))
            counts = pd.Series(pairs).value_counts()
            modal_pair = tuple(counts.index[0])
            stable = int(counts.iloc[0]) >= 3 and modal_pair != (0.0, 0.0)
            endpoint = bool(all(pair == (0.0, 0.0) for pair in pairs))
            # The registered +/-0.5 values are fixed MPR contrasts rather than
            # a continuous optimizer boundary.  Confounding is reserved for
            # an identical edge pair selected in all four folds.
            boundary = int(counts.iloc[0]) == 4 and modal_pair != (0.0, 0.0)
            direction = f"gamma_q={modal_pair[0]:g},gamma_g={modal_pair[1]:g}" if stable else "zero_or_unstable"
        rows.append({
            "model_id": model_id, "candidate": candidate,
            "parameter_stable": stable, "parent_endpoint_all_folds": endpoint,
            "candidate_boundary_confounded": boundary, "direction_label": direction,
        })
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "f2_parameter_stability.parquet", index=False)
    return frame


def parent_reproduction(predictions: pd.DataFrame) -> pd.DataFrame:
    authoritative = pd.read_parquet(TEST / "20260824_4" / "outputs" / "f1_temporal_oof_predictions.parquet")
    authoritative = authoritative.loc[authoritative.arm.eq("GAUSSIAN_PARENT")]
    current = predictions.loc[predictions.candidate.eq("GAUSSIAN_PROCESS_PARENT")]
    keys = ["station_key", "year", "month", "fold_id", "model_id", "layer"]
    joined = current[keys + ["pred_tn_mg_l"]].merge(
        authoritative[keys + ["pred_tn_mg_l"]], on=keys,
        suffixes=("_current", "_authoritative"), validate="one_to_one",
    )
    joined["abs_difference_mg_l"] = np.abs(joined.pred_tn_mg_l_current - joined.pred_tn_mg_l_authoritative)
    joined["relative_difference"] = joined.abs_difference_mg_l / np.maximum(
        np.abs(joined.pred_tn_mg_l_authoritative), 1e-12
    )
    summary = joined.groupby(["model_id", "layer"], as_index=False).agg(
        rows=("abs_difference_mg_l", "size"),
        max_abs_difference_mg_l=("abs_difference_mg_l", "max"),
        max_relative_difference=("relative_difference", "max"),
    )
    summary["pass"] = summary.max_relative_difference <= 1e-5
    summary.to_parquet(OUT / "f2_parent_reproduction.parquet", index=False)
    if not summary["pass"].all():
        raise RuntimeError("STOP_F2_PARENT_REPRODUCTION")
    return summary


def decision(
    gates: pd.DataFrame,
    parameters: pd.DataFrame,
    stability: pd.DataFrame,
    synthetic: dict[str, object],
    spin: pd.DataFrame,
    mass: pd.DataFrame,
) -> dict[str, object]:
    candidate_rows = []
    for candidate in CANDIDATES:
        def count(layer: str, metric: str, field: str) -> int:
            part = gates.loc[
                gates.candidate.eq(candidate) & gates.layer.eq(layer) & gates.metric.eq(metric)
            ]
            return int(part[field].sum())
        counts = {
            "P1_anomaly_point_improved": count("P1", "station_anomaly_rmse_log1p", "point_improved"),
            "P1_anomaly_simultaneously_improved": count("P1", "station_anomaly_rmse_log1p", "simultaneous_improved"),
            "P1_station_absolute_noninferior": count("P1", "station_absolute_rmse_log1p", "simultaneous_noninferior_0p005"),
            "P1_tree_absolute_noninferior": count("P1", "tree_absolute_rmse_log1p", "simultaneous_noninferior_0p005"),
            "P2_station_absolute_noninferior": count("P2", "station_absolute_rmse_log1p", "simultaneous_noninferior_0p005"),
            "P2_tree_absolute_noninferior": count("P2", "tree_absolute_rmse_log1p", "simultaneous_noninferior_0p005"),
        }
        par = parameters.loc[parameters.candidate.eq(candidate)]
        repeated_fit_boundary = par.groupby(["model_id", "layer"], observed=True).apply(
            lambda g: int((g.eta_boundary | g.vf_boundary | ~g.optimizer_success).sum()) >= 2,
            include_groups=False,
        )
        numerical_confounded_models = int(repeated_fit_boundary.groupby(level=0).any().sum())
        stable_models = int(stability.loc[
            stability.candidate.eq(candidate), "parameter_stable"
        ].sum())
        candidate_boundary_models = int(stability.loc[
            stability.candidate.eq(candidate), "candidate_boundary_confounded"
        ].sum())
        stability_required = candidate == CANDIDATES[0] or stable_models >= 10
        no_confounding = numerical_confounded_models <= 2 and candidate_boundary_models <= 2
        formal_pass = bool(
            counts["P1_anomaly_point_improved"] >= 10
            and counts["P1_anomaly_simultaneously_improved"] >= 10
            and counts["P1_station_absolute_noninferior"] >= 10
            and counts["P1_tree_absolute_noninferior"] >= 10
            and counts["P2_station_absolute_noninferior"] >= 10
            and counts["P2_tree_absolute_noninferior"] >= 10
            and stability_required and no_confounding
        )
        candidate_rows.append({
            "candidate": candidate, **counts,
            "parameter_stable_models": stable_models,
            "candidate_boundary_confounded_models": candidate_boundary_models,
            "fit_or_readout_confounded_models": numerical_confounded_models,
            "formal_gate_pass": formal_pass,
        })
    matrix = pd.DataFrame(candidate_rows)
    matrix.to_parquet(OUT / "f2_candidate_decision_matrix.parquet", index=False)
    numeric = bool(
        spin.converged.all()
        and mass.max_relative_mass_balance_error.max() <= MASS_REL_TOL
        and mass.minimum_state_or_flux_kg_n.min() >= -1e-8
    )
    if synthetic["status"] != "PASS":
        status = "IDENTIFIABILITY_BLOCKED"
    elif not numeric:
        status = "IMPLEMENTATION_BLOCKED"
    elif matrix.formal_gate_pass.any():
        status = "F2_TERRESTRIAL_TEMPORAL_UPGRADE_SUPPORTED_PENDING_SPATIAL"
    elif any(row["P2_station_absolute_noninferior"] >= 10 for row in candidate_rows):
        status = "F2_PROCESS_NOT_SUPPORTED_PREDICTION_PRESERVED"
    else:
        status = "NO_REGISTERED_F2_UPGRADE_SUPPORTED"
    winners = matrix.loc[matrix.formal_gate_pass, "candidate"].tolist()
    return {
        "status": status, "supported_candidates": winners,
        "candidate_matrix": candidate_rows,
        "synthetic_gate": synthetic["status"], "numeric_contract_pass": numeric,
        "F1_hinge_was_not_in_parent_comparison": True,
        "spatial_extrapolation_status": "not_yet_evaluated",
        "TN_years_read": [2016, 2017, 2018, 2019, 2020, 2021],
        "TN_2022_values_read": False,
        "claim_boundary": "A supported F2 candidate is an effective terrestrial-interface mechanism, not identification of real SON stock, manure fraction, mineralization rate, groundwater age or monthly agricultural timing.",
    }


def write_report(
    result: dict[str, object], metrics: pd.DataFrame, selections: pd.DataFrame,
    synthetic: dict[str, object],
) -> None:
    mean_metrics = metrics.groupby(["layer", "candidate"], as_index=False).agg(
        rmse_mg_l=("rmse_mg_l", "mean"),
        nse_mg_l=("nse_mg_l", "mean"),
        pearson_r2_mg_l=("pearson_r2_mg_l", "mean"),
        anomaly_rmse=("station_macro_anomaly_rmse_log1p", "mean"),
        absolute_log_rmse=("station_macro_absolute_rmse_log1p", "mean"),
        median_station_nse=("median_station_nse", "mean"),
        median_station_r2=("median_station_pearson_r2", "mean"),
    )
    selection_distribution: dict[str, list[dict[str, object]]] = {}
    for candidate, group in selections.groupby("candidate", observed=True):
        counts = group.groupby(
            ["lambda_wetness", "gamma_quick_slope", "gamma_gw_permeability"],
            dropna=False,
        ).size().reset_index(name="fold_count")
        selection_distribution[str(candidate)] = counts.fillna("NA").to_dict("records")
    lines = [
        "# F2 农田缓释与陆地—水体接口实验",
        "",
        f"正式裁决：`{result['status']}`。",
        "",
        "## 这轮到底测试了什么",
        "",
        "本轮没有在原模型后面再增加一个自由 lag。`L1` 保留同一个正式成员的 `mu`，但把记忆从仅地下水路径之后移动到快/慢路径分流之前，并删除原 post-GW T1；因此它是等复杂度的记忆位置挑战。`L2` 检验 Q72 前期湿润状态是否带来额外有效损失；`L3` 检验坡度与渗透率能否形成可迁移的路径 delivery multiplier。三者均独立对照 pre-F1 Gaussian H1 Parent。",
        "",
        "农业输入仍是年度重建的 `annual/12`。即使 L1 通过，也只能称为 effective terrestrial-memory operator，不能称为识别了真实农田矿化月份、SON库存或粪肥比例。",
        "",
        "## 合成可识别性",
        "",
        f"合成门：`{synthetic['status']}`。每个候选各使用500个null与500个material-alternative重复。",
        "",
        "| Candidate | false upgrade | power | condition number | max |r| |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in synthetic["candidate_reports"]:
        lines.append(
            f"| {row['candidate']} | {row['false_upgrade_rate']:.3f} | {row['power']:.3f} | "
            f"{row['design']['condition_number']:.3f} | {row['design']['maximum_absolute_offdiagonal_correlation']:.3f} |"
        )
    lines.extend([
        "",
        "## 2018–2021 OOF效果（12个正式成员的均值）",
        "",
        "| Layer | Candidate | RMSE mg/L | NSE | Pearson R² | station anomaly log-RMSE | station absolute log-RMSE | median station NSE | median station R² |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    order = ["GAUSSIAN_PROCESS_PARENT", *CANDIDATES]
    for layer in ("P1", "P2"):
        table = mean_metrics.loc[mean_metrics.layer.eq(layer)].set_index("candidate")
        for candidate in order:
            row = table.loc[candidate]
            lines.append(
                f"| {layer} | {candidate} | {row.rmse_mg_l:.4f} | {row.nse_mg_l:.4f} | "
                f"{row.pearson_r2_mg_l:.4f} | {row.anomaly_rmse:.4f} | {row.absolute_log_rmse:.4f} | "
                f"{row.median_station_nse:.4f} | {row.median_station_r2:.4f} |"
            )
    lines.extend([
        "",
        "## 预注册门禁",
        "",
        "| Candidate | P1 anomaly improved | P1 station NI | P1 tree NI | P2 station NI | P2 tree NI | stable | confounded | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in result["candidate_matrix"]:
        lines.append(
            f"| {row['candidate']} | {row['P1_anomaly_simultaneously_improved']}/12 | "
            f"{row['P1_station_absolute_noninferior']}/12 | {row['P1_tree_absolute_noninferior']}/12 | "
            f"{row['P2_station_absolute_noninferior']}/12 | {row['P2_tree_absolute_noninferior']}/12 | "
            f"{row['parameter_stable_models']}/12 | "
            f"{row['fit_or_readout_confounded_models'] + row['candidate_boundary_confounded_models']} | "
            f"{row['formal_gate_pass']} |"
        )
    lines.extend([
        "",
        "## 参数选择",
        "",
        "`L2` 的 `lambda=0` 是正式 Parent 端点；`L3` 的 `(gamma_q,gamma_g)=(0,0)` 是正式 Parent 端点。fold选择分布如下：",
        "",
        "```json",
        json.dumps(selection_distribution, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## 解释边界与下一步",
        "",
        "结构选择只由每折训练期 P1 完成，P2 没有反向选择 L1/L2/L3。F1 的 C–Q hinge 没有进入本轮父模型，因此若某个 F2 候选通过，可把增益归因于注册的 terrestrial interface；若均失败，则保留 F1 作为已监测站时间均值层升级，但不能把 F1 的正高流斜率解释成已验证的农田缓释机制。",
        "",
        "本轮没有读取2022 TN，也没有运行温度、水库、第二SON池、第二地下水库或source-specific lag。",
    ])
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require_runtime()
    for path in (OUT, REPORTS, LOCKS, CACHE):
        path.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    if stage0.get("status") != "PASS":
        raise RuntimeError("STOP_PARENT_STAGE0_NOT_PASS")
    lock = write_input_lock()
    spin, mass = prepare_l1_candidates()
    wetness, wetness_audit = prepare_wetness()
    covariates = prepare_mpr_covariates()
    endpoint = endpoint_audits(wetness, covariates)

    synth_path = REPORTS / "f2_synthetic_identifiability.json"
    if synth_path.exists() and (OUT / "f2_synthetic_recovery.parquet").exists():
        synthetic = json.loads(synth_path.read_text(encoding="utf-8"))
    else:
        _, synthetic = synthetic_recovery(wetness, covariates)
    if synthetic["status"] != "PASS":
        result = {
            "status": "IDENTIFIABILITY_BLOCKED",
            "reason": "At least one registered F2 candidate failed the pre-observed known-truth gate",
            "observed_candidate_fit_performed": False,
            "synthetic": synthetic,
            "TN_2022_values_read": False,
        }
        dump_json(REPORTS / "f2_decision.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    predictions, parameters, grid_scores, selections = run_formal_oof()
    reproduction = parent_reproduction(predictions)
    metrics = overall_metrics(predictions)
    gates, distributions = paired_simultaneous_gates(predictions)
    stability = parameter_stability(selections)
    result = decision(gates, parameters, stability, synthetic, spin, mass)
    result["input_lock_sha256"] = sha256(LOCKS / "f2_pre_observed_candidate_input_lock.json")
    result["observed_candidate_fit_performed"] = True
    result["parent_reproduction_pass"] = bool(reproduction["pass"].all())
    result["endpoint_nesting_pass"] = bool(endpoint["pass"].all())
    result["q72_wetness_zero_sd_rows"] = int(wetness_audit.zero_sd_rows.sum())
    dump_json(REPORTS / "f2_decision.json", result)
    write_report(result, metrics, selections, synthetic)

    continuation = json.loads(CONTINUATION.read_text(encoding="utf-8"))
    continuation["status"] = "completed"
    continuation["F2_status"] = result["status"]
    continuation["TN_2022_values_read"] = False
    dump_json(CONTINUATION, continuation)
    completion_files = [
        LOCKS / "f2_pre_observed_candidate_input_lock.json",
        REPORTS / "f2_synthetic_identifiability.json",
        REPORTS / "f2_decision.json", REPORTS / "technical_report.md",
        OUT / "f2_temporal_oof_predictions.parquet", OUT / "f2_fold_parameters.parquet",
        OUT / "f2_paired_simultaneous_gates.parquet", OUT / "f2_performance_metrics.parquet",
        OUT / "l1_spinup_audit.parquet", OUT / "l1_mass_balance_audit.parquet",
        OUT / "f2_parent_endpoint_nesting_audit.parquet", OUT / "f2_parent_reproduction.parquet",
    ]
    completion = {
        "status": "PASS",
        "decision": result["status"],
        "files": {str(path): sha256(path) for path in completion_files},
        "TN_2022_values_read": False,
    }
    dump_json(REPORTS / "f2_completion_audit.json", completion)
    print(json.dumps({
        "input_lock": lock["aggregate_sha256"], "decision": result,
        "completion_status": completion["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
