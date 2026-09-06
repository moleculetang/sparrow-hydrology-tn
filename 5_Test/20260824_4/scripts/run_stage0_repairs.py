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
HERE = TEST / "20260824_4"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"

MANIFEST = HERE / "program_manifest.json"
CHARTER = HERE / "program_charter.json"
CONTRACT = HERE / "experiment_contract.json"
THIS_SCRIPT = Path(__file__)

PARENT_LOCK = TEST / "20260824_1" / "locks" / "parent_lock_registry.json"
PARENT_DECISION = TEST / "20260824_1" / "reports" / "parent_readjudication.json"
OBS = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLDS = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
INTERFACE = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
Q72_ROUTED = TEST / "20260820_2" / "outputs" / "q72_routed_monthly_discharge_2006_2022.parquet"
SEGMENTS = TEST / "20260820_2" / "outputs" / "andreadis_500m_channel_segments.parquet"
EXPOSURE = TEST / "20260820_18" / "outputs" / "reach_month_aquatic_uptake_exposure_2006_2021.parquet"
H_SHARED = TEST / "20260820_19" / "scripts" / "hierarchical19_shared.py"
FULL_PARAMETERS = TEST / "20260820_19" / "outputs" / "full_development_parameters.parquet"
POSTERIOR = TEST / "20260820_19" / "outputs" / "posterior_diagnostics.parquet"
LAPLACE = TEST / "20260820_19" / "outputs" / "posterior_laplace_diagnostics.parquet"
STAGE3_DECISION = TEST / "20260824_3" / "reports" / "temporal_decision.json"
STAGE3_SELECTIONS = TEST / "20260824_3" / "outputs" / "fold_phi_selections.parquet"
STAGE3_PARAMETERS = TEST / "20260824_3" / "outputs" / "candidate_fold_parameters.parquet"
STAGE2_SCRIPT = TEST / "20260824_2" / "scripts" / "run_source_identifiability_preflight.py"

FORMAL_MUS = (12, 36, 60, 96, 144, 240)
FORMAL_MODELS = tuple(
    sorted(
        [f"S0_mu_{mu:03d}m" for mu in FORMAL_MUS]
        + [f"S1_tau_012m_mu_{mu:03d}m" for mu in FORMAL_MUS]
    )
)
H_IDENTITY_REL_TOL = 1e-12
Q_CLOSURE_ABS_TOL_MM = 1e-10
ROUTING_REL_TOL = 1e-12


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


def input_paths() -> list[Path]:
    paths = [
        MANIFEST,
        CHARTER,
        CONTRACT,
        THIS_SCRIPT,
        PARENT_LOCK,
        PARENT_DECISION,
        OBS,
        FOLDS,
        INTERFACE,
        Q72_ROUTED,
        SEGMENTS,
        EXPOSURE,
        H_SHARED,
        FULL_PARAMETERS,
        POSTERIOR,
        LAPLACE,
        STAGE3_DECISION,
        STAGE3_SELECTIONS,
        STAGE3_PARAMETERS,
        STAGE2_SCRIPT,
    ]
    return paths


def write_pre_candidate_lock() -> dict[str, object]:
    paths = input_paths()
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing registered input: {missing}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    if not manifest["registered_before_new_candidate_tn_results"]:
        raise RuntimeError("STOP_PROGRAM_NOT_REGISTERED")
    if not contract["registered_before_new_candidate_tn_results"]:
        raise RuntimeError("STOP_EXPERIMENT_NOT_REGISTERED")
    if parent["selected_full_development_process_parent"] != "H1_GLOBAL":
        raise RuntimeError("STOP_PARENT_NOT_H1_GLOBAL")
    if parent["TN_2022_values_read"]:
        raise RuntimeError("STOP_PARENT_TN_2022_BOUNDARY")
    hashes = {str(path): sha256(path) for path in paths}
    lock = {
        "lock_id": "20260824_4_stage0_pre_candidate_lock",
        "created_before_new_candidate_TN_fit": True,
        "TN_2022_values_read": False,
        "files": hashes,
        "aggregate_sha256": hashlib.sha256(
            "\n".join(f"{key}|{value}" for key, value in sorted(hashes.items())).encode("utf-8")
        ).hexdigest(),
    }
    dump_json(LOCKS / "stage0_pre_candidate_input_lock.json", lock)
    return lock


def segmentwise_h1_identity() -> tuple[pd.DataFrame, dict[str, object]]:
    segments = pd.read_parquet(SEGMENTS).sort_values(["reach_id", "segment_index"]).reset_index(drop=True)
    hydro = pd.read_parquet(Q72_ROUTED, filters=[("year", ">=", 2006), ("year", "<=", 2021)])
    hydro = hydro.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    exposure = pd.read_parquet(EXPOSURE)
    exposure = exposure.loc[
        exposure["year"].between(2006, 2021)
        & exposure["hydraulic_scenario"].eq("central_n0035")
    ].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    if segments["wqd_reference_discharge_used"].any() or hydro["andreadis_discharge_used"].any():
        raise RuntimeError("STOP_REFERENCE_DISCHARGE_USED")
    reach_ids = np.sort(hydro["reach_id"].unique().astype(int))
    if len(reach_ids) != 230:
        raise RuntimeError("STOP_REACH_COVERAGE")
    ridx = {int(value): i for i, value in enumerate(reach_ids)}
    seg_ridx = segments["reach_id"].astype(int).map(ridx).to_numpy(int)
    x = segments["segment_midpoint_fraction"].to_numpy(float)
    length = segments["segment_length_m"].to_numpy(float)
    width = segments["width_central_m"].to_numpy(float)
    mid_length = segments["midpoint_to_outlet_segment_length_m"].to_numpy(float)
    times = hydro[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    shape = (len(times), len(reach_ids))
    upstream = hydro["q72_upstream_discharge_m3_s"].to_numpy(float).reshape(shape)
    local = hydro["q72_local_discharge_m3_s"].to_numpy(float).reshape(shape)
    expected_full = np.empty(shape, dtype=float)
    expected_mid = np.empty(shape, dtype=float)
    for ti in range(len(times)):
        q_segment = upstream[ti, seg_ridx] + x * local[ti, seg_ridx]
        if np.any(q_segment <= 0):
            raise RuntimeError("STOP_NONPOSITIVE_SEGMENT_Q")
        expected_full[ti] = np.bincount(
            seg_ridx,
            weights=length * width / (86400.0 * q_segment),
            minlength=len(reach_ids),
        )
        expected_mid[ti] = np.bincount(
            seg_ridx,
            weights=mid_length * width / (86400.0 * q_segment),
            minlength=len(reach_ids),
        )
    observed_full = exposure["uptake_exposure_full_days_per_m"].to_numpy(float).reshape(shape)
    observed_mid = exposure["uptake_exposure_midpoint_to_outlet_days_per_m"].to_numpy(float).reshape(shape)
    rel_full = np.abs(observed_full - expected_full) / np.maximum(np.abs(expected_full), 1e-30)
    rel_mid = np.abs(observed_mid - expected_mid) / np.maximum(np.abs(expected_mid), 1e-30)
    rows = exposure[["reach_id", "year", "month"]].copy()
    rows["h_full_registered_days_per_m"] = observed_full.ravel()
    rows["h_full_segment_identity_days_per_m"] = expected_full.ravel()
    rows["h_full_relative_error"] = rel_full.ravel()
    rows["h_mid_registered_days_per_m"] = observed_mid.ravel()
    rows["h_mid_segment_identity_days_per_m"] = expected_mid.ravel()
    rows["h_mid_relative_error"] = rel_mid.ravel()
    audit = {
        "status": "PASS" if max(float(rel_full.max()), float(rel_mid.max())) <= H_IDENTITY_REL_TOL else "FAIL",
        "correct_identity": "H_reach=sum_i(L_i*W_i/(86400*Q_i))",
        "invalid_prior_audit": "sum(tau_i)/length_weighted_mean_depth",
        "production_H_values_changed": False,
        "rows": int(len(rows)),
        "reaches": int(rows["reach_id"].nunique()),
        "max_relative_error_full": float(rel_full.max()),
        "max_relative_error_midpoint": float(rel_mid.max()),
        "reference_discharge_used": False,
    }
    return rows, audit


def route_mass_closure(router: object, vf: np.ndarray) -> float:
    sf = np.exp(-router.h_full * vf[None, :])
    sm = np.exp(-router.h_mid * vf[None, :])
    worst = 0.0
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
        worst = max(worst, float(np.max(np.abs(balance) / scale)))
    return worst


def hydrology_and_routing_audit() -> tuple[pd.DataFrame, dict[str, object]]:
    interface = pd.read_parquet(INTERFACE, filters=[("year", ">=", 2006), ("year", "<=", 2021)])
    q_error = (
        interface["q_local_total_mm"]
        - interface["quick_release_mm"]
        - interface["gw_discharge_mm"]
    )
    h19 = load_module(H_SHARED, "tn24_4_hierarchical19")
    shared = h19.parent_shared()
    params = pd.read_parquet(FULL_PARAMETERS).set_index("model_id")
    rows: list[dict[str, object]] = []
    for model_id in FORMAL_MODELS:
        router = h19.build_router(model_id, shared)
        v_f = float(params.loc[model_id, "v_f_m_per_day"])
        closure = route_mass_closure(router, np.full(len(router.reach_ids), v_f))
        rows.append(
            {
                "model_id": model_id,
                "v_f_m_per_day": v_f,
                "routing_mass_balance_max_relative_error": closure,
                "routing_mass_balance_pass": closure <= ROUTING_REL_TOL,
            }
        )
    frame = pd.DataFrame(rows)
    report = {
        "status": "PASS"
        if float(q_error.abs().max()) <= Q_CLOSURE_ABS_TOL_MM
        and frame["routing_mass_balance_pass"].all()
        else "FAIL",
        "water_flux_source": sorted(interface["hydrology_source"].astype(str).unique().tolist()),
        "q_local_quick_plus_gw_max_abs_error_mm": float(q_error.abs().max()),
        "routing_mass_balance_max_relative_error": float(frame["routing_mass_balance_max_relative_error"].max()),
        "formal_models": len(frame),
        "TN_2022_values_read": False,
    }
    return frame, report


def inference_and_prior_experiment_supersession() -> dict[str, object]:
    full = pd.read_parquet(FULL_PARAMETERS)
    post = pd.read_parquet(POSTERIOR)
    laplace = pd.read_parquet(LAPLACE)
    production_vf = float(full["v_f_m_per_day"].median())
    posterior_vf = float(post.loc[post["parameter"].eq("v_f_m_per_day"), "mean"].iloc[0])
    laplace_vf = float(np.median([json.loads(value)[0] for value in laplace["map_json"]]))
    selections = pd.read_parquet(STAGE3_SELECTIONS)
    model_boundary = (
        selections.assign(boundary=selections["selected_phi_A"].eq(1.0))
        .groupby("mu_month", observed=True)["boundary"]
        .sum()
    )
    stage3 = json.loads(STAGE3_DECISION.read_text(encoding="utf-8"))
    stage2_source = STAGE2_SCRIPT.read_text(encoding="utf-8")
    return {
        "status": "PASS",
        "production_inference": {
            "production_v_f_m_per_day_median": production_vf,
            "posterior_surrogate_v_f_mean_m_per_day": posterior_vf,
            "laplace_map_v_f_median_m_per_day": laplace_vf,
            "posterior_to_production_ratio": posterior_vf / production_vf,
            "authoritative_uncertainty_status": "WITHDRAWN_NONISOMORPHIC_TO_PRODUCTION_OBJECTIVE",
            "reason": "The posterior/Laplace surrogate is centered on a different sample-SSE optimum rather than the production station/tree-macro bilevel optimum."
        },
        "production_fitting_contract": {
            "eta_inner_fit": "penalized sample-level log1p SSE",
            "structure_outer_score": "mean of station-macro and tree-macro log1p RMSE",
            "description": "bilevel fitting, not one coherent likelihood or posterior"
        },
        "20260824_2_synthetic_supersession": {
            "old_script_uses_linear_eta_objective": "np.linalg.lstsq" in stage2_source,
            "old_script_clips_noisy_log_tn": "np.maximum" in stage2_source,
            "old_decision_may_be_used_for_new_family_gating": False,
            "required_replacement": [
                "fit eta in the registered log1p objective",
                "include and score a real phi=0 endpoint",
                "simulate untruncated noisy log1p TN and transform only where the scoring contract requires concentration",
                "write an immutable input lock before candidate analysis"
            ],
            "scientific_source_identity_conclusion": "retained as strong practical-collinearity evidence, but its recovery percentages are non-authoritative until corrected"
        },
        "20260824_3_superseding_status": {
            "old_status": stage3.get("scientific_status"),
            "new_status": "PROCESS_NOT_SUPPORTED_PREDICTION_PRESERVED",
            "phi_1_fold_selections": int(selections["selected_phi_A"].eq(1.0).sum()),
            "models_phi_1_in_at_least_2_of_4_folds": int((model_boundary >= 2).sum()),
            "boundary_interpretation": "phi=1 is an endpoint boundary and must be reported as boundary confounding, not stable interior evidence",
        },
        "TN_2022_values_read": False,
    }


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)
    lock = write_pre_candidate_lock()

    h_rows, h_report = segmentwise_h1_identity()
    h_rows.to_parquet(OUT / "h1_segmentwise_identity_audit.parquet", index=False)
    dump_json(REPORTS / "h1_segmentwise_identity_audit.json", h_report)

    route_rows, route_report = hydrology_and_routing_audit()
    route_rows.to_parquet(OUT / "q72_h1_routing_mass_closure.parquet", index=False)
    dump_json(REPORTS / "q72_h1_hard_gate_audit.json", route_report)

    supersession = inference_and_prior_experiment_supersession()
    dump_json(REPORTS / "superseding_repairs_and_evidence_status.json", supersession)

    completion = {
        "status": "PASS"
        if h_report["status"] == "PASS" and route_report["status"] == "PASS"
        else "FAIL",
        "pre_candidate_lock_sha256": sha256(LOCKS / "stage0_pre_candidate_input_lock.json"),
        "segmentwise_h1_identity": h_report["status"],
        "q72_and_routing_hard_gates": route_report["status"],
        "uncertainty_withdrawal_recorded": supersession["production_inference"]["authoritative_uncertainty_status"].startswith("WITHDRAWN"),
        "stage3_reclassified": supersession["20260824_3_superseding_status"]["new_status"],
        "corrected_stage2_synthetic_required_before_reuse": True,
        "new_candidate_TN_fit_performed": False,
        "TN_2022_values_read": False,
    }
    dump_json(REPORTS / "stage0_repair_completion_audit.json", completion)
    if completion["status"] != "PASS":
        raise RuntimeError(f"STAGE0_REPAIR_FAILED: {completion}")
    print(json.dumps({"lock": lock["aggregate_sha256"], "completion": completion}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
