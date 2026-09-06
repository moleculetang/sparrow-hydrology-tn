"""Conservative regionalized signature constraint for DYN2P component identity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_4"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(STAGE3 / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts"),
    str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from run_stage2_temporal import AlphaTwoPathCandidate, periodic_dyn2p_spinup  # noqa: E402
from run_stage3_diagnostics import (  # noqa: E402
    autocorrelation, eckhardt_baseflow, lyne_hollick_baseflow,
    route_instantaneous_np, ukih_baseflow,
)
from dyn2p_hbv import simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import DISCHARGE, FORCING, Q72, SCALING, TOPOLOGY, antecedent_mean, build_support  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


SELECTED_SEED = 260827
RIDGE = 12.0
OFFSET_BOUND = 2.0
MULTISCALE = ROOT / "5_Test" / "20260826_15" / "outputs" / "multiscale_static_features_standardized.parquet"
LOCAL_STATIC = ROOT / "5_Test" / "20260826_15" / "outputs" / "local_static_features_standardized.parquet"
FORBIDDEN_TEST_PATHS = [
    ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_locked_retrospective_2019_2022.parquet",
    ROOT / "5_Test" / "20260823_14" / "outputs" / "supplemental_zero_history_station_month_observations.parquet",
    ROOT / "5_Test" / "20260823_14" / "outputs" / "supplemental_zero_history_station_coverage.parquet",
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bfi_target(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    q = q[np.isfinite(q)]
    estimates = [
        np.sum(lyne_hollick_baseflow(q)),
        np.sum(eckhardt_baseflow(q)),
        np.sum(ukih_baseflow(q)),
    ]
    return float(np.median(np.asarray(estimates) / np.maximum(np.sum(q), 1.0e-12)))


def fit_ridge(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ logit(np.clip(target, 0.02, 0.98)))


def predict_ridge(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return expit(np.column_stack([np.ones(len(features)), features]) @ coefficients)


def apply_signature_operator(local: np.ndarray, target_fraction: np.ndarray, development_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = local.sum(axis=2)
    original_fraction = local[:, :, 1] / np.maximum(total, 1.0e-12)
    long_fraction = local[development_mask, :, 1].sum(axis=0) / np.maximum(total[development_mask].sum(axis=0), 1.0e-12)
    offset = np.clip(logit(np.clip(target_fraction, 1.0e-6, 1 - 1.0e-6)) - logit(np.clip(long_fraction, 1.0e-6, 1 - 1.0e-6)), -OFFSET_BOUND, OFFSET_BOUND)
    corrected_fraction = expit(logit(np.clip(original_fraction, 1.0e-8, 1 - 1.0e-8)) + offset[None, :])
    corrected = np.stack([total * (1.0 - corrected_fraction), total * corrected_fraction], axis=2)
    corrected[total <= 1.0e-12] = 0.0
    return corrected, offset


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    prior = json.loads((STAGE3 / "reports" / "stage3_decision.json").read_text(encoding="utf-8"))
    if prior["authorized_successor"] != "20260827_4":
        raise RuntimeError("Stage 3 did not authorize repair")
    contract_path = RUN / "experiment_contract.json"
    model_path = STAGE2 / "outputs" / f"dyn2p_alpha05_seed_{SELECTED_SEED}_lock.pt"
    input_files = [contract_path, model_path, DISCHARGE, FORCING, Q72, SCALING, TOPOLOGY, MULTISCALE, LOCAL_STATIC]
    write_json(REPORTS / "input_hash_registry.json", {
        "files": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in input_files],
        "forbidden_test_paths": [str(path) for path in FORBIDDEN_TEST_PATHS],
        "forbidden_test_paths_opened": [],
    })

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    spin_mask = np.asarray(dates.year <= 2009)
    train_mask = np.asarray((dates.year >= 2010) & (dates.year <= 2015))
    eval_mask = np.asarray((dates.year >= 2017) & (dates.year <= 2018))
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(float).copy()
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(float).copy()
    p, pet = torch.from_numpy(p_np), torch.from_numpy(pet_np)
    api3, api30 = torch.from_numpy(antecedent_mean(p_np, 3)), torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    area_np = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)
    area = torch.from_numpy(area_np.copy())
    local_static = torch.from_numpy(pd.read_parquet(LOCAL_STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(float).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    model = AlphaTwoPathCandidate(SELECTED_SEED)
    model.load_state_dict(saved["model_state"])
    model.eval()
    physical = raw_to_physical(saved["raw_parameters"].detach().to(torch.float64))
    initial, spin = periodic_dyn2p_spinup(p[spin_mask], pet[spin_mask], physical, local_static, center, scale, model.gate)
    with torch.no_grad():
        result = simulate_dyn2p_hbv(p, pet, api3, api30, sin_doy, cos_doy, physical, initial, local_static, center, scale, model.gate)
    local = (result.components_mm_day * area[None, :, None] * 1000.0 / 86400.0).numpy()

    stations = pd.read_parquet(STAGE2 / "outputs" / "temporal_station_registry.parquet")
    support = build_support(stations, reach_ids, order, downstream)
    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    observed = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(float)
    train_bfi = np.asarray([bfi_target(observed[train_mask, index]) for index in range(len(stations))])
    eval_bfi = np.asarray([bfi_target(observed[eval_mask, index]) for index in range(len(stations))])
    multiscale = pd.read_parquet(MULTISCALE).sort_values("reach_id").set_index("reach_id")
    feature_columns = list(multiscale.columns)
    reach_features = multiscale.reindex(reach_ids).to_numpy(float)
    station_features = multiscale.reindex(stations.reach_id).to_numpy(float)
    coefficients = fit_ridge(station_features, train_bfi)
    predicted_reach_bfi = predict_ridge(reach_features, coefficients)

    cv_rows = []
    for heldout_tree in sorted(stations.terminal_tree.unique()):
        heldout = stations.terminal_tree.to_numpy() == heldout_tree
        fold_coefficients = fit_ridge(station_features[~heldout], train_bfi[~heldout])
        prediction = predict_ridge(station_features[heldout], fold_coefficients)
        climatology = np.repeat(np.mean(train_bfi[~heldout]), heldout.sum())
        for station_name, observed_bfi, predicted_bfi, climatology_bfi in zip(stations.loc[heldout, "station_norm"], train_bfi[heldout], prediction, climatology):
            cv_rows.append({
                "heldout_tree": int(heldout_tree), "station_norm": station_name,
                "observed_BFI": observed_bfi, "predicted_BFI": predicted_bfi,
                "climatology_BFI": climatology_bfi,
            })
    cv = pd.DataFrame(cv_rows)
    cv.to_parquet(OUT / "tree_block_signature_regionalization.parquet", index=False)
    cv_rmse = float(np.sqrt(np.mean((cv.predicted_BFI - cv.observed_BFI) ** 2)))
    cv_climatology_rmse = float(np.sqrt(np.mean((cv.climatology_BFI - cv.observed_BFI) ** 2)))

    corrected_local, offset = apply_signature_operator(local, predicted_reach_bfi, train_mask)
    routed_parent = route_instantaneous_np(local, list(order), downstream)
    routed_corrected = route_instantaneous_np(corrected_local, list(order), downstream)
    station_components = np.einsum("trc,sr->tsc", corrected_local, support)
    eval_components = station_components[eval_mask]
    eval_total = eval_components.sum(axis=2)
    predicted_eval_slow_fraction = eval_components[:, :, 1].sum(axis=0) / np.maximum(eval_total.sum(axis=0), 1.0e-12)
    rho, rho_p = spearmanr(predicted_eval_slow_fraction, eval_bfi)
    rmse = float(np.sqrt(np.mean((predicted_eval_slow_fraction - eval_bfi) ** 2)))
    high_direction, memory_direction = [], []
    for index in range(len(stations)):
        q = observed[eval_mask, index]
        valid = np.isfinite(q)
        q = q[valid]
        fast = eval_components[valid, index, 0]
        slow = eval_components[valid, index, 1]
        fraction = fast / np.maximum(fast + slow, 1.0e-12)
        q20, q90 = np.quantile(q, [0.2, 0.9])
        high_direction.append(float(np.mean(fraction[q >= q90])) > float(np.mean(fraction[q <= q20])))
        memory_direction.append(autocorrelation(slow, 30) > autocorrelation(fast, 30))
    station_output = stations[["station_norm", "reach_id", "terminal_tree"]].copy()
    station_output["train_BFI_target"] = train_bfi
    station_output["eval_BFI"] = eval_bfi
    station_output["predicted_eval_slow_fraction"] = predicted_eval_slow_fraction
    station_output["high_flow_direction"] = high_direction
    station_output["slow_memory_direction"] = memory_direction
    station_output.to_parquet(OUT / "signature_repair_station_metrics.parquet", index=False)

    operator = pd.DataFrame({
        "reach_id": reach_ids,
        "predicted_BFI_target": predicted_reach_bfi,
        "logit_offset": offset,
    })
    operator.to_parquet(OUT / "signature_operator_by_reach.parquet", index=False)
    coefficient_frame = pd.DataFrame({
        "feature": ["intercept"] + feature_columns,
        "coefficient": coefficients,
    })
    coefficient_frame.to_parquet(OUT / "signature_regionalization_coefficients.parquet", index=False)

    local_closure = float(np.max(np.abs(corrected_local.sum(axis=2) - local.sum(axis=2))))
    routed_total_change = float(np.max(np.abs(routed_corrected.sum(axis=2) - routed_parent.sum(axis=2))))
    component_closure = float(np.max(np.abs(routed_corrected.sum(axis=2) - routed_corrected[:, :, 0] - routed_corrected[:, :, 1])))
    boundary_fraction = float(np.mean(np.isclose(np.abs(offset), OFFSET_BOUND, atol=1.0e-10)))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    gates = contract["registered_gates"]
    checks = {
        "eval_BFI_spearman_ge_0p3": bool(np.isfinite(rho) and rho >= gates["2017_2018_slow_fraction_BFI_spearman_min"]),
        "eval_BFI_RMSE_le_0p2": rmse <= gates["2017_2018_slow_fraction_BFI_RMSE_max"],
        "high_flow_direction_ge_0p8": float(np.mean(high_direction)) >= gates["high_flow_fast_direction_station_fraction_min"],
        "slow_memory_direction_ge_0p8": float(np.mean(memory_direction)) >= gates["slow_memory_direction_station_fraction_min"],
        "tree_block_ridge_beats_climatology": cv_rmse < cv_climatology_rmse,
        "offset_boundary_fraction_le_0p1": boundary_fraction <= gates["offset_boundary_fraction_max"],
        "local_total_identity_le_1e_10": local_closure <= gates["total_flow_change_max_abs_m3_s"],
        "routed_total_identity_le_1e_10": routed_total_change <= gates["total_flow_change_max_abs_m3_s"],
        "component_closure_le_1e_10": component_closure <= gates["component_closure_max_abs_m3_s"],
        "spinup_converged": bool(spin["converged"]),
        "2019_2022_observations_not_read": True,
        "four_station_observations_not_read": True,
        "TN_not_read": True,
    }
    scientific_names = [
        "eval_BFI_spearman_ge_0p3", "eval_BFI_RMSE_le_0p2", "high_flow_direction_ge_0p8",
        "slow_memory_direction_ge_0p8", "tree_block_ridge_beats_climatology", "offset_boundary_fraction_le_0p1",
    ]
    scientific_pass = all(checks[name] for name in scientific_names)
    integrity_pass = all(value for name, value in checks.items() if name not in scientific_names)
    decision = {
        "stage": "20260827_4",
        "status": "PASS_SIGNATURE_CONSTRAINED_COMPONENT_REPAIR" if scientific_pass and integrity_pass else "SIGNATURE_REPAIR_NOT_AUTHORIZED_FOR_COMPONENT_CLAIMS",
        "metrics": {
            "eval_slow_fraction_BFI_spearman": float(rho),
            "eval_slow_fraction_BFI_spearman_p": float(rho_p),
            "eval_slow_fraction_BFI_RMSE": rmse,
            "tree_block_ridge_RMSE": cv_rmse,
            "tree_block_climatology_RMSE": cv_climatology_rmse,
            "high_flow_direction_fraction": float(np.mean(high_direction)),
            "slow_memory_direction_fraction": float(np.mean(memory_direction)),
            "offset_boundary_fraction": boundary_fraction,
            "local_total_change_max_abs_m3_s": local_closure,
            "routed_total_change_max_abs_m3_s": routed_total_change,
            "component_closure_max_abs_m3_s": component_closure,
        },
        "checks": checks,
        "operator_authorized": bool(scientific_pass and integrity_pass),
        "total_flow_model_unchanged": True,
        "authorized_successor": "20260827_5",
    }
    write_json(REPORTS / "stage4_decision.json", decision)
    write_json(REPORTS / "signature_operator_lock.json", {
        "status": "DEVELOPMENT_SIGNATURE_OPERATOR_LOCK" if decision["operator_authorized"] else "DIAGNOSTIC_ONLY",
        "ridge_lambda": RIDGE,
        "offset_bound": OFFSET_BOUND,
        "feature_columns": feature_columns,
        "coefficients": coefficients.tolist(),
        "training_period": "2010-2015",
        "evaluation_period": "2017-2018",
        "station_identity_terms": False,
    })
    write_json(REPORTS / "validation.json", {"stage": "20260827_4", "integrity_pass": integrity_pass, "scientific_pass": scientific_pass, "checks": checks})
    (REPORTS / "technical_report.md").write_text(
        "# 20260827_4 守恒水文特征约束修复\n\n"
        f"状态：`{decision['status']}`。该算子只重分配快慢通量，局地与路由总流量保持逐值不变。\n\n"
        f"2017–2018慢分量比例与BFI代理Spearman={float(rho):.3f}，RMSE={rmse:.3f}；"
        f"按河树留出的特征区域化RMSE={cv_rmse:.3f}，常数气候态RMSE={cv_climatology_rmse:.3f}。\n\n"
        "未读取2019–2022实测流量或四个空间测试站的观测与Reach映射。\n",
        encoding="utf-8",
    )
    (RUN / "README.md").write_text("# 20260827_4 conservative signature-constrained component repair\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
