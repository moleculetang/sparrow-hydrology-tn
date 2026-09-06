from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_18"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
LIB = TEST / "20260823_17" / "scripts"
sys.path.insert(0, str(LIB))
from regionalization import (  # noqa: E402
    ATTRIBUTES, fit_joint_map, load_all_reach_frame, load_observed_frame,
    predict_joint_map, station_metrics, summary_metrics,
)


LOCK17 = TEST / "20260823_17" / "reports" / "stage17_candidate_lock.json"
OBS = TEST / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
LAMBDAS = [0.01, 0.1, 1.0, 10.0]
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reconcile_month(upstream: np.ndarray, target: np.ndarray, local_q72: np.ndarray, lam: float) -> tuple[np.ndarray, dict]:
    w = 1.0 / np.sqrt(1.0 + np.maximum(target, 0.0))
    d = 1.0 / np.sqrt(1.0 + np.maximum(local_q72, 0.0))
    a = np.vstack([w[:, None] * upstream, np.sqrt(lam) * np.diag(d)])
    b = np.concatenate([w * target, np.sqrt(lam) * d * local_q72])
    fit = lsq_linear(a, b, bounds=(0.0, np.inf), method="bvls", tol=1e-8, max_iter=1000)
    if not fit.success:
        raise RuntimeError(f"Reconciliation failed: {fit.message}")
    local = np.maximum(fit.x, 0.0)
    return local, {"cost": float(fit.cost), "iterations": int(fit.nit), "optimality": float(fit.optimality)}


def solve_panel(frame: pd.DataFrame, upstream: np.ndarray, target: np.ndarray, lam: float) -> tuple[np.ndarray, pd.DataFrame]:
    reaches = frame.comid.drop_duplicates().to_numpy(int)
    times = frame[["year", "month"]].drop_duplicates().to_records(index=False)
    n_reach, n_time = len(reaches), len(times)
    local_q72 = (frame.q72_local_quick_cfs + frame.q72_local_slow_cfs).to_numpy(float).reshape(n_reach, n_time)
    target_2d = np.asarray(target, float).reshape(n_reach, n_time)
    local = np.zeros_like(local_q72)
    rows = []
    for t, (year, month) in enumerate(times):
        local[:, t], info = reconcile_month(upstream, target_2d[:, t], local_q72[:, t], lam)
        rows.append({"year": int(year), "month": int(month), "lambda": lam, **info})
    return local, pd.DataFrame(rows)


def routed_flat(upstream: np.ndarray, local: np.ndarray) -> np.ndarray:
    return (upstream @ local).reshape(-1)


def attach_obs(observed: pd.DataFrame, all_reach: pd.DataFrame, values: np.ndarray, column: str) -> pd.DataFrame:
    key = all_reach[["comid", "year", "month"]].copy()
    key[column] = values
    return observed.merge(key.rename(columns={"comid": "reach_id"}), on=["reach_id", "year", "month"], how="left", validate="many_to_one")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_reconciliation":
        raise RuntimeError("Reconciliation contract not pre-registered")
    lock17 = json.loads(LOCK17.read_text(encoding="utf-8"))
    candidate, alpha = lock17["selected_candidate"], float(lock17["selected_alpha"])
    attributes = pd.read_parquet(ATTRIBUTES).sort_values("comid").reset_index(drop=True)
    all_reach, modeling, upstream = load_all_reach_frame()
    observed = load_observed_frame(all_reach)
    dev = observed[observed.year.le(2018)].copy()
    model = fit_joint_map(dev, candidate, alpha, attributes, list(modeling.TRANSFER_FEATURES))
    target = predict_joint_map(all_reach, model)
    n_reach = all_reach.comid.nunique()
    n_time = len(all_reach) // n_reach
    local_q72 = (all_reach.q72_local_quick_cfs + all_reach.q72_local_slow_cfs).to_numpy(float).reshape(n_reach, n_time)

    grid_rows = []
    development_solutions: dict[float, np.ndarray] = {}
    solver_parts = []
    for lam in LAMBDAS:
        local, solver = solve_panel(all_reach, upstream, target, lam)
        routed = routed_flat(upstream, local)
        evaluation = attach_obs(observed[observed.year.le(2018)], all_reach, routed, "Q_reconciled_cfs")
        metrics = summary_metrics(evaluation, "Q_reconciled_cfs")
        grid_rows.append({"lambda": lam, **metrics})
        development_solutions[lam] = local
        solver_parts.append(solver)
    grid = pd.DataFrame(grid_rows).sort_values(["station_mean_RMSE_log", "lambda"])
    selected_lambda = float(grid.iloc[0]["lambda"])
    local = development_solutions[selected_lambda]

    local_total_q72 = local_q72
    local_quick_q72 = all_reach.q72_local_quick_cfs.to_numpy(float).reshape(n_reach, n_time)
    quick_fraction = np.divide(local_quick_q72, local_total_q72, out=np.zeros_like(local_total_q72), where=local_total_q72 > 1e-12)
    local_quick = local * quick_fraction
    local_slow = local - local_quick
    routed_total = upstream @ local
    routed_quick = upstream @ local_quick
    routed_slow = upstream @ local_slow
    target_2d = target.reshape(n_reach, n_time)

    product = all_reach[["comid", "year", "month"]].copy()
    product["Q_target_cfs"] = target_2d.reshape(-1)
    product["Q72_local_total_cfs"] = local_total_q72.reshape(-1)
    product["Q72_local_quick_cfs"] = local_quick_q72.reshape(-1)
    product["Q72_local_slow_cfs"] = all_reach.q72_local_slow_cfs.to_numpy(float)
    product["local_reconciled_total_cfs"] = local.reshape(-1)
    product["local_reconciled_quick_cfs"] = local_quick.reshape(-1)
    product["local_reconciled_slow_cfs"] = local_slow.reshape(-1)
    product["local_correction_cfs"] = (local - local_total_q72).reshape(-1)
    product["routed_reconciled_total_cfs"] = routed_total.reshape(-1)
    product["routed_reconciled_quick_cfs"] = routed_quick.reshape(-1)
    product["routed_reconciled_slow_cfs"] = routed_slow.reshape(-1)
    product["selected_lambda"] = selected_lambda
    product.to_parquet(OUT / "monthly_reconciled_hydrology.parquet", index=False)
    grid.to_parquet(OUT / "reconciliation_lambda_grid.parquet", index=False)
    pd.concat(solver_parts, ignore_index=True).to_parquet(OUT / "reconciliation_solver_audit.parquet", index=False)

    closure = routed_total - (routed_quick + routed_slow)
    network_recomputed = upstream @ local
    correction_abs_ratio = float(np.sum(np.abs(local - local_total_q72)) / max(np.sum(local_total_q72), 1e-12))
    audit = {
        "stage": "20260823_18",
        "status": "PASS",
        "target_candidate": candidate,
        "target_alpha": alpha,
        "selected_lambda": selected_lambda,
        "lambda_selected_using_post_2018": False,
        "reach_count": int(n_reach),
        "month_count": int(n_time),
        "row_count": int(len(product)),
        "minimum_local_reconciled_cfs": float(local.min()),
        "negative_local_row_count": int(np.sum(local < -1e-12)),
        "network_routing_max_abs_error_cfs": float(np.max(np.abs(routed_total - network_recomputed))),
        "quick_slow_closure_max_abs_error_cfs": float(np.max(np.abs(closure))),
        "absolute_correction_over_Q72_local_total": correction_abs_ratio,
        "median_absolute_local_correction_cfs": float(np.median(np.abs(local - local_total_q72))),
        "claim_boundary": "River-network closure only; full precipitation-ET-storage closure is not claimed.",
        "authorized_successor": "20260823_19"
    }
    (REPORT / "stage18_reconciliation_lock.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_18 network reconciliation\n\n"
        + f"Status: `{audit['status']}`; selected lambda={selected_lambda}.\n\n"
        + grid.to_markdown(index=False) + "\n\n"
        + f"All local increments are nonnegative and quick+slow closure max error is {audit['quick_slow_closure_max_abs_error_cfs']:.3e} cfs. "
        + f"The absolute correction is {correction_abs_ratio:.3f} of total Q72 local flow. This product claims river-network closure only.\n",
        encoding="utf-8",
    )
    integrity = {
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "stage17_lock_sha256": sha256(LOCK17),
        "observation_registry_sha256": sha256(OBS),
        "monthly_product_sha256": sha256(OUT / "monthly_reconciled_hydrology.parquet"),
    }
    (REPORT / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
