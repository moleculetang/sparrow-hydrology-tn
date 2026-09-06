from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_21"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
sys.path.insert(0, str(TEST / "20260823_20" / "scripts"))
sys.path.insert(0, str(TEST / "20260823_17" / "scripts"))
from run_stage20_network_native import (  # noqa: E402
    ATTRIBUTES, PARENT_STATE, NetworkNativeMAP5, build_bundle, gradient_audit,
    load_all_reach_frame, load_observed_frame, predict_for_observed,
    station_metrics, summary_metrics,
)


SMOOTH_GRID = [0.1, 1.0, 10.0, 100.0]
RIDGE_GRID = [0.1, 1.0, 10.0]
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class CorrectedMAP5(NetworkNativeMAP5):
    def __init__(self, bundle, smooth: float, ridge: float):
        super().__init__(bundle, smooth, ridge, attribute_precision=1.0, global_sigma=0.25)

    def objective(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        c, gamma, u = self.unpack(theta)
        routed, local, _, active = self.forward(theta)
        pred = routed[self.d.obs_reach, self.d.obs_time]
        residual = np.log1p(pred) - self.d.obs_y
        n = max(len(residual), 1)
        loss_data = float(np.mean(np.square(residual)))
        grad_routed = np.zeros_like(routed)
        np.add.at(grad_routed, (self.d.obs_reach, self.d.obs_time), 2.0 * residual / (n * (1.0 + pred)))
        grad_local = self.d.upstream.T @ grad_routed
        grad_s = grad_local * local * active
        grad_c = np.einsum("rt,rtg->g", grad_s, self.d.x_global)
        grad_b = np.einsum("rt,rtk->rk", grad_s, self.d.dynamic)

        diff = u[self.d.edges_i] - u[self.d.edges_j]
        loss_smooth = self.smooth * float(np.sum(np.square(diff))) / n
        grad_u = np.zeros_like(u)
        edge_grad = 2.0 * self.smooth * diff / n
        np.add.at(grad_u, self.d.edges_i, edge_grad)
        np.add.at(grad_u, self.d.edges_j, -edge_grad)
        loss_ridge = self.ridge * float(np.sum(np.square(u))) / n
        grad_u += 2.0 * self.ridge * u / n
        loss_attribute = self.attribute_precision * float(np.sum(np.square(gamma))) / n
        grad_gamma = self.d.attributes.T @ grad_b + 2.0 * self.attribute_precision * gamma / n
        loss_global = float(np.sum(np.square(c[1:])) / self.global_sigma**2 / n)
        grad_c[1:] += 2.0 * c[1:] / self.global_sigma**2 / n
        grad_u += grad_b
        loss = loss_data + loss_smooth + loss_ridge + loss_attribute + loss_global
        return float(loss), np.concatenate([grad_c, grad_gamma.ravel(), grad_u.ravel()])

    def fit(self, initial: np.ndarray | None = None, maxiter: int = 800) -> tuple[np.ndarray, dict]:
        x0 = np.zeros(self.np, float) if initial is None else np.asarray(initial, float).copy()
        result = minimize(
            fun=lambda x: self.objective(x), x0=x0, jac=True, method="L-BFGS-B",
            bounds=[(-5.0, 5.0)] * self.np,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxls": 60, "maxcor": 30},
        )
        return result.x, {
            "success": bool(result.success), "status": int(result.status), "message": str(result.message),
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "objective": float(result.fun), "gradient_max_abs": float(np.max(np.abs(result.jac))),
            "parameter_boundary_count": int(np.sum(np.abs(result.x) >= 4.999)),
        }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_fitting":
        raise RuntimeError("Stage 21 contract not pre-registered")
    attrs = pd.read_parquet(ATTRIBUTES).sort_values("comid").reset_index(drop=True)
    all_reach, modeling, upstream = load_all_reach_frame()
    observed = load_observed_frame(all_reach)
    features = list(modeling.TRANSFER_FEATURES)
    train_mask = observed.year.le(2014)
    validation = observed[observed.year.between(2015, 2018)].copy()
    bundle_train = build_bundle(all_reach, observed, upstream, attrs, train_mask, features)
    audit_model = CorrectedMAP5(bundle_train, 1.0, 1.0)
    gradient = gradient_audit(audit_model, np.zeros(audit_model.np))
    if gradient["max_relative_error"] > 1e-4:
        raise RuntimeError(f"Gradient audit failed: {gradient}")

    rows, solvers, solutions = [], [], {}
    for smooth in SMOOTH_GRID:
        for ridge in RIDGE_GRID:
            model = CorrectedMAP5(bundle_train, smooth, ridge)
            theta, info = model.fit(initial=None, maxiter=550)
            routed, _, _, _ = model.forward(theta)
            evaluation = predict_for_observed(validation, all_reach, routed, "Q_pred_cfs")
            metrics = summary_metrics(evaluation, "Q_pred_cfs")
            selection_score = float(metrics["station_mean_RMSE_log"] + 0.2 * abs(metrics["PBIAS_pct"]) / 100.0)
            rows.append({"smooth_precision": smooth, "ridge_precision": ridge, "selection_score": selection_score, **metrics})
            solvers.append({"phase": "hyperparameter", "smooth_precision": smooth, "ridge_precision": ridge, **info})
            solutions[(smooth, ridge)] = theta
            print(f"grid smooth={smooth}, ridge={ridge}, score={selection_score:.6f}, station_RMSE={metrics['station_mean_RMSE_log']:.6f}", flush=True)
    grid = pd.DataFrame(rows).sort_values(["selection_score", "smooth_precision", "ridge_precision"])
    best = grid.iloc[0]
    smooth, ridge = float(best.smooth_precision), float(best.ridge_precision)

    bundle_final = build_bundle(all_reach, observed, upstream, attrs, observed.year.le(2018), features)
    final_model = CorrectedMAP5(bundle_final, smooth, ridge)
    starts = {
        "selected_2006_2014": solutions[(smooth, ridge)],
        "zero": np.zeros(final_model.np),
    }
    final_trials = []
    for name, start in starts.items():
        theta, info = final_model.fit(initial=start, maxiter=1200)
        final_trials.append((info["objective"], name, theta, info))
        solvers.append({"phase": f"full_{name}", "smooth_precision": smooth, "ridge_precision": ridge, **info})
        print(f"final start={name}, objective={info['objective']:.8f}, iterations={info['iterations']}, success={info['success']}", flush=True)
    _, selected_start, theta, final_info = min(final_trials, key=lambda item: item[0])
    routed, local, b, active = final_model.forward(theta)
    c, gamma, u = final_model.unpack(theta)

    n_reach, n_time = local.shape
    q72_local = bundle_final.local_q72
    q72_quick = all_reach.q72_local_quick_cfs.to_numpy(float).reshape(n_reach, n_time)
    quick_fraction = np.divide(q72_quick, q72_local, out=np.zeros_like(q72_local), where=q72_local > 1e-12)
    local_quick = local * quick_fraction
    routed_quick = upstream @ local_quick
    routed_slow = upstream @ (local - local_quick)
    product = all_reach[["comid", "year", "month"]].copy()
    product["Q72_local_total_cfs"] = q72_local.reshape(-1)
    product["local_network_native_total_cfs"] = local.reshape(-1)
    product["local_network_native_quick_cfs"] = local_quick.reshape(-1)
    product["local_network_native_slow_cfs"] = (local - local_quick).reshape(-1)
    product["routed_network_native_total_cfs"] = routed.reshape(-1)
    product["routed_network_native_quick_cfs"] = routed_quick.reshape(-1)
    product["routed_network_native_slow_cfs"] = routed_slow.reshape(-1)
    product["local_multiplier"] = np.divide(local, q72_local, out=np.ones_like(local), where=q72_local > 1e-12).reshape(-1)
    product.to_parquet(OUT / "monthly_network_native_hydrology.parquet", index=False)

    params = pd.DataFrame({"reach_id": bundle_final.reach_ids})
    effect_names = ["intercept", "log_q72_total", "q72_quick_fraction", "month_sin", "month_cos"]
    for j, name in enumerate(effect_names):
        params[name] = b[:, j]
        params[f"structured_residual_{name}"] = u[:, j]
    params.to_parquet(OUT / "reach_map5_parameters.parquet", index=False)
    pd.DataFrame({"feature": bundle_final.feature_names, "coefficient": c}).to_parquet(OUT / "global_network_native_parameters.parquet", index=False)
    pd.DataFrame(gamma, index=bundle_final.attribute_names, columns=effect_names).reset_index(names="attribute").to_parquet(
        OUT / "attribute_map5_parameters.parquet", index=False
    )
    grid.to_parquet(OUT / "corrected_prior_hyperparameter_grid.parquet", index=False)
    pd.DataFrame(solvers).to_parquet(OUT / "solver_audit.parquet", index=False)
    np.savez_compressed(OUT / "final_model_parameters.npz", theta=theta, global_mean=bundle_final.global_mean, global_std=bundle_final.global_std)

    check = observed[observed.year.ge(2019) & observed.selected_for_four_group_check].copy()
    check = predict_for_observed(check, all_reach, routed, "Q_NETWORK_NATIVE_cfs")
    check["Q72_cfs"] = check.q72_routed_total_cfs
    parent = pd.read_parquet(PARENT_STATE)[["station_norm", "year", "month", "Q_MAP_cfs"]]
    parent["q_site"] = parent.station_norm.astype(str)
    check = check.merge(parent[["q_site", "year", "month", "Q_MAP_cfs"]], on=["q_site", "year", "month"], validate="one_to_one")
    temporal_rows, station_parts = [], []
    for model_name, col in [("LOCAL_STATION_MAP_UPPER_BOUND", "Q_MAP_cfs"), ("NETWORK_NATIVE_MAP5", "Q_NETWORK_NATIVE_cfs"), ("Q72", "Q72_cfs")]:
        temporal_rows.append({"model": model_name, **summary_metrics(check, col)})
        s = station_metrics(check, col)
        s.insert(0, "model", model_name)
        station_parts.append(s)
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_parquet(OUT / "locked_2019_2022_metrics.parquet", index=False)
    pd.concat(station_parts, ignore_index=True).to_parquet(OUT / "locked_2019_2022_station_metrics.parquet", index=False)
    check[["q_site", "reach_id", "year", "month", "Q_obsv_cfs", "Q_MAP_cfs", "Q_NETWORK_NATIVE_cfs", "Q72_cfs"]].to_parquet(
        OUT / "locked_2019_2022_predictions.parquet", index=False
    )
    upper = temporal[temporal.model.eq("LOCAL_STATION_MAP_UPPER_BOUND")].iloc[0]
    net = temporal[temporal.model.eq("NETWORK_NATIVE_MAP5")].iloc[0]
    gates = {
        "pooled_NSE_decline_le_0_02": bool(upper.NSE - net.NSE <= 0.02),
        "station_median_NSE_decline_le_0_03": bool(upper.station_median_NSE - net.station_median_NSE <= 0.03),
        "log_RMSE_increase_le_0_03": bool(net.RMSE_log - upper.RMSE_log <= 0.03),
        "absolute_PBIAS_le_5_pct": bool(abs(net.PBIAS_pct) <= 5.0),
    }
    audit = {
        "stage": "20260823_21",
        "status": "PRIOR_SCALE_REPAIR_RETENTION_PASS" if all(gates.values()) else "PRIOR_SCALE_REPAIR_RETENTION_FAIL",
        "selected_smooth_precision": smooth, "selected_ridge_precision": ridge,
        "selected_final_initialization": selected_start, "gradient_audit": gradient,
        "full_solver": final_info, "reach_count": int(len(params)), "free_station_identity_columns": 0,
        "network_closure_max_abs_cfs": float(np.max(np.abs(routed - upstream @ local))),
        "quick_slow_closure_max_abs_cfs": float(np.max(np.abs(routed - routed_quick - routed_slow))),
        "multiplier_clip_fraction": float(1.0 - active.mean()),
        "temporal_retention_gates": gates, "external_four_stations_read": False,
    }
    (REPORT / "stage21_decision.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_21 prior-scale repair\n\n"
        + f"Status: `{audit['status']}`.\n\n## Corrected MAP hyperparameter check\n\n"
        + grid.to_markdown(index=False) + "\n\n## Locked 2019-2022 check\n\n"
        + temporal.to_markdown(index=False) + "\n\n"
        + "Only Gaussian-prior scaling and solver convergence were changed from stage 20. The network-native equation is unchanged.\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "monthly_product_sha256": sha256(OUT / "monthly_network_native_hydrology.parquet"),
        "parameter_sha256": sha256(OUT / "final_model_parameters.npz"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(temporal.to_string(index=False))


if __name__ == "__main__":
    main()
