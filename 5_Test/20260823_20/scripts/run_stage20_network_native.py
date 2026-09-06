from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_20"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
sys.path.insert(0, str(TEST / "20260823_17" / "scripts"))
from regionalization import (  # noqa: E402
    ATTRIBUTES, base_attribute_matrix, load_all_reach_frame, load_observed_frame,
    station_metrics, summary_metrics,
)


PARENT_STATE = TEST / "20260823_15" / "final_outputs" / "model_and_assimilation_state_audit.parquet"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
SMOOTH_GRID = [0.001, 0.01, 0.1, 1.0]
RIDGE_GRID = [0.001, 0.01, 0.1]
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class DataBundle:
    local_q72: np.ndarray
    upstream: np.ndarray
    x_global: np.ndarray
    dynamic: np.ndarray
    attributes: np.ndarray
    edges_i: np.ndarray
    edges_j: np.ndarray
    obs_reach: np.ndarray
    obs_time: np.ndarray
    obs_y: np.ndarray
    reach_ids: np.ndarray
    times: pd.DataFrame
    feature_names: list[str]
    attribute_names: list[str]
    global_mean: np.ndarray
    global_std: np.ndarray


def build_bundle(all_reach: pd.DataFrame, observed: pd.DataFrame, upstream: np.ndarray, attrs: pd.DataFrame, fit_mask: pd.Series, features: list[str]) -> DataBundle:
    reaches = all_reach.comid.drop_duplicates().to_numpy(int)
    times = all_reach[["year", "month"]].drop_duplicates().reset_index(drop=True)
    n_reach, n_time = len(reaches), len(times)
    training = observed.loc[fit_mask]
    mean = training[features].mean().to_numpy(float)
    std = training[features].std(ddof=0).replace(0, 1.0).to_numpy(float)
    raw = all_reach[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    z = ((raw - mean) / std).reshape(n_reach, n_time, len(features))
    x_global = np.concatenate([np.ones((n_reach, n_time, 1)), z], axis=2)
    f_index = {name: i for i, name in enumerate(features)}
    dynamic = np.stack([
        np.ones((n_reach, n_time)), z[:, :, f_index["log_q72_total"]],
        z[:, :, f_index["q72_quick_fraction"]], z[:, :, f_index["month_sin"]],
        z[:, :, f_index["month_cos"]],
    ], axis=2)
    attribute_matrix, attribute_names = base_attribute_matrix(attrs)
    topo = pd.read_csv(TOPOLOGY)
    index = {int(r): i for i, r in enumerate(reaches)}
    ei, ej = [], []
    for row in topo.itertuples():
        if pd.isna(row.downstream_reach):
            continue
        ei.append(index[int(row.reach_id)])
        ej.append(index[int(row.downstream_reach)])
    obs = training.copy()
    time_index = {(int(row.year), int(row.month)): i for i, row in times.iterrows()}
    obs_reach = obs.reach_id.astype(int).map(index).to_numpy(int)
    obs_time = np.array([time_index[(int(y), int(m))] for y, m in zip(obs.year, obs.month)], int)
    local = (all_reach.q72_local_quick_cfs + all_reach.q72_local_slow_cfs).to_numpy(float).reshape(n_reach, n_time)
    return DataBundle(
        local_q72=local, upstream=np.asarray(upstream, float), x_global=x_global, dynamic=dynamic,
        attributes=attribute_matrix, edges_i=np.array(ei, int), edges_j=np.array(ej, int),
        obs_reach=obs_reach, obs_time=obs_time, obs_y=np.log1p(obs.Q_obsv_cfs.to_numpy(float)),
        reach_ids=reaches, times=times, feature_names=["intercept", *features], attribute_names=attribute_names,
        global_mean=mean, global_std=std,
    )


class NetworkNativeMAP5:
    def __init__(self, bundle: DataBundle, smooth: float, ridge: float, attribute_precision: float = 0.01, global_sigma: float = 0.25):
        self.d = bundle
        self.smooth = float(smooth)
        self.ridge = float(ridge)
        self.attribute_precision = float(attribute_precision)
        self.global_sigma = float(global_sigma)
        self.ng = bundle.x_global.shape[2]
        self.na = bundle.attributes.shape[1]
        self.nr = len(bundle.reach_ids)
        self.np = self.ng + self.na * 5 + self.nr * 5

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = theta[: self.ng]
        gamma = theta[self.ng : self.ng + self.na * 5].reshape(self.na, 5)
        u = theta[self.ng + self.na * 5 :].reshape(self.nr, 5)
        return c, gamma, u

    def forward(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c, gamma, u = self.unpack(theta)
        b = self.d.attributes @ gamma + u
        s = np.einsum("rtg,g->rt", self.d.x_global, c) + np.einsum("rtk,rk->rt", self.d.dynamic, b)
        clipped = np.clip(s, -8.0, 8.0)
        local = self.d.local_q72 * np.exp(clipped)
        routed = self.d.upstream @ local
        return routed, local, b, (np.abs(s) < 8.0)

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
        edge_scale = max(len(self.d.edges_i) * 5, 1)
        loss_smooth = self.smooth * float(np.sum(np.square(diff)) / edge_scale)
        grad_u_prior = np.zeros_like(u)
        edge_grad = 2.0 * self.smooth * diff / edge_scale
        np.add.at(grad_u_prior, self.d.edges_i, edge_grad)
        np.add.at(grad_u_prior, self.d.edges_j, -edge_grad)
        loss_ridge = self.ridge * float(np.mean(np.square(u)))
        grad_u_prior += 2.0 * self.ridge * u / u.size
        loss_attribute = self.attribute_precision * float(np.mean(np.square(gamma)))
        grad_gamma_prior = 2.0 * self.attribute_precision * gamma / gamma.size
        loss_global = float(np.sum(np.square(c[1:])) / self.global_sigma**2 / max(self.ng - 1, 1))
        grad_c[1:] += 2.0 * c[1:] / self.global_sigma**2 / max(self.ng - 1, 1)

        grad_gamma = self.d.attributes.T @ grad_b + grad_gamma_prior
        grad_u = grad_b + grad_u_prior
        loss = loss_data + loss_smooth + loss_ridge + loss_attribute + loss_global
        grad = np.concatenate([grad_c, grad_gamma.ravel(), grad_u.ravel()])
        return float(loss), grad

    def fit(self, initial: np.ndarray | None = None, maxiter: int = 250) -> tuple[np.ndarray, dict]:
        x0 = np.zeros(self.np, float) if initial is None else np.asarray(initial, float).copy()
        result = minimize(
            fun=lambda x: self.objective(x), x0=x0, jac=True, method="L-BFGS-B",
            bounds=[(-3.0, 3.0)] * self.np,
            options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-7, "maxls": 50},
        )
        info = {
            "success": bool(result.success), "status": int(result.status), "message": str(result.message),
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "objective": float(result.fun), "gradient_max_abs": float(np.max(np.abs(result.jac))),
            "parameter_boundary_count": int(np.sum(np.abs(result.x) >= 2.999)),
        }
        return result.x, info


def predict_for_observed(observed: pd.DataFrame, all_reach: pd.DataFrame, routed: np.ndarray, column: str) -> pd.DataFrame:
    key = all_reach[["comid", "year", "month"]].copy()
    key[column] = routed.reshape(-1)
    return observed.merge(key.rename(columns={"comid": "reach_id"}), on=["reach_id", "year", "month"], how="left", validate="many_to_one")


def gradient_audit(model: NetworkNativeMAP5, theta: np.ndarray, n_checks: int = 12) -> dict:
    loss, grad = model.objective(theta)
    rng = np.random.default_rng(20260823)
    indices = rng.choice(len(theta), n_checks, replace=False)
    errors = []
    step = 1e-6
    for idx in indices:
        plus, minus = theta.copy(), theta.copy()
        plus[idx] += step
        minus[idx] -= step
        numerical = (model.objective(plus)[0] - model.objective(minus)[0]) / (2 * step)
        scale = max(1e-8, abs(numerical), abs(grad[idx]))
        errors.append(abs(numerical - grad[idx]) / scale)
    return {"base_loss": loss, "checks": n_checks, "max_relative_error": float(max(errors)), "median_relative_error": float(np.median(errors))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_fitting":
        raise RuntimeError("Stage 20 contract not pre-registered")
    attrs = pd.read_parquet(ATTRIBUTES).sort_values("comid").reset_index(drop=True)
    all_reach, modeling, upstream = load_all_reach_frame()
    observed = load_observed_frame(all_reach)
    features = list(modeling.TRANSFER_FEATURES)
    train_mask = observed.year.le(2014)
    validation_obs = observed[observed.year.between(2015, 2018)].copy()
    bundle_train = build_bundle(all_reach, observed, upstream, attrs, train_mask, features)
    zero_model = NetworkNativeMAP5(bundle_train, 0.01, 0.01)
    gradient = gradient_audit(zero_model, np.zeros(zero_model.np))
    if gradient["max_relative_error"] > 1e-4:
        raise RuntimeError(f"Analytic gradient audit failed: {gradient}")

    grid_rows, solver_rows = [], []
    warm: np.ndarray | None = None
    for smooth in SMOOTH_GRID:
        for ridge in RIDGE_GRID:
            model = NetworkNativeMAP5(bundle_train, smooth, ridge)
            theta, info = model.fit(initial=warm, maxiter=220)
            warm = theta
            routed, _, _, _ = model.forward(theta)
            evaluation = predict_for_observed(validation_obs, all_reach, routed, "Q_pred_cfs")
            metrics = summary_metrics(evaluation, "Q_pred_cfs")
            grid_rows.append({"smooth_precision": smooth, "ridge_precision": ridge, **metrics})
            solver_rows.append({"phase": "hyperparameter", "smooth_precision": smooth, "ridge_precision": ridge, **info})
            print(f"grid smooth={smooth}, ridge={ridge}, station_mean_RMSE_log={metrics['station_mean_RMSE_log']:.6f}", flush=True)
    grid = pd.DataFrame(grid_rows).sort_values(["station_mean_RMSE_log", "smooth_precision", "ridge_precision"])
    best = grid.iloc[0]
    smooth, ridge = float(best.smooth_precision), float(best.ridge_precision)

    final_mask = observed.year.le(2018)
    bundle_final = build_bundle(all_reach, observed, upstream, attrs, final_mask, features)
    final_model = NetworkNativeMAP5(bundle_final, smooth, ridge)
    theta, final_info = final_model.fit(initial=None, maxiter=350)
    solver_rows.append({"phase": "full_2006_2018", "smooth_precision": smooth, "ridge_precision": ridge, **final_info})
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
    for j, name in enumerate(["intercept", "log_q72_total", "q72_quick_fraction", "month_sin", "month_cos"]):
        params[name] = b[:, j]
        params[f"structured_residual_{name}"] = u[:, j]
    params.to_parquet(OUT / "reach_map5_parameters.parquet", index=False)
    pd.DataFrame({"feature": bundle_final.feature_names, "coefficient": c}).to_parquet(OUT / "global_network_native_parameters.parquet", index=False)
    gamma_frame = pd.DataFrame(gamma, index=bundle_final.attribute_names, columns=["intercept", "log_q72_total", "q72_quick_fraction", "month_sin", "month_cos"])
    gamma_frame.reset_index(names="attribute").to_parquet(OUT / "attribute_map5_parameters.parquet", index=False)
    grid.to_parquet(OUT / "network_native_hyperparameter_grid.parquet", index=False)
    pd.DataFrame(solver_rows).to_parquet(OUT / "network_native_solver_audit.parquet", index=False)

    check = observed[observed.year.ge(2019) & observed.selected_for_four_group_check].copy()
    check = predict_for_observed(check, all_reach, routed, "Q_NETWORK_NATIVE_cfs")
    check["Q72_cfs"] = check.q72_routed_total_cfs
    parent = pd.read_parquet(PARENT_STATE)[["station_norm", "year", "month", "Q_MAP_cfs"]]
    parent["q_site"] = parent.station_norm.astype(str)
    check = check.merge(parent[["q_site", "year", "month", "Q_MAP_cfs"]], on=["q_site", "year", "month"], validate="one_to_one")
    rows = []
    for model_name, col in [("LOCAL_STATION_MAP_UPPER_BOUND", "Q_MAP_cfs"), ("NETWORK_NATIVE_MAP5", "Q_NETWORK_NATIVE_cfs"), ("Q72", "Q72_cfs")]:
        rows.append({"model": model_name, **summary_metrics(check, col)})
    temporal = pd.DataFrame(rows)
    temporal.to_parquet(OUT / "locked_2019_2022_metrics.parquet", index=False)
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
        "stage": "20260823_20",
        "status": "NETWORK_NATIVE_TEMPORAL_RETENTION_PASS" if all(gates.values()) else "NETWORK_NATIVE_TEMPORAL_RETENTION_FAIL",
        "selected_smooth_precision": smooth,
        "selected_ridge_precision": ridge,
        "gradient_audit": gradient,
        "full_solver": final_info,
        "reach_count": int(len(params)),
        "free_station_identity_columns": 0,
        "minimum_local_flow_cfs": float(local.min()),
        "network_closure_max_abs_cfs": float(np.max(np.abs(routed - upstream @ local))),
        "quick_slow_closure_max_abs_cfs": float(np.max(np.abs(routed - routed_quick - routed_slow))),
        "multiplier_clip_fraction": float(1.0 - active.mean()),
        "temporal_retention_gates": gates,
        "external_four_stations_read": False,
    }
    (REPORT / "stage20_decision.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_20 network-native MAP5\n\n"
        + f"Status: `{audit['status']}`.\n\n## Development-only hyperparameter check\n\n"
        + grid.to_markdown(index=False) + "\n\n## Locked 2019-2022 check\n\n"
        + temporal.to_markdown(index=False) + "\n\n"
        + "The five Reach fields act on nonnegative local Q72 flow before frozen forward routing. "
        + "No station-identity design block or post-hoc inverse reconciliation is present.\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "monthly_product_sha256": sha256(OUT / "monthly_network_native_hydrology.parquet"),
        "reach_parameters_sha256": sha256(OUT / "reach_map5_parameters.parquet"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(temporal.to_string(index=False))


if __name__ == "__main__":
    main()
