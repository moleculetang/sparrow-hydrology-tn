from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import hierarchical19_shared as h


def numerical_hessian(function, point: np.ndarray, step: np.ndarray) -> np.ndarray:
    n = len(point)
    out = np.zeros((n, n), dtype=float)
    f0 = float(function(point))
    for i in range(n):
        ei = np.zeros(n); ei[i] = step[i]
        out[i, i] = (function(point + ei) - 2.0 * f0 + function(point - ei)) / step[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = step[j]
            value = (
                function(point + ei + ej) - function(point + ei - ej)
                - function(point - ei + ej) + function(point - ei - ej)
            ) / (4.0 * step[i] * step[j])
            out[i, j] = out[j, i] = value
    return out


def laplace_diagnostic(router: h.HydraulicRouter, obs: pd.DataFrame, locked_vf: float,
                       locked_eta: np.ndarray, shared) -> dict[str, object]:
    initial_frame = router.frame(obs, np.full(len(router.reach_ids), locked_vf))
    initial_raw = shared.concentration(initial_frame, locked_eta)
    observed = np.log1p(obs.tn_mg_l.to_numpy(float))
    sigma = float(np.sqrt(np.mean((np.log1p(np.maximum(initial_raw, 0.0)) - observed) ** 2)))
    sigma = max(sigma, 1e-6)

    def nll(theta: np.ndarray) -> float:
        vf, eta_q, eta_g = map(float, theta)
        if not (0 <= vf <= 0.5 and 0 <= eta_q <= 1 and 0 <= eta_g <= 1):
            return 1e100
        frame = router.frame(obs, np.full(len(router.reach_ids), vf))
        raw = shared.concentration(frame, np.array([eta_q, eta_g]))
        residual = (np.log1p(np.maximum(raw, 0.0)) - observed) / sigma
        return float(0.5 * np.sum(residual ** 2) + 0.5 * ((eta_q - 1) ** 2 + (eta_g - 1) ** 2))

    opt = minimize(
        nll, np.array([locked_vf, *locked_eta]), method="L-BFGS-B",
        bounds=[(1e-8, 0.499999), (1e-8, 0.999999), (1e-8, 0.999999)],
        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 200},
    )
    point = np.asarray(opt.x, dtype=float)
    hessian = numerical_hessian(nll, point, np.array([1e-4, 1e-4, 1e-4]))
    eigenvalues = np.linalg.eigvalsh(hessian)
    positive = bool(np.all(eigenvalues > 0))
    covariance = np.linalg.inv(hessian) if positive else np.linalg.pinv(hessian)
    return {
        "map": point.tolist(), "covariance": covariance.tolist(),
        "hessian_eigenvalues": eigenvalues.tolist(), "positive_definite": positive,
        "optimizer_success": bool(opt.success), "optimizer_nfev": int(opt.nfev),
        "residual_sigma_log1p": sigma,
    }


def main() -> None:
    h.require_runtime()
    decision = json.loads((h.REPORTS / "spatial_hydraulic_decision.json").read_text(encoding="utf-8"))
    mechanism_lock = json.loads((h.REPORTS / "development_mechanism_lock.json").read_text(encoding="utf-8"))
    if decision["selected_process_structure"] != "H1_GLOBAL" or mechanism_lock["selected_process_structure"] != "H1_GLOBAL":
        raise RuntimeError("STOP_MECHANISM_NOT_LOCKED")
    obs = h.development_observations()
    shared = h.parent_shared()
    parameter_rows = []
    effect_rows = []
    laplace_rows = []
    for model_id in h.FORMAL_MODELS:
        router = h.build_router(model_id, shared)
        fit = h.fit_structure(router, obs, "H1_GLOBAL", shared)
        vf = np.asarray(fit["vf"])
        train = router.frame(obs, vf)
        p1 = shared.predict_layer(train, "P1", np.asarray(fit["eta"]), {})
        p2_fit = h.fit_selected_readout(train, "P2", shared)
        p2 = h.predict_selected(train, "P2", p2_fit, shared)
        parent_train = router.frame(obs, np.zeros(len(router.reach_ids)))
        parent_p1_fit = h.fit_selected_readout(parent_train, "P1", shared)
        parent_p2_fit = h.fit_selected_readout(parent_train, "P2", shared)
        parameter_rows.append({
            "model_id": model_id, "process_structure": "H1_GLOBAL",
            "v_f_m_per_day": float(fit["parameters"][0]),
            "P1_eta_quick": float(fit["eta"][0]), "P1_eta_gw": float(fit["eta"][1]),
            "P2_eta_quick": float(p2_fit["eta"][0]), "P2_eta_gw": float(p2_fit["eta"][1]),
            "H0_P1_eta_quick": float(parent_p1_fit["eta"][0]), "H0_P1_eta_gw": float(parent_p1_fit["eta"][1]),
            "H0_P2_eta_quick": float(parent_p2_fit["eta"][0]), "H0_P2_eta_gw": float(parent_p2_fit["eta"][1]),
            "P1_station_macro_rmse_log1p_training": h.station_macro_rmse(p1),
            "P1_tree_macro_rmse_log1p_training": h.tree_macro_rmse(p1),
            "P2_station_macro_rmse_log1p_training": h.station_macro_rmse(p2),
            "P2_tree_macro_rmse_log1p_training": h.tree_macro_rmse(p2),
            "vf_boundary": bool(float(fit["parameters"][0]) >= 0.49),
            "P1_eta_boundary": bool(fit["diagnostic"]["eta_boundary"]),
            "P2_eta_boundary": bool(p2_fit["diagnostic"]["eta_boundary"]),
        })
        for station, value in p2_fit["effects"].items():
            effect_rows.append({"model_id": model_id, "mechanism": "H1_GLOBAL", "station_key": station, "station_effect": float(value)})
        for station, value in parent_p2_fit["effects"].items():
            effect_rows.append({"model_id": model_id, "mechanism": "H0_PARENT", "station_key": station, "station_effect": float(value)})
        laplace = laplace_diagnostic(router, obs, float(fit["parameters"][0]), np.asarray(fit["eta"]), shared)
        laplace_rows.append({
            "model_id": model_id, "parameter_order": "v_f,eta_quick,eta_gw",
            "map_json": json.dumps(laplace["map"]), "covariance_json": json.dumps(laplace["covariance"]),
            "hessian_eigenvalues_json": json.dumps(laplace["hessian_eigenvalues"]),
            "positive_definite": laplace["positive_definite"],
            "optimizer_success": laplace["optimizer_success"], "optimizer_nfev": laplace["optimizer_nfev"],
            "residual_sigma_log1p": laplace["residual_sigma_log1p"],
        })
        print(json.dumps({"full_development_complete": model_id}), flush=True)
    parameters = pd.DataFrame(parameter_rows)
    effects = pd.DataFrame(effect_rows)
    laplace = pd.DataFrame(laplace_rows)
    parameters.to_parquet(h.OUT / "full_development_parameters.parquet", index=False)
    effects.to_parquet(h.OUT / "full_development_station_effects.parquet", index=False)
    laplace.to_parquet(h.OUT / "posterior_laplace_diagnostics.parquet", index=False)

    temporal = pd.read_parquet(h.OUT / "temporal_performance.parquet")
    p1_h1 = temporal.loc[temporal.layer.eq("P1") & temporal.mechanism.eq("H1_GLOBAL")]
    median_rmse = float(p1_h1.station_macro_rmse_log1p.median())
    p1_h1 = p1_h1.assign(distance=(p1_h1.station_macro_rmse_log1p - median_rmse).abs()).sort_values(["distance", "model_id"])
    representative = str(p1_h1.iloc[0].model_id)
    lock = {
        "lock": "full_development_parameter_lock", "written_before_2022_TN": True,
        "selected_process_structure": "H1_GLOBAL", "temperature": "closed_not_used",
        "models": 12, "representative_model_for_MCMC": representative,
        "v_f_range_m_per_day": [float(parameters.v_f_m_per_day.min()), float(parameters.v_f_m_per_day.max())],
        "all_laplace_hessians_positive_definite": bool(laplace.positive_definite.all()),
        "all_laplace_optimizers_successful": bool(laplace.optimizer_success.all()),
        "TN_2022_read": False,
    }
    h.dump_json(h.REPORTS / "full_development_parameter_lock.json", lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
