"""Block-coordinate numerical repair for the F25 H7 joint optimum."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"E:\SPARROW")
HERE = ROOT / "5_Test/20260904_4/scripts"
RUN = ROOT / "5_Test/20260904_7"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
sys.path.insert(0, str(HERE))
from f25_common import atomic_json, atomic_parquet, observations  # noqa: E402
from finalize_f25 import metrics, reach_monthly_product, sha256  # noqa: E402
from unified_fit import GAMMA_PRIOR_SD, SITE_RIDGE, FoldDesign, JointObjective, UnifiedTNModel, predict, s44  # noqa: E402


def solve_head(objective: JointObjective, physical: np.ndarray, gamma0: np.ndarray, site0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        _, base = objective.model.evaluate(
            objective.train, torch.tensor(physical), min(objective.training_years), max(objective.training_years)
        )
    base = base.detach()
    sigma = torch.exp(torch.tensor(physical[objective.model.names().index("log_sigma")]))
    gamma = torch.nn.Parameter(torch.tensor(gamma0.copy()))
    site = torch.nn.Parameter(torch.tensor(site0.copy()))

    def loss() -> torch.Tensor:
        population = base + objective.design.effect(gamma)[objective.row_reach]
        effects = objective.site_effect(site)
        conditional = population + effects[objective.row_station]
        data = 0.90 * objective.layer_loss(population, sigma) + 0.10 * objective.layer_loss(conditional, sigma)
        gamma_prior = 0.5 * torch.mean((gamma / GAMMA_PRIOR_SD).square())
        site_prior = 0.10 * 0.5 * SITE_RIDGE * torch.sum(effects.square())
        return data + (gamma_prior + site_prior) / 119.0

    adam = torch.optim.AdamW([gamma, site], lr=0.02, weight_decay=0.0)
    for _ in range(200):
        adam.zero_grad(); value = loss(); value.backward(); adam.step()
    lbfgs = torch.optim.LBFGS(
        [gamma, site], lr=1.0, max_iter=400, tolerance_grad=1.0e-12,
        tolerance_change=1.0e-15, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(); value = loss(); value.backward(); return value

    lbfgs.step(closure)
    gamma.grad = None; site.grad = None
    final = loss(); final.backward()
    print(json.dumps({
        "stage": "fast_head_solve", "objective_without_process_constant": float(final.detach()),
        "gamma_gradient_max": float(torch.max(torch.abs(gamma.grad))),
        "site_gradient_max": float(torch.max(torch.abs(site.grad))),
    }), flush=True)
    return gamma.detach().numpy().copy(), site.detach().numpy().copy()


def full_check(
    objective: JointObjective, physical: np.ndarray, gamma: np.ndarray, site: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    raw = torch.nn.Parameter(objective.model.to_raw(physical))
    gamma_tensor = torch.nn.Parameter(torch.tensor(gamma.copy()))
    site_tensor = torch.nn.Parameter(torch.tensor(site.copy()))
    value = objective.loss(raw, gamma_tensor, site_tensor)
    value.backward()
    process_site = s44.s44.projected_kkt(objective.model, raw, raw.grad, site_tensor.grad)
    gamma_kkt = float(torch.max(torch.abs(gamma_tensor.grad)))
    result = {
        "objective": float(value.detach()),
        "process_kkt": process_site["process_projected_kkt_max"],
        "site_kkt": process_site["site_gradient_max"],
        "gamma_kkt": gamma_kkt,
        "combined_kkt": max(float(process_site["combined_max"]), gamma_kkt),
        "process_parameter_states": process_site["parameter_states"],
    }
    return result, raw.detach().numpy().copy()


def solve_process(
    objective: JointObjective, physical: np.ndarray, gamma: np.ndarray, site: np.ndarray
) -> np.ndarray:
    # Near the optimum the logit Jacobian makes an already small physical
    # gradient unnecessarily ill-conditioned.  Refine the six interior
    # process coordinates directly; beta_contact is at its valid lower KKT
    # boundary and v_f is fixed by the registered discrete profile.
    fixed_indices = {objective.model.names().index("beta_contact"), objective.model.names().index("v_f")}
    free_indices = [index for index in range(len(physical)) if index not in fixed_indices]
    free = torch.nn.Parameter(torch.tensor(physical[free_indices].copy()))
    gamma_tensor = torch.tensor(gamma)
    site_tensor = torch.tensor(site)

    def vector() -> torch.Tensor:
        values = []
        cursor = 0
        for index, value in enumerate(physical):
            if index in fixed_indices:
                values.append(torch.tensor(float(value)))
            else:
                values.append(free[cursor]); cursor += 1
        return torch.stack(values)

    def physical_loss() -> torch.Tensor:
        candidate = vector()
        named = dict(zip(objective.model.names(), candidate))
        _, base = objective.model.evaluate(
            objective.train, candidate, min(objective.training_years), max(objective.training_years)
        )
        population = base + objective.design.effect(gamma_tensor)[objective.row_reach]
        effects = objective.site_effect(site_tensor)
        conditional = population + effects[objective.row_station]
        sigma = torch.exp(named["log_sigma"])
        data = 0.90 * objective.layer_loss(population, sigma) + 0.10 * objective.layer_loss(conditional, sigma)
        process_prior = 0.5 * (
            ((named["beta_contact"] - 1.0) / 0.35) ** 2
            + (named["delta_path"] / 0.5) ** 2
            + (named["beta_low"] / 0.35) ** 2
            + (named["beta_high"] / 0.35) ** 2
        ) + objective.model.extra_prior(named)
        gamma_prior = 0.5 * torch.mean((gamma_tensor / GAMMA_PRIOR_SD).square())
        site_prior = 0.10 * 0.5 * SITE_RIDGE * torch.sum(effects.square())
        return data + (process_prior + gamma_prior + site_prior) / 119.0

    lbfgs = torch.optim.LBFGS(
        [free], lr=1.0, max_iter=24, tolerance_grad=1.0e-9,
        tolerance_change=1.0e-15, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(); value = physical_loss(); value.backward(); return value

    lbfgs.step(closure)
    fitted = vector().detach().numpy()
    for index, name in enumerate(objective.model.names()):
        fitted[index] = np.clip(fitted[index], objective.model.lower[name], objective.model.upper[name])
    return fitted


def publish(model: UnifiedTNModel, objective: JointObjective, physical: np.ndarray, gamma: np.ndarray, site_raw: np.ndarray, check: dict[str, object]) -> None:
    design = objective.design
    site_effect = objective.site_effect(torch.tensor(site_raw)).detach().numpy()
    reach_effect = design.effect(torch.tensor(gamma)).detach().numpy()
    result = {
        "physical": physical, "gamma": gamma, "site_raw": site_raw, "site_effect": site_effect,
        "reach_effect": reach_effect, "stations": objective.stations, "design": design,
        "selected_v_f": float(physical[model.names().index("v_f")]),
        "training_years": [2021, 2022, 2023, 2024, 2025],
    }
    obs = observations()
    evaluation = obs.loc[obs.year.between(2016, 2025)].copy()
    layers = predict(model, result, evaluation)
    frames = []
    base = evaluation[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
    for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
        frame = base.copy()
        frame["pred_tn_mg_l"] = np.where(np.isfinite(layers[layer]), np.maximum(np.expm1(layers[layer]), 0.0), np.nan)
        frame["layer"] = layer
        frame["period"] = np.where(frame.year.between(2021, 2025), "F25_calibration", "2016_2020_backcast")
        frame["product"] = "F25_TRAINING_SENSITIVITY"
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    metric_rows = []
    for period in ["F25_calibration", "2016_2020_backcast"]:
        subset = predictions.loc[predictions.period.eq(period)]
        for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
            metric_rows.append(metrics(subset, layer, period))
    metrics_frame = pd.DataFrame(metric_rows)
    parameters = pd.read_parquet(OUT / "f25_parameters.parquet")
    parameters.loc[0, "objective"] = check["objective"]
    parameters.loc[0, "projected_kkt_max"] = check["combined_kkt"]
    parameters.loc[0, "gamma_kkt"] = check["gamma_kkt"]
    for name, value in zip(model.names(), physical):
        parameters.loc[0, name] = float(value)
    gamma_frame = pd.DataFrame({"feature": design.fields, "gamma": gamma, "product": "F25_TRAINING_SENSITIVITY"})
    sites = pd.DataFrame({"station_key": objective.stations, "station_residual_log_unit": site_effect, "product": "F25_TRAINING_SENSITIVITY"})
    offsets = pd.DataFrame({"reach_id": np.arange(1, 231), "transferable_offset_log_unit": reach_effect, "product": "F25_TRAINING_SENSITIVITY"})
    reach = reach_monthly_product(model, result)
    for frame, path in [
        (predictions, OUT / "f25_station_predictions_2016_2025.parquet"),
        (metrics_frame, OUT / "f25_performance_metrics.parquet"),
        (parameters, OUT / "f25_parameters.parquet"),
        (gamma_frame, OUT / "f25_gamma_coefficients.parquet"),
        (sites, OUT / "f25_station_residuals.parquet"),
        (offsets, OUT / "f25_reach_transferable_offsets.parquet"),
        (reach, OUT / "f25_reach_monthly_1961_2025.parquet"),
    ]:
        atomic_parquet(frame, path)
    old_report = json.loads((REPORTS / "f25_training_report.json").read_text(encoding="utf-8"))
    carrier = model.carrier_diagnostics(torch.tensor(physical))
    old_report["status"] = "PASS_F25_TRAINING_SENSITIVITY" if check["combined_kkt"] <= 1.0e-5 else "F25_NUMERIC_REVIEW_REQUIRED"
    old_report["checks"]["projected_kkt_le_1e_5"] = bool(check["combined_kkt"] <= 1.0e-5)
    old_report["optimization"]["block_coordinate_repair"] = check
    old_report["optimization"]["final_objective"] = check["objective"]
    old_report["optimization"]["final_kkt"] = check["combined_kkt"]
    old_report["carrier_diagnostics"] = carrier
    old_report["metrics"] = metrics_frame.to_dict("records")
    paths = {
        "station_predictions": OUT / "f25_station_predictions_2016_2025.parquet",
        "metrics": OUT / "f25_performance_metrics.parquet", "parameters": OUT / "f25_parameters.parquet",
        "gamma": OUT / "f25_gamma_coefficients.parquet", "sites": OUT / "f25_station_residuals.parquet",
        "offsets": OUT / "f25_reach_transferable_offsets.parquet", "reach_monthly": OUT / "f25_reach_monthly_1961_2025.parquet",
    }
    old_report["output_hashes"] = {key: sha256(path) for key, path in paths.items()}
    atomic_json(old_report, REPORTS / "f25_training_report.json")
    atomic_json({
        "stage": "20260904_7", "status": old_report["status"], "product": "F25_TRAINING_SENSITIVITY",
        "quality_flags": old_report["quality_flags"], "report_sha256": sha256(REPORTS / "f25_training_report.json"),
    }, LOCKS / "f25_training_sensitivity_lock.json")
    rows = [
        "该版本用2021–2025 TN单次联合拟合过程参数、H7可迁移空间属性头和强收缩站点残差。", "",
        "固定限制：`PET_EXTENSION_CONFOUNDED`、`SOURCE_2025_CARRYFORWARD_CONFOUNDED`、`2025_TN_INCOMPLETE_DECEMBER`、`SENSITIVITY_ONLY`。", "",
        "| period | layer | station log-RMSE | RMSE | MAE | pooled NSE | pooled R² | KGE | PBIAS | spatial r |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_frame.itertuples(index=False):
        rows.append(f"| {row.period} | {row.layer} | {row.station_macro_log_rmse:.4f} | {row.pooled_rmse_mg_l:.3f} | {row.pooled_mae_mg_l:.3f} | {row.pooled_nse:.3f} | {row.pooled_r2:.3f} | {row.pooled_kge:.3f} | {row.pooled_pbias_percent:.1f}% | {row.station_mean_spatial_r:.3f} |")
    rows += ["", f"最终联合KKT：`{check['combined_kkt']:.3e}`；质量闭合相对误差：`{carrier['mass_balance_relative']:.3e}`。", ""]
    (REPORTS / "technical_report.md").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    started = time.perf_counter()
    train = observations().loc[lambda frame: frame.year.between(2021, 2025)].copy()
    model = UnifiedTNModel("sensitivity")
    physical_table = pd.read_parquet(OUT / "f25_parameters.parquet")
    physical = np.asarray([physical_table.iloc[0][name] for name in model.names()], dtype=float)
    model.set_fixed_v_f(float(physical[model.names().index("v_f")]))
    design = FoldDesign.build("H7", train)
    objective = JointObjective(model, design, train, [2021, 2022, 2023, 2024, 2025])
    gamma_table = pd.read_parquet(OUT / "f25_gamma_coefficients.parquet").set_index("feature")
    gamma = np.asarray([gamma_table.loc[field, "gamma"] for field in design.fields], dtype=float)
    site_map = pd.read_parquet(OUT / "f25_station_residuals.parquet").set_index("station_key").station_residual_log_unit.to_dict()
    site = np.asarray([site_map[station] for station in objective.stations], dtype=float)
    for cycle in range(3):
        gamma, site = solve_head(objective, physical, gamma, site)
        check, _ = full_check(objective, physical, gamma, site)
        print(json.dumps({"stage": "full_kkt_check", "cycle": cycle, **check}), flush=True)
        if check["combined_kkt"] <= 1.0e-5:
            break
        if check["process_kkt"] > 1.0e-5:
            physical = solve_process(objective, physical, gamma, site)
    gamma, site = solve_head(objective, physical, gamma, site)
    check, _ = full_check(objective, physical, gamma, site)
    check["cycles"] = cycle + 1
    check["runtime_seconds"] = time.perf_counter() - started
    print(json.dumps({"stage": "final_full_kkt", **check}), flush=True)
    publish(model, objective, physical, gamma, site, check)
    if check["combined_kkt"] > 1.0e-5:
        raise RuntimeError("F25_NUMERIC_REVIEW_REQUIRED_AFTER_BLOCK_REPAIR")


if __name__ == "__main__":
    main()
