"""Reusable block-coordinate refinement for the expensive unified TN core."""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from unified_fit import GAMMA_PRIOR_SD, SITE_RIDGE, JointObjective, s44


def physical_loss(
    objective: JointObjective,
    physical: torch.Tensor,
    gamma: torch.Tensor,
    site: torch.Tensor,
) -> torch.Tensor:
    named = dict(zip(objective.model.names(), physical))
    _, base = objective.model.evaluate(
        objective.train, physical, min(objective.training_years), max(objective.training_years)
    )
    population = base + objective.design.effect(gamma)[objective.row_reach]
    effects = objective.site_effect(site)
    conditional = population + effects[objective.row_station]
    sigma = torch.exp(named["log_sigma"])
    data = 0.90 * objective.layer_loss(population, sigma) + 0.10 * objective.layer_loss(conditional, sigma)
    process_prior = 0.5 * (
        ((named["beta_contact"] - 1.0) / 0.35) ** 2
        + (named["delta_path"] / 0.5) ** 2
        + (named["beta_low"] / 0.35) ** 2
        + (named["beta_high"] / 0.35) ** 2
    ) + objective.model.extra_prior(named)
    gamma_prior = 0.5 * torch.mean((gamma / GAMMA_PRIOR_SD).square()) if len(gamma) else torch.zeros(())
    site_prior = 0.10 * 0.5 * SITE_RIDGE * torch.sum(effects.square())
    return data + (process_prior + gamma_prior + site_prior) / 119.0


def solve_head(
    objective: JointObjective,
    physical: np.ndarray,
    gamma0: np.ndarray,
    site0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
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
        gamma_prior = 0.5 * torch.mean((gamma / GAMMA_PRIOR_SD).square()) if len(gamma) else torch.zeros(())
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
    audit = {
        "head_objective_without_process_constant": float(final.detach()),
        "gamma_gradient_max": float(torch.max(torch.abs(gamma.grad))) if len(gamma) else 0.0,
        "site_gradient_max": float(torch.max(torch.abs(site.grad))),
    }
    return gamma.detach().numpy().copy(), site.detach().numpy().copy(), audit


def full_check(
    objective: JointObjective,
    physical: np.ndarray,
    gamma: np.ndarray,
    site: np.ndarray,
) -> dict[str, object]:
    raw = torch.nn.Parameter(objective.model.to_raw(physical))
    gamma_tensor = torch.nn.Parameter(torch.tensor(gamma.copy()))
    site_tensor = torch.nn.Parameter(torch.tensor(site.copy()))
    value = objective.loss(raw, gamma_tensor, site_tensor)
    value.backward()
    process_site = s44.s44.projected_kkt(objective.model, raw, raw.grad, site_tensor.grad)
    gamma_kkt = float(torch.max(torch.abs(gamma_tensor.grad))) if len(gamma) else 0.0
    return {
        "objective": float(value.detach()),
        "process_kkt": process_site["process_projected_kkt_max"],
        "site_kkt": process_site["site_gradient_max"],
        "gamma_kkt": gamma_kkt,
        "combined_kkt": max(float(process_site["combined_max"]), gamma_kkt),
        "process_parameter_states": process_site["parameter_states"],
    }


def solve_process(
    objective: JointObjective,
    physical: np.ndarray,
    gamma: np.ndarray,
    site: np.ndarray,
    max_iter: int = 8,
) -> np.ndarray:
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

    lbfgs = torch.optim.LBFGS(
        [free], lr=1.0, max_iter=max_iter, tolerance_grad=1.0e-9,
        tolerance_change=1.0e-15, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(); value = physical_loss(objective, vector(), gamma_tensor, site_tensor); value.backward(); return value

    lbfgs.step(closure)
    fitted = vector().detach().numpy()
    for index, name in enumerate(objective.model.names()):
        fitted[index] = np.clip(fitted[index], objective.model.lower[name], objective.model.upper[name])
    return fitted


def secant_gradient_candidate(
    history: list[tuple[np.ndarray, np.ndarray]], objective: JointObjective
) -> np.ndarray | None:
    """Propose physical coordinates at the component-wise gradient zeros.

    The extrapolation is only a proposal.  ``refine`` refits the complete head
    and accepts it only when both the unchanged objective and full projected
    KKT improve, so this cannot relax the registered numerical gate.
    """
    if len(history) < 2:
        return None
    x0, g0 = history[-2]
    x1, g1 = history[-1]
    denominator = g1 - g0
    candidate = x1.copy()
    usable = np.abs(denominator) > 1.0e-12
    candidate[usable] = x1[usable] - g1[usable] * (x1[usable] - x0[usable]) / denominator[usable]
    fixed = {
        objective.model.names().index("beta_contact"),
        objective.model.names().index("v_f"),
    }
    for index, name in enumerate(objective.model.names()):
        if index in fixed:
            candidate[index] = x1[index]
            continue
        span = objective.model.upper[name] - objective.model.lower[name]
        candidate[index] = np.clip(
            candidate[index],
            objective.model.lower[name] + 1.0e-8 * span,
            objective.model.upper[name] - 1.0e-8 * span,
        )
    return candidate if np.isfinite(candidate).all() else None


def bounded_coordinate_proposal(
    objective: JointObjective,
    physical: np.ndarray,
    gamma: np.ndarray,
    site: np.ndarray,
    check: dict[str, object],
    excluded: set[str] | None = None,
) -> tuple[np.ndarray, str] | None:
    """Minimize the largest residual interior physical coordinate.

    This is reserved for the near-optimum regime, where a full strong-Wolfe
    process block would spend dozens of complete 1961-history evaluations on
    a one-coordinate residual.  The returned point is still acceptance-gated
    after refitting the observation head and recomputing the full KKT.
    """
    fixed = {"beta_contact", "v_f", *(excluded or set())}
    states = sorted(
        check["process_parameter_states"],
        key=lambda row: float(row["kkt_residual"]),
        reverse=True,
    )
    target = next(
        (row for row in states if str(row["parameter"]) not in fixed and float(row["kkt_residual"]) > 1.0e-5),
        None,
    )
    if target is None:
        return None
    name = str(target["parameter"])
    index = objective.model.names().index(name)
    span = objective.model.upper[name] - objective.model.lower[name]
    center = float(physical[index])
    lower = max(objective.model.lower[name] + 1.0e-8 * span, center - 0.02 * span)
    upper = min(objective.model.upper[name] - 1.0e-8 * span, center + 0.02 * span)
    gamma_tensor = torch.tensor(np.asarray(gamma, dtype=float))
    site_tensor = torch.tensor(np.asarray(site, dtype=float))

    def coordinate_objective(value: float) -> float:
        trial = np.asarray(physical, dtype=float).copy()
        trial[index] = float(value)
        with torch.no_grad():
            return float(
                physical_loss(
                    objective,
                    torch.tensor(trial),
                    gamma_tensor,
                    site_tensor,
                )
            )

    result = minimize_scalar(
        coordinate_objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": max(1.0e-10, 1.0e-10 * span), "maxiter": 48},
    )
    if not result.success or not np.isfinite(result.fun):
        return None
    candidate = np.asarray(physical, dtype=float).copy()
    candidate[index] = float(result.x)
    return candidate, name


def refine(
    objective: JointObjective,
    physical: np.ndarray,
    gamma: np.ndarray,
    site: np.ndarray,
    max_cycles: int = 8,
    initial_gradient_history: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], list[dict[str, object]]]:
    history = []
    gradient_history: list[tuple[np.ndarray, np.ndarray]] = [
        (np.asarray(x, dtype=float).copy(), np.asarray(g, dtype=float).copy())
        for x, g in (initial_gradient_history or [])
    ]
    check: dict[str, object] = {"combined_kkt": float("inf")}
    for cycle in range(max_cycles):
        gamma, site, head = solve_head(objective, physical, gamma, site)
        check = full_check(objective, physical, gamma, site)
        gradient = np.asarray(
            [row["physical_gradient"] for row in check["process_parameter_states"]], dtype=float
        )
        gradient_history.append((np.asarray(physical, dtype=float).copy(), gradient))
        history.append({"cycle": cycle, **head, **check})
        print(json.dumps({"stage": "block_refine", **history[-1]}), flush=True)
        if float(check["combined_kkt"]) <= 1.0e-5:
            break
        accelerated = secant_gradient_candidate(gradient_history, objective)
        if accelerated is not None:
            accelerated_gamma, accelerated_site, accelerated_head = solve_head(
                objective, accelerated, gamma, site
            )
            accelerated_check = full_check(
                objective, accelerated, accelerated_gamma, accelerated_site
            )
            accepted = bool(
                np.isfinite(float(accelerated_check["objective"]))
                and float(accelerated_check["objective"]) <= float(check["objective"]) + 1.0e-10
                and float(accelerated_check["combined_kkt"]) < float(check["combined_kkt"])
            )
            history.append({
                "cycle": cycle,
                "proposal": "physical_gradient_secant",
                "accepted": accepted,
                **accelerated_head,
                **accelerated_check,
            })
            print(json.dumps({"stage": "block_refine_acceleration", **history[-1]}), flush=True)
            if accepted:
                physical = accelerated
                gamma = accelerated_gamma
                site = accelerated_site
                check = accelerated_check
                accelerated_gradient = np.asarray(
                    [row["physical_gradient"] for row in check["process_parameter_states"]], dtype=float
                )
                gradient_history[-1] = (accelerated.copy(), accelerated_gradient)
                if float(check["combined_kkt"]) <= 1.0e-5:
                    break
        if float(check["combined_kkt"]) <= 5.0e-3:
            rejected_coordinates: set[str] = set()
            for coordinate_round in range(12):
                proposal = bounded_coordinate_proposal(
                    objective, physical, gamma, site, check, rejected_coordinates
                )
                if proposal is None:
                    break
                coordinate_physical, coordinate_name = proposal
                coordinate_gamma, coordinate_site, coordinate_head = solve_head(
                    objective, coordinate_physical, gamma, site
                )
                coordinate_check = full_check(
                    objective, coordinate_physical, coordinate_gamma, coordinate_site
                )
                accepted = bool(
                    np.isfinite(float(coordinate_check["objective"]))
                    and float(coordinate_check["objective"]) <= float(check["objective"]) + 1.0e-10
                    and float(coordinate_check["combined_kkt"]) < float(check["combined_kkt"])
                )
                history.append({
                    "cycle": cycle,
                    "proposal": "bounded_physical_coordinate",
                    "coordinate_round": coordinate_round,
                    "parameter": coordinate_name,
                    "accepted": accepted,
                    **coordinate_head,
                    **coordinate_check,
                })
                print(json.dumps({"stage": "block_refine_coordinate", **history[-1]}), flush=True)
                if not accepted:
                    rejected_coordinates.add(coordinate_name)
                    continue
                physical = coordinate_physical
                gamma = coordinate_gamma
                site = coordinate_site
                check = coordinate_check
                rejected_coordinates.clear()
                coordinate_gradient = np.asarray(
                    [row["physical_gradient"] for row in check["process_parameter_states"]], dtype=float
                )
                gradient_history[-1] = (physical.copy(), coordinate_gradient)
                if float(check["combined_kkt"]) <= 1.0e-5:
                    break
            if float(check["combined_kkt"]) <= 1.0e-5:
                break
        physical = solve_process(objective, physical, gamma, site)
    if float(check["combined_kkt"]) > 1.0e-5:
        gamma, site, head = solve_head(objective, physical, gamma, site)
        check = full_check(objective, physical, gamma, site)
        history.append({"cycle": max_cycles, **head, **check})
    return physical, gamma, site, check, history
