"""Run one of four checkpointed nested-spatial shards for H22_TRANSFER_HEAD."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize, minimize_scalar


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / r"5_Test\20260902_5"
WORK = RUN / "work"
STAGE4 = ROOT / r"5_Test\20260902_4"
PARENT_PAR = ROOT / r"5_Test\20260824_47\work\mineral_lifetime_nested_parameters.parquet"
for path in [
    STAGE4 / "scripts", ROOT / r"5_Test\20260824_41\scripts", ROOT / r"5_Test\20260824_44\scripts",
    ROOT / r"5_Test\20260824_19\scripts", ROOT / r"5_Test\20260824_46\scripts",
]:
    sys.path.insert(0, str(path))
from spatial_head_model import FoldDesign, Objective, gamma_start  # noqa: E402
from stage46_models import Stage46Model  # noqa: E402
import run_stage41 as s41  # noqa: E402
import run_stage44 as s44  # noqa: E402
import run_stage19 as s19  # noqa: E402

s28 = s41.s28
SEED = 260905
N_SHARDS = 8


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def starts_from_parent(model: Stage46Model, row: pd.Series) -> list[np.ndarray]:
    anchor = np.asarray([row[name] for name in model.names()], dtype=float)
    starts = [anchor.copy() for _ in range(5)]
    starts[1][-1] += math.log(1.5)
    starts[2][-1] -= math.log(1.5)
    starts[3][4] += 0.10; starts[3][5] -= 0.10
    starts[4][4] -= 0.10; starts[4][5] += 0.10
    for array in starts:
        for index, name in enumerate(model.names()):
            array[index] = np.clip(array[index], model.lower[name] + 1e-6, model.upper[name] - 1e-6)
    return starts


def fit_fold(model: Stage46Model, train: pd.DataFrame, fold_id: str, parent_row: pd.Series) -> dict[str, object]:
    design = FoldDesign.build("H22_TRANSFER_HEAD", train)
    objective = Objective(model, design, train)
    trials = []
    for variant, process_start in enumerate(starts_from_parent(model, parent_row)):
        torch.manual_seed(SEED + variant)
        raw = torch.nn.Parameter(model.to_raw(process_start))
        gamma = torch.nn.Parameter(torch.tensor(gamma_start(len(design.fields), variant)))
        raw_site = torch.nn.Parameter(torch.tensor(s44.site_start(len(objective.stations), variant)))
        variables = [raw, gamma, raw_site]
        adam = torch.optim.AdamW(variables, lr=0.03, weight_decay=0.0)
        for _ in range(30):
            adam.zero_grad(); value = objective.loss(raw, gamma, raw_site); value.backward()
            torch.nn.utils.clip_grad_norm_(variables, 10.0); adam.step()
        lbfgs = torch.optim.LBFGS(
            variables, lr=1.0, max_iter=60, tolerance_grad=1e-11,
            tolerance_change=1e-13, line_search_fn="strong_wolfe",
        )
        def closure() -> torch.Tensor:
            lbfgs.zero_grad(); result = objective.loss(raw, gamma, raw_site); result.backward(); return result
        lbfgs.step(closure)
        raw.grad = gamma.grad = raw_site.grad = None
        final = objective.loss(raw, gamma, raw_site); final.backward()
        kkt_ps = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
        gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach())
        with torch.no_grad():
            physical = model.to_physical(raw).detach().numpy()
            effects = objective.site_effects(raw_site).detach().numpy()
            gamma_value = gamma.detach().numpy().copy()
            reach_effect = design.effect_reach(gamma).detach().numpy()
            score = float(final.detach())
        trials.append({
            "variant": variant, "objective": score,
            "physical": physical, "gamma": gamma_value, "effects": effects,
            "raw_site": raw_site.detach().numpy().copy(),
            "stations": objective.stations.copy(), "reach_effect": reach_effect,
            "kkt": max(kkt_ps["combined_max"], gamma_kkt),
            "process_site_kkt": kkt_ps["combined_max"], "gamma_kkt": gamma_kkt,
            "process_projected_kkt": kkt_ps["process_projected_kkt_max"],
            "site_gradient_kkt": kkt_ps["site_gradient_max"],
            "parameter_states": kkt_ps["parameter_states"],
            "finite": bool(np.isfinite(score) and np.isfinite(physical).all() and np.isfinite(gamma_value).all()),
        })
        del raw, gamma, raw_site, adam, lbfgs
        gc.collect()
    finite = [row for row in trials if row["finite"]]
    if len(finite) != 5:
        raise RuntimeError(f"{fold_id}: only {len(finite)}/5 starts finite")
    best = min(finite, key=lambda row: row["objective"])
    best["all_starts_json"] = json.dumps([{key: row[key] for key in ["variant", "objective", "kkt", "finite"]} for row in trials])
    # The five starts identify the basin of attraction.  Continue only the
    # selected solution so every nested fold satisfies the same numerical KKT
    # gate without multiplying the expensive long-history forward pass by five.
    raw = torch.nn.Parameter(model.to_raw(np.asarray(best["physical"], dtype=float)))
    gamma = torch.nn.Parameter(torch.tensor(np.asarray(best["gamma"], dtype=float)))
    raw_site = torch.nn.Parameter(torch.tensor(np.asarray(best["raw_site"], dtype=float)))
    variables = [raw, gamma, raw_site]
    polish_blocks = 0
    vf_coordinate_rescue = False
    coordinate_rescue_parameters: list[str] = []
    alternating_rescue_rounds = 0
    cached_head_coordinate_rounds = 0
    scipy_process_rescue_rounds = 0
    for _ in range(5):
        polish_blocks += 1
        optimizer = torch.optim.LBFGS(
            variables, lr=1.0, max_iter=200, tolerance_grad=1e-12,
            tolerance_change=1e-14, line_search_fn="strong_wolfe",
        )
        def polish_closure() -> torch.Tensor:
            optimizer.zero_grad(); value = objective.loss(raw, gamma, raw_site); value.backward(); return value
        optimizer.step(polish_closure)
        raw.grad = gamma.grad = raw_site.grad = None
        final = objective.loss(raw, gamma, raw_site); final.backward()
        kkt_ps = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
        gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach())
        combined = max(kkt_ps["combined_max"], gamma_kkt)
        if combined <= 1e-5:
            break
        state_by_name = {row["parameter"]: row for row in kkt_ps["parameter_states"]}
        vf_residual = float(state_by_name["v_f"]["kkt_residual"])
        other_process = max(
            (float(row["kkt_residual"]) for row in kkt_ps["parameter_states"] if row["parameter"] != "v_f"),
            default=0.0,
        )
        if (
            vf_residual > 1e-5
            and max(other_process, float(kkt_ps["site_gradient_max"]), gamma_kkt) <= 1e-5
        ):
            break
    # Near convergence, the bounded physical v_f gradient can remain slightly
    # above the registered KKT gate even when its raw-sigmoid gradient is tiny
    # enough for L-BFGS to stop on function change.  Apply an exact 1-D
    # physical-scale coordinate minimization and inspect the complete gradient
    # before any optional joint polish.  This changes neither the objective nor
    # the model; it only removes a known parameterization-conditioning artifact.
    if combined > 1e-5:
        vf_coordinate_rescue = True
        coordinate_rescue_parameters.append("v_f")
        vf_index = model.names().index("v_f")
        for rescue_round in range(3):
            physical_base = model.to_physical(raw).detach().numpy()

            def vf_objective(value: float) -> float:
                trial = physical_base.copy()
                trial[vf_index] = value
                trial_raw = model.to_raw(trial)
                with torch.no_grad():
                    return float(objective.loss(trial_raw, gamma, raw_site).detach())

            rescue = minimize_scalar(
                vf_objective, bounds=(model.lower["v_f"] + 1e-10, model.upper["v_f"] - 1e-10),
                method="bounded", options={"xatol": 1e-10, "maxiter": 100},
            )
            if not rescue.success or not np.isfinite(rescue.fun):
                raise RuntimeError(f"{fold_id}: v_f coordinate rescue failed: {rescue.message}")
            physical_base[vf_index] = float(rescue.x)
            with torch.no_grad():
                raw.copy_(model.to_raw(physical_base))
            raw.grad = gamma.grad = raw_site.grad = None
            final = objective.loss(raw, gamma, raw_site); final.backward()
            kkt_ps = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
            gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach())
            combined = max(kkt_ps["combined_max"], gamma_kkt)
            if combined <= 1e-5:
                break
            optimizer = torch.optim.LBFGS(
                variables, lr=1.0, max_iter=200, tolerance_grad=1e-12,
                tolerance_change=1e-14, line_search_fn="strong_wolfe",
            )
            def rescue_closure() -> torch.Tensor:
                optimizer.zero_grad(); value = objective.loss(raw, gamma, raw_site); value.backward(); return value
            optimizer.step(rescue_closure)
            raw.grad = gamma.grad = raw_site.grad = None
            final = objective.loss(raw, gamma, raw_site); final.backward()
            kkt_ps = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
            gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach())
            combined = max(kkt_ps["combined_max"], gamma_kkt)
            if combined <= 1e-5:
                break
    # If v_f correction exposes a different small physical-scale process
    # gradient, finish with deterministic Gauss-Seidel coordinate corrections.
    # This is a numerical KKT repair on the unchanged registered objective.
    for _ in range(8):
        if combined <= 1e-5:
            break
        states = sorted(kkt_ps["parameter_states"], key=lambda row: float(row["kkt_residual"]), reverse=True)
        top = states[0]
        if float(top["kkt_residual"]) > 1e-5:
            name = str(top["parameter"])
            index = model.names().index(name)
            physical_base = model.to_physical(raw).detach().numpy()

            def coordinate_objective(value: float) -> float:
                trial = physical_base.copy()
                trial[index] = value
                with torch.no_grad():
                    return float(objective.loss(model.to_raw(trial), gamma, raw_site).detach())

            coordinate = minimize_scalar(
                coordinate_objective,
                bounds=(model.lower[name] + 1e-10, model.upper[name] - 1e-10),
                method="bounded", options={"xatol": 1e-10, "maxiter": 100},
            )
            if not coordinate.success or not np.isfinite(coordinate.fun):
                raise RuntimeError(f"{fold_id}: {name} coordinate rescue failed: {coordinate.message}")
            physical_base[index] = float(coordinate.x)
            with torch.no_grad():
                raw.copy_(model.to_raw(physical_base))
            coordinate_rescue_parameters.append(name)
        else:
            optimizer = torch.optim.LBFGS(
                variables, lr=1.0, max_iter=200, tolerance_grad=1e-12,
                tolerance_change=1e-14, line_search_fn="strong_wolfe",
            )
            def residual_closure() -> torch.Tensor:
                optimizer.zero_grad(); value = objective.loss(raw, gamma, raw_site); value.backward(); return value
            optimizer.step(residual_closure)
        raw.grad = gamma.grad = raw_site.grad = None
        final = objective.loss(raw, gamma, raw_site); final.backward()
        kkt_ps = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
        gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach())
        combined = max(kkt_ps["combined_max"], gamma_kkt)
    # A held-out terminal tree can make the joint Hessian much more poorly
    # conditioned than a held-out reach: gamma, station residuals and process
    # coordinates may then all remain above tolerance even after exact process
    # coordinate corrections.  Finish those rare folds with deterministic
    # block-coordinate L-BFGS on the unchanged objective.  Separating the
    # observation-head/site block from the process block removes the scale
    # competition that causes joint L-BFGS to stop on function change.
    if combined > 1e-5:
        for _ in range(8):
            alternating_rescue_rounds += 1
            # Cache the long-history process output while optimizing the
            # observation head.  Omitting process-only constants preserves the
            # exact gamma/site gradient but avoids premature function-change
            # stopping caused by a large fixed objective offset.
            with torch.no_grad():
                physical_fixed = model.to_physical(raw).detach()
                _, base_fixed = model.evaluate(
                    objective.train, physical_fixed,
                    objective.train_start, objective.train_end,
                )
                value_map = dict(zip(model.names(), physical_fixed))
                sigma_fixed = torch.exp(value_map["log_sigma"]).detach()

            def head_only_loss() -> torch.Tensor:
                spatial = design.effect_reach(gamma)[objective.row_reach]
                population = base_fixed + spatial
                effects = objective.site_effects(raw_site)
                conditional = population + effects[objective.row_station]
                data = 0.90 * objective.layer_loss(population, sigma_fixed) + 0.10 * objective.layer_loss(conditional, sigma_fixed)
                site_prior = 0.10 * 0.5 * 12.0 * torch.sum(effects.square())
                return data + (design.prior(gamma) + site_prior) / 119.0

            head_optimizer = torch.optim.LBFGS(
                [gamma, raw_site], lr=1.0, max_iter=1000,
                tolerance_grad=1e-14, tolerance_change=1e-16,
                line_search_fn="strong_wolfe",
            )

            def head_closure() -> torch.Tensor:
                for variable in variables:
                    variable.grad = None
                value = head_only_loss()
                value.backward()
                return value

            head_optimizer.step(head_closure)
            optimizer = torch.optim.LBFGS(
                [raw], lr=1.0, max_iter=300, tolerance_grad=1e-13,
                tolerance_change=1e-15, line_search_fn="strong_wolfe",
            )

            def process_block_closure() -> torch.Tensor:
                for variable in variables:
                    variable.grad = None
                value = objective.loss(raw, gamma, raw_site)
                value.backward()
                return value

            optimizer.step(process_block_closure)
            raw.grad = gamma.grad = raw_site.grad = None
            final = objective.loss(raw, gamma, raw_site); final.backward()
            kkt_ps = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
            gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach())
            combined = max(kkt_ps["combined_max"], gamma_kkt)
            if combined <= 1e-5:
                break
            states = sorted(kkt_ps["parameter_states"], key=lambda row: float(row["kkt_residual"]), reverse=True)
            top = states[0]
            if float(top["kkt_residual"]) > 1e-5:
                name = str(top["parameter"])
                index = model.names().index(name)
                physical_base = model.to_physical(raw).detach().numpy()

                def alternating_coordinate_objective(value: float) -> float:
                    trial = physical_base.copy()
                    trial[index] = value
                    with torch.no_grad():
                        return float(objective.loss(model.to_raw(trial), gamma, raw_site).detach())

                coordinate = minimize_scalar(
                    alternating_coordinate_objective,
                    bounds=(model.lower[name] + 1e-10, model.upper[name] - 1e-10),
                    method="bounded", options={"xatol": 1e-10, "maxiter": 100},
                )
                if not coordinate.success or not np.isfinite(coordinate.fun):
                    raise RuntimeError(f"{fold_id}: alternating {name} coordinate rescue failed: {coordinate.message}")
                physical_base[index] = float(coordinate.x)
                with torch.no_grad():
                    raw.copy_(model.to_raw(physical_base))
                coordinate_rescue_parameters.append(name)
                raw.grad = gamma.grad = raw_site.grad = None
                final = objective.loss(raw, gamma, raw_site); final.backward()
                kkt_ps = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
                gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach())
                combined = max(kkt_ps["combined_max"], gamma_kkt)
                if combined <= 1e-5:
                    break
    # End on the conditionally optimized head, not on a process update.  This
    # final cached block-coordinate sweep is inexpensive for gamma/site because
    # the 1960-2024 process trajectory is evaluated only once per round.  A
    # short deterministic Adam bridge is used only when the conditional L-BFGS
    # gradient itself remains above the registered tolerance.
    if combined > 1e-5:
        for _ in range(24):
            cached_head_coordinate_rounds += 1
            with torch.no_grad():
                physical_fixed = model.to_physical(raw).detach()
                _, base_fixed = model.evaluate(
                    objective.train, physical_fixed,
                    objective.train_start, objective.train_end,
                )
                value_map = dict(zip(model.names(), physical_fixed))
                sigma_fixed = torch.exp(value_map["log_sigma"]).detach()

            def final_head_loss() -> torch.Tensor:
                spatial = design.effect_reach(gamma)[objective.row_reach]
                population = base_fixed + spatial
                effects = objective.site_effects(raw_site)
                conditional = population + effects[objective.row_station]
                data = 0.90 * objective.layer_loss(population, sigma_fixed) + 0.10 * objective.layer_loss(conditional, sigma_fixed)
                site_prior = 0.10 * 0.5 * 12.0 * torch.sum(effects.square())
                return data + (design.prior(gamma) + site_prior) / 119.0

            head_optimizer = torch.optim.LBFGS(
                [gamma, raw_site], lr=1.0, max_iter=1000,
                tolerance_grad=1e-14, tolerance_change=1e-16,
                line_search_fn="strong_wolfe",
            )

            def final_head_closure() -> torch.Tensor:
                for variable in variables:
                    variable.grad = None
                value = final_head_loss()
                value.backward()
                return value

            head_optimizer.step(final_head_closure)
            for variable in variables:
                variable.grad = None
            final_head = final_head_loss(); final_head.backward()
            conditional_kkt = max(
                float(torch.max(torch.abs(gamma.grad)).detach()),
                float(torch.max(torch.abs(raw_site.grad)).detach()),
            )
            if conditional_kkt > 1e-5:
                head_adam = torch.optim.Adam([gamma, raw_site], lr=0.003)
                for _ in range(1000):
                    head_adam.zero_grad(); value = final_head_loss(); value.backward(); head_adam.step()
                head_optimizer = torch.optim.LBFGS(
                    [gamma, raw_site], lr=1.0, max_iter=1000,
                    tolerance_grad=1e-14, tolerance_change=1e-16,
                    line_search_fn="strong_wolfe",
                )
                head_optimizer.step(final_head_closure)
            raw.grad = gamma.grad = raw_site.grad = None
            final = objective.loss(raw, gamma, raw_site); final.backward()
            kkt_ps = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
            gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach())
            combined = max(kkt_ps["combined_max"], gamma_kkt)
            if combined <= 1e-5:
                break
            states = sorted(kkt_ps["parameter_states"], key=lambda row: float(row["kkt_residual"]), reverse=True)
            top = states[0]
            if float(top["kkt_residual"]) <= 1e-5:
                continue
            name = str(top["parameter"])
            index = model.names().index(name)
            physical_base = model.to_physical(raw).detach().numpy()

            def final_coordinate_objective(value: float) -> float:
                trial = physical_base.copy()
                trial[index] = value
                with torch.no_grad():
                    return float(objective.loss(model.to_raw(trial), gamma, raw_site).detach())

            coordinate = minimize_scalar(
                final_coordinate_objective,
                bounds=(model.lower[name] + 1e-10, model.upper[name] - 1e-10),
                method="bounded", options={"xatol": 1e-10, "maxiter": 100},
            )
            if not coordinate.success or not np.isfinite(coordinate.fun):
                raise RuntimeError(f"{fold_id}: final {name} coordinate rescue failed: {coordinate.message}")
            physical_base[index] = float(coordinate.x)
            with torch.no_grad():
                raw.copy_(model.to_raw(physical_base))
            coordinate_rescue_parameters.append(name)
        raw.grad = gamma.grad = raw_site.grad = None
        final = objective.loss(raw, gamma, raw_site); final.backward()
        kkt_ps = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
        gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach())
        combined = max(kkt_ps["combined_max"], gamma_kkt)
    # Coordinate cycling indicates correlation among physical process
    # parameters. Resolve that final rare case by jointly optimizing the small
    # bounded physical process vector with an exact autograd gradient, then
    # refitting the cached observation head. L-BFGS-B works on physical scale,
    # avoiding sigmoid-conditioning artifacts near parameter bounds.
    if combined > 1e-5:
        process_names = model.names()
        lower = np.asarray([model.lower[name] for name in process_names], dtype=float)
        upper = np.asarray([model.upper[name] for name in process_names], dtype=float)
        span = upper - lower
        process_bounds = list(zip(lower + 1e-8 * span, upper - 1e-8 * span))
        for _ in range(6):
            scipy_process_rescue_rounds += 1
            gamma_fixed = gamma.detach().clone()
            site_fixed = raw_site.detach().clone()

            def process_value_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
                values = np.asarray(values, dtype=float)
                trial_raw = model.to_raw(values)
                trial_raw.requires_grad_(True)
                value = objective.loss(trial_raw, gamma_fixed, site_fixed)
                value.backward()
                fraction = np.clip((values - lower) / span, 1e-8, 1.0 - 1e-8)
                jacobian = span * fraction * (1.0 - fraction)
                gradient = trial_raw.grad.detach().numpy() / jacobian
                return float(value.detach()), np.asarray(gradient, dtype=float)

            process_start = model.to_physical(raw).detach().numpy()
            process_result = minimize(
                process_value_gradient, process_start, method="L-BFGS-B",
                jac=True, bounds=process_bounds,
                options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-10, "maxls": 50},
            )
            if not np.isfinite(process_result.fun) or not np.isfinite(process_result.x).all():
                raise RuntimeError(f"{fold_id}: physical L-BFGS-B rescue produced nonfinite values")
            with torch.no_grad():
                raw.copy_(model.to_raw(np.asarray(process_result.x, dtype=float)))

            # Refit gamma/site conditionally using a cached process trajectory.
            with torch.no_grad():
                physical_fixed = model.to_physical(raw).detach()
                _, base_fixed = model.evaluate(
                    objective.train, physical_fixed,
                    objective.train_start, objective.train_end,
                )
                value_map = dict(zip(model.names(), physical_fixed))
                sigma_fixed = torch.exp(value_map["log_sigma"]).detach()

            def scipy_head_loss() -> torch.Tensor:
                spatial = design.effect_reach(gamma)[objective.row_reach]
                population = base_fixed + spatial
                effects = objective.site_effects(raw_site)
                conditional = population + effects[objective.row_station]
                data = 0.90 * objective.layer_loss(population, sigma_fixed) + 0.10 * objective.layer_loss(conditional, sigma_fixed)
                site_prior = 0.10 * 0.5 * 12.0 * torch.sum(effects.square())
                return data + (design.prior(gamma) + site_prior) / 119.0

            head_optimizer = torch.optim.LBFGS(
                [gamma, raw_site], lr=1.0, max_iter=1000,
                tolerance_grad=1e-14, tolerance_change=1e-16,
                line_search_fn="strong_wolfe",
            )

            def scipy_head_closure() -> torch.Tensor:
                for variable in variables:
                    variable.grad = None
                value = scipy_head_loss()
                value.backward()
                return value

            head_optimizer.step(scipy_head_closure)
            for variable in variables:
                variable.grad = None
            head_check = scipy_head_loss(); head_check.backward()
            conditional_kkt = max(
                float(torch.max(torch.abs(gamma.grad)).detach()),
                float(torch.max(torch.abs(raw_site.grad)).detach()),
            )
            if conditional_kkt > 1e-5:
                head_adam = torch.optim.Adam([gamma, raw_site], lr=0.003)
                for _ in range(1000):
                    head_adam.zero_grad(); value = scipy_head_loss(); value.backward(); head_adam.step()
                head_optimizer = torch.optim.LBFGS(
                    [gamma, raw_site], lr=1.0, max_iter=1000,
                    tolerance_grad=1e-14, tolerance_change=1e-16,
                    line_search_fn="strong_wolfe",
                )
                head_optimizer.step(scipy_head_closure)
            raw.grad = gamma.grad = raw_site.grad = None
            final = objective.loss(raw, gamma, raw_site); final.backward()
            kkt_ps = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
            gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach())
            combined = max(kkt_ps["combined_max"], gamma_kkt)
            if combined <= 1e-5:
                break
    if combined > 1e-5:
        states = sorted(kkt_ps["parameter_states"], key=lambda row: float(row["kkt_residual"]), reverse=True)
        detail = {
            "combined": combined,
            "site": float(kkt_ps["site_gradient_max"]),
            "gamma": gamma_kkt,
            "top_process": states[:3],
            "coordinate_sequence": coordinate_rescue_parameters,
        }
        raise RuntimeError(f"{fold_id}: KKT rescue exhausted: {json.dumps(detail)}")
    with torch.no_grad():
        best.update({
            "objective": float(final.detach()),
            "physical": model.to_physical(raw).detach().numpy(),
            "gamma": gamma.detach().numpy().copy(),
            "effects": objective.site_effects(raw_site).detach().numpy(),
            "reach_effect": design.effect_reach(gamma).detach().numpy(),
            "kkt": combined,
            "process_site_kkt": kkt_ps["combined_max"],
            "gamma_kkt": gamma_kkt,
            "process_projected_kkt": kkt_ps["process_projected_kkt_max"],
            "site_gradient_kkt": kkt_ps["site_gradient_max"],
            "parameter_states": kkt_ps["parameter_states"],
            "polish_blocks": polish_blocks,
            "vf_coordinate_rescue": vf_coordinate_rescue,
            "coordinate_rescue_parameters_json": json.dumps(coordinate_rescue_parameters),
            "alternating_rescue_rounds": alternating_rescue_rounds,
            "cached_head_coordinate_rounds": cached_head_coordinate_rounds,
            "scipy_process_rescue_rounds": scipy_process_rescue_rounds,
        })
    del raw, gamma, raw_site, optimizer
    gc.collect()
    return best


def station_equal_mean(train: pd.DataFrame) -> float:
    return float(train.assign(y=np.log1p(train.tn_mg_l)).groupby("station_key").y.mean().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=range(N_SHARDS))
    parser.add_argument("--fold-id")
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    if args.shard is None and args.fold_id is None:
        parser.error("one of --shard or --fold-id is required")
    s28.require_runtime(); torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    decision = json.loads((STAGE4 / "reports" / "temporal_decision.json").read_text(encoding="utf-8"))
    if "H22_TRANSFER_HEAD" not in decision["eligible_candidates"]:
        raise RuntimeError("H22 was not authorized by Stage 4")
    observations = s19.build_observations()
    folds = s19.build_folds(observations, "full").loc[lambda x: ~x.holdout_type.eq("TEMPORAL")].reset_index(drop=True)
    if args.fold_id is not None:
        folds = folds.loc[folds.fold_id.astype(str).eq(str(args.fold_id))].copy()
        if len(folds) != 1:
            raise RuntimeError(f"Expected exactly one fold for {args.fold_id}, found {len(folds)}")
        suffix = "target_" + "".join(character if character.isalnum() or character in "_-" else "_" for character in str(args.fold_id))
    else:
        folds = folds.loc[folds.index % N_SHARDS == args.shard].copy()
        suffix = f"part{args.shard}_of{N_SHARDS}"
    parent = pd.read_parquet(PARENT_PAR).set_index("fold_id")
    model = Stage46Model("MINERAL_LIFETIME")
    pred_path = WORK / f"h22_nested_predictions_{suffix}.parquet"
    par_path = WORK / f"h22_nested_parameters_{suffix}.parquet"
    gamma_path = WORK / f"h22_nested_gamma_{suffix}.parquet"
    predictions = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    parameters = pd.read_parquet(par_path) if par_path.exists() else pd.DataFrame()
    gammas = pd.read_parquet(gamma_path) if gamma_path.exists() else pd.DataFrame()
    if not parameters.empty:
        invalid = set(parameters.loc[parameters.projected_kkt_max > 1e-5, "fold_id"].astype(str))
        if invalid:
            parameters = parameters.loc[~parameters.fold_id.astype(str).isin(invalid)].copy()
            predictions = predictions.loc[~predictions.fold_id.astype(str).isin(invalid)].copy()
            gammas = gammas.loc[~gammas.fold_id.astype(str).isin(invalid)].copy()
    # Earlier four-way preflight/checkpoint files remain immutable.  Reuse every
    # valid completed fold across both layouts so repartitioning never repeats or
    # discards a finished held-out fit.
    completed: set[str] = set()
    for existing in WORK.glob("h22_nested_parameters_*.parquet"):
        prior = pd.read_parquet(existing)
        if "projected_kkt_max" in prior:
            prior = prior.loc[prior.projected_kkt_max <= 1e-5]
        completed.update(prior.fold_id.astype(str))
    new_count = 0
    for sequence, (_, fold) in enumerate(folds.iterrows(), start=1):
        fold_id = str(fold.fold_id)
        if fold_id in completed:
            continue
        if args.max_new is not None and new_count >= args.max_new:
            break
        model._obs_index_cache.clear(); model._q_feature_cache.clear()
        train, test = s19.fold_frames(observations, fold)
        if fold_id not in parent.index:
            raise RuntimeError(f"No held-out-clean parent initialization for {fold_id}")
        result = fit_fold(model, train, fold_id, parent.loc[fold_id])
        physical = torch.tensor(np.asarray(result["physical"], dtype=float))
        gamma = torch.tensor(np.asarray(result["gamma"], dtype=float))
        design = FoldDesign.build("H22_TRANSFER_HEAD", train)
        effect_map = dict(zip(result["stations"], map(float, result["effects"])))
        with torch.no_grad():
            _, base_population = model.evaluate(test, physical, int(fold.train_start_year), int(fold.train_end_year))
            reach_effect = design.effect_reach(gamma).detach().numpy()
        population = base_population.detach().numpy() + reach_effect[test.reach_id.astype(int).to_numpy() - 1]
        known = test.station_key.astype(str).isin(effect_map).to_numpy()
        conditional = population.copy(); conditional[known] += np.asarray([effect_map.get(str(x), 0.0) for x in test.station_key])[known]; conditional[~known] = np.nan
        baseline = max(math.expm1(station_equal_mean(train)), 0.0)
        for layer, values in [("population_transferable", population), ("gauged_conditional", conditional)]:
            frame = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
            frame["pred_tn_mg_l"] = np.where(np.isfinite(values), np.maximum(np.expm1(values), 0.0), np.nan)
            frame["baseline_pred_tn_mg_l"] = baseline
            frame["conditional_available"] = known if layer == "gauged_conditional" else True
            frame["fold_id"] = fold_id; frame["holdout_type"] = str(fold.holdout_type); frame["holdout_id"] = str(fold.holdout_id); frame["layer"] = layer; frame["candidate"] = "H22_TRANSFER_HEAD"
            predictions = pd.concat([predictions, frame], ignore_index=True)
        par_row = {
            "candidate": "H22_TRANSFER_HEAD", "fold_id": fold_id,
            "holdout_type": str(fold.holdout_type), "holdout_id": str(fold.holdout_id),
            "train_rows": len(train), "test_rows": len(test), "objective": result["objective"],
            "projected_kkt_max": result["kkt"], "process_site_kkt_max": result["process_site_kkt"],
            "gamma_kkt_max": result["gamma_kkt"], "max_abs_transferable_offset": float(np.max(np.abs(result["reach_effect"]))),
            "process_projected_kkt_max": result["process_projected_kkt"],
            "site_gradient_kkt_max": result["site_gradient_kkt"],
            "parameter_states_json": json.dumps(result["parameter_states"]),
            "polish_blocks": result["polish_blocks"],
            "vf_coordinate_rescue": result["vf_coordinate_rescue"],
            "coordinate_rescue_parameters_json": result["coordinate_rescue_parameters_json"],
            "alternating_rescue_rounds": result["alternating_rescue_rounds"],
            "cached_head_coordinate_rounds": result["cached_head_coordinate_rounds"],
            "scipy_process_rescue_rounds": result["scipy_process_rescue_rounds"],
            "all_starts_json": result["all_starts_json"],
            **dict(zip(model.names(), map(float, result["physical"]))),
        }
        parameters = pd.concat([parameters, pd.DataFrame([par_row])], ignore_index=True)
        gamma_rows = pd.DataFrame({"fold_id": fold_id, "feature": design.fields, "gamma": result["gamma"]})
        gammas = pd.concat([gammas, gamma_rows], ignore_index=True)
        atomic_parquet(predictions, pred_path); atomic_parquet(parameters, par_path); atomic_parquet(gammas, gamma_path)
        new_count += 1
        if sequence % 5 == 0 or sequence == len(folds):
            current, peak = s28.memory_gib()
            print(json.dumps({"shard": args.shard, "target_fold": args.fold_id, "completed": fold_id, "sequence": sequence, "total": len(folds), "kkt": result["kkt"], "rss_gib": current, "peak_gib": peak}), flush=True)
    print(json.dumps({"status": "SHARD_COMPLETE", "shard": args.shard, "target_fold": args.fold_id, "folds": int(parameters.fold_id.nunique())}), flush=True)


if __name__ == "__main__":
    main()
