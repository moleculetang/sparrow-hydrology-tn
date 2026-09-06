"""Resume a saved near-optimum temporal fit on the unchanged joint objective.

This is an engineering-only repair for a final checkpoint that failed solely
because the projected KKT tolerance was not reached.  It does not revisit the
five-start selection, the v_f profile, predictions, bounds, priors or held-out
data.  Physical L-BFGS-B and the cached observation head are alternated, and a
proposal is accepted only when the full objective does not increase and the
independently recomputed full projected KKT strictly decreases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

ROOT = Path(r"E:\SPARROW")
HERE = ROOT / "5_Test/20260904_4/scripts"
RUN = ROOT / "5_Test/20260904_4"
WORK = RUN / "work"
sys.path.insert(0, str(HERE))

from block_refine import full_check, physical_loss, solve_head, solve_process  # noqa: E402
from temporal_common import FOLDS, atomic_json, json_default, observations  # noqa: E402
from unified_fit import FoldDesign, JointObjective, UnifiedTNModel, predict, s41  # noqa: E402


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def scipy_process_proposal(
    objective: JointObjective,
    physical: np.ndarray,
    gamma: np.ndarray,
    site: np.ndarray,
    max_iter: int = 20,
) -> tuple[np.ndarray, dict[str, object]]:
    names = objective.model.names()
    fixed = {names.index("beta_contact"), names.index("v_f")}
    free_indices = [index for index in range(len(names)) if index not in fixed]
    bounds = [
        (objective.model.lower[names[index]] + 1.0e-10, objective.model.upper[names[index]] - 1.0e-10)
        for index in free_indices
    ]
    gamma_tensor = torch.tensor(np.asarray(gamma, dtype=float))
    site_tensor = torch.tensor(np.asarray(site, dtype=float))
    evaluations = 0

    def evaluate(values: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        free = torch.tensor(np.asarray(values, dtype=float), requires_grad=True)
        pieces: list[torch.Tensor] = []
        cursor = 0
        for index, value in enumerate(physical):
            if index in fixed:
                pieces.append(torch.tensor(float(value)))
            else:
                pieces.append(free[cursor])
                cursor += 1
        candidate = torch.stack(pieces)
        loss = physical_loss(objective, candidate, gamma_tensor, site_tensor)
        loss.backward()
        return float(loss.detach()), free.grad.detach().numpy().copy()

    result = minimize(
        evaluate,
        np.asarray(physical[free_indices], dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter, "maxls": 40, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    candidate = np.asarray(physical, dtype=float).copy()
    candidate[free_indices] = result.x
    audit = {
        "method": "physical_scipy_lbfgsb",
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": int(evaluations),
        "reported_objective": float(result.fun),
    }
    return candidate, audit


def publish(
    capacity: str,
    fold: str,
    model: UnifiedTNModel,
    objective: JointObjective,
    test: pd.DataFrame,
    physical: np.ndarray,
    gamma: np.ndarray,
    site_raw: np.ndarray,
    check: dict[str, object],
    repair_history: list[dict[str, object]],
    repair_seconds: float,
) -> str:
    stem = f"{capacity.lower()}_{fold.lower()}_final"
    parameter_path = WORK / f"{stem}_parameters.parquet"
    parameter = pd.read_parquet(parameter_path)
    site_effect = objective.site_effect(torch.tensor(site_raw)).detach().numpy()
    reach_effect = objective.design.effect(torch.tensor(gamma)).detach().numpy()
    training_years, _ = FOLDS[fold]
    result = {
        "physical": physical,
        "gamma": gamma,
        "site_raw": site_raw,
        "site_effect": site_effect,
        "reach_effect": reach_effect,
        "stations": objective.stations,
        "design": objective.design,
        "selected_v_f": float(physical[model.names().index("v_f")]),
        "training_years": training_years,
    }
    layers = predict(model, result, test)
    base = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
    frames = []
    for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
        frame = base.copy()
        values = layers[layer]
        frame["pred_tn_mg_l"] = np.where(np.isfinite(values), np.maximum(np.expm1(values), 0.0), np.nan)
        frame["conditional_available"] = np.isfinite(values) if layer == "gauged_conditional" else True
        frame["capacity"] = capacity
        frame["fold_id"] = fold
        frame["layer"] = layer
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    carrier = model.carrier_diagnostics(torch.tensor(physical))

    old_history = json.loads(str(parameter.loc[0, "refinement_history_json"]))
    parameter.loc[0, "objective"] = float(check["objective"])
    parameter.loc[0, "projected_kkt_max"] = float(check["combined_kkt"])
    parameter.loc[0, "process_kkt"] = float(check["process_kkt"])
    parameter.loc[0, "site_kkt"] = float(check["site_kkt"])
    parameter.loc[0, "gamma_kkt"] = float(check["gamma_kkt"])
    parameter.loc[0, "runtime_seconds"] = float(parameter.loc[0, "runtime_seconds"]) + repair_seconds
    parameter.loc[0, "refinement_history_json"] = json.dumps(old_history + repair_history, default=json_default)
    parameter["numerical_repair_json"] = json.dumps(repair_history, default=json_default)
    parameter.loc[0, "mass_balance_relative"] = float(carrier["mass_balance_relative"])
    for name, value in zip(model.names(), physical):
        parameter.loc[0, name] = float(value)

    atomic_parquet(predictions, WORK / f"{stem}_predictions.parquet")
    atomic_parquet(parameter, parameter_path)
    atomic_parquet(
        pd.DataFrame({"feature": objective.design.fields, "gamma": gamma, "capacity": capacity, "fold_id": fold}),
        WORK / f"{stem}_gamma.parquet",
    )
    atomic_parquet(
        pd.DataFrame({
            "station_key": objective.stations,
            "site_raw": site_raw,
            "station_residual_log_unit": site_effect,
            "capacity": capacity,
            "fold_id": fold,
        }),
        WORK / f"{stem}_sites.parquet",
    )
    atomic_parquet(
        pd.DataFrame({
            "reach_id": np.arange(1, 231),
            "transferable_offset_log_unit": reach_effect,
            "capacity": capacity,
            "fold_id": fold,
        }),
        WORK / f"{stem}_offsets.parquet",
    )
    status = (
        "PASS_TEMPORAL_FIT_CHECKPOINT"
        if float(check["combined_kkt"]) <= 1.0e-5 and float(carrier["mass_balance_relative"]) <= 1.0e-10
        else "FAIL_TEMPORAL_FIT_CHECKPOINT"
    )
    atomic_json({
        "status": status,
        "capacity": capacity,
        "fold_id": fold,
        "kkt": check["combined_kkt"],
        "mass_balance_relative": carrier["mass_balance_relative"],
        "numerical_repair": "alternating cached-head and physical-scale optimization",
    }, WORK / f"{stem}_checkpoint.json")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", choices=["H7", "H14", "H22"], required=True)
    parser.add_argument("--fold", choices=list(FOLDS), required=True)
    parser.add_argument("--cycles", type=int, default=6)
    args = parser.parse_args()
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    started = time.perf_counter()
    stem = f"{args.capacity.lower()}_{args.fold.lower()}_final"
    checkpoint_path = WORK / f"{stem}_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("status") == "PASS_TEMPORAL_FIT_CHECKPOINT":
        print(json.dumps({"status": "ALREADY_PASS", "capacity": args.capacity, "fold": args.fold}), flush=True)
        return

    training_years, evaluation_year = FOLDS[args.fold]
    obs = observations()
    train = obs.loc[obs.year.isin(training_years)].copy()
    test = obs.loc[obs.year.eq(evaluation_year)].copy()
    model = UnifiedTNModel("formal")
    parameter = pd.read_parquet(WORK / f"{stem}_parameters.parquet")
    physical = np.asarray([parameter.loc[0, name] for name in model.names()], dtype=float)
    model.set_fixed_v_f(float(physical[model.names().index("v_f")]))
    design = FoldDesign.build(args.capacity, train)
    objective = JointObjective(model, design, train, training_years)
    gamma_table = pd.read_parquet(WORK / f"{stem}_gamma.parquet").set_index("feature")
    gamma = np.asarray([gamma_table.loc[field, "gamma"] for field in design.fields], dtype=float)
    site_table = pd.read_parquet(WORK / f"{stem}_sites.parquet").set_index("station_key")
    site = np.asarray([site_table.loc[station, "site_raw"] for station in objective.stations], dtype=float)
    history: list[dict[str, object]] = []

    gamma, site, head = solve_head(objective, physical, gamma, site)
    check = full_check(objective, physical, gamma, site)
    history.append({"repair_cycle": 0, "stage": "initial_refit_head", **head, **check})
    print(json.dumps(history[-1], default=json_default), flush=True)
    for cycle in range(1, args.cycles + 1):
        if float(check["combined_kkt"]) <= 1.0e-5:
            break
        candidate, process_audit = scipy_process_proposal(objective, physical, gamma, site)
        candidate_gamma, candidate_site, candidate_head = solve_head(objective, candidate, gamma, site)
        candidate_check = full_check(objective, candidate, candidate_gamma, candidate_site)
        accepted = bool(
            np.isfinite(float(candidate_check["objective"]))
            and float(candidate_check["objective"]) <= float(check["objective"]) + 1.0e-10
            and float(candidate_check["combined_kkt"]) < float(check["combined_kkt"])
        )
        record = {
            "repair_cycle": cycle,
            "stage": "physical_scipy_lbfgsb",
            "accepted": accepted,
            "optimizer": process_audit,
            **candidate_head,
            **candidate_check,
        }
        history.append(record)
        print(json.dumps(record, default=json_default), flush=True)
        if accepted:
            physical, gamma, site, check = candidate, candidate_gamma, candidate_site, candidate_check
            continue

        fallback = solve_process(objective, physical, gamma, site, max_iter=24)
        fallback_gamma, fallback_site, fallback_head = solve_head(objective, fallback, gamma, site)
        fallback_check = full_check(objective, fallback, fallback_gamma, fallback_site)
        fallback_accepted = bool(
            np.isfinite(float(fallback_check["objective"]))
            and float(fallback_check["objective"]) <= float(check["objective"]) + 1.0e-10
            and float(fallback_check["combined_kkt"]) < float(check["combined_kkt"])
        )
        fallback_record = {
            "repair_cycle": cycle,
            "stage": "physical_torch_lbfgs_fallback",
            "accepted": fallback_accepted,
            **fallback_head,
            **fallback_check,
        }
        history.append(fallback_record)
        print(json.dumps(fallback_record, default=json_default), flush=True)
        if fallback_accepted:
            physical, gamma, site, check = fallback, fallback_gamma, fallback_site, fallback_check
        else:
            break

    gamma, site, final_head = solve_head(objective, physical, gamma, site)
    check = full_check(objective, physical, gamma, site)
    history.append({"repair_cycle": len(history), "stage": "final_full_check", **final_head, **check})
    repair_seconds = time.perf_counter() - started
    status = publish(
        args.capacity, args.fold, model, objective, test, physical, gamma, site,
        check, history, repair_seconds,
    )
    print(json.dumps({
        "status": status,
        "capacity": args.capacity,
        "fold": args.fold,
        "kkt": check["combined_kkt"],
        "repair_seconds": repair_seconds,
    }, default=json_default), flush=True)
    if status != "PASS_TEMPORAL_FIT_CHECKPOINT":
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
