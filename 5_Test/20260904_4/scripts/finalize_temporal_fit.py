"""Profile, refine and checkpoint one capacity/fold temporal OOF fit."""

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

ROOT = Path(r"E:\SPARROW")
HERE = ROOT / "5_Test/20260904_4/scripts"
RUN = ROOT / "5_Test/20260904_4"
WORK = RUN / "work"
sys.path.insert(0, str(HERE))
from block_refine import refine  # noqa: E402
from temporal_common import FOLDS, atomic_json, json_default, load_trial, observations  # noqa: E402
from unified_fit import FoldDesign, JointObjective, UnifiedTNModel, predict, s41  # noqa: E402


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def accelerated_physical_from_log(
    path: Path, model: UnifiedTNModel
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Recover a deterministic physical-gradient secant proposal from a log."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("stage") != "block_refine" and not bool(payload.get("accepted")):
            continue
        states = {str(row["parameter"]): row for row in payload["process_parameter_states"]}
        physical = np.asarray([float(states[name]["physical"]) for name in model.names()], dtype=float)
        gradient = np.asarray([float(states[name]["physical_gradient"]) for name in model.names()], dtype=float)
        rows.append((physical, gradient))
    if len(rows) < 2:
        raise RuntimeError(f"Need at least two completed refine cycles in {path}")
    (x0, g0), (x1, g1) = rows[-2:]
    denominator = g1 - g0
    candidate = x1.copy()
    usable = np.abs(denominator) > 1.0e-12
    candidate[usable] = x1[usable] - g1[usable] * (x1[usable] - x0[usable]) / denominator[usable]
    fixed = {model.names().index("beta_contact"), model.names().index("v_f")}
    for index, name in enumerate(model.names()):
        if index in fixed:
            candidate[index] = x1[index]
            continue
        span = model.upper[name] - model.lower[name]
        candidate[index] = np.clip(
            candidate[index], model.lower[name] + 1.0e-8 * span, model.upper[name] - 1.0e-8 * span
        )
    if not np.isfinite(candidate).all():
        raise RuntimeError("Nonfinite interrupted-log acceleration proposal")
    # Resume from the last accepted point itself.  Preserve the preceding
    # point as history so ``refine`` can test the secant proposal under its
    # objective/KKT acceptance gate instead of accepting it implicitly.
    return x1.copy(), [(x0.copy(), g0.copy())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", choices=["H7", "H14", "H22"], required=True)
    parser.add_argument("--fold", choices=list(FOLDS), required=True)
    parser.add_argument("--resume-log", type=Path)
    args = parser.parse_args()
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    started = time.perf_counter()
    training_years, evaluation_year = FOLDS[args.fold]
    obs = observations()
    train = obs.loc[obs.year.isin(training_years)].copy()
    test = obs.loc[obs.year.eq(evaluation_year)].copy()
    trials = [load_trial(args.capacity, args.fold, variant) for variant in range(5)]
    if not all(bool(trial["finite"]) for trial in trials):
        raise RuntimeError("Not all temporal starts finite")
    selected_start = int(min(range(5), key=lambda index: float(trials[index]["objective"])))
    best = trials[selected_start]
    model = UnifiedTNModel("formal")
    design = FoldDesign.build(args.capacity, train)
    objective = JointObjective(model, design, train, training_years)
    anchor_vf = float(best["physical"][model.names().index("v_f")])
    profile_values = sorted({float(np.clip(anchor_vf + delta, 0.0, 0.5)) for delta in (-0.05, 0.0, 0.05)})
    profile = []
    for value in profile_values:
        model.set_fixed_v_f(value)
        candidate = best["physical"].copy(); candidate[model.names().index("v_f")] = value
        with torch.no_grad():
            score = float(objective.loss(model.to_raw(candidate), torch.tensor(best["gamma"]), torch.tensor(best["site_raw"])))
        profile.append({"v_f": value, "objective": score})
    selected_vf = float(min(profile, key=lambda row: row["objective"])["v_f"])
    resume_history = None
    if args.resume_log is not None:
        physical, resume_history = accelerated_physical_from_log(args.resume_log, model)
    else:
        physical = best["physical"].copy()
    physical[model.names().index("v_f")] = selected_vf
    model.set_fixed_v_f(selected_vf)
    physical, gamma, site_raw, check, history = refine(
        objective, physical, best["gamma"], best["site_raw"], max_cycles=10,
        initial_gradient_history=resume_history,
    )
    site_effect = objective.site_effect(torch.tensor(site_raw)).detach().numpy()
    reach_effect = design.effect(torch.tensor(gamma)).detach().numpy()
    result = {
        "physical": physical, "gamma": gamma, "site_raw": site_raw, "site_effect": site_effect,
        "reach_effect": reach_effect, "stations": objective.stations, "design": design,
        "selected_v_f": selected_vf, "training_years": training_years,
    }
    layers = predict(model, result, test)
    base = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
    frames = []
    for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
        frame = base.copy()
        frame["pred_tn_mg_l"] = np.where(np.isfinite(layers[layer]), np.maximum(np.expm1(layers[layer]), 0.0), np.nan)
        frame["conditional_available"] = np.isfinite(layers[layer]) if layer == "gauged_conditional" else True
        frame["capacity"] = args.capacity; frame["fold_id"] = args.fold; frame["layer"] = layer
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    carrier = model.carrier_diagnostics(torch.tensor(physical))
    stem = f"{args.capacity.lower()}_{args.fold.lower()}_final"
    parameter = {
        "capacity": args.capacity, "fold_id": args.fold, "selected_start": selected_start,
        "objective": check["objective"], "projected_kkt_max": check["combined_kkt"],
        "process_kkt": check["process_kkt"], "site_kkt": check["site_kkt"], "gamma_kkt": check["gamma_kkt"],
        "selected_v_f": selected_vf, "training_years_json": json.dumps(training_years),
        "evaluation_year": evaluation_year, "train_rows": len(train), "test_rows": len(test),
        "train_stations": train.station_key.nunique(), "runtime_seconds": time.perf_counter() - started,
        "all_starts_json": json.dumps([{"variant": i, "objective": t["objective"], "kkt": t["combined_kkt"]} for i, t in enumerate(trials)]),
        "v_f_profile_json": json.dumps(profile), "refinement_history_json": json.dumps(history, default=json_default),
        "mass_balance_relative": carrier["mass_balance_relative"], "design_audit_json": design.audit_json(),
    }
    parameter.update(dict(zip(model.names(), map(float, physical))))
    atomic_parquet(predictions, WORK / f"{stem}_predictions.parquet")
    atomic_parquet(pd.DataFrame([parameter]), WORK / f"{stem}_parameters.parquet")
    atomic_parquet(pd.DataFrame({"feature": design.fields, "gamma": gamma, "capacity": args.capacity, "fold_id": args.fold}), WORK / f"{stem}_gamma.parquet")
    atomic_parquet(pd.DataFrame({"station_key": objective.stations, "site_raw": site_raw, "station_residual_log_unit": site_effect, "capacity": args.capacity, "fold_id": args.fold}), WORK / f"{stem}_sites.parquet")
    atomic_parquet(pd.DataFrame({"reach_id": np.arange(1, 231), "transferable_offset_log_unit": reach_effect, "capacity": args.capacity, "fold_id": args.fold}), WORK / f"{stem}_offsets.parquet")
    status = "PASS_TEMPORAL_FIT_CHECKPOINT" if check["combined_kkt"] <= 1.0e-5 and carrier["mass_balance_relative"] <= 1.0e-10 else "FAIL_TEMPORAL_FIT_CHECKPOINT"
    atomic_json({"status": status, "capacity": args.capacity, "fold_id": args.fold, "kkt": check["combined_kkt"], "mass_balance_relative": carrier["mass_balance_relative"]}, WORK / f"{stem}_checkpoint.json")
    rss, peak = s41.s28.memory_gib()
    print(json.dumps({"status": status, "capacity": args.capacity, "fold": args.fold, "kkt": check["combined_kkt"], "runtime_seconds": time.perf_counter() - started, "rss_gib": rss, "peak_gib": peak}), flush=True)
    if status.startswith("FAIL"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
