"""Resumable shard worker for nested spatial R2 TN evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260831_4"
WORK = RUN / "work"
CONTRACT = RUN / "experiment_contract.json"
REFINEMENT_CONTRACT = RUN / "refinement_contract.json"
CORE = ROOT / "5_Test/20260831_2/scripts"
S3 = ROOT / "5_Test/20260831_3/scripts"
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(S3))
from reservoir_tn_core import ReservoirTNModel  # noqa: E402
from run_stage3_worker import Objective, optimize_at_vf  # noqa: E402
import stage3_common as common  # noqa: E402


OLD_PARAMETERS = ROOT / "5_Test/20260824_47/outputs/stage47_nested_spatial_parameters.parquet"
TEMPORAL_PARAMETERS = ROOT / "5_Test/20260831_3/outputs/objective_fold_parameters.parquet"
TEMPORAL_SITES = ROOT / "5_Test/20260831_3/outputs/objective_site_effects.parquet"


def fold_frames(obs: pd.DataFrame, row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_id = str(row.fold_id)
    temporal = fold_id.split("_", 1)[0]
    start, end, evaluation = common.FOLDS[temporal]
    train = obs.loc[obs.year.between(start, end)].copy()
    test = obs.loc[obs.year.eq(evaluation)].copy()
    if row.holdout_type == "REACH":
        holdout = int(row.holdout_id); train = train.loc[train.reach_id.ne(holdout)]; test = test.loc[test.reach_id.eq(holdout)]
    elif row.holdout_type == "TREE":
        holdout = int(row.holdout_id); train = train.loc[train.terminal_tree_id.ne(holdout)]; test = test.loc[test.terminal_tree_id.eq(holdout)]
    else:
        raise ValueError(row.holdout_type)
    if train.empty or test.empty:
        raise RuntimeError(f"Empty nested fold {fold_id}")
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--phase", choices=["screen", "refined_screen", "reference_screen", "full"], default="screen")
    args = parser.parse_args()
    if Path(sys.prefix).name.lower() != "sparrow": raise RuntimeError("conda sparrow required")
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    contract_path = REFINEMENT_CONTRACT if args.phase in ["refined_screen", "reference_screen"] else CONTRACT
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_status = "REGISTERED_BEFORE_REFINED_RESULTS" if args.phase in ["refined_screen", "reference_screen"] else "REGISTERED_BEFORE_RESULTS"
    if contract.get("status") != expected_status: raise RuntimeError(f"Stage4 {args.phase} not registered")
    old = pd.read_parquet(OLD_PARAMETERS).loc[lambda x: x.candidate.eq("MINERAL_LIFETIME") & x.holdout_type.isin(["REACH", "TREE"])].copy()
    if args.phase in ["screen", "refined_screen", "reference_screen"]: old = old.loc[old.fold_id.str.startswith("T3_")].copy()
    else: old = old.loc[old.fold_id.str.startswith(("T1_", "T2_"))].copy()
    old = old.sort_values("fold_id").reset_index(drop=True)
    assigned = old.loc[np.arange(len(old)) % args.shards == args.shard].copy()
    prefix = f"{args.phase}_s{args.shard:02d}of{args.shards:02d}"
    pred_path = WORK / f"{prefix}_predictions.parquet"
    par_path = WORK / f"{prefix}_parameters.parquet"
    existing_pred = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    existing_par = pd.read_parquet(par_path) if par_path.exists() else pd.DataFrame()
    if args.phase == "reference_screen" and not existing_par.empty:
        valid = set(existing_par.loc[existing_par.gradient_max_abs.le(0.005), "fold_id"].astype(str))
        existing_par = existing_par.loc[existing_par.fold_id.astype(str).isin(valid)].copy()
        if not existing_pred.empty:
            existing_pred = existing_pred.loc[existing_pred.fold_id.astype(str).isin(valid)].copy()
    complete = set(existing_par.fold_id.astype(str)) if not existing_par.empty else set()
    obs = common.observations(); model = ReservoirTNModel(reservoir_enabled=True)
    temporal_parameters = pd.read_parquet(TEMPORAL_PARAMETERS).loc[lambda x: x.objective_id.eq("O0_LOG_T4")].set_index("fold_id")
    temporal_sites = pd.read_parquet(TEMPORAL_SITES).loc[lambda x: x.objective_id.eq("O0_LOG_T4")]
    prediction_frames = [existing_pred] if not existing_pred.empty else []
    parameter_rows = existing_par.to_dict("records") if not existing_par.empty else []
    for row in assigned.itertuples(index=False):
        if row.fold_id in complete: continue
        started = time.perf_counter(); series = pd.Series(row._asdict())
        train, test = fold_frames(obs, series)
        # l0-v2 caches observation indices by Python object id.  DataFrame ids
        # can be reused after a previous nested fold is collected, so a long
        # worker must clear both caches at every fold boundary.
        model._obs_index_cache.clear(); model._q_feature_cache.clear()
        if args.phase == "reference_screen":
            temporal_fold = str(row.fold_id).split("_", 1)[0]
            temporal_row = temporal_parameters.loc[temporal_fold]
            anchor = np.asarray([temporal_row[name] for name in model.names()], dtype=float)
            initial_v_f = float(temporal_row.selected_v_f)
        else:
            anchor = np.asarray([getattr(row, name) for name in model.names()], dtype=float)
            initial_v_f = float(row.v_f)
        model.set_fixed_v_f(initial_v_f)
        if args.phase == "reference_screen":
            temporal_fold = str(row.fold_id).split("_", 1)[0]
            site_map = temporal_sites.loc[temporal_sites.fold_id.eq(temporal_fold)].set_index("station_key").b_raw_log_unit.to_dict()
            stations = sorted(train.station_key.astype(str).unique())
            site_start = np.asarray([float(site_map.get(station, 0.0)) for station in stations])
            adam_steps, adam_lr, lbfgs_steps = 0, 0.0, 8
        elif args.phase == "refined_screen":
            with torch.no_grad():
                _, anchor_population = model.evaluate(
                    train, torch.tensor(anchor), int(train.year.min()), int(train.year.max())
                )
            residual = np.log1p(train.tn_mg_l.to_numpy(float)) - anchor_population.numpy()
            station_values = train.station_key.astype(str)
            stations = sorted(station_values.unique())
            site_start = np.asarray([
                float(np.mean(residual[station_values.eq(station)])) for station in stations
            ])
            model._obs_index_cache.clear(); model._q_feature_cache.clear()
            adam_steps, adam_lr, lbfgs_steps = 3, 0.02, 6
        else:
            site_start = np.zeros(train.station_key.nunique(), dtype=float)
            adam_steps, adam_lr, lbfgs_steps = 1, 0.015, 2
        objective = Objective(model, train, "O0_LOG_T4", anchor, site_start)
        fit = optimize_at_vf(objective, anchor, site_start, initial_v_f, adam_steps, adam_lr, lbfgs_steps)
        candidates = sorted(set(float(np.clip(initial_v_f + offset, 0.0, 0.5)) for offset in [-0.05, 0.0, 0.05]))
        profile = []
        for v_f in candidates:
            model.set_fixed_v_f(v_f); physical = fit["physical"].copy(); physical[model.names().index("v_f")] = v_f
            with torch.no_grad(): value = float(objective.loss(model.to_raw(physical), torch.tensor(fit["site_raw"])))
            profile.append({"v_f": v_f, "objective": value})
        selected_vf = min(profile, key=lambda item: item["objective"])["v_f"]
        if abs(selected_vf - initial_v_f) > 1.0e-12:
            if args.phase == "reference_screen":
                fit = optimize_at_vf(objective, fit["physical"], fit["site_raw"], selected_vf, 0, 0.0, 8)
            elif args.phase == "refined_screen":
                fit = optimize_at_vf(objective, fit["physical"], fit["site_raw"], selected_vf, 3, 0.02, 6)
            else:
                fit = optimize_at_vf(objective, fit["physical"], fit["site_raw"], selected_vf, 1, 0.01, 1)
        final_vf_lbfgs_steps = lbfgs_steps
        topup_rounds = 0
        if args.phase == "reference_screen":
            while fit["gradient_max_abs"] > 0.005 and final_vf_lbfgs_steps < 20:
                fit = optimize_at_vf(objective, fit["physical"], fit["site_raw"], selected_vf, 0, 0.0, 4)
                final_vf_lbfgs_steps += 4
                topup_rounds += 1
        if not fit["finite"]:
            raise RuntimeError(f"Nonfinite Stage4 fit {row.fold_id}")
        model.set_fixed_v_f(selected_vf)
        model._obs_index_cache.clear(); model._q_feature_cache.clear()
        with torch.no_grad(): _, population = model.evaluate(test, torch.tensor(fit["physical"]), int(train.year.min()), int(train.year.max()))
        frame = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
        frame["pred_tn_mg_l"] = np.maximum(np.expm1(population.numpy()), 0.0)
        station_means = train.assign(y=np.log1p(train.tn_mg_l)).groupby("station_key").y.mean()
        baseline_log = float(station_means.mean())
        frame["baseline_pred_tn_mg_l"] = max(float(np.expm1(baseline_log)), 0.0)
        frame["fold_id"] = row.fold_id; frame["holdout_type"] = row.holdout_type; frame["holdout_id"] = str(row.holdout_id)
        frame["layer"] = "population_transferable"; frame["candidate"] = "R2_CONSERVATIVE_O0"
        prediction_frames.append(frame)
        parameter = {
            "fold_id": row.fold_id, "holdout_type": row.holdout_type, "holdout_id": str(row.holdout_id),
            "objective": fit["objective"], "gradient_max_abs": fit["gradient_max_abs"], "selected_v_f": selected_vf,
            "v_f_profile_json": json.dumps(profile), "train_rows": len(train), "test_rows": len(test),
            "site_initialization": (
                "temporal_o0_site_effect" if args.phase == "reference_screen" else
                "training_residual_mean" if args.phase == "refined_screen" else "zero"
            ),
            "adam_steps": adam_steps, "adam_lr": adam_lr, "lbfgs_steps": lbfgs_steps,
            "final_v_f_lbfgs_steps": final_vf_lbfgs_steps,
            "gradient_topup_rounds": topup_rounds,
            "gradient_target_met": bool(fit["gradient_max_abs"] <= 0.005) if args.phase in ["refined_screen", "reference_screen"] else None,
            "runtime_seconds": time.perf_counter() - started,
        }
        parameter.update(dict(zip(model.names(), map(float, fit["physical"]))))
        parameter_rows.append(parameter)
        common.atomic_parquet(pd.concat(prediction_frames, ignore_index=True), pred_path)
        common.atomic_parquet(pd.DataFrame(parameter_rows), par_path)
        print(json.dumps({"event": "fold_complete", "shard": args.shard, "fold": row.fold_id, "runtime_seconds": parameter["runtime_seconds"]}), flush=True)
        gc.collect()
    common.atomic_json({
        "status": "SHARD_COMPLETE", "phase": args.phase, "shard": args.shard, "shards": args.shards,
        "folds": len(assigned), "predictions_sha256": common.sha256(pred_path), "parameters_sha256": common.sha256(par_path),
    }, WORK / f"{prefix}_checkpoint.json")
    print(json.dumps({"status": "SHARD_COMPLETE", "phase": args.phase, "shard": args.shard, "folds": len(assigned)}), flush=True)


if __name__ == "__main__": main()
