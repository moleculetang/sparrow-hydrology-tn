"""Finalize the T3 spatial screen or the complete T1-T3 nested evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260831_4"
WORK = RUN / "work"
OUTPUTS = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"
REFINEMENT_CONTRACT = RUN / "refinement_contract.json"
S3 = ROOT / "5_Test/20260831_3/scripts"
sys.path.insert(0, str(S3))
import stage3_common as common  # noqa: E402


REFERENCE = ROOT / "5_Test/20260824_47/outputs/stage47_nested_spatial_predictions.parquet"
SEED = 2608314
REPLICATES = 10_000
MARGIN = 0.005


def combine(phase: str, kind: str) -> pd.DataFrame:
    frames = []
    if phase == "screen": phases = ["screen"]
    elif phase in ["refined_screen", "reference_screen"]: phases = [phase]
    else: phases = ["reference_screen", "full"]
    for item in phases:
        for shard in range(4):
            path = WORK / f"{item}_s{shard:02d}of04_{kind}.parquet"
            checkpoint = WORK / f"{item}_s{shard:02d}of04_checkpoint.json"
            if not path.exists() or not checkpoint.exists():
                raise RuntimeError(f"Missing Stage4 {item} shard {shard}")
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("status") != "SHARD_COMPLETE" or common.sha256(path) != saved[f"{kind}_sha256"]:
                raise RuntimeError(f"Invalid Stage4 {item} shard {shard}")
            frames.append(pd.read_parquet(path))
    combined = pd.concat(frames, ignore_index=True)
    if phase in ["reference_screen", "full"]:
        for rescue_path in WORK.glob(f"reference_rescue_*_{kind}.parquet"):
            rescue = pd.read_parquet(rescue_path)
            rescue_folds = set(rescue.fold_id.astype(str))
            combined = combined.loc[~combined.fold_id.astype(str).isin(rescue_folds)].copy()
            combined = pd.concat([combined, rescue], ignore_index=True)
    return combined


def block_comparison(joined: pd.DataFrame, holdout_type: str) -> dict[str, object]:
    frame = joined.loc[joined.holdout_type.eq(holdout_type)].copy()
    values = []
    for block, group in frame.groupby("holdout_id"):
        y = np.log1p(group.tn_mg_l.to_numpy(float))
        candidate = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate) - y) ** 2))
        reference = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_reference) - y) ** 2))
        values.append((block, candidate, reference))
    table = pd.DataFrame(values, columns=["block", "candidate", "reference"])
    delta = table.candidate.to_numpy() - table.reference.to_numpy()
    rng = np.random.default_rng(SEED + (0 if holdout_type == "REACH" else 1000))
    draw = np.empty(REPLICATES)
    for i in range(REPLICATES):
        index = rng.integers(0, len(delta), len(delta)); draw[i] = delta[index].mean()
    lower, upper = np.quantile(draw, [0.025, 0.975])
    candidate_error = np.log1p(frame.pred_tn_mg_l_candidate.to_numpy(float)) - np.log1p(frame.tn_mg_l.to_numpy(float))
    baseline_error = np.log1p(frame.baseline_pred_tn_mg_l.to_numpy(float)) - np.log1p(frame.tn_mg_l.to_numpy(float))
    skill = 1.0 - np.sum(candidate_error**2) / np.sum(baseline_error**2)
    skill_draw = np.empty(REPLICATES)
    groups = [group for _, group in frame.groupby("holdout_id")]
    for i in range(REPLICATES):
        index = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[j] for j in index], ignore_index=True)
        error = np.log1p(sample.pred_tn_mg_l_candidate.to_numpy(float)) - np.log1p(sample.tn_mg_l.to_numpy(float))
        base = np.log1p(sample.baseline_pred_tn_mg_l.to_numpy(float)) - np.log1p(sample.tn_mg_l.to_numpy(float))
        skill_draw[i] = 1.0 - np.sum(error**2) / np.sum(base**2)
    skill_lower, skill_upper = np.quantile(skill_draw, [0.025, 0.975])
    return {
        "holdout_type": holdout_type, "blocks": len(delta),
        "delta_candidate_minus_reference_log_rmse": float(delta.mean()),
        "delta_ci95_lower": float(lower), "delta_ci95_upper": float(upper),
        "noninferior_0p005": bool(upper < MARGIN), "improved": bool(upper < 0),
        "station_blind_skill_log": float(skill), "skill_ci95_lower": float(skill_lower),
        "skill_ci95_upper": float(skill_upper), "positive_point_skill": bool(skill > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=["screen", "refined_screen", "reference_screen", "full"], default="screen")
    args = parser.parse_args()
    contract_path = REFINEMENT_CONTRACT if args.phase in ["refined_screen", "reference_screen"] else CONTRACT
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_status = "REGISTERED_BEFORE_REFINED_RESULTS" if args.phase in ["refined_screen", "reference_screen"] else "REGISTERED_BEFORE_RESULTS"
    if contract.get("status") != expected_status: raise RuntimeError(f"Stage4 {args.phase} not registered")
    prediction = combine(args.phase, "predictions")
    parameter = combine(args.phase, "parameters")
    expected = 110 if args.phase in ["screen", "refined_screen", "reference_screen"] else 331
    if parameter.fold_id.nunique() != expected or prediction.fold_id.nunique() != expected:
        raise RuntimeError(f"Expected {expected} unique folds, got {parameter.fold_id.nunique()}")
    if prediction.duplicated(["fold_id", "station_key", "reach_id", "year", "month"]).any():
        raise RuntimeError("Nested prediction keys duplicated")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    common.atomic_parquet(prediction, OUTPUTS / f"{args.phase}_nested_predictions.parquet")
    common.atomic_parquet(parameter, OUTPUTS / f"{args.phase}_nested_parameters.parquet")

    reference = pd.read_parquet(REFERENCE).loc[
        lambda x: x.candidate.eq("MINERAL_LIFETIME") & x.layer.eq("population_transferable") & x.holdout_type.isin(["REACH", "TREE"])
    ].copy()
    if args.phase in ["screen", "refined_screen", "reference_screen"]: reference = reference.loc[reference.fold_id.str.startswith("T3_")]
    keys = ["fold_id", "holdout_type", "holdout_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    joined = prediction[keys + ["pred_tn_mg_l", "baseline_pred_tn_mg_l"]].merge(
        reference[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one",
    )
    primary = joined.loc[joined.reach_id.ne(17) & joined.terminal_tree_id.ne(163)].copy()
    comparisons = [block_comparison(primary, holdout) for holdout in ["REACH", "TREE"]]
    table = pd.DataFrame(comparisons)
    common.atomic_parquet(table, OUTPUTS / f"{args.phase}_spatial_comparisons.parquet")
    gates = {
        "LORO_noninferior": bool(table.loc[table.holdout_type.eq("REACH"), "noninferior_0p005"].iloc[0]),
        "LOTO_noninferior": bool(table.loc[table.holdout_type.eq("TREE"), "noninferior_0p005"].iloc[0]),
        "LORO_positive_point_skill": bool(table.loc[table.holdout_type.eq("REACH"), "positive_point_skill"].iloc[0]),
        "LOTO_positive_point_skill": bool(table.loc[table.holdout_type.eq("TREE"), "positive_point_skill"].iloc[0]),
    }
    optimizer_audit = {
        "gradient_target": 0.005 if args.phase in ["refined_screen", "reference_screen"] else None,
        "folds_above_target": int((parameter.gradient_max_abs > 0.005).sum()) if args.phase in ["refined_screen", "reference_screen"] else None,
        "gradient_max_abs_max": float(parameter.gradient_max_abs.max()),
        "gradient_max_abs_median": float(parameter.gradient_max_abs.median()),
    }
    optimizer_valid = args.phase not in ["refined_screen", "reference_screen"] or optimizer_audit["folds_above_target"] == 0
    passed = all(gates.values()) and optimizer_valid
    reach17 = joined.loc[joined.reach_id.eq(17)].copy()
    special = {
        "reach17_rows": len(reach17),
        "reach17_candidate_log_rmse": float(np.sqrt(np.mean((np.log1p(reach17.pred_tn_mg_l_candidate) - np.log1p(reach17.tn_mg_l)) ** 2))) if len(reach17) else None,
        "reach17_reference_log_rmse": float(np.sqrt(np.mean((np.log1p(reach17.pred_tn_mg_l_reference) - np.log1p(reach17.tn_mg_l)) ** 2))) if len(reach17) else None,
        "tree163_rows": int((joined.terminal_tree_id == 163).sum()),
    }
    if args.phase == "screen":
        status = "PASS_T3_SPATIAL_SCREEN_READY_FOR_FULL_NESTED" if passed else "STOP_T3_SPATIAL_SCREEN_FAILED"
        successor = "20260831_4 full" if passed else None
    elif args.phase == "refined_screen":
        if not optimizer_valid:
            status = "STOP_T3_REFINED_SCREEN_OPTIMIZER_CONFOUNDED"
            successor = None
        else:
            status = "PASS_T3_REFINED_SPATIAL_SCREEN_READY_FOR_FULL_NESTED" if passed else "STOP_T3_REFINED_SPATIAL_SCREEN_FAILED"
            successor = "20260831_4 full" if passed else None
    elif args.phase == "reference_screen":
        if not optimizer_valid:
            status = "STOP_T3_REFERENCE_EQUIVALENT_SCREEN_OPTIMIZER_CONFOUNDED"
            successor = None
        else:
            status = "PASS_T3_REFERENCE_EQUIVALENT_SPATIAL_SCREEN_READY_FOR_FULL_NESTED" if passed else "STOP_T3_REFERENCE_EQUIVALENT_SPATIAL_SCREEN_FAILED"
            successor = "20260831_4 full" if passed else None
    else:
        status = "PASS_FULL_SPATIAL_GATE_READY_FOR_RESERVOIR_REACTION" if passed else "STOP_FULL_SPATIAL_GATE_FAILED"
        successor = "20260831_5" if passed else None
    decision = {
        "stage": "20260831_4", "phase": args.phase, "status": status,
        "folds": expected, "gates": gates, "comparisons": comparisons,
        "optimizer_audit": optimizer_audit,
        "special_audits": special,
        "claim_boundary": "Spatial evidence is station-blind relative performance. Tree bootstrap has only seven independent formal blocks; its CI is supporting robustness, not a large-sample certainty statement.",
        "output_hashes": {
            "predictions": common.sha256(OUTPUTS / f"{args.phase}_nested_predictions.parquet"),
            "parameters": common.sha256(OUTPUTS / f"{args.phase}_nested_parameters.parquet"),
            "comparisons": common.sha256(OUTPUTS / f"{args.phase}_spatial_comparisons.parquet"),
        },
        "authorized_successor": successor,
    }
    common.atomic_json(decision, REPORTS / f"{args.phase}_spatial_decision.json")
    common.atomic_json({
        "stage": "20260831_4", "phase": args.phase, "status": status,
        "decision_sha256": common.sha256(REPORTS / f"{args.phase}_spatial_decision.json"),
        "authorized_successor": successor,
    }, LOCKS / f"{args.phase}_spatial_lock.json")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
