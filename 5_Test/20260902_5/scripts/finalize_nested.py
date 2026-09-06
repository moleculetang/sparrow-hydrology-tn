"""Aggregate H22 nested-spatial shards and apply the preregistered gates."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / r"5_Test\20260902_5"
WORK, OUT, REPORTS, LOCKS = (RUN / name for name in ("work", "outputs", "reports", "locks"))
PARENT_PATH = ROOT / r"5_Test\20260824_47\work\mineral_lifetime_nested_predictions.parquet"
CONTRACT = RUN / "experiment_contract.json"
MANIFEST = ROOT / r"5_Test\20260902_1\program_manifest.json"
MARGIN = 0.005
REPLICATES = 10_000
SEED = 260905
KEYS = ["fold_id", "holdout_type", "holdout_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def joined_frame(candidate: pd.DataFrame, parent: pd.DataFrame, holdout_type: str) -> pd.DataFrame:
    left = candidate.loc[candidate.holdout_type.eq(holdout_type), KEYS + ["pred_tn_mg_l", "baseline_pred_tn_mg_l"]]
    right = parent.loc[parent.holdout_type.eq(holdout_type), KEYS + ["pred_tn_mg_l"]]
    return left.merge(right, on=KEYS, suffixes=("_candidate", "_parent"), validate="one_to_one")


def comparison(candidate: pd.DataFrame, parent: pd.DataFrame, holdout_type: str) -> tuple[dict[str, object], pd.DataFrame]:
    block_key = {"REACH": "reach_id", "TREE": "terminal_tree_id", "FIRST_OBSERVED_2021": "station_key"}[holdout_type]
    joined = joined_frame(candidate, parent, holdout_type)
    rows: list[dict[str, object]] = []
    for block, group in joined.groupby(block_key, sort=True):
        obs = np.log1p(group.tn_mg_l.to_numpy(float))
        cand = np.log1p(group.pred_tn_mg_l_candidate.to_numpy(float))
        par = np.log1p(group.pred_tn_mg_l_parent.to_numpy(float))
        base = np.log1p(group.baseline_pred_tn_mg_l.to_numpy(float))
        rows.append({
            "holdout_type": holdout_type,
            "block": str(block),
            "rows": len(group),
            "candidate_rmse": float(np.sqrt(np.mean((cand - obs) ** 2))),
            "parent_rmse": float(np.sqrt(np.mean((par - obs) ** 2))),
            "candidate_sse": float(np.sum((cand - obs) ** 2)),
            "baseline_sse": float(np.sum((base - obs) ** 2)),
        })
    blocks = pd.DataFrame(rows)
    diff = blocks.candidate_rmse.to_numpy() - blocks.parent_rmse.to_numpy()
    rng = np.random.default_rng(SEED + sum(map(ord, holdout_type)))
    delta = np.empty(REPLICATES)
    skill = np.empty(REPLICATES)
    for index in range(REPLICATES):
        take = rng.integers(0, len(blocks), len(blocks))
        sample = blocks.iloc[take]
        delta[index] = diff[take].mean()
        skill[index] = 1.0 - sample.candidate_sse.sum() / sample.baseline_sse.sum()
    dl, du = np.quantile(delta, [0.025, 0.975])
    sl, su = np.quantile(skill, [0.025, 0.975])
    result = {
        "holdout_type": holdout_type,
        "blocks": len(blocks),
        "rows": len(joined),
        "delta_candidate_minus_parent": float(diff.mean()),
        "delta_ci95_lower": float(dl),
        "delta_ci95_upper": float(du),
        "noninferior_0p005": bool(du < MARGIN),
        "improved": bool(du < 0.0),
        "station_blind_skill_log": float(1.0 - blocks.candidate_sse.sum() / blocks.baseline_sse.sum()),
        "skill_ci95_lower": float(sl),
        "skill_ci95_upper": float(su),
        "positive_skill": bool(sl > 0.0),
    }
    blocks["delta_candidate_minus_parent"] = diff
    return result, blocks


def tree_sign_flip(candidate: pd.DataFrame, parent: pd.DataFrame) -> dict[str, object]:
    joined = joined_frame(candidate, parent, "TREE")
    values = []
    for tree, group in joined.groupby("terminal_tree_id", sort=True):
        obs = np.log1p(group.tn_mg_l.to_numpy(float))
        cand = np.log1p(group.pred_tn_mg_l_candidate.to_numpy(float))
        par = np.log1p(group.pred_tn_mg_l_parent.to_numpy(float))
        values.append((str(tree), float(np.sqrt(np.mean((cand - obs) ** 2)) - np.sqrt(np.mean((par - obs) ** 2)))))
    delta = np.asarray([value for _, value in values])
    permutations = np.asarray([np.mean(delta * np.asarray(signs)) for signs in itertools.product((-1.0, 1.0), repeat=len(delta))])
    point = float(delta.mean())
    return {
        "trees": len(delta),
        "tree_deltas": dict(values),
        "permutations": len(permutations),
        "observed_mean_delta": point,
        "two_sided_p": float(np.mean(np.abs(permutations) >= abs(point) - 1e-15)),
    }


def main() -> None:
    for path in (OUT, REPORTS, LOCKS):
        path.mkdir(parents=True, exist_ok=True)
    par_paths = sorted(WORK.glob("h22_nested_parameters_*.parquet"))
    if not par_paths:
        raise RuntimeError("Missing nested worker outputs")
    pred_parts, par_parts, gamma_parts = [], [], []
    for par_path in par_paths:
        suffix = par_path.name.removeprefix("h22_nested_parameters_").removesuffix(".parquet")
        pred_path = WORK / f"h22_nested_predictions_{suffix}.parquet"
        gamma_path = WORK / f"h22_nested_gamma_{suffix}.parquet"
        if not pred_path.exists() or not gamma_path.exists():
            raise RuntimeError(f"Checkpoint triplet incomplete: {suffix}")
        par = pd.read_parquet(par_path).loc[lambda frame: frame.projected_kkt_max <= 1e-5].copy()
        if par.empty:
            continue
        valid = set(par.fold_id.astype(str))
        par["_checkpoint"] = suffix
        pred = pd.read_parquet(pred_path).loc[lambda frame: frame.fold_id.astype(str).isin(valid)].copy()
        gam = pd.read_parquet(gamma_path).loc[lambda frame: frame.fold_id.astype(str).isin(valid)].copy()
        pred["_checkpoint"] = suffix; gam["_checkpoint"] = suffix
        par_parts.append(par); pred_parts.append(pred); gamma_parts.append(gam)
    if not par_parts:
        raise RuntimeError("No KKT-passing nested checkpoints")
    parameter_all = pd.concat(par_parts, ignore_index=True)
    selected = (
        parameter_all.sort_values(["fold_id", "projected_kkt_max"])
        .drop_duplicates("fold_id", keep="first")[["fold_id", "_checkpoint"]]
    )
    parameter = parameter_all.merge(selected, on=["fold_id", "_checkpoint"], how="inner", validate="many_to_one")
    prediction = pd.concat(pred_parts, ignore_index=True).merge(selected, on=["fold_id", "_checkpoint"], how="inner", validate="many_to_one")
    gamma = pd.concat(gamma_parts, ignore_index=True).merge(selected, on=["fold_id", "_checkpoint"], how="inner", validate="many_to_one")
    # Interrupted engineering probes may leave a completed but numerically
    # ineligible fold.  The formal worker recomputes it; retain only the valid
    # KKT-passing realization and its matching prediction/gamma records.
    parameter = parameter.drop(columns="_checkpoint").reset_index(drop=True)
    prediction = prediction.drop(columns="_checkpoint").reset_index(drop=True)
    gamma = gamma.drop(columns="_checkpoint").reset_index(drop=True)
    parent_all = pd.read_parquet(PARENT_PATH)
    parent = parent_all.loc[parent_all.layer.eq("population_transferable")].copy()
    candidate = prediction.loc[prediction.layer.eq("population_transferable")].copy()
    comparisons, block_frames = {}, []
    for holdout in ("REACH", "TREE", "FIRST_OBSERVED_2021"):
        result, blocks = comparison(candidate, parent, holdout)
        comparisons[holdout] = result
        block_frames.append(blocks)
    expected = {"REACH": 309, "TREE": 21, "FIRST_OBSERVED_2021": 1}
    fold_counts = parameter.groupby("holdout_type").fold_id.nunique().to_dict()
    heldout_conditional = prediction.loc[prediction.layer.eq("gauged_conditional")]
    checks = {
        "all_331_folds": all(int(fold_counts.get(kind, 0)) == count for kind, count in expected.items()),
        "parameter_fold_ids_unique": not parameter.fold_id.duplicated().any(),
        "all_projected_kkt_le_1e_5": bool((parameter.projected_kkt_max <= 1e-5).all()),
        "population_predictions_finite": bool(np.isfinite(candidate.pred_tn_mg_l).all()),
        "heldout_conditional_unavailable": bool(heldout_conditional.pred_tn_mg_l.isna().all()),
        "candidate_parent_keys_exact": set(map(tuple, candidate[KEYS].itertuples(index=False, name=None))) == set(map(tuple, parent[KEYS].itertuples(index=False, name=None))),
        "prediction_keys_unique": not prediction.duplicated(["layer"] + KEYS).any(),
        "gamma_keys_unique": not gamma.duplicated(["fold_id", "feature"]).any(),
        "no_partial_files": not any(RUN.rglob("*.part")),
    }
    spatial_pass = bool(
        comparisons["REACH"]["noninferior_0p005"]
        and comparisons["TREE"]["noninferior_0p005"]
        and comparisons["REACH"]["positive_skill"]
        and comparisons["TREE"]["positive_skill"]
        and comparisons["FIRST_OBSERVED_2021"]["noninferior_0p005"]
    )
    status = "PASS_H22_SPATIAL_GATES" if all(checks.values()) and spatial_pass else ("FAIL_H22_SPATIAL_GATES" if all(checks.values()) else "FAIL_STAGE5_ENGINEERING")
    paths = {
        "predictions": OUT / "h22_nested_spatial_predictions.parquet",
        "parameters": OUT / "h22_nested_spatial_parameters.parquet",
        "gamma": OUT / "h22_nested_gamma_coefficients.parquet",
        "block_metrics": OUT / "h22_nested_block_metrics.parquet",
    }
    atomic_parquet(prediction, paths["predictions"])
    atomic_parquet(parameter, paths["parameters"])
    atomic_parquet(gamma, paths["gamma"])
    atomic_parquet(pd.concat(block_frames, ignore_index=True), paths["block_metrics"])
    decision = {
        "stage": "20260902_5",
        "status": status,
        "candidate": "H22_TRANSFER_HEAD",
        "spatially_eligible": spatial_pass and all(checks.values()),
        "comparisons": comparisons,
        "tree_sign_flip": tree_sign_flip(candidate, parent),
        "checks": checks,
        "max_projected_kkt": float(parameter.projected_kkt_max.max()),
        "max_abs_transferable_offset": float(parameter.max_abs_transferable_offset.max()),
        "input_hashes": {str(path): sha256(path) for path in (CONTRACT, PARENT_PATH)},
        "output_hashes": {name: sha256(path) for name, path in paths.items()},
        "authorized_successor": "20260902_6" if status == "PASS_H22_SPATIAL_GATES" else None,
    }
    atomic_json(decision, REPORTS / "spatial_decision.json")
    atomic_json({
        "stage": "20260902_5",
        "status": status,
        "spatially_eligible": decision["spatially_eligible"],
        "decision_sha256": sha256(REPORTS / "spatial_decision.json"),
        "authorized_successor": decision["authorized_successor"],
    }, LOCKS / "stage5_lock.json")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["stage_status"]["20260902_5"] = "completed_pass" if status == "PASS_H22_SPATIAL_GATES" else "completed_fail"
    manifest["stage_status"]["20260902_6"] = "authorized_next" if status == "PASS_H22_SPATIAL_GATES" else "closed_not_authorized"
    if status != "PASS_H22_SPATIAL_GATES":
        manifest["status"] = "complete_without_spatial_promotion"
    atomic_json(manifest, MANIFEST)
    lines = [
        "# `20260902_5` H22 nested spatial validation", "", f"Status: `{status}`.", "",
        "| holdout | blocks | delta vs Parent | CI95 | noninferior | absolute Skill_log | Skill CI95 |", "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in comparisons.values():
        lines.append(f"| {row['holdout_type']} | {row['blocks']} | {row['delta_candidate_minus_parent']:.5f} | {row['delta_ci95_lower']:.5f}–{row['delta_ci95_upper']:.5f} | {row['noninferior_0p005']} | {row['station_blind_skill_log']:.3f} | {row['skill_ci95_lower']:.3f}–{row['skill_ci95_upper']:.3f} |")
    lines += ["", "All selection uses the population-transferable layer. Held-out conditional predictions are unavailable by contract."]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if status == "FAIL_STAGE5_ENGINEERING":
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
