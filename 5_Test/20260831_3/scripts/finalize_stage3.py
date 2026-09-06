"""Combine Stage-3 shards, compute preregistered OOF gates and lock one objective."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260831_3"
WORK = RUN / "work"
OUTPUTS = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"
HERE = RUN / "scripts"
sys.path.insert(0, str(HERE))
import stage3_common as common  # noqa: E402


OBJECTIVES = ["O0_LOG_T4", "O1_HET_T4", "O2_DYN_BALANCED"]
FOLDS = ["T1", "T2", "T3"]
LAYERS = ["raw_mass_process", "population_transferable", "gauged_conditional"]
SEED = 260831
REPLICATES = 10_000
MARGIN = 0.005


def primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.reach_id.ne(17) & frame.terminal_tree_id.ne(163)].copy()


def paired_frames(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys = common.KEYS
    return candidate[keys + ["pred_tn_mg_l"]].merge(
        reference[keys + ["pred_tn_mg_l"]], on=keys,
        suffixes=("_candidate", "_reference"), validate="one_to_one",
    )


def log_rmse_comparison(candidate: pd.DataFrame, reference: pd.DataFrame, block: str) -> dict[str, object]:
    joined = paired_frames(candidate, reference)
    values = []
    for key, group in joined.groupby(block):
        observed = np.log1p(group.tn_mg_l.to_numpy(float))
        candidate_error = np.log1p(group.pred_tn_mg_l_candidate.to_numpy(float)) - observed
        reference_error = np.log1p(group.pred_tn_mg_l_reference.to_numpy(float)) - observed
        values.append((key, np.sqrt(np.mean(candidate_error**2)), np.sqrt(np.mean(reference_error**2))))
    table = pd.DataFrame(values, columns=[block, "candidate", "reference"])
    delta = table.candidate.to_numpy() - table.reference.to_numpy()
    rng = np.random.default_rng(SEED + (0 if block == "station_key" else 100))
    draw = np.empty(REPLICATES)
    for i in range(REPLICATES):
        index = rng.integers(0, len(delta), len(delta)); draw[i] = delta[index].mean()
    lower, upper = np.quantile(draw, [0.025, 0.975])
    return {
        "metric": "macro_log_rmse", "block": block, "blocks": len(delta),
        "delta": float(delta.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper),
        "improved": bool(upper < 0), "noninferior_0p005": bool(upper < MARGIN),
    }


def station_nse_table(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    rows = []
    for station, group in frame.groupby("station_key"):
        y = group.tn_mg_l.to_numpy(float); p = group[prediction].to_numpy(float)
        denominator = np.sum((y - y.mean()) ** 2)
        rows.append({"station_key": station, "nse": 1.0 - np.sum((p - y) ** 2) / denominator if denominator > 0 else np.nan})
    return pd.DataFrame(rows).dropna()


def nse_comparison(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, object]:
    joined = paired_frames(candidate, reference)
    candidate_table = station_nse_table(joined.rename(columns={"pred_tn_mg_l_candidate": "prediction"}), "prediction")
    reference_table = station_nse_table(joined.rename(columns={"pred_tn_mg_l_reference": "prediction"}), "prediction")
    table = candidate_table.merge(reference_table, on="station_key", suffixes=("_candidate", "_reference"), validate="one_to_one")
    delta = (table.nse_candidate - table.nse_reference).to_numpy(float)
    rng = np.random.default_rng(SEED + 200)
    draw = np.empty(REPLICATES)
    for i in range(REPLICATES):
        index = rng.integers(0, len(delta), len(delta)); draw[i] = np.median(delta[index])
    lower, upper = np.quantile(draw, [0.025, 0.975])
    return {
        "metric": "paired_station_nse", "block": "station_key", "blocks": len(delta),
        "delta": float(np.median(delta)), "mean_delta": float(np.mean(delta)),
        "ci95_lower": float(lower), "ci95_upper": float(upper),
        "improved": bool(lower > 0),
    }


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_BEFORE_RESULTS":
        raise RuntimeError("Stage 3 not registered")
    predictions, parameters, sites = [], [], []
    checkpoint_audit = []
    for objective in OBJECTIVES:
        for fold in FOLDS:
            prefix = f"{objective.lower()}_{fold.lower()}"
            checkpoint_path = WORK / f"{prefix}_checkpoint.json"
            if not checkpoint_path.exists():
                raise RuntimeError(f"Missing checkpoint {objective} {fold}")
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            paths = {name: WORK / f"{prefix}_{name}.parquet" for name in ["predictions", "parameters", "sites"]}
            hashes_match = all(common.sha256(paths[name]) == checkpoint["hashes"][name] for name in paths)
            checkpoint_audit.append({"objective_id": objective, "fold_id": fold, "hashes_match": hashes_match})
            if checkpoint.get("status") != "SHARD_COMPLETE" or not hashes_match:
                raise RuntimeError(f"Invalid checkpoint {objective} {fold}")
            predictions.append(pd.read_parquet(paths["predictions"]))
            parameters.append(pd.read_parquet(paths["parameters"]))
            sites.append(pd.read_parquet(paths["sites"]))
    prediction = pd.concat(predictions, ignore_index=True)
    parameter = pd.concat(parameters, ignore_index=True)
    site = pd.concat(sites, ignore_index=True)
    if prediction.duplicated(common.KEYS + ["objective_id", "layer"]).any():
        raise RuntimeError("Stage3 prediction keys duplicated")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    common.atomic_parquet(prediction, OUTPUTS / "objective_temporal_oof_predictions.parquet")
    common.atomic_parquet(parameter, OUTPUTS / "objective_fold_parameters.parquet")
    common.atomic_parquet(site, OUTPUTS / "objective_site_effects.parquet")

    metric_rows = []
    for domain, selected in [("all_river", prediction), ("primary_gate", primary(prediction))]:
        for (objective, layer), frame in selected.groupby(["objective_id", "layer"]):
            metric_rows.append({"domain": domain, "objective_id": objective, "layer": layer, **common.metric_summary(frame)})
    metrics = pd.DataFrame(metric_rows)
    common.atomic_parquet(metrics, OUTPUTS / "objective_temporal_metrics.parquet")

    comparisons = []
    gate = primary(prediction)
    for candidate in ["O1_HET_T4", "O2_DYN_BALANCED"]:
        for layer in ["population_transferable", "gauged_conditional"]:
            cand = gate.loc[(gate.objective_id == candidate) & (gate.layer == layer)]
            ref = gate.loc[(gate.objective_id == "O0_LOG_T4") & (gate.layer == layer)]
            for block in ["station_key", "terminal_tree_id"]:
                comparisons.append({"candidate": candidate, "reference": "O0_LOG_T4", "layer": layer, **log_rmse_comparison(cand, ref, block)})
            comparisons.append({"candidate": candidate, "reference": "O0_LOG_T4", "layer": layer, **nse_comparison(cand, ref)})
    comparison = pd.DataFrame(comparisons)
    common.atomic_parquet(comparison, OUTPUTS / "objective_paired_comparisons.parquet")

    eligibility = {}
    eligible = []
    for candidate in ["O1_HET_T4", "O2_DYN_BALANCED"]:
        table = comparison.loc[comparison.candidate.eq(candidate)]
        log_gate = bool(table.loc[table.metric.eq("macro_log_rmse"), "noninferior_0p005"].all())
        conditional_nse = table.loc[(table.metric.eq("paired_station_nse")) & (table.layer.eq("gauged_conditional"))].iloc[0]
        population_nse = table.loc[(table.metric.eq("paired_station_nse")) & (table.layer.eq("population_transferable"))].iloc[0]
        dynamic_gate = bool(conditional_nse.ci95_lower > 0 and population_nse.delta >= 0)
        eligibility[candidate] = {
            "all_log_rmse_noninferior": log_gate,
            "conditional_station_nse_ci_lower_gt_zero": bool(conditional_nse.ci95_lower > 0),
            "population_station_nse_point_nonworse": bool(population_nse.delta >= 0),
            "eligible": bool(log_gate and dynamic_gate),
        }
        if log_gate and dynamic_gate:
            eligible.append(candidate)

    selection_reason = "no alternative passed dynamics; retain control"
    selected_objective = "O0_LOG_T4"
    unique = True
    head_to_head = None
    if len(eligible) == 1:
        selected_objective = eligible[0]; selection_reason = "one alternative passed every temporal gate"
    elif len(eligible) == 2:
        layer = "gauged_conditional"
        a = gate.loc[(gate.objective_id == eligible[0]) & (gate.layer == layer)]
        b = gate.loc[(gate.objective_id == eligible[1]) & (gate.layer == layer)]
        head_to_head = nse_comparison(a, b)
        if head_to_head["ci95_lower"] > 0:
            selected_objective = eligible[0]; selection_reason = "head-to-head station NSE significantly favored first eligible objective"
        elif head_to_head["ci95_upper"] < 0:
            selected_objective = eligible[1]; selection_reason = "head-to-head station NSE significantly favored second eligible objective"
        else:
            selected_objective = None; unique = False; selection_reason = "eligible alternatives were not significantly ordered"

    status = "PASS_UNIQUE_OBJECTIVE_READY_FOR_SPATIAL" if unique else "STOP_NO_UNIQUE_TEMPORAL_OBJECTIVE"
    decision = {
        "stage": "20260831_3", "status": status,
        "selected_objective": selected_objective, "selection_reason": selection_reason,
        "eligibility": eligibility, "head_to_head": head_to_head,
        "primary_gate_exclusions": {"reach_id": 17, "terminal_tree_id": 163},
        "checkpoint_audit": checkpoint_audit,
        "runtime": {
            "workers_requested_initially": 6,
            "initial_observed_total_rss_gib": 29.7,
            "workers_locked_after_audit": 4,
            "rss_hard_limit_gib": 28,
            "interrupted_and_resumed_shard": "O0_LOG_T4 T3",
            "all_final_shards_complete": True,
        },
        "output_hashes": {
            "predictions": common.sha256(OUTPUTS / "objective_temporal_oof_predictions.parquet"),
            "parameters": common.sha256(OUTPUTS / "objective_fold_parameters.parquet"),
            "sites": common.sha256(OUTPUTS / "objective_site_effects.parquet"),
            "metrics": common.sha256(OUTPUTS / "objective_temporal_metrics.parquet"),
            "comparisons": common.sha256(OUTPUTS / "objective_paired_comparisons.parquet"),
        },
        "authorized_successor": "20260831_4" if unique else None,
    }
    common.atomic_json(decision, REPORTS / "stage3_objective_decision.json")
    common.atomic_json({
        "stage": "20260831_3", "status": status, "selected_objective": selected_objective,
        "contract_sha256": common.sha256(CONTRACT),
        "decision_sha256": common.sha256(REPORTS / "stage3_objective_decision.json"),
        "authorized_successor": decision["authorized_successor"],
    }, LOCKS / "objective_lock.json")

    primary_metrics = metrics.loc[metrics.domain.eq("primary_gate")]
    lines = [
        "# 20260831 Stage 3 objective experiment", "", f"Status: `{status}`.", "",
        f"Selected objective: `{selected_objective}`.", "", selection_reason, "",
        "## Primary-gate OOF metrics", "",
        "| objective | layer | station log-RMSE | station median NSE | pooled NSE | R2 | KGE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary_metrics.itertuples(index=False):
        lines.append(
            f"| {row.objective_id} | {row.layer} | {row.station_macro_log_rmse:.4f} | "
            f"{row.station_median_nse:.4f} | {row.nse:.4f} | {row.r2:.4f} | {row.kge:.4f} |"
        )
    lines.extend(["", "## Runtime audit", "", "Six workers reached 29.7 GiB and violated the 28-GiB contract. One uncheckpointed shard was interrupted; the safe concurrency was locked to four, and all nine shards then completed with valid hashes."])
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
