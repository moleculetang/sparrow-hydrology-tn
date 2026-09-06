"""Run and aggregate complete-tree zero-target-history spatial evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_17"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE16 = ROOT / "5_Test" / "20260826_16"
TREES = [1, 20, 22, 26, 56, 166, 212, 217]
MODELS = ["DPL_HBV_LOCAL_STATIC", "DPL_HBV_MULTISCALE_STATIC"]
SEEDS = [260826, 260827, 260828]
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 26082617


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_workers() -> None:
    pending = []
    for tree in TREES:
        metadata = OUT / "folds" / f"tree_{tree}" / "metadata.json"
        if metadata.is_file() and json.loads(metadata.read_text(encoding="utf-8")).get("completed"):
            print(f"tree {tree} already complete", flush=True)
            continue
        log_path = REPORTS / f"tree_{tree}_worker.log"
        log_stream = log_path.open("w", encoding="utf-8")
        command = [sys.executable, str(RUN / "scripts" / "fold_worker.py"), "--tree", str(tree)]
        process = subprocess.Popen(command, stdout=log_stream, stderr=subprocess.STDOUT, text=True)
        pending.append({"tree": tree, "process": process, "log": log_stream, "path": log_path})
    while pending:
        for item in list(pending):
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["log"].close()
            pending.remove(item)
            if returncode != 0:
                tail = item["path"].read_text(encoding="utf-8", errors="replace")[-5000:]
                raise RuntimeError(f"tree {item['tree']} worker failed:\n{tail}")
            print(f"tree {item['tree']} complete; remaining={len(pending)}", flush=True)
        time.sleep(5)


def bootstrap(values: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1)
    return {
        "point_delta": float(values.mean()),
        "ci95_lower": float(np.quantile(sampled, 0.025)),
        "ci95_upper": float(np.quantile(sampled, 0.975)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    stage16 = json.loads((STAGE16 / "reports" / "stage16_decision.json").read_text(encoding="utf-8"))
    if stage16["status"] != "STATIC_DPL_TEMPORAL_GATE_FAILED":
        raise RuntimeError("Stage 17 diagnostic contract expects the observed temporal-gate failure")
    contract = {
        "stage": "20260826_17",
        "purpose": "Complete the already registered zero-target-history spatial evidence even though temporal high-flow failure precludes promotion.",
        "heldout_terminal_trees": TREES,
        "target_Q_2010_2018": "evaluation only after fold model lock; excluded from optimization, early stopping and selection",
        "fold_parent": "global eight-parameter HBV retrained on the seven remaining trees",
        "candidate_nesting": "candidate epoch 0 exactly equals its fold-specific global parent",
        "optimization": "same optimizer, prior, 2010-2015 training, 2016 early stopping and seeds as Stage 16",
        "promotion_possible": False,
        "reason_promotion_impossible": "Stage 16 high-flow noninferiority failed in all six candidate-seed runs",
        "2019_2022_read": False,
        "TN_read": False,
    }
    write_json(RUN / "experiment_contract.json", contract)
    run_workers()

    metrics = []
    summaries = []
    predictions = []
    for tree in TREES:
        fold = OUT / "folds" / f"tree_{tree}"
        metrics.append(pd.read_parquet(fold / "station_metrics.parquet"))
        summaries.append(pd.read_parquet(fold / "run_summary.parquet"))
        predictions.append(pd.read_parquet(fold / "predictions.parquet"))
    metric_frame = pd.concat(metrics, ignore_index=True)
    run_frame = pd.concat(summaries, ignore_index=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame.to_parquet(OUT / "nested_spatial_station_performance.parquet", index=False)
    run_frame.to_parquet(OUT / "nested_spatial_run_summary.parquet", index=False)
    prediction_frame.to_parquet(OUT / "nested_spatial_predictions.parquet", index=False)

    comparison_rows = []
    gate_rows = []
    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(SEEDS):
            candidate = metric_frame.loc[metric_frame.model_id.eq(model) & metric_frame.seed.eq(seed)].copy()
            parent = metric_frame.loc[metric_frame.model_id.eq("GLOBAL_HBV_FOLD_PARENT")].copy()
            merged = candidate.merge(
                parent,
                on=["station_norm", "reach_id", "terminal_tree"],
                suffixes=("_candidate", "_parent"),
                validate="one_to_one",
            )
            boot = {}
            for field_index, field in enumerate(("log_RMSE", "low_log_RMSE", "high_log_RMSE")):
                valid = merged[[f"{field}_candidate", f"{field}_parent"]].notna().all(axis=1)
                delta = (
                    merged.loc[valid, f"{field}_candidate"] - merged.loc[valid, f"{field}_parent"]
                )
                by_tree = delta.groupby(merged.loc[valid, "terminal_tree"]).mean().reindex(TREES).dropna()
                result = bootstrap(by_tree.to_numpy(np.float64), BOOTSTRAP_SEED + 1000 * model_index + 100 * seed_index + field_index)
                boot[field] = result
                comparison_rows.append(
                    {
                        "model_id": model,
                        "seed": seed,
                        "field": field,
                        "tree_count": len(by_tree),
                        **result,
                    }
                )
            candidate_median_nse = float(merged.NSE_candidate.median())
            parent_median_nse = float(merged.NSE_parent.median())
            candidate_abs_pbias = float(merged.PBIAS_pct_candidate.abs().median())
            parent_abs_pbias = float(merged.PBIAS_pct_parent.abs().median())
            gates = {
                "log_RMSE_noninferior": boot["log_RMSE"]["ci95_upper"] < 0.01,
                "median_NSE_noninferior": candidate_median_nse - parent_median_nse >= -0.02,
                "median_abs_PBIAS_noninferior": candidate_abs_pbias - parent_abs_pbias <= 2.0,
                "low_flow_noninferior": boot["low_log_RMSE"]["ci95_upper"] < 0.01,
                "high_flow_noninferior": boot["high_log_RMSE"]["ci95_upper"] < 0.01,
            }
            gate_rows.append(
                {
                    "model_id": model,
                    "seed": seed,
                    "candidate_station_median_NSE": candidate_median_nse,
                    "parent_station_median_NSE": parent_median_nse,
                    "candidate_station_median_absolute_PBIAS_pct": candidate_abs_pbias,
                    "parent_station_median_absolute_PBIAS_pct": parent_abs_pbias,
                    "all_spatial_gates_pass": all(gates.values()),
                    **{f"gate_{name}": value for name, value in gates.items()},
                }
            )
    comparison_frame = pd.DataFrame(comparison_rows)
    gates_frame = pd.DataFrame(gate_rows)
    comparison_frame.to_parquet(OUT / "nested_spatial_tree_block_comparisons.parquet", index=False)
    gates_frame.to_parquet(OUT / "nested_spatial_seed_gates.parquet", index=False)

    model_decisions = {}
    for model in MODELS:
        spatial_pass_count = int(gates_frame.loc[gates_frame.model_id.eq(model), "all_spatial_gates_pass"].sum())
        spatial_improvement_count = int(
            comparison_frame.loc[
                comparison_frame.model_id.eq(model) & comparison_frame.field.eq("log_RMSE"), "ci95_upper"
            ].lt(0.0).sum()
        )
        model_decisions[model] = {
            "spatial_seed_gate_pass_count": spatial_pass_count,
            "spatial_improvement_seed_count": spatial_improvement_count,
            "spatial_status": "SPATIAL_GATE_PASS" if spatial_pass_count >= 2 else "SPATIAL_GATE_FAIL",
            "promoted": False,
            "promotion_blocker": "Stage 16 temporal high-flow gate failed",
        }
    decision = {
        "stage": "20260826_17",
        "status": "SPATIAL_DIAGNOSTIC_COMPLETE_NO_PROMOTION_TEMPORAL_HIGH_FLOW_FAILED",
        "models": model_decisions,
        "target_Q_history_used": False,
        "spatially_promoted": [],
        "authorized_successor": "20260826_18",
    }
    write_json(REPORTS / "stage17_decision.json", decision)
    program = json.loads((STAGE16 / "program_manifest.json").read_text(encoding="utf-8"))
    program["stage_status"] = dict(program["stage_status"])
    program["stage_status"]["20260826_17"] = decision["status"]
    program["authorized_successor"] = "20260826_18"
    write_json(RUN / "program_manifest.json", program)
    report = f"""# 20260826_17 完整河树零目标历史空间评价

状态：`{decision['status']}`。八棵河树分别完整删除目标树2010–2018全部流量，父模型和候选仅在其余七棵树重训；目标流量只在模型锁定后用于评分。

空间结果为诊断证据。即使某候选空间门通过，也不能抵消`20260826_16`全部种子的时间高流非劣失败，因此本阶段没有升级任何全河网产品。

```text
{gates_frame.to_string(index=False)}
```
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    (RUN / "README.md").write_text(
        "# 20260826_17\n\nComplete terminal-tree zero-target-Q-history spatial diagnostic. Temporal high-flow failure makes promotion impossible.\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
