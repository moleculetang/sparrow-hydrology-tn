"""Register, execute, aggregate, and decide Stage 26 spatial folds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_27"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
WORKER = RUN / "scripts" / "spatial_2p_worker.py"
PYTHON = Path(sys.executable)
TREES = [1, 20, 22, 26, 56, 166, 212, 217]
MODELS = ["DYN2P"]
BOOTSTRAPS = 10000
SEED = 26082626


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_fold(tree: int) -> tuple[int, int, str]:
    log_path = REPORTS / f"tree_{tree}_worker.log"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    command = [str(PYTHON), str(WORKER), "--tree", str(tree)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    return tree, result.returncode, result.stdout[-1000:] + result.stderr[-1000:]


def bootstrap(metrics: pd.DataFrame, model: str, field: str, seed: int) -> dict[str, object]:
    pivot = metrics.loc[metrics.model_id.isin(["FOLD_PARENT", model])].pivot_table(
        index=["station_norm", "terminal_tree"], columns="model_id", values=field, aggfunc="mean"
    ).dropna()
    station_delta = pivot[model] - pivot["FOLD_PARENT"]
    tree_delta = station_delta.groupby(level="terminal_tree").mean().reindex(TREES).to_numpy(float)
    if np.any(~np.isfinite(tree_delta)):
        raise RuntimeError(f"Missing tree delta for {model} {field}")
    rng = np.random.default_rng(seed)
    draws = tree_delta[rng.integers(0, len(tree_delta), size=(BOOTSTRAPS, len(tree_delta)))].mean(axis=1)
    return {
        "model_id": model, "field": field, "point": float(tree_delta.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)), "tree_count": len(tree_delta),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    predecessor = json.loads((REPORTS / "stage27_temporal_decision.json").read_text(encoding="utf-8"))
    if not predecessor["spatial_two_path_test_required"]:
        raise RuntimeError("Temporal DYN2P did not authorize spatial evaluation")

    contract = {
        "stage": "20260826_27 spatial DYN2P substage",
        "registered_before_DYN2P_target_results_read": True,
        "hypothesis": "The nested two-response conserving correction transfers to an entirely unseen terminal tree without any target-gauge discharge history.",
        "heldout_terminal_trees": TREES,
        "formal_candidates": MODELS,
        "training": "2010-2015 non-target-tree complete gauges only",
        "early_stopping": "2016 non-target-tree gauges only",
        "evaluation": "all available 2010-2018 target-tree observations; never used for fitting, stopping, scaling or initialization",
        "fold_parent": "independently fit from the fixed physical prior under the same composite objective; no full-development parameter lock",
        "candidate_seed": 260826,
        "gates": {
            "overall_log_noninferior": "paired tree bootstrap CI95 upper < 0.01",
            "high_flow_noninferior": "paired tree bootstrap CI95 upper < 0.01",
            "low_flow_noninferior": "paired tree bootstrap CI95 upper < 0.01",
            "median_NSE_drop_max": 0.02,
            "median_abs_PBIAS_worsening_max_points": 2.0,
            "tree_direction": "candidate improves high-flow RMSE in at least 5 of 8 trees",
        },
        "selection": "DYN2P is formally selected only if it passes every zero-target-history spatial gate.",
        "TN_read": False,
        "2019_2022_read": False,
    }
    write_json(RUN / "experiment_contract.json", contract)

    pending = [tree for tree in TREES if not (OUT / "folds" / f"tree_{tree}" / "metadata.json").is_file()]
    if pending:
        with ThreadPoolExecutor(max_workers=min(8, len(pending))) as executor:
            futures = {executor.submit(run_fold, tree): tree for tree in pending}
            for future in as_completed(futures):
                tree, code, tail = future.result()
                print(f"TREE {tree} exit={code}\n{tail}", flush=True)
                if code != 0:
                    raise RuntimeError(f"Tree {tree} failed; see worker log")

    fold_dirs = [OUT / "folds" / f"tree_{tree}" for tree in TREES]
    runs = pd.concat([pd.read_parquet(path / "run_summary.parquet") for path in fold_dirs], ignore_index=True)
    metrics = pd.concat([pd.read_parquet(path / "station_metrics.parquet") for path in fold_dirs], ignore_index=True)
    predictions = pd.concat([pd.read_parquet(path / "predictions.parquet") for path in fold_dirs], ignore_index=True)
    runs.to_parquet(OUT / "spatial_run_summary.parquet", index=False)
    metrics.to_parquet(OUT / "spatial_station_metrics.parquet", index=False)
    predictions.to_parquet(OUT / "spatial_predictions.parquet", index=False)

    comparisons = []
    decisions: dict[str, dict[str, object]] = {}
    fields = ["log_RMSE", "high_log_RMSE", "low_log_RMSE"]
    parent = metrics.loc[metrics.model_id.eq("FOLD_PARENT")]
    for model_index, model in enumerate(MODELS):
        model_results = {}
        for field_index, field in enumerate(fields):
            value = bootstrap(metrics, model, field, SEED + model_index * 10 + field_index)
            comparisons.append(value)
            model_results[field] = value
        candidate = metrics.loc[metrics.model_id.eq(model)]
        paired = candidate.merge(parent, on=["station_norm", "terminal_tree"], suffixes=("_candidate", "_parent"))
        tree_high = paired.assign(delta=lambda frame: frame.high_log_RMSE_candidate - frame.high_log_RMSE_parent).groupby("terminal_tree").delta.mean()
        nse_drop = float(np.nanmedian(parent.NSE) - np.nanmedian(candidate.NSE))
        pbias_worsening = float(np.nanmedian(np.abs(candidate.PBIAS_pct)) - np.nanmedian(np.abs(parent.PBIAS_pct)))
        decisions[model] = {
            "overall_noninferior": model_results["log_RMSE"]["ci95_upper"] < 0.01,
            "high_flow_noninferior": model_results["high_log_RMSE"]["ci95_upper"] < 0.01,
            "low_flow_noninferior": model_results["low_log_RMSE"]["ci95_upper"] < 0.01,
            "median_NSE_drop": nse_drop,
            "median_abs_PBIAS_worsening_points": pbias_worsening,
            "high_flow_improved_tree_count": int((tree_high < 0).sum()),
        }
        decisions[model]["spatial_gate_pass"] = bool(
            decisions[model]["overall_noninferior"]
            and decisions[model]["high_flow_noninferior"]
            and decisions[model]["low_flow_noninferior"]
            and nse_drop <= 0.02
            and pbias_worsening <= 2.0
            and decisions[model]["high_flow_improved_tree_count"] >= 5
        )
    comparisons_frame = pd.DataFrame(comparisons)
    comparisons_frame.to_parquet(OUT / "spatial_tree_block_comparisons.parquet", index=False)

    selected = "DYN2P" if decisions["DYN2P"]["spatial_gate_pass"] else None
    status = "PASS_DYN2P_SPATIAL_AND_PATH_SELECTION" if selected else "DYN2P_FAILS_SPATIAL_RETAIN_DYN3P"
    decision = {
        "stage": "20260826_27", "status": status, "candidate_decisions": decisions,
        "formal_operational_path_selection": "TWO_PATH" if selected else "THREE_PATH",
        "target_tree_history_used_in_training": False,
        "2019_2022_read": False, "TN_read": False,
        "authorized_successor": "20260826_28",
    }
    write_json(REPORTS / "stage27_decision.json", decision)
    validation = {
        "stage": "20260826_27",
        "checks": {
            "eight_tree_folds_complete": len(runs.heldout_tree.unique()) == 8,
            "zero_target_history": bool(runs.target_observations_used_in_training.eq(0).all()),
            "all_spinups_converged": bool(runs.spinup_converged.all()),
            "all_land_mass_errors_bounded": bool(runs.land_mass_error_mm.le(1.0e-8).all()),
            "all_models_present": set(runs.model_id.unique()) == {"FOLD_PARENT", *MODELS},
        },
    }
    validation["all_checks_pass"] = all(validation["checks"].values())
    write_json(REPORTS / "validation.json", validation)
    (RUN / "README.md").write_text("# 20260826_27 two-path identifiability and spatial evaluation\n", encoding="utf-8")
    (REPORTS / "technical_report.md").write_text(
        "# 20260826_27 两路径辨识与空间评价\n\n"
        f"状态：`{status}`。八棵目标河树的全部流量历史均未进入各折训练、停止、初始化或缩放。\n\n"
        f"正式操作性路径选择：`{'TWO_PATH' if selected else 'THREE_PATH'}`。\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
