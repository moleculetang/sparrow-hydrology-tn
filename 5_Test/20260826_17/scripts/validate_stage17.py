"""Validate complete-tree spatial diagnostic artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUN = Path(r"E:\SPARROW\5_Test\20260826_17")
TREES = {1, 20, 22, 26, 56, 166, 212, 217}


def main() -> None:
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    decision = json.loads((RUN / "reports" / "stage17_decision.json").read_text(encoding="utf-8"))
    metrics = pd.read_parquet(RUN / "outputs" / "nested_spatial_station_performance.parquet")
    gates = pd.read_parquet(RUN / "outputs" / "nested_spatial_seed_gates.parquet")
    summaries = pd.read_parquet(RUN / "outputs" / "nested_spatial_run_summary.parquet")
    metadata = [
        json.loads((RUN / "outputs" / "folds" / f"tree_{tree}" / "metadata.json").read_text(encoding="utf-8"))
        for tree in sorted(TREES)
    ]
    checks = {
        "eight_trees": set(metrics.terminal_tree.unique()) == TREES,
        "six_candidate_seed_gates": len(gates) == 6,
        "zero_target_history": all(not row["target_Q_used_in_training_early_stopping_or_selection"] for row in metadata),
        "all_folds_complete": all(row["completed"] for row in metadata),
        "spinups_converged": bool(summaries.loc[summaries.model_id.ne("GLOBAL_HBV_FOLD_PARENT"), "periodic_spinup_converged"].all()),
        "mass_closed": bool((summaries.loc[summaries.model_id.ne("GLOBAL_HBV_FOLD_PARENT"), "mass_error_mm"] <= 1.0e-10).all()),
        "promotion_forbidden": not contract["promotion_possible"] and decision["spatially_promoted"] == [],
        "stage18_next": decision["authorized_successor"] == "20260826_18",
    }
    result = {"stage": "20260826_17", "checks": checks, "all_checks_pass": all(checks.values())}
    (RUN / "reports" / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["all_checks_pass"]:
        raise RuntimeError(f"Stage 17 validation failed: {checks}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
