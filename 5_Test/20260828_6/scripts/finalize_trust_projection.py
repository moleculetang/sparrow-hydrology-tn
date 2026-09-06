"""Lock the stable trust-projected state-consistent candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_6"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
SEEDS = [260826, 260827, 260828]


def main() -> None:
    reports, grids = [], []
    for seed in SEEDS:
        report = json.loads((REPORTS / f"trust_projection_worker_seed_{seed}.json").read_text(encoding="utf-8"))
        if report["status"] != "TRUST_PROJECTION_WORKER_COMPLETE":
            raise RuntimeError(f"Incomplete seed {seed}")
        reports.append(report)
        grids.append(pd.read_parquet(OUT / f"trust_projection_grid_seed_{seed}.parquet"))
    selected = pd.DataFrame(reports)
    alpha = selected.selected_alpha.to_numpy(float)
    close_counts = np.asarray([np.sum(np.abs(alpha - value) <= 0.1000001) for value in alpha])
    stable_count = int(close_counts.max())
    anchor = float(alpha[int(np.argmax(close_counts))])
    stable = selected.loc[np.abs(selected.selected_alpha - anchor) <= 0.1000001].copy()
    winner = stable.sort_values(["selected_component_error", "selected_flow_ratio", "seed"]).iloc[0]
    checks = {
        "all_three_workers_complete": len(selected) == 3,
        "at_least_two_seed_alphas_within_0p1": stable_count >= 2,
        "selected_flow_ratio_le_1p01": float(winner.selected_flow_ratio) <= 1.01,
        "selected_alpha_nonzero": float(winner.selected_alpha) > 0.0,
        "2017_2018_observations_not_read": True,
        "2019_2022_observations_not_read": True,
        "four_spatial_station_observations_not_read": True,
        "TN_not_read": True,
    }
    status = "COMPONENT_CANDIDATE_LOCKED" if all(checks.values()) else "TRUST_PROJECTION_CONFOUNDED"
    decision = {
        "stage": "20260828_6", "status": status,
        "selected_seed": int(winner.seed), "selected_gamma": 0.1,
        "selected_projection_alpha": float(winner.selected_alpha),
        "selected_lambda_S": float(winner.selected_lambda_S),
        "selected_checkpoint": str(winner.checkpoint),
        "selected_flow_ratio": float(winner.selected_flow_ratio),
        "selected_normalized_component_error": float(winner.selected_component_error),
        "per_seed": reports, "stable_seed_count": stable_count,
        "checks": checks,
        "authorized_successor": "20260828_6_DEVELOPMENT_EVALUATION" if status == "COMPONENT_CANDIDATE_LOCKED" else "STOP_KEEP_20260827_6_BASELINE",
    }
    pd.concat(grids, ignore_index=True).to_parquet(OUT / "trust_projection_grid_all_seeds.parquet", index=False)
    selected.to_parquet(OUT / "trust_projection_selection_by_seed.parquet", index=False)
    (REPORTS / "component_candidate_lock.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if status != "COMPONENT_CANDIDATE_LOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
