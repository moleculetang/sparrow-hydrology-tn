"""Lock the component-supervised candidate without reading 2017+ observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_5"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE8 = ROOT / "5_Test" / "20260828_2"
GAMMAS = [0.1, 0.3, 1.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    frames = []
    for gamma in GAMMAS:
        suffix = str(gamma).replace(".", "p")
        report = json.loads((REPORTS / f"component_worker_gamma_{suffix}.json").read_text(encoding="utf-8"))
        if report["status"] != "COMPONENT_WORKER_COMPLETE" or report["seed_count"] != 3 or not report["all_finite"]:
            raise RuntimeError(f"Incomplete component worker gamma={gamma}")
        frames.append(pd.read_parquet(OUT / f"component_run_summary_gamma_{suffix}.parquet"))
    runs = pd.concat(frames, ignore_index=True)
    if len(runs) != 9 or runs[["seed", "gamma"]].duplicated().any():
        raise RuntimeError("Incomplete 3-by-3 component grid")

    parents = pd.read_parquet(OUT / "same_scale_parent_stop_flow.parquet")
    runs = runs.merge(parents, on="seed", validate="many_to_one")
    runs["flow_ratio_to_parent"] = runs.best_stop_flow / runs.parent_stop_flow
    runs["flow_eligible"] = runs.flow_ratio_to_parent.le(1.01)
    eligible = runs.loc[runs.flow_eligible].copy()
    if eligible.seed.nunique() != 3:
        missing = sorted(set(runs.seed) - set(eligible.seed))
        per_seed_best = runs.sort_values(["seed", "flow_ratio_to_parent", "best_stop_component", "gamma"]).groupby("seed", as_index=False).first()
        decision = {
            "stage": "20260828_5",
            "status": "NO_FLOW_ELIGIBLE_COMPONENT_CANDIDATE",
            "missing_eligible_seeds": [int(value) for value in missing],
            "per_seed_best_available": [
                {"seed": int(row.seed), "gamma": float(row.gamma), "lambda_S": float(row.lambda_S),
                 "flow_ratio_to_parent": float(row.flow_ratio_to_parent),
                 "normalized_component_error": float(row.best_stop_component)}
                for row in per_seed_best.itertuples()
            ],
            "checks": {
                "all_9_runs_complete": len(runs) == 9,
                "all_seeds_have_flow_eligible_candidate": False,
                "2017_2018_observations_not_read": True,
                "2019_2022_observations_not_read": True,
                "four_spatial_station_observations_not_read": True,
                "TN_not_read": True,
            },
            "authorized_successor": "20260828_6_CONTROLLED_SMALL_GAMMA_REPAIR",
        }
        runs.to_parquet(OUT / "component_candidate_grid.parquet", index=False)
        per_seed_best.to_parquet(OUT / "component_candidate_selection_by_seed.parquet", index=False)
        write_json(REPORTS / "component_candidate_lock.json", decision)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return
    per_seed = eligible.sort_values(["seed", "best_stop_component", "gamma", "lambda_S"]).groupby("seed", as_index=False).first()
    gamma_counts = per_seed.gamma.value_counts().sort_index()
    stable_gamma = float(gamma_counts.idxmax())
    stable_count = int(gamma_counts.max())
    stability_pass = stable_count >= 2
    stable_pool = eligible.loc[eligible.gamma.eq(stable_gamma)]
    selected = stable_pool.sort_values(["best_stop_component", "best_stop_flow", "lambda_S", "seed"]).iloc[0]
    lambda_boundary_count = int((per_seed.lambda_S.ge(0.99) | per_seed.lambda_S.le(0.01)).sum())
    decision = {
        "stage": "20260828_5",
        "status": "COMPONENT_CANDIDATE_LOCKED" if stability_pass and lambda_boundary_count < 2 else "COMPONENT_CANDIDATE_CONFOUNDED",
        "selected_seed": int(selected.seed),
        "selected_gamma": float(selected.gamma),
        "selected_lambda_S": float(selected.lambda_S),
        "selected_checkpoint": str(OUT / f"component_seed_{int(selected.seed)}_gamma_{str(float(selected.gamma)).replace('.', 'p')}.pt"),
        "selection_metrics": {
            "stop_flow": float(selected.best_stop_flow),
            "parent_stop_flow": float(selected.parent_stop_flow),
            "flow_ratio_to_parent": float(selected.flow_ratio_to_parent),
            "normalized_component_error": float(selected.best_stop_component),
            "best_epoch": int(selected.best_epoch),
        },
        "per_seed_selected": [
            {"seed": int(row.seed), "gamma": float(row.gamma), "lambda_S": float(row.lambda_S),
             "flow_ratio_to_parent": float(row.flow_ratio_to_parent),
             "normalized_component_error": float(row.best_stop_component)}
            for row in per_seed.itertuples()
        ],
        "gamma_counts": {str(float(key)): int(value) for key, value in gamma_counts.items()},
        "gamma_stability_pass": stability_pass,
        "lambda_boundary_seed_count": lambda_boundary_count,
        "checks": {
            "all_9_runs_complete": len(runs) == 9,
            "all_seeds_have_flow_eligible_candidate": eligible.seed.nunique() == 3,
            "at_least_2_of_3_seeds_select_same_gamma": stability_pass,
            "lambda_boundary_not_repeated": lambda_boundary_count < 2,
            "2017_2018_observations_not_read": True,
            "2019_2022_observations_not_read": True,
            "four_spatial_station_observations_not_read": True,
            "TN_not_read": True,
        },
        "authorized_successor": "20260828_5_DEVELOPMENT_EVALUATION" if stability_pass and lambda_boundary_count < 2 else "STOP_KEEP_20260827_6",
        "hashes": {
            "contract": sha256(RUN / "experiment_contract.json"),
            "preflight": sha256(REPORTS / "learnable_operator_preflight.json"),
            "selected_checkpoint": sha256(Path(str(OUT / f"component_seed_{int(selected.seed)}_gamma_{str(float(selected.gamma)).replace('.', 'p')}.pt"))),
        },
    }
    runs.to_parquet(OUT / "component_candidate_grid.parquet", index=False)
    per_seed.to_parquet(OUT / "component_candidate_selection_by_seed.parquet", index=False)
    write_json(REPORTS / "component_candidate_lock.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if decision["status"] != "COMPONENT_CANDIDATE_LOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
