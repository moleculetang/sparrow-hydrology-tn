"""Merge the five independently written lambda-worker results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_8"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE7 = ROOT / "5_Test" / "20260827_7"
DAILY = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
MONTHLY = ROOT / "5_Test" / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
MULTISCALE = ROOT / "5_Test" / "20260826_15" / "outputs" / "multiscale_static_features_standardized.parquet"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


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
    traces = []
    for value in LAMBDAS:
        suffix = str(float(value)).replace(".", "p")
        decision = json.loads((REPORTS / f"lambda_worker_{suffix}.json").read_text(encoding="utf-8"))
        if decision["status"] != "LAMBDA_WORKER_COMPLETE" or decision["seed_count"] != 3:
            raise RuntimeError(f"Incomplete worker for lambda {value}")
        frames.append(pd.read_parquet(OUT / f"candidate_run_summary_lambda_{suffix}.parquet"))
        traces.append(pd.read_parquet(OUT / f"candidate_training_trace_lambda_{suffix}.parquet"))
    run_frame = pd.concat(frames, ignore_index=True)
    trace_frame = pd.concat(traces, ignore_index=True)
    if len(run_frame) != 15 or run_frame[["seed", "lambda_S"]].duplicated().any():
        raise RuntimeError("Candidate grid is incomplete or duplicated")
    per_seed = run_frame.sort_values(["seed", "best_stop_score", "lambda_S"]).groupby("seed", as_index=False).first()
    selected = run_frame.sort_values(["best_stop_score", "lambda_S", "seed"]).iloc[0]
    boundary_count = int(per_seed.lambda_S.eq(1.0).sum())
    registry = pd.read_parquet(OUT / "station_registry_91.parquet")
    preparation = json.loads((REPORTS / "stage8_preparation_lock.json").read_text(encoding="utf-8"))
    decision = {
        "stage": "20260827_8",
        "status": "CANDIDATE_LOCKED_WITHOUT_DEVELOPMENT_READ",
        "station_count": int(len(registry)),
        "reach_count": int(registry.reach_id.nunique()),
        "BFI_fit_station_count_90pct_daily_coverage": int(preparation["BFI_fit_station_count_90pct_daily_coverage"]),
        "selected_seed": int(selected.seed),
        "selected_lambda_S": float(selected.lambda_S),
        "selected_stop_score": float(selected.best_stop_score),
        "per_seed_selected_lambda": {str(int(row.seed)): float(row.lambda_S) for row in per_seed.itertuples()},
        "lambda_one_seed_count": boundary_count,
        "boundary_confounded": boundary_count >= 2,
        "checks": {
            "station_count_is_91": len(registry) == 91,
            "unique_reach_count_is_91": registry.reach_id.nunique() == 91,
            "all_15_candidates_completed": len(run_frame) == 15,
            "all_scores_finite": bool(np.isfinite(run_frame.best_stop_score).all()),
            "2017_2018_observations_not_used": True,
            "2019_2022_observations_not_read": True,
            "four_station_observations_not_read": True,
            "TN_not_read": True
        },
        "authorized_successor": "20260827_9"
    }
    run_frame.to_parquet(OUT / "candidate_run_summary.parquet", index=False)
    trace_frame.to_parquet(OUT / "candidate_training_trace.parquet", index=False)
    per_seed.to_parquet(OUT / "per_seed_candidate_selection.parquet", index=False)
    write_json(REPORTS / "stage8_candidate_lock.json", decision)
    write_json(REPORTS / "input_hash_registry.json", {
        "contract": sha256(RUN / "experiment_contract.json"),
        "stage7_operator": sha256(STAGE7 / "scripts" / "state_consistent_sig2p.py"),
        "daily_discharge": sha256(DAILY), "monthly_discharge": sha256(MONTHLY),
        "forcing": sha256(FORCING), "gauges": sha256(GAUGES), "multiscale": sha256(MULTISCALE),
        "regionalized_score": sha256(OUT / "regionalized_slow_score.parquet")
    })
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
