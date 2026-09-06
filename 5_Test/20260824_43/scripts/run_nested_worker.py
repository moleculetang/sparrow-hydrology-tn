"""Checkpointed nested spatial worker for Dryad ActiveLegacy-v2 vs KFAST."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
STAGE42_SCRIPTS = ROOT / "5_Test/20260824_42/scripts"
sys.path.insert(0, str(STAGE42_SCRIPTS))
import run_stage42 as s42  # noqa: E402
from active_legacy_v2_core import TorchActiveLegacyV2  # noqa: E402


RUN = ROOT / "5_Test/20260824_43"
SHARDS = RUN / "outputs/nested_shards"
STAGE42_AUDIT = ROOT / "5_Test/20260824_42/reports/stage42_validation.json"


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False)
    os.replace(part, path)


def starts(model: TorchActiveLegacyV2) -> list[np.ndarray]:
    """TN-independent cold multistarts required by the spatial contract."""
    return [model.initial(0), model.initial(1)]


def baseline_log_prediction(train: pd.DataFrame) -> float:
    station_means = train.assign(log_obs=np.log1p(train.tn_mg_l)).groupby("station_key").log_obs.mean()
    return float(station_means.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["kfast", "active"], required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--max-new-folds", type=int, default=None)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("invalid shard index")
    s42.s41.s28.require_runtime()
    audit = json.loads(STAGE42_AUDIT.read_text(encoding="utf-8"))
    if audit.get("scientific_decision") != "ACTIVELEGACY_V2_TEMPORAL_SUPPORTED_PENDING_NESTED_SPATIAL":
        raise RuntimeError("Stage42 Dryad temporal gate did not authorize nested validation")
    candidate_id = "ActiveLegacy_v2__KFAST_CONTROL" if args.candidate == "kfast" else "ActiveLegacy_v2__IMM0"
    instant = args.candidate == "kfast"
    observations = s42.s41.s28.s19.build_observations()
    folds = s42.s41.s28.s19.build_folds(observations, "full")
    folds = folds.loc[~folds.holdout_type.eq("TEMPORAL")].reset_index(drop=True)
    folds = folds.iloc[args.shard_index :: args.shards].reset_index(drop=True)
    prefix = f"{args.candidate}_s{args.shard_index:02d}of{args.shards:02d}"
    prediction_path = SHARDS / f"{prefix}_predictions.parquet"
    parameter_path = SHARDS / f"{prefix}_parameters.parquet"
    predictions = pd.read_parquet(prediction_path).to_dict(orient="records") if prediction_path.exists() else []
    parameters = pd.read_parquet(parameter_path).to_dict(orient="records") if parameter_path.exists() else []
    completed = {str(row["fold_id"]) for row in parameters}
    if predictions:
        prediction_folds = {str(row["fold_id"]) for row in predictions}
        completed &= prediction_folds
        parameters = [row for row in parameters if str(row["fold_id"]) in completed]
        predictions = [row for row in predictions if str(row["fold_id"]) in completed]
    model = TorchActiveLegacyV2(instant_control=instant)
    started = time.perf_counter()
    newly_completed = 0
    for position, fold in folds.iterrows():
        fold_id = str(fold.fold_id)
        if fold_id in completed:
            continue
        model._obs_index_cache.clear()
        model._q_feature_cache.clear()
        train, test = s42.s41.s28.s19.fold_frames(observations, fold)
        heldout_stations = set(test.station_key.astype(str))
        training_stations = set(train.station_key.astype(str))
        overlap = heldout_stations & training_stations
        if overlap:
            raise RuntimeError(f"held-out station leaked into training for {fold_id}: {sorted(overlap)[:3]}")
        fit = s42.fit_model(model, train, starts(model))
        layers = s42.s41.prediction_layers(
            model, test, fit["physical"], int(fold.train_start_year), int(fold.train_end_year), {}
        )
        baseline_log = baseline_log_prediction(train)
        prediction = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
        prediction["pred_tn_mg_l"] = np.maximum(np.expm1(layers["population_transferable"]), 0.0)
        prediction["baseline_tn_mg_l"] = max(math.expm1(baseline_log), 0.0)
        prediction["candidate"] = candidate_id
        prediction["fold_id"] = fold_id
        prediction["holdout_type"] = str(fold.holdout_type)
        prediction["holdout_id"] = str(fold.holdout_id)
        prediction["evaluation_year"] = int(fold.evaluation_year)
        predictions.extend(prediction.to_dict(orient="records"))
        parameter = {
            "candidate": candidate_id, "fold_id": fold_id,
            "holdout_type": str(fold.holdout_type), "holdout_id": str(fold.holdout_id),
            "train_start_year": int(fold.train_start_year), "train_end_year": int(fold.train_end_year),
            "evaluation_year": int(fold.evaluation_year), "train_rows": len(train), "test_rows": len(test),
            "train_stations": train.station_key.nunique(), "test_stations": test.station_key.nunique(),
            "heldout_training_station_overlap": len(overlap), "baseline_log_prediction": baseline_log,
            "objective": fit["objective"], "gradient_max_abs": fit["gradient_max_abs"],
            "selected_start": fit["start_variant"],
        }
        parameter.update(dict(zip(model.names(), map(float, fit["physical"]))))
        parameter["k_active_year_minus_1"] = float("inf") if instant else math.exp(parameter["log_k_active"])
        current, peak = s42.s41.s28.memory_gib()
        parameter["worker_rss_gib"] = current
        parameter["worker_peak_gib"] = peak
        parameters.append(parameter)
        atomic_parquet(pd.DataFrame(predictions), prediction_path)
        atomic_parquet(pd.DataFrame(parameters), parameter_path)
        completed.add(fold_id)
        newly_completed += 1
        print(json.dumps({
            "candidate": candidate_id, "shard": args.shard_index, "completed": len(completed),
            "assigned": len(folds), "fold_id": fold_id, "holdout_type": str(fold.holdout_type),
            "objective": fit["objective"], "rss_gib": current, "peak_gib": peak,
            "elapsed_seconds": time.perf_counter() - started,
        }), flush=True)
        gc.collect()
        if args.max_new_folds is not None and newly_completed >= args.max_new_folds:
            break
    print(json.dumps({
        "status": "PASS_NESTED_SHARD_COMPLETE", "candidate": candidate_id,
        "shard_index": args.shard_index, "shards": args.shards,
        "folds": len(folds), "prediction_rows": len(predictions),
    }), flush=True)


if __name__ == "__main__":
    main()
