"""Run one candidate-fold fit with atomic checkpoint output."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / r"5_Test\20260902_4"
OUT = RUN / "work"
HERE = RUN / "scripts"
STAGE41 = ROOT / r"5_Test\20260824_41\scripts"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(STAGE41))
from spatial_head_model import FoldDesign, fit, prediction_layers  # noqa: E402
import run_stage41 as s41  # noqa: E402

s28 = s41.s28
CANDIDATES = ["PARENT", "H22_TRANSFER_HEAD", "TN82_TRANSFER_HEAD"]


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--fold", choices=["T1", "T2", "T3"], required=True)
    args = parser.parse_args()
    s28.require_runtime()
    observations = s28.s19.build_observations()
    folds = s28.s19.build_folds(observations, "temporal")
    fold = folds.set_index("fold_id").loc[args.fold]
    train, test = s28.s19.fold_frames(observations, fold)
    prefix = f"{args.candidate.lower()}_{args.fold.lower()}"
    required_tables = [OUT / f"{prefix}_{suffix}.parquet" for suffix in ["predictions", "parameters", "gamma", "sites", "offsets"]]
    checkpoint = OUT / f"{prefix}_checkpoint.json"
    if checkpoint.exists():
        print(json.dumps({"status": "ALREADY_COMPLETE", "candidate": args.candidate, "fold": args.fold}), flush=True)
        return
    if all(path.exists() for path in required_tables):
        predictions = pd.read_parquet(required_tables[0])
        parameters = pd.read_parquet(required_tables[1])
        expected_layers = {"raw_mass_process", "population_transferable", "gauged_conditional"}
        if set(predictions.layer) != expected_layers or len(parameters) != 1:
            raise RuntimeError("Incomplete pre-checkpoint tables cannot be recovered")
        design = FoldDesign.build(args.candidate, train)
        atomic_json(design.audit(), OUT / f"{prefix}_preprocessing.json")
        rss, _ = s28.memory_gib()
        atomic_json({"status": "RECOVERED_COMPLETE", "candidate": args.candidate, "fold_id": args.fold, "rss_gib": rss}, checkpoint)
        print(json.dumps({"status": "RECOVERED_COMPLETE", "candidate": args.candidate, "fold": args.fold}), flush=True)
        return
    result = fit(args.candidate, args.fold, train)
    layers = prediction_layers(result, test, int(fold.train_start_year), int(fold.train_end_year))
    base = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
    predictions = []
    for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
        frame = base.copy()
        values = layers[layer]
        frame["pred_tn_mg_l"] = np.where(np.isfinite(values), np.maximum(np.expm1(values), 0.0), np.nan)
        frame["conditional_available"] = layers["known"] if layer == "gauged_conditional" else True
        frame["candidate"] = args.candidate; frame["fold_id"] = args.fold; frame["layer"] = layer
        predictions.append(frame)
    parameter = {
        "candidate": args.candidate,
        "fold_id": args.fold,
        "objective": result["objective"],
        "objective_spread": result["objective_spread"],
        "selected_start": result["variant"],
        "projected_kkt_max": result["combined_kkt"],
        "process_site_kkt_max": result["process_site_kkt"]["combined_max"],
        "gamma_gradient_max": result["gamma_gradient_max"],
        "all_starts_json": json.dumps(result["all_starts"]),
        "feature_count": len(result["gamma"]),
        "max_abs_transferable_offset": float(np.max(np.abs(result["reach_effect"]))),
        "train_rows": len(train),
        "train_stations": train.station_key.nunique(),
        **dict(zip(result["model"].names(), map(float, result["physical"]))),
    }
    gamma = pd.DataFrame({
        "candidate": args.candidate,
        "fold_id": args.fold,
        "feature": result["design"].fields,
        "group": [next(group for group, indices in result["design"].groups.items() if i in indices) for i in range(len(result["design"].fields))],
        "gamma": result["gamma"],
    })
    sites = pd.DataFrame({
        "candidate": args.candidate,
        "fold_id": args.fold,
        "station_key": result["stations"],
        "station_residual_log_unit": result["effects"],
    })
    offsets = pd.DataFrame({
        "candidate": args.candidate,
        "fold_id": args.fold,
        "reach_id": range(1, 231),
        "transferable_offset_log_unit": result["reach_effect"],
    })
    atomic_parquet(pd.concat(predictions, ignore_index=True), OUT / f"{prefix}_predictions.parquet")
    atomic_parquet(pd.DataFrame([parameter]), OUT / f"{prefix}_parameters.parquet")
    atomic_parquet(gamma, OUT / f"{prefix}_gamma.parquet")
    atomic_parquet(sites, OUT / f"{prefix}_sites.parquet")
    atomic_parquet(offsets, OUT / f"{prefix}_offsets.parquet")
    atomic_json(result["design_audit"], OUT / f"{prefix}_preprocessing.json")
    rss, _ = s28.memory_gib()
    if rss > 6.5:
        raise MemoryError(f"Worker RSS {rss:.3f} GiB exceeds 6.5 GiB")
    atomic_json({"status": "COMPLETE", "candidate": args.candidate, "fold_id": args.fold, "rss_gib": rss}, OUT / f"{prefix}_checkpoint.json")
    print(json.dumps({"status": "COMPLETE", "candidate": args.candidate, "fold": args.fold, "objective": result["objective"], "kkt": result["combined_kkt"], "rss_gib": rss}), flush=True)


if __name__ == "__main__":
    main()
