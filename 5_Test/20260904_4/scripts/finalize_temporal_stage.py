"""Aggregate and independently audit all Stage 4 temporal OOF checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260904_4"
WORK = RUN / "work"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CAPACITIES = ["H7", "H14", "H22"]
FOLDS = ["T1", "T2", "T3"]
LAYERS = ["raw_mass_process", "population_transferable", "gauged_conditional"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def metric(frame: pd.DataFrame, capacity: str, layer: str, fold_id: str) -> dict[str, object]:
    block = frame.loc[frame.pred_tn_mg_l.notna()].copy()
    y = block.tn_mg_l.to_numpy(float); p = block.pred_tn_mg_l.to_numpy(float)
    error = p - y; denominator = np.sum((y - np.mean(y)) ** 2)
    station_log = [float(np.sqrt(np.mean((np.log1p(g.pred_tn_mg_l) - np.log1p(g.tn_mg_l)) ** 2))) for _, g in block.groupby("station_key")]
    tree_log = [float(np.sqrt(np.mean((np.log1p(g.pred_tn_mg_l) - np.log1p(g.tn_mg_l)) ** 2))) for _, g in block.groupby("terminal_tree_id")]
    station_means = block.groupby("station_key")[["tn_mg_l", "pred_tn_mg_l"]].mean()
    return {
        "capacity": capacity, "layer": layer, "fold_id": fold_id,
        "rows": len(block), "stations": block.station_key.nunique(), "reaches": block.reach_id.nunique(), "trees": block.terminal_tree_id.nunique(),
        "station_macro_log_rmse": float(np.mean(station_log)), "tree_macro_log_rmse": float(np.mean(tree_log)),
        "rmse_mg_l": float(np.sqrt(np.mean(error**2))), "mae_mg_l": float(np.mean(np.abs(error))),
        "nse": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan,
        "r2": float(np.corrcoef(y, p)[0, 1] ** 2) if np.std(y) > 0 and np.std(p) > 0 else math.nan,
        "pbias_percent": float(100.0 * np.sum(error) / np.sum(y)),
        "station_mean_spatial_r": float(np.corrcoef(station_means.tn_mg_l, station_means.pred_tn_mg_l)[0, 1]),
    }


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow": raise RuntimeError("sparrow environment required")
    predictions = []; parameters = []; gamma = []; sites = []; offsets = []
    checkpoints = []
    for capacity in CAPACITIES:
        for fold in FOLDS:
            stem = f"{capacity.lower()}_{fold.lower()}_final"
            checkpoint_path = WORK / f"{stem}_checkpoint.json"
            if not checkpoint_path.exists(): raise RuntimeError(f"Missing {checkpoint_path}")
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")); checkpoints.append(checkpoint)
            if checkpoint.get("status") != "PASS_TEMPORAL_FIT_CHECKPOINT": raise RuntimeError(checkpoint)
            predictions.append(pd.read_parquet(WORK / f"{stem}_predictions.parquet"))
            parameters.append(pd.read_parquet(WORK / f"{stem}_parameters.parquet"))
            gamma.append(pd.read_parquet(WORK / f"{stem}_gamma.parquet"))
            sites.append(pd.read_parquet(WORK / f"{stem}_sites.parquet"))
            offsets.append(pd.read_parquet(WORK / f"{stem}_offsets.parquet"))
    prediction = pd.concat(predictions, ignore_index=True)
    parameter = pd.concat(parameters, ignore_index=True)
    gamma_frame = pd.concat(gamma, ignore_index=True)
    site_frame = pd.concat(sites, ignore_index=True)
    offset_frame = pd.concat(offsets, ignore_index=True)
    metric_rows = []
    for capacity in CAPACITIES:
        for layer in LAYERS:
            block = prediction.loc[prediction.capacity.eq(capacity) & prediction.layer.eq(layer)]
            metric_rows.append(metric(block, capacity, layer, "ALL_OOF"))
            for fold, group in block.groupby("fold_id"):
                metric_rows.append(metric(group, capacity, layer, str(fold)))
    metrics = pd.DataFrame(metric_rows)
    month = (
        prediction.assign(signed_log_residual=lambda d: np.log1p(d.tn_mg_l) - np.log1p(d.pred_tn_mg_l))
        .groupby(["capacity", "layer", "month"], as_index=False)
        .agg(rows=("tn_mg_l", "size"), signed_log_residual=("signed_log_residual", "mean"))
    )
    output_paths = {
        "predictions": OUT / "temporal_oof_predictions.parquet", "parameters": OUT / "fold_parameters.parquet",
        "gamma": OUT / "gamma_coefficients.parquet", "sites": OUT / "station_residuals.parquet",
        "offsets": OUT / "reach_transferable_offsets.parquet", "metrics": OUT / "performance_metrics.parquet",
        "month_residuals": OUT / "month_residuals.parquet",
    }
    for key, frame in {
        "predictions": prediction, "parameters": parameter, "gamma": gamma_frame, "sites": site_frame,
        "offsets": offset_frame, "metrics": metrics, "month_residuals": month,
    }.items(): atomic_parquet(frame, output_paths[key])
    checks = {
        "nine_checkpoints": len(checkpoints) == 9, "all_kkt": bool((parameter.projected_kkt_max <= 1e-5).all()),
        "all_mass_closure": bool((parameter.mass_balance_relative <= 1e-10).all()),
        "prediction_grain_unique": not prediction.duplicated(["capacity", "fold_id", "layer", "station_key", "year", "month"]).any(),
        "heldout_years_exact": set(prediction.groupby("fold_id").year.unique().map(lambda x: tuple(x))) == {(2022,), (2023,), (2024,)},
        "all_population_finite": prediction.loc[prediction.layer.eq("population_transferable"), "pred_tn_mg_l"].notna().all(),
        "no_parts": not any(RUN.rglob("*.part")),
    }
    status = "PASS_TEMPORAL_OOF_READY_FOR_NESTED" if all(checks.values()) else "FAIL_TEMPORAL_OOF"
    report = {
        "stage": "20260904_4", "status": status, "checks": {key: bool(value) for key, value in checks.items()},
        "metrics": metrics.loc[metrics.fold_id.eq("ALL_OOF")].to_dict("records"),
        "output_hashes": {key: sha256(path) for key, path in output_paths.items()},
    }
    atomic_json(report, REPORTS / "temporal_decision.json")
    atomic_json({"stage": "20260904_4", "status": status, "report_sha256": sha256(REPORTS / "temporal_decision.json")}, LOCKS / "temporal_oof_lock.json")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if status.startswith("FAIL"): raise RuntimeError(status)


if __name__ == "__main__":
    main()
