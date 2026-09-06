"""Finalize Stage 4 temporal OOF outputs and determine spatial-screen eligibility."""

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
RUN = ROOT / r"5_Test\20260902_4"
WORK = RUN / "work"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CANDIDATES = ["PARENT", "H22_TRANSFER_HEAD", "TN82_TRANSFER_HEAD"]
FOLDS = ["T1", "T2", "T3"]
LAYERS = ["raw_mass_process", "population_transferable", "gauged_conditional"]
MARGIN = 0.005
SEED = 260902
KEYS = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def macro_log_rmse(frame: pd.DataFrame, block: str) -> float:
    values = []
    for _, group in frame.groupby(block):
        e = np.log1p(group.pred_tn_mg_l.to_numpy(float)) - np.log1p(group.tn_mg_l.to_numpy(float))
        values.append(np.sqrt(np.mean(e * e)))
    return float(np.mean(values))


def paired_ci(candidate: pd.DataFrame, parent: pd.DataFrame, block: str, seed: int) -> dict[str, object]:
    joined = parent[KEYS + ["pred_tn_mg_l"]].merge(
        candidate[KEYS + ["pred_tn_mg_l"]], on=KEYS, suffixes=("_parent", "_candidate"), validate="one_to_one"
    )
    values = []
    for _, group in joined.groupby(block):
        y = np.log1p(group.tn_mg_l.to_numpy(float))
        p = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_parent.to_numpy(float)) - y) ** 2))
        c = np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate.to_numpy(float)) - y) ** 2))
        values.append(c - p)
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "block": block,
        "block_count": len(values),
        "delta_log_rmse": float(values.mean()),
        "ci95_lower": float(low),
        "ci95_upper": float(high),
        "noninferior_0p005": bool(high < MARGIN),
        "improved": bool(high < 0.0),
    }


def pooled_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame.tn_mg_l.to_numpy(float)
    p = frame.pred_tn_mg_l.to_numpy(float)
    e = p - y
    mean_y = float(np.mean(y))
    ss_res = float(np.sum(e * e))
    ss_tot = float(np.sum((y - mean_y) ** 2))
    correlation = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else float("nan")
    station_means = frame.groupby("station_key")[["tn_mg_l", "pred_tn_mg_l"]].mean()
    spatial_corr = float(station_means.corr().iloc[0, 1])
    temporal = []
    for _, group in frame.groupby("station_key"):
        if len(group) >= 3 and group.tn_mg_l.std(ddof=0) > 0 and group.pred_tn_mg_l.std(ddof=0) > 0:
            temporal.append(group[["tn_mg_l", "pred_tn_mg_l"]].corr().iloc[0, 1])
    return {
        "log_rmse_station_macro": macro_log_rmse(frame, "station_key"),
        "log_rmse_tree_macro": macro_log_rmse(frame, "terminal_tree_id"),
        "rmse_mg_l": float(np.sqrt(np.mean(e * e))),
        "mae_mg_l": float(np.mean(np.abs(e))),
        "pbias_percent": float(100.0 * np.sum(e) / np.sum(y)),
        "correlation": correlation,
        "r2_correlation_squared": correlation ** 2,
        "nse": float(1.0 - ss_res / ss_tot),
        "station_mean_spatial_correlation": spatial_corr,
        "median_within_station_temporal_correlation": float(np.median(temporal)) if temporal else float("nan"),
        "rows": len(frame),
        "stations": frame.station_key.nunique(),
        "trees": frame.terminal_tree_id.nunique(),
    }


def main() -> None:
    paths = []
    for candidate in CANDIDATES:
        for fold in FOLDS:
            prefix = f"{candidate.lower()}_{fold.lower()}"
            checkpoint = WORK / f"{prefix}_checkpoint.json"
            if not checkpoint.exists():
                raise RuntimeError(f"Missing checkpoint: {checkpoint}")
            paths.append((candidate, fold, prefix))
    predictions = pd.concat([pd.read_parquet(WORK / f"{prefix}_predictions.parquet") for _, _, prefix in paths], ignore_index=True)
    parameters = pd.concat([pd.read_parquet(WORK / f"{prefix}_parameters.parquet") for _, _, prefix in paths], ignore_index=True)
    gammas = pd.concat([pd.read_parquet(WORK / f"{prefix}_gamma.parquet") for _, _, prefix in paths], ignore_index=True)
    sites = pd.concat([pd.read_parquet(WORK / f"{prefix}_sites.parquet") for _, _, prefix in paths], ignore_index=True)
    offsets = pd.concat([pd.read_parquet(WORK / f"{prefix}_offsets.parquet") for _, _, prefix in paths], ignore_index=True)
    if predictions.duplicated(["candidate", "layer", *KEYS[:-1]]).any():
        raise RuntimeError("Prediction key duplication")

    metrics = []
    for (candidate, layer), group in predictions.loc[predictions.pred_tn_mg_l.notna()].groupby(["candidate", "layer"]):
        metrics.append({"candidate": candidate, "layer": layer, **pooled_metrics(group)})
    metrics_frame = pd.DataFrame(metrics)

    comparisons: dict[str, object] = {}
    parent = predictions.loc[predictions.candidate.eq("PARENT")]
    eligibility = {}
    for candidate in CANDIDATES[1:]:
        current = predictions.loc[predictions.candidate.eq(candidate)]
        for layer in ["population_transferable", "gauged_conditional"]:
            a = current.loc[current.layer.eq(layer) & current.pred_tn_mg_l.notna()]
            b = parent.loc[parent.layer.eq(layer) & parent.pred_tn_mg_l.notna()]
            for index, block in enumerate(["station_key", "terminal_tree_id"]):
                comparisons[f"{candidate}:{layer}:{block}"] = paired_ci(a, b, block, SEED + index)
        pop_station = comparisons[f"{candidate}:population_transferable:station_key"]
        pop_tree = comparisons[f"{candidate}:population_transferable:terminal_tree_id"]
        cond_station = comparisons[f"{candidate}:gauged_conditional:station_key"]
        cond_tree = comparisons[f"{candidate}:gauged_conditional:terminal_tree_id"]
        par = parameters.loc[parameters.candidate.eq(candidate)]
        offset_confounded = int((par.max_abs_transferable_offset > 1.0).sum()) >= 2
        eligibility[candidate] = bool(
            (par.projected_kkt_max <= 1.0e-5).all()
            and pop_station["noninferior_0p005"] and pop_tree["noninferior_0p005"]
            and (pop_station["improved"] or pop_tree["improved"])
            and cond_station["noninferior_0p005"] and cond_tree["noninferior_0p005"]
            and not offset_confounded
        )

    parent_zero = offsets.loc[offsets.candidate.eq("PARENT"), "transferable_offset_log_unit"].abs().max()
    checks = {
        "all_nine_checkpoints": len(paths) == 9,
        "parent_kkt_le_1e_5": bool((parameters.loc[parameters.candidate.eq("PARENT"), "projected_kkt_max"] <= 1.0e-5).all()),
        "at_least_one_spatial_candidate_kkt_le_1e_5": bool(
            any((parameters.loc[parameters.candidate.eq(candidate), "projected_kkt_max"] <= 1.0e-5).all() for candidate in CANDIDATES[1:])
        ),
        "parent_transferable_offset_zero_le_1e_10": float(parent_zero) <= 1.0e-10,
        "no_partial_files": not any(RUN.rglob("*.part")),
    }
    eligible = [candidate for candidate in CANDIDATES[1:] if eligibility[candidate]]
    status = "PASS_TEMPORAL_SCREEN_READY_FOR_SPATIAL" if eligible and all(checks.values()) else (
        "PASS_TEMPORAL_SCREEN_NO_ELIGIBLE_CANDIDATE" if all(checks.values()) else "FAIL_TEMPORAL_ENGINEERING"
    )
    decision = {
        "stage": "20260902_4",
        "status": status,
        "eligible_candidates": eligible,
        "eligibility": eligibility,
        "comparisons": comparisons,
        "checks": checks,
        "metrics": metrics,
        "offset_confounded": {
            candidate: int((parameters.loc[parameters.candidate.eq(candidate), "max_abs_transferable_offset"] > 1.0).sum()) >= 2
            for candidate in CANDIDATES[1:]
        },
    }
    for name, frame in [
        ("temporal_oof_predictions", predictions), ("fold_parameters", parameters),
        ("gamma_coefficients", gammas), ("station_residuals", sites),
        ("reach_transferable_offsets", offsets), ("performance_metrics", metrics_frame),
    ]:
        atomic_parquet(frame, OUT / f"{name}.parquet")
    atomic_json(decision, REPORTS / "temporal_decision.json")
    atomic_json({"stage": "20260902_4", "status": status, "eligible_candidates": eligible}, LOCKS / "temporal_screen_lock.json")
    program = json.loads((ROOT / r"5_Test\20260902_3\program_manifest.json").read_text(encoding="utf-8"))
    program["stage_status"]["20260902_4"] = status
    program["stage_status"]["20260902_5"] = "authorized_next" if eligible else "closed_no_temporal_candidate"
    atomic_json(program, RUN / "program_manifest.json")
    print(json.dumps({"status": status, "eligible_candidates": eligible, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
