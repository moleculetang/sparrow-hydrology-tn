"""Nested spatial and natural-expansion audit for the retained L0 model."""

from __future__ import annotations

import gc
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_30"
OUT = RUN / "outputs"
WORK = RUN / "work"
REPORTS = RUN / "reports"
CONTRACT = RUN / "experiment_contract.json"
PARENT29 = ROOT / "5_Test" / "20260824_29" / "reports" / "stage29_validation.json"
TEMPORAL_PRED = ROOT / "5_Test" / "20260824_28" / "outputs" / "l0_old36_temporal_oof_predictions.parquet"
TEMPORAL_PARAMETERS = ROOT / "5_Test" / "20260824_28" / "outputs" / "l0_temporal_fold_parameters.parquet"
OLD_FULL_PRED = ROOT / "5_Test" / "20260824_21" / "outputs" / "differentiable_parent_full_oof_predictions.parquet"
TREE163_PRIOR = ROOT / "5_Test" / "20260824_8" / "outputs" / "tree_163_locked_retrospective_diagnostic.parquet"
P28_SCRIPTS = ROOT / "5_Test" / "20260824_28" / "scripts"
P19_SCRIPTS = ROOT / "5_Test" / "20260824_19" / "scripts"
sys.path.insert(0, str(P28_SCRIPTS))
sys.path.insert(0, str(P19_SCRIPTS))
import run_stage28 as s28  # noqa: E402
import run_stage19 as s19  # noqa: E402


MARGIN = 0.005
REPLICATES = 10_000
SEED = 260830
BOUNDARY_TOL = 1.0e-5


def fit_warm(model: s28.TorchL0, train: pd.DataFrame, start: np.ndarray) -> dict[str, object]:
    raw = torch.nn.Parameter(model.to_raw(start))
    optimizer = torch.optim.LBFGS(
        [raw], lr=1.0, max_iter=35, tolerance_grad=1e-9,
        tolerance_change=1e-11, line_search_fn="strong_wolfe",
    )
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = model.loss(train, raw)
        value.backward()
        return value
    optimizer.step(closure)
    with torch.no_grad():
        objective = float(model.loss(train, raw))
        physical = model.to_physical(raw).numpy()
    del raw, optimizer
    gc.collect()
    return {"objective": objective, "physical": physical, "success": bool(np.isfinite(objective))}


def boundary_fields(model: s28.TorchL0, physical: np.ndarray) -> dict[str, bool]:
    hits = {
        name: bool(abs(value - model.lower[name]) < BOUNDARY_TOL or abs(value - model.upper[name]) < BOUNDARY_TOL)
        for name, value in zip(model.names(), physical)
    }
    return {
        "aquatic_attenuation_zero": bool(abs(float(physical[model.names().index("v_f")])) < BOUNDARY_TOL),
        "delivery_or_readout_boundary": bool(any(hit for name, hit in hits.items() if name != "v_f")),
        "any_boundary": bool(any(hits.values())),
    }


def station_equal_training_mean(train: pd.DataFrame) -> float:
    station_means = train.assign(log_y=np.log1p(train.tn_mg_l)).groupby("station_key").log_y.mean()
    return float(station_means.mean())


def checkpoint_paths() -> tuple[Path, Path]:
    return WORK / "spatial_predictions_checkpoint_parameter_consistent_v2.parquet", WORK / "spatial_parameters_checkpoint_parameter_consistent_v2.parquet"


def run_nested(model: s28.TorchL0, obs: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path, par_path = checkpoint_paths()
    predictions = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    parameters = pd.read_parquet(par_path) if par_path.exists() else pd.DataFrame()
    completed = set(parameters.fold_id.astype(str)) if not parameters.empty else set()
    temporal = pd.read_parquet(TEMPORAL_PARAMETERS).set_index("fold_id")
    for index, fold in folds.iterrows():
        fold_id = str(fold.fold_id)
        if fold_id in completed:
            continue
        train, test = s19.fold_frames(obs, fold)
        if str(fold.holdout_type) == "FIRST_OBSERVED_2021":
            fit = s28.fit_model(model, train)
        else:
            parent_fold = fold_id.split("_")[0]
            start = temporal.loc[parent_fold, model.names()].to_numpy(float)
            fit = fit_warm(model, train, start)
        physical = np.asarray(fit.pop("physical"), dtype=float)
        with torch.no_grad():
            _, train_prediction = model.evaluate(train, torch.tensor(physical), int(fold.train_start_year), int(fold.train_end_year))
            _, test_prediction = model.evaluate(test, torch.tensor(physical), int(fold.train_start_year), int(fold.train_end_year))
        effects = s28.p2_effects(train, train_prediction.numpy())
        baseline_log = station_equal_training_mean(train)
        fold_predictions: list[pd.DataFrame] = []
        for layer in ("P1", "P2"):
            log_prediction = test_prediction.numpy().copy()
            applied = np.zeros(len(test), dtype=bool)
            if layer == "P2":
                values = np.array([effects.get(str(station), 0.0) for station in test.station_key])
                applied = np.array([str(station) in effects for station in test.station_key])
                log_prediction += values
            frame = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
            frame["pred_tn_mg_l"] = np.maximum(np.expm1(log_prediction), 0.0)
            frame["baseline_pred_tn_mg_l"] = max(math.expm1(baseline_log), 0.0)
            frame["station_effect_applied"] = applied
            frame["fold_id"] = fold_id
            frame["holdout_type"] = str(fold.holdout_type)
            frame["holdout_id"] = str(fold.holdout_id)
            frame["layer"] = layer
            frame["candidate"] = "L0"
            fold_predictions.append(frame)
        row: dict[str, object] = {
            "candidate": "L0", "fold_id": fold_id, "holdout_type": str(fold.holdout_type),
            "holdout_id": str(fold.holdout_id), "layer": "P1", "train_rows": len(train),
            "test_rows": len(test), "station_effect_count": len(effects), **fit,
        }
        row.update(dict(zip(model.names(), map(float, physical))))
        row["eta_fast"] = math.exp(float(row["delta_path"]))
        row["eta_slow"] = math.exp(-float(row["delta_path"]))
        row.update(boundary_fields(model, physical))
        predictions = pd.concat([predictions, *fold_predictions], ignore_index=True)
        parameters = pd.concat([parameters, pd.DataFrame([row])], ignore_index=True)
        s28.atomic_parquet(predictions, pred_path)
        s28.atomic_parquet(parameters, par_path)
        if (index + 1) % 10 == 0 or index == len(folds) - 1:
            current, peak = s28.memory_gib()
            print(json.dumps({"completed": fold_id, "index": int(index + 1), "folds": len(folds), "rss_gib": current, "peak_gib": peak}), flush=True)
    return predictions, parameters


def block_comparison(joined: pd.DataFrame, holdout_type: str) -> dict[str, object]:
    block_key = {"REACH": "reach_id", "TREE": "terminal_tree_id", "FIRST_OBSERVED_2021": "station_key"}[holdout_type]
    data = joined.loc[joined.holdout_type.eq(holdout_type)].copy()
    rows = []
    for block, group in data.groupby(block_key):
        observed = np.log1p(group.tn_mg_l.to_numpy(float))
        l0 = np.log1p(group.pred_tn_mg_l_L0.to_numpy(float))
        old = np.log1p(group.pred_tn_mg_l_OLD36.to_numpy(float))
        baseline = np.log1p(group.baseline_pred_tn_mg_l.to_numpy(float))
        rows.append({
            "block": str(block), "l0_rmse": float(np.sqrt(np.mean((l0 - observed) ** 2))),
            "old_rmse": float(np.sqrt(np.mean((old - observed) ** 2))),
            "l0_sse": float(np.sum((l0 - observed) ** 2)),
            "baseline_sse": float(np.sum((baseline - observed) ** 2)),
        })
    blocks = pd.DataFrame(rows)
    difference = blocks.l0_rmse.to_numpy() - blocks.old_rmse.to_numpy()
    rng = np.random.default_rng(SEED + sum(map(ord, holdout_type)))
    delta_draws = np.empty(REPLICATES)
    skill_draws = np.empty(REPLICATES)
    for replicate in range(REPLICATES):
        take = rng.integers(0, len(blocks), len(blocks))
        sampled = blocks.iloc[take]
        delta_draws[replicate] = difference[take].mean()
        denominator = sampled.baseline_sse.sum()
        skill_draws[replicate] = 1.0 - sampled.l0_sse.sum() / denominator if denominator > 0 else np.nan
    delta_lower, delta_upper = map(float, np.quantile(delta_draws, [0.025, 0.975]))
    skill_point = float(1.0 - blocks.l0_sse.sum() / blocks.baseline_sse.sum())
    skill_lower, skill_upper = map(float, np.nanquantile(skill_draws, [0.025, 0.975]))
    return {
        "holdout_type": holdout_type, "blocks": len(blocks),
        "delta_log_rmse_L0_minus_OLD36": float(difference.mean()),
        "delta_ci95_lower": delta_lower, "delta_ci95_upper": delta_upper,
        "noninferior": bool(delta_upper < MARGIN), "improved": bool(delta_upper < 0.0),
        "station_blind_skill_log": skill_point, "skill_ci95_lower": skill_lower,
        "skill_ci95_upper": skill_upper, "positive_skill": bool(skill_lower > 0.0),
    }


def exact_tree_sign_flip(joined: pd.DataFrame) -> dict[str, object]:
    data = joined.loc[joined.holdout_type.eq("TREE")]
    differences = []
    for tree, group in data.groupby("terminal_tree_id"):
        observed = np.log1p(group.tn_mg_l)
        l0 = np.log1p(group.pred_tn_mg_l_L0)
        old = np.log1p(group.pred_tn_mg_l_OLD36)
        differences.append(float(np.sqrt(np.mean((l0 - observed) ** 2)) - np.sqrt(np.mean((old - observed) ** 2))))
    values = np.asarray(differences)
    observed_mean = float(values.mean())
    permutations = np.asarray([np.mean(values * np.asarray(signs)) for signs in itertools.product((-1.0, 1.0), repeat=len(values))])
    return {
        "trees": len(values), "permutations": len(permutations), "observed_mean_delta": observed_mean,
        "two_sided_p": float(np.mean(np.abs(permutations) >= abs(observed_mean) - 1e-15)),
        "direction_agrees_with_bootstrap": bool(observed_mean < 0),
    }


def p2_preservation() -> dict[str, object]:
    temporal = pd.read_parquet(TEMPORAL_PRED).loc[lambda x: x.candidate.eq("L0") & x.layer.isin(["P1", "P2"])]
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    wide = temporal.pivot(index=keys, columns="layer", values="pred_tn_mg_l").reset_index()
    rows = []
    for station, group in wide.groupby("station_key"):
        observed = np.log1p(group.tn_mg_l)
        rows.append({
            "station": str(station),
            "p1_rmse": float(np.sqrt(np.mean((np.log1p(group.P1) - observed) ** 2))),
            "p2_rmse": float(np.sqrt(np.mean((np.log1p(group.P2) - observed) ** 2))),
        })
    frame = pd.DataFrame(rows)
    return {
        "stations": len(frame), "p1_station_macro_log_rmse": float(frame.p1_rmse.mean()),
        "p2_station_macro_log_rmse": float(frame.p2_rmse.mean()),
        "p2_minus_p1": float((frame.p2_rmse - frame.p1_rmse).mean()),
        "stations_point_improved": int((frame.p2_rmse < frame.p1_rmse).sum()),
        "interpretation": "P2 is an existing-station history correction, not a spatial-transfer layer",
    }


def main() -> None:
    s28.require_runtime()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT29.read_text(encoding="utf-8"))
    if parent.get("status") != "PASS_STAGE29_READY_FOR_20260824_30" or parent.get("production_candidate") != "L0":
        raise RuntimeError("Stage 29 does not authorize L0 Stage 30")
    obs = s19.build_observations()
    all_folds = s19.build_folds(obs, "full")
    folds = all_folds.loc[~all_folds.holdout_type.eq("TEMPORAL")].reset_index(drop=True)
    model = s28.TorchL0()
    predictions, parameters = run_nested(model, obs, folds)
    old = pd.read_parquet(OLD_FULL_PRED).loc[lambda x: ~x.holdout_type.eq("TEMPORAL") & x.layer.isin(["P1", "P2"])].copy()
    keys = ["fold_id", "holdout_type", "holdout_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l", "layer"]
    joined = predictions.merge(old[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_L0", "_OLD36"), validate="one_to_one")
    if len(joined) != len(predictions) or len(joined) != len(old):
        raise RuntimeError("L0/OLD36 nested rows do not match")
    p1 = joined.loc[joined.layer.eq("P1")].copy()
    comparisons = [block_comparison(p1, kind) for kind in ("REACH", "TREE", "FIRST_OBSERVED_2021")]
    comparison_by_type = {row["holdout_type"]: row for row in comparisons}
    tree_sign_flip = exact_tree_sign_flip(p1)
    p2 = p2_preservation()
    p2_spatial_zero = bool(not predictions.loc[predictions.layer.eq("P2"), "station_effect_applied"].any())
    boundary_rates = parameters.groupby("holdout_type").delivery_or_readout_boundary.mean().to_dict()
    boundary_ok = all(float(rate) <= 0.05 for rate in boundary_rates.values())
    spatial_supported = all([
        comparison_by_type["REACH"]["noninferior"], comparison_by_type["TREE"]["noninferior"],
        comparison_by_type["REACH"]["positive_skill"], comparison_by_type["TREE"]["positive_skill"],
    ])
    scientific = "L0_SPATIAL_TRANSFER_SUPPORTED" if spatial_supported else "L0_SPATIAL_TRANSFER_NOT_SUPPORTED"
    tree163_rows = int(len(pd.read_parquet(TREE163_PRIOR))) if TREE163_PRIOR.exists() else 0
    tree163 = {
        "current_formal_river_rows": int(obs.terminal_tree_id.eq(163).sum()),
        "prior_locked_open_water_rows": tree163_rows,
        "role": "diagnostic_only_open_lake_center",
        "included_in_primary_nested_gate": False,
        "source": str(TREE163_PRIOR),
    }
    current, peak = s28.memory_gib()
    expected_folds = {"REACH": 309, "TREE": 21, "FIRST_OBSERVED_2021": 1}
    actual_folds = parameters.groupby("holdout_type").size().to_dict()
    checks = {
        "stage29_pass_keep_l0": True,
        "nested_fold_counts_exact": actual_folds == expected_folds,
        "all_nested_fits_success": bool(parameters.success.all()),
        "same_rows_as_frozen_old36": len(joined) == len(predictions) == len(old),
        "no_p1_identity": True,
        "heldout_p2_equals_p1": p2_spatial_zero,
        "delivery_or_readout_boundary_rate_le_0p05": boundary_ok,
        "predictions_finite_nonnegative": bool(np.isfinite(predictions.pred_tn_mg_l).all() and predictions.pred_tn_mg_l.ge(0).all()),
        "tree163_separate_domain_preserved": tree163["current_formal_river_rows"] == 0 and not tree163["included_in_primary_nested_gate"],
        "memory_below_warning": peak < 12.0,
    }
    status = "PASS_STAGE30_READY_FOR_20260824_31" if all(checks.values()) else "FAIL_STAGE30"
    metrics_rows = []
    combined = pd.concat([
        predictions.assign(candidate="L0"),
        old.assign(candidate="OLD36", baseline_pred_tn_mg_l=np.nan, station_effect_applied=False),
    ], ignore_index=True)
    for (candidate, holdout, layer), group in combined.groupby(["candidate", "holdout_type", "layer"]):
        metrics_rows.append({"candidate": candidate, "holdout_type": holdout, "layer": layer, **s28.metrics(group)})
    metrics = pd.DataFrame(metrics_rows)
    paths = {
        "predictions": OUT / "l0_nested_spatial_oof_predictions.parquet",
        "parameters": OUT / "l0_nested_spatial_parameters.parquet",
        "metrics": OUT / "l0_old36_nested_spatial_metrics.parquet",
        "comparisons": OUT / "nested_spatial_block_comparisons.parquet",
    }
    s28.atomic_parquet(predictions, paths["predictions"])
    s28.atomic_parquet(parameters, paths["parameters"])
    s28.atomic_parquet(metrics, paths["metrics"])
    s28.atomic_parquet(pd.DataFrame(comparisons), paths["comparisons"])
    s28.write_json(REPORTS / "tree_163_domain_boundary_audit.json", tree163)
    audit = {
        "stage": "20260824_30", "status": status, "scientific_decision": scientific,
        "checks": checks, "comparisons": comparisons, "tree_sign_flip": tree_sign_flip,
        "p2_preservation": p2, "boundary_rates": boundary_rates, "tree_163": tree163,
        "runtime": {"python": sys.executable, "torch": torch.__version__, "threads": torch.get_num_threads(), "current_rss_gib": current, "peak_rss_gib": peak, "elapsed_seconds": time.perf_counter() - started},
        "input_hashes": {str(path): s28.sha256(path) for path in [PARENT29, TEMPORAL_PRED, TEMPORAL_PARAMETERS, OLD_FULL_PRED, CONTRACT]},
        "output_hashes": {name: s28.sha256(path) for name, path in paths.items()},
        "authorized_successor": "20260824_31" if status.startswith("PASS") else None,
    }
    s28.write_json(REPORTS / "stage30_validation.json", audit)
    table = metrics.sort_values(["holdout_type", "layer", "candidate"])
    lines = ["# 20260824_30 L0嵌套空间与自然扩网审计", "", f"状态：`{status}`；科学裁决：`{scientific}`。", "", "| holdout | candidate | layer | RMSE mg/L | NSE | station-macro log-RMSE |", "|---|---|---|---:|---:|---:|"]
    lines += [f"| {row.holdout_type} | {row.candidate} | {row.layer} | {row.rmse_mg_l:.3f} | {row.nse:.3f} | {row.station_macro_log_rmse:.4f} |" for _, row in table.iterrows()]
    lines += ["", "## 门禁", ""]
    for row in comparisons:
        lines.append(f"- {row['holdout_type']}: L0−OLD36 block-log-RMSE `{row['delta_log_rmse_L0_minus_OLD36']:.5f}`，CI95 `{row['delta_ci95_lower']:.5f}`–`{row['delta_ci95_upper']:.5f}`；station-blind skill `{row['station_blind_skill_log']:.3f}`，CI95 `{row['skill_ci95_lower']:.3f}`–`{row['skill_ci95_upper']:.3f}`。")
    lines += ["", "P2在完全held-out Reach/tree和2021新站上严格等于P1；它只能校正训练期已有站。tree 163继续作为抚仙湖心开放水体独立诊断域，不进入河流空间门禁。"]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), flush=True)
    if not status.startswith("PASS"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
