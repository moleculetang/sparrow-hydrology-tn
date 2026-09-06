"""Checkpointed nested-spatial worker for one Stage47 model."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260824_47"
WORK = RUN / "work"
CONTRACT = RUN / "experiment_contract.json"
PARENT_LOCK = ROOT / "5_Test/20260824_46/locks/stage46_lock.json"
PARENT_PARAMETERS = ROOT / "5_Test/20260824_45/outputs/l0_v2_u3_fold_parameters.parquet"
PARENT_SITES = ROOT / "5_Test/20260824_45/outputs/l0_v2_u3_site_effects.parquet"
CANDIDATE_PARAMETERS = ROOT / "5_Test/20260824_46/outputs/stage46_fold_parameters.parquet"
CANDIDATE_SITES = ROOT / "5_Test/20260824_46/outputs/stage46_site_effects.parquet"
for path in [
    ROOT / "5_Test/20260824_46/scripts", ROOT / "5_Test/20260824_41/scripts",
    ROOT / "5_Test/20260824_44/scripts", ROOT / "5_Test/20260824_19/scripts",
]:
    sys.path.insert(0, str(path))

import l0_v2_core as l0  # noqa: E402
import run_stage41 as s41  # noqa: E402
import run_stage44 as s44  # noqa: E402
import run_stage19 as s19  # noqa: E402
from stage46_models import Stage46Model  # noqa: E402
from run_stage46 import Objective  # noqa: E402

s28 = l0.s28
MODELS = ["PARENT", "REG3_OBSERVATION", "DYN_HYDRO_DELIVERY", "MINERAL_LIFETIME", "AQ_SIZE"]
SEED = 260847


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class ParentAdapter(l0.TorchL0V2):
    candidate = "PARENT"

    def extra_names(self):
        return []

    def extra_prior(self, values):
        return torch.zeros(())


def make_model(name: str):
    return ParentAdapter(False) if name == "PARENT" else Stage46Model(name)


def temporal_tables(name: str):
    if name == "PARENT":
        parameters = pd.read_parquet(PARENT_PARAMETERS).set_index("fold_id")
        sites = pd.read_parquet(PARENT_SITES)
    else:
        parameters = pd.read_parquet(CANDIDATE_PARAMETERS).loc[lambda x: x.candidate.eq(name)].set_index("fold_id")
        sites = pd.read_parquet(CANDIDATE_SITES).loc[lambda x: x.candidate.eq(name)]
    return parameters, sites


def station_equal_mean(train: pd.DataFrame) -> float:
    station_means = train.assign(y=np.log1p(train.tn_mg_l)).groupby("station_key").y.mean()
    return float(station_means.mean())


def warm_fit(model, train: pd.DataFrame, fold_id: str, temporal_parameters: pd.DataFrame, temporal_sites: pd.DataFrame):
    parent_fold = fold_id.split("_")[0]
    row = temporal_parameters.loc[parent_fold]
    start = np.asarray([row[name] for name in model.names()], dtype=float)
    objective = Objective(model, train)
    site_map = temporal_sites.loc[temporal_sites.fold_id.eq(parent_fold)].set_index("station_key").b_station_log_unit.to_dict()
    site_start = np.asarray([site_map.get(station, 0.0) for station in objective.stations], dtype=float)
    raw = torch.nn.Parameter(model.to_raw(start))
    raw_site = torch.nn.Parameter(torch.tensor(site_start))
    optimizer = torch.optim.LBFGS(
        [raw, raw_site], lr=1.0, max_iter=35, tolerance_grad=1.0e-9,
        tolerance_change=1.0e-11, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(); value = objective.loss(raw, raw_site); value.backward(); return value

    optimizer.step(closure)
    raw.grad = None; raw_site.grad = None
    final = objective.loss(raw, raw_site); final.backward()
    with torch.no_grad():
        physical = model.to_physical(raw).detach().numpy()
        effects = objective.site_effects(raw_site).detach().numpy()
    result = {
        "objective": float(final.detach()), "physical": physical, "effects": effects,
        "stations": objective.stations.copy(),
        "gradient_max_abs": max(float(raw.grad.abs().max()), float(raw_site.grad.abs().max())),
        "success": bool(np.isfinite(float(final.detach())) and np.isfinite(physical).all() and np.isfinite(effects).all()),
        "starts": 1,
    }
    del raw, raw_site, optimizer; gc.collect(); return result


def generic_starts(model):
    nested = model.initial(0); second = model.initial(1)
    third = 0.5 * (nested + second)
    if getattr(model, "candidate", "PARENT") != "PARENT" and len(model.extra_names()):
        if model.candidate == "MINERAL_LIFETIME":
            nested[-1] = math.log(365.25); second[-1] = math.log(730.5); third[-1] = math.log(258.27)
        elif len(model.extra_names()) == 1:
            nested[-1] = 0.0; second[-1] = 0.20; third[-1] = -0.20
        else:
            nested[-3:] = 0.0; second[-3:] = [0.15, -0.15, 0.10]; third[-3:] = [-0.15, 0.15, -0.10]
    return [nested, second, third]


def natural_fit(model, train: pd.DataFrame):
    objective = Objective(model, train); trials = []
    for variant, start in enumerate(generic_starts(model)):
        torch.manual_seed(SEED + variant)
        raw = torch.nn.Parameter(model.to_raw(start)); raw_site = torch.nn.Parameter(torch.tensor(s44.site_start(len(objective.stations), variant)))
        adam = torch.optim.AdamW([raw, raw_site], lr=0.03, weight_decay=0.0)
        for _ in range(60):
            adam.zero_grad(); value = objective.loss(raw, raw_site); value.backward(); torch.nn.utils.clip_grad_norm_([raw, raw_site], 10.0); adam.step()
        optimizer = torch.optim.LBFGS([raw, raw_site], lr=1.0, max_iter=80, tolerance_grad=1.0e-11, tolerance_change=1.0e-13, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad(); value = objective.loss(raw, raw_site); value.backward(); return value
        optimizer.step(closure)
        raw.grad = None; raw_site.grad = None; final = objective.loss(raw, raw_site); final.backward()
        with torch.no_grad(): physical = model.to_physical(raw).detach().numpy(); effects = objective.site_effects(raw_site).detach().numpy()
        trials.append({"objective": float(final.detach()), "physical": physical, "effects": effects, "stations": objective.stations.copy(), "gradient_max_abs": max(float(raw.grad.abs().max()), float(raw_site.grad.abs().max())), "success": bool(np.isfinite(float(final.detach())) and np.isfinite(physical).all()), "starts": 3})
        del raw, raw_site, adam, optimizer; gc.collect()
    return min((item for item in trials if item["success"]), key=lambda item: item["objective"])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True, choices=MODELS); args = parser.parse_args()
    name = args.model; s28.require_runtime(); WORK.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8")); lock = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_BEFORE_RESULTS" or lock.get("status") != "PASS_TEMPORAL_SCREEN_READY_FOR_STAGE47": raise RuntimeError("Stage47 not authorized")
    observations = s19.build_observations(); folds = s19.build_folds(observations, "full").loc[lambda x: ~x.holdout_type.eq("TEMPORAL")].reset_index(drop=True)
    model = make_model(name); temporal_parameters, temporal_sites = temporal_tables(name)
    pred_path = WORK / f"{name.lower()}_nested_predictions.parquet"; par_path = WORK / f"{name.lower()}_nested_parameters.parquet"
    predictions = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame(); parameters = pd.read_parquet(par_path) if par_path.exists() else pd.DataFrame()
    completed = set(parameters.fold_id.astype(str)) if not parameters.empty else set()
    for index, fold in folds.iterrows():
        fold_id = str(fold.fold_id)
        if fold_id in completed: continue
        model._obs_index_cache.clear(); model._q_feature_cache.clear(); train, test = s19.fold_frames(observations, fold)
        fit = natural_fit(model, train) if str(fold.holdout_type) == "FIRST_OBSERVED_2021" else warm_fit(model, train, fold_id, temporal_parameters, temporal_sites)
        physical = np.asarray(fit.pop("physical"), dtype=float); effects = dict(zip(fit.pop("stations"), map(float, fit.pop("effects"))))
        with torch.no_grad(): _, population = model.evaluate(test, torch.tensor(physical), int(fold.train_start_year), int(fold.train_end_year))
        known = test.station_key.astype(str).isin(effects).to_numpy(); conditional = population.detach().numpy().copy(); conditional[known] += np.asarray([effects.get(str(station), 0.0) for station in test.station_key])[known]; conditional[~known] = np.nan
        baseline = station_equal_mean(train); rows = []
        for layer, log_prediction in [("population_transferable", population.detach().numpy()), ("gauged_conditional", conditional)]:
            frame = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy(); frame["pred_tn_mg_l"] = np.where(np.isfinite(log_prediction), np.maximum(np.expm1(log_prediction), 0.0), np.nan)
            frame["baseline_pred_tn_mg_l"] = max(math.expm1(baseline), 0.0); frame["conditional_available"] = known if layer == "gauged_conditional" else True
            frame["fold_id"] = fold_id; frame["holdout_type"] = str(fold.holdout_type); frame["holdout_id"] = str(fold.holdout_id); frame["layer"] = layer; frame["candidate"] = name; rows.append(frame)
        row = {"candidate": name, "fold_id": fold_id, "holdout_type": str(fold.holdout_type), "holdout_id": str(fold.holdout_id), "train_rows": len(train), "test_rows": len(test), **fit}
        row.update(dict(zip(model.names(), map(float, physical))))
        predictions = pd.concat([predictions, *rows], ignore_index=True); parameters = pd.concat([parameters, pd.DataFrame([row])], ignore_index=True)
        atomic_parquet(predictions, pred_path); atomic_parquet(parameters, par_path)
        if (index + 1) % 10 == 0 or index == len(folds) - 1:
            current, peak = s28.memory_gib(); print(json.dumps({"model": name, "completed": fold_id, "index": int(index + 1), "folds": len(folds), "rss_gib": current, "peak_gib": peak}), flush=True)
        gc.collect()
    print(json.dumps({"model": name, "status": "WORKER_COMPLETE", "folds": int(parameters.fold_id.nunique()), "prediction_rows": len(predictions)}), flush=True)


if __name__ == "__main__": main()
