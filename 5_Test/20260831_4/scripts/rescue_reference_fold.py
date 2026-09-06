"""Reference-equivalent 35-LBFGS rescue for the sole unconverged T3 fold."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260831_4"
WORK = RUN / "work"
CORE = ROOT / "5_Test/20260831_2/scripts"
S3 = ROOT / "5_Test/20260831_3/scripts"
S4 = RUN / "scripts"
for path in [CORE, S3, S4]: sys.path.insert(0, str(path))
from reservoir_tn_core import ReservoirTNModel  # noqa: E402
from run_stage3_worker import Objective, optimize_at_vf  # noqa: E402
from run_stage4_worker import fold_frames  # noqa: E402
import stage3_common as common  # noqa: E402

FOLD_ID = "T3_LORO_R213"
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
contract = json.loads((RUN / "refinement_contract.json").read_text(encoding="utf-8"))
if "reference_rescue_registered_before_spatial_metrics" not in contract:
    raise RuntimeError("Reference rescue is not registered")
nested = pd.read_parquet(ROOT / "5_Test/20260824_47/outputs/stage47_nested_spatial_parameters.parquet")
row = nested.loc[nested.candidate.eq("MINERAL_LIFETIME") & nested.fold_id.eq(FOLD_ID)].iloc[0]
obs = common.observations(); train, test = fold_frames(obs, row)
temporal_parameters = pd.read_parquet(ROOT / "5_Test/20260831_3/outputs/objective_fold_parameters.parquet")
temporal = temporal_parameters.loc[
    temporal_parameters.objective_id.eq("O0_LOG_T4") & temporal_parameters.fold_id.eq("T3")
].iloc[0]
temporal_sites = pd.read_parquet(ROOT / "5_Test/20260831_3/outputs/objective_site_effects.parquet")
temporal_sites = temporal_sites.loc[
    temporal_sites.objective_id.eq("O0_LOG_T4") & temporal_sites.fold_id.eq("T3")
]
model = ReservoirTNModel(reservoir_enabled=True)
anchor = np.asarray([temporal[name] for name in model.names()], dtype=float)
stations = sorted(train.station_key.astype(str).unique())
site_map = temporal_sites.set_index("station_key").b_raw_log_unit.to_dict()
site_start = np.asarray([float(site_map.get(station, 0.0)) for station in stations])
initial_v_f = float(temporal.selected_v_f)
model.set_fixed_v_f(initial_v_f); model._obs_index_cache.clear(); model._q_feature_cache.clear()
started = time.perf_counter(); objective = Objective(model, train, "O0_LOG_T4", anchor, site_start)
fit = optimize_at_vf(objective, anchor, site_start, initial_v_f, 0, 0.0, 35)
candidates = sorted(set(float(np.clip(initial_v_f + offset, 0.0, 0.5)) for offset in [-0.05, 0.0, 0.05]))
profile = []
for v_f in candidates:
    model.set_fixed_v_f(v_f); physical = fit["physical"].copy(); physical[model.names().index("v_f")] = v_f
    with torch.no_grad(): value = float(objective.loss(model.to_raw(physical), torch.tensor(fit["site_raw"])))
    profile.append({"v_f": v_f, "objective": value})
selected_v_f = min(profile, key=lambda item: item["objective"])["v_f"]
if abs(selected_v_f - initial_v_f) > 1.0e-12:
    fit = optimize_at_vf(objective, fit["physical"], fit["site_raw"], selected_v_f, 0, 0.0, 35)
if not fit["finite"]:
    raise RuntimeError("Nonfinite rescue fit")
model.set_fixed_v_f(selected_v_f); model._obs_index_cache.clear(); model._q_feature_cache.clear()
with torch.no_grad(): _, population = model.evaluate(test, torch.tensor(fit["physical"]), int(train.year.min()), int(train.year.max()))
prediction = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
prediction["pred_tn_mg_l"] = np.maximum(np.expm1(population.numpy()), 0.0)
station_means = train.assign(y=np.log1p(train.tn_mg_l)).groupby("station_key").y.mean()
prediction["baseline_pred_tn_mg_l"] = max(float(np.expm1(float(station_means.mean()))), 0.0)
prediction["fold_id"] = FOLD_ID; prediction["holdout_type"] = row.holdout_type
prediction["holdout_id"] = str(row.holdout_id); prediction["layer"] = "population_transferable"
prediction["candidate"] = "R2_CONSERVATIVE_O0"
parameter = {
    "fold_id": FOLD_ID, "holdout_type": row.holdout_type, "holdout_id": str(row.holdout_id),
    "objective": fit["objective"], "gradient_max_abs": fit["gradient_max_abs"],
    "selected_v_f": selected_v_f, "v_f_profile_json": json.dumps(profile),
    "train_rows": len(train), "test_rows": len(test), "site_initialization": "temporal_o0_site_effect",
    "adam_steps": 0, "adam_lr": 0.0, "lbfgs_steps": 35, "final_v_f_lbfgs_steps": 35,
    "gradient_topup_rounds": 0, "gradient_target_met": bool(fit["gradient_max_abs"] <= 0.005),
    "runtime_seconds": time.perf_counter() - started,
}
parameter.update(dict(zip(model.names(), map(float, fit["physical"]))))
pred_path = WORK / f"reference_rescue_{FOLD_ID}_predictions.parquet"
par_path = WORK / f"reference_rescue_{FOLD_ID}_parameters.parquet"
common.atomic_parquet(prediction, pred_path); common.atomic_parquet(pd.DataFrame([parameter]), par_path)
result = {
    "status": "RESCUE_COMPLETE" if parameter["gradient_target_met"] else "RESCUE_OPTIMIZER_CONFOUNDED",
    "fold_id": FOLD_ID, "gradient_max_abs": fit["gradient_max_abs"],
    "predictions_sha256": common.sha256(pred_path), "parameters_sha256": common.sha256(par_path),
}
common.atomic_json(result, WORK / f"reference_rescue_{FOLD_ID}_checkpoint.json")
print(json.dumps(result, indent=2))
