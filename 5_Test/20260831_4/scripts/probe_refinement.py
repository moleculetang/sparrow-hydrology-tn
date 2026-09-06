"""Pre-registered one-fold optimizer refinement probe."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"E:\SPARROW")
CORE = ROOT / "5_Test/20260831_2/scripts"
S3 = ROOT / "5_Test/20260831_3/scripts"
S4 = ROOT / "5_Test/20260831_4/scripts"
for path in [CORE, S3, S4]: sys.path.insert(0, str(path))
from reservoir_tn_core import ReservoirTNModel  # noqa: E402
from run_stage3_worker import Objective, optimize_at_vf  # noqa: E402
from run_stage4_worker import fold_frames  # noqa: E402
import stage3_common as common  # noqa: E402

torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
old = pd.read_parquet(ROOT / "5_Test/20260824_47/outputs/stage47_nested_spatial_parameters.parquet")
row = old.loc[old.candidate.eq("MINERAL_LIFETIME") & old.fold_id.eq("T3_LORO_R002")].iloc[0]
obs = common.observations(); train, test = fold_frames(obs, row)
model = ReservoirTNModel(); anchor = np.asarray([row[name] for name in model.names()], dtype=float)
model.set_fixed_v_f(float(row.v_f)); model._obs_index_cache.clear(); model._q_feature_cache.clear()
with torch.no_grad():
    _, population = model.evaluate(train, torch.tensor(anchor), 2021, 2023)
residual = np.log1p(train.tn_mg_l.to_numpy(float)) - population.numpy()
stations = sorted(train.station_key.astype(str).unique())
site_start = np.asarray([float(np.mean(residual[train.station_key.astype(str).eq(station)])) for station in stations])
started = time.perf_counter(); objective = Objective(model, train, "O0_LOG_T4", anchor, site_start)
fit = optimize_at_vf(objective, anchor, site_start, float(row.v_f), 1, 0.015, 2)
print(json.dumps({
    "fold_id": row.fold_id, "gradient_max_abs": fit["gradient_max_abs"],
    "objective": fit["objective"], "runtime_seconds": time.perf_counter() - started,
    "registered_gradient_target": 0.01,
}, indent=2))
