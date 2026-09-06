"""Pre-registered reference-equivalent optimizer probe on one T3 LORO fold."""

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
nested = pd.read_parquet(ROOT / "5_Test/20260824_47/outputs/stage47_nested_spatial_parameters.parquet")
row = nested.loc[nested.candidate.eq("MINERAL_LIFETIME") & nested.fold_id.eq("T3_LORO_R002")].iloc[0]
obs = common.observations(); train, _ = fold_frames(obs, row)
temporal_parameters = pd.read_parquet(ROOT / "5_Test/20260831_3/outputs/objective_fold_parameters.parquet")
temporal_row = temporal_parameters.loc[
    temporal_parameters.objective_id.eq("O0_LOG_T4") & temporal_parameters.fold_id.eq("T3")
].iloc[0]
temporal_sites = pd.read_parquet(ROOT / "5_Test/20260831_3/outputs/objective_site_effects.parquet")
temporal_sites = temporal_sites.loc[
    temporal_sites.objective_id.eq("O0_LOG_T4") & temporal_sites.fold_id.eq("T3")
]
model = ReservoirTNModel(); anchor = np.asarray([temporal_row[name] for name in model.names()], dtype=float)
stations = sorted(train.station_key.astype(str).unique())
site_map = temporal_sites.set_index("station_key").b_raw_log_unit.to_dict()
site_start = np.asarray([float(site_map.get(station, 0.0)) for station in stations])
v_f = float(temporal_row.selected_v_f)
model.set_fixed_v_f(v_f); model._obs_index_cache.clear(); model._q_feature_cache.clear()
started = time.perf_counter(); objective = Objective(model, train, "O0_LOG_T4", anchor, site_start)
fit = optimize_at_vf(objective, anchor, site_start, v_f, 0, 0.0, 35)
result = {
    "fold_id": row.fold_id,
    "gradient_max_abs": fit["gradient_max_abs"],
    "objective": fit["objective"],
    "runtime_seconds": time.perf_counter() - started,
    "registered_gradient_target": 0.005,
    "passed": bool(fit["gradient_max_abs"] <= 0.005),
}
common.atomic_json(result, S4.parent / "reports/reference_equivalent_probe.json")
print(json.dumps(result, indent=2))
