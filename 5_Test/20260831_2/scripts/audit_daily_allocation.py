"""Diagnose monthly TN mass not represented by registered daily path weights."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reservoir_tn_core import ReservoirTNModel  # noqa: E402


torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
model = ReservoirTNModel()
parameters = pd.read_parquet(
    r"E:\SPARROW\5_Test\20260824_46\outputs\by_candidate\mineral_lifetime_parameters.parquet"
)
rows = []
for fold_id in ["T1", "T2", "T3"]:
    row = parameters.loc[parameters.fold_id.eq(fold_id)].iloc[0]
    physical = torch.tensor([row[name] for name in model.names()])
    values = dict(zip(model.names(), physical))
    fast, slow = model.local_fluxes_2006(values)
    fast_sum = model.fast_day_weight.sum(dim=1)
    slow_sum = model.slow_day_weight.sum(dim=1)
    missing_fast = fast * (1.0 - fast_sum)
    missing_slow = slow * (1.0 - slow_sum)
    for path, mass, weights in [
        ("fast", fast, fast_sum), ("slow", slow, slow_sum),
    ]:
        missing = mass * (1.0 - weights)
        affected = (torch.abs(1.0 - weights) > 1.0e-12) & (mass > 0)
        rows.append({
            "fold_id": fold_id,
            "path": path,
            "monthly_mass_kg_n": float(mass.sum()),
            "missing_mass_kg_n": float(missing.sum()),
            "relative_missing": float(missing.sum() / torch.clamp(mass.sum(), min=1.0)),
            "affected_reach_months": int(affected.sum()),
            "maximum_affected_monthly_mass_kg_n": float(torch.max(torch.where(affected, mass, torch.zeros_like(mass)))),
        })
print(pd.DataFrame(rows).to_string(index=False))
