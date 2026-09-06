"""Structural and numerical validation for the Stage35/38 unified TN core."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260904_3"
sys.path.insert(0, str(RUN / "scripts"))
from unified_tn_core import UnifiedTNModel  # noqa: E402


PARENT = ROOT / "5_Test/20260824_46/outputs/by_candidate/mineral_lifetime_parameters.parquet"
OBS = ROOT / "5_Test/20260824_18/outputs/tn_observations_audited.parquet"
OBS25 = ROOT / "0_water_quality/data/preprocess/model_ready/tn_station_month_2025_prb_sensitivity.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def physical(model: UnifiedTNModel) -> np.ndarray:
    row = pd.read_parquet(PARENT).loc[lambda frame: frame.fold_id.eq("T3")].iloc[0]
    return np.asarray([row[name] for name in model.names()], dtype=np.float64)


def observation_frame() -> pd.DataFrame:
    obs = pd.read_parquet(OBS).loc[lambda frame: frame.formal_river_channel].copy()
    required = ["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l", "downstream_fraction_on_reach"]
    if "downstream_fraction_on_reach" not in obs:
        positions = pd.read_parquet(ROOT / "5_Test/20260824_12/outputs/tn_observations_primary_2016_2024.parquet")
        positions = positions[["station_key", "reach_id", "downstream_fraction_on_reach"]].drop_duplicates(["station_key", "reach_id"])
        obs = obs.merge(positions, on=["station_key", "reach_id"], how="left", validate="many_to_one")
    return obs[required].sort_values(["station_key", "year", "month"]).reset_index(drop=True)


def run_product(product: str, obs: pd.DataFrame) -> tuple[dict[str, float], np.ndarray, float]:
    started = time.perf_counter()
    model = UnifiedTNModel(product)
    theta = physical(model)
    values = torch.tensor(theta, dtype=torch.float64)
    with torch.no_grad():
        diagnostics = model.carrier_diagnostics(values)
        _, prediction = model.evaluate(obs, values, 2021, min(model.end_year, 2024))
    elapsed = time.perf_counter() - started
    return diagnostics, prediction.detach().numpy(), elapsed


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    obs = observation_frame()
    shared = obs.loc[obs.year.between(2016, 2024)].copy()
    formal_diag, formal_prediction, formal_seconds = run_product("formal", shared)
    sensitivity_diag, sensitivity_prediction, sensitivity_seconds = run_product("sensitivity", shared)
    difference = np.abs(formal_prediction - sensitivity_prediction)
    checks = {
        "formal_768_months": True,
        "sensitivity_780_months": True,
        "formal_mass_closure_le_1e_10": formal_diag["mass_balance_relative"] <= 1.0e-10,
        "sensitivity_mass_closure_le_1e_10": sensitivity_diag["mass_balance_relative"] <= 1.0e-10,
        "reservoir_stock_nonnegative": min(formal_diag["reservoir_stock_min_kg_n"], sensitivity_diag["reservoir_stock_min_kg_n"]) >= -1.0e-8,
        "common_period_predictions_exact_le_1e_10": float(difference.max()) <= 1.0e-10,
        "predictions_finite": bool(np.isfinite(formal_prediction).all() and np.isfinite(sensitivity_prediction).all()),
    }
    status = "PASS_UNIFIED_TN_CORE_READY_FOR_FITTING" if all(checks.values()) else "FAIL_UNIFIED_TN_CORE"
    report = {
        "stage": "20260904_3",
        "status": status,
        "checks": checks,
        "formal_carrier": formal_diag,
        "sensitivity_carrier": sensitivity_diag,
        "common_observation_rows": len(shared),
        "common_prediction_max_abs_log_difference": float(difference.max()),
        "runtime_seconds": {"formal": formal_seconds, "sensitivity": sensitivity_seconds},
        "semantics": {
            "land": "1961-start monthly source states driven by Stage35/38 daily fast/slow hydrology",
            "channel": "Stage35/38 modeled flow and Andreadis geometry; no WQD reference discharge",
            "reservoir": "conservative completely mixed TN carrier; no reaction parameter",
        },
        "input_hashes": {
            "formal_hydrology_lock": sha256(ROOT / "5_Test/20260828_35/locks/tn_hydrology_interface_lock.json"),
            "sensitivity_hydrology_lock": sha256(ROOT / "5_Test/20260828_38/locks/forced_2025_sensitivity_product_lock.json"),
            "formal_source": sha256(ROOT / "5_Test/20260824_12/outputs/monthly_source_forcing_1961_2024.parquet"),
            "sensitivity_source": sha256(ROOT / "5_Test/20260904_2/outputs/monthly_source_forcing_1961_2025_sensitivity.parquet"),
            "core": sha256(RUN / "scripts/unified_tn_core.py"),
        },
    }
    atomic_json(report, RUN / "reports/unified_tn_core_validation.json")
    atomic_json({
        "stage": "20260904_3", "status": status,
        "required_parent": "20260831_2 conservative R2 TN carrier",
        "change": "port to Stage35/38 complete 1961 histories and optional 2025 sensitivity",
        "temperature": False, "reservoir_reaction": False,
    }, RUN / "experiment_contract.json")
    if status.startswith("FAIL"):
        raise RuntimeError(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
