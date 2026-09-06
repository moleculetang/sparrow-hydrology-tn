"""Shared observation, weighting, metric and atomic-output utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260831_3"
OBS_AUDIT = ROOT / "5_Test/20260824_18/outputs/tn_observations_audited.parquet"
CURRENT_OBS = ROOT / "5_Test/20260824_12/outputs/tn_observations_primary_2016_2024.parquet"
KEYS = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
FOLDS = {
    "T1": (2021, 2021, 2022),
    "T2": (2021, 2022, 2023),
    "T3": (2021, 2023, 2024),
}


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
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def observations() -> pd.DataFrame:
    obs = pd.read_parquet(OBS_AUDIT).loc[lambda x: x.formal_river_channel].copy()
    positions = pd.read_parquet(CURRENT_OBS)[
        ["station_key", "reach_id", "downstream_fraction_on_reach", "station_to_assigned_reach_distance_m"]
    ].drop_duplicates(["station_key", "reach_id"])
    obs = obs.merge(positions, on=["station_key", "reach_id"], how="left", validate="many_to_one")
    if obs.downstream_fraction_on_reach.isna().any():
        raise RuntimeError("TN station position is incomplete")
    return obs.sort_values(["station_key", "year", "month"]).reset_index(drop=True)


def fold_frames(obs: pd.DataFrame, fold_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end, test_year = FOLDS[fold_id]
    train = obs.loc[obs.year.between(start, end)].copy()
    test = obs.loc[obs.year.eq(test_year)].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Empty temporal fold {fold_id}")
    return train, test


def data_weights(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    station_counts = frame.groupby("station_key").size()
    station_weight = frame.station_key.map(1.0 / (len(station_counts) * station_counts)).to_numpy(float)
    station_tree = frame[["station_key", "terminal_tree_id"]].drop_duplicates()
    tree_count = station_tree.terminal_tree_id.nunique()
    stations_per_tree = station_tree.groupby("terminal_tree_id").size()
    tree_weight = 1.0 / (
        tree_count * frame.terminal_tree_id.map(stations_per_tree).to_numpy(float)
        * frame.station_key.map(station_counts).to_numpy(float)
    )
    if not np.isclose(station_weight.sum(), 1.0) or not np.isclose(tree_weight.sum(), 1.0):
        raise RuntimeError("Balanced observation weights do not close")
    return station_weight, tree_weight


def center_weights(frame: pd.DataFrame, stations: list[str]) -> np.ndarray:
    station_tree = frame[["station_key", "terminal_tree_id"]].drop_duplicates().set_index("station_key")
    tree_count = station_tree.terminal_tree_id.nunique()
    stations_per_tree = station_tree.groupby("terminal_tree_id").size()
    result = np.asarray([
        0.5 / len(stations)
        + 0.5 / (tree_count * stations_per_tree.loc[station_tree.loc[station, "terminal_tree_id"]])
        for station in stations
    ])
    if not np.isclose(result.sum(), 1.0):
        raise RuntimeError("Site-effect center weights do not close")
    return result


def metric_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame.tn_mg_l.to_numpy(float)
    p = frame.pred_tn_mg_l.to_numpy(float)
    log_error = np.log1p(p) - np.log1p(y)
    residual = p - y
    station_log_rmse, station_nse = [], []
    for _, group in frame.groupby("station_key"):
        gy, gp = group.tn_mg_l.to_numpy(float), group.pred_tn_mg_l.to_numpy(float)
        station_log_rmse.append(float(np.sqrt(np.mean((np.log1p(gp) - np.log1p(gy)) ** 2))))
        denominator = float(np.sum((gy - gy.mean()) ** 2))
        station_nse.append(float(1.0 - np.sum((gp - gy) ** 2) / denominator) if denominator > 0 else math.nan)
    denominator = float(np.sum((y - y.mean()) ** 2))
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else math.nan
    alpha = float(np.std(p, ddof=0) / np.std(y, ddof=0)) if np.std(y) > 0 else math.nan
    beta = float(np.mean(p) / np.mean(y)) if np.mean(y) != 0 else math.nan
    kge = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {
        "rows": len(frame), "stations": frame.station_key.nunique(), "trees": frame.terminal_tree_id.nunique(),
        "station_macro_log_rmse": float(np.nanmean(station_log_rmse)),
        "station_median_nse": float(np.nanmedian(station_nse)),
        "pooled_log_rmse": float(np.sqrt(np.mean(log_error**2))),
        "rmse_mg_l": float(np.sqrt(np.mean(residual**2))), "mae_mg_l": float(np.mean(np.abs(residual))),
        "pbias_percent": float(100.0 * residual.sum() / y.sum()),
        "nse": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else math.nan,
        "r2": float(r**2), "kge": kge,
    }
