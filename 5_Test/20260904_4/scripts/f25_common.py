"""Shared observation and checkpoint helpers for the F25 sensitivity fit."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260904_7"


def observations() -> pd.DataFrame:
    old = pd.read_parquet(ROOT / "5_Test/20260824_18/outputs/tn_observations_audited.parquet")
    old = old.loc[old.formal_river_channel].copy()
    positions = pd.read_parquet(ROOT / "5_Test/20260824_12/outputs/tn_observations_primary_2016_2024.parquet")
    positions = positions[["station_key", "reach_id", "downstream_fraction_on_reach"]].drop_duplicates(
        ["station_key", "reach_id"]
    )
    old = old.merge(positions, on=["station_key", "reach_id"], how="left", validate="many_to_one")
    new = pd.read_parquet(
        ROOT / "0_water_quality/data/preprocess/model_ready/tn_station_month_2025_prb_sensitivity.parquet"
    )
    columns = [
        "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l",
        "downstream_fraction_on_reach",
    ]
    result = pd.concat([old[columns], new[columns]], ignore_index=True)
    if result.duplicated(["station_key", "year", "month"]).any():
        raise RuntimeError("Duplicate station-month after appending 2025")
    if result[columns].isna().any().any():
        raise RuntimeError("Missing required F25 observation field")
    return result.sort_values(["station_key", "year", "month"]).reset_index(drop=True)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def save_trial(variant: int, result: dict[str, object]) -> None:
    target = RUN / "work" / f"h7_f25_start{variant}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            physical=result["physical"], gamma=result["gamma"], site_raw=result["site_raw"],
            site_effect=result["site_effect"], reach_effect=result["reach_effect"],
        )
    os.replace(temporary, target)
    atomic_json({
        "variant": variant,
        "objective": result["objective"],
        "combined_kkt": result["combined_kkt"],
        "process_site_kkt": result["process_site_kkt"],
        "gamma_kkt": result["gamma_kkt"],
        "finite": result["finite"],
    }, RUN / "work" / f"h7_f25_start{variant}.json")


def load_trial(variant: int) -> dict[str, object]:
    arrays = np.load(RUN / "work" / f"h7_f25_start{variant}.npz")
    metadata = json.loads((RUN / "work" / f"h7_f25_start{variant}.json").read_text(encoding="utf-8"))
    return {**metadata, **{name: arrays[name].copy() for name in arrays.files}}
