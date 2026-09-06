"""I/O contracts shared by Stage 4 temporal OOF workers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260904_4"
WORK = RUN / "work"
FOLDS = {
    "T1": ([2021], 2022),
    "T2": ([2021, 2022], 2023),
    "T3": ([2021, 2022, 2023], 2024),
}


def observations() -> pd.DataFrame:
    old = pd.read_parquet(ROOT / "5_Test/20260824_18/outputs/tn_observations_audited.parquet")
    old = old.loc[old.formal_river_channel & old.year.between(2016, 2024)].copy()
    positions = pd.read_parquet(ROOT / "5_Test/20260824_12/outputs/tn_observations_primary_2016_2024.parquet")
    positions = positions[["station_key", "reach_id", "downstream_fraction_on_reach"]].drop_duplicates(
        ["station_key", "reach_id"]
    )
    old = old.merge(positions, on=["station_key", "reach_id"], how="left", validate="many_to_one")
    columns = [
        "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l",
        "downstream_fraction_on_reach",
    ]
    result = old[columns].sort_values(["station_key", "year", "month"]).reset_index(drop=True)
    if result[columns].isna().any().any() or result.duplicated(["station_key", "year", "month"]).any():
        raise RuntimeError("Invalid formal observation grain")
    return result


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def prefix(capacity: str, fold: str, variant: int) -> Path:
    return WORK / f"{capacity.lower()}_{fold.lower()}_start{variant}"


def save_trial(capacity: str, fold: str, variant: int, result: dict[str, object]) -> None:
    base = prefix(capacity, fold, variant)
    base.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(base) + ".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, physical=result["physical"], gamma=result["gamma"], site_raw=result["site_raw"],
            site_effect=result["site_effect"], reach_effect=result["reach_effect"],
        )
    os.replace(temporary, Path(str(base) + ".npz"))
    atomic_json({
        "capacity": capacity, "fold_id": fold, "variant": variant,
        "objective": result["objective"], "combined_kkt": result["combined_kkt"],
        "process_site_kkt": result["process_site_kkt"], "gamma_kkt": result["gamma_kkt"],
        "finite": result["finite"],
    }, Path(str(base) + ".json"))


def load_trial(capacity: str, fold: str, variant: int) -> dict[str, object]:
    base = prefix(capacity, fold, variant)
    arrays = np.load(Path(str(base) + ".npz"))
    metadata = json.loads(Path(str(base) + ".json").read_text(encoding="utf-8"))
    return {**metadata, **{name: arrays[name].copy() for name in arrays.files}}
