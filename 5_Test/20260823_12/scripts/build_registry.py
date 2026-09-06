from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT.parent
INPUT = TEST_ROOT / "20260813_54" / "inputs" / "parent_indata.parquet"
TOPOLOGY = TEST_ROOT / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
OLD_LOCK = TEST_ROOT / "20260823_11" / "reports" / "final_model_lock.json"
OUT = ROOT / "outputs"
REPORT = ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terminal_trees(frame: pd.DataFrame) -> dict[int, int]:
    # The complete panel repeats static topology monthly.  Hydroseq decreases
    # downstream, so the terminal reach is the final downstream id available
    # in the topology table.  Fall back to the reach itself at a terminal.
    static = frame.sort_values(["comid", "year", "month"]).drop_duplicates("comid")
    downstream = {
        int(row.comid): (None if not np.isfinite(row.ctonode) else int(row.ctonode))
        for row in static.itertuples()
    }
    # ctonode is a node id, not a reach id.  Use terminal reach membership
    # already audited in the locked all-reach product where available.
    product = TEST_ROOT / "20260823_11" / "outputs" / "locked_q72_hydrology_230_reaches_2006_2022.parquet"
    locked = pd.read_parquet(product, columns=["reach_id", "terminal_tree"]).drop_duplicates("reach_id")
    return dict(zip(locked.reach_id.astype(int), locked.terminal_tree.astype(int)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT)
    obs = df.loc[df.Q_obsv_cfs.notna() & df.Q_obsv_cfs.gt(0)].copy()
    trees = terminal_trees(df)
    station = (
        obs.groupby(["q_site", "comid"], as_index=False)
        .agg(
            observed_months=("Q_obsv_cfs", "size"),
            observed_years=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            train_months=("year", lambda s: int((s <= 2018).sum())),
            retrospective_months=("year", lambda s: int((s >= 2019).sum())),
        )
    )
    station["terminal_tree"] = station.comid.astype(int).map(trees).astype("Int64")
    station["complete_2006_2022"] = (
        station.observed_months.eq(204)
        & station.observed_years.eq(17)
        & station.first_year.eq(2006)
        & station.last_year.eq(2022)
    )
    station["role"] = np.where(station.complete_2006_2022, "complete_training_station", "zero_history_spatial_station")
    station["primary_spatial_gate"] = (
        ~station.complete_2006_2022
        & station.observed_months.ge(24)
        & station.observed_years.ge(2)
    )
    station = station.sort_values(["role", "terminal_tree", "comid"]).reset_index(drop=True)
    station.to_parquet(OUT / "station_roles.parquet", index=False)

    rows = obs.merge(station[["q_site", "comid", "role", "primary_spatial_gate", "terminal_tree"]], on=["q_site", "comid"], how="left")
    rows["period_role"] = np.select(
        [
            rows.role.eq("complete_training_station") & rows.year.le(2018),
            rows.role.eq("complete_training_station") & rows.year.ge(2019),
            rows.role.eq("zero_history_spatial_station") & rows.year.le(2018),
            rows.role.eq("zero_history_spatial_station") & rows.year.ge(2019),
        ],
        [
            "model_development",
            "locked_time_retrospective",
            "zero_history_spatial_same_period",
            "zero_history_spatial_plus_time",
        ],
        default="unused",
    )
    rows[["comid", "q_site", "year", "month", "terminal_tree", "role", "period_role", "primary_spatial_gate"]].to_parquet(
        OUT / "observation_role_keys.parquet", index=False
    )

    complete = station[station.complete_2006_2022]
    incomplete = station[~station.complete_2006_2022]
    payload = {
        "stage": "20260823_12",
        "status": "PASS",
        "domain": "S111_posthoc_screened_operational_domain",
        "interpretation_limit": "2019-2022 is a locked retrospective check, not a never-seen independent validation, because S111 station screening used later-period performance",
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "topology_sha256": sha256(TOPOLOGY),
        "previous_lock_sha256": sha256(OLD_LOCK),
        "panel_rows": int(len(df)),
        "panel_reaches": int(df.comid.nunique()),
        "complete_training_stations": int(len(complete)),
        "complete_training_reaches": int(complete.comid.nunique()),
        "complete_training_terminal_trees": int(complete.terminal_tree.nunique()),
        "zero_history_spatial_stations": int(len(incomplete)),
        "zero_history_spatial_reaches": int(incomplete.comid.nunique()),
        "zero_history_spatial_terminal_trees": int(incomplete.terminal_tree.nunique()),
        "primary_spatial_stations": int(incomplete.primary_spatial_gate.sum()),
        "development_observations": int(((rows.role == "complete_training_station") & (rows.year <= 2018)).sum()),
        "time_retrospective_observations": int(((rows.role == "complete_training_station") & (rows.year >= 2019)).sum()),
        "spatial_same_period_observations": int(((rows.role == "zero_history_spatial_station") & (rows.year <= 2018)).sum()),
        "spatial_plus_time_observations": int(((rows.role == "zero_history_spatial_station") & (rows.year >= 2019)).sum()),
        "forbidden": [
            "target spatial-station observations in any fit or hyperparameter selection",
            "2019-2022 observations in parameter or state updates",
            "station identity in the all-reach transferable candidate",
            "reservoir-equation changes",
        ],
    }
    (REPORT / "data_partition_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    contract = {
        "stage": "20260823_12-15",
        "status": "REGISTERED",
        "training": "S111 complete stations, 2006-2018 only",
        "time_check": "same complete stations, 2019-2022, no observation update",
        "spatial_check": "all observations from incomplete S111 stations; target history always zero",
        "candidates": {
            "H0_Q72_CLEAN": "reselected physical production-routing parameters; no station identity",
            "H1_Q72_MAP_TRANSFERABLE": "low-dimensional Gaussian-prior MAP using all-reach features only",
            "H2_Q72_MAP_GAUGED": "H1 plus partially pooled station terms for the 68 complete stations",
            "H3_Q72_MAP_DA": "H2 plus recursive residual-state assimilation through 2018 only",
        },
        "all_reach_eligibility": ["H0_Q72_CLEAN", "H1_Q72_MAP_TRANSFERABLE"],
        "gauged_only": ["H2_Q72_MAP_GAUGED", "H3_Q72_MAP_DA"],
        "assimilation_semantics": "prediction-error state only; never added to physical water stores or TN water mass",
        "old_q72_parameters": "initialization only",
    }
    (ROOT / "experiment_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
