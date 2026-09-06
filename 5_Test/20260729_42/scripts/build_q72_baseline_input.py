from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
SOURCE = (
    RUN / "inputs" / "authoritative_20260728_17" / "indata.parquet"
)
OUTPUT = RUN / "inputs" / "indata.parquet"
AUDIT = RUN / "inputs" / "source_metadata"
FIXED_EXCLUSIONS = {
    "劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"
}
PROTECTED = "石角站"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(SOURCE)
    required = {
        "comid", "year", "month", "q_site",
        "Q_obsv_cfs", "station_id", "ifmon1",
    }
    if not required <= set(frame.columns):
        raise RuntimeError(
            f"Missing authoritative columns: {sorted(required-set(frame.columns))}"
        )

    source_mask = (
        frame["q_site"].astype(str).eq("金鸡站")
        & frame["year"].between(2006, 2007)
        & frame["comid"].eq(6)
    )
    source_rows = frame.loc[
        source_mask,
        ["year", "month", "q_site", "Q_obsv_cfs", "station_id", "ifmon1"],
    ].copy()
    if len(source_rows) != 24:
        raise RuntimeError(
            f"Expected 24 legacy 金鸡站 rows on reach 6, found {len(source_rows)}"
        )
    target_index = (
        frame.loc[
            frame["comid"].eq(5) & frame["year"].between(2006, 2007),
            ["year", "month"],
        ]
        .reset_index()
        .rename(columns={"index": "target_index"})
    )
    transfer = source_rows.merge(
        target_index,
        on=["year", "month"],
        how="inner",
        validate="one_to_one",
    )
    if len(transfer) != 24:
        raise RuntimeError(
            f"Expected 24 金鸡站 target months on reach 5, found {len(transfer)}"
        )
    for row in transfer.itertuples(index=False):
        target = int(row.target_index)
        frame.loc[target, "q_site"] = row.q_site
        frame.loc[target, "Q_obsv_cfs"] = row.Q_obsv_cfs
        frame.loc[target, "station_id"] = row.station_id
        frame.loc[target, "ifmon1"] = row.ifmon1
    frame.loc[source_mask, ["q_site", "Q_obsv_cfs", "station_id"]] = np.nan
    frame.loc[source_mask, "ifmon1"] = 0.0

    active = set(frame["q_site"].dropna().astype(str))
    if active & FIXED_EXCLUSIONS:
        raise RuntimeError(
            f"Fixed exclusions entered Q72 input: {sorted(active & FIXED_EXCLUSIONS)}"
        )
    if PROTECTED not in active:
        raise RuntimeError("Protected station 石角站 is absent")

    frame.to_parquet(OUTPUT, index=False)
    policy_rows = [
        {
            "station_name": station,
            "q72_policy": (
                "must_retain" if station == PROTECTED else "fixed_exclusion"
            ),
            "active_in_q72_input": station in active,
        }
        for station in sorted(FIXED_EXCLUSIONS | {PROTECTED})
    ]
    pd.DataFrame(policy_rows).to_csv(
        AUDIT / "q72_station_policy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    patch = {
        "run_id": RUN.name,
        "authoritative_parent": "20260728_17",
        "purpose": (
            "Reproduce the later frozen Q72 blocked-fold mapping while keeping "
            "the 20260728_17 input as an immutable local source snapshot."
        ),
        "explicit_patch": {
            "station": "金鸡站",
            "period": [2006, 2007],
            "months": 24,
            "from_reach": 6,
            "to_reach": 5,
            "fields_transferred": [
                "q_site", "Q_obsv_cfs", "station_id", "ifmon1"
            ],
        },
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "active_station_count": int(frame["q_site"].nunique(dropna=True)),
        "fixed_exclusions_absent": not bool(active & FIXED_EXCLUSIONS),
        "protected_shijiao_present": PROTECTED in active,
        "source_sha256": sha256(SOURCE),
        "output_sha256": sha256(OUTPUT),
        "runtime": RUNTIME,
    }
    (AUDIT / "q72_input_build_audit.json").write_text(
        json.dumps(patch, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(patch, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
