from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from runtime_guard import assert_sparrow_runtime


assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "inputs" / "parent_indata.parquet"
TARGET = ROOT / "inputs" / "development_indata_2006_2018.parquet"
STATE_TARGET = ROOT / "inputs" / "state_forcing_2006_2022_no_locked_observations.parquet"
REPORT = ROOT / "reports" / "development_input_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    # Predicate pushdown prevents materializing locked-period rows or values.
    frame = pd.read_parquet(SOURCE, filters=[("year", "<=", 2018)])
    if int(frame["year"].max()) != 2018 or int(frame["year"].min()) != 2006:
        raise RuntimeError("Development year boundary failed")
    if len(frame) != 230 * 156:
        raise RuntimeError(f"Unexpected development rows: {len(frame)}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(TARGET, index=False)
    # State/interface trajectories may use locked-period meteorological forcing,
    # but never locked discharge observations.  Avoid reading that column at all.
    state_columns = [name for name in pd.read_parquet(SOURCE, columns=[]).columns]
    # Pandas returns no names for columns=[]; obtain schema names without values.
    import pyarrow.parquet as pq
    state_columns = [name for name in pq.ParquetFile(SOURCE).schema.names if name != "Q_obsv_cfs"]
    state_frame = pd.read_parquet(SOURCE, columns=state_columns)
    state_frame["Q_obsv_cfs"] = float("nan")
    state_frame.to_parquet(STATE_TARGET, index=False)
    payload = {
        "source": str(SOURCE),
        "target": str(TARGET),
        "predicate": "year <= 2018",
        "rows": int(len(frame)),
        "reaches": int(frame["comid"].nunique()),
        "months_per_reach": int(frame.groupby("comid").size().min()),
        "min_year": int(frame["year"].min()),
        "max_year": int(frame["year"].max()),
        "target_sha256": sha256(TARGET),
        "state_forcing_rows": int(len(state_frame)),
        "state_forcing_sha256": sha256(STATE_TARGET),
        "state_forcing_locked_discharge_observations_materialized": False,
        "locked_value_access": False,
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
