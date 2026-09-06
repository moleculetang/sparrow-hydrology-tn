from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
TOPOLOGY = ROOT / "5_Test/20260828_13/outputs/priority_reservoir_topology_audit.parquet"
REPORT = ROOT / "5_Test/20260828_13/reports/priority_reservoir_block_details.json"


def _json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    frame = pd.read_parquet(TOPOLOGY)
    blocked = frame.loc[~frame["operator_ready_for_formal_fit"].astype(bool)].copy()
    records = [
        {column: _json_value(value) for column, value in row.items()}
        for row in blocked.to_dict(orient="records")
    ]
    payload = {
        "stage": "20260828_13",
        "status": "BLOCK_DETAIL_AUDIT",
        "source": str(TOPOLOGY),
        "priority_rows": int(len(frame)),
        "blocked_rows": int(len(blocked)),
        "blocked": records,
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
