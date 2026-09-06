"""Verify that the sensitivity run did not mutate the formal product.

The common 1961-2024 period is hashed row-wise across every exported column,
so the comparison is independent of Parquet row-group layout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_38"
FORMAL = ROOT / "5_Test" / "20260828_35" / "outputs"
SENSITIVITY = RUN / "outputs"
FORMAL_LOCK = ROOT / "5_Test" / "20260828_37" / "locks" / "program_completion_lock.json"
REPORT = RUN / "reports" / "dual_product_crosscheck.json"
LOCK = RUN / "locks" / "dual_product_crosscheck_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_object(value: object) -> object:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def logical_hash(path: Path, time_column: str, cutoff: pd.Timestamp | None) -> tuple[str, int, list[str]]:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    digest = hashlib.sha256()
    rows = 0
    for batch in parquet.iter_batches(batch_size=16_384, columns=columns):
        frame = batch.to_pandas()
        if cutoff is not None:
            frame = frame[pd.to_datetime(frame[time_column]) <= cutoff]
        if frame.empty:
            continue
        for column in frame.select_dtypes(include=["object"]).columns:
            frame[column] = frame[column].map(stable_object)
        hashed = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(dtype="uint64")
        digest.update(hashed.tobytes())
        rows += len(frame)
    return digest.hexdigest(), rows, columns


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    formal_lock = json.loads(FORMAL_LOCK.read_text(encoding="utf-8"))
    specs = {
        "reach_daily": ("tn_hydrology_reach_daily.parquet", "date", pd.Timestamp("2024-12-31")),
        "reach_monthly": ("tn_hydrology_reach_monthly.parquet", "month", pd.Timestamp("2024-12-01")),
        "reservoir_daily": ("tn_hydrology_reservoir_daily.parquet", "date", pd.Timestamp("2024-12-31")),
        "reservoir_monthly": ("tn_hydrology_reservoir_monthly.parquet", "month", pd.Timestamp("2024-12-01")),
    }
    comparisons = {}
    for label, (filename, time_column, cutoff) in specs.items():
        formal_path = FORMAL / filename
        sensitivity_path = SENSITIVITY / filename
        formal_logical, formal_rows, formal_columns = logical_hash(formal_path, time_column, None)
        sensitivity_logical, sensitivity_rows, sensitivity_columns = logical_hash(sensitivity_path, time_column, cutoff)
        comparisons[label] = {
            "formal_file_hash_matches_completion_lock": sha256(formal_path) == formal_lock["files"][label],
            "schemas_identical": formal_columns == sensitivity_columns,
            "common_period_rows_identical": formal_rows == sensitivity_rows,
            "common_period_values_identical": formal_logical == sensitivity_logical,
            "formal_rows": formal_rows,
            "sensitivity_common_rows": sensitivity_rows,
            "formal_logical_sha256": formal_logical,
            "sensitivity_common_logical_sha256": sensitivity_logical,
        }
    static_formal = FORMAL / "tn_hydrology_reservoir_static_metadata.parquet"
    static_sensitivity = SENSITIVITY / "tn_hydrology_reservoir_static_metadata.parquet"
    comparisons["reservoir_static"] = {
        "formal_file_hash_matches_completion_lock": sha256(static_formal) == formal_lock["files"]["reservoir_static"],
        "files_identical": sha256(static_formal) == sha256(static_sensitivity),
    }
    passed = all(all(bool(value) for key, value in item.items() if isinstance(value, bool)) for item in comparisons.values())
    report = {
        "stage": "20260828_38",
        "status": "PASS_DUAL_PRODUCT_ISOLATION_AND_COMMON_PERIOD_IDENTITY" if passed else "FAIL_DUAL_PRODUCT_CROSSCHECK",
        "comparison_period": "1961-01-01 through 2024-12-31",
        "comparisons": comparisons,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = {
        "stage": "20260828_38",
        "status": report["status"],
        "report": sha256(REPORT),
        "verifier_code": sha256(Path(__file__)),
        "formal_completion_lock": sha256(FORMAL_LOCK),
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
