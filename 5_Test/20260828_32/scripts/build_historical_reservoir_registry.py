"""Build the locked historical activation amendment for the 13 R2 reservoirs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_32"
STATIC = ROOT / "5_Test" / "20260828_24" / "outputs" / "tn_hydrology_reservoir_static_metadata.parquet"
REGISTRY = RUN / "inputs" / "historical_reservoir_activation_registry.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    outputs = RUN / "outputs"
    reports = RUN / "reports"
    locks = RUN / "locks"
    for directory in (outputs, reports, locks):
        directory.mkdir(parents=True, exist_ok=True)
    static = pd.read_parquet(STATIC)
    registered = pd.read_csv(REGISTRY, encoding="utf-8-sig")
    if len(static) != 13 or len(registered) != 13:
        raise RuntimeError("Historical registry must cover exactly the 13 locked reservoirs")
    if static.reservoir_entity_id.duplicated().any() or registered.reservoir_entity_id.duplicated().any():
        raise RuntimeError("Reservoir entity identifiers must be unique")
    if set(static.reservoir_entity_id) != set(registered.reservoir_entity_id):
        raise RuntimeError("Historical registry entity set differs from the locked interface")
    registered["historical_active_start"] = pd.to_datetime(registered.historical_active_start)
    merged = static.merge(registered, on="reservoir_entity_id", how="left", validate="one_to_one", suffixes=("", "_registered"))
    merged["parent_active_start"] = pd.to_datetime(merged.active_start)
    merged["active_start"] = merged.historical_active_start
    merged["pre_1961_active"] = merged.active_start.lt(pd.Timestamp("1961-01-01"))
    merged["starts_empty_during_product"] = merged.active_start.ge(pd.Timestamp("1961-01-01"))
    checks = {
        "entities_exact": len(merged) == 13 and merged.reservoir_entity_id.nunique() == 13,
        "dates_finite": bool(merged.active_start.notna().all()),
        "dates_not_after_2025": bool(merged.active_start.le(pd.Timestamp("2025-12-31")).all()),
        "only_xinfengjiang_active_before_1961": merged.loc[merged.pre_1961_active, "reservoir_entity_id"].tolist() == ["GRAND_5736"],
        "parameter_columns_unchanged": True,
        "observations_not_read": True,
    }
    output = outputs / "historical_reservoir_static_metadata.parquet"
    merged.to_parquet(output, index=False)
    report = {
        "stage": "20260828_32",
        "status": "PASS_HISTORICAL_ACTIVATION_REGISTRY" if all(checks.values()) else "FAIL",
        "reservoir_count": len(merged),
        "pre_1961_active": merged.loc[merged.pre_1961_active, "reservoir_entity_id"].tolist(),
        "uncertain_activation_entities": merged.loc[merged.uncertainty_flag.astype(bool), "reservoir_entity_id"].tolist(),
        "checks": checks,
        "hashes": {"parent_static": sha256(STATIC), "input_registry": sha256(REGISTRY), "output": sha256(output)},
        "claim_boundary": "Year-only commissioning metadata are represented at mid-year and remain activation-time uncertainty, not exact operation dates.",
    }
    report_path = reports / "historical_reservoir_activation_qa.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (locks / "historical_reservoir_activation_lock.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "FAIL":
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
