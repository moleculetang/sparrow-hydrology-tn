"""Independently verify and lock the authoritative 1961-2024/2025 forcing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_31"
S29 = ROOT / "5_Test" / "20260828_29"
S30 = ROOT / "5_Test" / "20260828_30"
OLD = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"
CHM_2005 = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "precipitation" / "chm_pre_v2" / "daily" / "CHM_PRE_V2_daily_2005.nc"
REPORT = RUN / "reports" / "forcing_integrity_qa.json"
LOCK = RUN / "locks" / "forcing_integrity_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_forcing(path: Path, start: str, end: str, expected_rows: int) -> dict:
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    required = {"date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"}
    if not required.issubset(columns):
        raise RuntimeError(f"forcing lacks columns: {sorted(required - columns)}")
    rows = 0
    minimum_reach = 10**9
    maximum_reach = -1
    minimum_date = None
    maximum_date = None
    finite = True
    nonnegative = True
    keys: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(
        batch_size=250_000,
        columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"],
    ):
        frame = batch.to_pandas()
        frame["date"] = pd.to_datetime(frame["date"])
        values = frame[["precipitation_daily_mm", "pet_fao56_mm_day"]].to_numpy(float)
        finite &= bool(np.isfinite(values).all())
        nonnegative &= bool((values >= 0).all())
        rows += len(frame)
        minimum_reach = min(minimum_reach, int(frame.reach_id.min()))
        maximum_reach = max(maximum_reach, int(frame.reach_id.max()))
        bmin, bmax = frame.date.min(), frame.date.max()
        minimum_date = bmin if minimum_date is None else min(minimum_date, bmin)
        maximum_date = bmax if maximum_date is None else max(maximum_date, bmax)
        keys.append(frame[["date", "reach_id"]])
    key = pd.concat(keys, ignore_index=True)
    expected_days = len(pd.date_range(start, end, freq="D"))
    day_counts = key.groupby("date", sort=False).reach_id.agg(["count", "nunique"])
    checks = {
        "rows_exact": rows == expected_rows,
        "date_range_exact": minimum_date == pd.Timestamp(start) and maximum_date == pd.Timestamp(end),
        "reach_ids_exact": minimum_reach == 1 and maximum_reach == 230,
        "daily_grid_exact": len(day_counts) == expected_days and bool((day_counts["count"] == 230).all()) and bool((day_counts["nunique"] == 230).all()),
        "duplicates_zero": not key.duplicated(["date", "reach_id"]).any(),
        "forcing_finite": finite,
        "precipitation_and_pet_nonnegative": nonnegative,
    }
    return {
        "path": str(path),
        "rows": rows,
        "date_min": str(minimum_date.date()),
        "date_max": str(maximum_date.date()),
        "checks": checks,
    }


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    qa29_path = S29 / "reports" / "historical_forcing_qa.json"
    qa30_path = S30 / "reports" / "era5_pet_harmonization_qa.json"
    if not qa29_path.is_file():
        raise RuntimeError("Stage 29 QA must exist")
    qa29 = read_json(qa29_path)
    if qa29.get("status") != "PASS":
        raise RuntimeError("Stage 29 forcing did not pass")
    qa30 = read_json(qa30_path) if qa30_path.is_file() else None
    bridge = str(qa30.get("status")) if qa30 is not None else "PENDING_BACKGROUND_REPAIR"
    if bridge not in {"PASS", "PET_EXTENSION_CONFOUNDED", "PENDING_BACKGROUND_REPAIR"}:
        raise RuntimeError(f"Unexpected Stage 30 status: {bridge}")
    historical = S29 / "outputs" / "daily_hbv_forcing_1961_2024.parquet"
    if bridge == "PASS":
        formal = S30 / "outputs" / "daily_hbv_forcing_1961_2025.parquet"
        start, end, expected_rows = "1961-01-01", "2025-12-31", 5_460_430
        status = "PASS_FORCING_LOCK_1961_2025"
    elif bridge == "PET_EXTENSION_CONFOUNDED":
        formal = historical
        start, end, expected_rows = "1961-01-01", "2024-12-31", 5_376_480
        status = "PASS_FORCING_LOCK_1961_2024_WITH_2025_SENSITIVITY"
    else:
        # The independently accepted 1961-2024 forcing is a complete base
        # product.  A background 2025 PET repair must not block its lock.
        formal = historical
        start, end, expected_rows = "1961-01-01", "2024-12-31", 5_376_480
        status = "PASS_FORCING_LOCK_1961_2024_2025_PENDING"
    if not formal.is_file():
        raise RuntimeError(f"Formal forcing is missing: {formal}")
    grid = scan_forcing(formal, start, end, expected_rows)

    repair_chm = read_json(S29 / "reports" / "chm_pre_2005_official_reacquisition_audit.json")
    cmfd_repairs = {
        label: read_json(S29 / "reports" / filename)
        for label, filename in {
            "temp_196101": "cmfd_reacquire_temp_196101.json",
            "temp_199308": "cmfd_reacquire_temp_199308.json",
            "pres_199310": "cmfd_reacquire_pres_199310.json",
            "pres_199901": "cmfd_reacquire_pres_199901.json",
        }.items()
    }
    repair_checks = {
        "chm_2005_official_replacement_passed": (
            str(repair_chm.get("status", "")).startswith("PASS")
            and CHM_2005.is_file()
            and sha256(CHM_2005) == repair_chm.get("candidate", {}).get("sha256")
        ),
        "all_four_cmfd_deep_read_repairs_passed": all(
            record.get("status") == "PASS_DEEP_AUDIT" for record in cmfd_repairs.values()
        ),
        "accepted_2006_2024_source_hash_unchanged": qa29.get("hashes", {}).get("old_2006_2024") == sha256(OLD),
        "formal_grid_passed": all(grid["checks"].values()),
        "bridge_decision_consistent": (
            (bridge == "PASS" and (S30 / "outputs" / "daily_hbv_forcing_1961_2025.parquet").is_file())
            or (bridge != "PASS" and formal == historical)
        ),
    }
    report = {
        "stage": "20260828_31",
        "status": status if all(repair_checks.values()) else "FAIL_FORCING_INTEGRITY",
        "formal_period": f"{start} through {end}",
        "formal_forcing_path": str(formal),
        "pet_bridge_status": bridge,
        "pet_2025_extension_pending": bridge == "PENDING_BACKGROUND_REPAIR",
        "grid": grid,
        "repair_and_provenance_checks": repair_checks,
        "background_era5_2006_2020_used": False,
        "discharge_observations_read": False,
        "tn_observations_read": False,
        "hashes": {
            "formal_forcing": sha256(formal),
            "stage29_qa": sha256(qa29_path),
            "stage30_qa": sha256(qa30_path) if qa30_path.is_file() else None,
            "contract": sha256(RUN / "experiment_contract.json"),
            "verifier_code": sha256(Path(__file__)),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["hashes"]["qa"] = sha256(REPORT)
    LOCK.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "FAIL_FORCING_INTEGRITY":
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
