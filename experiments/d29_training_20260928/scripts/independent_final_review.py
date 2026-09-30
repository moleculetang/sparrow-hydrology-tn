"""Read-only, standard-library check of frozen campaign receipts and score summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EVAL = OUT / "evaluation"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def numeric(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def equal(a, b, *, tolerance=1e-9):
    return (a is None and b is None) or (
        a is not None and b is not None and math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
    )


def main():
    issues = []
    manifest = read_json(OUT / "campaign_manifest.json")
    records = manifest["training_records"]
    selected = manifest["selected_jobs"]
    accounted = manifest["accounted_paths"]
    if len(accounted) != 32 or len(selected) != 24 or len(set(accounted)) != 32:
        issues.append("campaign path counts or uniqueness differ from 32/24")
    if set(accounted) != set(records) or set(selected) - set(accounted):
        issues.append("manifest paths and training records differ")
    if not manifest.get("selection_training_only") or manifest.get("excluded_jobs"):
        issues.append("selection/exclusion contract differs from registered rule")

    status_counts = defaultdict(int)
    for job, rec in records.items():
        status_counts[rec["status"]] += 1
        folder = OUT / "jobs" / job
        for filename, field in (
            ("independent_audit.json", "audit_sha256"),
            ("physical_ledger.json", "ledger_sha256"),
        ):
            path = folder / filename
            if not path.is_file() or digest(path) != rec[field]:
                issues.append(f"{job}: {filename} hash mismatch or missing")
        audit = read_json(folder / "independent_audit.json")
        ledger = read_json(folder / "physical_ledger.json")
        if not (audit.get("objective_passed") and ledger.get("passed") and rec["objective_passed"] and rec["physical_passed"]):
            issues.append(f"{job}: objective or ledger gate failed")
        if not equal(audit.get("objective"), rec.get("objective")) or not equal(audit.get("projected_gradient"), rec.get("projected_gradient")):
            issues.append(f"{job}: manifest audit values differ")
        if bool(audit.get("numerically_sufficient")) != (numeric(rec.get("projected_gradient")) <= 1e-5):
            issues.append(f"{job}: projected-gradient sufficiency mismatch")
        if ledger.get("local_balance_max_kg", 0) > 1e-6 or ledger.get("source_balance_max_kg", 0) > 1e-6:
            issues.append(f"{job}: local/source balance tolerance exceeded")

    groups = defaultdict(list)
    for job in accounted:
        groups[job.rsplit("_s", 1)[0]].append(job)
    for config, jobs in groups.items():
        expected = min(jobs, key=lambda j: records[j]["objective"])
        if expected not in selected:
            issues.append(f"{config}: selected checkpoint is not minimum training objective")

    pairs_checked = 0
    pair_rows = read_json(EVAL / "paired_summaries.json")
    for pair in pair_rows:
        if pair["role"] not in ("time", "same_year_spatial", "space_time"):
            continue
        job, scope, year = pair["job"], pair["scope"], pair["year"]
        filename = f"{scope}_{year}_heldout_stations.csv"
        path = EVAL / job / filename
        if not path.is_file():
            issues.append(f"{job}: missing {filename}")
            continue
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        for key, column in (("NSE", "NSE"), ("month_centered_NSE", "month_centered_NSE")):
            if key == "month_centered_NSE" and scope != "daily":
                continue
            baseline = {r["station_key"]: numeric(r[column]) for r in rows if r["configuration"] == "baseline"}
            candidate = {r["station_key"]: numeric(r[column]) for r in rows if r["configuration"] == "candidate"}
            common = [(baseline[s], candidate[s]) for s in baseline.keys() & candidate.keys()
                      if baseline[s] is not None and candidate[s] is not None]
            given = pair[key]
            if len(common) != given["common_stations"]:
                issues.append(f"{job}/{scope}/{year}/{key}: common-station count differs")
                continue
            if common:
                before, after = zip(*common)
                computed = {
                    "baseline_median": statistics.median(before),
                    "candidate_median": statistics.median(after),
                    "median_difference": statistics.median(after) - statistics.median(before),
                    "median_paired_difference": statistics.median(b - a for a, b in common),
                    "improved_fraction": sum(b > a for a, b in common) / len(common),
                }
                for name, value in computed.items():
                    if not equal(value, given.get(name)):
                        issues.append(f"{job}/{scope}/{year}/{key}: {name} differs")
        pairs_checked += 1

    bootstrap_checked = 0
    for job in selected:
        for path in (EVAL / job).glob("*_bootstrap_receipt.json"):
            receipt = read_json(path)
            if receipt.get("seed") != 1729 or receipt.get("replicates") != 1000 or receipt.get("block_months") not in (1, 2):
                issues.append(f"{job}: {path.name} sampling contract differs")
            if len(receipt.get("draw_indices", [])) != 1000:
                issues.append(f"{job}: {path.name} contains an incomplete draw ledger")
            bootstrap_checked += 1

    event_receipts = 0
    for job in selected:
        for path in (EVAL / job).glob("events_*_support.json"):
            support = read_json(path)
            table = path.with_name(path.name.replace("_support.json", ".csv"))
            with table.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            if len(rows) != 2 * support["events_in_requested_spatial_support"]:
                issues.append(f"{job}: {table.name} row count differs from two configurations per event")
            if support["year"] == 2024 and support["registered_events"] != 57:
                issues.append(f"{job}: frozen 2024 event count is not 57")
            if not support.get("background_clipped_to_evaluation_year"):
                issues.append(f"{job}: event background crosses year boundary")
            event_receipts += 1

    outcome = {
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not issues,
        "issues": issues,
        "accounted_paths": len(accounted),
        "selected_jobs": len(selected),
        "numerically_insufficient_selected": [j for j in selected if not records[j]["numerically_sufficient"]],
        "training_status_counts": dict(status_counts),
        "heldout_pair_tables_checked": pairs_checked,
        "bootstrap_receipts_checked": bootstrap_checked,
        "event_support_receipts_checked": event_receipts,
        "scope": "receipt/hash/summary consistency; does not independently re-run full historical physical model",
    }
    target = OUT / "scientific_independent_review.json"
    target.write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
