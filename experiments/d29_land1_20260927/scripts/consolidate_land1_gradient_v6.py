"""Consolidate independent v6 partitions after the primary worker is stopped.

This preserves the original primary receipt and rejects overlapping, missing,
or non-identical implementation/reference evaluations.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.runtime import sha, write_json

out = ROOT / "outputs/land1_full_history_gradient_v6.json"
primary = json.loads(out.read_text(encoding="utf-8"))
if primary["status"] not in ("stopped_at_safe_evaluation_boundary", "passed", "requires_branch_review"):
    raise RuntimeError("PRIMARY_DERIVATIVE_WORKER_NOT_STOPPED")
backup = ROOT / "outputs/land1_full_history_gradient_v6_primary.json"
if backup.exists():
    if json.loads(backup.read_text(encoding="utf-8")) != primary:
        raise RuntimeError("PRIMARY_BACKUP_CONFLICT")
else:
    shutil.copyfile(out, backup)

if [row["index"] for row in primary["coordinates"][:6]] != list(range(6)):
    raise RuntimeError("PRIMARY_PREFIX_INCOMPLETE")
primary_prefix = dict(primary)
primary_prefix["coordinates"] = primary["coordinates"][:6]
parts = [primary_prefix]
segments = ((6, 11, 11), (11, 14, 17), (14, 17, 17),
            (17, 20, 23), (20, 23, 23))
for start, end, file_end in segments:
    path = ROOT / f"outputs/land1_gradient_v6_shard_{start:02d}_{file_end:02d}.json"
    item = json.loads(path.read_text(encoding="utf-8"))
    if item["status"] not in ("passed", "requires_branch_review", "stopped_at_safe_evaluation_boundary"):
        raise RuntimeError(f"SHARD_NOT_COMPLETE:{path}")
    if [row["index"] for row in item["coordinates"][:end-start]] != list(range(start, end)):
        raise RuntimeError(f"SHARD_COVERAGE:{path}")
    item["coordinates"] = item["coordinates"][:end-start]
    parts.append(item)

full_identity = parts[1]["identity"]
worker = json.loads((ROOT / "outputs/land1_precision_gradient_v6_worker.json").read_text(encoding="utf-8"))
for item in parts:
    if any(item["identity"].get(path) != digest for path, digest in primary["identity"].items()):
        raise RuntimeError("IMPLEMENTATION_IDENTITY_MISMATCH")
    if item is not parts[0] and item["identity"] != full_identity:
        raise RuntimeError("SHARD_IDENTITY_MISMATCH")
    if item["objective"] != primary["objective"] or item["analytic_gradient"] != primary["analytic_gradient"]:
        raise RuntimeError("REFERENCE_OBJECTIVE_OR_GRADIENT_MISMATCH")
for path, digest in full_identity.items():
    if sha(path) != digest:
        raise RuntimeError(f"IMPLEMENTATION_CHANGED:{path}")
    if path not in primary["identity"] and Path(path).stat().st_mtime > worker["created_unix"]:
        raise RuntimeError(f"PRIMARY_UNRECORDED_IDENTITY_CHANGED_DURING_SCAN:{path}")

rows = sorted((row for item in parts for row in item["coordinates"]), key=lambda row: row["index"])
if [row["index"] for row in rows] != list(range(23)):
    raise RuntimeError("MISSING_OR_OVERLAPPING_COORDINATES")
receipt = dict(primary)
receipt.update(status="passed" if all(row["passed"] for row in rows) else "requires_branch_review",
               identity=full_identity,
               coordinates=rows, pending={}, calls=sum(item["calls"] for item in parts),
               primary_identity_extended_from_shards_with_mtime_check=True,
               partition_receipts=[{"path": str(backup), "sha256": sha(backup)}] + [
                   {"path": str(ROOT / f"outputs/land1_gradient_v6_shard_{a:02d}_{file_end:02d}.json"),
                    "sha256": sha(ROOT / f"outputs/land1_gradient_v6_shard_{a:02d}_{file_end:02d}.json")}
                   for a, _, file_end in segments])
write_json(out, receipt)
print(json.dumps({"status": receipt["status"], "coordinates": len(rows),
                  "failed": [row["index"] for row in rows if not row["passed"]],
                  "calls": receipt["calls"]}))
