from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
PARENT = ROOT / "5_Test" / "20260729_11"
CSV_PATH = RUN_DIR / "inputs" / "reach_slope_direction_adjudicated.csv"
PARQUET_PATH = RUN_DIR / "inputs" / "reach_slope_direction_adjudicated.parquet"
STATIC_IN = PARENT / "inputs" / "reach_static_reviewed_slope.parquet"
STATIC_OUT = RUN_DIR / "inputs" / "reach_static_direction_adjudicated.parquet"
MANIFEST_PATH = RUN_DIR / "inputs_manifest" / "provenance_manifest.json"
MANIFEST_CSV = RUN_DIR / "inputs_manifest" / "provenance_manifest.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path, role: str, semantic_state: str = "derived") -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": semantic_state,
        "bytes": st.st_size,
        "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def main() -> None:
    slopes = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    slopes.to_parquet(PARQUET_PATH, index=False)

    static = pd.read_parquet(STATIC_IN)
    fields = [
        "reach_id",
        "direction_adjudication_class",
        "direction_gate_pass",
        "direction_support_basis",
        "direction_confidence",
        "recommended_model_direction",
        "slope_value_changed_by_direction_audit",
    ]
    for col in fields[1:]:
        if col in static.columns:
            static = static.drop(columns=col)
    static = static.merge(slopes[fields], on="reach_id", how="left", validate="one_to_one")
    static.to_parquet(STATIC_OUT, index=False)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    added_sources = [
        record(RUN_DIR / "README.md", "run_documentation", "reported"),
        record(RUN_DIR / "scripts" / "audit_unresolved_reach_direction.py", "executed_script", "reported"),
        record(RUN_DIR / "scripts" / "finalize_direction_products.py", "executed_script", "reported"),
        record(RUN_DIR / "scripts" / "validate_direction_audit.py", "executed_script", "reported"),
    ]
    added_products = [
        record(PARQUET_PATH, "direction_review_parquet"),
        record(STATIC_OUT, "direction_review_static_parquet"),
        record(RUN_DIR / "logs" / "runtime_incidents.md", "runtime_incident_log"),
    ]
    manifest["sources"] = list({
        rec["path"]: rec for rec in manifest["sources"] + added_sources
    }.values())
    manifest["products"] = list({
        rec["path"]: rec for rec in manifest["products"] + added_products
    }.values())
    manifest["parquet_runtime"] = "conda sparrow"
    manifest["parquet_finalized_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(manifest["sources"] + manifest["products"]).to_csv(
        MANIFEST_CSV, index=False, encoding="utf-8-sig"
    )
    print(f"finalized rows={len(slopes)} static_rows={len(static)}")


if __name__ == "__main__":
    main()
