from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
PARENT = ROOT / "5_Test" / "20260729_12"
CSV = RUN_DIR / "inputs" / "reach_slope_direction_final.csv"
PARQUET = RUN_DIR / "inputs" / "reach_slope_direction_final.parquet"
STATIC_IN = PARENT / "inputs" / "reach_static_direction_adjudicated.parquet"
STATIC_OUT = RUN_DIR / "inputs" / "reach_static_direction_final.parquet"
MANIFEST = RUN_DIR / "inputs_manifest" / "provenance_manifest.json"
MANIFEST_CSV = RUN_DIR / "inputs_manifest" / "provenance_manifest.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path, role: str, state: str = "derived") -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": state,
        "bytes": st.st_size,
        "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def main() -> None:
    current = pd.read_csv(CSV, encoding="utf-8-sig")
    current.to_parquet(PARQUET, index=False)
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
    static = static.drop(columns=[c for c in fields[1:] if c in static.columns])
    static = static.merge(current[fields], on="reach_id", how="left", validate="one_to_one")
    static.to_parquet(STATIC_OUT, index=False)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    added_sources = [
        record(RUN_DIR / "README.md", "run_documentation", "reported"),
        record(RUN_DIR / "scripts" / "adjudicate_xunjiang_named_confluence.py", "executed_script", "reported"),
        record(RUN_DIR / "scripts" / "finalize_xunjiang_direction.py", "executed_script", "reported"),
        record(RUN_DIR / "scripts" / "validate_xunjiang_direction.py", "executed_script", "reported"),
    ]
    added_products = [
        record(PARQUET, "final_direction_parquet"),
        record(STATIC_OUT, "final_direction_static_parquet"),
        record(RUN_DIR / "reports" / "xunjiang_direction_gate" / "gate.json", "direction_gate"),
        record(RUN_DIR / "reports" / "xunjiang_direction_gate" / "technical_report.md", "technical_report"),
        record(RUN_DIR / "reports" / "xunjiang_direction_gate" / "external_web_cross_validation.md", "external_cross_validation"),
    ]
    manifest["sources"] = list({r["path"]: r for r in manifest["sources"] + added_sources}.values())
    manifest["products"] = list({r["path"]: r for r in manifest["products"] + added_products}.values())
    manifest["parquet_runtime"] = "conda sparrow"
    manifest["parquet_finalized_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(manifest["sources"] + manifest["products"]).to_csv(
        MANIFEST_CSV, index=False, encoding="utf-8-sig"
    )
    print(f"finalized rows={len(current)} static_rows={len(static)}")


if __name__ == "__main__":
    main()
