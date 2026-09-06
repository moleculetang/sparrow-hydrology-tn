from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent.parent
INPUTS = RUN / "inputs"
MANIFESTS = RUN / "inputs_manifest"
REPORTS = RUN / "reports" / "slope_anomaly_review_gate"
PARENT_STATIC = (
    ROOT
    / "5_Test"
    / "20260729_10"
    / "inputs"
    / "reach_static_with_slope.parquet"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "role": role,
        "semantic_state": "derived",
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path),
    }


def main() -> None:
    reviewed = pd.read_csv(
        INPUTS / "reach_slope_reviewed.csv", encoding="utf-8-sig"
    )
    parent = pd.read_parquet(PARENT_STATIC)
    slope_path = INPUTS / "reach_slope_reviewed.parquet"
    reviewed.to_parquet(slope_path, index=False)
    static = parent.drop(
        columns=[
            "slope_m_m",
            "slope_status",
            "slope_source_status",
            "direct_dem_estimate",
        ],
        errors="ignore",
    ).merge(
        reviewed[
            [
                "reach_id",
                "slope_m_m",
                "slope_source_status",
                "direct_dem_estimate",
                "review_class",
                "direction_support_pass",
                "review_confidence",
            ]
        ],
        on="reach_id",
        how="left",
        validate="one_to_one",
    )
    static["slope_status"] = static["slope_source_status"]
    static_path = INPUTS / "reach_static_reviewed_slope.parquet"
    static.to_parquet(static_path, index=False)

    provenance_path = MANIFESTS / "provenance_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    new = [record(slope_path, "reviewed_slope_product"), record(static_path, "reviewed_static_product")]
    provenance["products"].extend(new)
    provenance["parquet_runtime"] = "conda sparrow"
    provenance["parquet_finalized_utc"] = datetime.now(timezone.utc).isoformat()
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(provenance["sources"] + provenance["products"]).to_csv(
        MANIFESTS / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest_path = REPORTS / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parquet_finalization_pending"] = False
    manifest["parquet_runtime"] = "conda sparrow"
    manifest["parquet_products"] = new
    manifest["provenance_sha256"] = sha256(provenance_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"SLOPE_REVIEW_FINALIZE reaches={len(static)}")


if __name__ == "__main__":
    main()
