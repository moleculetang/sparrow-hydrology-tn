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
REPORTS = RUN / "reports" / "reach_slope_quality_gate"
PARENT_STATIC = (
    ROOT / "5_Test" / "20260729_9" / "inputs" / "reach_static.parquet"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str, semantic_state: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "role": role,
        "semantic_state": semantic_state,
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path),
    }


def main() -> None:
    gate_path = REPORTS / "gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    slope = pd.read_csv(INPUTS / "reach_slope.csv", encoding="utf-8-sig")
    parent = pd.read_parquet(PARENT_STATIC)
    if len(slope) != 230 or slope["reach_id"].nunique() != 230:
        raise AssertionError("Slope CSV does not contain exactly 230 reaches")

    slope_parquet = INPUTS / "reach_slope.parquet"
    slope.to_parquet(slope_parquet, index=False)
    static = parent.drop(
        columns=["slope_m_m", "slope_status"], errors="ignore"
    ).merge(
        slope[
            [
                "reach_id",
                "slope_m_m",
                "slope_source_status",
                "direct_dem_estimate",
            ]
        ],
        on="reach_id",
        how="left",
        validate="one_to_one",
    )
    static["slope_status"] = static["slope_source_status"]
    static_path = INPUTS / "reach_static_with_slope.parquet"
    static.to_parquet(static_path, index=False)

    provenance_path = MANIFESTS / "provenance_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    new_products = [
        file_record(slope_parquet, "slope_product", "derived"),
        file_record(static_path, "shared_static_product", "derived"),
    ]
    existing_paths = {record["path"] for record in new_products}
    provenance["products"] = [
        record
        for record in provenance["products"]
        if record["path"] not in existing_paths
    ] + new_products
    provenance["parquet_finalized_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["parquet_runtime"] = "conda sparrow"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(provenance["sources"] + provenance["products"]).to_csv(
        MANIFESTS / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    run_manifest_path = REPORTS / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["parquet_finalization_pending"] = False
    run_manifest["parquet_runtime"] = "conda sparrow"
    run_manifest["parquet_finalized_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["provenance_sha256"] = sha256(provenance_path)
    run_manifest["gate_sha256"] = sha256(gate_path)
    run_manifest["parquet_products"] = new_products
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "REACH_SLOPE_FINALIZE "
        f"scientific_gate={gate['passed']} reaches={len(static)}"
    )


if __name__ == "__main__":
    main()
