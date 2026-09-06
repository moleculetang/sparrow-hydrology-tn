from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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


def main() -> None:
    gate = json.loads((REPORTS / "gate.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (MANIFESTS / "provenance_manifest.json").read_text(encoding="utf-8")
    )
    slope = pd.read_parquet(INPUTS / "reach_slope.parquet")
    static = pd.read_parquet(INPUTS / "reach_static_with_slope.parquet")
    parent = pd.read_parquet(PARENT_STATIC)

    hash_failures = []
    for record in provenance["sources"] + provenance["products"]:
        path = Path(record["path"])
        if not path.exists() or sha256(path) != record["sha256"]:
            hash_failures.append(str(path))

    unchanged_columns = [
        column
        for column in parent.columns
        if column not in {"slope_m_m", "slope_status"}
    ]
    parent_core = parent.sort_values("reach_id")[unchanged_columns].reset_index(
        drop=True
    )
    static_core = static.sort_values("reach_id")[unchanged_columns].reset_index(
        drop=True
    )
    checks = {
        "source_and_product_hashes_match": len(hash_failures) == 0,
        "slope_unique_230_reaches": (
            len(slope) == 230 and slope["reach_id"].nunique() == 230
        ),
        "static_unique_230_reaches": (
            len(static) == 230 and static["reach_id"].nunique() == 230
        ),
        "slopes_finite_positive": bool(
            np.isfinite(slope["slope_m_m"]).all()
            and slope["slope_m_m"].gt(0).all()
        ),
        "slope_status_present": slope["slope_source_status"].notna().all(),
        "parent_static_unchanged_except_slope": parent_core.equals(static_core),
        "static_slope_matches_slope_product": np.allclose(
            static.sort_values("reach_id")["slope_m_m"],
            slope.sort_values("reach_id")["slope_m_m"],
        ),
        "decision_consistent_with_pass": (
            (
                gate["passed"] is True
                and gate["authorized_next_action"]
                == "BUILD_Q78_NAT_CONSERVATION_CORE"
            )
            or (
                gate["passed"] is False
                and gate["authorized_next_action"]
                == "REVIEW_REACH_SLOPE_ANOMALIES"
            )
        ),
        "all_gate_checks_boolean": all(
            isinstance(value, bool) for value in gate["checks"].values()
        ),
        "burned_dem_not_used": gate["method"]["dem_is_unburned"] is True,
        "absolute_value_not_used": (
            gate["method"]["absolute_value_used_to_hide_direction_conflict"]
            is False
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(all(checks.values()))
    payload = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": "conda sparrow environment executable",
        "checks": checks,
        "hash_failures": hash_failures,
        "scientific_gate_passed": bool(gate["passed"]),
        "validation_passed": passed,
    }
    (REPORTS / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"REACH_SLOPE_VALIDATION passed={passed} "
        f"scientific_gate={gate['passed']} checks={len(checks)}"
    )
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit(f"Reach slope validation failed: {failed}; {hash_failures}")


if __name__ == "__main__":
    main()
