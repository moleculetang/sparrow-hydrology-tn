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
REPORTS = RUN / "reports" / "slope_anomaly_review_gate"
PARENT_SLOPE = (
    ROOT / "5_Test" / "20260729_10" / "inputs" / "reach_slope.parquet"
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
    reviewed = pd.read_parquet(INPUTS / "reach_slope_reviewed.parquet")
    static = pd.read_parquet(INPUTS / "reach_static_reviewed_slope.parquet")
    parent = pd.read_parquet(PARENT_SLOPE)
    hashes = []
    for item in provenance["sources"] + provenance["products"]:
        path = Path(item["path"])
        if not path.exists() or sha256(path) != item["sha256"]:
            hashes.append(str(path))
    checks = {
        "hashes_match": len(hashes) == 0,
        "reviewed_230_unique": len(reviewed) == 230
        and reviewed["reach_id"].nunique() == 230,
        "static_230_unique": len(static) == 230
        and static["reach_id"].nunique() == 230,
        "slope_values_unchanged": np.allclose(
            reviewed.sort_values("reach_id")["slope_m_m"],
            parent.sort_values("reach_id")["slope_m_m"],
        ),
        "all_slopes_positive_finite": bool(
            np.isfinite(reviewed["slope_m_m"]).all()
            and reviewed["slope_m_m"].gt(0).all()
        ),
        "all_reaches_have_review_metadata": reviewed[
            ["review_class", "direction_support_pass", "review_confidence"]
        ]
        .notna()
        .all()
        .all(),
        "scientific_decision_consistent": (
            (
                gate["passed"] is True
                and gate["authorized_next_action"]
                == "BUILD_Q78_NAT_CONSERVATION_CORE"
            )
            or (
                gate["passed"] is False
                and gate["authorized_next_action"]
                == "REPAIR_UNRESOLVED_REACH_DIRECTION"
            )
        ),
        "all_gate_checks_boolean": all(
            isinstance(value, bool) for value in gate["checks"].values()
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = bool(all(checks.values()))
    payload = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": "conda sparrow environment executable",
        "checks": checks,
        "hash_failures": hashes,
        "scientific_gate_passed": bool(gate["passed"]),
        "validation_passed": passed,
    }
    (REPORTS / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"SLOPE_ANOMALY_VALIDATION passed={passed} "
        f"scientific_gate={gate['passed']} checks={len(checks)}"
    )
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit(f"Validation failed: {failed}; hashes={hashes}")


if __name__ == "__main__":
    main()
