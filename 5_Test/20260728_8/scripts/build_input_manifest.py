from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SERIES = RUN.parent
OUT = RUN / "inputs_manifest"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    sources = [
        (
            SERIES
            / "20260728_6"
            / "reports"
            / "comprehensive_gate"
            / "candidate_contract.json",
            "candidate_contract",
        ),
        (SERIES / "20260728_7" / "reports" / "gate.json", "nested_gate_admission"),
        (
            SERIES
            / "20260728_7"
            / "reports"
            / "nested_oof_gate"
            / "outer_fold_station_gate_assignments.csv",
            "frozen_outer_station_gates",
        ),
        (
            SERIES
            / "20260728_7"
            / "reports"
            / "nested_oof_gate"
            / "nested_block_oof_predictions.csv",
            "nested_oof_amplitude_selection",
        ),
        (
            SERIES
            / "20260728_7"
            / "reports"
            / "nested_oof_gate"
            / "outer_fold_gate_summary.csv",
            "nested_gate_scientific_reference",
        ),
        (
            SERIES
            / "20260728_4"
            / "reports"
            / "q72_q78_complementarity"
            / "oof_predictions_2012_2018.csv",
            "strict_outer_oof_baseline",
        ),
        (
            SERIES
            / "20260728_5"
            / "reports"
            / "low_flow_heterogeneity"
            / "station_heterogeneity_classification.csv",
            "policy_and_reporting_context",
        ),
    ]
    missing = [str(path) for path, _ in sources if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing residual-pilot inputs:\n" + "\n".join(missing))
    rows = []
    for path, role in sources:
        stat = path.stat()
        rows.append(
            {
                "run_id": RUN.name,
                "path": str(path),
                "role": role,
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime)
                .astimezone()
                .isoformat(),
                "sha256": sha256(path),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "source_file_manifest.csv", index=False, encoding="utf-8-sig")
    payload = {
        "run_id": RUN.name,
        "files": len(rows),
        "all_present": True,
        "total_bytes": int(frame["bytes"].sum()),
        "records": rows,
    }
    (OUT / "source_file_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"files": len(rows), "all_present": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
