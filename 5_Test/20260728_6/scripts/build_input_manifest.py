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
            / "20260728_1"
            / "SPARROW_failure_analysis_and_revised_direction.md",
            "scientific_direction",
        ),
        (
            SERIES
            / "20260728_1"
            / "SPARROW_20260728_SERIES_EXECUTION_PLAN.md",
            "series_contract",
        ),
        (SERIES / "20260728_2" / "reports" / "gate.json", "baseline_reproduction_gate"),
        (
            SERIES / "20260728_2" / "reports" / "final_model_summary.csv",
            "accepted_baseline_metrics",
        ),
        (
            SERIES / "20260728_3" / "reports" / "gate.json",
            "structure_compensation_gate",
        ),
        (
            SERIES
            / "20260728_3"
            / "reports"
            / "structure_compensation"
            / "compensation_summary.csv",
            "structure_compensation_evidence",
        ),
        (
            SERIES
            / "20260728_3"
            / "inputs"
            / "source_metadata"
            / "run_experiment.json",
            "structure_phase_policy",
        ),
        (
            SERIES / "20260728_4" / "reports" / "gate.json",
            "complementarity_gate",
        ),
        (
            SERIES
            / "20260728_4"
            / "reports"
            / "q72_q78_complementarity"
            / "conditional_fusion_eligibility.csv",
            "complementarity_evidence",
        ),
        (
            SERIES
            / "20260728_4"
            / "inputs"
            / "source_metadata"
            / "run_experiment.json",
            "complementarity_phase_policy",
        ),
        (
            SERIES / "20260728_5" / "reports" / "gate.json",
            "heterogeneity_gate",
        ),
        (
            SERIES
            / "20260728_5"
            / "reports"
            / "low_flow_heterogeneity"
            / "scientific_gate.json",
            "heterogeneity_scientific_evidence",
        ),
        (
            SERIES
            / "20260728_5"
            / "reports"
            / "low_flow_heterogeneity"
            / "station_heterogeneity_classification.csv",
            "station_target_reference",
        ),
        (
            SERIES
            / "20260728_5"
            / "reports"
            / "low_flow_heterogeneity"
            / "separability_model_summary.csv",
            "gate_separability_evidence",
        ),
        (
            SERIES
            / "20260728_5"
            / "inputs"
            / "source_metadata"
            / "run_experiment.json",
            "heterogeneity_phase_policy",
        ),
    ]
    missing = [str(path) for path, _ in sources if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comprehensive-gate inputs:\n" + "\n".join(missing))
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
