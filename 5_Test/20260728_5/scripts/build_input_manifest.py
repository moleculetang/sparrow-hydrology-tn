from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REFERENCE = RUN.parent / "20260727_6"
PREDECESSOR = RUN.parent / "20260728_4"
OUT = RUN / "inputs_manifest"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    first_fold = (
        REFERENCE
        / "reports"
        / "station_screening"
        / "blocked_folds"
        / "fit_2006_2011_eval_2012_2013"
        / "reports"
        / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv"
    )
    paths_and_roles = [
        (
            PREDECESSOR
            / "reports"
            / "q72_q78_complementarity"
            / "oof_predictions_2012_2018.csv",
            "strict_oof_label_source",
        ),
        (PREDECESSOR / "reports" / "gate.json", "predecessor_integrity_gate"),
        (
            REFERENCE
            / "reports"
            / "intermediate"
            / "mass_reach_class"
            / "reach_month_forcing_panel.csv",
            "prediction_time_reach_and_climate_attributes",
        ),
        (first_fold, "common_2006_2011_gauged_signature_source"),
        (
            REFERENCE
            / "reports"
            / "input_preprocessing"
            / "same_reach_station_reliability.csv",
            "independent_data_quality_context",
        ),
        (
            REFERENCE
            / "reports"
            / "input_preprocessing"
            / "station_reach_match.csv",
            "location_sensitivity_source",
        ),
        (
            REFERENCE
            / "reports"
            / "main_model"
            / "station_diagnostic_labels.csv",
            "reservoir_scope_source",
        ),
        (
            REFERENCE
            / "inputs"
            / "source_metadata"
            / "station_screening_policy.csv",
            "fixed_station_policy_source",
        ),
    ]
    missing = [str(path) for path, _ in paths_and_roles if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    rows = []
    for path, role in paths_and_roles:
        stat = path.stat()
        rows.append(
            {
                "run_id": RUN.name,
                "logical_parent_run": REFERENCE.name,
                "diagnostic_predecessor": PREDECESSOR.name,
                "path": str(path),
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime)
                .astimezone()
                .isoformat(),
                "sha256": sha256(path),
                "role": role,
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "source_file_manifest.csv", index=False, encoding="utf-8-sig")
    payload = {
        "run_id": RUN.name,
        "logical_parent_run": REFERENCE.name,
        "diagnostic_predecessor": PREDECESSOR.name,
        "files": len(rows),
        "total_bytes": int(frame["bytes"].sum()),
        "all_present": True,
        "records": rows,
    }
    (OUT / "source_file_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"files": len(rows), "all_present": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
