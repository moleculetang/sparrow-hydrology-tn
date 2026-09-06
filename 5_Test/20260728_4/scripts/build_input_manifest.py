from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REFERENCE = RUN.parent / "20260727_6"
OUT = RUN / "inputs_manifest"

FOLDS = [
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = [
        REFERENCE
        / "reports"
        / "intermediate"
        / "mass_reach_class"
        / "reach_month_forcing_panel.csv",
        REFERENCE
        / "reports"
        / "intermediate"
        / "mass_reach_class"
        / "routed_class_local_source_basis.csv",
        REFERENCE / "reports" / "main_model" / "selected_reach_class_alpha.csv",
        REFERENCE / "reports" / "main_model" / "station_diagnostic_labels.csv",
        REFERENCE / "scripts" / "fit_global_mass_model.py",
        REFERENCE / "scripts" / "fit_reach_class_mass_model.py",
    ]
    for fold in FOLDS:
        fold_root = (
            REFERENCE
            / "reports"
            / "station_screening"
            / "blocked_folds"
            / fold
        )
        paths.extend(
            [
                fold_root / "evaluation_predictions.csv",
                fold_root
                / "reports"
                / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv",
            ]
        )

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    rows = []
    for path in paths:
        stat = path.stat()
        rows.append(
            {
                "run_id": RUN.name,
                "reference_run": REFERENCE.name,
                "path": str(path),
                "relative_to_reference": str(path.relative_to(REFERENCE)),
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                "sha256": sha256(path),
                "role": (
                    "q72_blocked_oof"
                    if path.name == "evaluation_predictions.csv"
                    else "q72_fold_training_observations"
                    if path.name.startswith("monthly_bayes")
                    else "observation_independent_mass_basis"
                    if path.name in {
                        "reach_month_forcing_panel.csv",
                        "routed_class_local_source_basis.csv",
                    }
                    else "policy_or_code_reference"
                ),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "source_file_manifest.csv", index=False, encoding="utf-8-sig")
    payload = {
        "run_id": RUN.name,
        "reference_run": REFERENCE.name,
        "files": len(rows),
        "total_bytes": int(frame["bytes"].sum()),
        "all_present": True,
        "sha256_unique_files": int(frame["sha256"].nunique()),
        "records": rows,
    }
    (OUT / "source_file_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "files": payload["files"],
                "total_bytes": payload["total_bytes"],
                "all_present": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
