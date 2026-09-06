from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SERIES = RUN.parent
BASELINE = SERIES / "20260727_6"
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
        (
            SERIES / "20260728_6" / "reports" / "gate.json",
            "contract_admission_gate",
        ),
        (
            BASELINE
            / "scripts"
            / "components"
            / "fit_monthly_bayes_seasonal_hysteresis.py",
            "q72_component",
        ),
        (BASELINE / "scripts" / "fit_global_mass_model.py", "q78_base_code"),
        (
            BASELINE / "scripts" / "fit_reach_class_mass_model.py",
            "q78_reach_class_code",
        ),
        (
            BASELINE
            / "reports"
            / "intermediate"
            / "mass_reach_class"
            / "reach_month_forcing_panel.csv",
            "q78_forcing_and_static_reach_data",
        ),
        (
            BASELINE
            / "reports"
            / "intermediate"
            / "mass_reach_class"
            / "routed_class_local_source_basis.csv",
            "q78_routed_basis",
        ),
        (
            BASELINE / "reports" / "main_model" / "selected_reach_class_alpha.csv",
            "frozen_class_alpha",
        ),
        (
            SERIES
            / "20260728_4"
            / "reports"
            / "q72_q78_complementarity"
            / "oof_predictions_2012_2018.csv",
            "existing_strict_oof_outer_and_later_inner_blocks",
        ),
        (
            SERIES
            / "20260728_5"
            / "reports"
            / "low_flow_heterogeneity"
            / "station_heterogeneity_classification.csv",
            "fixed_zero_overrides_and_comparison_labels",
        ),
    ]
    missing = [str(path) for path, _ in sources if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing nested-gate inputs:\n" + "\n".join(missing))
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
