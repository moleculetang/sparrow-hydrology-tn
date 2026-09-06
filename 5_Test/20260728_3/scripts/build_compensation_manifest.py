from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "5_Test"
OUT = RUN / "inputs_manifest"
RUN_IDS = ["20260727_6", "20260727_13", "20260727_15"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    relative_paths = [
        "inputs/indata.parquet",
        "scripts/experiment_settings.py",
        "scripts/components/fit_monthly_bayes_seasonal_hysteresis.py",
        "scripts/components/run_et_role_experiment.py",
        "scripts/components/run_slowflow_redundancy_experiment.py",
        "scripts/components/run_storage_timing_experiment.py",
        "scripts/components/run_hysteresis_gate_experiment.py",
        "reports/intermediate/base_regression/smearing_predictions_long.csv",
        "reports/workflow/run_manifest.csv",
    ]
    for run_id in RUN_IDS:
        source = TEST_ROOT / run_id
        for relative in relative_paths:
            path = source / relative
            required = not relative.endswith("experiment_settings.py") or (
                run_id != "20260727_6"
            )
            exists = path.exists()
            if required and not exists:
                raise FileNotFoundError(path)
            if exists:
                rows.append(
                    {
                        "run_id": run_id,
                        "relative_path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "compensation_source_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    input_hashes = (
        frame.loc[
            frame["relative_path"].eq("inputs/indata.parquet"),
            ["run_id", "sha256"],
        ]
        .set_index("run_id")["sha256"]
        .to_dict()
    )
    payload = {
        "run_id": RUN.name,
        "reference_run": "20260727_6",
        "candidate_runs": ["20260727_13", "20260727_15"],
        "files": int(len(frame)),
        "input_hashes": input_hashes,
        "all_input_hashes_identical": len(set(input_hashes.values())) == 1,
    }
    (OUT / "compensation_source_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["all_input_hashes_identical"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
