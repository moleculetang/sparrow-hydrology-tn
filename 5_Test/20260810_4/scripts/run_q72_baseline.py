from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME_IDENTITY = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
COMPONENT = RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py"
SCREENING = RUN / "reports" / "q72_baseline"
INPUT_PATH = RUN / "inputs" / "source_snapshot" / "baseline_reference" / "inputs" / "indata.parquet"
TOPOLOGY_PATH = RUN / "inputs" / "source_snapshot" / "baseline_reference" / "inputs" / "topology" / "topology_edges.csv"
SCENARIO_ID = "S01"

FOLDS = [
    {"fold_id": "fit_2006_2011_eval_2012_2013", "train_end": 2011, "eval_start": 2012, "eval_end": 2013},
    {"fold_id": "fit_2006_2013_eval_2014_2015", "train_end": 2013, "eval_start": 2014, "eval_end": 2015},
    {"fold_id": "fit_2006_2015_eval_2016_2018", "train_end": 2015, "eval_start": 2016, "eval_end": 2018},
]

FIXED_HYPERPARAMETERS = {
    "rho": 0.70,
    "wm": 480.0,
    "et_gamma": 0.75,
    "sas_rho": 0.93,
    "young_k": 1.5,
    "storage_scale": 720.0,
    "prod_capacity": 240.0,
    "runoff_gamma": 2.5,
    "quick_rho": 0.25,
    "base_rho": 0.85,
    "base_release": 0.10,
    "fixed_sigma": 3.0,
    "production_sigma": 1.5,
    "group_sigma": 1.5,
    "multistore_sigma": 0.30,
    "hysteresis_sigma": 3.0,
    "station_sigma": 1.0,
    "slope_sigma": 0.15,
    "regime_slope_sigma": 0.25,
    "anomaly_weight": 0.0,
    "flow_contrast_weight": 1.0,
}

# Hydrologic states must advance on the complete forcing calendar.  Q-observed
# rows are still selected downstream by the component before design/fitting.
STATE_CALENDAR_MODE = "full_forcing"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_component():
    spec = importlib.util.spec_from_file_location("screening_q72", COMPONENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COMPONENT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fold(fold: dict[str, object]) -> dict[str, object]:
    module = load_component()
    fold_dir = SCREENING / "blocked_folds" / str(fold["fold_id"])
    report_dir = fold_dir / "reports"
    figure_dir = fold_dir / "figure"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    module.REPORT_DIR = report_dir
    module.FIG_DIR = figure_dir
    module.INPUT_PATH = INPUT_PATH
    module.TOPOLOGY_PATH = TOPOLOGY_PATH
    module.CAL_END_YEAR = int(fold["train_end"])
    module.INNER_TRAIN_END_YEAR = int(fold["train_end"])
    module.STATE_CALENDAR_MODE = STATE_CALENDAR_MODE
    # The component's legacy writer targets the run root and would overwrite
    # main-workflow provenance.  Blocked folds own only their fold directory.
    module.write_readme = lambda *_args, **_kwargs: None

    def fixed_choice(_frame):
        row = {
            **FIXED_HYPERPARAMETERS,
            "selection": "fixed_from_20260620_44",
            "state_calendar_mode": STATE_CALENDAR_MODE,
        }
        return row, pd.DataFrame([row])

    module.choose_hyperparameters = fixed_choice
    started = datetime.now()
    module.main()
    ended = datetime.now()
    prediction_path = report_dir / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv"
    prediction = pd.read_csv(prediction_path, encoding="utf-8-sig")
    evaluation = prediction[prediction["year"].between(int(fold["eval_start"]), int(fold["eval_end"]))].copy()
    evaluation["fold_id"] = str(fold["fold_id"])
    evaluation["train_end"] = int(fold["train_end"])
    evaluation["eval_start"] = int(fold["eval_start"])
    evaluation["eval_end"] = int(fold["eval_end"])
    evaluation["state_calendar_mode"] = STATE_CALENDAR_MODE
    evaluation.to_csv(fold_dir / "evaluation_predictions.csv", index=False, encoding="utf-8-sig")
    return {
        **fold,
        "returncode": 0,
        "evaluation_rows": int(len(evaluation)),
        "stations": int(evaluation["q_site"].nunique()),
        "elapsed_seconds": round((ended - started).total_seconds(), 3),
        "prediction_path": str(prediction_path),
        "state_calendar_mode": STATE_CALENDAR_MODE,
        "scenario_id": SCENARIO_ID,
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen Q72 blocked-fold scenario")
    parser.add_argument("--scenario-id", default=SCENARIO_ID)
    parser.add_argument("--state-calendar-mode", choices=("observed_only", "full_forcing"), default=STATE_CALENDAR_MODE)
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=SCREENING)
    return parser.parse_args()


def main() -> None:
    global SCENARIO_ID, STATE_CALENDAR_MODE, INPUT_PATH, SCREENING
    args = parse_args()
    SCENARIO_ID = str(args.scenario_id)
    STATE_CALENDAR_MODE = str(args.state_calendar_mode)
    INPUT_PATH = Path(args.input_path).resolve()
    SCREENING = Path(args.output_dir).resolve()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    SCREENING.mkdir(parents=True, exist_ok=True)
    rows = [run_fold(fold) for fold in FOLDS]
    experiment_path = RUN / "inputs" / "source_metadata" / "run_experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.exists() else {}
    pd.DataFrame(rows).to_csv(SCREENING / "blocked_fold_manifest.csv", index=False, encoding="utf-8-sig")
    (SCREENING / "blocked_fold_manifest.json").write_text(
        json.dumps(
            {
                "run_id": RUN.name,
                "scenario_id": SCENARIO_ID,
                "state_calendar_mode": STATE_CALENDAR_MODE,
                "input_path": str(INPUT_PATH),
                "input_sha256": sha256(INPUT_PATH),
                "parent_run": experiment.get("parent_run", ""),
                "folds": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    oof_parts = []
    for fold in FOLDS:
        path = SCREENING / "blocked_folds" / str(fold["fold_id"]) / "evaluation_predictions.csv"
        oof_parts.append(pd.read_csv(path, encoding="utf-8-sig"))
    oof = pd.concat(oof_parts, ignore_index=True)
    oof["scenario_id"] = SCENARIO_ID
    oof["state_calendar_mode"] = STATE_CALENDAR_MODE
    oof.to_parquet(SCREENING / "q72_three_fold_oof_predictions.parquet", index=False)
    log_lines = [
        f"# {RUN.name} Blocked-Fold Log",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for row in rows:
        log_lines.append(
            f"- {row['fold_id']}: returncode={row['returncode']}, "
            f"stations={row['stations']}, rows={row['evaluation_rows']}, "
            f"elapsed={row['elapsed_seconds']}s"
        )
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    (SCREENING / "blocked_fold_log.md").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    (SCREENING / "runtime_identity.json").write_text(
        json.dumps(RUNTIME_IDENTITY, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
