from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
COMPONENT = RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py"
TOPOLOGY = RUN / "inputs" / "topology" / "topology_edges.csv"
FOLDS = [
    {"fold_id": "fit_2006_2011_eval_2012_2013", "train_end": 2011, "eval_start": 2012, "eval_end": 2013},
    {"fold_id": "fit_2006_2013_eval_2014_2015", "train_end": 2013, "eval_start": 2014, "eval_end": 2015},
    {"fold_id": "fit_2006_2015_eval_2016_2018", "train_end": 2015, "eval_start": 2016, "eval_end": 2018},
]
FIXED_HYPERPARAMETERS = {
    "rho": 0.70, "wm": 480.0, "et_gamma": 0.75, "sas_rho": 0.93,
    "young_k": 1.5, "storage_scale": 720.0, "prod_capacity": 240.0,
    "runoff_gamma": 2.5, "quick_rho": 0.25, "base_rho": 0.85,
    "base_release": 0.10, "fixed_sigma": 3.0, "production_sigma": 1.5,
    "group_sigma": 1.5, "multistore_sigma": 0.30, "hysteresis_sigma": 3.0,
    "station_sigma": 1.0, "slope_sigma": 0.15, "regime_slope_sigma": 0.25,
    "anomaly_weight": 0.0, "flow_contrast_weight": 1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_component():
    spec = importlib.util.spec_from_file_location("spatial_q72_component", COMPONENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COMPONENT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fold(scenario: str, input_path: Path, output: Path, fold: dict[str, object]) -> dict[str, object]:
    module = load_component()
    fold_dir = output / "blocked_folds" / str(fold["fold_id"])
    report_dir = fold_dir / "reports"
    figure_dir = fold_dir / "figure"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    module.REPORT_DIR = report_dir
    module.FIG_DIR = figure_dir
    module.INPUT_PATH = input_path
    module.TOPOLOGY_PATH = TOPOLOGY
    module.CAL_END_YEAR = int(fold["train_end"])
    module.INNER_TRAIN_END_YEAR = int(fold["train_end"])
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    module.write_readme = lambda *_args, **_kwargs: None

    def fixed_choice(_frame):
        row = {
            **FIXED_HYPERPARAMETERS,
            "selection": "fixed_from_20260620_44",
            "state_calendar_mode": "full_forcing",
            "forcing_semantics_mode": "prescribed_aet_balance",
            "mass_accounting_mode": "explicit_upstream_volume",
        }
        return row, pd.DataFrame([row])

    module.choose_hyperparameters = fixed_choice
    started = datetime.now()
    module.main()
    ended = datetime.now()
    path = report_dir / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv"
    prediction = pd.read_csv(path, encoding="utf-8-sig")
    evaluation = prediction[prediction["year"].between(int(fold["eval_start"]), int(fold["eval_end"]))].copy()
    evaluation["fold_id"] = str(fold["fold_id"])
    evaluation["scenario_id"] = scenario
    evaluation["forcing_semantics_mode"] = "prescribed_aet_balance"
    evaluation["mass_accounting_mode"] = "explicit_upstream_volume"
    evaluation.to_csv(fold_dir / "evaluation_predictions.csv", index=False, encoding="utf-8-sig")
    return {
        **fold, "scenario_id": scenario, "evaluation_rows": len(evaluation),
        "stations": int(evaluation["q_site"].nunique()),
        "elapsed_seconds": round((ended - started).total_seconds(), 3),
        "input_sha256": sha256(input_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=["S0", "S1"])
    args = parser.parse_args()
    scenario = args.scenario
    input_path = RUN / "inputs" / "scenarios" / f"{scenario}_indata.parquet"
    output = RUN / "outputs" / scenario
    output.mkdir(parents=True, exist_ok=True)
    rows = [run_fold(scenario, input_path, output, fold) for fold in FOLDS]
    pd.DataFrame(rows).to_csv(output / "blocked_fold_manifest.csv", index=False, encoding="utf-8-sig")
    parts = [
        pd.read_csv(output / "blocked_folds" / fold["fold_id"] / "evaluation_predictions.csv", encoding="utf-8-sig")
        for fold in FOLDS
    ]
    oof = pd.concat(parts, ignore_index=True)
    if len(oof) != 8738 or oof["fold_id"].nunique() != 3:
        raise RuntimeError(f"OOF gate failed for {scenario}: {len(oof)}")
    path = output / "q72_three_fold_oof_predictions.parquet"
    oof.to_parquet(path, index=False)
    payload = {
        "runtime": RUNTIME, "scenario": scenario, "input": str(input_path),
        "input_sha256": sha256(input_path), "oof_sha256": sha256(path), "folds": rows,
    }
    (output / "run_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
