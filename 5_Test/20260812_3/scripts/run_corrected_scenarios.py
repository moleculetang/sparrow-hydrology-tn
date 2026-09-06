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
COMPONENT = RUN / "scripts" / "components" / "corrected_dynamic_q72_component.py"
INPUT = RUN / "inputs" / "scenarios" / "B0_indata.parquet"
TOPOLOGY = RUN / "inputs" / "topology" / "topology_edges.csv"
FOLDS = [
    {"fold_id": "fit_2006_2011_eval_2012_2013", "train_end": 2011, "eval_start": 2012, "eval_end": 2013},
    {"fold_id": "fit_2006_2013_eval_2014_2015", "train_end": 2013, "eval_start": 2014, "eval_end": 2015},
    {"fold_id": "fit_2006_2015_eval_2016_2018", "train_end": 2015, "eval_start": 2016, "eval_end": 2018},
]
FIXED = {
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
    spec = importlib.util.spec_from_file_location("corrected_q72", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def run_fold(scenario: str, beta_w: float, fold: dict[str, object]) -> dict[str, object]:
    module = load_component()
    fold_dir = RUN / "outputs" / scenario / "blocked_folds" / str(fold["fold_id"])
    report_dir = fold_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "figure").mkdir(parents=True, exist_ok=True)
    module.REPORT_DIR = report_dir
    module.FIG_DIR = fold_dir / "figure"
    module.INPUT_PATH = INPUT
    module.TOPOLOGY_PATH = TOPOLOGY
    module.CAL_END_YEAR = int(fold["train_end"])
    module.INNER_TRAIN_END_YEAR = int(fold["train_end"])
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    module.DYNAMIC_BETA_W = float(beta_w)
    module.SCENARIO_ID = scenario
    module.write_readme = lambda *_args, **_kwargs: None

    def fixed_choice(_frame):
        row = {**FIXED, "selection": "fixed_from_20260620_44", "scenario": scenario, "beta_w": beta_w}
        return row, pd.DataFrame([row])

    module.choose_hyperparameters = fixed_choice
    started = datetime.now()
    module.main()
    ended = datetime.now()
    source = report_dir / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.parquet"
    prediction = pd.read_parquet(source)
    csv_source = report_dir / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv"
    csv_prediction = pd.read_csv(csv_source, encoding="utf-8-sig")
    if len(csv_prediction) != len(prediction):
        raise RuntimeError(
            f"Prediction persistence mismatch for {fold['fold_id']}: "
            f"parquet_rows={len(prediction)}, csv_rows={len(csv_prediction)}"
        )
    # A fold fit is allowed to use only observations through train_end, but
    # prediction must retain the complete observed panel.  In particular,
    # stations first observed after train_end receive zero (prior-mean) random
    # effects; they must never disappear from OOF evaluation.
    input_panel = pd.read_parquet(INPUT, columns=["comid", "q_site", "year", "month", "Q_obsv_cfs"])
    expected = input_panel.loc[
        input_panel["Q_obsv_cfs"].notna() & input_panel["Q_obsv_cfs"].gt(0),
        ["comid", "q_site", "year", "month", "Q_obsv_cfs"],
    ].copy()
    expected["q_site"] = expected["q_site"].astype(str)
    prediction["q_site"] = prediction["q_site"].astype(str)
    key = ["comid", "q_site", "year", "month"]
    if len(prediction) != len(expected) or prediction[key].duplicated().any():
        raise RuntimeError(
            f"Complete-observation prediction gate failed for {fold['fold_id']}: "
            f"prediction_rows={len(prediction)}, expected_rows={len(expected)}, "
            f"duplicate_keys={int(prediction[key].duplicated().sum())}"
        )
    key_a = prediction[key].sort_values(key).reset_index(drop=True)
    key_b = expected[key].sort_values(key).reset_index(drop=True)
    if not key_a.equals(key_b):
        missing = key_b.merge(key_a, on=key, how="left", indicator=True)["_merge"].eq("left_only").sum()
        extra = key_a.merge(key_b, on=key, how="left", indicator=True)["_merge"].eq("left_only").sum()
        raise RuntimeError(
            f"Complete-observation key gate failed for {fold['fold_id']}: "
            f"missing={int(missing)}, extra={int(extra)}"
        )
    evaluation = prediction[prediction.year.between(int(fold["eval_start"]), int(fold["eval_end"]))].copy()
    expected_evaluation = expected[
        expected.year.between(int(fold["eval_start"]), int(fold["eval_end"]))
    ].copy()
    if len(evaluation) != len(expected_evaluation):
        raise RuntimeError(
            f"Fold evaluation row gate failed for {fold['fold_id']}: "
            f"evaluation_rows={len(evaluation)}, expected_rows={len(expected_evaluation)}"
        )
    evaluation["fold_id"] = str(fold["fold_id"])
    evaluation["scenario_id"] = scenario
    evaluation["beta_w"] = float(beta_w)
    evaluation["accounting_contract"] = "literal_rho_incarea_topology_single_outlet_quick"
    evaluation_path = fold_dir / "evaluation_predictions.parquet"
    evaluation.to_parquet(evaluation_path, index=False)
    evaluation[[
        "comid", "q_site", "year", "month", "actual", "predict", "fold_id",
        "scenario_id", "beta_w", "accounting_contract",
    ]].to_csv(fold_dir / "evaluation_predictions.csv", index=False, encoding="utf-8-sig")
    return {**fold, "scenario": scenario, "beta_w": beta_w, "rows": len(evaluation), "elapsed_seconds": (ended-started).total_seconds()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=["I0", "I1"])
    parser.add_argument("--beta-w", type=float, required=True)
    args = parser.parse_args()
    if args.scenario == "I0" and abs(args.beta_w) > 1e-15:
        raise ValueError("I0 is the exact beta_w=0 fixed-0.35 control")
    rows = [run_fold(args.scenario, args.beta_w, fold) for fold in FOLDS]
    output = RUN / "outputs" / args.scenario
    pd.DataFrame(rows).to_csv(output / "blocked_fold_manifest.csv", index=False, encoding="utf-8-sig")
    parts = [pd.read_parquet(output / "blocked_folds" / fold["fold_id"] / "evaluation_predictions.parquet") for fold in FOLDS]
    oof = pd.concat(parts, ignore_index=True)
    if len(oof) != 8738 or oof.fold_id.nunique() != 3:
        raise RuntimeError(f"OOF gate failed: {len(oof)}")
    path = output / "q72_three_fold_oof_predictions.parquet"
    oof.to_parquet(path, index=False)
    payload = {"runtime": RUNTIME, "scenario": args.scenario, "beta_w": args.beta_w, "input_sha256": sha256(INPUT), "oof_sha256": sha256(path), "folds": rows}
    (output / "run_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
