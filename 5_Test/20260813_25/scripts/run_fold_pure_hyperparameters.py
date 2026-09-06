from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "scripts" / "components" / "q72_fold_pure_component.py"
INPUT = ROOT / "inputs" / "A1_indata.parquet"
TOPOLOGY = ROOT / "inputs" / "topology" / "topology_edges.csv"
REFERENCE_I0 = ROOT / "inputs" / "reference" / "B1_oof.parquet"
FOLDS = [
    {"fold_id": "fit_2006_2011_eval_2012_2013", "train_end": 2011, "inner_train_end": 2009, "eval_start": 2012, "eval_end": 2013},
    {"fold_id": "fit_2006_2013_eval_2014_2015", "train_end": 2013, "inner_train_end": 2011, "eval_start": 2014, "eval_end": 2015},
    {"fold_id": "fit_2006_2015_eval_2016_2018", "train_end": 2015, "inner_train_end": 2013, "eval_start": 2016, "eval_end": 2018},
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
SCENARIOS = {
    "P0": {"scale": 1.0, "semantics": "exact_20260813_1_B1_control", "repair": True, "fold_pure_hysteresis": False},
    "P1": {"scale": 1.0, "semantics": "fold_pure_hysteresis_selection", "repair": True, "fold_pure_hysteresis": True},
}
KEY = ["comid", "q_site", "year", "month", "fold_id"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_component(name: str):
    spec = importlib.util.spec_from_file_location(f"q72_scale_{name}", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def configure(module, scenario: str, fold: dict[str, object], report_dir: Path, figure_dir: Path) -> None:
    contract = SCENARIOS[scenario]
    module.REPORT_DIR = report_dir
    module.FIG_DIR = figure_dir
    module.INPUT_PATH = INPUT
    module.TOPOLOGY_PATH = TOPOLOGY
    module.CAL_END_YEAR = int(fold["train_end"])
    module.INNER_TRAIN_END_YEAR = int(fold["inner_train_end"])
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    module.DYNAMIC_BETA_W = 0.0
    module.configure_engineering_repair(bool(contract["repair"]))
    module.set_et_feature_block_mode("full")
    module.NETWORK_INPUT_SCALE = float(contract["scale"])
    module.NETWORK_INPUT_SEMANTICS = str(contract["semantics"])
    module.SCENARIO_ID = scenario
    module.write_readme = lambda *_args, **_kwargs: None

    original_choice = module.choose_hyperparameters

    def fixed_choice(_frame):
        row = {
            **FIXED,
            "selection": "fixed_from_20260620_44",
            "scenario": scenario,
            "network_input_scale": float(contract["scale"]),
        }
        return row, pd.DataFrame([row])

    module.choose_hyperparameters = original_choice if bool(contract["fold_pure_hysteresis"]) else fixed_choice


def audit_fold_features(module, scenario: str, fold: dict[str, object], fold_dir: Path) -> dict[str, object]:
    forcing = module.load_forcing_panel()
    featured = module.add_hydrologic_features(
        forcing,
        **{k: FIXED[k] for k in [
            "rho", "wm", "et_gamma", "sas_rho", "young_k", "storage_scale",
            "prod_capacity", "runoff_gamma", "quick_rho", "base_rho", "base_release",
        ]},
    )
    scale = float(SCENARIOS[scenario]["scale"])
    local = featured["local_positive_input_equivalent_cfs"].to_numpy(float)
    upstream = featured["upstream_positive_input_equivalent_cfs"].to_numpy(float)
    qcalc = featured["Q_calc_cfs"].to_numpy(float)
    explicit = featured["explicit_upstream_net_cfs"].to_numpy(float)
    closure = {
        "scenario": scenario,
        "fold_id": fold["fold_id"],
        "forcing_rows": int(len(featured)),
        "reaches": int(featured.comid.nunique()),
        "months_per_reach_min": int(featured.groupby("comid").size().min()),
        "months_per_reach_max": int(featured.groupby("comid").size().max()),
        "scale": scale,
        "max_abs_unscaled_topology_error_cfs": float(np.max(np.abs(upstream - explicit))),
        "max_abs_scaled_predictor_error_cfs": float(np.max(np.abs(qcalc - scale * explicit))),
        "max_abs_local_alias_error_cfs": float(np.max(np.abs(local - featured["local_net_cfs"].to_numpy(float)))),
        "negative_unscaled_input_count": int(np.sum(upstream < -1e-12)),
        "semantics_unique": sorted(featured["network_input_semantics"].astype(str).unique().tolist()),
    }
    (fold_dir / "feature_accounting_audit.json").write_text(
        json.dumps(closure, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return closure


def run_fold(scenario: str, fold: dict[str, object]) -> dict[str, object]:
    module = load_component(f"{scenario}_{fold['train_end']}")
    fold_dir = ROOT / "outputs" / scenario / "blocked_folds" / str(fold["fold_id"])
    report_dir = fold_dir / "reports"
    figure_dir = fold_dir / "figure"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure(module, scenario, fold, report_dir, figure_dir)
    accounting = audit_fold_features(module, scenario, fold, fold_dir)
    started = datetime.now()
    module.main()
    ended = datetime.now()
    source = report_dir / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.parquet"
    prediction = pd.read_parquet(source)
    csv_source = report_dir / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv"
    csv_prediction = pd.read_csv(csv_source, encoding="utf-8-sig")
    csv_rebuilt_from_authoritative_parquet = False
    if len(csv_prediction) != len(prediction):
        # The synchronized Test volume has intermittently yielded a truncated
        # large text mirror even though the authoritative Parquet is complete.
        # Rebuild the mirror atomically from the verified Parquet; never accept
        # or publish a partial CSV.
        mirror_columns = [
            "comid", "q_site", "year", "month", "split", "actual", "predict",
            "Q_calc_cfs", "Q_ma_cfs", "PPT", "AET", "PET", "scenario_id",
            "upstream_positive_input_equivalent_cfs", "network_input_predictor_cfs",
            "network_input_climatology_cfs", "network_input_scale",
            "network_input_semantics", "et_feature_block",
        ]
        temporary = csv_source.with_suffix(".csv.rebuild.tmp")
        prediction[mirror_columns].to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(csv_source)
        csv_prediction = pd.read_csv(csv_source, encoding="utf-8-sig")
        csv_rebuilt_from_authoritative_parquet = True
    if len(csv_prediction) != len(prediction):
        raise RuntimeError(
            f"Prediction mirror rebuild failed: parquet={len(prediction)}, csv={len(csv_prediction)}"
        )
    input_panel = pd.read_parquet(INPUT, columns=["comid", "q_site", "year", "month", "Q_obsv_cfs"])
    expected = input_panel.loc[
        input_panel["Q_obsv_cfs"].notna() & input_panel["Q_obsv_cfs"].gt(0),
        ["comid", "q_site", "year", "month"],
    ].copy()
    expected["q_site"] = expected["q_site"].astype(str)
    prediction["q_site"] = prediction["q_site"].astype(str)
    key4 = ["comid", "q_site", "year", "month"]
    if len(prediction) != len(expected) or prediction[key4].duplicated().any():
        raise RuntimeError("Complete observation population gate failed")
    if not prediction[key4].sort_values(key4).reset_index(drop=True).equals(
        expected[key4].sort_values(key4).reset_index(drop=True)
    ):
        raise RuntimeError("Complete observation key gate failed")
    evaluation = prediction[prediction.year.between(int(fold["eval_start"]), int(fold["eval_end"]))].copy()
    evaluation["fold_id"] = str(fold["fold_id"])
    evaluation["scenario_id"] = scenario
    evaluation["network_input_scale"] = float(SCENARIOS[scenario]["scale"])
    evaluation["accounting_contract"] = "literal_rho_incarea_topology_single_outlet_quick"
    evaluation.to_parquet(fold_dir / "evaluation_predictions.parquet", index=False)
    evaluation.to_csv(fold_dir / "evaluation_predictions.csv", index=False, encoding="utf-8-sig")
    return {
        **fold,
        "scenario": scenario,
        "network_input_scale": float(SCENARIOS[scenario]["scale"]),
        "rows": int(len(evaluation)),
        "elapsed_seconds": (ended - started).total_seconds(),
        "csv_rebuilt_from_authoritative_parquet": csv_rebuilt_from_authoritative_parquet,
        **{k: accounting[k] for k in [
            "max_abs_unscaled_topology_error_cfs", "max_abs_scaled_predictor_error_cfs",
        ]},
    }


def assemble(scenario: str) -> dict[str, object]:
    output = ROOT / "outputs" / scenario
    rows = [run_fold(scenario, fold) for fold in FOLDS]
    pd.DataFrame(rows).to_csv(output / "blocked_fold_manifest.csv", index=False, encoding="utf-8-sig")
    parts = [
        pd.read_parquet(output / "blocked_folds" / fold["fold_id"] / "evaluation_predictions.parquet")
        for fold in FOLDS
    ]
    oof = pd.concat(parts, ignore_index=True)
    if len(oof) != 8738 or oof.fold_id.nunique() != 3 or oof[KEY].duplicated().any():
        raise RuntimeError(f"OOF gate failed for {scenario}: rows={len(oof)}")
    path = output / "q72_three_fold_oof_predictions.parquet"
    oof.to_parquet(path, index=False)
    oof.to_csv(output / "q72_three_fold_oof_predictions.csv", index=False, encoding="utf-8-sig")
    payload = {
        "runtime": RUNTIME,
        "scenario": scenario,
        "network_input_scale": float(SCENARIOS[scenario]["scale"]),
        "input_sha256": sha256(INPUT),
        "component_sha256": sha256(COMPONENT),
        "oof_sha256": sha256(path),
        "folds": rows,
    }
    (output / "run_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def exact_a0_gate() -> dict[str, object]:
    a0 = pd.read_parquet(ROOT / "outputs" / "P0" / "q72_three_fold_oof_predictions.parquet")
    ref = pd.read_parquet(REFERENCE_I0)
    a0["q_site"] = a0["q_site"].astype(str)
    ref["q_site"] = ref["q_site"].astype(str)
    a0 = a0.sort_values(KEY).reset_index(drop=True)
    ref = ref.sort_values(KEY).reset_index(drop=True)
    key_equal = a0[KEY].equals(ref[KEY])
    actual_max = float(np.max(np.abs(a0.actual.to_numpy(float) - ref.actual.to_numpy(float))))
    prediction_max = float(np.max(np.abs(a0.predict.to_numpy(float) - ref.predict.to_numpy(float))))
    result = {
        "rows_a0": len(a0),
        "rows_reference_i0": len(ref),
        "keys_equal": bool(key_equal),
        "actual_max_abs_difference": actual_max,
        "prediction_max_abs_difference": prediction_max,
        "pass": bool(len(a0) == len(ref) == 8738 and key_equal and actual_max <= 1e-12 and prediction_max <= 1e-8),
        "reference_sha256": sha256(REFERENCE_I0),
    }
    (ROOT / "logs" / "P0_exact_reproduction_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not result["pass"]:
        raise RuntimeError(f"P0 exact reproduction failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=["P0", "P1", "all"])
    args = parser.parse_args()
    payload: dict[str, object] = {}
    if args.scenario in {"P0", "all"}:
        payload["P0"] = assemble("P0")
        payload["P0_reproduction"] = exact_a0_gate()
    if args.scenario in {"P1", "all"}:
        if not (ROOT / "logs" / "P0_exact_reproduction_gate.json").exists():
            raise RuntimeError("P1 is blocked until P0 exact reproduction has passed")
        gate = json.loads((ROOT / "logs" / "P0_exact_reproduction_gate.json").read_text(encoding="utf-8"))
        if not gate.get("pass"):
            raise RuntimeError("P1 is blocked by failed P0 reproduction")
        payload["P1"] = assemble("P1")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
