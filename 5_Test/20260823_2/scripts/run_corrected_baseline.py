from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT.parent
PARENT = TEST_ROOT / "20260813_54"
COMPONENT = PARENT / "scripts" / "components" / "q72_prior_semantics_component.py"
INPUT = PARENT / "inputs" / "parent_indata.parquet"
TOPOLOGY = PARENT / "inputs" / "topology" / "topology_edges.csv"
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
EPS = 1.0e-12
KEY = ["comid", "q_site", "year", "month", "fold_id"]
FOLDS = [
    {"fold_id": "fit_2006_2011_eval_2012_2013", "inner_end": 2009, "train_end": 2011, "eval_start": 2012, "eval_end": 2013},
    {"fold_id": "fit_2006_2013_eval_2014_2015", "inner_end": 2011, "train_end": 2013, "eval_start": 2014, "eval_end": 2015},
    {"fold_id": "fit_2006_2015_eval_2016_2018", "inner_end": 2013, "train_end": 2015, "eval_start": 2016, "eval_end": 2018},
]
BASE = {
    "rho": 0.70, "wm": 480.0, "et_gamma": 0.75, "sas_rho": 0.93,
    "young_k": 1.5, "storage_scale": 720.0, "prod_capacity": 240.0,
    "runoff_gamma": 2.5, "quick_rho": 0.25, "base_rho": 0.85,
    "base_release": 0.10, "highflow_scale": 1.0,
    "fixed_sigma": 3.0, "production_sigma": 1.5, "group_sigma": 1.5,
    "multistore_sigma": 0.30, "station_sigma": 1.0, "slope_sigma": 0.15,
    "regime_slope_sigma": 0.25, "anomaly_weight": 0.0,
    "flow_contrast_weight": 1.0,
}


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kge_2012(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[mask], pred[mask]
    if len(obs) < 2 or np.std(obs) <= 0 or np.mean(obs) == 0 or np.mean(pred) == 0:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else np.nan
    gamma = (np.std(pred) / np.mean(pred)) / (np.std(obs) / np.mean(obs))
    beta = np.mean(pred) / np.mean(obs)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (gamma - 1.0) ** 2 + (beta - 1.0) ** 2))


def load_component(label: str, highflow_scale: float = 1.0):
    spec = importlib.util.spec_from_file_location(f"controlled_q72_{label}", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.INPUT_PATH = INPUT
    module.TOPOLOGY_PATH = TOPOLOGY
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    module.NETWORK_INPUT_SCALE = 1.0
    module.NETWORK_INPUT_SEMANTICS = "literal_upstream_positive_input_equivalent"
    module.DYNAMIC_BETA_W = 0.0
    module.configure_engineering_repair(True)
    module.configure_full_gaussian_prior_space(True)
    module.ZERO_PRESERVING_FULL_PRIOR_MODE = True
    module.DETERMINISTIC_SPINUP_MODE = True
    module.ET_STATE_OPERATOR_MODE = "baseline_clip"
    module.set_et_feature_block_mode("full")
    module.kge_2012 = kge_2012
    original_simulator = module.simulate_production_variant

    def controlled_simulator(*args, **kwargs):
        if "highflow_scale" not in kwargs:
            kwargs["highflow_scale"] = float(highflow_scale)
        return original_simulator(*args, **kwargs)

    module.simulate_production_variant = controlled_simulator
    return module


def featured(module) -> pd.DataFrame:
    forcing = module.load_forcing_panel()
    return module.build_featured_observation_panel(
        forcing,
        rho=BASE["rho"], wm=BASE["wm"], et_gamma=BASE["et_gamma"],
        sas_rho=BASE["sas_rho"], young_k=BASE["young_k"],
        storage_scale=BASE["storage_scale"], prod_capacity=BASE["prod_capacity"],
        runoff_gamma=BASE["runoff_gamma"], quick_rho=BASE["quick_rho"],
        base_rho=BASE["base_rho"], base_release=BASE["base_release"],
        state_calendar_mode="full_forcing",
    )


def fit(module, frame: pd.DataFrame, stations: list[str], train_end: int, hs: float, report_dir: Path):
    train = frame[frame["year"] <= int(train_end)].copy()
    mean, std = module.standardize_fit(train)
    module.REPORT_DIR = report_dir
    beta = module.fit_map_ridge(
        train, stations, mean, std,
        fixed_sigma=BASE["fixed_sigma"], production_sigma=BASE["production_sigma"],
        group_sigma=BASE["group_sigma"], multistore_sigma=BASE["multistore_sigma"],
        hysteresis_sigma=float(hs), station_sigma=BASE["station_sigma"],
        slope_sigma=BASE["slope_sigma"], regime_slope_sigma=BASE["regime_slope_sigma"],
        anomaly_weight=BASE["anomaly_weight"], flow_contrast_weight=BASE["flow_contrast_weight"],
    )
    return beta, mean, std


def station_summary(module, frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    summary = module.station_median_metrics(frame, pred)
    return {str(k): float(v) if np.isfinite(v) else np.nan for k, v in summary.items()}


def choose_hysteresis(module, frame: pd.DataFrame, stations: list[str], fold: dict[str, object], report_dir: Path) -> tuple[float, pd.DataFrame]:
    tr = frame[frame["year"] <= int(fold["inner_end"])].copy()
    iv = frame[frame["year"].between(int(fold["inner_end"]) + 1, int(fold["train_end"]))].copy()
    mean, std = module.standardize_fit(tr)
    rows = []
    for hs in [0.30, 0.80, 1.50, 3.00]:
        module.REPORT_DIR = report_dir / f"inner_hs_{hs:g}"
        beta = module.fit_map_ridge(
            tr, stations, mean, std,
            fixed_sigma=BASE["fixed_sigma"], production_sigma=BASE["production_sigma"],
            group_sigma=BASE["group_sigma"], multistore_sigma=BASE["multistore_sigma"],
            hysteresis_sigma=hs, station_sigma=BASE["station_sigma"],
            slope_sigma=BASE["slope_sigma"], regime_slope_sigma=BASE["regime_slope_sigma"],
            anomaly_weight=BASE["anomaly_weight"], flow_contrast_weight=BASE["flow_contrast_weight"],
        )
        pred = np.exp(np.clip(module.predict_log(iv, beta, stations, mean, std), -20, 20))
        md = module.metric_dict(iv["Q_obsv_cfs"].to_numpy(float), pred)
        sm = station_summary(module, iv, pred)
        score = (
            sm["median_NSE_log"] + 1.5 * sm["median_KGE"]
            - 0.60 * min(abs(sm["median_alpha"] - 1.0), 2.0)
            - 0.005 * min(sm["median_abs_PBIAS"], 9999.0)
            + 0.01 * sm["good_count"]
        )
        rows.append({
            "hysteresis_sigma": hs, "inner_score": score,
            "inner_pooled_log_NSE": md["NSE_log"],
            "inner_pooled_KGE2012": md["KGE_2012"], **sm,
        })
    grid = pd.DataFrame(rows).sort_values(["inner_score", "hysteresis_sigma"], ascending=[False, True]).reset_index(drop=True)
    return float(grid.iloc[0]["hysteresis_sigma"]), grid


def parent_fold_hysteresis() -> dict[str, float]:
    parent = pd.read_parquet(PARENT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet")
    if "hysteresis_sigma" in parent.columns:
        return {str(k): float(v) for k, v in parent.groupby("fold_id")["hysteresis_sigma"].first().items()}
    result = {}
    for fold in FOLDS:
        path = PARENT / "outputs" / "P1" / "blocked_folds" / str(fold["fold_id"]) / "reports" / "monthly_bayes_seasonal_hysteresis_hyperparameter_grid.csv"
        grid = pd.read_csv(path, encoding="utf-8-sig")
        result[str(fold["fold_id"])] = float(grid.sort_values(["selected", "inner_score"], ascending=[False, False]).iloc[0]["hysteresis_sigma"]) if "selected" in grid else float(grid.sort_values("inner_score", ascending=False).iloc[0]["hysteresis_sigma"])
    return result


def run_model(model_id: str, reselect: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent_hs = parent_fold_hysteresis()
    parts = []
    selections = []
    for fold in FOLDS:
        fold_id = str(fold["fold_id"])
        print(f"[{model_id}] starting {fold_id}", flush=True)
        module = load_component(f"{model_id}_{fold_id}", highflow_scale=BASE["highflow_scale"])
        module.CAL_END_YEAR = int(fold["train_end"])
        module.INNER_TRAIN_END_YEAR = int(fold["inner_end"])
        frame = featured(module)
        stations = sorted(frame["q_site"].astype(str).unique())
        fold_dir = OUTPUTS / model_id / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        if reselect:
            hs, grid = choose_hysteresis(module, frame, stations, fold, fold_dir / "inner_artifacts")
            grid.to_csv(fold_dir / "hysteresis_grid_correct_kge2012.csv", index=False, encoding="utf-8-sig")
        else:
            hs = float(parent_hs[fold_id])
            grid = pd.DataFrame([{"hysteresis_sigma": hs, "selection": "frozen_parent"}])
        beta, mean, std = fit(module, frame, stations, int(fold["train_end"]), hs, fold_dir / "final_fit")
        evaluation = frame[frame["year"].between(int(fold["eval_start"]), int(fold["eval_end"]))].copy()
        prediction = np.exp(np.clip(module.predict_log(evaluation, beta, stations, mean, std), -20, 20))
        out = evaluation[["comid", "q_site", "year", "month", "Q_obsv_cfs", "Q_calc_cfs", "routed_quick_cfs", "routed_base_cfs", "sas_old_release_cfs", "production_mass_balance_error_mm"]].copy()
        out = out.rename(columns={"Q_obsv_cfs": "actual"})
        out["predict"] = prediction
        out["fold_id"] = fold_id
        out["model_id"] = model_id
        out["hysteresis_sigma"] = hs
        out["network_input_scale"] = 1.0
        out["highflow_scale"] = BASE["highflow_scale"]
        out.to_parquet(fold_dir / "evaluation_predictions.parquet", index=False)
        parts.append(out)
        selections.append({
            "model_id": model_id, "fold_id": fold_id, "hysteresis_sigma": hs,
            "network_input_scale": 1.0, "highflow_scale": BASE["highflow_scale"],
            "evaluation_rows": int(len(out)), "design_columns": int(len(module.design_column_metadata(stations))),
            "max_abs_process_mass_balance_error_mm": float(evaluation["production_mass_balance_error_mm"].abs().max()),
        })
        print(f"[{model_id}] completed {fold_id}: hs={hs:g}, rows={len(out)}", flush=True)
    oof = pd.concat(parts, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    if len(oof) != 7755 or oof[KEY].duplicated().any():
        raise RuntimeError(f"OOF population failure for {model_id}: {len(oof)}")
    model_dir = OUTPUTS / model_id
    oof.to_parquet(model_dir / "oof.parquet", index=False)
    selection_table = pd.DataFrame(selections)
    selection_table.to_csv(model_dir / "fold_parameter_selections.csv", index=False, encoding="utf-8-sig")
    return oof, selection_table


def pooled_metrics(frame: pd.DataFrame) -> dict[str, float]:
    obs = frame["actual"].to_numpy(float)
    pred = frame["predict"].to_numpy(float)
    logobs = np.log(np.clip(obs, EPS, None))
    logpred = np.log(np.clip(pred, EPS, None))
    return {
        "n": int(len(frame)),
        "raw_NSE": float(1.0 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)),
        "log_NSE": float(1.0 - np.sum((logpred - logobs) ** 2) / np.sum((logobs - logobs.mean()) ** 2)),
        "KGE_2012": kge_2012(obs, pred),
        "PBIAS_pct": float(100.0 * np.sum(pred - obs) / np.sum(obs)),
        "RMSE": float(np.sqrt(np.mean((pred - obs) ** 2))),
        "log_RMSE": float(np.sqrt(np.mean((logpred - logobs) ** 2))),
    }


def station_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, part in frame.groupby(frame["q_site"].astype(str), sort=True):
        metrics = pooled_metrics(part)
        rows.append({"q_site": str(site), **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    exact, exact_sel = run_model("H0_EXACT_REPRO", reselect=False)
    parent = pd.read_parquet(PARENT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet")
    parent["q_site"] = parent["q_site"].astype(str)
    exact["q_site"] = exact["q_site"].astype(str)
    joined = parent[KEY + ["predict"]].rename(columns={"predict": "parent_predict"}).merge(
        exact[KEY + ["predict"]].rename(columns={"predict": "reproduced_predict"}), on=KEY, validate="one_to_one"
    )
    joined["absolute_difference"] = (joined["parent_predict"] - joined["reproduced_predict"]).abs()
    reproduction = {
        "rows": int(len(joined)),
        "max_abs_prediction_difference_cfs": float(joined["absolute_difference"].max()),
        "mean_abs_prediction_difference_cfs": float(joined["absolute_difference"].mean()),
        "pass": bool(len(joined) == 7755 and joined["absolute_difference"].max() <= 1e-7),
    }
    dump(REPORTS / "exact_reproduction_gate.json", reproduction)
    corrected, corrected_sel = run_model("H0_CORRECTED", reselect=True)
    metric_rows = []
    station_tables = []
    for model_id, frame in [("PARENT_FROZEN", parent.rename(columns={"actual": "actual"})), ("H0_EXACT_REPRO", exact), ("H0_CORRECTED", corrected)]:
        if "actual" not in frame:
            raise RuntimeError(f"Missing actual column: {model_id}")
        metric_rows.append({"model_id": model_id, **pooled_metrics(frame)})
        table = station_metrics(frame)
        table.insert(0, "model_id", model_id)
        station_tables.append(table)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORTS / "corrected_baseline_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    station = pd.concat(station_tables, ignore_index=True)
    station.to_parquet(OUTPUTS / "corrected_baseline_station_metrics.parquet", index=False)
    parameter_lock = {
        "model_id": "H0_CORRECTED",
        "component_sha256": sha256(COMPONENT),
        "input_sha256": sha256(INPUT),
        "topology_sha256": sha256(TOPOLOGY),
        "network_input_scale": 1.0,
        "highflow_scale": 1.0,
        "hysteresis_by_fold": {str(r.fold_id): float(r.hysteresis_sigma) for r in corrected_sel.itertuples(index=False)},
        "parent_hysteresis_by_fold": {str(r.fold_id): float(r.hysteresis_sigma) for r in exact_sel.itertuples(index=False)},
        "kge_implementation": "KGE2012_CV_RATIO",
        "outer_prediction_exposure": "registered_evaluation_rows_only",
        "et_gamma_status": "inactive_compatibility_signature_not_interpreted_as_parameter",
    }
    dump(REPORTS / "corrected_model_lock.json", parameter_lock)
    passed = reproduction["pass"] and len(corrected) == 7755 and corrected[KEY].duplicated().sum() == 0
    gate = {
        "stage": "20260823_2",
        "status": "PASS_CORRECTED_BASELINE_LOCKED" if passed else "BLOCKED",
        "exact_reproduction": reproduction,
        "corrected_hysteresis_by_fold": parameter_lock["hysteresis_by_fold"],
        "corrected_metrics": metrics[metrics["model_id"] == "H0_CORRECTED"].iloc[0].to_dict(),
        "next_authorized_stage": "20260823_3" if passed else None,
    }
    dump(REPORTS / "stage_gate.json", gate)
    program_path = TEST_ROOT / "20260823_1" / "program_manifest.json"
    program = json.loads(program_path.read_text(encoding="utf-8"))
    program["stages"]["20260823_2"]["status"] = "passed" if passed else "failed"
    program["stages"]["20260823_2"]["stage_gate"] = "../20260823_2/reports/stage_gate.json"
    program_path.write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

