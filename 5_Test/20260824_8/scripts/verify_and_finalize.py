from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(r"E:\SPARROW\5_Test\20260824_8")
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    obs = frame.tn_mg_l.to_numpy(float)
    pred = frame.pred_tn_mg_l.to_numpy(float)
    error = pred - obs
    denom = float(np.sum(np.square(obs - obs.mean())))
    corr = float(np.corrcoef(obs, pred)[0, 1]) if np.std(obs) > 0 and np.std(pred) > 0 else math.nan
    station_abs = []
    station_anom = []
    for _, group in frame.groupby("station_key", observed=True):
        o = np.log1p(group.tn_mg_l.to_numpy(float)); p = np.log1p(group.pred_tn_mg_l.to_numpy(float))
        station_abs.append(float(np.sqrt(np.mean(np.square(p - o)))))
        station_anom.append(float(np.sqrt(np.mean(np.square((p - p.mean()) - (o - o.mean()))))))
    return {
        "n": float(len(frame)),
        "rmse_mg_l": float(np.sqrt(np.mean(np.square(error)))),
        "mae_mg_l": float(np.mean(np.abs(error))),
        "nse_mg_l": 1.0 - float(np.sum(np.square(error))) / denom,
        "pearson_r2_mg_l": corr * corr,
        "pbias_percent": 100.0 * float(np.sum(error)) / float(np.sum(obs)),
        "station_macro_rmse_log1p": float(np.mean(station_abs)),
        "station_macro_anomaly_rmse_log1p": float(np.mean(station_anom)),
    }


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    checks = []
    def check(name: str, passed: bool, evidence) -> None:
        checks.append({"check": name, "pass": bool(passed), "evidence": str(evidence)})

    required = [
        HERE / "experiment_contract.json",
        LOCKS / "development_mechanism_lock.json",
        LOCKS / "full_development_parameter_lock.json",
        LOCKS / "TN_2022_source_read_receipt.json",
        OUT / "full_development_parameters.parquet",
        OUT / "full_development_station_effects.parquet",
        OUT / "full_development_H1_parameters.parquet",
        OUT / "locked_2022_observations_snapshot.parquet",
        OUT / "reach_month_H1_exposure_2022.parquet",
        OUT / "locked_2022_retrospective_predictions.parquet",
        OUT / "locked_2022_retrospective_metrics.parquet",
        OUT / "locked_2022_month_flow_residuals.parquet",
        OUT / "locked_2022_retrospective_gates.parquet",
        OUT / "tree_163_locked_retrospective_diagnostic.parquet",
        REPORTS / "locked_2022_retrospective_audit.json",
        REPORTS / "final_program_decision.json",
        REPORTS / "technical_report.md",
    ]
    check("required_artifacts", all(path.exists() for path in required), [str(path) for path in required if not path.exists()])
    mechanism = json.loads((LOCKS / "development_mechanism_lock.json").read_text(encoding="utf-8"))
    parameter_lock = json.loads((LOCKS / "full_development_parameter_lock.json").read_text(encoding="utf-8"))
    receipt = json.loads((LOCKS / "TN_2022_source_read_receipt.json").read_text(encoding="utf-8"))
    decision = json.loads((REPORTS / "final_program_decision.json").read_text(encoding="utf-8"))
    check("mechanism_precedes_2022", mechanism["written_before_2022_TN_values_read"] and not mechanism["TN_2022_values_read"], mechanism["lock"])
    check("parameters_precede_2022", parameter_lock["written_before_2022_TN_values_read"] and not parameter_lock["TN_2022_values_read"], parameter_lock["lock"])
    locked_hash_errors = [path for path, value in parameter_lock["files"].items() if not Path(path).exists() or sha256(Path(path)) != value]
    check("parameter_lock_hashes", not locked_hash_errors, locked_hash_errors)
    check("single_2022_source_read", receipt["source_file_read_count"] == 1 and receipt["TN_2022_values_read"] is True, receipt["source_file_read_count"])
    check("snapshot_integrity", sha256(OUT / "locked_2022_observations_snapshot.parquet") == receipt["snapshot_sha256"], receipt["snapshot_sha256"])

    snapshot = pd.read_parquet(OUT / "locked_2022_observations_snapshot.parquet")
    check("snapshot_year", set(snapshot.year.unique()) == {2022}, sorted(snapshot.year.unique()))
    check("snapshot_rows", len(snapshot) == decision["TN_rows"], len(snapshot))
    parameters = pd.read_parquet(OUT / "full_development_parameters.parquet")
    h1 = pd.read_parquet(OUT / "full_development_H1_parameters.parquet")
    check("formal_parameter_keys", len(parameters) == 48 and parameters[["model_id", "layer", "arm"]].drop_duplicates().shape[0] == 48, len(parameters))
    check("formal_model_count", parameters.model_id.nunique() == 12 and len(h1) == 12, parameters.model_id.nunique())
    check("optimizer_success", bool(parameters.success.all() and h1.H1_outer_success.all() and h1.H1_inner_success.all()), int((~parameters.success).sum()))
    check("no_parameter_boundary", int(parameters.eta_boundary.sum()) == 0 and int(parameters.beta_boundary.sum()) == 0 and int(h1.H1_vf_boundary.sum()) == 0, {"eta": int(parameters.eta_boundary.sum()), "beta": int(parameters.beta_boundary.sum()), "vf": int(h1.H1_vf_boundary.sum())})

    exposure = pd.read_parquet(OUT / "reach_month_H1_exposure_2022.parquet")
    check("H1_exposure_coverage", len(exposure) == 2760 and exposure.reach_id.nunique() == 230 and set(exposure.month.unique()) == set(range(1, 13)), exposure.shape)
    check("Q72_only_water", set(exposure.water_source.astype(str)) == {"Q72_structural_canonical_main_only"}, exposure.water_source.unique())
    check("Andreadis_geometry_only", set(exposure.geometry_source.astype(str)) == {"Andreadis_width_depth_only"} and not exposure.wqd_reference_discharge_used.any(), exposure.geometry_source.unique())

    predictions = pd.read_parquet(OUT / "locked_2022_retrospective_predictions.parquet")
    check("prediction_groups", predictions.groupby(["model_id", "layer", "arm"], observed=True).ngroups == 48, predictions.groupby(["model_id", "layer", "arm"], observed=True).ngroups)
    sizes = predictions.groupby(["model_id", "layer", "arm"], observed=True).size()
    check("complete_2022_keys", bool(sizes.eq(len(snapshot)).all()), sizes.value_counts().to_dict())
    check("prediction_unique", not predictions.duplicated(["model_id", "layer", "arm", "station_key", "year", "month"]).any(), "unique")
    check("prediction_year", set(predictions.year.unique()) == {2022}, sorted(predictions.year.unique()))

    effects = pd.read_parquet(OUT / "full_development_station_effects.parquet")
    max_error = 0.0
    for (model_id, layer, arm), frame in predictions.groupby(["model_id", "layer", "arm"], observed=True):
        row = parameters.loc[parameters.model_id.eq(model_id) & parameters.layer.eq(layer) & parameters.arm.eq(arm)].iloc[0]
        base = np.divide(
            (float(row.eta_quick) * frame.routed_quick_tn_kg_n.to_numpy(float) + float(row.eta_gw) * frame.routed_gw_tn_kg_n.to_numpy(float)) * 1000.0,
            frame.routed_water_volume_m3.to_numpy(float),
            out=np.zeros(len(frame), dtype=float), where=frame.routed_water_volume_m3.to_numpy(float) > 1e-12,
        )
        multiplier = np.exp(np.clip(float(row.beta_low) * frame.cq_low.to_numpy(float) + float(row.beta_high) * frame.cq_high.to_numpy(float), -20, 20))
        effect = frame.station_effect.to_numpy(float) if layer == "P2" else np.zeros(len(frame))
        expected = np.maximum(np.expm1(np.log1p(base * multiplier) + effect), 0.0)
        max_error = max(max_error, float(np.max(np.abs(expected - frame.pred_tn_mg_l.to_numpy(float)))))
    check("locked_prediction_exact_recalculation", max_error <= 1e-12, max_error)

    stored_metrics = pd.read_parquet(OUT / "locked_2022_retrospective_metrics.parquet")
    metric_error = 0.0
    fields = ["n", "rmse_mg_l", "mae_mg_l", "nse_mg_l", "pearson_r2_mg_l", "pbias_percent", "station_macro_rmse_log1p", "station_macro_anomaly_rmse_log1p"]
    primary = predictions.loc[predictions.primary_river_domain]
    for keys, frame in primary.groupby(["model_id", "layer", "arm"], observed=True):
        row = stored_metrics.loc[
            stored_metrics.model_id.eq(keys[0]) & stored_metrics.layer.eq(keys[1]) & stored_metrics.arm.eq(keys[2])
        ].iloc[0]
        calculated = metric_values(frame)
        metric_error = max(metric_error, max(abs(float(row[field]) - calculated[field]) for field in fields))
    check("metric_exact_recalculation", metric_error <= 1e-12, metric_error)
    gates = pd.read_parquet(OUT / "locked_2022_retrospective_gates.parquet")
    reproduced = {
        f"{layer}_{block}_{field}": int(gates.loc[gates.layer.eq(layer) & gates.block.eq(block), field].sum())
        for layer in ("P1", "P2") for block in ("station", "tree")
        for field in ("noninferior", "predictively_improved")
    }
    check("gate_count_reproduction", reproduced == decision["gate_counts"], reproduced)
    check("terminal_state", decision["program_terminal_state"] == "MONITORED_STATION_PREDICTION_UPGRADE_ONLY" and not decision["architecture_changed_by_2022"] and not decision["parameters_changed_by_2022"], decision["program_terminal_state"])
    check("spatial_claim_boundary", decision["spatial_transfer_status"] == "NOT_SUPPORTED_FROM_DEVELOPMENT_NESTED_LOTO", decision["spatial_transfer_status"])
    check("temperature_closed", decision["temperature_used"] is False, decision["temperature_used"])

    status = "PASS" if all(row["pass"] for row in checks) else "FAIL"
    verification = {
        "status": status,
        "checks": checks,
        "max_locked_prediction_recalculation_error_mg_l": max_error,
        "max_metric_recalculation_error": metric_error,
        "TN_2022_source_file_read_count": receipt["source_file_read_count"],
        "program_terminal_state": decision["program_terminal_state"],
    }
    verification_path = REPORTS / "independent_verification.json"
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    if status != "PASS":
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        raise RuntimeError("FINAL_VERIFICATION_FAILED")

    excluded_manifest_targets = {
        HERE / "final_artifact_manifest.json",
        REPORTS / "completion_audit.json",
    }
    artifact_paths = sorted(
        path for path in HERE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path not in excluded_manifest_targets
    )
    manifest = {
        "status": "COMPLETE",
        "program_terminal_state": "MONITORED_STATION_PREDICTION_UPGRADE_ONLY",
        "development_support_period": [2016, 2021],
        "OOF_evaluation_years": [2018, 2019, 2020, 2021],
        "retrospective_locked_year": 2022,
        "TN_2022_source_file_read_count": 1,
        "files": {str(path): sha256(path) for path in artifact_paths},
    }
    (HERE / "final_artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    completion = {
        "status": "PASS",
        "program_complete": True,
        "terminal_state": manifest["program_terminal_state"],
        "verification_sha256": sha256(verification_path),
        "artifact_manifest_sha256": sha256(HERE / "final_artifact_manifest.json"),
        "scope": "monitored-station temporal upgrade only; spatial transfer unresolved",
    }
    (REPORTS / "completion_audit.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verification": verification, "completion": completion}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
