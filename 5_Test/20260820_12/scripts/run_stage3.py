from __future__ import annotations

import json

import numpy as np
import pandas as pd

from a0_shared import (
    HERE, OUT, PROGRAM, REPORTS, development_observations, dump_json, fit_harmonic, fit_uniform,
    formal_specs, hash_manifest, locked_2022_observations, metric_values, observation_basis,
    parameter_row, parent_shared, predict_fit, require_runtime, sha256,
)


def count_noninferior(frame: pd.DataFrame, layer: str, block: str) -> int:
    return int(frame.loc[frame.layer.eq(layer) & frame.block.eq(block), "noninferior"].sum())


def decision_registry() -> tuple[dict[str, object], pd.DataFrame]:
    temporal = pd.read_parquet(OUT / "temporal_paired_comparisons.parquet")
    nested = pd.read_parquet(OUT / "nested_spatial_metrics.parquet")
    fingerprints = pd.read_parquet(OUT / "registered_fingerprint_metrics.parquet")
    stability = pd.read_parquet(OUT / "harmonic_fold_stability.parquet")
    p1_fingerprint = fingerprints.loc[fingerprints.layer.eq("P1")].copy()
    p1_fingerprint["registered_gate_model"] = (
        p1_fingerprint.J6_improved
        & p1_fingerprint.Jphase_improved
        & p1_fingerprint.at_least_one_contrast_significantly_shrunk
        & p1_fingerprint.registered_months_abs_bias_shrunk.ge(4)
        & p1_fingerprint.registered_month_max_abs_worsening.le(0.01)
        & p1_fingerprint.contrast_max_abs_worsening.le(0.01)
    )
    boundary = stability.groupby("model_id", as_index=False).agg(
        amplitude_boundary_model=("amplitude_boundary_model", "max"),
        eta_boundary_model=("eta_boundary_model", "max"),
    )
    counts = {
        "P1_station_noninferior": count_noninferior(temporal, "P1", "station_key"),
        "P1_tree_noninferior": count_noninferior(temporal, "P1", "terminal_tree_id"),
        "P2_station_noninferior": count_noninferior(temporal, "P2", "station_key"),
        "P2_tree_noninferior": count_noninferior(temporal, "P2", "terminal_tree_id"),
        "nested_LOSO_noninferior": int(nested.loc[nested.evaluation.eq("LOSO"), "noninferior"].sum()),
        "nested_LOTO_noninferior": int(nested.loc[nested.evaluation.eq("LOTO"), "noninferior"].sum()),
        "J6_improved": int(p1_fingerprint.J6_improved.sum()),
        "Jphase_improved": int(p1_fingerprint.Jphase_improved.sum()),
        "registered_month_gate": int(p1_fingerprint.registered_gate_model.sum()),
        "phase_stable_models": int(stability.loc[stability.layer.eq("P1"), "seasonal_phase_stable"].sum()),
        "amplitude_boundary_models": int(boundary.amplitude_boundary_model.sum()),
        "eta_boundary_models": int(boundary.eta_boundary_model.sum()),
        "P1_station_improved": int(temporal.loc[temporal.layer.eq("P1") & temporal.block.eq("station_key"), "predictively_improved"].sum()),
        "P1_tree_improved": int(temporal.loc[temporal.layer.eq("P1") & temporal.block.eq("terminal_tree_id"), "predictively_improved"].sum()),
        "P2_station_improved": int(temporal.loc[temporal.layer.eq("P2") & temporal.block.eq("station_key"), "predictively_improved"].sum()),
        "P2_tree_improved": int(temporal.loc[temporal.layer.eq("P2") & temporal.block.eq("terminal_tree_id"), "predictively_improved"].sum()),
    }
    performance_fingerprint_pass = all([
        counts["P1_station_noninferior"] >= 10, counts["P1_tree_noninferior"] >= 10,
        counts["P2_station_noninferior"] >= 10, counts["P2_tree_noninferior"] >= 10,
        counts["nested_LOSO_noninferior"] >= 10, counts["nested_LOTO_noninferior"] >= 10,
        counts["J6_improved"] >= 10, counts["Jphase_improved"] >= 10,
        counts["registered_month_gate"] >= 10,
    ])
    if counts["amplitude_boundary_models"] > 2:
        decision = "seasonal_availability_boundary_confounded"
    elif counts["eta_boundary_models"] > 2:
        decision = "delivery_efficiency_confounded"
    elif performance_fingerprint_pass and counts["phase_stable_models"] >= 10:
        decision = "source_timing_supported"
    elif performance_fingerprint_pass:
        decision = "seasonal_availability_supported_phase_unresolved"
    elif counts["P2_station_improved"] >= 6 and counts["P2_tree_improved"] >= 6 and counts["P1_station_improved"] < 6:
        decision = "station_readout_compensated_not_source_timing_supported"
    else:
        decision = "source_timing_not_supported"
    selected = "A0_HARMONIC" if decision in {"source_timing_supported", "seasonal_availability_supported_phase_unresolved"} else "UNIFORM_PARENT"
    return {
        "scenario_id": "20260820_12", "status": "development_decision_complete", "decision": decision,
        "selected_mechanism": selected, "counts": counts,
        "performance_and_fingerprint_gate_pass": performance_fingerprint_pass,
        "A1_status": "closed_pending_new_registered_hypothesis",
        "A2_status": "closed",
        "forbidden_directions_remain_closed": ["HLEG_complexification", "new_mu", "temperature", "groundwater_age", "second_groundwater_store"],
    }, p1_fingerprint.merge(boundary, on="model_id", validate="one_to_one")


def main() -> None:
    require_runtime()
    nested_audit = json.loads((REPORTS / "stage2_nested_spatial_audit.json").read_text(encoding="utf-8"))
    if nested_audit["status"] != "PASS":
        raise RuntimeError("STOP_STAGE2_NOT_PASS")
    decision, model_gate = decision_registry()
    model_gate.to_parquet(OUT / "model_gate_registry.parquet", index=False)
    dump_json(REPORTS / "source_availability_decision.json", decision)
    dump_json(REPORTS / "development_mechanism_lock.json", {
        "scenario_id": "20260820_12", "development_period": [2016, 2021], "oof_years": [2018, 2019, 2020, 2021],
        "decision": decision["decision"], "selected_mechanism": decision["selected_mechanism"],
        "TN_2022_not_materialized": True,
    })

    observations = development_observations()
    shared = parent_shared()
    final_parameter_rows = []
    final_effect_rows = []
    final_fits: dict[tuple[str, str], dict[str, object]] = {}
    for index, spec in enumerate(formal_specs(), start=1):
        model_id = str(spec["model_id"])
        basis = observation_basis(model_id, observations)
        indices = np.arange(len(basis.base))
        for layer in ("P1", "P2"):
            fit = fit_harmonic(shared, basis, indices, layer) if decision["selected_mechanism"] == "A0_HARMONIC" else fit_uniform(shared, basis, indices, layer)
            final_fits[(model_id, layer)] = fit
            row = parameter_row(model_id, "FULL_2016_2021", layer, decision["selected_mechanism"], fit)
            final_parameter_rows.append(row)
            if layer == "P2":
                final_effect_rows.extend({"model_id": model_id, "layer": layer, "station_key": station, "station_effect": value} for station, value in fit["effects"].items())
        print(f"stage3 full refit {index:02d}/12 {model_id}", flush=True)
    final_parameters = pd.DataFrame(final_parameter_rows)
    final_parameters.to_parquet(OUT / "full_development_parameters.parquet", index=False)
    pd.DataFrame(final_effect_rows).to_parquet(OUT / "full_development_station_effects.parquet", index=False)
    dump_json(REPORTS / "full_development_parameter_lock.json", {
        "scenario_id": "20260820_12", "selected_mechanism": decision["selected_mechanism"],
        "development_period": [2016, 2021], "parameters_file": str(OUT / "full_development_parameters.parquet"),
        "parameters_sha256": sha256(OUT / "full_development_parameters.parquet"),
        "station_effects_sha256": sha256(OUT / "full_development_station_effects.parquet"),
        "TN_2022_not_materialized_when_lock_written": True,
    })

    # This is the first materialization of 2022 TN in this experiment.
    observations_2022 = locked_2022_observations()
    retrospective_parts = []
    for spec in formal_specs():
        model_id = str(spec["model_id"])
        basis = observation_basis(model_id, observations_2022)
        for layer in ("P1", "P2"):
            fit = final_fits[(model_id, layer)]
            pred = predict_fit(shared, basis, fit, np.arange(len(basis.base)), layer, decision["selected_mechanism"] == "UNIFORM_PARENT")
            pred["model_id"] = model_id
            pred["layer"] = layer
            pred["mechanism"] = decision["selected_mechanism"]
            pred["evaluation"] = "locked_2022_retrospective_temporal_check"
            retrospective_parts.append(pred)
    retrospective = pd.concat(retrospective_parts, ignore_index=True)
    retrospective.to_parquet(OUT / "locked_2022_retrospective_predictions.parquet", index=False)
    performance_rows = []
    temporal_metrics = pd.read_parquet(OUT / "temporal_oof_metrics.parquet")
    for row in temporal_metrics.to_dict("records"):
        performance_rows.append({"evaluation": "development_oof_2018_2021", **row})
    for (model_id, layer), group in retrospective.groupby(["model_id", "layer"]):
        performance_rows.append({"evaluation": "locked_2022_retrospective_temporal_check", "model_id": model_id, "mechanism": decision["selected_mechanism"], "layer": layer, "scope": "pooled", **metric_values(group)})
    pd.DataFrame(performance_rows).to_parquet(OUT / "final_performance_metrics.parquet", index=False)

    eta = pd.read_parquet(OUT / "availability_eta_compensation_diagnostic.parquet")
    eta_summary = {
        "rows": len(eta),
        "delta_eta_quick_median": float(eta.delta_eta_quick.median()),
        "delta_eta_gw_median": float(eta.delta_eta_gw.median()),
        "spearman_delta_eta_quick_vs_delta_training_rmse": float(eta.delta_eta_quick.corr(eta.delta_training_rmse_log1p, method="spearman")),
        "spearman_delta_eta_gw_vs_delta_training_rmse": float(eta.delta_eta_gw.corr(eta.delta_training_rmse_log1p, method="spearman")),
        "interpretation": "diagnostic only; A0 is evaluated with the same globally refitted pathway readout and does not independently identify a pure source process",
    }
    dump_json(REPORTS / "availability_eta_compensation_summary.json", eta_summary)

    counts = decision["counts"]
    report = f"""# 20260820_12 A0 Source Availability technical report

## Decision

**{decision['decision']}**  
Selected full-development mechanism: **{decision['selected_mechanism']}**

The experiment tests an `effective monthly diffuse-N availability operator`, not an observed fertilizer calendar. Q72, F00, the 12-member Legacy ensemble, fixed T1 and R0 remained frozen.

## Registered gates

| Gate | Models passing |
|---|---:|
| P1 station temporal noninferior | {counts['P1_station_noninferior']}/12 |
| P1 tree temporal noninferior | {counts['P1_tree_noninferior']}/12 |
| P2 station temporal noninferior | {counts['P2_station_noninferior']}/12 |
| P2 tree temporal noninferior | {counts['P2_tree_noninferior']}/12 |
| Nested LOSO noninferior | {counts['nested_LOSO_noninferior']}/12 |
| Nested LOTO noninferior | {counts['nested_LOTO_noninferior']}/12 |
| J6 improved | {counts['J6_improved']}/12 |
| Jphase improved | {counts['Jphase_improved']}/12 |
| Complete registered-month gate | {counts['registered_month_gate']}/12 |
| Stable phase | {counts['phase_stable_models']}/12 |

Amplitude-boundary models: {counts['amplitude_boundary_models']}/12. Eta-boundary models: {counts['eta_boundary_models']}/12.

## Interpretation boundary

The eta-compensation output is descriptive and is not an extra promotion gate. Even if A0 is retained, it means that a mass-conserving basin-wide availability perturbation improved the registered development evidence under the same refitted global pathway readout. It does not identify real fertilizer dates, a pure source coefficient, groundwater age, or denitrification.

2022 TN was read only after the development mechanism and full-development parameter locks were written. Its role is a locked retrospective temporal check and it did not select the mechanism.
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")

    manifest_path = PROGRAM / "program_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for scenario in manifest["scenarios"]:
        if scenario["scenario_id"] == "20260820_12":
            scenario["status"] = "passed" if decision["selected_mechanism"] == "A0_HARMONIC" else "confounded" if "confounded" in decision["decision"] else "failed"
            scenario["decision"] = decision["decision"]
        if scenario["scenario_id"] == "20260820_13":
            scenario["entry_condition_satisfied_by_A0_completion"] = True
            scenario["status"] = "closed"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    authoritative = [
        REPORTS / "development_mechanism_lock.json", REPORTS / "full_development_parameter_lock.json",
        REPORTS / "source_availability_decision.json", REPORTS / "technical_report.md",
        OUT / "final_performance_metrics.parquet", OUT / "registered_month_direction_metrics.parquet",
        OUT / "availability_eta_compensation_diagnostic.parquet",
    ]
    final_lock = {
        "scenario_id": "20260820_12", "status": "frozen_complete", "decision": decision["decision"],
        "selected_mechanism": decision["selected_mechanism"], "authoritative_outputs": hash_manifest(authoritative),
        "A1": "closed_pending_new_registered_hypothesis", "A2": "closed",
    }
    dump_json(HERE / "final_lock.json", final_lock)


if __name__ == "__main__":
    main()
