from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(r"E:\SPARROW\5_Test\20260820_1\scripts")))
from legacy20_shared import (  # noqa: E402
    PARENT_PRED_PATH, S20_2, STATION_BOOTSTRAP_SEED, TREE_BOOTSTRAP_SEED,
    dump_json, formal_registry, hash_manifest, paired_block_bootstrap, require_runtime,
)
from reaction20_core import fit_fold, load_context, path_diagnostics


S20_3 = Path(r"E:\SPARROW\5_Test\20260820_3")


def main() -> None:
    require_runtime()
    out = S20_3 / "outputs"
    reports = S20_3 / "reports"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    parent_paths = [
        PARENT_PRED_PATH, Path(r"E:\SPARROW\5_Test\20260820_1\experiment_contract.json"),
        S20_2 / "outputs" / "reach_month_hydraulic_exposure.parquet",
        S20_2 / "outputs" / "source_target_path_exposure_2006_2022.parquet",
    ]
    start = hash_manifest(parent_paths)
    dump_json(reports / "parent_hashes_start.json", start)
    context = load_context()
    models = formal_registry()
    fold_defs = context.folds[["fold_id", "train_start_year", "train_end_year", "evaluation_year"]].drop_duplicates().sort_values("evaluation_year")
    predictions, parameters, effects, local_rows, audits = [], [], [], [], []
    for spec_row in models.itertuples(index=False):
        spec = {"model_id": spec_row.model_id, "source_structure": spec_row.source_structure, "soil_tau_month": spec_row.soil_tau_month, "delivery_mu_month": int(spec_row.effective_tn_delivery_mu_month)}
        for layer in ("P1", "P2"):
            for fold in fold_defs.itertuples(index=False):
                pred, diagnostic, effect_map, local_eval, audit = fit_fold(
                    context, spec, fold, layer, 1.0, 1.0, "GW_STATIC_2020", "AQ_CLIM",
                )
                pred["model_id"] = spec_row.model_id
                pred["candidate_id"] = "reaction_null_q10gw1_q10aq1"
                path = path_diagnostics(context, local_eval, pred)
                pred = pred.merge(path, left_on=["reach_id", "year", "month"], right_on=["target_reach_id", "year", "month"], how="left", validate="many_to_one").drop(columns="target_reach_id")
                predictions.append(pred)
                parameters.append({"model_id": spec_row.model_id, "candidate_id": "reaction_null_q10gw1_q10aq1", **diagnostic})
                for station, value in effect_map.items():
                    effects.append({"model_id": spec_row.model_id, "fold_id": fold.fold_id, "layer": layer, "station_key": station, "station_effect": value})
                local_eval["model_id"] = spec_row.model_id
                local_eval["fold_id"] = fold.fold_id
                local_eval["layer"] = layer
                local_rows.append(local_eval)
                audits.append({"model_id": spec_row.model_id, "fold_id": fold.fold_id, "layer": layer, "local_max_abs_mass_balance_error_kg_n": audit["local"]["max_abs_mass_balance_error_kg_n"], "local_max_relative_mass_balance_error": audit["local"]["max_relative_mass_balance_error"], "routing_max_relative_closure": audit["routing"]["max_positive_mass_relative_closure"], "routing_max_zero_mass_abs_closure_kg_n": audit["routing"]["max_zero_mass_absolute_closure_kg_n"]})
                print(json.dumps({"model": spec_row.model_id, "layer": layer, "fold": fold.fold_id, "s": [diagnostic["s_gw_refmonth_20c"], diagnostic["s_aq_refmonth_20c"]]}, ensure_ascii=False), flush=True)
    candidate = pd.concat(predictions, ignore_index=True)
    params = pd.DataFrame(parameters)
    effects_frame = pd.DataFrame(effects)
    local_frame = pd.concat(local_rows, ignore_index=True)
    audit_frame = pd.DataFrame(audits)
    candidate.to_parquet(out / "reaction_null_oof_predictions_2018_2021.parquet", index=False)
    params.to_parquet(out / "candidate_fold_reaction_parameters.parquet", index=False)
    effects_frame.to_parquet(out / "candidate_station_effects.parquet", index=False)
    local_frame.to_parquet(out / "candidate_fold_local_releases.parquet", index=False)
    audit_frame.to_parquet(out / "reaction_mass_balance_audit.parquet", index=False)

    parent = pd.read_parquet(PARENT_PRED_PATH)
    parent = parent.loc[parent.layer.isin(["P1", "P2"])].copy()
    parent["terminal_tree_id"] = parent.terminal_tree_id.astype(int)
    gate_rows, distributions = [], []
    for (model_id, layer), cand in candidate.groupby(["model_id", "layer"]):
        par = parent.loc[parent.model_id.eq(model_id) & parent.layer.eq(layer)].copy()
        for scheme, block, seed in (("station", "station_key", STATION_BOOTSTRAP_SEED), ("observed_terminal_tree", "terminal_tree_id", TREE_BOOTSTRAP_SEED)):
            dist, summary = paired_block_bootstrap(par, cand, block, seed)
            gate_rows.append({"model_id": model_id, "layer": layer, "scheme": scheme, "subset": "all", **summary})
            distributions.append(pd.DataFrame({"model_id": model_id, "layer": layer, "scheme": scheme, "subset": "all", "replicate": np.arange(len(dist)), "delta_rmse_log1p": dist}))
            low = cand.loc[cand.low_ab_subset.fillna(False)].copy()
            key = ["station_key", "year", "month", "fold_id", "layer", "model_id"]
            if len(low) and low[block].nunique() >= 2:
                par_low = par.merge(low[key], on=key, how="inner", validate="one_to_one")
                dist_low, summary_low = paired_block_bootstrap(par_low, low, block, seed + 100)
                gate_rows.append({"model_id": model_id, "layer": layer, "scheme": scheme, "subset": "low_AB_F_le_0.05", **summary_low})
                distributions.append(pd.DataFrame({"model_id": model_id, "layer": layer, "scheme": scheme, "subset": "low_AB_F_le_0.05", "replicate": np.arange(len(dist_low)), "delta_rmse_log1p": dist_low}))
    gates = pd.DataFrame(gate_rows)
    gates.to_parquet(out / "reaction_architecture_gate_metrics.parquet", index=False)
    pd.concat(distributions, ignore_index=True).to_parquet(out / "paired_bootstrap_distributions.parquet", index=False)

    all_gate = gates.loc[gates.subset.eq("all")]
    model_gate = all_gate.pivot_table(index=["model_id", "layer"], columns="scheme", values=["noninferior", "direction_improved"], aggfunc="first").reset_index()
    model_gate.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else col for col in model_gate.columns]
    boundary = params.groupby(["model_id", "layer"], as_index=False).agg(s_gw_boundary_folds=("s_gw_boundary", "sum"), s_aq_boundary_folds=("s_aq_boundary", "sum"))
    timing = candidate.groupby(["model_id", "layer"], as_index=False).agg(mean_path_mass_fraction_exceeding_month=("path_mass_fraction_exceeding_month", "mean"), fraction_observations_timing_over_half=("path_mass_fraction_exceeding_month", lambda x: float(np.mean(x > 0.5))))
    timing["timing_class"] = np.where(timing.mean_path_mass_fraction_exceeding_month <= 0.05, "broadly_consistent", np.where(timing.mean_path_mass_fraction_exceeding_month <= 0.50, "quasi_steady_only", "timing_inconsistent"))
    model_gate = model_gate.merge(boundary, on=["model_id", "layer"]).merge(timing, on=["model_id", "layer"])
    model_gate["boundary_not_persistent_3of4"] = (model_gate.s_gw_boundary_folds < 3) & (model_gate.s_aq_boundary_folds < 3)
    model_gate["formal_model_gate"] = model_gate.noninferior_station & model_gate.noninferior_observed_terminal_tree & model_gate.boundary_not_persistent_3of4 & ~model_gate.timing_class.eq("timing_inconsistent")
    model_gate.to_parquet(out / "reaction_architecture_model_gate_matrix.parquet", index=False)

    summary = model_gate.groupby("layer", as_index=False).agg(models_passing=("formal_model_gate", "sum"), station_direction_improved=("direction_improved_station", "sum"), tree_direction_improved=("direction_improved_observed_terminal_tree", "sum"), timing_inconsistent_models=("timing_class", lambda x: int(np.sum(x == "timing_inconsistent"))))
    summary["layer_supported"] = (summary.models_passing >= 10) & (summary.station_direction_improved >= 10)
    summary.to_parquet(out / "reaction_architecture_layer_summary.parquet", index=False)
    p1 = bool(summary.loc[summary.layer.eq("P1"), "layer_supported"].iloc[0])
    p2 = bool(summary.loc[summary.layer.eq("P2"), "layer_supported"].iloc[0])
    decision = {
        "status": "reaction_architecture_supported" if p1 and p2 else "reaction_architecture_not_supported",
        "P1_reaction_process_layer_supported": p1,
        "P2_predictive_preservation_supported": p2,
        "eta_q_eta_gw_replaced_not_stacked": True,
        "temperature_stages_authorized": bool(p1 and p2),
        "criterion": "at least 10/12 model gates and at least 10/12 station primary D<0 in both P1 and P2",
    }
    dump_json(reports / "reaction_architecture_decision.json", decision)
    completion_checks = {
        "candidate_rows": len(candidate) == 12 * 2 * 4097,
        "all_optimizer_calls_dynamic": bool((params.optimizer_dynamic_replay_count > 0).all()),
        "all_evaluations_fresh_replay": bool(params.fresh_evaluation_replay.all()),
        "eta_fixed_one": bool((params.eta_quick.eq(1) & params.eta_gw.eq(1)).all()),
        "mass_balance_pass": bool((audit_frame.local_max_relative_mass_balance_error <= 1e-12).all() and (audit_frame.routing_max_relative_closure <= 1e-12).all()),
        "parent_hashes_unchanged": hash_manifest(parent_paths) == start,
    }
    dump_json(reports / "completion_audit.json", {
        "status": "PASS" if all(completion_checks.values()) else "FAIL",
        "candidate_rows": len(candidate),
        "expected_candidate_rows": 12 * 2 * 4097,
        **completion_checks,
    })
    dump_json(reports / "parent_hashes_end.json", hash_manifest(parent_paths))
    (S20_3 / "experiment_contract.json").write_text(json.dumps({
        "stage": "20260820_3", "question": "same-two-parameter reaction architecture vs frozen eta parent",
        "candidate": {"q10_gw": 1.0, "q10_aq": 1.0, "eta_q": 1.0, "eta_gw": 1.0},
        "optimizer": "each objective evaluation reruns parameter-specific pre1961 equilibrium and complete 1961-to-fold N states",
        "R1a": "upstream full exposure plus local actual midpoint-to-outlet exposure",
        "parent": "matched model/fold/layer frozen eta model",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not all(completion_checks.values()):
        raise RuntimeError(f"stage3 completion audit failed: {[k for k, v in completion_checks.items() if not v]}")
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
