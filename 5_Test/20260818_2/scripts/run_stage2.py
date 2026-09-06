from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STAGE1 = Path(r"E:\SPARROW\5_Test\20260818_1")
sys.path.insert(0, str(STAGE1 / "scripts"))
from legacy18_shared import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    FOLD_PATH,
    NONINFERIOR_MARGIN,
    OBS_PATH,
    OPERATORS,
    S16_4,
    S17_6,
    TEST,
    dump_json,
    formal_specs,
    hash_manifest,
    paired_rmse_bootstrap,
    parent_core,
    prepare_arrays,
    require_runtime,
    route_candidate,
    sha256,
    simulate_operator,
    spatial_holdout,
    station_macro_rmse,
    temporal_oof,
)


ROOT = TEST / "20260818_2"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
INTERFACE = TEST / "20260814_6" / "outputs" / "structural_canonical_main_interface_2006_2022.parquet"
PARENT_ROUTED = S16_4 / "outputs" / "candidate_routed_paths_2016_2021.parquet"
PARENT_OOF = S16_4 / "outputs" / "candidate_oof_predictions_2018_2021.parquet"


def q72_semantics() -> dict[str, object]:
    x = pd.read_parquet(INTERFACE)
    errors = {
        "positive_partition_max_abs_mm": float((x.positive_input_mm - x.quick_generated_mm - x.source_positive_input_to_store_mm).abs().max()),
        "quick_input_partition_max_abs_mm": float((x.quick_input_mm - x.quick_generated_mm - x.soil_overflow_to_quick_mm).abs().max()),
        "source_pre_recharge_max_abs_mm": float((x.source_store_pre_recharge_mm - x.source_store_start_mm - x.source_positive_input_to_store_mm + x.soil_overflow_to_quick_mm).abs().max()),
        "gw_recharge_max_abs_mm": float((x.gw_recharge_mm - x.k_p * x.source_store_pre_recharge_mm).abs().max()),
    }
    passed = max(errors.values()) <= 1e-9
    return {
        "status": "PASS" if passed else "FAIL",
        "quick_generated_includes_overflow": False if passed else None,
        "quick_generated_and_overflow_are_distinct_accounting_fluxes": bool(passed),
        "both_can_be_positive_same_month": True,
        "both_positive_rows": int(((x.quick_generated_mm > 0) & (x.soil_overflow_to_quick_mm > 0)).sum()),
        "rows": len(x),
        "identity_errors": errors,
        "parent_equation": "quick_input_mm = quick_generated_mm + soil_overflow_to_quick_mm",
    }


def compare_f00_oof(ours: pd.DataFrame, routed: pd.DataFrame) -> dict[str, object]:
    parent_oof = pd.read_parquet(PARENT_OOF)
    parent_routed = pd.read_parquet(PARENT_ROUTED)
    reports: dict[str, object] = {}
    passed = True
    for spec in formal_specs():
        model = str(spec["model_id"])
        a = ours.loc[(ours.operator.eq("F00")) & (ours.model_id.eq(model)) & (ours.layer.eq("P2"))].sort_values(["station_key", "year", "month"])
        b = parent_oof.loc[parent_oof.model_id.eq(model)].sort_values(["station_key", "year", "month"])
        key_equal = a[["station_key", "year", "month"]].reset_index(drop=True).equals(b[["station_key", "year", "month"]].reset_index(drop=True))
        pred_close = bool(np.allclose(a.pred_tn_mg_l, b.pred_tn_mg_l, rtol=1e-12, atol=1e-12))
        eta_close = bool(np.allclose(a[["eta_quick", "eta_gw"]], b[["eta_quick", "eta_gw"]], rtol=1e-12, atol=1e-12))
        ar = routed.loc[(routed.operator.eq("F00")) & routed.model_id.eq(model)].sort_values(["reach_id", "year", "month"])
        br = parent_routed.loc[parent_routed.model_id.eq(model)].sort_values(["reach_id", "year", "month"])
        route_close = all(
            np.allclose(ar[c], br[c], rtol=1e-12, atol=1e-9)
            for c in ("routed_quick_tn_kg_n", "routed_gw_tn_kg_n", "routed_water_volume_m3")
        )
        item_pass = len(a) == 4097 and key_equal and pred_close and eta_close and route_close
        reports[model] = {
            "rows": len(a),
            "key_equal": key_equal,
            "prediction_close": pred_close,
            "eta_close": eta_close,
            "routed_paths_close": bool(route_close),
            "pass": bool(item_pass),
        }
        passed &= item_pass
    return {"status": "PASS" if passed else "FAIL", "models": reports}


def compare_representative_states(
    reach_ids: np.ndarray,
    times: list[tuple[int, int]],
    arrays: dict[str, np.ndarray],
    early_positive: np.ndarray,
) -> dict[str, object]:
    core = parent_core()
    representatives = [
        ("S0_mu_036m", "S0", None, 36),
        ("S1_tau_012m_mu_036m", "S1", 12, 36),
        ("S1_tau_012m_mu_240m", "S1", 12, 240),
    ]
    common = [
        "son_state_end_kg_n", "mobile_state_end_kg_n", "quick_state_end_kg_n", "gw_state_end_kg_n",
        "quick_tn_release_kg_n", "gw_tn_release_kg_n", "local_tn_release_kg_n",
        "negative_removed_kg_n", "negative_unmet_kg_n",
    ]
    results: dict[str, object] = {}
    passed = True
    for model, structure, tau, mu in representatives:
        parent, _ = core.simulate_candidate_totals(model, structure, tau, mu, reach_ids, times, arrays, early_positive)
        ours, _, _, _ = simulate_operator("F00", model, structure, tau, mu, reach_ids, times, arrays, early_positive, 1961, 2021)
        parent = parent.loc[parent.year.le(2021)].sort_values(["year", "month", "reach_id"])
        ours = ours.sort_values(["year", "month", "reach_id"])
        key_equal = ours[["reach_id", "year", "month"]].reset_index(drop=True).equals(parent[["reach_id", "year", "month"]].reset_index(drop=True))
        column_results: dict[str, object] = {}
        item_pass = key_equal
        for column in common:
            close = bool(np.allclose(ours[column], parent[column], rtol=1e-12, atol=1e-9))
            column_results[column] = {
                "max_abs_difference": float(np.max(np.abs(ours[column].to_numpy(float) - parent[column].to_numpy(float)))),
                "pass": close,
            }
            item_pass &= close
        results[model] = {"key_equal": key_equal, "columns": column_results, "pass": bool(item_pass)}
        passed &= item_pass
    return {"status": "PASS" if passed else "FAIL", "models": results}


def subset_delta(parent: pd.DataFrame, candidate: pd.DataFrame) -> float:
    if len(parent) == 0 or len(candidate) == 0:
        return np.nan
    return station_macro_rmse(candidate) - station_macro_rmse(parent)


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parents = [INTERFACE, PARENT_ROUTED, PARENT_OOF, S17_6 / "final_lock.json", STAGE1 / "reports" / "verification.json"]
    hashes_start = hash_manifest(parents)
    dump_json(REPORTS / "parent_hashes_start.json", hashes_start)
    semantics = q72_semantics()
    dump_json(REPORTS / "q72_flux_semantics_audit.json", semantics)
    if semantics["status"] != "PASS":
        raise RuntimeError("STOP_F01_F11_Q72_FLUX_SEMANTICS_FAILED")

    reach_ids, times, arrays, early_positive = prepare_arrays()
    observations = pd.read_parquet(OBS_PATH)
    folds = pd.read_parquet(FOLD_PATH)
    regimes = pd.read_parquet(STAGE1 / "outputs" / "flow_regime_registry.parquet")
    regime_keep = [
        "station_key", "reach_id", "year", "month", "fold_id", "flow_regime",
        "high_quick_fraction", "wet_season", "routed_quick_fraction", "routed_total_water_m3",
    ]

    state_rows: list[pd.DataFrame] = []
    routed_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    parameter_rows: list[pd.DataFrame] = []
    effect_rows: list[pd.DataFrame] = []
    spatial_rows: list[pd.DataFrame] = []
    spatial_param_rows: list[pd.DataFrame] = []
    spinup_rows: list[dict[str, object]] = []
    spinup_state_rows: list[pd.DataFrame] = []
    engineering_rows: list[dict[str, object]] = []

    specs = formal_specs()
    total = len(OPERATORS) * len(specs)
    counter = 0
    for operator in OPERATORS:
        for spec in specs:
            counter += 1
            model = str(spec["model_id"])
            frame, engineering, spinup, spinup_states = simulate_operator(
                operator,
                model,
                str(spec["source_structure"]),
                spec["soil_tau_month"],
                int(spec["delivery_mu_month"]),
                reach_ids,
                times,
                arrays,
                early_positive,
            )
            candidate_id = f"{operator}__{model}"
            routed = route_candidate(frame, reach_ids)
            routed["operator"] = operator
            routed["model_id"] = model
            routed["candidate_id"] = candidate_id
            pred, params, effects = temporal_oof(routed.drop(columns=["operator", "model_id", "candidate_id"]), observations, folds, layers=("P1", "P2"))
            pred["operator"] = operator
            pred["model_id"] = model
            pred["candidate_id"] = candidate_id
            params["operator"] = operator
            params["model_id"] = model
            params["candidate_id"] = candidate_id
            effects["operator"] = operator
            effects["model_id"] = model
            effects["candidate_id"] = candidate_id
            pred = pred.merge(regimes[regime_keep], on=["station_key", "reach_id", "year", "month", "fold_id"], how="left", validate="many_to_one")
            sp_loso, spp_loso = spatial_holdout(routed.drop(columns=["operator", "model_id", "candidate_id"]), observations, "station_key")
            sp_loto, spp_loto = spatial_holdout(routed.drop(columns=["operator", "model_id", "candidate_id"]), observations, "terminal_tree_id")
            sp = pd.concat([sp_loso, sp_loto], ignore_index=True)
            sp = sp.loc[sp.layer.eq("P1")].copy()
            sp["spatial_scheme"] = np.where(sp.holdout_column.eq("station_key"), "LOSO", "LOTO")
            sp["fold_id"] = sp.spatial_scheme
            sp["operator"] = operator
            sp["model_id"] = model
            sp["candidate_id"] = candidate_id
            spp = pd.concat([spp_loso, spp_loto], ignore_index=True)
            spp = spp.loc[spp.layer.eq("P1")].copy()
            spp["operator"] = operator
            spp["model_id"] = model
            spp["candidate_id"] = candidate_id

            state_rows.append(frame)
            routed_rows.append(routed)
            prediction_rows.append(pred)
            parameter_rows.append(params)
            effect_rows.append(effects)
            spatial_rows.append(sp)
            spatial_param_rows.append(spp)
            spinup_rows.append(spinup)
            spinup_state_rows.append(spinup_states)
            engineering_rows.append(engineering)
            print(f"[{counter:02d}/{total}] {candidate_id}", flush=True)

    states = pd.concat(state_rows, ignore_index=True)
    routed_all = pd.concat(routed_rows, ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    parameters = pd.concat(parameter_rows, ignore_index=True)
    effects = pd.concat(effect_rows, ignore_index=True)
    spatial = pd.concat(spatial_rows, ignore_index=True)
    spatial_params = pd.concat(spatial_param_rows, ignore_index=True)
    spinups = pd.DataFrame(spinup_rows)
    spinup_states = pd.concat(spinup_state_rows, ignore_index=True)
    engineering = pd.DataFrame(engineering_rows)

    f00_oof = compare_f00_oof(predictions, routed_all)
    dump_json(REPORTS / "f00_all_model_oof_reproduction_audit.json", f00_oof)
    if f00_oof["status"] != "PASS":
        raise RuntimeError("STOP_F00_PARENT_NOT_REPRODUCED")
    representative = compare_representative_states(reach_ids, times, arrays, early_positive)
    dump_json(REPORTS / "f00_representative_state_flux_reproduction_audit.json", representative)
    if representative["status"] != "PASS":
        raise RuntimeError("STOP_F00_PARENT_NOT_REPRODUCED")

    states.to_parquet(OUT / "operator_state_flux_2016_2021.parquet", index=False)
    routed_all.to_parquet(OUT / "candidate_routed_paths_2016_2021.parquet", index=False)
    predictions.to_parquet(OUT / "candidate_oof_predictions_2018_2021.parquet", index=False)
    parameters.to_parquet(OUT / "candidate_fold_readout_parameters.parquet", index=False)
    effects.to_parquet(OUT / "candidate_station_effects.parquet", index=False)
    spatial.to_parquet(OUT / "candidate_spatial_holdout_predictions.parquet", index=False)
    spatial_params.to_parquet(OUT / "candidate_spatial_readout_parameters.parquet", index=False)
    spinups.to_parquet(OUT / "operator_spinup_audit.parquet", index=False)
    spinup_states.to_parquet(OUT / "operator_spinup_final_states.parquet", index=False)
    engineering.to_parquet(OUT / "operator_engineering_audit.parquet", index=False)

    comparison_rows: list[dict[str, object]] = []
    distributions: list[pd.DataFrame] = []
    matrix_rows: list[dict[str, object]] = []
    seed_offset = 0
    for operator in ("F10", "F01", "F11"):
        for spec in specs:
            model = str(spec["model_id"])
            parent_id = f"F00__{model}"
            candidate_id = f"{operator}__{model}"
            pbase = predictions.loc[predictions.candidate_id.eq(parent_id)]
            pcand = predictions.loc[predictions.candidate_id.eq(candidate_id)]
            sbase = spatial.loc[spatial.candidate_id.eq(parent_id)]
            scand = spatial.loc[spatial.candidate_id.eq(candidate_id)]
            gate: dict[str, object] = {"operator": operator, "model_id": model, "candidate_id": candidate_id}
            comparisons = [
                ("P2_temporal_station", pbase.loc[pbase.layer.eq("P2")], pcand.loc[pcand.layer.eq("P2")], "station_key"),
                ("P1_temporal_station", pbase.loc[pbase.layer.eq("P1")], pcand.loc[pcand.layer.eq("P1")], "station_key"),
                ("P1_temporal_tree", pbase.loc[pbase.layer.eq("P1")], pcand.loc[pcand.layer.eq("P1")], "terminal_tree_id"),
                ("P1_low_flow_station", pbase.loc[(pbase.layer.eq("P1")) & pbase.flow_regime.eq("low")], pcand.loc[(pcand.layer.eq("P1")) & pcand.flow_regime.eq("low")], "station_key"),
                ("P1_LOSO_station", sbase.loc[sbase.spatial_scheme.eq("LOSO")], scand.loc[scand.spatial_scheme.eq("LOSO")], "station_key"),
                ("P1_LOTO_tree", sbase.loc[sbase.spatial_scheme.eq("LOTO")], scand.loc[scand.spatial_scheme.eq("LOTO")], "terminal_tree_id"),
            ]
            for comparison_id, pb, pc, block in comparisons:
                seed_offset += 1
                dist, summary = paired_rmse_bootstrap(pb, pc, block, seed_offset)
                comparison_rows.append({"operator": operator, "model_id": model, "candidate_id": candidate_id, "comparison_id": comparison_id, "block": block, **summary})
                distributions.append(pd.DataFrame({
                    "operator": operator,
                    "model_id": model,
                    "comparison_id": comparison_id,
                    "replicate": np.arange(BOOTSTRAP_REPLICATES, dtype=int),
                    "delta_rmse_log1p": dist,
                }))
                gate[comparison_id + "_noninferior"] = bool(summary["noninferior"])
                gate[comparison_id + "_clear_failure"] = bool(summary["clear_failure"])
                gate[comparison_id + "_point_delta"] = float(summary["point_delta"])
            ph = pbase.loc[(pbase.layer.eq("P1")) & pbase.flow_regime.eq("high")]
            ch = pcand.loc[(pcand.layer.eq("P1")) & pcand.flow_regime.eq("high")]
            pq = pbase.loc[(pbase.layer.eq("P1")) & pbase.high_quick_fraction]
            cq = pcand.loc[(pcand.layer.eq("P1")) & pcand.high_quick_fraction]
            gate["P1_high_flow_point_delta"] = subset_delta(ph, ch)
            gate["P1_high_flow_point_improved"] = bool(gate["P1_high_flow_point_delta"] < 0)
            gate["P1_high_quick_point_delta"] = subset_delta(pq, cq)
            gate["P1_high_quick_point_improved"] = bool(gate["P1_high_quick_point_delta"] < 0)
            gate["P1_spatial_noninferior"] = bool(gate["P1_LOSO_station_noninferior"] and gate["P1_LOTO_tree_noninferior"])
            gate["no_clear_failure"] = not any(bool(v) for k, v in gate.items() if k.endswith("_clear_failure"))
            matrix_rows.append(gate)

    comparisons = pd.DataFrame(comparison_rows)
    bootstrap = pd.concat(distributions, ignore_index=True)
    matrix = pd.DataFrame(matrix_rows)
    comparisons.to_parquet(OUT / "paired_gate_metrics.parquet", index=False)
    bootstrap.to_parquet(OUT / "paired_bootstrap_distributions.parquet", index=False)
    matrix.to_parquet(OUT / "operator_model_gate_matrix.parquet", index=False)

    operator_rows: list[dict[str, object]] = []
    for operator, group in matrix.groupby("operator"):
        s0 = group.loc[group.model_id.str.startswith("S0_")]
        p2_count = int(group.P2_temporal_station_noninferior.sum())
        p1_count = int((group.P1_temporal_station_noninferior & group.P1_temporal_tree_noninferior).sum())
        spatial_count = int(group.P1_spatial_noninferior.sum())
        high_count = int(group.P1_high_flow_point_improved.sum())
        quick_count = int(group.P1_high_quick_point_improved.sum())
        low_count = int(group.P1_low_flow_station_noninferior.sum())
        screening = bool(
            s0.P2_temporal_station_noninferior.sum() >= 5
            and (s0.P1_temporal_station_noninferior & s0.P1_temporal_tree_noninferior).sum() >= 5
            and s0.P1_high_flow_point_improved.sum() >= 4
            and s0.P1_high_quick_point_improved.sum() >= 4
            and s0.P1_spatial_noninferior.sum() >= 5
            and s0.no_clear_failure.all()
        )
        final_pass = bool(
            screening and p2_count >= 10 and p1_count >= 10 and spatial_count >= 10
            and high_count >= 8 and quick_count >= 8 and low_count >= 10
            and group.no_clear_failure.all()
        )
        p1_params = parameters.loc[(parameters.operator.eq(operator)) & parameters.layer.eq("P1")]
        repeated = p1_params.groupby("model_id").eta_boundary.sum().ge(3)
        confounded_models = int(repeated.sum())
        operator_rows.append({
            "operator": operator,
            "screening_pass": screening,
            "final_gate_pass": final_pass,
            "p2_temporal_noninferior_count": p2_count,
            "p1_temporal_noninferior_count": p1_count,
            "p1_spatial_noninferior_count": spatial_count,
            "p1_high_flow_improved_count": high_count,
            "p1_high_quick_improved_count": quick_count,
            "p1_low_flow_noninferior_count": low_count,
            "clear_failure_count": int((~group.no_clear_failure).sum()),
            "pathway_boundary_repeated_model_count": confounded_models,
            "pathway_scaling_status": "confounded" if confounded_models >= 6 else "stable",
            "median_P1_temporal_delta": float(group.P1_temporal_station_point_delta.median()),
            "median_P1_high_flow_delta": float(group.P1_high_flow_point_delta.median()),
            "median_P1_spatial_delta": float(group[["P1_LOSO_station_point_delta", "P1_LOTO_tree_point_delta"]].mean(axis=1).median()),
        })
    operator_summary = pd.DataFrame(operator_rows)
    operator_summary.to_parquet(OUT / "operator_gate_summary.parquet", index=False)
    eligible = operator_summary.loc[operator_summary.final_gate_pass].copy()
    if len(eligible):
        eligible["switch_count"] = eligible.operator.map({"F10": 1, "F01": 1, "F11": 2})
        eligible = eligible.sort_values(
            ["p1_temporal_noninferior_count", "median_P1_temporal_delta", "median_P1_high_flow_delta", "median_P1_spatial_delta", "switch_count", "operator"],
            ascending=[False, True, True, True, True, True],
        )
        selected = str(eligible.iloc[0].operator)
        scaling = str(eligible.iloc[0].pathway_scaling_status)
        status = "process_consistent_but_pathway_scaling_confounded" if scaling == "confounded" else "source_water_operator_supported"
    else:
        selected = "F00"
        scaling = "not_applicable_parent_retained"
        compensated = operator_summary.loc[(operator_summary.p2_temporal_noninferior_count >= 10) & (~operator_summary.final_gate_pass)]
        status = "readout_compensated_not_process_supported" if len(compensated) else "no_alternative_supported_parent_retained"
    decision = {
        "scenario_id": "20260818_2",
        "selected_operator": selected,
        "operator_status": status,
        "pathway_scaling_status": scaling,
        "formal_mu_reselected": False,
        "formal_model_count": 12,
        "operator_count": 4,
        "F00_all_model_oof_reproduction": f00_oof["status"],
        "F00_representative_state_flux_reproduction": representative["status"],
        "all_spinups_converged": bool(spinups.converged.all()),
        "all_mass_balance_relative_le_1e_12": bool(engineering.max_relative_mass_balance_error.le(1e-12).all()),
        "locked_2022_used": False,
    }
    dump_json(REPORTS / "source_water_operator_decision.json", decision)
    dump_json(REPORTS / "numerical_gate_contract.json", {
        "delta_definition": "candidate_RMSE_minus_parent_RMSE",
        "point_improved": "delta < 0",
        "point_nonworse": "delta <= 0",
        "noninferior": "paired_bootstrap_CI95_upper < 0.005",
        "predictively_improved": "paired_bootstrap_CI95_upper < 0",
        "clear_failure": "delta > 0.01 and CI95_lower > 0",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "noninferiority_margin_log_rmse": NONINFERIOR_MARGIN,
    })
    hashes_end = hash_manifest(parents)
    dump_json(REPORTS / "parent_hashes_end.json", hashes_end)
    dump_json(REPORTS / "completion_audit.json", {
        "status": "PASS" if hashes_start == hashes_end else "FAIL",
        "parent_hashes_unchanged": hashes_start == hashes_end,
        "candidates": int(routed_all.candidate_id.nunique()),
        "spinups": len(spinups),
        "all_spinups_converged": bool(spinups.converged.all()),
        "F00_reproduced": f00_oof["status"] == "PASS" and representative["status"] == "PASS",
        "selected_operator": selected,
    })


if __name__ == "__main__":
    main()
