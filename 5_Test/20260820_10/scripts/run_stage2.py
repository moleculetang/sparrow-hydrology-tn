from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from hleg_shared import (
    BETA_GRID,
    FOLD_PATH,
    NONINFERIOR_MARGIN,
    OUT,
    REPORTS,
    development_observations,
    dump_json,
    formal_specs,
    metric_values,
    paired_bootstrap,
    parent_shared,
    require_runtime,
    select_beta,
    station_macro_rmse,
    tree_macro_rmse,
)


def tree_sign_flip(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    keys = ["station_key", "year", "month", "fold_id"]
    joined = reference[keys + ["terminal_tree_id", "tn_mg_l", "pred_tn_mg_l"]].merge(
        candidate[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_reference", "_candidate"), validate="one_to_one"
    )
    deltas = []
    for tree, group in joined.groupby("terminal_tree_id"):
        obs = np.log1p(group.tn_mg_l.to_numpy(float))
        ref = float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_reference) - obs) ** 2)))
        cand = float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate) - obs) ** 2)))
        deltas.append((int(tree), cand - ref))
    values = np.asarray([value for _, value in deltas], dtype=float)
    if len(values) > 12:
        raise RuntimeError("sign-flip sensitivity is registered only for the eight observed terminal trees")
    distribution = np.asarray([
        np.mean(values * np.asarray(signs, dtype=float))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    observed = float(values.mean())
    p_two = float(np.mean(np.abs(distribution) >= abs(observed)))
    return {
        "tree_count": int(len(values)),
        "enumerations": int(len(distribution)),
        "observed_mean_tree_delta": observed,
        "two_sided_randomization_p": p_two,
        "tree_deltas": [{"terminal_tree_id": tree, "delta_rmse_log1p": value} for tree, value in deltas],
    }


def comparison_state(a_to_b: dict[str, object], b_to_a: dict[str, object] | None = None) -> str:
    if bool(a_to_b["predictively_improved"]):
        return "improved"
    if bool(a_to_b["noninferior"]):
        if b_to_a is None or not bool(b_to_a.get("predictively_improved", False)):
            return "approximately_equal_noninferior_not_improved"
    if b_to_a is not None and bool(b_to_a.get("predictively_improved", False)):
        return "worse_reference_improved"
    return "comparison_inconclusive"


def main() -> None:
    require_runtime()
    stage1 = json.loads((REPORTS / "stage1_mechanism_audit.json").read_text(encoding="utf-8"))
    if stage1["status"] != "PASS":
        raise RuntimeError("STOP_STAGE1_NOT_PASS")
    shared = parent_shared()
    paths = pd.read_parquet(OUT / "candidate_development_observation_paths.parquet")
    observations = development_observations()
    if len(observations) != paths.loc[(paths.model_id == paths.model_id.iloc[0]) & paths.mechanism.eq("PARENT")].shape[0]:
        raise RuntimeError("candidate observation-key count mismatch")
    folds = pd.read_parquet(FOLD_PATH)[["fold_id", "train_start_year", "train_end_year", "evaluation_year"]].drop_duplicates().sort_values("evaluation_year")
    if folds.evaluation_year.tolist() != [2018, 2019, 2020, 2021]:
        raise RuntimeError("OOF year contract violation")

    predictions: list[pd.DataFrame] = []
    selection_surface: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    for model_index, spec in enumerate(formal_specs(), start=1):
        model_id = str(spec["model_id"])
        model_paths = paths.loc[paths.model_id.eq(model_id)]
        for fold in folds.itertuples():
            for mechanism in ("PARENT", "HLEG_BULK", "HLEG_AGE"):
                candidates = model_paths.loc[model_paths.mechanism.eq(mechanism)]
                if mechanism == "PARENT":
                    selected_beta = 0.0
                    surface = pd.DataFrame([{"beta_h": 0.0, "training_station_macro_rmse_log1p": np.nan}])
                    train = candidates.loc[candidates.year.between(fold.train_start_year, fold.train_end_year)].copy()
                    eta_p1, effects_p1, diag_p1 = shared.fit_readout(train, "P1")
                else:
                    selected_beta, surface, eta_p1, effects_p1 = select_beta(
                        shared, candidates, int(fold.train_start_year), int(fold.train_end_year), layer="P1"
                    )
                    train = candidates.loc[candidates.beta_h.eq(selected_beta) & candidates.year.between(fold.train_start_year, fold.train_end_year)].copy()
                    diag_p1 = surface.loc[surface.beta_h.eq(selected_beta)].iloc[0].to_dict()
                test = candidates.loc[candidates.beta_h.eq(selected_beta) & candidates.year.eq(fold.evaluation_year)].copy()
                pred_p1 = shared.predict_layer(test, "P1", eta_p1, effects_p1)
                pred_p1["fold_id"] = fold.fold_id
                pred_p1["selected_beta_h"] = selected_beta
                predictions.append(pred_p1)
                parameter_rows.append({
                    "model_id": model_id, "mechanism": mechanism, "fold_id": fold.fold_id,
                    "evaluation_year": int(fold.evaluation_year), "layer": "P1", "selected_beta_h": selected_beta,
                    "eta_quick": float(eta_p1[0]), "eta_gw": float(eta_p1[1]),
                    "eta_boundary": bool(np.any(eta_p1 <= 0.01) or np.any(eta_p1 >= 0.99)),
                })
                surface["model_id"] = model_id
                surface["mechanism"] = mechanism
                surface["fold_id"] = fold.fold_id
                surface["evaluation_year"] = int(fold.evaluation_year)
                surface["selected"] = surface.beta_h.eq(selected_beta)
                selection_surface.append(surface)

                # P2 receives the P1-selected beta but independently re-estimates the full readout.
                eta_p2, effects_p2, diag_p2 = shared.fit_readout(train, "P2")
                pred_p2 = shared.predict_layer(test, "P2", eta_p2, effects_p2)
                pred_p2["fold_id"] = fold.fold_id
                pred_p2["selected_beta_h"] = selected_beta
                predictions.append(pred_p2)
                parameter_rows.append({
                    "model_id": model_id, "mechanism": mechanism, "fold_id": fold.fold_id,
                    "evaluation_year": int(fold.evaluation_year), "layer": "P2", "selected_beta_h": selected_beta,
                    "eta_quick": float(eta_p2[0]), "eta_gw": float(eta_p2[1]),
                    "eta_boundary": bool(diag_p2["eta_boundary"]),
                })
                effect_rows.extend([
                    {"model_id": model_id, "mechanism": mechanism, "fold_id": fold.fold_id, "station_key": station, "station_effect": value}
                    for station, value in effects_p2.items()
                ])
        print(f"stage2 temporal {model_index:02d}/12 {model_id}", flush=True)

    oof = pd.concat(predictions, ignore_index=True)
    hydro = pd.read_parquet(OUT / "station_upstream_hydrologic_state_registry_1961_2022.parquet")
    oof = oof.merge(
        hydro[["reach_id", "year", "month", "H_up_standardized"]],
        on=["reach_id", "year", "month"], how="left", validate="many_to_one",
    )
    oof["hydrologic_state"] = np.where(
        oof.H_up_standardized >= 1.0, "high_H", np.where(oof.H_up_standardized <= -1.0, "low_H", "middle_H")
    )
    oof.to_parquet(OUT / "temporal_oof_predictions.parquet", index=False)
    pd.concat(selection_surface, ignore_index=True).to_parquet(OUT / "candidate_fold_beta_surface.parquet", index=False)
    pd.DataFrame(parameter_rows).to_parquet(OUT / "candidate_fold_readout_parameters.parquet", index=False)
    pd.DataFrame(effect_rows).to_parquet(OUT / "candidate_fold_station_effects.parquet", index=False)

    metric_rows = []
    for (model_id, mechanism, layer), group in oof.groupby(["model_id", "mechanism", "layer"]):
        metric_rows.append({"model_id": model_id, "mechanism": mechanism, "layer": layer, "scope": "pooled", **metric_values(group)})
        metric_rows.append({"model_id": model_id, "mechanism": mechanism, "layer": layer, "scope": "station_macro", "rmse_log1p": station_macro_rmse(group), "n": len(group)})
        metric_rows.append({"model_id": model_id, "mechanism": mechanism, "layer": layer, "scope": "terminal_tree_macro", "rmse_log1p": tree_macro_rmse(group), "n": len(group)})
    pd.DataFrame(metric_rows).to_parquet(OUT / "temporal_oof_metrics.parquet", index=False)

    comparison_rows = []
    state_rows = []
    signflip_rows = []
    seed_offset = 0
    for spec in formal_specs():
        model_id = str(spec["model_id"])
        for layer in ("P1", "P2"):
            groups = {
                mechanism: oof.loc[oof.model_id.eq(model_id) & oof.mechanism.eq(mechanism) & oof.layer.eq(layer)]
                for mechanism in ("PARENT", "HLEG_BULK", "HLEG_AGE")
            }
            for candidate_mechanism, reference_mechanism in (
                ("HLEG_BULK", "PARENT"), ("HLEG_AGE", "PARENT"), ("HLEG_AGE", "HLEG_BULK"),
            ):
                candidate, reference = groups[candidate_mechanism], groups[reference_mechanism]
                for block in ("station_key", "terminal_tree_id"):
                    seed_offset += 1
                    result = paired_bootstrap(reference, candidate, block, seed_offset)
                    comparison_rows.append({
                        "model_id": model_id, "candidate": candidate_mechanism, "reference": reference_mechanism,
                        "evaluation": "temporal_oof_2018_2021", "layer": layer, "block": block, **result,
                    })
                if layer == "P1":
                    signflip = tree_sign_flip(reference, candidate)
                    signflip_rows.append({
                        "model_id": model_id, "candidate": candidate_mechanism, "reference": reference_mechanism,
                        "tree_count": signflip["tree_count"], "enumerations": signflip["enumerations"],
                        "observed_mean_tree_delta": signflip["observed_mean_tree_delta"],
                        "two_sided_randomization_p": signflip["two_sided_randomization_p"],
                    })
                    for state in ("high_H", "low_H"):
                        ref_state = reference.loc[reference.hydrologic_state.eq(state)]
                        cand_state = candidate.loc[candidate.hydrologic_state.eq(state)]
                        seed_offset += 1
                        result = paired_bootstrap(ref_state, cand_state, "station_key", seed_offset)
                        state_rows.append({
                            "model_id": model_id, "candidate": candidate_mechanism, "reference": reference_mechanism,
                            "hydrologic_state": state, **result,
                        })
    comparisons = pd.DataFrame(comparison_rows)
    states = pd.DataFrame(state_rows)
    comparisons.to_parquet(OUT / "temporal_paired_comparisons.parquet", index=False)
    states.to_parquet(OUT / "hydrologic_state_performance.parquet", index=False)
    pd.DataFrame(signflip_rows).to_parquet(OUT / "terminal_tree_signflip_sensitivity.parquet", index=False)

    params = pd.DataFrame(parameter_rows)
    p1_dynamic = params.loc[params.layer.eq("P1") & params.mechanism.ne("PARENT")]
    stability_rows = []
    for (model_id, mechanism), group in p1_dynamic.groupby(["model_id", "mechanism"]):
        values = group.selected_beta_h.to_numpy(float)
        positive = int((values > 0).sum())
        negative = int((values < 0).sum())
        direction = (
            "positive_stable" if positive >= 3 and negative == 0
            else "negative_stable" if negative >= 3 and positive == 0
            else "direction_unstable_or_zero"
        )
        stability_rows.append({
            "model_id": model_id, "mechanism": mechanism, "positive_folds": positive, "negative_folds": negative,
            "zero_folds": int((values == 0).sum()), "direction_status": direction,
            "boundary_folds": int((np.abs(values) == 1.0).sum()),
            "boundary_confounded": bool((np.abs(values) == 1.0).sum() >= 2),
        })
    stability = pd.DataFrame(stability_rows)
    stability.to_parquet(OUT / "beta_fold_stability.parquet", index=False)

    station_comparisons = comparisons.loc[comparisons.block.eq("station_key")]
    evidence = []
    for candidate, reference in (("HLEG_BULK", "PARENT"), ("HLEG_AGE", "PARENT"), ("HLEG_AGE", "HLEG_BULK")):
        subset = station_comparisons.loc[station_comparisons.candidate.eq(candidate) & station_comparisons.reference.eq(reference) & station_comparisons.layer.eq("P1")]
        subset_p2 = station_comparisons.loc[station_comparisons.candidate.eq(candidate) & station_comparisons.reference.eq(reference) & station_comparisons.layer.eq("P2")]
        state_subset = states.loc[states.candidate.eq(candidate) & states.reference.eq(reference)]
        high = state_subset.loc[state_subset.hydrologic_state.eq("high_H")]
        low = state_subset.loc[state_subset.hydrologic_state.eq("low_H")]
        evidence.append({
            "candidate": candidate, "reference": reference,
            "temporal_noninferior_models": int(subset.noninferior.sum()),
            "temporal_improved_models": int(subset.predictively_improved.sum()),
            "P2_temporal_noninferior_models": int(subset_p2.noninferior.sum()),
            "P2_temporal_improved_models": int(subset_p2.predictively_improved.sum()),
            "high_H_noninferior_models": int(high.noninferior.sum()),
            "low_H_noninferior_models": int(low.noninferior.sum()),
            "high_H_improved_models": int(high.predictively_improved.sum()),
            "low_H_improved_models": int(low.predictively_improved.sum()),
        })
    dump_json(REPORTS / "temporal_evidence_summary.json", {
        "scenario_id": "20260820_10",
        "primary_layer": "P1_globally_scaled_pathway_layer",
        "primary_metric": "station_macro_RMSE_ln1p_TN",
        "noninferiority_margin": NONINFERIOR_MARGIN,
        "comparisons": evidence,
        "OOF_keys_per_model_mechanism_layer": int(oof.groupby(["model_id", "mechanism", "layer"]).size().min()),
        "TN_2022_rows_materialized": 0,
    })


if __name__ == "__main__":
    main()
