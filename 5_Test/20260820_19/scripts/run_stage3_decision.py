from __future__ import annotations

import json

import numpy as np
import pandas as pd

import hierarchical19_shared as h


def build_temporal_h1_readouts() -> pd.DataFrame:
    shared = h.parent_shared()
    obs = h.development_observations()
    folds = h.fold_registry()
    parameters = pd.read_parquet(h.OUT / "temporal_fold_parameters.parquet")
    parts = []
    for model_id in h.FORMAL_MODELS:
        router = h.build_router(model_id, shared)
        for fold in folds.itertuples(index=False):
            fold_id = str(fold.fold_id)
            train_obs = obs.loc[obs.year.between(int(fold.train_start_year), int(fold.train_end_year))]
            test_obs = obs.loc[obs.year.eq(int(fold.evaluation_year))]
            row = parameters.loc[
                parameters.model_id.eq(model_id) & parameters.fold_id.eq(fold_id)
                & parameters.structure.eq("H1_GLOBAL")
            ].iloc[0]
            vf_h1 = np.full(len(router.reach_ids), float(row.v0_m_per_day))
            for mechanism, vf in (("H0_PARENT", np.zeros(len(router.reach_ids))), ("H1_GLOBAL", vf_h1)):
                train = router.frame(train_obs, vf)
                test = router.frame(test_obs, vf)
                for layer in ("P2", "P2R"):
                    fit = h.fit_selected_readout(train, layer, shared)
                    pred = h.predict_selected(test, layer, fit, shared)
                    pred["model_id"] = model_id
                    pred["fold_id"] = fold_id
                    pred["mechanism"] = mechanism
                    parts.append(pred)
    return pd.concat(parts, ignore_index=True)


def comparisons(predictions: pd.DataFrame, mechanisms: tuple[str, str], label: str, seed_base: int) -> pd.DataFrame:
    rows = []
    seed = seed_base
    reference_name, candidate_name = mechanisms
    for model_id in h.FORMAL_MODELS:
        for layer in sorted(predictions.layer.unique()):
            subset = predictions.loc[predictions.model_id.eq(model_id) & predictions.layer.eq(layer)]
            reference = subset.loc[subset.mechanism.eq(reference_name)]
            candidate = subset.loc[subset.mechanism.eq(candidate_name)]
            for block in ("station_key", "terminal_tree_id"):
                seed += 1
                rows.append({
                    "comparison": label, "model_id": model_id, "layer": layer,
                    "block": "station" if block == "station_key" else "tree",
                    **h.paired_bootstrap(reference, candidate, block, seed),
                })
    return pd.DataFrame(rows)


def main() -> None:
    h.require_runtime()
    nested_audit = json.loads((h.REPORTS / "stage2_nested_audit.json").read_text(encoding="utf-8"))
    if nested_audit["status"] != "PASS" or nested_audit["TN_2022_read"]:
        raise RuntimeError("STOP_NESTED_NOT_PASS")

    temporal_readout = build_temporal_h1_readouts()
    temporal_readout.to_parquet(h.OUT / "temporal_h1_readout_predictions.parquet", index=False)
    temporal_readout_gates = comparisons(
        temporal_readout, ("H0_PARENT", "H1_GLOBAL"), "H1_vs_H0", 2026083000
    )
    temporal_readout_gates.to_parquet(h.OUT / "temporal_h1_readout_gates.parquet", index=False)

    temporal_process = pd.read_parquet(h.OUT / "temporal_gate_matrix.parquet")
    nested_gates = pd.read_parquet(h.OUT / "nested_spatial_metrics.parquet")
    nested_predictions = pd.read_parquet(h.OUT / "nested_spatial_predictions.parquet")
    regional_rows = []
    seed = 2026084000
    for model_id in h.FORMAL_MODELS:
        for evaluation, block in (("LOSO", "station_key"), ("LOTO", "terminal_tree_id")):
            subset = nested_predictions.loc[
                nested_predictions.model_id.eq(model_id)
                & nested_predictions.evaluation.eq(evaluation)
                & nested_predictions.mechanism.eq("NESTED_SELECTED")
            ]
            reference = subset.loc[subset.layer.eq("P1")]
            candidate = subset.loc[subset.layer.eq("P2R")]
            seed += 1
            regional_rows.append({
                "comparison": "P2R_vs_P1_on_H1", "model_id": model_id,
                "evaluation": evaluation, **h.paired_bootstrap(reference, candidate, block, seed),
            })
    regional = pd.DataFrame(regional_rows)
    regional.to_parquet(h.OUT / "p2r_absolute_spatial_gate.parquet", index=False)

    def count(frame: pd.DataFrame, filters: dict[str, str], column: str) -> int:
        mask = np.ones(len(frame), dtype=bool)
        for key, value in filters.items():
            mask &= frame[key].eq(value).to_numpy()
        return int(frame.loc[mask, column].sum())

    temporal_p1 = {
        f"{block}_{metric}": count(temporal_process, {"layer": "P1", "mechanism": "H1_GLOBAL", "block": block}, metric)
        for block in ("station", "tree") for metric in ("noninferior", "predictively_improved")
    }
    temporal_p2 = {
        f"{block}_{metric}": count(temporal_readout_gates, {"layer": "P2", "block": block}, metric)
        for block in ("station", "tree") for metric in ("noninferior", "predictively_improved")
    }
    nested_p1 = {
        f"{evaluation}_{metric}": count(nested_gates, {"layer": "P1", "evaluation": evaluation}, metric)
        for evaluation in ("LOSO", "LOTO") for metric in ("noninferior", "predictively_improved")
    }
    p2r_absolute = {
        f"{evaluation}_{metric}": count(regional, {"evaluation": evaluation}, metric)
        for evaluation in ("LOSO", "LOTO") for metric in ("noninferior", "predictively_improved")
    }
    params = pd.read_parquet(h.OUT / "temporal_fold_parameters.parquet")
    h1 = params.loc[params.structure.eq("H1_GLOBAL")]
    boundary_models = int((h1.groupby("model_id").vf_boundary.sum() >= 2).sum())
    eta_boundary_models = int((h1.groupby("model_id").eta_boundary.sum() >= 2).sum())
    process_pass = bool(
        temporal_p1["station_noninferior"] >= 10 and temporal_p1["tree_noninferior"] >= 10
        and max(temporal_p1["station_predictively_improved"], temporal_p1["tree_predictively_improved"]) >= 8
        and nested_p1["LOSO_noninferior"] >= 10 and nested_p1["LOTO_noninferior"] >= 10
        and max(nested_p1["LOSO_predictively_improved"], nested_p1["LOTO_predictively_improved"]) >= 8
        and boundary_models <= 2 and eta_boundary_models <= 2
    )
    monitored_p2_pass = bool(
        temporal_p2["station_noninferior"] >= 10 and temporal_p2["tree_noninferior"] >= 10
    )
    p2r_pass = bool(
        p2r_absolute["LOSO_noninferior"] >= 10 and p2r_absolute["LOTO_noninferior"] >= 10
        and max(p2r_absolute["LOSO_predictively_improved"], p2r_absolute["LOTO_predictively_improved"]) >= 8
    )
    decision = {
        "status": "river_hydraulic_exposure_supported" if process_pass else "river_hydraulic_exposure_not_supported",
        "selected_process_structure": "H1_GLOBAL" if process_pass else "H0_PARENT",
        "temperature": "closed_not_used",
        "temporal_P1_counts": temporal_p1,
        "temporal_P2_counts": temporal_p2,
        "nested_P1_counts": nested_p1,
        "P2R_vs_P1_counts": p2r_absolute,
        "boundary_confounded_models": boundary_models,
        "eta_boundary_confounded_models": eta_boundary_models,
        "monitored_station_prediction_layer": "P2" if monitored_p2_pass else "P1",
        "unmonitored_spatial_prediction_layer": "P2R" if p2r_pass else "P1",
        "P2R_status": "supported" if p2r_pass else "not_supported_absolute_spatial_skill",
        "tree_163_role": "diagnostic_only_open_lake_center; no lake operator added",
        "interpretation": (
            "A single Q72-plus-geometry hydraulic exposure coefficient transfers spatially. "
            "Station P2 remains valid only for monitored stations; Bayesian MAP partial pooling is not promoted unless it also beats P1 in absolute nested spatial prediction."
        ),
        "TN_2022_read": False,
    }
    h.dump_json(h.REPORTS / "spatial_hydraulic_decision.json", decision)
    lock = {
        "lock": "development_mechanism_lock", "written_before_2022_TN": True,
        "selected_process_structure": decision["selected_process_structure"],
        "monitored_station_prediction_layer": decision["monitored_station_prediction_layer"],
        "unmonitored_spatial_prediction_layer": decision["unmonitored_spatial_prediction_layer"],
        "temperature": "forbidden_in_20260820_19", "tree_163_role": decision["tree_163_role"],
        "development_years": [2016, 2021], "OOF_years": [2018, 2019, 2020, 2021],
    }
    h.dump_json(h.REPORTS / "development_mechanism_lock.json", lock)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
