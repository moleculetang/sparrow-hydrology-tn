from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from legacy18_shared import (
    FOLD_PATH,
    OBS_PATH,
    S16_4,
    S17_1,
    S17_4,
    S17_6,
    TEST,
    dump_json,
    flow_regime_registry,
    formal_specs,
    hash_manifest,
    hydrology_registry,
    metric_values,
    require_runtime,
    spatial_holdout,
    spatial_skill,
    spatial_skill_bootstrap,
    station_macro_rmse,
    temporal_oof,
    tree_macro_rmse,
)


ROOT = TEST / "20260818_1"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parents = [
        S16_4 / "outputs" / "candidate_routed_paths_2016_2021.parquet",
        S16_4 / "outputs" / "candidate_oof_predictions_2018_2021.parquet",
        S17_1 / "reports" / "structural_reconciliation_decision.json",
        S17_4 / "reports" / "performance_metric_contract.json",
        S17_6 / "final_lock.json",
        OBS_PATH,
        FOLD_PATH,
    ]
    start_hashes = hash_manifest(parents)
    dump_json(REPORTS / "parent_hashes_start.json", start_hashes)

    observations = pd.read_parquet(OBS_PATH)
    folds = pd.read_parquet(FOLD_PATH)
    routed_all = pd.read_parquet(parents[0])
    model_ids = [str(x["model_id"]) for x in formal_specs()]
    routed_all = routed_all.loc[routed_all.model_id.isin(model_ids)]
    hydro = hydrology_registry(2016, 2021)
    regimes = flow_regime_registry(observations, folds, hydro)
    regimes.to_parquet(OUT / "flow_regime_registry.parquet", index=False)

    temporal_rows: list[pd.DataFrame] = []
    parameter_rows: list[pd.DataFrame] = []
    effect_rows: list[pd.DataFrame] = []
    spatial_rows: list[pd.DataFrame] = []
    spatial_param_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    skill_rows: list[dict[str, object]] = []

    for i, model_id in enumerate(model_ids, start=1):
        routed = routed_all.loc[routed_all.model_id.eq(model_id)].drop(columns="model_id")
        pred, params, effects = temporal_oof(routed, observations, folds)
        pred["model_id"] = model_id
        params["model_id"] = model_id
        effects["model_id"] = model_id
        temporal_rows.append(pred)
        parameter_rows.append(params)
        effect_rows.append(effects)
        for layer, group in pred.groupby("layer"):
            base = {"model_id": model_id, "evaluation": "temporal_oof_2018_2021", "layer": layer}
            metric_rows.append({**base, "scope": "pooled", **metric_values(group)})
            metric_rows.append({**base, "scope": "station_macro", "rmse_log1p": station_macro_rmse(group), "n": len(group)})
            metric_rows.append({**base, "scope": "terminal_tree_macro", "rmse_log1p": tree_macro_rmse(group), "n": len(group)})

        for holdout_column, label in (("station_key", "LOSO"), ("terminal_tree_id", "LOTO")):
            sp, spp = spatial_holdout(routed, observations, holdout_column)
            sp["model_id"] = model_id
            sp["spatial_scheme"] = label
            spp["model_id"] = model_id
            spp["spatial_scheme"] = label
            spatial_rows.append(sp)
            spatial_param_rows.append(spp)
            for layer, group in sp.groupby("layer"):
                base = {"model_id": model_id, "evaluation": label, "layer": layer}
                metric_rows.append({**base, "scope": "pooled", **metric_values(group)})
                metric_rows.append({**base, "scope": "station_macro", "rmse_log1p": station_macro_rmse(group), "n": len(group)})
                metric_rows.append({**base, "scope": "terminal_tree_macro", "rmse_log1p": tree_macro_rmse(group), "n": len(group)})
            p1 = sp.loc[sp.layer.eq("P1")]
            block = "station_key" if label == "LOSO" else "terminal_tree_id"
            lo, hi, mean = spatial_skill_bootstrap(p1, block, seed_offset=i * 10 + (0 if label == "LOSO" else 1))
            skill_rows.append({
                "model_id": model_id,
                "spatial_scheme": label,
                "skill_log_station_macro": spatial_skill(p1),
                "bootstrap_mean": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "ci_lower_gt_zero": bool(lo > 0),
            })
        print(f"[{i:02d}/{len(model_ids)}] {model_id}", flush=True)

    temporal = pd.concat(temporal_rows, ignore_index=True)
    parameters = pd.concat(parameter_rows, ignore_index=True)
    effects = pd.concat(effect_rows, ignore_index=True)
    spatial = pd.concat(spatial_rows, ignore_index=True)
    spatial_params = pd.concat(spatial_param_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    skills = pd.DataFrame(skill_rows)

    temporal.to_parquet(OUT / "readout_level_predictions.parquet", index=False)
    parameters.to_parquet(OUT / "candidate_fold_readout_parameters.parquet", index=False)
    effects.to_parquet(OUT / "station_effects_by_fold.parquet", index=False)
    spatial.to_parquet(OUT / "spatial_holdout_predictions.parquet", index=False)
    spatial_params.to_parquet(OUT / "spatial_holdout_readout_parameters.parquet", index=False)
    metrics.to_parquet(OUT / "readout_skill_decomposition.parquet", index=False)
    skills.to_parquet(OUT / "station_blind_spatial_skill.parquet", index=False)

    p2_effect = effects.loc[effects.layer.eq("P2")]
    stability = p2_effect.groupby(["model_id", "station_key"], as_index=False).agg(
        fold_count=("fold_id", "nunique"),
        mean_b=("station_effect", "mean"),
        mean_abs_b=("station_effect", lambda x: float(np.mean(np.abs(x)))),
        sd_b=("station_effect", "std"),
        sign_consistency=("station_effect", lambda x: float(max(np.mean(x >= 0), np.mean(x <= 0)))),
    )
    stability.to_parquet(OUT / "station_effect_stability.parquet", index=False)

    pivot = skills.pivot(index="model_id", columns="spatial_scheme", values="ci_lower_gt_zero")
    supported_ids = pivot.index[(pivot.get("LOSO", False)) & (pivot.get("LOTO", False))].tolist()
    decision = {
        "scenario_id": "20260818_1",
        "formal_model_count": len(model_ids),
        "temporal_oof_keys_per_model_layer": int(temporal.groupby(["model_id", "layer"]).size().min()),
        "spatial_benchmark": "station_equal_training_mean_ln1p_TN",
        "skill_formula": "1 - station_macro_MSE(P1) / station_macro_MSE(station_blind_baseline)",
        "models_with_LOSO_and_LOTO_skill_ci_lower_gt_zero": supported_ids,
        "globally_scaled_spatial_skill_supported_for_all_models": len(supported_ids) == len(model_ids),
        "interpretation_boundary": "P0/P1/P2 are performance layers, not orthogonal variance attribution",
        "locked_2022_used": False,
    }
    dump_json(REPORTS / "readout_dependence_decision.json", decision)
    dump_json(REPORTS / "readout_training_contract.json", {
        "P0": "unscaled_process_layer",
        "P1": "globally_scaled_process_layer",
        "P2": "station_conditioned_prediction",
        "solver": "scipy.optimize.least_squares_trf",
        "x0": [0.8, 0.8],
        "bounds": [0.0, 1.0],
        "eta_ridge": 1.0,
        "station_ridge": 12.0,
        "xtol": 1e-12,
        "ftol": 1e-12,
        "gtol": 1e-12,
        "max_nfev": 2000,
        "candidate_specific_training_required": True,
    })
    end_hashes = hash_manifest(parents)
    dump_json(REPORTS / "parent_hashes_end.json", end_hashes)
    dump_json(REPORTS / "completion_audit.json", {
        "status": "PASS" if start_hashes == end_hashes else "FAIL",
        "parent_hashes_unchanged": start_hashes == end_hashes,
        "formal_models": len(model_ids),
        "temporal_prediction_rows": len(temporal),
        "spatial_prediction_rows": len(spatial),
        "locked_2022_used": False,
    })


if __name__ == "__main__":
    main()
