from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hleg_shared import (
    FOLD_PATH,
    NONINFERIOR_MARGIN,
    OUT,
    REPORTS,
    dump_json,
    formal_specs,
    paired_bootstrap,
    parent_shared,
    require_runtime,
    select_beta,
    station_macro_rmse,
    tree_macro_rmse,
)


def fit_nested_candidate(
    shared: object,
    candidates: pd.DataFrame,
    mechanism: str,
    train_start: int,
    train_end: int,
    evaluation_year: int,
    holdout_column: str,
    holdout: object,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    train_mask = candidates.year.between(train_start, train_end) & ~candidates[holdout_column].eq(holdout)
    test_mask = candidates.year.eq(evaluation_year) & candidates[holdout_column].eq(holdout)
    train_all = candidates.loc[train_mask].copy()
    test_all = candidates.loc[test_mask].copy()
    if train_all.empty or test_all.empty:
        raise RuntimeError("nested split unexpectedly empty")
    if mechanism == "PARENT":
        selected_beta = 0.0
        train = train_all
        test = test_all
        eta, effects, diagnostic = shared.fit_readout(train, "P1")
        surface = pd.DataFrame([{"beta_h": 0.0, "training_station_macro_rmse_log1p": station_macro_rmse(shared.predict_layer(train, "P1", eta, effects))}])
    else:
        selected_beta, surface, eta, effects = select_beta(shared, train_all, train_start, train_end, layer="P1")
        train = train_all.loc[train_all.beta_h.eq(selected_beta)]
        test = test_all.loc[test_all.beta_h.eq(selected_beta)]
        diagnostic = {"eta_quick": float(eta[0]), "eta_gw": float(eta[1]), "eta_boundary": bool(np.any(eta <= 0.01) or np.any(eta >= 0.99))}
    prediction = shared.predict_layer(test, "P1", eta, {})
    station_means = train.assign(y=np.log1p(train.tn_mg_l)).groupby("station_key").y.mean()
    prediction["baseline_log_station_equal"] = float(station_means.mean())
    prediction["selected_beta_h"] = selected_beta
    return prediction, {
        "selected_beta_h": selected_beta,
        "eta_quick": float(eta[0]),
        "eta_gw": float(eta[1]),
        "eta_boundary": bool(np.any(eta <= 0.01) or np.any(eta >= 0.99)),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
    }, surface


def spatial_skill(frame: pd.DataFrame) -> float:
    work = frame.copy()
    work["model_sq"] = (np.log1p(work.pred_tn_mg_l) - np.log1p(work.tn_mg_l)) ** 2
    work["base_sq"] = (work.baseline_log_station_equal - np.log1p(work.tn_mg_l)) ** 2
    station = work.groupby("station_key")[["model_sq", "base_sq"]].mean()
    return float(1.0 - station.model_sq.mean() / station.base_sq.mean())


def main() -> None:
    require_runtime()
    temporal = json.loads((REPORTS / "temporal_evidence_summary.json").read_text(encoding="utf-8"))
    if temporal["TN_2022_rows_materialized"] != 0:
        raise RuntimeError("2022 TN boundary violation")
    shared = parent_shared()
    paths = pd.read_parquet(OUT / "candidate_development_observation_paths.parquet")
    folds = pd.read_parquet(FOLD_PATH)[["fold_id", "train_start_year", "train_end_year", "evaluation_year"]].drop_duplicates().sort_values("evaluation_year")

    prediction_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    surface_rows: list[pd.DataFrame] = []
    for model_index, spec in enumerate(formal_specs(), start=1):
        model_id = str(spec["model_id"])
        model_paths = paths.loc[paths.model_id.eq(model_id)]
        for fold in folds.itertuples():
            eval_parent = model_paths.loc[
                model_paths.mechanism.eq("PARENT") & model_paths.year.eq(fold.evaluation_year)
            ]
            for scheme, column in (("LOSO", "station_key"), ("LOTO", "terminal_tree_id")):
                holdouts = sorted(eval_parent[column].dropna().unique(), key=str)
                for holdout in holdouts:
                    for mechanism in ("PARENT", "HLEG_BULK", "HLEG_AGE"):
                        candidates = model_paths.loc[model_paths.mechanism.eq(mechanism)]
                        pred, params, surface = fit_nested_candidate(
                            shared, candidates, mechanism, int(fold.train_start_year), int(fold.train_end_year),
                            int(fold.evaluation_year), column, holdout,
                        )
                        pred["fold_id"] = fold.fold_id
                        pred["spatial_scheme"] = scheme
                        pred["holdout_id"] = str(holdout)
                        prediction_rows.append(pred)
                        parameter_rows.append({
                            "model_id": model_id, "mechanism": mechanism, "fold_id": fold.fold_id,
                            "evaluation_year": int(fold.evaluation_year), "spatial_scheme": scheme,
                            "holdout_id": str(holdout), **params,
                        })
                        surface["model_id"] = model_id
                        surface["mechanism"] = mechanism
                        surface["fold_id"] = fold.fold_id
                        surface["spatial_scheme"] = scheme
                        surface["holdout_id"] = str(holdout)
                        surface["selected"] = surface.beta_h.eq(params["selected_beta_h"])
                        surface_rows.append(surface)
        print(f"stage3 nested {model_index:02d}/12 {model_id}", flush=True)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_parquet(OUT / "nested_spatial_predictions.parquet", index=False)
    pd.DataFrame(parameter_rows).to_parquet(OUT / "nested_spatial_readout_parameters.parquet", index=False)
    pd.concat(surface_rows, ignore_index=True).to_parquet(OUT / "nested_spatial_beta_surface.parquet", index=False)

    performance_rows = []
    metric_rows = []
    seed_offset = 5000
    for (model_id, scheme), group in predictions.groupby(["model_id", "spatial_scheme"]):
        mechanisms = {name: group.loc[group.mechanism.eq(name)] for name in ("PARENT", "HLEG_BULK", "HLEG_AGE")}
        block = "station_key" if scheme == "LOSO" else "terminal_tree_id"
        for mechanism, frame in mechanisms.items():
            metric_rows.append({
                "model_id": model_id, "mechanism": mechanism, "spatial_scheme": scheme,
                "station_macro_rmse_log1p": station_macro_rmse(frame),
                "terminal_tree_macro_rmse_log1p": tree_macro_rmse(frame),
                "station_blind_skill_log": spatial_skill(frame),
                "n": int(len(frame)),
            })
        for candidate_name, reference_name in (
            ("HLEG_BULK", "PARENT"), ("HLEG_AGE", "PARENT"), ("HLEG_AGE", "HLEG_BULK"),
        ):
            seed_offset += 1
            result = paired_bootstrap(mechanisms[reference_name], mechanisms[candidate_name], block, seed_offset)
            performance_rows.append({
                "model_id": model_id, "spatial_scheme": scheme,
                "candidate": candidate_name, "reference": reference_name, **result,
            })
    performance = pd.DataFrame(performance_rows)
    performance.to_parquet(OUT / "nested_spatial_performance.parquet", index=False)
    pd.DataFrame(metric_rows).to_parquet(OUT / "nested_spatial_metrics.parquet", index=False)

    evidence = []
    for candidate, reference in (("HLEG_BULK", "PARENT"), ("HLEG_AGE", "PARENT"), ("HLEG_AGE", "HLEG_BULK")):
        subset = performance.loc[performance.candidate.eq(candidate) & performance.reference.eq(reference)]
        evidence.append({
            "candidate": candidate, "reference": reference,
            "LOSO_noninferior_models": int(subset.loc[subset.spatial_scheme.eq("LOSO"), "noninferior"].sum()),
            "LOTO_noninferior_models": int(subset.loc[subset.spatial_scheme.eq("LOTO"), "noninferior"].sum()),
            "LOSO_improved_models": int(subset.loc[subset.spatial_scheme.eq("LOSO"), "predictively_improved"].sum()),
            "LOTO_improved_models": int(subset.loc[subset.spatial_scheme.eq("LOTO"), "predictively_improved"].sum()),
        })
    dump_json(REPORTS / "nested_spatial_evidence_summary.json", {
        "scenario_id": "20260820_10",
        "evaluation": "time_by_space_nested_P1",
        "noninferiority_margin": NONINFERIOR_MARGIN,
        "ensemble_noninferiority_requirement": "at_least_10_of_12_models_in_each_scheme",
        "comparisons": evidence,
        "TN_2022_rows_materialized": 0,
    })


if __name__ == "__main__":
    main()
