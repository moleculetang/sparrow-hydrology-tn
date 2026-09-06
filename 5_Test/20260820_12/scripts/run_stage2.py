from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd

from a0_shared import (
    CACHE, FOLD_PATH, OUT, REPORTS, development_observations, dump_json, fit_harmonic,
    fit_uniform, formal_specs, observation_basis, paired_bootstrap, parameter_row,
    parent_shared, predict_fit, require_runtime, station_macro_rmse, tree_macro_rmse,
)


def run_model(model_id: str, observations: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = parent_shared()
    basis = observation_basis(model_id, observations)
    years = basis.base.year.to_numpy(int)
    station_values = basis.base.station_key.astype(str).to_numpy()
    tree_values = basis.base.terminal_tree_id.to_numpy(int)
    prediction_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    for fold in folds.itertuples():
        train_time = (years >= fold.train_start_year) & (years <= fold.train_end_year)
        eval_time = years == fold.evaluation_year
        holdout_sets = (
            ("LOSO", "station_key", station_values, sorted(set(station_values[eval_time]))),
            ("LOTO", "terminal_tree_id", tree_values, sorted(set(tree_values[eval_time]))),
        )
        for evaluation, holdout_column, values, holdouts in holdout_sets:
            for number, holdout in enumerate(holdouts, start=1):
                train_idx = np.flatnonzero(train_time & (values != holdout))
                test_idx = np.flatnonzero(eval_time & (values == holdout))
                if not len(train_idx) or not len(test_idx):
                    continue
                for mechanism, fit, parent in (
                    ("UNIFORM_PARENT", fit_uniform(shared, basis, train_idx, "P1"), True),
                    ("A0_HARMONIC", fit_harmonic(shared, basis, train_idx, "P1"), False),
                ):
                    pred = predict_fit(shared, basis, fit, test_idx, "P1", parent)
                    pred["model_id"] = model_id
                    pred["fold_id"] = fold.fold_id
                    pred["evaluation"] = evaluation
                    pred["holdout_column"] = holdout_column
                    pred["holdout_id"] = str(holdout)
                    pred["mechanism"] = mechanism
                    pred["layer"] = "P1"
                    prediction_parts.append(pred)
                    row = parameter_row(model_id, str(fold.fold_id), "P1", mechanism, fit)
                    row.update({"evaluation": evaluation, "holdout_column": holdout_column, "holdout_id": str(holdout), "evaluation_year": int(fold.evaluation_year)})
                    parameter_rows.append(row)
            print(f"stage2 {model_id} {fold.fold_id} {evaluation} holdouts={len(holdouts)}", flush=True)
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(parameter_rows)


def run_model_worker(model_id: str) -> tuple[str, str, str]:
    observations = development_observations()
    folds = pd.read_parquet(FOLD_PATH)[["fold_id", "train_start_year", "train_end_year", "evaluation_year"]].drop_duplicates().sort_values("evaluation_year")
    nested_cache = CACHE / "nested"
    nested_cache.mkdir(parents=True, exist_ok=True)
    pred_path = nested_cache / f"{model_id}__predictions.parquet"
    param_path = nested_cache / f"{model_id}__parameters.parquet"
    if not (pred_path.exists() and param_path.exists()):
        predictions, parameters = run_model(model_id, observations, folds)
        predictions.to_parquet(pred_path, index=False)
        parameters.to_parquet(param_path, index=False)
    return model_id, str(pred_path), str(param_path)


def main() -> None:
    require_runtime()
    stage1 = pd.read_json(REPORTS / "stage1_temporal_audit.json", typ="series")
    if stage1["status"] != "PASS":
        raise RuntimeError("STOP_STAGE1_NOT_PASS")
    nested_cache = CACHE / "nested"
    nested_cache.mkdir(parents=True, exist_ok=True)
    prediction_parts = []
    parameter_parts = []
    model_ids = [str(spec["model_id"]) for spec in formal_specs()]
    max_workers = min(6, len(model_ids), os.cpu_count() or 1)
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_model_worker, model_id): model_id for model_id in model_ids}
        for future in as_completed(futures):
            model_id, pred_path, param_path = future.result()
            prediction_parts.append(pd.read_parquet(pred_path))
            parameter_parts.append(pd.read_parquet(param_path))
            completed += 1
            print(f"stage2 nested complete {completed:02d}/12 {model_id}", flush=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    parameters = pd.concat(parameter_parts, ignore_index=True)
    predictions.to_parquet(OUT / "nested_spatial_predictions.parquet", index=False)
    parameters.to_parquet(OUT / "nested_spatial_parameters.parquet", index=False)

    comparison_rows = []
    performance_rows = []
    seed_offset = 5000
    for model_id in sorted(predictions.model_id.unique()):
        for evaluation, block in (("LOSO", "station_key"), ("LOTO", "terminal_tree_id")):
            parent = predictions.loc[predictions.model_id.eq(model_id) & predictions.evaluation.eq(evaluation) & predictions.mechanism.eq("UNIFORM_PARENT")]
            candidate = predictions.loc[predictions.model_id.eq(model_id) & predictions.evaluation.eq(evaluation) & predictions.mechanism.eq("A0_HARMONIC")]
            seed_offset += 1
            comparison_rows.append({"model_id": model_id, "evaluation": evaluation, "block": block, **paired_bootstrap(parent, candidate, block, seed_offset)})
            performance_rows.extend([
                {"model_id": model_id, "evaluation": evaluation, "mechanism": "UNIFORM_PARENT", "rmse_log1p": station_macro_rmse(parent) if evaluation == "LOSO" else tree_macro_rmse(parent), "n": len(parent)},
                {"model_id": model_id, "evaluation": evaluation, "mechanism": "A0_HARMONIC", "rmse_log1p": station_macro_rmse(candidate) if evaluation == "LOSO" else tree_macro_rmse(candidate), "n": len(candidate)},
            ])
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_parquet(OUT / "nested_spatial_metrics.parquet", index=False)
    pd.DataFrame(performance_rows).to_parquet(OUT / "nested_spatial_performance.parquet", index=False)
    expected_parameter_rows = sum(
        2 * (predictions.loc[(predictions.model_id == model_id) & (predictions.mechanism == "UNIFORM_PARENT")][["fold_id", "evaluation", "holdout_id"]].drop_duplicates().shape[0])
        for model_id in predictions.model_id.unique()
    )
    audit = {
        "status": "PASS",
        "models": int(predictions.model_id.nunique()),
        "evaluations": sorted(predictions.evaluation.unique().tolist()),
        "prediction_rows": len(predictions), "parameter_rows": len(parameters),
        "unique_candidate_holdouts": int(parameters[["model_id", "fold_id", "evaluation", "holdout_id", "mechanism"]].drop_duplicates().shape[0]),
        "all_outer_optimizers_successful": bool(parameters.outer_success.all()),
        "all_inner_optimizers_successful": bool(parameters.inner_success.all()),
        "expected_parameter_rows_from_predictions": int(expected_parameter_rows),
    }
    dump_json(REPORTS / "stage2_nested_spatial_audit.json", audit)
    if not audit["all_outer_optimizers_successful"] or not audit["all_inner_optimizers_successful"]:
        raise RuntimeError("STOP_NESTED_OPTIMIZER_FAILURE")


if __name__ == "__main__":
    main()
