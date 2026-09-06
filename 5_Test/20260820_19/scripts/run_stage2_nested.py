from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

import numpy as np
import pandas as pd

import hierarchical19_shared as h


def _add_identity(frame: pd.DataFrame, model_id: str, fold_id: str, evaluation: str,
                  holdout_id: str, mechanism: str, selected: str) -> pd.DataFrame:
    frame["model_id"] = model_id
    frame["fold_id"] = fold_id
    frame["evaluation"] = evaluation
    frame["holdout_id"] = holdout_id
    frame["mechanism"] = mechanism
    frame["selected_structure"] = selected
    return frame


def run_model(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shared = h.parent_shared()
    obs = h.development_observations()
    folds = h.fold_registry()
    router = h.build_router(model_id, shared)
    reach_to_tree = dict(zip(router.reach_ids.astype(int), router.terminal_by_reach.astype(int)))
    obs["terminal_tree_id"] = obs.reach_id.astype(int).map(reach_to_tree).astype(int)
    predictions: list[pd.DataFrame] = []
    structure_rows: list[dict[str, object]] = []
    readout_rows: list[dict[str, object]] = []
    completed = 0
    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)
        train_time = obs.year.between(int(fold.train_start_year), int(fold.train_end_year))
        eval_time = obs.year.eq(int(fold.evaluation_year))
        evaluations = (
            ("LOSO", "station_key", sorted(obs.loc[eval_time, "station_key"].astype(str).unique())),
            ("LOTO", "terminal_tree_id", sorted(obs.loc[eval_time, "terminal_tree_id"].astype(int).unique())),
        )
        for evaluation, column, holdouts in evaluations:
            for holdout in holdouts:
                if column == "station_key":
                    excluded = obs.station_key.astype(str).eq(str(holdout))
                    tested = excluded
                else:
                    excluded = obs.terminal_tree_id.eq(int(holdout))
                    tested = excluded
                train_obs = obs.loc[train_time & ~excluded].copy()
                test_obs = obs.loc[eval_time & tested].copy()
                if train_obs.empty or test_obs.empty:
                    continue
                # Stage 1 admitted H1 and rejected H2/H3 at the P1 tree-block gate.
                # The fixed H1 architecture is now refit from scratch inside every holdout.
                h0 = h.fit_structure(router, train_obs, "H0_PARENT", shared)
                h1 = h.fit_structure(router, train_obs, "H1_GLOBAL", shared)
                fits = {"H0_PARENT": h0, "H1_GLOBAL": h1}
                selected = "H1_GLOBAL"
                holdout_id = str(holdout)
                for structure in ("H0_PARENT", "H1_GLOBAL"):
                    structure_rows.append(h.structure_parameter_row(
                        model_id, fold_id, structure, fits[structure], structure == selected,
                        evaluation=evaluation, holdout_id=holdout_id,
                    ) | {"evaluation_year": int(fold.evaluation_year)})

                for comparison_role, structure in (("H0_PARENT", "H0_PARENT"), ("NESTED_SELECTED", "H1_GLOBAL")):
                    vf = np.asarray(fits[structure]["vf"])
                    train = router.frame(train_obs, vf)
                    test = router.frame(test_obs, vf)
                    if comparison_role == "H0_PARENT":
                        p1_fit = fits["H0_PARENT"]
                    else:
                        p1_fit = fits[selected]
                    p1 = shared.predict_layer(test, "P1", np.asarray(p1_fit["eta"]), {})
                    predictions.append(_add_identity(
                        p1, model_id, fold_id, evaluation, holdout_id, comparison_role, selected
                    ))
                    for layer in ("P2", "P2R"):
                        readout = h.fit_selected_readout(train, layer, shared)
                        pred = h.predict_selected(test, layer, readout, shared)
                        predictions.append(_add_identity(
                            pred, model_id, fold_id, evaluation, holdout_id, comparison_role, selected
                        ))
                        diagnostic = readout["diagnostic"]
                        readout_rows.append({
                            "model_id": model_id, "fold_id": fold_id, "evaluation": evaluation,
                            "evaluation_year": int(fold.evaluation_year), "holdout_id": holdout_id,
                            "mechanism": comparison_role, "selected_structure": selected, "layer": layer,
                            "eta_quick": float(readout["eta"][0]), "eta_gw": float(readout["eta"][1]),
                            "eta_boundary": bool(diagnostic["eta_boundary"]),
                            "success": bool(diagnostic["success"]), "nfev": int(diagnostic.get("nfev", 0)),
                        })
                completed += 1
                if completed % 10 == 0:
                    print(json.dumps({"model": model_id, "nested_holdouts": completed}), flush=True)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(structure_rows), pd.DataFrame(readout_rows)


def worker(model_id: str) -> tuple[str, str, str, str]:
    cache = h.CACHE / "nested"
    cache.mkdir(parents=True, exist_ok=True)
    pred_path = cache / f"{model_id}__predictions.parquet"
    structure_path = cache / f"{model_id}__structures.parquet"
    readout_path = cache / f"{model_id}__readouts.parquet"
    if not (pred_path.exists() and structure_path.exists() and readout_path.exists()):
        pred, structures, readouts = run_model(model_id)
        pred.to_parquet(pred_path, index=False)
        structures.to_parquet(structure_path, index=False)
        readouts.to_parquet(readout_path, index=False)
    return model_id, str(pred_path), str(structure_path), str(readout_path)


def main() -> None:
    h.require_runtime()
    stage1 = json.loads((h.REPORTS / "stage1_temporal_audit.json").read_text(encoding="utf-8"))
    decision = json.loads((h.REPORTS / "stage1_structure_decision.json").read_text(encoding="utf-8"))
    if (stage1["status"] != "PASS" or stage1["TN_2022_read"]
            or not decision["nested_LOSO_LOTO_authorized"]
            or decision["registered_parsimony_decision"] != "H1_GLOBAL"):
        raise RuntimeError("STOP_STAGE1_NOT_PASS")
    results = []
    max_workers = min(8, len(h.FORMAL_MODELS), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, model): model for model in h.FORMAL_MODELS}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"stage2_model_complete": result[0], "count": len(results)}), flush=True)
    results.sort()
    predictions = pd.concat([pd.read_parquet(row[1]) for row in results], ignore_index=True)
    structures = pd.concat([pd.read_parquet(row[2]) for row in results], ignore_index=True)
    readouts = pd.concat([pd.read_parquet(row[3]) for row in results], ignore_index=True)
    predictions.to_parquet(h.OUT / "nested_spatial_predictions.parquet", index=False)
    structures.to_parquet(h.OUT / "nested_spatial_parameters.parquet", index=False)
    readouts.to_parquet(h.OUT / "nested_spatial_readout_parameters.parquet", index=False)

    comparison_rows = []
    performance_rows = []
    seed = 2026082500
    for model_id in h.FORMAL_MODELS:
        for evaluation, block in (("LOSO", "station_key"), ("LOTO", "terminal_tree_id")):
            for layer in ("P1", "P2", "P2R"):
                subset = predictions.loc[
                    predictions.model_id.eq(model_id) & predictions.evaluation.eq(evaluation)
                    & predictions.layer.eq(layer)
                ]
                parent = subset.loc[subset.mechanism.eq("H0_PARENT")]
                candidate = subset.loc[subset.mechanism.eq("NESTED_SELECTED")]
                seed += 1
                comparison_rows.append({
                    "model_id": model_id, "evaluation": evaluation, "layer": layer,
                    **h.paired_bootstrap(parent, candidate, block, seed),
                })
                for mechanism, frame in (("H0_PARENT", parent), ("NESTED_SELECTED", candidate)):
                    performance_rows.append({
                        "model_id": model_id, "evaluation": evaluation, "layer": layer,
                        "mechanism": mechanism, "n": len(frame),
                        "station_macro_rmse_log1p": h.station_macro_rmse(frame),
                        "tree_macro_rmse_log1p": h.tree_macro_rmse(frame),
                    })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_parquet(h.OUT / "nested_spatial_metrics.parquet", index=False)
    pd.DataFrame(performance_rows).to_parquet(h.OUT / "nested_spatial_performance.parquet", index=False)
    selected = structures.loc[structures.selected]
    audit = {
        "status": "PASS",
        "models": int(predictions.model_id.nunique()),
        "evaluations": sorted(predictions.evaluation.unique().tolist()),
        "prediction_rows": len(predictions), "structure_parameter_rows": len(structures),
        "readout_parameter_rows": len(readouts),
        "selected_structure_counts": selected.structure.value_counts().to_dict(),
        "all_structure_optimizers_successful": bool(structures.outer_success.all()),
        "all_readout_optimizers_successful": bool(readouts.success.all()),
        "LOSO_heldout_station_effect_rule": "station effect absent from training map and exactly zero in prediction",
        "LOTO_heldout_tree_effect_rule": "process and readout tree effects absent from training maps and exactly zero in prediction",
        "temperature_used": False, "TN_2022_read": False,
    }
    h.dump_json(h.REPORTS / "stage2_nested_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
