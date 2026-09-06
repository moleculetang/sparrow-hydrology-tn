from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

import numpy as np
import pandas as pd

import hierarchical19_shared as h


def _readout_parameter_row(model_id: str, fold_id: str, structure: str, layer: str, fit: dict[str, object]) -> dict[str, object]:
    diagnostic = fit["diagnostic"]
    row: dict[str, object] = {
        "model_id": model_id, "fold_id": fold_id, "structure": structure, "layer": layer,
        "eta_quick": float(fit["eta"][0]), "eta_gw": float(fit["eta"][1]),
        "eta_boundary": bool(diagnostic["eta_boundary"]), "success": bool(diagnostic["success"]),
        "nfev": int(diagnostic.get("nfev", 0)),
    }
    if layer == "P2R":
        row.update({
            "delta_json": json.dumps(np.asarray(fit["delta"]).tolist()),
            "tree_effect_json": json.dumps(fit["tree_effects"]),
            "station_effect_count": len(fit["station_effects"]),
        })
    else:
        row["station_effect_count"] = len(fit["effects"])
    return row


def run_model(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shared = h.parent_shared()
    obs = h.development_observations()
    folds = h.fold_registry()
    router = h.build_router(model_id, shared)
    prediction_parts: list[pd.DataFrame] = []
    structure_rows: list[dict[str, object]] = []
    readout_rows: list[dict[str, object]] = []
    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)
        train_obs = obs.loc[obs.year.between(int(fold.train_start_year), int(fold.train_end_year))].copy()
        test_obs = obs.loc[obs.year.eq(int(fold.evaluation_year))].copy()
        fits = h.fit_all_structures(router, train_obs, shared)
        selected = h.select_structure(fits)
        for structure in h.STRUCTURES:
            fit = fits[structure]
            structure_rows.append(h.structure_parameter_row(model_id, fold_id, structure, fit, structure == selected))
            test = router.frame(test_obs, np.asarray(fit["vf"]))
            pred = shared.predict_layer(test, "P1", np.asarray(fit["eta"]), fit["effects"])
            pred["model_id"] = model_id
            pred["fold_id"] = fold_id
            pred["mechanism"] = structure
            pred["selected_structure"] = selected
            prediction_parts.append(pred)

        for structure in tuple(dict.fromkeys(("H0_PARENT", selected))):
            vf = np.asarray(fits[structure]["vf"])
            train = router.frame(train_obs, vf)
            test = router.frame(test_obs, vf)
            for layer in ("P2", "P2R"):
                readout = h.fit_selected_readout(train, layer, shared)
                pred = h.predict_selected(test, layer, readout, shared)
                pred["model_id"] = model_id
                pred["fold_id"] = fold_id
                pred["mechanism"] = structure
                pred["selected_structure"] = selected
                prediction_parts.append(pred)
                readout_rows.append(_readout_parameter_row(model_id, fold_id, structure, layer, readout))
        print(json.dumps({"model": model_id, "fold": fold_id, "selected": selected}), flush=True)
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(structure_rows), pd.DataFrame(readout_rows)


def worker(model_id: str) -> tuple[str, str, str, str]:
    cache = h.CACHE / "temporal"
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
    preflight = json.loads((h.REPORTS / "stage0_preflight.json").read_text(encoding="utf-8"))
    if preflight["status"] != "PASS" or preflight["TN_2022_read"]:
        raise RuntimeError("STOP_STAGE0_NOT_PASS")
    h.CACHE.mkdir(parents=True, exist_ok=True)
    audit_router = h.build_router(h.FORMAL_MODELS[0], h.parent_shared())
    nesting = h.exact_nesting_audit(audit_router)
    nesting.to_parquet(h.OUT / "exact_nesting_audit.parquet", index=False)
    if not nesting["pass"].all():
        raise RuntimeError("STOP_EXACT_NESTING_FAILURE")

    results = []
    max_workers = min(4, len(h.FORMAL_MODELS), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, model): model for model in h.FORMAL_MODELS}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"stage1_complete": result[0], "count": len(results)}), flush=True)
    results.sort()
    predictions = pd.concat([pd.read_parquet(row[1]) for row in results], ignore_index=True)
    structures = pd.concat([pd.read_parquet(row[2]) for row in results], ignore_index=True)
    readouts = pd.concat([pd.read_parquet(row[3]) for row in results], ignore_index=True)
    predictions.to_parquet(h.OUT / "temporal_oof_predictions.parquet", index=False)
    structures.to_parquet(h.OUT / "temporal_fold_parameters.parquet", index=False)
    readouts.to_parquet(h.OUT / "temporal_readout_parameters.parquet", index=False)

    metric_rows = []
    gate_rows = []
    seed = 2026082000
    for model_id in h.FORMAL_MODELS:
        for layer in ("P1", "P2", "P2R"):
            subset = predictions.loc[predictions.model_id.eq(model_id) & predictions.layer.eq(layer)]
            reference = subset.loc[subset.mechanism.eq("H0_PARENT")]
            mechanisms = h.STRUCTURES if layer == "P1" else tuple(subset.mechanism.unique())
            for mechanism in mechanisms:
                candidate = subset.loc[subset.mechanism.eq(mechanism)]
                if candidate.empty:
                    continue
                metric_rows.append({
                    "model_id": model_id, "layer": layer, "mechanism": mechanism,
                    "station_macro_rmse_log1p": h.station_macro_rmse(candidate),
                    "tree_macro_rmse_log1p": h.tree_macro_rmse(candidate), "n": len(candidate),
                })
                if mechanism != "H0_PARENT":
                    for block in ("station_key", "terminal_tree_id"):
                        seed += 1
                        gate_rows.append({
                            "model_id": model_id, "layer": layer, "mechanism": mechanism,
                            "block": "station" if block == "station_key" else "tree",
                            **h.paired_bootstrap(reference, candidate, block, seed),
                        })
    pd.DataFrame(metric_rows).to_parquet(h.OUT / "temporal_performance.parquet", index=False)
    gates = pd.DataFrame(gate_rows)
    gates.to_parquet(h.OUT / "temporal_gate_matrix.parquet", index=False)
    selected_counts = structures.loc[structures.selected, "structure"].value_counts().to_dict()
    audit = {
        "status": "PASS",
        "models": int(predictions.model_id.nunique()),
        "folds": int(structures.fold_id.nunique()),
        "primary_stations": int(predictions.station_key.nunique()),
        "primary_trees": int(predictions.terminal_tree_id.nunique()),
        "selected_structure_counts": selected_counts,
        "all_structure_optimizers_successful": bool(structures.outer_success.all()),
        "all_readout_optimizers_successful": bool(readouts.success.all()),
        "exact_nesting_pass": bool(nesting["pass"].all()),
        "temperature_used": False, "TN_2022_read": False,
    }
    h.dump_json(h.REPORTS / "stage1_temporal_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
