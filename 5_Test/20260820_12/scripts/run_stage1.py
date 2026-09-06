from __future__ import annotations

import numpy as np
import pandas as pd

from a0_shared import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CACHE, FOLD_PATH, OUT, PARENT_OOF, REGISTERED_MONTHS,
    REPORTS, development_observations, dump_json, fit_harmonic, fit_uniform, formal_specs,
    metric_values, observation_basis, paired_bootstrap, parameter_row, parent_shared,
    predict_fit, require_runtime, season, station_macro_rmse, tree_macro_rmse,
)


def residual_arrays(reference: pd.DataFrame, candidate: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keys = ["station_key", "year", "month", "fold_id"]
    joined = reference[keys + ["tn_mg_l", "pred_tn_mg_l"]].merge(
        candidate[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_parent", "_candidate"), validate="one_to_one"
    )
    joined["res_parent"] = np.log1p(joined.tn_mg_l) - np.log1p(joined.pred_tn_mg_l_parent)
    joined["res_candidate"] = np.log1p(joined.tn_mg_l) - np.log1p(joined.pred_tn_mg_l_candidate)
    stations = sorted(joined.station_key.astype(str).unique())
    parent = np.full((len(stations), 12), np.nan)
    candidate_values = np.full((len(stations), 12), np.nan)
    for i, station in enumerate(stations):
        group = joined.loc[joined.station_key.astype(str).eq(station)]
        p = group.groupby("month").res_parent.mean()
        c = group.groupby("month").res_candidate.mean()
        for month in range(1, 13):
            if month in p.index:
                parent[i, month - 1] = float(p.loc[month])
                candidate_values[i, month - 1] = float(c.loc[month])
    return parent, candidate_values, stations


def fingerprint(values: np.ndarray) -> tuple[float, float, float, float]:
    b = np.nanmean(values, axis=0)
    registered = np.asarray([month - 1 for month in REGISTERED_MONTHS], dtype=int)
    j6 = float(np.sqrt(np.mean(b[registered] ** 2)))
    d23 = float(b[1] - b[2])
    d101112 = float(b[9] - 0.5 * (b[10] + b[11]))
    jphase = float(np.sqrt(0.5 * (d23 ** 2 + d101112 ** 2)))
    return j6, jphase, d23, d101112


def fingerprint_bootstrap(reference: pd.DataFrame, candidate: pd.DataFrame, seed_offset: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    parent, harmonic, stations = residual_arrays(reference, candidate)
    parent_point = np.nanmean(parent, axis=0)
    harmonic_point = np.nanmean(harmonic, axis=0)
    pj6, pjp, pd23, pd101112 = fingerprint(parent)
    hj6, hjp, hd23, hd101112 = fingerprint(harmonic)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    chosen = rng.integers(0, len(stations), size=(BOOTSTRAP_REPLICATES, len(stations)))
    deltas = np.zeros((BOOTSTRAP_REPLICATES, 4), dtype=float)
    for i, idx in enumerate(chosen):
        p = fingerprint(parent[idx])
        h = fingerprint(harmonic[idx])
        deltas[i] = [h[0] - p[0], h[1] - p[1], abs(h[2]) - abs(p[2]), abs(h[3]) - abs(p[3])]
    cis = np.quantile(deltas, [0.025, 0.975], axis=0)
    summary = {
        "station_blocks": len(stations),
        "parent_J6": pj6, "harmonic_J6": hj6, "delta_J6": hj6 - pj6,
        "delta_J6_ci_lower": float(cis[0, 0]), "delta_J6_ci_upper": float(cis[1, 0]), "J6_improved": bool(cis[1, 0] < 0),
        "parent_Jphase": pjp, "harmonic_Jphase": hjp, "delta_Jphase": hjp - pjp,
        "delta_Jphase_ci_lower": float(cis[0, 1]), "delta_Jphase_ci_upper": float(cis[1, 1]), "Jphase_improved": bool(cis[1, 1] < 0),
        "parent_D23": pd23, "harmonic_D23": hd23, "delta_abs_D23": abs(hd23) - abs(pd23),
        "delta_abs_D23_ci_upper": float(cis[1, 2]), "D23_significantly_shrunk": bool(cis[1, 2] < 0),
        "parent_D101112": pd101112, "harmonic_D101112": hd101112, "delta_abs_D101112": abs(hd101112) - abs(pd101112),
        "delta_abs_D101112_ci_upper": float(cis[1, 3]), "D101112_significantly_shrunk": bool(cis[1, 3] < 0),
        "at_least_one_contrast_significantly_shrunk": bool(cis[1, 2] < 0 or cis[1, 3] < 0),
        "contrast_max_abs_worsening": float(max(abs(hd23) - abs(pd23), abs(hd101112) - abs(pd101112))),
    }
    month_rows = []
    for month in REGISTERED_MONTHS:
        p = float(parent_point[month - 1])
        h = float(harmonic_point[month - 1])
        month_rows.append({
            "month": month, "parent_B": p, "harmonic_B": h,
            "delta_B": h - p, "delta_abs_B": abs(h) - abs(p), "absolute_bias_shrunk": bool(abs(h) < abs(p)),
        })
    summary["registered_months_abs_bias_shrunk"] = int(sum(row["absolute_bias_shrunk"] for row in month_rows))
    summary["registered_month_max_abs_worsening"] = float(max(row["delta_abs_B"] for row in month_rows))
    return summary, month_rows


def main() -> None:
    require_runtime()
    preflight = pd.read_json(REPORTS / "stage0_preflight.json", typ="series")
    if preflight["status"] != "PASS":
        raise RuntimeError("STOP_STAGE0_NOT_PASS")
    observations = development_observations()
    folds = pd.read_parquet(FOLD_PATH)[["fold_id", "train_start_year", "train_end_year", "evaluation_year"]].drop_duplicates().sort_values("evaluation_year")
    if folds.evaluation_year.tolist() != [2018, 2019, 2020, 2021]:
        raise RuntimeError("OOF year contract violation")
    shared = parent_shared()

    prediction_path = OUT / "temporal_oof_predictions.parquet"
    parameter_path = OUT / "candidate_fold_parameters.parquet"
    effect_path = OUT / "candidate_fold_station_effects.parquet"
    if prediction_path.exists() and parameter_path.exists() and effect_path.exists():
        predictions = pd.read_parquet(prediction_path)
        parameters = pd.read_parquet(parameter_path)
    else:
        prediction_parts: list[pd.DataFrame] = []
        parameter_rows: list[dict[str, object]] = []
        effect_rows: list[dict[str, object]] = []
        for model_index, spec in enumerate(formal_specs(), start=1):
            model_id = str(spec["model_id"])
            basis = observation_basis(model_id, observations)
            years = basis.base.year.to_numpy(int)
            for fold in folds.itertuples():
                train_idx = np.flatnonzero((years >= fold.train_start_year) & (years <= fold.train_end_year))
                test_idx = np.flatnonzero(years == fold.evaluation_year)
                for layer in ("P1", "P2"):
                    for mechanism, fit, is_parent in (
                        ("UNIFORM_PARENT", fit_uniform(shared, basis, train_idx, layer), True),
                        ("A0_HARMONIC", fit_harmonic(shared, basis, train_idx, layer), False),
                    ):
                        prediction = predict_fit(shared, basis, fit, test_idx, layer, is_parent)
                        prediction["model_id"] = model_id
                        prediction["fold_id"] = fold.fold_id
                        prediction["layer"] = layer
                        prediction["mechanism"] = mechanism
                        prediction_parts.append(prediction)
                        row = parameter_row(model_id, str(fold.fold_id), layer, mechanism, fit)
                        row["evaluation_year"] = int(fold.evaluation_year)
                        parameter_rows.append(row)
                        if layer == "P2":
                            effect_rows.extend({
                                "model_id": model_id, "fold_id": fold.fold_id, "mechanism": mechanism,
                                "station_key": station_key, "station_effect": value,
                            } for station_key, value in fit["effects"].items())
            print(f"stage1 temporal {model_index:02d}/12 {model_id}", flush=True)
        predictions = pd.concat(prediction_parts, ignore_index=True)
        predictions.to_parquet(prediction_path, index=False)
        parameters = pd.DataFrame(parameter_rows)
        parameters.to_parquet(parameter_path, index=False)
        parameters.loc[parameters.mechanism.eq("A0_HARMONIC")].to_parquet(OUT / "harmonic_weight_diagnostics.parquet", index=False)
        pd.DataFrame(effect_rows).to_parquet(effect_path, index=False)

    metric_rows = []
    for (model_id, mechanism, layer), group in predictions.groupby(["model_id", "mechanism", "layer"]):
        metric_rows.append({"model_id": model_id, "mechanism": mechanism, "layer": layer, "scope": "pooled", **metric_values(group)})
        metric_rows.append({"model_id": model_id, "mechanism": mechanism, "layer": layer, "scope": "station_macro", "rmse_log1p": station_macro_rmse(group), "n": len(group)})
        metric_rows.append({"model_id": model_id, "mechanism": mechanism, "layer": layer, "scope": "terminal_tree_macro", "rmse_log1p": tree_macro_rmse(group), "n": len(group)})
    pd.DataFrame(metric_rows).to_parquet(OUT / "temporal_oof_metrics.parquet", index=False)

    comparison_rows = []
    fingerprint_rows = []
    month_rows_all = []
    seed_offset = 0
    for spec in formal_specs():
        model_id = str(spec["model_id"])
        for layer in ("P1", "P2"):
            parent = predictions.loc[predictions.model_id.eq(model_id) & predictions.layer.eq(layer) & predictions.mechanism.eq("UNIFORM_PARENT")]
            candidate = predictions.loc[predictions.model_id.eq(model_id) & predictions.layer.eq(layer) & predictions.mechanism.eq("A0_HARMONIC")]
            for block in ("station_key", "terminal_tree_id"):
                seed_offset += 1
                comparison_rows.append({"model_id": model_id, "layer": layer, "block": block, **paired_bootstrap(parent, candidate, block, seed_offset)})
            seed_offset += 1
            fingerprint, month_rows = fingerprint_bootstrap(parent, candidate, seed_offset)
            fingerprint_rows.append({"model_id": model_id, "layer": layer, **fingerprint})
            month_rows_all.extend({"model_id": model_id, "layer": layer, **row} for row in month_rows)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_parquet(OUT / "temporal_paired_comparisons.parquet", index=False)
    fingerprints = pd.DataFrame(fingerprint_rows)
    fingerprints.to_parquet(OUT / "registered_fingerprint_metrics.parquet", index=False)
    pd.DataFrame(month_rows_all).to_parquet(OUT / "registered_month_direction_metrics.parquet", index=False)

    stability_rows = []
    harmonic_parameters = parameters.loc[parameters.mechanism.eq("A0_HARMONIC")]
    for (model_id, layer), group in harmonic_parameters.groupby(["model_id", "layer"]):
        evaluable = group.loc[group.phase_evaluable]
        counts = evaluable.peak_month.map(season).value_counts()
        stable = bool(len(evaluable) >= 3 and not counts.empty and counts.max() >= 3)
        stability_rows.append({
            "model_id": model_id, "layer": layer, "phase_evaluable_folds": int(len(evaluable)),
            "dominant_season": None if counts.empty else str(counts.index[0]),
            "dominant_season_fold_count": 0 if counts.empty else int(counts.iloc[0]),
            "seasonal_phase_stable": stable,
            "phase_status": "seasonal_phase_stable" if stable else "phase_not_identifiable" if len(evaluable) < 3 else "phase_unstable",
            "amplitude_boundary_fold_count": int(group.amplitude_boundary.sum()),
            "eta_boundary_fold_count": int(group.eta_boundary.sum()),
            "amplitude_boundary_model": bool(group.amplitude_boundary.sum() >= 2),
            "eta_boundary_model": bool(group.eta_boundary.sum() >= 2),
        })
    pd.DataFrame(stability_rows).to_parquet(OUT / "harmonic_fold_stability.parquet", index=False)

    parent_params = parameters.loc[parameters.mechanism.eq("UNIFORM_PARENT")].drop(columns=[column for column in parameters.columns if column.startswith("w_month_")])
    harmonic_params = parameters.loc[parameters.mechanism.eq("A0_HARMONIC")]
    keys = ["model_id", "fold_id", "layer"]
    eta = harmonic_params.merge(parent_params[keys + ["eta_quick", "eta_gw", "training_station_macro_rmse_log1p"]], on=keys, suffixes=("_A0", "_Parent"), validate="one_to_one")
    for pathway in ("quick", "gw"):
        eta[f"delta_eta_{pathway}"] = eta[f"eta_{pathway}_A0"] - eta[f"eta_{pathway}_Parent"]
        eta[f"relative_delta_eta_{pathway}"] = np.divide(
            eta[f"delta_eta_{pathway}"], eta[f"eta_{pathway}_Parent"], out=np.full(len(eta), np.nan), where=eta[f"eta_{pathway}_Parent"].abs() > 1e-12
        )
    eta["delta_training_rmse_log1p"] = eta.training_station_macro_rmse_log1p_A0 - eta.training_station_macro_rmse_log1p_Parent
    eta = eta.merge(fingerprints[keys[:1] + ["layer", "delta_J6", "delta_Jphase"]], on=["model_id", "layer"], how="left", validate="many_to_one")
    eta.to_parquet(OUT / "availability_eta_compensation_diagnostic.parquet", index=False)

    frozen = pd.read_parquet(PARENT_OOF)
    frozen = frozen.loc[frozen.mechanism.eq("PARENT")]
    reproduced = predictions.loc[predictions.mechanism.eq("UNIFORM_PARENT")]
    oof_keys = ["model_id", "layer", "station_key", "year", "month", "fold_id"]
    joined = frozen.merge(reproduced, on=oof_keys, suffixes=("_frozen", "_new"), validate="one_to_one")
    max_abs = float(np.max(np.abs(joined.pred_tn_mg_l_frozen - joined.pred_tn_mg_l_new)))
    reproduction = {"keys": len(joined), "expected_keys": len(frozen), "max_abs_prediction_mg_l": max_abs, "pass": bool(len(joined) == len(frozen) and max_abs <= 1e-12)}
    dump_json(REPORTS / "uniform_parent_oof_reproduction.json", reproduction)
    if not reproduction["pass"]:
        raise RuntimeError("STOP_UNIFORM_PARENT_OOF_NOT_REPRODUCED")
    dump_json(REPORTS / "stage1_temporal_audit.json", {"status": "PASS", "prediction_rows": len(predictions), "parameter_rows": len(parameters), "parent_reproduction": reproduction})


if __name__ == "__main__":
    main()
