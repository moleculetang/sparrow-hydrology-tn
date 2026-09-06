from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


RUN = Path(__file__).resolve().parents[1]
REFERENCE = RUN.parent / "20260727_6"
OUT = RUN / "reports" / "q72_q78_complementarity"
EPS = 1.0e-6

FOLDS = [
    {
        "fold_id": "fit_2006_2011_eval_2012_2013",
        "train_end": 2011,
        "eval_start": 2012,
        "eval_end": 2013,
    },
    {
        "fold_id": "fit_2006_2013_eval_2014_2015",
        "train_end": 2013,
        "eval_start": 2014,
        "eval_end": 2015,
    },
    {
        "fold_id": "fit_2006_2015_eval_2016_2018",
        "train_end": 2015,
        "eval_start": 2016,
        "eval_end": 2018,
    },
]

EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站"}
PROTECTED = "石角站"
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.01), 2)
MIN_CELL_MONTHS = 6


def load_reach_class_module():
    path = REFERENCE / "scripts" / "fit_reach_class_mass_model.py"
    spec = importlib.util.spec_from_file_location("reference_mass_reach_class", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred >= 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {
            "n": 0,
            "log_rmse": np.nan,
            "NSElog": np.nan,
            "PBIAS_pct": np.nan,
        }
    log_obs = np.log(obs + EPS)
    log_pred = np.log(pred + EPS)
    sse = float(np.sum((log_pred - log_obs) ** 2))
    denominator = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
    return {
        "n": int(len(obs)),
        "log_rmse": float(np.sqrt(sse / len(obs))),
        "NSElog": float(1.0 - sse / denominator) if denominator > 0 else np.nan,
        "PBIAS_pct": float(100.0 * np.sum(pred - obs) / np.sum(obs)),
    }


def fit_coefficients(
    obs_basis: pd.DataFrame,
    feature_names: list[str],
    train_end: int,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, object]]:
    columns = [f"routed_{name}_cfs" for name in feature_names]
    train = obs_basis[
        (obs_basis["year"] <= train_end)
        & (obs_basis["Q_obsv_cfs"] > 0)
    ].copy()
    train = train.dropna(subset=columns + ["Q_obsv_cfs"])
    if train.empty:
        raise RuntimeError(f"No Q78 training rows through {train_end}")
    if int(train["year"].max()) > train_end:
        raise RuntimeError("Q78 fit crossed the fold training cutoff")

    x = train[columns].to_numpy(dtype=float)
    y = train["Q_obsv_cfs"].to_numpy(dtype=float)
    scales = np.nanmedian(np.where(x > 0, x, np.nan), axis=0)
    global_scale = np.nanmedian(x[x > 0]) if np.any(x > 0) else 1.0
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, global_scale)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    x_scaled = x / scales
    y_log = np.log(y + EPS)
    prior_strength = 0.05

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = x_scaled @ theta
        log_residual = np.log(prediction + EPS) - y_log
        penalty = prior_strength * theta
        return np.concatenate([log_residual, penalty])

    result = least_squares(
        residual,
        np.full(len(feature_names), 0.02),
        bounds=(0.0, np.inf),
        max_nfev=5000,
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    scaled_coefficients = result.x
    coefficients = scaled_coefficients / scales
    coefficient_map = {
        name: float(value) for name, value in zip(feature_names, coefficients)
    }
    rows = []
    for index, name in enumerate(feature_names):
        reach_class, basis_name = name.split("__", 1)
        rows.append(
            {
                "reach_class": reach_class,
                "basis_name": basis_name,
                "feature_name": name,
                "coefficient": coefficient_map[name],
                "scaled_coefficient": float(scaled_coefficients[index]),
                "scale_cfs": float(scales[index]),
                "nonnegative_bound": 1,
            }
        )
    fit_info = {
        "training_rows": int(len(train)),
        "training_stations": int(train["q_site"].nunique()),
        "training_year_min": int(train["year"].min()),
        "training_year_max": int(train["year"].max()),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
        "minimum_coefficient": float(np.min(coefficients)),
        "maximum_coefficient": float(np.max(coefficients)),
    }
    return coefficient_map, pd.DataFrame(rows), fit_info


def normalize_q72_evaluation(path: Path, fold: dict[str, object]) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"q_site", "comid", "year", "month", "actual", "predict"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    frame = frame.rename(
        columns={
            "comid": "reach_id",
            "actual": "Q_obsv_cfs",
            "predict": "Q72_pred_cfs",
        }
    )
    frame["reach_id"] = pd.to_numeric(frame["reach_id"], errors="raise").astype(int)
    frame["year"] = frame["year"].astype(int)
    frame["month"] = frame["month"].astype(int)
    frame["q_site"] = frame["q_site"].astype(str)
    frame = frame[
        frame["year"].between(
            int(fold["eval_start"]),
            int(fold["eval_end"]),
        )
    ].copy()
    frame["fold_id"] = str(fold["fold_id"])
    frame["train_end"] = int(fold["train_end"])
    frame["eval_start"] = int(fold["eval_start"])
    frame["eval_end"] = int(fold["eval_end"])
    return frame[
        [
            "fold_id",
            "train_end",
            "eval_start",
            "eval_end",
            "q_site",
            "reach_id",
            "year",
            "month",
            "Q_obsv_cfs",
            "Q72_pred_cfs",
        ]
    ]


def load_fold_training_observations(
    path: Path,
    train_end: int,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["q_site", "comid", "year", "month", "actual"],
    )
    frame = frame.rename(columns={"comid": "reach_id", "actual": "Q_obsv_cfs"})
    frame["reach_id"] = pd.to_numeric(frame["reach_id"], errors="raise").astype(int)
    frame["year"] = frame["year"].astype(int)
    frame["month"] = frame["month"].astype(int)
    frame["q_site"] = frame["q_site"].astype(str)
    frame = frame[
        (frame["year"] <= train_end)
        & frame["Q_obsv_cfs"].notna()
        & (frame["Q_obsv_cfs"] > 0)
    ].copy()
    return frame.drop_duplicates(["q_site", "reach_id", "year", "month"])


def with_regime(frame: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    result = frame.merge(
        thresholds[["fold_id", "q_site", "training_q25_cfs", "training_q75_cfs"]],
        on=["fold_id", "q_site"],
        how="left",
        validate="many_to_one",
    )
    if result[["training_q25_cfs", "training_q75_cfs"]].isna().any().any():
        missing = result.loc[
            result["training_q25_cfs"].isna()
            | result["training_q75_cfs"].isna(),
            ["fold_id", "q_site"],
        ].drop_duplicates()
        raise RuntimeError(f"Missing training thresholds:\n{missing.to_string(index=False)}")
    result["flow_regime"] = np.where(
        result["Q_obsv_cfs"] <= result["training_q25_cfs"],
        "low",
        np.where(
            result["Q_obsv_cfs"] >= result["training_q75_cfs"],
            "high",
            "middle",
        ),
    )
    return result


def cell_metrics(part: pd.DataFrame, regime: str) -> dict[str, object]:
    obs = part["Q_obsv_cfs"].to_numpy(dtype=float)
    q72 = part["Q72_pred_cfs"].to_numpy(dtype=float)
    q78 = part["Q78_mass_cfs"].to_numpy(dtype=float)
    class_alpha = float(part["class_alpha"].iloc[0])
    original = (1.0 - class_alpha) * q72 + class_alpha * q78

    q72_metric = metric(obs, q72)
    q78_metric = metric(obs, q78)
    original_metric = metric(obs, original)
    log_obs = np.log(obs + EPS)
    residual72 = np.log(q72 + EPS) - log_obs
    residual78 = np.log(q78 + EPS) - log_obs
    residual_correlation = (
        float(np.corrcoef(residual72, residual78)[0, 1])
        if len(part) >= 3
        and np.std(residual72) > 0
        and np.std(residual78) > 0
        else np.nan
    )

    oracle_rmse = []
    for alpha in ALPHAS:
        prediction = (1.0 - alpha) * q72 + alpha * q78
        oracle_rmse.append(metric(obs, prediction)["log_rmse"])
    best_index = int(np.nanargmin(np.asarray(oracle_rmse, dtype=float)))
    oracle_alpha = float(ALPHAS[best_index])
    oracle_prediction = (1.0 - oracle_alpha) * q72 + oracle_alpha * q78
    oracle_metric = metric(obs, oracle_prediction)

    return {
        "fold_id": str(part["fold_id"].iloc[0]),
        "train_end": int(part["train_end"].iloc[0]),
        "eval_start": int(part["eval_start"].iloc[0]),
        "eval_end": int(part["eval_end"].iloc[0]),
        "q_site": str(part["q_site"].iloc[0]),
        "reach_id": int(part["reach_id"].iloc[0]),
        "reach_class": str(part["reach_class"].iloc[0]),
        "reservoir_flag": bool(part["reservoir_flag"].iloc[0]),
        "flow_regime": regime,
        "n": int(len(part)),
        "log_residual_correlation_q72_q78": residual_correlation,
        "q72_log_rmse": q72_metric["log_rmse"],
        "q72_NSElog": q72_metric["NSElog"],
        "q72_PBIAS_pct": q72_metric["PBIAS_pct"],
        "q78_log_rmse": q78_metric["log_rmse"],
        "q78_NSElog": q78_metric["NSElog"],
        "q78_PBIAS_pct": q78_metric["PBIAS_pct"],
        "original_class_alpha": class_alpha,
        "original_fusion_log_rmse": original_metric["log_rmse"],
        "original_fusion_NSElog": original_metric["NSElog"],
        "original_fusion_PBIAS_pct": original_metric["PBIAS_pct"],
        "original_fusion_log_rmse_gain_vs_q72": (
            q72_metric["log_rmse"] - original_metric["log_rmse"]
        ),
        "oracle_alpha": oracle_alpha,
        "oracle_log_rmse": oracle_metric["log_rmse"],
        "oracle_NSElog": oracle_metric["NSElog"],
        "oracle_PBIAS_pct": oracle_metric["PBIAS_pct"],
        "oracle_log_rmse_gain_vs_q72": (
            q72_metric["log_rmse"] - oracle_metric["log_rmse"]
        ),
        "oracle_NSElog_gain_vs_q72": (
            oracle_metric["NSElog"] - q72_metric["NSElog"]
        ),
        "oracle_abs_volume_bias_improvement_pct_points": (
            abs(q72_metric["PBIAS_pct"]) - abs(oracle_metric["PBIAS_pct"])
        ),
        "positive_oracle_gain": bool(
            q72_metric["log_rmse"] - oracle_metric["log_rmse"] > 1.0e-12
        ),
        "qualified_cell": bool(len(part) >= MIN_CELL_MONTHS),
    }


def summarize_cells(cells: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in cells.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        eligible = part[(~part["reservoir_flag"]) & part["qualified_cell"]]
        row.update(
            {
                "cells": int(len(part)),
                "stations": int(part["q_site"].nunique()),
                "reservoir_cells": int(part["reservoir_flag"].sum()),
                "eligible_nonreservoir_cells": int(len(eligible)),
                "eligible_nonreservoir_stations": int(eligible["q_site"].nunique()),
                "positive_station_fraction": float(
                    eligible.groupby("q_site")["oracle_log_rmse_gain_vs_q72"]
                    .median()
                    .gt(1.0e-12)
                    .mean()
                )
                if not eligible.empty
                else np.nan,
                "median_log_residual_correlation": float(
                    eligible["log_residual_correlation_q72_q78"].median()
                )
                if not eligible.empty
                else np.nan,
                "median_q72_log_rmse": float(eligible["q72_log_rmse"].median())
                if not eligible.empty
                else np.nan,
                "median_q78_log_rmse": float(eligible["q78_log_rmse"].median())
                if not eligible.empty
                else np.nan,
                "median_original_fusion_gain": float(
                    eligible["original_fusion_log_rmse_gain_vs_q72"].median()
                )
                if not eligible.empty
                else np.nan,
                "median_oracle_log_rmse_gain": float(
                    eligible["oracle_log_rmse_gain_vs_q72"].median()
                )
                if not eligible.empty
                else np.nan,
                "median_oracle_alpha": float(eligible["oracle_alpha"].median())
                if not eligible.empty
                else np.nan,
                "median_oracle_NSElog_gain": float(
                    eligible["oracle_NSElog_gain_vs_q72"].median()
                )
                if not eligible.empty
                else np.nan,
                "median_oracle_abs_volume_bias_improvement_pct_points": float(
                    eligible["oracle_abs_volume_bias_improvement_pct_points"].median()
                )
                if not eligible.empty
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_eligibility_gate(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    eligible = cells[(~cells["reservoir_flag"]) & cells["qualified_cell"]].copy()
    fold_summary = summarize_cells(eligible, ["fold_id", "flow_regime"])
    rows = []
    for regime, part in eligible.groupby("flow_regime", sort=True):
        station_gain = (
            part.groupby("q_site")["oracle_log_rmse_gain_vs_q72"].median()
        )
        folds = fold_summary[fold_summary["flow_regime"].eq(regime)].copy()
        direction_count = int(
            folds["median_oracle_log_rmse_gain"].gt(1.0e-12).sum()
        )
        positive_fraction = float(station_gain.gt(1.0e-12).mean())
        median_gain = float(part["oracle_log_rmse_gain_vs_q72"].median())
        median_correlation = float(
            part["log_residual_correlation_q72_q78"].median()
        )
        gates = {
            "direction_consistent_at_least_2_of_3_folds": direction_count >= 2,
            "positive_station_fraction_at_least_0_60": positive_fraction >= 0.60,
            "median_log_rmse_reduction_at_least_0_02": median_gain >= 0.02,
            "median_residual_correlation_below_0_80": median_correlation < 0.80,
        }
        rows.append(
            {
                "flow_regime": regime,
                "eligible_stations": int(station_gain.size),
                "eligible_cells": int(len(part)),
                "direction_positive_folds": direction_count,
                "positive_station_fraction": positive_fraction,
                "median_oracle_log_rmse_reduction": median_gain,
                "median_log_residual_correlation": median_correlation,
                **gates,
                "regime_eligible_for_conditional_fusion": bool(all(gates.values())),
            }
        )
    gate_frame = pd.DataFrame(rows)
    eligible_regimes = gate_frame.loc[
        gate_frame["regime_eligible_for_conditional_fusion"],
        "flow_regime",
    ].tolist()
    payload = {
        "run_id": RUN.name,
        "phase_id": "q72_q78_oof_complementarity",
        "reference_run": REFERENCE.name,
        "diagnostic_only": True,
        "oracle_weights_deployable": False,
        "thresholds": {
            "minimum_direction_consistent_folds": 2,
            "minimum_positive_station_fraction": 0.60,
            "minimum_median_log_rmse_reduction": 0.02,
            "maximum_median_log_residual_correlation": 0.80,
            "minimum_cell_months": MIN_CELL_MONTHS,
        },
        "eligible_regimes": eligible_regimes,
        "conditional_fusion_upper_bound_supported": bool(eligible_regimes),
        "decision": (
            "allow_one_preregistered_conditional_fusion_candidate_later"
            if eligible_regimes
            else "do_not_develop_conditional_fusion_from_current_branches"
        ),
        "regimes": gate_frame.to_dict(orient="records"),
    }
    return gate_frame, payload


def main() -> None:
    started = datetime.now().astimezone()
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_reach_class_module()
    topology = module.BASE.load_topology()
    order, topology_issues = module.BASE.topological_order(topology)
    if topology_issues:
        raise RuntimeError("Topology issues: " + "; ".join(topology_issues))

    forcing_path = (
        REFERENCE
        / "reports"
        / "intermediate"
        / "mass_reach_class"
        / "reach_month_forcing_panel.csv"
    )
    basis_path = (
        REFERENCE
        / "reports"
        / "intermediate"
        / "mass_reach_class"
        / "routed_class_local_source_basis.csv"
    )
    forcing = pd.read_csv(forcing_path, encoding="utf-8-sig")
    routed_basis = pd.read_csv(basis_path, encoding="utf-8-sig")
    feature_names = [
        column.removeprefix("routed_").removesuffix("_cfs")
        for column in routed_basis.columns
        if column.startswith("routed_") and column.endswith("_cfs")
    ]
    if len(feature_names) != 40:
        raise RuntimeError(f"Expected 40 reach-class features, found {len(feature_names)}")

    alpha_table = pd.read_csv(
        REFERENCE / "reports" / "main_model" / "selected_reach_class_alpha.csv",
        encoding="utf-8-sig",
    )
    alpha_map = alpha_table.set_index("reach_class")["class_alpha"].to_dict()
    labels = pd.read_csv(
        REFERENCE / "reports" / "main_model" / "station_diagnostic_labels.csv",
        encoding="utf-8-sig",
    )
    reservoir_map = (
        labels.drop_duplicates("q_site")
        .set_index("q_site")["reservoir_related"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
        .to_dict()
    )

    oof_parts = []
    threshold_parts = []
    coefficient_parts = []
    fit_rows = []
    mass_rows = []
    for fold in FOLDS:
        fold_id = str(fold["fold_id"])
        fold_root = (
            REFERENCE
            / "reports"
            / "station_screening"
            / "blocked_folds"
            / fold_id
        )
        q72_eval = normalize_q72_evaluation(
            fold_root / "evaluation_predictions.csv",
            fold,
        )
        training = load_fold_training_observations(
            fold_root
            / "reports"
            / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv",
            int(fold["train_end"]),
        )
        thresholds = (
            training.groupby("q_site")["Q_obsv_cfs"]
            .agg(
                training_months="count",
                training_q25_cfs=lambda values: float(values.quantile(0.25)),
                training_q75_cfs=lambda values: float(values.quantile(0.75)),
            )
            .reset_index()
        )
        thresholds.insert(0, "fold_id", fold_id)
        thresholds["train_end"] = int(fold["train_end"])
        threshold_parts.append(thresholds)

        obs_basis = training.merge(
            routed_basis,
            on=["reach_id", "year", "month"],
            how="left",
            validate="many_to_one",
        )
        coefficients, coefficient_frame, fit_info = fit_coefficients(
            obs_basis,
            feature_names,
            int(fold["train_end"]),
        )
        coefficient_frame.insert(0, "fold_id", fold_id)
        coefficient_frame.insert(1, "train_end", int(fold["train_end"]))
        coefficient_parts.append(coefficient_frame)

        route_panel = forcing[forcing["year"] <= int(fold["eval_end"])].copy()
        routed = module.route_with_class_coefficients(
            route_panel,
            topology,
            order,
            coefficients,
        )
        max_mass_residual = float(routed["mass_balance_residual_cfs"].abs().max())
        max_relative_residual = float(
            routed["relative_mass_balance_residual"].abs().max()
        )
        mass_rows.append(
            {
                "fold_id": fold_id,
                "routed_year_min": int(routed["year"].min()),
                "routed_year_max": int(routed["year"].max()),
                "routed_rows": int(len(routed)),
                "negative_Q_out_rows": int((routed["Q_out_cfs"] < 0).sum()),
                "max_abs_mass_balance_residual_cfs": max_mass_residual,
                "max_abs_relative_mass_balance_residual": max_relative_residual,
            }
        )
        q78_eval = routed[
            routed["year"].between(
                int(fold["eval_start"]),
                int(fold["eval_end"]),
            )
        ][
            [
                "reach_id",
                "year",
                "month",
                "Q_out_cfs",
                "reach_class",
                "is_reservoir_reach",
            ]
        ].rename(columns={"Q_out_cfs": "Q78_mass_cfs"})
        joined = q72_eval.merge(
            q78_eval,
            on=["reach_id", "year", "month"],
            how="left",
            validate="many_to_one",
        )
        joined["reservoir_flag"] = (
            joined["q_site"].map(reservoir_map).fillna(False).astype(bool)
        )
        joined["class_alpha"] = joined["reach_class"].map(alpha_map)
        if joined[["Q78_mass_cfs", "reach_class", "class_alpha"]].isna().any().any():
            raise RuntimeError(f"Q78 merge incomplete for {fold_id}")
        oof_parts.append(joined)
        fit_rows.append(
            {
                "fold_id": fold_id,
                "train_end": int(fold["train_end"]),
                "eval_start": int(fold["eval_start"]),
                "eval_end": int(fold["eval_end"]),
                "q72_eval_rows": int(len(q72_eval)),
                "q72_eval_stations": int(q72_eval["q_site"].nunique()),
                "q78_eval_rows": int(len(q78_eval)),
                "joined_eval_rows": int(len(joined)),
                "joined_eval_stations": int(joined["q_site"].nunique()),
                "excluded_station_rows": int(joined["q_site"].isin(EXCLUSIONS).sum()),
                "protected_station_rows": int(joined["q_site"].eq(PROTECTED).sum()),
                **fit_info,
            }
        )
        print(
            f"{fold_id}: fit_rows={fit_info['training_rows']} "
            f"eval_rows={len(joined)} max_mass_residual={max_mass_residual:.3e}",
            flush=True,
        )

    thresholds_all = pd.concat(threshold_parts, ignore_index=True)
    oof = with_regime(pd.concat(oof_parts, ignore_index=True), thresholds_all)
    oof["Q_original_fusion_cfs"] = (
        (1.0 - oof["class_alpha"]) * oof["Q72_pred_cfs"]
        + oof["class_alpha"] * oof["Q78_mass_cfs"]
    )
    key = ["fold_id", "q_site", "reach_id", "year", "month"]
    if oof.duplicated(key).any():
        raise RuntimeError("OOF prediction key is not unique")

    cell_rows = []
    for (_, _), part in oof.groupby(["fold_id", "q_site"], sort=True):
        cell_rows.append(cell_metrics(part, "all"))
        for regime, regime_part in part.groupby("flow_regime", sort=True):
            cell_rows.append(cell_metrics(regime_part, str(regime)))
    cells = pd.DataFrame(cell_rows)
    fold_summary = summarize_cells(cells, ["fold_id", "flow_regime"])
    reach_class_summary = summarize_cells(
        cells,
        ["reach_class", "flow_regime"],
    )
    gate_frame, gate_payload = build_eligibility_gate(cells)

    oof.to_csv(OUT / "oof_predictions_2012_2018.csv", index=False, encoding="utf-8-sig")
    thresholds_all.to_csv(OUT / "training_flow_thresholds.csv", index=False, encoding="utf-8-sig")
    pd.concat(coefficient_parts, ignore_index=True).to_csv(
        OUT / "q78_fold_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fit_rows).to_csv(
        OUT / "fold_fit_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(mass_rows).to_csv(
        OUT / "mass_balance_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cells.to_csv(
        OUT / "station_regime_fold_oracle.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_summary.to_csv(
        OUT / "fold_regime_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reach_class_summary.to_csv(
        OUT / "reach_class_regime_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate_frame.to_csv(
        OUT / "conditional_fusion_eligibility.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "conditional_fusion_eligibility.json").write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "run_id": RUN.name,
        "reference_run": REFERENCE.name,
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "folds": len(FOLDS),
        "oof_rows": int(len(oof)),
        "oof_stations": int(oof["q_site"].nunique()),
        "reservoir_stations": int(
            oof.loc[oof["reservoir_flag"], "q_site"].nunique()
        ),
        "protected_station_present": bool(oof["q_site"].eq(PROTECTED).any()),
        "forbidden_exclusion_rows": int(oof["q_site"].isin(EXCLUSIONS).sum()),
        "maximum_evidence_year": int(oof["year"].max()),
        "maximum_q78_training_year": int(
            pd.DataFrame(fit_rows)["training_year_max"].max()
        ),
        "maximum_mass_balance_residual_cfs": float(
            pd.DataFrame(mass_rows)["max_abs_mass_balance_residual_cfs"].max()
        ),
        "conditional_fusion_upper_bound_supported": gate_payload[
            "conditional_fusion_upper_bound_supported"
        ],
        "eligible_regimes": gate_payload["eligible_regimes"],
        "oracle_weights_deployable": False,
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
