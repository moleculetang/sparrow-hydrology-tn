from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RUN = Path(__file__).resolve().parents[1]
REFERENCE = RUN.parent / "20260727_6"
PREDECESSOR = RUN.parent / "20260728_4"
OUT = RUN / "reports" / "low_flow_heterogeneity"
RANDOM_SEED = 1729
PERMUTATIONS = 500
MIN_LOW_MONTHS = 3
MIN_ELIGIBLE_FOLDS = 2
MATERIAL_BIAS = 10.0
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站"}
PROTECTED = "石角站"
HETEROGENEITY_CLASSES = [
    "reservoir_or_regulation_possible",
    "data_quality_suspicious",
    "stable_low_flow_overprediction",
    "material_sign_reversal",
    "episodic_low_flow_overprediction",
    "not_stably_overpredicted",
]

STATIC_NUMERIC = [
    "log_inc_area_km2",
    "log_tot_area_km2",
    "log_length_km",
    "local_area_fraction",
    "upstream_count",
    "headwater",
    "terminal",
    "frac",
    "climate_ppt_mean",
    "climate_pet_mean",
    "climate_aet_mean",
    "climate_aridity",
    "climate_surplus_ratio",
    "climate_ppt_cv",
    "climate_wetness_mean",
]
LOCATION_NUMERIC = ["location_x", "location_y"]
SIGNATURE_NUMERIC = [
    "signature_log_mean_q",
    "signature_cv_q",
    "signature_q10_q50_ratio",
    "signature_q90_q50_ratio",
    "signature_dry_wet_ratio",
    "signature_monthly_climatology_cv",
    "signature_logq_lag1",
]


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)


def upstream_count(value: object) -> int:
    if pd.isna(value) or str(value).strip() in {"", "[]"}:
        return 0
    try:
        parsed = ast.literal_eval(str(value))
        return len(parsed) if isinstance(parsed, (list, tuple)) else 0
    except (ValueError, SyntaxError):
        return len([item for item in str(value).split(",") if item.strip()])


def safe_cv(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    return float(np.std(array, ddof=0) / mean) if mean != 0 else np.nan


def lag1_logq(part: pd.DataFrame) -> float:
    ordered = part.sort_values(["year", "month"])
    values = np.log(ordered["Q_obsv_cfs"].to_numpy(dtype=float) + 1.0e-6)
    if len(values) < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def load_fold_biases() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = (
        PREDECESSOR
        / "reports"
        / "q72_q78_complementarity"
        / "oof_predictions_2012_2018.csv"
    )
    oof = pd.read_csv(path, encoding="utf-8-sig")
    low = oof[oof["flow_regime"].eq("low")].copy()
    rows = []
    for (fold_id, site), part in low.groupby(["fold_id", "q_site"], sort=True):
        obs = part["Q_obsv_cfs"].to_numpy(dtype=float)
        pred = part["Q_original_fusion_cfs"].to_numpy(dtype=float)
        bias = float(100.0 * np.sum(pred - obs) / np.sum(obs))
        log_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        np.log(pred + 1.0e-6)
                        - np.log(obs + 1.0e-6)
                    )
                    ** 2
                )
            )
        )
        rows.append(
            {
                "fold_id": fold_id,
                "train_end": int(part["train_end"].iloc[0]),
                "eval_start": int(part["eval_start"].iloc[0]),
                "eval_end": int(part["eval_end"].iloc[0]),
                "q_site": site,
                "reach_id": int(part["reach_id"].iloc[0]),
                "reach_class": str(part["reach_class"].iloc[0]),
                "reservoir_flag_oof": bool(part["reservoir_flag"].iloc[0]),
                "low_months": int(len(part)),
                "low_flow_volume_bias_pct": bias,
                "low_flow_log_rmse": log_rmse,
                "fold_eligible": bool(len(part) >= MIN_LOW_MONTHS),
            }
        )
    return oof, pd.DataFrame(rows)


def build_static_attributes() -> pd.DataFrame:
    forcing_path = (
        REFERENCE
        / "reports"
        / "intermediate"
        / "mass_reach_class"
        / "reach_month_forcing_panel.csv"
    )
    forcing = pd.read_csv(forcing_path, encoding="utf-8-sig")
    static_columns = [
        "reach_id",
        "inc_area_km2",
        "tot_area_km2",
        "length_km",
        "upstream_ids",
        "frac",
        "headwater",
        "terminal",
        "reach_class",
    ]
    static = forcing[static_columns].drop_duplicates("reach_id").copy()
    static["log_inc_area_km2"] = np.log1p(static["inc_area_km2"].astype(float))
    static["log_tot_area_km2"] = np.log1p(static["tot_area_km2"].astype(float))
    static["log_length_km"] = np.log1p(static["length_km"].astype(float))
    static["local_area_fraction"] = (
        static["inc_area_km2"].astype(float)
        / static["tot_area_km2"].astype(float).clip(lower=1.0e-9)
    )
    static["upstream_count"] = static["upstream_ids"].apply(upstream_count)
    static["headwater"] = static["headwater"].fillna(0).astype(int)
    static["terminal"] = static["terminal"].fillna(0).astype(int)
    static["frac"] = static["frac"].fillna(1.0).astype(float)

    climate = forcing[forcing["year"].between(2006, 2011)].copy()
    climate_rows = []
    for reach_id, part in climate.groupby("reach_id", sort=True):
        ppt_mean = float(part["PPT"].mean())
        pet_mean = float(part["PET"].mean())
        aet_mean = float(part["AET"].mean())
        climate_rows.append(
            {
                "reach_id": int(reach_id),
                "climate_months": int(len(part)),
                "climate_year_min": int(part["year"].min()),
                "climate_year_max": int(part["year"].max()),
                "climate_ppt_mean": ppt_mean,
                "climate_pet_mean": pet_mean,
                "climate_aet_mean": aet_mean,
                "climate_aridity": pet_mean / max(ppt_mean, 1.0e-6),
                "climate_surplus_ratio": max(ppt_mean - aet_mean, 0.0)
                / max(ppt_mean, 1.0e-6),
                "climate_ppt_cv": safe_cv(part["PPT"]),
                "climate_wetness_mean": float(part["Wetness"].mean()),
            }
        )
    climate_attributes = pd.DataFrame(climate_rows)
    keep = ["reach_id", "reach_class", *STATIC_NUMERIC]
    return static.merge(
        climate_attributes,
        on="reach_id",
        how="left",
        validate="one_to_one",
    )[keep + ["climate_months", "climate_year_min", "climate_year_max"]]


def build_gauged_signatures() -> pd.DataFrame:
    path = (
        REFERENCE
        / "reports"
        / "station_screening"
        / "blocked_folds"
        / "fit_2006_2011_eval_2012_2013"
        / "reports"
        / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv"
    )
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["q_site", "comid", "year", "month", "actual"],
    ).rename(columns={"comid": "reach_id", "actual": "Q_obsv_cfs"})
    frame = frame[
        frame["year"].between(2006, 2011)
        & frame["Q_obsv_cfs"].notna()
        & frame["Q_obsv_cfs"].gt(0)
    ].copy()
    rows = []
    for site, part in frame.groupby("q_site", sort=True):
        values = part["Q_obsv_cfs"].astype(float)
        q10, q50, q90 = values.quantile([0.10, 0.50, 0.90]).tolist()
        wet = part[part["month"].between(4, 9)]["Q_obsv_cfs"].mean()
        dry = part[~part["month"].between(4, 9)]["Q_obsv_cfs"].mean()
        monthly = part.groupby("month")["Q_obsv_cfs"].mean()
        rows.append(
            {
                "q_site": str(site),
                "signature_reach_id": int(float(part["reach_id"].iloc[0])),
                "signature_months": int(len(part)),
                "signature_year_min": int(part["year"].min()),
                "signature_year_max": int(part["year"].max()),
                "signature_log_mean_q": float(np.log(values.mean() + 1.0e-6)),
                "signature_cv_q": safe_cv(values),
                "signature_q10_q50_ratio": float(q10 / max(q50, 1.0e-6)),
                "signature_q90_q50_ratio": float(q90 / max(q50, 1.0e-6)),
                "signature_dry_wet_ratio": float(dry / max(wet, 1.0e-6)),
                "signature_monthly_climatology_cv": safe_cv(monthly),
                "signature_logq_lag1": lag1_logq(part),
            }
        )
    return pd.DataFrame(rows)


def build_quality_and_context() -> pd.DataFrame:
    reliability = pd.read_csv(
        REFERENCE
        / "reports"
        / "input_preprocessing"
        / "same_reach_station_reliability.csv",
        encoding="utf-8-sig",
    )
    selected = reliability[bool_series(reliability["selected_for_reach"])].copy()
    selected = selected.sort_values(
        ["station_name", "usable_months"],
        ascending=[True, False],
    ).drop_duplicates("station_name")
    selected["topology_flow_plausible_bool"] = bool_series(
        selected["topology_flow_plausible"]
    )
    selected["adequate_coverage_bool"] = bool_series(
        selected["adequate_2010_2022_coverage"]
    )
    selected["small_flow_bool"] = bool_series(
        selected["small_flow_relative_to_reach"]
    )
    reason_columns = []
    for name, condition in [
        ("topology_flow_implausible", ~selected["topology_flow_plausible_bool"]),
        ("inadequate_2010_2022_coverage", ~selected["adequate_coverage_bool"]),
        ("usable_years_below_10", selected["usable_years"].fillna(0).lt(10)),
        ("small_flow_relative_to_reach", selected["small_flow_bool"]),
        ("snap_distance_above_5km", selected["snap_distance_m"].fillna(0).gt(5000)),
    ]:
        selected[name] = condition
        reason_columns.append(name)
    selected["data_quality_suspicious"] = selected[reason_columns].any(axis=1)
    selected["data_quality_reasons"] = selected.apply(
        lambda row: ";".join(name for name in reason_columns if bool(row[name])),
        axis=1,
    )

    labels = pd.read_csv(
        REFERENCE / "reports" / "main_model" / "station_diagnostic_labels.csv",
        encoding="utf-8-sig",
    ).drop_duplicates("q_site")
    labels["reservoir_related_bool"] = bool_series(labels["reservoir_related"])

    locations = pd.read_csv(
        REFERENCE
        / "reports"
        / "input_preprocessing"
        / "station_reach_match.csv",
        encoding="utf-8-sig",
    )
    locations = locations[bool_series(locations["used"])].copy()
    locations = locations.sort_values(
        ["station_norm", "usable_months"],
        ascending=[True, False],
    ).drop_duplicates("station_norm")
    locations = locations.rename(
        columns={
            "station_norm": "q_site",
            "x": "location_x",
            "y": "location_y",
        }
    )

    quality = selected.rename(columns={"station_name": "q_site"})[
        [
            "q_site",
            "usable_months",
            "usable_years",
            "snap_distance_m",
            "obs_to_reach_flow_ratio",
            "data_quality_suspicious",
            "data_quality_reasons",
        ]
    ]
    context = labels[
        [
            "q_site",
            "reservoir_related_bool",
            "reservoir_relation",
            "nearest_upstream_reservoir_name",
        ]
    ]
    return quality.merge(context, on="q_site", how="outer").merge(
        locations[["q_site", "location_x", "location_y"]],
        on="q_site",
        how="left",
    )


def classify_stations(
    oof: pd.DataFrame,
    fold_biases: pd.DataFrame,
    attributes: pd.DataFrame,
    signatures: pd.DataFrame,
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_base = (
        oof[
            ["q_site", "reach_id", "reach_class", "reservoir_flag"]
        ]
        .drop_duplicates("q_site")
        .copy()
    )
    eligible_bias = fold_biases[fold_biases["fold_eligible"]].copy()
    summaries = []
    for site, part in fold_biases.groupby("q_site", sort=True):
        eligible = part[part["fold_eligible"]]
        biases = eligible["low_flow_volume_bias_pct"].to_numpy(dtype=float)
        summaries.append(
            {
                "q_site": site,
                "folds_available": int(len(part)),
                "eligible_folds": int(len(eligible)),
                "low_months_total": int(eligible["low_months"].sum()),
                "median_low_flow_bias_pct": float(np.median(biases))
                if len(biases)
                else np.nan,
                "mean_low_flow_bias_pct": float(np.mean(biases))
                if len(biases)
                else np.nan,
                "min_low_flow_bias_pct": float(np.min(biases))
                if len(biases)
                else np.nan,
                "max_low_flow_bias_pct": float(np.max(biases))
                if len(biases)
                else np.nan,
                "positive_fold_fraction": float(np.mean(biases > 0))
                if len(biases)
                else np.nan,
                "material_positive_fold_fraction": float(
                    np.mean(biases >= MATERIAL_BIAS)
                )
                if len(biases)
                else np.nan,
                "material_negative_fold_fraction": float(
                    np.mean(biases <= -MATERIAL_BIAS)
                )
                if len(biases)
                else np.nan,
                "bias_sign_changes": bool(
                    len(biases)
                    and np.any(biases >= MATERIAL_BIAS)
                    and np.any(biases <= -MATERIAL_BIAS)
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    stations = (
        station_base.merge(summary, on="q_site", how="left")
        .merge(attributes, on=["reach_id", "reach_class"], how="left")
        .merge(signatures, on="q_site", how="left")
        .merge(context, on="q_site", how="left")
    )
    stations["reservoir_related_bool"] = (
        stations["reservoir_related_bool"].fillna(stations["reservoir_flag"])
        .astype(bool)
    )
    stations["data_quality_suspicious"] = (
        stations["data_quality_suspicious"].fillna(True).astype(bool)
    )
    stations["eligible_for_heterogeneity_label"] = (
        stations["eligible_folds"].fillna(0).ge(MIN_ELIGIBLE_FOLDS)
    )
    stable = (
        stations["eligible_for_heterogeneity_label"]
        & stations["positive_fold_fraction"].ge(2.0 / 3.0)
        & stations["median_low_flow_bias_pct"].ge(MATERIAL_BIAS)
        & stations["material_negative_fold_fraction"].eq(0)
    )
    sign_reversal = (
        stations["eligible_for_heterogeneity_label"]
        & stations["bias_sign_changes"]
    )
    episodic = (
        stations["eligible_for_heterogeneity_label"]
        & (
            stations["max_low_flow_bias_pct"].gt(0)
            | stations["median_low_flow_bias_pct"].gt(0)
        )
    )
    stations["heterogeneity_class"] = np.select(
        [
            stations["reservoir_related_bool"],
            stations["data_quality_suspicious"],
            stable,
            sign_reversal,
            episodic,
        ],
        [
            "reservoir_or_regulation_possible",
            "data_quality_suspicious",
            "stable_low_flow_overprediction",
            "material_sign_reversal",
            "episodic_low_flow_overprediction",
        ],
        default="not_stably_overpredicted",
    )
    stations["stable_low_flow_target"] = stations["heterogeneity_class"].eq(
        "stable_low_flow_overprediction"
    )
    stations["eligible_for_separability"] = (
        stations["eligible_for_heterogeneity_label"]
        & ~stations["reservoir_related_bool"]
        & ~stations["data_quality_suspicious"]
    )
    stations["active_exclusion_violation"] = stations["q_site"].isin(EXCLUSIONS)
    return stations, eligible_bias


def make_pipeline(numeric_columns: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_columns),
            (
                "reach_class",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["reach_class"],
            ),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("transform", transformer), ("model", model)])


def evaluate_model(
    stations: pd.DataFrame,
    model_id: str,
    numeric_columns: list[str],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = ["q_site", "reach_id", "reach_class", "stable_low_flow_target", *numeric_columns]
    data = stations[stations["eligible_for_separability"]][columns].dropna().copy()
    y = data["stable_low_flow_target"].astype(int).to_numpy()
    positives = int(np.sum(y))
    negatives = int(len(y) - positives)
    result = {
        "model_id": model_id,
        "rows": int(len(data)),
        "positive_stations": positives,
        "negative_stations": negatives,
        "numeric_features": len(numeric_columns),
        "categorical_features": 1,
        "cv_folds": 5,
        "status": "not_run_insufficient_class",
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "balanced_accuracy_at_0_5": np.nan,
        "sensitivity_at_0_5": np.nan,
        "specificity_at_0_5": np.nan,
        "permutation_pvalue_auc": np.nan,
        "permutations": PERMUTATIONS,
    }
    empty = pd.DataFrame()
    if positives < 5 or negatives < 5:
        return result, empty, empty, empty

    x = data[["reach_class", *numeric_columns]]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    splits = list(splitter.split(x, y))
    pipeline = make_pipeline(numeric_columns)
    probability = cross_val_predict(
        pipeline,
        x,
        y,
        cv=splits,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    observed_auc = float(roc_auc_score(y, probability))

    rng = np.random.default_rng(RANDOM_SEED)
    permutation_rows = []
    for index in range(PERMUTATIONS):
        permuted_y = rng.permutation(y)
        permuted_probability = cross_val_predict(
            clone(pipeline),
            x,
            permuted_y,
            cv=splits,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        permutation_rows.append(
            {
                "model_id": model_id,
                "permutation": index + 1,
                "roc_auc": float(roc_auc_score(permuted_y, permuted_probability)),
            }
        )
    permutation_frame = pd.DataFrame(permutation_rows)
    pvalue = float(
        (1 + (permutation_frame["roc_auc"] >= observed_auc).sum())
        / (1 + PERMUTATIONS)
    )

    fitted = clone(pipeline).fit(x, y)
    feature_names = fitted.named_steps["transform"].get_feature_names_out()
    coefficients = fitted.named_steps["model"].coef_[0]
    coefficient_frame = pd.DataFrame(
        {
            "model_id": model_id,
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)
    prediction_frame = data[["q_site", "reach_id", "reach_class"]].copy()
    prediction_frame["model_id"] = model_id
    prediction_frame["stable_low_flow_target"] = y
    prediction_frame["oof_probability"] = probability
    prediction_frame["oof_prediction_at_0_5"] = predicted

    result.update(
        {
            "status": "completed",
            "roc_auc": observed_auc,
            "average_precision": float(average_precision_score(y, probability)),
            "balanced_accuracy_at_0_5": float(
                balanced_accuracy_score(y, predicted)
            ),
            "sensitivity_at_0_5": float(tp / max(tp + fn, 1)),
            "specificity_at_0_5": float(tn / max(tn + fp, 1)),
            "permutation_pvalue_auc": pvalue,
        }
    )
    return result, prediction_frame, coefficient_frame, permutation_frame


def feature_class_summary(stations: pd.DataFrame) -> pd.DataFrame:
    eligible = stations[stations["eligible_for_separability"]].copy()
    rows = []
    for feature in [*STATIC_NUMERIC, *SIGNATURE_NUMERIC]:
        if feature not in eligible.columns:
            continue
        for target, part in eligible.groupby("stable_low_flow_target", sort=True):
            rows.append(
                {
                    "feature": feature,
                    "stable_low_flow_target": bool(target),
                    "stations": int(part[feature].notna().sum()),
                    "median": float(part[feature].median()),
                    "q25": float(part[feature].quantile(0.25)),
                    "q75": float(part[feature].quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    started = datetime.now().astimezone()
    OUT.mkdir(parents=True, exist_ok=True)
    predecessor_gate = json.loads(
        (PREDECESSOR / "reports" / "gate.json").read_text(encoding="utf-8")
    )
    if not predecessor_gate.get("gate_passed"):
        raise RuntimeError("20260728_4 diagnostic integrity gate did not pass")

    oof, fold_biases = load_fold_biases()
    attributes = build_static_attributes()
    signatures = build_gauged_signatures()
    context = build_quality_and_context()
    stations, eligible_biases = classify_stations(
        oof,
        fold_biases,
        attributes,
        signatures,
        context,
    )

    models = [
        ("deployable_static", STATIC_NUMERIC),
        ("static_plus_location_sensitivity", [*STATIC_NUMERIC, *LOCATION_NUMERIC]),
        (
            "static_plus_gauged_signature_upper_bound",
            [*STATIC_NUMERIC, *SIGNATURE_NUMERIC],
        ),
    ]
    model_rows = []
    prediction_parts = []
    coefficient_parts = []
    permutation_parts = []
    for model_id, numeric_columns in models:
        result, predictions, coefficients, permutations = evaluate_model(
            stations,
            model_id,
            numeric_columns,
        )
        model_rows.append(result)
        if not predictions.empty:
            prediction_parts.append(predictions)
            coefficient_parts.append(coefficients)
            permutation_parts.append(permutations)
        print(
            f"{model_id}: status={result['status']} rows={result['rows']} "
            f"positive={result['positive_stations']} auc={result['roc_auc']}",
            flush=True,
        )

    model_results = pd.DataFrame(model_rows)
    stable_sites = stations[stations["stable_low_flow_target"]]["q_site"]
    stable_fold = fold_biases[
        fold_biases["fold_eligible"] & fold_biases["q_site"].isin(stable_sites)
    ]
    stable_fold_summary = (
        stable_fold.groupby("fold_id")
        .agg(
            stable_stations=("q_site", "nunique"),
            median_low_flow_bias_pct=("low_flow_volume_bias_pct", "median"),
            positive_station_fraction=(
                "low_flow_volume_bias_pct",
                lambda values: float((values > 0).mean()),
            ),
            material_positive_station_fraction=(
                "low_flow_volume_bias_pct",
                lambda values: float((values >= MATERIAL_BIAS).mean()),
            ),
        )
        .reset_index()
    )

    class_summary = (
        stations.groupby("heterogeneity_class", dropna=False)
        .agg(
            stations=("q_site", "count"),
            median_low_flow_bias_pct=("median_low_flow_bias_pct", "median"),
            median_positive_fold_fraction=("positive_fold_fraction", "median"),
            median_eligible_folds=("eligible_folds", "median"),
        )
        .reset_index()
    )
    class_summary = (
        class_summary.set_index("heterogeneity_class")
        .reindex(HETEROGENEITY_CLASSES)
        .reset_index()
    )
    class_summary["stations"] = class_summary["stations"].fillna(0).astype(int)
    target_eligible = stations[stations["eligible_for_separability"]]
    stable_count = int(stations["stable_low_flow_target"].sum())
    stable_group_median = float(
        stations.loc[
            stations["stable_low_flow_target"],
            "median_low_flow_bias_pct",
        ].median()
    ) if stable_count else np.nan
    positive_pooled_folds = int(
        stable_fold_summary["median_low_flow_bias_pct"].gt(0).sum()
    )
    stable_signal_gates = {
        "minimum_stable_stations_10": stable_count >= 10,
        "minimum_positive_pooled_folds_2": positive_pooled_folds >= 2,
        "stable_group_median_bias_at_least_15_pct": (
            stable_group_median >= 15.0
        ) if np.isfinite(stable_group_median) else False,
    }
    stable_signal_supported = bool(all(stable_signal_gates.values()))

    static_row = model_results[
        model_results["model_id"].eq("deployable_static")
    ].iloc[0]
    static_gates = {
        "positive_class_at_least_10": int(static_row["positive_stations"]) >= 10,
        "negative_class_at_least_10": int(static_row["negative_stations"]) >= 10,
        "roc_auc_at_least_0_70": bool(
            np.isfinite(static_row["roc_auc"])
            and float(static_row["roc_auc"]) >= 0.70
        ),
        "balanced_accuracy_at_least_0_65": bool(
            np.isfinite(static_row["balanced_accuracy_at_0_5"])
            and float(static_row["balanced_accuracy_at_0_5"]) >= 0.65
        ),
        "permutation_pvalue_at_most_0_05": bool(
            np.isfinite(static_row["permutation_pvalue_auc"])
            and float(static_row["permutation_pvalue_auc"]) <= 0.05
        ),
    }
    static_gate_supported = bool(all(static_gates.values()))
    signature_row = model_results[
        model_results["model_id"].eq(
            "static_plus_gauged_signature_upper_bound"
        )
    ].iloc[0]
    signature_upper_bound_supported = bool(
        np.isfinite(signature_row["roc_auc"])
        and float(signature_row["roc_auc"]) >= 0.70
        and np.isfinite(signature_row["balanced_accuracy_at_0_5"])
        and float(signature_row["balanced_accuracy_at_0_5"]) >= 0.65
        and np.isfinite(signature_row["permutation_pvalue_auc"])
        and float(signature_row["permutation_pvalue_auc"]) <= 0.05
    )

    scientific_gate = {
        "run_id": RUN.name,
        "phase_id": "low_flow_heterogeneity_and_attribute_separability",
        "logical_parent_run": REFERENCE.name,
        "diagnostic_predecessor": PREDECESSOR.name,
        "diagnostic_only": True,
        "stable_signal": {
            "stable_stations": stable_count,
            "eligible_nonreservoir_nonquality_stations": int(len(target_eligible)),
            "stable_group_median_low_flow_bias_pct": stable_group_median,
            "positive_pooled_folds": positive_pooled_folds,
            "gates": stable_signal_gates,
            "supported": stable_signal_supported,
        },
        "deployable_static_attribute_gate": {
            "metrics": static_row.to_dict(),
            "gates": static_gates,
            "supported": static_gate_supported,
        },
        "gauged_signature_upper_bound": {
            "metrics": signature_row.to_dict(),
            "supported": signature_upper_bound_supported,
            "deployable_to_ungauged_reaches": False,
        },
        "location_sensitivity_can_promote": False,
        "decision": (
            "stable_station_target_and_static_attribute_gate_supported"
            if stable_signal_supported and static_gate_supported
            else "stable_station_target_supported_static_attribute_gate_not_supported"
            if stable_signal_supported
            else "stable_low_flow_target_not_supported"
        ),
    }

    fold_biases.to_csv(
        OUT / "station_fold_low_flow_bias.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stations.to_csv(
        OUT / "station_heterogeneity_classification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_summary.to_csv(
        OUT / "heterogeneity_class_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stable_fold_summary.to_csv(
        OUT / "stable_target_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_results.to_csv(
        OUT / "separability_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_class_summary(stations).to_csv(
        OUT / "feature_class_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if prediction_parts:
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            OUT / "separability_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.concat(coefficient_parts, ignore_index=True).to_csv(
            OUT / "separability_full_fit_coefficients.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.concat(permutation_parts, ignore_index=True).to_csv(
            OUT / "separability_permutation_null.csv",
            index=False,
            encoding="utf-8-sig",
        )
    (OUT / "scientific_gate.json").write_text(
        json.dumps(scientific_gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "run_id": RUN.name,
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "oof_rows": int(len(oof)),
        "stations": int(len(stations)),
        "stable_target_stations": stable_count,
        "reservoir_related_stations": int(stations["reservoir_related_bool"].sum()),
        "data_quality_suspicious_stations": int(
            stations["data_quality_suspicious"].sum()
        ),
        "eligible_for_separability_stations": int(
            stations["eligible_for_separability"].sum()
        ),
        "protected_station_present": bool(stations["q_site"].eq(PROTECTED).any()),
        "active_exclusion_rows": int(stations["q_site"].isin(EXCLUSIONS).sum()),
        "maximum_oof_year": int(oof["year"].max()),
        "maximum_signature_year": int(signatures["signature_year_max"].max()),
        "maximum_climate_attribute_year": int(attributes["climate_year_max"].max()),
        "stable_signal_supported": stable_signal_supported,
        "deployable_static_attribute_gate_supported": static_gate_supported,
        "gauged_signature_upper_bound_supported": signature_upper_bound_supported,
        "decision": scientific_gate["decision"],
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
