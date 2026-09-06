from __future__ import annotations

import importlib.util
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import roc_auc_score


RUN = Path(__file__).resolve().parents[1]
SERIES = RUN.parent
BASELINE = SERIES / "20260727_6"
OUT = RUN / "reports" / "nested_oof_gate"
EPS = 1.0e-6
MIN_LOW_MONTHS = 3
MIN_HISTORY_BLOCKS = 2
MATERIAL_BIAS = 10.0
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站"}
PROTECTED = "石角站"

NEW_BLOCKS = [
    {
        "block_id": "fit_2006_2007_eval_2008_2009",
        "train_end": 2007,
        "eval_start": 2008,
        "eval_end": 2009,
    },
    {
        "block_id": "fit_2006_2009_eval_2010_2011",
        "train_end": 2009,
        "eval_start": 2010,
        "eval_end": 2011,
    },
]
REUSED_BLOCKS = [
    {
        "block_id": "fit_2006_2011_eval_2012_2013",
        "source_fold_id": "fit_2006_2011_eval_2012_2013",
        "train_end": 2011,
        "eval_start": 2012,
        "eval_end": 2013,
    },
    {
        "block_id": "fit_2006_2013_eval_2014_2015",
        "source_fold_id": "fit_2006_2013_eval_2014_2015",
        "train_end": 2013,
        "eval_start": 2014,
        "eval_end": 2015,
    },
]
OUTER_FOLDS = [
    {
        "outer_fold": "fit_through_2011_eval_2012_2013",
        "source_fold_id": "fit_2006_2011_eval_2012_2013",
        "eval_start": 2012,
        "eval_end": 2013,
        "allowed_blocks": [
            "fit_2006_2007_eval_2008_2009",
            "fit_2006_2009_eval_2010_2011",
        ],
    },
    {
        "outer_fold": "fit_through_2013_eval_2014_2015",
        "source_fold_id": "fit_2006_2013_eval_2014_2015",
        "eval_start": 2014,
        "eval_end": 2015,
        "allowed_blocks": [
            "fit_2006_2007_eval_2008_2009",
            "fit_2006_2009_eval_2010_2011",
            "fit_2006_2011_eval_2012_2013",
        ],
    },
    {
        "outer_fold": "fit_through_2015_eval_2016_2018",
        "source_fold_id": "fit_2006_2015_eval_2016_2018",
        "eval_start": 2016,
        "eval_end": 2018,
        "allowed_blocks": [
            "fit_2006_2007_eval_2008_2009",
            "fit_2006_2009_eval_2010_2011",
            "fit_2006_2011_eval_2012_2013",
            "fit_2006_2013_eval_2014_2015",
        ],
    },
]
FIXED_HYPERPARAMETERS = {
    "rho": 0.70,
    "wm": 480.0,
    "et_gamma": 0.75,
    "sas_rho": 0.93,
    "young_k": 1.5,
    "storage_scale": 720.0,
    "prod_capacity": 240.0,
    "runoff_gamma": 2.5,
    "quick_rho": 0.25,
    "base_rho": 0.85,
    "base_release": 0.10,
    "fixed_sigma": 3.0,
    "production_sigma": 1.5,
    "group_sigma": 1.5,
    "multistore_sigma": 0.30,
    "hysteresis_sigma": 3.0,
    "station_sigma": 1.0,
    "slope_sigma": 0.15,
    "regime_slope_sigma": 0.25,
    "anomaly_weight": 0.0,
    "flow_contrast_weight": 1.0,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_q72_block(block: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    started = time.monotonic()
    component = load_module(
        f"nested_q72_{block['train_end']}",
        BASELINE
        / "scripts"
        / "components"
        / "fit_monthly_bayes_seasonal_hysteresis.py",
    )
    with tempfile.TemporaryDirectory(
        prefix=f"{block['block_id']}_",
        dir=str(RUN / "logs"),
    ) as temporary:
        root = Path(temporary)
        component.REPORT_DIR = root / "reports"
        component.FIG_DIR = root / "figures"
        component.CAL_END_YEAR = int(block["train_end"])
        component.INNER_TRAIN_END_YEAR = int(block["train_end"])
        component.write_readme = lambda *_args, **_kwargs: None

        def fixed_choice(_frame):
            row = {
                **FIXED_HYPERPARAMETERS,
                "selection": "fixed_from_20260727_6",
            }
            return row, pd.DataFrame([row])

        component.choose_hyperparameters = fixed_choice
        component.main()
        prediction_path = (
            component.REPORT_DIR
            / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv"
        )
        full = pd.read_csv(
            prediction_path,
            encoding="utf-8-sig",
            usecols=["q_site", "comid", "year", "month", "actual", "predict"],
        ).rename(
            columns={
                "comid": "reach_id",
                "actual": "Q_obsv_cfs",
                "predict": "Q72_pred_cfs",
            }
        )
    full["reach_id"] = pd.to_numeric(full["reach_id"], errors="raise").astype(int)
    full["year"] = full["year"].astype(int)
    full["month"] = full["month"].astype(int)
    training = full[
        (full["year"] <= int(block["train_end"]))
        & full["Q_obsv_cfs"].notna()
        & full["Q_obsv_cfs"].gt(0)
    ].copy()
    evaluation = full[
        full["year"].between(
            int(block["eval_start"]),
            int(block["eval_end"]),
        )
    ].copy()
    return training, evaluation, time.monotonic() - started


def fit_q78_coefficients(
    training: pd.DataFrame,
    routed_basis: pd.DataFrame,
    feature_names: list[str],
    train_end: int,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, object]]:
    columns = [f"routed_{name}_cfs" for name in feature_names]
    joined = training.merge(
        routed_basis,
        on=["reach_id", "year", "month"],
        how="left",
        validate="many_to_one",
    ).dropna(subset=columns + ["Q_obsv_cfs"])
    joined = joined[
        (joined["year"] <= train_end) & joined["Q_obsv_cfs"].gt(0)
    ]
    x = joined[columns].to_numpy(dtype=float)
    y = joined["Q_obsv_cfs"].to_numpy(dtype=float)
    scales = np.nanmedian(np.where(x > 0, x, np.nan), axis=0)
    global_scale = np.nanmedian(x[x > 0]) if np.any(x > 0) else 1.0
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, global_scale)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    x_scaled = x / scales
    y_log = np.log(y + EPS)

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = x_scaled @ theta
        return np.concatenate(
            [
                np.log(prediction + EPS) - y_log,
                0.05 * theta,
            ]
        )

    result = least_squares(
        residual,
        np.full(len(feature_names), 0.02),
        bounds=(0.0, np.inf),
        max_nfev=5000,
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    coefficients = result.x / scales
    mapping = {
        name: float(value) for name, value in zip(feature_names, coefficients)
    }
    coefficient_frame = pd.DataFrame(
        {
            "feature_name": feature_names,
            "coefficient": coefficients,
            "scaled_coefficient": result.x,
            "scale_cfs": scales,
        }
    )
    info = {
        "q78_training_rows": int(len(joined)),
        "q78_training_stations": int(joined["q_site"].nunique()),
        "q78_training_year_min": int(joined["year"].min()),
        "q78_training_year_max": int(joined["year"].max()),
        "q78_optimizer_success": bool(result.success),
        "q78_optimizer_cost": float(result.cost),
        "q78_optimizer_nfev": int(result.nfev),
        "q78_min_coefficient": float(np.min(coefficients)),
        "q78_max_coefficient": float(np.max(coefficients)),
    }
    return mapping, coefficient_frame, info


def add_regime_and_fusion(
    block: dict[str, object],
    training: pd.DataFrame,
    q72_eval: pd.DataFrame,
    q78_eval: pd.DataFrame,
    alpha_map: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = (
        training.groupby("q_site")["Q_obsv_cfs"]
        .agg(
            training_months="count",
            training_q25_cfs=lambda values: float(values.quantile(0.25)),
        )
        .reset_index()
    )
    thresholds["block_id"] = str(block["block_id"])
    joined = q72_eval.merge(
        q78_eval,
        on=["reach_id", "year", "month"],
        how="left",
        validate="many_to_one",
    ).merge(
        thresholds[["q_site", "training_q25_cfs"]],
        on="q_site",
        how="left",
        validate="many_to_one",
    )
    if joined[
        ["Q78_mass_cfs", "reach_class"]
    ].isna().any().any():
        raise RuntimeError(f"Incomplete early nested block {block['block_id']}")
    joined["class_alpha"] = joined["reach_class"].map(alpha_map)
    if joined["class_alpha"].isna().any():
        raise RuntimeError(f"Missing alpha in early nested block {block['block_id']}")
    joined["Q0_pred_cfs"] = (
        (1.0 - joined["class_alpha"]) * joined["Q72_pred_cfs"]
        + joined["class_alpha"] * joined["Q78_mass_cfs"]
    )
    joined["flow_regime"] = np.select(
        [
            joined["training_q25_cfs"].isna(),
            joined["Q_obsv_cfs"] <= joined["training_q25_cfs"],
        ],
        ["unclassified_no_training_q25", "low"],
        default="nonlow",
    )
    joined["block_id"] = str(block["block_id"])
    joined["train_end"] = int(block["train_end"])
    joined["eval_start"] = int(block["eval_start"])
    joined["eval_end"] = int(block["eval_end"])
    return joined, thresholds


def station_block_bias(predictions: pd.DataFrame) -> pd.DataFrame:
    low = predictions[predictions["flow_regime"].eq("low")]
    rows = []
    for (block_id, site), part in low.groupby(["block_id", "q_site"], sort=True):
        obs = part["Q_obsv_cfs"].to_numpy(dtype=float)
        pred = part["Q0_pred_cfs"].to_numpy(dtype=float)
        rows.append(
            {
                "block_id": block_id,
                "q_site": site,
                "reach_id": int(part["reach_id"].iloc[0]),
                "low_months": int(len(part)),
                "low_flow_volume_bias_pct": float(
                    100.0 * np.sum(pred - obs) / np.sum(obs)
                ),
                "fold_eligible": bool(len(part) >= MIN_LOW_MONTHS),
            }
        )
    return pd.DataFrame(rows)


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40, 40))))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    contract = json.loads(
        (
            SERIES
            / "20260728_6"
            / "reports"
            / "comprehensive_gate"
            / "candidate_contract.json"
        ).read_text(encoding="utf-8")
    )
    if not contract.get("admitted"):
        raise RuntimeError("Candidate contract is not admitted")
    context = pd.read_csv(
        SERIES
        / "20260728_5"
        / "reports"
        / "low_flow_heterogeneity"
        / "station_heterogeneity_classification.csv",
        encoding="utf-8-sig",
    )
    context = context[
        [
            "q_site",
            "reservoir_related_bool",
            "data_quality_suspicious",
            "stable_low_flow_target",
        ]
    ].drop_duplicates("q_site")
    for column in [
        "reservoir_related_bool",
        "data_quality_suspicious",
        "stable_low_flow_target",
    ]:
        context[column] = (
            context[column].astype(str).str.lower().eq("true")
        )

    mass_module = load_module(
        "nested_reach_class_mass",
        BASELINE / "scripts" / "fit_reach_class_mass_model.py",
    )
    topology = mass_module.BASE.load_topology()
    order, issues = mass_module.BASE.topological_order(topology)
    if issues:
        raise RuntimeError(";".join(issues))
    forcing = pd.read_csv(
        BASELINE
        / "reports"
        / "intermediate"
        / "mass_reach_class"
        / "reach_month_forcing_panel.csv",
        encoding="utf-8-sig",
    )
    routed_basis = pd.read_csv(
        BASELINE
        / "reports"
        / "intermediate"
        / "mass_reach_class"
        / "routed_class_local_source_basis.csv",
        encoding="utf-8-sig",
    )
    feature_names = [
        column.removeprefix("routed_").removesuffix("_cfs")
        for column in routed_basis.columns
        if column.startswith("routed_") and column.endswith("_cfs")
    ]
    alpha_table = pd.read_csv(
        BASELINE / "reports" / "main_model" / "selected_reach_class_alpha.csv",
        encoding="utf-8-sig",
    )
    alpha_map = alpha_table.set_index("reach_class")["class_alpha"].to_dict()

    prediction_parts = []
    threshold_parts = []
    coefficient_parts = []
    manifest_rows = []
    mass_rows = []
    for block in NEW_BLOCKS:
        training, q72_eval, q72_seconds = run_q72_block(block)
        coefficients, coefficient_frame, fit_info = fit_q78_coefficients(
            training,
            routed_basis,
            feature_names,
            int(block["train_end"]),
        )
        coefficient_frame.insert(0, "block_id", str(block["block_id"]))
        coefficient_parts.append(coefficient_frame)
        route_panel = forcing[forcing["year"] <= int(block["eval_end"])].copy()
        routed = mass_module.route_with_class_coefficients(
            route_panel,
            topology,
            order,
            coefficients,
        )
        q78_eval = routed[
            routed["year"].between(
                int(block["eval_start"]),
                int(block["eval_end"]),
            )
        ][
            [
                "reach_id",
                "year",
                "month",
                "Q_out_cfs",
                "reach_class",
            ]
        ].rename(columns={"Q_out_cfs": "Q78_mass_cfs"})
        joined, thresholds = add_regime_and_fusion(
            block,
            training,
            q72_eval,
            q78_eval,
            alpha_map,
        )
        prediction_parts.append(joined)
        threshold_parts.append(thresholds)
        max_residual = float(routed["mass_balance_residual_cfs"].abs().max())
        mass_rows.append(
            {
                "block_id": block["block_id"],
                "routed_year_max": int(routed["year"].max()),
                "routed_rows": int(len(routed)),
                "max_abs_mass_balance_residual_cfs": max_residual,
            }
        )
        manifest_rows.append(
            {
                **block,
                "source": "new_strict_block",
                "q72_training_rows": int(len(training)),
                "q72_training_stations": int(training["q_site"].nunique()),
                "q72_evaluation_rows": int(len(q72_eval)),
                "q72_evaluation_stations": int(q72_eval["q_site"].nunique()),
                "q72_elapsed_seconds": round(q72_seconds, 3),
                "joined_evaluation_rows": int(len(joined)),
                "joined_evaluation_stations": int(joined["q_site"].nunique()),
                "training_q25_missing_eval_rows": int(
                    joined["training_q25_cfs"].isna().sum()
                ),
                "training_q25_missing_eval_stations": int(
                    joined.loc[
                        joined["training_q25_cfs"].isna(),
                        "q_site",
                    ].nunique()
                ),
                "maximum_evidence_year": int(joined["year"].max()),
                **fit_info,
            }
        )
        print(
            f"{block['block_id']}: q72={q72_seconds:.1f}s "
            f"eval_rows={len(joined)} max_mass_residual={max_residual:.3e}",
            flush=True,
        )

    existing = pd.read_csv(
        SERIES
        / "20260728_4"
        / "reports"
        / "q72_q78_complementarity"
        / "oof_predictions_2012_2018.csv",
        encoding="utf-8-sig",
    )
    for block in REUSED_BLOCKS:
        part = existing[
            existing["fold_id"].eq(block["source_fold_id"])
        ].copy()
        part["block_id"] = str(block["block_id"])
        part["Q0_pred_cfs"] = part["Q_original_fusion_cfs"]
        part["flow_regime"] = np.where(
            part["flow_regime"].eq("low"),
            "low",
            "nonlow",
        )
        prediction_parts.append(part)
        manifest_rows.append(
            {
                **block,
                "source": "reused_20260728_4_strict_oof",
                "q72_training_rows": "",
                "q72_training_stations": "",
                "q72_evaluation_rows": int(len(part)),
                "q72_evaluation_stations": int(part["q_site"].nunique()),
                "q72_elapsed_seconds": 0.0,
                "joined_evaluation_rows": int(len(part)),
                "joined_evaluation_stations": int(part["q_site"].nunique()),
                "training_q25_missing_eval_rows": int(
                    part["training_q25_cfs"].isna().sum()
                ),
                "training_q25_missing_eval_stations": int(
                    part.loc[
                        part["training_q25_cfs"].isna(),
                        "q_site",
                    ].nunique()
                ),
                "maximum_evidence_year": int(part["year"].max()),
                "q78_training_rows": "",
                "q78_training_stations": "",
                "q78_training_year_min": 2006,
                "q78_training_year_max": int(block["train_end"]),
                "q78_optimizer_success": True,
                "q78_optimizer_cost": "",
                "q78_optimizer_nfev": "",
                "q78_min_coefficient": "",
                "q78_max_coefficient": "",
            }
        )

    nested_predictions = pd.concat(prediction_parts, ignore_index=True)
    nested_predictions = nested_predictions[
        [
            "block_id",
            "train_end",
            "eval_start",
            "eval_end",
            "q_site",
            "reach_id",
            "year",
            "month",
            "Q_obsv_cfs",
            "Q72_pred_cfs",
            "Q78_mass_cfs",
            "class_alpha",
            "Q0_pred_cfs",
            "training_q25_cfs",
            "flow_regime",
        ]
    ]
    block_bias = station_block_bias(nested_predictions)

    gate_rows = []
    fold_rows = []
    for outer in OUTER_FOLDS:
        outer_data = existing[
            existing["fold_id"].eq(outer["source_fold_id"])
        ].copy()
        outer_low = outer_data[outer_data["flow_regime"].eq("low")]
        outer_bias_rows = []
        for site, part in outer_low.groupby("q_site", sort=True):
            obs = part["Q_obsv_cfs"].to_numpy(dtype=float)
            pred = part["Q_original_fusion_cfs"].to_numpy(dtype=float)
            outer_bias_rows.append(
                {
                    "q_site": site,
                    "outer_low_months": int(len(part)),
                    "outer_low_flow_bias_pct": float(
                        100.0 * np.sum(pred - obs) / np.sum(obs)
                    ),
                    "outer_bias_eligible": bool(len(part) >= MIN_LOW_MONTHS),
                }
            )
        outer_bias = pd.DataFrame(outer_bias_rows)
        universe = (
            outer_data[["q_site", "reach_id"]]
            .drop_duplicates("q_site")
            .merge(context, on="q_site", how="left", validate="one_to_one")
            .merge(outer_bias, on="q_site", how="left", validate="one_to_one")
        )
        for column in [
            "reservoir_related_bool",
            "data_quality_suspicious",
            "stable_low_flow_target",
        ]:
            universe[column] = universe[column].fillna(False).astype(bool)

        for row in universe.itertuples(index=False):
            history = block_bias[
                block_bias["block_id"].isin(outer["allowed_blocks"])
                & block_bias["q_site"].eq(row.q_site)
                & block_bias["fold_eligible"]
            ]
            values = history["low_flow_volume_bias_pct"].to_numpy(dtype=float)
            eligible_blocks = int(len(values))
            median_bias = float(np.median(values)) if len(values) else np.nan
            positive_fraction = float(np.mean(values > 0)) if len(values) else np.nan
            material_negative_fraction = (
                float(np.mean(values <= -MATERIAL_BIAS)) if len(values) else np.nan
            )
            history_sufficient = eligible_blocks >= MIN_HISTORY_BLOCKS
            stable_history = bool(
                history_sufficient
                and positive_fraction >= 2.0 / 3.0
                and median_bias >= MATERIAL_BIAS
                and material_negative_fraction == 0
            )
            hard_zero_reason = ""
            if bool(row.reservoir_related_bool):
                hard_zero_reason = "reservoir_related"
            elif bool(row.data_quality_suspicious):
                hard_zero_reason = "data_quality_suspicious"
            elif not history_sufficient:
                hard_zero_reason = "insufficient_inner_blocks"
            elif not stable_history:
                hard_zero_reason = "history_not_stable_overprediction"
            gate_value = (
                sigmoid((median_bias - MATERIAL_BIAS) / 10.0)
                if stable_history and hard_zero_reason == ""
                else 0.0
            )
            gate_rows.append(
                {
                    "outer_fold": outer["outer_fold"],
                    "eval_start": outer["eval_start"],
                    "eval_end": outer["eval_end"],
                    "q_site": row.q_site,
                    "reach_id": int(row.reach_id),
                    "allowed_inner_blocks": "|".join(outer["allowed_blocks"]),
                    "eligible_inner_blocks": eligible_blocks,
                    "history_median_low_flow_bias_pct": median_bias,
                    "history_positive_fraction": positive_fraction,
                    "history_material_negative_fraction": material_negative_fraction,
                    "history_stable_overprediction": stable_history,
                    "reservoir_related": bool(row.reservoir_related_bool),
                    "data_quality_suspicious": bool(row.data_quality_suspicious),
                    "hard_zero_reason": hard_zero_reason,
                    "station_gate": gate_value,
                    "outer_low_months": row.outer_low_months,
                    "outer_low_flow_bias_pct": row.outer_low_flow_bias_pct,
                    "outer_bias_eligible": bool(row.outer_bias_eligible)
                    if not pd.isna(row.outer_bias_eligible)
                    else False,
                    "outer_material_overprediction": bool(
                        not pd.isna(row.outer_low_flow_bias_pct)
                        and bool(row.outer_bias_eligible)
                        and float(row.outer_low_flow_bias_pct) >= MATERIAL_BIAS
                    ),
                    "full_series_20260728_5_target_comparison_only": bool(
                        row.stable_low_flow_target
                    ),
                }
            )

        fold_gate = pd.DataFrame(
            [entry for entry in gate_rows if entry["outer_fold"] == outer["outer_fold"]]
        )
        clean_eval = fold_gate[
            ~fold_gate["reservoir_related"]
            & ~fold_gate["data_quality_suspicious"]
            & fold_gate["outer_bias_eligible"]
        ]
        target = clean_eval[clean_eval["station_gate"] > 0]
        nontarget = clean_eval[clean_eval["station_gate"] == 0]
        target_median = float(target["outer_low_flow_bias_pct"].median()) if len(target) else np.nan
        nontarget_median = float(nontarget["outer_low_flow_bias_pct"].median()) if len(nontarget) else np.nan
        contrast = (
            target_median - nontarget_median
            if np.isfinite(target_median) and np.isfinite(nontarget_median)
            else np.nan
        )
        y = clean_eval["outer_material_overprediction"].astype(int)
        auc = (
            float(roc_auc_score(y, clean_eval["station_gate"]))
            if len(clean_eval) and y.nunique() == 2
            else np.nan
        )
        target_positive_fraction = (
            float((target["outer_low_flow_bias_pct"] > 0).mean())
            if len(target)
            else np.nan
        )
        criteria = {
            "identified_targets_at_least_10": len(target) >= 10,
            "target_median_bias_at_least_10pct": bool(
                np.isfinite(target_median) and target_median >= 10.0
            ),
            "target_positive_fraction_at_least_0_60": bool(
                np.isfinite(target_positive_fraction)
                and target_positive_fraction >= 0.60
            ),
            "gate_auc_at_least_0_60": bool(np.isfinite(auc) and auc >= 0.60),
            "target_minus_nontarget_contrast_at_least_5pct_points": bool(
                np.isfinite(contrast) and contrast >= 5.0
            ),
        }
        full_target_set = set(
            fold_gate.loc[
                fold_gate["full_series_20260728_5_target_comparison_only"],
                "q_site",
            ]
        )
        predicted_set = set(target["q_site"])
        union = full_target_set | predicted_set
        fold_rows.append(
            {
                "outer_fold": outer["outer_fold"],
                "eval_start": outer["eval_start"],
                "eval_end": outer["eval_end"],
                "outer_stations": int(fold_gate["q_site"].nunique()),
                "clean_outer_bias_eligible_stations": int(len(clean_eval)),
                "identified_target_stations": int(len(target)),
                "clean_nontarget_stations": int(len(nontarget)),
                "target_median_low_flow_bias_pct": target_median,
                "nontarget_median_low_flow_bias_pct": nontarget_median,
                "target_minus_nontarget_median_bias_pct_points": contrast,
                "target_positive_station_fraction": target_positive_fraction,
                "gate_auc_material_overprediction": auc,
                "full_series_label_jaccard_comparison_only": (
                    float(len(full_target_set & predicted_set) / len(union))
                    if union
                    else np.nan
                ),
                "protected_station_gate": float(
                    fold_gate.loc[
                        fold_gate["q_site"].eq(PROTECTED),
                        "station_gate",
                    ].iloc[0]
                )
                if fold_gate["q_site"].eq(PROTECTED).any()
                else np.nan,
                **criteria,
                "outer_fold_scientific_gate_passed": bool(all(criteria.values())),
            }
        )

    gate_assignments = pd.DataFrame(gate_rows)
    fold_summary = pd.DataFrame(fold_rows)
    passing_folds = int(fold_summary["outer_fold_scientific_gate_passed"].sum())
    scientific = {
        "run_id": RUN.name,
        "phase_id": "nested_oof_station_gate_audit",
        "contract_id": contract["contract_id"],
        "outer_folds": len(OUTER_FOLDS),
        "passing_outer_folds": passing_folds,
        "minimum_passing_outer_folds": 2,
        "nested_station_gate_supported": passing_folds >= 2,
        "decision": (
            "permit_minimal_monotone_residual_pilot"
            if passing_folds >= 2
            else "evidence_stop_do_not_create_residual_pilot"
        ),
        "full_series_label_used_as_predictor": False,
        "residual_correction_applied": False,
        "folds": fold_summary.to_dict(orient="records"),
    }

    nested_predictions.to_csv(
        OUT / "nested_block_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    block_bias.to_csv(
        OUT / "station_inner_block_low_flow_bias.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(manifest_rows).to_csv(
        OUT / "nested_block_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if threshold_parts:
        pd.concat(threshold_parts, ignore_index=True).to_csv(
            OUT / "new_block_training_q25.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if coefficient_parts:
        pd.concat(coefficient_parts, ignore_index=True).to_csv(
            OUT / "new_block_q78_coefficients.csv",
            index=False,
            encoding="utf-8-sig",
        )
    pd.DataFrame(mass_rows).to_csv(
        OUT / "new_block_mass_balance_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate_assignments.to_csv(
        OUT / "outer_fold_station_gate_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_summary.to_csv(
        OUT / "outer_fold_gate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "scientific_gate.json").write_text(
        json.dumps(scientific, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "run_id": RUN.name,
        "new_nested_blocks": len(NEW_BLOCKS),
        "reused_nested_blocks": len(REUSED_BLOCKS),
        "nested_prediction_rows": int(len(nested_predictions)),
        "nested_prediction_year_min": int(nested_predictions["year"].min()),
        "nested_prediction_year_max": int(nested_predictions["year"].max()),
        "unclassified_no_training_q25_rows": int(
            nested_predictions["flow_regime"]
            .eq("unclassified_no_training_q25")
            .sum()
        ),
        "outer_gate_assignment_rows": int(len(gate_assignments)),
        "outer_folds": len(OUTER_FOLDS),
        "passing_outer_folds": passing_folds,
        "nested_station_gate_supported": passing_folds >= 2,
        "protected_station_present_all_outer_folds": bool(
            gate_assignments[
                gate_assignments["q_site"].eq(PROTECTED)
            ]["outer_fold"].nunique()
            == len(OUTER_FOLDS)
        ),
        "excluded_station_rows": int(
            gate_assignments["q_site"].isin(EXCLUSIONS).sum()
        ),
        "maximum_mass_balance_residual_cfs": float(
            pd.DataFrame(mass_rows)["max_abs_mass_balance_residual_cfs"].max()
        ),
        "residual_correction_applied": False,
        "decision": scientific["decision"],
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
