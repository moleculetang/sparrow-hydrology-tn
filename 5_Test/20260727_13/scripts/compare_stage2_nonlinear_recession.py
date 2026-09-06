from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_settings import (
    ABS_PBIAS_EFFECT_SIZE_PCT_POINTS,
    KGE_EFFECT_SIZE,
    LINEAR_RECESSION_REFERENCE_EXPONENT,
    LOW_FLOW_BIAS_EFFECT_SIZE_PCT_POINTS,
    LOW_FLOW_LOG_RMSE_EFFECT_SIZE,
    NONLINEAR_RECESSION_EXPONENT,
    NSELOG_EFFECT_SIZE,
    REFERENCE_RUN_ID,
)


RUN = Path(__file__).resolve().parents[1]
REFERENCE = RUN.parent / REFERENCE_RUN_ID
OUT = RUN / "reports" / "stage2_nonlinear_recession_diagnostics"
TOL = 1.0e-9
PERIODS = ("selection_2016_2018", "development_holdout_2019_2022")
POPULATIONS = ("all_stations", "nonreservoir_target")
BRANCHES = ("Q72", "Q78_mass", "main_fusion")
FOLDS = (
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
)
EPS = 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = [
            f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare_frame(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    keys: list[str],
    numeric_columns: list[str] | None = None,
) -> dict[str, object]:
    current = current.sort_values(keys).reset_index(drop=True)
    reference = reference.sort_values(keys).reset_index(drop=True)
    result: dict[str, object] = {
        "current_rows": int(len(current)),
        "reference_rows": int(len(reference)),
        "row_count_match": bool(len(current) == len(reference)),
        "key_set_match": False,
        "max_abs_numeric_difference": np.nan,
        "numeric_columns_compared": 0,
    }
    if len(current) != len(reference):
        return result
    key_match = current[keys].astype(str).equals(reference[keys].astype(str))
    result["key_set_match"] = bool(key_match)
    if not key_match:
        return result
    if numeric_columns is None:
        numeric_columns = sorted(
            set(current.select_dtypes(include=[np.number]).columns)
            & set(reference.select_dtypes(include=[np.number]).columns)
        )
    numeric_columns = [
        column
        for column in numeric_columns
        if column in current.columns and column in reference.columns
    ]
    differences: list[float] = []
    for column in numeric_columns:
        left = pd.to_numeric(current[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(left) & np.isnan(right)
        delta = np.abs(left - right)
        delta[both_nan] = 0.0
        differences.append(
            np.inf
            if np.isnan(delta).any()
            else (float(np.max(delta)) if len(delta) else 0.0)
        )
    result["numeric_columns_compared"] = int(len(numeric_columns))
    result["max_abs_numeric_difference"] = (
        float(max(differences)) if differences else 0.0
    )
    return result


def station_policy_gate() -> dict[str, object]:
    policy = pd.read_csv(
        RUN / "inputs" / "source_metadata" / "station_screening_policy.csv",
        encoding="utf-8-sig",
    )
    excluded = set(
        policy.loc[
            policy["exclude_before_training"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"}),
            "station_name",
        ].astype(str)
    )
    expected = {"劳村站", "富罗（二）站", "隆安站"}
    stone = policy.loc[policy["station_name"].eq("石角站")]
    return {
        "excluded_exact_match": bool(excluded == expected),
        "excluded_stations": sorted(excluded),
        "stone_present": bool(len(stone) == 1),
        "stone_not_excluded": bool(
            len(stone) == 1
            and str(stone.iloc[0]["exclude_before_training"]).lower()
            not in {"true", "1", "yes"}
        ),
    }


def metric_summary(frame: pd.DataFrame) -> dict[str, object]:
    good = (
        frame["n_months"].ge(24)
        & frame["NSElog"].ge(0.65)
        & frame["KGE"].ge(0.50)
        & frame["abs_PBIAS_pct"].le(25.0)
    )
    return {
        "stations": int(len(frame)),
        "median_NSEraw": float(frame["NSE_raw"].median()),
        "median_NSElog": float(frame["NSElog"].median()),
        "median_KGE": float(frame["KGE"].median()),
        "median_abs_PBIAS": float(frame["abs_PBIAS_pct"].median()),
        "median_high_flow_nrmse": float(
            frame["high_flow_normalized_RMSE_Q75"].median()
        ),
        "median_abs_low_flow_bias": float(
            frame["low_flow_volume_bias_pct_Q25"].abs().median()
        ),
        "median_low_flow_log_rmse": float(
            frame["low_flow_RMSElog_Q25"].median()
        ),
        "good": int(good.sum()),
        "abs_PBIAS_gt50": int(frame["abs_PBIAS_pct"].gt(50).sum()),
        "abs_PBIAS_gt100": int(frame["abs_PBIAS_pct"].gt(100).sum()),
    }


def raw_station_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for site, part in frame.groupby("q_site", sort=True):
        obs = pd.to_numeric(part["actual"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(part["predict"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
        obs = obs[mask]
        pred = pred[mask]
        if len(obs) < 3:
            continue
        lo = np.log(obs + EPS)
        lp = np.log(pred + EPS)
        denom = float(np.sum((lo - np.mean(lo)) ** 2))
        nse_log = (
            float(1.0 - np.sum((lp - lo) ** 2) / denom)
            if denom > 0
            else np.nan
        )
        if np.std(obs) <= 0 or np.std(pred) <= 0:
            kge = np.nan
        else:
            corr = float(np.corrcoef(obs, pred)[0, 1])
            beta = float(np.mean(pred) / np.mean(obs))
            cv_obs = float(np.std(obs) / np.mean(obs))
            cv_pred = float(np.std(pred) / np.mean(pred))
            gamma = cv_pred / cv_obs if cv_obs > 0 else np.nan
            kge = float(
                1.0
                - np.sqrt(
                    (corr - 1.0) ** 2
                    + (beta - 1.0) ** 2
                    + (gamma - 1.0) ** 2
                )
            )
        pbias = float(100.0 * np.sum(pred - obs) / np.sum(obs))
        raw_denom = float(np.sum((obs - np.mean(obs)) ** 2))
        nse_raw = (
            float(1.0 - np.sum((pred - obs) ** 2) / raw_denom)
            if raw_denom > 0
            else np.nan
        )
        high = obs >= np.quantile(obs, 0.75)
        low = obs <= np.quantile(obs, 0.25)
        high_flow_nrmse = float(
            np.sqrt(np.mean((pred[high] - obs[high]) ** 2))
            / (np.mean(obs) + EPS)
        )
        low_flow_bias = float(
            100.0 * np.sum(pred[low] - obs[low]) / np.sum(obs[low])
        )
        low_flow_log_rmse = float(
            np.sqrt(np.mean((lp[low] - lo[low]) ** 2))
        )
        rows.append(
            {
                "q_site": site,
                "n_months": int(len(obs)),
                "NSE_raw": nse_raw,
                "NSElog": nse_log,
                "KGE": kge,
                "abs_PBIAS_pct": abs(pbias),
                "high_flow_normalized_RMSE_Q75": high_flow_nrmse,
                "low_flow_volume_bias_pct_Q25": low_flow_bias,
                "low_flow_RMSElog_Q25": low_flow_log_rmse,
            }
        )
    return pd.DataFrame(rows)


def build_metric_comparison(
    nonreservoir: set[str],
) -> pd.DataFrame:
    current = pd.read_csv(
        OUT / "station_branch_period_metrics.csv", encoding="utf-8-sig"
    )
    reference = pd.read_csv(
        REFERENCE
        / "reports"
        / "stage0_diagnostics"
        / "station_branch_period_metrics.csv",
        encoding="utf-8-sig",
    )
    rows: list[dict[str, object]] = []
    for period in PERIODS:
        for population in POPULATIONS:
            allowed = None if population == "all_stations" else nonreservoir
            for branch in BRANCHES:
                cur = current[
                    current["period"].eq(period) & current["branch"].eq(branch)
                ].copy()
                ref = reference[
                    reference["period"].eq(period) & reference["branch"].eq(branch)
                ].copy()
                if allowed is not None:
                    cur = cur[cur["q_site"].isin(allowed)]
                    ref = ref[ref["q_site"].isin(allowed)]
                common = sorted(set(cur["q_site"]) & set(ref["q_site"]))
                cur = cur[cur["q_site"].isin(common)]
                ref = ref[ref["q_site"].isin(common)]
                cur_summary = metric_summary(cur)
                ref_summary = metric_summary(ref)
                rows.append(
                    {
                        "period": period,
                        "population": population,
                        "branch": branch,
                        "common_stations": len(common),
                        **{f"reference_{k}": v for k, v in ref_summary.items()},
                        **{f"candidate_{k}": v for k, v in cur_summary.items()},
                        "delta_NSEraw": cur_summary["median_NSEraw"]
                        - ref_summary["median_NSEraw"],
                        "delta_NSElog": cur_summary["median_NSElog"]
                        - ref_summary["median_NSElog"],
                        "delta_KGE": cur_summary["median_KGE"]
                        - ref_summary["median_KGE"],
                        "abs_PBIAS_reduction": ref_summary["median_abs_PBIAS"]
                        - cur_summary["median_abs_PBIAS"],
                        "high_flow_nrmse_reduction": ref_summary[
                            "median_high_flow_nrmse"
                        ]
                        - cur_summary["median_high_flow_nrmse"],
                        "abs_low_flow_bias_reduction": ref_summary[
                            "median_abs_low_flow_bias"
                        ]
                        - cur_summary["median_abs_low_flow_bias"],
                        "low_flow_log_rmse_reduction": ref_summary[
                            "median_low_flow_log_rmse"
                        ]
                        - cur_summary["median_low_flow_log_rmse"],
                        "delta_good": cur_summary["good"] - ref_summary["good"],
                        "delta_abs_PBIAS_gt50": cur_summary["abs_PBIAS_gt50"]
                        - ref_summary["abs_PBIAS_gt50"],
                        "delta_abs_PBIAS_gt100": cur_summary["abs_PBIAS_gt100"]
                        - ref_summary["abs_PBIAS_gt100"],
                    }
                )
    return pd.DataFrame(rows)


def build_low_flow_target_comparison(
    target_sites: set[str],
) -> pd.DataFrame:
    current = pd.read_csv(
        OUT / "station_branch_period_metrics.csv", encoding="utf-8-sig"
    )
    reference = pd.read_csv(
        REFERENCE
        / "reports"
        / "stage0_diagnostics"
        / "station_branch_period_metrics.csv",
        encoding="utf-8-sig",
    )
    rows: list[dict[str, object]] = []
    for period in PERIODS:
        for branch in ("Q72", "main_fusion"):
            cur = current[
                current["period"].eq(period)
                & current["branch"].eq(branch)
                & current["q_site"].isin(target_sites)
            ].copy()
            ref = reference[
                reference["period"].eq(period)
                & reference["branch"].eq(branch)
                & reference["q_site"].isin(target_sites)
            ].copy()
            common = sorted(set(cur["q_site"]) & set(ref["q_site"]))
            cur = cur[cur["q_site"].isin(common)]
            ref = ref[ref["q_site"].isin(common)]
            reference_low_bias = float(
                ref["low_flow_volume_bias_pct_Q25"].median()
            )
            candidate_low_bias = float(
                cur["low_flow_volume_bias_pct_Q25"].median()
            )
            reference_abs_low_bias = float(
                ref["low_flow_volume_bias_pct_Q25"].abs().median()
            )
            candidate_abs_low_bias = float(
                cur["low_flow_volume_bias_pct_Q25"].abs().median()
            )
            reference_low_rmse = float(
                ref["low_flow_RMSElog_Q25"].median()
            )
            candidate_low_rmse = float(
                cur["low_flow_RMSElog_Q25"].median()
            )
            rows.append(
                {
                    "period": period,
                    "branch": branch,
                    "frozen_failure_group": "low_flow_overprediction",
                    "common_stations": len(common),
                    "reference_stations": int(len(ref)),
                    "candidate_stations": int(len(cur)),
                    "reference_median_low_flow_bias_pct": (
                        reference_low_bias
                    ),
                    "candidate_median_low_flow_bias_pct": (
                        candidate_low_bias
                    ),
                    "reference_median_abs_low_flow_bias_pct": (
                        reference_abs_low_bias
                    ),
                    "candidate_median_abs_low_flow_bias_pct": (
                        candidate_abs_low_bias
                    ),
                    "abs_low_flow_bias_reduction_pct_points": (
                        reference_abs_low_bias
                        - candidate_abs_low_bias
                    ),
                    "reference_median_low_flow_log_rmse": (
                        reference_low_rmse
                    ),
                    "candidate_median_low_flow_log_rmse": (
                        candidate_low_rmse
                    ),
                    "low_flow_log_rmse_reduction": (
                        reference_low_rmse - candidate_low_rmse
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_blocked_fold_comparison(
    nonreservoir: set[str],
    low_flow_target: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in FOLDS:
        relative = (
            Path("reports")
            / "station_screening"
            / "blocked_folds"
            / fold
            / "evaluation_predictions.csv"
        )
        cur_raw = pd.read_csv(RUN / relative, encoding="utf-8-sig")
        ref_raw = pd.read_csv(REFERENCE / relative, encoding="utf-8-sig")
        cur_metrics = raw_station_metrics(cur_raw)
        ref_metrics = raw_station_metrics(ref_raw)
        for population, allowed in (
            ("all_stations", None),
            ("nonreservoir_target", nonreservoir),
            ("low_flow_target", low_flow_target),
        ):
            cur = cur_metrics.copy()
            ref = ref_metrics.copy()
            if allowed is not None:
                cur = cur[cur["q_site"].isin(allowed)]
                ref = ref[ref["q_site"].isin(allowed)]
            common = sorted(set(cur["q_site"]) & set(ref["q_site"]))
            cur = cur[cur["q_site"].isin(common)]
            ref = ref[ref["q_site"].isin(common)]
            cur_summary = metric_summary(cur)
            ref_summary = metric_summary(ref)
            rows.append(
                {
                    "fold_id": fold,
                    "population": population,
                    "common_stations": len(common),
                    **{f"reference_{k}": v for k, v in ref_summary.items()},
                    **{f"candidate_{k}": v for k, v in cur_summary.items()},
                    "delta_NSEraw": cur_summary["median_NSEraw"]
                    - ref_summary["median_NSEraw"],
                    "delta_NSElog": cur_summary["median_NSElog"]
                    - ref_summary["median_NSElog"],
                    "delta_KGE": cur_summary["median_KGE"]
                    - ref_summary["median_KGE"],
                    "abs_PBIAS_reduction": ref_summary["median_abs_PBIAS"]
                    - cur_summary["median_abs_PBIAS"],
                    "high_flow_nrmse_reduction": ref_summary[
                        "median_high_flow_nrmse"
                    ]
                    - cur_summary["median_high_flow_nrmse"],
                    "abs_low_flow_bias_reduction": ref_summary[
                        "median_abs_low_flow_bias"
                    ]
                    - cur_summary["median_abs_low_flow_bias"],
                    "low_flow_log_rmse_reduction": ref_summary[
                        "median_low_flow_log_rmse"
                    ]
                    - cur_summary["median_low_flow_log_rmse"],
                    "delta_good": cur_summary["good"] - ref_summary["good"],
                    "delta_abs_PBIAS_gt50": cur_summary["abs_PBIAS_gt50"]
                    - ref_summary["abs_PBIAS_gt50"],
                    "delta_abs_PBIAS_gt100": cur_summary["abs_PBIAS_gt100"]
                    - ref_summary["abs_PBIAS_gt100"],
                }
            )
    return pd.DataFrame(rows)


def no_material_regression(row: pd.Series) -> bool:
    return bool(
        row["delta_NSElog"] >= -NSELOG_EFFECT_SIZE
        and row["delta_KGE"] >= -KGE_EFFECT_SIZE
        and row["abs_PBIAS_reduction"]
        >= -ABS_PBIAS_EFFECT_SIZE_PCT_POINTS
        and row["delta_good"] >= 0
        and row["delta_abs_PBIAS_gt50"] <= 0
        and row["delta_abs_PBIAS_gt100"] <= 0
    )


def material_improvement(row: pd.Series) -> bool:
    material = bool(
        row["delta_NSElog"] >= NSELOG_EFFECT_SIZE
        or row["delta_KGE"] >= KGE_EFFECT_SIZE
        or row["abs_PBIAS_reduction"] >= ABS_PBIAS_EFFECT_SIZE_PCT_POINTS
    )
    return material and no_material_regression(row)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(
        REFERENCE
        / "reports"
        / "stage0_diagnostics"
        / "frozen_station_manifest.csv",
        encoding="utf-8-sig",
    )
    nonreservoir = set(
        frozen.loc[
            frozen["reservoir_relation"].eq("not_reservoir_related"), "q_site"
        ].astype(str)
    )
    low_flow_target_sites = set(
        frozen.loc[
            frozen["primary_failure_group"].eq("low_flow_overprediction"),
            "q_site",
        ].astype(str)
    )

    current_input_path = RUN / "inputs" / "indata.parquet"
    reference_input_path = REFERENCE / "inputs" / "indata.parquet"
    current_input = pd.read_parquet(current_input_path)
    reference_input = pd.read_parquet(reference_input_path)
    input_comparison = compare_frame(
        current_input, reference_input, ["comid", "year", "month"]
    )
    input_comparison.update(
        {
            "current_sha256": sha256(current_input_path),
            "reference_sha256": sha256(reference_input_path),
            "byte_hash_match": bool(
                sha256(current_input_path) == sha256(reference_input_path)
            ),
        }
    )

    q72_name = (
        Path("reports")
        / "intermediate"
        / "base_regression"
        / "smearing_predictions_long.csv"
    )
    current_q72 = pd.read_csv(RUN / q72_name, encoding="utf-8-sig")
    reference_q72 = pd.read_csv(REFERENCE / q72_name, encoding="utf-8-sig")
    q72_comparison = compare_frame(
        current_q72,
        reference_q72,
        ["q_site", "year", "month", "variant"],
        ["Q_obsv_cfs", "eta_log", "predict"],
    )
    q72_change_gate = bool(
        q72_comparison["row_count_match"]
        and q72_comparison["key_set_match"]
        and float(q72_comparison["max_abs_numeric_difference"]) > TOL
    )

    pred_name = (
        Path("reports")
        / "main_model"
        / "reach_class_selected_predictions_long.csv"
    )
    current_pred = pd.read_csv(RUN / pred_name, encoding="utf-8-sig")
    reference_pred = pd.read_csv(REFERENCE / pred_name, encoding="utf-8-sig")
    q78_prediction_invariants = compare_frame(
        current_pred,
        reference_pred,
        ["q_site", "reach_id", "year", "month"],
        ["Q_obsv_cfs", "Q78_mass_cfs"],
    )
    coef_name = (
        Path("reports")
        / "intermediate"
        / "mass_map_shrunk"
        / "map_reach_class_local_source_coefficients.csv"
    )
    current_coef = pd.read_csv(RUN / coef_name, encoding="utf-8-sig")
    reference_coef = pd.read_csv(REFERENCE / coef_name, encoding="utf-8-sig")
    q78_coefficient_invariants = compare_frame(
        current_coef,
        reference_coef,
        ["feature_name"],
        ["coefficient"],
    )

    recession_audit = pd.read_csv(
        RUN
        / "reports"
        / "intermediate"
        / "base_regression"
        / "nonlinear_recession_structure_audit.csv",
        encoding="utf-8-sig",
    )
    if len(recession_audit) != 1:
        raise RuntimeError("nonlinear recession audit must contain exactly one row")
    audit_row = recession_audit.iloc[0]
    recession_structure_gate = {
        "reference_exponent": float(audit_row["reference_exponent"]),
        "candidate_exponent": float(audit_row["candidate_exponent"]),
        "minimum_fitting_year": int(audit_row["minimum_fitting_year"]),
        "maximum_fitting_year": int(audit_row["maximum_fitting_year"]),
        "fitting_rows": int(audit_row["fitting_rows"]),
        "stations": int(audit_row["stations"]),
        "minimum_candidate_to_linear_ratio": float(
            audit_row["minimum_candidate_to_linear_ratio"]
        ),
        "median_candidate_to_linear_ratio": float(
            audit_row["median_candidate_to_linear_ratio"]
        ),
        "maximum_candidate_to_linear_ratio": float(
            audit_row["maximum_candidate_to_linear_ratio"]
        ),
        "candidate_release_nonnegative": bool(
            audit_row["candidate_release_nonnegative"]
        ),
        "candidate_release_not_above_linear": bool(
            audit_row["candidate_release_not_above_linear"]
        ),
        "changed_positive_release_rows": int(
            audit_row["changed_positive_release_rows"]
        ),
    }
    recession_structure_gate["passed"] = bool(
        recession_structure_gate["reference_exponent"]
        == LINEAR_RECESSION_REFERENCE_EXPONENT
        and recession_structure_gate["candidate_exponent"]
        == NONLINEAR_RECESSION_EXPONENT
        and recession_structure_gate["maximum_fitting_year"] <= 2018
        and recession_structure_gate["fitting_rows"] > 0
        and recession_structure_gate["stations"] > 0
        and recession_structure_gate["candidate_release_nonnegative"]
        and recession_structure_gate["candidate_release_not_above_linear"]
        and recession_structure_gate["changed_positive_release_rows"] > 0
        and 0.0
        <= recession_structure_gate["minimum_candidate_to_linear_ratio"]
        <= recession_structure_gate["maximum_candidate_to_linear_ratio"]
        <= 1.0 + TOL
    )

    mass_audit_paths = {
        "mass_skeleton": RUN
        / "reports"
        / "intermediate"
        / "mass_skeleton"
        / "mass_balance_audit.csv",
        "mass_global_nonnegative": RUN
        / "reports"
        / "intermediate"
        / "mass_global_nonnegative"
        / "mass_balance_audit.csv",
        "mass_reach_class": RUN
        / "reports"
        / "intermediate"
        / "mass_reach_class"
        / "mass_balance_audit.csv",
        "mass_map_shrunk": RUN
        / "reports"
        / "intermediate"
        / "mass_map_shrunk"
        / "mass_balance_audit.csv",
    }
    mass_balance_gate: dict[str, object] = {}
    for name, path in mass_audit_paths.items():
        audit = pd.read_csv(path, encoding="utf-8-sig")
        maximum = float(
            pd.to_numeric(
                audit["mass_balance_residual_cfs"], errors="coerce"
            ).abs().max()
        )
        mass_balance_gate[name] = {
            "max_abs_residual_cfs": maximum,
            "passed": bool(maximum <= TOL),
        }

    policy_gate = station_policy_gate()
    workflow = pd.read_csv(
        RUN / "reports" / "workflow" / "workflow_step_status.csv",
        encoding="utf-8-sig",
    )
    blocked = pd.read_csv(
        RUN / "reports" / "station_screening" / "blocked_fold_manifest.csv",
        encoding="utf-8-sig",
    )
    execution_gate = {
        "main_workflow_steps": int(len(workflow)),
        "main_workflow_all_zero": bool(
            len(workflow) == 12 and workflow["returncode"].eq(0).all()
        ),
        "blocked_folds": int(len(blocked)),
        "blocked_folds_all_zero": bool(
            len(blocked) == 3 and blocked["returncode"].eq(0).all()
        ),
    }

    metric_comparison = build_metric_comparison(nonreservoir)
    metric_comparison.to_csv(
        OUT / "candidate_vs_reference_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    low_flow_target_comparison = build_low_flow_target_comparison(
        low_flow_target_sites
    )
    low_flow_target_comparison.to_csv(
        OUT / "low_flow_target_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    blocked_comparison = build_blocked_fold_comparison(
        nonreservoir,
        low_flow_target_sites,
    )
    blocked_comparison.to_csv(
        OUT / "blocked_fold_candidate_vs_reference.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_population_gate = bool(
        len(metric_comparison) == len(PERIODS) * len(POPULATIONS) * len(BRANCHES)
        and (
            metric_comparison["common_stations"]
            == metric_comparison["candidate_stations"]
        ).all()
        and (
            metric_comparison["common_stations"]
            == metric_comparison["reference_stations"]
        ).all()
    )

    main_rows = metric_comparison[
        metric_comparison["branch"].eq("main_fusion")
    ].copy()
    main_rows["guardrail_passed"] = main_rows.apply(
        no_material_regression, axis=1
    )
    main_guardrails_passed = bool(
        len(main_rows) == len(PERIODS) * len(POPULATIONS)
        and main_rows["guardrail_passed"].all()
    )

    q72_rows = metric_comparison[
        metric_comparison["branch"].eq("Q72")
    ].copy()
    q72_rows["guardrail_passed"] = q72_rows.apply(
        no_material_regression, axis=1
    )
    q72_guardrails_passed = bool(
        len(q72_rows) == len(PERIODS) * len(POPULATIONS)
        and q72_rows["guardrail_passed"].all()
    )

    low_flow_population_gate = bool(
        len(low_flow_target_comparison) == len(PERIODS) * 2
        and (
            low_flow_target_comparison["common_stations"]
            == low_flow_target_comparison["candidate_stations"]
        ).all()
        and (
            low_flow_target_comparison["common_stations"]
            == low_flow_target_comparison["reference_stations"]
        ).all()
    )
    q72_low_flow = low_flow_target_comparison[
        low_flow_target_comparison["branch"].eq("Q72")
    ].copy()
    main_low_flow = low_flow_target_comparison[
        low_flow_target_comparison["branch"].eq("main_fusion")
    ].copy()
    selection_target = q72_low_flow[
        q72_low_flow["period"].eq("selection_2016_2018")
    ]
    holdout_target = q72_low_flow[
        q72_low_flow["period"].eq("development_holdout_2019_2022")
    ]
    q72_low_flow_selection_improved = bool(
        len(selection_target) == 1
        and selection_target.iloc[0][
            "abs_low_flow_bias_reduction_pct_points"
        ]
        >= LOW_FLOW_BIAS_EFFECT_SIZE_PCT_POINTS
        and selection_target.iloc[0]["low_flow_log_rmse_reduction"]
        >= LOW_FLOW_LOG_RMSE_EFFECT_SIZE
    )
    q72_low_flow_holdout_confirmed = bool(
        len(holdout_target) == 1
        and holdout_target.iloc[0][
            "abs_low_flow_bias_reduction_pct_points"
        ]
        >= LOW_FLOW_BIAS_EFFECT_SIZE_PCT_POINTS
        and holdout_target.iloc[0]["low_flow_log_rmse_reduction"]
        >= LOW_FLOW_LOG_RMSE_EFFECT_SIZE
    )
    main_low_flow_guardrail_passed = bool(
        len(main_low_flow) == len(PERIODS)
        and (
            main_low_flow["abs_low_flow_bias_reduction_pct_points"]
            >= -LOW_FLOW_BIAS_EFFECT_SIZE_PCT_POINTS
        ).all()
        and (
            main_low_flow["low_flow_log_rmse_reduction"]
            >= -LOW_FLOW_LOG_RMSE_EFFECT_SIZE
        ).all()
    )

    blocked_target = blocked_comparison[
        blocked_comparison["population"].eq("nonreservoir_target")
    ].copy()
    blocked_target["guardrail_passed"] = blocked_target.apply(
        no_material_regression, axis=1
    )
    blocked_fold_guardrails_passed = bool(
        len(blocked_target) == len(FOLDS)
        and blocked_target["guardrail_passed"].all()
    )
    blocked_low_flow = blocked_comparison[
        blocked_comparison["population"].eq("low_flow_target")
    ].copy()
    blocked_low_flow["target_improved"] = (
        blocked_low_flow["abs_low_flow_bias_reduction"].gt(0)
        & blocked_low_flow["low_flow_log_rmse_reduction"].ge(
            -LOW_FLOW_LOG_RMSE_EFFECT_SIZE
        )
    )
    blocked_fold_improvement_count = int(
        blocked_low_flow["target_improved"].sum()
    )
    blocked_fold_stability_passed = bool(
        blocked_fold_guardrails_passed
        and len(blocked_low_flow) == len(FOLDS)
        and blocked_fold_improvement_count >= 2
    )

    hard_gate = bool(
        input_comparison["byte_hash_match"]
        and input_comparison["row_count_match"]
        and input_comparison["key_set_match"]
        and float(input_comparison["max_abs_numeric_difference"]) <= TOL
        and q72_change_gate
        and q78_prediction_invariants["row_count_match"]
        and q78_prediction_invariants["key_set_match"]
        and float(q78_prediction_invariants["max_abs_numeric_difference"]) <= TOL
        and q78_coefficient_invariants["row_count_match"]
        and q78_coefficient_invariants["key_set_match"]
        and float(q78_coefficient_invariants["max_abs_numeric_difference"]) <= TOL
        and recession_structure_gate["passed"]
        and all(bool(value["passed"]) for value in mass_balance_gate.values())
        and policy_gate["excluded_exact_match"]
        and policy_gate["stone_present"]
        and policy_gate["stone_not_excluded"]
        and execution_gate["main_workflow_all_zero"]
        and execution_gate["blocked_folds_all_zero"]
        and metric_population_gate
        and low_flow_population_gate
    )
    promote = bool(
        hard_gate
        and main_guardrails_passed
        and q72_guardrails_passed
        and q72_low_flow_selection_improved
        and q72_low_flow_holdout_confirmed
        and main_low_flow_guardrail_passed
        and blocked_fold_stability_passed
    )
    decision = (
        "accept_20260727_13_as_nonlinear_recession_candidate"
        if promote
        else "reject_candidate_keep_20260727_6"
    )

    result = {
        "run_id": RUN.name,
        "reference_run": REFERENCE.name,
        "hypothesis": (
            "replace the primary production store's linear recession "
            "law with a fixed exponent-1.5 power-law recession"
        ),
        "tolerance": TOL,
        "effect_sizes": {
            "median_NSElog": NSELOG_EFFECT_SIZE,
            "median_KGE": KGE_EFFECT_SIZE,
            "median_abs_PBIAS_percentage_points": ABS_PBIAS_EFFECT_SIZE_PCT_POINTS,
            "low_flow_abs_bias_percentage_points": (
                LOW_FLOW_BIAS_EFFECT_SIZE_PCT_POINTS
            ),
            "low_flow_log_RMSE": LOW_FLOW_LOG_RMSE_EFFECT_SIZE,
        },
        "input_comparison": input_comparison,
        "q72_change_comparison": q72_comparison,
        "q72_change_gate_passed": q72_change_gate,
        "q78_prediction_invariants": q78_prediction_invariants,
        "q78_coefficient_invariants": q78_coefficient_invariants,
        "nonlinear_recession_structure_gate": recession_structure_gate,
        "mass_balance_gate": mass_balance_gate,
        "station_policy_gate": policy_gate,
        "execution_gate": execution_gate,
        "performance_gate": {
            "metric_population_gate": metric_population_gate,
            "low_flow_population_gate": low_flow_population_gate,
            "main_guardrails_passed": main_guardrails_passed,
            "q72_guardrails_passed": q72_guardrails_passed,
            "q72_low_flow_selection_improved": (
                q72_low_flow_selection_improved
            ),
            "q72_low_flow_holdout_confirmed": (
                q72_low_flow_holdout_confirmed
            ),
            "main_low_flow_guardrail_passed": (
                main_low_flow_guardrail_passed
            ),
            "blocked_fold_guardrails_passed": blocked_fold_guardrails_passed,
            "blocked_fold_improvement_count": blocked_fold_improvement_count,
            "blocked_fold_stability_passed": blocked_fold_stability_passed,
        },
        "hard_integrity_gate_passed": hard_gate,
        "promotion_passed": promote,
        "decision": decision,
    }
    (OUT / "nonlinear_recession_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    compact = metric_comparison[
        metric_comparison["branch"].isin(["Q72", "main_fusion"])
    ][
        [
            "period",
            "population",
            "branch",
            "common_stations",
            "delta_NSEraw",
            "delta_NSElog",
            "delta_KGE",
            "abs_PBIAS_reduction",
            "abs_low_flow_bias_reduction",
            "low_flow_log_rmse_reduction",
            "delta_good",
        ]
    ].copy()
    low_flow_compact = low_flow_target_comparison[
        [
            "period",
            "branch",
            "common_stations",
            "reference_median_low_flow_bias_pct",
            "candidate_median_low_flow_bias_pct",
            "abs_low_flow_bias_reduction_pct_points",
            "reference_median_low_flow_log_rmse",
            "candidate_median_low_flow_log_rmse",
            "low_flow_log_rmse_reduction",
        ]
    ].copy()
    blocked_compact = blocked_comparison[
        blocked_comparison["population"].eq("nonreservoir_target")
    ][
        [
            "fold_id",
            "common_stations",
            "delta_NSEraw",
            "delta_NSElog",
            "delta_KGE",
            "abs_PBIAS_reduction",
            "delta_good",
        ]
    ].copy()
    blocked_low_flow_compact = blocked_comparison[
        blocked_comparison["population"].eq("low_flow_target")
    ][
        [
            "fold_id",
            "common_stations",
            "abs_low_flow_bias_reduction",
            "low_flow_log_rmse_reduction",
            "delta_NSElog",
            "delta_KGE",
            "abs_PBIAS_reduction",
        ]
    ].copy()
    lines = [
        f"# {RUN.name} Stage-2A Nonlinear-Recession Gate",
        "",
        f"- reference: {REFERENCE.name}",
        f"- input byte-identical: {input_comparison['byte_hash_match']}",
        f"- Q72 changed as intended: {q72_change_gate}",
        f"- Q78 predictions unchanged: {float(q78_prediction_invariants['max_abs_numeric_difference']) <= TOL}",
        f"- Q78 coefficients unchanged: {float(q78_coefficient_invariants['max_abs_numeric_difference']) <= TOL}",
        f"- nonlinear-recession structure audit: {recession_structure_gate['passed']}",
        f"- mass-balance gate: {all(bool(value['passed']) for value in mass_balance_gate.values())}",
        f"- metric population gate: {metric_population_gate}",
        f"- low-flow target population gate: {low_flow_population_gate}",
        f"- main guardrails: {main_guardrails_passed}",
        f"- Q72 guardrails: {q72_guardrails_passed}",
        f"- Q72 low-flow selection improvement (bias >= {LOW_FLOW_BIAS_EFFECT_SIZE_PCT_POINTS:.1f} pp; log-RMSE >= {LOW_FLOW_LOG_RMSE_EFFECT_SIZE:.3f}): {q72_low_flow_selection_improved}",
        f"- Q72 low-flow holdout confirmation: {q72_low_flow_holdout_confirmed}",
        f"- main-fusion low-flow guardrail: {main_low_flow_guardrail_passed}",
        f"- blocked-fold stability: {blocked_fold_stability_passed} ({blocked_fold_improvement_count}/3 target folds reduced absolute low-flow bias without material low-flow log-RMSE regression)",
        f"- hard integrity gate: {'PASS' if hard_gate else 'FAIL'}",
        f"- promotion: {'PASS' if promote else 'FAIL'}",
        f"- decision: {decision}",
        "",
        "## Candidate minus reference",
        "",
        markdown_table(compact),
        "",
        "## Frozen low-flow-overprediction target group",
        "",
        markdown_table(low_flow_compact),
        "",
        "## Nonreservoir blocked folds",
        "",
        markdown_table(blocked_compact),
        "",
        "## Low-flow target blocked folds",
        "",
        markdown_table(blocked_low_flow_compact),
    ]
    (OUT / "nonlinear_recession_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    if not hard_gate:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
