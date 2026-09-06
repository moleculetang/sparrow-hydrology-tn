from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SERIES = RUN.parent
OUT = RUN / "reports" / "minimal_residual_pilot"
EPS = 1.0e-6
AMPLITUDES = [0.0, 0.05, 0.10, 0.15, 0.20]
TIME_GATE_SHARPNESS = 3.0
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站"}
PROTECTED = "石角站"
OUTER_MAP = {
    "fit_through_2011_eval_2012_2013": "fit_2006_2011_eval_2012_2013",
    "fit_through_2013_eval_2014_2015": "fit_2006_2013_eval_2014_2015",
    "fit_through_2015_eval_2016_2018": "fit_2006_2015_eval_2016_2018",
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def apply_correction(frame: pd.DataFrame, amplitude: float) -> np.ndarray:
    baseline = frame["Q0_pred_cfs"].to_numpy(dtype=float)
    if float(amplitude) == 0.0:
        return baseline.copy()
    station_gate = frame["station_gate"].to_numpy(dtype=float)
    delta = (
        station_gate
        * frame["time_gate"].to_numpy(dtype=float)
        * float(amplitude)
    )
    corrected = np.exp(
        np.log(baseline + EPS) - delta
    ) - EPS
    corrected = np.maximum(corrected, 0.0)
    corrected[station_gate == 0] = baseline[station_gate == 0]
    return corrected


def station_metric(part: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    obs = part["Q_obsv_cfs"].to_numpy(dtype=float)
    pred = part[prediction_column].to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred >= 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {
            "n": 0,
            "log_rmse": np.nan,
            "NSElog": np.nan,
            "KGE": np.nan,
            "PBIAS_pct": np.nan,
            "log_sse": np.nan,
        }
    log_obs = np.log(obs + EPS)
    log_pred = np.log(pred + EPS)
    log_sse = float(np.sum((log_pred - log_obs) ** 2))
    denominator = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
    nselog = float(1 - log_sse / denominator) if denominator > 0 else np.nan
    if len(obs) >= 2 and np.std(obs) > 0 and np.std(pred) > 0:
        correlation = float(np.corrcoef(obs, pred)[0, 1])
        beta = float(np.mean(pred) / np.mean(obs))
        cv_obs = float(np.std(obs) / np.mean(obs))
        cv_pred = float(np.std(pred) / np.mean(pred)) if np.mean(pred) else np.nan
        gamma = cv_pred / cv_obs if cv_obs > 0 else np.nan
        kge = float(
            1
            - np.sqrt(
                (correlation - 1) ** 2
                + (beta - 1) ** 2
                + (gamma - 1) ** 2
            )
        )
    else:
        kge = np.nan
    return {
        "n": int(len(obs)),
        "log_rmse": float(np.sqrt(log_sse / len(obs))),
        "NSElog": nselog,
        "KGE": kge,
        "PBIAS_pct": float(100 * np.sum(pred - obs) / np.sum(obs)),
        "log_sse": log_sse,
    }


def build_time_statistics(
    nested: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for outer, outer_assignments in assignments.groupby("outer_fold", sort=True):
        allowed = str(
            outer_assignments["allowed_inner_blocks"].iloc[0]
        ).split("|")
        history = nested[nested["block_id"].isin(allowed)]
        for site, part in history.groupby("q_site", sort=True):
            q0 = part["Q0_pred_cfs"].to_numpy(dtype=float)
            logq = np.log(q0 + EPS)
            rows.append(
                {
                    "outer_fold": outer,
                    "q_site": site,
                    "time_history_rows": int(len(part)),
                    "time_q25_Q0_cfs": float(np.quantile(q0, 0.25)),
                    "time_q75_Q0_cfs": float(np.quantile(q0, 0.75)),
                    "time_logq_iqr": float(
                        np.quantile(logq, 0.75) - np.quantile(logq, 0.25)
                    ),
                    "time_gate_scale": float(
                        max(
                            np.quantile(logq, 0.75)
                            - np.quantile(logq, 0.25),
                            0.25,
                        )
                    ),
                    "allowed_inner_blocks": "|".join(allowed),
                }
            )
    return pd.DataFrame(rows)


def attach_gate_and_time(
    frame: pd.DataFrame,
    outer_fold: str,
    assignments: pd.DataFrame,
    time_stats: pd.DataFrame,
) -> pd.DataFrame:
    gates = assignments[assignments["outer_fold"].eq(outer_fold)][
        [
            "q_site",
            "station_gate",
            "reservoir_related",
            "data_quality_suspicious",
            "hard_zero_reason",
        ]
    ]
    stats = time_stats[time_stats["outer_fold"].eq(outer_fold)].drop(
        columns=["outer_fold"]
    )
    result = frame.merge(gates, on="q_site", how="left", validate="many_to_one").merge(
        stats,
        on="q_site",
        how="left",
        validate="many_to_one",
    )
    result["station_gate"] = result["station_gate"].fillna(0.0)
    missing_time = result["time_gate_scale"].isna()
    result.loc[missing_time, "station_gate"] = 0.0
    numerator = (
        np.log(result["time_q25_Q0_cfs"].fillna(result["Q0_pred_cfs"]) + EPS)
        - np.log(result["Q0_pred_cfs"] + EPS)
    )
    denominator = result["time_gate_scale"].fillna(1.0)
    result["time_gate"] = sigmoid(
        (TIME_GATE_SHARPNESS * numerator / denominator).to_numpy(dtype=float)
    )
    result.loc[missing_time, "time_gate"] = 0.0
    return result


def low_station_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    low = frame[frame["flow_regime"].eq("low") & frame["station_gate"].gt(0)]
    for site, part in low.groupby("q_site", sort=True):
        before = station_metric(part, "Q0_pred_cfs")
        after = station_metric(part, "Qnew_pred_cfs")
        rows.append(
            {
                "q_site": site,
                "n": before["n"],
                "before_log_rmse": before["log_rmse"],
                "after_log_rmse": after["log_rmse"],
                "log_rmse_improvement": before["log_rmse"] - after["log_rmse"],
                "before_PBIAS_pct": before["PBIAS_pct"],
                "after_PBIAS_pct": after["PBIAS_pct"],
                "abs_PBIAS_improvement_pct_points": abs(before["PBIAS_pct"])
                - abs(after["PBIAS_pct"]),
                "before_log_sse": before["log_sse"],
                "after_log_sse": after["log_sse"],
                "log_sse_improvement": before["log_sse"] - after["log_sse"],
            }
        )
    result = pd.DataFrame(rows)
    return result[result["n"].ge(3)].copy() if not result.empty else result


def evaluate_amplitudes(
    outer_fold: str,
    nested: pd.DataFrame,
    assignments: pd.DataFrame,
    time_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    allowed = str(
        assignments.loc[
            assignments["outer_fold"].eq(outer_fold),
            "allowed_inner_blocks",
        ].iloc[0]
    ).split("|")
    training = attach_gate_and_time(
        nested[nested["block_id"].isin(allowed)].copy(),
        outer_fold,
        assignments,
        time_stats,
    )
    rows = []
    for amplitude in AMPLITUDES:
        part = training.copy()
        part["Qnew_pred_cfs"] = apply_correction(part, amplitude)
        part["abs_delta_logq"] = np.abs(
            np.log(part["Qnew_pred_cfs"] + EPS)
            - np.log(part["Q0_pred_cfs"] + EPS)
        )
        metrics = low_station_metrics(part)
        target_nonlow = part[
            part["station_gate"].gt(0)
            & ~part["flow_regime"].eq("low")
            & ~part["flow_regime"].eq("unclassified_no_training_q25")
        ]
        predicted_high = part[
            part["station_gate"].gt(0)
            & (part["Q0_pred_cfs"] >= part["time_q75_Q0_cfs"])
        ]
        nontarget = part[part["station_gate"].eq(0)]
        non_target_change = (
            float(nontarget["abs_delta_logq"].median())
            if len(nontarget)
            else np.nan
        )
        nonlow_change = (
            float(target_nonlow["abs_delta_logq"].median())
            if len(target_nonlow)
            else np.nan
        )
        high_change = (
            float(predicted_high["abs_delta_logq"].median())
            if len(predicted_high)
            else np.nan
        )
        guardrails = {
            "inner_clean_nontarget_exact": bool(
                np.isfinite(non_target_change) and non_target_change <= 1.0e-12
            ),
            "inner_target_nonlow_change_at_most_0_01": bool(
                np.isfinite(nonlow_change) and nonlow_change <= 0.01
            ),
            "inner_predicted_high_change_at_most_0_005": bool(
                np.isfinite(high_change) and high_change <= 0.005
            ),
        }
        rows.append(
            {
                "outer_fold": outer_fold,
                "amplitude": amplitude,
                "target_low_stations": int(len(metrics)),
                "target_low_rows": int(
                    (
                        part["station_gate"].gt(0)
                        & part["flow_regime"].eq("low")
                    ).sum()
                ),
                "median_target_low_log_rmse_before": float(
                    metrics["before_log_rmse"].median()
                )
                if len(metrics)
                else np.nan,
                "median_target_low_log_rmse_after": float(
                    metrics["after_log_rmse"].median()
                )
                if len(metrics)
                else np.nan,
                "median_target_low_log_rmse_improvement": float(
                    metrics["log_rmse_improvement"].median()
                )
                if len(metrics)
                else np.nan,
                "median_target_low_abs_pbias_before": float(
                    metrics["before_PBIAS_pct"].abs().median()
                )
                if len(metrics)
                else np.nan,
                "median_target_low_abs_pbias_after": float(
                    metrics["after_PBIAS_pct"].abs().median()
                )
                if len(metrics)
                else np.nan,
                "clean_nontarget_median_abs_delta_logq": non_target_change,
                "target_nonlow_median_abs_delta_logq": nonlow_change,
                "predicted_high_median_abs_delta_logq": high_change,
                **guardrails,
                "inner_guardrails_passed": bool(all(guardrails.values())),
            }
        )
    candidates = pd.DataFrame(rows)
    eligible = candidates[
        candidates["inner_guardrails_passed"]
        & candidates["median_target_low_log_rmse_after"].notna()
    ].sort_values(
        ["median_target_low_log_rmse_after", "amplitude"],
        ascending=[True, True],
    )
    if eligible.empty:
        raise RuntimeError(f"No guardrail-compliant amplitude for {outer_fold}")
    selected = float(eligible.iloc[0]["amplitude"])
    candidates["selected"] = candidates["amplitude"].eq(selected)
    return candidates, selected


def outer_station_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, part in frame.groupby("q_site", sort=True):
        for regime, regime_part in [
            ("all", part),
            ("low", part[part["flow_regime"].eq("low")]),
            ("nonlow", part[~part["flow_regime"].eq("low")]),
            ("high", part[part["flow_regime"].eq("high")]),
        ]:
            if regime_part.empty:
                continue
            before = station_metric(regime_part, "Q0_pred_cfs")
            after = station_metric(regime_part, "Qnew_pred_cfs")
            rows.append(
                {
                    "q_site": site,
                    "regime": regime,
                    "station_gate": float(part["station_gate"].iloc[0]),
                    "n": before["n"],
                    "before_log_rmse": before["log_rmse"],
                    "after_log_rmse": after["log_rmse"],
                    "log_rmse_improvement": before["log_rmse"]
                    - after["log_rmse"],
                    "before_NSElog": before["NSElog"],
                    "after_NSElog": after["NSElog"],
                    "before_KGE": before["KGE"],
                    "after_KGE": after["KGE"],
                    "before_PBIAS_pct": before["PBIAS_pct"],
                    "after_PBIAS_pct": after["PBIAS_pct"],
                    "abs_PBIAS_improvement_pct_points": abs(before["PBIAS_pct"])
                    - abs(after["PBIAS_pct"]),
                    "before_log_sse": before["log_sse"],
                    "after_log_sse": after["log_sse"],
                    "log_sse_improvement": before["log_sse"]
                    - after["log_sse"],
                }
            )
    return pd.DataFrame(rows)


def summarize_outer(
    outer_fold: str,
    frame: pd.DataFrame,
    station_metrics: pd.DataFrame,
    amplitude: float,
) -> dict[str, object]:
    target_low = station_metrics[
        station_metrics["regime"].eq("low")
        & station_metrics["station_gate"].gt(0)
        & station_metrics["n"].ge(3)
    ]
    all_metrics = station_metrics[station_metrics["regime"].eq("all")].copy()
    delta = np.abs(
        np.log(frame["Qnew_pred_cfs"] + EPS)
        - np.log(frame["Q0_pred_cfs"] + EPS)
    )
    frame = frame.copy()
    frame["abs_delta_logq"] = delta
    clean_nontarget = frame[frame["station_gate"].eq(0)]
    target_nonlow = frame[
        frame["station_gate"].gt(0)
        & ~frame["flow_regime"].eq("low")
    ]
    high = frame[
        frame["station_gate"].gt(0)
        & frame["flow_regime"].eq("high")
    ]
    newly_severe = int(
        (
            all_metrics["before_PBIAS_pct"].abs().le(50)
            & all_metrics["after_PBIAS_pct"].abs().gt(50)
        ).sum()
    )
    positive_sse = target_low["log_sse_improvement"].clip(lower=0)
    top_share = (
        float(positive_sse.max() / positive_sse.sum())
        if len(positive_sse) and positive_sse.sum() > 0
        else 0.0
    )
    top_gain_station = (
        str(target_low.loc[positive_sse.idxmax(), "q_site"])
        if len(positive_sse) and positive_sse.sum() > 0
        else ""
    )
    top_gain_log_sse = (
        float(positive_sse.max())
        if len(positive_sse) and positive_sse.sum() > 0
        else 0.0
    )
    before_good = (
        all_metrics["n"].ge(24)
        & all_metrics["before_NSElog"].ge(0.65)
        & all_metrics["before_KGE"].ge(0.50)
        & all_metrics["before_PBIAS_pct"].abs().le(25)
    )
    after_good = (
        all_metrics["n"].ge(24)
        & all_metrics["after_NSElog"].ge(0.65)
        & all_metrics["after_KGE"].ge(0.50)
        & all_metrics["after_PBIAS_pct"].abs().le(25)
    )
    target_abs_before = float(target_low["before_PBIAS_pct"].abs().median())
    target_abs_after = float(target_low["after_PBIAS_pct"].abs().median())
    target_log_before = float(target_low["before_log_rmse"].median())
    target_log_after = float(target_low["after_log_rmse"].median())
    nontarget_change = (
        float(clean_nontarget["abs_delta_logq"].median())
        if len(clean_nontarget)
        else np.nan
    )
    nonlow_change = (
        float(target_nonlow["abs_delta_logq"].median())
        if len(target_nonlow)
        else np.nan
    )
    high_change = (
        float(high["abs_delta_logq"].median()) if len(high) else np.nan
    )
    criteria = {
        "target_low_abs_pbias_improved": (
            target_abs_before - target_abs_after
        ) > 1.0e-12,
        "target_low_logrmse_improved": (
            target_log_before - target_log_after
        ) > 1.0e-12,
        "clean_nontarget_change_at_most_0_01": bool(
            np.isfinite(nontarget_change) and nontarget_change <= 0.01
        ),
        "target_nonlow_change_at_most_0_01": bool(
            np.isfinite(nonlow_change) and nonlow_change <= 0.01
        ),
        "high_flow_change_at_most_0_005": bool(
            np.isfinite(high_change) and high_change <= 0.005
        ),
        "no_new_severe_abs_pbias_station": newly_severe == 0,
        "single_station_gain_share_at_most_0_25": bool(
            np.isfinite(top_share) and top_share <= 0.25
        ),
    }
    return {
        "outer_fold": outer_fold,
        "selected_amplitude": amplitude,
        "target_low_stations": int(len(target_low)),
        "target_low_median_abs_pbias_before": target_abs_before,
        "target_low_median_abs_pbias_after": target_abs_after,
        "target_low_median_abs_pbias_improvement_pct_points": target_abs_before
        - target_abs_after,
        "target_low_median_logrmse_before": target_log_before,
        "target_low_median_logrmse_after": target_log_after,
        "target_low_median_logrmse_improvement": target_log_before
        - target_log_after,
        "clean_nontarget_median_abs_delta_logq": nontarget_change,
        "target_nonlow_median_abs_delta_logq": nonlow_change,
        "high_flow_median_abs_delta_logq": high_change,
        "new_severe_abs_pbias_stations": newly_severe,
        "maximum_single_station_positive_sse_gain_station": top_gain_station,
        "maximum_single_station_positive_log_sse_gain": top_gain_log_sse,
        "maximum_single_station_positive_sse_gain_share": top_share,
        "good_before": int(before_good.sum()),
        "good_after": int(after_good.sum()),
        "good_delta": int(after_good.sum() - before_good.sum()),
        **criteria,
        "outer_fold_all_contract_gates_passed": bool(all(criteria.values())),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predecessor_gate = json.loads(
        (SERIES / "20260728_7" / "reports" / "gate.json").read_text(
            encoding="utf-8"
        )
    )
    if not predecessor_gate["scientific_result"]["residual_pilot_permitted"]:
        raise RuntimeError("Nested station gate did not permit this pilot")
    assignments = pd.read_csv(
        SERIES
        / "20260728_7"
        / "reports"
        / "nested_oof_gate"
        / "outer_fold_station_gate_assignments.csv",
        encoding="utf-8-sig",
    )
    nested = pd.read_csv(
        SERIES
        / "20260728_7"
        / "reports"
        / "nested_oof_gate"
        / "nested_block_oof_predictions.csv",
        encoding="utf-8-sig",
    )
    outer_source = pd.read_csv(
        SERIES
        / "20260728_4"
        / "reports"
        / "q72_q78_complementarity"
        / "oof_predictions_2012_2018.csv",
        encoding="utf-8-sig",
    )
    time_stats = build_time_statistics(nested, assignments)

    selection_parts = []
    outer_prediction_parts = []
    station_metric_parts = []
    outer_rows = []
    selected_rows = []
    for outer_fold, source_fold in OUTER_MAP.items():
        selection, selected_amplitude = evaluate_amplitudes(
            outer_fold,
            nested,
            assignments,
            time_stats,
        )
        selection_parts.append(selection)
        selected_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_amplitude": selected_amplitude,
                "selection_rows": int(
                    selection.loc[selection["selected"], "target_low_rows"].iloc[0]
                ),
                "selection_stations": int(
                    selection.loc[
                        selection["selected"],
                        "target_low_stations",
                    ].iloc[0]
                ),
            }
        )
        outer = outer_source[
            outer_source["fold_id"].eq(source_fold)
        ].copy()
        outer = outer.rename(
            columns={
                "Q_original_fusion_cfs": "Q0_pred_cfs",
            }
        )
        outer = attach_gate_and_time(
            outer,
            outer_fold,
            assignments,
            time_stats,
        )
        outer["outer_fold"] = outer_fold
        outer["selected_amplitude"] = selected_amplitude
        outer["Qnew_pred_cfs"] = apply_correction(outer, selected_amplitude)
        outer["delta_logq"] = (
            np.log(outer["Qnew_pred_cfs"] + EPS)
            - np.log(outer["Q0_pred_cfs"] + EPS)
        )
        outer_prediction_parts.append(outer)
        metrics = outer_station_metrics(outer)
        metrics.insert(0, "outer_fold", outer_fold)
        metrics.insert(1, "selected_amplitude", selected_amplitude)
        station_metric_parts.append(metrics)
        outer_rows.append(
            summarize_outer(
                outer_fold,
                outer,
                metrics,
                selected_amplitude,
            )
        )
        print(
            f"{outer_fold}: selected_amplitude={selected_amplitude:.2f} "
            f"target_low_stations={outer_rows[-1]['target_low_stations']} "
            f"pbias_gain={outer_rows[-1]['target_low_median_abs_pbias_improvement_pct_points']:.3f} "
            f"logrmse_gain={outer_rows[-1]['target_low_median_logrmse_improvement']:.4f}",
            flush=True,
        )

    selection_all = pd.concat(selection_parts, ignore_index=True)
    predictions = pd.concat(outer_prediction_parts, ignore_index=True)
    station_metrics = pd.concat(station_metric_parts, ignore_index=True)
    outer_summary = pd.DataFrame(outer_rows)

    pbias_improvement_folds = int(
        outer_summary["target_low_abs_pbias_improved"].sum()
    )
    logrmse_improvement_folds = int(
        outer_summary["target_low_logrmse_improved"].sum()
    )
    protection_columns = [
        "clean_nontarget_change_at_most_0_01",
        "target_nonlow_change_at_most_0_01",
        "high_flow_change_at_most_0_005",
        "no_new_severe_abs_pbias_station",
        "single_station_gain_share_at_most_0_25",
    ]
    protection_all_folds = bool(
        outer_summary[protection_columns].all().all()
    )
    scientific_pass = bool(
        pbias_improvement_folds >= 2
        and logrmse_improvement_folds >= 2
        and protection_all_folds
    )
    failed_protection = [
        column
        for column in protection_columns
        if not bool(outer_summary[column].all())
    ]
    selected_zero_all = bool(
        all(row["selected_amplitude"] == 0.0 for row in selected_rows)
    )
    positive_rows = selection_all[selection_all["amplitude"].gt(0)]
    positive_target_signal_folds = int(
        positive_rows.groupby("outer_fold")[
            "median_target_low_log_rmse_improvement"
        ].max().gt(0).sum()
    )
    positive_nontarget_anchor_exact = bool(
        positive_rows["inner_clean_nontarget_exact"].all()
    )
    positive_time_guardrail_failures = bool(
        (
            ~positive_rows["inner_target_nonlow_change_at_most_0_01"]
            | ~positive_rows["inner_predicted_high_change_at_most_0_005"]
        ).all()
    )
    time_gate_breadth_diagnosed = bool(
        selected_zero_all
        and positive_target_signal_folds >= 2
        and positive_nontarget_anchor_exact
        and positive_time_guardrail_failures
    )
    if scientific_pass:
        decision = "minimal_pilot_passed_freeze_for_robustness_audit"
        next_action = "freeze_current_rule_and_run_temporal_robustness_audit"
    else:
        decision = "bounded_time_gate_sharpening_failed"
        next_action = "evidence_stop_current_candidate"

    scientific = {
        "run_id": RUN.name,
        "phase_id": "bounded_time_gate_sharpening_iteration",
        "contract_id": "C01_gauged_history_low_flow_residual_expert",
        "time_gate_sharpness": TIME_GATE_SHARPNESS,
        "bounded_iteration_consumed": True,
        "selected_amplitudes": selected_rows,
        "target_low_abs_pbias_improvement_folds": pbias_improvement_folds,
        "target_low_logrmse_improvement_folds": logrmse_improvement_folds,
        "required_improvement_folds": 2,
        "protection_all_folds": protection_all_folds,
        "failed_protection_gates": failed_protection,
        "selected_zero_all_folds": selected_zero_all,
        "positive_amplitude_target_signal_folds": positive_target_signal_folds,
        "positive_amplitude_nontarget_anchor_exact": positive_nontarget_anchor_exact,
        "positive_amplitude_time_guardrail_failures": positive_time_guardrail_failures,
        "time_gate_breadth_diagnosed": time_gate_breadth_diagnosed,
        "candidate_evidence_stop": not scientific_pass,
        "minimal_pilot_passed": scientific_pass,
        "decision": decision,
        "next_action": next_action,
        "folds": outer_summary.to_dict(orient="records"),
    }

    time_stats.to_csv(
        OUT / "outer_fold_time_gate_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection_all.to_csv(
        OUT / "nested_amplitude_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions[
        [
            "outer_fold",
            "fold_id",
            "q_site",
            "reach_id",
            "year",
            "month",
            "Q_obsv_cfs",
            "Q0_pred_cfs",
            "Qnew_pred_cfs",
            "flow_regime",
            "station_gate",
            "time_gate",
            "selected_amplitude",
            "delta_logq",
            "hard_zero_reason",
        ]
    ].to_csv(
        OUT / "outer_oof_corrected_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_metrics.to_csv(
        OUT / "outer_station_regime_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    outer_summary.to_csv(
        OUT / "outer_fold_pilot_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "scientific_gate.json").write_text(
        json.dumps(scientific, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "run_id": RUN.name,
        "phase_id": "bounded_time_gate_sharpening_iteration",
        "time_gate_sharpness": TIME_GATE_SHARPNESS,
        "bounded_iteration_consumed": True,
        "outer_prediction_rows": int(len(predictions)),
        "outer_stations": int(predictions["q_site"].nunique()),
        "outer_year_min": int(predictions["year"].min()),
        "outer_year_max": int(predictions["year"].max()),
        "selected_amplitudes": {
            row["outer_fold"]: row["selected_amplitude"]
            for row in selected_rows
        },
        "target_low_abs_pbias_improvement_folds": pbias_improvement_folds,
        "target_low_logrmse_improvement_folds": logrmse_improvement_folds,
        "protection_all_folds": protection_all_folds,
        "minimal_pilot_passed": scientific_pass,
        "candidate_evidence_stop": not scientific_pass,
        "excluded_station_rows": int(predictions["q_site"].isin(EXCLUSIONS).sum()),
        "protected_station_rows": int(predictions["q_site"].eq(PROTECTED).sum()),
        "decision": decision,
        "next_action": next_action,
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
