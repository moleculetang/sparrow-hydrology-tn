from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1.0e-6
FOLDS = (
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
)
CONTROL_RUN = Path(__file__).resolve().parents[1].parent / "20260721_1"
POLICY_PATH = CONTROL_RUN / "inputs" / "source_metadata" / "station_screening_decision_policy_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a station ablation with its accepted parent under policy v2.")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--candidate")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["q_site"] = frame["q_site"].astype(str)
    return frame


def metric_dict(frame: pd.DataFrame, obs_col: str, pred_col: str) -> dict[str, float | int | bool]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["good_thresholds"]
    obs = pd.to_numeric(frame[obs_col], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy(float)
    use = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[use], pred[use]
    n = int(len(obs))
    if n < 3:
        return {"n": n, "NSElog": np.nan, "KGE": np.nan, "PBIAS_pct": np.nan, "good": False}
    lo, lp = np.log(obs + EPS), np.log(pred + EPS)
    denom = float(np.sum((lo - lo.mean()) ** 2))
    nselog = np.nan if denom <= 0 else float(1.0 - np.sum((lp - lo) ** 2) / denom)
    if np.std(obs) <= 0 or np.std(pred) <= 0 or np.mean(obs) == 0:
        kge = np.nan
    else:
        corr = float(np.corrcoef(obs, pred)[0, 1])
        variability = float(np.std(pred) / np.std(obs))
        volume = float(np.mean(pred) / np.mean(obs))
        kge = float(1.0 - np.sqrt((corr - 1.0) ** 2 + (variability - 1.0) ** 2 + (volume - 1.0) ** 2))
    pbias = float(100.0 * np.sum(pred - obs) / np.sum(obs))
    good = bool(
        n >= int(policy["minimum_months"])
        and np.isfinite(nselog)
        and np.isfinite(kge)
        and nselog >= float(policy["NSElog_min"])
        and kge >= float(policy["KGE_min"])
        and abs(pbias) <= float(policy["abs_PBIAS_pct_max"])
    )
    return {"n": n, "NSElog": nselog, "KGE": kge, "PBIAS_pct": pbias, "good": good}


def station_metrics(frame: pd.DataFrame, obs_col: str, pred_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"q_site": str(site), **metric_dict(part, obs_col, pred_col)} for site, part in frame.groupby("q_site", sort=True)]
    )


def summarize(metrics: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    count = int(len(metrics))
    good_count = int(metrics["good"].astype(bool).sum())
    return {
        f"{prefix}_station_count": count,
        f"{prefix}_mean_NSElog": float(metrics["NSElog"].mean()),
        f"{prefix}_median_NSElog": float(metrics["NSElog"].median()),
        f"{prefix}_mean_KGE": float(metrics["KGE"].mean()),
        f"{prefix}_median_KGE": float(metrics["KGE"].median()),
        f"{prefix}_median_absPBIAS": float(metrics["PBIAS_pct"].abs().median()),
        f"{prefix}_mean_absPBIAS": float(metrics["PBIAS_pct"].abs().mean()),
        f"{prefix}_good_count": good_count,
        f"{prefix}_good_rate": float(good_count / count) if count else np.nan,
    }


def compare_common_frames(
    base_frame: pd.DataFrame,
    trial_frame: pd.DataFrame,
    obs_col: str,
    pred_col: str,
    scope: str,
    class_map: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    common_sites = sorted(set(base_frame["q_site"]) & set(trial_frame["q_site"]))
    base_metrics = station_metrics(base_frame[base_frame["q_site"].isin(common_sites)], obs_col, pred_col).set_index("q_site")
    trial_metrics = station_metrics(trial_frame[trial_frame["q_site"].isin(common_sites)], obs_col, pred_col).set_index("q_site")
    common_sites = sorted(set(base_metrics.index) & set(trial_metrics.index))
    base_metrics, trial_metrics = base_metrics.loc[common_sites], trial_metrics.loc[common_sites]
    row: dict[str, object] = {
        "scope": scope,
        **summarize(base_metrics.reset_index(), "base"),
        **summarize(trial_metrics.reset_index(), "trial"),
    }
    for metric in (
        "mean_NSElog",
        "median_NSElog",
        "mean_KGE",
        "median_KGE",
        "median_absPBIAS",
        "mean_absPBIAS",
        "good_count",
        "good_rate",
    ):
        row[f"delta_{metric}"] = float(row[f"trial_{metric}"]) - float(row[f"base_{metric}"])
    delta = pd.DataFrame(
        {
            "scope": scope,
            "q_site": common_sites,
            "reach_class": [class_map.get(site, "unknown") for site in common_sites],
            "base_NSElog": base_metrics["NSElog"].to_numpy(),
            "trial_NSElog": trial_metrics["NSElog"].to_numpy(),
            "base_KGE": base_metrics["KGE"].to_numpy(),
            "trial_KGE": trial_metrics["KGE"].to_numpy(),
            "base_PBIAS_pct": base_metrics["PBIAS_pct"].to_numpy(),
            "trial_PBIAS_pct": trial_metrics["PBIAS_pct"].to_numpy(),
            "base_good": base_metrics["good"].astype(bool).to_numpy(),
            "trial_good": trial_metrics["good"].astype(bool).to_numpy(),
        }
    )
    delta["delta_NSElog"] = delta["trial_NSElog"] - delta["base_NSElog"]
    delta["delta_KGE"] = delta["trial_KGE"] - delta["base_KGE"]
    delta["delta_absPBIAS"] = delta["trial_PBIAS_pct"].abs() - delta["base_PBIAS_pct"].abs()
    delta["good_change"] = delta["trial_good"].astype(int) - delta["base_good"].astype(int)
    class_rows = []
    for reach_class, part in delta.groupby("reach_class", sort=True):
        class_rows.append(
            {
                "scope": scope,
                "reach_class": reach_class,
                "station_count": int(len(part)),
                "base_median_NSElog": float(part["base_NSElog"].median()),
                "trial_median_NSElog": float(part["trial_NSElog"].median()),
                "delta_median_NSElog": float(part["trial_NSElog"].median() - part["base_NSElog"].median()),
                "base_median_KGE": float(part["base_KGE"].median()),
                "trial_median_KGE": float(part["trial_KGE"].median()),
                "delta_median_KGE": float(part["trial_KGE"].median() - part["base_KGE"].median()),
                "base_good_count": int(part["base_good"].sum()),
                "trial_good_count": int(part["trial_good"].sum()),
            }
        )
    return row, delta, pd.DataFrame(class_rows)


def compare_active_network(
    base_frame: pd.DataFrame,
    trial_frame: pd.DataFrame,
    obs_col: str,
    pred_col: str,
    scope: str,
) -> dict[str, object]:
    base_metrics = station_metrics(base_frame, obs_col, pred_col)
    trial_metrics = station_metrics(trial_frame, obs_col, pred_col)
    row: dict[str, object] = {
        "scope": scope,
        **summarize(base_metrics, "base"),
        **summarize(trial_metrics, "trial"),
    }
    for metric in (
        "mean_NSElog",
        "median_NSElog",
        "mean_KGE",
        "median_KGE",
        "median_absPBIAS",
        "mean_absPBIAS",
        "good_count",
        "good_rate",
    ):
        row[f"delta_{metric}"] = float(row[f"trial_{metric}"]) - float(row[f"base_{metric}"])
    return row


def main() -> None:
    args = parse_args()
    base, trial = args.base.resolve(), args.trial.resolve()
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    common_limits = policy["common_station_guardrails"]
    common_benefit = policy["benefit_common_stations"]
    negative_benefit = policy["remove_negative_contributor"]
    persistent_policy = policy["persistent_failure"]
    experiment_path = trial / "inputs" / "source_metadata" / "run_experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    candidate = str(args.candidate or experiment["candidate_station"])
    accepted_parent = str(experiment["parent_run"])
    output = trial / "reports" / "station_screening" / "ablation_comparison"
    output.mkdir(parents=True, exist_ok=True)
    base_main = read_csv(base / "reports" / "main_model" / "reach_class_selected_predictions_long.csv")
    trial_main = read_csv(trial / "reports" / "main_model" / "reach_class_selected_predictions_long.csv")
    class_map = base_main.drop_duplicates("q_site").set_index("q_site")["reach_class"]
    summaries, active_summaries, station_deltas, class_deltas, set_audits, candidate_rows = [], [], [], [], [], []

    def add_scope(base_frame: pd.DataFrame, trial_frame: pd.DataFrame, obs: str, pred: str, scope: str) -> None:
        base_sites, trial_sites = set(base_frame["q_site"]), set(trial_frame["q_site"])
        missing = sorted(base_sites - trial_sites)
        added = sorted(trial_sites - base_sites)
        set_audits.append(
            {
                "scope": scope,
                "base_stations": len(base_sites),
                "trial_stations": len(trial_sites),
                "common_stations": len(base_sites & trial_sites),
                "missing_from_trial": ";".join(missing),
                "added_in_trial": ";".join(added),
                "candidate_absent_from_trial": candidate not in trial_sites,
                "no_unexpected_missing_from_trial": set(missing).issubset({candidate}),
            }
        )
        summary, station_delta, class_delta = compare_common_frames(base_frame, trial_frame, obs, pred, scope, class_map)
        summaries.append(summary)
        active_summaries.append(compare_active_network(base_frame, trial_frame, obs, pred, scope))
        station_deltas.append(station_delta)
        class_deltas.append(class_delta)
        candidate_metric = station_metrics(base_frame[base_frame["q_site"].eq(candidate)], obs, pred)
        if candidate_metric.empty:
            candidate_rows.append({"scope": scope, "q_site": candidate, "n": 0, "NSElog": np.nan, "KGE": np.nan, "PBIAS_pct": np.nan, "good": False})
        else:
            candidate_rows.append({"scope": scope, **candidate_metric.iloc[0].to_dict()})

    for fold_id in FOLDS:
        relpath = Path("reports") / "station_screening" / "blocked_folds" / fold_id / "evaluation_predictions.csv"
        add_scope(read_csv(base / relpath), read_csv(trial / relpath), "actual", "predict", fold_id)
    for scope, start, end in (("main_2016_2018", 2016, 2018), ("confirmation_2019_2022", 2019, 2022)):
        add_scope(
            base_main[base_main["year"].between(start, end)].copy(),
            trial_main[trial_main["year"].between(start, end)].copy(),
            "Q_obsv_cfs",
            "Q_pred_cfs",
            scope,
        )

    calibration_base = base_main[base_main["split"].astype(str).eq("calibration") & base_main["year"].le(2018)].copy()
    calibration_metric = station_metrics(calibration_base[calibration_base["q_site"].eq(candidate)], "Q_obsv_cfs", "Q_pred_cfs")
    if calibration_metric.empty:
        calibration_row = {"scope": "calibration_2010_2018", "q_site": candidate, "n": 0, "NSElog": np.nan, "KGE": np.nan, "PBIAS_pct": np.nan, "good": False}
    else:
        calibration_row = {"scope": "calibration_2010_2018", **calibration_metric.iloc[0].to_dict()}
    candidate_rows.insert(0, calibration_row)

    summary = pd.DataFrame(summaries)
    active_summary = pd.DataFrame(active_summaries)
    station_delta = pd.concat(station_deltas, ignore_index=True)
    class_delta = pd.concat(class_deltas, ignore_index=True)
    set_audit = pd.DataFrame(set_audits)
    candidate_metrics = pd.DataFrame(candidate_rows)
    summary.to_csv(output / "scope_summary.csv", index=False, encoding="utf-8-sig")
    active_summary.to_csv(output / "active_network_summary.csv", index=False, encoding="utf-8-sig")
    station_delta.to_csv(output / "common_station_deltas.csv", index=False, encoding="utf-8-sig")
    class_delta.to_csv(output / "reach_class_deltas.csv", index=False, encoding="utf-8-sig")
    set_audit.to_csv(output / "station_set_audit.csv", index=False, encoding="utf-8-sig")
    candidate_metrics.to_csv(output / "candidate_metrics.csv", index=False, encoding="utf-8-sig")

    trial_input_path = trial / "inputs" / "indata.parquet"
    trial_input = pd.read_parquet(trial_input_path, columns=["q_site", "station_id"])
    candidate_input_rows = int(
        trial_input["q_site"].astype(str).eq(candidate).sum()
        + trial_input["station_id"].astype(str).eq(candidate).sum()
    )
    trial_policy = pd.read_csv(trial / "inputs" / "source_metadata" / "station_screening_policy.csv", encoding="utf-8-sig")
    policy_match = trial_policy[trial_policy["station_name"].astype(str).eq(candidate)]
    policy_excludes_candidate = bool(
        not policy_match.empty
        and str(policy_match.iloc[0]["exclude_before_training"]).strip().lower() in {"true", "1", "yes"}
    )
    candidate_main_rows = int(trial_main["q_site"].astype(str).eq(candidate).sum())
    candidate_fold_rows = int(
        sum(
            read_csv(
                trial
                / "reports"
                / "station_screening"
                / "blocked_folds"
                / fold_id
                / "evaluation_predictions.csv"
            )["q_site"]
            .astype(str)
            .eq(candidate)
            .sum()
            for fold_id in FOLDS
        )
    )
    zero_participation = {
        "candidate_station": candidate,
        "trial_run": trial.name,
        "policy_excludes_before_training": policy_excludes_candidate,
        "candidate_rows_in_trial_input_q_site_or_station_id": candidate_input_rows,
        "candidate_rows_in_trial_main_Q72_Q78_predictions": candidate_main_rows,
        "candidate_rows_in_trial_blocked_Q72_evaluations": candidate_fold_rows,
        "zero_participation_passed": bool(
            policy_excludes_candidate
            and candidate_input_rows == 0
            and candidate_main_rows == 0
            and candidate_fold_rows == 0
        ),
    }
    (output / "zero_participation_audit.json").write_text(
        json.dumps(zero_participation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    folds = summary[summary["scope"].isin(FOLDS)].copy()
    active_folds = active_summary[active_summary["scope"].isin(FOLDS)].copy()
    main_gate = summary[summary["scope"].eq("main_2016_2018")].iloc[0]
    positive_good_folds = int((folds["delta_good_count"] > 0).sum())
    cumulative_good_gain = int(folds["delta_good_count"].sum())
    mean_delta_nselog = float(folds["delta_median_NSElog"].mean())
    mean_delta_kge = float(folds["delta_median_KGE"].mean())
    benefit_good = bool(
        positive_good_folds >= int(common_benefit["minimum_positive_good_folds"])
        and cumulative_good_gain >= int(common_benefit["minimum_cumulative_good_gain"])
    )
    benefit_medians = bool(
        mean_delta_nselog >= float(common_benefit["minimum_mean_delta_median_NSElog"])
        and mean_delta_kge >= float(common_benefit["minimum_mean_delta_median_KGE"])
    )
    benefit_common_stations = bool(benefit_good or benefit_medians)

    candidate_fold_metrics = candidate_metrics[candidate_metrics["scope"].isin(FOLDS)].copy()
    candidate_calibration = candidate_metrics[candidate_metrics["scope"].eq("calibration_2010_2018")].iloc[0]
    candidate_failed_folds = int((~candidate_fold_metrics["good"].astype(bool)).sum())
    candidate_persistent_failure = bool(
        not bool(candidate_calibration["good"])
        and candidate_failed_folds >= int(persistent_policy["minimum_failed_blocked_folds"])
    )

    mean_active_delta_nselog = float(active_folds["delta_mean_NSElog"].mean())
    mean_active_delta_kge = float(active_folds["delta_mean_KGE"].mean())
    mean_active_delta_good_rate = float(active_folds["delta_good_rate"].mean())
    mean_active_delta_abs_pbias = float(active_folds["delta_mean_absPBIAS"].mean())
    active_network_benefit = bool(
        mean_active_delta_nselog > float(negative_benefit["minimum_mean_delta_active_network_mean_NSElog"]) + EPS
        and mean_active_delta_kge >= float(negative_benefit["minimum_mean_delta_active_network_mean_KGE"])
        and mean_active_delta_good_rate >= float(negative_benefit["minimum_mean_delta_active_network_good_rate"]) - EPS
        and mean_active_delta_abs_pbias <= float(negative_benefit["maximum_mean_delta_active_network_mean_abs_PBIAS_pct_points"]) + EPS
    )

    no_pbias_worsening = bool(
        float(folds["delta_mean_absPBIAS"].mean())
        <= float(common_limits["maximum_three_fold_mean_abs_PBIAS_worsening_pct_points"]) + EPS
    )
    no_fold_good_loss = bool((folds["delta_good_count"] >= -int(common_limits["maximum_good_count_loss_per_fold"])).all())
    no_fold_nselog_loss = bool((folds["delta_median_NSElog"] >= -float(common_limits["maximum_median_NSElog_loss_per_fold"])).all())
    no_fold_kge_loss = bool((folds["delta_median_KGE"] >= -float(common_limits["maximum_median_KGE_loss_per_fold"])).all())
    fold_classes = class_delta[class_delta["scope"].isin(FOLDS)]
    bad_class_counts = (
        fold_classes.assign(bad=fold_classes["delta_median_NSElog"] < -float(common_limits["maximum_reach_class_median_NSElog_loss"]))
        .groupby("reach_class")["bad"]
        .sum()
    )
    no_repeated_class_loss = bool((bad_class_counts <= int(common_limits["maximum_folds_with_repeated_reach_class_loss"])).all())
    fold_station_delta = station_delta[station_delta["scope"].isin(FOLDS)].copy()
    severe_threshold = float(common_limits["severe_station_NSElog_loss"])
    severe = fold_station_delta[fold_station_delta["delta_NSElog"] < -severe_threshold].copy()
    severe_counts_by_fold = severe.groupby("scope").size().reindex(FOLDS, fill_value=0)
    repeated_severe_stations = severe.groupby("q_site").size()
    no_concentrated_station_loss = bool(
        (severe_counts_by_fold <= int(common_limits["maximum_severely_damaged_stations_per_fold"])).all()
        and (repeated_severe_stations <= int(common_limits["maximum_folds_with_same_station_severely_damaged"])).all()
    )
    main_severe_count = int(
        (station_delta[station_delta["scope"].eq("main_2016_2018")]["delta_NSElog"] < -severe_threshold).sum()
    )
    main_guardrail = bool(
        float(main_gate["delta_good_count"]) >= -int(common_limits["maximum_good_count_loss_per_fold"])
        and float(main_gate["delta_median_NSElog"]) >= -float(common_limits["maximum_median_NSElog_loss_per_fold"])
        and float(main_gate["delta_median_KGE"]) >= -float(common_limits["maximum_median_KGE_loss_per_fold"])
        and float(main_gate["delta_mean_absPBIAS"]) <= float(common_limits["maximum_three_fold_mean_abs_PBIAS_worsening_pct_points"]) + EPS
        and main_severe_count <= int(common_limits["maximum_severely_damaged_stations_per_fold"])
    )
    candidate_absent = bool(set_audit["candidate_absent_from_trial"].all())
    zero_participation_passed = bool(zero_participation["zero_participation_passed"])
    no_unexpected_missing = bool(set_audit["no_unexpected_missing_from_trial"].all())
    common_guardrails_passed = bool(
        no_pbias_worsening
        and no_fold_good_loss
        and no_fold_nselog_loss
        and no_fold_kge_loss
        and no_repeated_class_loss
        and no_concentrated_station_loss
        and main_guardrail
        and candidate_absent
        and zero_participation_passed
        and no_unexpected_missing
    )
    benefit_remove_negative_contributor = bool(candidate_persistent_failure and active_network_benefit)
    accepted = bool(common_guardrails_passed and (benefit_common_stations or benefit_remove_negative_contributor))
    accepted_paths = []
    if benefit_common_stations:
        accepted_paths.append("benefit_common_stations")
    if benefit_remove_negative_contributor:
        accepted_paths.append("remove_negative_contributor")
    accepted_by = "+".join(accepted_paths) if accepted else ""

    gates = {
        "policy_version": str(policy["policy_version"]),
        "candidate_station": candidate,
        "metric_reference_run": base.name,
        "accepted_parent_run": accepted_parent,
        "trial_run": trial.name,
        "selection_years": "2006-2018_only",
        "candidate_calibration_period": "2010-2018",
        "confirmation_years_not_used_for_decision": "2019-2022",
        "accepted": accepted,
        "decision": "accept_exclusion" if accepted else "reject_exclusion",
        "accepted_by": accepted_by,
        "benefit_common_stations": benefit_common_stations,
        "benefit_remove_negative_contributor": benefit_remove_negative_contributor,
        "candidate_persistent_failure": candidate_persistent_failure,
        "candidate_calibration_good": bool(candidate_calibration["good"]),
        "candidate_failed_blocked_folds": candidate_failed_folds,
        "active_network_benefit": active_network_benefit,
        "active_network_mean_NSElog_strictly_positive": mean_active_delta_nselog > EPS,
        "mean_delta_active_network_mean_NSElog": mean_active_delta_nselog,
        "mean_delta_active_network_mean_KGE": mean_active_delta_kge,
        "mean_delta_active_network_good_rate": mean_active_delta_good_rate,
        "mean_delta_active_network_mean_absPBIAS": mean_active_delta_abs_pbias,
        "benefit_good": benefit_good,
        "positive_good_folds": positive_good_folds,
        "cumulative_good_gain": cumulative_good_gain,
        "benefit_medians": benefit_medians,
        "mean_delta_median_NSElog": mean_delta_nselog,
        "mean_delta_median_KGE": mean_delta_kge,
        "common_guardrails_passed": common_guardrails_passed,
        "no_mean_absPBIAS_worsening_gt_0_5": no_pbias_worsening,
        "mean_delta_common_station_mean_absPBIAS": float(folds["delta_mean_absPBIAS"].mean()),
        "mean_base_absPBIAS": float(folds["base_mean_absPBIAS"].mean()),
        "mean_trial_absPBIAS": float(folds["trial_mean_absPBIAS"].mean()),
        "no_fold_good_loss_gt_1": no_fold_good_loss,
        "no_fold_median_NSElog_loss_gt_0_01": no_fold_nselog_loss,
        "no_fold_median_KGE_loss_gt_0_02": no_fold_kge_loss,
        "no_reach_class_NSElog_loss_gt_0_02_in_two_folds": no_repeated_class_loss,
        "no_concentrated_station_NSElog_loss": no_concentrated_station_loss,
        "severe_station_loss_counts_by_fold": {str(k): int(v) for k, v in severe_counts_by_fold.items()},
        "repeated_severely_damaged_stations": [str(x) for x in repeated_severe_stations[repeated_severe_stations > 1].index],
        "main_2016_2018_guardrail": main_guardrail,
        "candidate_absent_from_all_trial_evaluations": candidate_absent,
        "zero_participation_passed": zero_participation_passed,
        "zero_participation_audit": zero_participation,
        "no_unexpected_missing_stations": no_unexpected_missing,
        "repeated_bad_reach_classes": [str(x) for x in bad_class_counts[bad_class_counts > 1].index],
    }
    (output / "decision.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fold_lines = [
        f"| {r.scope} | {int(r.base_station_count)} | {int(r.delta_good_count):+d} | {r.delta_median_NSElog:+.6f} | {r.delta_median_KGE:+.6f} | {r.delta_mean_absPBIAS:+.6f} |"
        for r in folds.itertuples(index=False)
    ]
    active_lines = [
        f"| {r.scope} | {int(r.base_station_count)}→{int(r.trial_station_count)} | {r.delta_mean_NSElog:+.6f} | {r.delta_mean_KGE:+.6f} | {r.delta_good_rate:+.6f} | {r.delta_mean_absPBIAS:+.6f} |"
        for r in active_folds.itertuples(index=False)
    ]
    confirmation = summary[summary["scope"].eq("confirmation_2019_2022")].iloc[0]
    report = f"""# {trial.name} Station Ablation Decision — Policy v2

- candidate: {candidate}
- accepted parent before test: {accepted_parent}
- metric reference: {base.name}
- decision: **{gates['decision']}**
- accepted by: {accepted_by or 'none'}
- selection evidence: 2006–2018 only; candidate calibration metrics cover 2010–2018
- 2019–2022: confirmation output only; not used in the decision

## Common-station causal effect

| fold | common stations | delta good | delta median NSElog | delta median KGE | delta mean abs PBIAS |
|---|---:|---:|---:|---:|---:|
{chr(10).join(fold_lines)}

## Active-network utility effect

| fold | station count | delta mean NSElog | delta mean KGE | delta good rate | delta mean abs PBIAS |
|---|---:|---:|---:|---:|---:|
{chr(10).join(active_lines)}

- candidate calibration good: {bool(candidate_calibration['good'])}
- candidate failed blocked folds: {candidate_failed_folds}/3
- candidate persistent failure: {candidate_persistent_failure}
- zero participation in input, Q72 and Q78: {zero_participation_passed}
- common-station benefit: {benefit_common_stations}
- negative-contributor benefit: {benefit_remove_negative_contributor}
- common guardrails passed: {common_guardrails_passed}

## Gate result

```json
{json.dumps(gates, ensure_ascii=False, indent=2)}
```

## 2019–2022 confirmation (not a selection gate)

- delta good: {int(confirmation['delta_good_count']):+d}
- delta median NSElog: {confirmation['delta_median_NSElog']:+.6f}
- delta median KGE: {confirmation['delta_median_KGE']:+.6f}
- delta mean abs PBIAS: {confirmation['delta_mean_absPBIAS']:+.6f}
"""
    (output / "decision_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
