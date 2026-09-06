from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SCREENING = RUN / "reports" / "station_screening"
INPUT_REPORTS = RUN / "reports" / "input_preprocessing"
MAIN = RUN / "reports" / "main_model"
POLICY_PATH = RUN / "inputs" / "source_metadata" / "station_screening_policy.csv"
EPS = 1.0e-6


def metric_dict(frame: pd.DataFrame, obs_col: str, pred_col: str) -> dict[str, float | int | bool]:
    obs = pd.to_numeric(frame[obs_col], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[mask]
    pred = pred[mask]
    n = int(len(obs))
    if n < 3:
        return {"n": n, "NSElog": np.nan, "KGE": np.nan, "PBIAS_pct": np.nan, "good": False}
    lo = np.log(obs + EPS)
    lp = np.log(pred + EPS)
    denom = float(np.sum((lo - np.mean(lo)) ** 2))
    nselog = np.nan if denom <= 0 else float(1.0 - np.sum((lp - lo) ** 2) / denom)
    if np.std(obs) <= 0 or np.std(pred) <= 0 or np.mean(obs) == 0:
        kge = np.nan
    else:
        r = float(np.corrcoef(obs, pred)[0, 1])
        alpha = float(np.std(pred) / np.std(obs))
        beta = float(np.mean(pred) / np.mean(obs))
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    pbias = float(100.0 * np.sum(pred - obs) / np.sum(obs))
    good = bool(n >= 24 and np.isfinite(nselog) and np.isfinite(kge) and nselog >= 0.65 and kge >= 0.50 and abs(pbias) <= 25.0)
    return {"n": n, "NSElog": nselog, "KGE": kge, "PBIAS_pct": pbias, "good": good}


def station_metrics(frame: pd.DataFrame, obs_col: str, pred_col: str, prefix: str) -> pd.DataFrame:
    rows = []
    for site, part in frame.groupby("q_site", sort=True):
        rows.append({"q_site": str(site), **{f"{prefix}_{k}": v for k, v in metric_dict(part, obs_col, pred_col).items()}})
    return pd.DataFrame(rows)


def maximum_equal_run(values: pd.Series) -> int:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    best = current = 0
    previous = np.nan
    for value in arr:
        if not np.isfinite(value):
            current = 0
            previous = np.nan
        elif np.isfinite(previous) and value == previous:
            current += 1
        else:
            current = 1
        best = max(best, current)
        previous = value
    return int(best)


def maximum_nonpositive_run(values: pd.Series) -> int:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    best = current = 0
    for value in arr:
        current = current + 1 if np.isfinite(value) and value <= 0 else 0
        best = max(best, current)
    return int(best)


def raw_station_qc() -> pd.DataFrame:
    raw = pd.read_csv(INPUT_REPORTS / "discharge_coverage_by_station_month.csv", encoding="utf-8-sig")
    raw = raw.sort_values(["station_norm", "year", "month"]).copy()
    mapping = pd.read_csv(INPUT_REPORTS / "same_reach_station_reliability.csv", encoding="utf-8-sig")
    mapping = mapping[mapping["selected_for_reach"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    duplicates = pd.read_csv(INPUT_REPORTS / "discharge_duplicate_station_years.csv", encoding="utf-8-sig")
    duplicate_counts = duplicates.groupby("station_norm").size().to_dict() if not duplicates.empty else {}
    rows: list[dict[str, object]] = []
    for site, part in raw.groupby("station_norm", sort=True):
        part = part.sort_values(["year", "month"]).copy()
        q = pd.to_numeric(part["Q_obsv_cfs"], errors="coerce")
        usable = part[part["usable"].astype(str).str.lower().isin({"true", "1", "yes"}) & q.notna()].copy()
        q_use = pd.to_numeric(usable["Q_obsv_cfs"], errors="coerce")
        nonpositive_fraction = float((q_use <= 0).mean()) if len(q_use) else np.nan
        jump_events = 0
        seasonal_outliers = 0
        if len(usable) >= 3:
            values = q_use.to_numpy(dtype=float)
            for i in range(1, len(values) - 1):
                neighbor = max(min(abs(values[i - 1]), abs(values[i + 1])), EPS)
                ratio = abs(values[i]) / neighbor
                month = int(usable.iloc[i]["month"])
                climatology = q_use[usable["month"].astype(int).eq(month)].to_numpy(dtype=float)
                med = float(np.nanmedian(climatology))
                mad = float(np.nanmedian(np.abs(climatology - med)))
                robust_z = abs(values[i] - med) / max(1.4826 * mad, EPS)
                if robust_z > 6:
                    seasonal_outliers += 1
                if ratio > 50 and robust_z > 6:
                    jump_events += 1
        early = usable[usable["year"].between(2006, 2009)].groupby("month")["Q_obsv_cfs"].median()
        later = usable[usable["year"].between(2010, 2012)].groupby("month")["Q_obsv_cfs"].median()
        common_months = early.index.intersection(later.index)
        transition_bad_months = 0
        if len(common_months):
            ratios = later.loc[common_months].to_numpy(dtype=float) / np.maximum(early.loc[common_months].to_numpy(dtype=float), EPS)
            transition_bad_months = int(((ratios < (1.0 / 3.0)) | (ratios > 3.0)).sum())
        rows.append(
            {
                "q_site": str(site),
                "raw_rows": int(len(part)),
                "usable_rows": int(len(usable)),
                "usable_fraction": float(len(usable) / max(len(part), 1)),
                "nonpositive_fraction": nonpositive_fraction,
                "max_nonpositive_run": maximum_nonpositive_run(q_use),
                "max_equal_value_run": maximum_equal_run(q_use),
                "seasonal_outlier_count_z6": int(seasonal_outliers),
                "jump_event_count_ratio50_z6": int(jump_events),
                "transition_bad_months_ratio3": int(transition_bad_months),
                "duplicate_station_year_count": int(duplicate_counts.get(site, 0)),
                "raw_median_q_cfs": float(np.nanmedian(q_use)) if len(q_use) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    map_cols = [
        "station_norm",
        "reach_id",
        "snap_distance_m",
        "obs_to_reach_flow_ratio",
        "topology_flow_plausible",
        "topology_selection_allowed",
        "reach_has_topology_flow_plausible_candidate",
        "adequate_2010_2022_coverage",
        "extended_2006_2022_coverage",
    ]
    out = out.merge(mapping[map_cols].rename(columns={"station_norm": "q_site"}), on="q_site", how="left")
    return out


def main() -> None:
    SCREENING.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(MAIN / "reach_class_selected_predictions_long.csv", encoding="utf-8-sig")
    calibration = predictions[predictions["year"].between(2010, 2018)].copy()
    cal_metrics = station_metrics(calibration, "Q_obsv_cfs", "Q_pred_cfs", "cal")

    fold_metric_parts = []
    manifest = pd.read_csv(SCREENING / "blocked_fold_manifest.csv", encoding="utf-8-sig")
    for fold in manifest.itertuples(index=False):
        path = SCREENING / "blocked_folds" / str(fold.fold_id) / "evaluation_predictions.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        metrics = station_metrics(frame, "actual", "predict", "fold")
        metrics["fold_id"] = str(fold.fold_id)
        fold_metric_parts.append(metrics)
    fold_metrics = pd.concat(fold_metric_parts, ignore_index=True)
    fold_metrics["fold_score"] = fold_metrics["fold_NSElog"] + fold_metrics["fold_KGE"] - fold_metrics["fold_PBIAS_pct"].abs() / 100.0
    fold_metrics["bottom10"] = fold_metrics.groupby("fold_id")["fold_score"].transform(lambda s: s <= s.quantile(0.10))
    fold_metrics.to_csv(SCREENING / "blocked_fold_station_metrics.csv", index=False, encoding="utf-8-sig")
    fold_summary = (
        fold_metrics.groupby("q_site")
        .agg(
            fold_count=("fold_id", "nunique"),
            fold_good_count=("fold_good", "sum"),
            fold_fail_count=("fold_good", lambda s: int((~s.astype(bool)).sum())),
            fold_bottom10_count=("bottom10", "sum"),
            fold_median_NSElog=("fold_NSElog", "median"),
            fold_median_KGE=("fold_KGE", "median"),
            fold_median_absPBIAS=("fold_PBIAS_pct", lambda s: float(np.nanmedian(np.abs(s)))),
        )
        .reset_index()
    )

    diag = pd.read_csv(MAIN / "station_diagnostic_labels.csv", encoding="utf-8-sig")
    diag_cols = ["q_site", "reach_id", "reach_class", "reservoir_relation", "NSE_log", "KGE_2012", "PBIAS_pct", "good", "failure_mode"]
    evidence = diag[diag_cols].rename(
        columns={"NSE_log": "confirm_2019_2022_NSElog", "KGE_2012": "confirm_2019_2022_KGE", "PBIAS_pct": "confirm_2019_2022_PBIAS", "good": "confirm_2019_2022_good"}
    )
    evidence = evidence.merge(cal_metrics, on="q_site", how="left").merge(fold_summary, on="q_site", how="left").merge(raw_station_qc(), on=["q_site", "reach_id"], how="left")

    log_ratio = np.log10(pd.to_numeric(evidence["obs_to_reach_flow_ratio"], errors="coerce").clip(lower=EPS))
    evidence["flow_ratio_class_robust_z"] = np.nan
    for reach_class, idx in evidence.groupby("reach_class").groups.items():
        values = log_ratio.loc[idx]
        med = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - med)))
        evidence.loc[idx, "flow_ratio_class_robust_z"] = (values - med).abs() / max(1.4826 * mad, EPS)

    evidence["reservoir_deferred"] = ~evidence["reservoir_relation"].eq("not_reservoir_related")
    evidence["flag_duplicate_conflict"] = evidence["duplicate_station_year_count"].fillna(0).gt(0)
    evidence["flag_nonpositive"] = evidence["nonpositive_fraction"].fillna(0).gt(0.05) | evidence["max_nonpositive_run"].fillna(0).ge(3)
    evidence["flag_frozen"] = evidence["max_equal_value_run"].fillna(0).ge(6)
    evidence["flag_extreme_jump"] = evidence["jump_event_count_ratio50_z6"].fillna(0).gt(0)
    evidence["flag_source_transition"] = evidence["transition_bad_months_ratio3"].fillna(0).ge(6)
    evidence["flag_station_reach_distance"] = evidence["snap_distance_m"].fillna(0).gt(500)
    evidence["flag_flow_scale"] = (
        (~pd.to_numeric(evidence["obs_to_reach_flow_ratio"], errors="coerce").between(0.1, 10.0))
        & evidence["flow_ratio_class_robust_z"].fillna(0).gt(3.5)
    )
    severe_cols = [
        "flag_duplicate_conflict",
        "flag_nonpositive",
        "flag_frozen",
        "flag_extreme_jump",
        "flag_source_transition",
        "flag_station_reach_distance",
        "flag_flow_scale",
    ]
    evidence["severe_data_flag_count"] = evidence[severe_cols].sum(axis=1).astype(int)
    evidence["persistent_model_failure"] = (~evidence["cal_good"].fillna(False).astype(bool)) & evidence["fold_fail_count"].fillna(0).ge(2)
    evidence["candidate_for_ablation"] = (~evidence["reservoir_deferred"]) & (
        evidence["persistent_model_failure"]
        | evidence["severe_data_flag_count"].gt(0)
        | (evidence["fold_bottom10_count"].fillna(0).ge(3) & evidence["severe_data_flag_count"].gt(0))
    )
    evidence["screening_status"] = np.select(
        [
            evidence["reservoir_deferred"],
            evidence["candidate_for_ablation"],
            evidence["persistent_model_failure"],
        ],
        ["defer_reservoir", "candidate", "retain_model_limitation"],
        default="retain_healthy",
    )
    evidence["reason_codes"] = evidence.apply(
        lambda row: ";".join(
            [
                *[col.removeprefix("flag_") for col in severe_cols if bool(row[col])],
                *(["persistent_model_failure"] if bool(row["persistent_model_failure"]) else []),
                *(["all_folds_bottom10"] if int(row.get("fold_bottom10_count", 0) or 0) >= 3 else []),
            ]
        ),
        axis=1,
    )
    evidence["candidate_priority"] = (
        100 * evidence["severe_data_flag_count"]
        + 20 * evidence["fold_fail_count"].fillna(0)
        + 10 * evidence["fold_bottom10_count"].fillna(0)
        + (~evidence["cal_good"].fillna(False).astype(bool)).astype(int)
    )
    evidence = evidence.sort_values(["reservoir_deferred", "candidate_for_ablation", "candidate_priority"], ascending=[True, False, False])
    evidence.to_csv(SCREENING / "station_evidence_matrix.csv", index=False, encoding="utf-8-sig")
    evidence[evidence["candidate_for_ablation"]].to_csv(SCREENING / "candidate_queue.csv", index=False, encoding="utf-8-sig")

    existing = pd.read_csv(POLICY_PATH, encoding="utf-8-sig")
    active_policy = evidence[["q_site", "screening_status", "reservoir_deferred", "reason_codes"]].rename(columns={"q_site": "station_name", "screening_status": "station_status"})
    active_policy["exclude_before_training"] = False
    active_policy["first_flagged_run"] = np.where(active_policy["station_status"].eq("candidate"), RUN.name, "")
    active_policy["last_tested_run"] = RUN.name
    active_policy["evidence_run"] = RUN.name
    active_policy["decision"] = np.where(active_policy["station_status"].eq("candidate"), "pending_ablation", "retain_or_defer")
    active_policy = active_policy[
        ["station_name", "station_status", "exclude_before_training", "reservoir_deferred", "reason_codes", "first_flagged_run", "last_tested_run", "evidence_run", "decision"]
    ]
    prior = existing[existing["exclude_before_training"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    policy = pd.concat([prior, active_policy], ignore_index=True).drop_duplicates("station_name", keep="first").sort_values("station_name")
    policy.to_csv(SCREENING / "station_screening_policy_proposed.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "run_id": RUN.name,
                "stations_total": int(evidence["q_site"].nunique()),
                "reservoir_deferred": int(evidence["reservoir_deferred"].sum()),
                "nonreservoir_checked": int((~evidence["reservoir_deferred"]).sum()),
                "candidates": int(evidence["candidate_for_ablation"].sum()),
                "persistent_model_failures": int(evidence["persistent_model_failure"].sum()),
                "stations_with_severe_data_flags": int(evidence["severe_data_flag_count"].gt(0).sum()),
            }
        ]
    )
    summary.to_csv(SCREENING / "screening_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(evidence[evidence["candidate_for_ablation"]][["q_site", "candidate_priority", "reason_codes"]].to_string(index=False))


if __name__ == "__main__":
    main()
