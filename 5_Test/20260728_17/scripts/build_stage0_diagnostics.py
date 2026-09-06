from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
MAIN = RUN / "reports" / "main_model"
OUT = RUN / "reports" / "stage0_diagnostics"
EPS = 1.0e-6

BRANCHES = {
    "Q72": "Q72_pred_cfs",
    "Q78_mass": "Q78_mass_cfs",
    "main_fusion": "Q_pred_cfs",
}
PERIODS = {
    "train_common_2010_2015": (2010, 2015),
    "selection_2016_2018": (2016, 2018),
    "development_holdout_2019_2022": (2019, 2022),
    "full_common_2010_2022": (2010, 2022),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    return float(1.0 - np.sum((pred - obs) ** 2) / denom) if denom > 0 else np.nan


def kge(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    if len(obs) < 2 or np.std(obs) <= 0 or np.mean(obs) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = float(np.corrcoef(obs, pred)[0, 1]) if np.std(pred) > 0 else 0.0
    beta = float(np.mean(pred) / np.mean(obs))
    cv_obs = float(np.std(obs) / np.mean(obs))
    cv_pred = float(np.std(pred) / np.mean(pred)) if np.mean(pred) != 0 else np.nan
    gamma = float(cv_pred / cv_obs) if cv_obs > 0 else np.nan
    if not all(np.isfinite(v) for v in (r, beta, gamma)):
        return np.nan, r, beta, gamma
    return (
        float(1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)),
        r,
        beta,
        gamma,
    )


def pct_bias(pred: np.ndarray, obs: np.ndarray) -> float:
    denom = float(np.sum(obs))
    return float(100.0 * (np.sum(pred) - denom) / denom) if denom != 0 else np.nan


def month_number(year: int, month: int) -> int:
    return int(year) * 12 + int(month)


def extract_events(values: np.ndarray, months: np.ndarray, quantile: float = 0.75) -> list[tuple[int, float]]:
    if len(values) < 3:
        return []
    threshold = float(np.quantile(values, quantile))
    candidates: list[tuple[int, float]] = []
    for i, value in enumerate(values):
        left = values[i - 1] if i > 0 else -np.inf
        right = values[i + 1] if i + 1 < len(values) else -np.inf
        if value >= threshold and value >= left and value >= right:
            candidates.append((int(months[i]), float(value)))
    collapsed: list[tuple[int, float]] = []
    for event in candidates:
        if collapsed and event[0] - collapsed[-1][0] < 2:
            if event[1] > collapsed[-1][1]:
                collapsed[-1] = event
        else:
            collapsed.append(event)
    return collapsed


def event_metrics(obs: np.ndarray, pred: np.ndarray, months: np.ndarray, window: int) -> dict[str, float]:
    observed = extract_events(obs, months)
    simulated = extract_events(pred, months)
    candidates: list[tuple[int, int, int]] = []
    for oi, (om, _) in enumerate(observed):
        for si, (sm, _) in enumerate(simulated):
            lag = sm - om
            if abs(lag) <= window:
                candidates.append((abs(lag), oi, si))
    used_o: set[int] = set()
    used_s: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, oi, si in sorted(candidates):
        if oi in used_o or si in used_s:
            continue
        used_o.add(oi)
        used_s.add(si)
        pairs.append((oi, si))
    tp = len(pairs)
    fp = len(simulated) - tp
    fn = len(observed) - tp
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    lags = np.array(
        [simulated[si][0] - observed[oi][0] for oi, si in pairs], dtype=float
    )
    amplitude = np.array(
        [
            np.log((simulated[si][1] + EPS) / (observed[oi][1] + EPS))
            for oi, si in pairs
        ],
        dtype=float,
    )
    return {
        f"event_count_obs_w{window}": float(len(observed)),
        f"event_count_pred_w{window}": float(len(simulated)),
        f"event_TP_w{window}": float(tp),
        f"event_precision_w{window}": float(precision),
        f"event_recall_w{window}": float(recall),
        f"event_F1_w{window}": float(f1),
        f"event_median_signed_lag_w{window}": float(np.median(lags)) if len(lags) else np.nan,
        f"event_median_abs_lag_w{window}": float(np.median(np.abs(lags))) if len(lags) else np.nan,
        f"event_median_abs_log_amplitude_error_w{window}": (
            float(np.median(np.abs(amplitude))) if len(amplitude) else np.nan
        ),
    }


def fdc_errors(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    order = np.argsort(-obs)
    log_error = np.log(pred[order] + EPS) - np.log(obs[order] + EPS)
    exceedance = (np.arange(len(obs), dtype=float) + 0.5) / len(obs) * 100.0
    out: dict[str, float] = {}
    for lo, hi in ((0, 10), (10, 30), (30, 70), (70, 90), (90, 100)):
        mask = (exceedance >= lo) & (exceedance < hi)
        out[f"FDC_logRMSE_exceedance_{lo}_{hi}"] = (
            float(np.sqrt(np.mean(log_error[mask] ** 2))) if mask.sum() >= 2 else np.nan
        )
    return out


def circular_month_difference(pred_month: int, obs_month: int) -> int:
    return int((pred_month - obs_month + 6) % 12 - 6)


def metric_block(part: pd.DataFrame, pred_col: str, ppt_threshold: float) -> tuple[dict[str, float], list[dict[str, float]]]:
    frame = part[["year", "month", "Q_obsv_cfs", pred_col, "PPT"]].dropna().copy()
    frame = frame[(frame["Q_obsv_cfs"] > 0) & (frame[pred_col] > 0)].sort_values(
        ["year", "month"]
    )
    if frame.empty:
        return {}, []
    obs = frame["Q_obsv_cfs"].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    log_obs = np.log(obs + EPS)
    log_pred = np.log(pred + EPS)
    kg, corr, beta, gamma = kge(obs, pred)
    q10, q25, q75, q90 = np.quantile(obs, [0.10, 0.25, 0.75, 0.90])
    low = obs <= q25
    high = obs >= q75
    very_low = obs <= q10
    very_high = obs >= q90
    months = np.array(
        [month_number(y, m) for y, m in zip(frame["year"], frame["month"])],
        dtype=int,
    )

    annual_rows: list[dict[str, float]] = []
    for year, year_part in frame.groupby("year"):
        annual_rows.append(
            {
                "year": int(year),
                "observed_volume_sum_cfs_month": float(year_part["Q_obsv_cfs"].sum()),
                "predicted_volume_sum_cfs_month": float(year_part[pred_col].sum()),
                "annual_volume_ratio": float(
                    year_part[pred_col].sum() / year_part["Q_obsv_cfs"].sum()
                ),
                "annual_volume_abs_error_pct": float(
                    100.0
                    * abs(
                        year_part[pred_col].sum() / year_part["Q_obsv_cfs"].sum()
                        - 1.0
                    )
                ),
            }
        )

    obs_monthly = frame.groupby("month")["Q_obsv_cfs"].mean()
    pred_monthly = frame.groupby("month")[pred_col].mean()
    obs_peak_month = int(obs_monthly.idxmax())
    pred_peak_month = int(pred_monthly.idxmax())

    prev_obs = frame["Q_obsv_cfs"].shift(1)
    contiguous = (
        frame["year"] * 12
        + frame["month"]
        - (frame["year"].shift(1) * 12 + frame["month"].shift(1))
    ).eq(1)
    recession = (
        contiguous
        & frame["Q_obsv_cfs"].lt(prev_obs)
        & frame["PPT"].le(ppt_threshold)
        & prev_obs.gt(0)
    )
    obs_recession = np.log(
        frame.loc[recession, "Q_obsv_cfs"].to_numpy(dtype=float)
        / prev_obs.loc[recession].to_numpy(dtype=float)
    )
    prev_pred = frame[pred_col].shift(1)
    pred_recession = np.log(
        frame.loc[recession, pred_col].to_numpy(dtype=float)
        / prev_pred.loc[recession].to_numpy(dtype=float)
    )

    metrics = {
        "n_months": float(len(frame)),
        "NSE_raw": nse(obs, pred),
        "NSElog": nse(log_obs, log_pred),
        "KGE": kg,
        "KGE_r": corr,
        "KGE_beta": beta,
        "KGE_gamma": gamma,
        "PBIAS_pct": pct_bias(pred, obs),
        "abs_PBIAS_pct": abs(pct_bias(pred, obs)),
        "RMSElog": float(np.sqrt(np.mean((log_pred - log_obs) ** 2))),
        "MAElog": float(np.mean(np.abs(log_pred - log_obs))),
        "low_flow_volume_bias_pct_Q25": pct_bias(pred[low], obs[low]),
        "high_flow_volume_bias_pct_Q75": pct_bias(pred[high], obs[high]),
        "very_low_flow_volume_bias_pct_Q10": pct_bias(pred[very_low], obs[very_low]),
        "very_high_flow_volume_bias_pct_Q90": pct_bias(pred[very_high], obs[very_high]),
        "low_flow_RMSElog_Q25": float(
            np.sqrt(np.mean((log_pred[low] - log_obs[low]) ** 2))
        ),
        "high_flow_normalized_RMSE_Q75": float(
            np.sqrt(np.mean((pred[high] - obs[high]) ** 2)) / (np.mean(obs) + EPS)
        ),
        "annual_ROCE_pct": float(
            np.mean([row["annual_volume_abs_error_pct"] for row in annual_rows])
        ),
        "seasonal_peak_obs_month": float(obs_peak_month),
        "seasonal_peak_pred_month": float(pred_peak_month),
        "seasonal_peak_phase_difference_months": float(
            circular_month_difference(pred_peak_month, obs_peak_month)
        ),
        "recession_pair_count": float(len(obs_recession)),
        "recession_median_log_ratio_obs": (
            float(np.median(obs_recession)) if len(obs_recession) else np.nan
        ),
        "recession_median_log_ratio_pred": (
            float(np.median(pred_recession)) if len(pred_recession) else np.nan
        ),
        "recession_median_log_ratio_error": (
            float(np.median(pred_recession) - np.median(obs_recession))
            if len(obs_recession)
            else np.nan
        ),
    }
    metrics.update(fdc_errors(obs, pred))
    metrics.update(event_metrics(obs, pred, months, window=1))
    metrics.update(event_metrics(obs, pred, months, window=2))
    return metrics, annual_rows


def deseason_cross_correlation(
    station: pd.DataFrame, pred_col: str, start: int, end: int
) -> tuple[float, float]:
    training = station[station["year"].between(2010, 2015)].copy()
    target = station[station["year"].between(start, end)].copy()
    if training.empty or len(target) < 12:
        return np.nan, np.nan
    obs_clim = training.assign(log_obs=np.log(training["Q_obsv_cfs"] + EPS)).groupby(
        "month"
    )["log_obs"].mean()
    pred_clim = training.assign(log_pred=np.log(training[pred_col] + EPS)).groupby(
        "month"
    )["log_pred"].mean()
    target = target.sort_values(["year", "month"]).copy()
    obs_anom = np.log(target["Q_obsv_cfs"].to_numpy(dtype=float) + EPS) - target[
        "month"
    ].map(obs_clim).to_numpy(dtype=float)
    pred_anom = np.log(target[pred_col].to_numpy(dtype=float) + EPS) - target[
        "month"
    ].map(pred_clim).to_numpy(dtype=float)
    best_corr = -np.inf
    best_lag = np.nan
    for lag in range(-3, 4):
        if lag < 0:
            x, y = obs_anom[-lag:], pred_anom[:lag]
        elif lag > 0:
            x, y = obs_anom[:-lag], pred_anom[lag:]
        else:
            x, y = obs_anom, pred_anom
        if len(x) < 6 or np.std(x) <= 0 or np.std(y) <= 0:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_lag = float(lag)
    return (
        float(best_corr) if np.isfinite(best_corr) else np.nan,
        float(best_lag),
    )


def branch_complementarity(frame: pd.DataFrame, period: str, start: int, end: int) -> pd.DataFrame:
    rows = []
    subset = frame[frame["year"].between(start, end)].copy()
    for site, part in subset.groupby("q_site"):
        part = part.dropna(
            subset=["Q_obsv_cfs", "Q72_pred_cfs", "Q78_mass_cfs", "Q_pred_cfs"]
        )
        if part.empty:
            continue
        obs = np.log(part["Q_obsv_cfs"].to_numpy(dtype=float) + EPS)
        e72 = np.log(part["Q72_pred_cfs"].to_numpy(dtype=float) + EPS) - obs
        e78 = np.log(part["Q78_mass_cfs"].to_numpy(dtype=float) + EPS) - obs
        emain = np.log(part["Q_pred_cfs"].to_numpy(dtype=float) + EPS) - obs
        residual_corr = (
            float(np.corrcoef(e72, e78)[0, 1])
            if np.std(e72) > 0 and np.std(e78) > 0
            else np.nan
        )
        best_branch_mae = min(float(np.mean(np.abs(e72))), float(np.mean(np.abs(e78))))
        rows.append(
            {
                "period": period,
                "q_site": site,
                "reach_id": int(part["reach_id"].iloc[0]),
                "reach_class": part["reach_class"].iloc[0],
                "n_months": int(len(part)),
                "q72_q78_log_residual_correlation": residual_corr,
                "q78_win_fraction_abs_log_error": float(
                    np.mean(np.abs(e78) < np.abs(e72))
                ),
                "q72_MAElog": float(np.mean(np.abs(e72))),
                "q78_MAElog": float(np.mean(np.abs(e78))),
                "main_MAElog": float(np.mean(np.abs(emain))),
                "main_gain_vs_best_branch_MAElog": float(
                    best_branch_mae - np.mean(np.abs(emain))
                ),
                "monthly_oracle_MAElog": float(
                    np.mean(np.minimum(np.abs(e72), np.abs(e78)))
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_primary(row: pd.Series) -> str:
    if row["reservoir_relation"] != "not_reservoir_related":
        return "reservoir_deferred"
    if abs(row["PBIAS_pct"]) > 100 or row["NSElog"] < -1:
        return "data_topology_audit"
    if row["NSElog"] < 0.65 and abs(row["PBIAS_pct"]) > 25:
        return "shape_and_volume"
    if row["low_flow_volume_bias_pct_Q25"] > 30:
        return "low_flow_overprediction"
    if (
        row["high_flow_volume_bias_pct_Q75"] < -30
        or row["PBIAS_pct"] < -25
    ):
        return "high_flow_or_volume_underprediction"
    if (
        row.get("event_F1_w1", np.nan) < 0.5
        or abs(row.get("event_median_signed_lag_w1", 0)) >= 1
    ):
        return "peak_timing"
    if row["NSElog"] < 0.65:
        return "general_shape_skill"
    return "good"


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(
        MAIN / "reach_class_selected_predictions_long.csv", encoding="utf-8-sig"
    )
    panel = pd.read_parquet(RUN / "inputs" / "indata.parquet")
    climate = panel[["comid", "year", "month", "PPT"]].drop_duplicates(
        ["comid", "year", "month"]
    )
    climate["reach_id"] = climate["comid"].astype(float).round().astype(int)
    predictions = predictions.merge(
        climate[["reach_id", "year", "month", "PPT"]],
        on=["reach_id", "year", "month"],
        how="left",
    )
    labels = pd.read_csv(MAIN / "station_diagnostic_labels.csv", encoding="utf-8-sig")
    relation = labels[
        ["q_site", "reservoir_relation", "nearest_upstream_reservoir_name"]
    ].drop_duplicates("q_site")

    metric_rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []
    for site, station in predictions.groupby("q_site"):
        training_ppt = station.loc[station["year"].between(2010, 2015), "PPT"]
        ppt_threshold = float(training_ppt.median()) if not training_ppt.empty else np.inf
        for period, (start, end) in PERIODS.items():
            part = station[station["year"].between(start, end)].copy()
            if part.empty:
                continue
            for branch, pred_col in BRANCHES.items():
                metrics, years = metric_block(part, pred_col, ppt_threshold)
                if not metrics:
                    continue
                corr, lag = deseason_cross_correlation(station, pred_col, start, end)
                metrics["deseason_max_cross_correlation"] = corr
                metrics["deseason_best_lag_pred_minus_obs_months"] = lag
                metric_rows.append(
                    {
                        "q_site": site,
                        "reach_id": int(part["reach_id"].iloc[0]),
                        "reach_class": part["reach_class"].iloc[0],
                        "period": period,
                        "branch": branch,
                        **metrics,
                    }
                )
                annual_rows.extend(
                    {
                        "q_site": site,
                        "reach_id": int(part["reach_id"].iloc[0]),
                        "period": period,
                        "branch": branch,
                        **row,
                    }
                    for row in years
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        OUT / "station_branch_period_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(annual_rows).to_csv(
        OUT / "annual_volume_metrics.csv", index=False, encoding="utf-8-sig"
    )

    complement = pd.concat(
        [
            branch_complementarity(predictions, period, start, end)
            for period, (start, end) in PERIODS.items()
        ],
        ignore_index=True,
    )
    complement.to_csv(
        OUT / "branch_complementarity.csv", index=False, encoding="utf-8-sig"
    )

    evaluation = metrics[
        metrics["period"].eq("development_holdout_2019_2022")
        & metrics["branch"].eq("main_fusion")
    ].merge(relation, on="q_site", how="left")
    evaluation["good"] = (
        evaluation["n_months"].ge(24)
        & evaluation["NSElog"].ge(0.65)
        & evaluation["KGE"].ge(0.50)
        & evaluation["abs_PBIAS_pct"].le(25.0)
    )
    evaluation["protected_station"] = evaluation["q_site"].eq("石角站")
    evaluation["primary_failure_group"] = evaluation.apply(classify_primary, axis=1)
    evaluation["issue_low_flow_overprediction"] = (
        evaluation["low_flow_volume_bias_pct_Q25"] > 30
    )
    evaluation["issue_high_flow_underprediction"] = (
        evaluation["high_flow_volume_bias_pct_Q75"] < -30
    )
    evaluation["issue_volume_bias"] = evaluation["abs_PBIAS_pct"] > 25
    evaluation["issue_peak_timing"] = (
        evaluation["event_F1_w1"].fillna(0) < 0.5
    ) | (evaluation["event_median_signed_lag_w1"].abs() >= 1)
    evaluation.to_csv(
        OUT / "frozen_station_manifest.csv", index=False, encoding="utf-8-sig"
    )

    group_summary = (
        evaluation.groupby("primary_failure_group", as_index=False)
        .agg(
            stations=("q_site", "size"),
            good=("good", "sum"),
            median_NSElog=("NSElog", "median"),
            median_KGE=("KGE", "median"),
            median_abs_PBIAS=("abs_PBIAS_pct", "median"),
            median_low_flow_bias=("low_flow_volume_bias_pct_Q25", "median"),
            median_high_flow_bias=("high_flow_volume_bias_pct_Q75", "median"),
            median_event_F1=("event_F1_w1", "median"),
        )
        .sort_values(["stations", "primary_failure_group"], ascending=[False, True])
    )
    group_summary.to_csv(
        OUT / "frozen_failure_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    population_rows = []
    for population, part in {
        "all_116": evaluation,
        "nonreservoir_target": evaluation[
            evaluation["reservoir_relation"].eq("not_reservoir_related")
        ],
        "reservoir_report_only": evaluation[
            ~evaluation["reservoir_relation"].eq("not_reservoir_related")
        ],
    }.items():
        population_rows.append(
            {
                "population": population,
                "stations": int(len(part)),
                "good": int(part["good"].sum()),
                "bad": int((~part["good"]).sum()),
                "median_NSElog": float(part["NSElog"].median()),
                "median_KGE": float(part["KGE"].median()),
                "median_abs_PBIAS": float(part["abs_PBIAS_pct"].median()),
                "abs_PBIAS_gt50": int((part["abs_PBIAS_pct"] > 50).sum()),
                "abs_PBIAS_gt100": int((part["abs_PBIAS_pct"] > 100).sum()),
            }
        )
    population = pd.DataFrame(population_rows)
    population.to_csv(
        OUT / "frozen_population_summary.csv", index=False, encoding="utf-8-sig"
    )

    q72_predictions = pd.read_csv(
        RUN
        / "reports"
        / "intermediate"
        / "base_regression"
        / "smearing_predictions_long.csv",
        encoding="utf-8-sig",
    )
    q72_predictions = q72_predictions[
        q72_predictions["variant"].eq("median_exp_eta")
    ]
    coverage_rows = []
    for year in range(2006, 2023):
        obs_count = int(
            panel.loc[
                panel["year"].eq(year) & panel["Q_obsv_cfs"].notna(),
                ["Q_obsv_cfs"],
            ].shape[0]
        )
        coverage_rows.append(
            {
                "year": year,
                "indata_observed_station_months": obs_count,
                "q72_prediction_rows": int(q72_predictions["year"].eq(year).sum()),
                "q78_and_fusion_prediction_rows": int(
                    predictions["year"].eq(year).sum()
                ),
                "q78_currently_includes_year": bool(year >= 2010),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(
        OUT / "q78_coverage_audit.csv", index=False, encoding="utf-8-sig"
    )

    input_hash = sha256(RUN / "inputs" / "indata.parquet")
    (RUN / "inputs_manifest" / "indata_sha256.txt").write_text(
        f"{input_hash}  {RUN / 'inputs' / 'indata.parquet'}\n", encoding="utf-8"
    )
    baseline = {
        "run_id": RUN.name,
        "indata_sha256": input_hash,
        "population": population.to_dict(orient="records"),
        "q78_common_prediction_start_year": int(predictions["year"].min()),
        "q78_common_prediction_end_year": int(predictions["year"].max()),
        "q72_prediction_start_year": int(q72_predictions["year"].min()),
        "q72_prediction_end_year": int(q72_predictions["year"].max()),
    }
    (OUT / "baseline_metrics.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        f"# {RUN.name} Stage-0 Frozen Baseline Diagnostics",
        "",
        "## Frozen populations",
        "",
        markdown_table(population),
        "",
        "## Mutually exclusive primary failure groups",
        "",
        markdown_table(group_summary),
        "",
        "## Data-coverage finding",
        "",
        f"- Q72 predictions cover {int(q72_predictions['year'].min())}-{int(q72_predictions['year'].max())}.",
        f"- Q78/fusion common predictions cover {int(predictions['year'].min())}-{int(predictions['year'].max())}.",
        "- Therefore the current Q78 branch does not use 2006-2009 station observations; this is recorded as the first conditional stage-1 coverage experiment, not changed in this baseline.",
        "",
        "## Boundaries",
        "",
        "- No model structure, model hyperparameter, fusion rule, or station exclusion was changed.",
        "- Reservoir-related stations are reported but are not a target group in this series.",
        "- Peak diagnostics use one-to-one local-event matching; the old full-period maximum-peak lag is not used for routing decisions.",
    ]
    (OUT / "stage0_diagnostic_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
