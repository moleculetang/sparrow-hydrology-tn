from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN_DIR = ROOT / "5_Test" / "20260603_11"
REPORT_DIR = RUN_DIR / "reports"
LOG_PATH = ROOT / "5_Test" / "20260603.log"
EPS = 1.0e-6
VALIDATION_START_PERIOD = 64
VALIDATION_END_PERIOD = 68


def hydrologic_metrics(part: pd.DataFrame) -> dict[str, float | int]:
    y = pd.to_numeric(part["actual"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(part["predict"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {"n": 0}
    err = p - y
    ly = np.log(y)
    lp = np.log(np.maximum(p, EPS))
    sse = float(np.sum(err**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    log_sse = float(np.sum((lp - ly) ** 2))
    log_sst = float(np.sum((ly - ly.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    obs_sd = float(np.std(y, ddof=0))
    pred_sd = float(np.std(p, ddof=0))
    corr = float(np.corrcoef(y, p)[0, 1]) if len(y) >= 2 and obs_sd > 0 and pred_sd > 0 else np.nan
    alpha = float(pred_sd / obs_sd) if obs_sd > 0 else np.nan
    beta = float(np.mean(p) / np.mean(y)) if np.mean(y) != 0 else np.nan
    kge = np.nan
    if np.isfinite(corr) and np.isfinite(alpha) and np.isfinite(beta):
        kge = float(1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    pbias = float(100 * np.sum(err) / np.sum(y)) if np.sum(y) != 0 else np.nan
    abs_pct = np.abs(err) / np.maximum(y, EPS) * 100
    peak_obs_i = int(np.argmax(y))
    peak_pred_i = int(np.argmax(p))
    return {
        "n": int(len(y)),
        "NSE_raw": float(1 - sse / sst) if sst > 0 else np.nan,
        "NSE_log": float(1 - log_sse / log_sst) if log_sst > 0 else np.nan,
        "KGE_2012": kge,
        "KGE_r": corr,
        "KGE_alpha_variability": alpha,
        "KGE_beta_bias_ratio": beta,
        "RSR": float(rmse / obs_sd) if obs_sd > 0 else np.nan,
        "PBIAS_pct": pbias,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_pct": float(np.mean(abs_pct)),
        "median_abs_pct_error": float(np.median(abs_pct)),
        "max_abs_pct_error": float(np.max(abs_pct)),
        "peak_obs": float(y[peak_obs_i]),
        "peak_pred_at_obs_peak": float(p[peak_obs_i]),
        "peak_pct_error_at_obs_peak": float(100 * (p[peak_obs_i] - y[peak_obs_i]) / y[peak_obs_i]),
        "pred_peak_period_offset": int(peak_pred_i - peak_obs_i),
        "mean_actual": float(np.mean(y)),
        "mean_pred": float(np.mean(p)),
    }


def markdown_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    part = df.head(n)[cols].copy()
    for col in part.columns:
        if pd.api.types.is_float_dtype(part[col]):
            part[col] = part[col].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
        else:
            part[col] = part[col].map(lambda v: "" if pd.isna(v) else str(v))
    header = "| " + " | ".join(part.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(part.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in part.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def nse_class(v: float) -> str:
    if not np.isfinite(v):
        return "NA"
    if v >= 0.75:
        return "very_good"
    if v >= 0.65:
        return "good"
    if v >= 0.50:
        return "satisfactory"
    return "unsatisfactory"


def kge_class(v: float) -> str:
    if not np.isfinite(v):
        return "NA"
    if v >= 0.75:
        return "very_good"
    if v >= 0.50:
        return "acceptable"
    return "poor"


def pbias_class(v: float) -> str:
    av = abs(v)
    if not np.isfinite(av):
        return "NA"
    if av <= 10:
        return "very_good"
    if av <= 15:
        return "good"
    if av <= 25:
        return "satisfactory"
    return "unsatisfactory"


def parse_downstream(value: object) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    out = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(float(token)))
        except ValueError:
            pass
    return out


def all_downstream_reaches(edges: pd.DataFrame) -> dict[int, set[int]]:
    immediate = {int(r.reach_id): set(parse_downstream(r.downstream_reach)) for r in edges.itertuples()}
    all_down = {}
    for rid in immediate:
        seen = set()
        q = deque(immediate.get(rid, set()))
        while q:
            cur = q.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            for nxt in immediate.get(cur, set()):
                if nxt not in seen:
                    q.append(nxt)
        all_down[rid] = seen
    return all_down


def quarter_problem_string(part: pd.DataFrame) -> str:
    rows = []
    for row in part.sort_values("period").itertuples(index=False):
        pct = float(row.pct_error)
        if abs(pct) >= 35 or abs(float(row.log_error)) >= 0.35:
            direction = "over" if pct > 0 else "under"
            rows.append(f"{int(row.year)}Q{int(row.quarter)}:{direction},{pct:+.0f}%")
    if rows:
        return "; ".join(rows)
    worst = part.iloc[part["abs_pct_error"].to_numpy().argmax()]
    direction = "over" if worst["pct_error"] > 0 else "under"
    return f"{int(worst['year'])}Q{int(worst['quarter'])}:{direction},{worst['pct_error']:+.0f}%"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    resids = pd.read_csv(RUN_DIR / "outputs" / "resids.csv", encoding="utf-8-sig")
    indata = pd.read_parquet(RUN_DIR / "inputs" / "indata.parquet")
    obs_info = indata[indata["Q_obsv_cfs"].notna()][["comid", "year", "quarter", "period", "q_site"]].copy()
    obs_info["comid"] = pd.to_numeric(obs_info["comid"], errors="coerce")
    resids = resids.drop(columns=["q_site"], errors="ignore").merge(obs_info, on=["comid", "year", "quarter", "period"], how="left")
    resids["q_site"] = resids["q_site"].fillna("UNKNOWN")
    resids["pct_error"] = 100 * (resids["predict"] - resids["actual"]) / resids["actual"].clip(lower=EPS)
    resids["abs_pct_error"] = resids["pct_error"].abs()
    resids["log_error"] = np.log(resids["predict"].clip(lower=EPS)) - np.log(resids["actual"].clip(lower=EPS))
    validation = resids[resids["period"].between(VALIDATION_START_PERIOD, VALIDATION_END_PERIOD)].copy()
    validation.to_csv(REPORT_DIR / "validation_resids_native_run.csv", index=False, encoding="utf-8-sig")

    rows = []
    for station, part in validation.groupby("q_site"):
        m = hydrologic_metrics(part)
        m["q_site"] = station
        rows.append(m)
    metrics = pd.DataFrame(rows)
    for col in ["NSE_raw", "NSE_log", "KGE_2012", "PBIAS_pct"]:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    metrics["NSE_raw_class"] = metrics["NSE_raw"].map(nse_class)
    metrics["NSE_log_class"] = metrics["NSE_log"].map(nse_class)
    metrics["KGE_class"] = metrics["KGE_2012"].map(kge_class)
    metrics["PBIAS_class"] = metrics["PBIAS_pct"].map(pbias_class)
    metrics["high_log_r2_station"] = metrics["NSE_log"] > 0.8
    metrics["problem_station"] = (
        (metrics["NSE_log"] <= 0.8)
        | (metrics["NSE_raw"] < 0.65)
        | (metrics["KGE_2012"] < 0.5)
        | (metrics["PBIAS_pct"].abs() > 25)
    )
    metrics = metrics.sort_values(["problem_station", "NSE_log", "KGE_2012"], ascending=[False, True, True])
    metrics.to_csv(REPORT_DIR / "hydrologic_metrics_by_station_validation.csv", index=False, encoding="utf-8-sig")
    high = metrics[metrics["high_log_r2_station"]].sort_values(["NSE_raw", "KGE_2012"], ascending=False)
    high.to_csv(REPORT_DIR / "hydrologic_metrics_high_log_r2_stations.csv", index=False, encoding="utf-8-sig")
    weak_high = high[(high["NSE_raw"] < 0.65) | (high["KGE_2012"] < 0.5) | (high["PBIAS_pct"].abs() > 25)].copy()
    weak_high.to_csv(REPORT_DIR / "high_log_r2_but_hydrologically_weak.csv", index=False, encoding="utf-8-sig")

    station_match = pd.read_csv(RUN_DIR / "reports" / "station_reach_match.csv", encoding="utf-8-sig")
    station_match = station_match[station_match["used"].astype(str).str.lower().isin(["true", "1"])]
    reach_by_station = station_match.drop_duplicates("station_name").set_index("station_name")["reach_id"].to_dict()
    edges = pd.read_csv(ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv", encoding="utf-8-sig")
    all_down = all_downstream_reaches(edges)
    stations_by_reach = defaultdict(list)
    for name, rid in reach_by_station.items():
        if pd.notna(rid):
            stations_by_reach[int(rid)].append(name)

    problem_set = set(metrics[metrics["problem_station"]]["q_site"])
    severe_set = set(metrics[(metrics["NSE_log"] < 0.5) | (metrics["NSE_raw"] < 0.0) | (metrics["PBIAS_pct"].abs() > 50)]["q_site"])
    metric_by_station = metrics.set_index("q_site").to_dict("index")
    problem_rows = []
    for station in sorted(problem_set):
        m = metric_by_station[station]
        part = validation[validation["q_site"] == station].copy()
        rid = reach_by_station.get(station, np.nan)
        downstream_stations: list[str] = []
        if pd.notna(rid):
            for dr in sorted(all_down.get(int(rid), set())):
                downstream_stations.extend(stations_by_reach.get(dr, []))
        downstream_stations = sorted(set(x for x in downstream_stations if x != station))
        poor_down = [x for x in downstream_stations if x in problem_set]
        severe_down = [x for x in downstream_stations if x in severe_set]
        mean_pct = float(part["pct_error"].mean()) if len(part) else np.nan
        pattern = "over" if mean_pct > 10 else "under" if mean_pct < -10 else "mixed/shape"
        priority = "A_upstream_cascade" if len(severe_down) or len(poor_down) >= 3 else "B_local_or_terminal"
        if len(downstream_stations) == 0:
            priority = "C_no_gauged_downstream"
        problem_rows.append(
            {
                "priority": priority,
                "station": station,
                "reach_id": int(rid) if pd.notna(rid) else np.nan,
                "validation_n": int(m["n"]),
                "NSE_raw": m["NSE_raw"],
                "NSE_log": m["NSE_log"],
                "KGE_2012": m["KGE_2012"],
                "PBIAS_pct": m["PBIAS_pct"],
                "mean_error_pattern": pattern,
                "problem_year_quarters": quarter_problem_string(part),
                "downstream_gauged_station_count": len(downstream_stations),
                "downstream_problem_station_count": len(poor_down),
                "downstream_severe_station_count": len(severe_down),
                "downstream_problem_stations": "; ".join(poor_down[:30]),
                "downstream_severe_stations": "; ".join(severe_down[:30]),
            }
        )
    problems = pd.DataFrame(problem_rows).sort_values(
        ["priority", "downstream_severe_station_count", "downstream_problem_station_count", "NSE_log"],
        ascending=[True, False, False, True],
    )
    problems.to_csv(REPORT_DIR / "problem_station_downstream_impact_table.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "group": "validation_all_stations",
                "stations": int(metrics["q_site"].nunique()),
                "rows": int(len(validation)),
                "median_NSE_raw": float(metrics["NSE_raw"].median()),
                "median_NSE_log": float(metrics["NSE_log"].median()),
                "median_KGE": float(metrics["KGE_2012"].median()),
                "median_abs_PBIAS_pct": float(metrics["PBIAS_pct"].abs().median()),
                "high_log_r2_gt_0p8": int((metrics["NSE_log"] > 0.8).sum()),
                "NSE_raw_gt_0p75": int((metrics["NSE_raw"] > 0.75).sum()),
                "KGE_gt_0p75": int((metrics["KGE_2012"] > 0.75).sum()),
                "abs_PBIAS_le_25pct": int((metrics["PBIAS_pct"].abs() <= 25).sum()),
                "problem_station_count": int(metrics["problem_station"].sum()),
                "cascade_priority_count": int((problems["priority"] == "A_upstream_cascade").sum()),
            }
        ]
    )
    summary.to_csv(REPORT_DIR / "validation_metric_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 20260603_11 split reservoir-body/downstream Q run and validation diagnostics",
        "",
        "This run extends 20260603_10 by splitting the reservoir mechanism into reservoir-body and near-downstream terms:",
        "`res_lag_self/res_decay_self` apply to named reservoir reaches, while `res_lag_down/res_decay_down` apply only to first and second downstream reaches.",
        f"`2021Q4-2022Q4` (`period` {VALIDATION_START_PERIOD}-{VALIDATION_END_PERIOD}).",
        "",
        "Important note: the native run was executed as a normal one-pass estimate+predict run. The",
        "validation-period diagnostics below describe that run output; they are not the same as the",
        "earlier custom strict train/cal/validation experimental chain.",
        "",
        "## Input And Run Counts",
        "",
        f"- input rows: {len(indata)}",
        f"- observed Q rows in input: {int(indata['Q_obsv_cfs'].notna().sum())}",
        f"- validation rows evaluated: {len(validation)}",
        f"- validation stations evaluated: {metrics['q_site'].nunique()}",
        "",
        "## Validation Summary",
        "",
        markdown_table(summary, list(summary.columns), 1),
        "",
        "## High log-R2 Stations",
        "",
        markdown_table(high, ["q_site", "n", "NSE_raw", "NSE_log", "KGE_2012", "RSR", "PBIAS_pct", "MAPE_pct", "max_abs_pct_error"], 20),
        "",
        "## High log-R2 But Hydrologically Weak",
        "",
        markdown_table(weak_high, ["q_site", "n", "NSE_raw", "NSE_log", "KGE_2012", "RSR", "PBIAS_pct", "MAPE_pct", "max_abs_pct_error", "NSE_raw_class", "KGE_class", "PBIAS_class"], 30),
        "",
        "## Problem Stations With Downstream Priority",
        "",
        markdown_table(problems, ["priority", "station", "reach_id", "NSE_raw", "NSE_log", "KGE_2012", "PBIAS_pct", "mean_error_pattern", "problem_year_quarters", "downstream_problem_station_count", "downstream_severe_station_count", "downstream_problem_stations"], 30),
        "",
        "## Files",
        "",
        "- `reports/hydrologic_metrics_by_station_validation.csv`",
        "- `reports/hydrologic_metrics_high_log_r2_stations.csv`",
        "- `reports/high_log_r2_but_hydrologically_weak.csv`",
        "- `reports/problem_station_downstream_impact_table.csv`",
        "- `reports/validation_resids_native_run.csv`",
    ]
    (RUN_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            "\n[2026-06-03] 20260603_11 split reservoir-body/downstream run used separate bounded lag/decay parameters for named reservoir reaches and first/second downstream reaches, "
            f"observed_rows={int(indata['Q_obsv_cfs'].notna().sum())}, validation_stations={metrics['q_site'].nunique()}, "
            f"high_log_r2={int((metrics['NSE_log'] > 0.8).sum())}, problem_stations={int(metrics['problem_station'].sum())}, "
            f"median_NSE_log={metrics['NSE_log'].median():.6f}, median_KGE={metrics['KGE_2012'].median():.6f}.\n"
        )


if __name__ == "__main__":
    main()
