"""Common-support TN metrics and synchronized paired calendar bootstrap.

Raw daily metrics use read-count weights. Month-centered metrics first demean
within each station/month using read counts, then give each month equal weight.
No epsilon is added to a zero observation variance. Bootstrap never regenerates
lags or events by concatenating blocks; repeated months have independent IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoveragePolicy:
    min_days: int = 2
    min_months: int = 1

    def __post_init__(self):
        if self.min_days < 2 or self.min_months < 1:
            raise ValueError("Coverage requires >=2 days and >=1 month")


def _validated(table):
    required = {"station_key", "date", "truth", "baseline", "candidate", "read_count"}
    if not required.issubset(table):
        raise ValueError(f"Missing paired columns: {sorted(required-set(table.columns))}")
    t = table.copy(deep=True)
    t["date"] = pd.to_datetime(t.date, errors="raise").dt.normalize()
    if t.date.isna().any() or t.station_key.isna().any():
        raise ValueError("Missing date or station identity")
    t["station_key"] = t.station_key.astype(str)
    if t.duplicated(["station_key", "date"]).any():
        raise ValueError("Paired table must contain unique station-days")
    counts = t.read_count.to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Read weights must be finite and positive")
    # Common finite support is formed once, before either model is evaluated.
    finite = np.isfinite(t[["truth", "baseline", "candidate"]].to_numpy(dtype=float)).all(axis=1)
    t = t.loc[finite].copy()
    t["month"] = t.date.dt.to_period("M").astype(str)
    return t


def weighted_metrics(truth, prediction, weights=None):
    y, p = np.asarray(truth, float), np.asarray(prediction, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if y.ndim != 1 or y.shape != p.shape or w.shape != y.shape:
        raise ValueError("Expected matching one-dimensional metric vectors")
    if not (np.isfinite(y).all() and np.isfinite(p).all() and np.isfinite(w).all()) or (w <= 0).any():
        raise ValueError("Metrics require finite values and positive weights")
    empty = {"nse": None, "rmse": None, "bias": None, "correlation": None,
             "amplitude_ratio": None, "observed_sd": None, "predicted_sd": None,
             "nse_status": "insufficient_support"}
    if len(y) < 2:
        return empty
    w = w/w.sum()
    ym, pm = float(w@y), float(w@p)
    vy = 0.0 if np.all(y == y[0]) else float(w@(y-ym)**2)
    vp = 0.0 if np.all(p == p[0]) else float(w@(p-pm)**2)
    error = float(w@(p-y)**2)
    return {"nse": float(1-error/vy) if vy > 0 else None,
            "nse_status": "defined" if vy > 0 else "zero_observed_variance",
            "rmse": float(np.sqrt(error)), "bias": float(w@(p-y)),
            "correlation": float(w@((y-ym)*(p-pm))/np.sqrt(vy*vp)) if vy > 0 and vp > 0 else None,
            "amplitude_ratio": float(np.sqrt(vp/vy)) if vy > 0 else None,
            "observed_sd": float(np.sqrt(vy)), "predicted_sd": float(np.sqrt(vp))}


def _station_metrics(frame, column, policy, month_col="month"):
    n, nm = len(frame), frame[month_col].nunique()
    if n < policy.min_days or nm < policy.min_months:
        out = weighted_metrics([], [])
        out["month_centered_nse"] = None
        out["month_centered_status"] = "insufficient_support"
        out["month_centered_rmse"] = None
        return out
    result = weighted_metrics(frame.truth, frame[column], frame.read_count)
    total_error, total_var = 0.0, 0.0
    for _, g in frame.groupby(month_col, sort=False):
        w = g.read_count.to_numpy(float)
        w = w / w.sum()
        y, p = g.truth.to_numpy(float), g[column].to_numpy(float)
        yc, pc = y-w@y, p-w@p
        if np.all(y == y[0]):
            yc = np.zeros_like(y)
        if np.all(p == p[0]):
            pc = np.zeros_like(p)
        total_error += float(w@(pc-yc)**2)
        total_var += float(w@yc**2)
    result.update(month_centered_nse=float(1-total_error/total_var) if total_var > 0 else None,
                  month_centered_status="defined" if total_var > 0 else "zero_observed_within_month_variance",
                  month_centered_rmse=float(np.sqrt(total_error/nm)),
                  month_centered_denominator=float(total_var/nm))
    return result


def _summarize(stations, metric):
    base, cand = "baseline_"+metric, "candidate_"+metric
    if stations.empty:
        valid = stations
    else:
        valid = stations[np.isfinite(pd.to_numeric(stations[base], errors="coerce")) &
                         np.isfinite(pd.to_numeric(stations[cand], errors="coerce"))]
    if valid.empty:
        return {"metric": metric, "n_common_stations": 0, "difference_of_medians": None,
                "median_paired_difference": None, "improvement_fraction": None,
                "baseline_median": None, "candidate_median": None}
    a, b = valid[base].to_numpy(float), valid[cand].to_numpy(float)
    diff = b-a
    higher_better = metric in ("nse", "month_centered_nse", "correlation")
    if metric == "bias":
        improve = np.abs(b) < np.abs(a)
        worsen = np.abs(b) > np.abs(a)
    elif metric == "amplitude_ratio":
        improve = np.abs(b-1) < np.abs(a-1)
        worsen = np.abs(b-1) > np.abs(a-1)
    else:
        improve = diff > 0 if higher_better else diff < 0
        worsen = diff < 0 if higher_better else diff > 0
    return {"metric": metric, "n_common_stations": len(valid),
            "baseline_median": float(np.median(a)), "candidate_median": float(np.median(b)),
            "difference_of_medians": float(np.median(b)-np.median(a)),
            "median_paired_difference": float(np.median(diff)),
            "improvement_fraction": float(np.mean(improve)),
            "improved_stations": valid.loc[improve, "station_key"].tolist(),
            "worsened_stations": valid.loc[worsen, "station_key"].tolist(),
            "unchanged_stations": valid.loc[~improve & ~worsen, "station_key"].tolist(),
            "not_improved_stations": valid.loc[~improve, "station_key"].tolist()}


def _evaluate_validated(t, policy, scope, month_col="month"):
    rows = []
    for station, g in t.groupby("station_key", sort=True):
        row = {"station_key": station, "scope": scope,
               "n_common_observations": len(g),
               "n_common_days": None if "monthly" in scope else len(g),
               "n_common_months": g[month_col].nunique()}
        for model in ("baseline", "candidate"):
            row.update({model+"_"+k: v for k,v in _station_metrics(g, model, policy, month_col).items()})
        for metric in ("nse", "month_centered_nse"):
            a, b = row["baseline_"+metric], row["candidate_"+metric]
            row["paired_"+metric+"_difference"] = b-a if a is not None and b is not None else None
        rows.append(row)
    station = pd.DataFrame(rows)
    summaries = [_summarize(station, key) for key in
                 ("nse", "month_centered_nse", "rmse", "bias", "correlation", "amplitude_ratio")]
    return station, pd.DataFrame(summaries)


def evaluate_pair(table, policy=CoveragePolicy(), scope="HF_daily"):
    """Each model is scored on exactly the same finite station-date support."""
    return _evaluate_validated(_validated(table), policy, scope)


def monthly_views(hf_table, original_monthly_table, min_hf_days=2):
    """Separate HF monthly means from original monthly reports, never pool them.

    The original monthly table uses the same paired columns; its date identifies
    the month, not a daily label. Original-month predictions must already have
    passed the actual monthly observation operator.
    """
    h = _validated(hf_table)
    records = []
    for (station, month), g in h.groupby(["station_key", "month"]):
        if len(g) < min_hf_days:
            continue
        w = g.read_count.to_numpy(float)
        w = w / w.sum()
        record = {"station_key": station, "date": pd.Timestamp(month+"-01"), "read_count": 1,
                  "n_hf_days": len(g), "total_read_count": float(g.read_count.sum())}
        record.update({k: float(w@g[k].to_numpy(float)) for k in ("truth", "baseline", "candidate")})
        records.append(record)
    columns = ["station_key", "date", "read_count", "truth", "baseline", "candidate"]
    hf_months = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
    m = _validated(original_monthly_table)
    if m.duplicated(["station_key", "month"]).any():
        raise ValueError("Original monthly report must be unique station-month")
    m["read_count"] = 1.0
    return {"HF_monthly": hf_months, "original_monthly": m}


def calendar_draws(months, n_bootstrap=1000, block_months=1, seed=1729):
    """Moving calendar blocks; gaps remain empty months, never joined in time."""
    if block_months not in (1, 2) or n_bootstrap < 1:
        raise ValueError("Use positive repetitions and registered 1/2-month blocks")
    if not months:
        raise ValueError("No calendar support")
    period = pd.PeriodIndex(sorted(set(months)), freq="M")
    calendar = pd.period_range(period.min(), period.max(), freq="M").astype(str).tolist()
    if len(calendar) < block_months:
        raise ValueError("Insufficient calendar length for requested block")
    rng = np.random.default_rng(seed)
    records = []
    for rep in range(n_bootstrap):
        drawn = 0
        block = 0
        while drawn < len(calendar):
            start = int(rng.integers(0, len(calendar)-block_months+1))
            for offset in range(block_months):
                if drawn >= len(calendar):
                    break
                records.append({"replicate": rep, "block_instance": block,
                                "sample_month_id": drawn, "source_month": calendar[start+offset],
                                "offset_in_block": offset, "block_months": block_months})
                drawn += 1
            block += 1
    return pd.DataFrame(records)


def paired_calendar_bootstrap(table, policy=CoveragePolicy(), n_bootstrap=1000,
                              block_months=1, seed=1729):
    """Frozen-model intervals; not parameter uncertainty or future validation.

    Sufficient monthly moments are precomputed on the original series. Every
    station receives the same sampled months; a repeated month adds a separate
    contribution. No new lag/event relationship is constructed at block joins.
    """
    t = _validated(table)
    ledger = calendar_draws(t.month.unique().tolist(), n_bootstrap, block_months, seed)
    _, base_summary = _evaluate_validated(t, policy, "HF_daily")
    stations = sorted(t.station_key.unique())
    calendar = sorted(ledger.source_month.unique())
    # Ensure months never sampled due to tiny replication counts remain indexed.
    calendar = pd.period_range(min(t.month), max(t.month), freq="M").astype(str).tolist()
    month_idx = {m:i for i,m in enumerate(calendar)}
    station_idx = {s:i for i,s in enumerate(stations)}
    # Per month/station: n, read sum, sum_y, sum_y2, [sum_p,sum_p2,sum_yp,sse,centered_sse]*2, centered_yvar.
    moments = np.zeros((len(calendar), len(stations), 15), float)
    present = np.zeros((len(calendar), len(stations)), bool)
    for (station, month), g in t.groupby(["station_key", "month"]):
        mi, si = month_idx[month], station_idx[station]
        w = g.read_count.to_numpy(float); y = g.truth.to_numpy(float)
        mass = w.sum(); yw = float(w@y); ym = yw/mass
        row = [len(g), mass, yw, float(w@y**2)]
        for key in ("baseline", "candidate"):
            p = g[key].to_numpy(float); pm = float(w@p)/mass
            row.extend([float(w@p), float(w@p**2), float(w@(y*p)),
                        float(w@(p-y)**2), float(w@((p-pm)-(y-ym))**2)/mass])
        row.append(0.0 if np.all(y==y[0]) else float(w@(y-ym)**2)/mass)
        moments[mi,si] = row
        present[mi,si] = True
    records = []
    for replicate, draw in ledger.groupby("replicate", sort=True):
        idx = np.array([month_idx[m] for m in draw.source_month])
        sums = moments[idx].sum(axis=0)
        month_n = present[idx].sum(axis=0)
        eligible = (sums[:,0] >= policy.min_days) & (month_n >= policy.min_months)
        read_sum = sums[:,1]
        # Exact zero variance is undefined; no artificial stabilizing variance.
        with np.errstate(divide="ignore", invalid="ignore"):
            variance_sum = sums[:,3]-sums[:,2]**2/read_sum
        # Cancellation can occur only in raw moment aggregation: direct stable fallback.
        for si in np.where(eligible & (variance_sum <= 1e-10*np.maximum(1.0, np.abs(sums[:,3]))))[0]:
            w_parts, y_parts = [], []
            for m in draw.source_month:
                g = t[(t.station_key==stations[si]) & (t.month==m)]
                w_parts.extend(g.read_count.tolist()); y_parts.extend(g.truth.tolist())
            wv,yv=np.asarray(w_parts),np.asarray(y_parts)
            variance_sum[si]=(0.0 if np.all(yv==yv[0]) else float(wv@(yv-(wv@yv)/wv.sum())**2))
        for metric, den, col_a, col_b in (("nse", variance_sum, 7, 12),
                                         ("month_centered_nse", sums[:,14], 8, 13)):
            valid = eligible & (den > 0)
            a = 1-sums[valid,col_a]/den[valid]
            b = 1-sums[valid,col_b]/den[valid]
            records.append({"replicate": int(replicate), "metric": metric,
                            "n_common_stations": int(valid.sum()),
                            "difference_of_medians": float(np.median(b)-np.median(a)) if len(a) else np.nan,
                            "median_paired_difference": float(np.median(b-a)) if len(a) else np.nan,
                            "improvement_fraction": float(np.mean(b>a)) if len(a) else np.nan})
    draws = pd.DataFrame(records)
    intervals=[]
    for metric,g in draws.groupby("metric"):
        for statistic in ("difference_of_medians", "median_paired_difference", "improvement_fraction"):
            values=g[statistic].dropna().to_numpy()
            q=np.quantile(values,[.025,.975]) if len(values) else [np.nan,np.nan]
            intervals.append({"metric":metric,"statistic":statistic,"lower_95":q[0],"upper_95":q[1],
                              "n_defined_replicates":len(values),"n_requested_replicates":n_bootstrap,
                              "block_months":block_months,"seed":seed,
                              "interpretation":"conditional_on_frozen_models_and_observed_calendar"})
    return {"summary": base_summary, "replicates": draws, "intervals": pd.DataFrame(intervals), "ledger": ledger}


def evaluate_events(table, events, evaluation_start=None):
    """Evaluate pre-frozen events on paired dates; no event or lag discovery."""
    t = _validated(table)
    required={"station_key","event_id","start","end","background_start","background_end"}
    if not required.issubset(events):
        raise ValueError("Frozen event support columns missing")
    records=[]
    for event in events.to_dict("records"):
        dates={k:pd.Timestamp(event[k]).normalize() for k in ("start","end","background_start","background_end")}
        if dates["start"]>dates["end"] or dates["background_start"]>dates["background_end"]:
            raise ValueError("Invalid frozen event dates")
        if evaluation_start is not None and dates["background_start"] < pd.Timestamp(evaluation_start):
            raise ValueError("Event background crosses training/evaluation boundary")
        if dates["background_end"] >= dates["start"]:
            raise ValueError("Background must precede event")
        g=t[t.station_key==str(event["station_key"])]
        win=g[g.date.between(dates["start"],dates["end"])].sort_values("date")
        bg=g[g.date.between(dates["background_start"],dates["background_end"])].sort_values("date")
        row={"station_key":str(event["station_key"]),"event_id":event["event_id"],
             "n_common_event_days":len(win),"n_common_background_days":len(bg),
             "status":"defined" if len(win)>=2 and len(bg)>=1 else "insufficient_support"}
        if row["status"] != "defined":
            row.update(baseline_nse=None,candidate_nse=None)
            records.append(row); continue
        ybg=float(np.average(bg.truth,weights=bg.read_count))
        ypeak=float(win.truth.max()); ydate=win.loc[win.truth.idxmax(),"date"]
        row.update(observed_amplitude=ypeak-ybg, observed_peak_date=str(ydate.date()),
                   observed_background=ybg, peak_tie_rule="first_common_date")
        for model in ("baseline","candidate"):
            result=_station_metrics(win,model,CoveragePolicy())
            row.update({model+"_"+k:v for k,v in result.items()})
            pbg=float(np.average(bg[model],weights=bg.read_count))
            ppeak=float(win[model].max()); pdate=win.loc[win[model].idxmax(),"date"]
            row.update({model+"_amplitude_error":ppeak-pbg-(ypeak-ybg),
                        model+"_peak_error":ppeak-ypeak, model+"_background_error":pbg-ybg,
                        model+"_peak_date":str(pdate.date()),model+"_peak_date_offset_days":int((pdate-ydate).days)})
        records.append(row)
    return pd.DataFrame(records)
