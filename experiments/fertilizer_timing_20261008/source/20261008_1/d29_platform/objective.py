"""Fold-local observation objective; no oracle, evaluation labels or file I/O.

Prediction vectors follow ``TrainingObjective.rows``. HF predictions are sampled
day values. Monthly predictions must be produced by the frozen month observation
operator upstream; this module never expands monthly labels into daily truth.
Both D terms are HALF squared errors and J = .5 D_level + .5 D_anomaly + R.
The analytic gradient returned here is dJ/d(prediction), for chaining to either
land model's full-history adjoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingPolicy:
    """Explicit fold-only scale floors, in mg N/L; never evaluation variance."""
    level_std_floor: float
    anomaly_std_floor: float
    min_hf_days_per_month: int = 2

    def __post_init__(self):
        if not (np.isfinite(self.level_std_floor) and self.level_std_floor > 0
                and np.isfinite(self.anomaly_std_floor) and self.anomaly_std_floor > 0):
            raise ValueError("Scale floors must be explicit, finite and positive")
        if self.min_hf_days_per_month < 2:
            raise ValueError("Anomaly support needs at least two HF days per month")


def clean_training_table(table: pd.DataFrame, kind: str, years: Sequence[int]) -> pd.DataFrame:
    """Validate caller-cleaned eligible labels; reject rather than trim leakage."""
    required = {"station_key", "date", "tn_mg_l", "eligible"}
    if kind == "HF":
        required.add("read_count")
    if not required.issubset(table.columns):
        raise ValueError(f"Missing {kind} columns: {sorted(required-set(table.columns))}")
    t = table.copy(deep=True)
    t["date"] = pd.to_datetime(t.date, errors="raise").dt.normalize()
    if t.date.isna().any() or not set(t.date.dt.year).issubset(set(years)):
        raise ValueError("Training table contains missing dates or non-training-year labels")
    if len(t) and (not pd.api.types.is_bool_dtype(t.eligible.dtype) or t.eligible.isna().any()):
        raise ValueError("eligible must be boolean")
    t = t[t.eligible.astype(bool)].copy()
    if t.station_key.isna().any() or t.station_key.astype(str).eq("").any():
        raise ValueError("Station identity missing")
    t["station_key"] = t.station_key.astype(str)
    t["tn_mg_l"] = pd.to_numeric(t.tn_mg_l, errors="raise").astype(float)
    if not np.isfinite(t.tn_mg_l).all() or (t.tn_mg_l < 0).any():
        raise ValueError("Eligible TN must be finite and nonnegative")
    t["month"] = t.date.dt.to_period("M").astype(str)
    if kind == "HF":
        t["read_count"] = pd.to_numeric(t.read_count, errors="raise").astype(float)
        if (not np.isfinite(t.read_count).all() or (t.read_count <= 0).any()
                or (t.read_count % 1 != 0).any()):
            raise ValueError("HF read_count must be positive integer")
        keys = ["station_key", "date"]
    else:
        t["read_count"] = 1
        keys = ["station_key", "month"]
    if t.duplicated(keys).any():
        raise ValueError(f"Duplicate {kind} labels on {keys}; resolve provenance upstream")
    t["observation_kind"] = kind
    return t


class TrainingObjective:
    """Frozen train-only support, scales, and analytic observation gradient."""
    def __init__(self, rows, groups, scales, identity):
        self._rows = rows.copy(deep=True)
        self.groups = tuple(groups)
        self.scales = scales.copy(deep=True)
        self.identity = dict(identity)
        self._truth = rows.tn_mg_l.to_numpy(dtype=np.float64, copy=True)
        self._truth.setflags(write=False)

    @property
    def rows(self):
        return self._rows.copy(deep=True)

    def evaluate(self, prediction, prior_terms: Mapping[str, float] | None = None):
        p = np.asarray(prediction, dtype=np.float64)
        if p.shape != self._truth.shape or not np.isfinite(p).all():
            raise ValueError("Prediction must be finite and match frozen rows")
        prior_terms = dict(prior_terms or {})
        if any(not np.isfinite(v) or v < 0 for v in prior_terms.values()):
            raise ValueError("Prior terms must be finite and nonnegative")
        gradient = np.zeros_like(p)
        level, anomaly = 0.0, 0.0
        for g in self.groups:
            idx, alpha = g["indices"], g["alpha"]
            residual = p[idx] - self._truth[idx]
            mean_error = float(alpha @ residual)
            level += .5 * g["level_weight"] * mean_error**2 / g["level_variance"]
            gradient[idx] += .5 * g["level_weight"] * mean_error * alpha / g["level_variance"]
            if g["dynamic_weight"] > 0:
                centered = residual - mean_error
                anomaly += .5 * g["dynamic_weight"] * float(alpha @ centered**2) / g["anomaly_variance"]
                gradient[idx] += .5 * g["dynamic_weight"] * alpha * centered / g["anomaly_variance"]
        terms = {"D_month_level": float(level), "D_month_anomaly": float(anomaly),
                 "weighted_month_level": .5*float(level), "weighted_month_anomaly": .5*float(anomaly),
                 "prior_terms": prior_terms, "prior_total": float(sum(prior_terms.values()))}
        return .5*(level+anomaly)+terms["prior_total"], gradient, terms


def build_training_objective(hf: pd.DataFrame, monthly: pd.DataFrame,
                             train_years: Sequence[int], policy: TrainingPolicy) -> TrainingObjective:
    """HF qualifying station-months win; monthly reports only fill other months.

    Station scales are population SD of unique training-month levels and the
    equal-month average read-weighted HF anomaly variance, floored ONLY by the
    caller's explicit training policy. No unseen station scale is inferred.
    """
    years = tuple(sorted(set(int(y) for y in train_years)))
    if not years:
        raise ValueError("Empty train_years")
    h = clean_training_table(hf, "HF", years)
    m = clean_training_table(monthly, "monthly", years)
    keys = ["station_key", "month"]
    if not h.empty:
        size = h.groupby(keys).date.transform("size")
        h = h[size >= policy.min_hf_days_per_month].copy()
    hf_keys = set(map(tuple, h[keys].to_numpy()))
    m = m[[tuple(row) not in hf_keys for row in m[keys].to_numpy()]].copy()
    columns = ["station_key", "date", "month", "tn_mg_l", "read_count", "observation_kind"]
    rows = pd.concat([h[columns], m[columns]], ignore_index=True).sort_values(
        ["station_key", "month", "date", "observation_kind"]).reset_index(drop=True)
    if rows.empty or h.empty:
        raise ValueError("Common objective requires both level and qualifying HF anomaly support")
    rows["prediction_id"] = np.arange(len(rows))
    station_month_count = rows[keys].drop_duplicates().station_key.value_counts()
    dynamic_month_count = h[keys].drop_duplicates().station_key.value_counts()
    n_level, n_dyn = len(station_month_count), len(dynamic_month_count)
    provisional, station_levels, station_anomalies = [], {}, {}
    for (station, month), frame in rows.groupby(keys, sort=True):
        idx = frame.index.to_numpy()
        weights = frame.read_count.to_numpy(dtype=np.float64)
        alpha = weights / weights.sum()
        observed = frame.tn_mg_l.to_numpy(dtype=np.float64)
        mean = float(alpha @ observed)
        is_hf = bool((frame.observation_kind == "HF").all())
        station_levels.setdefault(station, []).append(mean)
        if is_hf:
            station_anomalies.setdefault(station, []).append(float(alpha @ (observed-mean)**2))
        provisional.append({"station_key": station, "month": month, "indices": idx,
                            "alpha": alpha, "is_hf": is_hf})
    scale_rows = []
    for station in sorted(station_levels):
        level_var = float(np.var(station_levels[station]))
        anomaly_var = (float(np.mean(station_anomalies[station]))
                       if station in station_anomalies else None)
        scale_rows.append({"station_key": station, "level_empirical_variance": level_var,
                           "level_variance": max(level_var, policy.level_std_floor**2),
                           "anomaly_empirical_variance": anomaly_var,
                           "anomaly_variance": (max(anomaly_var, policy.anomaly_std_floor**2)
                                                if anomaly_var is not None else None),
                           "level_floor_used": level_var < policy.level_std_floor**2,
                           "anomaly_floor_used": (anomaly_var < policy.anomaly_std_floor**2
                                                  if anomaly_var is not None else None)})
    scales = pd.DataFrame(scale_rows)
    by_station = scales.set_index("station_key").to_dict("index")
    groups = []
    for g in provisional:
        station = g["station_key"]
        g.update(level_weight=1/(n_level*station_month_count[station]),
                 dynamic_weight=(1/(n_dyn*dynamic_month_count[station]) if g["is_hf"] else 0),
                 level_variance=by_station[station]["level_variance"],
                 anomaly_variance=by_station[station]["anomaly_variance"])
        g["indices"].setflags(write=False)
        g["alpha"].setflags(write=False)
        groups.append(g)
    serialized = rows.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g")
    identity = {"train_years": list(years), "policy": vars(policy),
                "support_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                "scale_sha256": hashlib.sha256(scales.to_json(orient="records", double_precision=15).encode()).hexdigest(),
                "unique_station_months": len(groups), "n_level_stations": n_level,
                "n_hf_stations": n_dyn, "objective": ".25*level_standardized_MSE+.25*anomaly_standardized_MSE+R",
                "evaluation_labels_read": False, "monthly_selection": "qualified_HF_first_monthly_fill"}
    identity["identity_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return TrainingObjective(rows, groups, scales, identity)
