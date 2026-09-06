from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260823_14"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
EPS = 1e-12


def corr_log(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) <= EPS or np.std(b) <= EPS:
        return float("nan")
    return float(np.corrcoef(np.log1p(a), np.log1p(b))[0, 1])


def md(frame: pd.DataFrame) -> str:
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    head = "| " + " | ".join(view.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([head, rule, *rows])


coverage = pd.read_parquet(OUT / "station_coverage_and_selection.parquet")
counts = coverage.groupby("reach_id").station_norm.transform("nunique")
conflicts = coverage[counts.gt(1)].copy()
monthly = pd.read_parquet(OUT / "all_monthly_discharge_after_exclusions.parquet")
monthly = monthly[monthly.Q_obsv_cfs.notna()][["station_norm", "year", "month", "Q_obsv_cfs"]]
reach_names = pd.read_csv(ROOT / "0_reach_topology" / "results" / "tables" / "reach_summary.csv")[["reach_id", "src_id"]]

rows = []
for rid, group in conflicts.groupby("reach_id", sort=True):
    stations = group.drop_duplicates("station_norm").set_index("station_norm")
    for a, b in itertools.combinations(stations.index.astype(str), 2):
        aa = monthly[monthly.station_norm.eq(a)].rename(columns={"Q_obsv_cfs": "qa"})
        bb = monthly[monthly.station_norm.eq(b)].rename(columns={"Q_obsv_cfs": "qb"})
        pair = aa.merge(bb, on=["year", "month"], how="inner")
        qa = pair.qa.to_numpy(float)
        qb = pair.qb.to_numpy(float)
        fa = float(stations.loc[a, "downstream_fraction_on_reach"])
        fb = float(stations.loc[b, "downstream_fraction_on_reach"])
        if fa <= fb:
            upstream, downstream, qu, qd, fu, fd = a, b, qa, qb, fa, fb
        else:
            upstream, downstream, qu, qd, fu, fd = b, a, qb, qa, fb, fa
        n = len(pair)
        relative = np.abs(qd - qu) / np.maximum(np.maximum(qd, qu), EPS) if n else np.array([])
        ratio = float(np.median(qd / np.maximum(qu, EPS))) if n else float("nan")
        r = corr_log(qu, qd)
        exact = bool(n >= 12 and np.max(relative) < 1e-10)
        downstream_ge = float(np.mean(qd >= qu)) if n else float("nan")
        if n < 12:
            flag = "INSUFFICIENT_OVERLAP"
        elif exact:
            flag = "EXACT_DUPLICATE_SERIES"
        elif (np.isfinite(r) and r < 0.5) or ratio < 0.5:
            flag = "HYDROLOGICALLY_SUSPICIOUS"
        elif ratio >= 0.8 and (not np.isfinite(r) or r >= 0.7):
            flag = "DIRECTION_PLAUSIBLE"
        else:
            flag = "MANUAL_REVIEW"
        rows.append({
            "reach_id": int(rid), "upstream_station": upstream, "downstream_station": downstream,
            "upstream_fraction": fu, "downstream_fraction": fd, "overlap_months": n,
            "median_downstream_upstream_flow_ratio": ratio, "log_flow_correlation": r,
            "fraction_months_downstream_ge_upstream": downstream_ge,
            "median_abs_relative_difference": float(np.median(relative)) if n else float("nan"),
            "pair_flag": flag,
        })

audit = pd.DataFrame(rows).merge(reach_names, on="reach_id", how="left")
audit = audit[["reach_id", "src_id", "upstream_station", "downstream_station", "overlap_months",
               "median_downstream_upstream_flow_ratio", "log_flow_correlation",
               "fraction_months_downstream_ge_upstream", "median_abs_relative_difference", "pair_flag"]]
audit.to_parquet(OUT / "same_reach_pairwise_flow_audit.parquet", index=False)

summary = audit.groupby(["reach_id", "src_id"], as_index=False).agg(
    candidate_pairs=("pair_flag", "size"),
    exact_duplicate_pairs=("pair_flag", lambda s: int((s == "EXACT_DUPLICATE_SERIES").sum())),
    suspicious_pairs=("pair_flag", lambda s: int((s == "HYDROLOGICALLY_SUSPICIOUS").sum())),
    insufficient_pairs=("pair_flag", lambda s: int((s == "INSUFFICIENT_OVERLAP").sum())),
    minimum_log_flow_correlation=("log_flow_correlation", "min"),
    minimum_downstream_upstream_ratio=("median_downstream_upstream_flow_ratio", "min"),
)
summary.to_parquet(OUT / "same_reach_pairwise_flow_summary.parquet", index=False)

report = "# Same-Reach pairwise flow audit\n\n"
report += "This is diagnostic evidence only. It does not automatically select a station.\n\n"
report += "## Reach summary\n\n" + md(summary) + "\n\n"
report += "## Pairs requiring attention\n\n" + md(audit[audit.pair_flag.ne("DIRECTION_PLAUSIBLE")]) + "\n"
(REPORT / "same_reach_pairwise_flow_audit.md").write_text(report, encoding="utf-8")

print(summary.to_string(index=False))
