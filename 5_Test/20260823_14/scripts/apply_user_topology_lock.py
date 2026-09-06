from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260823_14"
OUT = RUN / "outputs"
REPORT = RUN / "reports"

USER_REACH_OVERRIDES = {
    "博罗（二）": 23,
    "高要": 20,
    "梧州（四）": 18,
    "顺天": 54,
    "富阳": 91,
}
SAME_REACH_EXCEPTIONS = {64: "滃江"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


coverage = pd.read_parquet(OUT / "station_coverage_and_selection.parquet")
monthly = pd.read_parquet(OUT / "all_monthly_discharge_after_exclusions.parquet")
match = pd.read_csv(REPORT / "station_reach_match.csv", encoding="utf-8-sig")
exclusions = set(pd.read_parquet(OUT / "registered_station_exclusions.parquet").station_key.astype(str))

# The station identity contract already removes a terminal 站. Assert the
# explicit example from the user rather than adding a second alias system.
if "荆山" not in set(monthly.station_norm.astype(str)):
    raise RuntimeError("Expected normalized identity 荆山 was not found")
if any(str(name).endswith("站") for name in monthly.station_norm.astype(str).unique()):
    raise RuntimeError("Terminal 站 survived normalized station identity")

coverage["original_reach_id"] = coverage.reach_id.astype(int)
coverage["reach_assignment_method"] = "audited_spatial_match"
for station, rid in USER_REACH_OVERRIDES.items():
    mask = coverage.station_norm.eq(station)
    if mask.sum() != 1:
        raise RuntimeError(f"User override station {station} resolved to {mask.sum()} coverage rows")
    coverage.loc[mask, "reach_id"] = int(rid)
    coverage.loc[mask, "reach_assignment_method"] = "user_explicit_reaches_topology_id"
    coverage.loc[mask, "snap_distance_m"] = np.nan

development_flow = monthly[monthly.year.le(2018) & monthly.Q_obsv_cfs.notna()].groupby("station_norm", as_index=False).agg(
    development_mean_Q_cfs=("Q_obsv_cfs", "mean"),
    development_median_Q_cfs=("Q_obsv_cfs", "median"),
)
coverage = coverage.drop(columns=["selected_for_model", "selected_for_four_group_check", "representative_rank_for_reach"], errors="ignore")
coverage = coverage.merge(development_flow, on="station_norm", how="left", validate="one_to_one")

# Selection is only among registered training-eligible gauges. Larger mean
# 2006-2018 flow wins, except for explicit user exceptions.
coverage["user_exception_priority"] = 0
for rid, station in SAME_REACH_EXCEPTIONS.items():
    exists = coverage.reach_id.eq(rid) & coverage.station_norm.eq(station)
    if exists.sum() != 1:
        raise RuntimeError(f"User same-Reach exception {rid}={station} not uniquely available")
    coverage.loc[exists, "user_exception_priority"] = 1
coverage = coverage.sort_values(
    ["reach_id", "training_eligible", "user_exception_priority", "development_mean_Q_cfs", "development_months", "station_norm"],
    ascending=[True, False, False, False, False, True],
)
coverage["representative_rank_for_reach"] = coverage.groupby("reach_id").cumcount() + 1
coverage["selected_for_model"] = coverage.training_eligible & coverage.representative_rank_for_reach.eq(1)
coverage["selected_for_four_group_check"] = coverage.selected_for_model & coverage.four_group_check_eligible

selected = coverage.loc[coverage.selected_for_model, ["station_norm", "reach_id", "selected_for_four_group_check"]].copy()
selected_pairs = selected[["station_norm", "reach_id"]].drop_duplicates()
if selected_pairs.station_norm.duplicated().any() or selected_pairs.reach_id.duplicated().any():
    raise RuntimeError("Final selection is not one station per Reach and one Reach per station")
if any(any(len(item) >= 2 and (item in station or station in item) for item in exclusions) for station in selected.station_norm):
    raise RuntimeError("Registered exclusion family survived final station selection")

# Reassign only the five user-authorized stations; all other monthly rows keep
# their audited Reach. Non-selected, unmatched and >5-km stations remain out.
usable = pd.read_parquet(OUT / "all_monthly_discharge_after_exclusions.parquet")
usable = usable[usable.Q_obsv_cfs.notna() & usable.Q_obsv_cfs.gt(0)].copy()
station_reach = coverage[["station_norm", "reach_id"]].drop_duplicates("station_norm")
usable = usable.drop(columns=["reach_id"], errors="ignore").merge(station_reach, on="station_norm", how="inner", validate="many_to_one")
final_obs = usable.merge(selected, on=["station_norm", "reach_id"], how="inner")
canonical_names = coverage.set_index("station_norm").q_site.to_dict()
final_obs["q_site"] = final_obs.station_norm.map(canonical_names)
final_obs["station_type"] = final_obs.station_norm.map(coverage.set_index("station_norm").station_type)
final_obs["snap_distance_m"] = final_obs.station_norm.map(coverage.set_index("station_norm").snap_distance_m)
final_obs = final_obs[[
    "station_norm", "q_site", "reach_id", "year", "month", "Q_obsv_cfs", "q_m3s", "coverage",
    "valid_days", "total_days", "source", "source_window", "station_type", "snap_distance_m",
    "selected_for_four_group_check",
]].sort_values(["station_norm", "year", "month"])

if final_obs.duplicated(["reach_id", "year", "month"]).any():
    raise RuntimeError("Duplicate Reach-month survived final station lock")
if final_obs.duplicated(["station_norm", "year", "month"]).any():
    raise RuntimeError("Duplicate station-month survived final station lock")

coverage.to_parquet(OUT / "final_user_locked_station_coverage.parquet", index=False)
final_path = OUT / "final_model_station_month_observations.parquet"
final_obs.to_parquet(final_path, index=False)

decisions = []
for rid, part in coverage[coverage.training_eligible].groupby("reach_id", sort=True):
    if part.station_norm.nunique() <= 1:
        continue
    chosen = part.loc[part.selected_for_model, "station_norm"]
    decisions.append({
        "reach_id": int(rid),
        "candidates": part.station_norm.astype(str).tolist(),
        "candidate_development_mean_Q_cfs": {str(r.station_norm): float(r.development_mean_Q_cfs) for r in part.itertuples(index=False)},
        "selected_station_norm": str(chosen.iloc[0]),
        "rule": "USER_EXCEPTION" if int(rid) in SAME_REACH_EXCEPTIONS else "MAX_2006_2018_MEAN_OBSERVED_FLOW",
    })

lock = {
    "stage": "20260823_14",
    "status": "USER_TOPOLOGY_LOCK_APPLIED",
    "station_identity_rule": "NFKC; whitespace removed; parentheses unified; terminal 站 removed; terminal _2/_3/_4 normalized",
    "same_reach_rule": "maximum mean observed monthly flow in 2006-2018 among training-eligible candidates",
    "same_reach_exceptions": {str(k): v for k, v in SAME_REACH_EXCEPTIONS.items()},
    "user_reach_overrides": USER_REACH_OVERRIDES,
    "unmatched_or_outside_topology_policy": "EXCLUDED_UNTIL_EXPLICIT_REACH_MAPPING",
    "single_structure_station_policy": "retained as observation and flagged; no reservoir equation introduced",
    "selected_training_stations": int(selected.station_norm.nunique()),
    "selected_training_reaches": int(selected.reach_id.nunique()),
    "selected_four_group_check_stations": int(selected.selected_for_four_group_check.sum()),
    "development_rows": int(final_obs.year.le(2018).sum()),
    "frozen_check_rows": int((final_obs.year.ge(2019) & final_obs.selected_for_four_group_check).sum()),
    "post_lock_duplicate_reach_months": int(final_obs.duplicated(["reach_id", "year", "month"]).sum()),
    "post_lock_exclusion_residual_count": 0,
    "final_observation_sha256": sha256(final_path),
    "same_reach_decisions": decisions,
}
(REPORT / "applied_user_topology_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# Applied user topology lock", "", f"Status: `{lock['status']}`", "",
    f"Selected training stations/Reaches: {lock['selected_training_stations']}/{lock['selected_training_reaches']}",
    f"Four-group checking stations: {lock['selected_four_group_check_stations']}", "",
    "## User Reach overrides", "",
]
for station, rid in USER_REACH_OVERRIDES.items():
    lines.append(f"- {station} -> Reach {rid}")
lines.extend(["", "## Same-Reach selections", ""])
for decision in decisions:
    lines.append(f"- Reach {decision['reach_id']}: {decision['selected_station_norm']} ({decision['rule']})")
(REPORT / "applied_user_topology_lock.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in lock.items() if k != "same_reach_decisions"}, ensure_ascii=False, indent=2))
