from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260823_14"
OUT = RUN / "outputs"
REPORT = RUN / "reports"


def markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    head = "| " + " | ".join(view.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([head, rule, *rows])


coverage = pd.read_parquet(OUT / "station_coverage_and_selection.parquet")
topology = pd.read_csv(ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv")
reach = pd.read_csv(ROOT / "0_reach_topology" / "results" / "tables" / "reach_summary.csv")
reach_name = reach[["reach_id", "src_id"]].drop_duplicates("reach_id")
counts = coverage.groupby("reach_id").station_norm.transform("nunique")
conflicts = coverage[counts.gt(1)].copy().merge(reach_name, on="reach_id", how="left")
conflicts = conflicts[[
    "reach_id", "src_id", "station_norm", "q_site", "station_type", "training_eligible",
    "four_group_check_eligible", "development_months", "check_months", "downstream_fraction_on_reach",
    "snap_distance_m", "line_catchment_override", "selected_for_model",
]].sort_values(["reach_id", "downstream_fraction_on_reach"])
conflicts["user_decision"] = ""
conflicts.to_parquet(OUT / "manual_same_reach_station_decisions.parquet", index=False)

match = pd.read_csv(REPORT / "station_reach_match.csv", encoding="utf-8-sig")
outside = match[~match.used.astype(bool)].copy().merge(reach_name, on="reach_id", how="left")
outside = outside[["station_norm", "station_name", "reach_id", "src_id", "match_method", "best_line_distance_m", "best_catchment_reach_id", "best_line_reach_id"]]
outside["user_confirmed_reach_id"] = ""
outside.to_parquet(OUT / "manual_outside_topology_station_decisions.parquet", index=False)
risk = match[
    match.used.astype(bool)
    & (match.best_line_distance_m.gt(500) | match.line_overrode_catchment.astype(bool))
].copy().merge(reach_name, left_on="reach_id", right_on="reach_id", how="left")
risk = risk[[
    "station_norm", "station_name", "reach_id", "src_id", "match_method", "best_line_distance_m",
    "best_catchment_reach_id", "best_line_reach_id", "line_overrode_catchment",
]].sort_values(["line_overrode_catchment", "best_line_distance_m"], ascending=[False, False])
risk["user_decision"] = ""
risk.to_parquet(OUT / "manual_spatial_risk_station_decisions.parquet", index=False)

special = coverage[coverage.station_type.ne("ordinary_river_gauge")].copy().merge(reach_name, on="reach_id", how="left")
special = special[[
    "station_norm", "q_site", "station_type", "reach_id", "src_id", "development_months", "check_months",
    "downstream_fraction_on_reach", "snap_distance_m", "selected_for_model",
]].sort_values(["selected_for_model", "reach_id"], ascending=[False, True])
special["user_decision"] = ""
special.to_parquet(OUT / "manual_structure_station_decisions.parquet", index=False)

unmatched = pd.read_csv(REPORT / "unmatched_discharge_station_names.csv", encoding="utf-8-sig")
coords = pd.read_csv(REPORT / "station_shapefile_attributes.csv", encoding="utf-8-sig")
coord_names = coords[["station_norm", "station_name_shp"]].drop_duplicates("station_norm")
alias_rows = []
for row in unmatched.itertuples(index=False):
    source = str(row.station_norm)
    candidates = []
    for cand in coord_names.itertuples(index=False):
        target = str(cand.station_norm)
        score = SequenceMatcher(None, source, target).ratio()
        containment = source.endswith(target) or target.endswith(source) or source in target or target in source
        candidates.append((containment, score, target, str(cand.station_name_shp)))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    containment, score, target, display = candidates[0]
    alias_rows.append({
        "unmatched_station_norm": source,
        "unmatched_station_name": str(row.station_name),
        "usable_months": int(row.usable_months),
        "suggested_coordinate_station_norm": target,
        "suggested_coordinate_station_name": display,
        "name_similarity": float(score),
        "suffix_or_containment_relation": bool(containment),
        "high_priority_manual_review": bool(containment or score >= 0.85),
        "user_confirmed_coordinate_station": "",
        "user_confirmed_reach_id": "",
    })
aliases = pd.DataFrame(alias_rows).sort_values(["high_priority_manual_review", "usable_months", "name_similarity"], ascending=[False, False, False])
aliases.to_parquet(OUT / "manual_unmatched_alias_decisions.parquet", index=False)

conflict_view = conflicts[[
    "reach_id", "src_id", "station_norm", "development_months", "check_months",
    "downstream_fraction_on_reach", "snap_distance_m", "selected_for_model", "user_decision",
]]
risk_view = risk[["station_norm", "reach_id", "src_id", "match_method", "best_line_distance_m", "best_catchment_reach_id", "best_line_reach_id", "user_decision"]]
special_view = special[["station_norm", "station_type", "reach_id", "src_id", "development_months", "check_months", "selected_for_model", "user_decision"]]
alias_view = aliases.loc[aliases.high_priority_manual_review, [
    "unmatched_station_norm", "usable_months", "suggested_coordinate_station_norm", "name_similarity",
    "suffix_or_containment_relation", "user_confirmed_reach_id",
]]

text = "# Manual station-to-Reach review\n\n"
text += "These tables are decision aids. `selected_for_model` is provisional and must not be treated as the final user lock.\n\n"
text += f"## Same-Reach conflicts ({conflicts.reach_id.nunique()} Reaches, {len(conflicts)} stations)\n\n{markdown(conflict_view)}\n\n"
text += f"## Spatial-risk matches ({len(risk)} stations)\n\n{markdown(risk_view)}\n\n"
text += f"## Coordinate stations outside the 5-km topology gate ({len(outside)} stations)\n\n{markdown(outside.head(80))}\n\n"
text += f"## Channel/reservoir/structure-like stations ({len(special)} stations)\n\n{markdown(special_view)}\n\n"
text += f"## High-priority unmatched aliases ({len(alias_view)} of {len(aliases)} unmatched stations)\n\n{markdown(alias_view)}\n"
(REPORT / "manual_topology_review.md").write_text(text, encoding="utf-8")

summary = {
    "status": "NEEDS_USER_TOPOLOGY_DECISIONS",
    "same_reach_conflict_reaches": int(conflicts.reach_id.nunique()),
    "same_reach_candidate_stations": int(len(conflicts)),
    "spatial_risk_stations": int(len(risk)),
    "coordinate_stations_outside_topology_gate": int(len(outside)),
    "structure_like_stations": int(len(special)),
    "unmatched_stations": int(len(aliases)),
    "high_priority_unmatched_aliases": int(aliases.high_priority_manual_review.sum()),
    "decision_files": [
        str(OUT / "manual_same_reach_station_decisions.parquet"),
        str(OUT / "manual_spatial_risk_station_decisions.parquet"),
        str(OUT / "manual_structure_station_decisions.parquet"),
        str(OUT / "manual_unmatched_alias_decisions.parquet"),
        str(OUT / "manual_outside_topology_station_decisions.parquet"),
    ],
}
(REPORT / "manual_topology_review.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

lock_template = {
    "status": "AWAITING_USER_DECISIONS",
    "instructions": {
        "same_reach": "For every entry, set selected_station_norm to exactly one listed candidate or EXCLUDE_REACH.",
        "spatial_override": "Set decision to ACCEPT_CURRENT_REACH, USE_CATCHMENT_REACH, EXCLUDE_STATION, or an integer Reach id.",
        "structure_station": "Set decision to INCLUDE or EXCLUDE.",
        "unmatched_and_outside": "Remain outside training unless an explicit user_confirmed_reach_id is supplied later.",
    },
    "same_reach_decisions": [],
    "spatial_override_decisions": [],
    "structure_station_decisions": [],
}
for (rid, src), part in conflicts.groupby(["reach_id", "src_id"], sort=True):
    lock_template["same_reach_decisions"].append({
        "reach_id": int(rid),
        "reach_name": str(src),
        "candidates": [
            {
                "station_norm": str(row.station_norm),
                "development_months": int(row.development_months),
                "check_months": int(row.check_months),
                "downstream_fraction": float(row.downstream_fraction_on_reach),
                "station_type": str(row.station_type),
            }
            for row in part.itertuples(index=False)
        ],
        "selected_station_norm": None,
        "decision_reason": "",
    })
for row in risk[risk.line_overrode_catchment.astype(bool)].itertuples(index=False):
    lock_template["spatial_override_decisions"].append({
        "station_norm": str(row.station_norm),
        "nearest_line_reach_id": int(row.best_line_reach_id),
        "catchment_reach_id": int(row.best_catchment_reach_id),
        "nearest_line_distance_m": float(row.best_line_distance_m),
        "decision": None,
        "decision_reason": "",
    })
for row in special.itertuples(index=False):
    lock_template["structure_station_decisions"].append({
        "station_norm": str(row.station_norm),
        "current_reach_id": int(row.reach_id),
        "station_type": str(row.station_type),
        "currently_selected": bool(row.selected_for_model),
        "decision": None,
        "decision_reason": "",
    })
(REPORT / "user_topology_lock_template.json").write_text(json.dumps(lock_template, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
