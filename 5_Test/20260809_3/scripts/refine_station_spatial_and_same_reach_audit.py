from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORTS = RUN / "reports"
BASE = ROOT / "5_Test" / "20260805_2"

STATION_SPATIAL = REPORTS / "all_station_reach_catchment_audit.csv"
SELECTED = BASE / "inputs" / "source_metadata" / "selected_representative_stations.csv"
SAME_REACH = BASE / "inputs" / "source_metadata" / "same_reach_selection_audit.csv"


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def strip_station_suffix(value: object) -> str:
    text = str(value).strip().replace("（", "(").replace("）", ")")
    text = re.sub(r"\(重复\)$", "", text)
    text = re.sub(r"_\d+$", "", text)
    return text[:-1] if text.endswith("站") else text


def classify(row: pd.Series) -> tuple[str, str]:
    if pd.isna(row.assigned_reach_id):
        return "UNMAPPED_NOT_USED_TO_JUDGE_REACH", "No assigned Reach"
    if bool_value(row.assigned_reach_catchment_contains_station):
        if bool_value(row.assigned_reach_is_geometric_nearest):
            if float(row.assigned_reach_distance_m_recomputed) <= 2000:
                return "SPATIALLY_CONSISTENT_MAIN_REACH", "Inside assigned incremental Catchment; assigned Reach is nearest"
            return "PLAUSIBLE_MINOR_TRIBUTARY_OR_GENERALIZED_MAIN_REACH", "Inside assigned Catchment; nearest represented Reach is assigned but simplified network is distant"
        return "PLAUSIBLE_WITHIN_ASSIGNED_CATCHMENT_DIFFERENT_NEAREST_REACH", "Inside assigned Catchment; another represented Reach is slightly nearer"
    distance = float(row.distance_to_assigned_catchment_m) if pd.notna(row.distance_to_assigned_catchment_m) else np.inf
    containing = str(row.containing_incremental_catchment_ids).strip()
    if distance > 5000 and containing:
        return "CONFIRMED_POINT_IN_DIFFERENT_INCREMENTAL_CATCHMENT", "Coordinate is >5 km outside assigned Catchment and lies in a different incremental Catchment"
    if distance > 5000:
        return "HIGH_PRIORITY_COORDINATE_OR_OMITTED_TRIBUTARY_REVIEW", "Coordinate is >5 km outside assigned Catchment but no alternative Catchment contains it"
    if distance > 2000:
        return "STATION_CATCHMENT_BOUNDARY_REVIEW", "Coordinate is 2-5 km outside assigned Catchment; coarse coordinates/generalization remain plausible"
    return "LIKELY_COORDINATE_ROUNDING_OR_BOUNDARY_GENERALIZATION", "Coordinate is <=2 km outside assigned Catchment"


def main() -> None:
    spatial = pd.read_csv(STATION_SPATIAL, encoding="utf-8-sig")
    selected = pd.read_csv(SELECTED, encoding="utf-8-sig")
    same = pd.read_csv(SAME_REACH, encoding="utf-8-sig")

    spatial["station_key"] = spatial.station.map(strip_station_suffix)
    selected["station_key"] = selected.station_norm.map(strip_station_suffix)
    selected_keys = set(selected.station_key)
    spatial["q72_selected_representative"] = spatial.station_key.isin(selected_keys)
    decisions = spatial.apply(classify, axis=1, result_type="expand")
    decisions.columns = ["coordinate_spatial_class", "coordinate_spatial_reason"]
    out = pd.concat([spatial, decisions], axis=1)
    out.to_csv(REPORTS / "all_station_coordinate_based_spatial_classification.csv", index=False, encoding="utf-8-sig")

    same["reach_id"] = pd.to_numeric(same.reach_id, errors="raise").astype(int)
    same["median_q_cfs"] = pd.to_numeric(same.median_q_cfs, errors="coerce")
    same["selection_rank"] = pd.to_numeric(same.selection_rank, errors="coerce")
    same["selected_for_reach_bool"] = same.selected_for_reach.map(bool_value)
    group_rows = []
    violations = []
    for rid, group in same.groupby("reach_id", sort=True):
        chosen = group[group.selected_for_reach_bool]
        max_flow = group.median_q_cfs.max()
        chosen_flow = chosen.median_q_cfs.iloc[0] if len(chosen) == 1 else np.nan
        correct = bool(len(chosen) == 1 and (np.isnan(max_flow) or abs(chosen_flow - max_flow) <= 1e-9))
        if not correct:
            violations.append(int(rid))
        group_rows.append(
            {
                "reach_id": int(rid),
                "station_count_on_reach": len(group),
                "station_names": "|".join(group.station_norm.astype(str)),
                "selected_station": chosen.station_norm.iloc[0] if len(chosen) == 1 else "",
                "selected_median_q_cfs": chosen_flow,
                "maximum_median_q_cfs": max_flow,
                "one_and_only_one_selected": len(chosen) == 1,
                "highest_flow_station_selected": correct,
            }
        )
    groups = pd.DataFrame(group_rows)
    groups.to_csv(REPORTS / "same_reach_representative_selection_validation.csv", index=False, encoding="utf-8-sig")

    selected_out = out[out.q72_selected_representative].drop_duplicates(["station_key", "assigned_reach_id"]).copy()
    mapped_unique = out[out.assigned_reach_id.notna()].drop_duplicates(["station_key", "assigned_reach_id"]).copy()
    summary = {
        "all_station_rows": int(len(out)),
        "mapped_station_mapping_rows": int(out.assigned_reach_id.notna().sum()),
        "mapped_unique_station_reach_pairs": int(len(mapped_unique)),
        "q72_selected_representative_unique_station_reach_pairs": int(len(selected_out)),
        "q72_selected_spatial_class_counts": selected_out.coordinate_spatial_class.value_counts().to_dict(),
        "all_mapped_spatial_class_counts": mapped_unique.coordinate_spatial_class.value_counts().to_dict(),
        "same_reach_unique_reaches": int(len(groups)),
        "same_reach_collision_reaches": int((groups.station_count_on_reach > 1).sum()),
        "same_reach_collision_station_rows": int(groups.loc[groups.station_count_on_reach > 1, "station_count_on_reach"].sum()),
        "same_reach_highest_flow_selection_violations": violations,
        "decision_basis": "coordinate, nearest Reach, and incremental Catchment containment; station and river names are identifiers only",
    }
    (REPORTS / "station_spatial_and_same_reach_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
