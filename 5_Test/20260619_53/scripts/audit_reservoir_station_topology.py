from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260619_53"
REPORTS = RUN / "reports"
LOGS = RUN / "logs"
DAILY_LOG = ROOT / "5_Test" / "20260619.log"

TABLES = ROOT / "0_reach_topology" / "results" / "tables"
REACH_SUMMARY = TABLES / "reach_summary.csv"
TOPOLOGY = TABLES / "topology_edges.csv"
NODE_SUMMARY = TABLES / "node_summary.csv"
STATION_MATCH = ROOT / "5_Test" / "20260618_1" / "reports" / "input_preprocessing" / "station_reach_match.csv"
MODEL_RES_COV = ROOT / "5_Test" / "20260619_50" / "reports" / "model_reservoir_observation_coverage.csv"
STATION_COV = ROOT / "5_Test" / "20260619_50" / "reports" / "reservoir_related_station_operation_coverage.csv"


def clean_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def norm(x: object) -> str:
    s = "" if pd.isna(x) else str(x)
    for token in ["水库", "水电站", "水利枢纽", "枢纽", "（红水河）", "(盘阳河)", "（一级）", "(一级)", "一级"]:
        s = s.replace(token, "")
    return re.sub(r"[()\[\]（）\s·]", "", s)


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_fraction_on_chord(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float]:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom <= 0:
        return np.nan, np.nan
    t = (wx * vx + wy * vy) / denom
    t_clamped = max(0.0, min(1.0, t))
    cx, cy = ax + t_clamped * vx, ay + t_clamped * vy
    return float(t_clamped), dist((px, py), (cx, cy))


def build_path(start: int, target: int, downstream: dict[int, float], max_steps: int = 50) -> list[int]:
    path = [int(start)]
    cur = int(start)
    seen = {cur}
    while cur != int(target) and len(path) <= max_steps:
        nxt = downstream.get(cur)
        if pd.isna(nxt):
            break
        nxt = int(float(nxt))
        if nxt in seen:
            path.append(nxt)
            break
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    return path


def markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in frame.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    reach = pd.read_csv(REACH_SUMMARY)
    topo = pd.read_csv(TOPOLOGY)
    node = pd.read_csv(NODE_SUMMARY)
    station_match = pd.read_csv(STATION_MATCH, encoding="utf-8-sig")
    res_cov = pd.read_csv(MODEL_RES_COV, encoding="utf-8-sig")
    station_cov = pd.read_csv(STATION_COV, encoding="utf-8-sig")

    reach_idx = reach.set_index("reach_id")
    topo_idx = topo.set_index("reach_id")
    node_idx = node.set_index("node_id")
    downstream = topo_idx["downstream_reach"].to_dict()

    station_cov["good_bool"] = clean_bool(station_cov["good_bool"])
    linked_bad = station_cov[
        station_cov["nearest_upstream_reservoir_name"].notna()
        & station_cov["nearest_upstream_reservoir_name"].astype(str).str.strip().ne("")
        & (~station_cov["good_bool"])
    ].copy()

    # Map nearest reservoir name to a reservoir reach. Prefer exact source name;
    # otherwise conservative normalized-name match.
    res_rows = []
    for _, r in linked_bad.iterrows():
        nearest = r["nearest_upstream_reservoir_name"]
        match = res_cov[res_cov["src_id"].astype(str).eq(str(nearest))]
        if len(match) == 0:
            key = norm(nearest)
            match = res_cov[res_cov["src_id"].map(norm).eq(key)]
        if len(match) == 0:
            match = res_cov[res_cov["src_id"].map(norm).apply(lambda x: key and (key in x or x in key))]
        if len(match) > 0:
            m = match.iloc[0]
            res_reach = int(m["reach_id"])
            observed_std = m.get("observed_reservoir_std", np.nan)
            has_observed = str(m.get("has_observed_operation", "")).lower() in ["true", "1", "yes"]
        else:
            res_reach = np.nan
            observed_std = np.nan
            has_observed = False
        res_rows.append((r["q_site"], res_reach, observed_std, has_observed))
    res_map = pd.DataFrame(res_rows, columns=["q_site", "reservoir_reach_id", "observed_reservoir_std_from_inventory", "inventory_has_observed_operation"])
    linked_bad = linked_bad.merge(res_map, on="q_site", how="left")

    sm = station_match[["station_name", "reach_id", "x", "y", "snap_distance_m", "best_line_distance_m", "line_overrode_catchment", "match_method"]]
    linked_bad = linked_bad.merge(sm, left_on="q_site", right_on="station_name", how="left", suffixes=("", "_match"))

    rows = []
    path_rows = []
    for _, r in linked_bad.iterrows():
        site = r["q_site"]
        station_reach = int(float(r["reach_id"]))
        res_reach = r["reservoir_reach_id"]
        if pd.isna(res_reach):
            path = []
            path_reaches_text = ""
            reaches_between = np.nan
            path_length = np.nan
            local_area_added = np.nan
            local_area_added_pct = np.nan
            reaches_between_names = ""
        else:
            res_reach = int(float(res_reach))
            path = build_path(res_reach, station_reach, downstream)
            path_reaches_text = "->".join(map(str, path))
            reaches_between = max(0, len(path) - 2) if path and path[-1] == station_reach else np.nan
            path_length = float(reach_idx.loc[path, "length_km"].sum()) if set(path).issubset(reach_idx.index) else np.nan
            res_area = float(reach_idx.loc[res_reach, "tot_area_km2"]) if res_reach in reach_idx.index else np.nan
            sta_area = float(reach_idx.loc[station_reach, "tot_area_km2"]) if station_reach in reach_idx.index else np.nan
            local_area_added = sta_area - res_area if np.isfinite(res_area) and np.isfinite(sta_area) else np.nan
            local_area_added_pct = 100.0 * local_area_added / res_area if np.isfinite(res_area) and res_area != 0 else np.nan
            reaches_between_names = "；".join(
                [
                    f"{int(pid)}:{reach_idx.loc[pid, 'src_id']}"
                    for pid in path[1:-1]
                    if pid in reach_idx.index
                ]
            )
            for seq, pid in enumerate(path):
                if pid in reach_idx.index:
                    path_rows.append(
                        {
                            "q_site": site,
                            "seq": seq,
                            "reach_id": pid,
                            "src_id": reach_idx.loc[pid, "src_id"],
                            "length_km": reach_idx.loc[pid, "length_km"],
                            "inc_area_km2": reach_idx.loc[pid, "inc_area_km2"],
                            "tot_area_km2": reach_idx.loc[pid, "tot_area_km2"],
                            "dist_to_outlet_km": reach_idx.loc[pid, "dist_to_outlet_km"],
                        }
                    )

        station_position_frac = np.nan
        station_distance_to_chord_m = np.nan
        distance_to_downstream_node_m = np.nan
        distance_to_upstream_node_m = np.nan
        if station_reach in topo_idx.index and station_reach in reach_idx.index and pd.notna(r.get("x")):
            fnode = topo_idx.loc[station_reach, "fnode"]
            tnode = topo_idx.loc[station_reach, "tnode"]
            if fnode in node_idx.index and tnode in node_idx.index:
                ax, ay = float(node_idx.loc[fnode, "x"]), float(node_idx.loc[fnode, "y"])
                bx, by = float(node_idx.loc[tnode, "x"]), float(node_idx.loc[tnode, "y"])
                px, py = float(r["x"]), float(r["y"])
                station_position_frac, station_distance_to_chord_m = point_fraction_on_chord(px, py, ax, ay, bx, by)
                distance_to_upstream_node_m = dist((px, py), (ax, ay))
                distance_to_downstream_node_m = dist((px, py), (bx, by))

        if pd.isna(res_reach):
            risk = "missing_reservoir_reach_mapping"
        elif station_reach == int(res_reach):
            if pd.notna(station_position_frac) and station_position_frac < 0.75:
                risk = "station_inside_reservoir_reach_not_near_outlet"
            else:
                risk = "station_same_reservoir_reach_near_outlet_or_unknown"
        elif pd.notna(local_area_added_pct) and local_area_added_pct > 15:
            risk = "large_intervening_area_between_reservoir_and_station"
        elif pd.notna(reaches_between) and reaches_between >= 1:
            risk = "intervening_reach_mixing"
        else:
            risk = "topology_close_enough_for_direct_release_test"

        rows.append(
            {
                "q_site": site,
                "station_reach_id": station_reach,
                "station_reach_name": reach_idx.loc[station_reach, "src_id"] if station_reach in reach_idx.index else "",
                "reservoir_name": r["nearest_upstream_reservoir_name"],
                "reservoir_reach_id": res_reach,
                "reservoir_reach_name": reach_idx.loc[int(res_reach), "src_id"] if pd.notna(res_reach) and int(res_reach) in reach_idx.index else "",
                "observed_reservoir_std": r.get("observed_reservoir_std", ""),
                "inventory_has_observed_operation": bool(r["inventory_has_observed_operation"]),
                "reported_downstream_order": r["reservoir_downstream_order"],
                "topology_path_reaches": path_reaches_text,
                "topology_path_length_km": path_length,
                "intervening_reach_count": reaches_between,
                "intervening_reach_names": reaches_between_names,
                "area_added_between_res_and_station_km2": local_area_added,
                "area_added_pct_of_res_area": local_area_added_pct,
                "station_position_fraction_fnode_to_tnode": station_position_frac,
                "station_distance_to_upstream_node_m": distance_to_upstream_node_m,
                "station_distance_to_downstream_node_m": distance_to_downstream_node_m,
                "station_distance_to_reach_chord_m": station_distance_to_chord_m,
                "snap_distance_m": r.get("snap_distance_m", np.nan),
                "best_line_distance_m": r.get("best_line_distance_m", np.nan),
                "match_method": r.get("match_method", ""),
                "line_overrode_catchment": r.get("line_overrode_catchment", ""),
                "NSE_log": r["NSE_log"],
                "KGE_2012": r["KGE_2012"],
                "PBIAS_pct": r["PBIAS_pct"],
                "failure_mode": r["failure_mode"],
                "topology_risk_class": risk,
            }
        )

    diag = pd.DataFrame(rows)
    path_df = pd.DataFrame(path_rows)
    diag.to_csv(REPORTS / "bad_reservoir_station_topology_diagnostics.csv", index=False, encoding="utf-8-sig")
    path_df.to_csv(REPORTS / "bad_reservoir_station_topology_paths.csv", index=False, encoding="utf-8-sig")

    risk_summary = (
        diag.groupby("topology_risk_class")
        .agg(
            station_count=("q_site", "count"),
            sites=("q_site", lambda x: "；".join(x)),
            median_NSElog=("NSE_log", lambda x: pd.to_numeric(x, errors="coerce").median()),
            median_KGE=("KGE_2012", lambda x: pd.to_numeric(x, errors="coerce").median()),
        )
        .reset_index()
    )
    risk_summary.to_csv(REPORTS / "bad_reservoir_station_topology_risk_summary.csv", index=False, encoding="utf-8-sig")

    direct_observed = diag[diag["inventory_has_observed_operation"]].copy()
    md = f"""# Reservoir-Station Topology Diagnostic

Run folder: `20260619_53`

## Purpose

Diagnose whether reservoir-related bad stations are topologically suitable for a direct reservoir release-rule test.

This answers the concern that a reservoir operator may look like "special tuning" if the station is not actually measuring near-dam release, or if substantial intervening drainage mixes the reservoir signal.

## Main Diagnostic Table

{markdown_table(diag[[
    "q_site",
    "station_reach_id",
    "station_reach_name",
    "reservoir_name",
    "reservoir_reach_id",
    "topology_path_reaches",
    "topology_path_length_km",
    "intervening_reach_count",
    "area_added_between_res_and_station_km2",
    "area_added_pct_of_res_area",
    "station_position_fraction_fnode_to_tnode",
    "topology_risk_class",
    "failure_mode",
]])}

## Risk Summary

{markdown_table(risk_summary)}

## Direct Observed-Reservoir Candidates

{markdown_table(direct_observed[[
    "q_site",
    "reservoir_name",
    "observed_reservoir_std",
    "topology_risk_class",
    "area_added_pct_of_res_area",
    "station_position_fraction_fnode_to_tnode",
    "NSE_log",
    "KGE_2012",
    "PBIAS_pct",
]])}

## Interpretation

1. `天生桥站` is one reach downstream of `天生桥一级水电站水库`, but the station reach accumulates a large additional drainage area before the station. A pure release-rule correction is therefore not enough; it must be coupled with intervening local runoff.
2. `天峨站` is one reach downstream of `龙滩水电站水库`, with smaller added area than 天生桥 but still not a direct dam-gauge equivalence.
3. `武宣（二）站` is assigned to the same reach as `大藤峡枢纽水库`. This is especially risky: a same-reach station may not represent the modeled reservoir outlet unless it is very near the downstream end of that reach.
4. East River bad stations remain outside this observed-operation workbook: their issue cannot be resolved by the current reservoir table.

## Decision

The current evidence does not support a broad reservoir correction. The next safe direction is:

```text
reservoir release rule only where:
1. operation data overlaps the simulation period, or is used only as a weak prior;
2. station is close to reservoir outlet or intervening drainage is explicitly modeled;
3. same-reach reservoir stations are manually checked against station geometry;
4. missing key reservoirs, especially 新丰江水库 and 枫树坝水库, receive their own operation data.
```

## Files Written

- `reports/bad_reservoir_station_topology_diagnostics.csv`
- `reports/bad_reservoir_station_topology_paths.csv`
- `reports/bad_reservoir_station_topology_risk_summary.csv`
"""
    (REPORTS / "reservoir_station_topology_diagnostic.md").write_text(md, encoding="utf-8-sig")
    (RUN / "README.md").write_text(
        "# 20260619_53 Reservoir-Station Topology Diagnostic\n\n"
        "Audits topology, path length, added drainage area, and station position for reservoir-related bad stations.\n",
        encoding="utf-8-sig",
    )
    (LOGS / "run_log.md").write_text(
        "# Run Log\n\n"
        "- Loaded reach topology, node geometry, station-reach matching, and reservoir coverage reports.\n"
        "- Built reservoir-to-station downstream paths for linked bad stations.\n"
        "- Computed path length, added cumulative drainage area, and station position along matched reach chord.\n"
        "- Conclusion: direct reservoir release-rule tests are structurally risky unless station/outlet relation and intervening drainage are handled.\n",
        encoding="utf-8-sig",
    )
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n## 20260619_53 reservoir-station topology diagnostic\n"
            "- Diagnosed linked bad stations from reservoir reach to station reach using topology_edges, reach_summary, node_summary, and station_reach_match.\n"
            "- Outputs: reservoir_station_topology_diagnostic.md, bad_reservoir_station_topology_diagnostics.csv, bad_reservoir_station_topology_paths.csv.\n"
            "- Key conclusion: direct reservoir correction is structurally risky where station is same reach as reservoir or where substantial intervening area mixes reservoir release with local runoff.\n"
        )

    print("Wrote", REPORTS / "reservoir_station_topology_diagnostic.md")
    print(diag[["q_site", "reservoir_name", "topology_path_reaches", "area_added_pct_of_res_area", "station_position_fraction_fnode_to_tnode", "topology_risk_class"]].to_string(index=False))


if __name__ == "__main__":
    main()
