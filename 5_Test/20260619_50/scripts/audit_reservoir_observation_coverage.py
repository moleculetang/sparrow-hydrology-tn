from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260619_50"
REPORTS = RUN / "reports"
LOGS = RUN / "logs"
INPUTS = RUN / "inputs"
DAILY_LOG = ROOT / "5_Test" / "20260619.log"

MODEL_BASE = ROOT / "5_Test" / "20260618_1" / "reports"
RES_OBS = ROOT / "5_Test" / "20260619_49"


MODEL_RESERVOIRS = MODEL_BASE / "input_preprocessing" / "large_reservoir_reach_inventory.csv"
INFLUENCE = MODEL_BASE / "main_model" / "reservoir_reach_influence_inventory.csv"
STATION_DETAIL = MODEL_BASE / "main_model" / "reservoir_related_station_detail.csv"
OBS_SUMMARY = RES_OBS / "reports" / "reservoir_water_level_authenticity_summary.csv"


def norm_name(x: object) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.strip()
    for token in [
        "水库",
        "水电站",
        "水利枢纽",
        "枢纽",
        "（红水河）",
        "(盘阳河)",
        "（坝上）",
        "(一级)",
        "（一级）",
        "一级",
    ]:
        s = s.replace(token, "")
    s = re.sub(r"[()\[\]（）\s·]", "", s)
    return s


MANUAL_OBS_ALIAS = {
    "天生桥一级水电站水库": "天生桥一级",
    "龙滩水电站水库": "龙滩",
    "岩滩水库（红水河）": "岩滩",
    "岩滩水库(盘阳河)": "岩滩",
    "长洲水利枢纽水库": "长洲枢纽",
    "大藤峡枢纽水库": "大藤峡水利枢纽",
    "西津水库": "西津",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    INPUTS.mkdir(parents=True, exist_ok=True)

    model_res = read_csv(MODEL_RESERVOIRS)
    infl = read_csv(INFLUENCE)
    stations = read_csv(STATION_DETAIL)
    obs = read_csv(OBS_SUMMARY)

    obs["obs_key"] = obs["reservoir_std"].map(norm_name)
    obs_lookup = {row.obs_key: row.reservoir_std for row in obs.itertuples()}

    def map_observed(src_id: object) -> str | None:
        src = "" if pd.isna(src_id) else str(src_id)
        if src in MANUAL_OBS_ALIAS:
            return MANUAL_OBS_ALIAS[src]
        key = norm_name(src)
        if key in obs_lookup:
            return obs_lookup[key]
        for ok, oname in obs_lookup.items():
            if key and ok and (key in ok or ok in key):
                return oname
        return None

    model_res["observed_reservoir_std"] = model_res["src_id"].map(map_observed)
    model_res["has_observed_operation"] = model_res["observed_reservoir_std"].notna()
    model_res = model_res.merge(
        obs[
            [
                "reservoir_std",
                "n_records",
                "date_start",
                "date_end",
                "h_min",
                "h_max",
                "outlier_count",
                "water_level_status",
            ]
        ],
        left_on="observed_reservoir_std",
        right_on="reservoir_std",
        how="left",
    )
    model_res["operation_data_action"] = model_res["has_observed_operation"].map(
        {
            True: "can_test_low_parameter_release_rule_for_this_reservoir",
            False: "no_operation_observation_do_not_train_release_rule_from_this_excel",
        }
    )

    # Some observed reservoirs may not be in the large-reservoir model inventory.
    mapped_obs = set(model_res["observed_reservoir_std"].dropna())
    obs["in_model_large_reservoir_inventory"] = obs["reservoir_std"].isin(mapped_obs)
    obs["model_mapping_note"] = obs["in_model_large_reservoir_inventory"].map(
        {
            True: "mapped_to_model_large_reservoir_reach",
            False: "not_mapped_to_current_large_reservoir_inventory_or_outside_model_scope",
        }
    )

    # Link station performance detail to whether its upstream reservoir has observed operation data.
    stations["observed_reservoir_std"] = stations["nearest_upstream_reservoir_name"].map(map_observed)
    stations["has_observed_upstream_reservoir_operation"] = stations["observed_reservoir_std"].notna()
    stations["good_bool"] = stations["good"].astype(str).str.lower().isin(["true", "1", "yes"])
    stations["reservoir_downstream_order_num"] = pd.to_numeric(
        stations["reservoir_downstream_order"], errors="coerce"
    )
    stations["direct_or_one_step"] = stations["reservoir_downstream_order_num"].le(1)
    stations["operation_data_use_class"] = "no_observed_operation_for_linked_reservoir"
    stations.loc[
        stations["has_observed_upstream_reservoir_operation"] & stations["direct_or_one_step"],
        "operation_data_use_class",
    ] = "suitable_for_direct_release_rule_diagnosis"
    stations.loc[
        stations["has_observed_upstream_reservoir_operation"] & ~stations["direct_or_one_step"],
        "operation_data_use_class",
    ] = "observed_reservoir_but_effect_is_indirect_or_farther_downstream"

    group_summary = (
        stations.groupby(["has_observed_upstream_reservoir_operation", "operation_data_use_class"], dropna=False)
        .agg(
            n_stations=("q_site", "count"),
            good_count=("good_bool", "sum"),
            bad_count=("good_bool", lambda x: int((~x).sum())),
            median_NSE_log=("NSE_log", lambda x: pd.to_numeric(x, errors="coerce").median()),
            median_KGE=("KGE_2012", lambda x: pd.to_numeric(x, errors="coerce").median()),
            median_abs_PBIAS=("PBIAS_pct", lambda x: pd.to_numeric(x, errors="coerce").abs().median()),
        )
        .reset_index()
    )

    bad_stations = stations[~stations["good_bool"]].copy()
    bad_focus = bad_stations[
        [
            "q_site",
            "reach_id",
            "reach_name",
            "nearest_upstream_reservoir_name",
            "reservoir_downstream_order",
            "observed_reservoir_std",
            "operation_data_use_class",
            "NSE_log",
            "KGE_2012",
            "PBIAS_pct",
            "failure_mode",
        ]
    ].sort_values(["operation_data_use_class", "nearest_upstream_reservoir_name", "q_site"])

    model_res.to_csv(REPORTS / "model_reservoir_observation_coverage.csv", index=False, encoding="utf-8-sig")
    obs.to_csv(REPORTS / "observed_reservoir_model_mapping.csv", index=False, encoding="utf-8-sig")
    stations.to_csv(REPORTS / "reservoir_related_station_operation_coverage.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(REPORTS / "reservoir_operation_coverage_station_summary.csv", index=False, encoding="utf-8-sig")
    bad_focus.to_csv(REPORTS / "bad_reservoir_related_station_action_table.csv", index=False, encoding="utf-8-sig")

    n_model = len(model_res)
    n_model_obs = int(model_res["has_observed_operation"].sum())
    n_obs = len(obs)
    n_obs_mapped = int(obs["in_model_large_reservoir_inventory"].sum())
    n_station = len(stations)
    linked_mask = stations["nearest_upstream_reservoir_name"].notna() & stations[
        "nearest_upstream_reservoir_name"
    ].astype(str).str.strip().ne("")
    bad_linked_mask = linked_mask & (~stations["good_bool"])
    n_linked_station = int(linked_mask.sum())
    n_bad_linked_station = int(bad_linked_mask.sum())
    n_linked_station_obs = int(
        (linked_mask & stations["has_observed_upstream_reservoir_operation"]).sum()
    )
    n_bad_linked_station_obs = int(
        (bad_linked_mask & stations["has_observed_upstream_reservoir_operation"]).sum()
    )
    n_station_obs = int(stations["has_observed_upstream_reservoir_operation"].sum())
    n_bad = len(bad_stations)
    n_bad_obs = int(bad_stations["has_observed_upstream_reservoir_operation"].sum())

    direct_candidates = stations[
        (stations["operation_data_use_class"] == "suitable_for_direct_release_rule_diagnosis")
        & (~stations["good_bool"])
    ]

    cand_txt = "None" if not len(direct_candidates) else _markdown_table(
        direct_candidates[
            [
                "q_site",
                "reach_id",
                "reach_name",
                "nearest_upstream_reservoir_name",
                "reservoir_downstream_order",
                "observed_reservoir_std",
                "NSE_log",
                "KGE_2012",
                "PBIAS_pct",
                "failure_mode",
            ]
        ]
    )
    md = f"""# Reservoir Observation Coverage Audit

Run folder: `20260619_50`

Purpose: diagnose whether the observed reservoir operation workbook can legitimately support a reservoir release-rule experiment for the model's reservoir-related bad stations.

## Main Findings

1. The current model large-reservoir inventory has **{n_model} reservoir reaches**.
2. The cleaned operation workbook from `_49` maps to **{n_model_obs} / {n_model} model reservoir reaches**.
3. The workbook contains **{n_obs} observed reservoir names**, of which **{n_obs_mapped}** map to the current large-reservoir inventory.
4. Among all current stations, **{n_linked_station} / {n_station}** have a linked upstream/model reservoir within the current influence inventory.
5. Among these linked-reservoir stations, **{n_linked_station_obs} / {n_linked_station}** have observed operation data in this workbook.
6. Among linked-reservoir bad stations, only **{n_bad_linked_station_obs} / {n_bad_linked_station}** have observed operation data for their linked upstream reservoir.

This means the workbook is useful, but it cannot support a global reservoir-operator replacement. It supports only a **targeted mechanism test** for reservoirs that are both:

```text
in the model reservoir inventory
and present in the cleaned operation workbook
and topologically close enough to the station being diagnosed
```

## Why A Global Reservoir Correction Looked Like Special Tuning

The earlier concern was justified. A single reservoir operator applied broadly is not well-posed because:

1. Many model reservoir reaches lack observed water-level/inflow/outflow data in this workbook.
2. Some observed reservoirs are not in the current large-reservoir inventory or are outside the present model scope.
3. Some stations are 2 reaches or more downstream, where the observed reservoir signal is mixed with intervening local runoff and tributaries.
4. East River bad stations are mostly linked to `新丰江水库` and `枫树坝水库`, but those two key reservoirs are **not** in this workbook.

Therefore, the problem is not that water level is fake. The problem is **coverage and linkage**:

```text
real operation data exists only for part of the reservoir set
and should only be used where model reach, reservoir observation, and station topology align
```

## Direct Bad-Station Candidates For A Reservoir Release-Rule Test

These bad stations have observed operation data for their nearest upstream reservoir and are direct/one-step downstream:

{cand_txt}

## Files Written

- `reports/model_reservoir_observation_coverage.csv`
- `reports/observed_reservoir_model_mapping.csv`
- `reports/reservoir_related_station_operation_coverage.csv`
- `reports/reservoir_operation_coverage_station_summary.csv`
- `reports/bad_reservoir_related_station_action_table.csv`

## Decision

Do not build a global reservoir correction from `珠江水情水库.xlsx`.

Next safe experiment should be one of:

1. **Direct observed-operation release-rule test** only for mapped reservoirs with close downstream bad stations.
2. **No-operation-data reservoir diagnosis** for East River reservoirs (`新丰江水库`, `枫树坝水库`) using external reservoir data after the user supplements them.
3. **Topology-distance attenuation experiment** that limits reservoir operators to reservoir reach and immediate downstream reach, then checks whether farther downstream stations stop being disturbed.
"""

    (REPORTS / "reservoir_observation_coverage_audit.md").write_text(md, encoding="utf-8-sig")
    (RUN / "README.md").write_text(
        "# 20260619_50 Reservoir Observation Coverage Audit\n\n"
        "Audits whether cleaned reservoir operation observations can be mapped to current model reservoir reaches and reservoir-related stations.\n",
        encoding="utf-8-sig",
    )
    (LOGS / "run_log.md").write_text(
        "# Run Log\n\n"
        "- Loaded model large-reservoir inventory from 20260618_1.\n"
        "- Loaded cleaned operation observation summary from 20260619_49.\n"
        "- Mapped observed reservoirs to model reservoir reaches using conservative aliases.\n"
        "- Classified reservoir-related stations by whether their upstream reservoir has operation observations and by downstream order.\n"
        "- Conclusion: operation workbook supports targeted reservoir tests only, not a global operator replacement.\n",
        encoding="utf-8-sig",
    )

    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n## 20260619_50 reservoir operation observation coverage audit\n"
            f"- Model large-reservoir reaches: {n_model}; mapped to cleaned operation observations: {n_model_obs}.\n"
            f"- Observed reservoir names in workbook: {n_obs}; mapped to model large-reservoir inventory: {n_obs_mapped}.\n"
            f"- Stations with linked upstream/model reservoir: {n_linked_station}/{n_station}; linked-reservoir stations with observed operation data: {n_linked_station_obs}/{n_linked_station}.\n"
            f"- Bad linked-reservoir stations with observed operation data: {n_bad_linked_station_obs}/{n_bad_linked_station}; all bad stations with observed operation data: {n_bad_obs}/{n_bad}.\n"
            "- Key conclusion: the workbook is not suitable for a global reservoir correction. It can only support targeted release-rule tests where observed reservoir, model reach, and station topology align.\n"
            "- East River problem reservoirs 新丰江水库 and 枫树坝水库 are not covered by this workbook, so their bad stations cannot be solved by this specific water-level table.\n"
        )

    print("Wrote", REPORTS / "reservoir_observation_coverage_audit.md")
    print(f"model reservoir reaches: {n_model}; observed mapped: {n_model_obs}")
    print(
        f"linked-reservoir stations: {n_linked_station}/{n_station}; "
        f"linked+observed: {n_linked_station_obs}/{n_linked_station}; "
        f"bad linked+observed: {n_bad_linked_station_obs}/{n_bad_linked_station}"
    )
    print(group_summary.to_string(index=False))


def _markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    rows = []
    for _, r in frame.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                row.append("")
            elif isinstance(v, float):
                row.append(f"{v:.3f}")
            else:
                row.append(str(v))
        rows.append(row)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()
