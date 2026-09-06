from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260619_55"
REPORTS = RUN / "reports"
PROCESSED = RUN / "inputs" / "processed"
LOGS = RUN / "logs"
DAILY_LOG = ROOT / "5_Test" / "20260619.log"

DATASET = ROOT / "0_hydro_sediment_data" / "reservoir" / "2010-2021年中国338座水库高分辨率水位与蓄水量变化数据集"
ATTR_XLSX = DATASET / "01 res_loc" / "01 res_loc" / "reservoir attributes.xlsx"
MODEL_RES = ROOT / "5_Test" / "20260618_1" / "reports" / "input_preprocessing" / "large_reservoir_reach_inventory.csv"
BAD_TOPO = ROOT / "5_Test" / "20260619_53" / "reports" / "bad_reservoir_station_topology_diagnostics.csv"


MANUAL_MODEL_TO_GRAND = {
    "天生桥一级水电站水库": 5692,
    "岩滩水库（红水河）": 5718,
    "岩滩水库(盘阳河)": 5718,
    "新丰江水库": 5736,
    "西津水库": 5763,
    "百色水库": 7028,
    "长洲水利枢纽水库": 7036,
    "龙滩水电站水库": 7196,
    # Not found by direct name in the 338-reservoir attributes in this first audit.
    "枫树坝水库": None,
    "大藤峡枢纽水库": None,
    "六陈水库": None,
    "南水水库": None,
    "龟石水库": None,
    "澄碧河水库": None,
    "抚仙湖水库": None,
}


def norm_name(x: object) -> str:
    s = "" if pd.isna(x) else str(x).lower()
    s = s.replace("reservoir", "").replace("dam", "")
    for token in ["水库", "水电站", "水利枢纽", "枢纽", "（红水河）", "(盘阳河)", "（一级）", "(一级)", "一级"]:
        s = s.replace(token.lower(), "")
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", s)


def extract_grand_id(path: Path) -> int | None:
    m = re.search(r"_(\d+)\.csv$", path.name)
    return int(m.group(1)) if m else None


def list_csv_index(root: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for p in root.rglob(pattern):
        if "__MACOSX" in str(p) or ".DS_Store" in p.name:
            continue
        gid = extract_grand_id(p)
        if gid is not None:
            rows.append({"GRAND_ID": gid, "path": str(p), "bytes": p.stat().st_size})
    return pd.DataFrame(rows).drop_duplicates(["GRAND_ID", "path"])


def read_csv_light(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NA", "999", 999])


def summarize_wse(path: str | Path) -> dict:
    df = read_csv_light(path)
    if not {"year", "month", "day"}.issubset(df.columns):
        return {}
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    value_col = "meanALLwithsalcs2" if "meanALLwithsalcs2" in df.columns else None
    if value_col is None:
        candidates = [c for c in df.columns if c not in {"year", "month", "day", "demical_year"}]
        value_col = candidates[-1] if candidates else None
    vals = pd.to_numeric(df[value_col], errors="coerce") if value_col else pd.Series(dtype=float)
    valid = vals.notna() & df["date"].notna()
    in_model = valid & df["date"].between("2010-01-01", "2021-12-31")
    return {
        "wse_col": value_col,
        "wse_date_min": df.loc[valid, "date"].min(),
        "wse_date_max": df.loc[valid, "date"].max(),
        "wse_valid_days_2010_2021": int(in_model.sum()),
        "wse_valid_months_2010_2021": int(df.loc[in_model, "date"].dt.to_period("M").nunique()),
        "wse_value_min": float(vals[valid].min()) if valid.any() else np.nan,
        "wse_value_max": float(vals[valid].max()) if valid.any() else np.nan,
    }


def summarize_monthly(path: str | Path, value_name: str, period_start="2010-01-01", period_end="2021-12-31") -> dict:
    df = read_csv_light(path)
    date_col = "dates" if "dates" in df.columns else df.columns[0]
    val_col = value_name if value_name in df.columns else df.columns[-1]
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    vals = pd.to_numeric(df[val_col], errors="coerce")
    valid = vals.notna() & df["date"].notna()
    in_period = valid & df["date"].between(period_start, period_end)
    prefix = "rwsc" if value_name == "SATVariation" else "swa"
    return {
        f"{prefix}_col": val_col,
        f"{prefix}_date_min": df.loc[valid, "date"].min(),
        f"{prefix}_date_max": df.loc[valid, "date"].max(),
        f"{prefix}_valid_months_2010_2021": int(df.loc[in_period, "date"].dt.to_period("M").nunique()),
        f"{prefix}_value_min": float(vals[valid].min()) if valid.any() else np.nan,
        f"{prefix}_value_max": float(vals[valid].max()) if valid.any() else np.nan,
    }


def choose_rwsc_index(dataset: Path) -> pd.DataFrame:
    rwsc_files = list_csv_index(dataset / "04 res_rwsc", "rwsc_*.csv")
    if rwsc_files.empty:
        return rwsc_files
    paths = rwsc_files["path"].astype(str)
    rwsc_files["rwsc_method"] = np.select(
        [
            paths.str.contains(r"WSE-SWA\\EM\\obs\\rwsc", regex=True),
            paths.str.contains(r"WSE-SWA\\EM\\noobs\\rwsc", regex=True),
            paths.str.contains(r"WSE-SWA\\SM\\obs\\rwsc\\rwsc", regex=True),
            paths.str.contains(r"WSE-SWA\\SM\\noobs\\rwsc", regex=True),
            paths.str.contains(r"DEM\\93res\\rwsc", regex=True),
            paths.str.contains(r"DEM\\noOBS\\rwsc", regex=True),
            paths.str.contains(r"DEM\\326res\\rwsc", regex=True),
        ],
        ["WSE-SWA_EM_obs", "WSE-SWA_EM_noobs", "WSE-SWA_SM_obs", "WSE-SWA_SM_noobs", "DEM_93res", "DEM_noOBS", "DEM_326res"],
        default="other",
    )
    priority = {
        "WSE-SWA_EM_obs": 1,
        "WSE-SWA_EM_noobs": 2,
        "WSE-SWA_SM_obs": 3,
        "WSE-SWA_SM_noobs": 4,
        "DEM_93res": 5,
        "DEM_noOBS": 6,
        "DEM_326res": 7,
        "other": 9,
    }
    rwsc_files["priority"] = rwsc_files["rwsc_method"].map(priority).fillna(9)
    return rwsc_files.sort_values(["GRAND_ID", "priority", "path"]).drop_duplicates("GRAND_ID")


def monthly_wse(path: str | Path) -> pd.DataFrame:
    df = read_csv_light(path)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    col = "meanALLwithsalcs2" if "meanALLwithsalcs2" in df.columns else df.columns[-2]
    df["wse_m"] = pd.to_numeric(df[col], errors="coerce")
    out = (
        df.dropna(subset=["date", "wse_m"])
        .assign(date_month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("date_month")
        .agg(wse_median_m=("wse_m", "median"), wse_obs_days=("wse_m", "size"))
        .reset_index()
    )
    return out


def monthly_simple(path: str | Path, value_col: str, out_col: str) -> pd.DataFrame:
    df = read_csv_light(path)
    date_col = "dates" if "dates" in df.columns else df.columns[0]
    val_col = value_col if value_col in df.columns else df.columns[-1]
    df["date_month"] = pd.to_datetime(df[date_col], errors="coerce")
    df[out_col] = pd.to_numeric(df[val_col], errors="coerce")
    return df[["date_month", out_col]].dropna(subset=["date_month"])


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
    PROCESSED.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    attrs = pd.read_excel(ATTR_XLSX, sheet_name="338res")
    attrs["name_norm"] = attrs["DAM_NAME"].map(norm_name)

    model_res = pd.read_csv(MODEL_RES, encoding="utf-8-sig")
    bad_topo = pd.read_csv(BAD_TOPO, encoding="utf-8-sig")

    wse_idx = list_csv_index(DATASET / "02 res_wse", "wse_time-series_*.csv")
    wse_idx["wse_mode"] = np.where(wse_idx["path"].str.contains("Enhanced Measurement", regex=False), "enhanced", "standard")
    wse_idx["obs_group"] = np.where(wse_idx["path"].str.contains(r"\\OBS\\", regex=True), "OBS", "noOBS")
    # Prefer enhanced OBS over enhanced noOBS, then standard.
    wse_idx["priority"] = np.select(
        [
            (wse_idx["wse_mode"].eq("enhanced") & wse_idx["obs_group"].eq("OBS")),
            (wse_idx["wse_mode"].eq("enhanced") & wse_idx["obs_group"].eq("noOBS")),
            wse_idx["wse_mode"].eq("standard"),
        ],
        [1, 2, 3],
        default=9,
    )
    wse_best = wse_idx.sort_values(["GRAND_ID", "priority", "path"]).drop_duplicates("GRAND_ID")

    swa_idx = list_csv_index(DATASET / "03 res_swa", "swa_*.csv")
    rwsc_best = choose_rwsc_index(DATASET)

    file_inventory = pd.DataFrame(
        [
            {"data_part": "attributes", "count": 1, "preferred_use": "GRanD_ID/name/location/capacity metadata"},
            {"data_part": "WSE", "count": len(wse_best), "preferred_use": "daily water level, aggregate to monthly state"},
            {"data_part": "SWA", "count": len(swa_idx), "preferred_use": "monthly surface water area"},
            {"data_part": "RWSC", "count": len(rwsc_best), "preferred_use": "monthly reservoir water storage change"},
        ]
    )

    rows = []
    for _, m in model_res.iterrows():
        src = m["src_id"]
        gid = MANUAL_MODEL_TO_GRAND.get(src, "unmapped")
        if gid == "unmapped":
            key = norm_name(src)
            cand = attrs[attrs["name_norm"].eq(key)]
            gid = int(cand["GRAND_ID"].iloc[0]) if len(cand) else None
        attr = attrs[attrs["GRAND_ID"].eq(gid)].iloc[0].to_dict() if gid in set(attrs["GRAND_ID"]) else {}
        rows.append(
            {
                "model_reach_id": m["reach_id"],
                "model_reservoir_name": src,
                "downstream_reach": m.get("downstream_reach", ""),
                "GRAND_ID": gid,
                "GRAND_DAM_NAME": attr.get("DAM_NAME", ""),
                "LONG": attr.get("LONG", np.nan),
                "LAT": attr.get("LAT", np.nan),
                "CAP_REP_MCM": attr.get("CAP_REP", np.nan),
                "AREA_SKM": attr.get("AREA_SKM", np.nan),
                "CATCH_SKM": attr.get("CATCH_SKM", np.nan),
                "has_wse": bool(gid in set(wse_best["GRAND_ID"])) if gid else False,
                "has_swa": bool(gid in set(swa_idx["GRAND_ID"])) if gid else False,
                "has_rwsc": bool(gid in set(rwsc_best["GRAND_ID"])) if gid else False,
                "mapping_note": "manual_or_exact_match" if gid else "not_found_in_338res_first_audit",
            }
        )
    mapping = pd.DataFrame(rows)

    # Summarize all mapped model reservoirs with available files.
    summary_rows = []
    for _, r in mapping.dropna(subset=["GRAND_ID"]).iterrows():
        gid = int(r["GRAND_ID"])
        row = r.to_dict()
        w = wse_best[wse_best["GRAND_ID"].eq(gid)]
        s = swa_idx[swa_idx["GRAND_ID"].eq(gid)]
        q = rwsc_best[rwsc_best["GRAND_ID"].eq(gid)]
        if len(w):
            row.update(summarize_wse(w.iloc[0]["path"]))
            row["wse_path"] = w.iloc[0]["path"]
        if len(s):
            row.update(summarize_monthly(s.iloc[0]["path"], "water_area"))
            row["swa_path"] = s.iloc[0]["path"]
        if len(q):
            row.update(summarize_monthly(q.iloc[0]["path"], "SATVariation"))
            row["rwsc_path"] = q.iloc[0]["path"]
            row["rwsc_method"] = q.iloc[0]["rwsc_method"]
        summary_rows.append(row)
    coverage = pd.DataFrame(summary_rows)

    target_names = [
        "天生桥一级水电站水库",
        "龙滩水电站水库",
        "新丰江水库",
        "枫树坝水库",
        "长洲水利枢纽水库",
        "西津水库",
        "岩滩水库（红水河）",
        "百色水库",
    ]
    target = mapping[mapping["model_reservoir_name"].isin(target_names)].copy()

    # Build compact monthly merged series only for mapped target reservoirs.
    monthly_parts = []
    for _, r in target.dropna(subset=["GRAND_ID"]).iterrows():
        gid = int(r["GRAND_ID"])
        frames = []
        w = wse_best[wse_best["GRAND_ID"].eq(gid)]
        s = swa_idx[swa_idx["GRAND_ID"].eq(gid)]
        q = rwsc_best[rwsc_best["GRAND_ID"].eq(gid)]
        if len(w):
            frames.append(monthly_wse(w.iloc[0]["path"]))
        if len(s):
            frames.append(monthly_simple(s.iloc[0]["path"], "water_area", "swa_km2"))
        if len(q):
            frames.append(monthly_simple(q.iloc[0]["path"], "SATVariation", "rwsc_km3_or_dataset_unit"))
        if frames:
            merged = frames[0]
            for f in frames[1:]:
                merged = merged.merge(f, on="date_month", how="outer")
            merged["GRAND_ID"] = gid
            merged["model_reservoir_name"] = r["model_reservoir_name"]
            merged["GRAND_DAM_NAME"] = r["GRAND_DAM_NAME"]
            monthly_parts.append(merged)
    if monthly_parts:
        monthly_target = pd.concat(monthly_parts, ignore_index=True)
        monthly_target = monthly_target[
            monthly_target["date_month"].between("2010-01-01", "2021-12-31")
        ].sort_values(["model_reservoir_name", "date_month"])
    else:
        monthly_target = pd.DataFrame()

    file_inventory.to_csv(REPORTS / "new_reservoir_dataset_file_inventory.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(REPORTS / "model_reservoir_to_grand_mapping.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(REPORTS / "mapped_reservoir_timeseries_coverage.csv", index=False, encoding="utf-8-sig")
    target.to_csv(REPORTS / "priority_reservoir_mapping_status.csv", index=False, encoding="utf-8-sig")
    if not monthly_target.empty:
        monthly_target.to_csv(PROCESSED / "priority_reservoir_monthly_state_2010_2021.csv", index=False, encoding="utf-8-sig")

    mapped_count = int(mapping["GRAND_ID"].notna().sum())
    full_count = int((mapping["has_wse"] & mapping["has_swa"] & mapping["has_rwsc"]).sum())
    target_txt = markdown_table(
        target[
            [
                "model_reservoir_name",
                "GRAND_ID",
                "GRAND_DAM_NAME",
                "has_wse",
                "has_swa",
                "has_rwsc",
                "mapping_note",
            ]
        ]
    )
    coverage_short = coverage[
        [
            "model_reservoir_name",
            "GRAND_ID",
            "GRAND_DAM_NAME",
            "wse_valid_months_2010_2021",
            "swa_valid_months_2010_2021",
            "rwsc_valid_months_2010_2021",
            "rwsc_method",
        ]
    ].copy()
    md = f"""# New 338-Reservoir Dataset Parsing And Use Plan

Run folder: `20260619_55`

Dataset:

```text
{DATASET}
```

## Dataset Meaning

This is the Shen et al. remotely sensed dataset:

```text
High-resolution water level and storage variation datasets for 338 reservoirs in China during 2010-2021
```

It contains:

```text
01 res_loc   reservoir metadata and GRanD_ID locations
02 res_wse   water surface elevation time series
03 res_swa   monthly surface water area
04 res_rwsc  monthly reservoir water storage change
```

Important difference from the previous `珠江水情水库.xlsx`:

```text
previous workbook: 2023-2026 daily H/Qin/Qout, no overlap with model period
new dataset:       2010-2021 H/SWA/RWSC, overlaps model period but has no Qin/Qout
```

Therefore the old workbook should be **abandoned as the primary reservoir input**. It can remain only as a supplemental diagnostic reference.

## File Inventory

{markdown_table(file_inventory)}

## Model Reservoir Mapping

Current model large-reservoir reaches:

```text
{len(mapping)}
```

Mapped to a GRanD_ID in this first audit:

```text
{mapped_count}
```

Mapped with WSE+SWA+RWSC available:

```text
{full_count}
```

## Priority Reservoirs

{target_txt}

## Time-Series Coverage For Mapped Reservoirs

{markdown_table(coverage_short)}

## How This Changes The Reservoir Strategy

This dataset should not be used as direct observed release.

It should be used as a **storage-change constraint**:

```text
Qout_r,t = Qin_r,t + Qlocal_r,t + Qboundary_r,t - dS_r,t / dt
```

where:

```text
dS_r,t  comes from RWSC, if available
H_r,t   comes from WSE and can define storage state / seasonal pool behavior
SWA_r,t comes from monthly water area and can help estimate evaporation over reservoir surface
```

This is more physically appropriate than the previous release-ratio prior because the data overlap the model period.

## Proposed Future Pipeline

Do not insert this into the model yet. Next steps should be:

```text
1. manually verify GRanD_ID mapping for each model reservoir reach;
2. prefer WSE-SWA RWSC over DEM-only RWSC where available;
3. convert RWSC from dataset units to cfs-equivalent monthly storage flux;
4. align reservoir reach topology:
   reservoir storage change should apply at the modeled reservoir reach,
   not automatically at a downstream station;
5. test only mapped reservoirs first:
   天生桥一级, 龙滩, 新丰江, 岩滩, 西津, 百色, 长洲;
6. keep 枫树坝 marked unresolved unless a GRanD_ID or external dataset match is confirmed;
7. evaluate effect only as a process-layer term:
   Qout = Qin + Qlocal - dS/dt,
   not as post-hoc station residual correction.
```

## Key Caveats

```text
1. It covers 2010-2021, not 2022.
2. It provides storage change, not observed outflow.
3. It may not include every model reservoir, notably 枫树坝 in this first audit.
4. Unit handling for SATVariation must be confirmed before any model run.
5. Same-reach reservoir/station problems, especially 武宣/大藤峡, still need topology handling.
```

## Files Written

- `reports/new_reservoir_dataset_file_inventory.csv`
- `reports/model_reservoir_to_grand_mapping.csv`
- `reports/mapped_reservoir_timeseries_coverage.csv`
- `reports/priority_reservoir_mapping_status.csv`
- `inputs/processed/priority_reservoir_monthly_state_2010_2021.csv`
"""
    (REPORTS / "new_338_reservoir_dataset_parse_and_plan.md").write_text(md, encoding="utf-8-sig")
    (RUN / "README.md").write_text(
        "# 20260619_55 New 338-Reservoir Dataset Parse And Plan\n\n"
        "Parses the 2010-2021 China 338-reservoir WSE/SWA/RWSC dataset, maps it to current model reservoir reaches, and defines a non-computational integration plan. The previous 2023-2026 water-regime workbook is demoted to supplemental diagnostic status.\n",
        encoding="utf-8-sig",
    )
    (LOGS / "run_log.md").write_text(
        "# Run Log\n\n"
        "- Read dataset readme and file structure.\n"
        "- Parsed reservoir attributes, WSE/SWA/RWSC file indices, and current model reservoir inventory.\n"
        "- Created first GRanD_ID mapping for model large reservoirs.\n"
        "- Generated compact monthly target-reservoir state table for 2010-2021, but did not insert it into the model.\n"
        "- Marked previous 2023-2026 reservoir workbook as deprecated for primary reservoir modeling.\n",
        encoding="utf-8-sig",
    )
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n## 20260619_55 new 338-reservoir dataset parse and plan\n"
            "- Parsed `2010-2021年中国338座水库高分辨率水位与蓄水量变化数据集`.\n"
            "- Dataset contains GRanD_ID metadata, WSE, SWA, and RWSC; it overlaps 2010-2021 but does not provide observed Qin/Qout.\n"
            f"- Model reservoir reaches mapped to GRanD_ID in first audit: {mapped_count}/{len(mapping)}; WSE+SWA+RWSC available: {full_count}/{len(mapping)}.\n"
            "- Key matches include 天生桥一级=5692, 岩滩=5718, 新丰江=5736, 西津=5763, 百色=7028, 长洲=7036, 龙滩=7196. 枫树坝 remains unresolved in this first audit.\n"
            "- Previous `珠江水情水库.xlsx` is no longer the primary reservoir data source; keep only as supplemental diagnostic reference.\n"
            "- No model calculation was run. Next reservoir integration should use RWSC as a process-layer storage-change constraint, not post-hoc correction.\n"
        )

    print("Wrote", REPORTS / "new_338_reservoir_dataset_parse_and_plan.md")
    print(f"mapped {mapped_count}/{len(mapping)}, full WSE+SWA+RWSC {full_count}/{len(mapping)}")
    print(target[["model_reservoir_name", "GRAND_ID", "GRAND_DAM_NAME", "has_wse", "has_swa", "has_rwsc"]].to_string(index=False))


if __name__ == "__main__":
    main()
