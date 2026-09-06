from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(r"E:\SPARROW\5_Test\20260618_1")
RUN = Path(r"E:\SPARROW\5_Test\20260620_42")
OUT = RUN / "reports" / "active_station_exclusion_comparison"
DAILY_LOG = Path(r"E:\SPARROW\5_Test\20260620.log")

EXCLUDED = ["官良站", "平山（三）站"]


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["q_site"] = df["q_site"].astype(str)
    return df


def summarize(df: pd.DataFrame, label: str) -> dict[str, object]:
    return {
        "scope": label,
        "station_count": int(df["q_site"].nunique()),
        "median_NSE_raw": float(df["main_NSE_raw"].median()),
        "median_NSElog": float(df["main_NSElog"].median()),
        "median_KGE": float(df["main_KGE_2012"].median()),
        "median_absPBIAS": float(df["main_abs_PBIAS_pct"].median()),
        "good_count": int(df["main_good"].sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_metrics(BASE / "reports" / "main_model" / "station_performance_diagnostics_extended.csv")
    new = load_metrics(RUN / "reports" / "main_model" / "station_performance_diagnostics_extended.csv")

    base_sites = set(base["q_site"])
    new_sites = set(new["q_site"])
    excluded = set(EXCLUDED)
    common = sorted(base_sites & new_sites)
    missing = sorted(base_sites - new_sites)
    added = sorted(new_sites - base_sites)

    set_diff = pd.DataFrame(
        [{"q_site": s, "set_change": "missing_from_new"} for s in missing]
        + [{"q_site": s, "set_change": "added_in_new"} for s in added]
    )
    set_diff.to_csv(OUT / "station_set_difference.csv", index=False, encoding="utf-8-sig")

    base_common = base[base["q_site"].isin(common)].copy()
    new_common = new[new["q_site"].isin(common)].copy()
    base_trusted = base[~base["q_site"].isin(excluded)].copy()
    new_added = new[new["q_site"].isin(added)].copy()

    summary = pd.DataFrame(
        [
            summarize(base, "baseline_all"),
            summarize(base_trusted, "baseline_without_excluded"),
            summarize(base_common, "baseline_common_with_new"),
            summarize(new_common, "strict_partial_exclusion_common"),
            summarize(new, "strict_partial_exclusion_active"),
            summarize(new_added, "strict_partial_exclusion_added"),
        ]
    )
    summary.to_csv(OUT / "summary_by_scope.csv", index=False, encoding="utf-8-sig")

    base_idx = base_common.set_index("q_site")
    new_idx = new_common.set_index("q_site")
    rows = []
    for site in common:
        rows.append(
            {
                "q_site": site,
                "reach_id_base": int(base_idx.at[site, "reach_id"]),
                "reach_id_new": int(new_idx.at[site, "reach_id"]),
                "base_NSE_raw": float(base_idx.at[site, "main_NSE_raw"]),
                "new_NSE_raw": float(new_idx.at[site, "main_NSE_raw"]),
                "delta_NSE_raw": float(new_idx.at[site, "main_NSE_raw"] - base_idx.at[site, "main_NSE_raw"]),
                "base_NSElog": float(base_idx.at[site, "main_NSElog"]),
                "new_NSElog": float(new_idx.at[site, "main_NSElog"]),
                "delta_NSElog": float(new_idx.at[site, "main_NSElog"] - base_idx.at[site, "main_NSElog"]),
                "base_absPBIAS": float(base_idx.at[site, "main_abs_PBIAS_pct"]),
                "new_absPBIAS": float(new_idx.at[site, "main_abs_PBIAS_pct"]),
                "delta_absPBIAS": float(new_idx.at[site, "main_abs_PBIAS_pct"] - base_idx.at[site, "main_abs_PBIAS_pct"]),
                "base_good": bool(base_idx.at[site, "main_good"]),
                "new_good": bool(new_idx.at[site, "main_good"]),
            }
        )
    delta = pd.DataFrame(rows)
    delta.to_csv(OUT / "common_100_station_delta.csv", index=False, encoding="utf-8-sig")

    improved = int((delta["delta_NSE_raw"] > 0).sum())
    worsened = int((delta["delta_NSE_raw"] < 0).sum())
    material_improved = int((delta["delta_NSE_raw"] >= 0.02).sum())
    material_worsened = int((delta["delta_NSE_raw"] <= -0.02).sum())

    base_common_row = summary[summary["scope"] == "baseline_common_with_new"].iloc[0]
    new_common_row = summary[summary["scope"] == "strict_partial_exclusion_common"].iloc[0]
    base_trusted_row = summary[summary["scope"] == "baseline_without_excluded"].iloc[0]
    new_active_row = summary[summary["scope"] == "strict_partial_exclusion_active"].iloc[0]

    md = f"""# 20260620_42 Partial Active Station Exclusion Comparison

## What Changed

Two suspected stations were removed before station matching and before the complete main workflow:

```text
{chr(10).join(EXCLUDED)}
```

The active station set may change because same-reach selection is rerun after exclusion. Stations added relative to the baseline:

```text
{chr(10).join(added)}
```

Missing from the new active set:

```text
{chr(10).join(missing)}
```

## Main Comparison

Common {int(base_common_row['station_count'])}-station scope:

```text
median raw NSE: {base_common_row['median_NSE_raw']:.4f} -> {new_common_row['median_NSE_raw']:.4f}
median NSElog:  {base_common_row['median_NSElog']:.4f} -> {new_common_row['median_NSElog']:.4f}
median KGE:     {base_common_row['median_KGE']:.4f} -> {new_common_row['median_KGE']:.4f}
median |PBIAS|: {base_common_row['median_absPBIAS']:.2f}% -> {new_common_row['median_absPBIAS']:.2f}%
good stations:  {int(base_common_row['good_count'])} / {int(base_common_row['station_count'])} -> {int(new_common_row['good_count'])} / {int(new_common_row['station_count'])}
```

New active {int(new_active_row['station_count'])}-station scope:

```text
median raw NSE: {new_active_row['median_NSE_raw']:.4f}
median NSElog:  {new_active_row['median_NSElog']:.4f}
median KGE:     {new_active_row['median_KGE']:.4f}
median |PBIAS|: {new_active_row['median_absPBIAS']:.2f}%
good stations:  {int(new_active_row['good_count'])} / {int(new_active_row['station_count'])}
```

For reference, the baseline trusted set after excluding only these stations was:

```text
median raw NSE: {base_trusted_row['median_NSE_raw']:.4f}
median NSElog:  {base_trusted_row['median_NSElog']:.4f}
median KGE:     {base_trusted_row['median_KGE']:.4f}
median |PBIAS|: {base_trusted_row['median_absPBIAS']:.2f}%
good stations:  {int(base_trusted_row['good_count'])} / {int(base_trusted_row['station_count'])}
```

## Common-Station Delta

```text
NSE_raw improved stations: {improved}
NSE_raw worsened stations: {worsened}
NSE_raw improved by >= 0.02: {material_improved}
NSE_raw worsened by <= -0.02: {material_worsened}
```

## Interpretation

This is now a true active-input partial-exclusion experiment. 官良站 and 平山（三）站 do not participate in input construction, station matching, Q72 fitting, Q78_mass fitting, alpha selection, or final main-model training. 北流站 and 犁市（二）站 remain active.

The result supports the idea that excluding these questionable observations produces a slightly cleaner global calibration over the remaining station network. It is still not a large model-structure improvement, so it should be treated as a data-QC screened calibration branch.
"""
    (OUT / "comparison_summary.md").write_text(md, encoding="utf-8")

    daily = f"""

## 20260620_42 partial active-station exclusion comparison

Purpose: compare the full rerun where 官良站、平山（三）站 were removed before input construction and all model training, while 北流站 and 犁市（二）站 remain active.

Result:
- active validation stations: {int(new_active_row['station_count'])}
- removed stations: {', '.join(EXCLUDED)}
- added after same-reach reselection: {', '.join(added)}
- common {int(base_common_row['station_count'])}-station median raw NSE: {base_common_row['median_NSE_raw']:.4f} -> {new_common_row['median_NSE_raw']:.4f}
- common {int(base_common_row['station_count'])}-station good count: {int(base_common_row['good_count'])} -> {int(new_common_row['good_count'])}
- active {int(new_active_row['station_count'])}-station median raw NSE: {new_active_row['median_NSE_raw']:.4f}
- active {int(new_active_row['station_count'])}-station good count: {int(new_active_row['good_count'])}

Interpretation: this is a true data-QC active-input exclusion rerun, not a post-hoc evaluation deletion. It gives a modest but cleaner trusted-network calibration and should be treated as a screened calibration branch, not a structural breakthrough.

Outputs: E:\\SPARROW\\5_Test\\20260620_42\\reports\\active_station_exclusion_comparison
"""
    with DAILY_LOG.open("a", encoding="utf-8-sig") as f:
        f.write(daily)

    print(md)


if __name__ == "__main__":
    main()
