from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test")
OUT = ROOT / "20260620_42" / "reports" / "station_subset_decision"
DAILY_LOG = ROOT / "20260620.log"


def load_summary(run: str) -> pd.DataFrame:
    path = ROOT / run / "reports" / "active_station_exclusion_comparison" / "summary_by_scope.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.insert(0, "run", run)
    return df


def pick_scope(df: pd.DataFrame, prefix: str) -> pd.Series:
    hit = df[df["scope"].astype(str).str.startswith(prefix)]
    if hit.empty:
        available = ", ".join(df["scope"].astype(str).tolist())
        raise ValueError(f"Missing scope prefix {prefix!r}. Available scopes: {available}")
    return hit.iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    s41 = load_summary("20260620_41")
    s42 = load_summary("20260620_42")

    rows = []

    b41_common = pick_scope(s41, "baseline_common_with_new")
    r41_common = pick_scope(s41, "strict_exclusion_common")
    r41_active = pick_scope(s41, "strict_exclusion_active")

    b42_common = pick_scope(s42, "baseline_common_with_new")
    r42_common = pick_scope(s42, "strict_partial_exclusion_common")
    r42_active = pick_scope(s42, "strict_partial_exclusion_active")

    rows.extend(
        [
            {
                "strategy": "baseline_without_four_candidate_scope",
                "description": "Baseline 20260618_1 evaluated after removing 北流、官良、平山（三）、犁市（二） from the station set.",
                "station_scope": "100 common/trusted stations",
                "median_NSE_raw": b41_common["median_NSE_raw"],
                "median_NSElog": b41_common["median_NSElog"],
                "median_KGE": b41_common["median_KGE"],
                "median_absPBIAS": b41_common["median_absPBIAS"],
                "good_count": b41_common["good_count"],
                "station_count": b41_common["station_count"],
                "decision_role": "reference",
            },
            {
                "strategy": "strict_exclude_all_4",
                "description": "20260620_41: remove 北流、官良、平山（三）、犁市（二） before input construction and full model training.",
                "station_scope": "100 common/trusted stations",
                "median_NSE_raw": r41_common["median_NSE_raw"],
                "median_NSElog": r41_common["median_NSElog"],
                "median_KGE": r41_common["median_KGE"],
                "median_absPBIAS": r41_common["median_absPBIAS"],
                "good_count": r41_common["good_count"],
                "station_count": r41_common["station_count"],
                "decision_role": "best_screened_calibration_candidate",
            },
            {
                "strategy": "strict_exclude_all_4_active_set",
                "description": "20260620_41 active station set after same-reach reselection; adds 坪石（二）站 and 罗定古榄站.",
                "station_scope": "102 active stations",
                "median_NSE_raw": r41_active["median_NSE_raw"],
                "median_NSElog": r41_active["median_NSElog"],
                "median_KGE": r41_active["median_KGE"],
                "median_absPBIAS": r41_active["median_absPBIAS"],
                "good_count": r41_active["good_count"],
                "station_count": r41_active["station_count"],
                "decision_role": "best_active_set_result",
            },
            {
                "strategy": "baseline_without_two_candidate_scope",
                "description": "Baseline 20260618_1 evaluated after removing 官良、平山（三） from the station set.",
                "station_scope": "102 common/trusted stations",
                "median_NSE_raw": b42_common["median_NSE_raw"],
                "median_NSElog": b42_common["median_NSElog"],
                "median_KGE": b42_common["median_KGE"],
                "median_absPBIAS": b42_common["median_absPBIAS"],
                "good_count": b42_common["good_count"],
                "station_count": b42_common["station_count"],
                "decision_role": "reference",
            },
            {
                "strategy": "strict_exclude_worst_2_keep_beiliu_lishi",
                "description": "20260620_42: remove 官良、平山（三） before input construction; keep 北流 and 犁市（二） active.",
                "station_scope": "102 common/trusted stations",
                "median_NSE_raw": r42_common["median_NSE_raw"],
                "median_NSElog": r42_common["median_NSElog"],
                "median_KGE": r42_common["median_KGE"],
                "median_absPBIAS": r42_common["median_absPBIAS"],
                "good_count": r42_common["good_count"],
                "station_count": r42_common["station_count"],
                "decision_role": "not_promoted",
            },
            {
                "strategy": "strict_exclude_worst_2_active_set",
                "description": "20260620_42 active station set after same-reach reselection; adds 罗定古榄站.",
                "station_scope": "103 active stations",
                "median_NSE_raw": r42_active["median_NSE_raw"],
                "median_NSElog": r42_active["median_NSElog"],
                "median_KGE": r42_active["median_KGE"],
                "median_absPBIAS": r42_active["median_absPBIAS"],
                "good_count": r42_active["good_count"],
                "station_count": r42_active["station_count"],
                "decision_role": "not_promoted",
            },
        ]
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "strict_station_subset_decision_summary.csv", index=False, encoding="utf-8-sig")

    md = f"""# Strict Station-Subset Decision Summary

## Question

If several difficult stations may have observation or representativeness problems, should they be excluded from active model calibration, or should only the most suspicious subset be excluded?

## Strict Experiments

```text
20260620_41:
  exclude 北流站、官良站、平山（三）站、犁市（二）站 before input construction.

20260620_42:
  exclude 官良站、平山（三）站 before input construction;
  keep 北流站 and 犁市（二）站 active.
```

## Key Result

Strict exclude-all-4 on the common 100 trusted stations:

```text
median raw NSE: {r41_common['median_NSE_raw']:.4f}
median NSElog:  {r41_common['median_NSElog']:.4f}
median KGE:     {r41_common['median_KGE']:.4f}
median |PBIAS|: {r41_common['median_absPBIAS']:.2f}%
good stations:  {int(r41_common['good_count'])} / {int(r41_common['station_count'])}
```

Strict exclude-worst-2 but keep 北流/犁市 on the common 102 trusted stations:

```text
median raw NSE: {r42_common['median_NSE_raw']:.4f}
median NSElog:  {r42_common['median_NSElog']:.4f}
median KGE:     {r42_common['median_KGE']:.4f}
median |PBIAS|: {r42_common['median_absPBIAS']:.2f}%
good stations:  {int(r42_common['good_count'])} / {int(r42_common['station_count'])}
```

## Decision

The partial-retention version is not better. It keeps more data, but the common-station performance does not improve and the good-station count drops in the 102-station comparison.

Recommended branch:

```text
20260620_41 strict_exclude_all_4
```

Interpretation:

```text
Use it as a data-QC screened calibration branch, not as a structural model breakthrough.
```

The evidence so far suggests that if these four stations are considered questionable, excluding all four is cleaner than keeping 北流站 and 犁市（二）站 active.
"""
    (OUT / "strict_station_subset_decision_summary.md").write_text(md, encoding="utf-8")

    with DAILY_LOG.open("a", encoding="utf-8-sig") as f:
        f.write(
            f"""

## 20260620_42 strict station-subset decision

Purpose: compare strict full-workflow station exclusion branches: `_41` excludes all four difficult stations; `_42` excludes only 官良站 and 平山（三）站 while retaining 北流站 and 犁市（二）站.

Result:
- `_41` common trusted scope median raw NSE: {r41_common['median_NSE_raw']:.4f}; good {int(r41_common['good_count'])}/{int(r41_common['station_count'])}
- `_42` common trusted scope median raw NSE: {r42_common['median_NSE_raw']:.4f}; good {int(r42_common['good_count'])}/{int(r42_common['station_count'])}
- `_42` active set median raw NSE: {r42_active['median_NSE_raw']:.4f}; good {int(r42_active['good_count'])}/{int(r42_active['station_count'])}

Decision: partial retention is not promoted. If these difficult stations are treated as questionable observations, the cleaner screened calibration branch is `20260620_41` strict exclude-all-4.

Outputs: E:\\SPARROW\\5_Test\\20260620_42\\reports\\station_subset_decision
"""
        )

    print(md)


if __name__ == "__main__":
    main()
