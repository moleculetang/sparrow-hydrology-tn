from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TEST_ROOT = Path(r"E:\SPARROW\5_Test")
CONTROL = TEST_ROOT / "20260721_1"
SOURCE_RUN = TEST_ROOT / "20260721_206"
HISTORICAL_FINAL = TEST_ROOT / "20260721_205"
OUT = CONTROL / "reports" / "dynamic_station_screening"


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def main() -> None:
    raw = pd.read_csv(
        SOURCE_RUN / "reports" / "input_preprocessing" / "discharge_coverage_by_station_month.csv",
        encoding="utf-8-sig",
    )
    terminal = pd.read_csv(
        CONTROL / "reports" / "dynamic_station_screening" / "final_station_status_ledger.csv",
        encoding="utf-8-sig",
    )
    historical = pd.read_csv(
        HISTORICAL_FINAL / "reports" / "final_station_screening" / "frozen_exclusion_set.csv",
        encoding="utf-8-sig",
    )
    decision_policy = json.loads(
        (CONTROL / "inputs" / "source_metadata" / "station_screening_decision_policy_v2.json").read_text(
            encoding="utf-8"
        )
    )
    protected = {
        str(row["station_name"])
        for row in decision_policy["domain_constraints"]["protected_from_exclusion"]
    }
    target_names = set(historical["station_name"].astype(str)) | protected
    universe = set(terminal["station_name"].astype(str))

    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw["q_m3s"] = pd.to_numeric(raw["q_m3s"], errors="coerce")
    raw = raw[
        truthy(raw["usable"])
        & raw["station_name"].astype(str).isin(universe)
        & raw["q_m3s"].gt(0)
    ].copy()
    periods = (
        ("selection_2006_2018", 2006, 2018),
        ("calibration_2010_2018", 2010, 2018),
        ("confirmation_2019_2022", 2019, 2022),
    )
    rows: list[pd.DataFrame] = []
    for period, start, end in periods:
        part = raw[raw["year"].between(start, end)].copy()
        profile = (
            part.groupby("station_name", as_index=False)
            .agg(
                usable_months=("q_m3s", "size"),
                usable_years=("year", "nunique"),
                mean_q_m3s=("q_m3s", "mean"),
                median_q_m3s=("q_m3s", "median"),
                p90_q_m3s=("q_m3s", lambda values: values.quantile(0.90)),
                max_q_m3s=("q_m3s", "max"),
            )
        )
        profile["mean_rank_desc"] = profile["mean_q_m3s"].rank(method="min", ascending=False).astype(int)
        profile["mean_percentile_from_low"] = profile["mean_q_m3s"].rank(pct=True) * 100.0
        profile["multiple_of_station_median_mean"] = (
            profile["mean_q_m3s"] / profile["mean_q_m3s"].median()
        )
        years = {
            station: sorted(set(range(start, end + 1)) - set(group["year"].dropna().astype(int)))
            for station, group in part.groupby("station_name")
        }
        profile["missing_years"] = profile["station_name"].map(
            lambda station: ";".join(str(year) for year in years.get(station, []))
        )
        profile["period"] = period
        profile["station_universe_count"] = int(len(profile))
        profile["station_median_of_mean_q_m3s"] = float(profile["mean_q_m3s"].median())
        rows.append(profile[profile["station_name"].astype(str).isin(target_names)].copy())

    result = pd.concat(rows, ignore_index=True)
    historical_status = dict(
        zip(terminal["station_name"].astype(str), terminal["final_status"].astype(str))
    )
    result["v2_1_status"] = result["station_name"].map(historical_status).fillna("")
    result["v2_2_status"] = result.apply(
        lambda row: "retain_domain_protected"
        if str(row["station_name"]) in protected
        else str(row["v2_1_status"]),
        axis=1,
    )
    result = result.sort_values(["period", "mean_rank_desc", "station_name"]).reset_index(drop=True)
    result.to_csv(OUT / "station_flow_scale_audit.csv", index=False, encoding="utf-8-sig")

    selection = result[result["period"].eq("selection_2006_2018")].copy()
    lines = [
        "# Station Flow-Scale Audit for the v2.1 Exclusion Set",
        "",
        "- source run: `20260721_206`",
        "- comparison universe: 105 modeled stations",
        "- ranking measure: mean usable monthly observed discharge, 2006–2018",
        "- policy consequence: 石角站 is protected from exclusion under v2.2",
        "",
        "| station | mean Q (m³/s) | median Q (m³/s) | rank / 105 | percentile | relative to station median | missing years | v2.2 status |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in selection.itertuples(index=False):
        lines.append(
            f"| {row.station_name} | {row.mean_q_m3s:.3f} | {row.median_q_m3s:.3f} | "
            f"{row.mean_rank_desc} / {row.station_universe_count} | {row.mean_percentile_from_low:.1f}% | "
            f"{row.multiple_of_station_median_mean:.2f}× | {row.missing_years or 'none'} | {row.v2_2_status} |"
        )
    lines.extend(
        [
            "",
            "The six stations are not uniformly large-flow stations. 迁江、都安（二） and 石角 are top-seven "
            "stations by mean flow; 隆安 is upper-quintile; 劳村 and 富罗（二） are below the station median. "
            "石角 is a good-performing, high-flow control station whose former exclusion was supported only by "
            "common-station ablation benefit, not by persistent station failure or proven raw-data invalidity.",
        ]
    )
    (OUT / "station_flow_scale_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "source_run": SOURCE_RUN.name,
        "station_universe_count": int(selection["station_universe_count"].iloc[0]),
        "protected_from_exclusion": sorted(protected),
        "selection_period_rows": selection[
            [
                "station_name",
                "mean_q_m3s",
                "median_q_m3s",
                "mean_rank_desc",
                "mean_percentile_from_low",
                "multiple_of_station_median_mean",
                "missing_years",
                "v2_2_status",
            ]
        ].to_dict(orient="records"),
    }
    (OUT / "station_flow_scale_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
