from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260810_4")
SOURCE = Path(
    r"E:\SPARROW\1_Inputs\DischargeData\monthly_mean_2006_2009"
    r"\水文年鉴录入表-珠江流域2006-2009-202512.xlsx"
)
UNRESOLVED = ROOT / "overlay" / "unresolved_keys.csv"
OUT_CSV = ROOT / "reports" / "tables" / "authoritative_excel_coverage.csv"
OUT_JSON = ROOT / "reports" / "authoritative_excel_coverage_summary.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"站$", "", text)
    return text


def numeric_days(frame: pd.DataFrame) -> pd.DataFrame:
    day_cols = [f"{day}日" for day in range(1, 32)]
    values = frame[day_cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.copy()
    frame["monthly_mean_from_daily"] = values.mean(axis=1, skipna=True)
    frame["n_days_from_daily"] = values.notna().sum(axis=1)
    return frame


def format_values(values: pd.Series) -> str:
    return "|".join(f"{float(value):.12g}" for value in values)


def main() -> None:
    unresolved = pd.read_csv(UNRESOLVED, encoding="utf-8")
    runoff = pd.read_excel(SOURCE, sheet_name="径流m3 s")
    runoff = numeric_days(runoff)
    runoff["station_norm"] = runoff["站点"].map(norm)
    runoff["river_norm"] = runoff["河流"].map(norm)
    runoff["year_int"] = pd.to_numeric(runoff["年"], errors="coerce").astype("Int64")
    runoff["month_int"] = pd.to_numeric(runoff["月"], errors="coerce").astype("Int64")
    runoff["excel_row"] = np.arange(len(runoff), dtype=int) + 2

    output_rows: list[dict[str, object]] = []
    for row in unresolved.itertuples(index=False):
        station_norm = norm(row.station)
        river_norm = norm(row.river)
        base = (runoff["year_int"] == int(row.year)) & (
            runoff["month_int"] == int(row.month)
        )
        exact = runoff[
            base
            & (runoff["station_norm"] == station_norm)
            & (runoff["river_norm"] == river_norm)
        ]
        swapped = runoff[
            base
            & (runoff["station_norm"] == river_norm)
            & (runoff["river_norm"] == station_norm)
        ]

        if len(exact) == 1:
            coverage = "UNIQUE_EXACT_STATION_RIVER_MONTH"
            selected = exact
        elif len(exact) > 1:
            coverage = "AMBIGUOUS_WITHIN_EXACT_STATION_RIVER_MONTH"
            selected = exact
        elif len(swapped) == 1:
            coverage = "UNIQUE_AFTER_STATION_RIVER_SWAP"
            selected = swapped
        elif len(swapped) > 1:
            coverage = "AMBIGUOUS_AFTER_STATION_RIVER_SWAP"
            selected = swapped
        else:
            coverage = "NO_STATION_RIVER_MONTH_MATCH"
            selected = exact

        values = (
            format_values(selected["monthly_mean_from_daily"])
            if len(selected)
            else ""
        )
        excel_rows = (
            "|".join(map(str, selected["excel_row"].astype(int)))
            if len(selected)
            else ""
        )
        output_rows.append(
            {
                "station": row.station,
                "river": row.river,
                "year": int(row.year),
                "month": int(row.month),
                "previous_resolution_status": row.resolution_status,
                "authority_coverage_status": coverage,
                "authority_candidate_count": int(len(selected)),
                "authority_candidate_monthly_means": values,
                "authority_excel_rows": excel_rows,
                "old_q": float(row.old_q),
                "candidate_values_from_monthly_workbook": row.supported_candidate_values,
            }
        )

    result = pd.DataFrame(output_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    counts = result["authority_coverage_status"].value_counts().to_dict()
    station_year_counts = (
        result.groupby("authority_coverage_status")[["station", "year"]]
        .apply(lambda frame: frame.drop_duplicates().shape[0])
        .to_dict()
    )
    summary = {
        "authoritative_source_path": str(SOURCE),
        "authoritative_source_sha256": sha256(SOURCE),
        "unresolved_input_rows": int(len(unresolved)),
        "unresolved_input_station_years": int(
            unresolved[["station", "year"]].drop_duplicates().shape[0]
        ),
        "coverage_status_counts": {key: int(value) for key, value in counts.items()},
        "coverage_station_year_counts": {
            key: int(value) for key, value in station_year_counts.items()
        },
        "interpretation": {
            "unique_exact_or_swap": (
                "The workbook can select a single row after using station, river, "
                "year and month as the key (or correcting an evident column swap)."
            ),
            "ambiguous_exact": (
                "The workbook itself contains more than one daily row for the same "
                "station, river, year and month, so declaring the workbook authoritative "
                "does not select a unique value."
            ),
            "no_match": (
                "The workbook has no row at the intended composite key and cannot cover "
                "the conflict without an additional mapping or image-level source."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
