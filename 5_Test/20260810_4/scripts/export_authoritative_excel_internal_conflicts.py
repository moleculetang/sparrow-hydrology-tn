from __future__ import annotations

import calendar
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from reaudit_all_198_authoritative_excel import (
    AUTHORITY,
    ROOT,
    correct_roles,
    norm_river,
    norm_station,
    sha256,
)


REAUDIT = ROOT / "reports" / "tables" / "reaudit_198_authoritative_summary.csv"
OUT_KEYS = (
    ROOT
    / "reports"
    / "tables"
    / "authoritative_excel_internal_conflict_keys.csv"
)
OUT_SEQUENCES = (
    ROOT
    / "reports"
    / "tables"
    / "authoritative_excel_internal_conflicting_daily_sequences.csv"
)
OUT_VALIDATION = ROOT / "reports" / "authoritative_excel_internal_conflicts_validation.json"


def canonical_cell(value: object) -> object:
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if not math.isfinite(number):
        return None
    return format(number, ".12g")


def sequence_hash(
    station: str,
    river: str,
    year: int,
    month: int,
    daily_values: list[object],
) -> str:
    payload = {
        "station": station,
        "river": river,
        "year": year,
        "month": month,
        "daily_values_within_calendar_month": [canonical_cell(v) for v in daily_values],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def numeric_mean(values: list[object]) -> tuple[float | None, int]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return None, 0
    return float(numeric.mean()), int(numeric.size)


def pipe(values: list[object]) -> str:
    return "|".join(str(value) for value in values)


def main() -> None:
    reaudited = pd.read_csv(REAUDIT, encoding="utf-8-sig")
    ambiguous = reaudited[
        reaudited["reaudit_status"].eq(
            "AMBIGUOUS_MULTIPLE_AUTHORITY_VALUES_SAME_COMPOSITE_KEY"
        )
    ].copy()
    if len(ambiguous) != 126:
        raise RuntimeError(f"Expected 126 ambiguous keys, found {len(ambiguous)}")

    source = pd.read_excel(AUTHORITY, sheet_name="径流m3 s")
    source["source_excel_row"] = np.arange(len(source), dtype=int) + 2
    corrected = source.apply(
        lambda row: correct_roles(row["站点"], row["河流"]), axis=1, result_type="expand"
    )
    corrected.columns = ["station_corrected", "river_corrected", "roles_swapped"]
    source = pd.concat([source, corrected], axis=1)
    source["station_key"] = source["station_corrected"].map(norm_station)
    source["river_key"] = source["river_corrected"].map(norm_river)
    source["year_int"] = pd.to_numeric(source["年"], errors="coerce").astype("Int64")
    source["month_int"] = pd.to_numeric(source["月"], errors="coerce").astype("Int64")

    detail_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for target in ambiguous.itertuples(index=False):
        station = str(target.station_corrected_target)
        river = str(target.river_corrected_target)
        year = int(target.year)
        month = int(target.month)
        calendar_days = calendar.monthrange(year, month)[1]
        matches = source[
            (source["station_key"] == norm_station(station))
            & (source["river_key"] == norm_river(river))
            & (source["year_int"] == year)
            & (source["month_int"] == month)
        ].copy()
        if len(matches) < 2:
            raise RuntimeError(
                f"Ambiguous key no longer has multiple rows: {station}/{river}/{year}/{month}"
            )

        per_key: list[dict[str, object]] = []
        for candidate_rank, (_, match) in enumerate(matches.iterrows(), start=1):
            valid_daily = [match[f"{day}日"] for day in range(1, calendar_days + 1)]
            mean_q, n_days = numeric_mean(valid_daily)
            outside_values = [
                match[f"{day}日"]
                for day in range(calendar_days + 1, 32)
                if not pd.isna(match[f"{day}日"])
            ]
            digest = sequence_hash(station, river, year, month, valid_daily)
            row = {
                "source_file": str(AUTHORITY),
                "source_sha256": sha256(AUTHORITY),
                "source_sheet": "径流m3 s",
                "source_excel_row": int(match["source_excel_row"]),
                "station": station,
                "river": river,
                "year": year,
                "month": month,
                "candidate_rank_within_key": candidate_rank,
                "source_station_raw": match["站点"],
                "source_river_raw": match["河流"],
                "source_roles_swapped": bool(match["roles_swapped"]),
                "calendar_days": calendar_days,
                "numeric_days_within_calendar": n_days,
                "monthly_mean_m3_s_from_daily": mean_q,
                "nonempty_cells_outside_calendar": len(outside_values),
                "daily_sequence_sha256": digest,
            }
            for day in range(1, 32):
                row[f"day_{day}"] = match[f"{day}日"]
            detail_rows.append(row)
            per_key.append(row)

        distinct_hashes = sorted({str(item["daily_sequence_sha256"]) for item in per_key})
        distinct_means = sorted(
            {
                round(float(item["monthly_mean_m3_s_from_daily"]), 12)
                for item in per_key
                if item["monthly_mean_m3_s_from_daily"] is not None
            }
        )
        if len(distinct_hashes) < 2 or len(distinct_means) < 2:
            raise RuntimeError(
                f"Listed key is not a true distinct-value conflict: {station}/{river}/{year}/{month}"
            )
        key_rows.append(
            {
                "source_file": str(AUTHORITY),
                "source_sha256": sha256(AUTHORITY),
                "source_sheet": "径流m3 s",
                "station": station,
                "river": river,
                "year": year,
                "month": month,
                "candidate_row_count": len(per_key),
                "distinct_daily_sequence_count": len(distinct_hashes),
                "distinct_monthly_mean_count": len(distinct_means),
                "source_excel_rows": pipe([item["source_excel_row"] for item in per_key]),
                "candidate_monthly_means_m3_s": pipe(
                    [format(value, ".12g") for value in distinct_means]
                ),
                "daily_sequence_sha256s": pipe(distinct_hashes),
                "status": "INTERNAL_CONFLICT_SAME_STATION_RIVER_YEAR_MONTH",
            }
        )

    detail = pd.DataFrame(detail_rows).sort_values(
        ["station", "river", "year", "month", "source_excel_row"]
    )
    keys = pd.DataFrame(key_rows).sort_values(["station", "river", "year", "month"])
    OUT_KEYS.parent.mkdir(parents=True, exist_ok=True)
    keys.to_csv(OUT_KEYS, index=False, encoding="utf-8-sig")
    detail.to_csv(OUT_SEQUENCES, index=False, encoding="utf-8-sig")

    validation = {
        "source_file": str(AUTHORITY),
        "source_sha256": sha256(AUTHORITY),
        "source_sheet": "径流m3 s",
        "conflict_keys": int(len(keys)),
        "candidate_source_rows": int(len(detail)),
        "candidate_count_distribution": {
            str(key): int(value)
            for key, value in keys["candidate_row_count"].value_counts().sort_index().items()
        },
        "all_keys_have_multiple_distinct_daily_sequences": bool(
            keys["distinct_daily_sequence_count"].ge(2).all()
        ),
        "all_keys_have_multiple_distinct_monthly_means": bool(
            keys["distinct_monthly_mean_count"].ge(2).all()
        ),
        "output_key_csv": str(OUT_KEYS),
        "output_sequence_csv": str(OUT_SEQUENCES),
    }
    OUT_VALIDATION.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
