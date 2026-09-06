from __future__ import annotations

import hashlib
import json
import math
import re
import calendar
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260810_4")
AUTHORITY = Path(
    r"E:\SPARROW\1_Inputs\DischargeData\monthly_mean_2006_2009"
    r"\水文年鉴录入表-珠江流域2006-2009-202512.xlsx"
)
OLD_DECISIONS = ROOT / "overlay" / "resolved_month_overrides.csv"
OUT_DETAIL = ROOT / "reports" / "tables" / "reaudit_198_authoritative_detail.csv"
OUT_SUMMARY_CSV = ROOT / "reports" / "tables" / "reaudit_198_authoritative_summary.csv"
OUT_SUMMARY_JSON = ROOT / "reports" / "reaudit_198_authoritative_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = text.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", text)


def norm_station(value: object) -> str:
    return re.sub(r"站$", "", clean_text(value))


def norm_river(value: object) -> str:
    return clean_text(value)


def correct_roles(station: object, river: object) -> tuple[str, str, bool]:
    station_text = clean_text(station)
    river_text = clean_text(river)
    # The authority workbook has a small number of blocks where the river name
    # is stored under 站点 and the station name (ending in 站) under 河流.
    swapped = (not station_text.endswith("站")) and river_text.endswith("站")
    if swapped:
        return river_text, station_text, True
    return station_text, river_text, False


def finite_mean_and_count(values: pd.Series) -> tuple[float | None, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return None, 0
    # Zero discharge is valid and must not be dropped.  This intentionally
    # corrects the former audit's positive-only fingerprint error.
    return float(numeric.mean()), int(numeric.size)


def close(left: float, right: float) -> bool:
    # The derived monthly workbook stores six decimal places.
    return round(float(left), 6) == round(float(right), 6)


def pipe(values: list[object]) -> str:
    return "|".join(map(str, values))


def main() -> None:
    old = pd.read_csv(OLD_DECISIONS, encoding="utf-8")
    if len(old) != 198:
        raise RuntimeError(f"Expected 198 frozen conflict decisions, found {len(old)}")

    authority = pd.read_excel(AUTHORITY, sheet_name="径流m3 s")
    day_columns = [f"{day}日" for day in range(1, 32)]
    authority["authority_excel_row"] = np.arange(len(authority), dtype=int) + 2
    corrected = authority.apply(
        lambda row: correct_roles(row["站点"], row["河流"]), axis=1, result_type="expand"
    )
    corrected.columns = ["station_corrected", "river_corrected", "roles_swapped"]
    authority = pd.concat([authority, corrected], axis=1)
    authority["station_key"] = authority["station_corrected"].map(norm_station)
    authority["river_key"] = authority["river_corrected"].map(norm_river)
    authority["year_int"] = pd.to_numeric(authority["年"], errors="coerce").astype("Int64")
    authority["month_int"] = pd.to_numeric(authority["月"], errors="coerce").astype("Int64")
    means_counts: list[tuple[float | None, int]] = []
    for _, row in authority.iterrows():
        try:
            year = int(row["年"])
            month = int(row["月"])
            valid_calendar_days = calendar.monthrange(year, month)[1]
        except (TypeError, ValueError):
            means_counts.append((None, 0))
            continue
        row_values = row[[f"{day}日" for day in range(1, valid_calendar_days + 1)]]
        means_counts.append(finite_mean_and_count(row_values))
    authority["q_from_daily"] = [item[0] for item in means_counts]
    authority["n_days_from_daily"] = [item[1] for item in means_counts]

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for target in old.itertuples(index=False):
        target_station, target_river, target_swapped = correct_roles(target.station, target.river)
        station_key = norm_station(target_station)
        river_key = norm_river(target_river)
        matches = authority[
            (authority["station_key"] == station_key)
            & (authority["river_key"] == river_key)
            & (authority["year_int"] == int(target.year))
            & (authority["month_int"] == int(target.month))
        ].copy()

        finite_q = sorted(
            {
                round(float(value), 12)
                for value in matches["q_from_daily"].dropna().tolist()
                if math.isfinite(float(value))
            }
        )
        old_match = [value for value in finite_q if close(value, float(target.old_q))]
        previous_new_match = [value for value in finite_q if close(value, float(target.new_q))]

        if not len(matches):
            status = "NO_AUTHORITY_ROW_AT_CORRECTED_COMPOSITE_KEY"
            selected_q = np.nan
            selected_n = np.nan
        elif len(finite_q) == 1:
            status = (
                "UNIQUE_AUTHORITY_VALUE_SINGLE_ROW"
                if len(matches) == 1
                else "UNIQUE_AUTHORITY_VALUE_DUPLICATE_IDENTICAL_ROWS"
            )
            selected_q = finite_q[0]
            selected_n_values = sorted(set(matches["n_days_from_daily"].astype(int)))
            selected_n = selected_n_values[0] if len(selected_n_values) == 1 else np.nan
        else:
            status = "AMBIGUOUS_MULTIPLE_AUTHORITY_VALUES_SAME_COMPOSITE_KEY"
            selected_q = np.nan
            selected_n = np.nan

        summary_rows.append(
            {
                "station_raw_target": target.station,
                "river_raw_target": target.river,
                "station_corrected_target": target_station,
                "river_corrected_target": target_river,
                "target_roles_swapped": target_swapped,
                "year": int(target.year),
                "month": int(target.month),
                "old_q": float(target.old_q),
                "previous_new_q": float(target.new_q),
                "previous_resolution_status": target.resolution_status,
                "authority_match_rows": int(len(matches)),
                "authority_distinct_q_values": int(len(finite_q)),
                "authority_q_values": pipe([format(value, ".12g") for value in finite_q]),
                "old_q_present_in_authority": bool(old_match),
                "previous_new_q_present_in_authority": bool(previous_new_match),
                "reaudit_status": status,
                "reaudit_selected_q": selected_q,
                "reaudit_selected_n_days": selected_n,
            }
        )
        for match in matches.itertuples(index=False):
            detail_rows.append(
                {
                    "target_station": target.station,
                    "target_river": target.river,
                    "target_year": int(target.year),
                    "target_month": int(target.month),
                    "target_old_q": float(target.old_q),
                    "target_previous_new_q": float(target.new_q),
                    "previous_resolution_status": target.resolution_status,
                    "authority_excel_row": int(match.authority_excel_row),
                    "authority_station_raw": match.站点,
                    "authority_river_raw": match.河流,
                    "authority_station_corrected": match.station_corrected,
                    "authority_river_corrected": match.river_corrected,
                    "authority_roles_swapped": bool(match.roles_swapped),
                    "authority_q_from_daily": match.q_from_daily,
                    "authority_n_days_from_daily": int(match.n_days_from_daily),
                }
            )

    summary_frame = pd.DataFrame(summary_rows)
    detail_frame = pd.DataFrame(detail_rows)
    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    detail_frame.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    summary_frame.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    old_resolved = summary_frame[
        summary_frame["previous_resolution_status"].eq(
            "RESOLVED_L3_UNIQUE_FROZEN_DAILY_VECTOR"
        )
    ]
    result = {
        "authority_path": str(AUTHORITY),
        "authority_sha256": sha256(AUTHORITY),
        "all_conflict_keys_reaudited": int(len(summary_frame)),
        "reaudit_status_counts": {
            key: int(value)
            for key, value in summary_frame["reaudit_status"].value_counts().items()
        },
        "corrected_target_role_swap_keys": int(summary_frame["target_roles_swapped"].sum()),
        "former_27_status_counts": {
            key: int(value)
            for key, value in old_resolved["reaudit_status"].value_counts().items()
        },
        "former_27_previous_new_q_still_present": int(
            old_resolved["previous_new_q_present_in_authority"].sum()
        ),
        "zero_values_are_valid": True,
        "authority_decision_rule": (
            "A key is automatically selectable only when the corrected "
            "station-river-year-month composite key has exactly one distinct daily-derived "
            "monthly mean. Multiple identical source rows are deterministic duplicates; "
            "multiple distinct values remain unresolved."
        ),
    }
    OUT_SUMMARY_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
