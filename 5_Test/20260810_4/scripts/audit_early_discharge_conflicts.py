"""Read-only lineage audit for 2006--2009 discharge-workbook conflicts.

This program creates only files below ``5_Test/20260810_4``.  It never edits
the workbooks, stage-2 CSVs, JSON work packages, or model inputs.  A filename
is used only as a locator.  Candidate-to-source assertions require a strict
12-month numerical fingerprint (monthly positive-value mean and valid-day
count), then JSON fields are used to recover page/image provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT = Path(r"E:\SPARROW")
RUN = PROJECT / "5_Test" / "20260810_4"
OUT = RUN / "reports" / "tables"

SNAPSHOT = RUN / "inputs" / "source_snapshot"
# These are the only workbook inputs read by this script.  External workbooks
# may be compared by a separate human-controlled check, but are not dependencies
# of this reproducible overlay audit.
WORKBOOKS = {
    "frozen_monthly_snapshot": SNAPSHOT / "DischargeData_2006_2009.xlsx",
    "frozen_daily_entry_snapshot": SNAPSHOT / "水文年鉴录入表-珠江流域2006-2009-202512.xlsx",
}
MONTHLY_WORKBOOK_LABEL = "frozen_monthly_snapshot"
STAGE2_ROOTS = {
    "stage2_result": PROJECT / "0_hydro_sediment_data" / "discharge" / "stage2_result",
    "stage2_result_revised": PROJECT / "0_hydro_sediment_data" / "discharge" / "stage2_result_revised",
}
STAGE2_WORK = PROJECT / "0_hydro_sediment_data" / "discharge" / "stage2_work"
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
MONTH_ALIASES = {
    "jan": ("jan", "1月", "一月"), "feb": ("feb", "2月", "二月"), "mar": ("mar", "3月", "三月"),
    "apr": ("apr", "4月", "四月"), "may": ("may", "5月", "五月"), "jun": ("jun", "6月", "六月"),
    "jul": ("jul", "7月", "七月"), "aug": ("aug", "8月", "八月"), "sep": ("sep", "9月", "九月"),
    "oct": ("oct", "10月", "十月"), "nov": ("nov", "11月", "十一月"), "dec": ("dec", "12月", "十二月"),
}
YEAR_RE = re.compile(r"(?:^|[^0-9])(20\d{2})(?:[^0-9]|$)")
PATH_EVIDENCE_CACHE: dict[str, tuple[str, bool, str]] = {}


def norm_station(value: Any) -> str:
    """Normalize only typography, retaining meaningful Chinese qualifiers."""
    text = str(value or "").strip()
    text = text.replace("（", "(").replace("）", ")").replace(" ", "")
    text = re.sub(r"站$", "", text)
    return text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relocate_stage2_path(value: Any) -> tuple[str, bool, str]:
    """Rebase legacy stage2_work paths into the current workspace and hash them.

    A missing rebased path means only that this referenced asset is not mounted
    here; it is never used as a claim that no original page exists elsewhere.
    """
    raw = str(value or "").strip()
    if not raw:
        return "", False, ""
    if raw in PATH_EVIDENCE_CACHE:
        return PATH_EVIDENCE_CACHE[raw]
    normalized = raw.replace("\\", "/")
    marker = "stage2_work/"
    lower = normalized.lower()
    if marker in lower:
        suffix = normalized[lower.index(marker) + len(marker):]
        local = STAGE2_WORK.joinpath(*[part for part in suffix.split("/") if part])
    else:
        candidate = Path(raw)
        local = candidate if candidate.is_absolute() and str(candidate).startswith(str(STAGE2_WORK)) else None
    if local is None:
        result = ("", False, "")
    else:
        exists = local.is_file()
        result = (str(local), exists, sha256(local) if exists else "")
    PATH_EVIDENCE_CACHE[raw] = result
    return result


def finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def csv_year(path: Path, root: Path) -> int | None:
    for part in path.relative_to(root).parts[:-1]:
        if re.fullmatch(r"20\d{2}", part):
            return int(part)
    for part in path.parts:
        hit = YEAR_RE.search(part)
        if hit:
            return int(hit.group(1))
    return None


def read_workbook(label: str, path: Path) -> pd.DataFrame:
    required = {"station", "year", "month", "monthly_mean_m3_s", "n_days_used"}
    raw = pd.read_excel(path, sheet_name="monthly_mean", engine="openpyxl")
    absent = required.difference(raw.columns)
    if absent:
        raise RuntimeError(f"{path} missing required columns: {sorted(absent)}")
    out = raw.loc[:, sorted(required)].copy()
    out.insert(0, "workbook_label", label)
    out.insert(1, "workbook_path", str(path))
    out.insert(2, "workbook_sha256", sha256(path))
    out.insert(3, "excel_row", np.arange(2, len(out) + 2, dtype=int))
    out["station_name"] = out["station"].astype(str).str.strip()
    out["station_norm"] = out["station_name"].map(norm_station)
    for col in ("year", "month", "monthly_mean_m3_s", "n_days_used"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["year"].between(2006, 2009) & out["month"].between(1, 12)].copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    return out


def daily_semantic_hash(station: str, river: str, year: int, month: int, day_values: list[Any]) -> str:
    payload = {
        "station_norm": norm_station(station), "river": str(river or "").strip(),
        "year": int(year), "month": int(month),
        "daily_values": [None if finite_positive(v) is None else format(float(v), ".12g") for v in day_values],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_daily_entry_snapshot(path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Read the second frozen XLSX and record its runoff-table contract.

    It is not silently substituted for the monthly workbook: this audit keeps
    all monthly-label candidates intact and only records that the original daily
    entry workbook is locally frozen and readable for later evidence review.
    """
    sheets = pd.ExcelFile(path, engine="openpyxl").sheet_names
    runoff_sheet = next((name for name in sheets if "径流" in str(name)), None)
    if runoff_sheet is None:
        raise RuntimeError(f"No runoff sheet found in frozen daily-entry workbook: {path}")
    frame = pd.read_excel(path, sheet_name=runoff_sheet, engine="openpyxl")
    required = {"站点", "年", "月"}
    absent = required.difference(frame.columns)
    if absent:
        raise RuntimeError(f"{path}:{runoff_sheet} missing {sorted(absent)}")
    day_columns = [col for col in frame.columns if re.fullmatch(r"\d+日", str(col))]
    rows: list[dict[str, Any]] = []
    for source_pos, values in enumerate(frame.itertuples(index=False, name=None), start=2):
        record = dict(zip(frame.columns, values))
        try:
            year, month = int(record["年"]), int(record["月"])
        except (TypeError, ValueError):
            continue
        if not (2006 <= year <= 2009 and 1 <= month <= 12):
            continue
        daily_values = [record.get(col) for col in day_columns]
        positive = [value for value in (finite_positive(v) for v in daily_values) if value is not None]
        station = str(record.get("站点") or "").strip()
        river = str(record.get("河流") or "").strip()
        rows.append({
            "daily_entry_excel_row": source_pos, "station_name": station, "station_norm": norm_station(station),
            "river_name": river, "year": year, "month": month,
            "monthly_mean_m3_s": float(np.mean(positive)) if positive else None,
            "n_days_used": len(positive),
            "daily_semantic_sha256": daily_semantic_hash(station, river, year, month, daily_values),
            "daily_values_json": json.dumps([None if finite_positive(v) is None else float(v) for v in daily_values], ensure_ascii=False),
        })
    daily_rows = pd.DataFrame(rows)
    if daily_rows.empty:
        raise RuntimeError(f"No 2006--2009 runoff daily rows parsed from {path}")
    daily_rows["daily_candidate_rank"] = daily_rows.groupby(["station_norm", "year", "month"], sort=False).cumcount() + 1
    annual_rows: list[dict[str, Any]] = []
    for (station, year, rank), part in daily_rows.groupby(["station_norm", "year", "daily_candidate_rank"], sort=False):
        by_month = part.set_index("month")
        qvec = [finite_positive(by_month.at[m, "monthly_mean_m3_s"]) if m in by_month.index else None for m in range(1, 13)]
        nvec = [int(by_month.at[m, "n_days_used"]) if m in by_month.index else 0 for m in range(1, 13)]
        hashes = [str(by_month.at[m, "daily_semantic_sha256"]) if m in by_month.index else "" for m in range(1, 13)]
        annual_rows.append({
            "station_norm": station, "station_name": str(part["station_name"].iloc[0]),
            "river_name": str(part["river_name"].iloc[0]), "year": int(year), "daily_candidate_rank": int(rank),
            "months_present": int(part["month"].nunique()), "complete_12_month_vector": bool(part["month"].nunique() == 12),
            "daily_entry_excel_rows": "|".join(map(str, part.sort_values("month")["daily_entry_excel_row"].astype(int))),
            "daily_semantic_hashes": "|".join(hashes), "q_vector_m3_s": json.dumps(qvec, ensure_ascii=False),
            "n_days_vector": json.dumps(nvec, ensure_ascii=False),
        })
    meta = {
        "path": str(path),
        "sha256": sha256(path),
        "sheet_names": sheets,
        "runoff_sheet": runoff_sheet,
        "runoff_rows": int(len(frame)),
        "runoff_station_year_month_rows": int(frame[["站点", "年", "月"]].dropna(how="all").shape[0]),
        "daily_value_columns": int(len(day_columns)),
    }
    return meta, daily_rows, pd.DataFrame(annual_rows)


def fingerprint_daily_table(path: Path) -> tuple[list[float | None], list[int]] | None:
    """Return Jan--Dec positive-value means and counts; reject non-daily matrices."""
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path, engine="openpyxl" if path.suffix.lower() == ".xlsx" else None)
        else:
            frame = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="gb18030")
    except Exception:
        return None
    names = {str(col).strip().lower(): col for col in frame.columns}
    selected = {month: next((names[alias] for alias in MONTH_ALIASES[month] if alias in names), None) for month in MONTHS}
    if any(value is None for value in selected.values()):
        return None
    means: list[float | None] = []
    counts: list[int] = []
    for month in MONTHS:
        values = pd.to_numeric(frame[selected[month]], errors="coerce").to_numpy(dtype=float)
        valid = values[np.isfinite(values) & (values > 0.0)]
        counts.append(int(valid.size))
        means.append(float(valid.mean()) if valid.size else None)
    return means, counts


def fingerprint_embedded_csv_rows(rows: Any) -> tuple[list[float | None], list[int]] | None:
    """Fingerprint worker_result.station_results[*].csv_rows without exporting it."""
    if not isinstance(rows, list) or len(rows) < 2 or not isinstance(rows[0], list):
        return None
    try:
        frame = pd.DataFrame(rows[1:], columns=rows[0])
    except Exception:
        return None
    names = {str(col).strip().lower(): col for col in frame.columns}
    selected = {month: next((names[alias] for alias in MONTH_ALIASES[month] if alias in names), None) for month in MONTHS}
    if any(value is None for value in selected.values()):
        return None
    means, counts = [], []
    for month in MONTHS:
        values = pd.to_numeric(frame[selected[month]], errors="coerce").to_numpy(dtype=float)
        valid = values[np.isfinite(values) & (values > 0.0)]
        means.append(float(valid.mean()) if valid.size else None)
        counts.append(int(valid.size))
    return means, counts


def iter_stage2_daily_artifacts() -> Iterable[dict[str, Any]]:
    for kind, root in STAGE2_ROOTS.items():
        for path in root.rglob("*.csv"):
            fp = fingerprint_daily_table(path)
            if fp is None:
                continue
            yield {
                "catalog_kind": kind,
                "daily_csv_path": str(path),
                "station_name": path.stem,
                "station_norm": norm_station(path.stem),
                "year": csv_year(path, root),
                "monthly_means": fp[0],
                "n_days_used": fp[1],
                "csv_sha256": sha256(path),
                "work_run_dir": "",
            }
    work_output_paths = [
        path for path in STAGE2_WORK.rglob("*")
        if path.is_file() and path.parent.name == "04-output" and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
    ]
    for path in work_output_paths:
        fp = fingerprint_daily_table(path)
        if fp is None:
            continue
        run_dir = next((parent for parent in path.parents if (parent / "run.json").exists()), None)
        yield {
            "catalog_kind": f"stage2_work_output_{path.suffix.lower().lstrip('.')}",
            "daily_csv_path": str(path),
            "station_name": path.stem,
            "station_norm": norm_station(path.stem),
            "year": csv_year(path, STAGE2_WORK),
            "monthly_means": fp[0],
            "n_days_used": fp[1],
            "csv_sha256": sha256(path),
            "work_run_dir": str(run_dir) if run_dir else "",
        }


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def walk_json_values(obj: Any, key_name: str | None = None) -> Iterable[tuple[str | None, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from walk_json_values(value, str(key))
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_json_values(value, key_name)
    else:
        yield key_name, obj


def extract_json_index() -> tuple[pd.DataFrame, dict[str, dict[str, Any]], int]:
    """Index lineage-bearing worker/run/location JSON; retain all parsed fields needed for proof."""
    rows: list[dict[str, Any]] = []
    run_meta: dict[str, dict[str, Any]] = {}
    all_json_count = 0
    for path in STAGE2_WORK.rglob("*.json"):
        all_json_count += 1
        lname = path.name.lower()
        is_worker = lname == "worker_result.json"
        is_run = lname == "run.json"
        is_location = lname.endswith("station_table_location.json")
        if not (is_worker or is_run or is_location):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(path.read_text(encoding="gb18030"))
        except Exception as exc:
            rows.append({"json_path": str(path), "json_type": "parse_error", "parse_error": str(exc)})
            continue
        run_dir = next((parent for parent in path.parents if (parent / "run.json").exists()), path.parent)
        key = str(run_dir)
        if is_run:
            run_meta.setdefault(key, {}).update(
                {
                    "run_json_path": str(path),
                    "run_dir": key,
                    "run_image_path": as_text(data.get("image_path")),
                    "run_agent": as_text(data.get("agent")),
                    "run_group": as_text(data.get("group")),
                }
            )
        if is_worker:
            run_meta.setdefault(key, {}).update(
                {
                    "worker_json_path": str(path),
                    "run_dir": key,
                    "worker_image_path": as_text(data.get("image_path")),
                    "worker_station_results": as_text(data.get("station_results")),
                    "worker_detected_tables": as_text(data.get("detected_tables")),
                }
            )
            # A worker result can contain several station tables.  Expand each
            # result, including its embedded CSV matrix fingerprint, rather than
            # retaining an opaque JSON blob that cannot be joined or reviewed.
            for result in data.get("station_results", []) if isinstance(data.get("station_results"), list) else []:
                if not isinstance(result, dict):
                    continue
                station = as_text(result.get("station_name_from_station_list") or result.get("station_name_best_effort"))
                csv_rows = result.get("csv_rows")
                csv_rows_hash = hashlib.sha256(json.dumps(csv_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest() if csv_rows is not None else ""
                embedded_fp = fingerprint_embedded_csv_rows(csv_rows)
                row = {
                    "json_path": str(path), "json_type": "worker_station_result", "run_dir": key,
                    "station_name": station, "station_norm": norm_station(station),
                    "workflow_stage": as_text(result.get("workflow_stage")), "status": as_text(result.get("status")),
                    "review_status": as_text(result.get("review_status")), "data_type": as_text(result.get("data_type")),
                    "csv_rows_count": len(csv_rows) if isinstance(csv_rows, list) else 0, "csv_rows_sha256": csv_rows_hash,
                    "embedded_monthly_means": json.dumps(embedded_fp[0], ensure_ascii=False) if embedded_fp else "",
                    "embedded_n_days_used": json.dumps(embedded_fp[1], ensure_ascii=False) if embedded_fp else "",
                    "source_image": as_text(data.get("image_path")), "station_query": station,
                    "parse_error": "",
                }
                for field in ("xlsx_path", "table_image_path", "paddle_location_json", "paddle_review_json", "review_crops_manifest", "station_dir"):
                    raw_value = as_text(result.get(field))
                    rebased, exists, digest = relocate_stage2_path(raw_value)
                    row[field] = raw_value
                    row[f"{field}_rebased_path"] = rebased
                    row[f"{field}_exists"] = exists
                    row[f"{field}_sha256"] = digest
                image_rebased, image_exists, image_hash = relocate_stage2_path(row["source_image"])
                row.update({"source_image_rebased_path": image_rebased, "source_image_exists": image_exists, "source_image_sha256": image_hash})
                rows.append(row)
            for detected in data.get("detected_tables", []) if isinstance(data.get("detected_tables"), list) else []:
                if not isinstance(detected, dict):
                    continue
                station = as_text(detected.get("station_name_best_effort"))
                rows.append({
                    "json_path": str(path), "json_type": "worker_detected_table", "run_dir": key,
                    "station_name": station, "station_norm": norm_station(station),
                    "station_query": station, "source_image": as_text(data.get("image_path")),
                    "table_index": as_text(detected.get("table_index")), "data_type": as_text(detected.get("data_type")),
                    "title_preview": as_text(detected.get("title_text_best_effort")), "parse_error": "",
                })
        if is_location:
            station_query = as_text(data.get("station_query"))
            source_image = as_text(data.get("source_image"))
            table = data.get("table_block") if isinstance(data.get("table_block"), dict) else {}
            title = data.get("title_block") if isinstance(data.get("title_block"), dict) else {}
            row = {
                    "json_path": str(path),
                    "json_type": "station_table_location",
                    "run_dir": key,
                    "station_name": station_query,
                    "station_norm": norm_station(station_query),
                    "source_image": source_image,
                    "table_path": as_text(table.get("table_path")),
                    "table_crop_path": as_text(table.get("crop_path")),
                    "table_bbox": as_text(table.get("block_bbox")),
                    "title_crop_path": as_text(title.get("crop_path")),
                    "title_preview": as_text(title.get("block_content_preview")),
                    "station_query": station_query,
                    "data_type": as_text(data.get("data_type")),
                    "parse_error": "",
                }
            for field in ("source_image", "table_path", "table_crop_path", "title_crop_path"):
                rebased, exists, digest = relocate_stage2_path(row.get(field, ""))
                row[f"{field}_rebased_path"] = rebased
                row[f"{field}_exists"] = exists
                row[f"{field}_sha256"] = digest
            rows.append(row)
    # Add one lineage row for worker/run JSONs even when no station table was found.
    for run_dir, meta in run_meta.items():
        row = {
                "json_path": meta.get("worker_json_path") or meta.get("run_json_path", ""),
                "json_type": "run_worker_lineage",
                "run_dir": run_dir,
                "station_name": "",
                "station_norm": "",
                "source_image": meta.get("worker_image_path") or meta.get("run_image_path", ""),
                "table_path": "",
                "table_crop_path": "",
                "table_bbox": "",
                "title_crop_path": "",
                "title_preview": "",
                "station_query": "",
                "data_type": "",
                "worker_json_path": meta.get("worker_json_path", ""),
                "run_json_path": meta.get("run_json_path", ""),
                "run_agent": meta.get("run_agent", ""),
                "run_group": meta.get("run_group", ""),
                "parse_error": "",
            }
        rebased, exists, digest = relocate_stage2_path(row["source_image"])
        row.update({"source_image_rebased_path": rebased, "source_image_exists": exists, "source_image_sha256": digest})
        rows.append(row)
    return pd.DataFrame(rows), run_meta, all_json_count


def strict_match(candidate_q: list[float | None], candidate_n: list[int], source_q: list[float | None], source_n: list[int]) -> tuple[bool, int, float]:
    matched_months = 0
    max_rel_error = 0.0
    for left, left_n, right, right_n in zip(candidate_q, candidate_n, source_q, source_n):
        if int(left_n) != int(right_n):
            return False, matched_months, float("inf")
        if left is None and right is None:
            matched_months += 1
            continue
        if left is None or right is None:
            return False, matched_months, float("inf")
        rel = abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1.0)
        max_rel_error = max(max_rel_error, rel)
        if rel > 1e-6:
            return False, matched_months, max_rel_error
        matched_months += 1
    return matched_months == 12, matched_months, max_rel_error


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    frame.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = read_workbook(MONTHLY_WORKBOOK_LABEL, WORKBOOKS[MONTHLY_WORKBOOK_LABEL]).copy()
    daily_entry_meta, daily_entry_rows, daily_entry_annual = read_daily_entry_snapshot(WORKBOOKS["frozen_daily_entry_snapshot"])
    write_csv(daily_entry_rows.sort_values(["station_name", "year", "month", "daily_candidate_rank"]), "daily_entry_month_rows.csv")
    write_csv(daily_entry_annual.sort_values(["station_name", "year", "daily_candidate_rank"]), "daily_entry_annual_vectors.csv")
    key = ["station_norm", "year", "month"]
    frozen["candidate_rank"] = frozen.groupby(key, sort=False).cumcount() + 1
    counts = frozen.groupby(key, sort=False).size().rename("candidate_count").reset_index()
    duplicate_keys = counts[counts["candidate_count"] > 1].copy()
    conflict_rows = frozen.merge(duplicate_keys[key], on=key, how="inner", validate="many_to_one").copy()
    q_counts = (
        conflict_rows.groupby(key, sort=False)["monthly_mean_m3_s"]
        .agg(lambda values: len({round(float(x), 12) for x in values if pd.notna(x)}))
        .rename("distinct_q_values")
        .reset_index()
    )
    registry = duplicate_keys.merge(q_counts, on=key, validate="one_to_one")
    registry["q_conflict"] = registry["distinct_q_values"] > 1
    candidate_q = (
        conflict_rows.groupby(key, sort=False)["monthly_mean_m3_s"]
        .agg(lambda s: "|".join("" if pd.isna(x) else format(float(x), ".12g") for x in s))
        .rename("candidate_q_m3_s")
        .reset_index()
    )
    candidate_n = (
        conflict_rows.groupby(key, sort=False)["n_days_used"]
        .agg(lambda s: "|".join("" if pd.isna(x) else str(int(x)) for x in s))
        .rename("candidate_n_days_used")
        .reset_index()
    )
    station_name = frozen.groupby("station_norm", sort=False)["station_name"].first().rename("station_name").reset_index()
    registry = registry.merge(station_name, on="station_norm", validate="many_to_one").merge(candidate_q, on=key).merge(candidate_n, on=key)
    registry = registry.sort_values(["station_name", "year", "month"])
    write_csv(registry, "conflict_registry.csv")
    write_csv(conflict_rows.sort_values(["station_name", "year", "month", "candidate_rank"]), "candidate_month_rows.csv")

    # Preserve candidate rank across interleaved duplicate-month rows: rank 1 is the
    # first occurrence of every month, rank 2 the second, etc.; it is not a 12-row slice.
    yearly_rows: list[dict[str, Any]] = []
    for (station, year, rank), part in conflict_rows.groupby(["station_norm", "year", "candidate_rank"], sort=False):
        by_month = part.set_index("month")
        months = list(range(1, 13))
        qvec = [finite_positive(by_month.at[m, "monthly_mean_m3_s"]) if m in by_month.index else None for m in months]
        nvec = [int(by_month.at[m, "n_days_used"]) if m in by_month.index and pd.notna(by_month.at[m, "n_days_used"]) else 0 for m in months]
        yearly_rows.append(
            {
                "station_norm": station,
                "station_name": str(part["station_name"].iloc[0]),
                "year": int(year),
                "candidate_rank": int(rank),
                "months_present": int(part["month"].nunique()),
                "complete_12_month_vector": bool(part["month"].nunique() == 12),
                "excel_rows": "|".join(map(str, part.sort_values("month")["excel_row"].astype(int))),
                "q_vector_m3_s": json.dumps(qvec, ensure_ascii=False),
                "n_days_vector": json.dumps(nvec, ensure_ascii=False),
            }
        )
    annual = pd.DataFrame(yearly_rows).sort_values(["station_name", "year", "candidate_rank"])
    write_csv(annual, "candidate_annual_vectors.csv")

    # Direct frozen daily-entry evidence: only station/year-constrained strict
    # vectors can link a workbook candidate to a daily-entry candidate.
    daily_match_rows: list[dict[str, Any]] = []
    daily_entry_lookup = defaultdict(list)
    for row in daily_entry_annual.to_dict("records"):
        daily_entry_lookup[(row["station_norm"], int(row["year"]))].append(row)
    daily_match_by_candidate: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for candidate in annual.to_dict("records"):
        qvec, nvec = json.loads(candidate["q_vector_m3_s"]), json.loads(candidate["n_days_vector"])
        exact = []
        for entry in daily_entry_lookup[(candidate["station_norm"], int(candidate["year"]))]:
            matched, month_count, rel_error = strict_match(qvec, nvec, json.loads(entry["q_vector_m3_s"]), json.loads(entry["n_days_vector"]))
            if matched:
                exact.append({**entry, "matched_months": month_count, "max_relative_error": rel_error})
        if not candidate["complete_12_month_vector"]:
            match_status = "incomplete_workbook_candidate_vector"
        elif not exact:
            match_status = "no_station_year_strict_daily_entry_match"
        elif len(exact) == 1:
            match_status = "unique_station_year_strict_daily_entry_match"
        else:
            match_status = "ambiguous_multiple_station_year_strict_daily_entry_matches"
        daily_match_by_candidate[(candidate["station_norm"], int(candidate["year"]), int(candidate["candidate_rank"]))] = exact
        if not exact:
            daily_match_rows.append({**candidate, "daily_entry_match_status": match_status, "daily_candidate_rank": "", "daily_entry_excel_rows": "", "daily_semantic_hashes": "", "matched_months": 0, "max_relative_error": ""})
        else:
            for entry in exact:
                daily_match_rows.append({**candidate, "daily_entry_match_status": match_status, "daily_candidate_rank": entry["daily_candidate_rank"], "daily_entry_excel_rows": entry["daily_entry_excel_rows"], "daily_semantic_hashes": entry["daily_semantic_hashes"], "matched_months": entry["matched_months"], "max_relative_error": entry["max_relative_error"]})
    daily_matches = pd.DataFrame(daily_match_rows)
    write_csv(daily_matches, "daily_entry_candidate_matches.csv")
    # Month-level coverage is reported separately from annual-vector linkage:
    # a single daily row can support a conflict-month candidate without proving
    # which complete annual candidate is the authoritative source.
    monthly_probe = conflict_rows.copy()
    daily_probe = daily_entry_rows.copy()
    monthly_probe["_q_key"] = monthly_probe["monthly_mean_m3_s"].round(12)
    daily_probe["_q_key"] = daily_probe["monthly_mean_m3_s"].round(12)
    monthly_probe["_n_key"] = monthly_probe["n_days_used"].fillna(-1).astype(int)
    daily_probe["_n_key"] = daily_probe["n_days_used"].fillna(-1).astype(int)
    direct_month_rows = monthly_probe.merge(
        daily_probe[["station_norm", "year", "month", "_q_key", "_n_key", "daily_entry_excel_row", "daily_semantic_sha256"]],
        on=["station_norm", "year", "month", "_q_key", "_n_key"], how="inner",
    )
    direct_month_keys = direct_month_rows[["station_norm", "year", "month"]].drop_duplicates()
    direct_q_conflict_keys = registry[registry["q_conflict"]].merge(direct_month_keys, on=["station_norm", "year", "month"], how="inner")

    json_index, run_meta, all_json_count = extract_json_index()
    write_csv(json_index.sort_values(["run_dir", "json_type", "station_name"], na_position="last"), "stage2_json_lineage_index.csv")
    json_by_run = defaultdict(list)
    for row in json_index[json_index["json_type"].eq("station_table_location")].fillna("").to_dict("records"):
        json_by_run[str(row["run_dir"])].append(row)

    catalog = list(iter_stage2_daily_artifacts())
    # The JSON-embedded CSV matrix is a separate numerical evidence channel.
    # It remains linked to its worker_result even when a materialized XLSX/CSV
    # is unavailable in the current stage2_work package.
    for row in json_index[json_index["json_type"].eq("worker_station_result")].fillna("").to_dict("records"):
        if not row.get("embedded_monthly_means"):
            continue
        run_year = csv_year(Path(str(row["run_dir"])), STAGE2_WORK)
        catalog.append({
            "catalog_kind": "stage2_worker_embedded_csv_rows",
            "daily_csv_path": str(row.get("json_path", "")),
            "station_name": str(row.get("station_name", "")),
            "station_norm": str(row.get("station_norm", "")),
            "year": run_year,
            "monthly_means": json.loads(str(row["embedded_monthly_means"])),
            "n_days_used": json.loads(str(row["embedded_n_days_used"])),
            "csv_sha256": str(row.get("csv_rows_sha256", "")),
            "work_run_dir": str(row.get("run_dir", "")),
        })
    catalog_df = pd.DataFrame(catalog)
    original = catalog_df[catalog_df["catalog_kind"].eq("stage2_result")].copy()
    revised = catalog_df[catalog_df["catalog_kind"].eq("stage2_result_revised")].copy()
    pair_keys = sorted(set(zip(original.station_norm, original.year)).union(zip(revised.station_norm, revised.year)))
    original_map = {(row.station_norm, row.year): row for row in original.itertuples(index=False)}
    revised_map = {(row.station_norm, row.year): row for row in revised.itertuples(index=False)}
    comparison: list[dict[str, Any]] = []
    for item in pair_keys:
        left, right = original_map.get(item), revised_map.get(item)
        if left is None:
            status = "revised_only"
            eq, months, err = False, 0, float("nan")
        elif right is None:
            status = "original_only"
            eq, months, err = False, 0, float("nan")
        else:
            eq, months, err = strict_match(left.monthly_means, left.n_days_used, right.monthly_means, right.n_days_used)
            status = "strict_equal" if eq else "different_daily_fingerprint"
        comparison.append(
            {
                "station_norm": item[0], "year": item[1], "comparison_status": status,
                "original_csv_path": getattr(left, "daily_csv_path", ""),
                "revised_csv_path": getattr(right, "daily_csv_path", ""),
                "matched_months": months, "max_relative_error": err,
            }
        )
    write_csv(pd.DataFrame(comparison), "stage2_original_vs_revised.csv")

    lineage_rows: list[dict[str, Any]] = []
    # Preserve all discovered page/image evidence even if no workbook candidate
    # fingerprints it.  An empty candidate match is evidence of non-linkage, not
    # evidence that the original image is absent.
    source_manifest_rows: list[dict[str, Any]] = [
        {
            "json_path": row.get("json_path", ""),
            "json_type": row.get("json_type", ""),
            "run_dir": row.get("run_dir", ""),
            "source_image": row.get("source_image", ""),
            "source_image_rebased_path": row.get("source_image_rebased_path", ""),
            "source_image_exists": row.get("source_image_exists", False),
            "source_image_sha256": row.get("source_image_sha256", ""),
            "table_path": row.get("table_path", ""),
            "table_path_rebased_path": row.get("table_path_rebased_path", ""),
            "table_path_exists": row.get("table_path_exists", False),
            "table_path_sha256": row.get("table_path_sha256", ""),
            "table_crop_path": row.get("table_crop_path", ""),
            "table_crop_path_rebased_path": row.get("table_crop_path_rebased_path", ""),
            "table_crop_path_exists": row.get("table_crop_path_exists", False),
            "table_crop_path_sha256": row.get("table_crop_path_sha256", ""),
            "table_bbox": row.get("table_bbox", ""),
            "json_station_query": row.get("station_query", ""),
            "xlsx_path": row.get("xlsx_path", ""),
            "xlsx_path_rebased_path": row.get("xlsx_path_rebased_path", ""),
            "xlsx_path_exists": row.get("xlsx_path_exists", False),
            "xlsx_path_sha256": row.get("xlsx_path_sha256", ""),
            "table_image_path": row.get("table_image_path", ""),
            "table_image_path_rebased_path": row.get("table_image_path_rebased_path", ""),
            "table_image_path_exists": row.get("table_image_path_exists", False),
            "table_image_path_sha256": row.get("table_image_path_sha256", ""),
            "candidate_station_name": "",
            "candidate_year": "",
            "candidate_rank": "",
            "candidate_lineage_status": "unlinked_no_strict_candidate_match",
            "daily_csv_path": "",
            "daily_csv_sha256": "",
        }
        for row in json_index.fillna("").to_dict("records")
        if row.get("source_image") or row.get("table_path") or row.get("table_crop_path")
    ]
    for candidate in annual.to_dict("records"):
        qvec = json.loads(candidate["q_vector_m3_s"])
        nvec = json.loads(candidate["n_days_vector"])
        all_numeric_matches: list[dict[str, Any]] = []
        for source in catalog:
            matched, month_count, rel_error = strict_match(qvec, nvec, source["monthly_means"], source["n_days_used"])
            if matched:
                all_numeric_matches.append({**source, "matched_months": month_count, "max_relative_error": rel_error})
        station_year_matches = [source for source in all_numeric_matches if source["year"] == candidate["year"] and source["station_norm"] == candidate["station_norm"]]
        json_station_year_matches = [
            source for source in all_numeric_matches
            if source["year"] == candidate["year"]
            and any(loc.get("station_norm") == candidate["station_norm"] for loc in json_by_run.get(source["work_run_dir"], []))
        ]
        year_only_matches = [source for source in all_numeric_matches if source["year"] == candidate["year"]]
        if station_year_matches:
            matches, match_scope = station_year_matches, "strict_station_year_numeric_match"
        elif json_station_year_matches:
            matches, match_scope = json_station_year_matches, "strict_year_json_station_numeric_match"
        elif year_only_matches:
            matches, match_scope = year_only_matches, "strict_year_only_numeric_match"
        else:
            matches, match_scope = all_numeric_matches, "strict_global_numeric_match"
        daily_direct = daily_match_by_candidate[(candidate["station_norm"], int(candidate["year"]), int(candidate["candidate_rank"]))]
        daily_status_rows = daily_matches[
            (daily_matches["station_norm"] == candidate["station_norm"])
            & (daily_matches["year"] == candidate["year"])
            & (daily_matches["candidate_rank"] == candidate["candidate_rank"])
        ]
        daily_status = str(daily_status_rows["daily_entry_match_status"].iloc[0]) if len(daily_status_rows) else "not_evaluated"
        daily_rows_joined = "|".join(str(row["daily_entry_excel_rows"]) for row in daily_direct)
        daily_hashes_joined = "|".join(str(row["daily_semantic_hashes"]) for row in daily_direct)
        if not candidate["complete_12_month_vector"]:
            status = "incomplete_candidate_vector"
        elif not matches:
            status = "no_strict_numeric_fingerprint_match"
        elif len(matches) == 1:
            status = f"unique_{match_scope}"
        else:
            status = f"ambiguous_multiple_{match_scope}"
        if not matches:
            lineage_rows.append({**candidate, "lineage_status": status, "daily_entry_match_status": daily_status, "daily_entry_excel_rows": daily_rows_joined, "daily_entry_semantic_hashes": daily_hashes_joined, "daily_csv_path": "", "catalog_kind": "", "json_path": "", "source_image": "", "table_path": "", "matched_months": 0, "max_relative_error": ""})
            continue
        for source in matches:
            locs = json_by_run.get(source["work_run_dir"], [])
            station_locs = [x for x in locs if x.get("station_norm") == candidate["station_norm"]]
            selected_locs = station_locs or locs
            if not selected_locs:
                worker = run_meta.get(source["work_run_dir"], {})
                selected_locs = [{"json_path": worker.get("worker_json_path", ""), "source_image": worker.get("worker_image_path") or worker.get("run_image_path", ""), "table_path": "", "table_crop_path": "", "table_bbox": "", "station_query": ""}]
            for loc in selected_locs:
                row = {
                    **candidate,
                    "lineage_status": status,
                    "daily_entry_match_status": daily_status,
                    "daily_entry_excel_rows": daily_rows_joined,
                    "daily_entry_semantic_hashes": daily_hashes_joined,
                    "daily_csv_path": source["daily_csv_path"],
                    "catalog_kind": source["catalog_kind"],
                    "daily_csv_sha256": source["csv_sha256"],
                    "json_path": loc.get("json_path", ""),
                    "source_image": loc.get("source_image", ""),
                    "table_path": loc.get("table_path", ""),
                    "table_crop_path": loc.get("table_crop_path", ""),
                    "table_bbox": loc.get("table_bbox", ""),
                    "json_station_query": loc.get("station_query", ""),
                    "matched_months": source["matched_months"],
                    "max_relative_error": source["max_relative_error"],
                }
                lineage_rows.append(row)
                source_manifest_rows.append({
                    "json_path": loc.get("json_path", ""), "json_type": "strict_candidate_link",
                    "run_dir": source.get("work_run_dir", ""), "source_image": loc.get("source_image", ""),
                    "source_image_rebased_path": loc.get("source_image_rebased_path", ""),
                    "source_image_exists": loc.get("source_image_exists", False),
                    "source_image_sha256": loc.get("source_image_sha256", ""),
                    "table_path": loc.get("table_path", ""), "table_crop_path": loc.get("table_crop_path", ""),
                    "table_bbox": loc.get("table_bbox", ""), "json_station_query": loc.get("station_query", ""),
                    "candidate_station_name": candidate["station_name"], "candidate_year": candidate["year"],
                    "candidate_rank": candidate["candidate_rank"], "candidate_lineage_status": status,
                    "daily_csv_path": source["daily_csv_path"], "daily_csv_sha256": source["csv_sha256"],
                })
    lineage = pd.DataFrame(lineage_rows)
    write_csv(lineage, "candidate_to_json_lineage.csv")
    write_csv(pd.DataFrame(source_manifest_rows), "source_image_manifest.csv")

    # Compact run manifest enables reproducible reporting without changing source assets.
    report = {
        "workbooks": {
            MONTHLY_WORKBOOK_LABEL: {"path": str(WORKBOOKS[MONTHLY_WORKBOOK_LABEL]), "sha256": sha256(WORKBOOKS[MONTHLY_WORKBOOK_LABEL]), "rows_2006_2009": int(len(frozen))},
            "frozen_daily_entry_snapshot": daily_entry_meta,
        },
        "frozen_duplicate_keys": int(len(duplicate_keys)),
        "frozen_q_conflict_keys": int(registry["q_conflict"].sum()),
        "candidate_year_vectors": int(len(annual)),
        "complete_candidate_year_vectors": int(annual["complete_12_month_vector"].sum()),
        "daily_entry_month_rows": int(len(daily_entry_rows)),
        "daily_entry_year_vectors": int(len(daily_entry_annual)),
        "daily_entry_candidate_match_status_counts": dict(Counter(daily_matches.drop_duplicates(["station_norm", "year", "candidate_rank"])["daily_entry_match_status"])),
        "daily_entry_exact_month_candidate_rows": int(len(direct_month_rows)),
        "daily_entry_exact_duplicate_month_keys": int(len(direct_month_keys)),
        "daily_entry_exact_q_conflict_keys": int(len(direct_q_conflict_keys)),
        "stage2_daily_artifact_catalog_rows": int(len(catalog)),
        "stage2_daily_artifact_catalog_by_kind": dict(Counter(item["catalog_kind"] for item in catalog)),
        "stage2_work_json_files_scanned": all_json_count,
        "stage2_lineage_index_rows": int(len(json_index)),
        "stage2_original_vs_revised_status_counts": dict(Counter(item["comparison_status"] for item in comparison)),
        "candidate_lineage_status_counts": dict(Counter(lineage["lineage_status"])) if len(lineage) else {},
        "created_files": sorted(str(path.relative_to(RUN)) for path in OUT.glob("*.csv")),
    }
    (RUN / "reports" / "audit_run_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
