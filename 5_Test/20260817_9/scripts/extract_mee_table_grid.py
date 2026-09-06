"""Extract MEE annual WWTP inventories from ruled PDF page images.

The official 2007-2014 inventories are image PDFs.  Full-page OCR loses the
column boundaries, so this extractor uses the printed grid as primary
structure, removes the grid before OCR, then assigns TSV words back to their
original cells by coordinates.  The document row order is retained separately
from the OCR of the printed serial number; the two are compared in QA.

Run only with the ``sparrow`` conda environment, for example::

    D:/ProgramData/anaconda3/envs/sparrow/python.exe \
      scripts/extract_mee_table_grid.py --year 2009 --jobs 6
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260817_9"
TESSERACT = Path(r"D:\Program Files\Tesseract-OCR\tesseract.exe")
EXPECTED = {
    2007: 1178,
    2008: 1521,
    2009: 1916,
    2010: 2739,
    2011: 3184,
    2012: 3836,
    2013: 4136,
    2014: 4436,
}


def clusters(indices: np.ndarray) -> list[int]:
    if len(indices) == 0:
        return []
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return [int(np.median(group)) for group in groups if len(group)]


def detect_layout(first_page: Path) -> list[int]:
    gray = np.asarray(Image.open(first_page).convert("L"))
    score = (gray < 100).sum(axis=0)
    x_lines = clusters(np.where(score > score.max() * 0.70)[0])
    # The inventories have seven columns in 2007-2008 and eight thereafter.
    if len(x_lines) not in (8, 9):
        raise RuntimeError(f"Unexpected vertical grid in {first_page}: {x_lines}")
    return x_lines


def detect_horizontal_lines(gray: np.ndarray, x_lines: list[int]) -> list[int]:
    # All record boundaries traverse the six rightmost columns.  Province/city
    # cells are vertically merged, so the full table width cannot be used.
    x0, x1 = x_lines[-6], x_lines[-1]
    score = (gray[:, x0 : x1 + 1] < 100).sum(axis=1)
    y_lines = clusters(np.where(score > (x1 - x0) * 0.65)[0])
    if len(y_lines) < 3:
        raise RuntimeError(f"Could not identify table rows; horizontal lines={y_lines}")
    return y_lines


def remove_grid(gray: np.ndarray, x_lines: list[int], y_lines: list[int]) -> np.ndarray:
    cleaned = gray.copy()
    table_width = x_lines[-1] - x_lines[0]
    full_score = (gray[:, x_lines[0] : x_lines[-1] + 1] < 100).sum(axis=1)
    # Include partial header and merged-cell rules in addition to the record
    # boundaries.  Text strokes never span 30% of the table width.
    all_horizontal = clusters(np.where(full_score > table_width * 0.30)[0])
    for x in x_lines:
        cleaned[:, max(0, x - 3) : min(cleaned.shape[1], x + 4)] = 255
    for y in sorted(set(y_lines + all_horizontal)):
        cleaned[max(0, y - 3) : min(cleaned.shape[0], y + 4), :] = 255
    return cleaned


def make_tsv(page: Path, tsv_path: Path, x_lines: list[int], force: bool) -> None:
    if tsv_path.exists() and not force:
        return
    gray = np.asarray(Image.open(page).convert("L"))
    y_lines = detect_horizontal_lines(gray, x_lines)
    cleaned = remove_grid(gray, x_lines, y_lines)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mee_grid_") as temp_dir:
        clean_page = Path(temp_dir) / page.name
        Image.fromarray(cleaned).save(clean_page)
        output_stem = Path(temp_dir) / "ocr"
        subprocess.run(
            [
                str(TESSERACT),
                str(clean_page),
                str(output_stem),
                "-l",
                "chi_sim+eng",
                "--psm",
                "4",
                "tsv",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        produced = output_stem.with_suffix(".tsv")
        tsv_path.write_bytes(produced.read_bytes())


def page_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def join_cell(words: pd.DataFrame, x0: int, x1: int, y0: int, y1: int) -> tuple[str, float | None]:
    selected = words[
        words.x_center.between(x0 + 2, x1 - 2)
        & words.y_center.between(y0 + 2, y1 - 2)
    ].copy()
    if selected.empty:
        return "", None
    # Tesseract gives characters on the same printed line slightly different
    # tops/heights.  Sorting directly by top scrambles Chinese text.  Cluster
    # centres into visual lines first, then read left-to-right within each line.
    by_y = selected.sort_values("y_center")
    line_number = 0
    previous = None
    line_numbers: dict[int, int] = {}
    for index, y_center in zip(by_y.index, by_y.y_center):
        if previous is not None and y_center - previous > 25:
            line_number += 1
        line_numbers[index] = line_number
        previous = y_center
    selected["visual_line"] = pd.Series(line_numbers)
    selected = selected.sort_values(["visual_line", "left"])
    # Chinese words should be contiguous.  Slashes and Latin process names also
    # remain unambiguous without injected spaces.
    text = "".join(selected.text.astype(str)).replace("|", "").strip()
    conf = float(pd.to_numeric(selected.conf, errors="coerce").mean())
    return text, conf


def merged_cell_at(
    words: pd.DataFrame,
    gray: np.ndarray,
    x0: int,
    x1: int,
    center_y: float,
) -> tuple[str, float | None]:
    """Read a province/city cell that may span several facility rows."""
    score = (gray[:, x0 : x1 + 1] < 100).sum(axis=1)
    boundaries = clusters(np.where(score > (x1 - x0) * 0.65)[0])
    lower = max((y for y in boundaries if y < center_y), default=0)
    upper = min((y for y in boundaries if y > center_y), default=gray.shape[0] - 1)
    return join_cell(words, x0, x1, lower, upper)


def parse_float(raw: str) -> float | None:
    value = raw.strip().replace("，", ".").replace(",", ".")
    value = value.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
    value = re.sub(r"[^0-9.\-]", "", value)
    if not value or value in {".", "-", "-."}:
        return None
    # Keep one decimal point if OCR duplicated punctuation.
    if value.count(".") > 1:
        first = value.index(".")
        value = value[: first + 1] + value[first + 1 :].replace(".", "")
    try:
        return float(value)
    except ValueError:
        return None


def parse_identifier(raw: str) -> int | None:
    value = raw.translate(
        str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "L": "1", "T": "7", "S": "5", "B": "8"})
    )
    digits = re.sub(r"\D", "", value)
    return int(digits) if digits else None


def parse_date(raw: str) -> tuple[int | None, int | None]:
    value = raw.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
    year_match = re.search(r"((?:19|20)\d{2})", value)
    if not year_match:
        return None, None
    tail = value[year_match.end() :]
    month_match = re.search(r"(1[0-2]|0?[1-9])", tail)
    return int(year_match.group(1)), int(month_match.group(1)) if month_match else None


def parse_page(page: Path, tsv_path: Path, x_lines: list[int], year: int) -> list[dict[str, object]]:
    gray = np.asarray(Image.open(page).convert("L"))
    y_lines = detect_horizontal_lines(gray, x_lines)
    # OCR text can legitimately contain an unmatched quote.  TSV has no CSV
    # quoting contract, so quotes must be treated as ordinary cell text.
    raw = pd.read_csv(
        tsv_path,
        sep="\t",
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )
    words = raw[(raw.level == 5) & raw.text.astype(str).str.strip().ne("")].copy()
    for column in ("left", "top", "width", "height", "conf"):
        words[column] = pd.to_numeric(words[column], errors="coerce")
    words["x_center"] = words.left + words.width / 2
    words["y_center"] = words.top + words.height / 2
    names = (
        ["facility_id_raw_ocr", "province_raw_ocr", "facility_name_raw_ocr", "process_raw_ocr", "date_raw_ocr", "design_raw_ocr", "mean_flow_raw_ocr"]
        if len(x_lines) == 8
        else ["facility_id_raw_ocr", "province_raw_ocr", "city_raw_ocr", "facility_name_raw_ocr", "process_raw_ocr", "date_raw_ocr", "design_raw_ocr", "mean_flow_raw_ocr"]
    )
    records: list[dict[str, object]] = []
    # The first interval is a repeated header in 2007-2013.  The 2014 PDF is a
    # different export: only page 1 has a header and every continuation page
    # starts directly with a facility row.
    if year == 2014 and page_number(page) > 1:
        row_intervals = zip(y_lines[:-1], y_lines[1:])
    else:
        row_intervals = zip(y_lines[1:-1], y_lines[2:])
    for local_row, (y0, y1) in enumerate(row_intervals, start=1):
        item: dict[str, object] = {
            "source_year": year,
            "source_page": page_number(page),
            "source_row_number": local_row,
            "grid_y0": y0,
            "grid_y1": y1,
        }
        confidences = []
        for field, x0, x1 in zip(names, x_lines[:-1], x_lines[1:]):
            value, confidence = join_cell(words, x0, x1, y0, y1)
            item[field] = value
            item[field.replace("raw_ocr", "ocr_confidence")] = confidence
            if confidence is not None:
                confidences.append(confidence)
        # Province and city are commonly printed once in a vertically merged
        # cell.  Read the complete merged cell rather than only the row holding
        # the centred glyphs.
        for field, column_index in (("province_raw_ocr", 1), ("city_raw_ocr", 2)):
            if field not in names:
                continue
            value, confidence = merged_cell_at(
                words, gray, x_lines[column_index], x_lines[column_index + 1], (y0 + y1) / 2
            )
            item[field] = value
            item[field.replace("raw_ocr", "ocr_confidence")] = confidence
        raw_id = str(item["facility_id_raw_ocr"])
        item["facility_id_ocr"] = parse_identifier(raw_id)
        commission_year, commission_month = parse_date(str(item["date_raw_ocr"]))
        item["commission_year"] = commission_year
        item["commission_month"] = commission_month
        design = parse_float(str(item["design_raw_ocr"]))
        mean_flow = parse_float(str(item["mean_flow_raw_ocr"]))
        item["design_capacity_source_value"] = design
        item["mean_daily_flow_source_value"] = mean_flow
        # 2007 reports tonnes/day.  For municipal wastewater the conventional
        # density conversion 1 tonne water = 1 m3 is explicit and auditable.
        factor = 1.0 if year == 2007 else 10000.0
        item["source_flow_unit"] = "tonne/day" if year == 2007 else "10^4 m3/day"
        item["design_capacity_m3_d"] = design * factor if design is not None else None
        item["mean_daily_flow_m3_d"] = mean_flow * factor if mean_flow is not None else None
        item["row_mean_ocr_confidence"] = float(np.mean(confidences)) if confidences else None
        item["raw_row_ocr"] = " | ".join(str(item[field]) for field in names)
        records.append(item)
    return records


def run_year(year: int, jobs: int, force: bool, first: int | None, last: int | None) -> dict[str, object]:
    page_dir = RUN / "work" / "mee_ocr" / str(year) / "pages"
    pages = sorted(page_dir.glob("page-*.png"), key=page_number)
    if first is not None:
        pages = [page for page in pages if page_number(page) >= first]
    if last is not None:
        pages = [page for page in pages if page_number(page) <= last]
    if not pages:
        raise RuntimeError(f"No rendered pages for {year}: {page_dir}")
    all_pages = sorted(page_dir.glob("page-*.png"), key=page_number)
    x_lines = detect_layout(all_pages[0])
    tsv_dir = RUN / "work" / "mee_grid" / str(year) / "tsv"
    tsv_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(make_tsv, page, tsv_dir / f"{page.stem}.tsv", x_lines, force): page
            for page in pages
        }
        for future in as_completed(futures):
            future.result()
    records: list[dict[str, object]] = []
    per_page_rows: dict[int, int] = {}
    for page in pages:
        parsed = parse_page(page, tsv_dir / f"{page.stem}.tsv", x_lines, year)
        per_page_rows[page_number(page)] = len(parsed)
        records.extend(parsed)
    out = pd.DataFrame(records).sort_values(["source_page", "source_row_number"]).reset_index(drop=True)
    out.insert(0, "facility_id_grid_sequence", np.arange(1, len(out) + 1))
    out["facility_id_ocr_matches_grid"] = (
        pd.to_numeric(out.facility_id_ocr, errors="coerce") == out.facility_id_grid_sequence
    )
    for field in ("province_raw_ocr", "city_raw_ocr"):
        if field in out:
            out[field.replace("raw_ocr", "filled")] = out[field].replace("", pd.NA).ffill()
    valid_date = out.commission_year.between(1900, year) & out.commission_month.between(1, 12)
    valid_flows = out.design_capacity_m3_d.ge(0) & out.mean_daily_flow_m3_d.ge(0)
    name_ok = out.facility_name_raw_ocr.astype(str).str.strip().ne("")
    process_ok = out.process_raw_ocr.astype(str).str.strip().ne("")
    location_ok = out.province_filled.notna()
    complete = valid_date & valid_flows & name_ok & process_ok & location_ok
    out["automatic_six_field_parse_ok"] = complete
    staged = RUN / "inputs" / "staged" / "mee_grid_candidates"
    staged.mkdir(parents=True, exist_ok=True)
    out.to_parquet(staged / f"mee_{year}_grid_candidates.parquet", index=False)
    out.to_csv(staged / f"mee_{year}_grid_candidates.csv", index=False, encoding="utf-8-sig")
    qa = {
        "year": year,
        "rendered_pages_parsed": len(pages),
        "vertical_grid_lines": x_lines,
        "grid_rows": int(len(out)),
        "expected_official_facilities": EXPECTED[year],
        "official_count_closed": bool(len(out) == EXPECTED[year]),
        "printed_ids_matching_grid_sequence": int(out.facility_id_ocr_matches_grid.sum()),
        "valid_commission_year_month_rows": int(valid_date.sum()),
        "nonnegative_both_flow_rows": int(valid_flows.sum()),
        "nonempty_facility_name_rows": int(name_ok.sum()),
        "nonempty_process_rows": int(process_ok.sum()),
        "automatic_six_field_parse_ok_rows": int(complete.sum()),
        "minimum_rows_on_page": min(per_page_rows.values()),
        "maximum_rows_on_page": max(per_page_rows.values()),
        "status": "candidate_only" if not bool(complete.all() and len(out) == EXPECTED[year]) else "structural_and_field_qa_pass",
        "note": "facility_id_grid_sequence is document row order; facility_id_ocr remains independent OCR evidence",
    }
    qa_dir = RUN / "inputs" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / f"mee_{year}_grid_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, action="append", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--first", type=int)
    parser.add_argument("--last", type=int)
    args = parser.parse_args()
    if not TESSERACT.exists():
        raise FileNotFoundError(TESSERACT)
    for year in args.year:
        if year not in EXPECTED:
            raise ValueError(year)
        run_year(year, max(1, args.jobs), args.force, args.first, args.last)


if __name__ == "__main__":
    main()
