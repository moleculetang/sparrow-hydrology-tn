"""Targeted cell OCR repairs for MEE grid candidates.

The grid extractor deliberately performs one OCR call per page.  This second
pass re-OCRs only fields that need cell segmentation, especially vertically
merged province labels and failed dates/flows.  Both the page OCR and repaired
cell text are retained for audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

from extract_mee_table_grid import (
    EXPECTED,
    RUN,
    TESSERACT,
    clusters,
    detect_horizontal_lines,
    detect_layout,
    parse_date,
    parse_float,
    remove_grid,
)


PROVINCE_MAP = {
    "北京": "Beijing", "天津": "Tianjin", "河北": "Hebei", "山西": "Shanxi",
    "内蒙古": "Inner Mongolia", "辽宁": "Liaoning", "吉林": "Jilin", "黑龙江": "Heilongjiang",
    "上海": "Shanghai", "江苏": "Jiangsu", "浙江": "Zhejiang", "安徽": "Anhui",
    "福建": "Fujian", "江西": "Jiangxi", "山东": "Shandong", "河南": "Henan",
    "湖北": "Hubei", "湖南": "Hunan", "广东": "Guangdong", "广西": "Guangxi",
    "海南": "Hainan", "重庆": "Chongqing", "四川": "Sichuan", "贵州": "Guizhou",
    "云南": "Yunnan", "西藏": "Tibet", "陕西": "Shaanxi", "甘肃": "Gansu",
    "青海": "Qinghai", "宁夏": "Ningxia", "新疆": "Xinjiang",
}
PROVINCE_ORDER = list(PROVINCE_MAP.values())


def compact(text: str) -> str:
    return re.sub(r"[\s\"'“”‘’，。:：;；|]+", "", str(text)).strip()


def normalize_province(text: str) -> tuple[str | None, str | None]:
    value = compact(text)
    for chinese, english in PROVINCE_MAP.items():
        if chinese in value:
            return chinese, english
    chinese_only = "".join(re.findall(r"[\u4e00-\u9fff]", value))
    if 2 <= len(chinese_only) <= 4:
        scores = sorted(
            ((SequenceMatcher(None, chinese_only, candidate).ratio(), candidate, english)
             for candidate, english in PROVINCE_MAP.items()),
            reverse=True,
        )
        if scores[0][0] >= 0.50 and (scores[0][0] - scores[1][0]) >= 0.15:
            return scores[0][1], scores[0][2]
    return None, None


def chinese_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", str(text)))


def looks_like_blank_cell_hallucination(text: object) -> bool:
    value = compact(text)
    if not value:
        return True
    chinese = chinese_count(value)
    ascii_alnum = len(re.findall(r"[A-Za-z0-9]", value))
    return ascii_alnum > max(2, chinese * 2)


def column_segment_bounds(gray: np.ndarray, x0: int, x1: int, center_y: float) -> tuple[int, int]:
    score = (gray[:, x0 : x1 + 1] < 100).sum(axis=1)
    boundaries = clusters(np.where(score > (x1 - x0) * 0.65)[0])
    lower = max((y for y in boundaries if y < center_y), default=0)
    upper = min((y for y in boundaries if y > center_y), default=gray.shape[0] - 1)
    return lower + 4, upper - 4


def ocr_cell(cleaned: np.ndarray, bbox: tuple[int, int, int, int], language: str, psm: int) -> str:
    x0, y0, x1, y1 = bbox
    crop = cleaned[max(0, y0) : min(cleaned.shape[0], y1), max(0, x0) : min(cleaned.shape[1], x1)]
    if crop.size == 0:
        return ""
    with tempfile.TemporaryDirectory(prefix="mee_cell_") as temp_dir:
        temp = Path(temp_dir)
        image_path = temp / "cell.png"
        output_stem = temp / "ocr"
        Image.fromarray(crop).save(image_path)
        subprocess.run(
            [str(TESSERACT), str(image_path), str(output_stem), "-l", language, "--psm", str(psm)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        text_path = output_stem.with_suffix(".txt")
        return "" if not text_path.exists() else "".join(text_path.read_text(encoding="utf-8", errors="replace").split())


def repair_page(year: int, page_number: int, rows: pd.DataFrame, x_lines: list[int]) -> pd.DataFrame:
    page_dir = RUN / "work" / "mee_ocr" / str(year) / "pages"
    candidates = list(page_dir.glob(f"page-*{page_number:02d}.png")) + list(page_dir.glob(f"page-*{page_number:03d}.png"))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        # Avoid suffix ambiguity by resolving through the numeric stem.
        candidates = [p for p in page_dir.glob("page-*.png") if int(re.search(r"(\d+)$", p.stem).group(1)) == page_number]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one image for {year} page {page_number}, got {candidates}")
    gray = np.asarray(Image.open(candidates[0]).convert("L"))
    y_lines = detect_horizontal_lines(gray, x_lines)
    cleaned = remove_grid(gray, x_lines, y_lines)
    out = rows.copy()
    province_cache: dict[tuple[int, int], tuple[str, str | None, str | None]] = {}
    city_cache: dict[tuple[int, int], str] = {}
    for index, row in out.iterrows():
        y0, y1 = int(row.grid_y0), int(row.grid_y1)
        center = (y0 + y1) / 2
        py0, py1 = column_segment_bounds(gray, x_lines[1], x_lines[2], center)
        province_key = (py0, py1)
        if province_key not in province_cache:
            cell_text = ocr_cell(cleaned, (x_lines[1] + 4, py0, x_lines[2] - 4, py1), "chi_sim", 6)
            chinese, english = normalize_province(cell_text)
            province_cache[province_key] = (cell_text, chinese, english)
        cell_text, province_chinese, province_english = province_cache[province_key]
        out.at[index, "province_cell_ocr"] = cell_text
        out.at[index, "province_chinese"] = province_chinese
        out.at[index, "province_english"] = province_english
        out.at[index, "province_segment_y0"] = py0
        out.at[index, "province_segment_y1"] = py1

        if len(x_lines) == 9:
            cy0, cy1 = column_segment_bounds(gray, x_lines[2], x_lines[3], center)
            city_key = (cy0, cy1)
            existing_city = str(row.get("city_raw_ocr", ""))
            if city_key not in city_cache:
                city_cache[city_key] = (
                    existing_city
                    if chinese_count(existing_city) >= 1
                    else ocr_cell(cleaned, (x_lines[2] + 4, cy0, x_lines[3] - 4, cy1), "chi_sim", 6)
                )
            out.at[index, "city_repaired"] = compact(city_cache[city_key])

        raw_date = str(row.date_raw_ocr)
        commission_year, commission_month = parse_date(raw_date)
        date_repaired = False
        if commission_year is None or commission_month is None:
            date_text = ocr_cell(cleaned, (x_lines[-4] + 4, y0 + 4, x_lines[-3] - 4, y1 - 4), "chi_sim+eng", 7)
            retry_year, retry_month = parse_date(date_text)
            if retry_year is not None:
                commission_year = retry_year
            if retry_month is not None:
                commission_month = retry_month
            out.at[index, "date_cell_ocr"] = date_text
            date_repaired = bool(date_text)
        out.at[index, "commission_year_repaired"] = commission_year
        out.at[index, "commission_month_repaired"] = commission_month
        out.at[index, "date_cell_repair_used"] = date_repaired

        for raw_field, repaired_raw, output_field, left_index, right_index in (
            ("design_raw_ocr", "design_cell_ocr", "design_capacity_source_value_repaired", -3, -2),
            ("mean_flow_raw_ocr", "mean_flow_cell_ocr", "mean_daily_flow_source_value_repaired", -2, -1),
        ):
            value = parse_float(str(row[raw_field]))
            used = False
            if value is None or value < 0:
                cell_text = ocr_cell(
                    cleaned,
                    (x_lines[left_index] + 4, y0 + 4, x_lines[right_index] - 4, y1 - 4),
                    "eng",
                    7,
                )
                retry = parse_float(cell_text)
                if retry is not None:
                    value = retry
                out.at[index, repaired_raw] = cell_text
                used = bool(cell_text)
            out.at[index, output_field] = value
            out.at[index, output_field.replace("source_value_repaired", "cell_repair_used")] = used

        name = str(row.facility_name_raw_ocr)
        if chinese_count(name) < 2:
            name = ocr_cell(
                cleaned,
                (x_lines[-6] + 4, y0 + 4, x_lines[-5] - 4, y1 - 4),
                "chi_sim+eng",
                6 if y1 - y0 > 120 else 7,
            )
            out.at[index, "facility_name_cell_repair_used"] = True
        else:
            out.at[index, "facility_name_cell_repair_used"] = False
        out.at[index, "facility_name_repaired"] = compact(name)
    return out


def impute_province_sequence(out: pd.DataFrame) -> pd.DataFrame:
    """Resolve unreadable province labels using audited national table order.

    Imputation is performed on printed merged-cell segments, never arbitrary
    rows.  A blank segment at the top of a continuation page inherits the
    preceding province; a nonblank unreadable segment between two provinces
    with exactly one official province between them receives that middle
    province.  All such rows retain an explicit method flag.
    """
    result = out.sort_values("facility_id_grid_sequence").copy()
    result["province_assignment_method"] = np.where(
        result.province_english.notna(), "cell_ocr", "unresolved"
    )
    segment_columns = ["source_page", "province_segment_y0", "province_segment_y1"]
    segments = (
        result.groupby(segment_columns, dropna=False, sort=False)
        .agg(
            first_id=("facility_id_grid_sequence", "min"),
            cell_text=("province_cell_ocr", "first"),
            province=("province_english", "first"),
        )
        .reset_index()
        .sort_values("first_id")
        .reset_index(drop=True)
    )
    for position in segments.index[segments.province.isna()]:
        previous = next((segments.at[i, "province"] for i in range(position - 1, -1, -1) if pd.notna(segments.at[i, "province"])), None)
        following = next((segments.at[i, "province"] for i in range(position + 1, len(segments)) if pd.notna(segments.at[i, "province"])), None)
        assigned = None
        method = "unresolved"
        immediate_previous_province = segments.at[position - 1, "province"] if position > 0 else None
        same_ocr_as_previous_segment = (
            position > 0
            and compact(segments.at[position, "cell_text"])
            and compact(segments.at[position, "cell_text"]) == compact(segments.at[position - 1, "cell_text"])
        )
        is_top_blank_continuation = (
            looks_like_blank_cell_hallucination(segments.at[position, "cell_text"])
            and float(segments.at[position, "province_segment_y0"]) < 500
            and previous is not None
        )
        if same_ocr_as_previous_segment and pd.notna(immediate_previous_province):
            assigned, method = immediate_previous_province, "sequence_repeated_segment_ocr"
        elif is_top_blank_continuation:
            assigned, method = previous, "sequence_top_page_continuation"
        elif previous is None and following in PROVINCE_ORDER and int(segments.at[position, "first_id"]) == 1:
            following_index = PROVINCE_ORDER.index(following)
            if following_index == 0:
                assigned, method = following, "sequence_leading_same_province"
            elif following_index == 1:
                assigned, method = PROVINCE_ORDER[0], "sequence_leading_missing_beijing"
        elif following is None and previous is not None:
            assigned, method = previous, "sequence_trailing_continuation"
        elif previous == following and previous is not None:
            assigned, method = previous, "sequence_same_neighbors"
        elif previous in PROVINCE_ORDER and following in PROVINCE_ORDER:
            previous_index = PROVINCE_ORDER.index(previous)
            following_index = PROVINCE_ORDER.index(following)
            if following_index - previous_index == 2:
                assigned, method = PROVINCE_ORDER[previous_index + 1], "sequence_single_missing_province"
            elif following_index - previous_index == 1:
                chinese_only = "".join(re.findall(r"[\u4e00-\u9fff]", str(segments.at[position, "cell_text"])))
                previous_chinese = next(k for k, v in PROVINCE_MAP.items() if v == previous)
                following_chinese = next(k for k, v in PROVINCE_MAP.items() if v == following)
                previous_similarity = SequenceMatcher(None, chinese_only, previous_chinese).ratio()
                following_similarity = SequenceMatcher(None, chinese_only, following_chinese).ratio()
                if max(previous_similarity, following_similarity) > 0 and previous_similarity != following_similarity:
                    assigned = previous if previous_similarity > following_similarity else following
                    method = "sequence_adjacent_province_ocr_similarity"
                # At an adjacent-province transition, a blank segment on the
                # same page immediately after a valid segment belongs to the
                # preceding merged group; otherwise leave it unresolved.
                prior_page = int(segments.at[position - 1, "source_page"]) if position > 0 else None
                if assigned is None and looks_like_blank_cell_hallucination(segments.at[position, "cell_text"]) and prior_page == int(segments.at[position, "source_page"]):
                    assigned, method = previous, "sequence_blank_adjacent_continuation"
        if assigned is None:
            continue
        key = tuple(segments.loc[position, segment_columns])
        mask = (
            result.source_page.eq(key[0])
            & result.province_segment_y0.eq(key[1])
            & result.province_segment_y1.eq(key[2])
        )
        result.loc[mask, "province_english"] = assigned
        result.loc[mask, "province_chinese"] = next(k for k, v in PROVINCE_MAP.items() if v == assigned)
        result.loc[mask, "province_assignment_method"] = method
        segments.at[position, "province"] = assigned
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, action="append", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--impute-only", action="store_true", help="Reuse existing cell OCR and rerun province normalization/sequence QA")
    args = parser.parse_args()
    staged = RUN / "inputs" / "staged" / "mee_grid_candidates"
    qa_dir = RUN / "inputs" / "qa"
    for year in args.year:
        path = staged / f"mee_{year}_grid_candidates.parquet"
        source = pd.read_parquet(path)
        output_path = staged / f"mee_{year}_grid_repaired.parquet"
        if args.impute_only:
            if not output_path.exists():
                raise FileNotFoundError(output_path)
            out = pd.read_parquet(output_path)
            normalized = out.province_cell_ocr.map(normalize_province)
            out["province_chinese"] = normalized.map(lambda x: x[0])
            out["province_english"] = normalized.map(lambda x: x[1])
        else:
            first_page = sorted((RUN / "work" / "mee_ocr" / str(year) / "pages").glob("page-*.png"))[0]
            x_lines = detect_layout(first_page)
            repaired_pages = []
            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
                futures = {
                    pool.submit(repair_page, year, int(page), rows.copy(), x_lines): int(page)
                    for page, rows in source.groupby("source_page", sort=True)
                }
                for future in as_completed(futures):
                    repaired_pages.append(future.result())
            out = pd.concat(repaired_pages, ignore_index=True).sort_values("facility_id_grid_sequence")
        out = impute_province_sequence(out)
        factor = 1.0 if year == 2007 else 10000.0
        out["design_capacity_m3_d_repaired"] = out.design_capacity_source_value_repaired * factor
        out["mean_daily_flow_m3_d_repaired"] = out.mean_daily_flow_source_value_repaired * factor
        valid_date = out.commission_year_repaired.between(1900, year) & out.commission_month_repaired.between(1, 12)
        valid_flow = out.design_capacity_m3_d_repaired.ge(0) & out.mean_daily_flow_m3_d_repaired.ge(0)
        valid_province = out.province_english.notna()
        out["model_activity_fields_ok"] = valid_flow & valid_province
        out.to_parquet(output_path, index=False)
        out.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        qa = {
            "year": year,
            "rows": int(len(out)),
            "expected_rows": EXPECTED[year],
            "official_count_closed": bool(len(out) == EXPECTED[year]),
            "normalized_province_rows": int(valid_province.sum()),
            "province_assignment_methods": {str(k): int(v) for k, v in out.province_assignment_method.value_counts().items()},
            "valid_both_flow_rows": int(valid_flow.sum()),
            "valid_commission_date_rows": int(valid_date.sum()),
            "model_activity_fields_ok_rows": int(out.model_activity_fields_ok.sum()),
            "date_source_or_ocr_missing_rows": int((~valid_date).sum()),
            "flow_source_or_ocr_missing_rows": int((~valid_flow).sum()),
            "status": "PASS" if len(out) == EXPECTED[year] and valid_province.all() else "REVIEW",
        }
        (qa_dir / f"mee_{year}_field_repair_qa.json").write_text(
            json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
