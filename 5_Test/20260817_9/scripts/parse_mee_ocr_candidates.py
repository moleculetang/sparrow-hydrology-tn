"""Parse auditable *candidate* rows from page-level MEE OCR text.

This parser is intentionally conservative: a row can be structurally found
without being declared model-ready.  Each emitted candidate carries its source
page and the raw OCR block for visual/field-level checking.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260817_9"
EXPECTED = {2007: 1178, 2008: 1521, 2009: 1916, 2010: 2739, 2011: 3184, 2012: 3836, 2013: 4136, 2014: 4436}
# Repairs are limited to visually inspected page-local sequence breaks. They
# remain visible in `facility_id_raw_ocr` and never raise a candidate to
# model-ready status.
ID_GLYPH_REPAIRS = {"LL": "11", "TT": "77", "ALO": "410", "ATT": "477", "SLL": "511", "B1": "81", "Tee": "85", "Cee": "90", "gos": "805", "asp": "852", "L197": "1197"}
SPECIAL_ID_PATTERN = "|".join(re.escape(token) for token in sorted(ID_GLYPH_REPAIRS, key=len, reverse=True))
ID_TOKEN = rf"(?:\d{{1,5}}\.?|{SPECIAL_ID_PATTERN})"
ID_START = re.compile(rf"^({ID_TOKEN})(?:\s*[|)\]〉}}]|\s{{2,}}(?=[^\d\s])|(?<=Cee)\s+)")
ROW_START = re.compile(rf"(?m)^\s*({ID_TOKEN})(?:\s*[|)\]〉}}]|\s{{2,}}(?=[^\d\s])|(?<=Cee)\s+)")


def clean(value: str) -> str:
    return " ".join(value.replace("|", " | ").split()).strip()


def parse_block(block: str) -> dict[str, object]:
    raw_block = block.lstrip()
    identifier = ID_START.match(raw_block)
    normal = clean(raw_block)
    date = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*(?:月|A)?", normal)
    numbers = re.search(r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$", normal)
    record: dict[str, object] = {
        "facility_id_raw_ocr": identifier.group(1) if identifier else pd.NA,
        "facility_id_ocr": pd.NA,
        "facility_id_repaired_from_glyphs": False,
        "commission_year_ocr": int(date.group(1)) if date else pd.NA,
        "commission_month_ocr": int(date.group(2)) if date else pd.NA,
        "design_capacity_t_d_ocr": float(numbers.group(1)) if numbers else pd.NA,
        "mean_daily_flow_t_d_ocr": float(numbers.group(2)) if numbers else pd.NA,
        "province_ocr": pd.NA,
        "facility_name_ocr": pd.NA,
        "treatment_process_ocr": pd.NA,
    }
    if identifier:
        raw_id = identifier.group(1)
        normalized_id = ID_GLYPH_REPAIRS.get(raw_id, raw_id.translate(str.maketrans({"I": "1", "l": "1", "L": "1", "O": "0", "o": "0", "T": "7", "t": "7", "A": "4", "S": "5", "B": "8"})).rstrip("."))
        if normalized_id.isdigit():
            record["facility_id_ocr"] = int(normalized_id)
            record["facility_id_repaired_from_glyphs"] = raw_id != normalized_id
    if identifier and date:
        prefix = normal[len(identifier.group(1)):date.start()].strip(" |")
        fields = [x.strip() for x in prefix.split("|") if x.strip()]
        if len(fields) >= 2:
            record["province_ocr"] = fields[0]
            name_process = " ".join(fields[1:])
        else:
            name_process = prefix
        # OCR generally preserves the original fixed-width gap before process.
        pieces = re.split(r"\s{2,}", name_process)
        if len(pieces) >= 2:
            record["facility_name_ocr"] = pieces[0].strip()
            record["treatment_process_ocr"] = pieces[-1].strip()
        else:
            record["facility_name_ocr"] = name_process.strip() or pd.NA
    record["six_field_parse_ok"] = bool(
        pd.notna(record["facility_id_ocr"])
        and pd.notna(record["province_ocr"])
        and pd.notna(record["facility_name_ocr"])
        and pd.notna(record["commission_year_ocr"])
        and pd.notna(record["design_capacity_t_d_ocr"])
        and pd.notna(record["mean_daily_flow_t_d_ocr"])
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    text_dir = RUN / "work" / "mee_ocr" / str(args.year) / "text"
    pages = sorted(text_dir.glob("page-*.txt"))
    if not pages:
        raise RuntimeError(f"No OCR text pages found: {text_dir}")
    records: list[dict[str, object]] = []
    for page in pages:
        page_number = int(re.search(r"(\d+)$", page.stem).group(1))
        text = page.read_text(encoding="utf-8", errors="replace")
        starts = list(ROW_START.finditer(text))
        for row_number, match in enumerate(starts, start=1):
            block = text[match.start():starts[row_number].start() if row_number < len(starts) else len(text)]
            item = parse_block(block)
            item.update({"source_year": args.year, "source_page": page_number, "source_row_number": row_number, "raw_ocr": clean(block)})
            records.append(item)
    out = pd.DataFrame(records).sort_values(["source_page", "source_row_number"])
    output_dir = RUN / "inputs" / "staged" / "mee_ocr_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / f"mee_{args.year}_candidates.csv", index=False, encoding="utf-8-sig")
    ids = pd.to_numeric(out.facility_id_ocr, errors="coerce")
    valid_date = out.commission_year_ocr.between(1900, args.year) & out.commission_month_ocr.between(1, 12)
    valid_flows = out.design_capacity_t_d_ocr.ge(0) & out.mean_daily_flow_t_d_ocr.ge(0)
    qa = {
        "year": args.year,
        "ocr_pages": len(pages),
        "candidate_rows": int(len(out)),
        "expected_official_facilities": EXPECTED[args.year],
        "official_count_closed": bool(len(out) == EXPECTED[args.year]),
        "unique_facility_ids": int(ids.nunique()),
        "ids_are_exact_1_to_expected": bool(set(ids.dropna().astype(int)) == set(range(1, EXPECTED[args.year] + 1))),
        "six_field_parse_ok_rows": int(out.six_field_parse_ok.sum()),
        "valid_commission_year_month_rows": int(valid_date.sum()),
        "nonnegative_both_flow_rows": int(valid_flows.sum()),
        "status": "candidate_only; no row may enter the model until field QA and province continuity checks pass",
    }
    qa_dir = RUN / "inputs" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / f"mee_{args.year}_ocr_candidate_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
