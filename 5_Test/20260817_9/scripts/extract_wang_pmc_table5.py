"""Extract a labelled official Wang table from the open-access PMC JATS XML."""
from __future__ import annotations

import csv
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(r"E:\SPARROW")
SOURCE = ROOT / "0_reach_topology" / "data" / "raw" / "point_sources" / "wastewater" / "wang_wwtp_2006_2019" / "metadata" / "PMC9203788_fulltext.xml"
OUT_DIR = ROOT / "5_Test" / "20260817_9" / "work" / "wang_supplement"


def normalized_text(element: ET.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=int, default=5)
    args = parser.parse_args()
    root = ET.parse(SOURCE).getroot()
    target = None
    for table_wrap in root.findall(".//table-wrap"):
        label = normalized_text(table_wrap.find("label")) if table_wrap.find("label") is not None else ""
        if label == f"Table {args.table}":
            target = table_wrap
            break
    if target is None:
        raise RuntimeError(f"Official Table {args.table} was not found in PMC full-text XML")
    rows = []
    for row in target.findall(".//tr"):
        cells = [normalized_text(cell) for cell in list(row) if cell.tag in {"th", "td"}]
        if cells:
            rows.append(cells)
    out = OUT_DIR / f"official_table_{args.table}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerows(rows)
    for row in rows:
        print(" | ".join(row))
    print(f"rows={len(rows)} output={out}")


if __name__ == "__main__":
    main()
