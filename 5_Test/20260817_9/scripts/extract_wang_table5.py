"""Extract Wang et al. official Table 5 from its archived HTML page."""
from __future__ import annotations

import csv
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
SOURCE = ROOT / "0_reach_topology" / "data" / "raw" / "point_sources" / "wastewater" / "wang_wwtp_2006_2019" / "metadata" / "s41597-022-01439-7_table_5.html"
OUT = ROOT / "5_Test" / "20260817_9" / "work" / "wang_supplement" / "official_table_5_discharge_pathway_efs.csv"


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.rows: list[list[str]] = []
        self.row: list[str] = []
        self.cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row, self.row = True, []
        elif self.in_row and tag in {"th", "td"}:
            self.in_cell, self.cell = True, []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.in_cell:
            self.row.append(" ".join("".join(self.cell).split()))
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.row:
                self.rows.append(self.row)
            self.in_row = False
        elif tag == "table":
            self.in_table = False


def main() -> None:
    parser = TableParser()
    parser.feed(SOURCE.read_text(encoding="utf-8", errors="replace"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerows(parser.rows)
    for row in parser.rows:
        print(" | ".join(row))
    print(f"rows={len(parser.rows)} output={OUT}")


if __name__ == "__main__":
    main()
