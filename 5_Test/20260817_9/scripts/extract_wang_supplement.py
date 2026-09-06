"""Extract readable text and tables from Wang et al. supplementary DOCX.

Uses only Python's standard library so it remains runnable inside the project
``sparrow`` environment; it writes no model values and preserves the source.
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(r"E:\SPARROW")
SOURCE = (
    ROOT / "0_reach_topology" / "data" / "raw" / "point_sources" / "wastewater"
    / "wang_wwtp_2006_2019" / "metadata" / "41597_2022_1439_MOESM1_ESM.docx"
)
OUT = ROOT / "5_Test" / "20260817_9" / "work" / "wang_supplement"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.findall(".//w:t", NS)).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs = [text(p) for p in root.findall(".//w:p", NS)]
    paragraphs = [p for p in paragraphs if p]
    (OUT / "paragraphs.txt").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    tables = root.findall(".//w:tbl", NS)
    for index, table in enumerate(tables, start=1):
        rows = []
        for row in table.findall("./w:tr", NS):
            rows.append([text(cell) for cell in row.findall("./w:tc", NS)])
        with (OUT / f"table_{index:02d}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerows(rows)
    print(f"paragraphs={len(paragraphs)} tables={len(tables)} output={OUT}")


if __name__ == "__main__":
    main()
