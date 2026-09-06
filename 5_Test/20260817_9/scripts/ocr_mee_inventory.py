"""Render and OCR one MEE annual WWTP inventory to auditable page-level files.

Usage: sparrow/python ocr_mee_inventory.py --year 2007 [--first 1 --last 66]
The output is deliberately *not* parsed directly into model loads.  Each OCR
line retains page provenance for later record and numeric verification.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260817_9"
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "point_sources" / "wastewater" / "mee_wwtp_inventory"
PDFTOPPM = Path(r"C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe")
TESSERACT = Path(r"D:\Program Files\Tesseract-OCR\tesseract.exe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int)
    parser.add_argument("--ocr-first", type=int, help="First already-rendered page to OCR; defaults to --first")
    parser.add_argument("--ocr-last", type=int, help="Last already-rendered page to OCR; defaults to --last")
    parser.add_argument("--skip-render", action="store_true", help="OCR existing PNG pages only")
    args = parser.parse_args()
    pdfs = list((RAW / str(args.year) / "archives").glob("*.pdf"))
    if len(pdfs) != 1:
        raise RuntimeError(f"Expected one PDF for {args.year}, found {len(pdfs)}")
    target = RUN / "work" / "mee_ocr" / str(args.year)
    images, texts = target / "pages", target / "text"
    images.mkdir(parents=True, exist_ok=True); texts.mkdir(parents=True, exist_ok=True)
    if not args.skip_render:
        command = [str(PDFTOPPM), "-r", "300", "-png", "-f", str(args.first)]
        if args.last is not None:
            command += ["-l", str(args.last)]
        command += [str(pdfs[0]), str(images / "page")]
        subprocess.run(command, check=True)
    ocr_first = args.ocr_first if args.ocr_first is not None else args.first
    ocr_last = args.ocr_last if args.ocr_last is not None else args.last
    for image in sorted(images.glob("*.png")):
        match = re.search(r"(\d+)$", image.stem)
        if match is None:
            continue
        page = int(match.group(1))
        if page < ocr_first or (ocr_last is not None and page > ocr_last):
            continue
        output_stem = texts / image.stem
        if output_stem.with_suffix(".txt").exists():
            continue
        subprocess.run([str(TESSERACT), str(image), str(output_stem), "-l", "chi_sim+eng", "--psm", "4"], check=True)
    print(f"OCR staged at {target}; verify names, dates and both flow columns before parsing.")


if __name__ == "__main__":
    main()
