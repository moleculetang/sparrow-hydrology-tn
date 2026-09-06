from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN_DIR = ROOT / "5_Test" / "20260603_4"
BOOK = ROOT / "1_Inputs" / "DischargeData" / "monthly_mean_2006_2009" / "DischargeData_2006_2009.xlsx"


def main() -> None:
    out_dir = RUN_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    xl = pd.ExcelFile(BOOK)
    rows = []
    preview_lines = [f"# Workbook audit: {BOOK}", "", f"Sheets: {', '.join(xl.sheet_names)}", ""]
    for sheet in xl.sheet_names:
        df = pd.read_excel(BOOK, sheet_name=sheet, nrows=12)
        rows.append(
            {
                "sheet": sheet,
                "preview_rows": len(df),
                "preview_cols": len(df.columns),
                "columns": " | ".join(str(c) for c in df.columns),
            }
        )
        preview_lines.append(f"## {sheet}")
        preview_lines.append("")
        preview_lines.append("```text")
        preview_lines.append(df.head(8).to_string(index=False))
        preview_lines.append("```")
        preview_lines.append("")
    pd.DataFrame(rows).to_csv(out_dir / "workbook_2006_2009_audit.csv", index=False, encoding="utf-8-sig")
    (out_dir / "workbook_2006_2009_preview.md").write_text("\n".join(preview_lines), encoding="utf-8")
    print(f"sheets={len(xl.sheet_names)}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
