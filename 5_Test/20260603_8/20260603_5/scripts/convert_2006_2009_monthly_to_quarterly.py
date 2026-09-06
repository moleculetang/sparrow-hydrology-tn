from __future__ import annotations

import calendar
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN_DIR = ROOT / "5_Test" / "20260603_5"
BOOK = ROOT / "1_Inputs" / "DischargeData" / "monthly_mean_2006_2009" / "DischargeData_2006_2009.xlsx"
M3S_TO_CFS = 35.3146667
MIN_QUARTER_COVERAGE = 0.75


def norm_name(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace(" ", "").replace("\u3000", "")
    text = text.replace("(", "（").replace(")", "）")
    return text.strip()


def main() -> None:
    out_dir = RUN_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_excel(BOOK, sheet_name="monthly_mean")
    monthly.columns = [str(c).strip() for c in monthly.columns]
    required = {"station", "year", "month", "monthly_mean_m3_s", "n_days_used"}
    missing = required - set(monthly.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    monthly["station_name"] = monthly["station"].astype(str).str.strip()
    monthly["station_norm"] = monthly["station_name"].map(norm_name)
    monthly["year"] = pd.to_numeric(monthly["year"], errors="coerce").astype("Int64")
    monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce").astype("Int64")
    monthly["q_m3s_month"] = pd.to_numeric(monthly["monthly_mean_m3_s"], errors="coerce")
    monthly["n_days_used"] = pd.to_numeric(monthly["n_days_used"], errors="coerce").fillna(0.0)
    monthly = monthly[monthly["year"].between(2006, 2009) & monthly["month"].between(1, 12)].copy()
    monthly["quarter"] = ((monthly["month"].astype(int) - 1) // 3 + 1).astype(int)
    monthly["month_days"] = [
        calendar.monthrange(int(y), int(m))[1] for y, m in zip(monthly["year"], monthly["month"])
    ]
    monthly.loc[monthly["q_m3s_month"] <= 0, "q_m3s_month"] = np.nan
    monthly["weighted_q"] = monthly["q_m3s_month"] * monthly["n_days_used"]
    grouped = (
        monthly.groupby(["station_norm", "station_name", "year", "quarter"], as_index=False)
        .agg(
            q_weighted_sum=("weighted_q", "sum"),
            valid_days=("n_days_used", "sum"),
            total_days=("month_days", "sum"),
            months=("month", "nunique"),
        )
    )
    grouped["q_m3s"] = grouped["q_weighted_sum"] / grouped["valid_days"].replace(0, np.nan)
    grouped["coverage"] = grouped["valid_days"] / grouped["total_days"].replace(0, np.nan)
    grouped["usable"] = (grouped["coverage"] >= MIN_QUARTER_COVERAGE) & grouped["q_m3s"].notna()
    grouped["Q_obsv_cfs"] = np.where(grouped["usable"], grouped["q_m3s"] * M3S_TO_CFS, np.nan)
    grouped["source"] = "monthly_mean_2006_2009"
    grouped.to_csv(out_dir / "monthly_mean_2006_2009_quarterly_all.csv", index=False, encoding="utf-8-sig")
    usable = grouped[grouped["Q_obsv_cfs"].notna()].copy()
    usable.to_csv(out_dir / "monthly_mean_2006_2009_quarterly.csv", index=False, encoding="utf-8-sig")
    print(f"monthly rows={len(monthly)}")
    print(f"quarter rows={len(grouped)} usable={len(usable)} stations={usable['station_norm'].nunique()}")


if __name__ == "__main__":
    main()
