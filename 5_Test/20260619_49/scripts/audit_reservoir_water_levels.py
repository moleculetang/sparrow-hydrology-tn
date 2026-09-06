from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260619_49"
RAW = ROOT / "0_hydro_sediment_data" / "reservoir" / "珠江水情水库.xlsx"
REPORTS = RUN / "reports"
PROCESSED = RUN / "inputs" / "processed"
LOG = ROOT / "5_Test" / "20260619.log"


NAME_MAP = {
    "天生桥(一级)": "天生桥一级",
    "天生桥一级": "天生桥一级",
    "天生桥（一级）": "天生桥一级",
    "飞来峡": "飞来峡",
    "飞来峡（坝上）": "飞来峡",
}


# These are conservative plausibility envelopes, not exact operation rules.
# They are used only to decide whether the Excel "water level" is a real
# elevation-like water level and to remove physically impossible spikes.
REFERENCE = [
    {
        "reservoir_std": "天生桥一级",
        "reference_level_m": 780.0,
        "expected_min_m": 730.0,
        "expected_max_m": 781.0,
        "reference_basis": "public Hongshui River cascade table gives normal pool level 780 m",
        "source_url": "https://zh.wikipedia.org/wiki/%E7%BA%A2%E6%B0%B4%E6%B2%B3",
        "confidence": "high",
    },
    {
        "reservoir_std": "龙滩",
        "reference_level_m": 400.0,
        "expected_min_m": 330.0,
        "expected_max_m": 401.0,
        "reference_basis": "public Hongshui River cascade table gives normal pool level 400 m; observed period is below that level",
        "source_url": "https://zh.wikipedia.org/wiki/%E7%BA%A2%E6%B0%B4%E6%B2%B3",
        "confidence": "high",
    },
    {
        "reservoir_std": "岩滩",
        "reference_level_m": 223.0,
        "expected_min_m": 218.0,
        "expected_max_m": 224.0,
        "reference_basis": "public Hongshui River cascade table gives normal pool level 223 m",
        "source_url": "https://zh.wikipedia.org/wiki/%E7%BA%A2%E6%B0%B4%E6%B2%B3",
        "confidence": "high",
    },
    {
        "reservoir_std": "百色水库",
        "reference_level_m": 228.0,
        "expected_min_m": 203.0,
        "expected_max_m": 229.0,
        "reference_basis": "public normal pool level around 228 m",
        "source_url": "https://baike.baidu.com/item/%E7%99%BE%E8%89%B2%E6%B0%B4%E5%BA%93",
        "confidence": "medium",
    },
    {
        "reservoir_std": "大藤峡水利枢纽",
        "reference_level_m": 61.0,
        "expected_min_m": 44.0,
        "expected_max_m": 62.0,
        "reference_basis": "public Hongshui River cascade table gives normal pool level 61 m",
        "source_url": "https://zh.wikipedia.org/wiki/%E7%BA%A2%E6%B0%B4%E6%B2%B3",
        "confidence": "high",
    },
    {
        "reservoir_std": "飞来峡",
        "reference_level_m": 24.0,
        "expected_min_m": 17.0,
        "expected_max_m": 27.0,
        "reference_basis": "public/engineering descriptions put operation level near 24 m",
        "source_url": "",
        "confidence": "medium",
    },
    {
        "reservoir_std": "西津",
        "reference_level_m": 61.5,
        "expected_min_m": 56.0,
        "expected_max_m": 62.5,
        "reference_basis": "engineering descriptions and observed range near 61 m",
        "source_url": "",
        "confidence": "medium",
    },
    {
        "reservoir_std": "长洲枢纽",
        "reference_level_m": 20.6,
        "expected_min_m": 16.0,
        "expected_max_m": 26.0,
        "reference_basis": "low-head navigation/hydropower project; observed operational range mostly 18-25 m",
        "source_url": "",
        "confidence": "medium",
    },
    {
        "reservoir_std": "棉花滩水库",
        "reference_level_m": 173.0,
        "expected_min_m": 145.0,
        "expected_max_m": 174.0,
        "reference_basis": "public normal pool level around 173 m; outside Pearl River model domain unless explicitly mapped",
        "source_url": "https://baike.baidu.com/item/%E6%A3%89%E8%8A%B1%E6%BB%A9%E6%B0%B4%E7%94%B5%E7%AB%99",
        "confidence": "medium",
    },
]


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def q05(x: pd.Series) -> float:
    return float(x.quantile(0.05))


def q95(x: pd.Series) -> float:
    return float(x.quantile(0.95))


def status_from_row(row: pd.Series) -> str:
    hmin = row["h_min"]
    hmax = row["h_max"]
    emin = row["expected_min_m"]
    emax = row["expected_max_m"]
    if pd.isna(emin) or pd.isna(emax):
        return "needs_reference"
    if hmax < emin or hmin > emax:
        return "not_matching_reference"
    if row["outlier_count"] > 0:
        return "true_level_with_outliers"
    return "true_elevation_level"


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(frame.columns)

    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    rows = [[fmt(v) for v in row] for row in frame.to_numpy()]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(RAW)
    raw = raw[raw["站名"].notna()].copy()
    raw["date"] = pd.to_datetime(raw["时间"], errors="coerce")
    raw["reservoir_raw"] = raw["站名"].astype(str).str.strip()
    raw["reservoir_std"] = raw["reservoir_raw"].map(NAME_MAP).fillna(raw["reservoir_raw"])
    raw["water_level_m"] = pd.to_numeric(raw["水位-m"], errors="coerce")
    raw["qout_m3s"] = as_float(raw["出库-m3/s"])
    raw["qin_m3s"] = as_float(raw["入库-m3/s"])

    ref = pd.DataFrame(REFERENCE)
    df = raw.merge(ref, on="reservoir_std", how="left")
    df["water_level_outlier"] = (
        df["water_level_m"].notna()
        & (
            (df["water_level_m"] < df["expected_min_m"])
            | (df["water_level_m"] > df["expected_max_m"])
        )
    )
    df["water_level_adjusted_m"] = df["water_level_m"].mask(df["water_level_outlier"])
    df["water_level_adjustment_note"] = df["water_level_outlier"].map(
        {True: "set NA: outside reference/plausibility envelope", False: ""}
    )

    # Relative storage proxy: use observed min/max after outlier removal. This is
    # deliberately not a volume conversion; without an H-S curve, H can only
    # support a dimensionless storage-state proxy.
    grouped = df.groupby("reservoir_std")["water_level_adjusted_m"]
    h_clean_min = grouped.transform("min")
    h_clean_max = grouped.transform("max")
    denom = h_clean_max - h_clean_min
    df["storage_state_proxy_0_1"] = (df["water_level_adjusted_m"] - h_clean_min) / denom
    df.loc[denom.abs() < 1e-9, "storage_state_proxy_0_1"] = pd.NA

    stats = (
        df.groupby("reservoir_std", dropna=False)
        .agg(
            raw_names=("reservoir_raw", lambda x: "；".join(sorted(set(map(str, x))))),
            n_records=("water_level_m", "size"),
            n_days=("date", "nunique"),
            date_start=("date", "min"),
            date_end=("date", "max"),
            h_min=("water_level_m", "min"),
            h_p05=("water_level_m", q05),
            h_mean=("water_level_m", "mean"),
            h_p95=("water_level_m", q95),
            h_max=("water_level_m", "max"),
            h_sd=("water_level_m", "std"),
            clean_h_min=("water_level_adjusted_m", "min"),
            clean_h_max=("water_level_adjusted_m", "max"),
            outlier_count=("water_level_outlier", "sum"),
            qout_missing=("qout_m3s", lambda x: int(x.isna().sum())),
            qin_missing=("qin_m3s", lambda x: int(x.isna().sum())),
        )
        .reset_index()
        .merge(ref, on="reservoir_std", how="left")
    )
    stats["water_level_status"] = stats.apply(status_from_row, axis=1)
    stats["reference_match_note"] = stats.apply(
        lambda r: (
            "Excel water level has absolute-elevation magnitude; use as observed H, but convert to storage proxy unless H-S curve exists."
            if r["water_level_status"] in {"true_elevation_level", "true_level_with_outliers"}
            else "Do not use as physical H until independently verified."
        ),
        axis=1,
    )

    outliers = df[df["water_level_outlier"]].copy()
    cols_clean = [
        "date",
        "reservoir_std",
        "reservoir_raw",
        "water_level_m",
        "water_level_adjusted_m",
        "storage_state_proxy_0_1",
        "qout_m3s",
        "qin_m3s",
        "water_level_adjustment_note",
    ]
    df[cols_clean].to_csv(PROCESSED / "reservoir_daily_observed_clean.csv", index=False, encoding="utf-8-sig")

    monthly = (
        df.set_index("date")
        .groupby("reservoir_std")
        .resample("MS")
        .agg(
            water_level_mean_m=("water_level_adjusted_m", "mean"),
            water_level_start_m=("water_level_adjusted_m", "first"),
            water_level_end_m=("water_level_adjusted_m", "last"),
            storage_state_proxy_mean=("storage_state_proxy_0_1", "mean"),
            qout_mean_m3s=("qout_m3s", "mean"),
            qin_mean_m3s=("qin_m3s", "mean"),
            n_days=("reservoir_raw", "size"),
        )
        .reset_index()
    )
    monthly.to_csv(PROCESSED / "reservoir_monthly_observed_clean.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(REPORTS / "reservoir_water_level_authenticity_summary.csv", index=False, encoding="utf-8-sig")
    outliers[
        [
            "date",
            "reservoir_std",
            "reservoir_raw",
            "water_level_m",
            "expected_min_m",
            "expected_max_m",
            "qout_m3s",
            "qin_m3s",
            "water_level_adjustment_note",
        ]
    ].to_csv(REPORTS / "reservoir_water_level_outliers.csv", index=False, encoding="utf-8-sig")
    ref.to_csv(REPORTS / "reservoir_reference_levels.csv", index=False, encoding="utf-8-sig")

    table = stats[
        [
            "reservoir_std",
            "raw_names",
            "n_records",
            "date_start",
            "date_end",
            "h_min",
            "h_p05",
            "h_mean",
            "h_p95",
            "h_max",
            "reference_level_m",
            "expected_min_m",
            "expected_max_m",
            "outlier_count",
            "water_level_status",
            "confidence",
        ]
    ].copy()
    for c in ["date_start", "date_end"]:
        table[c] = pd.to_datetime(table[c]).dt.strftime("%Y-%m-%d")
    md_table = markdown_table(table, floatfmt=".3f")

    outlier_text = "None"
    if not outliers.empty:
        outlier_frame = outliers[
            ["date", "reservoir_std", "reservoir_raw", "water_level_m", "expected_min_m", "expected_max_m"]
        ].copy()
        outlier_frame["date"] = pd.to_datetime(outlier_frame["date"]).dt.strftime("%Y-%m-%d")
        outlier_text = markdown_table(outlier_frame, floatfmt=".3f")

    md = f"""# Reservoir Water-Level Authenticity Audit

Run folder: `20260619_49`

Raw workbook: `{RAW}`

## Main Conclusion

The `水位-m` field is **mostly a real absolute water-level elevation** rather than a normalized fluctuation index.

Evidence:

1. Different reservoirs have physically meaningful, reservoir-specific elevation ranges, for example `天生桥一级` near 742-780 m, `龙滩` near 337-375 m, `岩滩` near 219-223 m, `百色水库` near 206-228 m, `西津` near 58-61 m, and `飞来峡` near 18-26 m.
2. These ranges match public design/normal-pool magnitudes where references are available.
3. A relative fluctuation index would not preserve these reservoir-specific datum elevations.

However, the field is **not clean enough to use directly as storage volume**:

1. There is at least one physically implausible water-level spike.
2. Some reservoir names are aliases and were standardized.
3. Without reservoir elevation-storage curves, `H` should be used as an observed water-level state or normalized storage-state proxy, not as volume `S`.

## Reservoir-Level Summary

{md_table}

## Flagged Water-Level Outliers

{outlier_text}

## Adjustment Made In This Folder

This folder does not rewrite the raw workbook. It creates adjusted downstream inputs:

- `inputs/processed/reservoir_daily_observed_clean.csv`
- `inputs/processed/reservoir_monthly_observed_clean.csv`
- `reports/reservoir_water_level_authenticity_summary.csv`
- `reports/reservoir_water_level_outliers.csv`
- `reports/reservoir_reference_levels.csv`

Adjustment rules:

```text
reservoir aliases -> standardized reservoir names
water_level_adjusted_m = NA if outside conservative reservoir-specific plausibility envelope
storage_state_proxy_0_1 = (H_adjusted - min(H_adjusted)) / (max(H_adjusted) - min(H_adjusted))
```

This means the next reservoir operator should use:

```text
H_obs_m                    = real water level when available and QC-passed
storage_state_proxy_0_1    = dimensionless storage-state proxy when no H-S curve exists
```

It should not use:

```text
S = H_obs_m
```

because water level is not storage volume.

## Model Implication

The earlier storage-state proxy experiment should not be interpreted as a test of true reservoir storage unless an elevation-storage relation is added. The safe interpretation is:

```text
observed/reconstructed water-level state -> bounded release-state proxy
```

not:

```text
observed water level -> physical reservoir volume
```

Therefore the next reservoir experiment should separate:

1. QC-passed observed water level `H`.
2. Relative storage state derived from `H`.
3. True storage `S`, only if a water-level-storage curve is available.
4. Release rule parameters, calibrated without station-specific special tuning.
"""
    (REPORTS / "reservoir_water_level_authenticity_audit.md").write_text(md, encoding="utf-8-sig")

    readme = """# 20260619_49 Reservoir Water-Level Authenticity Audit

Purpose: verify whether `珠江水情水库.xlsx` water levels are real reservoir water-level elevations or relative fluctuations.

Result: mostly real elevation-like water levels, but with aliases and at least one outlier. Clean daily/monthly inputs are generated under `inputs/processed`.
"""
    (RUN / "README.md").write_text(readme, encoding="utf-8-sig")

    run_log = """# Run Log

- Read raw workbook `珠江水情水库.xlsx`.
- Standardized reservoir aliases.
- Compared water-level ranges against public reservoir level magnitudes.
- Flagged water-level outliers and generated clean daily/monthly reservoir observation inputs.
- Conclusion: `水位-m` is mostly real water-level elevation, not a relative fluctuation index; use it as H/state proxy, not directly as storage volume S.
"""
    (RUN / "logs" / "run_log.md").write_text(run_log, encoding="utf-8-sig")

    with LOG.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n## 20260619_49 reservoir water-level authenticity audit\n"
            "- Checked `珠江水情水库.xlsx` water-level magnitudes against public reservoir level ranges.\n"
            "- Conclusion: `水位-m` is mostly true absolute reservoir water-level elevation, not a normalized relative fluctuation.\n"
            "- Standardized aliases for 天生桥一级 and 飞来峡; generated QC daily/monthly reservoir observation inputs.\n"
            "- Flagged physically implausible water-level records, especially 长洲枢纽 2024-03-12 H=75.40 m.\n"
            "- Important modeling rule: use QC-passed H as water-level state or normalized storage proxy; do not treat H directly as storage volume S without H-S curve.\n"
        )

    print("Wrote", REPORTS / "reservoir_water_level_authenticity_audit.md")
    print("Outliers:", len(outliers))
    print(stats[["reservoir_std", "h_min", "h_max", "reference_level_m", "outlier_count", "water_level_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
