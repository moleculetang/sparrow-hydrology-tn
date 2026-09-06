from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
MAIN = RUN / "reports" / "main_model"
OUT = MAIN / "bad_station_numeric_diagnosis"


ISSUES = {
    "shape_or_timing": lambda d: d["main_NSElog"] < 0.65,
    "systematic_volume_bias": lambda d: d["main_PBIAS_pct"].abs() > 25,
    "low_correlation": lambda d: d["main_KGE_r"] < 0.60,
    "volume_ratio": lambda d: (d["main_KGE_beta_volume_ratio"] - 1.0).abs() > 0.25,
    "variability_amplitude": lambda d: (d["main_KGE_gamma_variability_ratio"] - 1.0).abs() > 0.35,
    "peak_timing": lambda d: d["main_peak_lag_months_pred_minus_obs"].abs() >= 2,
    "peak_magnitude": lambda d: d["main_peak_magnitude_bias_pct"].abs() > 35,
    "high_flow_bias": lambda d: d["main_high_flow_volume_bias_pct_obs_ge_q75"].abs() > 30,
    "low_flow_bias": lambda d: d["main_low_flow_volume_bias_pct_obs_le_q25"].abs() > 30,
    "seasonal_bias": lambda d: (d["main_wet_month_volume_bias_pct_Apr_Sep"] - d["main_dry_month_volume_bias_pct_Oct_Mar"]).abs() > 30,
}


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(MAIN / "station_performance_diagnostics_extended.csv", encoding="utf-8-sig")
    data["main_good"] = data["main_good"].astype(str).str.lower().isin({"true", "1", "yes"})
    bad = data[~data["main_good"]].copy()
    for name, rule in ISSUES.items():
        bad[name] = rule(bad).fillna(False)
    issue_summary = pd.DataFrame(
        [
            {"numeric_problem": name, "stations": int(flag.sum()), "share_of_bad_stations": float(flag.mean())}
            for name, flag in bad[list(ISSUES)].items()
        ]
    ).sort_values(["stations", "numeric_problem"], ascending=[False, True])
    issue_summary.to_csv(OUT / "issue_counts.csv", index=False, encoding="utf-8-sig")

    by_mode = (
        bad.groupby("failure_mode", as_index=False)
        .agg(
            stations=("q_site", "size"),
            median_NSElog=("main_NSElog", "median"),
            median_KGE=("main_KGE_2012", "median"),
            median_absPBIAS=("main_abs_PBIAS_pct", "median"),
            median_r=("main_KGE_r", "median"),
            median_volume_ratio=("main_KGE_beta_volume_ratio", "median"),
            median_variability_ratio=("main_KGE_gamma_variability_ratio", "median"),
            median_high_flow_bias=("main_high_flow_volume_bias_pct_obs_ge_q75", "median"),
            median_low_flow_bias=("main_low_flow_volume_bias_pct_obs_le_q25", "median"),
            reservoir_related=("reservoir_relation", lambda s: int(s.astype(str).ne("not_reservoir_related").sum())),
        )
        .sort_values("stations", ascending=False)
    )
    by_mode.to_csv(OUT / "failure_mode_numeric_summary.csv", index=False, encoding="utf-8-sig")

    front = [
        "q_site", "reach_id", "reach_class", "reservoir_relation", "failure_mode", "diagnostic_problem_signature",
        "main_NSElog", "main_KGE_2012", "main_PBIAS_pct", "main_abs_PBIAS_pct", "main_KGE_r", "main_KGE_beta_volume_ratio",
        "main_KGE_gamma_variability_ratio", "main_high_flow_volume_bias_pct_obs_ge_q75",
        "main_low_flow_volume_bias_pct_obs_le_q25", "main_peak_lag_months_pred_minus_obs", "main_peak_magnitude_bias_pct",
    ]
    profile = bad[front + list(ISSUES)].sort_values(["main_NSElog", "main_KGE_2012", "main_abs_PBIAS_pct"], ascending=[True, True, False])
    profile.to_csv(OUT / "bad_station_numeric_profile.csv", index=False, encoding="utf-8-sig")

    bad_summary = {
        "bad_station_count": int(len(bad)),
        "total_validation_station_count": int(len(data)),
        "bad_station_share": float(len(bad) / len(data)),
        "median_NSElog": float(bad["main_NSElog"].median()),
        "median_KGE": float(bad["main_KGE_2012"].median()),
        "median_absPBIAS": float(bad["main_abs_PBIAS_pct"].median()),
        "overprediction_bias_over_25pct": int((bad["main_PBIAS_pct"] > 25).sum()),
        "underprediction_bias_below_minus25pct": int((bad["main_PBIAS_pct"] < -25).sum()),
    }
    overview = pd.DataFrame([bad_summary])
    overview.to_csv(OUT / "bad_station_overview.csv", index=False, encoding="utf-8-sig")
    top = profile.head(15).copy()
    report = [
        f"# {RUN.name} Bad-Station Numeric Diagnosis",
        "",
        "## Scope and definition",
        "",
        "- Window: strict validation, 2019-2022; 48 monthly values per station.",
        "- A station is `good` only if NSElog >= 0.65, KGE >= 0.50, and |PBIAS| <= 25%.",
        f"- Bad stations: {bad_summary['bad_station_count']}/{bad_summary['total_validation_station_count']} ({bad_summary['bad_station_share']:.1%}).",
        f"- Bad-station medians: NSElog {bad_summary['median_NSElog']:.3f}, KGE {bad_summary['median_KGE']:.3f}, |PBIAS| {bad_summary['median_absPBIAS']:.1f}%.",
        f"- Volume-bias direction: {bad_summary['overprediction_bias_over_25pct']} overpredict by >25%; {bad_summary['underprediction_bias_below_minus25pct']} underpredict by >25%.",
        "",
        "## Numeric problems across bad stations",
        "",
        markdown_table(issue_summary),
        "",
        "## Failure-mode numeric summary",
        "",
        markdown_table(by_mode),
        "",
        "## Fifteen lowest-NSElog stations",
        "",
        markdown_table(top),
        "",
        "Negative PBIAS/high- or low-flow bias means underprediction; positive means overprediction.",
    ]
    (OUT / "bad_station_numeric_diagnosis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
