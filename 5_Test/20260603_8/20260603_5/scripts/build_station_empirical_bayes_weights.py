from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
PILOT_DIR = ROOT / "5_Test" / "20260603_4"
RUN_DIR = ROOT / "5_Test" / "20260603_5"
VALIDATION_START_PERIOD = 64


def main() -> int:
    reports = RUN_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    resids = pd.read_csv(PILOT_DIR / "outputs" / "resids.csv", encoding="utf-8-sig")
    reliability = pd.read_csv(PILOT_DIR / "reports" / "same_reach_station_reliability.csv", encoding="utf-8-sig")
    selected = reliability[reliability["selected_for_reach"].astype(str).str.lower().isin(["true", "1"])].copy()

    train = resids[pd.to_numeric(resids["period"], errors="coerce") < VALIDATION_START_PERIOD].copy()
    train["abs_log_resid"] = pd.to_numeric(train["ln_resid"], errors="coerce").abs()
    train["q_site"] = train["q_site"].astype(str)

    station = (
        train.groupby("q_site", as_index=False)
        .agg(
            train_obs=("ln_resid", "size"),
            median_abs_log_resid=("abs_log_resid", "median"),
            p90_abs_log_resid=("abs_log_resid", lambda s: float(np.nanpercentile(s, 90))),
        )
    )
    station = station.merge(
        selected[["station_name", "usable_quarters", "usable_years", "snap_distance_m"]],
        left_on="q_site",
        right_on="station_name",
        how="left",
    )
    coverage = pd.to_numeric(station["usable_quarters"], errors="coerce").fillna(station["train_obs"])
    coverage_factor = np.clip(coverage / 64.0, 0.35, 1.0)
    residual_scale = pd.to_numeric(station["median_abs_log_resid"], errors="coerce").fillna(0.75)
    robust_precision = 1.0 / (1.0 + (residual_scale / 0.60) ** 2)
    station["empirical_bayes_weight"] = np.clip(coverage_factor * robust_precision, 0.20, 1.0)
    station["weight_reason"] = "pilot_train_log_residual_precision_x_station_coverage"
    station = station.sort_values(["empirical_bayes_weight", "q_site"])
    station.to_csv(reports / "station_empirical_bayes_weights.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "pilot_run": "20260603_4",
                "target_run": "20260603_5",
                "validation_periods_excluded_from_weight_build": f"{VALIDATION_START_PERIOD}-68",
                "stations_weighted": int(station["q_site"].nunique()),
                "median_weight": float(station["empirical_bayes_weight"].median()),
                "min_weight": float(station["empirical_bayes_weight"].min()),
                "max_weight": float(station["empirical_bayes_weight"].max()),
            }
        ]
    )
    summary.to_csv(reports / "station_empirical_bayes_weight_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
