from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = RUN_DIR / "reports"
PILOT_OUTPUT_DIR = RUN_DIR / "outputs_pilot_equal_weight"
CALIBRATION_END_YEAR = 2018


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    resids = pd.read_csv(PILOT_OUTPUT_DIR / "resids.csv", encoding="utf-8-sig")
    resids = resids[pd.to_numeric(resids["year"], errors="coerce") <= CALIBRATION_END_YEAR].copy()
    resids["q_site"] = resids["q_site"].astype(str)
    resids["ln_resid"] = pd.to_numeric(resids["ln_resid"], errors="coerce")

    station = (
        resids.groupby("q_site", as_index=False)
        .agg(
            train_obs=("ln_resid", "size"),
            abs_log_resid_median=("ln_resid", lambda s: float(np.nanmedian(np.abs(s)))),
            abs_log_resid_mean=("ln_resid", lambda s: float(np.nanmean(np.abs(s)))),
            log_resid_std=("ln_resid", lambda s: float(np.nanstd(s, ddof=0))),
        )
    )
    # Precision-like weight from pilot calibration residuals only. Clipping keeps
    # a few exceptionally clean/noisy stations from dominating the strict rerun.
    station["residual_scale"] = (
        station["abs_log_resid_median"].fillna(station["abs_log_resid_mean"]).fillna(1.0) + 0.05
    )
    station["coverage_factor"] = np.sqrt(station["train_obs"].clip(lower=1) / station["train_obs"].median())
    station["raw_weight"] = station["coverage_factor"] / station["residual_scale"]
    median_raw = float(station["raw_weight"].median())
    station["empirical_bayes_weight"] = (station["raw_weight"] / median_raw).clip(lower=0.25, upper=4.0)
    station["weight_reason"] = "strict_calibration_2006_2018_pilot_residual_precision"
    station = station.sort_values("empirical_bayes_weight", ascending=False)

    station.to_csv(REPORT_DIR / "station_empirical_bayes_weights.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            {
                "source": str(PILOT_OUTPUT_DIR / "resids.csv"),
                "calibration_years": "2006-2018",
                "stations": int(station["q_site"].nunique()),
                "rows_used": int(len(resids)),
                "median_weight": float(station["empirical_bayes_weight"].median()),
                "min_weight": float(station["empirical_bayes_weight"].min()),
                "max_weight": float(station["empirical_bayes_weight"].max()),
                "validation_years_excluded_from_weight_build": "2019-2022",
            }
        ]
    )
    summary.to_csv(REPORT_DIR / "station_empirical_bayes_weight_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
