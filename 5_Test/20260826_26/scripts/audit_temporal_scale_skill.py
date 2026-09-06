"""Separate daily skill from the monthly skill relevant to the TN bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
PREDICTIONS = ROOT / "5_Test" / "20260826_25" / "outputs" / "temporal_predictions_2017_2018.parquet"
AUTHORITATIVE = ROOT / "5_Test" / "20260826_21" / "reports" / "final_program_decision.json"
OUT = ROOT / "5_Test" / "20260826_26" / "outputs" / "temporal_scale_skill_audit.parquet"
REPORT = ROOT / "5_Test" / "20260826_26" / "reports" / "temporal_scale_skill_audit.json"


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1.0 - np.sum((predicted - observed) ** 2) / denominator) if denominator > 0 else float("nan")


def main() -> None:
    values = pd.read_parquet(PREDICTIONS)
    values["date"] = pd.to_datetime(values.date)
    values["period"] = values.date.dt.to_period("M")
    monthly = values.groupby(["model_id", "seed", "station_norm", "period"], as_index=False).agg(
        observed_m3_s=("observed_m3_s", "mean"), predicted_m3_s=("predicted_m3_s", "mean")
    )
    rows = []
    for (model_id, seed), frame in monthly.groupby(["model_id", "seed"]):
        station_nse = frame.groupby("station_norm").apply(
            lambda group: nse(group.observed_m3_s.to_numpy(), group.predicted_m3_s.to_numpy()),
            include_groups=False,
        )
        rows.append({
            "model_id": model_id,
            "seed": int(seed),
            "period": "2017-2018",
            "time_scale": "monthly_mean_discharge",
            "pooled_NSE": nse(frame.observed_m3_s.to_numpy(), frame.predicted_m3_s.to_numpy()),
            "station_median_NSE": float(np.nanmedian(station_nse)),
            "station_mean_NSE": float(np.nanmean(station_nse)),
            "PBIAS_pct": float(100.0 * np.nansum(frame.predicted_m3_s - frame.observed_m3_s) / np.nansum(frame.observed_m3_s)),
        })
    result = pd.DataFrame(rows)
    result.to_parquet(OUT, index=False)
    old = json.loads(AUTHORITATIVE.read_text(encoding="utf-8"))["parent_decision_unchanged"]
    dyn = result.loc[result.model_id.eq("DYN_FLUX")]
    audit = {
        "status": "SCALE_CONFUSION_REPAIRED",
        "current_daily_skill_source": "20260826_25 locked 2017-2018 temporal predictions",
        "current_monthly_skill": {
            "DYN_FLUX_pooled_NSE_range": [float(dyn.pooled_NSE.min()), float(dyn.pooled_NSE.max())],
            "DYN_FLUX_station_median_NSE_range": [float(dyn.station_median_NSE.min()), float(dyn.station_median_NSE.max())],
        },
        "old_authoritative_monthly_skill_different_period": {
            "pooled_NSE": old["monthly_pooled_NSE"],
            "station_median_NSE": old["monthly_station_median_NSE"],
            "source_period": "2019-2022 retrospective",
        },
        "comparison_limit": "The values use different locked periods and are descriptive, not a paired promotion test.",
        "TN_relevant_interpretation": "The new candidate is materially stronger at monthly than daily scale, but remains slightly below the old authoritative monthly benchmark.",
        "promotion_rule_added": "Final promotion requires locked monthly performance to be noninferior to the old authoritative TN hydrology, in addition to component and spatial gates.",
    }
    REPORT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
