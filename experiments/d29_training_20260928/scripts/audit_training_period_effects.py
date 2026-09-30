"""Recompute 2016-2022 training-support TN metrics from frozen predictions.

F23 is the clean fold: T1/T2 train through 2022, while T0 uses only
2021-2022. The 2016-2020 T0 backcast is deliberately excluded here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from d29_training.metrics import station_table


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def one_job(job_id: str):
    training = ROOT / "data/training_contracts" / job_id
    output = ROOT / "outputs/jobs" / job_id
    rows = pd.read_parquet(training / "rows.parquet")
    audit = json.loads((output / "independent_audit.json").read_text(encoding="utf-8"))
    frozen = json.loads((output / "prediction_freeze.json").read_text(encoding="utf-8"))
    if not audit["objective_passed"] or audit["checkpoint_sha256"] != frozen["parameter_sha256"]:
        raise RuntimeError("UNACCEPTED_OR_UNFROZEN_PATH " + job_id)
    monthly_obs = rows.loc[rows.kind.eq("monthly") & rows.eligible, ["station_key", "year", "month", "tn_mg_l", "date"]]
    daily_obs = rows.loc[rows.kind.eq("HF") & rows.eligible, ["station_key", "date", "tn_mg_l", "read_count"]]
    if monthly_obs.duplicated(["station_key", "year", "month"]).any() or daily_obs.duplicated(["station_key", "date"]).any():
        raise RuntimeError("DUPLICATE_TRAINING_SUPPORT " + job_id)
    mp = pd.read_parquet(output / "frozen_station_months.parquet")
    dp = pd.read_parquet(output / "frozen_station_days.parquet")
    monthly = monthly_obs.merge(mp[["station_key", "year", "month", "prediction_mg_l"]],
                                on=["station_key", "year", "month"], how="left", validate="one_to_one")
    daily = daily_obs.merge(dp[["station_key", "date", "prediction_mg_l"]],
                            on=["station_key", "date"], how="left", validate="one_to_one")
    for frame in (monthly, daily):
        if frame.prediction_mg_l.isna().any() or not np.isfinite(frame[["tn_mg_l", "prediction_mg_l"]].to_numpy(float)).all():
            raise RuntimeError("MISSING_OR_INVALID_TRAINING_PREDICTION " + job_id)
        frame.rename(columns={"tn_mg_l": "observed", "prediction_mg_l": "simulated"}, inplace=True)
    return monthly, daily, audit


def summarize(frame: pd.DataFrame, daily: bool, job_id: str, period: str, out: Path):
    if frame.empty:
        raise RuntimeError("EMPTY_REGISTERED_TRAINING_PERIOD " + job_id + "/" + period)
    scores = station_table(frame, predictions=("simulated",), daily=daily, minimum_coverage=True)
    scores.insert(0, "job", job_id)
    scores.insert(1, "period", period)
    scores.to_csv(out / f"{job_id}_{period}_{'daily' if daily else 'monthly'}_stations.csv", index=False)
    valid = scores.loc[scores.nse_eligible]
    result = dict(job=job_id, period=period, scale="HF_daily" if daily else "monthly_report",
                  actual_training_rows=len(frame), observed_stations=frame.station_key.nunique(),
                  nse_defined_stations=len(valid), median_station_NSE=float(valid.NSE.median()),
                  positive_NSE_stations=int(valid.NSE.gt(0).sum()),
                  median_station_RMSE=float(valid.RMSE.median()),
                  median_station_bias=float(valid.bias.median()),
                  median_station_correlation=float(valid.correlation.median()),
                  median_station_amplitude_ratio=float(valid.amplitude_ratio.median()))
    if daily:
        centered = valid.month_centered_NSE.replace([np.inf, -np.inf], np.nan).dropna()
        result.update(month_centered_nse_defined_stations=len(centered),
                      median_station_month_centered_NSE=float(centered.median()))
    return result


def main():
    manifest = json.loads((ROOT / "outputs/campaign_manifest.json").read_text(encoding="utf-8"))
    selected = [j for j in manifest["selected_jobs"] if j.startswith(("U_F23_", "LAND1_F23_"))]
    if len(selected) != 6:
        raise RuntimeError("EXPECTED_SIX_SELECTED_F23_PATHS")
    out = ROOT / "outputs/training_period_effects"
    out.mkdir(exist_ok=True)
    summaries = []
    for job_id in selected:
        monthly, daily, audit = one_job(job_id)
        strategy = job_id.split("_")[2]
        periods = [("2021_2022", monthly[monthly.year.between(2021, 2022)])]
        if strategy in ("T1", "T2"):
            identity = json.loads((ROOT / "data/training_contracts" / job_id / "identity.json")
                                  .read_text(encoding="utf-8"))
            long_sites = set(identity["long_stations"])
            if len(long_sites) != 54:
                raise RuntimeError("LONG_STATION_GROUP_CHANGED " + job_id)
            periods.extend((("2016_2020", monthly[monthly.year.between(2016, 2020)]),
                            ("2016_2022", monthly[monthly.year.between(2016, 2022)]),
                            ("2016_2020_long54", monthly[monthly.year.between(2016, 2020)
                                                         & monthly.station_key.isin(long_sites)]),
                            ("2016_2022_long54", monthly[monthly.year.between(2016, 2022)
                                                         & monthly.station_key.isin(long_sites)])))
        for label, frame in periods:
            summaries.append(summarize(frame, False, job_id, label, out))
        summaries.append(summarize(daily[daily.date.dt.year.between(2021, 2022)], True,
                                   job_id, "2021_2022", out))
    table = pd.DataFrame(summaries)
    table.to_csv(out / "training_period_summary.csv", index=False)
    receipt = dict(fold="F23", training_end_year=2022,
                   source="registered training rows and accepted frozen full-history predictions",
                   monthly_support="2016-2020 and 2021-2022 separately; T0 has no 2016-2020 training rows",
                   HF_support="2021-2022 only; 4-hour readings collapsed to eligible daily means",
                   station_NSE="median over stations with >=8 monthly readings or >=30 HF days across >=3 months",
                   month_centered_NSE="per-station monthly centered metric on registered HF daily support",
                   training_objectives_not_comparable_across_strategies=True,
                   physical_and_optimization_NSE="not applicable", rows=json_clean(summaries))
    (out / "training_period_summary.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(table[["job", "period", "scale", "actual_training_rows", "nse_defined_stations",
                 "median_station_NSE", "median_station_month_centered_NSE"]
                if "median_station_month_centered_NSE" in table else
                ["job", "period", "scale", "actual_training_rows", "nse_defined_stations", "median_station_NSE"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
