"""Audit observation availability and score the immutable hydrology product."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_36"
OUT, REPORTS, LOCKS = RUN / "outputs", RUN / "reports", RUN / "locks"
S35 = ROOT / "5_Test" / "20260828_35"
REGISTRY_JSON = ROOT / "1_Inputs" / "DischargeData" / "registry" / "discharge_station_year_registry.json"
STATIONS = ROOT / "5_Test" / "20260828_2" / "outputs" / "station_registry_91.parquet"
OBS_DEV = [
    ROOT / "5_Test" / "20260828_1" / "inputs" / "monthly_observations_2010_2016.parquet",
    ROOT / "5_Test" / "20260828_1" / "inputs" / "monthly_observations_2017_2018.parquet",
]
OBS_TIME = ROOT / "5_Test" / "20260823_21" / "outputs" / "locked_2019_2022_predictions.parquet"
OBS_FOUR = ROOT / "5_Test" / "20260823_34" / "outputs" / "four_station_locked_retrospective_predictions.parquet"
CFS_TO_M3S = 0.028316846592


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kge_parts(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    correlation = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 else np.nan
    variability = float(np.std(pred, ddof=1) / np.std(obs, ddof=1)) if np.std(obs, ddof=1) > 0 else np.nan
    bias = float(np.mean(pred) / np.mean(obs)) if np.mean(obs) != 0 else np.nan
    kge = float(1.0 - np.sqrt((correlation - 1) ** 2 + (variability - 1) ** 2 + (bias - 1) ** 2))
    return kge, correlation, variability, bias


def metric_row(obs: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs >= 0) & (pred >= 0)
    obs, pred = obs[valid], pred[valid]
    if len(obs) < 2:
        return {"n": int(len(obs)), "nse": np.nan, "log_rmse": np.nan, "pbias_pct": np.nan, "kge": np.nan, "correlation": np.nan, "variability_ratio": np.nan, "bias_ratio": np.nan}
    denominator = float(np.sum((obs - obs.mean()) ** 2))
    nse = float(1 - np.sum((pred - obs) ** 2) / denominator) if denominator > 0 else np.nan
    kge, correlation, variability, bias = kge_parts(obs, pred)
    return {
        "n": int(len(obs)), "nse": nse,
        "log_rmse": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2))),
        "pbias_pct": float(100 * (pred.sum() - obs.sum()) / obs.sum()) if obs.sum() > 0 else np.nan,
        "kge": kge, "correlation": correlation,
        "variability_ratio": variability, "bias_ratio": bias,
    }


def score(frame: pd.DataFrame, station: str, stratum: str) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for name, group in frame.groupby(station, sort=True):
        row = {"stratum": stratum, "station": str(name), **metric_row(group.observed_m3_s.to_numpy(float), group.predicted_m3_s.to_numpy(float))}
        rows.append(row)
    station_metrics = pd.DataFrame(rows)
    pooled = metric_row(frame.observed_m3_s.to_numpy(float), frame.predicted_m3_s.to_numpy(float))
    eligible = station_metrics[station_metrics.n >= 12]
    summary = {
        "stratum": stratum, "stations": int(frame[station].nunique()), "station_months": int(len(frame)),
        "pooled": pooled,
        "station_mean_nse": float(eligible.nse.mean()),
        "station_median_nse": float(eligible.nse.median()),
        "station_mean_log_rmse": float(eligible.log_rmse.mean()),
        "station_median_log_rmse": float(eligible.log_rmse.median()),
    }
    return station_metrics, summary


def station_prediction(observations: pd.DataFrame, stations: pd.DataFrame, monthly: pd.DataFrame, identity: str) -> pd.DataFrame:
    if identity == "reach_id":
        mapping = stations[["reach_id", "downstream_fraction_on_reach"]].drop_duplicates("reach_id")
        obs = observations.merge(
            mapping, on="reach_id", how="inner", validate="many_to_one",
        )
        obs = obs.merge(monthly, on=["year", "month", "reach_id"], validate="many_to_one")
        obs["predicted_m3_s"] = (
            obs.routed_total_m3_s
            - (1.0 - obs.downstream_fraction_on_reach)
            * (obs.local_fast_response_m3_s + obs.local_slow_response_m3_s)
        ).clip(lower=0)
        return obs
    station_columns = [identity, "reach_id", "downstream_fraction_on_reach"]
    mapping = stations[station_columns].drop_duplicates(identity).rename(
        columns={"reach_id": "mapped_reach_id"},
    )
    obs = observations.merge(
        mapping,
        on=identity, how="inner", validate="many_to_one",
    )
    if "reach_id" in obs:
        if not np.array_equal(
            obs["reach_id"].to_numpy(int), obs["mapped_reach_id"].to_numpy(int),
        ):
            raise RuntimeError(f"Locked observation and station registry disagree on Reach for {identity}")
        obs = obs.drop(columns="mapped_reach_id")
    else:
        obs = obs.rename(columns={"mapped_reach_id": "reach_id"})
    obs = obs.merge(monthly, on=["year", "month", "reach_id"], validate="many_to_one")
    obs["predicted_m3_s"] = (
        obs.routed_total_m3_s
        - (1.0 - obs.downstream_fraction_on_reach) * (obs.local_fast_response_m3_s + obs.local_slow_response_m3_s)
    ).clip(lower=0)
    return obs


def main() -> None:
    for folder in (OUT, REPORTS, LOCKS):
        folder.mkdir(parents=True, exist_ok=True)
    interface_lock = json.loads((S35 / "locks" / "tn_hydrology_interface_lock.json").read_text(encoding="utf-8"))
    if interface_lock.get("status") != "PASS_TN_READY_HYDROLOGY_INTERFACE":
        raise RuntimeError("TN-ready hydrology must be locked before observations are opened")
    registry = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    years = [int(row["year"]) for row in registry["station_year"] if str(row.get("year", "")).isdigit()]
    availability = {
        "authoritative_registry": str(REGISTRY_JSON),
        "year_min": min(years), "year_max": max(years),
        "pre_2006_records": int(sum(year < 2006 for year in years)),
        "conclusion": "NO_PRE2006_DISCHARGE_OBSERVATIONS_AVAILABLE" if not any(year < 2006 for year in years) else "PRE2006_RECORDS_AVAILABLE",
    }
    monthly = pd.read_parquet(S35 / "outputs" / "tn_hydrology_reach_monthly.parquet")
    monthly["month"] = pd.to_datetime(monthly.month)
    monthly["year"] = monthly.month.dt.year
    monthly["month_number"] = monthly.month.dt.month
    monthly = monthly.rename(columns={"month_number": "month"}).drop(columns=["month"], errors="ignore") if False else monthly
    # Keep the timestamp in month_start and expose integer month for joins.
    monthly = monthly.rename(columns={"month": "month_start"})
    monthly["year"] = monthly.month_start.dt.year
    monthly["month"] = monthly.month_start.dt.month
    monthly_fields = monthly[[
        "year", "month", "reach_id", "routed_total_m3_s",
        "local_fast_response_m3_s", "local_slow_response_m3_s",
    ]]
    stations = pd.read_parquet(STATIONS)

    development_obs = pd.concat([pd.read_parquet(path) for path in OBS_DEV], ignore_index=True)
    development_obs = development_obs.rename(columns={"q_m3s": "observed_m3_s"})
    development = station_prediction(development_obs, stations, monthly_fields, "station_norm")
    development["stratum"] = "2010_2018_in_sample_diagnostic"

    temporal_obs = pd.read_parquet(OBS_TIME)
    temporal_obs["observed_m3_s"] = temporal_obs.Q_obsv_cfs.to_numpy(float) * CFS_TO_M3S
    # The legacy 2019-2022 table contains mojibake station labels, while its
    # locked Reach IDs are intact.  The 91-station registry has one station per
    # Reach, so join on that unique spatial key and retain q_site for scoring.
    temporal = station_prediction(temporal_obs, stations, monthly_fields, "reach_id")
    temporal["stratum"] = "2019_2022_temporal_evaluation"

    four_obs = pd.read_parquet(OBS_FOUR)
    four_obs = four_obs[four_obs.usable.fillna(False)].copy()
    four_obs["observed_m3_s"] = four_obs.q_m3s.to_numpy(float)
    four = four_obs.merge(monthly_fields, on=["year", "month", "reach_id"], validate="many_to_one")
    four["predicted_m3_s"] = four.routed_total_m3_s.clip(lower=0)
    four["stratum"] = "four_station_zero_history_retrospective"

    metric_frames, summaries = [], []
    for frame, identity, label in [
        (development, "station_norm", "2010_2018_in_sample_diagnostic"),
        (temporal, "q_site", "2019_2022_temporal_evaluation"),
        (four, "station_norm", "four_station_zero_history_retrospective"),
    ]:
        station_metrics, summary = score(frame, identity, label)
        metric_frames.append(station_metrics)
        summaries.append(summary)
    predictions = pd.concat([
        development[["stratum", "station_norm", "year", "month", "reach_id", "observed_m3_s", "predicted_m3_s"]].rename(columns={"station_norm": "station"}),
        temporal[["stratum", "q_site", "year", "month", "reach_id", "observed_m3_s", "predicted_m3_s"]].rename(columns={"q_site": "station"}),
        four[["stratum", "station_norm", "year", "month", "reach_id", "observed_m3_s", "predicted_m3_s"]].rename(columns={"station_norm": "station"}),
    ], ignore_index=True)
    station_metrics = pd.concat(metric_frames, ignore_index=True)
    predictions_path = OUT / "post_lock_discharge_predictions.parquet"
    metrics_path = OUT / "post_lock_station_metrics.parquet"
    predictions.to_parquet(predictions_path, index=False, compression="zstd")
    station_metrics.to_parquet(metrics_path, index=False, compression="zstd")
    report = {
        "stage": "20260828_36",
        "status": "COMPLETE_POST_LOCK_DIAGNOSTIC_WITH_PRE2006_EVIDENCE_GAP" if availability["pre_2006_records"] == 0 else "COMPLETE_POST_LOCK_DIAGNOSTIC",
        "availability": availability,
        "summaries": summaries,
        "station_exclusions_selected_after_results": False,
        "parameter_refit": False,
        "decision_role": "diagnostic_only",
        "claim_boundary": "No discharge evidence is available to validate 1961-2005 flow; forcing coverage and numerical conservation do not substitute for observations.",
    }
    report_path = REPORTS / "post_lock_discharge_diagnostic.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = {
        "stage": "20260828_36", "status": report["status"],
        "files": {
            "interface_lock": sha256(S35 / "locks" / "tn_hydrology_interface_lock.json"),
            "registry": sha256(REGISTRY_JSON), "predictions": sha256(predictions_path),
            "station_metrics": sha256(metrics_path), "report": sha256(report_path),
            "runner_code": sha256(Path(__file__)),
        },
    }
    (LOCKS / "post_lock_discharge_diagnostic_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
