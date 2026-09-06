"""Evaluate locked R0/R1/R2 reservoir structures on 2019-2022 without refitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test" / "20260828_23"
S20 = ROOT / "5_Test" / "20260828_20"
S21 = ROOT / "5_Test" / "20260828_21"
S22 = ROOT / "5_Test" / "20260828_22"
S14 = ROOT / "5_Test" / "20260828_14"
S15 = ROOT / "5_Test" / "20260828_15"
OBSERVATIONS = (
    ROOT
    / "5_Test"
    / "20260823_14"
    / "outputs"
    / "final_model_station_month_observations.parquet"
)

for folder in [S20 / "scripts", S15 / "scripts", S14 / "scripts"]:
    sys.path.insert(0, str(folder))

from reservoir_development_core import (  # noqa: E402
    load_static_context,
    metric_summary,
    prepare_reservoir_metadata,
    simulate_candidate,
    station_monthly_prediction,
)


SPATIAL_STATIONS = {"珠坑", "昭平", "瓦村（二）", "盘江桥（三）"}
LIMITS = {
    "pooled_log_RMSE": 0.01,
    "pooled_NSE": 0.02,
    "station_median_NSE": 0.03,
    "station_median_absolute_PBIAS_pct": 2.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked_observations(context: dict[str, object]) -> pd.DataFrame:
    source = pd.read_parquet(OBSERVATIONS)
    if "station_norm" not in source:
        raise RuntimeError("Retrospective observation source lacks station_norm")
    if "year" not in source or "month" not in source:
        if "date" not in source:
            raise RuntimeError("Retrospective observations lack year/month and date")
        source["date"] = pd.to_datetime(source["date"])
        source["year"] = source["date"].dt.year
        source["month"] = source["date"].dt.month
    q_column = next(
        (name for name in ["q_m3_s", "q_m3s", "observed_m3_s", "Q_obs_m3_s"] if name in source),
        None,
    )
    if q_column is None:
        raise RuntimeError(f"No recognized discharge field in {list(source.columns)}")
    legal = context["stations"]["station_norm"].astype(str).tolist()
    if SPATIAL_STATIONS.intersection(legal):
        raise RuntimeError("A registered spatial-test station leaked into the 89-gauge cohort")
    source = source[
        source["station_norm"].astype(str).isin(legal)
        & source["year"].between(2019, 2022)
    ].copy()
    result = source.pivot_table(
        index=["year", "month"], columns="station_norm", values=q_column, aggfunc="mean"
    ).reindex(columns=legal)
    expected_index = pd.MultiIndex.from_product(
        [range(2019, 2023), range(1, 13)], names=["year", "month"]
    )
    result = result.reindex(expected_index)
    if np.isfinite(result.to_numpy(float)).sum() == 0:
        raise RuntimeError("No valid 2019-2022 observations were opened")
    return result


def station_metrics(obs: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, prediction in predictions.items():
        aligned = prediction.reindex(obs.index)
        for station in obs.columns:
            observed = obs[station].to_numpy(float)
            predicted = aligned[station].to_numpy(float)
            valid = np.isfinite(observed) & np.isfinite(predicted)
            if valid.sum() < 12:
                continue
            observed = observed[valid]
            predicted = predicted[valid]
            denominator = float(np.sum((observed - observed.mean()) ** 2))
            rows.append(
                {
                    "model": model,
                    "station_norm": station,
                    "n_month": int(valid.sum()),
                    "NSE": 1.0 - float(np.sum((predicted - observed) ** 2)) / denominator
                    if denominator > 0
                    else np.nan,
                    "log_RMSE": float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(observed)) ** 2))),
                    "PBIAS_pct": 100.0 * float(predicted.sum() - observed.sum()) / float(observed.sum())
                    if observed.sum() > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    outputs = STAGE / "outputs"
    reports = STAGE / "reports"
    locks = STAGE / "locks"
    for folder in [outputs, reports, locks]:
        folder.mkdir(parents=True, exist_ok=True)

    initial = json.loads(
        (S22 / "reports" / "initial_state_sensitivity.json").read_text(encoding="utf-8")
    )
    if initial["status"] != "PASS_INITIAL_STATE_CONVERGED":
        raise RuntimeError("Stage 22 did not authorize retrospective opening")
    development = json.loads(
        (S21 / "locks" / "boundary_complete_development_lock.json").read_text(encoding="utf-8")
    )
    context = load_static_context("2022-12-31")
    obs = load_locked_observations(context)
    metadata = prepare_reservoir_metadata(context)

    parent_total = context["routed_total_rate"]
    r1 = development["selected_candidates"]["R1"]
    r2 = development["selected_candidates"]["R2"]
    r1_total, _, _ = simulate_candidate(
        context, metadata, "R1", r1["tau_day"], r1["inflow_response"], r1["drawdown_fraction"]
    )
    r2_total, _, _ = simulate_candidate(
        context, metadata, "R2", r2["tau_day"], r2["inflow_response"], r2["drawdown_fraction"]
    )
    predictions = {
        "R0": station_monthly_prediction(context, parent_total),
        "R1": station_monthly_prediction(context, r1_total),
        "R2": station_monthly_prediction(context, r2_total),
    }
    metrics = {model: metric_summary(obs, pred, (2019, 2022)) for model, pred in predictions.items()}
    station_detail = station_metrics(obs, predictions)
    station_detail.to_parquet(outputs / "locked_2019_2022_station_metrics.parquet", index=False)

    long_parts = []
    for model, prediction in predictions.items():
        part = prediction.reindex(obs.index).stack(future_stack=True).rename("predicted_m3_s").reset_index()
        observed = obs.stack(future_stack=True).rename("observed_m3_s").reset_index()
        part = part.merge(observed, on=["year", "month", "station_norm"], how="left", validate="one_to_one")
        part.insert(0, "model", model)
        long_parts.append(part)
    pd.concat(long_parts, ignore_index=True).to_parquet(
        outputs / "locked_2019_2022_predictions.parquet", index=False
    )

    gates: dict[str, dict[str, bool]] = {}
    for model in ["R1", "R2"]:
        gates[model] = {
            "pooled_log_RMSE": metrics[model]["pooled_log_RMSE"] - metrics["R0"]["pooled_log_RMSE"] <= LIMITS["pooled_log_RMSE"],
            "pooled_NSE": metrics["R0"]["pooled_NSE"] - metrics[model]["pooled_NSE"] <= LIMITS["pooled_NSE"],
            "station_median_NSE": metrics["R0"]["station_median_NSE"] - metrics[model]["station_median_NSE"] <= LIMITS["station_median_NSE"],
            "station_median_absolute_PBIAS_pct": metrics[model]["station_median_absolute_PBIAS_pct"] - metrics["R0"]["station_median_absolute_PBIAS_pct"] <= LIMITS["station_median_absolute_PBIAS_pct"],
        }
        gates[model]["all"] = all(gates[model].values())

    status = "R2_RETROSPECTIVE_NONINFERIOR_TN_INTERFACE_AUTHORIZED" if gates["R2"]["all"] else "R2_RETROSPECTIVE_CONFOUNDED_RETAIN_R0"
    report = {
        "stage": "20260828_23",
        "status": status,
        "evaluation_period": "2019-2022",
        "parameters_refit": False,
        "metrics": metrics,
        "noninferiority_against_R0": gates,
        "selected_tn_interface": "R2" if gates["R2"]["all"] else "R0",
        "claim_boundary": "Locked retrospective temporal check; shared reservoir structure only, not observed individual operation.",
        "spatial_four_station_observations_opened": False,
        "tn_observations_opened": False,
        "hashes": {
            "contract": sha256(STAGE / "experiment_contract.json"),
            "stage21_lock": sha256(S21 / "locks" / "boundary_complete_development_lock.json"),
            "stage22_report": sha256(S22 / "reports" / "initial_state_sensitivity.json"),
            "parent": sha256(S20 / "../20260828_9/outputs/canonical_reach_daily_2006_2024.parquet"),
            "observation_source": sha256(OBSERVATIONS),
            "runner_code": sha256(Path(__file__)),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (reports / "locked_retrospective_decision.json").write_text(text, encoding="utf-8")
    (locks / "retrospective_tn_interface_lock.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
