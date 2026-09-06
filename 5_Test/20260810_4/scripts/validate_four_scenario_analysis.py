from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "scenario_diagnostics"
SCENARIOS = ("S00_NATIVE", "S10", "S01", "S11")
KEYS = ["fold_id", "comid", "year", "month"]
CFS_PER_M3S = 35.31466672148859
EPS = 1.0e-12


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.square(observed - observed.mean()).sum()
    return float(1.0 - np.square(predicted - observed).sum() / denominator)


def load(scenario: str) -> pd.DataFrame:
    path = RUN / "outputs" / "scenarios" / scenario / "q72_three_fold_oof_predictions.parquet"
    return pd.read_parquet(path).sort_values(KEYS).reset_index(drop=True)


def station_log_nse(frame: pd.DataFrame) -> pd.Series:
    values = {}
    for comid, group in frame.groupby("comid", sort=True):
        observed = np.log1p(group["actual"].to_numpy(float) / CFS_PER_M3S)
        predicted = np.log1p(group["predict"].to_numpy(float) / CFS_PER_M3S)
        values[int(comid)] = nse(observed, predicted)
    return pd.Series(values, dtype=float)


def legacy_lowflow_log_rmse(frame: pd.DataFrame, reaches: set[int]) -> pd.Series:
    frame = frame[frame["comid"].isin(reaches)].copy()
    frame["lowflow"] = False
    for _, group in frame.groupby(["comid", "fold_id"], sort=False):
        count = max(1, int(math.ceil(len(group) * 0.25)))
        indices = group.sort_values(["actual", "year", "month"]).head(count).index
        frame.loc[indices, "lowflow"] = True
    values = {}
    for comid, group in frame[frame["lowflow"]].groupby("comid", sort=True):
        observed = np.log1p(group["actual"].to_numpy(float) / CFS_PER_M3S)
        predicted = np.log1p(group["predict"].to_numpy(float) / CFS_PER_M3S)
        values[int(comid)] = float(np.sqrt(np.mean(np.square(predicted - observed))))
    return pd.Series(values, dtype=float)


def main() -> None:
    frames = {scenario: load(scenario) for scenario in SCENARIOS}
    reference = frames["S00_NATIVE"]
    scenario_metrics = pd.read_csv(REPORT / "scenario_metrics.csv", encoding="utf-8-sig").set_index("scenario")
    station_metrics = pd.read_csv(REPORT / "station_metrics.csv", encoding="utf-8-sig")
    registry = pd.read_csv(
        RUN / "inputs" / "source_snapshot" / "experiment_reference" / "canonical_signal_registry.csv",
        encoding="utf-8-sig",
    )
    membership = registry["legacy_canonical_membership"].astype(str).str.casefold().isin({"true", "1"})
    legacy_reaches = set(pd.to_numeric(registry.loc[membership, "reach_id"], errors="coerce").dropna().astype(int))

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    for scenario, frame in frames.items():
        checks[f"{scenario}_rows_8738"] = len(frame) == 8738
        checks[f"{scenario}_keys_unique"] = not bool(frame.duplicated(KEYS).any())
        checks[f"{scenario}_keys_equal_s00"] = frame[KEYS].equals(reference[KEYS])
        checks[f"{scenario}_actual_equal_s00"] = bool(np.array_equal(frame["actual"].to_numpy(), reference["actual"].to_numpy()))
        pooled_log_observed = np.log1p(frame["actual"].to_numpy(float) / CFS_PER_M3S)
        pooled_log_predicted = np.log1p(frame["predict"].to_numpy(float) / CFS_PER_M3S)
        pooled_log_nse = nse(pooled_log_observed, pooled_log_predicted)
        station_values = station_log_nse(frame)
        reported_pooled = float(scenario_metrics.loc[scenario, "log_nse"])
        reported_median = float(station_metrics[station_metrics["scenario"].eq(scenario)]["log_nse"].median())
        checks[f"{scenario}_pooled_log_nse_recomputed"] = abs(pooled_log_nse - reported_pooled) <= 1.0e-12
        checks[f"{scenario}_median_station_log_nse_recomputed"] = abs(float(station_values.median()) - reported_median) <= 1.0e-12
        details[scenario] = {
            "pooled_log_nse_recomputed": pooled_log_nse,
            "pooled_log_nse_reported": reported_pooled,
            "median_station_log_nse_recomputed": float(station_values.median()),
            "median_station_log_nse_reported": reported_median,
        }

    expected = json.loads(
        (RUN / "inputs" / "source_snapshot" / "baseline_reference" / "contracts" / "q72_model_gate.json").read_text(encoding="utf-8")
    )["three_fold_oof"]["median_log_nse"]
    recomputed_s00 = float(station_log_nse(reference).median())
    checks["s00_matches_frozen_canonical_median_log_nse"] = abs(recomputed_s00 - expected) <= 5.0e-11
    details["s00_canonical_comparison"] = {"recomputed": recomputed_s00, "frozen_gate": expected, "absolute_difference": abs(recomputed_s00 - expected)}

    lowflow = {scenario: legacy_lowflow_log_rmse(frame, legacy_reaches) for scenario, frame in frames.items()}
    common = lowflow["S00_NATIVE"].index.intersection(lowflow["S01"].index)
    s01_improved = int((lowflow["S01"].loc[common] < lowflow["S00_NATIVE"].loc[common]).sum())
    reported_legacy = pd.read_csv(REPORT / "legacy_lowflow_metrics.csv", encoding="utf-8-sig")
    reported_s01 = reported_legacy[reported_legacy["scenario"].eq("S01")]
    checks["legacy_evaluable_27"] = len(common) == 27
    checks["s01_legacy_lowflow_improvement_count_recomputed"] = s01_improved == int(reported_s01["lowflow_log_rmse_improved_vs_s00"].sum())
    details["legacy_lowflow"] = {"evaluable": int(len(common)), "s01_improved_recomputed": s01_improved}

    chart_results = {}
    for chart in sorted((REPORT / "charts").glob("*.png")):
        with Image.open(chart) as image:
            width, height = image.size
            chart_results[chart.name] = {"width": width, "height": height, "mode": image.mode}
            checks[f"chart_{chart.stem}_readable"] = width >= 800 and height >= 400
    checks["three_required_charts_present"] = len(chart_results) == 3
    details["charts"] = chart_results

    result = {
        "runtime": RUNTIME,
        "checks": checks,
        "passed_count": int(sum(checks.values())),
        "check_count": int(len(checks)),
        "details": details,
        "passed": all(checks.values()),
        "confidence": "Ready to share with explicit provisional-overlay caveat" if all(checks.values()) else "Needs revision",
    }
    (REPORT / "analysis_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
