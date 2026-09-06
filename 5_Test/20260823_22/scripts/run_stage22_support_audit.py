from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_22"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
sys.path.insert(0, str(TEST / "20260823_17" / "scripts"))
from regionalization import summary_metrics  # noqa: E402


COVERAGE = TEST / "20260823_14" / "outputs" / "final_user_locked_station_coverage.parquet"
Q72 = TEST / "20260823_15" / "final_outputs" / "q72_physical_reference_predictions.parquet"
LOCAL_PARAMS = TEST / "20260823_16" / "outputs" / "local_map5_parameter_audit.parquet"
PRED21 = TEST / "20260823_21" / "outputs" / "locked_2019_2022_predictions.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    good = a.notna() & b.notna()
    return float(np.corrcoef(a[good], b[good])[0, 1]) if good.sum() >= 3 else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_audit":
        raise RuntimeError("Support audit was not pre-registered")
    coverage = pd.read_parquet(COVERAGE)
    coverage["q_site"] = coverage.station_norm.astype(str)
    coverage = coverage[coverage.selected_for_model].copy()
    q72 = pd.read_parquet(Q72)
    q72["q_site"] = q72.station_norm.astype(str)
    dev = q72[q72.year.le(2018)].copy()
    dev["q72_obs_ratio"] = dev.Q72_PROCESS_cfs / dev.Q_obsv_cfs.clip(lower=1e-12)
    dev["log_q72_minus_obs"] = np.log1p(dev.Q72_PROCESS_cfs) - np.log1p(dev.Q_obsv_cfs)
    station = dev.groupby(["q_site", "reach_id"], as_index=False).agg(
        n_months=("Q_obsv_cfs", "size"), observed_mean_cfs=("Q_obsv_cfs", "mean"),
        q72_mean_cfs=("Q72_PROCESS_cfs", "mean"), median_q72_obs_ratio=("q72_obs_ratio", "median"),
        median_log_q72_minus_obs=("log_q72_minus_obs", "median"),
    )
    keep_cols = [
        "q_site", "reach_id", "downstream_fraction_on_reach", "snap_distance_m", "line_catchment_override",
        "best_line_distance_m", "reach_assignment_method", "development_mean_Q_cfs",
    ]
    station = station.merge(coverage[keep_cols], on=["q_site", "reach_id"], validate="one_to_one")
    params = pd.read_parquet(LOCAL_PARAMS)[["q_site", "reach_id", "intercept", "log_q72_total", "q72_quick_fraction", "month_sin", "month_cos"]]
    station = station.merge(params, on=["q_site", "reach_id"], validate="one_to_one")
    station["interior_location"] = station.downstream_fraction_on_reach.lt(0.8)
    station["moderate_scale_mismatch"] = ~station.median_q72_obs_ratio.between(1 / 3, 3)
    station["severe_scale_mismatch"] = ~station.median_q72_obs_ratio.between(0.1, 10)
    station["spatial_match_risk"] = station.snap_distance_m.gt(5000) | station.line_catchment_override.fillna(False)
    station["outlet_compatible_diagnostic"] = (~station.moderate_scale_mismatch) & (~station.spatial_match_risk) & station.downstream_fraction_on_reach.ge(0.8)
    station["negative_log_scale_mismatch"] = -station.median_log_q72_minus_obs
    station.to_parquet(OUT / "station_reach_support_audit.parquet", index=False)

    by_location = station.groupby("interior_location", as_index=False).agg(
        station_count=("q_site", "size"), moderate_mismatch_fraction=("moderate_scale_mismatch", "mean"),
        severe_mismatch_fraction=("severe_scale_mismatch", "mean"),
        median_abs_log_mismatch=("median_log_q72_minus_obs", lambda x: float(np.median(np.abs(x)))),
    )
    by_location.to_parquet(OUT / "support_mismatch_by_location.parquet", index=False)
    correlations = []
    for coefficient in ["intercept", "log_q72_total", "q72_quick_fraction", "month_sin", "month_cos"]:
        correlations.append({
            "coefficient": coefficient,
            "correlation_with_negative_log_scale_mismatch": safe_corr(station[coefficient], station.negative_log_scale_mismatch),
            "correlation_with_downstream_fraction": safe_corr(station[coefficient], station.downstream_fraction_on_reach),
        })
    corr = pd.DataFrame(correlations)
    corr.to_parquet(OUT / "map5_support_correlations.parquet", index=False)

    # Sensitivity only: authoritative all-station metrics remain untouched.
    pred = pd.read_parquet(PRED21)
    pred["outlet_compatible_diagnostic"] = pred.q_site.map(dict(zip(station.q_site, station.outlet_compatible_diagnostic))).fillna(False)
    sensitivity_rows = []
    for subset, frame in [("ALL_AUTHORITATIVE", pred), ("OUTLET_COMPATIBLE_DIAGNOSTIC", pred[pred.outlet_compatible_diagnostic])]:
        for model, col in [("LOCAL_STATION_MAP", "Q_MAP_cfs"), ("NETWORK_NATIVE_MAP5", "Q_NETWORK_NATIVE_cfs"), ("Q72", "Q72_cfs")]:
            sensitivity_rows.append({"subset": subset, "model": model, **summary_metrics(frame, col)})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_parquet(OUT / "outlet_compatibility_performance_sensitivity.parquet", index=False)

    summary = {
        "stage": "20260823_22",
        "status": "SUPPORT_MISMATCH_PRESENT" if station.severe_scale_mismatch.any() else "NO_SEVERE_SUPPORT_MISMATCH",
        "station_count": int(len(station)),
        "interior_location_count": int(station.interior_location.sum()),
        "moderate_scale_mismatch_count": int(station.moderate_scale_mismatch.sum()),
        "severe_scale_mismatch_count": int(station.severe_scale_mismatch.sum()),
        "spatial_match_risk_count": int(station.spatial_match_risk.sum()),
        "outlet_compatible_diagnostic_count": int(station.outlet_compatible_diagnostic.sum()),
        "intercept_correlation_with_negative_log_scale_mismatch": float(corr.loc[corr.coefficient.eq("intercept"), "correlation_with_negative_log_scale_mismatch"].iloc[0]),
        "max_median_q72_observation_ratio": float(station.median_q72_obs_ratio.max()),
        "max_ratio_station": str(station.loc[station.median_q72_obs_ratio.idxmax(), "q_site"]),
        "authoritative_station_removed": 0,
        "external_four_stations_read": False,
        "interpretation": "Station MAP mixes an observation-support operator with Reach-flow correction when scale mismatch is large."
    }
    (REPORT / "stage22_decision.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    extremes = station.sort_values("median_q72_obs_ratio", ascending=False).head(15)[[
        "q_site", "reach_id", "downstream_fraction_on_reach", "median_q72_obs_ratio", "intercept", "spatial_match_risk"
    ]]
    (REPORT / "technical_report.md").write_text(
        "# 20260823_22 station–Reach support audit\n\n"
        + f"Status: `{summary['status']}`. No station was removed.\n\n## Summary\n\n"
        + pd.DataFrame([summary]).to_markdown(index=False) + "\n\n## Location comparison\n\n"
        + by_location.to_markdown(index=False) + "\n\n## MAP5 relationship\n\n"
        + corr.to_markdown(index=False) + "\n\n## Largest scale mismatches\n\n"
        + extremes.to_markdown(index=False) + "\n\n## Performance sensitivity (diagnostic only)\n\n"
        + sensitivity.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "coverage_sha256": sha256(COVERAGE), "q72_sha256": sha256(Q72),
        "audit_sha256": sha256(OUT / "station_reach_support_audit.parquet"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(by_location.to_string(index=False))
    print(corr.to_string(index=False))
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
