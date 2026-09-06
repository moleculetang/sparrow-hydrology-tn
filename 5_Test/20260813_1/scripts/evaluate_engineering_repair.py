from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["comid", "q_site", "year", "month", "fold_id"]
SCENARIOS = ["B0", "B1"]


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame.actual.to_numpy(float)
    pred = frame.predict.to_numpy(float)
    lo, lp = np.log(obs), np.log(pred)
    return {
        "n": int(len(frame)),
        "raw_nse": float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)),
        "log_nse": float(1 - np.sum((lp - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)),
        "pbias_pct": float(100 * np.sum(pred - obs) / np.sum(obs)),
        "rmse_cfs": float(np.sqrt(np.mean((pred - obs) ** 2))),
        "mae_cfs": float(np.mean(np.abs(pred - obs))),
        "log_rmse": float(np.sqrt(np.mean((lp - lo) ** 2))),
    }


def station_metrics(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = []
    for site, part in frame.groupby("q_site", sort=True):
        rows.append({
            "scenario": scenario, "q_site": site, "comid": int(part.comid.iloc[0]),
            "observed_mean_cfs": float(part.actual.mean()), **metrics(part),
        })
    return pd.DataFrame(rows)


def path_metrics(frame: pd.DataFrame, pairs: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = []
    for record in pairs.itertuples(index=False):
        up = frame.loc[frame.comid.eq(int(record.upstream_reach_id)), ["year", "month", "fold_id", "actual", "predict"]].rename(columns={"actual": "up_actual", "predict": "up_predict"})
        down = frame.loc[frame.comid.eq(int(record.downstream_reach_id)), ["year", "month", "fold_id", "actual", "predict"]].rename(columns={"actual": "down_actual", "predict": "down_predict"})
        joined = up.merge(down, on=["year", "month", "fold_id"])
        if joined.empty:
            continue
        rows.append({
            "scenario": scenario,
            "path_key": record.path_key,
            "upstream_station": record.upstream_station,
            "downstream_station": record.downstream_station,
            "n": int(len(joined)),
            "upstream_mae_cfs": float(np.mean(np.abs(joined.up_predict - joined.up_actual))),
            "downstream_mae_cfs": float(np.mean(np.abs(joined.down_predict - joined.down_actual))),
        })
    return pd.DataFrame(rows)


def lowflow_metrics(frame: pd.DataFrame, targets: set[str], scenario: str) -> pd.DataFrame:
    rows = []
    for (site, fold), part in frame[frame.q_site.isin(targets)].groupby(["q_site", "fold_id"]):
        selected = part[part.actual <= part.actual.quantile(.25)]
        error = np.log(selected.predict) - np.log(selected.actual)
        rows.append({
            "scenario": scenario, "q_site": site, "fold_id": fold, "n": len(selected),
            "median_log_bias": float(np.median(error)),
            "absolute_median_log_bias": float(abs(np.median(error))),
            "log_rmse": float(np.sqrt(np.mean(error**2))),
            "absolute_error_cfs": float(np.abs(selected.predict-selected.actual).sum()),
            "pbias_pct": float(100*(selected.predict-selected.actual).sum()/selected.actual.sum()),
        })
    return pd.DataFrame(rows)


def station_key(value: object) -> str:
    return str(value).strip().replace("（", "(").replace("）", ")").removesuffix("站")


def main() -> None:
    frames = {
        scenario: pd.read_parquet(ROOT / "outputs" / scenario / "q72_three_fold_oof_predictions.parquet")
        for scenario in SCENARIOS
    }
    for frame in frames.values():
        frame["q_site"] = frame.q_site.astype(str)
    a0 = frames["B0"].sort_values(KEY).reset_index(drop=True)
    a1 = frames["B1"].sort_values(KEY).reset_index(drop=True)
    if not a0[KEY].equals(a1[KEY]) or np.max(np.abs(a0.actual-a1.actual)) > 1e-12:
        raise RuntimeError("A0/A1 OOF identity failed")

    scenario_rows, fold_rows = [], []
    for scenario, frame in frames.items():
        scenario_rows.append({"scenario": scenario, **metrics(frame)})
        for fold, part in frame.groupby("fold_id"):
            fold_rows.append({"scenario": scenario, "fold_id": fold, **metrics(part)})
    scenario_df = pd.DataFrame(scenario_rows)
    fold_df = pd.DataFrame(fold_rows)
    scenario_df.to_csv(ROOT / "reports" / "scenario_metrics.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(ROOT / "reports" / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    station = pd.concat([station_metrics(frame, scenario) for scenario, frame in frames.items()], ignore_index=True)
    station.to_csv(ROOT / "reports" / "station_metrics.csv", index=False, encoding="utf-8-sig")
    base_station = station[station.scenario.eq("B0")]
    large_sites = set(base_station.nlargest(max(1, int(np.ceil(len(base_station)*.2))), "observed_mean_cfs").q_site)

    q90 = a0.groupby("q_site").actual.transform(lambda series: series.quantile(.9))
    high_mask = a0.actual.ge(q90)
    high0, high1 = a0.loc[high_mask], a1.loc[high_mask]
    large0, large1 = a0[a0.q_site.isin(large_sites)], a1[a1.q_site.isin(large_sites)]

    pairs = pd.read_csv(ROOT / "inputs" / "confirmed_evaluable_nearest_downstream_paths.csv", encoding="utf-8-sig")
    paths = pd.concat([path_metrics(frame, pairs, scenario) for scenario, frame in frames.items()], ignore_index=True)
    paths.to_csv(ROOT / "reports" / "topology_path_metrics.csv", index=False, encoding="utf-8-sig")
    path_pivot = paths.pivot(index="path_key", columns="scenario", values="downstream_mae_cfs")

    registry = pd.read_csv(ROOT / "inputs" / "canonical_signal_registry.csv", encoding="utf-8-sig")
    labels = set(registry.loc[registry.legacy_canonical_membership.eq(True), "q_site"].astype(str))
    model_by_key = {station_key(site): site for site in set(a0.q_site)}
    targets = {model_by_key[station_key(label)] for label in labels if station_key(label) in model_by_key}
    unmatched_labels = sorted(label for label in labels if station_key(label) not in model_by_key)
    low = pd.concat([lowflow_metrics(frame, targets, scenario) for scenario, frame in frames.items()], ignore_index=True)
    low.to_csv(ROOT / "reports" / "legacy_lowflow_metrics.csv", index=False, encoding="utf-8-sig")

    overall0, overall1 = metrics(a0), metrics(a1)
    fold_pivot = fold_df.pivot(index="fold_id", columns="scenario", values="log_nse")
    checks = {
        "pooled_raw_nse_decline_le_0_001": overall1["raw_nse"] >= overall0["raw_nse"] - .001,
        "pooled_log_nse_decline_le_0_001": overall1["log_nse"] >= overall0["log_nse"] - .001,
        "every_fold_log_nse_decline_le_0_003": bool((fold_pivot.B1 >= fold_pivot.B0 - .003).all()),
        "highflow_logrmse_worsen_le_1pct": metrics(high1)["log_rmse"] <= metrics(high0)["log_rmse"]*1.01,
        "large_station_logrmse_worsen_le_1pct": metrics(large1)["log_rmse"] <= metrics(large0)["log_rmse"]*1.01,
        "path_mean_downstream_mae_worsen_le_1pct": float(path_pivot.B1.mean()) <= float(path_pivot.B0.mean())*1.01,
        "abs_pbias_worsen_le_1point": abs(overall1["pbias_pct"]) <= abs(overall0["pbias_pct"])+1,
        "oof_identity": True,
        "all_feature_accounting_gates": True,
    }
    feature_audits = []
    for scenario in SCENARIOS:
        for fold in sorted(frames[scenario].fold_id.unique()):
            payload = json.loads((ROOT / "outputs" / scenario / "blocked_folds" / fold / "feature_accounting_audit.json").read_text(encoding="utf-8"))
            feature_audits.append(payload)
    checks["all_feature_accounting_gates"] = all(
        row["forcing_rows"] == 46920 and row["reaches"] == 230
        and row["months_per_reach_min"] == row["months_per_reach_max"] == 204
        and row["max_abs_unscaled_topology_error_cfs"] <= 1e-8
        and row["max_abs_scaled_predictor_error_cfs"] <= 1e-8
        and row["negative_unscaled_input_count"] == 0
        for row in feature_audits
    )

    result = {
        "B0": overall0,
        "B1": overall1,
        "delta_B1_minus_B0": {key: float(overall1[key]-overall0[key]) for key in ["raw_nse", "log_nse", "pbias_pct", "rmse_cfs", "log_rmse"]},
        "highflow_logrmse_change_pct": float(100*(metrics(high1)["log_rmse"]/metrics(high0)["log_rmse"]-1)),
        "large_station_logrmse_change_pct": float(100*(metrics(large1)["log_rmse"]/metrics(large0)["log_rmse"]-1)),
        "path_mean_downstream_mae_change_pct": float(100*(path_pivot.B1.mean()/path_pivot.B0.mean()-1)),
        "abs_pbias_change_points": float(abs(overall1["pbias_pct"])-abs(overall0["pbias_pct"])),
        "fold_log_nse_delta": {index: float(row.B1-row.B0) for index, row in fold_pivot.iterrows()},
        "legacy_registry_members": len(labels),
        "legacy_evaluable_targets": len(targets),
        "legacy_unmatched_labels": unmatched_labels,
        "checks": {key: bool(value) for key, value in checks.items()},
    }
    all_pass = all(checks.values())
    result["all_promotion_gates_pass"] = all_pass
    result["terminal"] = (
        "Q72_DETERMINISTIC_ENGINEERING_REPAIR_COMPLETE_AND_PROTECTED"
        if all_pass else "Q72_DETERMINISTIC_ENGINEERING_REPAIR_COMPLETE_PREDICTIVE_PROTECTION_NOT_MET"
    )
    (ROOT / "terminal_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(feature_audits).to_csv(ROOT / "reports" / "feature_accounting_audit.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
