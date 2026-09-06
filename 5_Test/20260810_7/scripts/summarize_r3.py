from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
BASELINE = RUN / "inputs" / "baseline" / "q72_clean_oof.parquet"
SIGNALS = RUN / "inputs" / "baseline" / "canonical_signal_registry.csv"
TABLES = RUN / "reports" / "tables"
SCENARIOS = ["R3_00", "R3_10", "R3_01", "R3_11"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {"comid": "reach_id", "q_site": "station_name", "actual": "observed_cfs", "predict": "predicted_cfs"}
    out = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns}).copy()
    required = ["reach_id", "station_name", "year", "month", "fold_id", "observed_cfs", "predicted_cfs"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise RuntimeError(f"OOF columns missing: {missing}")
    return out


def station_key(value: object) -> str:
    text = str(value).strip().replace("(", "（").replace(")", "）")
    if text.endswith("站"):
        text = text[:-1]
    return "".join(text.split())


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame["observed_cfs"].to_numpy(float)
    pred = frame["predicted_cfs"].to_numpy(float)
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[mask], pred[mask]
    if not len(obs):
        return {name: np.nan for name in ["NSE_raw", "NSE_log", "KGE", "PBIAS_pct", "RMSE_cfs", "log_RMSE"]} | {"n": 0}
    err = pred - obs
    log_obs, log_pred = np.log(obs), np.log(pred)
    raw_sst = np.sum((obs - obs.mean()) ** 2)
    log_sst = np.sum((log_obs - log_obs.mean()) ** 2)
    corr = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 and np.std(obs) > 0 and np.std(pred) > 0 else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) else np.nan
    kge = 1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) if np.isfinite(corr) else np.nan
    return {
        "n": int(len(obs)),
        "NSE_raw": float(1 - np.sum(err**2) / raw_sst) if raw_sst > 0 else np.nan,
        "NSE_log": float(1 - np.sum((log_pred - log_obs) ** 2) / log_sst) if log_sst > 0 else np.nan,
        "KGE": float(kge),
        "PBIAS_pct": float(100 * np.sum(err) / np.sum(obs)),
        "RMSE_cfs": float(np.sqrt(np.mean(err**2))),
        "log_RMSE": float(np.sqrt(np.mean((log_pred - log_obs) ** 2))),
    }


def lowflow_mask(reference: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=reference.index)
    for _, part in reference.groupby(["station_name", "fold_id"], sort=False):
        count = max(1, int(np.ceil(len(part) * 0.25)))
        selected = part.sort_values(["observed_cfs", "year", "month"], kind="stable").index[:count]
        mask.loc[selected] = True
    return mask


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    baseline = normalize(pd.read_parquet(BASELINE)).sort_values(["reach_id", "station_name", "year", "month", "fold_id"]).reset_index(drop=True)
    frames = {}
    key_columns = ["reach_id", "station_name", "year", "month", "fold_id"]
    key_checks = []
    for scenario in SCENARIOS:
        path = RUN / "outputs" / scenario / "q72_three_fold_oof_predictions.parquet"
        frame = normalize(pd.read_parquet(path)).sort_values(key_columns).reset_index(drop=True)
        if len(frame) != 8738:
            raise RuntimeError(f"{scenario} OOF rows: {len(frame)}")
        merged = baseline[key_columns].merge(frame[key_columns], on=key_columns, how="outer", indicator=True)
        identical = bool(merged["_merge"].eq("both").all() and len(merged) == 8738)
        key_checks.append({"scenario": scenario, "rows": len(frame), "keys_identical": identical})
        if not identical:
            raise RuntimeError(f"{scenario} keys differ from baseline")
        frames[scenario] = frame
    pd.DataFrame(key_checks).to_csv(TABLES / "oof_key_gate.csv", index=False, encoding="utf-8-sig")

    r300 = frames["R3_00"]
    prediction_max_abs = float(np.max(np.abs(r300["predicted_cfs"].to_numpy(float) - baseline["predicted_cfs"].to_numpy(float))))
    observed_max_abs = float(np.max(np.abs(r300["observed_cfs"].to_numpy(float) - baseline["observed_cfs"].to_numpy(float))))
    reproduction_pass = prediction_max_abs <= 1e-8 and observed_max_abs <= 1e-8

    signal = pd.read_csv(SIGNALS)
    legacy = set(signal.loc[
        signal["stable_nonreservoir_target"].fillna(False).astype(bool), "q_site"
    ].map(station_key))
    low_mask = lowflow_mask(r300)
    mean_flow = r300.groupby("station_name")["observed_cfs"].mean()
    large = set(mean_flow[mean_flow >= mean_flow.quantile(0.75)].index.astype(str))
    metric_rows = []
    station_rows = []
    low_rows = []
    for scenario, frame in frames.items():
        scopes = {
            "pooled": np.ones(len(frame), dtype=bool),
            "legacy_lowflow_targets_all_months": frame["station_name"].map(station_key).isin(legacy).to_numpy(),
            "large_stations": frame["station_name"].isin(large).to_numpy(),
            "shijiao": frame["station_name"].str.contains("石角", regex=False, na=False).to_numpy(),
        }
        for scope, mask in scopes.items():
            metric_rows.append({"scenario": scenario, "scope": scope, "scope_id": "ALL", **metrics(frame.loc[mask])})
        for fold, part in frame.groupby("fold_id", sort=False):
            metric_rows.append({"scenario": scenario, "scope": "fold", "scope_id": fold, **metrics(part)})
        per_station = []
        for (reach, station), part in frame.groupby(["reach_id", "station_name"], sort=False):
            row = {"scenario": scenario, "reach_id": int(reach), "station_name": station, **metrics(part)}
            row["legacy_target"] = station_key(station) in legacy
            row["large_station"] = str(station) in large
            row["shijiao"] = "石角" in str(station)
            station_rows.append(row)
            per_station.append(row)
        per_station_frame = pd.DataFrame(per_station)
        medians = {f"median_{name}": float(per_station_frame[name].median()) for name in ["NSE_raw", "NSE_log", "KGE", "PBIAS_pct", "RMSE_cfs", "log_RMSE"]}
        metric_rows.append({"scenario": scenario, "scope": "station_median", "scope_id": "ALL", "n": len(per_station_frame), **medians})

        lf = frame.loc[low_mask].copy()
        lf["log_error"] = np.log(lf["predicted_cfs"]) - np.log(lf["observed_cfs"])
        for (reach, station), part in lf.groupby(["reach_id", "station_name"], sort=False):
            low_rows.append({
                "scenario": scenario, "reach_id": int(reach), "station_name": station,
                "legacy_target": station_key(station) in legacy,
                "n_lowflow": len(part),
                "median_log_bias": float(part["log_error"].median()),
                "absolute_median_log_bias": float(abs(part["log_error"].median())),
                "log_RMSE": float(np.sqrt(np.mean(part["log_error"] ** 2))),
                "absolute_error_cfs": float(np.sum(np.abs(part["predicted_cfs"] - part["observed_cfs"]))),
                "observed_volume_cfs_months": float(part["observed_cfs"].sum()),
                "PBIAS_pct": float(100 * np.sum(part["predicted_cfs"] - part["observed_cfs"]) / part["observed_cfs"].sum()),
            })

    metric_frame = pd.DataFrame(metric_rows)
    station_frame = pd.DataFrame(station_rows)
    low_frame = pd.DataFrame(low_rows)
    metric_frame.to_csv(TABLES / "scenario_oof_metrics.csv", index=False, encoding="utf-8-sig")
    station_frame.to_csv(TABLES / "station_oof_metrics.csv", index=False, encoding="utf-8-sig")
    low_frame.to_csv(TABLES / "station_lowflow_metrics.csv", index=False, encoding="utf-8-sig")

    baseline_metrics = metric_frame[(metric_frame["scenario"] == "R3_00") & metric_frame["NSE_log"].notna()].copy()
    paired_rows = []
    for scenario in SCENARIOS[1:]:
        current = metric_frame[(metric_frame["scenario"] == scenario) & metric_frame["NSE_log"].notna()].copy()
        paired = baseline_metrics.merge(current, on=["scope", "scope_id"], suffixes=("_baseline", "_scenario"))
        for row in paired.itertuples(index=False):
            item = {"scenario": scenario, "scope": row.scope, "scope_id": row.scope_id}
            for name in ["NSE_raw", "NSE_log", "KGE", "PBIAS_pct", "RMSE_cfs", "log_RMSE"]:
                item[f"delta_{name}"] = float(getattr(row, f"{name}_scenario") - getattr(row, f"{name}_baseline"))
            paired_rows.append(item)
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(TABLES / "paired_scenario_effects.csv", index=False, encoding="utf-8-sig")

    low_base = low_frame[low_frame["scenario"].eq("R3_00")][["reach_id", "station_name", "legacy_target", "absolute_median_log_bias", "log_RMSE"]]
    low_effect_rows = []
    for scenario in SCENARIOS[1:]:
        current = low_frame[low_frame["scenario"].eq(scenario)][["reach_id", "station_name", "absolute_median_log_bias", "log_RMSE"]]
        paired = low_base.merge(current, on=["reach_id", "station_name"], suffixes=("_baseline", "_scenario"))
        paired["scenario"] = scenario
        paired["delta_absolute_median_log_bias"] = paired["absolute_median_log_bias_scenario"] - paired["absolute_median_log_bias_baseline"]
        paired["delta_log_RMSE"] = paired["log_RMSE_scenario"] - paired["log_RMSE_baseline"]
        low_effect_rows.append(paired)
    low_effect = pd.concat(low_effect_rows, ignore_index=True)
    low_effect.to_csv(TABLES / "paired_lowflow_effects.csv", index=False, encoding="utf-8-sig")

    input_audit = json.loads((RUN / "reports" / "r3_input_audit.json").read_text(encoding="utf-8"))
    input_build = json.loads((RUN / "reports" / "r3_input_build.json").read_text(encoding="utf-8"))
    run_gate = json.loads((RUN / "logs" / "scenario_run_gate.json").read_text(encoding="utf-8"))
    pooled = metric_frame[(metric_frame["scope"].eq("pooled")) & (metric_frame["scope_id"].eq("ALL"))].set_index("scenario")
    station_medians = metric_frame[metric_frame["scope"].eq("station_median")].set_index("scenario")
    combined_gain = float(pooled.loc["R3_11", "NSE_log"] - pooled.loc["R3_00", "NSE_log"])
    legacy_combined = low_effect[(low_effect["scenario"].eq("R3_11")) & low_effect["legacy_target"]]
    evaluable_legacy = set(r300["station_name"].map(station_key)) & legacy
    engineering_valid = bool(
        input_audit["input_contract_pass"] and input_build["max_upstream_area_relative_error"] <= 1e-8
        and run_gate["passed"] and reproduction_pass
    )
    if not reproduction_pass:
        terminal = "R3_BASELINE_REPRODUCTION_FAILURE"
    elif not input_audit["input_contract_pass"]:
        terminal = "R3_INPUT_CONTRACT_FAILURE"
    elif not engineering_valid:
        terminal = "R3_RUN_INCOMPLETE"
    elif combined_gain > 0:
        terminal = "R3_ENGINEERING_VALID_WITH_OOF_IMPROVEMENT"
    else:
        terminal = "R3_ENGINEERING_VALID_BUT_NO_OOF_IMPROVEMENT"
    result = {
        "runtime": RUNTIME,
        "baseline_oof_sha256": sha256(BASELINE),
        "baseline_reproduction": {
            "prediction_max_abs_difference_cfs": prediction_max_abs,
            "observed_max_abs_difference_cfs": observed_max_abs,
            "passed": reproduction_pass,
        },
        "engineering_valid": engineering_valid,
        "dem_corrected_et0_completed": input_audit["dem_corrected_et0_completed"],
        "dem_elevation_note": input_audit["dem_correction_status"],
        "pooled_metrics": pooled[["NSE_raw", "NSE_log", "KGE", "PBIAS_pct", "RMSE_cfs", "log_RMSE"]].to_dict(orient="index"),
        "combined_pooled_log_nse_gain": combined_gain,
        "legacy_target_count_in_registry": len(legacy),
        "legacy_target_count_evaluable": len(evaluable_legacy),
        "combined_station_median_log_nse_change": float(
            station_medians.loc["R3_11", "median_NSE_log"] - station_medians.loc["R3_00", "median_NSE_log"]
        ),
        "combined_station_median_raw_nse_change": float(
            station_medians.loc["R3_11", "median_NSE_raw"] - station_medians.loc["R3_00", "median_NSE_raw"]
        ),
        "combined_legacy_lowflow_abs_bias_improved_stations": int(
            legacy_combined["delta_absolute_median_log_bias"].lt(0).sum()
        ),
        "combined_legacy_lowflow_median_delta_abs_log_bias": float(
            legacy_combined["delta_absolute_median_log_bias"].median()
        ),
        "combined_legacy_lowflow_median_delta_log_rmse": float(
            legacy_combined["delta_log_RMSE"].median()
        ),
        "terminal": terminal,
    }
    (RUN / "terminal_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
