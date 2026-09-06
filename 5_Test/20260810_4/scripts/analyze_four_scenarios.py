from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "scenario_diagnostics"
SCENARIOS = ("S00_NATIVE", "S10", "S01", "S11")
SCENARIO_LABEL = {
    "S00_NATIVE": "S00 original + observed-only",
    "S10": "S10 overlay + observed-only",
    "S01": "S01 original + full-forcing",
    "S11": "S11 overlay + full-forcing",
}
KEYS = ["fold_id", "comid", "year", "month"]
EPS = 1.0e-12
CFS_PER_M3S = 35.31466672148859
LARGE_FLOW_THRESHOLD_M3S = 100.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    denominator = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((pred - obs) ** 2) / denominator) if denominator > EPS else np.nan


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 4 or obs.std(ddof=0) <= EPS or abs(obs.mean()) <= EPS or pred.std(ddof=0) <= EPS:
        return np.nan
    correlation = np.corrcoef(obs, pred)[0, 1]
    if not np.isfinite(correlation):
        return np.nan
    variability_ratio = pred.std(ddof=0) / obs.std(ddof=0)
    bias_ratio = pred.mean() / obs.mean()
    return float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (variability_ratio - 1.0) ** 2 + (bias_ratio - 1.0) ** 2))


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame["actual"].to_numpy(float)
    pred = np.maximum(frame["predict"].to_numpy(float), 0.0)
    # This is the frozen Q72 audit definition: log1p after converting cfs to m3/s.
    log_obs = np.log1p(np.maximum(obs, 0.0) / CFS_PER_M3S)
    log_pred = np.log1p(pred / CFS_PER_M3S)
    residual = log_pred - log_obs
    return {
        "rows": int(len(frame)),
        "nse": nse(obs, pred),
        "log_nse": nse(log_obs, log_pred),
        "kge": kge(obs, pred),
        "pbias_pct": float(100.0 * (pred.sum() - obs.sum()) / max(obs.sum(), EPS)),
        "rmse_cfs": float(np.sqrt(np.mean((pred - obs) ** 2))),
        "mae_cfs": float(np.mean(np.abs(pred - obs))),
        "log_rmse": float(np.sqrt(np.mean(residual**2))),
        "median_log_bias": float(np.median(residual)),
        "absolute_error_cfs": float(np.sum(np.abs(pred - obs))),
        "observed_volume_cfs_months": float(np.sum(obs)),
    }


def read_selected_station_map() -> pd.DataFrame:
    path = RUN / "inputs" / "source_snapshot" / "baseline_reference" / "contracts" / "selected_representative_stations.csv"
    selected = pd.read_csv(path, encoding="utf-8-sig")
    selected = selected[selected["selected_for_reach"].astype(str).str.casefold().isin({"true", "1"})].copy()
    selected["comid"] = pd.to_numeric(selected["reach_id"], errors="raise").astype(int)
    selected["station_name"] = selected["station_name"].astype(str)
    return selected[["comid", "station_name", "median_q_cfs", "usable_months", "snap_distance_m"]]


def load_scenarios() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    integrity: dict[str, Any] = {"runtime": RUNTIME, "scenarios": {}}
    reference_keys: pd.DataFrame | None = None
    reference_actual: pd.Series | None = None
    expected = {
        "S00_NATIVE": ("observed_only", "6d6f178cf96bc22d095ce13588d225edd0f8143422b1b67b2163c93ba6fd0f18"),
        "S10": ("observed_only", "5309d76af5ab225ea7e7e0c41d8b472beacbbbdff3e2754bbdc83dec797ec846"),
        "S01": ("full_forcing", "6d6f178cf96bc22d095ce13588d225edd0f8143422b1b67b2163c93ba6fd0f18"),
        "S11": ("full_forcing", "5309d76af5ab225ea7e7e0c41d8b472beacbbbdff3e2754bbdc83dec797ec846"),
    }
    for scenario in SCENARIOS:
        scenario_dir = RUN / "outputs" / "scenarios" / scenario
        prediction_path = scenario_dir / "q72_three_fold_oof_predictions.parquet"
        manifest_path = scenario_dir / "blocked_fold_manifest.json"
        frame = pd.read_parquet(prediction_path).sort_values(KEYS).reset_index(drop=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mode, input_hash = expected[scenario]
        scenario_checks = {
            "rows_8738": len(frame) == 8738,
            "reaches_110": frame["comid"].nunique() == 110,
            "folds_3": frame["fold_id"].nunique() == 3,
            "keys_unique": not bool(frame.duplicated(KEYS).any()),
            "predictions_positive_finite": bool(np.isfinite(frame["predict"]).all() and frame["predict"].gt(0).all()),
            "state_calendar_mode_expected": bool(frame["state_calendar_mode"].eq(mode).all() and manifest["state_calendar_mode"] == mode),
            "input_hash_expected": manifest["input_sha256"] == input_hash,
            "folds_returncode_zero": all(item["returncode"] == 0 for item in manifest["folds"]),
        }
        if reference_keys is None:
            reference_keys = frame[KEYS].copy()
            reference_actual = frame["actual"].copy()
            scenario_checks["keys_equal_s00"] = True
            scenario_checks["actual_equal_s00"] = True
            scenario_checks["maximum_actual_difference_s00"] = 0.0
        else:
            scenario_checks["keys_equal_s00"] = bool(reference_keys.equals(frame[KEYS]))
            actual_difference = np.abs(reference_actual.to_numpy(float) - frame["actual"].to_numpy(float))
            scenario_checks["actual_equal_s00"] = bool(actual_difference.max(initial=0.0) <= 1.0e-12)
            scenario_checks["maximum_actual_difference_s00"] = float(actual_difference.max(initial=0.0))
        integrity["scenarios"][scenario] = {
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256(prediction_path),
            "input_path": manifest["input_path"],
            "input_sha256": manifest["input_sha256"],
            "checks": scenario_checks,
            "passed": all(value for key, value in scenario_checks.items() if key != "maximum_actual_difference_s00"),
        }
        frames[scenario] = frame
    integrity["passed"] = all(value["passed"] for value in integrity["scenarios"].values())
    return frames, integrity


def add_fixed_annotations(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    station_map = read_selected_station_map()
    station_names = dict(zip(station_map["comid"], station_map["station_name"]))
    original_input = pd.read_parquet(
        RUN / "inputs" / "source_snapshot" / "baseline_reference" / "inputs" / "indata.parquet",
        columns=["comid", "year", "month", "Q_obsv_cfs", "Q_ma_cfs", "CumAreaKm2"],
    ).sort_values(["comid", "year", "month"]).reset_index(drop=True)
    overlay_input = pd.read_parquet(
        RUN / "inputs" / "scenarios" / "provisional_overlay_indata.parquet",
        columns=["comid", "year", "month", "Q_obsv_cfs"],
    ).sort_values(["comid", "year", "month"]).reset_index(drop=True)
    if not original_input[["comid", "year", "month"]].equals(overlay_input[["comid", "year", "month"]]):
        raise ValueError("Original and overlay input keys differ")
    left = original_input["Q_obsv_cfs"].to_numpy(float)
    right = overlay_input["Q_obsv_cfs"].to_numpy(float)
    changed_mask = ~(np.isclose(left, right, rtol=0.0, atol=1.0e-12, equal_nan=True))
    changed_rows = original_input.loc[changed_mask, ["comid", "year", "month", "Q_obsv_cfs"]].copy()
    changed_rows["overlay_q_obsv_cfs"] = overlay_input.loc[changed_mask, "Q_obsv_cfs"].to_numpy()
    conflict_reaches = set(changed_rows["comid"].astype(int))
    gaps = pd.read_csv(REPORT.parent / "state_calendar" / "observed_internal_gaps.csv")
    gap_summary = gaps.groupby("comid", as_index=False).agg(
        gap_event_count=("skipped_forcing_months", "size"),
        skipped_forcing_months_total=("skipped_forcing_months", "sum"),
        max_skipped_forcing_months=("skipped_forcing_months", "max"),
    )
    gap_map = gap_summary.set_index("comid").to_dict("index")
    reach_qma = original_input.groupby("comid")["Q_ma_cfs"].median().to_dict()
    reach_area = original_input.groupby("comid")["CumAreaKm2"].median().to_dict()

    reference = frames["S00_NATIVE"].copy()
    reference["row_number"] = np.arange(len(reference))
    low_rows: set[int] = set()
    high_rows: set[int] = set()
    for _, group in reference.groupby(["comid", "fold_id"], sort=False):
        count = max(1, int(math.ceil(len(group) * 0.25)))
        ordered = group.sort_values(["actual", "year", "month", "row_number"])
        low_rows.update(ordered.head(count)["row_number"].astype(int))
        high_rows.update(ordered.tail(count)["row_number"].astype(int))

    annotated: dict[str, pd.DataFrame] = {}
    for scenario, source in frames.items():
        frame = source.copy()
        frame["station_name"] = frame["comid"].map(station_names).fillna(frame["q_site"].astype(str))
        frame["row_number"] = np.arange(len(frame))
        frame["lowflow"] = frame["row_number"].isin(low_rows)
        frame["highflow"] = frame["row_number"].isin(high_rows)
        frame["conflict_reach"] = frame["comid"].isin(conflict_reaches)
        frame["legacy_target"] = False
        frame["qma_median_cfs"] = frame["comid"].map(reach_qma)
        frame["qma_median_m3s"] = frame["qma_median_cfs"] / CFS_PER_M3S
        frame["cum_area_km2"] = frame["comid"].map(reach_area)
        frame["large_flow_station"] = frame["qma_median_m3s"].ge(LARGE_FLOW_THRESHOLD_M3S)
        frame["headwater_class"] = np.where(frame["group_headwater_area"].ge(0.5), "headwater", "non_headwater")
        frame["max_skipped_forcing_months"] = frame["comid"].map(
            {key: value["max_skipped_forcing_months"] for key, value in gap_map.items()}
        ).fillna(0).astype(int)
        frame["gap_exposure_class"] = pd.cut(
            frame["max_skipped_forcing_months"], bins=[-1, 0, 6, 24, np.inf], labels=["none", "1_to_6", "7_to_24", "over_24"]
        ).astype(str)
        annotated[scenario] = frame

    registry_path = RUN / "inputs" / "source_snapshot" / "experiment_reference" / "canonical_signal_registry.csv"
    registry = pd.read_csv(registry_path, encoding="utf-8-sig")
    membership = registry["legacy_canonical_membership"].astype(str).str.casefold().isin({"true", "1"})
    legacy_reaches = set(pd.to_numeric(registry.loc[membership, "reach_id"], errors="coerce").dropna().astype(int))
    for frame in annotated.values():
        frame["legacy_target"] = frame["comid"].isin(legacy_reaches)
    metadata = {
        "station_map_rows": int(len(station_map)),
        "changed_model_label_rows": int(changed_mask.sum()),
        "changed_model_reaches": sorted(int(value) for value in conflict_reaches),
        "changed_model_reach_count": int(len(conflict_reaches)),
        "legacy_registry_members": int(len(legacy_reaches)),
        "legacy_registry_reaches": sorted(int(value) for value in legacy_reaches),
        "legacy_evaluable_reaches": int(len(legacy_reaches.intersection(set(reference["comid"].unique())))),
        "shijiao_reach": int(station_map.loc[station_map["station_name"].eq("石角站"), "comid"].iloc[0]),
    }
    changed_rows["station_name"] = changed_rows["comid"].map(station_names)
    changed_rows.to_csv(REPORT / "model_label_changes.csv", index=False, encoding="utf-8-sig")
    return annotated, metadata


def build_metric_tables(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    for scenario, frame in frames.items():
        scenario_rows.append({"scenario": scenario, "scenario_label": SCENARIO_LABEL[scenario], **metrics(frame)})
        for fold_id, group in frame.groupby("fold_id", sort=True):
            fold_rows.append({"scenario": scenario, "fold_id": fold_id, **metrics(group)})
        for (comid, station_name), group in frame.groupby(["comid", "station_name"], sort=True):
            row = {
                "scenario": scenario,
                "comid": int(comid),
                "station_name": station_name,
                "folds": int(group["fold_id"].nunique()),
                "conflict_reach": bool(group["conflict_reach"].iloc[0]),
                "legacy_target": bool(group["legacy_target"].iloc[0]),
                "large_flow_station": bool(group["large_flow_station"].iloc[0]),
                "headwater_class": str(group["headwater_class"].iloc[0]),
                "gap_exposure_class": str(group["gap_exposure_class"].iloc[0]),
                "max_skipped_forcing_months": int(group["max_skipped_forcing_months"].iloc[0]),
                "qma_median_cfs": float(group["qma_median_cfs"].iloc[0]),
                "qma_median_m3s": float(group["qma_median_m3s"].iloc[0]),
                **metrics(group),
            }
            low = metrics(group[group["lowflow"]])
            high = metrics(group[group["highflow"]])
            row.update({f"lowflow_{key}": value for key, value in low.items()})
            row.update({f"highflow_{key}": value for key, value in high.items()})
            station_rows.append(row)
        for regime, mask in [("lowflow", frame["lowflow"]), ("middleflow", ~(frame["lowflow"] | frame["highflow"])), ("highflow", frame["highflow"])]:
            flow_rows.append({"scenario": scenario, "flow_regime": regime, **metrics(frame[mask])})
    return pd.DataFrame(scenario_rows), pd.DataFrame(fold_rows), pd.DataFrame(station_rows), pd.DataFrame(flow_rows)


def residual_acf_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, frame in frames.items():
        work = frame.copy()
        work["residual"] = np.log1p(np.maximum(work["predict"], 0.0) / CFS_PER_M3S) - np.log1p(
            np.maximum(work["actual"], 0.0) / CFS_PER_M3S
        )
        work["calendar_index"] = work["year"].astype(int) * 12 + work["month"].astype(int)
        for lag in (1, 3, 6):
            station_values: list[dict[str, Any]] = []
            for (comid, fold_id), group in work.groupby(["comid", "fold_id"], sort=False):
                left = group[["calendar_index", "residual"]].rename(columns={"residual": "residual_now"})
                right = group[["calendar_index", "residual"]].copy()
                right["calendar_index"] = right["calendar_index"] + lag
                right = right.rename(columns={"residual": "residual_lag"})
                paired = left.merge(right, on="calendar_index", how="inner", validate="one_to_one")
                if len(paired) >= 4 and paired["residual_now"].std(ddof=0) > EPS and paired["residual_lag"].std(ddof=0) > EPS:
                    correlation = float(paired[["residual_now", "residual_lag"]].corr().iloc[0, 1])
                    station_values.append({"comid": int(comid), "fold_id": fold_id, "pairs": int(len(paired)), "acf": correlation})
            values = pd.DataFrame(station_values)
            if values.empty:
                rows.append({"scenario": scenario, "lag_months": lag, "station_fold_series": 0, "pairs": 0, "weighted_acf": np.nan, "median_acf": np.nan, "weighted_absolute_acf": np.nan})
                continue
            rows.append({
                "scenario": scenario,
                "lag_months": lag,
                "station_fold_series": int(len(values)),
                "pairs": int(values["pairs"].sum()),
                "weighted_acf": float(np.average(values["acf"], weights=values["pairs"])),
                "median_acf": float(values["acf"].median()),
                "weighted_absolute_acf": float(np.average(values["acf"].abs(), weights=values["pairs"])),
            })
    return pd.DataFrame(rows)


def effect_table(scenario_metrics: pd.DataFrame, station_metrics: pd.DataFrame, acf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = ["nse", "log_nse", "kge", "pbias_pct", "rmse_cfs", "log_rmse", "median_log_bias"]
    station_summary = station_metrics.groupby("scenario", as_index=False).agg(
        median_station_nse=("nse", "median"),
        median_station_log_nse=("log_nse", "median"),
        median_station_kge=("kge", "median"),
        median_station_absolute_pbias_pct=("pbias_pct", lambda values: float(np.median(np.abs(values)))),
        median_station_lowflow_log_rmse=("lowflow_log_rmse", "median"),
        median_station_highflow_log_rmse=("highflow_log_rmse", "median"),
        median_station_absolute_lowflow_log_bias=("lowflow_median_log_bias", lambda values: float(np.median(np.abs(values)))),
    )
    scenario_wide = scenario_metrics.set_index("scenario")
    station_wide = station_summary.set_index("scenario")
    acf_wide = acf.pivot(index="scenario", columns="lag_months", values="weighted_absolute_acf")
    source_values: dict[str, pd.Series] = {metric: scenario_wide[metric] for metric in selected}
    source_values.update({column: station_wide[column] for column in station_wide.columns})
    for lag in (1, 3, 6):
        source_values[f"residual_abs_acf_lag_{lag}"] = acf_wide[lag]
    rows: list[dict[str, Any]] = []
    for metric, values in source_values.items():
        s00, s10, s01, s11 = (float(values.loc[key]) for key in SCENARIOS)
        direction = "higher_is_better" if metric in {"nse", "log_nse", "kge", "median_station_nse", "median_station_log_nse", "median_station_kge"} else "closer_to_zero_or_lower_is_better"
        rows.extend([
            {"metric": metric, "effect": "S10_minus_S00_overlay", "value": s10 - s00, "direction": direction},
            {"metric": metric, "effect": "S01_minus_S00_state_continuity", "value": s01 - s00, "direction": direction},
            {"metric": metric, "effect": "S11_minus_S00_joint", "value": s11 - s00, "direction": direction},
            {"metric": metric, "effect": "interaction_S11_minus_S10_minus_S01_plus_S00", "value": s11 - s10 - s01 + s00, "direction": direction},
        ])
    station_wide_all = station_metrics.pivot(index=["comid", "station_name"], columns="scenario")
    paired_rows: list[dict[str, Any]] = []
    for (comid, station_name), _ in station_wide_all.iterrows():
        base = station_metrics[(station_metrics["comid"].eq(comid)) & (station_metrics["scenario"].eq("S00_NATIVE"))].iloc[0]
        for metric in ["log_nse", "kge", "pbias_pct", "lowflow_log_rmse", "lowflow_median_log_bias", "highflow_log_rmse"]:
            values = {scenario: float(station_metrics[(station_metrics["comid"].eq(comid)) & (station_metrics["scenario"].eq(scenario))][metric].iloc[0]) for scenario in SCENARIOS}
            paired_rows.append({
                "comid": int(comid), "station_name": station_name,
                "conflict_reach": bool(base["conflict_reach"]), "legacy_target": bool(base["legacy_target"]),
                "large_flow_station": bool(base["large_flow_station"]), "headwater_class": base["headwater_class"],
                "gap_exposure_class": base["gap_exposure_class"], "metric": metric,
                "S00": values["S00_NATIVE"], "S10": values["S10"], "S01": values["S01"], "S11": values["S11"],
                "overlay_effect": values["S10"] - values["S00_NATIVE"],
                "state_continuity_effect": values["S01"] - values["S00_NATIVE"],
                "joint_effect": values["S11"] - values["S00_NATIVE"],
                "interaction": values["S11"] - values["S10"] - values["S01"] + values["S00_NATIVE"],
            })
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def segment_table(station_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = {
        "conflict_status": station_metrics["conflict_reach"].map({True: "conflict_reach", False: "non_conflict_reach"}),
        "legacy_status": station_metrics["legacy_target"].map({True: "legacy_target", False: "other_station"}),
        "flow_scale": station_metrics["large_flow_station"].map({True: "qma_ge_100_m3s", False: "qma_lt_100_m3s"}),
        "headwater_status": station_metrics["headwater_class"],
        "gap_exposure": station_metrics["gap_exposure_class"],
    }
    for dimension, labels in dimensions.items():
        work = station_metrics.copy()
        work["segment"] = labels
        for (scenario, segment), group in work.groupby(["scenario", "segment"], sort=True):
            rows.append({
                "dimension": dimension, "segment": segment, "scenario": scenario, "stations": int(len(group)),
                "median_log_nse": float(group["log_nse"].median()),
                "median_kge": float(group["kge"].median()),
                "median_absolute_pbias_pct": float(group["pbias_pct"].abs().median()),
                "median_lowflow_log_rmse": float(group["lowflow_log_rmse"].median()),
                "median_absolute_lowflow_log_bias": float(group["lowflow_median_log_bias"].abs().median()),
                "median_highflow_log_rmse": float(group["highflow_log_rmse"].median()),
            })
    return pd.DataFrame(rows)


def legacy_table(station_metrics: pd.DataFrame) -> pd.DataFrame:
    legacy = station_metrics[station_metrics["legacy_target"]].copy()
    base = legacy[legacy["scenario"].eq("S00_NATIVE")][["comid", "lowflow_log_rmse", "lowflow_median_log_bias", "lowflow_absolute_error_cfs"]].rename(
        columns={
            "lowflow_log_rmse": "s00_lowflow_log_rmse",
            "lowflow_median_log_bias": "s00_lowflow_median_log_bias",
            "lowflow_absolute_error_cfs": "s00_lowflow_absolute_error_cfs",
        }
    )
    legacy = legacy.merge(base, on="comid", how="left", validate="many_to_one")
    legacy["lowflow_log_rmse_improved_vs_s00"] = legacy["lowflow_log_rmse"] < legacy["s00_lowflow_log_rmse"]
    legacy["absolute_lowflow_log_bias_improved_vs_s00"] = legacy["lowflow_median_log_bias"].abs() < legacy["s00_lowflow_median_log_bias"].abs()
    legacy["lowflow_absolute_error_improved_vs_s00"] = legacy["lowflow_absolute_error_cfs"] < legacy["s00_lowflow_absolute_error_cfs"]
    return legacy


def shijiao_table(station_metrics: pd.DataFrame, shijiao_reach: int) -> pd.DataFrame:
    return station_metrics[station_metrics["comid"].eq(shijiao_reach)].copy()


def make_charts(station_metrics: pd.DataFrame, effect: pd.DataFrame, legacy: pd.DataFrame) -> None:
    chart_dir = REPORT / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    order = list(SCENARIOS)
    colors = {"S00_NATIVE": "#315C8C", "S10": "#8FA9C1", "S01": "#C47A2C", "S11": "#E2B36F"}

    summary = station_metrics.groupby("scenario").agg(
        log_nse=("log_nse", "median"), kge=("kge", "median"),
        abs_pbias=("pbias_pct", lambda values: float(np.median(np.abs(values)))),
        lowflow_log_rmse=("lowflow_log_rmse", "median"),
    ).reindex(order)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, column, title, ylabel in zip(
        axes.flat,
        ["log_nse", "kge", "abs_pbias", "lowflow_log_rmse"],
        ["Median station log-NSE", "Median station KGE", "Median station |PBIAS|", "Median station low-flow log-RMSE"],
        ["score", "score", "%", "log1p(m3/s)"],
    ):
        values = summary[column]
        bars = axis.bar(range(len(order)), values, color=[colors[item] for item in order], edgecolor="#24313F", linewidth=0.8)
        axis.set_xticks(range(len(order)), [item.replace("_NATIVE", "") for item in order])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", color="#D8DEE6", linewidth=0.7)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Q72 four-scenario station-median comparison (8,738 OOF rows)", fontsize=14, color="#202833")
    fig.savefig(chart_dir / "scenario_station_medians.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    paired = effect[(effect["metric"].eq("log_nse"))].copy()
    paired = paired.sort_values("state_continuity_effect")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharex=True)
    x_values = np.arange(len(paired))
    for axis in axes:
        axis.scatter(x_values, paired["state_continuity_effect"], s=24, color="#C47A2C", edgecolor="#24313F", linewidth=0.4)
        axis.axhline(0, color="#24313F", linewidth=1)
        axis.set_xlabel("Stations sorted by S01-S00 log-NSE")
        axis.grid(axis="y", color="#D8DEE6", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Paired log-NSE change")
    axes[0].set_title("All stations, full range")
    lower, upper = paired["state_continuity_effect"].quantile([0.02, 0.98])
    padding = max((upper - lower) * 0.15, 0.005)
    axes[1].set_ylim(min(lower - padding, -0.005), max(upper + padding, 0.005))
    axes[1].set_title("Central 96% (focused scale)")
    fig.suptitle("Station-level effect of advancing states across no-Q months", fontsize=14, color="#202833")
    fig.savefig(chart_dir / "state_continuity_station_log_nse_delta.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    legacy_counts = legacy[legacy["scenario"].ne("S00_NATIVE")].groupby("scenario")["absolute_lowflow_log_bias_improved_vs_s00"].agg(["sum", "count"]).reindex(["S10", "S01", "S11"])
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = axis.bar(legacy_counts.index, legacy_counts["sum"], color=[colors[item] for item in legacy_counts.index], edgecolor="#24313F", linewidth=0.8)
    axis.axhline(legacy_counts["count"].iloc[0] / 2, color="#24313F", linestyle="--", linewidth=1, label="Half of evaluable targets")
    axis.set_ylim(0, max(legacy_counts["count"].max(), legacy_counts["sum"].max()) + 2)
    axis.set_ylabel("Legacy targets improved")
    axis.set_title("Legacy low-flow targets with smaller absolute median log bias")
    for bar, value, total in zip(bars, legacy_counts["sum"], legacy_counts["count"]):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}/{int(total)}", ha="center", va="bottom")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.7)
    fig.savefig(chart_dir / "legacy_lowflow_improvement_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    frames, integrity = load_scenarios()
    if not integrity["passed"]:
        (REPORT / "scenario_integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("Scenario integrity gate failed")
    frames, metadata = add_fixed_annotations(frames)
    scenario_metrics, fold_metrics, station_metrics, flow_metrics = build_metric_tables(frames)
    residual_acf = residual_acf_table(frames)
    attribution, station_effects = effect_table(scenario_metrics, station_metrics, residual_acf)
    segments = segment_table(station_metrics)
    legacy = legacy_table(station_metrics)
    shijiao = shijiao_table(station_metrics, metadata["shijiao_reach"])

    combined = pd.concat([frame.assign(scenario=scenario) for scenario, frame in frames.items()], ignore_index=True)
    combined.to_parquet(REPORT / "scenario_oof_predictions.parquet", index=False)
    tables = {
        "scenario_metrics.csv": scenario_metrics,
        "fold_metrics.csv": fold_metrics,
        "station_metrics.csv": station_metrics,
        "flow_regime_metrics.csv": flow_metrics,
        "residual_acf_metrics.csv": residual_acf,
        "paired_effect_attribution.csv": attribution,
        "station_paired_effects.csv": station_effects,
        "segment_metrics.csv": segments,
        "legacy_lowflow_metrics.csv": legacy,
        "shijiao_metrics.csv": shijiao,
    }
    for name, table in tables.items():
        table.to_csv(REPORT / name, index=False, encoding="utf-8-sig")
    make_charts(station_metrics, station_effects, legacy)

    station_summary = station_metrics.groupby("scenario").agg(
        station_count=("comid", "size"),
        median_nse=("nse", "median"),
        median_log_nse=("log_nse", "median"),
        median_kge=("kge", "median"),
        median_absolute_pbias_pct=("pbias_pct", lambda values: float(np.median(np.abs(values)))),
        median_lowflow_log_rmse=("lowflow_log_rmse", "median"),
        median_absolute_lowflow_log_bias=("lowflow_median_log_bias", lambda values: float(np.median(np.abs(values)))),
        median_highflow_log_rmse=("highflow_log_rmse", "median"),
    ).reset_index()
    legacy_summary = legacy.groupby("scenario").agg(
        evaluable_targets=("comid", "size"),
        lowflow_log_rmse_improved=("lowflow_log_rmse_improved_vs_s00", "sum"),
        absolute_lowflow_log_bias_improved=("absolute_lowflow_log_bias_improved_vs_s00", "sum"),
        lowflow_absolute_error_improved=("lowflow_absolute_error_improved_vs_s00", "sum"),
    ).reset_index()
    state_feature_columns = [
        "antecedent_wetness", "sas_storage_mm", "sas_young_fraction", "sas_young_cfs", "sas_old_release_cfs",
        "production_storage_mm", "production_saturation", "production_quick_cfs", "production_base_cfs",
        "routed_quick_cfs", "routed_base_cfs", "ms_slow_large_storage_mm", "ms_slow_large_routed_base_cfs",
    ]
    s00 = frames["S00_NATIVE"]
    s01 = frames["S01"]
    feature_difference = np.abs(s01[state_feature_columns].to_numpy(float) - s00[state_feature_columns].to_numpy(float))
    state_change_summary = {
        "compared_state_columns": state_feature_columns,
        "rows_with_any_state_change_s01_vs_s00": int((feature_difference.max(axis=1) > 1.0e-12).sum()),
        "maximum_state_feature_absolute_change": float(np.nanmax(feature_difference)),
        "gap_exposed_oof_rows": int(s00["max_skipped_forcing_months"].gt(0).sum()),
        "gap_exposed_rows_with_any_state_change": int(((feature_difference.max(axis=1) > 1.0e-12) & s00["max_skipped_forcing_months"].gt(0).to_numpy()).sum()),
    }
    summary = {
        "run_id": RUN.name,
        "runtime": RUNTIME,
        "scenario_integrity_passed": integrity["passed"],
        "metadata": metadata,
        "station_summary": station_summary.to_dict("records"),
        "pooled_summary": scenario_metrics.to_dict("records"),
        "legacy_summary": legacy_summary.to_dict("records"),
        "residual_acf": residual_acf.to_dict("records"),
        "state_change_summary": state_change_summary,
        "interpretation_boundary": "Performance is diagnostic only; it does not adjudicate discharge truth or engineering correctness.",
    }
    integrity["annotation_metadata"] = metadata
    (REPORT / "scenario_integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "diagnostics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
