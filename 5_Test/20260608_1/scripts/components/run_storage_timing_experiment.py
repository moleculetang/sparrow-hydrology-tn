from __future__ import annotations

import csv
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260607_64"
SRC30 = ROOT / "5_Test" / "20260607_30"
SRC62 = ROOT / "5_Test" / "20260607_62"
SRC63 = ROOT / "5_Test" / "20260607_63"
REPORTS = RUN / "reports"
EPS = 1.0e-6


VARIANTS = [
    {
        "variant": "storage_current_reference",
        "young_state": "current_clipped",
        "sas_reservoir": "current_pre_release",
        "delta_state": "clipped_delta",
    },
    {
        "variant": "storage_lagged_young",
        "young_state": "previous_clipped",
        "sas_reservoir": "current_pre_release",
        "delta_state": "clipped_delta",
    },
    {
        "variant": "storage_standard_linear",
        "young_state": "current_clipped",
        "sas_reservoir": "standard_post_release",
        "delta_state": "clipped_delta",
    },
    {
        "variant": "storage_raw_delta_hysteresis",
        "young_state": "current_clipped",
        "sas_reservoir": "current_pre_release",
        "delta_state": "raw_delta",
    },
    {
        "variant": "storage_combined_timing",
        "young_state": "previous_clipped",
        "sas_reservoir": "standard_post_release",
        "delta_state": "raw_delta",
    },
]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_module(name: str, script: Path):
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model30():
    return load_module("model30", SRC30 / "scripts" / "fit_monthly_bayes_seasonal_hysteresis.py")


def load_et62():
    return load_module("et62", SRC62 / "scripts" / "run_et_role_experiment.py")


def load_slow63():
    return load_module("slow63", SRC63 / "scripts" / "run_slowflow_redundancy_experiment.py")


def parse_hyperparams() -> dict[str, float]:
    manifest = pd.read_csv(SRC30 / "reports" / "run_manifest.csv", encoding="utf-8-sig")
    raw = str(manifest.loc[0, "selected_hyperparameters"])
    out: dict[str, float] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out


def capture_original_lists(model30) -> dict[str, list[str]]:
    return {
        "FIXED_FEATURES": list(model30.FIXED_FEATURES),
        "PRODUCTION_FEATURES": list(model30.PRODUCTION_FEATURES),
        "MULTISTORE_FEATURES": list(model30.MULTISTORE_FEATURES),
        "HYSTERESIS_FEATURES": list(model30.HYSTERESIS_FEATURES),
        "RANDOM_SLOPE_FEATURES": list(model30.RANDOM_SLOPE_FEATURES),
        "REGIME_SLOPE_FEATURES": list(model30.REGIME_SLOPE_FEATURES),
        "REGIME_GATES": list(model30.REGIME_GATES),
        "SPATIAL_GROUP_FEATURES": list(model30.SPATIAL_GROUP_FEATURES),
        "SPATIAL_GROUP_GATES": list(model30.SPATIAL_GROUP_GATES),
    }


def apply_lists(model30, lists: dict[str, list[str]]) -> None:
    for k, v in lists.items():
        setattr(model30, k, v)


def base_slow_reduced_lists(slow63, original: dict[str, list[str]]) -> dict[str, list[str]]:
    cfg = {
        "variant": "slow_no_multistore_hys_slow",
        "remove_features": slow63.SLOW_MULTISTORE | slow63.SLOW_HYSTERESIS,
        "add_merged": False,
    }
    return slow63.lists_for_variant(original, cfg)


def recompute_state(frame: pd.DataFrame, hp: dict[str, float]) -> pd.DataFrame:
    out = frame.sort_values(["comid", "year", "month"]).copy()
    raw = np.zeros(len(out), dtype=float)
    clipped = np.zeros(len(out), dtype=float)
    prev_clipped = np.zeros(len(out), dtype=float)
    raw_delta = np.zeros(len(out), dtype=float)
    clipped_delta = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        last_raw = 0.0
        last_clip = 0.0
        for pos in idx:
            current_raw = hp["rho"] * last_clip + float(out.at[pos, "wet_input"])
            current_clip = float(np.clip(current_raw, -3.0, 3.0))
            raw[pos] = current_raw
            clipped[pos] = current_clip
            prev_clipped[pos] = last_clip
            raw_delta[pos] = current_raw - last_raw
            clipped_delta[pos] = current_clip - last_clip
            last_raw = current_raw
            last_clip = current_clip
    out["antecedent_wetness_raw"] = raw
    out["antecedent_wetness"] = clipped
    out["antecedent_wetness_prev"] = prev_clipped
    out["wetness_delta_raw"] = raw_delta
    out["wetness_delta_clipped"] = clipped_delta
    return out


def recompute_sas(frame: pd.DataFrame, hp: dict[str, float], young_state: str, reservoir_form: str) -> pd.DataFrame:
    out = frame.sort_values(["comid", "year", "month"]).copy()
    days = np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(out["year"], out["month"])])
    seconds = days.astype(float) * 86400.0
    cumarea = out["CumAreaKm2"].fillna(out["CumAreaKm2"].median()).clip(lower=1).to_numpy(dtype=float)
    storage = np.zeros(len(out), dtype=float)
    young_frac_arr = np.zeros(len(out), dtype=float)
    young_cfs = np.zeros(len(out), dtype=float)
    old_cfs = np.zeros(len(out), dtype=float)
    storage_pre = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        last_post = 0.0
        for pos in idx:
            eff_mm = float(out.at[pos, "sas_effective_mm"])
            wet = float(out.at[pos, "antecedent_wetness_prev"] if young_state == "previous_clipped" else out.at[pos, "antecedent_wetness"])
            young_frac = float(np.clip(1.0 / (1.0 + np.exp(-hp["young_k"] * wet)), 0.05, 0.95))
            if reservoir_form == "standard_post_release":
                pre = last_post + (1.0 - young_frac) * eff_mm
                old_release_mm = (1.0 - hp["sas_rho"]) * pre
                last_post = max(pre - old_release_mm, 0.0)
            else:
                pre = hp["sas_rho"] * last_post + (1.0 - young_frac) * eff_mm
                old_release_mm = (1.0 - hp["sas_rho"]) * pre
                last_post = max(pre - old_release_mm, 0.0)
            area = float(out.at[pos, "CumAreaKm2"]) if pd.notna(out.at[pos, "CumAreaKm2"]) else float(np.nanmedian(cumarea))
            factor = area * 1_000_000.0 / 1000.0 / float(seconds[pos]) * 35.3146667
            storage_pre[pos] = pre
            storage[pos] = last_post
            young_frac_arr[pos] = young_frac
            young_cfs[pos] = young_frac * eff_mm * factor
            old_cfs[pos] = old_release_mm * factor
    out["sas_storage_pre_mm"] = storage_pre
    out["sas_storage_mm"] = storage
    out["sas_young_fraction"] = young_frac_arr
    out["sas_young_cfs"] = young_cfs
    out["sas_old_release_cfs"] = old_cfs
    out["sas_old_fraction"] = 1.0 - out["sas_young_fraction"]
    out["sas_storage_scaled"] = out["sas_storage_mm"] / max(hp["storage_scale"], EPS)
    return out


def recompute_dependent_features(frame: pd.DataFrame, hp: dict[str, float], delta_state: str) -> pd.DataFrame:
    out = frame.copy()
    if delta_state == "raw_delta":
        delta = out["wetness_delta_raw"].to_numpy(dtype=float)
    else:
        delta = out["wetness_delta_clipped"].to_numpy(dtype=float)
    state = out["antecedent_wetness"].to_numpy(dtype=float)
    out["wetness_delta"] = delta
    out["wetting_state"] = np.clip(delta, 0.0, 2.0)
    out["drying_state"] = np.clip(-delta, 0.0, 2.0)
    out["sas_young_wet_interaction"] = np.log1p(out["sas_young_cfs"].clip(lower=0.0)) * np.maximum(state, 0.0)
    out["sas_old_dry_release"] = np.log1p(out["sas_old_release_cfs"].clip(lower=0.0)) * np.maximum(-state, 0.0)
    quick_gate = 1.0 / (1.0 + np.exp(-2.0 * state))
    young_gate = out["sas_young_fraction"].to_numpy(dtype=float)
    out["high_flow_regime_gate"] = np.clip(0.55 * quick_gate + 0.45 * young_gate, 0.05, 0.95)
    out["low_flow_regime_gate"] = 1.0 - out["high_flow_regime_gate"]
    out["wet_high_regime_gate"] = out["wet_season_gate"] * out["high_flow_regime_gate"]
    out["wet_low_regime_gate"] = out["wet_season_gate"] * out["low_flow_regime_gate"]
    out["dry_high_regime_gate"] = out["dry_season_gate"] * out["high_flow_regime_gate"]
    out["dry_low_regime_gate"] = out["dry_season_gate"] * out["low_flow_regime_gate"]
    out["hys_early_wetting_quick"] = np.log1p(out["routed_quick_cfs"].clip(lower=0.0)) * out["early_wet_gate"] * out["wetting_state"]
    out["hys_early_threshold_flush"] = np.log1p(out["basin_threshold_cfs"].clip(lower=0.0)) * out["early_wet_gate"] * np.maximum(out["antecedent_wetness"], 0.0)
    out["hys_peak_highflow_pulse"] = np.log1p(out["production_highflow_mass_cfs"].clip(lower=0.0)) * out["peak_rain_gate"] * out["high_flow_regime_gate"]
    out["hys_peak_saturation_flush"] = out["production_saturation_wetness"] * out["peak_rain_gate"]
    out["hys_late_storage_release"] = np.log1p((out["routed_base_cfs"] + out["sas_old_release_cfs"]).clip(lower=0.0)) * out["late_recession_gate"]
    out["hys_late_drying_attenuation"] = np.log1p(out["production_highflow_mass_cfs"].clip(lower=0.0)) * out["late_recession_gate"] * out["drying_state"]
    out["hys_late_storage_excess"] = out["late_recession_gate"] * (
        out["sas_storage_scaled"].fillna(0.0) + out["production_storage_mm"].fillna(0.0) / max(hp["prod_capacity"], EPS)
    )
    out["hys_dry_recharge_memory"] = np.log1p(out["routed_base_cfs"].clip(lower=0.0)) * out["dry_recharge_gate"] * np.maximum(-out["antecedent_wetness"], 0.0)
    return out


def metric_by_station(model30, frame: pd.DataFrame, pred: np.ndarray, variant: str, split: str) -> pd.DataFrame:
    rows = []
    tmp = frame[["q_site", "Q_obsv_cfs"]].copy()
    tmp["predict"] = pred
    for site, part in tmp.groupby("q_site", sort=False):
        md = model30.metric_dict(part["Q_obsv_cfs"].to_numpy(dtype=float), part["predict"].to_numpy(dtype=float))
        md["q_site"] = site
        md["variant"] = variant
        md["split"] = split
        md["abs_PBIAS"] = abs(md["PBIAS_pct"]) if pd.notna(md["PBIAS_pct"]) else np.nan
        md["good"] = md["n"] >= 24 and md["NSE_log"] >= 0.65 and md["KGE_2012"] >= 0.50 and md["abs_PBIAS"] <= 25.0
        rows.append(md)
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame, variant: str, split: str, complexity: dict[str, int]) -> dict[str, object]:
    return {
        "variant": variant,
        "split": split,
        **complexity,
        "stations": int(len(metrics)),
        "median_NSE_raw": float(metrics["NSE_raw"].median()),
        "median_NSE_log": float(metrics["NSE_log"].median()),
        "median_KGE": float(metrics["KGE_2012"].median()),
        "median_abs_PBIAS_pct": float(metrics["abs_PBIAS"].median()),
        "median_trend_r": float(metrics["trend_r"].median()),
        "median_amplitude_ratio": float(metrics["amplitude_ratio"].median()),
        "good_validation_station_count": int(metrics["good"].sum()) if split == "validation" else "",
    }


def run_variant(model30, et62, slow63, original: dict[str, list[str]], observed: pd.DataFrame, hp: dict[str, float], cfg: dict[str, str]):
    lists = base_slow_reduced_lists(slow63, original)
    apply_lists(model30, lists)
    et_cfg = {
        "variant": "et_surplus_water_stress_state",
        "water_for_sas": "surplus",
        "water_for_production": "surplus",
        "wetness": "stress_adjusted",
        "production_demand_fraction": 0.00,
        "stress_interaction": "dimensionless_dry_stress",
        "remove_stress_features": False,
    }
    featured = et62.add_hydrologic_features_et_variant(model30, observed, hp, et_cfg)
    featured = recompute_state(featured, hp)
    featured = recompute_sas(featured, hp, cfg["young_state"], cfg["sas_reservoir"])
    featured = recompute_dependent_features(featured, hp, cfg["delta_state"])
    featured = model30.prepare_design(featured)
    featured = et62.add_depth_features(featured)
    featured = slow63.add_slow_indices(featured)

    train = featured[featured["year"] <= 2018].copy()
    validation = featured[featured["year"] >= 2019].copy()
    stations = sorted(featured["q_site"].astype(str).unique())
    mean, std = model30.standardize_fit(train)
    beta = model30.fit_map_ridge(
        train,
        stations,
        mean,
        std,
        fixed_sigma=hp["fixed_sigma"],
        production_sigma=hp["production_sigma"],
        group_sigma=hp["group_sigma"],
        multistore_sigma=hp["multistore_sigma"],
        hysteresis_sigma=hp["hysteresis_sigma"],
        station_sigma=hp["station_sigma"],
        slope_sigma=hp["slope_sigma"],
        regime_slope_sigma=hp["regime_slope_sigma"],
        anomaly_weight=0.0,
        flow_contrast_weight=hp.get("flow_contrast_weight", 1.0),
    )
    pred_log = model30.predict_log(featured, beta, stations, mean, std)
    pred = np.exp(np.clip(pred_log, -20, 20))
    train_mask = featured["year"] <= 2018
    val_mask = featured["year"] >= 2019
    train_metrics = metric_by_station(model30, featured.loc[train_mask], pred[train_mask.to_numpy()], cfg["variant"], "calibration")
    val_metrics = metric_by_station(model30, validation, pred[val_mask.to_numpy()], cfg["variant"], "validation")
    complexity = {
        "fixed_feature_count": len(lists["FIXED_FEATURES"]),
        "station_intercept_count": len(stations),
        "station_random_slope_count": len(lists["RANDOM_SLOPE_FEATURES"]) * len(stations),
        "regime_random_slope_count": len(lists["REGIME_SLOPE_FEATURES"]) * len(lists["REGIME_GATES"]) * len(stations),
        "total_parameter_count": len(beta),
    }
    state_audit = {
        "variant": cfg["variant"],
        "median_sas_old_release_cfs": float(featured.loc[val_mask, "sas_old_release_cfs"].median()),
        "median_sas_storage_mm": float(featured.loc[val_mask, "sas_storage_mm"].median()),
        "median_wetting_state": float(featured.loc[val_mask, "wetting_state"].median()),
        "p95_wetting_state": float(featured.loc[val_mask, "wetting_state"].quantile(0.95)),
        "median_young_fraction": float(featured.loc[val_mask, "sas_young_fraction"].median()),
    }
    summary = pd.DataFrame([summarize(train_metrics, cfg["variant"], "calibration", complexity), summarize(val_metrics, cfg["variant"], "validation", complexity)])
    return summary, pd.concat([train_metrics, val_metrics], ignore_index=True), state_audit, lists


def make_delta(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = metrics_df[metrics_df["split"] == "validation"].copy()
    base = val[val["variant"] == "storage_current_reference"].set_index("q_site")
    rows = []
    for variant, part in val.groupby("variant"):
        if variant == "storage_current_reference":
            continue
        cur = part.set_index("q_site")
        for site in cur.index.intersection(base.index):
            rows.append(
                {
                    "variant": variant,
                    "q_site": site,
                    "delta_NSE_log": float(cur.at[site, "NSE_log"] - base.at[site, "NSE_log"]),
                    "delta_KGE": float(cur.at[site, "KGE_2012"] - base.at[site, "KGE_2012"]),
                    "delta_abs_PBIAS": float(cur.at[site, "abs_PBIAS"] - base.at[site, "abs_PBIAS"]),
                    "base_good": bool(base.at[site, "good"]),
                    "variant_good": bool(cur.at[site, "good"]),
                }
            )
    delta = pd.DataFrame(rows)
    summary_rows = []
    for variant, part in delta.groupby("variant"):
        summary_rows.append(
            {
                "variant": variant,
                "station_count": int(len(part)),
                "NSElog_improved_station_count": int((part["delta_NSE_log"] > 0).sum()),
                "KGE_improved_station_count": int((part["delta_KGE"] > 0).sum()),
                "absPBIAS_improved_station_count": int((part["delta_abs_PBIAS"] < 0).sum()),
                "gained_good_station_count": int((~part["base_good"] & part["variant_good"]).sum()),
                "lost_good_station_count": int((part["base_good"] & ~part["variant_good"]).sum()),
                "median_delta_NSElog": float(part["delta_NSE_log"].median()),
                "median_delta_KGE": float(part["delta_KGE"].median()),
                "median_delta_absPBIAS": float(part["delta_abs_PBIAS"].median()),
            }
        )
    return delta, pd.DataFrame(summary_rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    model30 = load_model30()
    et62 = load_et62()
    slow63 = load_slow63()
    hp = parse_hyperparams()
    observed = model30.load_observed_panel()
    original = capture_original_lists(model30)

    summaries = []
    metrics = []
    audits = []
    inventory = []
    for cfg in VARIANTS:
        summary, by_station, state_audit, lists = run_variant(model30, et62, slow63, original, observed, hp, cfg)
        summaries.append(summary)
        metrics.append(by_station)
        audits.append(state_audit)
        inventory.append(
            {
                "variant": cfg["variant"],
                "young_state": cfg["young_state"],
                "sas_reservoir": cfg["sas_reservoir"],
                "delta_state": cfg["delta_state"],
                "fixed_feature_count": len(lists["FIXED_FEATURES"]),
                "random_slope_features": "|".join(lists["RANDOM_SLOPE_FEATURES"]),
                "regime_slope_features": "|".join(lists["REGIME_SLOPE_FEATURES"]),
            }
        )
    apply_lists(model30, original)

    summary_df = pd.concat(summaries, ignore_index=True)
    metrics_df = pd.concat(metrics, ignore_index=True)
    delta_df, delta_summary_df = make_delta(metrics_df)
    summary_df.to_csv(REPORTS / "storage_timing_validation_comparison.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(REPORTS / "storage_timing_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(REPORTS / "storage_state_audit.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(REPORTS / "storage_station_delta_vs_current.csv", index=False, encoding="utf-8-sig")
    delta_summary_df.to_csv(REPORTS / "storage_variant_delta_summary.csv", index=False, encoding="utf-8-sig")
    write_csv(
        REPORTS / "storage_timing_feature_inventory.csv",
        inventory,
        ["variant", "young_state", "sas_reservoir", "delta_state", "fixed_feature_count", "random_slope_features", "regime_slope_features"],
    )

    val = summary_df[summary_df["split"] == "validation"].copy()
    best = val.sort_values(["good_validation_station_count", "median_KGE", "median_NSE_log"], ascending=False).iloc[0]
    cur = val[val["variant"] == "storage_current_reference"].iloc[0]
    combined = val[val["variant"] == "storage_combined_timing"].iloc[0]
    if best["variant"] != "storage_current_reference":
        decision = "storage_timing_change_improves_validation"
    elif int(cur["good_validation_station_count"]) - int(combined["good_validation_station_count"]) <= 2 and float(cur["median_KGE"] - combined["median_KGE"]) <= 0.015:
        decision = "stricter_storage_timing_close_but_not_better"
    else:
        decision = "current_storage_timing_empirically_preferred"

    report = f"""# 20260607_64 Storage Timing Experiment

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

This run tests the review concern that the SAS/storage timing definition is ambiguous. It separates three issues: whether young-water partitioning should use current-month or previous-month wetness; whether the SAS old-water store should use the current pre-release recurrence or a standard linear-reservoir post-release recurrence; and whether hysteresis should use clipped or raw wetness change.

The base is the cleaner `_63` branch: `20260607_62` clean ET plus `slow_no_multistore_hys_slow`.

## Validation Summary

Best validation variant: `{best['variant']}`.

```text
current_reference good = {int(cur['good_validation_station_count'])}
current_reference median_NSElog = {float(cur['median_NSE_log']):.6f}
current_reference median_KGE = {float(cur['median_KGE']):.6f}
current_reference median_abs_PBIAS = {float(cur['median_abs_PBIAS_pct']):.6f}

combined_timing good = {int(combined['good_validation_station_count'])}
combined_timing median_NSElog = {float(combined['median_NSE_log']):.6f}
combined_timing median_KGE = {float(combined['median_KGE']):.6f}
combined_timing median_abs_PBIAS = {float(combined['median_abs_PBIAS_pct']):.6f}
```

## Decision

`{decision}`

If stricter timing is close or better, the model manual should adopt the clearer pre/post-release notation even if the predictive gain is small. If it is worse, keep the current empirical recurrence but explicitly name it as an empirical storage index rather than a strict physical reservoir.
"""
    (REPORTS / "storage_timing_report.md").write_text(report, encoding="utf-8")

    reflection = f"""# 20260607_64 Reflection

Decision: `{decision}`.

This experiment directly targets SAS/storage timing rather than changing the data, stations, validation split, or Bayesian fitting machinery. It uses the simplified slow-flow branch from `_63`, so any difference mainly reflects young-water timing, storage recurrence, or clipped-vs-raw hysteresis state.

The next independent rigor check should be hysteresis gate overlap if storage timing does not produce a major improvement, or adoption of the best stricter timing branch if it does.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    method = """# 20260607_64 Method Record

## Run Type

SAS/storage timing rigor experiment.

## Base

Uses `20260607_62` clean ET and the `20260607_63` reduced slow-flow branch (`slow_no_multistore_hys_slow`).

## Variants

- `storage_current_reference`: current clipped state, current SAS recurrence, clipped DeltaS.
- `storage_lagged_young`: previous clipped state controls young-water fraction.
- `storage_standard_linear`: standard linear-reservoir post-release storage.
- `storage_raw_delta_hysteresis`: raw, unclipped DeltaS drives wetting/drying hysteresis.
- `storage_combined_timing`: previous clipped state, standard linear reservoir, and raw DeltaS together.

## Split

Fit uses 2006-2018 observations. Strict validation uses 2019-2022 observations only after fitting.
"""
    (REPORTS / "model_equation_and_method.md").write_text(method, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260607_64"},
        {"item": "run_type", "value": "storage_timing_experiment"},
        {"item": "parent_model", "value": "20260607_30"},
        {"item": "base_branch", "value": "20260607_63_slow_no_multistore_hys_slow"},
        {"item": "variants", "value": "|".join(v["variant"] for v in VARIANTS)},
        {"item": "model_fitted", "value": "true"},
        {"item": "strict_validation", "value": "2019-2022"},
        {"item": "decision", "value": decision},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260607_64

This folder tests SAS/storage timing rigor.

Key outputs:

- `reports/storage_timing_validation_comparison.csv`
- `reports/storage_variant_delta_summary.csv`
- `reports/storage_state_audit.csv`
- `reports/storage_timing_feature_inventory.csv`
- `reports/storage_timing_report.md`
- `reports/reflection_summary.md`
- `reports/model_equation_and_method.md`
- `reports/run_manifest.csv`
"""
    (RUN / "README_20260607_64.md").write_text(readme, encoding="utf-8")

    print(summary_df[summary_df["split"] == "validation"].to_string(index=False))
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
