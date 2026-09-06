from __future__ import annotations

import csv
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260607_61"
SRC30 = ROOT / "5_Test" / "20260607_30"
REPORTS = RUN / "reports"
EPS = 1.0e-6


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_model30():
    script = SRC30 / "scripts" / "fit_monthly_bayes_seasonal_hysteresis.py"
    spec = importlib.util.spec_from_file_location("model30", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def add_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ppt = out["PPT"].fillna(0).clip(lower=0)
    aet = out["AET"].fillna(0).clip(lower=0)
    pet = out["PET"].fillna(0).clip(lower=0)
    out["net_depth_mm"] = np.maximum(ppt - aet, 0.0)
    out["threshold_depth_mm"] = np.maximum(ppt - 0.8 * pet, 0.0)
    out["log_net_depth"] = np.log1p(out["net_depth_mm"].fillna(0).clip(lower=0))
    out["log_threshold_depth"] = np.log1p(out["threshold_depth_mm"].fillna(0).clip(lower=0))
    return out


BASE_FEATURES = [
    "log_qcalc",
    "log_qma",
    "log_cumarea",
    "log_net_depth",
    "log_threshold_depth",
    "month_sin",
    "month_cos",
    "log_res_lag_self",
    "log_res_lag_down",
    "is_reservoir_reach",
    "downstream_reservoir",
]

ET_S_FEATURES = [
    "aridity",
    "log_aet",
    "log_pet",
    "log_et_deficit",
    "aet_pet_ratio",
    "aet_ppt_ratio",
    "pet_ppt_ratio",
    "log_lag1_aet",
    "log_lag1_et_deficit",
    "antecedent_wetness",
    "et_deficit_wetness",
]

SAS_FEATURES = [
    "log_sas_young",
    "log_sas_old_release",
    "sas_young_fraction",
    "sas_old_fraction",
    "sas_storage_scaled",
    "sas_young_wet_interaction",
    "sas_old_dry_release",
]

PRODUCTION_FEATURES = [
    "wet_quickflow",
    "log_production_quick",
    "log_production_base",
    "log_production_overflow",
    "log_routed_quick",
    "log_routed_base",
    "log_production_highflow_mass",
    "production_saturation",
    "production_saturation_wetness",
    "production_dry_base_release",
]


VARIANTS = [
    ("M0_base_only", ["base"], False, False, False),
    ("M1_base_ET_S", ["base", "et_s"], False, False, False),
    ("M2_base_ET_S_SAS", ["base", "et_s", "sas"], False, False, False),
    ("M3_base_ET_S_production", ["base", "et_s", "production"], False, False, False),
    ("M4_base_ET_S_SAS_production", ["base", "et_s", "sas", "production"], False, False, False),
    ("M5_plus_multistore_spatial", ["base", "et_s", "sas", "production", "multistore"], True, False, False),
    ("M6_plus_hysteresis", ["base", "et_s", "sas", "production", "multistore", "hysteresis"], True, False, False),
    ("M7_plus_station_random_slopes", ["base", "et_s", "sas", "production", "multistore", "hysteresis"], True, True, False),
    ("M8_plus_regime_random_slopes", ["base", "et_s", "sas", "production", "multistore", "hysteresis"], True, True, True),
]


def unique(seq: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def capture_original_lists(model30) -> dict[str, list[str]]:
    return {
        "PRODUCTION_FEATURES": list(model30.PRODUCTION_FEATURES),
        "MULTISTORE_FEATURES": list(model30.MULTISTORE_FEATURES),
        "HYSTERESIS_FEATURES": list(model30.HYSTERESIS_FEATURES),
        "RANDOM_SLOPE_FEATURES": list(model30.RANDOM_SLOPE_FEATURES),
        "REGIME_SLOPE_FEATURES": list(model30.REGIME_SLOPE_FEATURES),
        "REGIME_GATES": list(model30.REGIME_GATES),
        "SPATIAL_GROUP_FEATURES": list(model30.SPATIAL_GROUP_FEATURES),
        "SPATIAL_GROUP_GATES": list(model30.SPATIAL_GROUP_GATES),
    }


def lists_for_variant(original: dict[str, list[str]], modules: list[str], use_spatial: bool, use_station_slopes: bool, use_regime_slopes: bool) -> dict[str, list[str]]:
    fixed: list[str] = []
    if "base" in modules:
        fixed += BASE_FEATURES
    if "et_s" in modules:
        fixed += ET_S_FEATURES
    if "sas" in modules:
        fixed += SAS_FEATURES
    if "production" in modules:
        fixed += PRODUCTION_FEATURES
    if "multistore" in modules:
        fixed += list(original["MULTISTORE_FEATURES"])
    if "hysteresis" in modules:
        fixed += list(original["HYSTERESIS_FEATURES"])
    fixed = unique([f for f in fixed if f])

    spatial_features = [f for f in original["SPATIAL_GROUP_FEATURES"] if f in fixed] if use_spatial else []
    spatial_gates = list(original["SPATIAL_GROUP_GATES"]) if use_spatial and spatial_features else []

    random_features = []
    if use_station_slopes:
        for f in original["RANDOM_SLOPE_FEATURES"]:
            f2 = "log_net_depth" if f == "log_basin_net" else "log_threshold_depth" if f == "log_basin_threshold" else f
            if f2 in fixed:
                random_features.append(f2)

    regime_features = []
    regime_gates = []
    if use_regime_slopes:
        for f in original["REGIME_SLOPE_FEATURES"]:
            f2 = "log_net_depth" if f == "log_basin_net" else "log_threshold_depth" if f == "log_basin_threshold" else f
            if f2 in fixed:
                regime_features.append(f2)
        if regime_features:
            regime_gates = list(original["REGIME_GATES"])

    return {
        "FIXED_FEATURES": fixed,
        "RANDOM_SLOPE_FEATURES": random_features,
        "REGIME_SLOPE_FEATURES": regime_features,
        "REGIME_GATES": regime_gates,
        "SPATIAL_GROUP_FEATURES": spatial_features,
        "SPATIAL_GROUP_GATES": spatial_gates,
        "PRODUCTION_FEATURES": [f for f in original["PRODUCTION_FEATURES"] if f in fixed],
        "MULTISTORE_FEATURES": [f for f in original["MULTISTORE_FEATURES"] if f in fixed],
        "HYSTERESIS_FEATURES": [f for f in original["HYSTERESIS_FEATURES"] if f in fixed],
    }


def apply_lists(model30, lists: dict[str, list[str]]) -> None:
    for k, v in lists.items():
        setattr(model30, k, v)


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
        md["good"] = (
            md["n"] >= 24
            and pd.notna(md["NSE_log"])
            and md["NSE_log"] >= 0.65
            and pd.notna(md["KGE_2012"])
            and md["KGE_2012"] >= 0.50
            and pd.notna(md["abs_PBIAS"])
            and md["abs_PBIAS"] <= 25.0
        )
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


def run_variant(model30, original: dict[str, list[str]], featured: pd.DataFrame, hp: dict[str, float], variant: str, modules: list[str], use_spatial: bool, use_station_slopes: bool, use_regime_slopes: bool):
    lists = lists_for_variant(original, modules, use_spatial, use_station_slopes, use_regime_slopes)
    apply_lists(model30, lists)
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
    train_metrics = metric_by_station(model30, featured.loc[train_mask], pred[train_mask.to_numpy()], variant, "calibration")
    val_metrics = metric_by_station(model30, validation, pred[val_mask.to_numpy()], variant, "validation")
    complexity = {
        "fixed_feature_count": len(lists["FIXED_FEATURES"]),
        "spatial_group_parameter_count": len(lists["SPATIAL_GROUP_FEATURES"]) * len(lists["SPATIAL_GROUP_GATES"]),
        "station_intercept_count": len(stations),
        "station_random_slope_count": len(lists["RANDOM_SLOPE_FEATURES"]) * len(stations),
        "regime_random_slope_count": len(lists["REGIME_SLOPE_FEATURES"]) * len(lists["REGIME_GATES"]) * len(stations),
        "total_parameter_count": len(beta),
    }
    summary = pd.DataFrame([summarize(train_metrics, variant, "calibration", complexity), summarize(val_metrics, variant, "validation", complexity)])
    return summary, pd.concat([train_metrics, val_metrics], ignore_index=True), lists


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    model30 = load_model30()
    hp = parse_hyperparams()
    observed = model30.load_observed_panel()
    featured = model30.prepare_design(
        model30.add_hydrologic_features(
            observed,
            rho=hp["rho"],
            wm=hp["wm"],
            et_gamma=hp["et_gamma"],
            sas_rho=hp["sas_rho"],
            young_k=hp["young_k"],
            storage_scale=hp["storage_scale"],
            prod_capacity=hp["prod_capacity"],
            runoff_gamma=hp["runoff_gamma"],
            quick_rho=hp["quick_rho"],
            base_rho=hp["base_rho"],
            base_release=hp["base_release"],
        )
    )
    featured = add_depth_features(featured)

    original = capture_original_lists(model30)
    summaries = []
    metrics = []
    feature_rows = []
    for variant, modules, use_spatial, use_station_slopes, use_regime_slopes in VARIANTS:
        summary, by_station, lists = run_variant(model30, original, featured, hp, variant, modules, use_spatial, use_station_slopes, use_regime_slopes)
        summaries.append(summary)
        metrics.append(by_station)
        feature_rows.append(
            {
                "variant": variant,
                "modules": "|".join(modules),
                "fixed_features": "|".join(lists["FIXED_FEATURES"]),
                "random_slope_features": "|".join(lists["RANDOM_SLOPE_FEATURES"]),
                "regime_slope_features": "|".join(lists["REGIME_SLOPE_FEATURES"]),
                "spatial_group_features": "|".join(lists["SPATIAL_GROUP_FEATURES"]),
            }
        )
    apply_lists(model30, original)

    summary_df = pd.concat(summaries, ignore_index=True)
    metrics_df = pd.concat(metrics, ignore_index=True)
    val = summary_df[summary_df["split"] == "validation"].copy().reset_index(drop=True)
    increments = []
    for i in range(1, len(val)):
        prev = val.iloc[i - 1]
        cur = val.iloc[i]
        increments.append(
            {
                "from_variant": prev["variant"],
                "to_variant": cur["variant"],
                "delta_fixed_features": int(cur["fixed_feature_count"] - prev["fixed_feature_count"]),
                "delta_total_parameters": int(cur["total_parameter_count"] - prev["total_parameter_count"]),
                "delta_good": int(cur["good_validation_station_count"] - prev["good_validation_station_count"]),
                "delta_NSE_log": float(cur["median_NSE_log"] - prev["median_NSE_log"]),
                "delta_KGE": float(cur["median_KGE"] - prev["median_KGE"]),
                "delta_abs_PBIAS": float(cur["median_abs_PBIAS_pct"] - prev["median_abs_PBIAS_pct"]),
            }
        )
    inc_df = pd.DataFrame(increments)

    # Simple decision labels: strict enough to highlight weak/complex additions.
    decisions = []
    for row in increments:
        if row["delta_KGE"] >= 0.02 or row["delta_good"] >= 5 or row["delta_NSE_log"] >= 0.02:
            decision = "keep_major_signal"
        elif row["delta_KGE"] < -0.01 or row["delta_good"] <= -3:
            decision = "negative_or_destabilizing_signal"
        elif abs(row["delta_KGE"]) < 0.005 and abs(row["delta_NSE_log"]) < 0.005 and abs(row["delta_good"]) < 2:
            decision = "drop_or_merge_candidate_tiny_gain"
        else:
            decision = "mixed_minor_signal"
        decisions.append({**row, "decision": decision})
    decision_df = pd.DataFrame(decisions)

    summary_df.to_csv(REPORTS / "module_ablation_validation_comparison.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(REPORTS / "module_ablation_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    inc_df.to_csv(REPORTS / "module_incremental_gain.csv", index=False, encoding="utf-8-sig")
    decision_df.to_csv(REPORTS / "module_keep_drop_recommendation.csv", index=False, encoding="utf-8-sig")
    write_csv(
        REPORTS / "module_feature_inventory.csv",
        feature_rows,
        ["variant", "modules", "fixed_features", "random_slope_features", "regime_slope_features", "spatial_group_features"],
    )

    best = val.sort_values(["good_validation_station_count", "median_KGE", "median_NSE_log"], ascending=False).iloc[0]
    m8 = val[val["variant"] == "M8_plus_regime_random_slopes"].iloc[0]
    m6 = val[val["variant"] == "M6_plus_hysteresis"].iloc[0]
    random_gain = float(m8["median_KGE"] - m6["median_KGE"])
    random_good_gain = int(m8["good_validation_station_count"] - m6["good_validation_station_count"])
    if best["variant"] == "M8_plus_regime_random_slopes":
        overall_decision = "full_depth_only_branch_best_but_check_complexity"
    elif random_good_gain >= 5 or random_gain >= 0.02:
        overall_decision = "random_slope_layers_material_even_if_not_best"
    else:
        overall_decision = "simpler_module_stack_is_preferred_candidate"

    report = f"""# 20260607_61 Module Ablation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

This run tests whether the `_30`-family modules repeatedly explain the same monthly water signal. It uses the cleaner `area_depth_only` formulation from `20260607_60` so the ablation is not dominated by duplicate area/cfs effects.

## Variants

M0 base only; M1 adds ET/antecedent wetness; M2 adds SAS; M3 adds production; M4 combines SAS+production; M5 adds multistore/spatial response; M6 adds hysteresis; M7 adds station random slopes; M8 adds regime random slopes.

Station random intercepts are retained in all variants because this remains known-station temporal validation.

## Best Validation Variant

```text
best_variant = {best['variant']}
good = {int(best['good_validation_station_count'])}
median_NSElog = {float(best['median_NSE_log']):.6f}
median_KGE = {float(best['median_KGE']):.6f}
median_abs_PBIAS = {float(best['median_abs_PBIAS_pct']):.6f}
```

## Overall Decision

`{overall_decision}`

The key output is not only the best score, but the incremental gain table. Modules with tiny gains and large parameter increases should be merged or dropped in the refined branch.
"""
    (REPORTS / "module_ablation_report.md").write_text(report, encoding="utf-8")

    reflection = f"""# 20260607_61 Reflection

This is the first direct ablation of the `_30` mechanism stack after the terminology and area-collinearity audits.

Overall decision: `{overall_decision}`.

The important next action is to inspect `module_incremental_gain.csv` before adding any new physics. If random slopes or regime slopes carry most of the gain, the model should be reported as known-station prediction rather than ungauged SPARROW. If SAS/production/multistore increments are small, the refined branch should merge redundant water-signal modules before continuing.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    method = """# 20260607_61 Method Record

## Run Type

Module ablation and complexity diagnostic.

## Base

Uses the `area_depth_only` representation from `20260607_60`: `log_net_depth` and `log_threshold_depth` replace cfs basin forcing terms while `log_cumarea` remains explicit.

## Split

- Fit: 2006-2018
- Strict validation: 2019-2022

No 2019-2022 observations are used in fitting or feature selection. Station random intercepts are retained in all variants; station random slopes are introduced only in M7, and regime random slopes only in M8.
"""
    (REPORTS / "model_equation_and_method.md").write_text(method, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260607_61"},
        {"item": "run_type", "value": "module_ablation_M0_to_M8"},
        {"item": "parent_model", "value": "20260607_30"},
        {"item": "area_base", "value": "20260607_60_area_depth_only"},
        {"item": "variants", "value": "|".join(v[0] for v in VARIANTS)},
        {"item": "model_fitted", "value": "true"},
        {"item": "strict_validation", "value": "2019-2022"},
        {"item": "decision", "value": overall_decision},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260607_61

This folder runs the M0-M8 module ablation requested in the rigor plan.

It uses the `area_depth_only` base from `20260607_60` and tests the incremental value of ET/S, SAS, production, multistore/spatial response, hysteresis, station random slopes, and regime random slopes.

Key outputs:

- `reports/module_ablation_validation_comparison.csv`
- `reports/module_incremental_gain.csv`
- `reports/module_keep_drop_recommendation.csv`
- `reports/module_feature_inventory.csv`
- `reports/module_ablation_report.md`
- `reports/reflection_summary.md`
- `reports/model_equation_and_method.md`
- `reports/run_manifest.csv`
"""
    (RUN / "README_20260607_61.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
