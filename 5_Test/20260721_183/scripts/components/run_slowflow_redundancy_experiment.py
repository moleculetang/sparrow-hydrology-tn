from __future__ import annotations

import csv
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[2]
SRC30 = ROOT / "5_Test" / "20260607_30"
SRC62 = ROOT / "5_Test" / "20260607_62"
REPORTS = RUN / "reports"


SLOW_SAS = {"log_sas_old_release", "sas_old_fraction", "sas_old_dry_release"}
SLOW_PRODUCTION = {"log_production_base", "log_routed_base", "production_dry_base_release"}
SLOW_MULTISTORE = {
    "log_ms_headwater_flash_base",
    "log_ms_large_slow_base",
    "log_ms_large_slow_storage",
    "log_ms_reservoir_buffer_base",
    "log_ms_dry_headwater_base",
}
SLOW_HYSTERESIS = {"hys_late_storage_release", "hys_late_storage_excess", "hys_dry_recharge_memory"}
SLOW_ALL = SLOW_SAS | SLOW_PRODUCTION | SLOW_MULTISTORE | SLOW_HYSTERESIS
MERGED_SLOW = ["log_slow_release_index", "slow_dry_release_index", "slow_late_release_index", "slow_storage_index"]


VARIANTS = [
    {
        "variant": "slow_full_clean_et_reference",
        "remove_features": set(),
        "add_merged": False,
        "description": "clean ET branch with all SAS/production/multistore/hysteresis slow-flow terms",
    },
    {
        "variant": "slow_no_sas_old",
        "remove_features": SLOW_SAS,
        "add_merged": False,
        "description": "remove SAS old-water release terms, keep production and multistore slow-flow terms",
    },
    {
        "variant": "slow_no_production_base",
        "remove_features": SLOW_PRODUCTION,
        "add_merged": False,
        "description": "remove production base/routed-base slow terms, keep SAS old release",
    },
    {
        "variant": "slow_no_multistore_hys_slow",
        "remove_features": SLOW_MULTISTORE | SLOW_HYSTERESIS,
        "add_merged": False,
        "description": "keep core SAS/production slow terms but remove multistore and hysteresis slow-release basis functions",
    },
    {
        "variant": "slow_merged_index",
        "remove_features": SLOW_ALL,
        "add_merged": True,
        "description": "replace individual slow-flow terms with merged release/storage indices",
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


def replace_depth(items: list[str]) -> list[str]:
    return [
        "log_net_depth" if x == "log_basin_net" else "log_threshold_depth" if x == "log_basin_threshold" else x
        for x in items
    ]


def unique(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def lists_for_variant(original: dict[str, list[str]], cfg: dict[str, object]) -> dict[str, list[str]]:
    remove = set(cfg["remove_features"])
    lists = {k: list(v) for k, v in original.items()}
    for key in ["FIXED_FEATURES", "RANDOM_SLOPE_FEATURES", "REGIME_SLOPE_FEATURES"]:
        lists[key] = replace_depth(lists[key])
        lists[key] = [x for x in lists[key] if x not in remove]
    for key in ["SPATIAL_GROUP_FEATURES", "PRODUCTION_FEATURES", "MULTISTORE_FEATURES", "HYSTERESIS_FEATURES"]:
        lists[key] = [x for x in lists[key] if x not in remove]
    if cfg["add_merged"]:
        lists["FIXED_FEATURES"] = unique(lists["FIXED_FEATURES"] + MERGED_SLOW)
        lists["RANDOM_SLOPE_FEATURES"] = unique(lists["RANDOM_SLOPE_FEATURES"] + ["log_slow_release_index", "slow_dry_release_index"])
        lists["REGIME_SLOPE_FEATURES"] = unique(lists["REGIME_SLOPE_FEATURES"] + ["log_slow_release_index"])
    fixed = set(lists["FIXED_FEATURES"])
    lists["SPATIAL_GROUP_FEATURES"] = [x for x in lists["SPATIAL_GROUP_FEATURES"] if x in fixed]
    lists["RANDOM_SLOPE_FEATURES"] = [x for x in lists["RANDOM_SLOPE_FEATURES"] if x in fixed]
    lists["REGIME_SLOPE_FEATURES"] = [x for x in lists["REGIME_SLOPE_FEATURES"] if x in fixed]
    return lists


def apply_lists(model30, lists: dict[str, list[str]]) -> None:
    for k, v in lists.items():
        setattr(model30, k, v)


def add_slow_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    slow_release = (
        out["sas_old_release_cfs"].fillna(0).clip(lower=0)
        + out["routed_base_cfs"].fillna(0).clip(lower=0)
        + out["production_base_cfs"].fillna(0).clip(lower=0)
    )
    storage = out["sas_storage_scaled"].fillna(0).clip(lower=0) + out["production_storage_mm"].fillna(0).clip(lower=0) / 240.0
    dry = np.maximum(-out["antecedent_wetness"].fillna(0), 0)
    out["log_slow_release_index"] = np.log1p(slow_release)
    out["slow_dry_release_index"] = out["log_slow_release_index"] * dry
    out["slow_late_release_index"] = out["log_slow_release_index"] * out["late_recession_gate"].fillna(0)
    out["slow_storage_index"] = storage
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


def run_variant(model30, et62, original: dict[str, list[str]], observed: pd.DataFrame, hp: dict[str, float], cfg: dict[str, object]):
    lists = lists_for_variant(original, cfg)
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
    featured = model30.prepare_design(et62.add_hydrologic_features_et_variant(model30, observed, hp, et_cfg))
    featured = et62.add_depth_features(featured)
    featured = add_slow_indices(featured)
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
    train_metrics = metric_by_station(model30, featured.loc[train_mask], pred[train_mask.to_numpy()], str(cfg["variant"]), "calibration")
    val_metrics = metric_by_station(model30, validation, pred[val_mask.to_numpy()], str(cfg["variant"]), "validation")
    complexity = {
        "fixed_feature_count": len(lists["FIXED_FEATURES"]),
        "station_intercept_count": len(stations),
        "station_random_slope_count": len(lists["RANDOM_SLOPE_FEATURES"]) * len(stations),
        "regime_random_slope_count": len(lists["REGIME_SLOPE_FEATURES"]) * len(lists["REGIME_GATES"]) * len(stations),
        "total_parameter_count": len(beta),
    }
    summary = pd.DataFrame([summarize(train_metrics, str(cfg["variant"]), "calibration", complexity), summarize(val_metrics, str(cfg["variant"]), "validation", complexity)])
    return summary, pd.concat([train_metrics, val_metrics], ignore_index=True), lists


def make_delta(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = metrics_df[metrics_df["split"] == "validation"].copy()
    base = val[val["variant"] == "slow_full_clean_et_reference"].set_index("q_site")
    rows = []
    for variant, part in val.groupby("variant"):
        if variant == "slow_full_clean_et_reference":
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
    hp = parse_hyperparams()
    observed = model30.load_observed_panel()
    original = capture_original_lists(model30)

    summaries = []
    metrics = []
    inventory = []
    for cfg in VARIANTS:
        summary, by_station, lists = run_variant(model30, et62, original, observed, hp, cfg)
        summaries.append(summary)
        metrics.append(by_station)
        inventory.append(
            {
                "variant": cfg["variant"],
                "description": cfg["description"],
                "removed_features": "|".join(sorted(cfg["remove_features"])),
                "added_merged_features": "|".join(MERGED_SLOW if cfg["add_merged"] else []),
                "fixed_feature_count": len(lists["FIXED_FEATURES"]),
                "random_slope_features": "|".join(lists["RANDOM_SLOPE_FEATURES"]),
                "regime_slope_features": "|".join(lists["REGIME_SLOPE_FEATURES"]),
            }
        )
    apply_lists(model30, original)

    summary_df = pd.concat(summaries, ignore_index=True)
    metrics_df = pd.concat(metrics, ignore_index=True)
    delta_df, delta_summary_df = make_delta(metrics_df)
    summary_df.to_csv(REPORTS / "slowflow_redundancy_validation_comparison.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(REPORTS / "slowflow_redundancy_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(REPORTS / "slowflow_station_delta_vs_full.csv", index=False, encoding="utf-8-sig")
    delta_summary_df.to_csv(REPORTS / "slowflow_variant_delta_summary.csv", index=False, encoding="utf-8-sig")
    write_csv(
        REPORTS / "slowflow_feature_inventory.csv",
        inventory,
        ["variant", "description", "removed_features", "added_merged_features", "fixed_feature_count", "random_slope_features", "regime_slope_features"],
    )

    val = summary_df[summary_df["split"] == "validation"].copy()
    best = val.sort_values(["good_validation_station_count", "median_KGE", "median_NSE_log"], ascending=False).iloc[0]
    full = val[val["variant"] == "slow_full_clean_et_reference"].iloc[0]
    merged = val[val["variant"] == "slow_merged_index"].iloc[0]
    merged_close = (
        int(full["good_validation_station_count"]) - int(merged["good_validation_station_count"]) <= 2
        and float(full["median_KGE"] - merged["median_KGE"]) <= 0.015
    )
    if best["variant"] != "slow_full_clean_et_reference":
        decision = "reduced_or_merged_slowflow_variant_improves_validation"
    elif merged_close:
        decision = "merged_slowflow_close_enough_to_prefer_for_interpretation"
    else:
        decision = "full_slowflow_stack_retained_but_redundancy_risk_documented"

    report = f"""# 20260607_63 Slow-Flow Redundancy Experiment

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

This run tests whether SAS old release, production base/routed base, multistore slow/base terms, and seasonal hysteresis slow-release terms are redundantly explaining the same low-flow/storage signal.

The working base is the cleaner ET formulation from `20260607_62`: surplus water drives SAS and production, while ET deficit is treated as dry-stress/state information.

## Validation Summary

Best validation variant: `{best['variant']}`.

```text
full_reference good = {int(full['good_validation_station_count'])}
full_reference median_NSElog = {float(full['median_NSE_log']):.6f}
full_reference median_KGE = {float(full['median_KGE']):.6f}
full_reference median_abs_PBIAS = {float(full['median_abs_PBIAS_pct']):.6f}

best_reduced variant = {best['variant']}
best_reduced good = {int(best['good_validation_station_count'])}
best_reduced median_NSElog = {float(best['median_NSE_log']):.6f}
best_reduced median_KGE = {float(best['median_KGE']):.6f}
best_reduced median_abs_PBIAS = {float(best['median_abs_PBIAS_pct']):.6f}

merged_index good = {int(merged['good_validation_station_count'])}
merged_index median_NSElog = {float(merged['median_NSE_log']):.6f}
merged_index median_KGE = {float(merged['median_KGE']):.6f}
merged_index median_abs_PBIAS = {float(merged['median_abs_PBIAS_pct']):.6f}
```

## Decision

`{decision}`

If a reduced or merged slow-flow variant is close to the full stack, the individual slow-flow coefficients should not be interpreted as separate physical reservoirs.
"""
    (REPORTS / "slowflow_redundancy_report.md").write_text(report, encoding="utf-8")

    reflection = f"""# 20260607_63 Reflection

Decision: `{decision}`.

This experiment is a mechanism/rigor check, not a feature search. It uses the same stations, split, and Bayesian/MAP fitting machinery as `_62`, then removes or merges slow-flow modules that may be explaining the same storage-release signal.

The next independent review concern should be storage timing/SAS state definition or hysteresis-gate overlap, depending on whether this run shows the slow-flow stack can be simplified without a meaningful validation loss.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    method = """# 20260607_63 Method Record

## Run Type

SAS/production/multistore slow-flow redundancy experiment.

## Base

Uses `20260607_62` cleaner ET branch (`et_surplus_water_stress_state`) plus the `area_depth_only` feature representation and the full known-station Bayesian/MAP parameter structure.

## Variants

- `slow_full_clean_et_reference`
- `slow_no_sas_old`
- `slow_no_production_base`
- `slow_no_multistore_hys_slow`
- `slow_merged_index`

## Split

Fit uses 2006-2018 observations. Strict validation uses 2019-2022 observations only after fitting.
"""
    (REPORTS / "model_equation_and_method.md").write_text(method, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260607_63"},
        {"item": "run_type", "value": "slowflow_redundancy_experiment"},
        {"item": "parent_model", "value": "20260607_30"},
        {"item": "base_branch", "value": "20260607_62_et_surplus_water_stress_state"},
        {"item": "variants", "value": "|".join(str(v["variant"]) for v in VARIANTS)},
        {"item": "model_fitted", "value": "true"},
        {"item": "strict_validation", "value": "2019-2022"},
        {"item": "decision", "value": decision},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260607_63

This folder tests SAS/production/multistore slow-flow redundancy.

Key outputs:

- `reports/slowflow_redundancy_validation_comparison.csv`
- `reports/slowflow_variant_delta_summary.csv`
- `reports/slowflow_station_delta_vs_full.csv`
- `reports/slowflow_feature_inventory.csv`
- `reports/slowflow_redundancy_report.md`
- `reports/reflection_summary.md`
- `reports/model_equation_and_method.md`
- `reports/run_manifest.csv`
"""
    (RUN / "README_20260607_63.md").write_text(readme, encoding="utf-8")

    print(summary_df[summary_df["split"] == "validation"].to_string(index=False))
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
