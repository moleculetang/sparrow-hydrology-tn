from __future__ import annotations

import csv
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260607_65"
SRC30 = ROOT / "5_Test" / "20260607_30"
SRC62 = ROOT / "5_Test" / "20260607_62"
SRC63 = ROOT / "5_Test" / "20260607_63"
SRC64 = ROOT / "5_Test" / "20260607_64"
REPORTS = RUN / "reports"


VARIANTS = [
    {"variant": "gate_current_overlap", "gate_form": "current_overlap"},
    {"variant": "gate_nonoverlap_calendar", "gate_form": "nonoverlap_calendar"},
    {"variant": "gate_smooth_partition", "gate_form": "smooth_partition"},
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


def apply_gate_form(frame: pd.DataFrame, gate_form: str) -> pd.DataFrame:
    out = frame.copy()
    m = out["month"].astype(int)
    if gate_form == "current_overlap":
        out["early_wet_gate"] = m.between(2, 5).astype(float)
        out["peak_rain_gate"] = m.between(5, 7).astype(float)
        out["late_recession_gate"] = m.between(8, 11).astype(float)
        out["dry_recharge_gate"] = ((m >= 12) | (m <= 2)).astype(float)
    elif gate_form == "nonoverlap_calendar":
        out["dry_recharge_gate"] = ((m == 12) | (m == 1)).astype(float)
        out["early_wet_gate"] = m.between(2, 4).astype(float)
        out["peak_rain_gate"] = m.between(5, 7).astype(float)
        out["late_recession_gate"] = m.between(8, 11).astype(float)
    elif gate_form == "smooth_partition":
        weights = {
            1: (0.0, 0.0, 0.0, 1.0),
            2: (0.65, 0.0, 0.0, 0.35),
            3: (1.0, 0.0, 0.0, 0.0),
            4: (0.70, 0.30, 0.0, 0.0),
            5: (0.25, 0.75, 0.0, 0.0),
            6: (0.0, 1.0, 0.0, 0.0),
            7: (0.0, 0.75, 0.25, 0.0),
            8: (0.0, 0.25, 0.75, 0.0),
            9: (0.0, 0.0, 1.0, 0.0),
            10: (0.0, 0.0, 1.0, 0.0),
            11: (0.0, 0.0, 0.65, 0.35),
            12: (0.0, 0.0, 0.0, 1.0),
        }
        arr = np.array([weights[int(mm)] for mm in m], dtype=float)
        out["early_wet_gate"] = arr[:, 0]
        out["peak_rain_gate"] = arr[:, 1]
        out["late_recession_gate"] = arr[:, 2]
        out["dry_recharge_gate"] = arr[:, 3]
    else:
        raise ValueError(gate_form)
    out["hysteresis_gate_sum"] = out["early_wet_gate"] + out["peak_rain_gate"] + out["late_recession_gate"] + out["dry_recharge_gate"]
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


def gate_audit(frame: pd.DataFrame, variant: str) -> dict[str, object]:
    months = frame[["month", "early_wet_gate", "peak_rain_gate", "late_recession_gate", "dry_recharge_gate", "hysteresis_gate_sum"]].drop_duplicates().sort_values("month")
    return {
        "variant": variant,
        "months_with_gate_sum_gt_1": int((months["hysteresis_gate_sum"] > 1.000001).sum()),
        "months_with_gate_sum_lt_1": int((months["hysteresis_gate_sum"] < 0.999999).sum()),
        "max_gate_sum": float(months["hysteresis_gate_sum"].max()),
        "min_gate_sum": float(months["hysteresis_gate_sum"].min()),
        "gate_table": ";".join(
            f"{int(r.month)}:{r.early_wet_gate:.2f},{r.peak_rain_gate:.2f},{r.late_recession_gate:.2f},{r.dry_recharge_gate:.2f}"
            for r in months.itertuples(index=False)
        ),
    }


def run_variant(model30, et62, slow63, storage64, original: dict[str, list[str]], observed: pd.DataFrame, hp: dict[str, float], cfg: dict[str, str]):
    lists = storage64.base_slow_reduced_lists(slow63, original)
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
    featured = storage64.recompute_state(featured, hp)
    featured = storage64.recompute_sas(featured, hp, "current_clipped", "current_pre_release")
    featured = apply_gate_form(featured, cfg["gate_form"])
    featured = storage64.recompute_dependent_features(featured, hp, "clipped_delta")
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
    summary = pd.DataFrame([summarize(train_metrics, cfg["variant"], "calibration", complexity), summarize(val_metrics, cfg["variant"], "validation", complexity)])
    return summary, pd.concat([train_metrics, val_metrics], ignore_index=True), gate_audit(featured, cfg["variant"]), lists


def make_delta(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = metrics_df[metrics_df["split"] == "validation"].copy()
    base = val[val["variant"] == "gate_current_overlap"].set_index("q_site")
    rows = []
    for variant, part in val.groupby("variant"):
        if variant == "gate_current_overlap":
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
    model30 = load_module("model30", SRC30 / "scripts" / "fit_monthly_bayes_seasonal_hysteresis.py")
    et62 = load_module("et62", SRC62 / "scripts" / "run_et_role_experiment.py")
    slow63 = load_module("slow63", SRC63 / "scripts" / "run_slowflow_redundancy_experiment.py")
    storage64 = load_module("storage64", SRC64 / "scripts" / "run_storage_timing_experiment.py")
    hp = parse_hyperparams()
    observed = model30.load_observed_panel()
    original = capture_original_lists(model30)
    summaries = []
    metrics = []
    audits = []
    inventory = []
    for cfg in VARIANTS:
        summary, by_station, audit, lists = run_variant(model30, et62, slow63, storage64, original, observed, hp, cfg)
        summaries.append(summary)
        metrics.append(by_station)
        audits.append(audit)
        inventory.append(
            {
                "variant": cfg["variant"],
                "gate_form": cfg["gate_form"],
                "fixed_feature_count": len(lists["FIXED_FEATURES"]),
                "hysteresis_features": "|".join(lists["HYSTERESIS_FEATURES"]),
            }
        )
    apply_lists(model30, original)

    summary_df = pd.concat(summaries, ignore_index=True)
    metrics_df = pd.concat(metrics, ignore_index=True)
    delta_df, delta_summary_df = make_delta(metrics_df)
    summary_df.to_csv(REPORTS / "hysteresis_gate_validation_comparison.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(REPORTS / "hysteresis_gate_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(REPORTS / "hysteresis_gate_station_delta_vs_current.csv", index=False, encoding="utf-8-sig")
    delta_summary_df.to_csv(REPORTS / "hysteresis_gate_delta_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(REPORTS / "hysteresis_gate_audit.csv", index=False, encoding="utf-8-sig")
    write_csv(REPORTS / "hysteresis_gate_feature_inventory.csv", inventory, ["variant", "gate_form", "fixed_feature_count", "hysteresis_features"])

    val = summary_df[summary_df["split"] == "validation"].copy()
    best = val.sort_values(["good_validation_station_count", "median_KGE", "median_NSE_log"], ascending=False).iloc[0]
    cur = val[val["variant"] == "gate_current_overlap"].iloc[0]
    smooth = val[val["variant"] == "gate_smooth_partition"].iloc[0]
    if best["variant"] != "gate_current_overlap":
        decision = "nonoverlap_or_smooth_gate_improves_validation"
    elif int(cur["good_validation_station_count"]) - int(smooth["good_validation_station_count"]) <= 2 and float(cur["median_KGE"] - smooth["median_KGE"]) <= 0.015:
        decision = "smooth_gate_close_but_current_overlap_best"
    else:
        decision = "current_overlap_empirically_preferred_but_must_be_documented"

    report = f"""# 20260607_65 Hysteresis Gate Experiment

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

This run tests whether overlapping hysteresis calendar gates accidentally double-count seasonal phase effects. Current gates overlap in February and May. Alternatives are a non-overlap calendar split and a smooth partition with gate weights summing to 1.

The base is the cleaner `_64` current-reference branch: clean ET, reduced slow-flow stack, current SAS timing.

## Validation Summary

Best validation variant: `{best['variant']}`.

```text
current_overlap good = {int(cur['good_validation_station_count'])}
current_overlap median_NSElog = {float(cur['median_NSE_log']):.6f}
current_overlap median_KGE = {float(cur['median_KGE']):.6f}
current_overlap median_abs_PBIAS = {float(cur['median_abs_PBIAS_pct']):.6f}

smooth_partition good = {int(smooth['good_validation_station_count'])}
smooth_partition median_NSElog = {float(smooth['median_NSE_log']):.6f}
smooth_partition median_KGE = {float(smooth['median_KGE']):.6f}
smooth_partition median_abs_PBIAS = {float(smooth['median_abs_PBIAS_pct']):.6f}
```

## Decision

`{decision}`

If a non-overlap or smooth gate is close or better, it is easier to explain physically because every month has a controlled seasonal weight rather than unintended double membership.
"""
    (REPORTS / "hysteresis_gate_report.md").write_text(report, encoding="utf-8")

    reflection = f"""# 20260607_65 Reflection

Decision: `{decision}`.

This experiment directly tests hysteresis gate overlap while keeping the station set, split, clean ET branch, slow-flow simplification, and Bayesian/MAP fitting machinery fixed. The gate audit records whether each variant double-counts months or partitions seasonal phase cleanly.

The next independent rigor check should target MAP penalty scaling/retransformation if gate overlap does not produce a clear model replacement.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    method = """# 20260607_65 Method Record

## Run Type

Hysteresis seasonal-gate overlap experiment.

## Base

Uses the `_64` current-reference branch: clean ET from `_62`, slow-flow simplification from `_63`, and current SAS storage timing.

## Variants

- `gate_current_overlap`: Feb belongs to dry+early, May belongs to early+peak.
- `gate_nonoverlap_calendar`: dry Dec-Jan, early Feb-Apr, peak May-Jul, late Aug-Nov.
- `gate_smooth_partition`: month-specific seasonal weights with gate sums equal to 1.

## Split

Fit uses 2006-2018 observations. Strict validation uses 2019-2022 observations only after fitting.
"""
    (REPORTS / "model_equation_and_method.md").write_text(method, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260607_65"},
        {"item": "run_type", "value": "hysteresis_gate_overlap_experiment"},
        {"item": "parent_model", "value": "20260607_30"},
        {"item": "base_branch", "value": "20260607_64_storage_current_reference"},
        {"item": "variants", "value": "|".join(v["variant"] for v in VARIANTS)},
        {"item": "model_fitted", "value": "true"},
        {"item": "strict_validation", "value": "2019-2022"},
        {"item": "decision", "value": decision},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260607_65

This folder tests hysteresis gate overlap.

Key outputs:

- `reports/hysteresis_gate_validation_comparison.csv`
- `reports/hysteresis_gate_delta_summary.csv`
- `reports/hysteresis_gate_audit.csv`
- `reports/hysteresis_gate_report.md`
- `reports/reflection_summary.md`
- `reports/model_equation_and_method.md`
- `reports/run_manifest.csv`
"""
    (RUN / "README_20260607_65.md").write_text(readme, encoding="utf-8")

    print(summary_df[summary_df["split"] == "validation"].to_string(index=False))
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
