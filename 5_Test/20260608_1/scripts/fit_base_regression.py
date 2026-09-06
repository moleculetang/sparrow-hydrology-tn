from __future__ import annotations

import csv
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260608_1"
SRC30 = RUN
SRC62 = RUN
SRC63 = RUN
SRC64 = RUN
SRC65 = RUN
REPORTS = RUN / "reports" / "intermediate" / "base_regression"
EPS = 1.0e-6


VARIANTS = [
    "median_exp_eta",
    "global_lognormal_mean",
    "global_duan_smearing",
    "month_duan_smearing",
    "station_duan_smearing",
    "station_month_duan_smearing",
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


def build_base_eta(model30, et62, slow63, storage64, gate65, hp: dict[str, float]) -> tuple[pd.DataFrame, np.ndarray, dict[str, list[str]], list[str]]:
    observed = model30.load_observed_panel()
    original = capture_original_lists(model30)
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
    featured = gate65.apply_gate_form(featured, "current_overlap")
    featured = storage64.recompute_dependent_features(featured, hp, "clipped_delta")
    featured = model30.prepare_design(featured)
    featured = et62.add_depth_features(featured)
    featured = slow63.add_slow_indices(featured)
    train = featured[featured["year"] <= 2018].copy()
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
    eta = model30.predict_log(featured, beta, stations, mean, std)
    featured["eta_log"] = eta
    featured["log_residual_train_only"] = np.nan
    train_mask = featured["year"] <= 2018
    featured.loc[train_mask, "log_residual_train_only"] = featured.loc[train_mask, "log_obs"] - featured.loc[train_mask, "eta_log"]
    apply_lists(model30, original)
    return featured, eta, lists, stations


def smearing_factors(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    train = frame[frame["year"] <= 2018].copy()
    resid = train["log_residual_train_only"].dropna().to_numpy(dtype=float)
    global_duan = float(np.mean(np.exp(resid)))
    sigma = float(np.std(resid, ddof=1))
    global_lognormal = float(np.exp(0.5 * sigma * sigma))
    month_factor = train.groupby("month")["log_residual_train_only"].apply(lambda s: float(np.mean(np.exp(s.dropna())))).to_dict()
    station_factor = train.groupby("q_site")["log_residual_train_only"].apply(lambda s: float(np.mean(np.exp(s.dropna())))).to_dict()
    station_month_factor = (
        train.groupby(["q_site", "month"])["log_residual_train_only"].apply(lambda s: float(np.mean(np.exp(s.dropna())))).to_dict()
    )
    rows = [
        {"factor_scope": "global_duan", "q_site": "", "month": "", "factor": global_duan, "n": int(len(resid))},
        {"factor_scope": "global_lognormal", "q_site": "", "month": "", "factor": global_lognormal, "n": int(len(resid))},
    ]
    for m, factor in month_factor.items():
        n = int(((train["month"] == m) & train["log_residual_train_only"].notna()).sum())
        rows.append({"factor_scope": "month_duan", "q_site": "", "month": int(m), "factor": factor, "n": n})
    for site, factor in station_factor.items():
        n = int(((train["q_site"] == site) & train["log_residual_train_only"].notna()).sum())
        rows.append({"factor_scope": "station_duan", "q_site": site, "month": "", "factor": factor, "n": n})
    for (site, month), factor in station_month_factor.items():
        n = int(((train["q_site"] == site) & (train["month"] == month) & train["log_residual_train_only"].notna()).sum())
        rows.append({"factor_scope": "station_month_duan", "q_site": site, "month": int(month), "factor": factor, "n": n})
    factors = {
        "global_duan": global_duan,
        "global_lognormal": global_lognormal,
        "month": month_factor,
        "station": station_factor,
        "station_month": station_month_factor,
    }
    return factors, pd.DataFrame(rows)


def predictions_for_variant(frame: pd.DataFrame, variant: str, factors: dict[str, object]) -> np.ndarray:
    base = np.exp(np.clip(frame["eta_log"].to_numpy(dtype=float), -20, 20))
    if variant == "median_exp_eta":
        return base
    if variant == "global_lognormal_mean":
        return base * float(factors["global_lognormal"])
    if variant == "global_duan_smearing":
        return base * float(factors["global_duan"])
    if variant == "month_duan_smearing":
        fac = frame["month"].map(factors["month"]).fillna(float(factors["global_duan"])).to_numpy(dtype=float)
        return base * fac
    if variant == "station_duan_smearing":
        fac = frame["q_site"].map(factors["station"]).fillna(float(factors["global_duan"])).to_numpy(dtype=float)
        return base * fac
    if variant == "station_month_duan_smearing":
        lookup = factors["station_month"]
        global_duan = float(factors["global_duan"])
        station_lookup = factors["station"]
        month_lookup = factors["month"]
        fac = []
        for site, month in zip(frame["q_site"], frame["month"]):
            fac.append(
                float(
                    lookup.get(
                        (site, int(month)),
                        station_lookup.get(site, month_lookup.get(int(month), global_duan)),
                    )
                )
            )
        return base * np.array(fac, dtype=float)
    raise ValueError(variant)


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


def summarize(metrics: pd.DataFrame, variant: str, split: str) -> dict[str, object]:
    return {
        "variant": variant,
        "split": split,
        "stations": int(len(metrics)),
        "median_NSE_raw": float(metrics["NSE_raw"].median()),
        "median_NSE_log": float(metrics["NSE_log"].median()),
        "median_KGE": float(metrics["KGE_2012"].median()),
        "median_abs_PBIAS_pct": float(metrics["abs_PBIAS"].median()),
        "median_trend_r": float(metrics["trend_r"].median()),
        "median_amplitude_ratio": float(metrics["amplitude_ratio"].median()),
        "good_validation_station_count": int(metrics["good"].sum()) if split == "validation" else "",
    }


def raw_bias_summary(frame: pd.DataFrame, pred: np.ndarray, variant: str) -> dict[str, object]:
    val = frame[frame["year"] >= 2019].copy()
    pred_val = pred[frame["year"].to_numpy(dtype=int) >= 2019]
    obs = val["Q_obsv_cfs"].to_numpy(dtype=float)
    return {
        "variant": variant,
        "validation_total_obs_cfs_month": float(np.sum(obs)),
        "validation_total_pred_cfs_month": float(np.sum(pred_val)),
        "validation_total_PBIAS_pct": float(100.0 * (np.sum(pred_val) - np.sum(obs)) / np.sum(obs)),
        "median_pred_obs_ratio": float(np.median(pred_val / np.maximum(obs, EPS))),
        "mean_pred_obs_ratio": float(np.mean(pred_val / np.maximum(obs, EPS))),
    }


def make_delta(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = metrics_df[metrics_df["split"] == "validation"].copy()
    base = val[val["variant"] == "median_exp_eta"].set_index("q_site")
    rows = []
    for variant, part in val.groupby("variant"):
        if variant == "median_exp_eta":
            continue
        cur = part.set_index("q_site")
        for site in cur.index.intersection(base.index):
            rows.append(
                {
                    "variant": variant,
                    "q_site": site,
                    "delta_NSE_raw": float(cur.at[site, "NSE_raw"] - base.at[site, "NSE_raw"]),
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
                "NSEraw_improved_station_count": int((part["delta_NSE_raw"] > 0).sum()),
                "NSElog_improved_station_count": int((part["delta_NSE_log"] > 0).sum()),
                "KGE_improved_station_count": int((part["delta_KGE"] > 0).sum()),
                "absPBIAS_improved_station_count": int((part["delta_abs_PBIAS"] < 0).sum()),
                "gained_good_station_count": int((~part["base_good"] & part["variant_good"]).sum()),
                "lost_good_station_count": int((part["base_good"] & ~part["variant_good"]).sum()),
                "median_delta_NSEraw": float(part["delta_NSE_raw"].median()),
                "median_delta_NSElog": float(part["delta_NSE_log"].median()),
                "median_delta_KGE": float(part["delta_KGE"].median()),
                "median_delta_absPBIAS": float(part["delta_abs_PBIAS"].median()),
            }
        )
    return delta, pd.DataFrame(summary_rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    model30 = load_module("model30", RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py")
    et62 = load_module("et62", RUN / "scripts" / "components" / "run_et_role_experiment.py")
    slow63 = load_module("slow63", RUN / "scripts" / "components" / "run_slowflow_redundancy_experiment.py")
    storage64 = load_module("storage64", RUN / "scripts" / "components" / "run_storage_timing_experiment.py")
    gate65 = load_module("gate65", RUN / "scripts" / "components" / "run_hysteresis_gate_experiment.py")
    hp = parse_hyperparams()
    frame, eta, lists, stations = build_base_eta(model30, et62, slow63, storage64, gate65, hp)
    factors, factor_df = smearing_factors(frame)

    summaries = []
    metrics = []
    bias_rows = []
    pred_rows = []
    for variant in VARIANTS:
        pred = predictions_for_variant(frame, variant, factors)
        train_mask = frame["year"] <= 2018
        val_mask = frame["year"] >= 2019
        train_metrics = metric_by_station(model30, frame.loc[train_mask], pred[train_mask.to_numpy()], variant, "calibration")
        val_metrics = metric_by_station(model30, frame.loc[val_mask], pred[val_mask.to_numpy()], variant, "validation")
        metrics.append(pd.concat([train_metrics, val_metrics], ignore_index=True))
        summaries.append(pd.DataFrame([summarize(train_metrics, variant, "calibration"), summarize(val_metrics, variant, "validation")]))
        bias_rows.append(raw_bias_summary(frame, pred, variant))
        pred_part = frame[["q_site", "year", "month", "Q_obsv_cfs", "eta_log"]].copy()
        pred_part["variant"] = variant
        pred_part["predict"] = pred
        pred_rows.append(pred_part)

    summary_df = pd.concat(summaries, ignore_index=True)
    metrics_df = pd.concat(metrics, ignore_index=True)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    delta_df, delta_summary_df = make_delta(metrics_df)
    summary_df.to_csv(REPORTS / "smearing_validation_comparison.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(REPORTS / "smearing_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    factor_df.to_csv(REPORTS / "smearing_factors_train_only.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bias_rows).to_csv(REPORTS / "smearing_raw_bias_summary.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(REPORTS / "smearing_predictions_long.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(REPORTS / "smearing_station_delta_vs_median.csv", index=False, encoding="utf-8-sig")
    delta_summary_df.to_csv(REPORTS / "smearing_delta_summary.csv", index=False, encoding="utf-8-sig")

    val = summary_df[summary_df["split"] == "validation"].copy()
    best = val.sort_values(["good_validation_station_count", "median_KGE", "median_NSE_raw"], ascending=False).iloc[0]
    median = val[val["variant"] == "median_exp_eta"].iloc[0]
    global_duan = val[val["variant"] == "global_duan_smearing"].iloc[0]
    station = val[val["variant"] == "station_duan_smearing"].iloc[0]
    if best["variant"] != "median_exp_eta":
        decision = "smearing_retransformation_improves_raw_validation"
    else:
        decision = "median_exp_eta_empirically_preferred_for_current_metrics"

    report = f"""# 20260608_1 q72 stage Recommended Branch Computation

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

This run checks the review concern that `Q_pred = exp(eta)` from a log-flow model is a conditional median, not a conditional raw-flow mean. The experiment estimates retransformation factors using 2006-2018 training residuals only and applies them to 2019-2022 validation predictions.

## Validation Summary

Best validation variant: `{best['variant']}`.

```text
median_exp_eta good = {int(median['good_validation_station_count'])}
median_exp_eta median_NSEraw = {float(median['median_NSE_raw']):.6f}
median_exp_eta median_NSElog = {float(median['median_NSE_log']):.6f}
median_exp_eta median_KGE = {float(median['median_KGE']):.6f}
median_exp_eta median_abs_PBIAS = {float(median['median_abs_PBIAS_pct']):.6f}

global_duan good = {int(global_duan['good_validation_station_count'])}
global_duan median_NSEraw = {float(global_duan['median_NSE_raw']):.6f}
global_duan median_NSElog = {float(global_duan['median_NSE_log']):.6f}
global_duan median_KGE = {float(global_duan['median_KGE']):.6f}
global_duan median_abs_PBIAS = {float(global_duan['median_abs_PBIAS_pct']):.6f}

station_duan good = {int(station['good_validation_station_count'])}
station_duan median_NSEraw = {float(station['median_NSE_raw']):.6f}
station_duan median_NSElog = {float(station['median_NSE_log']):.6f}
station_duan median_KGE = {float(station['median_KGE']):.6f}
station_duan median_abs_PBIAS = {float(station['median_abs_PBIAS_pct']):.6f}
```

## Decision

`{decision}`

Smearing is mathematically legitimate as a log-model retransformation correction when factors are estimated only from training residuals. Station-level smearing is valid only for known-station temporal prediction and should not be presented as ungauged reach prediction.
"""
    (REPORTS / "smearing_retransformation_report.md").write_text(report, encoding="utf-8")

    reflection = f"""# 20260608_1 q72 stage Reflection

Decision: `{decision}`.

This experiment does not refit or retune the hydrologic equation. It only checks whether raw-flow evaluation should use the lognormal mean correction or Duan smearing instead of the median `exp(eta)`. All correction factors are estimated from 2006-2018 residuals only, so the 2019-2022 validation period remains hidden.

The next independent rigor check should target MAP penalty scaling and residual-variance interpretation.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    method = """# 20260608_1 q72 stage Method Record

## Run Type

Log-model retransformation and Duan smearing experiment.

## Base

Uses the `_65` current-overlap branch: clean ET, reduced slow-flow stack, current SAS timing, current shoulder-season hysteresis gates.

## Variants

- `median_exp_eta`: raw prediction is `exp(eta)`, the conditional median under lognormal errors.
- `global_lognormal_mean`: multiply by `exp(sigma_train^2 / 2)`.
- `global_duan_smearing`: multiply by `mean_train(exp(residual))`.
- `month_duan_smearing`: month-specific training smearing factors.
- `station_duan_smearing`: station-specific training smearing factors.
- `station_month_duan_smearing`: station-month training smearing factors with fallback to station/month/global.

## Split

Model fit and smearing factors use 2006-2018 observations. Strict validation uses 2019-2022 observations only after fitting and factor estimation.
"""
    (REPORTS / "model_equation_and_method.md").write_text(method, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260608_1"},
        {"item": "run_type", "value": "log_retransformation_smearing_experiment"},
        {"item": "parent_model", "value": "20260607_30"},
        {"item": "base_branch", "value": "20260607_65_gate_current_overlap"},
        {"item": "variants", "value": "|".join(VARIANTS)},
        {"item": "model_fitted", "value": "true"},
        {"item": "strict_validation", "value": "2019-2022"},
        {"item": "smearing_factor_fit_period", "value": "2006-2018"},
        {"item": "decision", "value": decision},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260608_1 q72 stage

This folder tests log-model retransformation and Duan smearing corrections.

Key outputs:

- `reports/smearing_validation_comparison.csv`
- `reports/smearing_delta_summary.csv`
- `reports/smearing_factors_train_only.csv`
- `reports/smearing_raw_bias_summary.csv`
- `reports/smearing_retransformation_report.md`
- `reports/reflection_summary.md`
- `reports/model_equation_and_method.md`
- `reports/run_manifest.csv`
"""
    (RUN / "README_20260608_1.md").write_text(readme, encoding="utf-8")

    print(summary_df[summary_df["split"] == "validation"].to_string(index=False))
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
