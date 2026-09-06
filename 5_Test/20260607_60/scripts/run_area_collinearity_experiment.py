from __future__ import annotations

import csv
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260607_60"
SRC30 = ROOT / "5_Test" / "20260607_30"
REPORTS = RUN / "reports"
FIG_DIR = RUN / "figure"
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


def variant_feature_lists(model30, variant: str) -> dict[str, list[str]]:
    fixed = list(model30.FIXED_FEATURES)
    random = list(model30.RANDOM_SLOPE_FEATURES)
    regime = list(model30.REGIME_SLOPE_FEATURES)

    def replace_depth(items: list[str]) -> list[str]:
        return [
            "log_net_depth" if x == "log_basin_net" else "log_threshold_depth" if x == "log_basin_threshold" else x
            for x in items
        ]

    if variant == "area_current":
        pass
    elif variant == "area_depth_only":
        fixed = replace_depth(fixed)
        random = replace_depth(random)
        regime = replace_depth(regime)
    elif variant == "area_flow_only":
        fixed = [x for x in fixed if x != "log_cumarea"]
    else:
        raise ValueError(variant)

    return {
        "FIXED_FEATURES": fixed,
        "RANDOM_SLOPE_FEATURES": random,
        "REGIME_SLOPE_FEATURES": regime,
        "REGIME_GATES": list(model30.REGIME_GATES),
        "SPATIAL_GROUP_FEATURES": list(model30.SPATIAL_GROUP_FEATURES),
        "SPATIAL_GROUP_GATES": list(model30.SPATIAL_GROUP_GATES),
        "PRODUCTION_FEATURES": list(model30.PRODUCTION_FEATURES),
        "MULTISTORE_FEATURES": list(model30.MULTISTORE_FEATURES),
        "HYSTERESIS_FEATURES": list(model30.HYSTERESIS_FEATURES),
    }


def apply_feature_lists(model30, lists: dict[str, list[str]]) -> None:
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


def summarize_metrics(metrics: pd.DataFrame, variant: str, split: str) -> dict[str, object]:
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


def vif_table(frame: pd.DataFrame, variant: str, features: list[str]) -> pd.DataFrame:
    rows = []
    available = [f for f in features if f in frame.columns]
    xdf = frame[available].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xdf = (xdf - xdf.mean()) / xdf.std(ddof=0).replace(0, 1.0)
    x = xdf.to_numpy(dtype=float)
    for j, feat in enumerate(available):
        y = x[:, j]
        others = np.delete(x, j, axis=1)
        if others.shape[1] == 0 or np.std(y) <= EPS:
            r2 = np.nan
            vif = np.nan
        else:
            xx = np.column_stack([np.ones(len(others)), others])
            beta, *_ = np.linalg.lstsq(xx, y, rcond=None)
            pred = xx @ beta
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            ss_res = float(np.sum((y - pred) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else np.nan
            vif = 1.0 / max(1.0 - r2, EPS) if pd.notna(r2) else np.nan
        rows.append({"variant": variant, "feature": feat, "r2_against_other_area_features": r2, "vif": vif})
    return pd.DataFrame(rows)


def correlation_table(frame: pd.DataFrame, variant: str, features: list[str]) -> pd.DataFrame:
    available = [f for f in features if f in frame.columns]
    corr = frame[available].replace([np.inf, -np.inf], np.nan).fillna(0.0).corr()
    rows = []
    for i, a in enumerate(available):
        for b in available[i + 1 :]:
            rows.append({"variant": variant, "feature_a": a, "feature_b": b, "corr": float(corr.loc[a, b])})
    return pd.DataFrame(rows)


def selected_area_features_for_variant(variant: str) -> list[str]:
    if variant == "area_current":
        return ["log_cumarea", "log_basin_net", "log_basin_threshold", "log_qcalc", "log_qma"]
    if variant == "area_depth_only":
        return ["log_cumarea", "log_net_depth", "log_threshold_depth", "log_qcalc", "log_qma"]
    if variant == "area_flow_only":
        return ["log_basin_net", "log_basin_threshold", "log_qcalc", "log_qma"]
    raise ValueError(variant)


def run_variant(model30, base_featured: pd.DataFrame, hp: dict[str, float], variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lists = variant_feature_lists(model30, variant)
    apply_feature_lists(model30, lists)
    featured = add_depth_features(base_featured)
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
    featured["predict_log"] = model30.predict_log(featured, beta, stations, mean, std)
    featured["predict"] = np.exp(np.clip(featured["predict_log"], -20, 20))
    train_metrics = metric_by_station(model30, featured[featured["year"] <= 2018], featured.loc[featured["year"] <= 2018, "predict"].to_numpy(), variant, "calibration")
    val_metrics = metric_by_station(model30, validation, featured.loc[featured["year"] >= 2019, "predict"].to_numpy(), variant, "validation")

    summary = pd.DataFrame(
        [
            summarize_metrics(train_metrics, variant, "calibration"),
            summarize_metrics(val_metrics, variant, "validation"),
        ]
    )

    by_station = pd.concat([train_metrics, val_metrics], ignore_index=True)
    return summary, by_station, featured


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model30 = load_model30()
    hp = parse_hyperparams()
    observed = model30.load_observed_panel()
    base_featured = model30.prepare_design(
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
    base_featured = add_depth_features(base_featured)

    variants = ["area_current", "area_depth_only", "area_flow_only"]
    summaries = []
    all_metrics = []
    all_vif = []
    all_corr = []
    original_lists = variant_feature_lists(model30, "area_current")
    for variant in variants:
        summary, metrics, featured = run_variant(model30, base_featured, hp, variant)
        summaries.append(summary)
        all_metrics.append(metrics)
        train = featured[featured["year"] <= 2018].copy()
        area_features = selected_area_features_for_variant(variant)
        all_vif.append(vif_table(train, variant, area_features))
        all_corr.append(correlation_table(train, variant, area_features))
    apply_feature_lists(model30, original_lists)

    summary_df = pd.concat(summaries, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    vif_df = pd.concat(all_vif, ignore_index=True)
    corr_df = pd.concat(all_corr, ignore_index=True)

    val = metrics_df[metrics_df["split"] == "validation"].copy()
    current = val[val["variant"] == "area_current"].set_index("q_site")
    deltas = []
    for variant in ["area_depth_only", "area_flow_only"]:
        other = val[val["variant"] == variant].set_index("q_site")
        common = current.index.intersection(other.index)
        for site in common:
            deltas.append(
                {
                    "variant": variant,
                    "q_site": site,
                    "delta_NSE_log": other.at[site, "NSE_log"] - current.at[site, "NSE_log"],
                    "delta_KGE": other.at[site, "KGE_2012"] - current.at[site, "KGE_2012"],
                    "delta_abs_PBIAS": other.at[site, "abs_PBIAS"] - current.at[site, "abs_PBIAS"],
                    "current_good": bool(current.at[site, "good"]),
                    "variant_good": bool(other.at[site, "good"]),
                }
            )
    delta_df = pd.DataFrame(deltas)
    delta_summary = (
        delta_df.groupby("variant")
        .agg(
            stations=("q_site", "count"),
            median_delta_NSE_log=("delta_NSE_log", "median"),
            median_delta_KGE=("delta_KGE", "median"),
            median_delta_abs_PBIAS=("delta_abs_PBIAS", "median"),
            improved_NSElog_count=("delta_NSE_log", lambda s: int((s > 0).sum())),
            improved_KGE_count=("delta_KGE", lambda s: int((s > 0).sum())),
            improved_absPBIAS_count=("delta_abs_PBIAS", lambda s: int((s < 0).sum())),
            gained_good=("variant_good", lambda s: 0),
        )
        .reset_index()
    )
    for idx, row in delta_summary.iterrows():
        part = delta_df[delta_df["variant"] == row["variant"]]
        delta_summary.at[idx, "gained_good"] = int((~part["current_good"] & part["variant_good"]).sum())
        delta_summary.at[idx, "lost_good"] = int((part["current_good"] & ~part["variant_good"]).sum())

    summary_df.to_csv(REPORTS / "area_variant_validation_comparison.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(REPORTS / "area_variant_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    vif_df.to_csv(REPORTS / "area_collinearity_vif.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(REPORTS / "area_feature_correlation.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(REPORTS / "area_station_delta_vs_20260607_30.csv", index=False, encoding="utf-8-sig")
    delta_summary.to_csv(REPORTS / "area_variant_delta_summary.csv", index=False, encoding="utf-8-sig")

    val_summary = summary_df[summary_df["split"] == "validation"].set_index("variant")
    current_good = int(val_summary.at["area_current", "good_validation_station_count"])
    depth_good = int(val_summary.at["area_depth_only", "good_validation_station_count"])
    flow_good = int(val_summary.at["area_flow_only", "good_validation_station_count"])
    current_kge = float(val_summary.at["area_current", "median_KGE"])
    depth_kge = float(val_summary.at["area_depth_only", "median_KGE"])
    flow_kge = float(val_summary.at["area_flow_only", "median_KGE"])
    depth_viable = depth_good >= current_good - 2 and depth_kge >= current_kge - 0.015
    flow_viable = flow_good >= current_good - 2 and flow_kge >= current_kge - 0.015
    if depth_viable and flow_viable:
        decision = "both_depth_only_and_flow_only_are_viable_tightening_candidates"
    elif depth_viable:
        decision = "depth_only_is_viable_tightening_candidate"
    elif flow_viable:
        decision = "flow_only_is_viable_tightening_candidate"
    else:
        decision = "current_area_structure_predictively_stronger_but_interpretation_risky"

    report = f"""# 20260607_60 Area Collinearity Tightening Experiment

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

This run tests the user's concern that `_30` repeats area information through `log_cumarea`, `log_basin_net_cfs`, `log_basin_threshold_cfs`, `Q_calc`, and `Q_ma`.

## Variants

- `area_current`: current `_30` formulation.
- `area_depth_only`: replaces `log_basin_net` and `log_basin_threshold` with depth-only `log_net_depth` and `log_threshold_depth`; keeps `log_cumarea`.
- `area_flow_only`: keeps cfs forcing terms but removes `log_cumarea`.

All other hydrologic states, Bayesian/MAP priors, station random effects, and strict train/validation splits remain the same.

## Strict Validation Result

```text
area_current good={current_good}, median_KGE={current_kge:.6f}
area_depth_only good={depth_good}, median_KGE={depth_kge:.6f}
area_flow_only good={flow_good}, median_KGE={flow_kge:.6f}
```

## Decision

`{decision}`

This is a rigor experiment, not a search for a small performance gain. If a simpler area formulation is close to `_30`, it should be preferred in the refined branch because it reduces parameter interpretation risk.
"""
    (REPORTS / "area_collinearity_report.md").write_text(report, encoding="utf-8")

    method = """# 20260607_60 Method Record

## Run Type

Independent rigor-tightening experiment targeting area duplication and collinearity.

## Split

- Final fit: 2006-2018
- Strict validation: 2019-2022

No 2019-2022 observations are used for feature construction, fitting, or variant selection.

## Interpretation Rule

This run does not attempt to make a larger model. It tests whether a simpler area/depth representation can retain performance while reducing ambiguity in parameter interpretation.
"""
    (REPORTS / "model_equation_and_method.md").write_text(method, encoding="utf-8")

    reflection = f"""# 20260607_60 Reflection

This folder directly tests the highest-risk interpretability issue from the user's review: area is repeated inside several flow-like predictors.

Decision from this run: `{decision}`.

The next step should depend on this result:

- If depth-only is close, continue the rigor branch with depth-only as the cleaner base.
- If current remains much stronger, keep `_30` as the predictive reference but mark its area coefficients as non-interpretable.
- In either case, do not add new features until the module ablation experiment checks whether SAS, production, multistore, and hysteresis are redundantly explaining the same water signal.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260607_60"},
        {"item": "run_type", "value": "area_collinearity_tightening_experiment"},
        {"item": "parent_model", "value": "20260607_30"},
        {"item": "variants", "value": "|".join(variants)},
        {"item": "model_fitted", "value": "true"},
        {"item": "strict_validation", "value": "2019-2022"},
        {"item": "decision", "value": decision},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260607_60

This folder tests whether `20260607_30` repeats area information too aggressively.

Variants:

- `area_current`
- `area_depth_only`
- `area_flow_only`

Key outputs:

- `reports/area_variant_validation_comparison.csv`
- `reports/area_collinearity_vif.csv`
- `reports/area_feature_correlation.csv`
- `reports/area_station_delta_vs_20260607_30.csv`
- `reports/area_collinearity_report.md`
- `reports/reflection_summary.md`
- `reports/model_equation_and_method.md`
- `reports/run_manifest.csv`
"""
    (RUN / "README_20260607_60.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
