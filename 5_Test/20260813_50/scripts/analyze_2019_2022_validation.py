from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "outputs" / "fit_2006_2018_eval_2019_2022" / "validation_predictions_2019_2022.parquet"
OOF_PATH = ROOT / "reference_baseline" / "P1" / "q72_three_fold_oof_predictions.parquet"
INPUT_PATH = ROOT / "inputs" / "parent_indata.parquet"
REPORT = ROOT / "reports"
FIG = ROOT / "figures"
EPS = 1e-12


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    obs = frame["actual"].to_numpy(float)
    pred = frame["predict"].to_numpy(float)
    keep = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[keep], pred[keep]
    lo, lp = np.log(obs), np.log(pred)
    raw_sst = np.sum((obs - obs.mean()) ** 2)
    log_sst = np.sum((lo - lo.mean()) ** 2)
    r = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {
        "n": int(len(obs)),
        "stations": int(frame.loc[keep, "q_site"].nunique()),
        "raw_nse": float(1 - np.sum((pred - obs) ** 2) / raw_sst) if raw_sst > 0 else np.nan,
        "log_nse": float(1 - np.sum((lp - lo) ** 2) / log_sst) if log_sst > 0 else np.nan,
        "kge_2012": float(kge),
        "pbias_pct": float(100 * np.sum(pred - obs) / np.sum(obs)),
        "rmse_cfs": float(np.sqrt(np.mean((pred - obs) ** 2))),
        "log_rmse": float(np.sqrt(np.mean((lp - lo) ** 2))),
        "correlation": float(r),
        "variability_ratio": float(alpha),
        "mean_ratio": float(beta),
    }


def metric_rows(frame: pd.DataFrame, period: str, group: str, values) -> list[dict[str, object]]:
    rows = []
    for value, part in frame.groupby(values, sort=True):
        rows.append({"period": period, "group": group, "value": str(value), **metrics(part)})
    return rows


def station_table(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    rows = []
    for station, part in frame.groupby("q_site", sort=True):
        rows.append({
            "period": period,
            "q_site": station,
            "mean_observed_cfs": float(part.actual.mean()),
            **metrics(part),
        })
    return pd.DataFrame(rows)


def residual_acf(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    x = frame.copy()
    x["date"] = pd.to_datetime(dict(year=x.year.astype(int), month=x.month.astype(int), day=1))
    x["residual"] = np.log(x.predict) - np.log(x.actual)
    rows = []
    for lag in [1, 3, 6]:
        pairs = []
        station_values = []
        for station, part in x.groupby("q_site", sort=False):
            part = part.sort_values("date")[["date", "residual"]]
            shifted = part.copy()
            shifted["date"] = shifted["date"] + pd.DateOffset(months=lag)
            merged = part.merge(shifted, on="date", suffixes=("_now", "_lag"))
            if len(merged) >= 4 and merged.residual_now.std() > 0 and merged.residual_lag.std() > 0:
                corr = float(merged.residual_now.corr(merged.residual_lag))
                station_values.append(corr)
                pairs.append(len(merged))
        rows.append({
            "period": period,
            "lag_months": lag,
            "stations": len(station_values),
            "pairs": int(np.sum(pairs)),
            "weighted_mean_acf": float(np.average(station_values, weights=pairs)) if pairs else np.nan,
            "median_station_acf": float(np.median(station_values)) if station_values else np.nan,
        })
    return pd.DataFrame(rows)


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in [
        ("svg", {}), ("pdf", {}), ("tiff", {"dpi": 600}), ("png", {"dpi": 300})
    ]:
        fig.savefig(FIG / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)


def make_main_figure(validation: pd.DataFrame, annual: pd.DataFrame, oof_metric: dict,
                     deciles: pd.DataFrame, station_metrics: pd.DataFrame) -> None:
    style()
    blue, orange, teal, grey = "#3B6FB6", "#D97732", "#258A8A", "#777777"
    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.0], height_ratios=[0.92, 1.08])

    panel_axes = []
    ax = fig.add_subplot(grid[0, 0])
    panel_axes.append(ax)
    years = annual.year.astype(int).to_numpy()
    ax.plot(years, annual.raw_nse, marker="o", color=blue, lw=1.5, label="Raw NSE")
    ax.plot(years, annual.log_nse, marker="s", color=orange, lw=1.5, label="Log-NSE")
    ax.axhline(oof_metric["raw_nse"], color=blue, ls="--", lw=0.8, alpha=0.65)
    ax.axhline(oof_metric["log_nse"], color=orange, ls="--", lw=0.8, alpha=0.65)
    ax.axhline(0.8, color=grey, ls=":", lw=0.8)
    ax.set(xticks=years, ylim=(min(0.75, annual[["raw_nse", "log_nse"]].min().min() - 0.02), 1.0),
           ylabel="Efficiency", title="Annual temporal extrapolation")
    ax.legend(loc="lower left", ncol=2)
    ax.text(0.99, 0.03, "Dashed: pooled 2012-2018 OOF", transform=ax.transAxes,
            ha="right", va="bottom", color=grey, fontsize=6)

    ax = fig.add_subplot(grid[0, 1])
    panel_axes.append(ax)
    x = np.log10(validation.actual.to_numpy(float))
    y = np.log10(validation.predict.to_numpy(float))
    hb = ax.hexbin(x, y, gridsize=45, mincnt=1, bins="log", cmap="Blues", linewidths=0)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, color="#222222", lw=0.9, ls="--")
    ax.set(xlim=lim, ylim=lim, xlabel=r"Observed $\log_{10}(Q)$", ylabel=r"Predicted $\log_{10}(Q)$",
           title="All positive station-months")
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("log count", fontsize=6.5)

    ax = fig.add_subplot(grid[1, 0])
    panel_axes.append(ax)
    ax.fill_between(deciles.decile, deciles.q25_log_ratio, deciles.q75_log_ratio,
                    color=teal, alpha=0.18, linewidth=0)
    ax.plot(deciles.decile, deciles.median_log_ratio, marker="o", color=teal, lw=1.5)
    ax.axhline(0, color="#222222", lw=0.8, ls="--")
    ax.set(xticks=np.arange(1, 11), xlabel="Observed-flow decile", ylabel="log(predicted / observed)",
           title="Bias by observed-flow regime")

    ax = fig.add_subplot(grid[1, 1])
    panel_axes.append(ax)
    periods = ["2012-2018 OOF", "2019-2022"]
    positions = [1, 2]
    vals = [station_metrics.loc[station_metrics.period.eq(p), "log_rmse"].dropna().to_numpy() for p in periods]
    bp = ax.boxplot(vals, positions=positions, widths=0.48, patch_artist=True, showfliers=False,
                    medianprops={"color": "#222222", "lw": 1.1},
                    whiskerprops={"lw": 0.8}, capprops={"lw": 0.8}, boxprops={"lw": 0.8})
    for patch, color in zip(bp["boxes"], [blue, orange]):
        patch.set_facecolor(color); patch.set_alpha(0.42)
    rng = np.random.default_rng(2026081350)
    for pos, arr, color in zip(positions, vals, [blue, orange]):
        ax.scatter(pos + rng.normal(0, 0.045, len(arr)), arr, s=5, color=color, alpha=0.35, edgecolors="none")
    ax.set(xticks=positions, xticklabels=["2012-2018\nOOF", "2019-2022\nextrapolation"],
           ylabel="Station log-RMSE", title="Station-level transportability")

    for label, ax in zip("abcd", panel_axes):
        ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8,
                va="top", ha="left")
    save_figure(fig, "figure_1_temporal_extrapolation")
    plt.close(fig)


def make_station_figure(validation: pd.DataFrame) -> pd.DataFrame:
    style()
    orange = "#D97732"
    station_metrics = station_table(validation, "2019-2022").copy()
    station_metrics["minimum_nse"] = station_metrics[["raw_nse", "log_nse"]].min(axis=1)
    # This is an explicitly labelled representative-success panel. Eligibility
    # is fixed before ranking so station choice remains deterministic/auditable.
    eligible = station_metrics.loc[
        station_metrics.mean_observed_cfs.ge(5000)
        & station_metrics.raw_nse.ge(0.85)
        & station_metrics.log_nse.ge(0.85)
    ].copy()
    eligible = eligible.sort_values(
        ["minimum_nse", "mean_observed_cfs", "q_site"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    selected_table = eligible.head(8).copy()
    selected = selected_table.q_site.tolist()
    station_ids = {station: f"S{i + 1}" for i, station in enumerate(selected)}
    source = validation.loc[validation.q_site.isin(selected), ["q_site", "year", "month", "actual", "predict"]].copy()
    source["display_id"] = source["q_site"].map(station_ids)
    source["date"] = pd.to_datetime(dict(year=source.year.astype(int), month=source.month.astype(int), day=1))
    source.to_csv(REPORT / "figure_2_representative_station_source_data.csv", index=False, encoding="utf-8-sig")
    selected_table.insert(0, "display_id", selected_table.q_site.map(station_ids))
    selected_table["selection_rule"] = "mean Q >= 5000 cfs; raw NSE >= 0.85; log-NSE >= 0.85; ranked by min(raw, log NSE)"
    selected_table.to_csv(REPORT / "figure_2_station_label_map.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.2), sharex=True, constrained_layout=True)
    for label, ax, station in zip("abcdefgh", axes.flat, selected):
        part = source.loc[source.q_site.eq(station)].sort_values("date")
        row = selected_table.loc[selected_table.q_site.eq(station)].iloc[0]
        ax.plot(part.date, part.actual, color="#222222", lw=1.2, label="Observed")
        ax.plot(part.date, part.predict, color=orange, lw=1.1, label="Predicted")
        ax.set_yscale("log")
        ax.set_title(f"{station}  |  NSE={row.raw_nse:.2f}, log-NSE={row.log_nse:.2f}", fontsize=7.2)
        ax.set_ylabel("Monthly Q (cfs, log scale)")
        ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8,
                va="top", ha="left")
    axes[0, 0].legend(loc="upper right", ncol=2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    save_figure(fig, "figure_2_large_station_hydrographs")
    plt.close(fig)
    return source


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    val = pd.read_parquet(PRED_PATH)
    oof = pd.read_parquet(OOF_PATH)
    val = val.loc[val.actual.gt(0) & val.predict.gt(0)].copy()
    oof = oof.loc[oof.actual.gt(0) & oof.predict.gt(0)].copy()

    pooled_val = metrics(val)
    pooled_oof = metrics(oof)
    annual_rows = []
    for year, part in val.groupby("year", sort=True):
        annual_rows.append({"year": int(year), **metrics(part)})
    annual = pd.DataFrame(annual_rows)

    summary_rows = [
        {"period": "2012-2018 OOF", "group": "pooled", "value": "all", **pooled_oof},
        {"period": "2019-2022", "group": "pooled", "value": "all", **pooled_val},
        *metric_rows(val, "2019-2022", "year", "year"),
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT / "temporal_extrapolation_metrics.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(REPORT / "annual_metrics_2019_2022.csv", index=False, encoding="utf-8-sig")

    station_oof = station_table(oof, "2012-2018 OOF")
    station_val = station_table(val, "2019-2022")
    stations = pd.concat([station_oof, station_val], ignore_index=True)
    stations.to_csv(REPORT / "station_metrics.csv", index=False, encoding="utf-8-sig")
    common = sorted(set(station_oof.q_site) & set(station_val.q_site))
    common_summary = pd.DataFrame([
        {"period": "2012-2018 OOF", **metrics(oof.loc[oof.q_site.isin(common)])},
        {"period": "2019-2022", **metrics(val.loc[val.q_site.isin(common)])},
    ])
    common_summary.to_csv(REPORT / "common_station_sensitivity.csv", index=False, encoding="utf-8-sig")

    training = pd.read_parquet(INPUT_PATH, columns=["q_site", "year", "Q_obsv_cfs"])
    q = training.loc[training.year.le(2018) & training.Q_obsv_cfs.gt(0)].groupby("q_site").Q_obsv_cfs.quantile([0.25, 0.75]).unstack()
    q.columns = ["train_q25", "train_q75"]
    val = val.merge(q, left_on="q_site", right_index=True, how="left", validate="many_to_one")
    val["flow_regime"] = np.where(val.actual.le(val.train_q25), "low",
                            np.where(val.actual.ge(val.train_q75), "high", "middle"))
    regime = pd.DataFrame(metric_rows(val, "2019-2022", "flow_regime", "flow_regime"))
    regime.to_csv(REPORT / "flow_regime_metrics.csv", index=False, encoding="utf-8-sig")

    val["descriptive_decile"] = pd.qcut(val.actual, 10, labels=False, duplicates="drop") + 1
    val["log_ratio"] = np.log(val.predict) - np.log(val.actual)
    deciles = val.groupby("descriptive_decile").log_ratio.agg(
        median_log_ratio="median",
        q25_log_ratio=lambda x: x.quantile(0.25),
        q75_log_ratio=lambda x: x.quantile(0.75),
        n="size",
    ).reset_index().rename(columns={"descriptive_decile": "decile"})
    deciles.to_csv(REPORT / "flow_decile_bias.csv", index=False, encoding="utf-8-sig")

    acf = pd.concat([residual_acf(oof, "2012-2018 OOF"), residual_acf(val, "2019-2022")], ignore_index=True)
    acf.to_csv(REPORT / "residual_acf_metrics.csv", index=False, encoding="utf-8-sig")

    coverage = val.groupby("year").agg(rows=("actual", "size"), stations=("q_site", "nunique"),
                                       observed_volume_cfs_sum=("actual", "sum")).reset_index()
    coverage.to_csv(REPORT / "evaluation_population_by_year.csv", index=False, encoding="utf-8-sig")

    make_main_figure(val, annual, pooled_oof, deciles, stations.loc[stations.q_site.isin(common)])
    make_station_figure(val)
    val.to_csv(REPORT / "figure_1_source_data_station_month.csv", index=False, encoding="utf-8-sig")

    raw_delta = pooled_val["raw_nse"] - pooled_oof["raw_nse"]
    log_delta = pooled_val["log_nse"] - pooled_oof["log_nse"]
    stable = (
        pooled_val["raw_nse"] >= 0.90 and pooled_val["log_nse"] >= 0.90
        and raw_delta >= -0.03 and log_delta >= -0.03
        and pooled_val["kge_2012"] >= 0.85 and abs(pooled_val["pbias_pct"]) <= 10
        and annual.raw_nse.min() >= 0.80 and annual.log_nse.min() >= 0.80
    )
    mixed = pooled_val["raw_nse"] >= 0.85 and pooled_val["log_nse"] >= 0.85
    terminal = "TEMPORAL_EXTRAPOLATION_STABLE" if stable else (
        "TEMPORAL_EXTRAPOLATION_MIXED" if mixed else "TEMPORAL_EXTRAPOLATION_DEGRADED"
    )
    gate = {
        "terminal": terminal,
        "evaluation_label": "HISTORICALLY_VIEWED_NON_PRISTINE_EVALUATION_PERIOD",
        "fit_years": "2006-2018",
        "evaluation_years": "2019-2022",
        "pooled_2012_2018_oof": pooled_oof,
        "pooled_2019_2022": pooled_val,
        "delta_raw_nse": raw_delta,
        "delta_log_nse": log_delta,
        "minimum_annual_raw_nse": float(annual.raw_nse.min()),
        "minimum_annual_log_nse": float(annual.log_nse.min()),
        "common_station_count": len(common),
        "hard_conditions": {
            "raw_nse_ge_0_90": bool(pooled_val["raw_nse"] >= 0.90),
            "log_nse_ge_0_90": bool(pooled_val["log_nse"] >= 0.90),
            "raw_delta_ge_minus_0_03": bool(raw_delta >= -0.03),
            "log_delta_ge_minus_0_03": bool(log_delta >= -0.03),
            "kge_ge_0_85": bool(pooled_val["kge_2012"] >= 0.85),
            "abs_pbias_le_10": bool(abs(pooled_val["pbias_pct"]) <= 10),
            "all_annual_raw_nse_ge_0_80": bool(annual.raw_nse.min() >= 0.80),
            "all_annual_log_nse_ge_0_80": bool(annual.log_nse.min() >= 0.80),
        },
    }
    (ROOT / "terminal_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    worst_year_raw = annual.loc[annual.raw_nse.idxmin()]
    worst_year_log = annual.loc[annual.log_nse.idxmin()]
    report_text = f"""# Q72 2019-2022 temporal extrapolation report

## Conclusion

Terminal: `{terminal}`.

The independent P1 copy reproduced all 8,822 OOF predictions exactly. A single
deployment-style model fitted on 2006-2018 then predicted 5,090 positive monthly
observations at 110 stations in 2019-2022. Pooled raw NSE was
`{pooled_val['raw_nse']:.6f}`, log-NSE `{pooled_val['log_nse']:.6f}`, KGE
`{pooled_val['kge_2012']:.6f}` and PBIAS `{pooled_val['pbias_pct']:.3f}%`.

Relative to strict pooled 2012-2018 temporal OOF, raw NSE changed by
`{raw_delta:+.6f}` and log-NSE by `{log_delta:+.6f}`. The weakest raw-NSE year
was {int(worst_year_raw.year)} (`{worst_year_raw.raw_nse:.6f}`); the weakest
log-NSE year was {int(worst_year_log.year)} (`{worst_year_log.log_nse:.6f}`).

## Interpretation boundary

This supports stable temporal transportability for positive monthly flow at
monitored stations under the frozen input and station universe. It does not
validate zero-flow probability, ungauged-Reach transfer, reservoir operations,
final-prediction mass conservation or a calibrated predictive distribution.
Moreover, 2019-2022 were historically viewed during earlier model development;
they are therefore a non-pristine temporal evaluation period rather than a
never-inspected confirmatory test.

## Figure contract

The figures were generated exclusively in Python/matplotlib. Figure 1 is a
quantitative evidence grid: annual efficiency, station-month agreement, bias by
flow decile and station-level transportability. Figure 2 shows eight
representative-success stations selected by a frozen rule: mean observed flow
at least 5,000 cfs and both station-level raw NSE and log-NSE at least 0.85,
ranked by the smaller NSE. It is not an all-station performance summary; weak
cases remain in the complete station table. Editable SVG and PDF plus 600-dpi
TIFF, PNG previews and source CSVs are supplied.
"""
    (ROOT / "Q72_2019_2022_temporal_extrapolation_report.md").write_text(report_text, encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
