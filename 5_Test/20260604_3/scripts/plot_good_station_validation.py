from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260604_3")
REPORT_DIR = ROOT / "reports"
FIG_DIR = ROOT / "figure"
STATION_FIG_DIR = FIG_DIR / "station_timeseries"

METRICS_PATH = REPORT_DIR / "hydrologic_metrics_by_station_validation.csv"
RESID_PATH = REPORT_DIR / "validation_resids_native_run.csv"


def configure_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def classify_area(area: float, q1: float, q2: float) -> str:
    if area <= q1:
        return "上游"
    if area <= q2:
        return "中游"
    return "下游"


def make_quality_table(metrics: pd.DataFrame, resids: pd.DataFrame) -> pd.DataFrame:
    area = (
        resids.groupby("q_site", as_index=False)
        .agg(
            nested_area_mean=("nested_area", "mean"),
            validation_rows=("q_site", "size"),
            validation_mean_actual=("actual", "mean"),
            validation_mean_predict=("predict", "mean"),
        )
    )
    out = metrics.merge(area, on="q_site", how="inner")
    q1, q2 = out["nested_area_mean"].quantile([1 / 3, 2 / 3]).tolist()
    out["river_position"] = out["nested_area_mean"].apply(lambda x: classify_area(x, q1, q2))
    out["abs_PBIAS_pct"] = out["PBIAS_pct"].abs()

    # Hydrologic fit score: high NSE_log/KGE are good; large bias is penalized.
    # NSE_raw is included lightly so that magnitude-space behavior still matters.
    out["quality_score"] = (
        out["NSE_log"].clip(lower=-2, upper=1)
        + out["KGE_2012"].clip(lower=-2, upper=1)
        + 0.35 * out["NSE_raw"].clip(lower=-2, upper=1)
        - 0.01 * out["abs_PBIAS_pct"].clip(upper=200)
    )
    return out.sort_values(
        ["river_position", "quality_score", "NSE_log", "KGE_2012"],
        ascending=[True, False, False, False],
    )


def select_balanced_stations(quality: pd.DataFrame) -> pd.DataFrame:
    targets = {"上游": 3, "中游": 3, "下游": 4}
    selected_parts = []
    selected_names: set[str] = set()

    for pos, n in targets.items():
        part = quality[quality["river_position"] == pos].sort_values(
            ["quality_score", "NSE_log", "KGE_2012"], ascending=False
        )
        take = part.head(n)
        selected_parts.append(take)
        selected_names.update(take["q_site"].tolist())

    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) < 10:
        fill = quality[~quality["q_site"].isin(selected_names)].sort_values(
            ["quality_score", "NSE_log", "KGE_2012"], ascending=False
        )
        selected = pd.concat([selected, fill.head(10 - len(selected))], ignore_index=True)

    position_order = {"上游": 0, "中游": 1, "下游": 2}
    selected = selected.copy()
    selected["river_position_order"] = selected["river_position"].map(position_order)
    return selected.sort_values(
        ["river_position_order", "nested_area_mean", "quality_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def period_label(row: pd.Series) -> str:
    return f"{int(row['year'])}Q{int(row['quarter'])}"


def plot_panel(selected: pd.DataFrame, resids: pd.DataFrame) -> None:
    names = selected["q_site"].tolist()
    plot_df = resids[resids["q_site"].isin(names)].copy()
    plot_df["period_label"] = plot_df.apply(period_label, axis=1)
    period_order = (
        plot_df[["year", "quarter", "period_label"]]
        .drop_duplicates()
        .sort_values(["year", "quarter"])["period_label"]
        .tolist()
    )
    period_to_x = {p: i for i, p in enumerate(period_order)}
    plot_df["x"] = plot_df["period_label"].map(period_to_x)

    n = len(names)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.65 * nrows), squeeze=False)

    metric_lookup = selected.set_index("q_site").to_dict("index")
    for ax, name in zip(axes.flat, names):
        one = plot_df[plot_df["q_site"] == name].sort_values(["year", "quarter"])
        m = metric_lookup[name]
        ax.plot(one["x"], one["actual"], marker="o", linewidth=2, label="实测")
        ax.plot(one["x"], one["predict"], marker="s", linewidth=2, label="模拟")
        ax.set_title(
            f"{m['river_position']} | {name} | NSElog={m['NSE_log']:.2f}, "
            f"KGE={m['KGE_2012']:.2f}, PBIAS={m['PBIAS_pct']:.1f}%",
            fontsize=11,
        )
        ax.set_ylabel("Q")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(range(len(period_order)))
        ax.set_xticklabels(period_order, rotation=0)
        ax.ticklabel_format(axis="y", style="plain")

    for ax in axes.flat[n:]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("20260604_3 上中下游优选站点验证期实测-模拟对比", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0.022, 1, 0.965))
    fig.savefig(FIG_DIR / "good_station_validation_timeseries.png", dpi=220)
    plt.close(fig)


def plot_scatter(selected: pd.DataFrame, resids: pd.DataFrame) -> None:
    names = selected["q_site"].tolist()
    plot_df = resids[resids["q_site"].isin(names)].copy()
    metric_lookup = selected.set_index("q_site")["river_position"].to_dict()

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    colors = {"上游": "#2E7D32", "中游": "#1565C0", "下游": "#B45309"}
    for name in names:
        one = plot_df[plot_df["q_site"] == name]
        pos = metric_lookup[name]
        ax.scatter(
            one["actual"],
            one["predict"],
            s=48,
            alpha=0.82,
            label=f"{pos}-{name}",
            color=colors[pos],
            edgecolor="white",
            linewidth=0.6,
        )

    all_values = pd.concat([plot_df["actual"], plot_df["predict"]]).dropna()
    lower = max(all_values.min() * 0.8, 1e-6)
    upper = all_values.max() * 1.2
    ax.plot([lower, upper], [lower, upper], color="#333333", linewidth=1.2, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("实测 Q（log）")
    ax.set_ylabel("模拟 Q（log）")
    ax.set_title("20260604_3 优选站点观测-模拟散点")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "good_station_observed_vs_predicted.png", dpi=220)
    plt.close(fig)


def plot_individual(selected: pd.DataFrame, resids: pd.DataFrame) -> None:
    STATION_FIG_DIR.mkdir(parents=True, exist_ok=True)
    metric_lookup = selected.set_index("q_site").to_dict("index")
    for name in selected["q_site"]:
        one = resids[resids["q_site"] == name].copy().sort_values(["year", "quarter"])
        one["period_label"] = one.apply(period_label, axis=1)
        m = metric_lookup[name]
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        ax.plot(one["period_label"], one["actual"], marker="o", linewidth=2.2, label="实测")
        ax.plot(one["period_label"], one["predict"], marker="s", linewidth=2.2, label="模拟")
        ax.set_title(
            f"{m['river_position']} | {name} | NSElog={m['NSE_log']:.2f}, "
            f"KGE={m['KGE_2012']:.2f}, PBIAS={m['PBIAS_pct']:.1f}%"
        )
        ax.set_xlabel("验证期")
        ax.set_ylabel("Q")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        ax.ticklabel_format(axis="y", style="plain")
        fig.tight_layout()
        safe_name = (
            str(name)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )
        fig.savefig(STATION_FIG_DIR / f"{safe_name}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    configure_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(METRICS_PATH)
    resids = pd.read_csv(RESID_PATH)
    quality = make_quality_table(metrics, resids)
    selected = select_balanced_stations(quality)

    selected_cols = [
        "river_position",
        "q_site",
        "nested_area_mean",
        "n",
        "NSE_raw",
        "NSE_log",
        "KGE_2012",
        "PBIAS_pct",
        "RMSE",
        "MAE",
        "quality_score",
        "validation_mean_actual",
        "validation_mean_predict",
    ]
    selected[selected_cols].to_csv(FIG_DIR / "selected_good_stations.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(FIG_DIR / "station_quality_rank_all.csv", index=False, encoding="utf-8-sig")

    plot_panel(selected, resids)
    plot_scatter(selected, resids)
    plot_individual(selected, resids)

    print("Selected stations:")
    print(selected[selected_cols].to_string(index=False))
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
