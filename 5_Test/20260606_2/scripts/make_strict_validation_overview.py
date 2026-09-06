from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = RUN_DIR / "reports"
FIG_DIR = RUN_DIR / "figure"


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(REPORT_DIR / "strict_metric_summary_2006_2022.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(REPORT_DIR / "strict_metrics_by_station_2006_2022.csv", encoding="utf-8-sig")
    good = pd.read_csv(REPORT_DIR / "strict_good_validation_stations.csv", encoding="utf-8-sig")
    not_good = pd.read_csv(REPORT_DIR / "strict_not_good_validation_stations.csv", encoding="utf-8-sig")
    no_val = pd.read_csv(REPORT_DIR / "strict_no_validation_observation_stations.csv", encoding="utf-8-sig")
    pred_obs = pd.read_csv(REPORT_DIR / "strict_prediction_vs_observed_2006_2022.csv", encoding="utf-8-sig")
    return summary, metrics, good, not_good, no_val, pred_obs


def plot_metric_distribution(metrics: pd.DataFrame) -> None:
    val = metrics[metrics["val_n"] > 0].copy()
    good = val["good_validation"].astype(bool)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.5))
    axes = axes.reshape(-1)

    axes[0].hist(val.loc[~good, "val_NSE_log"].dropna(), bins=24, color="#9CA3AF", alpha=0.75, label="Not good")
    axes[0].hist(val.loc[good, "val_NSE_log"].dropna(), bins=10, color="#1F77B4", alpha=0.9, label="Good")
    axes[0].axvline(0.65, color="#B91C1C", lw=1.2, ls="--")
    axes[0].set_title("Validation NSElog distribution")
    axes[0].set_xlabel("NSElog")
    axes[0].set_ylabel("Station count")

    axes[1].hist(val.loc[~good, "val_KGE_2012"].dropna(), bins=24, color="#9CA3AF", alpha=0.75)
    axes[1].hist(val.loc[good, "val_KGE_2012"].dropna(), bins=10, color="#1F77B4", alpha=0.9)
    axes[1].axvline(0.50, color="#B91C1C", lw=1.2, ls="--")
    axes[1].set_title("Validation KGE distribution")
    axes[1].set_xlabel("KGE")

    axes[2].hist(val.loc[~good, "abs_val_PBIAS_pct"].dropna(), bins=24, color="#9CA3AF", alpha=0.75)
    axes[2].hist(val.loc[good, "abs_val_PBIAS_pct"].dropna(), bins=10, color="#1F77B4", alpha=0.9)
    axes[2].axvline(25, color="#B91C1C", lw=1.2, ls="--")
    axes[2].set_title("Validation absolute PBIAS distribution")
    axes[2].set_xlabel("|PBIAS| (%)")
    axes[2].set_ylabel("Station count")

    colors = np.where(good, "#1F77B4", "#9CA3AF")
    axes[3].scatter(val["val_NSE_log"], val["val_KGE_2012"], c=colors, s=36, alpha=0.82, edgecolor="white", linewidth=0.4)
    axes[3].axvline(0.65, color="#B91C1C", lw=1.0, ls="--")
    axes[3].axhline(0.50, color="#B91C1C", lw=1.0, ls="--")
    axes[3].set_title("NSElog-KGE validation space")
    axes[3].set_xlabel("NSElog")
    axes[3].set_ylabel("KGE")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strict_validation_metric_distributions.png", dpi=300)
    plt.close(fig)


def plot_status_by_river(metrics: pd.DataFrame, no_val: pd.DataFrame) -> None:
    val = metrics[metrics["val_n"] > 0].copy()
    rows = []
    for pos in ["上游", "中游", "下游"]:
        part = val[val["river_position"] == pos]
        rows.append(
            {
                "river_position": pos,
                "Good": int(part["good_validation"].sum()),
                "Not good": int((~part["good_validation"].astype(bool)).sum()),
                "No validation": int((no_val["river_position"] == pos).sum()) if "river_position" in no_val.columns else 0,
            }
        )
    df = pd.DataFrame(rows).set_index("river_position")
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    bottom = np.zeros(len(df))
    colors = {"Good": "#1F77B4", "Not good": "#D97706", "No validation": "#6B7280"}
    for col in ["Good", "Not good", "No validation"]:
        vals = df[col].to_numpy()
        ax.bar(df.index, vals, bottom=bottom, label=col, color=colors[col])
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, bottom[i] + v / 2, str(int(v)), ha="center", va="center", color="white", weight="bold")
        bottom += vals
    ax.set_ylabel("Station count")
    ax.set_title("Validation status by upstream-midstream-downstream group")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strict_validation_status_by_river_position.png", dpi=300)
    plt.close(fig)


def plot_rank(metrics: pd.DataFrame) -> None:
    val = metrics[metrics["val_n"] > 0].sort_values("validation_rank_score", ascending=False).copy()
    show = pd.concat([val.head(20), val.tail(20)]).drop_duplicates("q_site")
    show = show.sort_values("validation_rank_score", ascending=True)
    colors = np.where(show["good_validation"].astype(bool), "#1F77B4", "#D97706")
    fig, ax = plt.subplots(figsize=(8.5, 9.2))
    y = np.arange(len(show))
    ax.barh(y, show["validation_rank_score"], color=colors, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(show["q_site"], fontsize=8)
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_xlabel("Validation rank score")
    ax.set_title("Top and bottom validation stations")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strict_validation_station_rank_top_bottom.png", dpi=300)
    plt.close(fig)


def plot_obs_pred_scatter(metrics: pd.DataFrame, pred_obs: pd.DataFrame) -> None:
    val = pred_obs[pred_obs["split"] == "validation"].copy()
    status = metrics.set_index("q_site")["good_validation"].to_dict()
    val["status"] = val["q_site"].map(lambda s: "Good" if bool(status.get(s, False)) else "Not good")
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    colors = {"Good": "#1F77B4", "Not good": "#9CA3AF"}
    for status_name, part in val.groupby("status"):
        ax.scatter(part["actual"], part["predict"], s=18, alpha=0.72, color=colors[status_name], label=status_name, linewidth=0)
    vals = pd.concat([val["actual"], val["predict"]]).dropna()
    lo = max(vals[vals > 0].min() * 0.75, 1e-6)
    hi = vals.max() * 1.25
    ax.plot([lo, hi], [lo, hi], color="#111827", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed Q (cfs)")
    ax.set_ylabel("Predicted Q (cfs)")
    ax.set_title("Strict validation: all observed-predicted points (2019-2022)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strict_validation_observed_predicted_scatter_all.png", dpi=300)
    plt.close(fig)


def plot_representative_hydrographs(metrics: pd.DataFrame, pred_obs: pd.DataFrame) -> None:
    good_names = metrics[(metrics["val_n"] > 0) & (metrics["good_validation"])].sort_values(
        "validation_rank_score", ascending=False
    )["q_site"].head(8)
    bad_names = metrics[(metrics["val_n"] > 0) & (~metrics["good_validation"].astype(bool))].sort_values(
        "validation_rank_score", ascending=True
    )["q_site"].head(4)
    names = list(good_names) + list(bad_names)
    data = pred_obs[pred_obs["q_site"].isin(names)].copy()
    data["date"] = pd.to_datetime(
        {"year": data["year"].astype(int), "month": data["month"].astype(int), "day": 1},
        errors="coerce",
    ) + pd.offsets.MonthEnd(0)
    metric_lookup = metrics.set_index("q_site").to_dict("index")

    fig, axes = plt.subplots(6, 2, figsize=(11.0, 12.0), sharex=True)
    axes = axes.reshape(-1)
    for ax, name in zip(axes, names):
        one = data[data["q_site"] == name].sort_values(["year", "month"])
        m = metric_lookup[name]
        ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"), color="#E9EEF6", zorder=0)
        ax.plot(one["date"], one["actual"], color="#1F4E79", lw=1.25, marker="o", markersize=2.0, label="Observed")
        ax.plot(one["date"], one["predict"], color="#C44E24", lw=1.2, marker="s", markersize=1.8, label="Predicted")
        status = "Good" if bool(m["good_validation"]) else "Not good"
        ax.set_title(
            f"{status} | {m['river_position']} | {name} | NSElog={m['val_NSE_log']:.2f}, "
            f"KGE={m['val_KGE_2012']:.2f}, PBIAS={m['val_PBIAS_pct']:.1f}%",
            fontsize=8.4,
        )
        ax.grid(True, alpha=0.22)
        ax.ticklabel_format(axis="y", style="plain")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FIG_DIR / "strict_validation_representative_hydrographs.png", dpi=300)
    plt.close(fig)


def html_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    part = df[cols].copy()
    if n is not None:
        part = part.head(n)
    for col in part.columns:
        if pd.api.types.is_float_dtype(part[col]):
            part[col] = part[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return part.to_html(index=False, classes="data-table", escape=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| |\n|-|\n| |"
    part = df.copy()
    for col in part.columns:
        if pd.api.types.is_float_dtype(part[col]):
            part[col] = part[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            part[col] = part[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(part.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(part.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in part.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_reports(summary: pd.DataFrame, metrics: pd.DataFrame, good: pd.DataFrame, not_good: pd.DataFrame, no_val: pd.DataFrame) -> None:
    good_cols = ["river_position", "q_site", "val_n", "val_NSE_log", "val_KGE_2012", "val_PBIAS_pct"]
    bad_cols = good_cols + ["failure_reason"]
    no_cols = ["river_position", "q_site", "years_present", "months_present", "val_n"]
    figs = [
        "strict_validation_status_by_river_position.png",
        "strict_validation_metric_distributions.png",
        "strict_validation_observed_predicted_scatter_all.png",
        "strict_validation_station_rank_top_bottom.png",
        "strict_validation_representative_hydrographs.png",
    ]

    reason_counts: dict[str, int] = {}
    for reason in not_good["failure_reason"].fillna(""):
        for token in str(reason).split(";"):
            token = token.strip()
            if token:
                reason_counts[token] = reason_counts.get(token, 0) + 1
    reason_df = pd.DataFrame(sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True), columns=["failure_reason", "station_count"])
    reason_df.to_csv(REPORT_DIR / "strict_validation_failure_reason_counts.csv", index=False, encoding="utf-8-sig")

    html = [
        "<html><head><meta charset='utf-8'><title>20260606_2 monthly strict validation overview</title>",
        "<style>body{font-family:'Microsoft YaHei',Arial,sans-serif;margin:28px;color:#111827}"
        ".cards{display:flex;gap:12px;flex-wrap:wrap}.card{border:1px solid #ddd;border-radius:6px;padding:12px 16px;min-width:150px}"
        ".big{font-size:26px;font-weight:700;color:#1F4E79}.data-table{border-collapse:collapse;font-size:13px;margin:10px 0 22px 0}"
        ".data-table th{background:#1F4E79;color:white}.data-table th,.data-table td{border:1px solid #ddd;padding:5px 8px}"
        "img{max-width:100%;border:1px solid #eee;margin:12px 0 24px 0}</style></head><body>",
        "<h1>20260606_2 Monthly Strict Validation Overview</h1>",
        "<p><b>Strict split:</b> 2006-2018 calibration; 2019-2022 validation hidden from fitting and station weights.</p>",
        "<div class='cards'>",
        f"<div class='card'><div>Validation stations</div><div class='big'>{int((metrics['val_n'] > 0).sum())}</div></div>",
        f"<div class='card'><div>Good</div><div class='big'>{len(good)}</div></div>",
        f"<div class='card'><div>Not good</div><div class='big'>{len(not_good)}</div></div>",
        f"<div class='card'><div>No validation obs</div><div class='big'>{len(no_val)}</div></div>",
        "</div>",
        "<h2>Metric Summary</h2>",
        summary.to_html(index=False, classes="data-table", escape=False),
        "<h2>Figures</h2>",
    ]
    for fig in figs:
        html.append(f"<h3>{fig}</h3><img src='../figure/{fig}' />")
    html += [
        "<h2>Good Validation Stations</h2>",
        html_table(good.sort_values("validation_rank_score", ascending=False), good_cols),
        "<h2>Worst Not-Good Stations (lowest rank score)</h2>",
        html_table(not_good.sort_values("validation_rank_score", ascending=True), bad_cols, n=30),
        "<h2>Failure Reason Counts</h2>",
        reason_df.to_html(index=False, classes="data-table", escape=False),
        "<h2>No Validation Observation Stations</h2>",
        html_table(no_val, no_cols),
        "</body></html>",
    ]
    (REPORT_DIR / "strict_validation_overview_20260606_2.html").write_text("\n".join(html), encoding="utf-8")

    md = [
        "# 20260606_2 Monthly Strict Validation Overview",
        "",
        "Strict split: 2006-2018 calibration; 2019-2022 validation hidden from fitting and station weights.",
        "",
        f"- Validation stations: {int((metrics['val_n'] > 0).sum())}",
        f"- Good: {len(good)}",
        f"- Not good: {len(not_good)}",
        f"- No validation observations: {len(no_val)}",
        "",
        "## Figures",
        "",
    ]
    for fig in figs:
        md.append(f"- `figure/{fig}`")
    md += [
        "",
        "## Good Validation Stations",
        "",
        markdown_table(good.sort_values("validation_rank_score", ascending=False)[good_cols]),
        "",
        "## No Validation Observation Stations",
        "",
        markdown_table(no_val[no_cols]),
        "",
    ]
    (REPORT_DIR / "strict_validation_overview_20260606_2.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary, metrics, good, not_good, no_val, pred_obs = load_tables()
    plot_metric_distribution(metrics)
    plot_status_by_river(metrics, no_val)
    plot_rank(metrics)
    plot_obs_pred_scatter(metrics, pred_obs)
    plot_representative_hydrographs(metrics, pred_obs)
    write_reports(summary, metrics, good, not_good, no_val)
    print(REPORT_DIR / "strict_validation_overview_20260606_2.html")


if __name__ == "__main__":
    main()
