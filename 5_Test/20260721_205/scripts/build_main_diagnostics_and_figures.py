from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
REPORTS = RUN / "reports"
FIG = RUN / "figure"
MAIN = REPORTS / "main_model"
MAIN_FIG = FIG / "main_model"
TOPO = ROOT / "0_reach_topology" / "results" / "tables"
EPS = 1.0e-6


def parse_ids(value: object) -> list[int]:
    if pd.isna(value):
        return []
    out: list[int] = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(float(token)))
        except ValueError:
            pass
    return out


def reservoir_influence() -> pd.DataFrame:
    topo = pd.read_csv(TOPO / "topology_edges.csv", encoding="utf-8-sig")
    topo["reach_id"] = topo["reach_id"].astype(int)
    topo["downstream_ids"] = topo["downstream_reach"].apply(parse_ids)
    tokens = ["水库", "湖", "水电站", "枢纽"]
    is_res = topo["src_id"].fillna("").astype(str).map(lambda x: any(t in x for t in tokens))
    reservoir_ids = set(topo.loc[is_res, "reach_id"].astype(int))
    name_by = dict(zip(topo["reach_id"].astype(int), topo["src_id"].astype(str)))
    downstream = dict(zip(topo["reach_id"].astype(int), topo["downstream_ids"]))
    rows: dict[int, dict[str, object]] = {}
    for rid in sorted(reservoir_ids):
        rows[rid] = {
            "reach_id": rid,
            "reach_name": name_by.get(rid, ""),
            "reservoir_self": 1,
            "reservoir_downstream_order": 0,
            "nearest_upstream_reservoir_reach": rid,
            "nearest_upstream_reservoir_name": name_by.get(rid, ""),
        }
        frontier = [(rid, 0)]
        seen = {rid}
        while frontier:
            current, order = frontier.pop(0)
            if order >= 2:
                continue
            for nxt in downstream.get(current, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                next_order = order + 1
                existing = rows.get(nxt)
                if existing is None or next_order < int(existing["reservoir_downstream_order"]):
                    rows[nxt] = {
                        "reach_id": nxt,
                        "reach_name": name_by.get(nxt, ""),
                        "reservoir_self": 0,
                        "reservoir_downstream_order": next_order,
                        "nearest_upstream_reservoir_reach": rid,
                        "nearest_upstream_reservoir_name": name_by.get(rid, ""),
                    }
                frontier.append((nxt, next_order))
    return pd.DataFrame(rows.values()).sort_values(["reservoir_downstream_order", "reach_id"])


def classify_failure(row: pd.Series) -> str:
    nse = float(row["NSE_log"])
    kge = float(row["KGE_2012"])
    pbias = float(row["PBIAS_pct"])
    if bool(row["good"]):
        return "good"
    if abs(pbias) > 25 and nse >= 0.50:
        return "bias_amplitude_problem"
    if nse < 0.20 and abs(pbias) <= 25:
        return "shape_timing_problem"
    if kge < 0.50 and nse >= 0.50:
        return "kge_component_problem"
    if nse < 0.20 and abs(pbias) > 25:
        return "shape_and_bias_problem"
    if 0.20 <= nse < 0.65:
        return "moderate_shape_skill_gap"
    return "threshold_edge_case"


def summarize_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(MAIN / "reach_class_station_metrics.csv", encoding="utf-8-sig")
    val = metrics[(metrics["variant"].eq("reach_class_alpha")) & (metrics["split"].eq("validation_2019_2022"))].copy()
    val["good"] = val["good"].astype(str).str.lower().isin(["true", "1", "yes"])
    res = reservoir_influence()
    val = val.merge(res, on="reach_id", how="left")
    val["reservoir_related"] = val["nearest_upstream_reservoir_reach"].notna()
    val["reservoir_relation"] = np.where(
        val["reservoir_self"].fillna(0).astype(int).eq(1),
        "reservoir_reach",
        np.where(val["reservoir_related"], "downstream_1_2_reaches", "not_reservoir_related"),
    )
    val["failure_mode"] = val.apply(classify_failure, axis=1)
    val["abs_PBIAS"] = val["PBIAS_pct"].abs()

    reservoir_summary = (
        val.groupby("reservoir_relation", dropna=False)
        .agg(
            stations=("q_site", "nunique"),
            good_count=("good", "sum"),
            bad_count=("good", lambda s: int((~s).sum())),
            median_NSElog=("NSE_log", "median"),
            median_KGE=("KGE_2012", "median"),
            median_absPBIAS=("abs_PBIAS", "median"),
        )
        .reset_index()
    )
    reservoir_detail = val[
        [
            "q_site",
            "reach_id",
            "reach_name",
            "reservoir_relation",
            "nearest_upstream_reservoir_name",
            "reservoir_downstream_order",
            "NSE_log",
            "KGE_2012",
            "PBIAS_pct",
            "good",
            "failure_mode",
        ]
    ].sort_values(["reservoir_relation", "good", "NSE_log"])
    failure_summary = (
        val.groupby("failure_mode")
        .agg(
            stations=("q_site", "nunique"),
            median_NSElog=("NSE_log", "median"),
            median_KGE=("KGE_2012", "median"),
            median_absPBIAS=("abs_PBIAS", "median"),
            reservoir_related_count=("reservoir_related", "sum"),
        )
        .reset_index()
        .sort_values(["stations", "median_NSElog"], ascending=[False, True])
    )
    val.to_csv(MAIN / "station_diagnostic_labels.csv", index=False, encoding="utf-8-sig")
    reservoir_summary.to_csv(MAIN / "reservoir_related_good_bad_summary.csv", index=False, encoding="utf-8-sig")
    reservoir_detail.to_csv(MAIN / "reservoir_related_station_detail.csv", index=False, encoding="utf-8-sig")
    failure_summary.to_csv(MAIN / "failure_mode_summary.csv", index=False, encoding="utf-8-sig")
    res.to_csv(MAIN / "reservoir_reach_influence_inventory.csv", index=False, encoding="utf-8-sig")
    return val, reservoir_summary, failure_summary


def choose_station_figures(val: pd.DataFrame) -> list[str]:
    chosen: list[str] = []

    def add(names: list[str]) -> None:
        for name in names:
            if name not in chosen:
                chosen.append(name)

    add(val[val["reservoir_related"]].sort_values(["good", "NSE_log"], ascending=[True, True])["q_site"].head(8).tolist())
    add(val[val["good"]].sort_values("NSE_log", ascending=False)["q_site"].head(6).tolist())
    add(val[~val["good"]].sort_values("NSE_log", ascending=True)["q_site"].head(10).tolist())
    add(val.sort_values("abs_PBIAS", ascending=False)["q_site"].head(8).tolist())
    return chosen[:24]


def make_station_hydrographs(val: pd.DataFrame) -> None:
    pred = pd.read_csv(MAIN / "reach_class_selected_predictions_long.csv", encoding="utf-8-sig")
    pred["date"] = pd.to_datetime(
        {"year": pred["year"].astype(int), "month": pred["month"].astype(int), "day": 1}
    )
    station_dir = MAIN_FIG / "station_hydrographs"
    station_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans SC",
                "SimHei",
                "SimSun",
                "DengXian",
                "DejaVu Sans",
            ],
            "figure.dpi": 160,
            "savefig.dpi": 350,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    selected = choose_station_figures(val)
    rows = []
    for site in selected:
        part = pred[pred["q_site"].eq(site)].sort_values("date").copy()
        if part.empty:
            continue
        meta = val[val["q_site"].eq(site)].iloc[0]
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        ax.plot(part["date"], part["Q_obsv_cfs"], color="#111111", lw=1.15, label="Observed")
        ax.plot(part["date"], part["Q_pred_cfs"], color="#2F6B9A", lw=1.15, label="Main model")
        ax.plot(part["date"], part["Q72_pred_cfs"], color="#D9822B", lw=0.85, alpha=0.55, label="Base regression")
        ax.plot(part["date"], part["Q78_mass_cfs"], color="#5C946E", lw=0.85, alpha=0.45, label="Mass baseline")
        ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"), color="#D9E6F2", alpha=0.28, lw=0)
        ax.set_yscale("log")
        ax.set_ylabel("Q (cfs, log)")
        ax.set_xlabel("")
        ax.set_title(
            f"{site} | reach {int(meta['reach_id'])} | NSElog={float(meta['NSE_log']):.3f}, "
            f"KGE={float(meta['KGE_2012']):.3f}, PBIAS={float(meta['PBIAS_pct']):.1f}%"
        )
        ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        safe = (
            str(site)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace("\"", "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )
        out = station_dir / f"{safe}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        rows.append(
            {
                "q_site": site,
                "figure": str(out),
                "reach_id": int(meta["reach_id"]),
                "good": bool(meta["good"]),
                "failure_mode": meta["failure_mode"],
                "reservoir_relation": meta["reservoir_relation"],
                "NSE_log": float(meta["NSE_log"]),
                "KGE_2012": float(meta["KGE_2012"]),
                "PBIAS_pct": float(meta["PBIAS_pct"]),
            }
        )
    pd.DataFrame(rows).to_csv(MAIN / "station_hydrograph_index.csv", index=False, encoding="utf-8-sig")

    # Compact overview: 12 representative stations, validation period only.
    overview_sites = selected[:12]
    fig, axes = plt.subplots(4, 3, figsize=(12.0, 9.0), sharex=True)
    for ax, site in zip(axes.flat, overview_sites):
        part = pred[pred["q_site"].eq(site) & pred["split"].eq("validation")].sort_values("date")
        meta = val[val["q_site"].eq(site)].iloc[0]
        ax.plot(part["date"], part["Q_obsv_cfs"], color="#111111", lw=0.9)
        ax.plot(part["date"], part["Q_pred_cfs"], color="#2F6B9A", lw=0.9)
        ax.set_yscale("log")
        ax.set_title(f"{site}\nNSElog {float(meta['NSE_log']):.2f}, KGE {float(meta['KGE_2012']):.2f}", fontsize=8)
    for ax in axes.flat[len(overview_sites) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(MAIN_FIG / "representative_station_validation_hydrographs.png", bbox_inches="tight")
    fig.savefig(MAIN_FIG / "representative_station_validation_hydrographs.pdf", bbox_inches="tight")
    plt.close(fig)


def write_analysis_report(val: pd.DataFrame, reservoir_summary: pd.DataFrame, failure_summary: pd.DataFrame) -> None:
    related = val[val["reservoir_related"]]
    direct = val[val["reservoir_relation"].eq("reservoir_reach")]
    lines = [
        "# 主线模型诊断报告",
        "",
        "## 水库相关站点",
        "",
        f"- 直接落在水库 reach 的站点：{direct['q_site'].nunique()} 个，其中 good={int(direct['good'].sum())}，未达 good={int((~direct['good']).sum())}。",
        f"- 水库 reach 及其下游 1-2 级影响范围内站点：{related['q_site'].nunique()} 个，其中 good={int(related['good'].sum())}，未达 good={int((~related['good']).sum())}。",
        "",
        "水库关系汇总表见 `reports/main_model/reservoir_related_good_bad_summary.csv`。",
        "",
        "## 未达标站点的主要问题",
        "",
    ]
    for r in failure_summary.itertuples(index=False):
        lines.append(
            f"- {r.failure_mode}: {int(r.stations)} 个站；median NSElog={float(r.median_NSElog):.3f}，"
            f"KGE={float(r.median_KGE):.3f}，median |PBIAS|={float(r.median_absPBIAS):.1f}%。"
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "当前主线模型暂时不适合把严格水库调度方程作为第一优先改造项。依据是：直接落在水库 reach 的验证站点数量很少，且前面严格质量守恒骨架的预测技能明显低于当前轻约束主线。更实际的问题是：未达标站点主要表现为过程形状/时序不足或峰值幅度/系统偏差不足，这些问题会被水库调节、reach/catchment 匹配误差、跨境/边界入流、人为取退水等未显式表达的过程放大。",
            "",
            "站点流量过程图已保存到 `figure/main_model/station_hydrographs`，索引见 `reports/main_model/station_hydrograph_index.csv`。",
        ]
    )
    (MAIN / "main_model_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    MAIN.mkdir(parents=True, exist_ok=True)
    MAIN_FIG.mkdir(parents=True, exist_ok=True)
    val, reservoir_summary, failure_summary = summarize_metrics()
    make_station_hydrographs(val)
    write_analysis_report(val, reservoir_summary, failure_summary)


if __name__ == "__main__":
    main()
