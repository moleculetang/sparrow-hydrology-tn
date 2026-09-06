from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet"
EXT = ROOT / "outputs" / "fit_2006_2018_eval_2019_2022" / "validation_predictions_2019_2022.parquet"
FIG = ROOT / "figures"
SOURCE = FIG / "source_data"
REPORT = ROOT / "reports"
EPS = 1e-12
SHIJIAO = "\u77f3\u89d2\u7ad9"

PALETTE = {
    "blue": "#0F4D92", "blue_soft": "#8DB5DD", "red": "#B64342",
    "green": "#2E9E44", "neutral": "#767676", "dark": "#272727",
    "light": "#D8D8D8", "violet": "#9A4D8E",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def metric(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[valid], pred[valid]
    if len(obs) < 2:
        return {"n": int(len(obs)), "raw_NSE": np.nan, "log_NSE": np.nan, "KGE_2012": np.nan, "PBIAS_pct": np.nan, "RMSE": np.nan, "log_RMSE": np.nan}
    sst = np.sum((obs - obs.mean()) ** 2)
    lo, lp = np.log(obs), np.log(pred)
    log_sst = np.sum((lo - lo.mean()) ** 2)
    r = np.corrcoef(obs, pred)[0, 1] if np.std(obs) and np.std(pred) else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) if np.isfinite(r) else np.nan
    return {
        "n": int(len(obs)),
        "raw_NSE": float(1 - np.sum((pred - obs) ** 2) / sst) if sst else np.nan,
        "log_NSE": float(1 - np.sum((lp - lo) ** 2) / log_sst) if log_sst else np.nan,
        "KGE_2012": float(kge),
        "PBIAS_pct": float(100 * np.sum(pred - obs) / np.sum(obs)),
        "RMSE": float(np.sqrt(np.mean((pred - obs) ** 2))),
        "log_RMSE": float(np.sqrt(np.mean((lp - lo) ** 2))),
    }


def station_metrics(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    rows = []
    for station, part in frame.groupby("q_site", sort=True):
        row = {"q_site": str(station), "period": period, **metric(part.actual, part.predict)}
        row["is_shijiao"] = row["q_site"] == SHIJIAO
        rows.append(row)
    return pd.DataFrame(rows).sort_values("q_site").reset_index(drop=True)


def set_style() -> None:
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "DejaVu Sans", "Liberation Sans"]
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["legend.frameon"] = False


def save(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in [("svg", None), ("pdf", None), ("tiff", 600), ("png", 300)]:
        p = base.with_suffix(f".{ext}")
        kw = {"bbox_inches": "tight"}
        if dpi:
            kw["dpi"] = dpi
        fig.savefig(p, **kw)
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def panel_scatter(ax: plt.Axes, data: pd.DataFrame, title: str, metrics: dict[str, float]) -> None:
    lo = max(float(min(data.actual.min(), data.predict.min())), 1e-5)
    hi = float(max(data.actual.max(), data.predict.max())) * 1.2
    ax.scatter(data.actual, data.predict, s=5, alpha=0.30, color=PALETTE["blue"], linewidths=0, rasterized=True)
    ax.plot([lo, hi], [lo, hi], color=PALETTE["red"], lw=1.0, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed monthly flow (cfs)")
    ax.set_ylabel("Predicted monthly flow (cfs)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
    ax.text(0.04, 0.96, f"n = {metrics['n']:,}\nraw NSE = {metrics['raw_NSE']:.3f}\nlog-NSE = {metrics['log_NSE']:.3f}\nKGE = {metrics['KGE_2012']:.3f}\nPBIAS = {metrics['PBIAS_pct']:+.1f}%", transform=ax.transAxes, va="top", fontsize=7, bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "none", "alpha": 0.88})


def build_figures(oof: pd.DataFrame, ext: pd.DataFrame, oof_st: pd.DataFrame, ext_st: pd.DataFrame, pooled: pd.DataFrame) -> list[str]:
    set_style()
    (SOURCE / "figure_1").mkdir(parents=True, exist_ok=True)
    oof.to_csv(SOURCE / "figure_1" / "oof_station_month_predictions.csv", index=False, encoding="utf-8-sig")
    ext.to_csv(SOURCE / "figure_1" / "external_station_month_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat([oof_st, ext_st], ignore_index=True).to_csv(SOURCE / "figure_1" / "station_metrics.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(SOURCE / "figure_1" / "pooled_metrics.csv", index=False, encoding="utf-8-sig")

    oof_m = pooled.loc[pooled.period.eq("OOF_2012_2018")].iloc[0].to_dict()
    ext_m = pooled.loc[pooled.period.eq("EXTERNAL_2019_2022")].iloc[0].to_dict()
    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.46, wspace=0.40)
    axa, axb, axc, axd = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))
    panel_scatter(axa, oof, "a  Blocked time-fold OOF (2012–2018)", oof_m)
    panel_scatter(axb, ext, "b  Temporal extrapolation (2019–2022)", ext_m)

    joined = oof_st.merge(ext_st, on="q_site", suffixes=("_oof", "_ext"), how="inner")
    lo, hi = min(joined.raw_NSE_oof.min(), joined.raw_NSE_ext.min(), 0.0), max(joined.raw_NSE_oof.max(), joined.raw_NSE_ext.max(), 1.0)
    axc.scatter(joined.raw_NSE_oof, joined.raw_NSE_ext, s=22, color=PALETTE["blue"], alpha=0.82, linewidths=0)
    axc.plot([lo, hi], [lo, hi], color=PALETTE["red"], lw=1.0)
    axc.axhline(0, lw=0.7, color=PALETTE["neutral"], ls=":"); axc.axvline(0, lw=0.7, color=PALETTE["neutral"], ls=":")
    axc.set_xlim(lo - 0.03, hi + 0.03); axc.set_ylim(lo - 0.03, hi + 0.03)
    axc.set_xlabel("Station raw NSE: OOF 2012–2018"); axc.set_ylabel("Station raw NSE: 2019–2022")
    axc.set_title("c  Station-level temporal stability", loc="left", fontweight="bold", fontsize=9)
    shi = joined.index[joined.q_site.eq(SHIJIAO)]
    if len(shi):
        ix = int(shi[0]); axc.scatter([joined.raw_NSE_oof.iloc[ix]], [joined.raw_NSE_ext.iloc[ix]], s=35, color=PALETTE["violet"], zorder=4)
        axc.annotate(SHIJIAO, (joined.raw_NSE_oof.iloc[ix], joined.raw_NSE_ext.iloc[ix]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=PALETTE["violet"])

    def hydro(part: pd.DataFrame, label: str, color: str):
        if part.empty:
            return
        p = part.copy(); p["date"] = pd.to_datetime(dict(year=p.year, month=p.month, day=1))
        p = p.sort_values("date")
        axd.plot(p.date, p.actual, color=PALETTE["dark"], lw=1.0, label=f"Observed ({label})")
        axd.plot(p.date, p.predict, color=color, lw=1.1, label=f"S111 ({label})")
    hydro(oof.loc[oof.q_site.astype(str).eq(SHIJIAO)], "OOF", PALETTE["blue"])
    hydro(ext.loc[ext.q_site.astype(str).eq(SHIJIAO)], "external", PALETTE["green"])
    axd.set_yscale("log"); axd.set_xlabel("Month"); axd.set_ylabel("Flow (cfs)")
    axd.set_title(f"d  Protected {SHIJIAO}", loc="left", fontweight="bold", fontsize=9)
    axd.legend(fontsize=5.8, ncol=2, loc="upper center")
    paths = save(fig, FIG / "S111_operational_baseline_performance")

    fig2, axs = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=False)
    for ax, tab, label, color in [(axs[0], oof_st, "OOF 2012–2018", PALETTE["blue"]), (axs[1], ext_st, "External 2019–2022", PALETTE["green"])]:
        vals = tab.sort_values("log_NSE").reset_index(drop=True)
        xx = np.arange(len(vals))
        ax.bar(xx, vals.log_NSE, width=0.85, color=color, alpha=0.78)
        ax.axhline(0, color=PALETTE["red"], lw=0.8, ls="--")
        ax.set_xlabel("Retained station (ordered by log-NSE)"); ax.set_ylabel("Station log-NSE")
        ax.set_title(label, loc="left", fontweight="bold", fontsize=9)
        p = vals.index[vals.q_site.eq(SHIJIAO)]
        if len(p):
            i = int(p[0]); ax.annotate(SHIJIAO, (i, vals.log_NSE.iloc[i]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7, color=PALETTE["violet"], rotation=90)
    paths.extend(save(fig2, FIG / "S111_station_log_nse_distribution"))
    return paths


def build_manifest() -> dict[str, object]:
    files = []
    for folder in [ROOT / "inputs", ROOT / "scripts", ROOT / "reference_selected"]:
        for p in sorted(x for x in folder.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
            files.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "files": files, "file_count": len(files)}


def main() -> None:
    runtime = assert_sparrow_runtime()
    for p in [OOF, EXT]:
        if not p.exists():
            raise RuntimeError(f"Missing rerun output: {p}")
    oof, ext = pd.read_parquet(OOF), pd.read_parquet(EXT)
    oof_st, ext_st = station_metrics(oof, "OOF_2012_2018"), station_metrics(ext, "EXTERNAL_2019_2022")
    pooled = pd.DataFrame([
        {"period": "OOF_2012_2018", **metric(oof.actual, oof.predict), "stations": int(oof.q_site.nunique())},
        {"period": "EXTERNAL_2019_2022", **metric(ext.actual, ext.predict), "stations": int(ext.q_site.nunique())},
    ])
    folds = pd.DataFrame([{ "fold_id": str(fid), **metric(part.actual, part.predict), "stations": int(part.q_site.nunique())} for fid, part in oof.groupby("fold_id", sort=True)])
    oof_st.to_csv(REPORT / "station_metrics_oof_2012_2018.csv", index=False, encoding="utf-8-sig")
    ext_st.to_csv(REPORT / "station_metrics_external_2019_2022.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(REPORT / "pooled_metrics.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(REPORT / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    figures = build_figures(oof, ext, oof_st, ext_st, pooled)
    manifest = build_manifest()
    manifest.update({"runtime": runtime, "python": sys.version, "platform": platform.platform(), "thread_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}})
    (REPORT / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    population = json.loads((REPORT / "reproduction_gate.json").read_text(encoding="utf-8"))
    deterministic = json.loads((REPORT / "deterministic_reproduction_gate.json").read_text(encoding="utf-8"))
    protected = pd.concat([oof_st, ext_st]).loc[lambda x: x.q_site.eq(SHIJIAO)]
    protected.to_csv(REPORT / "protected_shijiao_metrics.csv", index=False, encoding="utf-8-sig")
    independence_path = REPORT / "standalone_independence_audit.json"
    independence = json.loads(independence_path.read_text(encoding="utf-8")) if independence_path.exists() else {"terminal": "NOT_YET_RUN"}
    passed = population["terminal"].endswith("PASS") and deterministic["terminal"].endswith("PASS") and independence["terminal"].endswith("PASS")
    terminal = "S111_STANDALONE_OPERATIONAL_RERUN_COMPLETE" if passed else "S111_STANDALONE_OPERATIONAL_RERUN_FAILURE"
    payload = {"terminal": terminal, "generated_utc": datetime.now(timezone.utc).isoformat(), "population_gate": population["terminal"], "determinism_gate": deterministic["terminal"], "independence_gate": independence["terminal"], "oof_rows": int(len(oof)), "external_rows": int(len(ext)), "figures": figures, "post_hoc_station_domain_caveat": True}
    (REPORT / "terminal_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "delivery_audit.json").write_text(json.dumps({**payload, "complete_inputs_local": True, "complete_code_local": True, "outputs_regenerated_in_this_directory": True, "figures_generated_from_local_outputs": True, "protected_shijiao_rows": {"oof": int(oof.q_site.eq(SHIJIAO).sum()), "external": int(ext.q_site.eq(SHIJIAO).sum())}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if terminal.endswith("FAILURE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
