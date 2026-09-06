from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260608_2"
BASE_REPORT = RUN / "reports" / "main_model"
REPORT = RUN / "reports" / "human_water_use_model"
FIG = RUN / "figure" / "human_water_use_model"
HSWUD_CSV = RUN / "inputs" / "hswud_reach_monthly.csv"
EPS = 1.0e-6


@dataclass
class Design:
    columns: list[str]
    scaler: StandardScaler


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom <= 0:
        return np.nan
    return 1.0 - float(np.sum((sim - obs) ** 2) / denom)


def kge_2012(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if len(obs) < 2 or np.std(obs) <= 0 or np.mean(obs) == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1] if np.std(sim) > 0 else 0.0
    beta = np.mean(sim) / np.mean(obs)
    cv_obs = np.std(obs) / np.mean(obs)
    cv_sim = np.std(sim) / np.mean(sim) if np.mean(sim) != 0 else np.nan
    gamma = cv_sim / cv_obs if cv_obs != 0 else np.nan
    if not np.isfinite(r) or not np.isfinite(beta) or not np.isfinite(gamma):
        return np.nan
    return 1.0 - float(np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2))


def station_metrics(df: pd.DataFrame, pred_col: str, split_name: str, variant: str) -> pd.DataFrame:
    rows = []
    for (site, rid, rclass), g in df.groupby(["q_site", "reach_id", "reach_class"], sort=False):
        obs = g["Q_obsv_cfs"].to_numpy(dtype=float)
        sim = g[pred_col].to_numpy(dtype=float)
        valid = np.isfinite(obs) & np.isfinite(sim) & (obs > 0) & (sim > 0)
        obs = obs[valid]
        sim = sim[valid]
        if len(obs) < 12:
            continue
        nse_raw = nse(obs, sim)
        nse_log = nse(np.log(obs + EPS), np.log(sim + EPS))
        kge = kge_2012(obs, sim)
        pbias = float((np.sum(sim - obs) / np.sum(obs)) * 100.0)
        rows.append(
            {
                "q_site": site,
                "reach_id": int(rid),
                "reach_class": rclass,
                "variant": variant,
                "split": split_name,
                "n": int(len(obs)),
                "NSE_raw": nse_raw,
                "NSE_log": nse_log,
                "KGE_2012": kge,
                "PBIAS_pct": pbias,
                "abs_PBIAS": abs(pbias),
                "good": bool(np.isfinite(nse_log) and np.isfinite(kge) and nse_log >= 0.65 and kge >= 0.50 and abs(pbias) <= 25.0),
            }
        )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["variant", "split"], as_index=False)
        .agg(
            stations=("q_site", "nunique"),
            median_NSElog=("NSE_log", "median"),
            median_KGE=("KGE_2012", "median"),
            median_absPBIAS=("abs_PBIAS", "median"),
            good_count=("good", "sum"),
        )
        .sort_values(["split", "median_NSElog"], ascending=[True, False])
    )


def load_panel() -> pd.DataFrame:
    pred = pd.read_csv(BASE_REPORT / "reach_class_selected_predictions_long.csv", encoding="utf-8-sig")
    hswud = pd.read_csv(HSWUD_CSV, encoding="utf-8-sig")
    pred["reach_id"] = pred["reach_id"].astype(int)
    hswud["reach_id"] = hswud["reach_id"].astype(int)
    df = pred.merge(hswud, on=["reach_id", "year", "month"], how="left")
    hcols = [c for c in df.columns if c.startswith("hswud_")]
    df[hcols] = df[hcols].fillna(0.0)
    df["period3"] = np.select(
        [df["year"].between(2010, 2015), df["year"].between(2016, 2018), df["year"].between(2019, 2022)],
        ["calibration_2010_2015", "inner_2016_2018", "validation_2019_2022"],
        default="other",
    )
    return df[df["period3"].ne("other")].copy()


def add_hswud_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    groups = ["irrigation", "urban_industrial", "thermal"]
    for group in groups:
        for scope in ["loc", "up"]:
            w = out[f"hswud_{group}_{scope}_cfs"].clip(lower=0.0)
            out[f"H_{group}_{scope}"] = np.log1p(w / (out["Q72_pred_cfs"].clip(lower=1.0)))
            logw = np.log1p(w)
            train_mean = (
                out[out["period3"].eq("calibration_2010_2015")]
                .groupby(["reach_id", "month"])[f"hswud_{group}_{scope}_cfs"]
                .transform(lambda x: np.nan)
            )
            # Build a stable reach-month climatology from calibration years, then map it to all rows.
            clim = (
                out[out["period3"].eq("calibration_2010_2015")]
                .assign(_logw=logw[out["period3"].eq("calibration_2010_2015")].to_numpy())
                .groupby(["reach_id", "month"])["_logw"]
                .mean()
            )
            out[f"A_{group}_{scope}"] = logw - out.set_index(["reach_id", "month"]).index.map(clim).fillna(logw.median()).to_numpy()
            del train_mean
    return out


def build_residual_design(df: pd.DataFrame, fit_mask: pd.Series) -> tuple[np.ndarray, Design]:
    base = [c for c in df.columns if c.startswith("H_") or c.startswith("A_")]
    classes = sorted(df["reach_class"].dropna().unique().tolist())
    design = pd.DataFrame(index=df.index)
    for c in classes:
        flag = (df["reach_class"] == c).astype(float)
        design[f"class_{c}"] = flag
        for b in base:
            design[f"{b}__{c}"] = df[b].astype(float) * flag
    design = design.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(design.loc[fit_mask])
    x = scaler.transform(design)
    return x, Design(columns=design.columns.tolist(), scaler=scaler)


def fit_q72_residual(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df["period3"].eq("calibration_2010_2015")
    inner = df["period3"].eq("inner_2016_2018")
    x, design = build_residual_design(df, train)
    y = np.log(df["Q_obsv_cfs"].clip(lower=EPS)) - np.log(df["Q72_pred_cfs"].clip(lower=EPS))
    alpha_grid = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    rows = []
    best_alpha = alpha_grid[0]
    best_score = -np.inf
    best_pred = None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x[train], y[train])
        ehat = model.predict(x)
        q = df["Q72_pred_cfs"].to_numpy(dtype=float) * np.exp(ehat)
        tmp = df.copy()
        tmp["Q72H_pred_cfs"] = np.clip(q, EPS, None)
        m = station_metrics(tmp[inner], "Q72H_pred_cfs", "inner_2016_2018", f"q72_hswud_residual_alpha_{alpha:g}")
        score = float(m["NSE_log"].median()) if not m.empty else -np.inf
        rows.append(
            {
                "alpha": alpha,
                "inner_median_NSElog": score,
                "inner_median_KGE": float(m["KGE_2012"].median()) if not m.empty else np.nan,
                "inner_good_count": int(m["good"].sum()) if not m.empty else 0,
            }
        )
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_pred = tmp["Q72H_pred_cfs"].to_numpy(dtype=float)
    out = df.copy()
    out["Q72H_pred_cfs"] = best_pred
    coef = pd.DataFrame({"feature": design.columns, "coefficient": Ridge(alpha=best_alpha).fit(x[train], y[train]).coef_})
    coef["selected_alpha"] = best_alpha
    coef.to_csv(REPORT / "q72_hswud_residual_coefficients.csv", index=False, encoding="utf-8-sig")
    return out, pd.DataFrame(rows)


def mass_design(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[tuple[float, float]]]:
    classes = sorted(df["reach_class"].dropna().unique().tolist())
    groups = [
        ("irrigation", 0.0, 0.80),
        ("urban_industrial", 0.0, 0.35),
        ("thermal", 0.0, 0.05),
    ]
    cols = []
    bounds = []
    parts = []
    for c in classes:
        flag = (df["reach_class"] == c).astype(float).to_numpy()
        for group, lo, hi in groups:
            cols.append(f"delta_{group}__{c}")
            bounds.append((lo, hi))
            parts.append(df[f"hswud_{group}_loc_cfs"].to_numpy(dtype=float) * flag)
    return np.vstack(parts).T, cols, bounds


def fit_mass_depletion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df["period3"].eq("calibration_2010_2015").to_numpy()
    inner = df["period3"].eq("inner_2016_2018")
    x, cols, bounds = mass_design(df)
    obs_log = np.log(df["Q_obsv_cfs"].clip(lower=EPS).to_numpy(dtype=float))
    qmass = df["Q78_mass_cfs"].clip(lower=EPS).to_numpy(dtype=float)
    lambda_grid = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    rows = []
    best = None
    best_score = -np.inf
    best_params = None

    for lam in lambda_grid:
        def objective(params: np.ndarray) -> float:
            depletion = x[train] @ params
            pred = np.clip(qmass[train] - depletion, EPS, None)
            loss = np.mean((np.log(pred) - obs_log[train]) ** 2)
            penalty = lam * np.mean(params**2)
            return float(loss + penalty)

        x0 = np.array([(lo + hi) / 8.0 for lo, hi in bounds], dtype=float)
        res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
        params = res.x
        tmp = df.copy()
        tmp["Q78H_mass_cfs"] = np.clip(qmass - x @ params, EPS, None)
        m = station_metrics(tmp[inner], "Q78H_mass_cfs", "inner_2016_2018", f"mass_hswud_lambda_{lam:g}")
        score = float(m["NSE_log"].median()) if not m.empty else -np.inf
        rows.append(
            {
                "lambda": lam,
                "success": bool(res.success),
                "objective": float(res.fun),
                "inner_median_NSElog": score,
                "inner_median_KGE": float(m["KGE_2012"].median()) if not m.empty else np.nan,
                "inner_good_count": int(m["good"].sum()) if not m.empty else 0,
            }
        )
        if score > best_score:
            best_score = score
            best = tmp["Q78H_mass_cfs"].to_numpy(dtype=float)
            best_params = params

    out = df.copy()
    out["Q78H_mass_cfs"] = best
    coef = pd.DataFrame({"parameter": cols, "delta_effective_depletion": best_params})
    coef.to_csv(REPORT / "mass_hswud_effective_depletion_coefficients.csv", index=False, encoding="utf-8-sig")
    return out, pd.DataFrame(rows), coef


def select_fusion_alpha(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inner = df["period3"].eq("inner_2016_2018")
    alpha_grid = np.round(np.arange(0.0, 0.325, 0.025), 3)
    selected = {}
    rows = []
    for rclass, part in df[inner].groupby("reach_class"):
        best_alpha = 0.0
        best_score = -np.inf
        for alpha in alpha_grid:
            tmp = part.copy()
            tmp["QH_pred_cfs"] = np.exp(
                (1.0 - alpha) * np.log(tmp["Q72H_pred_cfs"].clip(lower=EPS))
                + alpha * np.log(tmp["Q78H_mass_cfs"].clip(lower=EPS))
            )
            m = station_metrics(tmp, "QH_pred_cfs", "inner_2016_2018", "hswud_fusion")
            score = float(m["NSE_log"].median()) if not m.empty else -np.inf
            rows.append(
                {
                    "reach_class": rclass,
                    "alpha": alpha,
                    "inner_median_NSElog": score,
                    "inner_good_count": int(m["good"].sum()) if not m.empty else 0,
                }
            )
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
        selected[rclass] = best_alpha
    out = df.copy()
    out["hswud_alpha"] = out["reach_class"].map(selected).fillna(0.0)
    out["QH_pred_cfs"] = np.exp(
        (1.0 - out["hswud_alpha"]) * np.log(out["Q72H_pred_cfs"].clip(lower=EPS))
        + out["hswud_alpha"] * np.log(out["Q78H_mass_cfs"].clip(lower=EPS))
    )
    selected_df = pd.DataFrame({"reach_class": list(selected.keys()), "hswud_alpha": list(selected.values())})
    return out, pd.DataFrame(rows).merge(selected_df, on="reach_class", how="left")


def evaluate_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = [
        ("baseline_q72", "Q72_pred_cfs"),
        ("baseline_mass", "Q78_mass_cfs"),
        ("baseline_main", "Q_pred_cfs"),
        ("q72_hswud_residual", "Q72H_pred_cfs"),
        ("mass_hswud_depletion", "Q78H_mass_cfs"),
        ("hswud_fusion", "QH_pred_cfs"),
    ]
    splits = {
        "inner_2016_2018": df["period3"].eq("inner_2016_2018"),
        "validation_2019_2022": df["period3"].eq("validation_2019_2022"),
    }
    all_metrics = []
    for split, mask in splits.items():
        for variant, col in variants:
            all_metrics.append(station_metrics(df[mask], col, split, variant))
    metrics = pd.concat(all_metrics, ignore_index=True)
    return metrics, summarize(metrics)


def diagnostic_slices(df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    val = df[df["period3"].eq("validation_2019_2022")].copy()
    pressure = (
        val.groupby(["q_site", "reach_id"], as_index=False)
        .agg(
            mean_total_pressure=("H_irrigation_up", "mean"),
            mean_total_up_cfs=("hswud_total_up_cfs", "mean"),
            mean_irr_up_cfs=("hswud_irrigation_up_cfs", "mean"),
            mean_urban_up_cfs=("hswud_urban_industrial_up_cfs", "mean"),
        )
    )
    base = metrics[(metrics["variant"].eq("baseline_main")) & (metrics["split"].eq("validation_2019_2022"))]
    new = metrics[(metrics["variant"].eq("hswud_fusion")) & (metrics["split"].eq("validation_2019_2022"))]
    comp = base[["q_site", "reach_id", "NSE_log", "KGE_2012", "PBIAS_pct", "good"]].merge(
        new[["q_site", "reach_id", "NSE_log", "KGE_2012", "PBIAS_pct", "good"]],
        on=["q_site", "reach_id"],
        suffixes=("_baseline", "_hswud"),
    )
    comp["delta_NSElog"] = comp["NSE_log_hswud"] - comp["NSE_log_baseline"]
    comp["delta_absPBIAS"] = comp["PBIAS_pct_hswud"].abs() - comp["PBIAS_pct_baseline"].abs()
    comp = comp.merge(pressure, on=["q_site", "reach_id"], how="left")
    comp.to_csv(REPORT / "station_delta_vs_mainline.csv", index=False, encoding="utf-8-sig")
    return comp


def write_figures(summary: pd.DataFrame, comp: pd.DataFrame, df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    val = summary[summary["split"].eq("validation_2019_2022")].copy()
    order = ["baseline_q72", "baseline_main", "q72_hswud_residual", "mass_hswud_depletion", "hswud_fusion", "baseline_mass"]
    val["variant"] = pd.Categorical(val["variant"], categories=order, ordered=True)
    val = val.sort_values("variant")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "SimSun", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 350,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    axes[0].bar(val["variant"].astype(str), val["median_NSElog"], color="#2F6B9A")
    axes[0].set_ylabel("Median NSElog")
    axes[1].bar(val["variant"].astype(str), val["median_KGE"], color="#5C946E")
    axes[1].set_ylabel("Median KGE")
    axes[2].bar(val["variant"].astype(str), val["good_count"], color="#D9822B")
    axes[2].set_ylabel("Good stations")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "hswud_model_validation_summary.png", bbox_inches="tight")
    fig.savefig(FIG / "hswud_model_validation_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    selected = comp.sort_values("mean_total_up_cfs", ascending=False)["q_site"].head(8).tolist()
    selected += comp.sort_values("delta_NSElog", ascending=False)["q_site"].head(4).tolist()
    selected = list(dict.fromkeys(selected))[:12]
    fig, axes = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
    val_panel = df[df["period3"].eq("validation_2019_2022")].copy()
    val_panel["date"] = pd.to_datetime({"year": val_panel["year"], "month": val_panel["month"], "day": 1})
    for ax, site in zip(axes.flat, selected):
        part = val_panel[val_panel["q_site"].eq(site)].sort_values("date")
        if part.empty:
            ax.axis("off")
            continue
        meta = comp[comp["q_site"].eq(site)].iloc[0]
        ax.plot(part["date"], part["Q_obsv_cfs"], color="#111111", lw=0.9, label="Obs")
        ax.plot(part["date"], part["Q_pred_cfs"], color="#777777", lw=0.85, label="Main")
        ax.plot(part["date"], part["QH_pred_cfs"], color="#2F6B9A", lw=0.95, label="HSWUD")
        ax.set_yscale("log")
        ax.set_title(f"{site}\nDelta NSElog {meta['delta_NSElog']:+.2f}", fontsize=8)
    for ax in axes.flat[len(selected) :]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "hswud_high_pressure_station_hydrographs.png", bbox_inches="tight")
    fig.savefig(FIG / "hswud_high_pressure_station_hydrographs.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, comp: pd.DataFrame, q72_grid: pd.DataFrame, mass_grid: pd.DataFrame, alpha_grid: pd.DataFrame) -> None:
    val = summary[summary["split"].eq("validation_2019_2022")].copy()
    pivot = val.set_index("variant")
    base = pivot.loc["baseline_main"]
    h = pivot.loc["hswud_fusion"]
    q72h = pivot.loc["q72_hswud_residual"]
    massh = pivot.loc["mass_hswud_depletion"]
    improved = int((comp["delta_NSElog"] > 0).sum())
    worsened = int((comp["delta_NSElog"] < 0).sum())
    lines = [
        "# HSWUD Human Water Use Experiment",
        "",
        "## Method",
        "",
        "HSWUD gross withdrawals were not directly subtracted from observed or predicted discharge. They were first aggregated from grid cells to reach-local monthly withdrawals with equal-area catchment intersections, accumulated upstream through the reach topology, and transformed into pressure variables relative to the baseline natural-flow proxy.",
        "",
        "Three branches were tested with the same strict split rule: calibration 2010-2015, inner selection 2016-2018, strict validation 2019-2022.",
        "",
        "- `q72_hswud_residual`: HSWUD explains residuals of the high-skill base regression.",
        "- `mass_hswud_depletion`: HSWUD enters the mass branch as sector x reach-class effective net streamflow depletion.",
        "- `hswud_fusion`: log-space fusion of the two HSWUD-adjusted branches with reach-class alpha selected on 2016-2018 only.",
        "",
        "## Strict Validation Result",
        "",
        f"- baseline_main median NSElog={base.median_NSElog:.6f}, KGE={base.median_KGE:.6f}, median |PBIAS|={base.median_absPBIAS:.6f}, good={int(base.good_count)}.",
        f"- q72_hswud_residual median NSElog={q72h.median_NSElog:.6f}, KGE={q72h.median_KGE:.6f}, median |PBIAS|={q72h.median_absPBIAS:.6f}, good={int(q72h.good_count)}.",
        f"- mass_hswud_depletion median NSElog={massh.median_NSElog:.6f}, KGE={massh.median_KGE:.6f}, median |PBIAS|={massh.median_absPBIAS:.6f}, good={int(massh.good_count)}.",
        f"- hswud_fusion median NSElog={h.median_NSElog:.6f}, KGE={h.median_KGE:.6f}, median |PBIAS|={h.median_absPBIAS:.6f}, good={int(h.good_count)}.",
        "",
        f"Station-level NSElog improved at {improved} validation stations and worsened at {worsened} stations relative to the current mainline.",
        "",
        "## Interpretation",
        "",
    ]
    if h.median_NSElog > base.median_NSElog:
        lines.append("The HSWUD-adjusted fusion improves the global median validation NSElog. This suggests that human water-use pressure contains useful information beyond the current hydrologic feature regression and light mass-constraint branch.")
    else:
        lines.append("The HSWUD-adjusted fusion does not improve the global median validation NSElog. This does not prove HSWUD is useless; it indicates that this first conservative sector x reach-class representation is not yet a better global mainline than the current model.")
    lines.extend(
        [
            "",
            "The result should be read together with station-level deltas and high-pressure slices, because HSWUD is expected to help most in high-withdrawal, dry-season, irrigation, urban-industrial, and regulated reaches rather than uniformly across the basin.",
            "",
            "## Key Files",
            "",
            "- `inputs/hswud_reach_monthly.csv`",
            "- `reports/human_water_use_model/hswud_model_summary.csv`",
            "- `reports/human_water_use_model/station_delta_vs_mainline.csv`",
            "- `figure/human_water_use_model/hswud_model_validation_summary.png`",
            "- `figure/human_water_use_model/hswud_high_pressure_station_hydrographs.png`",
        ]
    )
    (REPORT / "hswud_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    q72_grid.to_csv(REPORT / "q72_hswud_residual_alpha_selection.csv", index=False, encoding="utf-8-sig")
    mass_grid.to_csv(REPORT / "mass_hswud_lambda_selection.csv", index=False, encoding="utf-8-sig")
    alpha_grid.to_csv(REPORT / "fusion_hswud_alpha_selection.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = add_hswud_features(load_panel())
    df, q72_grid = fit_q72_residual(df)
    df, mass_grid, _ = fit_mass_depletion(df)
    df, alpha_grid = select_fusion_alpha(df)
    metrics, summary = evaluate_all(df)
    comp = diagnostic_slices(df, metrics)
    df.to_csv(REPORT / "hswud_selected_predictions_long.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(REPORT / "hswud_station_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORT / "hswud_model_summary.csv", index=False, encoding="utf-8-sig")
    write_figures(summary, comp, df)
    write_report(summary, comp, q72_grid, mass_grid, alpha_grid)
    print(summary.to_string(index=False))
    print(f"Wrote {REPORT / 'hswud_experiment_report.md'}")


if __name__ == "__main__":
    main()
