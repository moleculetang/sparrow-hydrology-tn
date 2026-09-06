from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260620_7")
INPUT_OUTPUTS = ROOT / "inputs" / "source_outputs"
INPUT_FORCING = ROOT / "inputs" / "forcing"
REPORT_DIR = ROOT / "reports"
FIG_DIR = ROOT / "figure" / "hydro_response_candidate"
LOG_PATH = Path(r"E:\SPARROW\5_Test\20260620.log")
EPS = 1e-6


FEATURES = [
    "wet_gate",
    "dry_gate",
    "surplus_high_gate",
    "surplus_low_gate",
    "stress_gate",
]


def period(year: int) -> str:
    if 2010 <= year <= 2015:
        return "train"
    if 2016 <= year <= 2018:
        return "inner"
    if 2019 <= year <= 2021:
        return "strict"
    return "other"


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 3:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom <= 0:
        return np.nan
    return 1 - np.sum((obs - sim) ** 2) / denom


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 3 or np.std(obs) <= 0 or np.mean(obs) == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1] if np.std(sim) > 0 else np.nan
    if not np.isfinite(r):
        return np.nan
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) == 0 or np.sum(obs) == 0:
        return np.nan
    return 100 * np.sum(sim - obs) / np.sum(obs)


def df_to_md(df: pd.DataFrame, n: int | None = None, float_digits: int = 4) -> str:
    if n is not None:
        df = df.head(n)
    if df.empty:
        return "_No rows._"
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
        else:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(shown.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in shown[cols].astype(str).to_numpy()]
    return "\n".join([header, sep] + rows)


def add_forcing_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    train = data[data["period"] == "train"].copy()
    clim = train.groupby("reach_id").agg(
        ppt_mu=("PPT_rainfall2_mm", "mean"),
        ppt_sd=("PPT_rainfall2_mm", "std"),
        surplus_mu=("P_surplus_mm", "mean"),
        surplus_sd=("P_surplus_mm", "std"),
        vpd_mu=("VPD_cmfd_kpa", "mean"),
        vpd_sd=("VPD_cmfd_kpa", "std"),
        pet_mu=("PET_cmfd_mm", "mean"),
        pet_sd=("PET_cmfd_mm", "std"),
    ).reset_index()
    data = data.merge(clim, on="reach_id", how="left")
    for col in ["ppt_sd", "surplus_sd", "vpd_sd", "pet_sd"]:
        data[col] = data[col].fillna(data[col].median()).replace(0, np.nan).fillna(data[col].median())
    data["ppt_z"] = (data["PPT_rainfall2_mm"] - data["ppt_mu"]) / data["ppt_sd"]
    data["surplus_z"] = (data["P_surplus_mm"] - data["surplus_mu"]) / data["surplus_sd"]
    data["vpd_z"] = (data["VPD_cmfd_kpa"] - data["vpd_mu"]) / data["vpd_sd"]
    data["pet_z"] = (data["PET_cmfd_mm"] - data["pet_mu"]) / data["pet_sd"]
    data["wet_gate"] = np.maximum(data["ppt_z"] - 0.5, 0)
    data["dry_gate"] = np.maximum(-data["ppt_z"] - 0.5, 0)
    data["surplus_high_gate"] = np.maximum(data["surplus_z"] - 0.5, 0)
    data["surplus_low_gate"] = np.maximum(-data["surplus_z"] - 0.5, 0)
    data["stress_gate"] = np.maximum(np.maximum(data["vpd_z"], data["pet_z"]) - 0.5, 0)
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    return data


def fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[ok]
    y = y[ok]
    if len(y) < X.shape[1] + 5:
        return np.zeros(X.shape[1])
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def fit_coefficients(data: pd.DataFrame, mode: str, ridge: float, class_shrink: float) -> pd.DataFrame:
    train = data[data["period"] == "train"].copy()
    train["target"] = np.log(np.maximum(train["Q_obsv_cfs"], EPS)) - np.log(np.maximum(train["Q_pred_cfs"], EPS))
    Xg = np.column_stack([np.ones(len(train)), train[FEATURES].to_numpy(float)])
    bg = fit_ridge(Xg, train["target"].to_numpy(float), ridge)
    rows = []
    if mode == "global":
        rows.append({"coefficient_group": "GLOBAL", **{f"b_{name}": bg[i + 1] for i, name in enumerate(FEATURES)}, "b_intercept": bg[0]})
        return pd.DataFrame(rows)
    for cls, g in train.groupby("reach_class", dropna=False):
        Xc = np.column_stack([np.ones(len(g)), g[FEATURES].to_numpy(float)])
        bc_raw = fit_ridge(Xc, g["target"].to_numpy(float), ridge)
        bc = class_shrink * bc_raw + (1 - class_shrink) * bg
        rows.append({"coefficient_group": str(cls), **{f"b_{name}": bc[i + 1] for i, name in enumerate(FEATURES)}, "b_intercept": bc[0]})
    return pd.DataFrame(rows)


def apply_coefficients(data: pd.DataFrame, coeffs: pd.DataFrame, mode: str, clip_max: float, variant: str) -> pd.DataFrame:
    out = data.copy()
    if mode == "global":
        row = coeffs.iloc[0]
        beta = np.array([row["b_intercept"]] + [row[f"b_{f}"] for f in FEATURES], dtype=float)
        X = np.column_stack([np.ones(len(out)), out[FEATURES].to_numpy(float)])
        delta = X @ beta
    else:
        delta = np.zeros(len(out))
        coeff_map = {str(r["coefficient_group"]): r for _, r in coeffs.iterrows()}
        for cls, idx in out.groupby("reach_class", dropna=False).groups.items():
            r = coeff_map.get(str(cls))
            if r is None:
                continue
            beta = np.array([r["b_intercept"]] + [r[f"b_{f}"] for f in FEATURES], dtype=float)
            X = np.column_stack([np.ones(len(idx)), out.loc[idx, FEATURES].to_numpy(float)])
            delta[out.index.get_indexer(idx)] = X @ beta
    delta = np.clip(delta, -clip_max, clip_max)
    out[f"log_delta_{variant}"] = delta
    out[f"Q_candidate_{variant}_cfs"] = np.exp(np.log(np.maximum(out["Q_pred_cfs"], EPS)) + delta)
    return out


def station_metrics(df: pd.DataFrame, pred_col: str, variant: str, eval_period: str) -> pd.DataFrame:
    rows = []
    d = df[df["period"] == eval_period].copy()
    for site, g in d.groupby("q_site", sort=False):
        obs = g["Q_obsv_cfs"].to_numpy(float)
        sim = g[pred_col].to_numpy(float)
        log_obs = np.log(np.maximum(obs, EPS))
        log_sim = np.log(np.maximum(sim, EPS))
        row = {
            "variant": variant,
            "eval_period": eval_period,
            "q_site": site,
            "reach_id": g["reach_id"].iloc[0],
            "reach_class": g["reach_class"].iloc[0],
            "reservoir_relation": g["reservoir_relation"].iloc[0],
            "NSE_raw": nse(obs, sim),
            "NSElog": nse(log_obs, log_sim),
            "KGE_2012": kge(obs, sim),
            "PBIAS_pct": pbias(obs, sim),
        }
        row["good"] = bool(row["NSE_raw"] >= 0.5 and row["KGE_2012"] >= 0.5 and abs(row["PBIAS_pct"]) <= 25)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame, diag: pd.DataFrame, variant: str, eval_period: str) -> dict:
    bad_sites = set(diag.loc[diag["is_bad"], "q_site"].astype(str))
    very_bad_sites = set(diag.loc[diag["is_very_bad"], "q_site"].astype(str))
    out = {"variant": variant, "eval_period": eval_period}
    for label, sites in {
        "all": set(metrics["q_site"].astype(str)),
        "bad": bad_sites,
        "very_bad": very_bad_sites,
    }.items():
        m = metrics[metrics["q_site"].astype(str).isin(sites)].copy()
        out[f"{label}_n"] = int(len(m))
        out[f"{label}_good"] = int(m["good"].sum())
        out[f"{label}_median_NSE_raw"] = float(m["NSE_raw"].median()) if len(m) else np.nan
        out[f"{label}_median_NSElog"] = float(m["NSElog"].median()) if len(m) else np.nan
        out[f"{label}_median_KGE_2012"] = float(m["KGE_2012"].median()) if len(m) else np.nan
        out[f"{label}_median_abs_PBIAS"] = float(m["PBIAS_pct"].abs().median()) if len(m) else np.nan
    return out


def make_figures(data: pd.DataFrame, selected_variant: str, diag: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    col = f"Q_candidate_{selected_variant}_cfs"
    examples = diag.sort_values("badness_score", ascending=False).head(12)["q_site"].tolist()
    for site in examples:
        g = data[data["q_site"] == site].sort_values("date")
        if g.empty or col not in g:
            continue
        plt.figure(figsize=(11, 4.8))
        plt.plot(g["date"], g["Q_obsv_cfs"], color="black", linewidth=2, label="Observed")
        plt.plot(g["date"], g["Q_pred_cfs"], color="#4c78a8", linewidth=1.2, label="Current")
        plt.plot(g["date"], g[col], color="#f58518", linewidth=1.2, label="Candidate")
        plt.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31"), color="#f0f0f0", alpha=0.7)
        plt.title(site)
        plt.ylabel("Q (cfs)")
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(site))
        plt.savefig(FIG_DIR / f"{safe}_selected_candidate.png", dpi=180)
        plt.close()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(INPUT_OUTPUTS / "current_main_predictions_2010_2021.csv", encoding="utf-8-sig", low_memory=False)
    diag = pd.read_csv(INPUT_OUTPUTS / "strict_station_diagnostics_all.csv", encoding="utf-8-sig")
    ppt = pd.read_csv(INPUT_FORCING / "chm_pre_v2_monthly_by_reach_2006_2022.csv", encoding="utf-8-sig")
    cmfd = pd.read_csv(INPUT_FORCING / "cmfd_monthly_by_reach_2006_2022.csv", encoding="utf-8-sig")

    force = ppt.merge(cmfd, on=["reach_id", "year", "month"], how="left")
    force["P_surplus_mm"] = force["PPT_rainfall2_mm"] - force["PET_cmfd_mm"]
    data = pred[[
        "q_site", "reach_id", "reach_class", "year", "month", "date", "Q_obsv_cfs",
        "Q_pred_cfs", "Q72_pred_cfs", "Q78_mass_cfs", "reservoir_relation",
    ]].merge(force[["reach_id", "year", "month", "PPT_rainfall2_mm", "PET_cmfd_mm", "VPD_cmfd_kpa", "P_surplus_mm"]], on=["reach_id", "year", "month"], how="left")
    data["date"] = pd.to_datetime(data["date"])
    data["period"] = data["year"].map(period)
    data = add_forcing_features(data[data["period"].isin(["train", "inner", "strict"])].copy())

    variants = []
    for mode in ["global", "class"]:
        for ridge in [10.0, 50.0, 200.0]:
            for clip_max in [0.05, 0.10, 0.20]:
                if mode == "global":
                    variants.append({"mode": mode, "ridge": ridge, "clip": clip_max, "class_shrink": 0.0})
                else:
                    for shrink in [0.25, 0.50]:
                        variants.append({"mode": mode, "ridge": ridge, "clip": clip_max, "class_shrink": shrink})

    working = data.copy()
    coeff_records = []
    metric_frames = []
    summary_rows = []
    base_inner = station_metrics(working, "Q_pred_cfs", "current_main", "inner")
    base_strict = station_metrics(working, "Q_pred_cfs", "current_main", "strict")
    metric_frames += [base_inner, base_strict]
    summary_rows += [
        summarize(base_inner, diag, "current_main", "inner"),
        summarize(base_strict, diag, "current_main", "strict"),
    ]

    for spec in variants:
        name = f"{spec['mode']}_ridge{int(spec['ridge'])}_clip{str(spec['clip']).replace('.', 'p')}_shrink{str(spec['class_shrink']).replace('.', 'p')}"
        coeffs = fit_coefficients(working, spec["mode"], spec["ridge"], spec["class_shrink"])
        coeffs["variant"] = name
        coeffs["mode"] = spec["mode"]
        coeffs["ridge"] = spec["ridge"]
        coeffs["clip"] = spec["clip"]
        coeffs["class_shrink"] = spec["class_shrink"]
        coeff_records.append(coeffs)
        working = apply_coefficients(working, coeffs, spec["mode"], spec["clip"], name)
        col = f"Q_candidate_{name}_cfs"
        for ep in ["inner", "strict"]:
            m = station_metrics(working, col, name, ep)
            metric_frames.append(m)
            summary_rows.append(summarize(m, diag, name, ep))

    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    coeff_all = pd.concat(coeff_records, ignore_index=True)

    # Select only on inner all-station behavior, not strict bad labels.
    inner = summary[summary["eval_period"] == "inner"].copy()
    base_inner_row = inner[inner["variant"] == "current_main"].iloc[0]
    inner["selection_score"] = (
        (inner["all_good"] - base_inner_row["all_good"]) * 1.0
        + (inner["all_median_NSE_raw"] - base_inner_row["all_median_NSE_raw"]) * 10.0
        + (inner["all_median_KGE_2012"] - base_inner_row["all_median_KGE_2012"]) * 5.0
        - np.maximum(inner["all_median_abs_PBIAS"] - base_inner_row["all_median_abs_PBIAS"], 0) * 0.05
    )
    inner = inner.sort_values(["selection_score", "all_good", "all_median_NSE_raw"], ascending=False)
    selected = inner.iloc[0]["variant"]
    best_nonbaseline = inner[inner["variant"] != "current_main"].iloc[0]["variant"]
    report_variant = selected if selected != "current_main" else best_nonbaseline
    selected_strict = summary[(summary["variant"] == selected) & (summary["eval_period"] == "strict")].iloc[0]
    report_variant_strict = summary[(summary["variant"] == report_variant) & (summary["eval_period"] == "strict")].iloc[0]
    base_strict_row = summary[(summary["variant"] == "current_main") & (summary["eval_period"] == "strict")].iloc[0]

    selected_metrics = metrics[(metrics["variant"].isin(["current_main", report_variant])) & (metrics["eval_period"] == "strict")].copy()
    base_by_site = selected_metrics[selected_metrics["variant"] == "current_main"].set_index("q_site")
    cand_by_site = selected_metrics[selected_metrics["variant"] == report_variant].set_index("q_site")
    delta = cand_by_site[["NSE_raw", "NSElog", "KGE_2012", "PBIAS_pct", "good"]].join(
        base_by_site[["NSE_raw", "NSElog", "KGE_2012", "PBIAS_pct", "good"]],
        lsuffix="_candidate",
        rsuffix="_base",
    )
    delta["q_site"] = delta.index
    delta["delta_NSE_raw"] = delta["NSE_raw_candidate"] - delta["NSE_raw_base"]
    delta["delta_NSElog"] = delta["NSElog_candidate"] - delta["NSElog_base"]
    delta["delta_KGE_2012"] = delta["KGE_2012_candidate"] - delta["KGE_2012_base"]
    delta["delta_abs_PBIAS"] = delta["PBIAS_pct_candidate"].abs() - delta["PBIAS_pct_base"].abs()
    delta["good_gain"] = delta["good_candidate"].astype(int) - delta["good_base"].astype(int)
    bad_sites = set(diag.loc[diag["is_bad"], "q_site"].astype(str))
    very_bad_sites = set(diag.loc[diag["is_very_bad"], "q_site"].astype(str))
    bad_delta = delta[delta["q_site"].astype(str).isin(bad_sites)].reset_index(drop=True)
    very_bad_delta = delta[delta["q_site"].astype(str).isin(very_bad_sites)].reset_index(drop=True)

    continuation = (
        selected != "current_main"
        and selected_strict["bad_good"] >= base_strict_row["bad_good"] + 3
        and selected_strict["bad_median_NSE_raw"] >= base_strict_row["bad_median_NSE_raw"] + 0.03
        and selected_strict["very_bad_median_NSE_raw"] >= base_strict_row["very_bad_median_NSE_raw"] + 0.03
        and selected_strict["all_median_abs_PBIAS"] <= base_strict_row["all_median_abs_PBIAS"] + 3.0
        and int((bad_delta["delta_NSE_raw"] > 0.05).sum()) >= 5
    )

    summary.to_csv(REPORT_DIR / "variant_summary_inner_strict.csv", index=False, encoding="utf-8-sig")
    inner.to_csv(REPORT_DIR / "inner_selection_table.csv", index=False, encoding="utf-8-sig")
    coeff_all[coeff_all["variant"] == report_variant].to_csv(REPORT_DIR / "selected_coefficients.csv", index=False, encoding="utf-8-sig")
    selected_metrics.to_csv(REPORT_DIR / "strict_station_metrics_selected.csv", index=False, encoding="utf-8-sig")
    bad_delta.to_csv(REPORT_DIR / "strict_bad_station_delta_selected.csv", index=False, encoding="utf-8-sig")
    very_bad_delta.to_csv(REPORT_DIR / "strict_very_bad_station_delta_selected.csv", index=False, encoding="utf-8-sig")
    working.to_csv(REPORT_DIR / "candidate_predictions_long.csv", index=False, encoding="utf-8-sig")

    make_figures(working, report_variant, diag)

    selected_summary = summary[summary["variant"].isin(["current_main", report_variant])].sort_values(["eval_period", "variant"])
    top_bad_delta = bad_delta.sort_values("delta_NSE_raw", ascending=False).head(15)
    worst_bad_delta = bad_delta.sort_values("delta_NSE_raw", ascending=True).head(10)

    if continuation:
        decision = "Continuation gate passed. This shared hydroclimate response idea deserves a formal mechanism-level process equation experiment."
    else:
        decision = "Continuation gate failed. Do not promote this bounded response operator as a model branch; use it only as evidence for process-equation redesign."

    report = [
        "# Shared Hydroclimate Response Candidate",
        "",
        "## Scope",
        "",
        "This branch tests shared hydroclimate-response parameters. It is not station-specific and does not use strict-period residuals for fitting or selection.",
        "",
        "## Selected Variant",
        "",
        f"- Selected by inner all-station score: `{selected}`",
        f"- Best non-baseline candidate retained for diagnostics: `{report_variant}`",
        f"- Continuation gate passed: `{bool(continuation)}`",
        "",
        "## Inner/Strict Summary",
        "",
        df_to_md(selected_summary),
        "",
        "## Strict Bad-Station Improvements",
        "",
        df_to_md(top_bad_delta[["q_site", "NSE_raw_base", "NSE_raw_candidate", "delta_NSE_raw", "KGE_2012_base", "KGE_2012_candidate", "PBIAS_pct_base", "PBIAS_pct_candidate", "good_gain"]]),
        "",
        "## Strict Bad-Station Worsening",
        "",
        df_to_md(worst_bad_delta[["q_site", "NSE_raw_base", "NSE_raw_candidate", "delta_NSE_raw", "KGE_2012_base", "KGE_2012_candidate", "PBIAS_pct_base", "PBIAS_pct_candidate", "good_gain"]]),
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Interpretation",
        "",
        "- If this candidate improves broadly, the next step should move the gates into the process equation or parameter-pooling layer.",
        "- If it fails, the diagnosis from `20260620_6` is still useful, but a bounded post-main response operator is not strong enough.",
        "- No station-specific switch or per-station parameter was used.",
        "",
    ]
    (REPORT_DIR / "hydro_response_candidate_report.md").write_text("\n".join(report), encoding="utf-8")

    with (ROOT / "README.md").open("a", encoding="utf-8") as f:
        f.write("\n## Run Result\n\n")
        f.write(f"- Selected variant: `{selected}`.\n")
        f.write(f"- Best non-baseline diagnostic candidate: `{report_variant}`.\n")
        f.write(f"- Continuation gate passed: `{bool(continuation)}`.\n")
        f.write(f"- Decision: {decision}\n")
        f.write("- See `reports/hydro_response_candidate_report.md`.\n")

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n## 20260620_7 shared hydroclimate response candidate\n")
        f.write("- Tested bounded shared hydroclimate response operators using train-fitted global/class parameters and inner all-station selection.\n")
        f.write(f"- Selected variant: {selected}; best non-baseline diagnostic candidate: {report_variant}; continuation gate passed: {bool(continuation)}.\n")
        f.write(
            f"- Strict current bad: good {int(base_strict_row['bad_good'])}/{int(base_strict_row['bad_n'])}, "
            f"median raw NSE {base_strict_row['bad_median_NSE_raw']:.4f}, KGE {base_strict_row['bad_median_KGE_2012']:.4f}, "
            f"|PBIAS| {base_strict_row['bad_median_abs_PBIAS']:.4f}.\n"
        )
        f.write(
            f"- Strict diagnostic candidate bad: good {int(report_variant_strict['bad_good'])}/{int(report_variant_strict['bad_n'])}, "
            f"median raw NSE {report_variant_strict['bad_median_NSE_raw']:.4f}, KGE {report_variant_strict['bad_median_KGE_2012']:.4f}, "
            f"|PBIAS| {report_variant_strict['bad_median_abs_PBIAS']:.4f}.\n"
        )
        f.write(f"- Decision: {decision}\n")
        f.write("- No station-specific correction was used.\n")

    print("Hydro response candidate complete:", ROOT)
    print("selected", selected)
    print("best_nonbaseline", report_variant)
    print("continuation", bool(continuation))
    print(selected_summary.to_string(index=False))


if __name__ == "__main__":
    main()
