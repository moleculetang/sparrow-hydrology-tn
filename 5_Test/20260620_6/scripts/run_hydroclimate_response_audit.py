from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260620_6")
INPUT_OUTPUTS = ROOT / "inputs" / "source_outputs"
INPUT_FORCING = ROOT / "inputs" / "forcing"
REPORT_DIR = ROOT / "reports"
FIG_DIR = ROOT / "figure" / "hydroclimate_response"
LOG_PATH = Path(r"E:\SPARROW\5_Test\20260620.log")
EPS = 1e-6


def period(year: int) -> str:
    if 2010 <= year <= 2015:
        return "train"
    if 2016 <= year <= 2018:
        return "inner"
    if 2019 <= year <= 2021:
        return "strict"
    return "other"


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    if len(obs) == 0 or np.sum(obs) == 0:
        return np.nan
    return 100.0 * np.sum(sim - obs) / np.sum(obs)


def slope(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 8 or np.std(x) <= 0:
        return np.nan
    b, _ = np.polyfit(x, y, 1)
    return float(b)


def corr(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 8 or np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


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


def station_response_metrics(g: pd.DataFrame) -> dict:
    g = g.sort_values("date").copy()
    strict = g[g["period"] == "strict"].copy()
    calib = g[g["period"].isin(["train", "inner"])].copy()
    if strict.empty:
        return {}

    obs = strict["Q_obsv_cfs"].to_numpy(float)
    sim = strict["Q_pred_cfs"].to_numpy(float)
    log_obs = np.log(np.maximum(obs, EPS))
    log_sim = np.log(np.maximum(sim, EPS))

    # Station-local thresholds are used only for evaluation stratification, not model training.
    ppt_q25 = calib["PPT_rainfall2_mm"].quantile(0.25)
    ppt_q75 = calib["PPT_rainfall2_mm"].quantile(0.75)
    surplus_q25 = calib["P_surplus_mm"].quantile(0.25)
    surplus_q75 = calib["P_surplus_mm"].quantile(0.75)
    vpd_q75 = calib["VPD_cmfd_kpa"].quantile(0.75)
    pet_q75 = calib["PET_cmfd_mm"].quantile(0.75)

    wet = strict["PPT_rainfall2_mm"] >= ppt_q75
    dry = strict["PPT_rainfall2_mm"] <= ppt_q25
    surplus_high = strict["P_surplus_mm"] >= surplus_q75
    surplus_low = strict["P_surplus_mm"] <= surplus_q25
    stress = (strict["VPD_cmfd_kpa"] >= vpd_q75) | (strict["PET_cmfd_mm"] >= pet_q75)

    def reg_p(mask: pd.Series) -> float:
        if int(mask.sum()) < 3:
            return np.nan
        return pbias(strict.loc[mask, "Q_obsv_cfs"].to_numpy(float), strict.loc[mask, "Q_pred_cfs"].to_numpy(float))

    z_ppt = (strict["PPT_rainfall2_mm"] - calib["PPT_rainfall2_mm"].mean()) / max(calib["PPT_rainfall2_mm"].std(), EPS)
    z_surplus = (strict["P_surplus_mm"] - calib["P_surplus_mm"].mean()) / max(calib["P_surplus_mm"].std(), EPS)
    z_pet = (strict["PET_cmfd_mm"] - calib["PET_cmfd_mm"].mean()) / max(calib["PET_cmfd_mm"].std(), EPS)
    z_vpd = (strict["VPD_cmfd_kpa"] - calib["VPD_cmfd_kpa"].mean()) / max(calib["VPD_cmfd_kpa"].std(), EPS)

    obs_ppt_slope = slope(log_obs, z_ppt.to_numpy(float))
    sim_ppt_slope = slope(log_sim, z_ppt.to_numpy(float))
    obs_surplus_slope = slope(log_obs, z_surplus.to_numpy(float))
    sim_surplus_slope = slope(log_sim, z_surplus.to_numpy(float))
    obs_pet_slope = slope(log_obs, z_pet.to_numpy(float))
    sim_pet_slope = slope(log_sim, z_pet.to_numpy(float))
    obs_vpd_slope = slope(log_obs, z_vpd.to_numpy(float))
    sim_vpd_slope = slope(log_sim, z_vpd.to_numpy(float))

    return {
        "strict_months": int(len(strict)),
        "strict_overall_PBIAS_pct": pbias(obs, sim),
        "wet_month_count": int(wet.sum()),
        "dry_month_count": int(dry.sum()),
        "surplus_high_count": int(surplus_high.sum()),
        "surplus_low_count": int(surplus_low.sum()),
        "stress_month_count": int(stress.sum()),
        "wet_PBIAS_pct": reg_p(wet),
        "dry_PBIAS_pct": reg_p(dry),
        "surplus_high_PBIAS_pct": reg_p(surplus_high),
        "surplus_low_PBIAS_pct": reg_p(surplus_low),
        "climate_stress_PBIAS_pct": reg_p(stress),
        "obs_logQ_ppt_slope": obs_ppt_slope,
        "sim_logQ_ppt_slope": sim_ppt_slope,
        "delta_logQ_ppt_slope": sim_ppt_slope - obs_ppt_slope if pd.notna(sim_ppt_slope) and pd.notna(obs_ppt_slope) else np.nan,
        "obs_logQ_surplus_slope": obs_surplus_slope,
        "sim_logQ_surplus_slope": sim_surplus_slope,
        "delta_logQ_surplus_slope": sim_surplus_slope - obs_surplus_slope if pd.notna(sim_surplus_slope) and pd.notna(obs_surplus_slope) else np.nan,
        "obs_logQ_pet_slope": obs_pet_slope,
        "sim_logQ_pet_slope": sim_pet_slope,
        "delta_logQ_pet_slope": sim_pet_slope - obs_pet_slope if pd.notna(sim_pet_slope) and pd.notna(obs_pet_slope) else np.nan,
        "obs_logQ_vpd_slope": obs_vpd_slope,
        "sim_logQ_vpd_slope": sim_vpd_slope,
        "delta_logQ_vpd_slope": sim_vpd_slope - obs_vpd_slope if pd.notna(sim_vpd_slope) and pd.notna(obs_vpd_slope) else np.nan,
        "obs_corr_logQ_ppt": corr(log_obs, z_ppt.to_numpy(float)),
        "sim_corr_logQ_ppt": corr(log_sim, z_ppt.to_numpy(float)),
        "obs_corr_logQ_surplus": corr(log_obs, z_surplus.to_numpy(float)),
        "sim_corr_logQ_surplus": corr(log_sim, z_surplus.to_numpy(float)),
    }


def assign_flags(row: pd.Series) -> str:
    flags = []
    if pd.notna(row.get("wet_PBIAS_pct")) and row["wet_PBIAS_pct"] >= 30:
        flags.append("wet_month_overprediction")
    if pd.notna(row.get("wet_PBIAS_pct")) and row["wet_PBIAS_pct"] <= -30:
        flags.append("wet_month_underprediction")
    if pd.notna(row.get("dry_PBIAS_pct")) and row["dry_PBIAS_pct"] >= 30:
        flags.append("dry_month_overprediction")
    if pd.notna(row.get("dry_PBIAS_pct")) and row["dry_PBIAS_pct"] <= -30:
        flags.append("dry_month_underprediction")
    if pd.notna(row.get("surplus_high_PBIAS_pct")) and row["surplus_high_PBIAS_pct"] >= 30:
        flags.append("high_surplus_overprediction")
    if pd.notna(row.get("surplus_high_PBIAS_pct")) and row["surplus_high_PBIAS_pct"] <= -30:
        flags.append("high_surplus_underprediction")
    if pd.notna(row.get("climate_stress_PBIAS_pct")) and row["climate_stress_PBIAS_pct"] >= 30:
        flags.append("stress_month_overprediction")
    if pd.notna(row.get("climate_stress_PBIAS_pct")) and row["climate_stress_PBIAS_pct"] <= -30:
        flags.append("stress_month_underprediction")
    if pd.notna(row.get("delta_logQ_ppt_slope")) and row["delta_logQ_ppt_slope"] >= 0.25:
        flags.append("rainfall_elasticity_too_high")
    if pd.notna(row.get("delta_logQ_ppt_slope")) and row["delta_logQ_ppt_slope"] <= -0.25:
        flags.append("rainfall_elasticity_too_low")
    if pd.notna(row.get("delta_logQ_surplus_slope")) and row["delta_logQ_surplus_slope"] >= 0.25:
        flags.append("surplus_elasticity_too_high")
    if pd.notna(row.get("delta_logQ_surplus_slope")) and row["delta_logQ_surplus_slope"] <= -0.25:
        flags.append("surplus_elasticity_too_low")
    if not flags and row.get("is_bad", False):
        flags.append("bad_but_no_clear_hydroclimate_flag")
    return ";".join(flags)


def make_figures(audit: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bad_group = np.where(audit["is_bad"], "bad", "not bad")
    plot = audit.copy()
    plot["bad_group"] = bad_group

    plt.figure(figsize=(8, 5))
    for label, g in plot.groupby("bad_group"):
        plt.scatter(g["delta_logQ_ppt_slope"], g["NSE_raw"], s=28, alpha=0.75, label=label)
    plt.axvline(0, color="black", linewidth=1)
    plt.axvline(-0.25, color="tab:orange", linestyle="--", linewidth=1)
    plt.axvline(0.25, color="tab:orange", linestyle="--", linewidth=1)
    plt.xlabel("simulated - observed logQ rainfall elasticity")
    plt.ylabel("strict raw NSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rainfall_elasticity_delta_vs_nse.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for label, g in plot.groupby("bad_group"):
        plt.scatter(g["wet_PBIAS_pct"], g["dry_PBIAS_pct"], s=28, alpha=0.75, label=label)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.axhline(30, color="tab:orange", linestyle="--", linewidth=1)
    plt.axhline(-30, color="tab:orange", linestyle="--", linewidth=1)
    plt.axvline(30, color="tab:orange", linestyle="--", linewidth=1)
    plt.axvline(-30, color="tab:orange", linestyle="--", linewidth=1)
    plt.xlabel("wet-month PBIAS (%)")
    plt.ylabel("dry-month PBIAS (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wet_vs_dry_pbias.png", dpi=180)
    plt.close()

    flag_cols = [
        "wet_month_overprediction", "wet_month_underprediction",
        "dry_month_overprediction", "dry_month_underprediction",
        "high_surplus_overprediction", "high_surplus_underprediction",
        "stress_month_overprediction", "stress_month_underprediction",
        "rainfall_elasticity_too_high", "rainfall_elasticity_too_low",
        "surplus_elasticity_too_high", "surplus_elasticity_too_low",
    ]
    rows = []
    for flag in flag_cols:
        for label, g in plot.groupby("bad_group"):
            rows.append({"flag": flag, "group": label, "rate": float(g[flag].mean()) if len(g) else np.nan})
    rates = pd.DataFrame(rows)
    pivot = rates.pivot(index="flag", columns="group", values="rate").fillna(0)
    pivot.plot(kind="bar", figsize=(12, 5), color=["#4c78a8", "#f58518"])
    plt.ylabel("station fraction")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hydroclimate_flag_rates_bad_vs_not_bad.png", dpi=180)
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
    force["aridity_ratio"] = force["PET_cmfd_mm"] / np.maximum(force["PPT_rainfall2_mm"], EPS)

    cols = [
        "reach_id", "year", "month", "PPT_rainfall2_mm", "PET_cmfd_mm", "ET0_cmfd_mm_day",
        "VPD_cmfd_kpa", "T2M_C_cmfd", "P_surplus_mm", "aridity_ratio",
    ]
    data = pred[[
        "q_site", "reach_id", "reach_class", "year", "month", "date", "Q_obsv_cfs",
        "Q_pred_cfs", "Q72_pred_cfs", "Q78_mass_cfs", "reservoir_relation",
    ]].merge(force[cols], on=["reach_id", "year", "month"], how="left")
    data["date"] = pd.to_datetime(data["date"])
    data["period"] = data["year"].map(period)
    data = data[data["period"].isin(["train", "inner", "strict"])].copy()

    rows = []
    for site, g in data.groupby("q_site", sort=False):
        metrics = station_response_metrics(g)
        if not metrics:
            continue
        first = g.iloc[0]
        metrics.update({
            "q_site": site,
            "reach_id": first["reach_id"],
            "reach_class": first["reach_class"],
            "reservoir_relation": first["reservoir_relation"],
        })
        rows.append(metrics)
    resp = pd.DataFrame(rows)
    audit = diag.merge(resp, on=["q_site", "reach_id", "reach_class", "reservoir_relation"], how="left")

    audit["hydroclimate_flags"] = audit.apply(assign_flags, axis=1)
    flag_names = sorted(set(";".join(audit["hydroclimate_flags"].fillna("")).split(";")) - {""})
    for flag in flag_names:
        audit[flag] = audit["hydroclimate_flags"].fillna("").str.contains(flag, regex=False)

    scored_flags = [f for f in flag_names if f != "bad_but_no_clear_hydroclimate_flag"]
    audit["hydroclimate_priority_score"] = audit[scored_flags].sum(axis=1).astype(float) if scored_flags else 0.0
    audit.loc[~audit["is_bad"], "hydroclimate_priority_score"] *= 0.25

    summary_rows = []
    for flag in scored_flags:
        for label, condition in {
            "all": pd.Series(True, index=audit.index),
            "bad": audit["is_bad"],
            "very_bad": audit["is_very_bad"],
            "not_bad": ~audit["is_bad"],
            "nonreservoir_bad": audit["is_bad"] & (audit["reservoir_relation"].fillna("not_reservoir_related") == "not_reservoir_related"),
        }.items():
            g = audit[condition]
            summary_rows.append({
                "flag": flag,
                "group": label,
                "station_count": int(len(g)),
                "flag_count": int(g[flag].sum()) if flag in g else 0,
                "flag_rate": float(g[flag].mean()) if len(g) else np.nan,
                "median_NSE_raw": float(g["NSE_raw"].median()) if len(g) else np.nan,
                "median_abs_PBIAS": float(g["PBIAS_pct"].abs().median()) if len(g) else np.nan,
            })
    flag_summary = pd.DataFrame(summary_rows)

    bad_priority = audit[audit["is_bad"]].sort_values(
        ["hydroclimate_priority_score", "badness_score"], ascending=[False, False]
    )

    audit.to_csv(REPORT_DIR / "station_hydroclimate_response_audit.csv", index=False, encoding="utf-8-sig")
    bad_priority.to_csv(REPORT_DIR / "hydroclimate_bad_station_priority.csv", index=False, encoding="utf-8-sig")
    flag_summary.to_csv(REPORT_DIR / "hydroclimate_flag_summary.csv", index=False, encoding="utf-8-sig")

    make_figures(audit)

    if flag_summary.empty:
        concentrated = pd.DataFrame()
    else:
        piv = flag_summary.pivot(index="flag", columns="group", values="flag_rate").reset_index()
        piv["bad_minus_not_bad"] = piv.get("bad", 0) - piv.get("not_bad", 0)
        concentrated = piv[(piv.get("bad", 0) >= 0.30) & (piv["bad_minus_not_bad"] >= 0.15)].sort_values("bad_minus_not_bad", ascending=False)

    bad = audit[audit["is_bad"]]
    very_bad = audit[audit["is_very_bad"]]
    clear_bad = bad[~bad["hydroclimate_flags"].str.contains("bad_but_no_clear", regex=False, na=False)]
    clear_very_bad = very_bad[~very_bad["hydroclimate_flags"].str.contains("bad_but_no_clear", regex=False, na=False)]

    if len(concentrated):
        decision = "This audit supports a process-diagnostic branch focused on shared hydroclimate response errors, but not a station-specific correction."
    else:
        decision = "This audit does not show a single concentrated hydroclimate-response failure strong enough to justify a broad immediate model correction."

    lines = [
        "# Hydroclimate Response Audit",
        "",
        "## Summary",
        "",
        f"- Total stations: {len(audit)}",
        f"- Bad stations: {len(bad)}",
        f"- Very bad stations: {len(very_bad)}",
        f"- Bad stations with at least one hydroclimate flag: {len(clear_bad)} / {len(bad)}",
        f"- Very bad stations with at least one hydroclimate flag: {len(clear_very_bad)} / {len(very_bad)}",
        "",
        "## Decision",
        "",
        decision,
        "",
        "This branch does not modify predictions. It identifies process directions that may deserve a future mechanism-level experiment.",
        "",
        "## Concentrated Flags",
        "",
        df_to_md(concentrated[["flag", "bad", "not_bad", "bad_minus_not_bad"]] if len(concentrated) else concentrated),
        "",
        "## Top Bad Stations By Hydroclimate Priority",
        "",
        df_to_md(bad_priority[[
            "q_site", "primary_diagnosis", "NSE_raw", "PBIAS_pct", "wet_PBIAS_pct",
            "dry_PBIAS_pct", "surplus_high_PBIAS_pct", "delta_logQ_ppt_slope",
            "delta_logQ_surplus_slope", "hydroclimate_priority_score", "hydroclimate_flags",
        ]], n=25),
        "",
        "## Flag Summary",
        "",
        df_to_md(flag_summary.pivot(index="flag", columns="group", values="flag_rate").reset_index() if len(flag_summary) else flag_summary),
        "",
        "## Interpretation",
        "",
        "- Wet/dry and surplus flags point to process-response errors, not station-specific tuning targets.",
        "- If flags are concentrated among bad stations, future work should adjust process equations or parameter pooling globally/by reach class.",
        "- If flags are diffuse, continuing to add small empirical corrections would violate the no-overfitting/no-minor-gain rule.",
        "",
        "## Files",
        "",
        "- `station_hydroclimate_response_audit.csv`",
        "- `hydroclimate_bad_station_priority.csv`",
        "- `hydroclimate_flag_summary.csv`",
        "- `figure/hydroclimate_response/*.png`",
        "",
    ]
    (REPORT_DIR / "hydroclimate_response_audit_report.md").write_text("\n".join(lines), encoding="utf-8")

    with (ROOT / "README.md").open("a", encoding="utf-8") as f:
        f.write("\n## Run Result\n\n")
        f.write(f"- Bad stations with hydroclimate flags: {len(clear_bad)} / {len(bad)}.\n")
        f.write(f"- Very bad stations with hydroclimate flags: {len(clear_very_bad)} / {len(very_bad)}.\n")
        f.write(f"- Concentrated process flags: {len(concentrated)}.\n")
        f.write(f"- Decision: {decision}\n")
        f.write("- See `reports/hydroclimate_response_audit_report.md`.\n")

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n## 20260620_6 hydroclimate response audit\n")
        f.write("- Audited wet/dry, surplus, climate-stress, rainfall-elasticity, and surplus-elasticity errors using CHM_PRE and CMFD forcing.\n")
        f.write(f"- Bad stations with hydroclimate flags: {len(clear_bad)}/{len(bad)}; very bad: {len(clear_very_bad)}/{len(very_bad)}.\n")
        f.write(f"- Concentrated process flags meeting continuation screen: {len(concentrated)}.\n")
        if len(concentrated):
            f.write("- Concentrated flags: " + "; ".join(concentrated["flag"].astype(str).tolist()) + "\n")
        f.write(f"- Decision: {decision}\n")
        f.write("- No station-specific correction or prediction change was performed.\n")

    print("Hydroclimate response audit complete:", ROOT)
    print(f"Bad flagged: {len(clear_bad)}/{len(bad)}")
    print(f"Very bad flagged: {len(clear_very_bad)}/{len(very_bad)}")
    print(f"Concentrated flags: {len(concentrated)}")
    if len(concentrated):
        print(concentrated[["flag", "bad", "not_bad", "bad_minus_not_bad"]].to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
