from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
REPRO = ROOT.parent / "20260813_53"
FIG = ROOT / "figures"
KEY_OOF = ["comid", "q_site", "year", "month", "fold_id"]
KEY_VAL = ["comid", "q_site", "year", "month"]
TOL = -1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(frame: pd.DataFrame, pred_col: str = "predict") -> dict[str, float]:
    obs, pred = frame.actual.to_numpy(float), frame[pred_col].to_numpy(float)
    keep = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[keep], pred[keep]
    lo, lp = np.log(obs), np.log(pred)
    r = np.corrcoef(obs, pred)[0, 1]
    alpha, beta = np.std(pred) / np.std(obs), np.mean(pred) / np.mean(obs)
    return {
        "n": len(obs), "stations": int(frame.loc[keep, "q_site"].nunique()),
        "raw_nse": 1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2),
        "log_nse": 1 - np.sum((lp - lo) ** 2) / np.sum((lo - lo.mean()) ** 2),
        "kge_2012": 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
        "pbias_pct": 100 * np.sum(pred - obs) / np.sum(obs),
        "rmse_cfs": np.sqrt(np.mean((pred - obs) ** 2)),
        "log_rmse": np.sqrt(np.mean((lp - lo) ** 2)),
    }


def paired_parent_effect(parent: pd.DataFrame, selected: pd.DataFrame, keys: list[str], period: str) -> list[dict[str, object]]:
    a = parent[keys + ["actual", "predict"]].rename(columns={"predict": "parent_predict"})
    b = selected[keys + ["actual", "predict"]].rename(columns={"actual": "selected_actual", "predict": "selected_predict"})
    p = a.merge(b, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(p.actual, p.selected_actual, rtol=0, atol=0):
        raise RuntimeError("Paired actual mismatch")
    rows = []
    for scenario, col in [("S000_parent_on_selected_population", "parent_predict"), ("S111_selected_refit", "selected_predict")]:
        x = p.rename(columns={col: "predict"})
        rows.append({"period": period, "scenario_id": scenario, **metrics(x)})
    return rows


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in [("svg", {}), ("pdf", {}), ("tiff", {"dpi": 600}), ("png", {"dpi": 300})]:
        fig.savefig(FIG / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)


def figure_sensitivity(scenario: pd.DataFrame) -> None:
    style()
    order = ["S000", "S100", "S010", "S001", "S110", "S101", "S011", "S111"]
    x = scenario.set_index("scenario_id").loc[order].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    width = 0.36
    xx = np.arange(len(x))
    axes[0].bar(xx - width / 2, x.oof_negative_station_count, width, color="#3B6FB6", label="2012-2018 OOF")
    axes[0].bar(xx + width / 2, x.eval_negative_station_count, width, color="#D97732", label="2019-2022")
    axes[0].set(xticks=xx, xticklabels=x.scenario_id, ylabel="Stations with a negative metric", title="Negative-station sensitivity")
    axes[0].legend(fontsize=5.8)
    axes[1].scatter(x.excluded_station_count, x.oof_log_nse, s=36, color="#3B6FB6", label="OOF", zorder=3)
    axes[1].scatter(x.excluded_station_count, x.eval_log_nse, s=36, facecolor="white", edgecolor="#D97732", linewidth=1.2, label="2019-2022", zorder=3)
    for _, r in x.iterrows():
        axes[1].annotate(r.scenario_id, (r.excluded_station_count, r.eval_log_nse), xytext=(2, 2), textcoords="offset points", fontsize=5.5)
    axes[1].set(xlabel="Excluded stations", ylabel="Pooled log-NSE", title="Coverage-performance trade-off")
    axes[1].legend(fontsize=5.8)
    axes[2].scatter(x.eval_stations, x.combined_negative_station_count, s=35, color="#6A51A3", edgecolor="white", lw=0.5)
    label_offsets = {
        "S000": (6, 3), "S100": (6, -8), "S010": (-20, 5), "S001": (4, 4),
        "S110": (4, 4), "S101": (4, 4), "S011": (4, -8), "S111": (4, 4),
    }
    for _, r in x.iterrows():
        axes[2].annotate(
            r.scenario_id, (r.eval_stations, r.combined_negative_station_count),
            xytext=label_offsets[r.scenario_id], textcoords="offset points", fontsize=5.5,
        )
    axes[2].axvline(90, color="#777777", ls="--", lw=0.8)
    axes[2].set(xlabel="Retained 2019-2022 stations", ylabel="Union negative stations", title="Zero-negative coverage gate")
    for label, ax in zip("abc", axes):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
    save(fig, "figure_1_station_combination_sensitivity")
    plt.close(fig)


def figure_heatmap(station: pd.DataFrame) -> None:
    style()
    problem = sorted(set(station.loc[station.negative_or_invalid, "q_site"].astype(str)))
    order = ["S000", "S100", "S010", "S001", "S110", "S101", "S011", "S111"]
    periods = ["2012-2018 OOF", "2019-2022 diagnostic"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True, constrained_layout=True)
    for ax, period in zip(axes, periods):
        p = station.loc[station.period.eq(period) & station.q_site.astype(str).isin(problem)]
        matrix = p.pivot(index="q_site", columns="scenario_id", values="minimum_efficiency").reindex(index=problem, columns=order)
        data = matrix.clip(-2, 1).to_numpy(float)
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap="RdYlBu", vmin=-2, vmax=1, aspect="auto")
        ax.set(xticks=np.arange(len(order)), xticklabels=order, yticks=np.arange(len(problem)), yticklabels=problem, title=period)
        for i in range(len(problem)):
            for j in range(len(order)):
                value = matrix.iloc[i, j]
                text = "excluded" if not np.isfinite(value) else ("<−2" if value < -2 else f"{value:.2f}")
                ax.text(j, i, text, ha="center", va="center", fontsize=4.5, color="white" if np.isfinite(value) and value < -0.6 else "#222222", rotation=90 if text == "excluded" else 0)
    fig.colorbar(im, ax=axes, label="Minimum of raw NSE, log-NSE and KGE", shrink=0.75)
    for label, ax in zip("ab", axes):
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
    save(fig, "figure_2_problem_station_scenario_heatmap")
    plt.close(fig)


def figure_shijiao(oof: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    style()
    parts = []
    for period, frame in [("2012-2018 OOF", oof), ("2019-2022 diagnostic", validation)]:
        x = frame.loc[frame.q_site.astype(str).eq("石角站")].copy().sort_values(["year", "month"])
        if x.empty:
            raise RuntimeError("Protected 石角站 missing")
        x["period_label"] = period
        x["date"] = pd.to_datetime(dict(year=x.year.astype(int), month=x.month.astype(int), day=1))
        parts.append(x)
    source = pd.concat(parts, ignore_index=True)
    source[["period_label", "q_site", "comid", "year", "month", "actual", "predict"]].to_csv(
        ROOT / "stone_corner_shijiao_audit.csv", index=False, encoding="utf-8-sig"
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), constrained_layout=True)
    metric_rows = []
    for label, ax, (period, x) in zip("ab", axes, source.groupby("period_label", sort=False)):
        m = metrics(x)
        metric_rows.append({"period": period, "q_site": "石角站", **m})
        ax.plot(x.date, x.actual, color="#222222", lw=1.1, label="Observed")
        ax.plot(x.date, x.predict, color="#D97732", lw=1.0, label="Predicted")
        ax.set_yscale("log")
        ax.set_ylabel("Monthly Q (cfs, log scale)")
        ax.set_title(f"石角站 — {period} | NSE={m['raw_nse']:.3f}, log-NSE={m['log_nse']:.3f}, KGE={m['kge_2012']:.3f}")
        ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("Year")
    save(fig, "figure_3_shijiao_protected_hydrograph")
    plt.close(fig)
    pd.DataFrame(metric_rows).to_csv(ROOT / "shijiao_metrics.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(metric_rows)


def audit_figures(stems: list[str]) -> list[dict[str, object]]:
    rows = []
    for stem in stems:
        paths = {s: FIG / f"{stem}.{s}" for s in ["svg", "pdf", "tiff", "png"]}
        svg = paths["svg"].read_text(encoding="utf-8")
        pages = len(re.findall(rb"/Type\s*/Page\b", paths["pdf"].read_bytes()))
        with Image.open(paths["tiff"]) as image:
            dpi = [float(v) for v in image.info.get("dpi", (0, 0))]
        rows.append({"stem": stem, "all_formats": all(p.exists() and p.stat().st_size > 0 for p in paths.values()), "editable_svg_text": "<text" in svg, "pdf_pages": pages, "tiff_dpi": dpi})
    return rows


def main() -> None:
    FIG.mkdir(exist_ok=True)
    scenario = pd.read_csv(ROOT / "scenario_metrics.csv", encoding="utf-8-sig")
    station = pd.read_csv(ROOT / "scenario_station_metrics.csv", encoding="utf-8-sig")
    selected_work = SCENARIOS / "S111"
    oof = pd.read_parquet(selected_work / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet")
    val = pd.read_parquet(selected_work / "outputs" / "fit_2006_2018_eval_2019_2022" / "validation_predictions_2019_2022.parquet")
    parent_oof = pd.read_parquet(SCENARIOS / "S000" / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet")
    parent_val = pd.read_parquet(SCENARIOS / "S000" / "outputs" / "fit_2006_2018_eval_2019_2022" / "validation_predictions_2019_2022.parquet")
    paired = pd.DataFrame(
        paired_parent_effect(parent_oof, oof, KEY_OOF, "2012-2018 OOF")
        + paired_parent_effect(parent_val, val, KEY_VAL, "2019-2022 diagnostic")
    )
    paired.to_csv(ROOT / "selected_same_population_protection_metrics.csv", index=False, encoding="utf-8-sig")

    fold_rows = []
    for scenario_id in sorted(scenario.scenario_id):
        p = pd.read_parquet(SCENARIOS / scenario_id / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet")
        for fold, x in p.groupby("fold_id", sort=True):
            fold_rows.append({"scenario_id": scenario_id, "fold_id": fold, **metrics(x)})
    pd.DataFrame(fold_rows).to_csv(ROOT / "scenario_fold_metrics.csv", index=False, encoding="utf-8-sig")

    selected_station = station.loc[station.scenario_id.eq("S111")]
    negative = selected_station.loc[selected_station.negative_or_invalid]
    selected_row = scenario.set_index("scenario_id").loc["S111"]
    parent_selected = paired.set_index(["period", "scenario_id"])
    protection = {}
    for period in ["2012-2018 OOF", "2019-2022 diagnostic"]:
        base = parent_selected.loc[(period, "S000_parent_on_selected_population")]
        new = parent_selected.loc[(period, "S111_selected_refit")]
        protection[period] = {
            "delta_raw_nse": float(new.raw_nse - base.raw_nse),
            "delta_log_nse": float(new.log_nse - base.log_nse),
            "delta_kge": float(new.kge_2012 - base.kge_2012),
            "delta_abs_pbias_percentage_points": float(abs(new.pbias_pct) - abs(base.pbias_pct)),
        }
    contract = json.loads((selected_work / "scenario_contract.json").read_text(encoding="utf-8"))
    selection = {
        "selected_scenario": "S111", "selection_is_post_hoc": True,
        "excluded_station_count": contract["excluded_station_count"],
        "excluded_stations": contract["excluded_stations"], "protected_station": "石角站",
        "oof_rows": len(oof), "oof_stations": int(oof.q_site.nunique()),
        "evaluation_rows": len(val), "evaluation_stations": int(val.q_site.nunique()),
        "zero_negative_station_count": int(len(negative)), "protection_effects": protection,
    }
    (ROOT / "selected_station_configuration.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(selected_work / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet", ROOT / "final_oof_predictions.parquet")
    shutil.copy2(selected_work / "outputs" / "fit_2006_2018_eval_2019_2022" / "validation_predictions_2019_2022.parquet", ROOT / "final_2019_2022_predictions.parquet")

    figure_sensitivity(scenario)
    figure_heatmap(station)
    shijiao = figure_shijiao(oof, val)
    stems = ["figure_1_station_combination_sensitivity", "figure_2_problem_station_scenario_heatmap", "figure_3_shijiao_protected_hydrograph"]
    figure_audit = audit_figures(stems)
    deterministic = json.loads((REPRO / "reports" / "deterministic_reproduction_gate.json").read_text(encoding="utf-8"))
    shijiao_ok = bool((shijiao[["raw_nse", "log_nse", "kge_2012"]].to_numpy(float) >= TOL).all())
    shijiao_parent = []
    for period, frame in [("2012-2018 OOF", parent_oof), ("2019-2022 diagnostic", parent_val)]:
        shijiao_parent.append({"period": period, **metrics(frame.loc[frame.q_site.astype(str).eq("石角站")])})
    shijiao_parent = pd.DataFrame(shijiao_parent).set_index("period")
    shijiao_selected = shijiao.set_index("period")
    shijiao_degradation_ok = all(
        shijiao_selected.loc[p, m] - shijiao_parent.loc[p, m] >= -0.05
        for p in shijiao_selected.index for m in ["raw_nse", "log_nse"]
    )
    population_protection_ok = all(
        effect["delta_raw_nse"] >= -0.005
        and effect["delta_log_nse"] >= -0.005
        and effect["delta_kge"] >= -0.01
        and effect["delta_abs_pbias_percentage_points"] <= 2.0
        for effect in protection.values()
    )
    input_integrity = {
        "oof_duplicate_keys": int(oof.duplicated(KEY_OOF).sum()),
        "evaluation_duplicate_keys": int(val.duplicated(KEY_VAL).sum()),
        "oof_missing_actual_or_prediction": int(oof[["actual", "predict"]].isna().any(axis=1).sum()),
        "evaluation_missing_actual_or_prediction": int(val[["actual", "predict"]].isna().any(axis=1).sum()),
        "oof_nonpositive_actual_or_prediction": int(((oof.actual <= 0) | (oof.predict <= 0)).sum()),
        "evaluation_nonpositive_actual_or_prediction": int(((val.actual <= 0) | (val.predict <= 0)).sum()),
    }
    all_checks = {
        "eight_scenarios_complete": len(scenario) == 8,
        "s111_unique_passing_initial_scenario": scenario.loc[scenario.zero_negative_gate & scenario.coverage_gate_ge_90_eval_stations, "scenario_id"].tolist() == ["S111"],
        "selected_zero_negative": len(negative) == 0,
        "selected_eval_coverage_ge_90": int(selected_row.eval_stations) >= 90,
        "shijiao_protected_and_nonnegative": shijiao_ok and "石角站" not in contract["excluded_stations"],
        "shijiao_degradation_within_0_05": shijiao_degradation_ok,
        "deterministic_reproduction": deterministic["terminal"] == "S111_DETERMINISTIC_REPRODUCTION_PASS",
        "forcing_and_reach_identity": contract["source_rows"] == 46920 and contract["reaches"] == 230,
        "figure_audit": all(r["all_formats"] and r["editable_svg_text"] and r["pdf_pages"] == 1 and min(r["tiff_dpi"]) >= 599 for r in figure_audit),
        "selected_prediction_integrity": all(value == 0 for value in input_integrity.values()),
    }
    core_checks = [
        all_checks["eight_scenarios_complete"], all_checks["s111_unique_passing_initial_scenario"],
        all_checks["selected_zero_negative"], all_checks["selected_eval_coverage_ge_90"],
        all_checks["shijiao_protected_and_nonnegative"], all_checks["shijiao_degradation_within_0_05"],
        all_checks["deterministic_reproduction"], all_checks["forcing_and_reach_identity"],
        all_checks["figure_audit"],
        all_checks["selected_prediction_integrity"],
    ]
    terminal = "COVERAGE_ADEQUATE_ZERO_NEGATIVE_STATION_CONFIGURATION_FOUND" if all(core_checks) else "STATION_COMBINATION_RUN_OR_REPRODUCTION_FAILURE"
    operational = (
        "ZERO_NEGATIVE_CONFIGURATION_FOUND_BUT_PROTECTION_NOT_ACCEPTABLE"
        if terminal.startswith("COVERAGE_ADEQUATE") and not population_protection_ok
        else "ZERO_NEGATIVE_CONFIGURATION_OPERATIONALLY_ACCEPTABLE"
    )
    gate = {
        "terminal": terminal,
        "operational_terminal": operational,
        "secondary_terminal": "POST_HOC_STATION_COMBINATION_SENSITIVITY_COMPLETE",
        "checks": {**all_checks, "same_population_protection_gate": population_protection_ok},
        "selected": selection, "selected_prediction_integrity": input_integrity,
        "population_accounting": {
            "parent_oof_stations": int(parent_oof.q_site.nunique()),
            "parent_2019_2022_stations": int(parent_val.q_site.nunique()),
            "selected_oof_stations": int(oof.q_site.nunique()),
            "selected_2019_2022_stations": int(val.q_site.nunique()),
        },
        "figure_audit": figure_audit,
    }
    (ROOT / "terminal_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if terminal.endswith("FAILURE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
