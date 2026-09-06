from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
PARENT_AUDIT = RUN.parent / "20260728_19"
COMPENSATION = RUN.parent / "20260728_20"
OUT = RUN / "reports" / "q72_lite_interaction_candidate"
EPS = 1.0e-12


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(model, frame: pd.DataFrame, eta: np.ndarray) -> pd.DataFrame:
    rows = []
    work = frame[["q_site", "Q_obsv_cfs"]].copy()
    work["prediction"] = np.exp(np.clip(eta, -20, 20))
    for site, part in work.groupby("q_site", sort=False):
        values = model.metric_dict(part["Q_obsv_cfs"].to_numpy(dtype=float), part["prediction"].to_numpy(dtype=float))
        values["q_site"] = str(site)
        values["abs_PBIAS"] = abs(float(values["PBIAS_pct"]))
        values["good"] = bool(values["n"] >= 24 and values["NSE_log"] >= 0.65 and values["KGE_2012"] >= 0.50 and values["abs_PBIAS"] <= 25.0)
        values["severe_pbias"] = bool(values["abs_PBIAS"] > 50.0)
        rows.append(values)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_module("audit_parent", PARENT_AUDIT / "scripts" / "run_q72_frozen_module_audit.py")
    audit.RUN = RUN
    compensation = load_module("compensation", COMPENSATION / "scripts" / "run_q72_module_compensation_audit.py")
    model = load_module("q72_model", RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py")
    et = load_module("et_module", RUN / "scripts" / "components" / "run_et_role_experiment.py")
    slow = load_module("slow_module", RUN / "scripts" / "components" / "run_slowflow_redundancy_experiment.py")
    storage = load_module("storage_module", RUN / "scripts" / "components" / "run_storage_timing_experiment.py")
    gate_module = load_module("gate_module", RUN / "scripts" / "components" / "run_hysteresis_gate_experiment.py")
    hp = audit.parse_hyperparams()
    featured, stations = audit.build_featured(model, et, slow, storage, gate_module, hp)
    layout = audit.design_layout(model, stations)
    mask = layout["module"].eq("interaction").to_numpy()
    if not mask.any() or layout.loc[mask, "term_type"].eq("intercept").any():
        raise RuntimeError("Interaction deletion mask is invalid")
    layout.to_csv(OUT / "q72_design_column_registry.csv", index=False, encoding="utf-8-sig")
    fold_rows = []
    station_rows = []
    beta_reproduction = []
    for fold_id, train_end, eval_start, eval_end in audit.FOLDS:
        train = featured[featured["year"] <= train_end].copy()
        evaluation = featured[featured["year"].between(eval_start, eval_end)].copy()
        mean, std = model.standardize_fit(train)
        x_aug, y_aug = compensation.augmented_design(model, train, stations, mean, std, hp)
        beta_base, *_ = np.linalg.lstsq(x_aug, y_aug, rcond=None)
        beta_reference = model.fit_map_ridge(train, stations, mean, std, hp["fixed_sigma"], hp["production_sigma"], hp["group_sigma"], hp["multistore_sigma"], hp["hysteresis_sigma"], hp["station_sigma"], hp["slope_sigma"], hp["regime_slope_sigma"], 0.0, hp.get("flow_contrast_weight", 1.0))
        beta_reproduction.append(float(np.max(np.abs(beta_base - beta_reference))))
        keep = ~mask
        beta_lite, *_ = np.linalg.lstsq(x_aug[:, keep], y_aug, rcond=None)
        x_eval, _ = model.build_matrix(evaluation, stations, mean, std)
        eta_base = x_eval @ beta_base
        eta_lite = x_eval[:, keep] @ beta_lite
        base_metrics = metrics(model, evaluation, eta_base).set_index("q_site")
        lite_metrics = metrics(model, evaluation, eta_lite).set_index("q_site")
        common = base_metrics.index.intersection(lite_metrics.index)
        base_metrics = base_metrics.loc[common]
        lite_metrics = lite_metrics.loc[common]
        delta = lite_metrics[["NSE_log", "KGE_2012", "abs_PBIAS"]] - base_metrics[["NSE_log", "KGE_2012", "abs_PBIAS"]]
        lost_good = int((base_metrics["good"] & ~lite_metrics["good"]).sum())
        new_severe = int((~base_metrics["severe_pbias"] & lite_metrics["severe_pbias"]).sum())
        fold_gate = bool(delta["NSE_log"].median() >= -0.005 and delta["KGE_2012"].median() >= -0.01 and delta["abs_PBIAS"].median() <= 1.0 and lost_good <= 1 and new_severe == 0)
        fold_rows.append({
            "fold_id": fold_id, "train_end_year": train_end, "eval_start_year": eval_start, "eval_end_year": eval_end,
            "removed_columns": int(mask.sum()), "baseline_good_count": int(base_metrics["good"].sum()), "lite_good_count": int(lite_metrics["good"].sum()),
            "lost_good_count": lost_good, "new_severe_pbias_count": new_severe,
            "median_delta_NSElog": float(delta["NSE_log"].median()), "median_delta_KGE": float(delta["KGE_2012"].median()), "median_delta_absPBIAS": float(delta["abs_PBIAS"].median()),
            "fold_gate_passed": fold_gate,
        })
        for site in common:
            station_rows.append({
                "fold_id": fold_id, "q_site": site,
                "base_NSElog": float(base_metrics.at[site, "NSE_log"]), "lite_NSElog": float(lite_metrics.at[site, "NSE_log"]),
                "delta_NSElog": float(delta.at[site, "NSE_log"]), "base_KGE": float(base_metrics.at[site, "KGE_2012"]),
                "lite_KGE": float(lite_metrics.at[site, "KGE_2012"]), "delta_KGE": float(delta.at[site, "KGE_2012"]),
                "base_absPBIAS": float(base_metrics.at[site, "abs_PBIAS"]), "lite_absPBIAS": float(lite_metrics.at[site, "abs_PBIAS"]),
                "delta_absPBIAS": float(delta.at[site, "abs_PBIAS"]), "base_good": bool(base_metrics.at[site, "good"]), "lite_good": bool(lite_metrics.at[site, "good"]),
            })
    fold_frame = pd.DataFrame(fold_rows)
    station_frame = pd.DataFrame(station_rows)
    fold_frame.to_csv(OUT / "interaction_deletion_by_fold.csv", index=False, encoding="utf-8-sig")
    station_frame.to_csv(OUT / "interaction_deletion_by_station.csv", index=False, encoding="utf-8-sig")
    payload = {
        "run_id": RUN.name, "parent_run": "20260728_20", "candidate": "remove_q72_interaction_module",
        "used_year_max": int(featured["year"].max()), "confirmation_years_used": False, "removed_design_columns": int(mask.sum()),
        "baseline_reproduction_max_abs_beta_delta": max(beta_reproduction, default=float("inf")),
        "fold_gate_pass_count": int(fold_frame["fold_gate_passed"].sum()), "fold_count": int(len(fold_frame)),
        "all_folds_passed": bool(fold_frame["fold_gate_passed"].all()),
    }
    payload["passed"] = bool(payload["used_year_max"] == 2018 and payload["baseline_reproduction_max_abs_beta_delta"] <= 1e-10 and payload["all_folds_passed"])
    (OUT / "gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Q72-lite interaction-deletion candidate", "", "| fold | Δ median NSElog | Δ median KGE | Δ median abs PBIAS (pp) | lost good | new severe PBIAS | gate |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in fold_frame.itertuples(index=False):
        lines.append(f"| {row.fold_id} | {row.median_delta_NSElog:+.6f} | {row.median_delta_KGE:+.6f} | {row.median_delta_absPBIAS:+.3f} | {row.lost_good_count} | {row.new_severe_pbias_count} | {'PASS' if row.fold_gate_passed else 'FAIL'} |")
    lines.extend(["", f"- Removed design columns: {int(mask.sum())}", f"- Q72 baseline reproduction max beta delta: {payload['baseline_reproduction_max_abs_beta_delta']:.3e}", f"- Candidate gate: {'PASS' if payload['passed'] else 'FAIL'}", "- 2019-2022 used: False"])
    (OUT / "interaction_deletion_candidate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not payload["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
