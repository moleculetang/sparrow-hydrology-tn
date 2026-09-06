from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SOURCE_RUN = RUN.parent / "20260728_30"
AUDIT_PARENT = RUN.parent / "20260728_19"
COMPENSATION_PARENT = RUN.parent / "20260728_20"
FORMAL_BASELINE = RUN.parent / "20260728_18"
OUT = RUN / "reports" / "lowflow_oof_canonical_baseline"
EPS = 1.0e-6
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_class_map() -> pd.DataFrame:
    path = FORMAL_BASELINE / "reports" / "main_model" / "reach_class_selected_predictions_long.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    out = frame[["q_site", "reach_class"]].drop_duplicates()
    if out["q_site"].duplicated().any() or out["reach_class"].isna().any():
        raise RuntimeError("Formal baseline canonical reach-class mapping is not one-to-one")
    return out


def old_area_class(area: float) -> str:
    if area < 5000:
        return "headwater"
    if area < 25000:
        return "mid_nonheadwater"
    return "large_nonheadwater"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_module("audit_parent", AUDIT_PARENT / "scripts" / "run_q72_frozen_module_audit.py")
    audit.RUN = SOURCE_RUN
    compensation = load_module("compensation", COMPENSATION_PARENT / "scripts" / "run_q72_module_compensation_audit.py")
    model = load_module("q72_model", SOURCE_RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py")
    et = load_module("et_module", SOURCE_RUN / "scripts" / "components" / "run_et_role_experiment.py")
    slow = load_module("slow_module", SOURCE_RUN / "scripts" / "components" / "run_slowflow_redundancy_experiment.py")
    storage = load_module("storage_module", SOURCE_RUN / "scripts" / "components" / "run_storage_timing_experiment.py")
    gate_module = load_module("gate_module", SOURCE_RUN / "scripts" / "components" / "run_hysteresis_gate_experiment.py")

    hp = audit.parse_hyperparams()
    featured, stations = audit.build_featured(model, et, slow, storage, gate_module, hp)
    if int(featured["year"].max()) != 2018:
        raise RuntimeError("Development diagnostic must not include years after 2018")
    layout = audit.design_layout(model, stations)
    interaction_mask = layout["module"].eq("interaction").to_numpy()
    if int(interaction_mask.sum()) != 62 or layout.loc[interaction_mask, "term_type"].eq("intercept").any():
        raise RuntimeError("Interaction-lite mask is not the frozen 62-column candidate")

    class_map = canonical_class_map()
    oof_rows: list[pd.DataFrame] = []
    low_rows: list[dict[str, object]] = []
    beta_deltas: list[float] = []
    for fold_id, train_end, eval_start, eval_end in audit.FOLDS:
        train = featured[featured["year"] <= train_end].copy()
        evaluation = featured[featured["year"].between(eval_start, eval_end)].copy()
        mean, std = model.standardize_fit(train)
        x_aug, y_aug = compensation.augmented_design(model, train, stations, mean, std, hp)
        beta_full, *_ = np.linalg.lstsq(x_aug, y_aug, rcond=None)
        beta_reference = model.fit_map_ridge(
            train, stations, mean, std,
            hp["fixed_sigma"], hp["production_sigma"], hp["group_sigma"], hp["multistore_sigma"],
            hp["hysteresis_sigma"], hp["station_sigma"], hp["slope_sigma"], hp["regime_slope_sigma"],
            0.0, hp.get("flow_contrast_weight", 1.0),
        )
        beta_deltas.append(float(np.max(np.abs(beta_full - beta_reference))))
        beta_lite, *_ = np.linalg.lstsq(x_aug[:, ~interaction_mask], y_aug, rcond=None)
        x_eval, _ = model.build_matrix(evaluation, stations, mean, std)
        reach_key = "reach_id" if "reach_id" in evaluation.columns else "comid"
        work = evaluation[[
            "q_site", reach_key, "year", "month", "Q_obsv_cfs", "CumAreaKm2",
            "is_reservoir_reach", "downstream_reservoir",
        ]].copy().rename(columns={reach_key: "reach_id"})
        work["fold_id"] = fold_id
        work["train_end_year"] = train_end
        work["q72_full_cfs"] = np.exp(np.clip(x_eval @ beta_full, -20, 20))
        work["q72_interaction_lite_cfs"] = np.exp(np.clip(x_eval[:, ~interaction_mask] @ beta_lite, -20, 20))
        q25 = train.groupby("q_site")["Q_obsv_cfs"].quantile(0.25).rename("train_q25_cfs")
        work = work.merge(q25, on="q_site", how="left")
        work["is_low_flow"] = work["Q_obsv_cfs"].le(work["train_q25_cfs"])
        oof_rows.append(work)
        for site, part in work.groupby("q_site", sort=False):
            low = part[part["is_low_flow"]].copy()
            if len(low) < 4:
                continue
            obs = low["Q_obsv_cfs"].to_numpy(dtype=float)
            pred = low["q72_interaction_lite_cfs"].to_numpy(dtype=float)
            first = part.iloc[0]
            low_rows.append({
                "fold_id": fold_id,
                "q_site": str(site),
                "reach_id": int(first["reach_id"]),
                "low_flow_months": int(len(low)),
                "low_flow_PBIAS_pct": float(100.0 * (pred.sum() - obs.sum()) / max(obs.sum(), EPS)),
                "low_flow_log_RMSE": float(np.sqrt(np.mean((np.log(pred + EPS) - np.log(obs + EPS)) ** 2))),
                "low_flow_overprediction": bool(100.0 * (pred.sum() - obs.sum()) / max(obs.sum(), EPS) > 25.0),
                "reservoir_related": bool(float(first.get("is_reservoir_reach", 0)) > 0 or float(first.get("downstream_reservoir", 0)) > 0),
                "area_audit_class": old_area_class(float(first["CumAreaKm2"])),
            })

    oof = pd.concat(oof_rows, ignore_index=True).merge(class_map, on="q_site", how="left", validate="many_to_one")
    if oof["reach_class"].isna().any():
        missing = sorted(oof.loc[oof["reach_class"].isna(), "q_site"].astype(str).unique())
        raise RuntimeError(f"Missing canonical reach classes: {missing}")
    low = pd.DataFrame(low_rows).merge(class_map, on="q_site", how="left", validate="many_to_one")
    summary = (
        low.groupby(["q_site", "reach_id", "reservoir_related", "area_audit_class", "reach_class"], as_index=False)
        .agg(
            eligible_folds=("fold_id", "nunique"),
            overprediction_folds=("low_flow_overprediction", "sum"),
            has_2016_2018=("fold_id", lambda s: "fit_2006_2015_eval_2016_2018" in set(s)),
            median_low_flow_PBIAS=("low_flow_PBIAS_pct", "median"),
            median_low_flow_log_RMSE=("low_flow_log_RMSE", "median"),
        )
    )
    summary["stable_nonreservoir_target"] = (
        summary["overprediction_folds"].ge(2)
        & summary["has_2016_2018"]
        & ~summary["reservoir_related"]
        & ~summary["q_site"].isin(EXCLUSIONS)
    )
    targets = summary[summary["stable_nonreservoir_target"]].copy()
    reconciliation = (
        summary[["q_site", "reach_id", "area_audit_class", "reach_class", "stable_nonreservoir_target"]]
        .drop_duplicates()
        .assign(class_changed=lambda d: d["area_audit_class"].ne(d["reach_class"]))
    )
    oof.to_csv(OUT / "q72_interaction_lite_oof_predictions.csv", index=False, encoding="utf-8-sig")
    low.to_csv(OUT / "low_flow_signature_by_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "low_flow_signature_summary_canonical.csv", index=False, encoding="utf-8-sig")
    reconciliation.to_csv(OUT / "reach_class_reconciliation.csv", index=False, encoding="utf-8-sig")
    layout.to_csv(OUT / "q72_design_column_registry.csv", index=False, encoding="utf-8-sig")

    source_panel = SOURCE_RUN / "inputs" / "indata.parquet"
    payload = {
        "run_id": RUN.name,
        "diagnostic_type": "interaction_lite_q72_oof_lowflow_baseline_with_canonical_reach_class",
        "source_run": SOURCE_RUN.name,
        "formal_baseline": FORMAL_BASELINE.name,
        "used_year_max": int(featured["year"].max()),
        "confirmation_years_used": False,
        "source_input_sha256": sha256(source_panel),
        "interaction_columns_removed": int(interaction_mask.sum()),
        "baseline_reproduction_max_abs_beta_delta": max(beta_deltas, default=float("inf")),
        "oof_key_unique": not oof.duplicated(["q_site", "reach_id", "year", "month", "fold_id"]).any(),
        "canonical_class_missing_count": int(oof["reach_class"].isna().sum()),
        "excluded_station_rows": {name: int((oof["q_site"] == name).sum()) for name in sorted(EXCLUSIONS)},
        "shijiao_present": bool((oof["q_site"] == "石角站").any()),
        "eligible_station_count": int(summary["q_site"].nunique()),
        "stable_nonreservoir_target_count": int(len(targets)),
        "stable_target_count_by_canonical_class": {str(k): int(v) for k, v in targets.groupby("reach_class").size().items()},
        "area_vs_canonical_class_changed_station_count": int(reconciliation["class_changed"].sum()),
    }
    payload["engineering_passed"] = bool(
        payload["used_year_max"] == 2018
        and payload["baseline_reproduction_max_abs_beta_delta"] <= 1e-10
        and payload["interaction_columns_removed"] == 62
        and payload["oof_key_unique"]
        and payload["canonical_class_missing_count"] == 0
        and all(v == 0 for v in payload["excluded_station_rows"].values())
        and payload["shijiao_present"]
    )
    payload["scientific_status"] = "PASS_BASELINE_REBUILT" if payload["engineering_passed"] else "FAIL_ENGINEERING"
    (OUT / "gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Interaction-lite low-flow OOF canonical-baseline gate",
        "",
        f"- Engineering gate: {'PASS' if payload['engineering_passed'] else 'FAIL'}",
        f"- Development years through: {payload['used_year_max']}",
        f"- Interaction columns removed: {payload['interaction_columns_removed']}",
        f"- Q72 baseline reproduction max beta delta: {payload['baseline_reproduction_max_abs_beta_delta']:.3e}",
        f"- Stable nonreservoir targets: {payload['stable_nonreservoir_target_count']}",
        f"- Canonical target counts: {payload['stable_target_count_by_canonical_class']}",
        f"- Stations whose area audit class differs from canonical class: {payload['area_vs_canonical_class_changed_station_count']}",
        "- 2019-2022 used: False",
    ]
    (OUT / "gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not payload["engineering_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
