from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
PARENT = RUN.parent / "20260728_19"
OUT = RUN / "reports" / "q72_module_compensation_audit"
EPS = 1.0e-12


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def augmented_design(model, train: pd.DataFrame, stations: list[str], mean: pd.Series, std: pd.Series, hp: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    x, y = model.build_matrix(train, stations, mean, std)
    x_parts = [x]
    y_parts = [y]
    contrast_x = []
    contrast_y = []
    root_weight = float(np.sqrt(hp.get("flow_contrast_weight", 1.0)))
    if root_weight > 0:
        for _, positions in train.groupby("q_site", sort=False).indices.items():
            pos = np.asarray(positions, dtype=int)
            if len(pos) < 36:
                continue
            local_x = x[pos, :]
            local_y = y[pos]
            local_q = train.iloc[pos]["Q_obsv_cfs"].to_numpy(dtype=float)
            q25, q50, q75, q90 = np.nanquantile(local_q, [0.25, 0.50, 0.75, 0.90])
            low, high = local_q <= q25, local_q >= q75
            mid, peak = (local_q >= q25) & (local_q <= q75), local_q >= q90
            if low.sum() >= 6 and high.sum() >= 6:
                contrast_x.append((local_x[high].mean(axis=0) - local_x[low].mean(axis=0)) * root_weight)
                contrast_y.append((local_y[high].mean() - local_y[low].mean()) * root_weight)
            if peak.sum() >= 3 and mid.sum() >= 12:
                contrast_x.append((local_x[peak].mean(axis=0) - local_x[mid].mean(axis=0)) * root_weight)
                contrast_y.append((local_y[peak].mean() - local_y[mid].mean()) * root_weight)
    if contrast_x:
        x_parts.append(np.vstack(contrast_x))
        y_parts.append(np.asarray(contrast_y, dtype=float))
    n_fixed_base = 1 + len(model.FIXED_FEATURES)
    n_group = len(model.SPATIAL_GROUP_GATES) * len(model.SPATIAL_GROUP_FEATURES)
    n_fixed = n_fixed_base + n_group
    n_station = len(stations)
    n_slope = len(model.RANDOM_SLOPE_FEATURES) * n_station
    n_regime = len(model.REGIME_GATES) * len(model.REGIME_SLOPE_FEATURES) * n_station
    penalty = np.zeros(n_fixed + n_station + n_slope + n_regime, dtype=float)
    penalty[1:n_fixed_base] = 1.0 / max(hp["fixed_sigma"], EPS)
    for index, feature in enumerate(model.FIXED_FEATURES, start=1):
        if feature in model.PRODUCTION_FEATURES:
            penalty[index] = 1.0 / max(hp["production_sigma"], EPS)
        if feature in model.MULTISTORE_FEATURES:
            penalty[index] = 1.0 / max(hp["multistore_sigma"], EPS)
        if feature in model.HYSTERESIS_FEATURES:
            penalty[index] = 1.0 / max(hp["hysteresis_sigma"], EPS)
    penalty[n_fixed_base:n_fixed] = 1.0 / max(hp["group_sigma"], EPS)
    penalty[n_fixed:n_fixed + n_station] = 1.0 / max(hp["station_sigma"], EPS)
    penalty[n_fixed + n_station:n_fixed + n_station + n_slope] = 1.0 / max(hp["slope_sigma"], EPS)
    penalty[n_fixed + n_station + n_slope:] = 1.0 / max(hp["regime_slope_sigma"], EPS)
    return np.vstack([*x_parts, np.diag(penalty)]), np.concatenate([*y_parts, np.zeros(len(penalty), dtype=float)])


def station_delta(audit, model, frame: pd.DataFrame, base_eta: np.ndarray, other_eta: np.ndarray) -> tuple[float, float, float]:
    base = audit.station_metrics(model, frame, base_eta).set_index("q_site")
    other = audit.station_metrics(model, frame, other_eta).set_index("q_site")
    delta = other.loc[base.index] - base
    return float(delta["NSE_log"].median()), float(delta["KGE_2012"].median()), float(delta["abs_PBIAS"].median())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_module("audit_parent", PARENT / "scripts" / "run_q72_frozen_module_audit.py")
    audit.RUN = RUN
    model = load_module("q72_model", RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py")
    et = load_module("et_module", RUN / "scripts" / "components" / "run_et_role_experiment.py")
    slow = load_module("slow_module", RUN / "scripts" / "components" / "run_slowflow_redundancy_experiment.py")
    storage = load_module("storage_module", RUN / "scripts" / "components" / "run_storage_timing_experiment.py")
    gate = load_module("gate_module", RUN / "scripts" / "components" / "run_hysteresis_gate_experiment.py")
    hp = audit.parse_hyperparams()
    featured, stations = audit.build_featured(model, et, slow, storage, gate, hp)
    layout = audit.design_layout(model, stations)
    layout.to_csv(OUT / "q72_design_column_registry.csv", index=False, encoding="utf-8-sig")
    rows = []
    reproduction_deltas = []
    for fold_id, train_end, eval_start, eval_end in audit.FOLDS:
        train = featured[featured["year"] <= train_end].copy()
        evaluation = featured[featured["year"].between(eval_start, eval_end)].copy()
        mean, std = model.standardize_fit(train)
        x_aug, y_aug = augmented_design(model, train, stations, mean, std, hp)
        beta, *_ = np.linalg.lstsq(x_aug, y_aug, rcond=None)
        beta_reference = model.fit_map_ridge(train, stations, mean, std, hp["fixed_sigma"], hp["production_sigma"], hp["group_sigma"], hp["multistore_sigma"], hp["hysteresis_sigma"], hp["station_sigma"], hp["slope_sigma"], hp["regime_slope_sigma"], 0.0, hp.get("flow_contrast_weight", 1.0))
        reproduction_deltas.append(float(np.max(np.abs(beta - beta_reference))))
        x_eval, _ = model.build_matrix(evaluation, stations, mean, std)
        eta_base = x_eval @ beta
        for module in audit.MODULES:
            mask = layout["module"].eq(module).to_numpy()
            beta_frozen = beta.copy()
            beta_frozen[mask] = 0.0
            eta_frozen = x_eval @ beta_frozen
            residual = y_aug - x_aug[:, ~mask] @ beta[~mask]
            local_part, *_ = np.linalg.lstsq(x_aug[:, mask], residual, rcond=None)
            beta_local = beta.copy()
            beta_local[mask] = local_part
            eta_local = x_eval @ beta_local
            keep = ~mask
            beta_full, *_ = np.linalg.lstsq(x_aug[:, keep], y_aug, rcond=None)
            eta_full = x_eval[:, keep] @ beta_full
            frozen_distance = float(np.linalg.norm(eta_frozen - eta_base))
            local_distance = float(np.linalg.norm(eta_local - eta_base))
            full_distance = float(np.linalg.norm(eta_full - eta_base))
            d_nse, d_kge, d_pbias = station_delta(audit, model, evaluation, eta_base, eta_full)
            rows.append({
                "fold_id": fold_id, "train_end_year": train_end, "eval_start_year": eval_start, "eval_end_year": eval_end,
                "module": module, "module_column_count": int(mask.sum()), "frozen_distance": frozen_distance,
                "local_distance": local_distance, "full_distance": full_distance,
                "local_compensation_index": float(1.0 - local_distance / (frozen_distance + EPS)),
                "full_compensation_index": float(1.0 - full_distance / (frozen_distance + EPS)),
                "full_median_delta_NSElog": d_nse, "full_median_delta_KGE": d_kge, "full_median_delta_absPBIAS": d_pbias,
            })
    by_fold = pd.DataFrame(rows)
    summary = by_fold.groupby("module", as_index=False).agg(
        module_column_count=("module_column_count", "first"), folds=("fold_id", "nunique"),
        median_local_CI=("local_compensation_index", "median"), median_full_CI=("full_compensation_index", "median"),
        folds_full_CI_ge_075=("full_compensation_index", lambda s: int((s >= 0.75).sum())),
        median_full_delta_NSElog=("full_median_delta_NSElog", "median"), median_full_delta_KGE=("full_median_delta_KGE", "median"),
        median_full_delta_absPBIAS=("full_median_delta_absPBIAS", "median"),
    )
    summary["compensation_dominated"] = summary["folds_full_CI_ge_075"] >= 2
    by_fold.to_csv(OUT / "module_compensation_by_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "module_compensation_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "run_id": RUN.name, "parent_run": PARENT.name, "used_year_max": int(featured["year"].max()),
        "forbidden_confirmation_years_used": bool(featured["year"].max() > 2018), "modules": audit.MODULES,
        "fold_module_rows": int(len(by_fold)), "baseline_fit_reproduction_max_abs_beta_delta": max(reproduction_deltas, default=float("inf")),
    }
    payload["passed"] = bool(payload["used_year_max"] == 2018 and len(by_fold) == len(audit.FOLDS) * len(audit.MODULES) and payload["baseline_fit_reproduction_max_abs_beta_delta"] <= 1e-10)
    (OUT / "gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Q72 module compensation audit", "", "| module | median local CI | median full CI | folds full CI >= 0.75 | compensation dominated |", "| --- | ---: | ---: | ---: | --- |"]
    for row in summary.sort_values("median_full_CI", ascending=False).itertuples(index=False):
        lines.append(f"| {row.module} | {row.median_local_CI:.3f} | {row.median_full_CI:.3f} | {row.folds_full_CI_ge_075} | {row.compensation_dominated} |")
    lines.extend(["", f"- Baseline coefficient reproduction max delta: {payload['baseline_fit_reproduction_max_abs_beta_delta']:.3e}", f"- Engineering gate: {'PASS' if payload['passed'] else 'FAIL'}", "- 2019-2022 used: False"])
    (OUT / "module_compensation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not payload["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
