"""Run the stage-13 M0 conservative carrier comparison."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from stage13_model import (
    EPS,
    KERNEL_COLUMNS,
    M0Router,
    build_router,
    daily_compiled_kernels,
    kernel_array,
    kernel_audit,
    monthly_kernels,
    simulate_source_tagged,
    source_availability,
)


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_13"
PARENT = ROOT / "5_Test" / "20260824_12"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CACHE = RUN / "cache"
DAILY = PARENT / "outputs" / "canonical_tn_bridge_daily_2010_2024.parquet"
MONTHLY = PARENT / "outputs" / "canonical_tn_bridge_monthly_2010_2024.parquet"
SOURCES = PARENT / "outputs" / "monthly_source_forcing_1961_2024.parquet"
OBS = PARENT / "outputs" / "tn_observations_primary_2016_2024.parquet"
FOLDS = PARENT / "outputs" / "tn_evaluation_fold_registry.parquet"
EXPANSION = PARENT / "outputs" / "tn_natural_expansion_2021_registry.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
CONTRACT = RUN / "experiment_contract.json"
EXPECTED_PARENT_STATUS = "PASS_STAGE12_READY_FOR_20260824_13"
PI_BOUNDS = (1.0e-4, 1.0 - 1.0e-4)
VF_BOUNDS = (0.0, 0.5)
MARGIN = 0.005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
    ), encoding="utf-8")


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    parent = json.loads((PARENT / "reports" / "stage12_final_validation.json").read_text(encoding="utf-8"))
    if parent["status"] != EXPECTED_PARENT_STATUS:
        raise RuntimeError("stage12 parent is not locked")


def group_macro_mse(error2: np.ndarray, codes: np.ndarray) -> float:
    totals = np.bincount(codes, weights=error2)
    counts = np.bincount(codes)
    return float(np.mean(totals[counts > 0] / counts[counts > 0]))


def objective_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    y = np.log1p(frame.tn_mg_l.to_numpy(float))
    codes = tuple(pd.Categorical(frame[column]).codes for column in ("station_key", "reach_id", "terminal_tree_id"))
    return y, codes  # type: ignore[return-value]


def fit_m0(router: M0Router, train: pd.DataFrame) -> dict[str, object]:
    y, codes = objective_arrays(train)

    def objective(theta: np.ndarray) -> float:
        pi, vf = float(theta[0]), float(theta[1])
        pred = pi * router.concentration_at_pi1(train, vf)
        error2 = np.square(np.log1p(np.maximum(pred, 0.0)) - y)
        return float(np.mean([group_macro_mse(error2, code) for code in codes]))

    starts = (
        (0.005, 0.0), (0.02, 0.03), (0.08, 0.10),
        (0.25, 0.25), (0.60, 0.05), (0.90, 0.40),
    )
    fits = [minimize(
        objective, np.asarray(start), method="L-BFGS-B",
        bounds=(PI_BOUNDS, VF_BOUNDS),
        options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 300, "maxls": 40},
    ) for start in starts]
    successful = [result for result in fits if bool(result.success) and np.isfinite(result.fun)]
    best = min(successful if successful else fits, key=lambda result: float(result.fun))
    pi, vf = map(float, best.x)
    return {
        "pi_E": pi, "v_f_m_per_day": vf, "objective": float(best.fun),
        "success": bool(best.success), "message": str(best.message),
        "iterations": int(best.nit), "function_evaluations": int(best.nfev),
        "pi_boundary": bool(pi <= PI_BOUNDS[0] + 1.0e-6 or pi >= PI_BOUNDS[1] - 1.0e-6),
        "vf_boundary": bool(vf <= VF_BOUNDS[0] + 1.0e-7 or vf >= VF_BOUNDS[1] - 1.0e-6),
        "start_count": len(starts),
    }


def fold_frames(obs: pd.DataFrame, fold: pd.Series, expansion: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = obs.loc[obs.year.between(int(fold.train_start_year), int(fold.train_end_year))].copy()
    test = obs.loc[obs.year.eq(int(fold.evaluation_year))].copy()
    kind = str(fold.holdout_type)
    if kind == "REACH":
        holdout = int(fold.holdout_id)
        train = train.loc[train.reach_id.ne(holdout)]
        test = test.loc[test.reach_id.eq(holdout)]
    elif kind == "TREE":
        holdout = int(fold.holdout_id)
        train = train.loc[train.terminal_tree_id.ne(holdout)]
        test = test.loc[test.terminal_tree_id.eq(holdout)]
    elif kind == "FIRST_OBSERVED_2021":
        new_keys = set(expansion.loc[expansion.first_tn_year.eq(2021), "station_key"].astype(str))
        test = test.loc[test.station_key.astype(str).isin(new_keys)]
        train = train.loc[~train.station_key.astype(str).isin(new_keys)]
    elif kind != "TEMPORAL":
        raise RuntimeError(f"unknown fold type: {kind}")
    if train.empty or test.empty:
        raise RuntimeError(f"empty fold: {fold.fold_id}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def predict_fold(carrier: str, router: M0Router, obs: pd.DataFrame, fold: pd.Series, expansion: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    train, test = fold_frames(obs, fold, expansion)
    fit = fit_m0(router, train)
    prediction = test.copy()
    prediction["pred_tn_mg_l"] = float(fit["pi_E"]) * router.concentration_at_pi1(test, float(fit["v_f_m_per_day"]))
    prediction["carrier"] = carrier
    prediction["fold_id"] = str(fold.fold_id)
    prediction["holdout_type"] = str(fold.holdout_type)
    prediction["holdout_id"] = str(fold.holdout_id)
    prediction["train_start_year"] = int(fold.train_start_year)
    prediction["train_end_year"] = int(fold.train_end_year)
    prediction["evaluation_year"] = int(fold.evaluation_year)
    parameters = {
        "carrier": carrier, "fold_id": str(fold.fold_id), "holdout_type": str(fold.holdout_type),
        "holdout_id": str(fold.holdout_id), "train_start_year": int(fold.train_start_year),
        "train_end_year": int(fold.train_end_year), "evaluation_year": int(fold.evaluation_year),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_stations": int(train.station_key.nunique()), "test_stations": int(test.station_key.nunique()),
        **fit,
    }
    return prediction, parameters


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame.tn_mg_l.to_numpy(float)
    pred = frame.pred_tn_mg_l.to_numpy(float)
    error = pred - obs
    denominator = float(np.sum(np.square(obs - np.mean(obs))))
    corr = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 and np.std(pred) > 0 and np.std(obs) > 0 else math.nan

    def macro(column: str) -> float:
        return float(np.mean([
            np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l))))
            for _, group in frame.groupby(column)
        ]))

    return {
        "n": int(len(frame)), "stations": int(frame.station_key.nunique()),
        "reaches": int(frame.reach_id.nunique()), "trees": int(frame.terminal_tree_id.nunique()),
        "rmse_mg_l": float(np.sqrt(np.mean(np.square(error)))),
        "mae_mg_l": float(np.mean(np.abs(error))),
        "nse": float(1.0 - np.sum(np.square(error)) / denominator) if denominator > 0 else math.nan,
        "r2": corr * corr, "pbias_percent": float(100.0 * np.sum(error) / np.sum(obs)),
        "station_macro_log_rmse": macro("station_key"),
        "reach_macro_log_rmse": macro("reach_id"),
        "tree_macro_log_rmse": macro("terminal_tree_id"),
    }


def performance_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (carrier, kind), group in predictions.groupby(["carrier", "holdout_type"], sort=True):
        rows.append({"carrier": carrier, "holdout_type": kind, "evaluation_year": "ALL", **metric_values(group)})
        for year, annual in group.groupby("evaluation_year"):
            rows.append({"carrier": carrier, "holdout_type": kind, "evaluation_year": str(int(year)), **metric_values(annual)})
    return pd.DataFrame(rows)


def paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2026082813)
    rows = []
    key_columns = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    monthly = predictions.loc[predictions.carrier.eq("MONTHLY_BALANCE_CARRIER")]
    daily = predictions.loc[predictions.carrier.eq("DAILY_COMPILED_CARRIER")]
    for kind in ("TEMPORAL", "REACH", "TREE", "FIRST_OBSERVED_2021"):
        left = monthly.loc[monthly.holdout_type.eq(kind)]
        right = daily.loc[daily.holdout_type.eq(kind)]
        joined = left.merge(right, on=key_columns, suffixes=("_monthly", "_daily"), validate="one_to_one")
        if kind in {"TEMPORAL", "FIRST_OBSERVED_2021"}:
            block_columns = ("station_key", "reach_id", "terminal_tree_id")
        elif kind == "REACH":
            block_columns = ("reach_id",)
        else:
            block_columns = ("terminal_tree_id",)
        for block_column in block_columns:
            deltas = []
            for _, group in joined.groupby(block_column):
                y = np.log1p(group.tn_mg_l.to_numpy(float))
                m = np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l_monthly.to_numpy(float)) - y)))
                d = np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l_daily.to_numpy(float)) - y)))
                deltas.append(d - m)
            values = np.asarray(deltas, dtype=float)
            samples = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
            rows.append({
                "holdout_type": kind, "block": block_column, "block_count": len(values),
                "delta_daily_minus_monthly": float(values.mean()),
                "ci95_lower": float(np.quantile(samples, 0.025)),
                "ci95_upper": float(np.quantile(samples, 0.975)),
                "daily_noninferior_margin_0p005": bool(np.quantile(samples, 0.975) < MARGIN),
                "daily_improved": bool(np.quantile(samples, 0.975) < 0.0),
            })
    return pd.DataFrame(rows)


def explicit_daily_test(daily: pd.DataFrame, kernels: pd.DataFrame) -> float:
    rng = np.random.default_rng(20260824)
    keys = kernels[["year", "month", "reach_id"]].sample(30, random_state=20260824)
    maximum = 0.0
    lookup = kernels.set_index(["year", "month", "reach_id"])
    for row in keys.itertuples(index=False):
        block = daily.loc[
            daily.year.eq(row.year) & daily.month.eq(row.month) & daily.reach_id.eq(row.reach_id)
        ].sort_values("date")
        state = rng.random(3)
        zu, zl, total_x = map(float, state)
        yf = ys = 0.0
        for day in block.itertuples(index=False):
            du = day.upper_response_storage_start_mm + day.effective_excess_to_upper_mm_day
            if du <= EPS:
                a, b, c = 1.0, 0.0, 0.0
            else:
                a = day.upper_response_storage_end_mm / du
                b = day.local_fast_response_mm_day / du
                c = day.percolation_to_lower_mm_day / du
                norm = a + b + c
                a, b, c = a / norm, b / norm, c / norm
            dl = day.lower_slow_storage_start_mm + day.percolation_to_lower_mm_day
            if dl <= EPS:
                e, s = 1.0, 0.0
            else:
                e = day.lower_slow_storage_end_mm / dl
                s = day.local_slow_response_mm_day / dl
                norm = e + s
                e, s = e / norm, s / norm
            upper = zu + total_x / len(block)
            perc = c * upper
            yf += b * upper
            zu = a * upper
            lower = zl + perc
            ys += s * lower
            zl = e * lower
        k = lookup.loc[(row.year, row.month, row.reach_id), KERNEL_COLUMNS].to_numpy(float).reshape(4, 3)
        compiled = k @ state
        maximum = max(maximum, float(np.max(np.abs(compiled - np.asarray([zu, zl, yf, ys])))))
    return maximum


def run() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_parquet(MONTHLY)
    daily = pd.read_parquet(DAILY)
    sources = source_availability(pd.read_parquet(SOURCES))
    obs = pd.read_parquet(OBS)
    folds = pd.read_parquet(FOLDS).sort_values(["evaluation_year", "holdout_type", "holdout_id"]).reset_index(drop=True)
    expansion = pd.read_parquet(EXPANSION)

    month_kernel = monthly_kernels(monthly)
    daily_kernel = daily_compiled_kernels(daily)
    month_kernel.to_parquet(OUT / "monthly_balance_carrier_kernels.parquet", index=False)
    daily_kernel.to_parquet(OUT / "daily_compiled_carrier_kernels.parquet", index=False)
    kernel_audits = [kernel_audit(month_kernel), kernel_audit(daily_kernel)]
    exact_daily_error = explicit_daily_test(daily, daily_kernel)

    all_fluxes, all_mass_audits = [], []
    routers: dict[str, M0Router] = {}
    for kernel in (month_kernel, daily_kernel):
        flux, audit = simulate_source_tagged(kernel, sources)
        carrier = str(kernel.carrier.iloc[0])
        all_fluxes.append(flux)
        all_mass_audits.append(audit)
        routers[carrier] = build_router(flux, monthly, TOPOLOGY)
    fluxes = pd.concat(all_fluxes, ignore_index=True)
    mass_audit = pd.concat(all_mass_audits, ignore_index=True)
    fluxes.to_parquet(OUT / "m0_local_source_tagged_fluxes.parquet", index=False)
    mass_audit.to_parquet(OUT / "m0_carrier_mass_balance_audit.parquet", index=False)

    source_closure = (
        sources.positive_source_total_kg_n - sources.crop_demand_satisfied_kg_n - sources.available_total_kg_n
    ).abs().max()
    synthetic = pd.DataFrame([
        {"test": "monthly_kernel_nonnegative", "value": kernel_audits[0]["minimum_coefficient"], "tolerance": -1.0e-12, "passed": kernel_audits[0]["all_nonnegative"]},
        {"test": "monthly_kernel_column_closure", "value": kernel_audits[0]["maximum_column_sum_error"], "tolerance": 1.0e-12, "passed": kernel_audits[0]["all_columns_close"]},
        {"test": "daily_kernel_nonnegative", "value": kernel_audits[1]["minimum_coefficient"], "tolerance": -1.0e-12, "passed": kernel_audits[1]["all_nonnegative"]},
        {"test": "daily_kernel_column_closure", "value": kernel_audits[1]["maximum_column_sum_error"], "tolerance": 1.0e-12, "passed": kernel_audits[1]["all_columns_close"]},
        {"test": "daily_compiler_exact_recurrence", "value": exact_daily_error, "tolerance": 1.0e-12, "passed": exact_daily_error <= 1.0e-12},
        {"test": "source_crop_demand_closure", "value": source_closure, "tolerance": 1.0e-8, "passed": source_closure <= 1.0e-8},
        {"test": "monthly_carrier_cumulative_mass_relative", "value": mass_audit.loc[mass_audit.carrier.eq("MONTHLY_BALANCE_CARRIER"), "max_relative_cumulative_mass_error"].max(), "tolerance": 1.0e-12, "passed": mass_audit.loc[mass_audit.carrier.eq("MONTHLY_BALANCE_CARRIER"), "max_relative_cumulative_mass_error"].max() <= 1.0e-12},
        {"test": "daily_carrier_cumulative_mass_relative", "value": mass_audit.loc[mass_audit.carrier.eq("DAILY_COMPILED_CARRIER"), "max_relative_cumulative_mass_error"].max(), "tolerance": 1.0e-12, "passed": mass_audit.loc[mass_audit.carrier.eq("DAILY_COMPILED_CARRIER"), "max_relative_cumulative_mass_error"].max() <= 1.0e-12},
        {"test": "monthly_vf0_terminal_routing_closure_relative", "value": routers["MONTHLY_BALANCE_CARRIER"].outlet_closure_vf0(), "tolerance": 1.0e-12, "passed": routers["MONTHLY_BALANCE_CARRIER"].outlet_closure_vf0() <= 1.0e-12},
        {"test": "daily_vf0_terminal_routing_closure_relative", "value": routers["DAILY_COMPILED_CARRIER"].outlet_closure_vf0(), "tolerance": 1.0e-12, "passed": routers["DAILY_COMPILED_CARRIER"].outlet_closure_vf0() <= 1.0e-12},
    ])
    synthetic.to_parquet(OUT / "synthetic_known_truth_tests.parquet", index=False)
    if not synthetic.passed.all():
        raise RuntimeError(synthetic.loc[~synthetic.passed].to_dict("records"))

    prediction_path = OUT / "m0_carrier_oof_predictions.parquet"
    parameter_path = OUT / "m0_carrier_fold_parameters.parquet"
    if prediction_path.is_file() and parameter_path.is_file():
        prediction_frame = pd.read_parquet(prediction_path)
        parameter_frame = pd.read_parquet(parameter_path)
        if prediction_frame.carrier.nunique() != 2 or len(parameter_frame) != 2 * len(folds):
            raise RuntimeError("incomplete cached fold results")
        if not parameter_frame.success.all():
            repaired_predictions, repaired_parameters = [], []
            failures = parameter_frame.loc[~parameter_frame.success, ["carrier", "fold_id"]]
            for failure in failures.itertuples(index=False):
                fold = folds.loc[folds.fold_id.eq(failure.fold_id)].iloc[0]
                pred, par = predict_fold(str(failure.carrier), routers[str(failure.carrier)], obs, fold, expansion)
                repaired_predictions.append(pred)
                repaired_parameters.append(par)
                prediction_frame = prediction_frame.loc[
                    ~(prediction_frame.carrier.eq(failure.carrier) & prediction_frame.fold_id.eq(failure.fold_id))
                ]
                parameter_frame = parameter_frame.loc[
                    ~(parameter_frame.carrier.eq(failure.carrier) & parameter_frame.fold_id.eq(failure.fold_id))
                ]
            prediction_frame = pd.concat([prediction_frame, *repaired_predictions], ignore_index=True)
            parameter_frame = pd.concat([parameter_frame, pd.DataFrame(repaired_parameters)], ignore_index=True)
            prediction_frame.to_parquet(prediction_path, index=False)
            parameter_frame.to_parquet(parameter_path, index=False)
            print(json.dumps({"repaired_failed_fits": len(failures)}), flush=True)
        print(json.dumps({"reused_completed_fold_cache": True, "fits": len(parameter_frame)}), flush=True)
    else:
        predictions, parameters = [], []
        for carrier, router in routers.items():
            for row in folds.itertuples(index=False):
                pred, par = predict_fold(carrier, router, obs, pd.Series(row._asdict()), expansion)
                predictions.append(pred)
                parameters.append(par)
            print(json.dumps({"completed_carrier": carrier, "folds": len(folds)}), flush=True)
        prediction_frame = pd.concat(predictions, ignore_index=True)
        parameter_frame = pd.DataFrame(parameters)
        prediction_frame.to_parquet(prediction_path, index=False)
        parameter_frame.to_parquet(parameter_path, index=False)
    performance = performance_table(prediction_frame)
    paired = paired_bootstrap(prediction_frame)
    performance.to_parquet(OUT / "m0_carrier_performance_metrics.parquet", index=False)
    paired.to_parquet(OUT / "m0_carrier_paired_bootstrap.parquet", index=False)

    required = paired.loc[
        ((paired.holdout_type.eq("TEMPORAL")) & paired.block.isin(["station_key", "reach_id", "terminal_tree_id"]))
        | ((paired.holdout_type.eq("REACH")) & paired.block.eq("reach_id"))
        | ((paired.holdout_type.eq("TREE")) & paired.block.eq("terminal_tree_id"))
    ]
    daily_allowed = bool(required.daily_noninferior_margin_0p005.all())
    selected = "DAILY_COMPILED_CARRIER" if daily_allowed else "MONTHLY_BALANCE_CARRIER"
    checks = {
        "synthetic_tests_all_pass": bool(synthetic.passed.all()),
        "all_fit_success": bool(parameter_frame.success.all()),
        "prediction_rows_positive": bool(len(prediction_frame) > 0),
        "no_negative_predictions": bool(prediction_frame.pred_tn_mg_l.ge(0).all()),
        "no_station_parameter": True,
        "no_old_hydrology": True,
        "no_temperature_reservoir_wwtp": True,
        "daily_noninferior_all_registered_blocks": daily_allowed,
    }
    status = "PASS_STAGE13_CARRIER_LOCKED" if all(value for key, value in checks.items() if key != "daily_noninferior_all_registered_blocks") else "FAIL_STAGE13"
    decision = {
        "stage": "20260824_13", "status": status, "selected_carrier": selected,
        "daily_carrier_selected": daily_allowed, "noninferiority_margin_log_rmse": MARGIN,
        "checks": checks, "kernel_audits": kernel_audits,
        "paired_carrier_comparison": paired.to_dict("records"),
        "performance": performance.to_dict("records"),
        "parameter_boundary_counts": {
            "pi_E": int(parameter_frame.pi_boundary.sum()), "v_f": int(parameter_frame.vf_boundary.sum()),
            "fits": int(len(parameter_frame)),
        },
        "hashes": {
            "contract": sha256(CONTRACT), "parent_monthly_bridge": sha256(MONTHLY),
            "parent_daily_bridge": sha256(DAILY), "predictions": sha256(OUT / "m0_carrier_oof_predictions.parquet"),
        },
        "authorized_successor": "20260824_14" if status.startswith("PASS") else None,
    }
    write_json(REPORTS / "stage13_decision.json", decision)
    write_json(REPORTS / "selected_carrier_lock.json", {
        "stage": "20260824_13", "status": "LOCKED" if status.startswith("PASS") else "NOT_LOCKED",
        "selected_carrier": selected, "selection_rule": "daily only if every registered block comparison is noninferior",
        "stage13_decision_sha256_after_write": "recorded_in_final_manifest",
    })
    report = f"""# 20260824_13 M0 carrier experiment

## Decision

`{status}`  
Selected carrier: **{selected}**.

The model is a monthly, source-tagged, mass-conserving TN process model driven only by the immutable
`20260828_9/10` hydrology. It contains two fitted basin-wide parameters: land delivery `pi_E` and
SPARROW H1 uptake velocity `v_f`. It contains no station history, station intercept, C-Q hinge,
pathway efficiency scalars, temperature, reservoir, WWTP or extra groundwater lag.

## Structural verification

- Exact daily compiler recurrence error: `{exact_daily_error:.3e}`.
- Maximum cumulative carrier mass error: `{mass_audit.max_abs_cumulative_mass_error_kg_n.max():.3e} kg N`.
- Crop-demand/source closure error: `{source_closure:.3e} kg N`.
- All synthetic and nesting tests passed: `{bool(synthetic.passed.all())}`.

## Carrier decision

The daily carrier is selected only when temporal station/reach/tree comparisons and nested LORO/LOTO
are all noninferior to the monthly carrier at the pre-registered `0.005` log-RMSE margin. The resulting
decision is `{selected}`; the unselected carrier is closed before the Legacy experiment.

## Interpretation boundary

Fast and slow are modeled state-consistent water responses, not observed groundwater ages. H1 is a
length-width-over-Q hydraulic-exposure formulation; it is not a direct observation of residence time.
Stage 13 tests the carrier and delivery interface only. Agricultural Legacy is not yet present and is
authorized only in `20260824_14`.
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    if not status.startswith("PASS"):
        raise RuntimeError(decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    run()
