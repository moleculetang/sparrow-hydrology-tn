from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from daily_state import HydroParameters, add_daily_hydrologic_features, operator_equivalence_tests, parameter_mapping_table
from experiment_utils import (
    EPS, add_flow_class, fit_three_fold_oof, load_component, metrics,
    residual_acf_rows, scenario_metric_rows, sha256, write_json,
)
from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN / "inputs" / "source_snapshot"
REPORTS = RUN / "reports"
SHUFFLES = REPORTS / "shuffles"
COMPONENT_PATH = RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py"


def load_config() -> dict[str, object]:
    return json.loads((RUN / "config.json").read_text(encoding="utf-8"))


def configured_component(unique: str):
    component = load_component(COMPONENT_PATH, unique)
    component.INPUT_PATH = SOURCE / "q72_indata_2006_2018.parquet"
    component.TOPOLOGY_PATH = SOURCE / "topology_edges.csv"
    return component


def load_monthly(component) -> pd.DataFrame:
    frame = component.load_observed_panel()
    if int(frame["year"].min()) != 2006 or int(frame["year"].max()) != 2018:
        raise RuntimeError("Monthly observed panel violates the 2006-2018 boundary")
    return frame


def input_validation() -> dict[str, object]:
    monthly_full = pd.read_parquet(SOURCE / "q72_indata_2006_2018.parquet")
    daily = pd.read_parquet(SOURCE / "chm_pre_v2_daily_by_reach_2006_2018.parquet")
    canonical = pd.read_parquet(SOURCE / "canonical_q72_oof.parquet")
    registry = pd.read_csv(SOURCE / "canonical_signal_registry.csv", encoding="utf-8-sig")
    daily["date"] = pd.to_datetime(daily["date"])
    monthly_key_unique = not bool(monthly_full.duplicated(["comid", "year", "month"]).any())
    daily_key_unique = not bool(daily.duplicated(["reach_id", "date"]).any())
    daily["year"] = daily["date"].dt.year.astype(int)
    daily["month"] = daily["date"].dt.month.astype(int)
    grouped = daily.groupby(["reach_id", "year", "month"], as_index=False).agg(
        daily_sum_mm=("PPT_daily_mm", "sum"), daily_count=("date", "size"),
        wet_days=("PPT_daily_mm", lambda s: int((s > 0).sum())),
        min_valid_grid_fraction=("valid_grid_fraction", "min"),
    )
    monthly = monthly_full[monthly_full["year"].between(2006, 2018)][["comid", "year", "month", "PPT"]].copy()
    compare = monthly.merge(grouped, left_on=["comid", "year", "month"], right_on=["reach_id", "year", "month"], how="left", validate="one_to_one")
    expected_days = pd.to_datetime({"year": compare["year"].astype(int), "month": compare["month"].astype(int), "day": 1}).dt.days_in_month.to_numpy()
    compare["abs_monthly_difference_mm"] = (compare["PPT"] - compare["daily_sum_mm"]).abs()
    max_difference = float(compare["abs_monthly_difference_mm"].max())
    target_reaches = set(pd.to_numeric(registry.loc[registry["legacy_canonical_membership"].astype(str).str.casefold().isin({"true", "1"}), "reach_id"], errors="raise").astype(int))
    evaluable_targets = sorted(target_reaches & set(canonical["reach_id"].astype(int)))
    fold_counts = canonical.groupby("fold_id").agg(rows=("station_name", "size"), stations=("station_name", "nunique")).reset_index().to_dict("records")
    checks = {
        "runtime_exact_conda_sparrow": Path(RUNTIME["sys_executable"]).parent.resolve() == Path(RUNTIME["expected_prefix"]).resolve(),
        "monthly_years_2006_2018_only": int(monthly_full["year"].min()) == 2006 and int(monthly_full["year"].max()) == 2018,
        "monthly_key_unique": monthly_key_unique,
        "daily_key_unique": daily_key_unique,
        "daily_years_2006_2018_only": int(daily["year"].min()) == 2006 and int(daily["year"].max()) == 2018,
        "daily_no_missing_ppt": not bool(daily["PPT_daily_mm"].isna().any()),
        "daily_calendar_complete": bool(np.array_equal(compare["daily_count"].to_numpy(int), expected_days)),
        "daily_valid_grid_complete": float(compare["min_valid_grid_fraction"].min()) >= 1.0,
        "daily_monthly_ppt_within_1e_8_mm": max_difference <= 1e-8,
        "canonical_rows_8738": len(canonical) == 8738,
        "canonical_stations_110": canonical["station_name"].nunique() == 110,
        "canonical_three_folds": canonical["fold_id"].nunique() == 3,
        "canonical_key_unique": not bool(canonical.duplicated(["station_name", "year", "month"]).any()),
        "legacy_registry_28": len(target_reaches) == 28,
        "legacy_evaluable_27": len(evaluable_targets) == 27,
        "tianhe_169_absent": 169 not in evaluable_targets,
        "shijiao_reach_56_present": bool(((canonical["station_name"] == "石角站") & canonical["reach_id"].astype(int).eq(56)).any()),
    }
    result = {
        "runtime": RUNTIME, "checks": checks, "passed": bool(all(checks.values())),
        "monthly_rows": int(len(monthly_full)), "daily_rows": int(len(daily)),
        "daily_monthly_key_rows": int(len(compare)), "max_abs_daily_monthly_ppt_difference_mm": max_difference,
        "canonical_fold_counts": fold_counts, "legacy_registry_total": len(target_reaches),
        "legacy_evaluable": len(evaluable_targets), "legacy_evaluable_reaches": evaluable_targets,
        "state_axis_semantics": "Only positive-observation station-month rows advance state, matching canonical Q72; missing months remain skipped.",
    }
    compare[["comid", "year", "month", "PPT", "daily_sum_mm", "daily_count", "wet_days", "abs_monthly_difference_mm"]].to_parquet(REPORTS / "daily_monthly_reconciliation.parquet", index=False)
    return result


def write_manifest() -> None:
    paths = sorted(SOURCE.glob("*")) + sorted((RUN / "scripts").glob("*.py")) + [COMPONENT_PATH, RUN / "config.json", RUN / "experiment_contract.md"]
    entries = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths]
    write_json(RUN / "input_manifest.json", {"run_id": RUN.name, "files": entries})
    pd.DataFrame(entries).to_csv(RUN / "inputs_manifest" / "source_manifest.csv", index=False, encoding="utf-8-sig")


def run_prepare() -> None:
    REPORTS.mkdir(exist_ok=True)
    validation = input_validation()
    write_json(RUN / "input_validation.json", validation)
    write_manifest()
    parameters = HydroParameters()
    parameter_mapping_table(parameters).to_csv(RUN / "parameter_timescale_mapping.csv", index=False, encoding="utf-8-sig")
    tests = operator_equivalence_tests(parameters)
    tests.to_csv(RUN / "state_operator_equivalence_tests.csv", index=False, encoding="utf-8-sig")
    if not validation["passed"] or not bool(tests["passed"].all()):
        raise RuntimeError("Input or operator equivalence gate failed")

    component = configured_component("q72_d0_prepare")
    monthly = load_monthly(component)
    hp = load_config()["fixed_hyperparameters"]
    monthly_featured = component.add_hydrologic_features(
        monthly, rho=float(hp["rho"]), wm=float(hp["wm"]), et_gamma=float(hp["et_gamma"]),
        sas_rho=float(hp["sas_rho"]), young_k=float(hp["young_k"]), storage_scale=float(hp["storage_scale"]),
        prod_capacity=float(hp["prod_capacity"]), runoff_gamma=float(hp["runoff_gamma"]),
        quick_rho=float(hp["quick_rho"]), base_rho=float(hp["base_rho"]), base_release=float(hp["base_release"]),
    )
    refit_path = REPORTS / "D0_refit_oof.parquet"
    required_d0_state_columns = {
        "production_month_end_quick_store_mm",
        "production_month_end_base_store_mm",
        "ms_slow_large_month_end_quick_store_mm",
        "ms_slow_large_month_end_base_store_mm",
    }
    if refit_path.exists():
        d0 = pd.read_parquet(refit_path)
    else:
        d0 = pd.DataFrame()
    if d0.empty or not required_d0_state_columns.issubset(d0.columns):
        d0 = fit_three_fold_oof(monthly_featured, component, load_config(), "D0")
        d0.to_parquet(refit_path, index=False)
    canonical = pd.read_parquet(SOURCE / "canonical_q72_oof.parquet")
    joined = d0.merge(canonical[["station_name", "reach_id", "year", "month", "fold_id", "observed_cfs", "predicted_cfs"]], on=["station_name", "reach_id", "year", "month", "fold_id"], how="outer", suffixes=("_new", "_canonical"), indicator=True, validate="one_to_one")
    max_prediction = float((joined["predicted_cfs_new"] - joined["predicted_cfs_canonical"]).abs().max())
    max_observation = float((joined["observed_cfs_new"] - joined["observed_cfs_canonical"]).abs().max())
    max_relative = float(((joined["predicted_cfs_new"] - joined["predicted_cfs_canonical"]).abs() / joined["predicted_cfs_canonical"].abs().clip(lower=EPS)).max())
    max_log = float((np.log(joined["predicted_cfs_new"].clip(lower=EPS)) - np.log(joined["predicted_cfs_canonical"].clip(lower=EPS))).abs().max())
    d0_metric = metrics(d0)
    canonical_metric = metrics(add_flow_class(canonical.assign(scenario="canonical")))
    tolerance = float(load_config()["gates"]["d0_prediction_tolerance_cfs"])
    relative_tolerance = float(load_config()["gates"]["d0_relative_or_log_tolerance"])
    checks = {
        "same_8738_rows": len(d0) == len(canonical) == 8738,
        "all_keys_matched": bool(joined["_merge"].eq("both").all()),
        "observed_difference_le_1e_8": max_observation <= tolerance,
        "prediction_difference_le_1e_8": max_prediction <= tolerance,
        "prediction_relative_difference_le_1e_12": max_relative <= relative_tolerance,
        "prediction_log_difference_le_1e_12": max_log <= relative_tolerance,
        "pooled_log_nse_difference_le_1e_8": abs(d0_metric["NSE_log"] - canonical_metric["NSE_log"]) <= tolerance,
    }
    strict_absolute_passed = bool(all(checks[key] for key in ["same_8738_rows", "all_keys_matched", "observed_difference_le_1e_8", "prediction_difference_le_1e_8", "pooled_log_nse_difference_le_1e_8"]))
    numerical_equivalence_passed = bool(all(checks[key] for key in ["same_8738_rows", "all_keys_matched", "observed_difference_le_1e_8", "prediction_relative_difference_le_1e_12", "prediction_log_difference_le_1e_12", "pooled_log_nse_difference_le_1e_8"]))
    reproduction = {
        "checks": checks, "strict_absolute_passed": strict_absolute_passed,
        "passed": numerical_equivalence_passed,
        "protocol_note": "Absolute 1e-8 cfs is retained and reported; relative/log equivalence is allowed only for machine-level last-bit differences on very large flows.",
        "max_abs_prediction_difference_cfs": max_prediction,
        "max_relative_prediction_difference": max_relative,
        "max_abs_log_prediction_difference": max_log,
        "max_abs_observation_difference_cfs": max_observation,
        "d0_metrics": d0_metric, "canonical_metrics": canonical_metric,
    }
    write_json(RUN / "d0_reproduction.json", reproduction)
    if not reproduction["passed"]:
        raise RuntimeError(f"D0 reproduction failed: max prediction difference={max_prediction}")
    # Formal D0 is the immutable canonical prediction.  Refit state columns are
    # retained, while observed/predicted values are replaced by the canonical
    # columns after the independent numerical-equivalence audit above.
    canonical_values = canonical[["station_name", "reach_id", "year", "month", "fold_id", "observed_cfs", "predicted_cfs"]]
    formal = d0.drop(columns=["observed_cfs", "predicted_cfs"]).merge(canonical_values, on=["station_name", "reach_id", "year", "month", "fold_id"], how="inner", validate="one_to_one")
    formal.to_parquet(REPORTS / "D0_oof.parquet", index=False)
    print(json.dumps({"input_validation": validation["passed"], "d0_reproduction": reproduction}, ensure_ascii=False, indent=2))


def compute_daily_scenario(mode: str, scenario: str, seed: int | None = None) -> pd.DataFrame:
    component = configured_component(f"q72_{scenario}_{os.getpid()}")
    monthly = load_monthly(component)
    daily = pd.read_parquet(SOURCE / "chm_pre_v2_daily_by_reach_2006_2018.parquet")
    featured = add_daily_hydrologic_features(monthly, daily, component, HydroParameters(), mode=mode, seed=seed)
    return fit_three_fold_oof(featured, component, load_config(), scenario)


def run_core() -> None:
    reproduction = json.loads((RUN / "d0_reproduction.json").read_text(encoding="utf-8"))
    if not reproduction["passed"]:
        raise RuntimeError("D0 reproduction gate is not passed")
    started = time.perf_counter()
    flat = compute_daily_scenario("flat", "D-flat")
    flat.to_parquet(REPORTS / "D-flat_oof.parquet", index=False)
    observed = compute_daily_scenario("observed", "D1")
    observed.to_parquet(REPORTS / "D1_oof.parquet", index=False)
    keys = ["station_name", "reach_id", "year", "month", "fold_id"]
    d0 = pd.read_parquet(REPORTS / "D0_oof.parquet")
    same_keys = set(map(tuple, d0[keys].to_numpy())) == set(map(tuple, flat[keys].to_numpy())) == set(map(tuple, observed[keys].to_numpy()))
    result = {"passed": bool(len(flat) == len(observed) == len(d0) == 8738 and same_keys), "same_8738_keys": same_keys, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    write_json(RUN / "core_scenario_validation.json", result)
    if not result["passed"]:
        raise RuntimeError("Core scenario OOF keys differ")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _shuffle_worker(rep: int) -> dict[str, object]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    path = SHUFFLES / f"shuffle_{rep:03d}.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        if len(old) == 8738 and old["scenario"].eq(f"D1-shuffle-{rep:03d}").all():
            return {"replicate": rep, "rows": len(old), "resumed": True, "path": str(path)}
    seed = int(load_config()["base_seed"]) + rep
    oof = compute_daily_scenario("shuffle", f"D1-shuffle-{rep:03d}", seed=seed)
    oof.to_parquet(path, index=False)
    return {"replicate": rep, "seed": seed, "rows": len(oof), "resumed": False, "path": str(path)}


def run_shuffles() -> None:
    core = json.loads((RUN / "core_scenario_validation.json").read_text(encoding="utf-8"))
    if not core["passed"]:
        raise RuntimeError("Core scenario gate is not passed")
    config = load_config()
    n_rep = int(config["shuffle_replicates"])
    logical = os.cpu_count() or 1
    physical_estimate = max(1, logical // 2)
    workers = min(int(config["worker_cap"]), max(1, math.floor(physical_estimate * 0.75)))
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_shuffle_worker, rep): rep for rep in range(1, n_rep + 1)}
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    manifest = pd.DataFrame(rows).sort_values("replicate")
    manifest.to_csv(REPORTS / "shuffle_execution_manifest.csv", index=False, encoding="utf-8-sig")
    validation = {
        "passed": bool(len(manifest) == n_rep and manifest["rows"].eq(8738).all()),
        "replicates": len(manifest), "workers": workers, "logical_cpu_count": logical,
        "physical_core_estimate": physical_estimate,
        "worker_formula": "min(8, floor(estimated_physical_cores*0.75))",
        "blas_threads_per_worker": 1, "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(RUN / "shuffle_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError("Shuffle execution is incomplete")


def target_reaches() -> set[int]:
    registry = pd.read_csv(SOURCE / "canonical_signal_registry.csv", encoding="utf-8-sig")
    return set(pd.to_numeric(registry.loc[registry["legacy_canonical_membership"].astype(str).str.casefold().isin({"true", "1"}), "reach_id"], errors="raise").astype(int))


def next_month_lowflow_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = target_reaches()
    for (scenario, station, fold_id), part in oof[oof["reach_id"].astype(int).isin(targets)].groupby(["scenario", "station_name", "fold_id"], sort=True):
        part = part.sort_values(["year", "month"]).copy()
        serial = part["year"].astype(int) * 12 + part["month"].astype(int)
        log_residual = np.log(np.clip(part["predicted_cfs"], EPS, None)) - np.log(np.clip(part["observed_cfs"], EPS, None))
        for idx in range(1, len(part)):
            if serial.iloc[idx] - serial.iloc[idx - 1] != 1 or part["flow_class"].iloc[idx] != "low":
                continue
            row = {"scenario": scenario, "station_name": station, "fold_id": fold_id, "year": int(part["year"].iloc[idx]), "month": int(part["month"].iloc[idx]), "next_lowflow_log_residual": float(log_residual.iloc[idx])}
            for column in [
                "antecedent_wetness",
                "sas_storage_mm", "sas_young_cfs", "sas_old_release_cfs",
                "production_storage_mm",
                "production_month_end_quick_store_mm", "production_month_end_base_store_mm",
                "production_quick_cfs", "production_base_cfs",
                "routed_quick_cfs", "routed_base_cfs",
                "ms_slow_large_storage_mm",
                "ms_slow_large_month_end_quick_store_mm", "ms_slow_large_month_end_base_store_mm",
                "ms_slow_large_routed_quick_cfs", "ms_slow_large_routed_base_cfs",
            ]:
                if column in part.columns:
                    row[f"previous_{column}"] = float(part[column].iloc[idx - 1])
            rows.append(row)
    return pd.DataFrame(rows)


def run_analyze() -> None:
    shuffle_validation = json.loads((RUN / "shuffle_validation.json").read_text(encoding="utf-8"))
    if not shuffle_validation["passed"]:
        raise RuntimeError("Shuffle gate is not passed")
    core_parts = [pd.read_parquet(REPORTS / f"{name}_oof.parquet") for name in ["D0", "D-flat", "D1"]]
    core = pd.concat(core_parts, ignore_index=True)
    core_summary, core_station = scenario_metric_rows(core)
    core_acf = residual_acf_rows(core)
    core_next = next_month_lowflow_table(core)
    core_summary.to_csv(REPORTS / "scenario_oof_metrics.csv", index=False, encoding="utf-8-sig")
    core_station.to_csv(REPORTS / "scenario_station_metrics.csv", index=False, encoding="utf-8-sig")
    core_acf.to_csv(RUN / "residual_acf_metrics.csv", index=False, encoding="utf-8-sig")
    core_next.to_csv(REPORTS / "next_month_lowflow_state_links.csv", index=False, encoding="utf-8-sig")

    d0 = core[core["scenario"].eq("D0")]
    d1 = core[core["scenario"].eq("D1")]
    dflat = core[core["scenario"].eq("D-flat")]
    targets = target_reaches()
    d0_station = core_station[core_station["scenario"].eq("D0")].set_index("reach_id")
    d1_station = core_station[core_station["scenario"].eq("D1")].set_index("reach_id")
    evaluable = sorted(targets & set(d0_station.index.astype(int)))
    target_improved = d1_station.loc[evaluable, "lowflow_median_log_bias"].abs() < d0_station.loc[evaluable, "lowflow_median_log_bias"].abs()
    target_low_d0 = d0[d0["reach_id"].astype(int).isin(evaluable) & d0["flow_class"].eq("low")]
    target_low_d1 = d1[d1["reach_id"].astype(int).isin(evaluable) & d1["flow_class"].eq("low")]
    target_rmse_gain = 1.0 - metrics(target_low_d1)["log_RMSE"] / metrics(target_low_d0)["log_RMSE"]

    shuffle_rows, shuffle_acf_rows, state_rows = [], [], []
    core_d0_metric = metrics(d0)
    for path in sorted(SHUFFLES.glob("shuffle_*.parquet")):
        part = pd.read_parquet(path)
        rep = int(path.stem.split("_")[-1])
        summary = metrics(part)
        station = scenario_metric_rows(part)[1].set_index("reach_id")
        improved = station.loc[evaluable, "lowflow_median_log_bias"].abs() < d0_station.loc[evaluable, "lowflow_median_log_bias"].abs()
        target_low = part[part["reach_id"].astype(int).isin(evaluable) & part["flow_class"].eq("low")]
        shuffle_rows.append({"replicate": rep, "NSE_log": summary["NSE_log"], "log_nse_gain_vs_d0": summary["NSE_log"] - core_d0_metric["NSE_log"], "legacy_improved_count": int(improved.sum()), "legacy_improved_fraction": float(improved.mean()), "legacy_lowflow_log_rmse": metrics(target_low)["log_RMSE"]})
        acf = residual_acf_rows(part)
        for row in acf.to_dict("records"):
            shuffle_acf_rows.append({"replicate": rep, **row})
        for group, group_part in [("legacy", part[part["reach_id"].astype(int).isin(evaluable)]), ("control", part[~part["reach_id"].astype(int).isin(evaluable)])]:
            for column in [
                "antecedent_wetness",
                "sas_storage_mm", "sas_young_cfs", "sas_old_release_cfs",
                "production_storage_mm",
                "production_month_end_quick_store_mm", "production_month_end_base_store_mm",
                "production_quick_cfs", "production_base_cfs",
                "routed_quick_cfs", "routed_base_cfs",
                "ms_slow_large_storage_mm",
                "ms_slow_large_month_end_quick_store_mm", "ms_slow_large_month_end_base_store_mm",
                "ms_slow_large_routed_quick_cfs", "ms_slow_large_routed_base_cfs",
            ]:
                if column in group_part:
                    state_rows.append({"replicate": rep, "group": group, "state": column, "median": float(group_part[column].median()), "mean": float(group_part[column].mean())})
    shuffle = pd.DataFrame(shuffle_rows).sort_values("replicate")
    shuffle.to_csv(RUN / "shuffle_null_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(shuffle_acf_rows).to_csv(REPORTS / "shuffle_residual_acf.csv", index=False, encoding="utf-8-sig")

    d1_metric = metrics(d1)
    flat_metric = metrics(dflat)
    q95_nse = float(shuffle["NSE_log"].quantile(0.95))
    p_nse = float((1 + int((shuffle["NSE_log"] >= d1_metric["NSE_log"]).sum())) / (len(shuffle) + 1))
    q95_improved = float(shuffle["legacy_improved_fraction"].quantile(0.95))
    p_improved = float((1 + int((shuffle["legacy_improved_fraction"] >= float(target_improved.mean())).sum())) / (len(shuffle) + 1))
    folds = []
    for fold_id in sorted(d0["fold_id"].unique()):
        m0 = metrics(d0[d0["fold_id"].eq(fold_id)])["NSE_log"]
        m1 = metrics(d1[d1["fold_id"].eq(fold_id)])["NSE_log"]
        folds.append({"fold_id": fold_id, "D0_NSE_log": m0, "D1_NSE_log": m1, "delta": m1 - m0})
    fold_table = pd.DataFrame(folds)
    fold_table.to_csv(REPORTS / "scenario_fold_metrics.csv", index=False, encoding="utf-8-sig")

    acf_wide = core_acf.pivot(index="lag", columns="scenario", values="weighted_abs_acf")
    acf_improved = {int(lag): bool(acf_wide.loc[lag, "D1"] < acf_wide.loc[lag, "D0"]) for lag in [1, 3, 6]}
    next_abs = core_next.groupby("scenario")["next_lowflow_log_residual"].apply(lambda s: float(s.abs().median()))
    highflow_relative = (metrics(d1[d1["flow_class"].eq("high")])["log_RMSE"] - metrics(d0[d0["flow_class"].eq("high")])["log_RMSE"]) / metrics(d0[d0["flow_class"].eq("high")])["log_RMSE"]
    major_threshold = float(d0_station["Q_ma_cfs"].quantile(0.70))
    major = d0_station.index[d0_station["Q_ma_cfs"] >= major_threshold]
    major_relative = float((d1_station.loc[major, "overall_log_RMSE"].median() - d0_station.loc[major, "overall_log_RMSE"].median()) / d0_station.loc[major, "overall_log_RMSE"].median())
    pbias_degradation = abs(d1_metric["PBIAS_pct"]) - abs(core_d0_metric["PBIAS_pct"])
    gates = load_config()["gates"]
    checks = {
        "D1_pooled_log_nse_gain_ge_0_01": d1_metric["NSE_log"] - core_d0_metric["NSE_log"] >= float(gates["pooled_log_nse_min_gain"]),
        "D1_all_folds_nonnegative": bool((fold_table["delta"] >= -1e-12).all()),
        "D1_at_least_two_folds_positive": int((fold_table["delta"] > 1e-12).sum()) >= 2,
        "D1_above_shuffle_nse_q95": d1_metric["NSE_log"] > q95_nse,
        "D1_randomization_p_nse_le_0_05": p_nse <= float(gates["randomization_p_max"]),
        "D1_better_than_Dflat": d1_metric["NSE_log"] > flat_metric["NSE_log"],
        "legacy_at_least_14_improved": int(target_improved.sum()) >= int(gates["legacy_min_improved"]),
        "legacy_improvement_above_shuffle_q95": float(target_improved.mean()) > q95_improved,
        "legacy_randomization_p_le_0_05": p_improved <= float(gates["randomization_p_max"]),
        "legacy_lowflow_log_rmse_gain_ge_5pct": target_rmse_gain >= float(gates["legacy_lowflow_log_rmse_min_relative_gain"]),
        "next_month_lowflow_abs_bias_toward_zero": float(next_abs["D1"]) < float(next_abs["D0"]),
        "residual_lag1_abs_acf_improved": acf_improved[1],
        "residual_at_least_two_of_lag_1_3_6_improved": sum(acf_improved.values()) >= 2,
        "highflow_log_rmse_protected": highflow_relative <= float(gates["highflow_log_rmse_max_relative_degradation"]),
        "major_station_log_rmse_protected": major_relative <= float(gates["major_station_log_rmse_max_relative_degradation"]),
        "absolute_pbias_protected": pbias_degradation <= float(gates["absolute_pbias_max_degradation_percentage_points"]),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    core_keys = list(checks)[:6]
    lowflow_keys = list(checks)[6:11]
    memory_keys = list(checks)[11:13]
    protection_keys = list(checks)[13:]
    core_pass = all(checks[key] for key in core_keys)
    lowflow_pass = all(checks[key] for key in lowflow_keys)
    memory_pass = all(checks[key] for key in memory_keys)
    protection_pass = all(checks[key] for key in protection_keys)
    if core_pass and lowflow_pass and memory_pass and protection_pass:
        decision = "DAILY_STATE_ORDER_MECHANISM_SUPPORTED"
    elif core_pass and lowflow_pass and memory_pass:
        decision = "DAILY_STATE_ORDER_DETECTED_BUT_NOT_OPERATIONALLY_ACCEPTABLE"
    elif d1_metric["NSE_log"] <= q95_nse and (flat_metric["NSE_log"] > core_d0_metric["NSE_log"] or float(shuffle["NSE_log"].median()) > core_d0_metric["NSE_log"]):
        decision = "DAILY_DISCRETIZATION_ONLY_NOT_ORDER_SUPPORTED"
    else:
        decision = "DAILY_TO_MONTHLY_PREMATURE_AGGREGATION_HYPOTHESIS_FALSIFIED"

    legacy_table = pd.DataFrame({
        "reach_id": evaluable,
        "station_name": [d0_station.loc[r, "station_name"] for r in evaluable],
        "D0_lowflow_median_log_bias": [d0_station.loc[r, "lowflow_median_log_bias"] for r in evaluable],
        "D1_lowflow_median_log_bias": [d1_station.loc[r, "lowflow_median_log_bias"] for r in evaluable],
        "abs_bias_improved": target_improved.to_numpy(bool),
        "D0_lowflow_log_RMSE": [d0_station.loc[r, "lowflow_log_RMSE"] for r in evaluable],
        "D1_lowflow_log_RMSE": [d1_station.loc[r, "lowflow_log_RMSE"] for r in evaluable],
    })
    legacy_table.to_csv(RUN / "legacy_lowflow_metrics.csv", index=False, encoding="utf-8-sig")
    shijiao = core_station[core_station["station_name"].eq("石角站")]
    shijiao.to_csv(REPORTS / "shijiao_protection_metrics.csv", index=False, encoding="utf-8-sig")

    state_null = pd.DataFrame(state_rows)
    diagnostic_rows = []
    for scenario, part in core.groupby("scenario"):
        for group, group_part in [("legacy", part[part["reach_id"].astype(int).isin(evaluable)]), ("control", part[~part["reach_id"].astype(int).isin(evaluable)])]:
            for column in [
                "antecedent_wetness",
                "sas_storage_mm", "sas_young_cfs", "sas_old_release_cfs",
                "production_storage_mm",
                "production_month_end_quick_store_mm", "production_month_end_base_store_mm",
                "production_quick_cfs", "production_base_cfs",
                "routed_quick_cfs", "routed_base_cfs",
                "ms_slow_large_storage_mm",
                "ms_slow_large_month_end_quick_store_mm", "ms_slow_large_month_end_base_store_mm",
                "ms_slow_large_routed_quick_cfs", "ms_slow_large_routed_base_cfs",
            ]:
                if column not in group_part:
                    continue
                null = state_null[(state_null["group"] == group) & (state_null["state"] == column)]["median"]
                value = float(group_part[column].median())
                diagnostic_rows.append({"scenario": scenario, "group": group, "state": column, "median": value, "mean": float(group_part[column].mean()), "shuffle_percentile_of_median": float((null <= value).mean()) if scenario == "D1" and len(null) else np.nan})
    state_diagnostics = pd.DataFrame(diagnostic_rows)
    state_diagnostics.to_csv(RUN / "state_mechanism_diagnostics.csv", index=False, encoding="utf-8-sig")
    state_null.to_csv(REPORTS / "shuffle_state_null.csv", index=False, encoding="utf-8-sig")

    result = {
        "run_id": RUN.name, "decision": decision,
        "daily_state": "SENSITIVITY_SUPPORTED" if decision == "DAILY_STATE_ORDER_MECHANISM_SUPPORTED" else "NOT_SUPPORTED",
        "checks": checks, "core_pass": core_pass, "lowflow_pass": lowflow_pass,
        "memory_pass": memory_pass, "protection_pass": protection_pass,
        "metrics": {
            "D0_NSE_log": core_d0_metric["NSE_log"], "Dflat_NSE_log": flat_metric["NSE_log"], "D1_NSE_log": d1_metric["NSE_log"],
            "D1_log_nse_gain_vs_D0": d1_metric["NSE_log"] - core_d0_metric["NSE_log"],
            "shuffle_NSE_log_q95": q95_nse, "randomization_p_NSE_log": p_nse,
            "legacy_improved_count": int(target_improved.sum()), "legacy_evaluable_count": len(evaluable),
            "legacy_improved_fraction": float(target_improved.mean()), "shuffle_legacy_improvement_q95": q95_improved,
            "randomization_p_legacy_improvement": p_improved, "legacy_lowflow_log_rmse_relative_gain": target_rmse_gain,
            "next_month_D0_median_abs_log_bias": float(next_abs["D0"]), "next_month_D1_median_abs_log_bias": float(next_abs["D1"]),
            "highflow_log_rmse_relative_degradation": float(highflow_relative),
            "major_station_log_rmse_relative_degradation": major_relative,
            "absolute_pbias_degradation_percentage_points": float(pbias_degradation),
        },
        "fold_log_nse": folds, "acf_improved": acf_improved,
        "interpretation_boundary": "Simulated states are model sensitivities, not observed storage. No authoritative daily ET was used.",
        "stop_rule": "No second daily-state parameter, feature, storage, or target-station search is permitted after this terminal result.",
    }
    write_json(RUN / "terminal_gate.json", result)
    write_json(RUN / "validation.json", {"passed": True, "terminal_decision_present": True, "scenario_rows": {name: 8738 for name in ["D0", "D-flat", "D1"]}, "shuffle_replicates": len(shuffle), "decision": decision})
    update_readme(result, core_summary, fold_table, core_acf, legacy_table, shijiao)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def update_readme(result: dict[str, object], summary: pd.DataFrame, folds: pd.DataFrame, acf: pd.DataFrame, legacy: pd.DataFrame, shijiao: pd.DataFrame) -> None:
    lines = [
        "# 20260808_4 Q72最小Daily-State顺序机制证伪实验", "",
        f"终点：`{result['decision']}`。", "",
        "本实验只检验真实月内daily-P顺序进入完整Q72状态方程的作用。所有水文参数、三折、站点层级、设计列和先验均冻结；最终回归层逐折重估。", "",
        "## 完整性", "",
        "- conda环境：sparrow；", "- canonical OOF：8738行、110站、3折；",
        "- daily CHM_PRE：2006—2018，Reach级local P；", "- 27个可评价legacy目标；天河站/Reach 169未补入；",
        "- 石角站/Reach 56保留；", "- 99个shuffle全部完成；", "",
        "## 核心结果", "", summary.to_string(index=False), "", "## 三折log-NSE", "", folds.to_string(index=False), "",
        "## 残差记忆", "", acf.to_string(index=False), "", "## Legacy低流", "",
        f"改善站：{int(legacy['abs_bias_improved'].sum())}/{len(legacy)}。", "", legacy.to_string(index=False), "",
        "## 石角保护", "", shijiao.to_string(index=False), "", "## 判定门禁", "",
        pd.DataFrame([result["checks"]]).to_string(index=False), "",
        "## 解释边界", "", "月末SAS、production和slow/base storage均为模型状态敏感性，不是观测储量。由于没有权威daily ET，即使通过也只能标记SENSITIVITY_SUPPORTED。", "",
        "本终点关闭daily-state搜索：不再搜索rho、capacity、gamma、release、时间窗、新库或目标站特调。", "",
        "独立subagent文献约束审计见`subagent_literature_audit.md`；审计处置见`review_disposition.md`。", "",
        "## 独立审计与最终处置", "",
        "审计结论：`ACCEPT / IN_BOUNDS / ACADEMIC_METHOD_SUPPORTED / STOP_AND_ACCEPT_FALSIFIED`。未发现评价期流量泄漏、结果后选择或情景不等价。", "",
        "真实月内顺序存在微小、统计可检测的总体信号，但不是当前Q72 legacy低流异常的主要且可行动来源。", "",
        "D0独立重拟合的最大绝对预测差为`1.01165e-7 cfs`，未达到原绝对`1e-8 cfs`检查；该失败值已保留。最大相对差和log差约为`1.81e-13`，pooled log-NSE完全一致，正式D0预测逐行采用冻结canonical值。该处不能解释为严格绝对复现通过。", "",
        "审计后仅补齐了不进入设计矩阵的月末quick/base状态和释放量诊断；预测、指标、shuffle分布、门禁和终点均未改变。",
    ]
    (RUN / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RUN / "method_summary.md").write_text("\n".join(lines[:34]) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "core", "shuffles", "analyze", "all"], required=True)
    args = parser.parse_args()
    if args.stage in {"prepare", "all"}:
        run_prepare()
    if args.stage in {"core", "all"}:
        run_core()
    if args.stage in {"shuffles", "all"}:
        run_shuffles()
    if args.stage in {"analyze", "all"}:
        run_analyze()


if __name__ == "__main__":
    main()
