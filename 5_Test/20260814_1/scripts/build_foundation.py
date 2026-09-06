from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from hydrology_core import BRANCHES, crosscheck_component, load_component, prepare_forcing, simulate_local_interface
from runtime_guard import assert_sparrow_runtime


warnings.filterwarnings("ignore")
RUNTIME = assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]
DEV_INPUT = ROOT / "inputs" / "development_indata_2006_2018.parquet"
OOF = ROOT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet"
REFERENCE = ROOT / "reference_selected" / "oof.parquet"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
REG = ROOT / "inputs" / "registries"
KEY = ["comid", "q_site", "year", "month", "fold_id"]
FOLDS = [
    ("fit_2006_2011_eval_2012_2013", 2011, 2012, 2013),
    ("fit_2006_2013_eval_2014_2015", 2013, 2014, 2015),
    ("fit_2006_2015_eval_2016_2018", 2015, 2016, 2018),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def audit_h0() -> dict[str, object]:
    new = pd.read_parquet(OOF)
    ref = pd.read_parquet(REFERENCE)
    if new[KEY].duplicated().any() or ref[KEY].duplicated().any():
        raise RuntimeError("H0 duplicate OOF keys")
    merged = new.merge(ref, on=KEY, suffixes=("_new", "_ref"), validate="one_to_one")
    diff = float(np.max(np.abs(merged["predict_new"] - merged["predict_ref"])))
    actual_diff = float(np.max(np.abs(merged["actual_new"] - merged["actual_ref"])))
    payload = {
        "rows_new": int(len(new)), "rows_reference": int(len(ref)), "matched_rows": int(len(merged)),
        "max_abs_prediction_difference_cfs": diff,
        "max_abs_observation_difference_cfs": actual_diff,
        "prediction_tolerance_cfs": 1e-8,
        "pass": bool(len(new) == len(ref) == len(merged) == 7755 and diff <= 1e-8 and actual_diff == 0.0),
        "new_oof_sha256": sha256(OOF), "reference_oof_sha256": sha256(REFERENCE),
    }
    if not payload["pass"]:
        raise RuntimeError(f"H0 reproduction failed: {payload}")
    json_write(REPORTS / "h0_reproduction_audit.json", payload)
    return payload


def configured_development_component():
    module = load_component("foundation_development")
    module.INPUT_PATH = DEV_INPUT
    module.set_et_feature_block_mode("full")
    return module


def build_design_audit(module) -> dict[str, object]:
    dev = pd.read_parquet(DEV_INPUT, columns=["q_site", "Q_obsv_cfs"])
    stations = sorted(dev.loc[dev.Q_obsv_cfs.notna() & dev.Q_obsv_cfs.gt(0), "q_site"].astype(str).unique())
    h0_meta = module.design_column_metadata(stations)
    h0_count = int(len(h0_meta))
    h0_features = list(module.FIXED_FEATURES)
    multistore = set(module.MULTISTORE_FEATURES)
    module.FIXED_FEATURES = [f for f in module.FIXED_FEATURES if f not in multistore]
    module.MULTISTORE_FEATURES = []
    single_meta = module.design_column_metadata(stations)
    payload = {
        "H0_design_columns": h0_count,
        "single_production_design_columns": int(len(single_meta)),
        "H0_fixed_features": h0_features,
        "single_fixed_features": list(module.FIXED_FEATURES),
        "removed_multistore_features": sorted(multistore.intersection(h0_features)),
        "stations": len(stations),
        "preprocessing": "fold_training_mean_and_population_std_as_frozen_H0",
        "prior_space": "full_independent_Gaussian_column_space_with_zero_preserving_gates",
        "interaction_order": "prepare_design_then_fold_training_standardization_then_matrix_gates",
        "pass": h0_count == 3391 and len(single_meta) == 3383,
    }
    if not payload["pass"]:
        raise RuntimeError(f"Design audit failed: {payload}")
    h0_meta.to_csv(REPORTS / "h0_design_column_manifest.csv", index=False, encoding="utf-8-sig")
    single_meta.to_csv(REPORTS / "single_design_column_manifest.csv", index=False, encoding="utf-8-sig")
    json_write(REPORTS / "baseline_feature_transform_audit.json", payload)
    return payload


def build_interfaces() -> tuple[pd.DataFrame, dict[str, object]]:
    module = load_component("foundation_interfaces")
    forcing = prepare_forcing(module)
    parts = []
    checks: dict[str, object] = {}
    for name, spec in BRANCHES.items():
        state = simulate_local_interface(module, forcing, spec)
        checks[name] = crosscheck_component(module, forcing, state, spec)
        parts.append(state)
    states = pd.concat(parts, ignore_index=True)
    path = OUT / "fixed_branch_local_states.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    states.to_parquet(path, index=False)
    states[["branch_id", "source_water_capacity_mm", "runoff_gamma", "quick_rho", "k_p", "rho_b", "highflow_scale"]].drop_duplicates().to_csv(OUT / "fixed_branch_parameters.csv", index=False, encoding="utf-8-sig")

    physical = [c for c in states.columns if c.endswith("_mm") and c != "mass_balance_error_mm"]
    min_physical = float(states[physical].min().min())
    max_mass = float(states.mass_balance_error_mm.abs().max())
    alias = float(np.max(np.abs(states.q_local_total_mm - states.quick_release_mm - states.gw_discharge_mm)))
    pre = float(np.max(np.abs(states.source_store_pre_recharge_mm - (states.source_store_start_mm + states.source_positive_input_to_store_mm - states.soil_overflow_to_quick_mm))))
    recharge = float(np.max(np.abs(states.gw_recharge_mm - states.k_p * states.source_store_pre_recharge_mm)))
    source_end = float(np.max(np.abs(states.source_store_end_mm - (states.source_store_pre_recharge_mm - states.gw_recharge_mm))))
    quick_input = float(np.max(np.abs(states.quick_input_mm - states.quick_generated_mm - states.soil_overflow_to_quick_mm)))
    cross_max = max(float(v) for branch in checks.values() for v in branch.values())
    payload = {
        "rows": int(len(states)), "branches": int(states.branch_id.nunique()),
        "reaches": int(states.comid.nunique()), "months": int(states[["year", "month"]].drop_duplicates().shape[0]),
        "minimum_physical_storage_or_flux_mm": min_physical,
        "max_abs_mass_balance_error_mm": max_mass,
        "max_q_local_total_identity_error_mm": alias,
        "max_source_pre_recharge_identity_error_mm": pre,
        "max_gw_recharge_identity_error_mm": recharge,
        "max_source_end_identity_error_mm": source_end,
        "max_quick_input_identity_error_mm": quick_input,
        "max_component_crosscheck_error": cross_max,
        "component_crosschecks": checks,
        "pass": bool(len(states) == 234600 and min_physical >= -1e-10 and max_mass <= 1e-9 and max(alias, pre, recharge, source_end, quick_input, cross_max) <= 1e-8),
        "sha256": sha256(path),
    }
    if not payload["pass"]:
        raise RuntimeError(f"Interface gate failed: {payload}")
    json_write(REPORTS / "local_interface_audit.json", payload)
    return states, payload


def build_wet_dry(states: pd.DataFrame) -> dict[str, object]:
    forcing = states.loc[states.branch_id.eq("main"), ["comid", "year", "month", "positive_input_mm"]].copy()
    rows = []
    for fold, train_end, es, ee in FOLDS:
        train = forcing[forcing.year <= train_end]
        thresholds = train.groupby("comid").positive_input_mm.quantile([1/3, 2/3]).unstack()
        thresholds.columns = ["dry_threshold_mm", "wet_threshold_mm"]
        evaluation = forcing[forcing.year.between(es, ee)].merge(thresholds, on="comid", validate="many_to_one")
        evaluation["wet_dry_class"] = np.where(evaluation.positive_input_mm <= evaluation.dry_threshold_mm, "dry", np.where(evaluation.positive_input_mm >= evaluation.wet_threshold_mm, "wet", "middle"))
        evaluation["fold_id"] = fold
        rows.append(evaluation)
    registry = pd.concat(rows, ignore_index=True)
    path = REG / "wet_dry_registry.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(path, index=False)
    payload = {"rows": int(len(registry)), "folds": int(registry.fold_id.nunique()), "hash_shared_by_all_branches": sha256(path), "candidate_specific": False}
    json_write(REPORTS / "wet_dry_registry_audit.json", payload)
    return payload


def build_tail_registry(module) -> dict[str, object]:
    forcing = prepare_forcing(module)
    featured = module.add_hydrologic_features(
        forcing, rho=0.70, wm=480.0, et_gamma=0.75, sas_rho=0.93,
        young_k=1.5, storage_scale=720.0, prod_capacity=240.0,
        runoff_gamma=2.5, quick_rho=0.25, base_rho=0.85, base_release=0.10,
    )
    cols = ["comid", "q_site", "year", "month", "Q_obsv_cfs", "upstream_positive_input_equivalent_cfs"]
    frame = featured[cols].dropna(subset=["q_site", "Q_obsv_cfs"]).copy()
    frame = frame[frame.Q_obsv_cfs > 0]
    events, months = [], []
    for fold, train_end, es, ee in FOLDS:
        for site, allsite in frame.groupby("q_site", sort=True):
            train = allsite[allsite.year <= train_end]
            evaluation = allsite[allsite.year.between(es, ee)].sort_values(["year", "month"]).reset_index(drop=True)
            if train.empty or evaluation.empty:
                continue
            q75 = float(train.Q_obsv_cfs.quantile(0.75))
            wet75 = float(train.upstream_positive_input_equivalent_cfs.quantile(0.75))
            prior_rows = allsite[(allsite.year < es)].sort_values(["year", "month"])
            if not prior_rows.empty and int(prior_rows.iloc[-1].year) == es - 1 and int(prior_rows.iloc[-1].month) == 12:
                prior_q = float(prior_rows.iloc[-1].Q_obsv_cfs)
            else:
                # A local peak requires an observed immediately preceding month.
                prior_q = np.inf
            for i in range(len(evaluation)):
                # Freeze events strictly inside the evaluation block so neither
                # edge can manufacture a one-sided local maximum.
                if i == 0 or i + 1 >= len(evaluation):
                    continue
                q = float(evaluation.at[i, "Q_obsv_cfs"])
                left = float(evaluation.at[i-1, "Q_obsv_cfs"])
                right = float(evaluation.at[i+1, "Q_obsv_cfs"]) if i+1 < len(evaluation) else -np.inf
                if not (q >= q75 and q >= left and q > right):
                    continue
                tail_rows = []
                previous = q
                stopped = False
                for j in range(i+1, min(i+7, len(evaluation))):
                    p0 = pd.Period(f"{int(evaluation.at[j-1,'year'])}-{int(evaluation.at[j-1,'month']):02d}")
                    p1 = pd.Period(f"{int(evaluation.at[j,'year'])}-{int(evaluation.at[j,'month']):02d}")
                    cur = float(evaluation.at[j, "Q_obsv_cfs"])
                    wet = float(evaluation.at[j, "upstream_positive_input_equivalent_cfs"])
                    if p1.ordinal-p0.ordinal != 1 or wet > wet75 or cur > previous:
                        stopped = True
                        break
                    tail_rows.append((j, cur, wet))
                    previous = cur
                if len(tail_rows) < 2:
                    continue
                event_id = f"{fold}::{site}::{int(evaluation.at[i,'year']):04d}-{int(evaluation.at[i,'month']):02d}"
                events.append({"event_id": event_id, "fold_id": fold, "comid": int(evaluation.at[i,"comid"]), "q_site": str(site), "peak_year": int(evaluation.at[i,"year"]), "peak_month": int(evaluation.at[i,"month"]), "peak_observed_cfs": q, "train_q75_cfs": q75, "train_upstream_positive_input_q75_cfs": wet75, "tail_month_count": len(tail_rows), "right_censored": False})
                for lag, (j, cur, wet) in enumerate(tail_rows, start=1):
                    months.append({"event_id": event_id, "fold_id": fold, "comid": int(evaluation.at[j,"comid"]), "q_site": str(site), "year": int(evaluation.at[j,"year"]), "month": int(evaluation.at[j,"month"]), "tail_lag": lag, "observed_cfs": cur, "upstream_positive_input_equivalent_cfs": wet})
    event_df = pd.DataFrame(events)
    month_df = pd.DataFrame(months)
    event_path = REG / "tail_event_registry.parquet"
    month_path = REG / "tail_month_registry.parquet"
    event_df.to_parquet(event_path, index=False)
    month_df.to_parquet(month_path, index=False)
    counts = event_df.groupby("fold_id").size().reindex([f[0] for f in FOLDS]).fillna(0).astype(int).tolist()
    payload = {"events": int(len(event_df)), "tail_months": int(len(month_df)), "events_by_fold": counts, "event_sha256": sha256(event_path), "month_sha256": sha256(month_path), "algorithm": "train_Q75_local_peak_then_2_to_6_monotone_tail_below_train_upstream_input_Q75_excluding_right_censor", "pass": len(event_df)==369 and len(month_df)==1206 and counts==[132,76,161]}
    if not payload["pass"]:
        raise RuntimeError(f"Tail registry count gate failed: {payload}")
    json_write(REPORTS / "tail_registry_audit.json", payload)
    return payload


def write_contract_manifests(h0: dict[str, object], design: dict[str, object], interface: dict[str, object]) -> None:
    import geopandas as gpd
    import pyarrow
    import scipy
    import shapely
    json_write(REPORTS / "environment.json", {
        "runtime_guard": RUNTIME, "python": sys.version, "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "geopandas": gpd.__version__, "shapely": shapely.__version__, "pyarrow": pyarrow.__version__,
        "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]},
    })
    json_write(REPORTS / "spinup_contract.json", {"mode":"deterministic_periodic_spinup", "initialization":"module_periodic_production_state", "state_timing":"start_pre_positive_input_end_post_release", "state_forcing_sha256": sha256(ROOT/"inputs"/"state_forcing_2006_2022_no_locked_observations.parquet")})
    json_write(REPORTS / "instrumentation_neutrality.json", {"h0_design_columns":design["H0_design_columns"], "oof_max_abs_difference_cfs":h0["max_abs_prediction_difference_cfs"], "interface_instrumentation_external_to_model_matrix":True, "pass":h0["pass"] and design["pass"]})
    parameters = {name: spec.__dict__ for name, spec in BRANCHES.items()}
    parameters["statistical_fixed"] = {"rho":0.70,"wm":480.0,"et_gamma":0.75,"sas_rho":0.93,"young_k":1.5,"storage_scale":720.0,"fixed_sigma":3.0,"production_sigma":1.5,"group_sigma":1.5,"multistore_sigma":0.30,"hysteresis_sigma":3.0,"station_sigma":1.0,"slope_sigma":0.15,"regime_slope_sigma":0.25}
    json_write(REPORTS / "parameter_manifest.json", parameters)
    schema = {}
    for field in pd.read_parquet(OUT/"fixed_branch_local_states.parquet").columns:
        cls = "statistical_feature"
        if field.endswith("_mm") and field != "mass_balance_error_mm": cls = "physical_flux" if any(token in field for token in ["input","generated","overflow","release","recharge","discharge","withdrawn","total"]) else "physical_storage"
        if field == "mass_balance_error_mm": cls = "signed_diagnostic"
        schema[field] = {"variable_class":cls, "nonnegative_required":cls in ["physical_flux","physical_storage"]}
    json_write(REPORTS / "interface_schema.json", schema)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True); REG.mkdir(parents=True, exist_ok=True)
    h0 = audit_h0()
    module = configured_development_component()
    design = build_design_audit(module)
    states, interface = build_interfaces()
    wetdry = build_wet_dry(states)
    tail = build_tail_registry(configured_development_component())
    write_contract_manifests(h0, design, interface)
    payload = {"h0":h0,"design":design,"interface":interface,"wet_dry":wetdry,"tail":tail,"pass":True}
    json_write(REPORTS/"foundation_non_gis_gate.json", payload)
    print(json.dumps({"pass":True,"h0_max_diff":h0["max_abs_prediction_difference_cfs"],"state_rows":interface["rows"],"tail_events":tail["events"],"tail_months":tail["tail_months"]}, indent=2))


if __name__ == "__main__":
    main()
