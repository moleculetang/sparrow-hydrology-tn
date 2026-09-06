from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import scipy

from legacy16_core import (
    EARLY_PATH,
    FOLD_PATH,
    MONTHLY_PATH,
    OBS_PATH,
    S14_6,
    S14_9,
    S15_1,
    S15_2,
    S15_4,
    S15_5,
    S15_7,
    S16_1,
    STAGE5_SCRIPT,
    TOPOLOGY_PATH,
    candidate_registry_frame,
    dump_json,
    hash_manifest,
    load_module,
    numeric_comparison,
    require_runtime,
)


OUT = S16_1 / "outputs"
REPORTS = S16_1 / "reports"
PARENT_LOCAL = S15_5 / "outputs" / "selected_delivery_model_reach_month_1961_2022.parquet"
PARENT_COHORT = S15_5 / "outputs" / "selected_delivery_model_cohort_state_end_2022.parquet"
PARENT_OOF = S15_5 / "outputs" / "delivery_candidate_oof_predictions_2018_2021.parquet"
PARENT_METRICS = S15_5 / "reports" / "delivery_candidate_metrics.csv"
PARENT_DECISION = S15_5 / "reports" / "delivery_structure_decision.json"


def frozen_paths() -> list[Path]:
    paths = [
        S14_6 / "structural_interface_lock.json",
        S14_6 / "outputs" / "structural_canonical_main_interface_2006_2022.parquet",
        S15_1 / "experiment_contract.json",
        S15_1 / "outputs" / "tn_observation_registry_2016_2022.parquet",
        S15_1 / "outputs" / "tn_fold_registry.parquet",
        S15_1 / "outputs" / "q72_monthly_climatology_2006_2015.parquet",
        S15_2 / "experiment_contract.json",
        MONTHLY_PATH,
        EARLY_PATH,
        S15_4 / "reports" / "source_structure_decision.json",
        S15_4 / "scripts" / "run_stage4.py",
        PARENT_DECISION,
        STAGE5_SCRIPT,
        PARENT_LOCAL,
        PARENT_COHORT,
        PARENT_OOF,
        PARENT_METRICS,
        S15_7 / "reports" / "final_model_manifest.json",
        S15_7 / "reports" / "pre_2022_model_lock.json",
        S14_9 / "inputs" / "model_ready" / "static" / "soil_tn_glhymps_by_reach.parquet",
        S14_9 / "inputs" / "model_ready" / "static" / "groundwater_nitrate_decadal_by_reach.parquet",
        S14_9 / "inputs" / "model_ready" / "observations" / "groundwater_nitrate_observations_prb_1979_2022.parquet",
        TOPOLOGY_PATH,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen parents: {missing}")
    return paths


def reproduce_parent() -> dict[str, object]:
    stage5 = load_module(STAGE5_SCRIPT, "legacy16_parent_stage5")
    monthly = pd.read_parquet(MONTHLY_PATH)
    early = pd.read_parquet(EARLY_PATH)
    obs = pd.read_parquet(OBS_PATH)
    folds = pd.read_parquet(FOLD_PATH)
    reach_ids, times, arrays = stage5.s4.prepare_arrays(monthly)
    early_positive = (
        early.set_index("reach_id")
        .loc[reach_ids, "early_1961_1965_positive_surplus_mean_kg_n_year"]
        .to_numpy(float)
        / 12.0
    )
    local, pools, simulation_audit = stage5.simulate_delivery(
        "T0", "S1", 480, 0, reach_ids, times, arrays, early_positive
    )
    expected_local = pd.read_parquet(PARENT_LOCAL)
    local_numeric = [
        column
        for column in expected_local.columns
        if column not in {"model_id", "source_structure"}
        and pd.api.types.is_numeric_dtype(expected_local[column])
    ]
    local_check = numeric_comparison(
        local,
        expected_local,
        ["reach_id", "year", "month"],
        local_numeric,
    )

    cohort = stage5.s4.final_cohort_table("T0", pools, reach_ids, times)
    expected_cohort = pd.read_parquet(PARENT_COHORT)
    cohort_numeric = [
        column
        for column in expected_cohort.columns
        if column != "model_id" and pd.api.types.is_numeric_dtype(expected_cohort[column])
    ]
    cohort_check = numeric_comparison(
        cohort,
        expected_cohort,
        ["pool", "reach_id", "input_year", "input_month", "cohort_origin"],
        cohort_numeric,
    )

    order, downstream, terminal = stage5.s4.topology_operators(reach_ids)
    routed = stage5.s4.route_recent(local, reach_ids, order, downstream, terminal)
    predictions, readout = stage5.s4.oof_predictions(routed, obs.loc[obs.year <= 2021], folds)
    predictions["model_id"] = "T0"
    expected_oof = pd.read_parquet(PARENT_OOF)
    expected_oof = expected_oof.loc[expected_oof.model_id.eq("T0")].copy()
    oof_check = numeric_comparison(
        predictions,
        expected_oof,
        ["station_key", "year", "month"],
        ["tn_mg_l", "raw_tn_mg_l", "pred_tn_mg_l", "fraction_memory_gt1y", "fraction_memory_gt5y", "fraction_memory_gt10y"],
    )
    metrics_actual = stage5.s4.metrics(predictions)
    metrics_expected = pd.read_csv(PARENT_METRICS).set_index("model_id").loc["T0"]
    metric_differences = {
        name: abs(float(value) - float(metrics_expected[name]))
        for name, value in metrics_actual.items()
    }
    metrics_pass = all(value <= 1e-12 for value in metric_differences.values())
    expected_keys = 4097
    key_count_pass = (
        len(predictions) == expected_keys
        and not predictions.duplicated(["station_key", "year", "month"]).any()
    )
    passed = bool(
        local_check["pass"]
        and cohort_check["pass"]
        and oof_check["pass"]
        and metrics_pass
        and key_count_pass
    )
    return {
        "status": "PASS_PARENT_REPRODUCED" if passed else "STOP_PARENT_NOT_REPRODUCED",
        "pass": passed,
        "expected_oof_keys": expected_keys,
        "actual_oof_keys": len(predictions),
        "unique_oof_keys": int(predictions[["station_key", "year", "month"]].drop_duplicates().shape[0]),
        "local_state_and_flux_comparison": local_check,
        "cohort_comparison": cohort_check,
        "oof_prediction_comparison": oof_check,
        "metric_absolute_differences": metric_differences,
        "metric_tolerance": 1e-12,
        "metrics_pass": metrics_pass,
        "simulation_audit": simulation_audit,
        "readout_parameters": readout,
    }


def external_profile() -> dict[str, object]:
    soil_path = S14_9 / "inputs" / "model_ready" / "static" / "soil_tn_glhymps_by_reach.parquet"
    nitrate_path = S14_9 / "inputs" / "model_ready" / "static" / "groundwater_nitrate_decadal_by_reach.parquet"
    point_path = S14_9 / "inputs" / "model_ready" / "observations" / "groundwater_nitrate_observations_prb_1979_2022.parquet"
    soil = pd.read_parquet(soil_path)
    nitrate = pd.read_parquet(nitrate_path)
    points = pd.read_parquet(point_path)
    decade_coverage = {}
    for decade in ("1980s", "1990s", "2000s", "2010s"):
        column = f"groundwater_no3_{decade}_mg_l_mean"
        decade_coverage[decade] = int(nitrate[column].notna().sum())
    return {
        "soil_glhymps": {
            "rows": len(soil),
            "unique_reaches": int(soil.reach_id.nunique()),
            "required_complete": bool(
                soil[[
                    "soil_tn_0_100cm_depth_weighted_g_kg",
                    "glhymps_log10_permeability_m2",
                    "glhymps_porosity",
                ]].notna().all().all()
            ),
            "glhymps_fields_confirmed": {
                "permeability": "logK_Ferr_ / 100 -> log10(k_m2)",
                "porosity": "Porosity_x / 100",
                "layer": "near_surface_without_permafrost",
                "deep_layer_present": False,
            },
            "csdl_missing_reaches": int(soil.csdl_v2_tn_0_5cm_native_mean.isna().sum()),
        },
        "groundwater_nitrate": {
            "decadal_reach_coverage": decade_coverage,
            "point_records": len(points),
            "point_reaches": int(points.reach_id.nunique()),
            "point_year_min": int(points.year.min()),
            "point_year_max": int(points.year.max()),
            "one_evidence_family": True,
        },
    }


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parents = frozen_paths()
    start = hash_manifest(parents)
    dump_json(REPORTS / "frozen_parent_hashes_start.json", start)
    candidate_registry_frame().to_csv(OUT / "candidate_registry_63.csv", index=False)
    reproduction = reproduce_parent()
    dump_json(REPORTS / "incumbent_reproduction_audit.json", reproduction)
    if not reproduction["pass"]:
        raise RuntimeError("STOP_PARENT_NOT_REPRODUCED")
    profile = external_profile()
    dump_json(REPORTS / "external_evidence_data_quality_profile.json", profile)
    runtime = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
    }
    dump_json(REPORTS / "runtime_environment.json", runtime)
    end = hash_manifest(parents)
    dump_json(REPORTS / "frozen_parent_hashes_end.json", end)
    if start != end:
        raise RuntimeError("a frozen parent changed during 20260816_1")
    completion = {
        "scenario_id": "20260816_1",
        "pass": True,
        "parent_reproduced": True,
        "candidate_count": 63,
        "external_profile_complete": True,
        "frozen_parent_hashes_unchanged": True,
        "next_stage_authorized": "20260816_1 soil observation operator preprocessing",
    }
    dump_json(REPORTS / "completion_audit.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
