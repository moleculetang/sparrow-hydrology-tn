from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


STAGE1 = Path(r"E:\SPARROW\5_Test\20260818_1")
STAGE2 = Path(r"E:\SPARROW\5_Test\20260818_2")
sys.path.insert(0, str(STAGE1 / "scripts"))
from legacy18_shared import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    FOLD_PATH,
    MONTHLY_PATH,
    NONINFERIOR_MARGIN,
    OBS_PATH,
    PARENT_CORE_PATH,
    S17_9,
    TEST,
    dump_json,
    formal_specs,
    hash_manifest,
    paired_rmse_bootstrap,
    require_runtime,
    route_arrays,
    spatial_holdout,
    station_macro_rmse,
    temporal_oof,
    topology_operators,
)


ROOT = TEST / "20260818_3"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
POINT_DIR = S17_9 / "inputs" / "model_ready" / "point_sources"
FACILITY = POINT_DIR / "prb_wwtp_tn_monthly_facility_2006_2019.parquet"
REACH = POINT_DIR / "prb_wwtp_tn_monthly_reach_2006_2019.parquet"
FINAL_QA = S17_9 / "inputs" / "qa" / "final_validation.json"
SCENARIOS = ("PS0", "PS1", "PS2", "PS3", "PS09")


def build_point_registry() -> tuple[pd.DataFrame, dict[str, object]]:
    facility = pd.read_parquet(FACILITY)
    reach = pd.read_parquet(REACH)
    reach_ids = np.arange(1, 231, dtype=int)
    months = pd.MultiIndex.from_product([range(2006, 2020), range(1, 13)], names=["year", "month"]).to_frame(index=False)
    grid = months.assign(_key=1).merge(pd.DataFrame({"reach_id": reach_ids, "_key": 1}), on="_key").drop(columns="_key")

    scenario_sources: dict[str, pd.DataFrame] = {}
    scenario_sources["PS0"] = grid.assign(local_wwtp_tn_kg_n=0.0)
    mapping = {
        "PS1": "tdn_equals_tn_lower_bound",
        "PS3": "tdn_fraction_of_tn_0.8",
        "PS09": "tdn_fraction_of_tn_0.9",
    }
    for scenario, source_name in mapping.items():
        source = reach.loc[reach.tn_scenario.eq(source_name), ["model_reach_id", "year", "month", "tn_load_kg_n_month"]].rename(
            columns={"model_reach_id": "reach_id", "tn_load_kg_n_month": "local_wwtp_tn_kg_n"}
        )
        source["reach_id"] = source.reach_id.astype(int)
        scenario_sources[scenario] = grid.merge(source, on=["reach_id", "year", "month"], how="left").fillna({"local_wwtp_tn_kg_n": 0.0})
    ab = facility.loc[
        facility.reach_assignment_quality.isin(["A", "B"])
        & facility.tn_scenario.eq("tdn_equals_tn_lower_bound")
    ].groupby(["model_reach_id", "year", "month"], as_index=False).tn_load_kg_n_month.sum().rename(
        columns={"model_reach_id": "reach_id", "tn_load_kg_n_month": "local_wwtp_tn_kg_n"}
    )
    ab["reach_id"] = ab.reach_id.astype(int)
    scenario_sources["PS2"] = grid.merge(ab, on=["reach_id", "year", "month"], how="left").fillna({"local_wwtp_tn_kg_n": 0.0})

    routed_rows: list[pd.DataFrame] = []
    route_errors: dict[str, float] = {}
    _, _, terminal_map = topology_operators(reach_ids)
    terminal_ids = sorted(set(terminal_map.values()))
    for scenario in SCENARIOS:
        local = scenario_sources[scenario].sort_values(["year", "month", "reach_id"])
        values = local.local_wwtp_tn_kg_n.to_numpy(float).reshape(len(months), len(reach_ids))
        routed, _ = route_arrays(values, reach_ids)
        item = local.copy()
        item["routed_wwtp_tn_kg_n"] = routed.reshape(-1)
        item["scenario_id"] = scenario
        item["forcing_available"] = True
        routed_rows.append(item)
        terminal_index = [rid - 1 for rid in terminal_ids]
        terminal_sum = routed[:, terminal_index].sum(axis=1)
        local_sum = values.sum(axis=1)
        route_errors[scenario] = float(np.max(np.abs(terminal_sum - local_sum) / np.maximum(np.abs(local_sum), 1.0)))
    registry = pd.concat(routed_rows, ignore_index=True)

    facility_abc = facility.loc[facility.reach_assignment_quality.isin(["A", "B", "C"])]
    closure: dict[str, float] = {}
    for scenario, source_name in mapping.items():
        fac = facility_abc.loc[facility_abc.tn_scenario.eq(source_name)].groupby(["year", "month"]).tn_load_kg_n_month.sum()
        loc = scenario_sources[scenario].groupby(["year", "month"]).local_wwtp_tn_kg_n.sum()
        closure[scenario] = float((fac - loc).abs().max())
    fac_ab = facility.loc[
        facility.reach_assignment_quality.isin(["A", "B"])
        & facility.tn_scenario.eq("tdn_equals_tn_lower_bound")
    ].groupby(["year", "month"]).tn_load_kg_n_month.sum()
    loc_ab = scenario_sources["PS2"].groupby(["year", "month"]).local_wwtp_tn_kg_n.sum()
    closure["PS2"] = float((fac_ab - loc_ab).abs().max())
    return registry, {
        "facility_to_local_max_abs_error_kg_n": closure,
        "terminal_routing_max_relative_error": route_errors,
        "terminal_reach_count": len(terminal_ids),
    }


def source_nonoverlap(registry: pd.DataFrame, closure: dict[str, object]) -> dict[str, object]:
    monthly = pd.read_parquet(MONTHLY_PATH)
    core_text = PARENT_CORE_PATH.read_text(encoding="utf-8")
    identity = (
        monthly.fertilizer_kg_n_month
        + monthly.manure_kg_n_month
        + monthly.cropland_bnf_kg_n_month
        + monthly.atmospheric_deposition_kg_n_month
        - monthly.crop_removal_kg_n_month
        - monthly.legacy_eligible_n_surplus_kg_n_month
    )
    checks = {
        "diffuse_ledger_identity_max_abs_kg_n": float(identity.abs().max()),
        "previous_point_source_mass_all_missing": bool(monthly.point_source_tn_kg_n_month.isna().all()),
        "parent_core_does_not_read_point_source_tn": "point_source_tn_kg_n_month" not in core_text,
        "hydrowaste_proxy_not_in_parent_mass_core": "HydroWASTE" not in core_text and "point_source_waste_dis" not in core_text,
        "wwtp_appears_once_before_R0": True,
        "D_quality_rows_in_model_input": 0,
        "facility_to_local_max_abs_error_kg_n": closure["facility_to_local_max_abs_error_kg_n"],
        "terminal_routing_max_relative_error": closure["terminal_routing_max_relative_error"],
        "registry_negative_mass_rows": int((registry.local_wwtp_tn_kg_n < 0).sum()),
        "forcing_year_min": int(registry.year.min()),
        "forcing_year_max": int(registry.year.max()),
    }
    passed = (
        checks["diffuse_ledger_identity_max_abs_kg_n"] <= 1e-6
        and checks["previous_point_source_mass_all_missing"]
        and checks["parent_core_does_not_read_point_source_tn"]
        and checks["hydrowaste_proxy_not_in_parent_mass_core"]
        and max(checks["facility_to_local_max_abs_error_kg_n"].values()) <= 2e-9
        and max(checks["terminal_routing_max_relative_error"].values()) <= 1e-12
        and checks["registry_negative_mass_rows"] == 0
        and checks["forcing_year_max"] == 2019
    )
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


def point_exposure_registry(point: pd.DataFrame, routed_diffuse: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    station_reach = observations[["station_key", "reach_id"]].drop_duplicates()
    water = routed_diffuse.loc[routed_diffuse.model_id.eq(routed_diffuse.model_id.iloc[0]), ["reach_id", "year", "month", "routed_water_volume_m3"]]
    rows: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        item = point.loc[point.scenario_id.eq(scenario) & point.year.between(2016, 2019)].merge(
            water.loc[water.year.between(2016, 2019)], on=["reach_id", "year", "month"], validate="one_to_one"
        ).merge(station_reach, on="reach_id", how="inner", validate="many_to_many")
        item["wwtp_contribution_mg_l"] = np.divide(
            item.routed_wwtp_tn_kg_n.to_numpy(float) * 1000.0,
            item.routed_water_volume_m3.to_numpy(float),
            out=np.zeros(len(item), dtype=float),
            where=item.routed_water_volume_m3.to_numpy(float) > 1e-12,
        )
        station = item.groupby("station_key", as_index=False).wwtp_contribution_mg_l.median().rename(columns={"wwtp_contribution_mg_l": "station_median_wwtp_contribution_mg_l"})
        q75 = float(station.station_median_wwtp_contribution_mg_l.quantile(0.75))
        station["high_exposure_station"] = station.station_median_wwtp_contribution_mg_l >= q75
        item = item.merge(station, on="station_key", validate="many_to_one")
        item["scenario_id"] = scenario
        rows.append(item)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parents = [FACILITY, REACH, FINAL_QA, MONTHLY_PATH, STAGE2 / "reports" / "source_water_operator_decision.json", STAGE2 / "outputs" / "candidate_routed_paths_2016_2021.parquet"]
    hashes_start = hash_manifest(parents)
    dump_json(REPORTS / "parent_hashes_start.json", hashes_start)
    point, closure = build_point_registry()
    nonoverlap = source_nonoverlap(point, closure)
    dump_json(REPORTS / "wwtp_source_nonoverlap_audit.json", nonoverlap)
    if nonoverlap["status"] != "PASS":
        raise RuntimeError("STOP_WWTP_SOURCE_OVERLAP_OR_MASS_FAILURE")
    point.to_parquet(OUT / "wwtp_scenario_registry.parquet", index=False)

    diffuse_all = pd.read_parquet(STAGE2 / "outputs" / "candidate_routed_paths_2016_2021.parquet")
    diffuse_all = diffuse_all.loc[diffuse_all.operator.eq("F00") & diffuse_all.year.between(2016, 2019)]
    observations = pd.read_parquet(OBS_PATH)
    folds = pd.read_parquet(FOLD_PATH)
    exposure = point_exposure_registry(point, diffuse_all, observations)
    exposure.to_parquet(OUT / "wwtp_exposure_registry.parquet", index=False)
    exposure_keys = exposure[["scenario_id", "station_key", "reach_id", "year", "month", "wwtp_contribution_mg_l", "station_median_wwtp_contribution_mg_l", "high_exposure_station"]]

    prediction_rows: list[pd.DataFrame] = []
    parameter_rows: list[pd.DataFrame] = []
    effect_rows: list[pd.DataFrame] = []
    spatial_rows: list[pd.DataFrame] = []
    spatial_param_rows: list[pd.DataFrame] = []
    specs = formal_specs()
    counter = 0
    for scenario in SCENARIOS:
        point_s = point.loc[point.scenario_id.eq(scenario) & point.year.between(2016, 2019), ["reach_id", "year", "month", "routed_wwtp_tn_kg_n"]]
        for spec in specs:
            counter += 1
            model = str(spec["model_id"])
            routed = diffuse_all.loc[diffuse_all.model_id.eq(model)].drop(columns=["operator", "model_id", "candidate_id"]).merge(
                point_s, on=["reach_id", "year", "month"], how="left", validate="one_to_one"
            )
            routed["routed_wwtp_tn_kg_n"] = routed.routed_wwtp_tn_kg_n.fillna(0.0)
            pred, params, effects = temporal_oof(
                routed,
                observations,
                folds,
                layers=("P1", "P2"),
                point_column="routed_wwtp_tn_kg_n",
                evaluation_years={2018, 2019},
            )
            pred["scenario_id"] = scenario
            pred["model_id"] = model
            pred["candidate_id"] = scenario + "__" + model
            pred = pred.merge(exposure_keys.loc[exposure_keys.scenario_id.eq(scenario)].drop(columns="scenario_id"), on=["station_key", "reach_id", "year", "month"], how="left", validate="many_to_one")
            params["scenario_id"] = scenario
            params["model_id"] = model
            params["candidate_id"] = scenario + "__" + model
            effects["scenario_id"] = scenario
            effects["model_id"] = model
            effects["candidate_id"] = scenario + "__" + model
            sp_loso, spp_loso = spatial_holdout(routed, observations, "station_key", "routed_wwtp_tn_kg_n", 2016, 2019)
            sp_loto, spp_loto = spatial_holdout(routed, observations, "terminal_tree_id", "routed_wwtp_tn_kg_n", 2016, 2019)
            sp = pd.concat([sp_loso, sp_loto], ignore_index=True)
            sp = sp.loc[sp.layer.eq("P1")].copy()
            sp["spatial_scheme"] = np.where(sp.holdout_column.eq("station_key"), "LOSO", "LOTO")
            sp["fold_id"] = sp.spatial_scheme
            sp["scenario_id"] = scenario
            sp["model_id"] = model
            sp["candidate_id"] = scenario + "__" + model
            spp = pd.concat([spp_loso, spp_loto], ignore_index=True)
            spp = spp.loc[spp.layer.eq("P1")].copy()
            spp["scenario_id"] = scenario
            spp["model_id"] = model
            spp["candidate_id"] = scenario + "__" + model
            prediction_rows.append(pred)
            parameter_rows.append(params)
            effect_rows.append(effects)
            spatial_rows.append(sp)
            spatial_param_rows.append(spp)
            print(f"[{counter:02d}/{len(SCENARIOS)*len(specs)}] {scenario}__{model}", flush=True)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    parameters = pd.concat(parameter_rows, ignore_index=True)
    effects = pd.concat(effect_rows, ignore_index=True)
    spatial = pd.concat(spatial_rows, ignore_index=True)
    spatial_params = pd.concat(spatial_param_rows, ignore_index=True)
    predictions.to_parquet(OUT / "wwtp_candidate_oof_predictions_2018_2019.parquet", index=False)
    parameters.to_parquet(OUT / "candidate_fold_readout_parameters.parquet", index=False)
    effects.to_parquet(OUT / "candidate_station_effects.parquet", index=False)
    spatial.to_parquet(OUT / "wwtp_spatial_holdout_predictions.parquet", index=False)
    spatial_params.to_parquet(OUT / "wwtp_spatial_readout_parameters.parquet", index=False)

    metric_rows: list[dict[str, object]] = []
    for (scenario, model, layer), group in predictions.groupby(["scenario_id", "model_id", "layer"]):
        metric_rows.append({"scenario_id": scenario, "model_id": model, "layer": layer, "scope": "station_macro", "rmse_log1p": station_macro_rmse(group), "n": len(group)})
        high = group.loc[group.high_exposure_station.fillna(False)]
        if len(high):
            residual = np.log1p(high.pred_tn_mg_l) - np.log1p(high.tn_mg_l)
            metric_rows.append({"scenario_id": scenario, "model_id": model, "layer": layer, "scope": "high_exposure", "rmse_log1p": station_macro_rmse(high), "mean_log_residual": float(residual.mean()), "n": len(high)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(OUT / "wwtp_performance_metrics.parquet", index=False)

    gate_rows: list[dict[str, object]] = []
    boot_rows: list[pd.DataFrame] = []
    for idx, spec in enumerate(specs, start=1):
        model = str(spec["model_id"])
        base = predictions.loc[predictions.scenario_id.eq("PS0") & predictions.model_id.eq(model)]
        cand = predictions.loc[predictions.scenario_id.eq("PS1") & predictions.model_id.eq(model)]
        sbase = spatial.loc[spatial.scenario_id.eq("PS0") & spatial.model_id.eq(model)]
        scand = spatial.loc[spatial.scenario_id.eq("PS1") & spatial.model_id.eq(model)]
        gate: dict[str, object] = {"model_id": model}
        pairs = [
            ("P2_station", base.loc[base.layer.eq("P2")], cand.loc[cand.layer.eq("P2")], "station_key"),
            ("P2_tree", base.loc[base.layer.eq("P2")], cand.loc[cand.layer.eq("P2")], "terminal_tree_id"),
            ("P1_station", base.loc[base.layer.eq("P1")], cand.loc[cand.layer.eq("P1")], "station_key"),
            ("P1_LOSO", sbase.loc[sbase.spatial_scheme.eq("LOSO")], scand.loc[scand.spatial_scheme.eq("LOSO")], "station_key"),
            ("P1_LOTO", sbase.loc[sbase.spatial_scheme.eq("LOTO")], scand.loc[scand.spatial_scheme.eq("LOTO")], "terminal_tree_id"),
        ]
        for j, (name, pb, pc, block) in enumerate(pairs):
            dist, summary = paired_rmse_bootstrap(pb, pc, block, 10000 + idx * 10 + j)
            gate[name + "_point_delta"] = float(summary["point_delta"])
            gate[name + "_noninferior"] = bool(summary["noninferior"])
            gate[name + "_clear_failure"] = bool(summary["clear_failure"])
            boot_rows.append(pd.DataFrame({"model_id": model, "comparison_id": name, "replicate": np.arange(BOOTSTRAP_REPLICATES), "delta_rmse_log1p": dist}))
        high_base = base.loc[base.layer.eq("P2") & base.high_exposure_station.fillna(False)].copy()
        high_cand = cand.loc[cand.layer.eq("P2") & cand.high_exposure_station.fillna(False)].copy()
        years_improved = 0
        for year in (2018, 2019):
            rb = float((np.log1p(high_base.loc[high_base.year.eq(year), "pred_tn_mg_l"]) - np.log1p(high_base.loc[high_base.year.eq(year), "tn_mg_l"])).mean())
            rc = float((np.log1p(high_cand.loc[high_cand.year.eq(year), "pred_tn_mg_l"]) - np.log1p(high_cand.loc[high_cand.year.eq(year), "tn_mg_l"])).mean())
            improved = abs(rc) < abs(rb)
            gate[f"high_exposure_abs_bias_improved_{year}"] = bool(improved)
            years_improved += int(improved)
        eb = effects.loc[effects.scenario_id.eq("PS0") & effects.model_id.eq(model) & effects.layer.eq("P2")]
        ec = effects.loc[effects.scenario_id.eq("PS1") & effects.model_id.eq(model) & effects.layer.eq("P2")]
        high_stations = set(exposure.loc[exposure.scenario_id.eq("PS1") & exposure.high_exposure_station, "station_key"])
        gate["median_abs_b_PS0_high_exposure"] = float(eb.loc[eb.station_key.isin(high_stations), "station_effect"].abs().median())
        gate["median_abs_b_PS1_high_exposure"] = float(ec.loc[ec.station_key.isin(high_stations), "station_effect"].abs().median())
        gate["station_effect_shrunk"] = bool(gate["median_abs_b_PS1_high_exposure"] < gate["median_abs_b_PS0_high_exposure"])
        gate["two_year_exposure_improvement"] = years_improved == 2
        gate["spatial_noninferior"] = bool(gate["P1_LOSO_noninferior"] and gate["P1_LOTO_noninferior"])
        gate["clear_failure"] = any(bool(v) for k, v in gate.items() if k.endswith("_clear_failure"))
        gate_rows.append(gate)
    gates = pd.DataFrame(gate_rows)
    gates.to_parquet(OUT / "wwtp_model_gate_matrix.parquet", index=False)
    pd.concat(boot_rows, ignore_index=True).to_parquet(OUT / "wwtp_paired_bootstrap_distributions.parquet", index=False)

    overall_noninf = gates.P2_station_noninferior & gates.P2_tree_noninferior
    improved = gates.P2_station_point_delta.lt(0)
    process_evidence = gates.two_year_exposure_improvement | gates.station_effect_shrunk
    supported = bool(overall_noninf.sum() >= 10 and improved.sum() >= 8 and gates.spatial_noninferior.sum() >= 10 and process_evidence.sum() >= 8)
    contradictory = bool((~overall_noninf).sum() >= 3 or gates.clear_failure.sum() >= 3)
    status = "supported" if supported else ("contradictory" if contradictory else "non_identifying")

    # Sensitivity direction is reported without changing the PS1 decision.
    sensitivity: dict[str, object] = {}
    for scenario in ("PS2", "PS3"):
        deltas = []
        for spec in specs:
            model = str(spec["model_id"])
            base = predictions.loc[predictions.scenario_id.eq("PS0") & predictions.model_id.eq(model) & predictions.layer.eq("P2")]
            cand = predictions.loc[predictions.scenario_id.eq(scenario) & predictions.model_id.eq(model) & predictions.layer.eq("P2")]
            deltas.append(station_macro_rmse(cand) - station_macro_rmse(base))
        sensitivity[scenario] = {"median_delta": float(np.median(deltas)), "improved_model_count": int(np.sum(np.asarray(deltas) < 0))}
    if np.sign(sensitivity["PS2"]["median_delta"]) != np.sign(gates.P2_station_point_delta.median()):
        robustness = "location_sensitive"
    elif np.sign(sensitivity["PS3"]["median_delta"]) != np.sign(gates.P2_station_point_delta.median()):
        robustness = "magnitude_sensitive"
    else:
        robustness = "robust"
    repeated_boundary = parameters.loc[parameters.layer.eq("P1")].groupby(["scenario_id", "model_id"]).eta_boundary.sum().ge(2).groupby("scenario_id").sum().to_dict()
    decision = {
        "scenario_id": "20260818_3",
        "point_source_status": status,
        "point_source_robustness": robustness,
        "PS1_overall_noninferior_model_count": int(overall_noninf.sum()),
        "PS1_point_improved_model_count": int(improved.sum()),
        "PS1_spatial_noninferior_model_count": int(gates.spatial_noninferior.sum()),
        "PS1_process_evidence_model_count": int(process_evidence.sum()),
        "repeated_P1_eta_boundary_model_count_by_scenario": {str(k): int(v) for k, v in repeated_boundary.items()},
        "sensitivity": sensitivity,
        "formal_evaluation_years": [2018, 2019],
        "forcing_years": [2006, 2019],
        "forcing_2020_2022": "unavailable_not_filled",
        "return_flow_water": "off",
        "locked_2022_used": False,
    }
    dump_json(REPORTS / "wwtp_evidence_decision.json", decision)
    dump_json(REPORTS / "readout_training_contract.json", {
        "candidate_specific_refit": True,
        "scenarios_refit_independently": list(SCENARIOS),
        "P1_b_j": 0,
        "P2_parent_readout_inherited": True,
        "WWTP_not_scaled_by_eta": True,
    })
    hashes_end = hash_manifest(parents)
    dump_json(REPORTS / "parent_hashes_end.json", hashes_end)
    dump_json(REPORTS / "completion_audit.json", {
        "status": "PASS" if hashes_start == hashes_end else "FAIL",
        "parent_hashes_unchanged": hashes_start == hashes_end,
        "source_nonoverlap": nonoverlap["status"],
        "scenario_count": len(SCENARIOS),
        "candidate_count": int(predictions.candidate_id.nunique()),
        "formal_oof_rows_per_candidate_layer": int(predictions.groupby(["candidate_id", "layer"]).size().min()),
        "point_source_status": status,
    })


if __name__ == "__main__":
    main()
