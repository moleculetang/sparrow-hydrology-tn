from __future__ import annotations

import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr


S16_1 = Path(r"E:\SPARROW\5_Test\20260816_1")
sys.path.insert(0, str(S16_1 / "scripts"))
from legacy16_core import (  # noqa: E402
    FOLD_PATH,
    MUS,
    OBS_PATH,
    S14_9,
    S16_1 as STAGE1,
    TOPOLOGY_PATH,
    all_candidate_specs,
    dump_json,
    hash_manifest,
    prepare_model_arrays,
    require_runtime,
    simulate_candidate_totals,
)


ROOT = Path(r"E:\SPARROW\5_Test\20260816_4")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
WORK = ROOT / "work"
S16_3 = Path(r"E:\SPARROW\5_Test\20260816_3")
GW_DECADAL = S14_9 / "inputs" / "model_ready" / "static" / "groundwater_nitrate_decadal_by_reach.parquet"
GW_POINTS = S14_9 / "inputs" / "model_ready" / "observations" / "groundwater_nitrate_observations_prb_1979_2022.parquet"
AQUIFER_ID_RASTER = Path(r"E:\SPARROW\0_reach_topology\data\raw\hydrology\groundwater_nitrate_global_1979_2022\data\global_nitrate_dataset\decadal_aquiferNO3_avg_datasets\global_aquifer_ID92.tif")
NITRATE_MASK = S14_9 / "work" / "new_soil_groundwater" / "catchment_mask_nitrate_5arcmin.dat"
GDAL_BIN = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin")
GDALINFO = GDAL_BIN / "gdalinfo.exe"
GDALTRANSLATE = GDAL_BIN / "gdal_translate.exe"
ETA_LAMBDA = 1.0
STATION_LAMBDA = 12.0
ETA_BOUNDARY_TOL = 0.01


def topology_operators(reach_ids: np.ndarray) -> tuple[list[int], dict[int, tuple[int, float]], dict[int, int]]:
    topo = pd.read_csv(TOPOLOGY_PATH)
    topo["reach_id"] = topo.reach_id.astype(int)
    downstream: dict[int, tuple[int, float]] = {}
    for row in topo.itertuples():
        if pd.notna(row.downstream_reach):
            downstream[int(row.reach_id)] = (int(row.downstream_reach), float(row.frac))
    nodes = list(map(int, reach_ids))
    indegree = {rid: 0 for rid in nodes}
    for rid, (down, _) in downstream.items():
        if rid not in indegree or down not in indegree:
            raise RuntimeError("topology references a reach outside the frozen domain")
        indegree[down] += 1
    queue = deque(sorted(rid for rid, degree in indegree.items() if degree == 0))
    order: list[int] = []
    while queue:
        rid = queue.popleft()
        order.append(rid)
        if rid in downstream:
            down = downstream[rid][0]
            indegree[down] -= 1
            if indegree[down] == 0:
                queue.append(down)
    if len(order) != len(nodes):
        raise RuntimeError("frozen topology is cyclic or incomplete")
    terminal: dict[int, int] = {}
    for rid in nodes:
        current = rid
        seen: set[int] = set()
        while current in downstream:
            if current in seen:
                raise RuntimeError("topology cycle")
            seen.add(current)
            current = downstream[current][0]
        terminal[rid] = current
    return order, downstream, terminal


def route_recent(
    frame: pd.DataFrame,
    reach_ids: np.ndarray,
    order: list[int],
    downstream: dict[int, tuple[int, float]],
    terminal: dict[int, int],
) -> pd.DataFrame:
    recent = frame.loc[frame.year.between(2016, 2021)].sort_values(["year", "month", "reach_id"])
    index = {int(rid): i for i, rid in enumerate(reach_ids)}
    rows: list[pd.DataFrame] = []
    for (year, month), block in recent.groupby(["year", "month"], sort=True):
        b = block.set_index("reach_id").loc[reach_ids]
        values = np.column_stack([
            b.quick_tn_release_kg_n.to_numpy(float),
            b.gw_tn_release_kg_n.to_numpy(float),
            (b.q_local_total_mm * b.catchment_area_km2 * 1000.0).to_numpy(float),
        ])
        for rid in order:
            if rid in downstream:
                down, fraction = downstream[rid]
                values[index[down]] += values[index[rid]] * fraction
        item = pd.DataFrame({
            "reach_id": reach_ids,
            "year": int(year),
            "month": int(month),
            "routed_quick_tn_kg_n": values[:, 0],
            "routed_gw_tn_kg_n": values[:, 1],
            "routed_water_volume_m3": values[:, 2],
        })
        item["terminal_tree_id"] = item.reach_id.map(terminal).astype(int)
        rows.append(item)
    routed = pd.concat(rows, ignore_index=True)
    routed["model_id"] = str(frame.model_id.iloc[0])
    return routed


def station_effects(log_residual: np.ndarray, station_index: np.ndarray, n_stations: int) -> np.ndarray:
    sums = np.bincount(station_index, weights=log_residual, minlength=n_stations)
    counts = np.bincount(station_index, minlength=n_stations).astype(float)
    return -sums / (counts + STATION_LAMBDA)


def fit_eta(train: pd.DataFrame) -> tuple[np.ndarray, dict[str, float], dict[str, object]]:
    levels = sorted(train.station_key.astype(str).unique())
    lookup = {station: i for i, station in enumerate(levels)}
    station_index = train.station_key.astype(str).map(lookup).to_numpy(int)
    q = train.routed_quick_tn_kg_n.to_numpy(float)
    g = train.routed_gw_tn_kg_n.to_numpy(float)
    water = train.routed_water_volume_m3.to_numpy(float)
    observed_log = np.log1p(train.tn_mg_l.to_numpy(float))

    def components(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        concentration = np.divide(
            (eta[0] * q + eta[1] * g) * 1000.0,
            water,
            out=np.zeros_like(water),
            where=water > 0,
        )
        raw_residual = np.log1p(np.maximum(concentration, 0.0)) - observed_log
        effects = station_effects(raw_residual, station_index, len(levels))
        return raw_residual + effects[station_index], effects

    def residual(eta: np.ndarray) -> np.ndarray:
        data_residual, effects = components(eta)
        return np.concatenate([
            data_residual,
            np.sqrt(ETA_LAMBDA) * (eta - 1.0),
            np.sqrt(STATION_LAMBDA) * effects,
        ])

    result = least_squares(
        residual,
        x0=np.array([0.8, 0.8]),
        bounds=(np.zeros(2), np.ones(2)),
        method="trf",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )
    _, effects = components(result.x)
    effect_map = {station: float(effects[i]) for i, station in enumerate(levels)}
    diagnostic = {
        "eta_quick": float(result.x[0]),
        "eta_gw": float(result.x[1]),
        "success": bool(result.success),
        "status": int(result.status),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "station_effect_count": len(levels),
        "delivery_efficiency_confounded": bool(np.any(result.x <= ETA_BOUNDARY_TOL) or np.any(result.x >= 1.0 - ETA_BOUNDARY_TOL)),
    }
    return result.x, effect_map, diagnostic


def oof_predictions(routed: pd.DataFrame, observations: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    joined = observations.merge(
        routed.drop(columns="model_id"),
        on=["reach_id", "year", "month"],
        validate="many_to_one",
    )
    fold_defs = folds[["fold_id", "train_start_year", "train_end_year", "evaluation_year"]].drop_duplicates().sort_values("evaluation_year")
    outputs: list[pd.DataFrame] = []
    parameters: list[dict[str, object]] = []
    for fold in fold_defs.itertuples():
        train = joined.loc[joined.year.between(fold.train_start_year, fold.train_end_year)].copy()
        test = joined.loc[joined.year.eq(fold.evaluation_year)].copy()
        eta, effects, diagnostic = fit_eta(train)
        concentration = np.divide(
            (eta[0] * test.routed_quick_tn_kg_n.to_numpy(float) + eta[1] * test.routed_gw_tn_kg_n.to_numpy(float)) * 1000.0,
            test.routed_water_volume_m3.to_numpy(float),
            out=np.zeros(len(test), dtype=float),
            where=test.routed_water_volume_m3.to_numpy(float) > 0,
        )
        effect = test.station_key.astype(str).map(effects).fillna(0.0).to_numpy(float)
        test["pred_tn_mg_l"] = np.maximum(np.expm1(np.log1p(np.maximum(concentration, 0.0)) + effect), 0.0)
        test["raw_eta_scaled_tn_mg_l"] = concentration
        test["eta_quick"] = float(eta[0])
        test["eta_gw"] = float(eta[1])
        test["fold_id"] = fold.fold_id
        outputs.append(test)
        parameters.append({"fold_id": fold.fold_id, **diagnostic})
    result = pd.concat(outputs, ignore_index=True)
    result["model_id"] = str(routed.model_id.iloc[0])
    for row in parameters:
        row["model_id"] = str(routed.model_id.iloc[0])
    return result, parameters


def metrics(prediction: pd.DataFrame) -> dict[str, float]:
    observed = prediction.tn_mg_l.to_numpy(float)
    predicted = prediction.pred_tn_mg_l.to_numpy(float)
    lo = np.log1p(observed)
    lp = np.log1p(predicted)
    annual = prediction.assign(log_observed=lo, log_predicted=lp).groupby(["station_key", "year"], as_index=False)[["log_observed", "log_predicted"]].median()
    return {
        "n_oof": len(prediction),
        "stations": int(prediction.station_key.nunique()),
        "terminal_trees": int(prediction.terminal_tree_id.nunique()),
        "rmse_log1p": float(np.sqrt(np.mean((lp - lo) ** 2))),
        "rmse_raw_mg_l": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "pbias_percent": float(100.0 * np.sum(predicted - observed) / np.sum(observed)),
        "correlation": float(np.corrcoef(observed, predicted)[0, 1]) if np.std(observed) > 0 and np.std(predicted) > 0 else np.nan,
        "annual_station_median_log_rmse": float(np.sqrt(np.mean((annual.log_predicted - annual.log_observed) ** 2))),
    }


def aquifer_mapping() -> pd.DataFrame:
    info = json.loads(subprocess.check_output([str(GDALINFO), "-json", str(AQUIFER_ID_RASTER)], text=True, encoding="utf-8"))
    width, height = map(int, info["size"])
    target = WORK / "global_aquifer_ID92.dat"
    if not target.exists():
        subprocess.run([str(GDALTRANSLATE), "-of", "ENVI", str(AQUIFER_ID_RASTER), str(target)], check=True)
    dtype_name = str(info["bands"][0]["type"])
    dtype = {"Byte": np.uint8, "Int16": "<i2", "UInt16": "<u2", "Int32": "<i4", "UInt32": "<u4"}.get(dtype_name)
    if dtype is None:
        raise RuntimeError(f"unsupported aquifer ID raster type: {dtype_name}")
    aquifer = np.memmap(target, dtype=dtype, mode="r", shape=(height, width))
    mask = np.memmap(NITRATE_MASK, dtype=np.int32, mode="r", shape=(height, width))
    valid = (mask > 0) & (aquifer > 0)
    pairs = pd.DataFrame({"reach_id": np.asarray(mask[valid], dtype=int), "aquifer_id": np.asarray(aquifer[valid], dtype=int)})
    counts = pairs.value_counts(["reach_id", "aquifer_id"]).rename("n_cells").reset_index()
    totals = counts.groupby("reach_id").n_cells.transform("sum")
    counts["aquifer_cell_fraction"] = counts.n_cells / totals
    modal = counts.sort_values(["reach_id", "n_cells", "aquifer_id"], ascending=[True, False, True]).drop_duplicates("reach_id")
    all_reaches = pd.DataFrame({"reach_id": np.arange(1, 231)})
    result = all_reaches.merge(modal, on="reach_id", how="left", validate="one_to_one")
    result["aquifer_mapping_status"] = np.where(result.aquifer_id.notna(), "modal_aquifer_assigned", "no_valid_aquifer_id")
    return result


def groundwater_context(frame: pd.DataFrame, mapping: pd.DataFrame, observed_grid: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[pd.DataFrame] = []
    decade_specs = (("1980s", 1980, 1989), ("1990s", 1990, 1999), ("2000s", 2000, 2009), ("2010s", 2010, 2019))
    for label, start, end in decade_specs:
        block = frame.loc[frame.year.between(start, end)].groupby("reach_id", as_index=False).agg(
            model_gw_tn_kg_n=("gw_tn_release_kg_n", "sum"),
            model_gw_water_m3=("gw_discharge_mm", lambda x: 0.0),
        )
        # The discharge volume needs the reach-specific area before aggregation.
        source = frame.loc[frame.year.between(start, end)].copy()
        source["gw_water_m3"] = source.gw_discharge_mm * source.catchment_area_km2 * 1000.0
        block = source.groupby("reach_id", as_index=False).agg(
            model_gw_tn_kg_n=("gw_tn_release_kg_n", "sum"),
            model_gw_water_m3=("gw_water_m3", "sum"),
        )
        obs_mean = f"groundwater_no3_{label}_mg_l_mean"
        obs_cells = f"groundwater_no3_{label}_n_cells"
        block = block.merge(observed_grid[["reach_id", obs_mean, obs_cells]], on="reach_id", validate="one_to_one").merge(mapping, on="reach_id", how="left", validate="one_to_one")
        block = block.loc[block.aquifer_id.notna()].copy()
        grouped_rows = []
        for aquifer_id, group in block.groupby("aquifer_id"):
            valid_obs = group[obs_mean].notna() & group[obs_cells].gt(0)
            observed = float(np.average(group.loc[valid_obs, obs_mean], weights=group.loc[valid_obs, obs_cells])) if valid_obs.any() else np.nan
            model_mass = float(group.model_gw_tn_kg_n.sum())
            model_water = float(group.model_gw_water_m3.sum())
            grouped_rows.append({
                "model_id": str(frame.model_id.iloc[0]),
                "aquifer_id": int(aquifer_id),
                "decade": label,
                "observed_no3_mg_l": observed,
                "observed_n_cells": int(group.loc[valid_obs, obs_cells].sum()) if valid_obs.any() else 0,
                "model_gw_pattern_mg_l_as_tn": model_mass * 1000.0 / model_water if model_water > 0 else np.nan,
                "model_gw_tn_kg_n": model_mass,
                "model_gw_water_m3": model_water,
                "n_reaches": int(group.reach_id.nunique()),
                "eta_gw_included": False,
            })
        rows.append(pd.DataFrame(grouped_rows))
    panel = pd.concat(rows, ignore_index=True)
    valid = panel.dropna(subset=["observed_no3_mg_l", "model_gw_pattern_mg_l_as_tn"])
    spatial = []
    for decade, group in valid.groupby("decade"):
        rho = float(spearmanr(group.observed_no3_mg_l, group.model_gw_pattern_mg_l_as_tn).statistic) if len(group) >= 3 else np.nan
        spatial.append({"decade": decade, "n_aquifers": len(group), "spearman": rho})
    matched = valid.sort_values(["aquifer_id", "decade"])
    direction_pairs = 0
    direction_agreements = 0
    for _, group in matched.groupby("aquifer_id"):
        group = group.assign(decade_start=group.decade.str[:4].astype(int)).sort_values("decade_start")
        if len(group) < 2:
            continue
        obs_delta = np.diff(group.observed_no3_mg_l.to_numpy(float))
        model_delta = np.diff(group.model_gw_pattern_mg_l_as_tn.to_numpy(float))
        direction_pairs += len(obs_delta)
        direction_agreements += int(np.sum(np.sign(obs_delta) == np.sign(model_delta)))
    summary = {
        "model_id": str(frame.model_id.iloc[0]),
        "formal_pattern_excludes_eta_gw": True,
        "independent_unit": "aquifer_id_x_decade",
        "spatial_by_decade": spatial,
        "matched_same_aquifer_direction_pairs": direction_pairs,
        "matched_same_aquifer_direction_agreement_fraction": float(direction_agreements / direction_pairs) if direction_pairs else np.nan,
        "context_status": "context_available" if len(valid) else "non_identifying_no_valid_panel",
    }
    return panel, summary


def point_context(frame: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    monthly = frame.copy()
    monthly["gw_water_m3"] = monthly.gw_discharge_mm * monthly.catchment_area_km2 * 1000.0
    annual = monthly.groupby(["reach_id", "year"], as_index=False).agg(
        model_gw_tn_kg_n=("gw_tn_release_kg_n", "sum"),
        model_gw_water_m3=("gw_water_m3", "sum"),
    )
    annual["model_gw_pattern_mg_l_as_tn"] = np.divide(
        annual.model_gw_tn_kg_n * 1000.0,
        annual.model_gw_water_m3,
        out=np.full(len(annual), np.nan),
        where=annual.model_gw_water_m3.to_numpy(float) > 0,
    )
    observed = points.copy()
    observed["year"] = observed.year.astype(int)
    joined = observed.merge(annual, on=["reach_id", "year"], how="left", validate="many_to_one")
    joined["model_id"] = str(frame.model_id.iloc[0])
    joined["eta_gw_included"] = False
    ratio = np.divide(joined.model_gw_pattern_mg_l_as_tn, joined.no3_mg_l, out=np.full(len(joined), np.nan), where=joined.no3_mg_l.to_numpy(float) > 0)
    joined["gross_order_of_magnitude_contradiction"] = (ratio > 100.0) | (ratio < 0.01)
    joined["diagnostic_role"] = "contextual_not_hard_gate_species_and_support_mismatch"
    return joined


def main() -> None:
    require_runtime()
    for directory in (OUT, REPORTS, WORK):
        directory.mkdir(parents=True, exist_ok=True)
    stage3_audit = S16_3 / "reports" / "completion_audit.json"
    if not json.loads(stage3_audit.read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("20260816_3 did not pass")
    protected = [
        stage3_audit,
        STAGE1 / "reports" / "incumbent_reproduction_audit.json",
        STAGE1 / "outputs" / "candidate_registry_63.csv",
        OBS_PATH,
        FOLD_PATH,
        GW_DECADAL,
        GW_POINTS,
        AQUIFER_ID_RASTER,
        NITRATE_MASK,
        TOPOLOGY_PATH,
    ]
    start_hashes = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_start.json", start_hashes)
    reach_ids, times, arrays, early_positive = prepare_model_arrays()
    observations = pd.read_parquet(OBS_PATH).loc[lambda x: x.year.between(2016, 2021)].copy()
    folds = pd.read_parquet(FOLD_PATH)
    order, downstream, terminal = topology_operators(reach_ids)
    mapping = aquifer_mapping()
    mapping.to_parquet(OUT / "reach_modal_aquifer_mapping.parquet", index=False)
    observed_grid = pd.read_parquet(GW_DECADAL)
    points = pd.read_parquet(GW_POINTS)

    all_oof: list[pd.DataFrame] = []
    all_routed: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    engineering: dict[str, object] = {}
    state_rows: list[pd.DataFrame] = []
    panel_rows: list[pd.DataFrame] = []
    gw_summaries: list[dict[str, object]] = []
    point_rows: list[pd.DataFrame] = []

    for candidate_index, spec in enumerate(all_candidate_specs(), start=1):
        model_id = str(spec["model_id"])
        frame, audit = simulate_candidate_totals(
            model_id,
            str(spec["source_structure"]),
            None if pd.isna(spec["soil_tau_month"]) else int(spec["soil_tau_month"]),
            int(spec["delivery_mu_month"]),
            reach_ids,
            times,
            arrays,
            early_positive,
        )
        engineering[model_id] = audit
        state = frame.loc[frame.year.between(2010, 2018)].copy()
        state["source_legacy_state_kg_n"] = state.son_state_end_kg_n + state.mobile_state_end_kg_n
        state_summary = state.groupby("reach_id", as_index=False).agg(
            source_legacy_state_mean_kg_n=("source_legacy_state_kg_n", "mean"),
            source_legacy_state_max_kg_n=("source_legacy_state_kg_n", "max"),
            gw_transit_state_mean_kg_n=("gw_state_end_kg_n", "mean"),
            gw_tn_release_mean_kg_n_month=("gw_tn_release_kg_n", "mean"),
        )
        state_summary["model_id"] = model_id
        state_rows.append(state_summary)
        routed = route_recent(frame, reach_ids, order, downstream, terminal)
        prediction, parameters = oof_predictions(routed, observations, folds)
        all_routed.append(routed)
        all_oof.append(prediction)
        parameter_rows.extend(parameters)
        confounded = any(bool(item["delivery_efficiency_confounded"]) for item in parameters)
        metric_rows.append({
            "model_id": model_id,
            "source_structure": spec["source_structure"],
            "soil_tau_month": spec["soil_tau_month"],
            "delivery_mu_month": spec["delivery_mu_month"],
            **metrics(prediction),
            "delivery_efficiency_confounded": confounded,
        })
        panel, gw_summary = groundwater_context(frame, mapping, observed_grid)
        panel_rows.append(panel)
        gw_summaries.append(gw_summary)
        point_rows.append(point_context(frame, points))
        print(f"[{candidate_index:02d}/63] {model_id}", flush=True)

    routed_all = pd.concat(all_routed, ignore_index=True)
    routed_all.to_parquet(OUT / "candidate_routed_paths_2016_2021.parquet", index=False)
    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_parquet(OUT / "candidate_oof_predictions_2018_2021.parquet", index=False)
    pd.DataFrame(parameter_rows).to_parquet(OUT / "candidate_fold_eta_parameters.parquet", index=False)
    pd.DataFrame(metric_rows).to_csv(REPORTS / "candidate_oof_metrics.csv", index=False)
    pd.concat(state_rows, ignore_index=True).to_parquet(OUT / "candidate_state_summary_2010_2018.parquet", index=False)
    pd.concat(panel_rows, ignore_index=True).to_parquet(OUT / "groundwater_aquifer_decade_panel.parquet", index=False)
    pd.DataFrame(gw_summaries).to_json(REPORTS / "groundwater_context_summary.json", orient="records", indent=2)
    pd.concat(point_rows, ignore_index=True).to_parquet(OUT / "groundwater_point_context_diagnostics.parquet", index=False)
    dump_json(REPORTS / "candidate_engineering_and_mass_audit.json", engineering)

    aquifer_audit = {
        "reach_count": len(mapping),
        "reaches_with_modal_aquifer": int(mapping.aquifer_id.notna().sum()),
        "aquifer_count": int(mapping.aquifer_id.nunique(dropna=True)),
        "minimum_modal_cell_fraction": float(mapping.aquifer_cell_fraction.min()),
        "trend_requires_same_aquifer": True,
    }
    dump_json(REPORTS / "aquifer_mapping_audit.json", aquifer_audit)
    end_hashes = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_end.json", end_hashes)
    if start_hashes != end_hashes:
        raise RuntimeError("protected parent changed during stage 4")
    completion = {
        "scenario_id": "20260816_4",
        "pass": True,
        "candidate_count": int(oof_all.model_id.nunique()),
        "oof_rows_per_candidate": int(oof_all.groupby("model_id").size().min()),
        "eta_parameter_rows": len(parameter_rows),
        "conditional_grid_candidate_count": 0,
        "locked_2022_read": False,
        "formal_groundwater_operator_excludes_eta_gw": True,
    }
    dump_json(REPORTS / "completion_audit.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
