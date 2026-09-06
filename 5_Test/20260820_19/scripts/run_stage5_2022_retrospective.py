from __future__ import annotations

import json

import numpy as np
import pandas as pd

import hierarchical19_shared as h


SEGMENTS = h.TEST / "20260820_2" / "outputs" / "andreadis_500m_channel_segments.parquet"
Q72 = h.TEST / "20260820_2" / "outputs" / "q72_routed_monthly_discharge_2006_2022.parquet"


def manning_depth(q: np.ndarray, width: np.ndarray, slope: np.ndarray, roughness: float) -> np.ndarray:
    hvalue = np.maximum((q * roughness / (width * np.sqrt(slope))) ** (3.0 / 5.0), 1e-8)
    for _ in range(80):
        area = width * hvalue
        perimeter = width + 2.0 * hvalue
        radius = area / perimeter
        modeled = area * radius ** (2.0 / 3.0) * np.sqrt(slope) / roughness
        dlog = 1.0 / hvalue + (2.0 / 3.0) * (1.0 / hvalue - 2.0 / perimeter)
        update = (modeled - q) / np.maximum(modeled * dlog, 1e-30)
        hvalue = np.maximum(hvalue - update, 1e-10)
    return hvalue


def build_2022_exposure() -> pd.DataFrame:
    segments = pd.read_parquet(SEGMENTS).sort_values(["reach_id", "segment_index"]).reset_index(drop=True)
    hydro = pd.read_parquet(Q72, filters=[("year", "==", 2022)]).sort_values(["year", "month", "reach_id"])
    if segments.wqd_reference_discharge_used.any() or hydro.andreadis_discharge_used.any():
        raise RuntimeError("STOP_WQD_REFERENCE_DISCHARGE_USED")
    reach_ids = np.sort(hydro.reach_id.unique().astype(int))
    rindex = {int(v): i for i, v in enumerate(reach_ids)}
    seg_ridx = segments.reach_id.map(rindex).to_numpy(int)
    x = segments.segment_midpoint_fraction.to_numpy(float)
    length = segments.segment_length_m.to_numpy(float)
    mid_length = segments.midpoint_to_outlet_segment_length_m.to_numpy(float)
    width = segments.width_central_m.to_numpy(float)
    slope = segments.slope_used.to_numpy(float)
    times = hydro[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    local_q = hydro.q72_local_discharge_m3_s.to_numpy(float).reshape(len(times), len(reach_ids))
    upstream_q = hydro.q72_upstream_discharge_m3_s.to_numpy(float).reshape(len(times), len(reach_ids))
    rows = []
    for ti, (year, month) in enumerate(times.itertuples(index=False)):
        q = upstream_q[ti, seg_ridx] + x * local_q[ti, seg_ridx]
        depth = manning_depth(q, width, slope, 0.035)
        velocity = q / (width * depth)
        tau = length / velocity
        tau_mid = mid_length / velocity
        rows.append(pd.DataFrame({
            "reach_id": reach_ids, "year": int(year), "month": int(month),
            "uptake_exposure_full_days_per_m": np.bincount(seg_ridx, weights=tau / 86400.0 / depth, minlength=len(reach_ids)),
            "uptake_exposure_midpoint_to_outlet_days_per_m": np.bincount(seg_ridx, weights=tau_mid / 86400.0 / depth, minlength=len(reach_ids)),
            "water_source": "Q72_structural_canonical_main_only",
            "geometry_source": "Andreadis_width_depth_only", "wqd_reference_discharge_used": False,
        }))
    result = pd.concat(rows, ignore_index=True).sort_values(["year", "month", "reach_id"])
    if len(result) != 2760 or result.reach_id.nunique() != 230:
        raise RuntimeError("STOP_2022_EXPOSURE_COVERAGE")
    return result


def build_2022_router(model_id: str, exposure: pd.DataFrame, shared) -> h.HydraulicRouter:
    local = pd.read_parquet(h.LOCAL / f"{model_id}.parquet", filters=[("year", "==", 2022)]).sort_values(["year", "month", "reach_id"])
    parent = pd.read_parquet(h.PARENT_ROUTED / f"{model_id}.parquet", filters=[("year", "==", 2022)]).sort_values(["year", "month", "reach_id"])
    cov = pd.read_parquet(h.COVARIATES).sort_values("reach_id")
    reach_ids = np.sort(local.reach_id.unique().astype(int))
    times = local[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    order, downstream, terminal = shared.topology_operators(reach_ids)
    lookup = {int(v): i for i, v in enumerate(reach_ids)}
    shape = (12, 230)
    return h.HydraulicRouter(
        model_id=model_id, reach_ids=reach_ids, years=times.year.to_numpy(int), months=times.month.to_numpy(int),
        local_q=local.quick_tn_release_kg_n.to_numpy(float).reshape(shape),
        local_g=local.gw_tn_release_kg_n.to_numpy(float).reshape(shape),
        h_full=exposure.uptake_exposure_full_days_per_m.to_numpy(float).reshape(shape),
        h_mid=exposure.uptake_exposure_midpoint_to_outlet_days_per_m.to_numpy(float).reshape(shape),
        water=parent.routed_water_volume_m3.to_numpy(float).reshape(shape),
        x=cov[list(h.COVARIATE_COLUMNS)].to_numpy(float),
        terminal_by_reach=np.array([terminal[int(v)] for v in reach_ids], dtype=int),
        order_index=[lookup[int(v)] for v in order],
        downstream_index={lookup[int(k)]: (lookup[int(v[0])], float(v[1])) for k, v in downstream.items()},
    )


def classify(name: str) -> tuple[str, bool]:
    value = str(name)
    if any(token in value for token in ("湖心", "库心")):
        return "open_lake_or_reservoir_water", False
    if any(token in value for token in ("坝下", "出口", "出水口")):
        return "dam_or_reservoir_outlet", True
    if any(token in value for token in ("水库", "湖")):
        return "uncertain_domain", False
    return "river_channel", True


def main() -> None:
    h.require_runtime()
    mechanism_lock = json.loads((h.REPORTS / "development_mechanism_lock.json").read_text(encoding="utf-8"))
    parameter_lock = json.loads((h.REPORTS / "full_development_parameter_lock.json").read_text(encoding="utf-8"))
    posterior = json.loads((h.REPORTS / "posterior_sampling_audit.json").read_text(encoding="utf-8"))
    if not (mechanism_lock["written_before_2022_TN"] and parameter_lock["written_before_2022_TN"]):
        raise RuntimeError("STOP_LOCKS_NOT_WRITTEN")
    if parameter_lock["TN_2022_read"] or posterior["TN_2022_read"]:
        raise RuntimeError("STOP_2022_ALREADY_READ")

    exposure = build_2022_exposure()
    exposure.to_parquet(h.OUT / "reach_month_aquatic_uptake_exposure_2022.parquet", index=False)
    # First and only TN read after both development locks.
    observations = pd.read_parquet(h.OBS, filters=[("year", "==", 2022)]).copy()
    if observations.year.nunique() != 1 or int(observations.year.iloc[0]) != 2022:
        raise RuntimeError("STOP_2022_TN_BOUNDARY")
    domain = observations.station_key.map(lambda _: None)
    classified = observations.station.map(classify)
    observations["observation_domain"] = [v[0] for v in classified]
    observations["primary_river_domain"] = [v[1] for v in classified]
    parameters = pd.read_parquet(h.OUT / "full_development_parameters.parquet").set_index("model_id")
    effects = pd.read_parquet(h.OUT / "full_development_station_effects.parquet")
    shared = h.parent_shared()
    parts = []
    for model_id in h.FORMAL_MODELS:
        router = build_2022_router(model_id, exposure, shared)
        row = parameters.loc[model_id]
        for mechanism, vf, p1_eta, p2_eta in (
            ("H0_PARENT", 0.0, np.array([row.H0_P1_eta_quick, row.H0_P1_eta_gw]), np.array([row.H0_P2_eta_quick, row.H0_P2_eta_gw])),
            ("H1_GLOBAL", float(row.v_f_m_per_day), np.array([row.P1_eta_quick, row.P1_eta_gw]), np.array([row.P2_eta_quick, row.P2_eta_gw])),
        ):
            frame = router.frame(observations, np.full(len(router.reach_ids), vf))
            for layer, eta in (("P1", p1_eta), ("P2", p2_eta)):
                effect_map = (
                    effects.loc[effects.model_id.eq(model_id) & effects.mechanism.eq(mechanism)]
                    .set_index("station_key").station_effect.to_dict() if layer == "P2" else {}
                )
                pred = shared.predict_layer(frame, layer, eta, effect_map)
                pred["model_id"] = model_id
                pred["fold_id"] = "R2022"
                pred["mechanism"] = mechanism
                pred["retrospective_role"] = np.where(pred.primary_river_domain, "primary_river", "diagnostic_only")
                parts.append(pred)
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_parquet(h.OUT / "locked_2022_retrospective_predictions.parquet", index=False)

    primary = predictions.loc[predictions.primary_river_domain]
    gate_rows = []
    seed = 2026085000
    for model_id in h.FORMAL_MODELS:
        for layer in ("P1", "P2"):
            subset = primary.loc[primary.model_id.eq(model_id) & primary.layer.eq(layer)]
            parent = subset.loc[subset.mechanism.eq("H0_PARENT")]
            candidate = subset.loc[subset.mechanism.eq("H1_GLOBAL")]
            for block in ("station_key", "terminal_tree_id"):
                seed += 1
                gate_rows.append({
                    "model_id": model_id, "layer": layer,
                    "block": "station" if block == "station_key" else "tree",
                    **h.paired_bootstrap(parent, candidate, block, seed),
                })
    gates = pd.DataFrame(gate_rows)
    gates.to_parquet(h.OUT / "locked_2022_retrospective_gates.parquet", index=False)
    tree163 = predictions.loc[predictions.terminal_tree_id.eq(163)].copy()
    tree163.to_parquet(h.OUT / "tree_163_locked_retrospective_diagnostic.parquet", index=False)
    report = {
        "status": "locked_2022_retrospective_complete",
        "TN_rows_read": int(len(observations)), "primary_river_rows": int(observations.primary_river_domain.sum()),
        "diagnostic_only_rows": int((~observations.primary_river_domain).sum()),
        "primary_stations": int(observations.loc[observations.primary_river_domain, "station_key"].nunique()),
        "tree_163_rows": int(len(tree163)),
        "P1_station_noninferior_models": int(gates.loc[gates.layer.eq("P1") & gates.block.eq("station"), "noninferior"].sum()),
        "P1_tree_noninferior_models": int(gates.loc[gates.layer.eq("P1") & gates.block.eq("tree"), "noninferior"].sum()),
        "P2_station_noninferior_models": int(gates.loc[gates.layer.eq("P2") & gates.block.eq("station"), "noninferior"].sum()),
        "P2_tree_noninferior_models": int(gates.loc[gates.layer.eq("P2") & gates.block.eq("tree"), "noninferior"].sum()),
        "tree_163_interpretation": "open-lake-center diagnostic only; not evidence against river-channel H1",
        "temperature_used": False, "TN_2022_read": True,
    }
    h.dump_json(h.REPORTS / "locked_2022_retrospective_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
