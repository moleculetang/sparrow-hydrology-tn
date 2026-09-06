from __future__ import annotations

import calendar
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(r"E:\SPARROW\5_Test\20260820_1\scripts")))
from legacy20_shared import (  # noqa: E402
    FOLD_PATH, MASS_REL_TOL, MASS_ZERO_ABS_TOL_KG_N, OBS_PATH, REFERENCE_DAYS,
    S20_1, S20_2, SURVIVAL_BOUNDARY_HIGH, SURVIVAL_BOUNDARY_LOW,
    SURVIVAL_INITIAL, SURVIVAL_LOWER, SURVIVAL_RIDGE_LAMBDA, SURVIVAL_UPPER,
    WATER_EPS, mass_balance_pass, prepare_model_arrays, topology_operators,
)


Q_RHO = 0.25
STATION_RIDGE = 12.0
SPINUP_TOL_KG_N = 1e-9
SPINUP_MAX_CYCLES = 5000
HAZARD_EPS = 1e-15
HYDRAULIC_PATH = S20_2 / "outputs" / "reach_month_hydraulic_exposure.parquet"
HYDRO_PATH = S20_2 / "outputs" / "q72_routed_monthly_discharge_2006_2022.parquet"
PATH_EXPOSURE_PATH = S20_2 / "outputs" / "source_target_path_exposure_2006_2022.parquet"
GW_FORCING_PATH = S20_1 / "outputs" / "groundwater_temperature_forcing_1961_2022.parquet"
AQ_FORCING_PATH = S20_1 / "outputs" / "aquatic_temperature_forcing_1961_2022.parquet"


def _withdraw(pool: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    removed = np.minimum(pool, demand)
    pool -= removed
    return removed, demand - removed


def _water_partitions(arrays: dict[str, np.ndarray], t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive_water = arrays["positive_input_mm"][t]
    quick_generated = arrays["quick_generated_mm"][t]
    bypass = np.clip(np.divide(quick_generated, positive_water, out=np.zeros_like(positive_water), where=positive_water > WATER_EPS), 0.0, 1.0)
    overflow = arrays["soil_overflow_to_quick_mm"][t]
    recharge = arrays["gw_recharge_mm"][t]
    contact = overflow + recharge
    capacity = arrays["source_water_capacity_mm"][t]
    flush = np.where(contact > WATER_EPS, 1.0 - np.exp(-contact / capacity), 0.0)
    quick_share = np.divide(overflow, contact, out=np.zeros_like(contact), where=contact > WATER_EPS)
    gw_share = np.divide(recharge, contact, out=np.zeros_like(contact), where=contact > WATER_EPS)
    return bypass, flush, quick_share, gw_share


@dataclass
class ReactionContext:
    reach_ids: np.ndarray
    times: list[tuple[int, int]]
    arrays: dict[str, np.ndarray]
    early_positive: np.ndarray
    observations: pd.DataFrame
    folds: pd.DataFrame
    exposure: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    routed_water: dict[tuple[int, int], np.ndarray]
    gw_temperature: dict[str, np.ndarray]
    aq_temperature: dict[str, np.ndarray]
    order: list[int]
    downstream: dict[int, tuple[int, float]]
    terminal: dict[int, int]


def _forcing_array(frame: pd.DataFrame, mode_column: str, mode: str, value_column: str, times: list[tuple[int, int]], reach_ids: np.ndarray) -> np.ndarray:
    work = frame.loc[frame[mode_column].eq(mode), ["reach_id", "year", "month", value_column]].copy()
    ordered = work.set_index(["year", "month", "reach_id"])
    keys = [(int(y), int(m), int(r)) for y, m in times for r in reach_ids]
    return ordered.loc[keys, value_column].to_numpy(float).reshape(len(times), len(reach_ids))


def load_context() -> ReactionContext:
    reach_ids, times, arrays, early = prepare_model_arrays()
    reach_ids = np.asarray(reach_ids, dtype=int)
    obs = pd.read_parquet(OBS_PATH).copy()
    obs["reach_id"] = obs.reach_id.astype(int)
    folds = pd.read_parquet(FOLD_PATH)
    exp = pd.read_parquet(HYDRAULIC_PATH)
    exp = exp.loc[exp.hydraulic_scenario.eq("central_n0035")].copy()
    climatology = exp.loc[exp.year.between(2006, 2015)].groupby(["reach_id", "month"], as_index=False).agg(
        travel_time_full_days=("travel_time_full_days", "mean"),
        travel_time_midpoint_to_outlet_days=("travel_time_midpoint_to_outlet_days", "mean"),
        above_bankfull_travel_seconds_full=("above_bankfull_travel_seconds_full", "mean"),
        above_bankfull_travel_seconds_midpoint=("above_bankfull_travel_seconds_midpoint", "mean"),
    )
    exposure: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    eindex = exp.set_index(["year", "month", "reach_id"])
    cindex = climatology.set_index(["month", "reach_id"])
    for year, month in times:
        if year >= 2006:
            block = eindex.loc[[(year, month, int(r)) for r in reach_ids]]
        else:
            block = cindex.loc[[(month, int(r)) for r in reach_ids]]
        exposure[(year, month)] = (
            block.travel_time_full_days.to_numpy(float),
            block.travel_time_midpoint_to_outlet_days.to_numpy(float),
            block.above_bankfull_travel_seconds_full.to_numpy(float),
            block.above_bankfull_travel_seconds_midpoint.to_numpy(float),
        )
    hydro = pd.read_parquet(HYDRO_PATH).set_index(["year", "month", "reach_id"])
    routed_water = {}
    for year, month in times:
        if year >= 2006:
            routed_water[(year, month)] = hydro.loc[[(year, month, int(r)) for r in reach_ids], "q72_outlet_water_volume_m3"].to_numpy(float)
        else:
            # No scored TN observations occur before 2006. This value only
            # supports complete historical ledgers for the stateless R1a operator.
            month_block = hydro.reset_index().loc[lambda x: x.year.between(2006, 2015) & x.month.eq(month)]
            routed_water[(year, month)] = month_block.groupby("reach_id").q72_outlet_water_volume_m3.mean().loc[reach_ids].to_numpy(float)
    gw_frame = pd.read_parquet(GW_FORCING_PATH)
    aq_frame = pd.read_parquet(AQ_FORCING_PATH)
    gw_temperature = {mode: _forcing_array(gw_frame, "gw_temperature_mode", mode, "groundwater_temperature_c", times, reach_ids) for mode in gw_frame.gw_temperature_mode.unique()}
    aq_temperature = {mode: _forcing_array(aq_frame, "aquatic_temperature_mode", mode, "aquatic_temperature_c", times, reach_ids) for mode in aq_frame.aquatic_temperature_mode.unique()}
    order, downstream, terminal = topology_operators(reach_ids)
    return ReactionContext(reach_ids, times, arrays, early, obs, folds, exposure, routed_water, gw_temperature, aq_temperature, order, downstream, terminal)


def _groundwater_step(pre: np.ndarray, rho_t: float, has_discharge: np.ndarray, s_gw: float, q10: float, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_transport = np.where(has_discharge, -np.log(rho_t), 0.0)
    k_reaction = -np.log(s_gw) * q10 ** ((temperature - 20.0) / 10.0)
    total = k_transport + k_reaction
    event = 1.0 - np.exp(-total)
    active = total > HAZARD_EPS
    transport_fraction = np.divide(k_transport, total, out=np.zeros_like(total), where=active)
    reaction_fraction = np.divide(k_reaction, total, out=np.zeros_like(total), where=active)
    delivery = pre * event * transport_fraction
    removed = pre * event * reaction_fraction
    end = np.where(active, pre * np.exp(-total), pre)
    return delivery, removed, end


def spinup(
    context: ReactionContext, structure: str, soil_tau_month: int | None,
    delivery_mu_month: int, s_gw: float, q10_gw: float, gw_mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    state = {name: np.zeros(len(context.reach_ids), dtype=float) for name in ("son", "mobile", "quick", "gw")}
    rho_s = soil_tau_month / (1.0 + soil_tau_month) if soil_tau_month is not None else 0.0
    rho_t = delivery_mu_month / (1.0 + delivery_mu_month)
    temperature = context.gw_temperature[gw_mode]
    delta = np.inf
    for cycle in range(1, SPINUP_MAX_CYCLES + 1):
        before = np.concatenate([x.copy() for x in state.values()])
        for t in range(12):
            bypass, flush, quick_share, gw_share = _water_partitions(context.arrays, t)
            direct = context.early_positive * bypass
            remaining = context.early_positive - direct
            if structure == "S0":
                state["mobile"] += remaining
            elif structure == "S1":
                state["son"] += remaining
                mineralized = state["son"] * (1.0 - rho_s)
                state["son"] -= mineralized
                state["mobile"] += mineralized
            else:
                raise ValueError(structure)
            source_release = state["mobile"] * flush
            state["mobile"] -= source_release
            state["quick"] += direct + source_release * quick_share
            state["gw"] += source_release * gw_share
            quick_release = np.where(context.arrays["quick_release_mm"][t] > WATER_EPS, (1.0 - Q_RHO) * state["quick"], 0.0)
            state["quick"] -= quick_release
            _, _, state["gw"] = _groundwater_step(
                state["gw"], rho_t, context.arrays["gw_discharge_mm"][t] > WATER_EPS,
                s_gw, q10_gw, temperature[t],
            )
        delta = float(np.max(np.abs(np.concatenate(list(state.values())) - before)))
        if delta <= SPINUP_TOL_KG_N:
            break
    audit = {"cycles": cycle, "terminal_max_abs_delta_kg_n": delta, "converged": bool(delta <= SPINUP_TOL_KG_N), "minimum_state_kg_n": float(min(x.min() for x in state.values()))}
    if not audit["converged"]:
        raise RuntimeError("parameter-specific pre-1961 equilibrium did not converge")
    return state, audit


def simulate_local(
    context: ReactionContext, spec: dict[str, object], s_gw: float,
    q10_gw: float, gw_mode: str, end_year: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    structure = str(spec["source_structure"])
    tau = None if pd.isna(spec.get("soil_tau_month")) else int(spec["soil_tau_month"])
    mu = int(spec["delivery_mu_month"])
    state, spin = spinup(context, structure, tau, mu, s_gw, q10_gw, gw_mode)
    rho_s = tau / (1.0 + tau) if tau is not None else 0.0
    rho_t = mu / (1.0 + mu)
    gw_temperature = context.gw_temperature[gw_mode]
    records = []
    max_abs, max_rel, min_value = 0.0, 0.0, np.inf
    for t, (year, month) in enumerate(context.times):
        if year > end_year:
            break
        start = sum(x.astype(np.longdouble) for x in state.values())
        positive = context.arrays["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        negative = context.arrays["negative_legacy_eligible_n_surplus_kg_n_month"][t]
        bypass, flush, quick_share, gw_share = _water_partitions(context.arrays, t)
        direct = positive * bypass
        pool_input = positive - direct
        removed_total = np.zeros(len(context.reach_ids))
        if structure == "S0":
            removed, unmet = _withdraw(state["mobile"], negative.copy())
            removed_total += removed
            state["mobile"] += pool_input
        else:
            removed, left = _withdraw(state["mobile"], negative.copy())
            removed_total += removed
            removed_son, unmet = _withdraw(state["son"], left)
            removed_total += removed_son
            state["son"] += pool_input
            mineralized = state["son"] * (1.0 - rho_s)
            state["son"] -= mineralized
            state["mobile"] += mineralized
        source_release = state["mobile"] * flush
        state["mobile"] -= source_release
        state["quick"] += direct + source_release * quick_share
        state["gw"] += source_release * gw_share
        quick_release = np.where(context.arrays["quick_release_mm"][t] > WATER_EPS, (1.0 - Q_RHO) * state["quick"], 0.0)
        state["quick"] -= quick_release
        gw_release, gw_removed, state["gw"] = _groundwater_step(
            state["gw"], rho_t, context.arrays["gw_discharge_mm"][t] > WATER_EPS,
            s_gw, q10_gw, gw_temperature[t],
        )
        end = sum(x.astype(np.longdouble) for x in state.values())
        balance = positive.astype(np.longdouble) + start - quick_release - gw_release - gw_removed - removed_total - end
        reference = np.abs(positive.astype(float)) + np.abs(np.asarray(start, dtype=float)) + np.abs(quick_release) + np.abs(gw_release) + np.abs(gw_removed) + np.abs(removed_total) + np.abs(np.asarray(end, dtype=float))
        passes = mass_balance_pass(np.asarray(balance, dtype=float), reference)
        if not passes.all():
            raise RuntimeError("reactive local mass balance failed")
        max_abs = max(max_abs, float(np.max(np.abs(balance))))
        positive_ref = reference > MASS_ZERO_ABS_TOL_KG_N
        if positive_ref.any():
            max_rel = max(max_rel, float(np.max(np.abs(np.asarray(balance, dtype=float)[positive_ref]) / reference[positive_ref])))
        min_value = min(min_value, *(float(x.min()) for x in state.values()), float(quick_release.min()), float(gw_release.min()), float(gw_removed.min()))
        if year >= 2006:
            records.append(pd.DataFrame({
                "reach_id": context.reach_ids, "year": year, "month": month,
                "local_quick_tn_kg_n": quick_release, "local_gw_tn_kg_n": gw_release,
                "local_pre_aquatic_tn_kg_n": quick_release + gw_release,
                "gw_reaction_removed_kg_n": gw_removed,
                "gw_state_end_kg_n": state["gw"],
            }))
    frame = pd.concat(records, ignore_index=True)
    audit = {"spinup": spin, "max_abs_mass_balance_error_kg_n": max_abs, "max_relative_mass_balance_error": max_rel, "minimum_state_or_flux_kg_n": min_value, "delivery_mode": "T1", "effective_tn_delivery_mu_month": mu, "rho_T": rho_t}
    return frame, audit


def route_reactive(
    context: ReactionContext, local: pd.DataFrame, s_aq: float,
    q10_aq: float, aq_mode: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    index = {rid: i for i, rid in enumerate(context.reach_ids)}
    tindex = {time: i for i, time in enumerate(context.times)}
    aq_temperature = context.aq_temperature[aq_mode]
    records = []
    max_abs, max_rel = 0.0, 0.0
    k20_day = -np.log(s_aq) / REFERENCE_DAYS
    for (year, month), block in local.groupby(["year", "month"], sort=True):
        b = block.set_index("reach_id").loc[context.reach_ids]
        local_q = b.local_quick_tn_kg_n.to_numpy(float)
        local_g = b.local_gw_tn_kg_n.to_numpy(float)
        full_days, mid_days, _, _ = context.exposure[(int(year), int(month))]
        temp = aq_temperature[tindex[(int(year), int(month))]]
        rate = k20_day * q10_aq ** ((temp - 20.0) / 10.0)
        survival_full = np.exp(-rate * full_days)
        survival_mid = np.exp(-rate * mid_days)
        upstream_q = np.zeros(len(context.reach_ids))
        upstream_g = np.zeros(len(context.reach_ids))
        out_q = np.zeros(len(context.reach_ids))
        out_g = np.zeros(len(context.reach_ids))
        removed = np.zeros(len(context.reach_ids))
        for rid in context.order:
            i = index[rid]
            out_q[i] = upstream_q[i] * survival_full[i] + local_q[i] * survival_mid[i]
            out_g[i] = upstream_g[i] * survival_full[i] + local_g[i] * survival_mid[i]
            removed[i] = upstream_q[i] + upstream_g[i] + local_q[i] + local_g[i] - out_q[i] - out_g[i]
            closure = upstream_q[i] + upstream_g[i] + local_q[i] + local_g[i] - out_q[i] - out_g[i] - removed[i]
            reference = upstream_q[i] + upstream_g[i] + local_q[i] + local_g[i]
            if reference > MASS_ZERO_ABS_TOL_KG_N:
                max_rel = max(max_rel, abs(closure) / reference)
            else:
                max_abs = max(max_abs, abs(closure))
            if rid in context.downstream:
                down, fraction = context.downstream[rid]
                upstream_q[index[down]] += fraction * out_q[i]
                upstream_g[index[down]] += fraction * out_g[i]
        water = context.routed_water[(int(year), int(month))]
        item = pd.DataFrame({
            "reach_id": context.reach_ids, "year": int(year), "month": int(month),
            "routed_quick_tn_kg_n": out_q, "routed_gw_tn_kg_n": out_g,
            "aquatic_removed_kg_n": removed, "routed_water_volume_m3": water,
        })
        item["raw_reaction_tn_mg_l"] = np.divide((out_q + out_g) * 1000.0, water, out=np.zeros(len(water)), where=water > WATER_EPS)
        item["terminal_tree_id"] = item.reach_id.map(context.terminal).astype(int)
        records.append(item)
    audit = {"max_zero_mass_absolute_closure_kg_n": max_abs, "max_positive_mass_relative_closure": max_rel, "zero_mass_absolute_tolerance_kg_n": MASS_ZERO_ABS_TOL_KG_N, "relative_tolerance": MASS_REL_TOL}
    if max_abs > MASS_ZERO_ABS_TOL_KG_N or max_rel > MASS_REL_TOL:
        raise RuntimeError("R1a mass closure failed")
    return pd.concat(records, ignore_index=True), audit


def station_effects(raw_residual: np.ndarray, station_index: np.ndarray, n_stations: int) -> np.ndarray:
    sums = np.bincount(station_index, weights=raw_residual, minlength=n_stations)
    counts = np.bincount(station_index, minlength=n_stations).astype(float)
    return -sums / (counts + STATION_RIDGE)


def observation_predictions(
    context: ReactionContext, routed: pd.DataFrame, observation_subset: pd.DataFrame,
    layer: str, effect_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    joined = observation_subset.merge(routed, on=["reach_id", "year", "month"], how="inner", validate="many_to_one")
    if len(joined) != len(observation_subset):
        raise RuntimeError("reaction prediction did not preserve observation keys")
    effects = joined.station_key.astype(str).map(effect_map or {}).fillna(0.0).to_numpy(float) if layer == "P2" else np.zeros(len(joined))
    joined["station_effect"] = effects
    joined["pred_tn_mg_l"] = np.maximum(np.expm1(np.log1p(joined.raw_reaction_tn_mg_l.to_numpy(float)) + effects), 0.0)
    joined["layer"] = layer
    joined["eta_quick"] = 1.0
    joined["eta_gw"] = 1.0
    return joined


def evaluate_parameters(
    context: ReactionContext, spec: dict[str, object], params: np.ndarray,
    q10_gw: float, q10_aq: float, gw_mode: str, aq_mode: str,
    end_year: int, observation_subset: pd.DataFrame, layer: str,
    effect_map: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    local, local_audit = simulate_local(context, spec, float(params[0]), q10_gw, gw_mode, end_year)
    routed, route_audit = route_reactive(context, local, float(params[1]), q10_aq, aq_mode)
    pred = observation_predictions(context, routed, observation_subset, layer, effect_map)
    return pred, local, {"local": local_audit, "routing": route_audit}


def fit_fold(
    context: ReactionContext, spec: dict[str, object], fold,
    layer: str, q10_gw: float, q10_aq: float, gw_mode: str, aq_mode: str,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float], pd.DataFrame, dict[str, object]]:
    train_obs = context.observations.loc[context.observations.year.between(int(fold.train_start_year), int(fold.train_end_year))].copy()
    test_obs = context.observations.loc[context.observations.year.eq(int(fold.evaluation_year))].copy()
    levels = sorted(train_obs.station_key.astype(str).unique())
    lookup = {station: i for i, station in enumerate(levels)}
    station_index = train_obs.station_key.astype(str).map(lookup).to_numpy(int)
    observed_log = np.log1p(train_obs.tn_mg_l.to_numpy(float))
    replay_count = 0

    def residual(params: np.ndarray) -> np.ndarray:
        nonlocal replay_count
        replay_count += 1
        pred, _, _ = evaluate_parameters(context, spec, params, q10_gw, q10_aq, gw_mode, aq_mode, int(fold.train_end_year), train_obs, "P1")
        raw = np.log1p(pred.raw_reaction_tn_mg_l.to_numpy(float)) - observed_log
        effects = station_effects(raw, station_index, len(levels)) if layer == "P2" else np.zeros(len(levels))
        pieces = [raw + effects[station_index], np.sqrt(SURVIVAL_RIDGE_LAMBDA) * (params - 1.0)]
        if layer == "P2":
            pieces.append(np.sqrt(STATION_RIDGE) * effects)
        return np.concatenate(pieces)

    result = least_squares(
        residual, x0=np.array([SURVIVAL_INITIAL, SURVIVAL_INITIAL]),
        bounds=(np.array([SURVIVAL_LOWER, SURVIVAL_LOWER]), np.array([SURVIVAL_UPPER, SURVIVAL_UPPER])),
        method="trf", xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=120,
    )
    train_p1, _, _ = evaluate_parameters(context, spec, result.x, q10_gw, q10_aq, gw_mode, aq_mode, int(fold.train_end_year), train_obs, "P1")
    raw = np.log1p(train_p1.raw_reaction_tn_mg_l.to_numpy(float)) - observed_log
    effects = station_effects(raw, station_index, len(levels)) if layer == "P2" else np.zeros(len(levels))
    effect_map = {station: float(effects[i]) for station, i in lookup.items()}
    # Fresh evaluation replay with parameters and station effects frozen.
    test, local, audit = evaluate_parameters(context, spec, result.x, q10_gw, q10_aq, gw_mode, aq_mode, int(fold.evaluation_year), test_obs, layer, effect_map)
    test["fold_id"] = fold.fold_id
    diagnostics = {
        "fold_id": fold.fold_id, "evaluation_year": int(fold.evaluation_year), "layer": layer,
        "s_gw_refmonth_20c": float(result.x[0]), "s_aq_refmonth_20c": float(result.x[1]),
        "success": bool(result.success), "status": int(result.status), "cost": float(result.cost),
        "optimality": float(result.optimality), "nfev": int(result.nfev),
        "optimizer_dynamic_replay_count": replay_count,
        "fresh_evaluation_replay": True,
        "s_gw_boundary": bool(result.x[0] <= SURVIVAL_BOUNDARY_LOW or result.x[0] >= SURVIVAL_BOUNDARY_HIGH),
        "s_aq_boundary": bool(result.x[1] <= SURVIVAL_BOUNDARY_LOW or result.x[1] >= SURVIVAL_BOUNDARY_HIGH),
        "eta_quick": 1.0, "eta_gw": 1.0, "station_effect_count": len(effect_map) if layer == "P2" else 0,
    }
    return test, diagnostics, effect_map, local.loc[local.year.eq(int(fold.evaluation_year))].copy(), audit


def path_diagnostics(context: ReactionContext, local_eval: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    paths = pd.read_parquet(PATH_EXPOSURE_PATH)
    targets = predictions[["reach_id", "year", "month"]].drop_duplicates().rename(columns={"reach_id": "target_reach_id"})
    p = paths.merge(targets, on=["target_reach_id", "year", "month"], how="inner")
    mass = local_eval[["reach_id", "year", "month", "local_pre_aquatic_tn_kg_n"]].rename(columns={"reach_id": "source_reach_id"})
    p = p.merge(mass, on=["source_reach_id", "year", "month"], validate="many_to_one")
    p["path_weight_kg_n"] = p.local_pre_aquatic_tn_kg_n * p.cumulative_routing_fraction
    p["weighted_ab"] = p.path_weight_kg_n * p.path_ab_fraction
    p["weighted_over_month"] = p.path_weight_kg_n * p.path_exceeds_month.astype(float)
    d = p.groupby(["target_reach_id", "year", "month"], as_index=False).agg(
        path_source_weight_kg_n=("path_weight_kg_n", "sum"), weighted_ab=("weighted_ab", "sum"),
        weighted_over_month=("weighted_over_month", "sum"),
    )
    d["path_mass_weighted_ab_fraction"] = np.divide(d.weighted_ab, d.path_source_weight_kg_n, out=np.zeros(len(d)), where=d.path_source_weight_kg_n > 0)
    d["path_mass_fraction_exceeding_month"] = np.divide(d.weighted_over_month, d.path_source_weight_kg_n, out=np.zeros(len(d)), where=d.path_source_weight_kg_n > 0)
    d["low_ab_subset"] = d.path_mass_weighted_ab_fraction <= 0.05
    return d.drop(columns=["weighted_ab", "weighted_over_month"])
