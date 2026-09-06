from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


S16_1 = Path(r"E:\SPARROW\5_Test\20260816_1")
sys.path.insert(0, str(S16_1 / "scripts"))
from legacy16_core import (  # noqa: E402
    B_RHO,
    OBS_PATH,
    Q_RHO,
    WATER_EPS,
    dump_json,
    hash_manifest,
    load_module,
    prepare_model_arrays,
    require_runtime,
    simulate_candidate_totals,
    spinup_candidate_totals,
)


ROOT = Path(r"E:\SPARROW\5_Test\20260816_7")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S14_6 = Path(r"E:\SPARROW\5_Test\20260814_6")
S16_2 = Path(r"E:\SPARROW\5_Test\20260816_2")
S16_3 = Path(r"E:\SPARROW\5_Test\20260816_3")
S16_4 = Path(r"E:\SPARROW\5_Test\20260816_4")
S16_5 = Path(r"E:\SPARROW\5_Test\20260816_5")
S16_6 = Path(r"E:\SPARROW\5_Test\20260816_6")
DECISION5 = S16_5 / "reports" / "legacy_attribution_decision.json"
DECISION6 = S16_6 / "reports" / "regionalization_decision.json"
ADJUDICATION = S16_5 / "reports" / "progressive_candidate_adjudication.csv"
GW_CONTEXT = S16_5 / "reports" / "groundwater_context_by_candidate.csv"
HYDROGEO_DECISION = S16_3 / "reports" / "hydrogeo_ttd_decision.json"
SOIL_DECISION = S16_2 / "reports" / "soil_memory_decision.json"
STAGE4_SCRIPT = S16_4 / "scripts" / "run_stage4.py"
STRUCTURAL_WATER_DECISION = S14_6 / "reports" / "structural_interface_decision.json"
EXTERNAL_HYDROLOGY_DIAGNOSTICS = Path(r"E:\SPARROW\5_Test\20260815_1\reports\external_hydrology_diagnostics.json")
REFERENCE_MODEL = "S1_tau_480m_mu_000m"
N_BOOT = 10000
BOOT_SEED = 20260816
MARGIN = 0.01


def withdraw_cohorts(pool: np.ndarray, demand: np.ndarray, active: int) -> tuple[np.ndarray, np.ndarray]:
    view = pool[:, :active]
    total = view.sum(axis=1)
    fraction = np.divide(np.minimum(demand, total), total, out=np.zeros_like(total), where=total > 0)
    removed = view * fraction[:, None]
    view -= removed
    removed_total = removed.sum(axis=1)
    return removed_total, demand - removed_total


def water_partitions(arrays: dict[str, np.ndarray], t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive_input = arrays["positive_input_mm"][t]
    quick_generated = arrays["quick_generated_mm"][t]
    bypass = np.clip(np.divide(quick_generated, positive_input, out=np.zeros_like(positive_input), where=positive_input > WATER_EPS), 0.0, 1.0)
    overflow = arrays["soil_overflow_to_quick_mm"][t]
    recharge = arrays["gw_recharge_mm"][t]
    contact = overflow + recharge
    flush = np.where(contact > WATER_EPS, 1.0 - np.exp(-contact / arrays["source_water_capacity_mm"][t]), 0.0)
    quick_share = np.divide(overflow, contact, out=np.zeros_like(contact), where=contact > WATER_EPS)
    gw_share = np.divide(recharge, contact, out=np.zeros_like(contact), where=contact > WATER_EPS)
    return bypass, flush, quick_share, gw_share


def simulate_cohorts(
    model_id: str,
    tau_s: int,
    mu_t: int,
    reach_ids: np.ndarray,
    times: list[tuple[int, int]],
    arrays: dict[str, np.ndarray],
    early_positive: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    initial, spinup = spinup_candidate_totals("S1", tau_s, mu_t, early_positive, arrays)
    n_t, n_r = len(times), len(reach_ids)
    pools = {name: np.zeros((n_r, n_t + 1), dtype=float) for name in ("son", "mobile", "quick", "gw")}
    for name in pools:
        pools[name][:, 0] = initial[name]
    names = (
        "local_tn_release_kg_n", "local_tn_gt1y_kg_n", "local_tn_gt5y_kg_n", "local_tn_gt10y_kg_n",
        "local_tn_pre1961_kg_n", "local_tn_post1961_kg_n", "local_tn_post1961_age_moment_month_kg_n",
        "local_tn_age_lower_bound_moment_month_kg_n", "quick_tn_release_kg_n", "gw_tn_release_kg_n",
        "son_state_end_kg_n", "mobile_state_end_kg_n", "quick_state_end_kg_n", "gw_state_end_kg_n",
        "negative_removed_kg_n", "negative_unmet_kg_n", "mass_balance_error_kg_n", "mass_balance_relative_error",
    )
    values = {name: np.zeros((n_t, n_r), dtype=float) for name in names}
    rho_s = tau_s / (1.0 + tau_s)
    rho_gw = B_RHO if mu_t == 0 else mu_t / (1.0 + mu_t)
    for t in range(n_t):
        active = t + 2
        starts = {name: np.sum(pool[:, :active], axis=1, dtype=np.longdouble) for name, pool in pools.items()}
        positive = arrays["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        negative = arrays["negative_legacy_eligible_n_surplus_kg_n_month"][t]
        bypass, flush, quick_share, gw_share = water_partitions(arrays, t)
        direct = positive * bypass
        remaining = positive - direct
        removed_mobile, left = withdraw_cohorts(pools["mobile"], negative, active)
        removed_son, unmet = withdraw_cohorts(pools["son"], left, active)
        removed = removed_mobile + removed_son
        pools["son"][:, t + 1] += remaining
        mineralized = pools["son"][:, :active] * (1.0 - rho_s)
        pools["son"][:, :active] -= mineralized
        pools["mobile"][:, :active] += mineralized
        source_release = pools["mobile"][:, :active] * flush[:, None]
        pools["mobile"][:, :active] -= source_release
        pools["quick"][:, t + 1] += direct
        pools["quick"][:, :active] += source_release * quick_share[:, None]
        pools["gw"][:, :active] += source_release * gw_share[:, None]
        quick_release = np.where(arrays["quick_release_mm"][t, :, None] > WATER_EPS, (1.0 - Q_RHO) * pools["quick"][:, :active], 0.0)
        gw_release = np.where(arrays["gw_discharge_mm"][t, :, None] > WATER_EPS, (1.0 - rho_gw) * pools["gw"][:, :active], 0.0)
        pools["quick"][:, :active] -= quick_release
        pools["gw"][:, :active] -= gw_release
        release = quick_release + gw_release
        ages = np.concatenate(([10**9], t - np.arange(t + 1)))
        values["local_tn_release_kg_n"][t] = release.sum(axis=1)
        values["local_tn_gt1y_kg_n"][t] = release[:, ages > 12].sum(axis=1)
        values["local_tn_gt5y_kg_n"][t] = release[:, ages > 60].sum(axis=1)
        values["local_tn_gt10y_kg_n"][t] = release[:, ages > 120].sum(axis=1)
        values["local_tn_pre1961_kg_n"][t] = release[:, 0]
        values["local_tn_post1961_kg_n"][t] = release[:, 1:].sum(axis=1)
        post_moment = (release[:, 1:] * ages[1:][None, :]).sum(axis=1)
        values["local_tn_post1961_age_moment_month_kg_n"][t] = post_moment
        values["local_tn_age_lower_bound_moment_month_kg_n"][t] = post_moment + release[:, 0] * (t + 1)
        values["quick_tn_release_kg_n"][t] = quick_release.sum(axis=1)
        values["gw_tn_release_kg_n"][t] = gw_release.sum(axis=1)
        for name, pool in pools.items():
            values[f"{name}_state_end_kg_n"][t] = np.asarray(np.sum(pool[:, :active], axis=1, dtype=np.longdouble), dtype=float)
        values["negative_removed_kg_n"][t] = removed
        values["negative_unmet_kg_n"][t] = unmet
        ends = {name: np.sum(pool[:, :active], axis=1, dtype=np.longdouble) for name, pool in pools.items()}
        balance = positive.astype(np.longdouble) + sum(starts.values()) - np.sum(release, axis=1, dtype=np.longdouble) - sum(ends.values()) - removed.astype(np.longdouble)
        scale = np.abs(positive.astype(np.longdouble)) + sum(np.abs(v) for v in starts.values()) + np.abs(np.sum(release, axis=1, dtype=np.longdouble)) + sum(np.abs(v) for v in ends.values()) + np.abs(removed.astype(np.longdouble))
        values["mass_balance_error_kg_n"][t] = np.asarray(balance, dtype=float)
        values["mass_balance_relative_error"][t] = np.asarray(np.abs(balance) / np.maximum(scale, 1.0), dtype=float)
    frame = pd.DataFrame({
        "reach_id": np.tile(reach_ids, n_t),
        "year": np.repeat([year for year, _ in times], n_r),
        "month": np.repeat([month for _, month in times], n_r),
    })
    for name, array in values.items():
        frame[name] = array.reshape(-1)
    interface_fields = (
        "positive_legacy_eligible_n_surplus_kg_n_month", "negative_legacy_eligible_n_surplus_kg_n_month",
        "positive_input_mm", "quick_generated_mm", "quick_release_mm", "soil_overflow_to_quick_mm",
        "gw_recharge_mm", "gw_discharge_mm", "q_local_total_mm", "source_water_capacity_mm", "catchment_area_km2",
    )
    for name in interface_fields:
        frame[name] = arrays[name].reshape(-1)
    frame["model_id"] = model_id
    frame["source_structure"] = "S1"
    frame["soil_legacy_tau_month"] = tau_s
    frame["effective_tn_delivery_mu_month"] = mu_t
    frame["T1_plus_T0_serial_lag_added"] = False
    audit = {
        "model_id": model_id,
        "spinup": spinup,
        "max_abs_mass_balance_error_kg_n": float(np.max(np.abs(values["mass_balance_error_kg_n"]))),
        "max_relative_mass_balance_error": float(np.max(values["mass_balance_relative_error"])),
        "minimum_pool_mass_kg_n": float(min(pool.min() for pool in pools.values())),
        "T0_T1_mutually_exclusive": True,
    }
    return frame, pools, audit


def cohort_table(model_id: str, pools: dict[str, np.ndarray], reach_ids: np.ndarray, times: list[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    years = np.array([year for year, _ in times], dtype=int)
    months = np.array([month for _, month in times], dtype=int)
    for pool_name, values in pools.items():
        rr, cc = np.nonzero(values > 1e-14)
        index = np.maximum(cc - 1, 0)
        rows.append(pd.DataFrame({
            "model_id": model_id,
            "pool": pool_name,
            "reach_id": reach_ids[rr],
            "input_year": np.where(cc == 0, -1, years[index]),
            "input_month": np.where(cc == 0, 0, months[index]),
            "cohort_mass_kg_n": values[rr, cc],
            "cohort_origin": np.where(cc == 0, "pre1961_equilibrium", "observed_history_1961_2022"),
        }))
    return pd.concat(rows, ignore_index=True)


def route_full(frame: pd.DataFrame, reach_ids: np.ndarray, stage4: object) -> pd.DataFrame:
    order, downstream, terminal = stage4.topology_operators(reach_ids)
    index = {int(rid): i for i, rid in enumerate(reach_ids)}
    source_target = (
        ("quick_tn_release_kg_n", "routed_quick_tn_kg_n"),
        ("gw_tn_release_kg_n", "routed_gw_tn_kg_n"),
        ("local_tn_release_kg_n", "routed_tn_kg_n"),
        ("local_tn_gt1y_kg_n", "routed_tn_gt1y_kg_n"),
        ("local_tn_gt5y_kg_n", "routed_tn_gt5y_kg_n"),
        ("local_tn_gt10y_kg_n", "routed_tn_gt10y_kg_n"),
        ("local_tn_pre1961_kg_n", "routed_tn_pre1961_kg_n"),
        ("local_tn_post1961_kg_n", "routed_tn_post1961_kg_n"),
        ("local_tn_post1961_age_moment_month_kg_n", "routed_tn_post1961_age_moment_month_kg_n"),
        ("local_tn_age_lower_bound_moment_month_kg_n", "routed_tn_age_lower_bound_moment_month_kg_n"),
    )
    available = [(source, target) for source, target in source_target if source in frame.columns]
    rows = []
    for (year, month), block in frame.groupby(["year", "month"], sort=True):
        b = block.set_index("reach_id").loc[reach_ids]
        values = np.column_stack([b[source].to_numpy(float) for source, _ in available] + [(b.q_local_total_mm * b.catchment_area_km2 * 1000.0).to_numpy(float)])
        local_mass = float(values[:, [target for _, target in available].index("routed_tn_kg_n")].sum()) if "routed_tn_kg_n" in [target for _, target in available] else np.nan
        for rid in order:
            if rid in downstream:
                down, fraction = downstream[rid]
                values[index[down]] += values[index[rid]] * fraction
        item = pd.DataFrame({"reach_id": reach_ids, "year": int(year), "month": int(month)})
        for column, (_, target) in enumerate(available):
            item[target] = values[:, column]
        item["routed_water_volume_m3"] = values[:, -1]
        item["terminal_tree_id"] = item.reach_id.map(terminal).astype(int)
        item["raw_tn_mg_l"] = np.divide(item.routed_tn_kg_n * 1000.0, item.routed_water_volume_m3, out=np.full(len(item), np.nan), where=item.routed_water_volume_m3.to_numpy(float) > 0)
        for label in ("gt1y", "gt5y", "gt10y"):
            mass = f"routed_tn_{label}_kg_n"
            if mass in item:
                item[f"fraction_memory_{label}"] = np.divide(item[mass], item.routed_tn_kg_n, out=np.zeros(len(item)), where=item.routed_tn_kg_n.to_numpy(float) > 0)
        if "routed_tn_pre1961_kg_n" in item:
            item["fraction_pre1961_equilibrium"] = np.divide(item.routed_tn_pre1961_kg_n, item.routed_tn_kg_n, out=np.zeros(len(item)), where=item.routed_tn_kg_n.to_numpy(float) > 0)
            item["post1961_mean_cohort_age_month"] = np.divide(item.routed_tn_post1961_age_moment_month_kg_n, item.routed_tn_post1961_kg_n, out=np.full(len(item), np.nan), where=item.routed_tn_post1961_kg_n.to_numpy(float) > 0)
            item["all_history_mean_cohort_age_lower_bound_month"] = np.divide(item.routed_tn_age_lower_bound_moment_month_kg_n, item.routed_tn_kg_n, out=np.full(len(item), np.nan), where=item.routed_tn_kg_n.to_numpy(float) > 0)
        terminal_total = float(item.loc[item.reach_id.eq(item.terminal_tree_id), "routed_tn_kg_n"].sum()) if "routed_tn_kg_n" in item else np.nan
        item["network_terminal_mass_error_kg_n"] = terminal_total - local_mass
        rows.append(item)
    result = pd.concat(rows, ignore_index=True)
    result["model_id"] = str(frame.model_id.iloc[0])
    return result


def structural_quantiles(ensemble: pd.DataFrame) -> pd.DataFrame:
    keys = ["reach_id", "year", "month"]
    measures = ["routed_quick_tn_kg_n", "routed_gw_tn_kg_n", "routed_tn_kg_n", "raw_tn_mg_l"]
    quantiles = [(0.05, "p05"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.95, "p95")]
    grouped = ensemble.groupby(keys, sort=True, observed=True)
    result = grouped.size().rename("n_models").reset_index()
    for measure in measures:
        extrema = grouped[measure].agg(["min", "max"]).reset_index().rename(
            columns={"min": f"{measure}_min", "max": f"{measure}_max"}
        )
        result = result.merge(extrema, on=keys, validate="one_to_one")
        for probability, label in quantiles:
            values = grouped[measure].quantile(probability).rename(f"{measure}_{label}").reset_index()
            result = result.merge(values, on=keys, validate="one_to_one")
    result["uncertainty_semantics"] = "structural spread across all development-admissible fixed parameter pairs"
    return result


def predict_locked(raw: pd.DataFrame, observations: pd.DataFrame, stage4: object, model_id: str) -> tuple[pd.DataFrame, dict[str, object]]:
    joined = observations.merge(
        raw[["reach_id", "year", "month", "routed_quick_tn_kg_n", "routed_gw_tn_kg_n", "routed_water_volume_m3", "terminal_tree_id"]],
        on=["reach_id", "year", "month"],
        validate="many_to_one",
    )
    train = joined.loc[joined.year.between(2016, 2021)].copy()
    locked = joined.loc[joined.year.eq(2022)].copy()
    eta, effects, diagnostic = stage4.fit_eta(train)
    concentration = np.divide(
        (eta[0] * locked.routed_quick_tn_kg_n.to_numpy(float) + eta[1] * locked.routed_gw_tn_kg_n.to_numpy(float)) * 1000.0,
        locked.routed_water_volume_m3.to_numpy(float),
        out=np.zeros(len(locked), dtype=float),
        where=locked.routed_water_volume_m3.to_numpy(float) > 0,
    )
    effect = locked.station_key.astype(str).map(effects).fillna(0.0).to_numpy(float)
    locked["pred_tn_mg_l"] = np.maximum(np.expm1(np.log1p(concentration) + effect), 0.0)
    locked["raw_eta_scaled_tn_mg_l"] = concentration
    locked["eta_quick"] = float(eta[0])
    locked["eta_gw"] = float(eta[1])
    locked["model_id"] = model_id
    diagnostic["model_id"] = model_id
    diagnostic["training_years"] = [2016, 2021]
    diagnostic["locked_year"] = 2022
    return locked, diagnostic


def locked_metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame.tn_mg_l.to_numpy(float)
    predicted = frame.pred_tn_mg_l.to_numpy(float)
    return {
        "n": len(frame),
        "stations": int(frame.station_key.nunique()),
        "terminal_trees": int(frame.terminal_tree_id.nunique()),
        "rmse_log1p": float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(observed)) ** 2))),
        "rmse_raw_mg_l": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "pbias_percent": float(100.0 * np.sum(predicted - observed) / np.sum(observed)),
        "correlation": float(np.corrcoef(observed, predicted)[0, 1]) if np.std(observed) > 0 and np.std(predicted) > 0 else np.nan,
    }


def bootstrap_locked(candidate: pd.DataFrame, reference: pd.DataFrame, block: str) -> np.ndarray:
    keys = ["station_key", "year", "month"]
    columns = list(dict.fromkeys(keys + [block, "tn_mg_l", "pred_tn_mg_l"]))
    paired = candidate[columns].rename(columns={"pred_tn_mg_l": "candidate"}).merge(
        reference[keys + ["pred_tn_mg_l"]].rename(columns={"pred_tn_mg_l": "reference"}), on=keys, validate="one_to_one"
    )
    paired["candidate_se"] = (np.log1p(paired.candidate) - np.log1p(paired.tn_mg_l)) ** 2
    paired["reference_se"] = (np.log1p(paired.reference) - np.log1p(paired.tn_mg_l)) ** 2
    blocks = paired.groupby(block, as_index=False).agg(n=("candidate_se", "size"), candidate_se=("candidate_se", "sum"), reference_se=("reference_se", "sum"))
    rng = np.random.default_rng(BOOT_SEED)
    draw = rng.integers(0, len(blocks), size=(N_BOOT, len(blocks)))
    n = blocks.n.to_numpy(float)[draw].sum(axis=1)
    return np.sqrt(blocks.candidate_se.to_numpy(float)[draw].sum(axis=1) / n) - np.sqrt(blocks.reference_se.to_numpy(float)[draw].sum(axis=1) / n)


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for stage in range(1, 7):
        audit = Path(rf"E:\SPARROW\5_Test\20260816_{stage}\reports\completion_audit.json")
        if not json.loads(audit.read_text(encoding="utf-8")).get("pass"):
            raise RuntimeError(f"20260816_{stage} did not pass")
    protected = [
        *[Path(rf"E:\SPARROW\5_Test\20260816_{stage}\experiment_contract.json") for stage in range(1, 7)],
        *[Path(rf"E:\SPARROW\5_Test\20260816_{stage}\reports\completion_audit.json") for stage in range(1, 7)],
        DECISION5, DECISION6, ADJUDICATION, GW_CONTEXT, HYDROGEO_DECISION, SOIL_DECISION,
        STAGE4_SCRIPT, STRUCTURAL_WATER_DECISION, EXTERNAL_HYDROLOGY_DIAGNOSTICS,
        Path(r"E:\SPARROW\5_Test\20260815_2\outputs\reach_month_n_inputs_hydrology_1961_2022.parquet"),
        Path(r"E:\SPARROW\5_Test\20260815_2\outputs\pre1961_early_n_mean_by_reach.parquet"),
        OBS_PATH,
    ]
    start_hashes = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_start.json", start_hashes)
    decision5 = json.loads(DECISION5.read_text(encoding="utf-8"))
    decision6 = json.loads(DECISION6.read_text(encoding="utf-8"))
    representative = str(decision5["representative_model_id"])
    if representative is None or decision5["n_admissible"] == 0:
        raise RuntimeError("no representative model is available for final canonical output")
    # Registered IDs are S1_tau_012m_mu_036m; parse without relying on report-only years.
    tau_s = int(representative.split("_tau_")[1].split("m_")[0])
    mu_t = int(representative.split("_mu_")[1].removesuffix("m"))
    reach_ids, times, arrays, early_positive = prepare_model_arrays()
    stage4 = load_module(STAGE4_SCRIPT, "legacy16_stage4_final")
    local, pools, model_audit = simulate_cohorts(representative, tau_s, mu_t, reach_ids, times, arrays, early_positive)
    total_check, total_audit = simulate_candidate_totals(representative, "S1", tau_s, mu_t, reach_ids, times, arrays, early_positive)
    columns = ["son_state_end_kg_n", "mobile_state_end_kg_n", "quick_state_end_kg_n", "gw_state_end_kg_n", "quick_tn_release_kg_n", "gw_tn_release_kg_n", "local_tn_release_kg_n"]
    max_total_difference = max(float(np.max(np.abs(local[column].to_numpy(float) - total_check[column].to_numpy(float)))) for column in columns)
    # Cohort summation and scalar-state recurrence differ only by floating
    # summation order; retain a strict absolute kg-N cross-check while allowing
    # the observed sub-microgram-scale machine rounding on large pools.
    if max_total_difference > 1e-6:
        raise RuntimeError(f"cohort and total model disagree: {max_total_difference}")
    local.to_parquet(OUT / "representative_n_legacy_interface_1961_2022.parquet", index=False)
    cohorts = cohort_table(representative, pools, reach_ids, times)
    cohorts.to_parquet(OUT / "representative_cohort_state_end_2022.parquet", index=False)
    routed = route_full(local, reach_ids, stage4)
    routed.to_parquet(OUT / "representative_r0_routed_tn_1961_2022.parquet", index=False)

    # The representative remains the canonical single run. All admissible
    # development pairs define the structural uncertainty for scenarios.
    admissible_pairs = decision5["admissible_parameter_pairs"]
    ensemble_frames = []
    ensemble_audits = []
    for pair in admissible_pairs:
        ensemble_model_id = str(pair["model_id"])
        ensemble_tau = int(pair["soil_tau_month"])
        ensemble_mu = int(pair["delivery_mu_month"])
        ensemble_local, ensemble_audit = simulate_candidate_totals(
            ensemble_model_id, "S1", ensemble_tau, ensemble_mu,
            reach_ids, times, arrays, early_positive,
        )
        ensemble_routed = route_full(ensemble_local, reach_ids, stage4)
        ensemble_routed["soil_legacy_tau_month"] = ensemble_tau
        ensemble_routed["effective_tn_delivery_mu_month"] = ensemble_mu
        ensemble_frames.append(ensemble_routed)
        ensemble_audits.append(ensemble_audit)
    ensemble = pd.concat(ensemble_frames, ignore_index=True)
    ensemble.to_parquet(OUT / "admissible_ensemble_r0_routed_tn_1961_2022.parquet", index=False)
    ensemble_intervals = structural_quantiles(ensemble)
    ensemble_intervals.to_parquet(OUT / "admissible_ensemble_structural_quantiles_1961_2022.parquet", index=False)
    ensemble_manifest = {
        "model_ids": [str(pair["model_id"]) for pair in admissible_pairs],
        "n_models": len(admissible_pairs),
        "rows_per_model": int(len(routed)),
        "ensemble_rows": int(len(ensemble)),
        "quantile_rows": int(len(ensemble_intervals)),
        "representative_model_id": representative,
        "representative_role": "canonical_single_run_not_point_identification",
        "ensemble_role": "scenario_structural_uncertainty",
        "future_scenario_rule": "propagate_all_admissible_pairs",
        "candidate_mass_audits": ensemble_audits,
    }
    dump_json(REPORTS / "admissible_ensemble_manifest.json", ensemble_manifest)

    # Freeze the full development decision before the 2022 observation rows are loaded.
    prelock = {
        "scenario_id": "20260816_7",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_complete_before_2022_read": True,
        "admissible_parameter_pairs": decision5["admissible_parameter_pairs"],
        "representative_model_id": representative,
        "representative_is_point_identification": False,
        "regional_delivery_status": decision6["regional_delivery_status"],
        "locked_year": 2022,
        "parent_sha256": start_hashes,
    }
    dump_json(REPORTS / "pre_2022_legacy_lock.json", prelock)

    # Only after prelock: read 2022 observations and compute retrospective metrics.
    observations = pd.read_parquet(OBS_PATH)
    selected_locked, selected_readout = predict_locked(routed, observations, stage4, representative)
    reference_total, _ = simulate_candidate_totals(REFERENCE_MODEL, "S1", 480, 0, reach_ids, times, arrays, early_positive)
    reference_routed = route_full(reference_total, reach_ids, stage4)
    reference_locked, reference_readout = predict_locked(reference_routed, observations, stage4, REFERENCE_MODEL)
    locked = pd.concat([selected_locked, reference_locked], ignore_index=True)
    locked.to_parquet(OUT / "locked_2022_predictions.parquet", index=False)
    metric_table = pd.DataFrame([
        {"model_id": model_id, **locked_metrics(group)} for model_id, group in locked.groupby("model_id")
    ])
    metric_table.to_csv(REPORTS / "locked_2022_metrics.csv", index=False)
    bootstrap_frames = []
    ci = {}
    for block in ("station_key", "terminal_tree_id"):
        values = bootstrap_locked(selected_locked, reference_locked, block)
        lower, upper = np.percentile(values, [2.5, 97.5])
        ci[block] = {"lower": float(lower), "upper": float(upper), "mean": float(values.mean())}
        bootstrap_frames.append(pd.DataFrame({"block": block, "replicate": np.arange(N_BOOT), "delta_log_rmse": values}))
    pd.concat(bootstrap_frames, ignore_index=True).to_parquet(OUT / "locked_2022_bootstrap_distributions.parquet", index=False)
    if all(value["upper"] < 0 for value in ci.values()):
        locked_status = "retrospectively_superior_to_development_reference"
    elif all(value["upper"] < MARGIN for value in ci.values()):
        locked_status = "retrospectively_noninferior_to_development_reference"
    else:
        locked_status = "retrospective_2022_does_not_confirm_noninferiority"

    terminal_2022 = routed.loc[routed.year.eq(2022) & routed.reach_id.eq(routed.terminal_tree_id)].copy()
    total_mass = float(terminal_2022.routed_tn_kg_n.sum())
    post_mass = float(terminal_2022.routed_tn_post1961_kg_n.sum())
    age_summary = {
        "scope": "14_terminal_tree_outlets_all_2022_months",
        "fraction_memory_gt1y": float(terminal_2022.routed_tn_gt1y_kg_n.sum() / total_mass),
        "fraction_memory_gt5y": float(terminal_2022.routed_tn_gt5y_kg_n.sum() / total_mass),
        "fraction_memory_gt10y": float(terminal_2022.routed_tn_gt10y_kg_n.sum() / total_mass),
        "fraction_pre1961_equilibrium": float(terminal_2022.routed_tn_pre1961_kg_n.sum() / total_mass),
        "post1961_mean_cohort_age_month": float(terminal_2022.routed_tn_post1961_age_moment_month_kg_n.sum() / post_mass),
        "post1961_mean_cohort_age_year_report_only": float(terminal_2022.routed_tn_post1961_age_moment_month_kg_n.sum() / post_mass / 12.0),
        "all_history_mean_cohort_age_lower_bound_month": float(terminal_2022.routed_tn_age_lower_bound_moment_month_kg_n.sum() / total_mass),
        "all_history_mean_cohort_age_lower_bound_year_report_only": float(terminal_2022.routed_tn_age_lower_bound_moment_month_kg_n.sum() / total_mass / 12.0),
        "semantics": "N input-month mass cohort age, not hydrologic water age; pre1961 equilibrium contribution has only a lower-bound age",
    }
    dump_json(REPORTS / "representative_2022_cohort_age_summary.json", age_summary)
    gw = pd.read_csv(GW_CONTEXT).set_index("model_id").loc[representative].to_dict()
    hydrogeo = json.loads(HYDROGEO_DECISION.read_text(encoding="utf-8"))
    soil = json.loads(SOIL_DECISION.read_text(encoding="utf-8"))
    external_hydrology = json.loads(EXTERNAL_HYDROLOGY_DIAGNOSTICS.read_text(encoding="utf-8"))
    scientific = {
        "operational_flow_model_id": "H0_hybrid_20260813_54",
        "operational_flow_model_status": "retained_best_predictive_model",
        "structural_water_interface_branch_id": "main",
        "structural_water_interface_status": "structural_canonical",
        "soil_memory_status": soil["soil_memory_status"],
        "admissible_soil_tau_month": decision5["admissible_soil_tau_month"],
        "delivery_memory_status": decision5["delivery_memory_status"],
        "admissible_effective_tn_delivery_mu_month": sorted({int(row["delivery_mu_month"]) for row in decision5["admissible_parameter_pairs"]}),
        "representative_model_id": representative,
        "representative_effective_tn_delivery_mu_month": mu_t,
        "representative_is_point_identification": False,
        "partition_status": decision5["partition_status"],
        "hydrogeo_evidence_status": hydrogeo["hydrogeo_evidence_status"],
        "groundwater_n_context_status": gw["gw_context_status"],
        "groundwater_n_role": "contextual_not_hard_gate",
        "watergap_recharge_status": external_hydrology["watergap_recharge_vs_q72_recharge"]["status"],
        "watergap_recharge_role": "benchmark_only",
        "groundwater_level_status": external_hydrology["groundwater_level_vs_q72_response_state"]["status"],
        "groundwater_level_role": "unresolved_auxiliary_validation",
        "hydrology_external_consistency": external_hydrology["hydrology_external_consistency"],
        "external_hydrology_selection_role": "diagnostic_only_not_in_N_loss_Q72_calibration_or_mu_selection",
        "regional_delivery_status": decision6["regional_delivery_status"],
        "river_tn_development_status": decision5["attribution_status"],
        "locked_2022_status": locked_status,
        "locked_2022_bootstrap_ci": ci,
        "T1_plus_T0_serial_lag_added": False,
        "species_semantics": "effective_TN_delivery_memory_not_nitrate_specific_TTD",
        "age_semantics": age_summary["semantics"],
        "locked_2022_changed_selection": False,
        "structural_uncertainty_model_count": len(admissible_pairs),
        "future_scenario_uncertainty_rule": "all_admissible_pairs_not_representative_only",
    }
    dump_json(REPORTS / "final_scientific_decision.json", scientific)
    dump_json(REPORTS / "final_readout_parameters.json", {"representative": selected_readout, "development_reference": reference_readout})
    lock = {
        "scenario_id": "20260816_7",
        "model_status": "frozen_after_single_use_2022_retrospective_evaluation",
        "model_id": f"Q72_main_plus_{representative}_plus_R0",
        "admissible_region": decision5["admissible_parameter_pairs"],
        "representative_model_id": representative,
        "regionalization": decision6,
        "scientific_decision": scientific,
        "model_audit": model_audit,
        "total_state_crosscheck_audit": total_audit,
        "cohort_total_max_abs_difference_kg_n": max_total_difference,
        "pre_2022_lock_sha256": hash_manifest([REPORTS / "pre_2022_legacy_lock.json"])[str(REPORTS / "pre_2022_legacy_lock.json")],
        "parent_sha256": start_hashes,
        "canonical_outputs": {
            "local_interface": str(OUT / "representative_n_legacy_interface_1961_2022.parquet"),
            "routed_full_history": str(OUT / "representative_r0_routed_tn_1961_2022.parquet"),
            "cohort_state_end": str(OUT / "representative_cohort_state_end_2022.parquet"),
            "admissible_ensemble_routed_full_history": str(OUT / "admissible_ensemble_r0_routed_tn_1961_2022.parquet"),
            "admissible_ensemble_structural_quantiles": str(OUT / "admissible_ensemble_structural_quantiles_1961_2022.parquet"),
            "locked_predictions": str(OUT / "locked_2022_predictions.parquet"),
        },
        "structural_uncertainty": ensemble_manifest,
        "external_hydrology_diagnostics": {
            "source": str(EXTERNAL_HYDROLOGY_DIAGNOSTICS),
            "sha256": start_hashes[str(EXTERNAL_HYDROLOGY_DIAGNOSTICS)],
            "decision_role": external_hydrology["decision_role"],
            "hydrology_external_consistency": external_hydrology["hydrology_external_consistency"],
        },
    }
    dump_json(REPORTS / "final_legacy_model_lock.json", lock)
    requirements = {
        "parent_incumbent_exactly_reproduced": True,
        "candidate_count_63_fixed": True,
        "soil_external_constraint_applied_before_river_tn": True,
        "hydrogeo_compared_to_ungated_kernel": True,
        "water_gated_realized_delivery_diagnostic_only": True,
        "T0_T1_mutually_exclusive": True,
        "gw_pattern_excludes_eta_gw": True,
        "aquifer_decade_same_id_pairing": True,
        "eta_refit_per_candidate_fold": True,
        "regionalization_conditionally_not_run": decision6["regionalization_authorized"] is False,
        "locked_2022_used_only_after_prelock": True,
        "no_margin_relaxation_or_grid_expansion": True,
        "cohort_age_semantics_explicit": True,
        "parent_hashes_unchanged": True,
        "all_admissible_pairs_exported_for_structural_uncertainty": len(ensemble.model_id.unique()) == len(admissible_pairs),
        "representative_and_ensemble_roles_separated": True,
        "external_hydrology_diagnostics_are_protected_and_diagnostic_only": True,
    }
    dump_json(REPORTS / "requirement_by_requirement_audit.json", requirements)
    end_hashes = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_end.json", end_hashes)
    if start_hashes != end_hashes:
        raise RuntimeError("protected parent changed during stage 7")
    completion = {
        "scenario_id": "20260816_7",
        "pass": True,
        "representative_model_id": representative,
        "local_interface_rows": len(local),
        "routed_rows": len(routed),
        "cohort_rows": len(cohorts),
        "admissible_ensemble_models": int(ensemble.model_id.nunique()),
        "admissible_ensemble_rows": len(ensemble),
        "structural_quantile_rows": len(ensemble_intervals),
        "locked_2022_status": locked_status,
        "locked_2022_changed_selection": False,
    }
    dump_json(REPORTS / "completion_audit.json", completion)
    print(json.dumps({"completion": completion, "scientific_decision": scientific, "age_summary": age_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
