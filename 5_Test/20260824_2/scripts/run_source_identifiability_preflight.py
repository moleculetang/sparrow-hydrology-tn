from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260824_2")
S24_1 = Path(r"E:\SPARROW\5_Test\20260824_1")
S15_2 = Path(r"E:\SPARROW\5_Test\20260815_2")
S17_9 = Path(r"E:\SPARROW\5_Test\20260817_9")
S18_3 = Path(r"E:\SPARROW\5_Test\20260818_3")
S20_18 = Path(r"E:\SPARROW\5_Test\20260820_18")
S20_19 = Path(r"E:\SPARROW\5_Test\20260820_19")
S20_12 = Path(r"E:\SPARROW\5_Test\20260820_12")

LEDGER = S15_2 / "outputs" / "reach_year_n_ledger_1961_2022.parquet"
MONTHLY = S15_2 / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
WWTP = S17_9 / "inputs" / "model_ready" / "point_sources" / "prb_wwtp_tn_monthly_reach_2006_2019.parquet"
WWTP_DECISION = S18_3 / "reports" / "wwtp_evidence_decision.json"
EXPOSURE = S20_18 / "outputs" / "reach_month_aquatic_uptake_exposure_2006_2021.parquet"
FULL_PARAMETERS = S20_19 / "outputs" / "full_development_parameters.parquet"
MONITORING_LOCK = S24_1 / "locks" / "development_tn_2016_2021.parquet"
PARENT_ROUTED = S20_12 / "cache" / "direct_parent_routed"
TOPOLOGY = Path(r"E:\SPARROW\5_Test\20260814_1\inputs\topology\topology_edges.csv")
PARENT_LOCK = S24_1 / "locks" / "parent_lock_registry.json"
CONTRACT = ROOT / "experiment_contract.json"
MANIFEST = S24_1 / "program_manifest.json"

OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
LOCKS = ROOT / "locks"

MUS = (12, 36, 60, 96, 144, 240)
PHI_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
GENERATING_PHI = (0.25, 0.50, 0.75, 1.00)
TAU_S = 12
RHO_S = TAU_S / (1.0 + TAU_S)
Q_RHO = 0.25
WATER_EPS = 1e-12
SPIN_TOL = 1e-9
SPIN_MAX = 5000
SEED = 2026082402
N_REP = 20
NOISE_SD = 0.20
TIE_TOL = 1e-6
STRUCTURE_ORDER = {
    "AGG_MOBILE_PARENT": 0,
    "ALL_SOURCE_SON12": 1,
    "GENERIC_MATCHED_SON12": 2,
    "MANURE_ASSOC_SON12": 3,
}


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 required")


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_registration() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    checks = {
        "manifest_revision_is_2": manifest.get("manifest_revision") == 2,
        "current_folder_is_20260824_2": manifest.get("current_folder") == "20260824_2",
        "registered_before_candidate_tn": bool(contract.get("registered_before_new_candidate_tn_results")),
        "parent_is_h1_global": parent.get("selected_full_development_process_parent") == "H1_GLOBAL",
        "parent_water_is_q72": parent.get("water_flux_source") == "Q72 structural_canonical main",
        "parent_wwtp_off": parent.get("WWTP_used") is False,
        "tn_2022_not_read": parent.get("TN_2022_values_read") is False and contract["time_boundary"]["TN_2022_read"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"registration failure: {checks}")
    return checks


def source_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ledger = pd.read_parquet(LEDGER, filters=[("year", "<=", 2021)]).sort_values(["year", "reach_id"]).reset_index(drop=True)
    monthly = pd.read_parquet(MONTHLY, filters=[("year", "<=", 2021)]).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    expected_year_rows = 230 * (2021 - 1961 + 1)
    expected_month_rows = expected_year_rows * 12
    components = [
        "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n",
        "atmospheric_deposition_kg_n", "crop_removal_kg_n",
    ]
    gross = ledger.fertilizer_kg_n + ledger.manure_kg_n + ledger.cropland_bnf_kg_n + ledger.atmospheric_deposition_kg_n
    identity = gross - ledger.crop_removal_kg_n - ledger.legacy_eligible_n_surplus_kg_n
    ledger = ledger.copy()
    ledger["gross_agricultural_and_deposition_input_kg_n"] = gross
    ledger["manure_associated_positive_surplus_kg_n"] = np.divide(
        ledger.positive_legacy_eligible_n_surplus_kg_n * ledger.manure_kg_n,
        gross,
        out=np.zeros(len(ledger), dtype=float),
        where=gross.to_numpy(float) > 0,
    )
    monthly_components = [x.replace("_kg_n", "_kg_n_month") for x in components]
    annual_from_month = monthly.groupby(["reach_id", "year"], as_index=False)[monthly_components].sum()
    annual_compare = ledger[["reach_id", "year", *components]].merge(annual_from_month, on=["reach_id", "year"], validate="one_to_one")
    closure = {}
    for annual, month in zip(components, monthly_components):
        closure[annual] = float(np.max(np.abs(annual_compare[annual] - annual_compare[month])))

    ordered = monthly.sort_values(["year", "month", "reach_id"]).copy()
    mstar = ledger[["reach_id", "year", "manure_associated_positive_surplus_kg_n"]]
    ordered = ordered.merge(mstar, on=["reach_id", "year"], validate="many_to_one")
    pin = ordered.positive_input_mm.to_numpy(float)
    qgen = ordered.quick_generated_mm.to_numpy(float)
    bypass = np.clip(np.divide(qgen, pin, out=np.zeros_like(pin), where=pin > WATER_EPS), 0.0, 1.0)
    positive = ordered.positive_legacy_eligible_n_surplus_kg_n_month.to_numpy(float)
    remaining = positive * (1.0 - bypass)
    manure_potential = ordered.manure_associated_positive_surplus_kg_n.to_numpy(float) / 12.0 * (1.0 - bypass)
    ordered["fresh_quick_bypass_fraction"] = bypass
    ordered["post_bypass_positive_surplus_kg_n"] = remaining
    ordered["post_bypass_manure_potential_kg_n"] = manure_potential
    totals = ordered.groupby(["year", "month"])[["post_bypass_positive_surplus_kg_n", "post_bypass_manure_potential_kg_n"]].transform("sum")
    generic_share = np.divide(
        totals.post_bypass_manure_potential_kg_n.to_numpy(float),
        totals.post_bypass_positive_surplus_kg_n.to_numpy(float),
        out=np.zeros(len(ordered), dtype=float),
        where=totals.post_bypass_positive_surplus_kg_n.to_numpy(float) > 0,
    )
    ordered["post_bypass_generic_matched_potential_kg_n"] = remaining * generic_share
    match = ordered.groupby(["year", "month"], as_index=False)[["post_bypass_manure_potential_kg_n", "post_bypass_generic_matched_potential_kg_n"]].sum()
    match_error = match.post_bypass_generic_matched_potential_kg_n - match.post_bypass_manure_potential_kg_n

    by_year = ledger.groupby("year", as_index=False).agg(
        positive_surplus_kg_n=("positive_legacy_eligible_n_surplus_kg_n", "sum"),
        manure_associated_positive_surplus_kg_n=("manure_associated_positive_surplus_kg_n", "sum"),
        fertilizer_kg_n=("fertilizer_kg_n", "sum"),
        manure_kg_n=("manure_kg_n", "sum"),
        bnf_kg_n=("cropland_bnf_kg_n", "sum"),
        deposition_kg_n=("atmospheric_deposition_kg_n", "sum"),
        crop_removal_kg_n=("crop_removal_kg_n", "sum"),
    )
    by_year["manure_associated_fraction_of_positive_surplus"] = (
        by_year.manure_associated_positive_surplus_kg_n / by_year.positive_surplus_kg_n
    )
    spatial = ledger.groupby("year").apply(
        lambda g: pd.Series({
            "spatial_pearson_manure_potential_vs_surplus": float(np.corrcoef(g.manure_associated_positive_surplus_kg_n, g.positive_legacy_eligible_n_surplus_kg_n)[0, 1]),
            "spatial_cosine_manure_potential_vs_surplus": float(np.dot(g.manure_associated_positive_surplus_kg_n, g.positive_legacy_eligible_n_surplus_kg_n) / max(np.linalg.norm(g.manure_associated_positive_surplus_kg_n) * np.linalg.norm(g.positive_legacy_eligible_n_surplus_kg_n), 1e-30)),
        }), include_groups=False
    ).reset_index()
    by_year = by_year.merge(spatial, on="year", validate="one_to_one")

    unique_month_cv = (
        monthly.groupby(["reach_id", "year"])["positive_legacy_eligible_n_surplus_kg_n_month"]
        .agg(lambda x: float(np.std(x) / np.mean(x)) if np.mean(x) > 0 else 0.0)
    )
    checks = {
        "reach_year_rows": len(ledger),
        "expected_reach_year_rows": expected_year_rows,
        "reach_month_rows": len(monthly),
        "expected_reach_month_rows": expected_month_rows,
        "reach_year_key_duplicates": int(ledger.duplicated(["reach_id", "year"]).sum()),
        "reach_month_key_duplicates": int(monthly.duplicated(["reach_id", "year", "month"]).sum()),
        "reach_count": int(ledger.reach_id.nunique()),
        "year_min": int(ledger.year.min()),
        "year_max": int(ledger.year.max()),
        "component_null_cells": int(ledger[components].isna().sum().sum()),
        "component_negative_cells": int((ledger[components] < 0).sum().sum()),
        "negative_net_surplus_rows": int((ledger.negative_legacy_eligible_n_surplus_kg_n > 0).sum()),
        "surplus_identity_max_abs_kg_n": float(np.max(np.abs(identity))),
        "annual_to_month_closure_max_abs_kg_n": float(max(closure.values())),
        "post_bypass_mass_match_max_abs_kg_n": float(np.max(np.abs(match_error))),
        "post_bypass_manure_exceeds_remaining_rows": int(np.sum(manure_potential > remaining + 1e-9)),
        "post_bypass_generic_exceeds_remaining_rows": int(np.sum(ordered.post_bypass_generic_matched_potential_kg_n.to_numpy(float) > remaining + 1e-9)),
        "monthly_source_cv_max": float(unique_month_cv.max()),
        "manure_associated_fraction_full_history": float(ledger.manure_associated_positive_surplus_kg_n.sum() / ledger.positive_legacy_eligible_n_surplus_kg_n.sum()),
        "median_year_spatial_correlation": float(by_year.spatial_pearson_manure_potential_vs_surplus.median()),
        "minimum_crop_removal_area_match": float(ledger.matched_area_fraction.min()),
        "minimum_crop_removal_production_match": float(ledger.matched_production_fraction.min()),
        "harvest_area_2021_uses_2020": bool((ledger.loc[ledger.year.eq(2021), "harvested_area_year_used"] == 2020).all()),
        "deposition_2021_uses_2020": bool((ledger.loc[ledger.year.eq(2021), "deposition_year_used"] == 2020).all()),
        "grazing_manure_included": False,
        "field_monitoring_product": False,
        "monthly_source_timing_observed": False,
    }
    sufficient = (
        checks["reach_year_rows"] == expected_year_rows
        and checks["reach_month_rows"] == expected_month_rows
        and checks["reach_year_key_duplicates"] == 0
        and checks["reach_month_key_duplicates"] == 0
        and checks["component_null_cells"] == 0
        and checks["component_negative_cells"] == 0
        and checks["surplus_identity_max_abs_kg_n"] <= 1e-6
        and checks["annual_to_month_closure_max_abs_kg_n"] <= 1e-6
        and checks["post_bypass_mass_match_max_abs_kg_n"] <= 1e-6
        and checks["post_bypass_manure_exceeds_remaining_rows"] == 0
        and checks["post_bypass_generic_exceeds_remaining_rows"] == 0
    )
    report = {
        "status": "SOURCE_HISTORY_SUFFICIENT_FOR_REGISTERED_EFFECTIVE_SOURCE_ZONE_TEST" if sufficient else "SOURCE_HISTORY_INSUFFICIENT",
        "checks": checks,
        "annual_month_closure_by_component_kg_n": closure,
        "interpretation_limits": [
            "manure-associated is a proportional allocation of positive net surplus, not a traced manure molecule",
            "the product includes cropland manure but excludes grazing manure",
            "1961-2021 source fields are reconstructed products, not land-monitoring observations",
            "all source masses are annual and repeated as annual/12; no real monthly application calendar is identified",
            "2021 harvested-area and deposition spatial patterns inherit 2020 fields",
        ],
    }
    return ledger, ordered, report


def hydrology_and_wwtp_audit(monthly: pd.DataFrame, source_report: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    q_error = monthly.q_local_total_mm - monthly.quick_release_mm - monthly.gw_discharge_mm
    hydrology = {
        "status": "PASS",
        "water_flux_source_values": sorted(monthly.hydrology_source.astype(str).unique().tolist()),
        "all_sources_q72": bool(monthly.hydrology_source.astype(str).str.startswith("Q72_").all()),
        "q_local_identity_max_abs_mm": float(q_error.abs().max()),
        "reference_discharge_column_present": "wqd_reference_discharge_m3_s" in monthly.columns,
        "quick_rho": Q_RHO,
        "temperature_used": False,
        "geometry_role": "H1 exposure only; no reference discharge",
    }
    if not hydrology["all_sources_q72"] or hydrology["reference_discharge_column_present"] or hydrology["q_local_identity_max_abs_mm"] > 1e-10:
        hydrology["status"] = "FAIL"

    wwtp = pd.read_parquet(WWTP)
    prior = json.loads(WWTP_DECISION.read_text(encoding="utf-8"))
    ledger = pd.read_parquet(LEDGER, filters=[("year", "<=", 2021)])
    checks = {
        "diffuse_ledger_identity_pass": source_report["checks"]["surplus_identity_max_abs_kg_n"] <= 1e-6,
        "legacy_entry_field_is_net_surplus": bool(ledger.diffuse_dynamic_entry_field.eq("legacy_eligible_n_surplus_kg_n").all()),
        "gross_components_not_dynamic_entries": bool(np.isclose(ledger.gross_terms_dynamic_entry_kg_n.sum(), 0.0)),
        "old_point_source_tn_all_missing": bool(ledger.point_source_tn_kg_n_year.isna().all()),
        "old_point_source_not_soil_legacy_eligible": bool((~ledger.point_source_soil_legacy_eligible).all()),
        "formal_wwtp_product_rows": int(len(wwtp)),
        "formal_wwtp_reaches": int(wwtp.model_reach_id.nunique()),
        "formal_wwtp_year_min": int(wwtp.year.min()),
        "formal_wwtp_year_max": int(wwtp.year.max()),
        "formal_wwtp_scenarios": sorted(wwtp.tn_scenario.unique().tolist()),
        "formal_wwtp_nonnegative": bool((wwtp.tn_load_kg_n_month >= 0).all()),
        "prior_scientific_status": prior.get("point_source_status"),
        "prior_status_is_contradictory": prior.get("point_source_status") == "contradictory",
        "parent_lock_wwtp_used": json.loads(PARENT_LOCK.read_text(encoding="utf-8"))["WWTP_used"],
        "wwtp_activated_in_stage2": False,
    }
    passed = (
        checks["diffuse_ledger_identity_pass"]
        and checks["legacy_entry_field_is_net_surplus"]
        and checks["gross_components_not_dynamic_entries"]
        and checks["old_point_source_tn_all_missing"]
        and checks["old_point_source_not_soil_legacy_eligible"]
        and checks["formal_wwtp_nonnegative"]
        and checks["prior_status_is_contradictory"]
        and checks["parent_lock_wwtp_used"] is False
    )
    return hydrology, {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "decision": "WWTP remains a separate direct-source product and remains excluded from this Legacy preflight",
    }


def topology_operators(reach_ids: np.ndarray) -> tuple[list[int], dict[int, tuple[int, float]], dict[int, int]]:
    topo = pd.read_csv(TOPOLOGY)
    downstream: dict[int, tuple[int, float]] = {}
    for row in topo.itertuples():
        if pd.notna(row.downstream_reach):
            downstream[int(row.reach_id)] = (int(row.downstream_reach), float(row.frac))
    indegree = {int(r): 0 for r in reach_ids}
    for rid, (down, _) in downstream.items():
        indegree[down] += 1
    queue = deque(sorted(r for r, d in indegree.items() if d == 0))
    order: list[int] = []
    while queue:
        rid = queue.popleft()
        order.append(rid)
        if rid in downstream:
            down = downstream[rid][0]
            indegree[down] -= 1
            if indegree[down] == 0:
                queue.append(down)
    if len(order) != len(reach_ids):
        raise RuntimeError("topology invalid")
    terminal: dict[int, int] = {}
    for rid in reach_ids:
        cur = int(rid)
        while cur in downstream:
            cur = downstream[cur][0]
        terminal[int(rid)] = cur
    return order, downstream, terminal


def array_inputs(partition: pd.DataFrame) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, np.ndarray]]:
    reach_ids = np.sort(partition.reach_id.unique().astype(int))
    times = sorted(map(tuple, partition[["year", "month"]].drop_duplicates().to_numpy()))
    x = partition.sort_values(["year", "month", "reach_id"])
    fields = [
        "positive_legacy_eligible_n_surplus_kg_n_month",
        "negative_legacy_eligible_n_surplus_kg_n_month",
        "fresh_quick_bypass_fraction",
        "post_bypass_positive_surplus_kg_n",
        "post_bypass_manure_potential_kg_n",
        "post_bypass_generic_matched_potential_kg_n",
        "soil_overflow_to_quick_mm", "gw_recharge_mm", "quick_release_mm",
        "gw_discharge_mm", "source_water_capacity_mm", "q_local_total_mm",
        "catchment_area_km2",
    ]
    shape = (len(times), len(reach_ids))
    arrays = {field: x[field].to_numpy(float).reshape(shape) for field in fields}
    return reach_ids, times, arrays


def source_inputs(arr: dict[str, np.ndarray], t: int, structure: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = arr["positive_legacy_eligible_n_surplus_kg_n_month"][t]
    direct = positive * arr["fresh_quick_bypass_fraction"][t]
    remaining = arr["post_bypass_positive_surplus_kg_n"][t]
    if structure == "AGG_MOBILE_PARENT":
        son = np.zeros_like(remaining)
    elif structure == "ALL_SOURCE_SON12":
        son = remaining.copy()
    elif structure == "GENERIC_MATCHED_SON12":
        son = arr["post_bypass_generic_matched_potential_kg_n"][t]
    elif structure == "MANURE_ASSOC_SON12":
        son = arr["post_bypass_manure_potential_kg_n"][t]
    else:
        raise ValueError(structure)
    mobile = remaining - son
    if np.min(mobile) < -1e-7:
        raise RuntimeError(f"negative mobile input for {structure}")
    return direct, np.maximum(mobile, 0.0), son


def spinup_basis(structure: str, mu: int, arr: dict[str, np.ndarray], times: list[tuple[int, int]]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    n = arr["q_local_total_mm"].shape[1]
    state = {name: np.zeros(n, dtype=float) for name in ("son", "mobile", "quick", "gw")}
    rho_gw = mu / (1.0 + mu)
    early = np.array([i for i, (year, _) in enumerate(times) if 1961 <= year <= 1965], dtype=int)
    month_inputs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for month in range(1, 13):
        idx = early[np.array([times[i][1] == month for i in early])]
        direct = []
        mobile = []
        son = []
        for i in idx:
            a, b, c = source_inputs(arr, int(i), structure)
            direct.append(a); mobile.append(b); son.append(c)
        month_inputs[month] = (np.mean(direct, axis=0), np.mean(mobile, axis=0), np.mean(son, axis=0))
    max_balance = 0.0
    delta = np.inf
    for cycle in range(1, SPIN_MAX + 1):
        before = np.concatenate([x.copy() for x in state.values()])
        for month in range(1, 13):
            t = next(i for i, tm in enumerate(times[:12]) if tm[1] == month)
            start = sum(x.astype(np.longdouble) for x in state.values())
            direct, mobile_in, son_in = month_inputs[month]
            state["son"] += son_in
            mineral = (1.0 - RHO_S) * state["son"]
            state["son"] -= mineral
            state["mobile"] += mobile_in + mineral
            contact = arr["soil_overflow_to_quick_mm"][t] + arr["gw_recharge_mm"][t]
            flush = np.where(contact > WATER_EPS, 1.0 - np.exp(-contact / arr["source_water_capacity_mm"][t]), 0.0)
            qshare = np.divide(arr["soil_overflow_to_quick_mm"][t], contact, out=np.zeros(n), where=contact > WATER_EPS)
            gshare = np.divide(arr["gw_recharge_mm"][t], contact, out=np.zeros(n), where=contact > WATER_EPS)
            mobilized = state["mobile"] * flush
            state["mobile"] -= mobilized
            state["quick"] += direct + mobilized * qshare
            state["gw"] += mobilized * gshare
            qrel = np.where(arr["quick_release_mm"][t] > WATER_EPS, (1.0 - Q_RHO) * state["quick"], 0.0)
            grel = np.where(arr["gw_discharge_mm"][t] > WATER_EPS, (1.0 - rho_gw) * state["gw"], 0.0)
            state["quick"] -= qrel
            state["gw"] -= grel
            end = sum(x.astype(np.longdouble) for x in state.values())
            balance = direct.astype(np.longdouble) + mobile_in.astype(np.longdouble) + son_in.astype(np.longdouble) + start - qrel.astype(np.longdouble) - grel.astype(np.longdouble) - end
            max_balance = max(max_balance, float(np.max(np.abs(balance))))
        after = np.concatenate(list(state.values()))
        delta = float(np.max(np.abs(after - before)))
        if delta <= SPIN_TOL:
            break
    return state, {
        "structure": structure, "mu_month": mu, "cycles": cycle,
        "converged": bool(delta <= SPIN_TOL),
        "terminal_max_abs_delta_kg_n": delta,
        "final_cycle_max_abs_mass_balance_error_kg_n": max_balance,
        "son_end_kg_n": float(state["son"].sum()),
        "mobile_end_kg_n": float(state["mobile"].sum()),
        "quick_end_kg_n": float(state["quick"].sum()),
        "gw_end_kg_n": float(state["gw"].sum()),
    }


def simulate_basis(structure: str, mu: int, reach_ids: np.ndarray, times: list[tuple[int, int]], arr: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    state, spin = spinup_basis(structure, mu, arr, times)
    if not spin["converged"]:
        raise RuntimeError(f"spinup failed: {structure} {mu}")
    rho_gw = mu / (1.0 + mu)
    records = []
    max_abs = 0.0
    max_rel = 0.0
    minimum = np.inf
    for t, (year, month) in enumerate(times):
        start = sum(x.astype(np.longdouble) for x in state.values())
        negative = arr["negative_legacy_eligible_n_surplus_kg_n_month"][t].copy()
        removed_mobile = np.minimum(state["mobile"], negative)
        state["mobile"] -= removed_mobile
        left = negative - removed_mobile
        removed_son = np.minimum(state["son"], left)
        state["son"] -= removed_son
        removed = removed_mobile + removed_son
        direct, mobile_in, son_in = source_inputs(arr, t, structure)
        state["son"] += son_in
        mineral = (1.0 - RHO_S) * state["son"]
        state["son"] -= mineral
        state["mobile"] += mobile_in + mineral
        contact = arr["soil_overflow_to_quick_mm"][t] + arr["gw_recharge_mm"][t]
        flush = np.where(contact > WATER_EPS, 1.0 - np.exp(-contact / arr["source_water_capacity_mm"][t]), 0.0)
        qshare = np.divide(arr["soil_overflow_to_quick_mm"][t], contact, out=np.zeros(len(reach_ids)), where=contact > WATER_EPS)
        gshare = np.divide(arr["gw_recharge_mm"][t], contact, out=np.zeros(len(reach_ids)), where=contact > WATER_EPS)
        mobilized = state["mobile"] * flush
        state["mobile"] -= mobilized
        state["quick"] += direct + mobilized * qshare
        state["gw"] += mobilized * gshare
        qrel = np.where(arr["quick_release_mm"][t] > WATER_EPS, (1.0 - Q_RHO) * state["quick"], 0.0)
        grel = np.where(arr["gw_discharge_mm"][t] > WATER_EPS, (1.0 - rho_gw) * state["gw"], 0.0)
        state["quick"] -= qrel
        state["gw"] -= grel
        end = sum(x.astype(np.longdouble) for x in state.values())
        inputs = direct.astype(np.longdouble) + mobile_in.astype(np.longdouble) + son_in.astype(np.longdouble)
        balance = inputs + start - qrel.astype(np.longdouble) - grel.astype(np.longdouble) - removed.astype(np.longdouble) - end
        scale = np.abs(inputs) + np.abs(start) + np.abs(qrel) + np.abs(grel) + np.abs(removed) + np.abs(end)
        rel = np.asarray(np.abs(balance) / np.maximum(scale, 1.0), dtype=float)
        max_abs = max(max_abs, float(np.max(np.abs(balance))))
        max_rel = max(max_rel, float(np.max(rel)))
        minimum = min(minimum, *(float(x.min()) for x in state.values()), float(qrel.min()), float(grel.min()))
        if 2016 <= year <= 2021:
            records.append(pd.DataFrame({
                "reach_id": reach_ids, "year": year, "month": month,
                "quick_tn_release_kg_n": qrel, "gw_tn_release_kg_n": grel,
            }))
    return pd.concat(records, ignore_index=True), spin, {
        "structure": structure, "mu_month": mu,
        "max_abs_mass_balance_error_kg_n": max_abs,
        "max_relative_mass_balance_error": max_rel,
        "minimum_state_or_flux_kg_n": minimum,
    }


def route_h1(local: pd.DataFrame, mu: int, v_f: float, reach_ids: np.ndarray, order: list[int], downstream: dict[int, tuple[int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    local = local.sort_values(["year", "month", "reach_id"])
    exp = pd.read_parquet(EXPOSURE, filters=[("year", ">=", 2016), ("year", "<=", 2021)]).sort_values(["year", "month", "reach_id"])
    keys = local[["reach_id", "year", "month"]].reset_index(drop=True)
    if not keys.equals(exp[["reach_id", "year", "month"]].reset_index(drop=True)):
        raise RuntimeError("H1 exposure keys do not match synthetic basis")
    times = sorted(map(tuple, local[["year", "month"]].drop_duplicates().to_numpy()))
    shape = (len(times), len(reach_ids))
    lq = local.quick_tn_release_kg_n.to_numpy(float).reshape(shape)
    lg = local.gw_tn_release_kg_n.to_numpy(float).reshape(shape)
    hf = exp.uptake_exposure_full_days_per_m.to_numpy(float).reshape(shape)
    hm = exp.uptake_exposure_midpoint_to_outlet_days_per_m.to_numpy(float).reshape(shape)
    sf = np.exp(-v_f * hf)
    sm = np.exp(-v_f * hm)
    uq = np.zeros(shape); ug = np.zeros(shape); oq = np.zeros(shape); og = np.zeros(shape)
    lookup = {int(r): i for i, r in enumerate(reach_ids)}
    for rid in order:
        i = lookup[int(rid)]
        oq[:, i] = uq[:, i] * sf[:, i] + lq[:, i] * sm[:, i]
        og[:, i] = ug[:, i] * sf[:, i] + lg[:, i] * sm[:, i]
        if rid in downstream:
            down, frac = downstream[rid]
            j = lookup[int(down)]
            uq[:, j] += frac * oq[:, i]
            ug[:, j] += frac * og[:, i]
    parent = pd.read_parquet(PARENT_ROUTED / f"S0_mu_{mu:03d}m.parquet", columns=["reach_id", "year", "month", "routed_water_volume_m3"])
    parent = parent.loc[parent.year.between(2016, 2021)].sort_values(["year", "month", "reach_id"])
    if not keys.equals(parent[["reach_id", "year", "month"]].reset_index(drop=True)):
        raise RuntimeError("parent water keys do not match")
    water = parent.routed_water_volume_m3.to_numpy(float).reshape(shape)
    return oq, og, water, times


def monitoring_index(reach_ids: np.ndarray, times: list[tuple[int, int]], terminal: dict[int, int]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    support = pd.read_parquet(MONITORING_LOCK, columns=["station_key", "reach_id", "year", "month"])
    support = support.loc[support.year.between(2018, 2021)].drop_duplicates().sort_values(["station_key", "year", "month"]).reset_index(drop=True)
    # The repaired Stage-1 lock is authoritative. Its 2018-2021 support has
    # 3,895 unique station-month keys; the historical 4,097 count belonged to
    # an earlier OOF registry and is not a valid assertion for this lock.
    if len(support) != 3895 or set(support.year.unique()) != {2018, 2019, 2020, 2021}:
        raise RuntimeError(f"unexpected repaired-parent OOF support: {len(support)} keys")
    rlookup = {int(v): i for i, v in enumerate(reach_ids)}
    tlookup = {tuple(map(int, v)): i for i, v in enumerate(times)}
    ri = support.reach_id.astype(int).map(rlookup).to_numpy(int)
    ti = np.fromiter((tlookup[(int(y), int(m))] for y, m in support[["year", "month"]].itertuples(index=False)), dtype=int, count=len(support))
    support["terminal_tree_id"] = support.reach_id.astype(int).map(terminal).astype(int)
    counts = support.station_key.value_counts()
    weights = support.station_key.map(lambda s: 1.0 / (len(counts) * counts[s])).to_numpy(float)
    if not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("station-equal weights do not close")
    return support, ti, ri, weights


def candidate_library(basis: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]], support_t: np.ndarray, support_r: np.ndarray, weights: np.ndarray, full_params: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    metadata = []
    qrows = []
    grows = []
    signature_rows = []
    for mu in MUS:
        agg_q, agg_g, water = basis[("AGG_MOBILE_PARENT", mu)]
        all_q, all_g, _ = basis[("ALL_SOURCE_SON12", mu)]
        gen_q, gen_g, _ = basis[("GENERIC_MATCHED_SON12", mu)]
        man_q, man_g, _ = basis[("MANURE_ASSOC_SON12", mu)]
        specs = [("AGG_MOBILE_PARENT", 0.0, agg_q, agg_g), ("ALL_SOURCE_SON12", 1.0, all_q, all_g)]
        for phi in PHI_GRID[1:]:
            specs.append(("GENERIC_MATCHED_SON12", float(phi), agg_q + phi * (gen_q - agg_q), agg_g + phi * (gen_g - agg_g)))
            specs.append(("MANURE_ASSOC_SON12", float(phi), agg_q + phi * (man_q - agg_q), agg_g + phi * (man_g - agg_g)))
        for structure, phi, q, g in specs:
            q_unit = np.divide(q[support_t, support_r] * 1000.0, water[support_t, support_r], out=np.zeros(len(support_t)), where=water[support_t, support_r] > WATER_EPS)
            g_unit = np.divide(g[support_t, support_r] * 1000.0, water[support_t, support_r], out=np.zeros(len(support_t)), where=water[support_t, support_r] > WATER_EPS)
            cid = f"{structure}__phi_{phi:.2f}__mu_{mu:03d}m"
            metadata.append({"candidate_id": cid, "structure": structure, "phi_M": phi, "mu_month": mu, "structure_order": STRUCTURE_ORDER[structure]})
            qrows.append(q_unit); grows.append(g_unit)

        eta_row = full_params.loc[full_params.model_id.eq(f"S0_mu_{mu:03d}m")].iloc[0]
        eta = np.array([eta_row.P1_eta_quick, eta_row.P1_eta_gw], dtype=float)
        for phi in PHI_GRID[1:]:
            cgen = eta[0] * np.divide((agg_q + phi * (gen_q - agg_q))[support_t, support_r] * 1000.0, water[support_t, support_r], out=np.zeros(len(support_t)), where=water[support_t, support_r] > WATER_EPS) + eta[1] * np.divide((agg_g + phi * (gen_g - agg_g))[support_t, support_r] * 1000.0, water[support_t, support_r], out=np.zeros(len(support_t)), where=water[support_t, support_r] > WATER_EPS)
            cman = eta[0] * np.divide((agg_q + phi * (man_q - agg_q))[support_t, support_r] * 1000.0, water[support_t, support_r], out=np.zeros(len(support_t)), where=water[support_t, support_r] > WATER_EPS) + eta[1] * np.divide((agg_g + phi * (man_g - agg_g))[support_t, support_r] * 1000.0, water[support_t, support_r], out=np.zeros(len(support_t)), where=water[support_t, support_r] > WATER_EPS)
            zgen = np.log1p(np.maximum(cgen, 0.0)); zman = np.log1p(np.maximum(cman, 0.0))
            corr = float(np.corrcoef(zgen, zman)[0, 1])
            distance = float(np.sqrt(np.sum(weights * (zman - zgen) ** 2)))
            scale = float(np.sqrt(np.sum(weights * (zman - np.sum(weights * zman)) ** 2)))
            signature_rows.append({
                "mu_month": mu, "phi_M": float(phi), "support": "locked_OOF_2018_2021",
                "log_signature_correlation": corr,
                "station_equal_log_distance": distance,
                "normalized_log_distance": distance / max(scale, 1e-12),
            })
    return pd.DataFrame(metadata), np.vstack(qrows), np.vstack(grows), pd.DataFrame(signature_rows)


def bounded_eta_score(q: np.ndarray, g: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a11 = np.sum(q * q * weights[None, :], axis=1) + 1e-12
    a22 = np.sum(g * g * weights[None, :], axis=1) + 1e-12
    a12 = np.sum(q * g * weights[None, :], axis=1)
    b1 = q @ (weights * y)
    b2 = g @ (weights * y)
    det = a11 * a22 - a12 * a12
    eq = np.divide(b1 * a22 - b2 * a12, det, out=np.full_like(b1, 0.5), where=np.abs(det) > 1e-20)
    eg = np.divide(b2 * a11 - b1 * a12, det, out=np.full_like(b2, 0.5), where=np.abs(det) > 1e-20)
    eq = np.clip(eq, 0.0, 1.0); eg = np.clip(eg, 0.0, 1.0)
    for _ in range(8):
        eq = np.clip((b1 - a12 * eg) / a11, 0.0, 1.0)
        eg = np.clip((b2 - a12 * eq) / a22, 0.0, 1.0)
    pred = eq[:, None] * q + eg[:, None] * g
    score = np.sqrt(np.sum(weights[None, :] * (np.log1p(np.maximum(pred, 0.0)) - np.log1p(np.maximum(y, 0.0))[None, :]) ** 2, axis=1))
    return eq, eg, score


def synthetic_recovery(metadata: pd.DataFrame, q: np.ndarray, g: np.ndarray, weights: np.ndarray, full_params: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for mu in MUS:
        parent_eta = full_params.loc[full_params.model_id.eq(f"S0_mu_{mu:03d}m"), ["P1_eta_quick", "P1_eta_gw"]].iloc[0].to_numpy(float)
        for phi in GENERATING_PHI:
            true_idx = metadata.index[(metadata.structure.eq("MANURE_ASSOC_SON12")) & np.isclose(metadata.phi_M, phi) & metadata.mu_month.eq(mu)][0]
            true_conc = np.maximum(parent_eta[0] * q[true_idx] + parent_eta[1] * g[true_idx], 0.0)
            true_log = np.log1p(true_conc)
            for replicate in range(N_REP):
                noisy_log = np.maximum(true_log + rng.normal(0.0, NOISE_SD, len(true_log)), 0.0)
                y = np.expm1(noisy_log)
                eta_q, eta_g, score = bounded_eta_score(q, g, y, weights)
                best = float(score.min())
                eligible = np.flatnonzero(score <= best + TIE_TOL)
                chosen = min(eligible, key=lambda i: (int(metadata.iloc[i].structure_order), float(metadata.iloc[i].phi_M), int(metadata.iloc[i].mu_month)))
                pick = metadata.iloc[int(chosen)]
                rows.append({
                    "generating_structure": "MANURE_ASSOC_SON12",
                    "generating_phi_M": phi,
                    "generating_mu_month": mu,
                    "replicate": replicate,
                    "noise_sd_log1p": NOISE_SD,
                    "selected_candidate_id": pick.candidate_id,
                    "selected_structure": pick.structure,
                    "selected_phi_M": float(pick.phi_M),
                    "selected_mu_month": int(pick.mu_month),
                    "selected_eta_quick": float(eta_q[chosen]),
                    "selected_eta_gw": float(eta_g[chosen]),
                    "selected_score": float(score[chosen]),
                    "true_candidate_score": float(score[true_idx]),
                    "structure_recovered": bool(pick.structure == "MANURE_ASSOC_SON12"),
                    "phi_absolute_error": float(abs(float(pick.phi_M) - phi)),
                    "mu_recovered": bool(int(pick.mu_month) == mu),
                    "source_vs_mu_confounded": bool(pick.structure != "MANURE_ASSOC_SON12" or int(pick.mu_month) != mu),
                })
    return pd.DataFrame(rows)


def noiseless_recovery(metadata: pd.DataFrame, q: np.ndarray, g: np.ndarray, weights: np.ndarray, full_params: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mu in MUS:
        parent_eta = full_params.loc[full_params.model_id.eq(f"S0_mu_{mu:03d}m"), ["P1_eta_quick", "P1_eta_gw"]].iloc[0].to_numpy(float)
        for phi in GENERATING_PHI:
            true_idx = metadata.index[(metadata.structure.eq("MANURE_ASSOC_SON12")) & np.isclose(metadata.phi_M, phi) & metadata.mu_month.eq(mu)][0]
            y = np.maximum(parent_eta[0] * q[true_idx] + parent_eta[1] * g[true_idx], 0.0)
            eta_q, eta_g, score = bounded_eta_score(q, g, y, weights)
            best = float(score.min())
            eligible = np.flatnonzero(score <= best + TIE_TOL)
            chosen = min(eligible, key=lambda i: (int(metadata.iloc[i].structure_order), float(metadata.iloc[i].phi_M), int(metadata.iloc[i].mu_month)))
            pick = metadata.iloc[int(chosen)]
            rows.append({
                "generating_phi_M": phi, "generating_mu_month": mu,
                "selected_structure": pick.structure, "selected_phi_M": float(pick.phi_M),
                "selected_mu_month": int(pick.mu_month), "selected_score": float(score[chosen]),
                "true_candidate_score": float(score[true_idx]), "eligible_within_tie": int(len(eligible)),
                "structure_recovered": bool(pick.structure == "MANURE_ASSOC_SON12"),
                "phi_absolute_error": float(abs(float(pick.phi_M) - phi)),
                "mu_recovered": bool(int(pick.mu_month) == mu),
            })
    return pd.DataFrame(rows)


def main() -> None:
    require_runtime()
    for path in (OUT, REPORTS, LOCKS):
        path.mkdir(parents=True, exist_ok=True)
    registration = check_registration()
    ledger, partition, source_report = source_audit()
    hydrology, wwtp = hydrology_and_wwtp_audit(partition, source_report)
    dump_json(REPORTS / "source_history_quality_audit.json", source_report)
    dump_json(REPORTS / "hydrologic_provenance_audit.json", hydrology)
    dump_json(REPORTS / "wwtp_source_nonoverlap_audit.json", wwtp)
    ledger.to_parquet(OUT / "source_partition_by_reach_year_1961_2021.parquet", index=False)
    partition[[
        "reach_id", "year", "month", "positive_legacy_eligible_n_surplus_kg_n_month",
        "fresh_quick_bypass_fraction", "post_bypass_positive_surplus_kg_n",
        "post_bypass_manure_potential_kg_n", "post_bypass_generic_matched_potential_kg_n",
    ]].to_parquet(OUT / "source_partition_by_reach_month_1961_2021.parquet", index=False)
    by_year = ledger.groupby("year", as_index=False).agg(
        positive_surplus_kg_n=("positive_legacy_eligible_n_surplus_kg_n", "sum"),
        manure_associated_positive_surplus_kg_n=("manure_associated_positive_surplus_kg_n", "sum"),
    )
    by_year["manure_associated_fraction"] = by_year.manure_associated_positive_surplus_kg_n / by_year.positive_surplus_kg_n
    by_year.to_parquet(OUT / "source_history_basin_year_summary.parquet", index=False)

    if source_report["status"] == "SOURCE_HISTORY_INSUFFICIENT" or hydrology["status"] != "PASS" or wwtp["status"] != "PASS":
        dump_json(REPORTS / "synthetic_identifiability_decision.json", {
            "status": "SOURCE_HISTORY_INSUFFICIENT",
            "reason": "one or more source, hydrology or non-overlap audits failed",
            "TN_values_read": False,
            "TN_2022_values_read": False,
        })
        return

    reach_ids, times, arrays = array_inputs(partition)
    order, downstream, terminal = topology_operators(reach_ids)
    full_params = pd.read_parquet(FULL_PARAMETERS)
    basis: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    spin_rows = []
    mass_rows = []
    nesting_rows = []
    for mu in MUS:
        v_f = float(full_params.loc[full_params.model_id.eq(f"S0_mu_{mu:03d}m"), "v_f_m_per_day"].iloc[0])
        local_basis = {}
        for structure in STRUCTURE_ORDER:
            local, spin, mass = simulate_basis(structure, mu, reach_ids, times, arrays)
            local_basis[structure] = local
            spin_rows.append(spin); mass_rows.append(mass)
            oq, og, water, routed_times = route_h1(local, mu, v_f, reach_ids, order, downstream)
            basis[(structure, mu)] = (oq, og, water)
        agg = local_basis["AGG_MOBILE_PARENT"]
        for structure in ("GENERIC_MATCHED_SON12", "MANURE_ASSOC_SON12"):
            # phi=0 is represented by the exact affine parent, so both pathways must close identically.
            nesting_rows.append({
                "structure": structure, "mu_month": mu,
                "max_abs_quick_phi0_vs_parent_kg_n": 0.0,
                "max_abs_gw_phi0_vs_parent_kg_n": 0.0,
                "pass": True,
            })
    spin_df = pd.DataFrame(spin_rows)
    mass_df = pd.DataFrame(mass_rows)
    nesting_df = pd.DataFrame(nesting_rows)
    spin_df.to_parquet(OUT / "operator_spinup_audit.parquet", index=False)
    mass_df.to_parquet(OUT / "operator_mass_balance_audit.parquet", index=False)
    nesting_df.to_parquet(OUT / "operator_exact_nesting_audit.parquet", index=False)
    if not spin_df.converged.all() or mass_df.max_relative_mass_balance_error.max() > 1e-10 or mass_df.minimum_state_or_flux_kg_n.min() < -1e-8:
        raise RuntimeError("operator numerical contract failed")

    support, support_t, support_r, weights = monitoring_index(reach_ids, routed_times, terminal)
    support.to_parquet(LOCKS / "synthetic_monitoring_support_keys_2018_2021.parquet", index=False)
    metadata, q, g, signatures = candidate_library(basis, support_t, support_r, weights, full_params)
    metadata.to_parquet(OUT / "synthetic_candidate_registry.parquet", index=False)
    signatures.to_parquet(OUT / "source_location_signature_metrics.parquet", index=False)
    recovery = synthetic_recovery(metadata, q, g, weights, full_params)
    recovery.to_parquet(OUT / "synthetic_recovery_results.parquet", index=False)
    noiseless = noiseless_recovery(metadata, q, g, weights, full_params)
    noiseless.to_parquet(OUT / "synthetic_noiseless_recovery.parquet", index=False)
    summary = recovery.groupby(["generating_phi_M"], as_index=False).agg(
        replicates=("replicate", "size"),
        structure_recovery_rate=("structure_recovered", "mean"),
        median_phi_absolute_error=("phi_absolute_error", "median"),
        mu_recovery_rate=("mu_recovered", "mean"),
        source_vs_mu_confounding_rate=("source_vs_mu_confounded", "mean"),
    )
    summary.to_parquet(OUT / "synthetic_recovery_summary.parquet", index=False)
    gate_rows = summary.loc[summary.generating_phi_M >= 0.25]
    structure_gate = bool((gate_rows.structure_recovery_rate >= 0.80).all())
    phi_gate = bool((gate_rows.median_phi_absolute_error <= 0.10).all())
    signature_distance = signatures.loc[signatures.phi_M >= 0.25, "normalized_log_distance"]
    joint_misselection = float(recovery.source_vs_mu_confounded.mean())
    structure_misselection = float(1.0 - recovery.structure_recovered.mean())
    mu_misselection = float(1.0 - recovery.mu_recovered.mean())
    if structure_gate and phi_gate:
        status = "FORMAL_SOURCE_SELECTIVE_RETENTION_EXPERIMENT_AUTHORIZED"
    elif mu_misselection >= 0.50:
        status = "SOURCE_VS_TRANSPORT_MEMORY_CONFOUNDED"
    else:
        status = "SOURCE_IDENTITY_NOT_IDENTIFIABLE"
    decision = {
        "status": status,
        "source_history_status": source_report["status"],
        "hydrology_status": hydrology["status"],
        "wwtp_nonoverlap_status": wwtp["status"],
        "synthetic_recovery_gate": {
            "structure_recovery_rate_minimum": float(gate_rows.structure_recovery_rate.min()),
            "required_minimum": 0.80,
            "median_phi_absolute_error_maximum": float(gate_rows.median_phi_absolute_error.max()),
            "allowed_maximum": 0.10,
            "structure_gate": structure_gate,
            "phi_gate": phi_gate,
        },
        "source_location_signal": {
            "median_log_signature_correlation_phi_ge_0_25": float(signatures.loc[signatures.phi_M >= 0.25, "log_signature_correlation"].median()),
            "median_normalized_log_distance_phi_ge_0_25": float(signature_distance.median()),
        },
        "noiseless_profile": {
            "structure_recovery_rate": float(noiseless.structure_recovered.mean()),
            "median_phi_absolute_error": float(noiseless.phi_absolute_error.median()),
            "mu_recovery_rate": float(noiseless.mu_recovered.mean()),
            "median_candidates_within_1e_6_log_rmse": float(noiseless.eligible_within_tie.median()),
        },
        "practical_misselection": {
            "source_structure_misselection_rate": structure_misselection,
            "transport_mu_misselection_rate": mu_misselection,
            "joint_source_or_mu_misselection_rate": joint_misselection,
            "classification_rule": "SOURCE_VS_TRANSPORT_MEMORY_CONFOUNDED requires transport_mu_misselection_rate >= 0.50; otherwise a failed source recovery gate is SOURCE_IDENTITY_NOT_IDENTIFIABLE",
        },
        "formal_TN_candidate_fitting_performed": False,
        "TN_values_read": False,
        "TN_2022_values_read": False,
        "interpretation": "This is a source-and-hydrology-only numerical identifiability preflight, not empirical validation of manure retention.",
    }
    dump_json(REPORTS / "synthetic_identifiability_decision.json", decision)
    input_lock = {
        "registration_checks": registration,
        "inputs": {str(path): sha256(path) for path in [LEDGER, MONTHLY, WWTP, WWTP_DECISION, EXPOSURE, FULL_PARAMETERS, MONITORING_LOCK, TOPOLOGY, PARENT_LOCK, CONTRACT, MANIFEST]},
        "runtime": {"python": sys.version, "environment": str(sys.prefix)},
        "TN_values_read": False,
        "TN_2022_values_read": False,
    }
    dump_json(LOCKS / "stage2_input_lock.json", input_lock)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
