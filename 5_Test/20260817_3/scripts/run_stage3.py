from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW\5_Test\20260817_3")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S17_1 = Path(r"E:\SPARROW\5_Test\20260817_1")
S17_2 = Path(r"E:\SPARROW\5_Test\20260817_2")
S16_4 = Path(r"E:\SPARROW\5_Test\20260816_4")
sys.path.insert(0, str(S17_2 / "scripts"))
import legacy17_core as core  # noqa: E402

AGE_BINS = (
    ("young_0_1yr", 0, 12),
    ("age_gt1_to5yr", 13, 60),
    ("age_gt5_to10yr", 61, 120),
    ("age_gt10_to20yr", 121, 240),
    ("age_gt20_to50yr", 241, 600),
    ("age_gt50yr", 601, None),
)


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(sys.prefix)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 required")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class FrameWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None

    def write(self, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def withdraw(pool: np.ndarray, demand: np.ndarray, active: int) -> tuple[np.ndarray, np.ndarray]:
    view = pool[:, :active]
    total = view.sum(axis=1)
    fraction = np.divide(np.minimum(demand, total), total, out=np.zeros_like(total), where=total > 0)
    removed = view * fraction[:, None]
    view -= removed
    removed_total = removed.sum(axis=1)
    return removed_total, demand - removed_total


def cohort_history(model_id: str, reach_ids: np.ndarray, times: list[tuple[int, int]], arrays: dict[str, np.ndarray], early: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    parent = core.parent_core()
    structure, tau_s, mu_t = core.model_spec(model_id)
    initial, spinup = parent.spinup_candidate_totals(structure, tau_s, mu_t, early, arrays)
    n_t, n_r = len(times), len(reach_ids)
    pools = {name: np.zeros((n_r, n_t + 1), dtype=float) for name in ("son", "mobile", "quick", "gw")}
    for name in pools:
        pools[name][:, 0] = initial[name]
    labels = [row[0] for row in AGE_BINS]
    q_bins = {label: np.zeros((n_t, n_r), dtype=float) for label in labels}
    g_bins = {label: np.zeros((n_t, n_r), dtype=float) for label in labels}
    q_pre = np.zeros((n_t, n_r), dtype=float)
    g_pre = np.zeros((n_t, n_r), dtype=float)
    q_post_moment = np.zeros((n_t, n_r), dtype=float)
    g_post_moment = np.zeros((n_t, n_r), dtype=float)
    q_pre_lower_moment = np.zeros((n_t, n_r), dtype=float)
    g_pre_lower_moment = np.zeros((n_t, n_r), dtype=float)
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_gw = core.B_RHO if mu_t == 0 else mu_t / (1.0 + mu_t)
    max_relative_balance = 0.0
    min_pool = float("inf")
    for t in range(n_t):
        active = t + 2
        starts = {name: np.sum(pool[:, :active], axis=1, dtype=np.longdouble) for name, pool in pools.items()}
        positive = arrays["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        negative = arrays["negative_legacy_eligible_n_surplus_kg_n_month"][t]
        positive_water = arrays["positive_input_mm"][t]
        quick_generated = arrays["quick_generated_mm"][t]
        bypass = np.clip(np.divide(quick_generated, positive_water, out=np.zeros(n_r), where=positive_water > core.WATER_EPS), 0.0, 1.0)
        contact = arrays["soil_overflow_to_quick_mm"][t] + arrays["gw_recharge_mm"][t]
        flush = np.where(contact > core.WATER_EPS, 1.0 - np.exp(-contact / arrays["source_water_capacity_mm"][t]), 0.0)
        quick_share = np.divide(arrays["soil_overflow_to_quick_mm"][t], contact, out=np.zeros(n_r), where=contact > core.WATER_EPS)
        gw_share = np.divide(arrays["gw_recharge_mm"][t], contact, out=np.zeros(n_r), where=contact > core.WATER_EPS)
        direct = positive * bypass
        remaining = positive - direct
        removed = np.zeros(n_r)
        sink = np.zeros(n_r)
        if structure == "M0":
            source_release = np.zeros((n_r, active), dtype=float)
            source_release[:, t + 1] = remaining * flush
            sink = remaining - source_release[:, t + 1]
        elif structure == "S0":
            pools["mobile"][:, t + 1] += remaining
            removed, _ = withdraw(pools["mobile"], negative, active)
            source_release = pools["mobile"][:, :active] * flush[:, None]
            pools["mobile"][:, :active] -= source_release
        elif structure == "S1":
            removed_mobile, left = withdraw(pools["mobile"], negative, active)
            removed_son, _ = withdraw(pools["son"], left, active)
            removed = removed_mobile + removed_son
            pools["son"][:, t + 1] += remaining
            mineralized = pools["son"][:, :active] * (1.0 - rho_s)
            pools["son"][:, :active] -= mineralized
            pools["mobile"][:, :active] += mineralized
            source_release = pools["mobile"][:, :active] * flush[:, None]
            pools["mobile"][:, :active] -= source_release
        else:
            raise ValueError(structure)
        pools["quick"][:, t + 1] += direct
        pools["quick"][:, :active] += source_release * quick_share[:, None]
        pools["gw"][:, :active] += source_release * gw_share[:, None]
        q = np.where(arrays["quick_release_mm"][t, :, None] > core.WATER_EPS, (1.0 - core.Q_RHO) * pools["quick"][:, :active], 0.0)
        g = np.where(arrays["gw_discharge_mm"][t, :, None] > core.WATER_EPS, (1.0 - rho_gw) * pools["gw"][:, :active], 0.0)
        pools["quick"][:, :active] -= q
        pools["gw"][:, :active] -= g
        ages = np.concatenate(([-1], t - np.arange(t + 1)))
        q_pre[t] = q[:, 0]
        g_pre[t] = g[:, 0]
        post_ages = ages[1:]
        for label, lower, upper in AGE_BINS:
            mask = post_ages >= lower if upper is None else ((post_ages >= lower) & (post_ages <= upper))
            q_bins[label][t] = q[:, 1:][:, mask].sum(axis=1)
            g_bins[label][t] = g[:, 1:][:, mask].sum(axis=1)
        q_post_moment[t] = (q[:, 1:] * post_ages[None, :]).sum(axis=1)
        g_post_moment[t] = (g[:, 1:] * post_ages[None, :]).sum(axis=1)
        q_pre_lower_moment[t] = q[:, 0] * (t + 1)
        g_pre_lower_moment[t] = g[:, 0] * (t + 1)
        release_total = q.sum(axis=1) + g.sum(axis=1)
        ends = {name: np.sum(pool[:, :active], axis=1, dtype=np.longdouble) for name, pool in pools.items()}
        balance = positive.astype(np.longdouble) + sum(starts.values()) - release_total.astype(np.longdouble) - sum(ends.values()) - removed.astype(np.longdouble) - sink.astype(np.longdouble)
        scale = np.abs(positive.astype(np.longdouble)) + sum(np.abs(v) for v in starts.values()) + np.abs(release_total.astype(np.longdouble)) + sum(np.abs(v) for v in ends.values()) + np.abs(removed.astype(np.longdouble)) + np.abs(sink.astype(np.longdouble))
        max_relative_balance = max(max_relative_balance, float(np.max(np.abs(balance) / np.maximum(scale, 1.0))))
        min_pool = min(min_pool, min(float(pool[:, :active].min()) for pool in pools.values()))
    values: dict[str, np.ndarray] = {}
    for label in labels:
        values[f"quick_{label}"] = q_bins[label]
        values[f"gw_{label}"] = g_bins[label]
    values.update({
        "quick_pre1961": q_pre,
        "gw_pre1961": g_pre,
        "quick_post1961_age_moment": q_post_moment,
        "gw_post1961_age_moment": g_post_moment,
        "quick_pre1961_age_lower_bound_moment": q_pre_lower_moment,
        "gw_pre1961_age_lower_bound_moment": g_pre_lower_moment,
    })
    audit = {
        "model_id": model_id,
        "source_structure": structure,
        "soil_tau_month": tau_s,
        "delivery_mu_month": mu_t,
        "spinup": spinup,
        "max_relative_mass_balance_error": max_relative_balance,
        "minimum_pool_mass_kg_n": min_pool,
    }
    return values, audit


def route_values(values: dict[str, np.ndarray], reach_ids: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, dict[int, int]]:
    routed: dict[str, np.ndarray] = {}
    terminals = np.array([], dtype=int)
    terminal_map: dict[int, int] = {}
    for name, array in values.items():
        routed[name], terminals, terminal_map = core.route_matrix(array, reach_ids)
    return routed, terminals, terminal_map


def fit_development_readout(model_id: str, routed: dict[str, np.ndarray], reach_ids: np.ndarray, times: list[tuple[int, int]], routed_water: np.ndarray, terminal_map: dict[int, int], stage4: object, observations: pd.DataFrame) -> tuple[np.ndarray, dict[str, float], dict[str, object]]:
    labels = [row[0] for row in AGE_BINS]
    quick = sum(routed[f"quick_{label}"] for label in labels) + routed["quick_pre1961"]
    gw = sum(routed[f"gw_{label}"] for label in labels) + routed["gw_pre1961"]
    n_t, n_r = quick.shape
    raw = pd.DataFrame({
        "reach_id": np.tile(reach_ids, n_t),
        "year": np.repeat([x[0] for x in times], n_r),
        "month": np.repeat([x[1] for x in times], n_r),
        "routed_quick_tn_kg_n": quick.reshape(-1),
        "routed_gw_tn_kg_n": gw.reshape(-1),
        "routed_water_volume_m3": routed_water.reshape(-1),
    })
    raw["terminal_tree_id"] = raw.reach_id.map(terminal_map).astype(int)
    joined = observations.merge(raw, on=["reach_id", "year", "month"], validate="many_to_one")
    train = joined.loc[joined.year.between(2016, 2021)].copy()
    if train.year.min() != 2016 or train.year.max() != 2021 or (train.year == 2022).any():
        raise RuntimeError("development readout boundary violation")
    eta, effects, diagnostic = stage4.fit_eta(train)
    diagnostic.update({
        "model_id": model_id,
        "fit_support_years": [2016, 2021],
        "locked_2022_rows_used_for_fit": 0,
        "eta_boundary_confounding_role": "diagnostic_only_not_hard_gate",
    })
    return eta, effects, diagnostic


def output_frame(model_id: str, routed: dict[str, np.ndarray], eta: np.ndarray, reach_ids: np.ndarray, times: list[tuple[int, int]], terminal_map: dict[int, int]) -> pd.DataFrame:
    labels = [row[0] for row in AGE_BINS]
    n_t, n_r = routed["quick_pre1961"].shape
    frame = pd.DataFrame({
        "model_id": model_id,
        "reach_id": np.tile(reach_ids, n_t),
        "year": np.repeat([x[0] for x in times], n_r),
        "month": np.repeat([x[1] for x in times], n_r),
    })
    frame["terminal_tree_id"] = frame.reach_id.map(terminal_map).astype(int)
    structural_total = np.zeros((n_t, n_r), dtype=float)
    weighted_total = np.zeros((n_t, n_r), dtype=float)
    for label in labels + ["pre1961"]:
        q = routed[f"quick_{label}"]
        g = routed[f"gw_{label}"]
        structural = q + g
        weighted = eta[0] * q + eta[1] * g
        frame[f"structural_routed_{label}_kg_n"] = structural.reshape(-1)
        frame[f"eta_weighted_{label}_kg_n"] = weighted.reshape(-1)
        structural_total += structural
        weighted_total += weighted
    structural_post_moment = routed["quick_post1961_age_moment"] + routed["gw_post1961_age_moment"]
    weighted_post_moment = eta[0] * routed["quick_post1961_age_moment"] + eta[1] * routed["gw_post1961_age_moment"]
    structural_pre_lower = routed["quick_pre1961_age_lower_bound_moment"] + routed["gw_pre1961_age_lower_bound_moment"]
    weighted_pre_lower = eta[0] * routed["quick_pre1961_age_lower_bound_moment"] + eta[1] * routed["gw_pre1961_age_lower_bound_moment"]
    frame["structural_routed_total_kg_n"] = structural_total.reshape(-1)
    frame["eta_weighted_total_kg_n"] = weighted_total.reshape(-1)
    frame["structural_routed_post1961_age_moment_month_kg_n"] = structural_post_moment.reshape(-1)
    frame["eta_weighted_post1961_age_moment_month_kg_n"] = weighted_post_moment.reshape(-1)
    frame["structural_routed_pre1961_age_lower_bound_moment_month_kg_n"] = structural_pre_lower.reshape(-1)
    frame["eta_weighted_pre1961_age_lower_bound_moment_month_kg_n"] = weighted_pre_lower.reshape(-1)
    frame["eta_quick"] = float(eta[0])
    frame["eta_gw"] = float(eta[1])
    return frame


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    labels = [row[0] for row in AGE_BINS] + ["pre1961"]
    terminal = frame.loc[frame.reach_id.eq(frame.terminal_tree_id) & frame.year.between(2016, 2022)]
    rows = []
    for (model_id, year), block in terminal.groupby(["model_id", "year"], sort=True):
        for semantics, prefix, total_column, moment_column in (
            ("structural_routed_n_age", "structural_routed", "structural_routed_total_kg_n", "structural_routed_post1961_age_moment_month_kg_n"),
            ("eta_weighted_predicted_n_age", "eta_weighted", "eta_weighted_total_kg_n", "eta_weighted_post1961_age_moment_month_kg_n"),
        ):
            total = float(block[total_column].sum())
            item = {"model_id": model_id, "year": int(year), "age_semantics": semantics, "total_mass_kg_n": total}
            for label in labels:
                mass = float(block[f"{prefix}_{label}_kg_n"].sum())
                item[f"{label}_fraction"] = mass / total if total > 0 else np.nan
            post = total - float(block[f"{prefix}_pre1961_kg_n"].sum())
            item["post1961_mean_age_month"] = float(block[moment_column].sum() / post) if post > 0 else np.nan
            item["pre1961_age_status"] = "right_censored_origin_before_1961"
            rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not json.loads((S17_2 / "reports" / "completion_audit.json").read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("stage2 incomplete")
    protected = [
        S17_1 / "outputs" / "analysis_model_registry_16.parquet",
        S17_2 / "stage_lock.json",
        core.MONTHLY,
        core.PARENT_CORE,
        S16_4 / "scripts" / "run_stage4.py",
    ]
    start_hash = {str(path): sha256(path) for path in protected}
    dump(ROOT / "upstream_manifest.json", start_hash)
    registry = pd.read_parquet(S17_1 / "outputs" / "analysis_model_registry_16.parquet")
    model_ids = registry.model_id.astype(str).tolist()
    parent = core.parent_core()
    reach_ids, times, arrays, early = parent.prepare_model_arrays()
    stage4 = core.load_module(S16_4 / "scripts" / "run_stage4.py", "legacy16_stage4_for_17_3")
    observations = pd.read_parquet(parent.OBS_PATH)
    local_water = arrays["q_local_total_mm"] * arrays["catchment_area_km2"] * 1000.0
    routed_water, _, terminal_map = core.route_matrix(local_water, reach_ids)
    writer = FrameWriter(OUT / "historical_routed_n_age_1961_2022.parquet")
    audits = []
    eta_rows = []
    effect_rows = []
    summary_frames = []
    for index, model_id in enumerate(model_ids, start=1):
        values, audit = cohort_history(model_id, reach_ids, times, arrays, early)
        routed, terminals, terminal_map = route_values(values, reach_ids)
        eta, effects, diagnostic = fit_development_readout(model_id, routed, reach_ids, times, routed_water, terminal_map, stage4, observations)
        frame = output_frame(model_id, routed, eta, reach_ids, times, terminal_map)
        writer.write(frame)
        summary_frames.append(summary(frame))
        scalar = core.simulate_history_replay(model_id)
        labels = [row[0] for row in AGE_BINS]
        local_quick = sum(values[f"quick_{label}"] for label in labels) + values["quick_pre1961"]
        local_gw = sum(values[f"gw_{label}"] for label in labels) + values["gw_pre1961"]
        q_diff = float(np.max(np.abs(local_quick.reshape(-1) - scalar.quick_tn_release_kg_n.to_numpy(float))))
        g_diff = float(np.max(np.abs(local_gw.reshape(-1) - scalar.gw_tn_release_kg_n.to_numpy(float))))
        audit.update({
            "cohort_vs_scalar_quick_max_abs_kg_n": q_diff,
            "cohort_vs_scalar_gw_max_abs_kg_n": g_diff,
            "cohort_vs_scalar_pass": bool(
                np.allclose(local_quick.reshape(-1), scalar.quick_tn_release_kg_n.to_numpy(float), rtol=1e-12, atol=1e-6)
                and np.allclose(local_gw.reshape(-1), scalar.gw_tn_release_kg_n.to_numpy(float), rtol=1e-12, atol=1e-6)
            ),
        })
        audits.append(audit)
        eta_rows.append(diagnostic)
        effect_rows.extend({"model_id": model_id, "station_key": station, "station_log_effect": value, "fit_support_start_year": 2016, "fit_support_end_year": 2021} for station, value in effects.items())
        print(f"cohort {index}/{len(model_ids)} {model_id}", flush=True)
    writer.close()
    pd.concat(summary_frames, ignore_index=True).to_parquet(OUT / "terminal_historical_n_age_summary_2016_2022.parquet", index=False)
    pd.DataFrame(eta_rows).to_parquet(OUT / "development_eta_parameters.parquet", index=False)
    pd.DataFrame(effect_rows).to_parquet(OUT / "development_station_effects.parquet", index=False)
    dump(REPORTS / "cohort_mass_balance_and_reproduction_audit.json", {
        "pass": bool(all(row["cohort_vs_scalar_pass"] and row["max_relative_mass_balance_error"] <= 1e-12 and row["minimum_pool_mass_kg_n"] >= -1e-9 for row in audits)),
        "models": audits,
    })
    dump(REPORTS / "age_bin_contract.json", {
        "unit": "month",
        "bins": [
            {"field": "young_0_1yr", "logic": "age_month <= 12", "lower_inclusive": 0, "upper_inclusive": 12},
            {"field": "age_gt1_to5yr", "logic": "12 < age_month <= 60", "lower_exclusive": 12, "upper_inclusive": 60},
            {"field": "age_gt5_to10yr", "logic": "60 < age_month <= 120", "lower_exclusive": 60, "upper_inclusive": 120},
            {"field": "age_gt10_to20yr", "logic": "120 < age_month <= 240", "lower_exclusive": 120, "upper_inclusive": 240},
            {"field": "age_gt20_to50yr", "logic": "240 < age_month <= 600", "lower_exclusive": 240, "upper_inclusive": 600},
            {"field": "age_gt50yr", "logic": "age_month > 600", "lower_exclusive": 600, "upper_inclusive": None},
        ],
        "coverage": "complete_nonoverlapping_for_post1961_monthly_cohorts",
        "pre1961": "separate right-censored cohort; not assigned an exact age bin",
        "van_meter_compatible_labels": "display_only; all calculations use registered month cutpoints",
    })
    dump(REPORTS / "locked_2022_no_refit_audit.json", {
        "pass": True,
        "fit_calls": len(model_ids),
        "all_fit_support": "2016-2021",
        "locked_2022_rows_used_for_fit": 0,
        "eta_and_station_effects_frozen_for_stage4_2022_prediction": True,
    })
    if not json.loads((REPORTS / "cohort_mass_balance_and_reproduction_audit.json").read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("cohort audit failed")
    end_hash = {str(path): sha256(path) for path in protected}
    if start_hash != end_hash:
        raise RuntimeError("protected input changed")
    completion = {
        "scenario_id": "20260817_3",
        "pass": True,
        "models": len(model_ids),
        "reaches": len(reach_ids),
        "months": len(times),
        "terminal_trees": len(set(terminal_map.values())),
        "age_semantics": ["structural_routed_n_age", "eta_weighted_predicted_n_age"],
        "pre1961_separate": True,
        "locked_2022_refit": False,
        "eta_boundary_confounding_hard_gate": False,
    }
    dump(REPORTS / "completion_audit.json", completion)
    dump(ROOT / "stage_lock.json", {"status": "complete", "scenario_id": "20260817_3", "completion_sha256": sha256(REPORTS / "completion_audit.json")})
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
