from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW\5_Test\20260815_4")
P2 = Path(r"E:\SPARROW\5_Test\20260815_2")
P3 = Path(r"E:\SPARROW\5_Test\20260815_3")
P1 = Path(r"E:\SPARROW\5_Test\20260815_1")
MONTHLY_PATH = P2 / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
EARLY_PATH = P2 / "outputs" / "pre1961_early_n_mean_by_reach.parquet"
OBS_PATH = P1 / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLD_PATH = P1 / "outputs" / "tn_fold_registry.parquet"
TOPOLOGY_PATH = Path(r"E:\SPARROW\5_Test\20260814_1\inputs\topology\topology_edges.csv")
PARENT_AUDIT = P3 / "reports" / "completion_audit.json"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
Q_RHO = 0.25
B_RHO = 0.85
WATER_EPS = 1e-12
TAUS = [12, 36, 60, 96, 144, 240, 480]
N_BOOT = 10000
BOOT_SEED = 20260815
MARGIN = 0.01


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow conda environment required")
    if any(os.environ.get(k) != "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]):
        raise RuntimeError("thread limits must all equal 1")


def withdraw(pool: np.ndarray, demand: np.ndarray, active_cols: int) -> tuple[np.ndarray, np.ndarray]:
    view = pool[:, :active_cols]
    total = view.sum(axis=1)
    fraction = np.divide(np.minimum(demand, total), total, out=np.zeros_like(total), where=total > 0)
    removed = view * fraction[:, None]
    view -= removed
    return removed.sum(axis=1), demand - removed.sum(axis=1)


def prepare_arrays(monthly: pd.DataFrame) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, np.ndarray]]:
    reach_ids = np.array(sorted(monthly.reach_id.unique()), dtype=int)
    times = sorted(map(tuple, monthly[["year", "month"]].drop_duplicates().to_numpy()))
    ordered = monthly.sort_values(["year", "month", "reach_id"])
    fields = [
        "positive_legacy_eligible_n_surplus_kg_n_month",
        "negative_legacy_eligible_n_surplus_kg_n_month",
        "positive_input_mm",
        "quick_generated_mm",
        "quick_release_mm",
        "soil_overflow_to_quick_mm",
        "gw_recharge_mm",
        "gw_discharge_mm",
        "q_local_total_mm",
        "source_water_capacity_mm",
        "catchment_area_km2",
    ]
    arrays = {name: ordered[name].to_numpy(float).reshape(len(times), len(reach_ids)) for name in fields}
    return reach_ids, times, arrays


def water_partitions(arr: dict[str, np.ndarray], t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pin = arr["positive_input_mm"][t]
    qgen = arr["quick_generated_mm"][t]
    bypass = np.clip(np.divide(qgen, pin, out=np.zeros_like(pin), where=pin > WATER_EPS), 0.0, 1.0)
    overflow = arr["soil_overflow_to_quick_mm"][t]
    recharge = arr["gw_recharge_mm"][t]
    contact = overflow + recharge
    flush = np.where(contact > WATER_EPS, 1.0 - np.exp(-contact / arr["source_water_capacity_mm"][t]), 0.0)
    quick_share = np.divide(overflow, contact, out=np.zeros_like(contact), where=contact > WATER_EPS)
    base_share = np.divide(recharge, contact, out=np.zeros_like(contact), where=contact > WATER_EPS)
    return bypass, flush, quick_share, base_share


def spinup_totals(structure: str, tau: int | None, early_positive: np.ndarray, arr: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    n = len(early_positive)
    son = np.zeros(n)
    mobile = np.zeros(n)
    quick = np.zeros(n)
    base = np.zeros(n)
    rho_s = tau / (1.0 + tau) if tau is not None else 0.0
    delta = np.inf
    for cycle in range(1, 5001):
        starts = np.concatenate([son, mobile, quick, base])
        for t in range(12):
            bypass, flush, qshare, bshare = water_partitions(arr, t)
            direct = early_positive * bypass
            remaining = early_positive - direct
            if structure == "M0":
                source_release = remaining * flush
            elif structure == "S0":
                mobile += remaining
                source_release = mobile * flush
                mobile -= source_release
            else:
                son += remaining
                mineral = (1.0 - rho_s) * son
                son -= mineral
                mobile += mineral
                source_release = mobile * flush
                mobile -= source_release
            quick += direct + source_release * qshare
            base += source_release * bshare
            qrel = np.where(arr["quick_release_mm"][t] > WATER_EPS, (1.0 - Q_RHO) * quick, 0.0)
            brel = np.where(arr["gw_discharge_mm"][t] > WATER_EPS, (1.0 - B_RHO) * base, 0.0)
            quick -= qrel
            base -= brel
        ends = np.concatenate([son, mobile, quick, base])
        delta = float(np.max(np.abs(ends - starts)))
        if delta <= 1e-9:
            break
    return {"son": son, "mobile": mobile, "quick": quick, "base": base}, {"cycles": cycle, "terminal_max_abs_delta_kg_n": delta, "converged": delta <= 1e-9}


def simulate_candidate(
    model_id: str,
    structure: str,
    tau: int | None,
    reach_ids: np.ndarray,
    times: list[tuple[int, int]],
    arr: dict[str, np.ndarray],
    early_positive: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    n_t, n_r = len(times), len(reach_ids)
    n_c = n_t + 1
    totals, spin = spinup_totals(structure, tau, early_positive, arr)
    pools = {name: np.zeros((n_r, n_c), dtype=float) for name in ["son", "mobile", "quick", "base"]}
    for name in pools:
        pools[name][:, 0] = totals[name]
    out_fields = {name: np.zeros((n_t, n_r), dtype=float) for name in [
        "local_tn_release_kg_n", "local_tn_gt1y_kg_n", "local_tn_gt5y_kg_n", "local_tn_gt10y_kg_n",
        "quick_tn_release_kg_n", "base_tn_release_kg_n", "son_state_end_kg_n", "mobile_state_end_kg_n",
        "quick_state_end_kg_n", "base_state_end_kg_n", "negative_removed_kg_n", "negative_unmet_kg_n",
        "same_month_unmobilized_sink_kg_n", "mass_balance_error_kg_n", "mass_balance_relative_error"
    ]}
    rho_s = tau / (1.0 + tau) if tau is not None else 0.0
    max_cohort_closure = 0.0
    for t in range(n_t):
        active = t + 2
        starts = {
            name: np.sum(pools[name][:, :active], axis=1, dtype=np.longdouble)
            for name in pools
        }
        positive = arr["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        negative = arr["negative_legacy_eligible_n_surplus_kg_n_month"][t]
        bypass, flush, qshare, bshare = water_partitions(arr, t)
        direct = positive * bypass
        remaining = positive - direct
        negative_removed = np.zeros(n_r)
        negative_unmet = negative.copy()
        sink = np.zeros(n_r)

        if structure == "M0":
            source_release = np.zeros((n_r, active))
            source_release[:, t + 1] = remaining * flush
            sink = remaining * (1.0 - flush)
        elif structure == "S0":
            pools["mobile"][:, t + 1] += remaining
            removed, negative_unmet = withdraw(pools["mobile"], negative, active)
            negative_removed += removed
            source_release = pools["mobile"][:, :active] * flush[:, None]
            pools["mobile"][:, :active] -= source_release
        else:
            removed, left = withdraw(pools["mobile"], negative, active)
            negative_removed += removed
            removed_son, negative_unmet = withdraw(pools["son"], left, active)
            negative_removed += removed_son
            pools["son"][:, t + 1] += remaining
            mineral = pools["son"][:, :active] * (1.0 - rho_s)
            pools["son"][:, :active] -= mineral
            pools["mobile"][:, :active] += mineral
            source_release = pools["mobile"][:, :active] * flush[:, None]
            pools["mobile"][:, :active] -= source_release

        quick_input = source_release * qshare[:, None]
        base_input = source_release * bshare[:, None]
        pools["quick"][:, t + 1] += direct
        pools["quick"][:, :active] += quick_input
        pools["base"][:, :active] += base_input
        qrel = np.where(
            arr["quick_release_mm"][t, :, None] > WATER_EPS,
            (1.0 - Q_RHO) * pools["quick"][:, :active],
            0.0,
        )
        brel = np.where(
            arr["gw_discharge_mm"][t, :, None] > WATER_EPS,
            (1.0 - B_RHO) * pools["base"][:, :active],
            0.0,
        )
        pools["quick"][:, :active] -= qrel
        pools["base"][:, :active] -= brel
        release = qrel + brel
        age_month = np.concatenate(([10**9], t - np.arange(t + 1)))
        out_fields["local_tn_release_kg_n"][t] = release.sum(axis=1)
        out_fields["local_tn_gt1y_kg_n"][t] = release[:, age_month > 12].sum(axis=1)
        out_fields["local_tn_gt5y_kg_n"][t] = release[:, age_month > 60].sum(axis=1)
        out_fields["local_tn_gt10y_kg_n"][t] = release[:, age_month > 120].sum(axis=1)
        out_fields["quick_tn_release_kg_n"][t] = qrel.sum(axis=1)
        out_fields["base_tn_release_kg_n"][t] = brel.sum(axis=1)
        for name in pools:
            out_fields[f"{name}_state_end_kg_n"][t] = np.asarray(
                np.sum(pools[name][:, :active], axis=1, dtype=np.longdouble),
                dtype=float,
            )
        out_fields["negative_removed_kg_n"][t] = negative_removed
        out_fields["negative_unmet_kg_n"][t] = negative_unmet
        out_fields["same_month_unmobilized_sink_kg_n"][t] = sink
        ends = {
            name: np.sum(pools[name][:, :active], axis=1, dtype=np.longdouble)
            for name in pools
        }
        balance = (
            positive.astype(np.longdouble)
            + sum(starts.values())
            - np.sum(release, axis=1, dtype=np.longdouble)
            - sum(ends.values())
            - negative_removed.astype(np.longdouble)
            - sink.astype(np.longdouble)
        )
        out_fields["mass_balance_error_kg_n"][t] = np.asarray(balance, dtype=float)
        balance_scale = (
            np.abs(positive.astype(np.longdouble))
            + sum(np.abs(value) for value in starts.values())
            + np.abs(np.sum(release, axis=1, dtype=np.longdouble))
            + sum(np.abs(value) for value in ends.values())
            + np.abs(negative_removed.astype(np.longdouble))
            + np.abs(sink.astype(np.longdouble))
        )
        out_fields["mass_balance_relative_error"][t] = np.asarray(
            np.divide(np.abs(balance), np.maximum(balance_scale, 1.0)), dtype=float
        )
        max_cohort_closure = max(max_cohort_closure, float(np.max(np.abs(release.sum(axis=1) - qrel.sum(axis=1) - brel.sum(axis=1)))))

    years = np.repeat([y for y, _ in times], n_r)
    months = np.repeat([m for _, m in times], n_r)
    frame = pd.DataFrame({"reach_id": np.tile(reach_ids, n_t), "year": years, "month": months})
    for name, values in out_fields.items():
        frame[name] = values.reshape(-1)
    for name in ["q_local_total_mm", "catchment_area_km2", "quick_release_mm", "gw_discharge_mm"]:
        frame[name] = arr[name].reshape(-1)
    frame["model_id"] = model_id
    frame["source_structure"] = structure
    frame["soil_legacy_tau_month"] = np.nan if tau is None else tau
    audit = {
        "model_id": model_id,
        "spinup": spin,
        "max_abs_mass_balance_error_kg_n": float(np.max(np.abs(out_fields["mass_balance_error_kg_n"]))),
        "max_relative_mass_balance_error": float(np.max(out_fields["mass_balance_relative_error"])),
        "max_cohort_release_closure_error_kg_n": max_cohort_closure,
        "minimum_pool_mass_kg_n": float(min(pool.min() for pool in pools.values())),
    }
    return frame, pools, audit


def topology_operators(reach_ids: np.ndarray) -> tuple[list[int], dict[int, tuple[int, float]], dict[int, int]]:
    topo = pd.read_csv(TOPOLOGY_PATH)
    topo["reach_id"] = topo.reach_id.astype(int)
    downstream: dict[int, tuple[int, float]] = {}
    for row in topo.itertuples():
        if pd.notna(row.downstream_reach):
            downstream[int(row.reach_id)] = (int(row.downstream_reach), float(row.frac))
    # The source file's hydseq is not guaranteed to use one universal
    # orientation, so derive the order from the frozen directed edges.
    nodes = list(map(int, reach_ids))
    indegree = {rid: 0 for rid in nodes}
    for rid, (down, _) in downstream.items():
        if rid not in indegree or down not in indegree:
            raise RuntimeError("topology references a reach outside the 230-reach domain")
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
        raise RuntimeError("topology is cyclic or incomplete")
    terminal: dict[int, int] = {}
    for rid in reach_ids:
        cur = int(rid)
        seen: set[int] = set()
        while cur in downstream:
            if cur in seen:
                raise RuntimeError("topology cycle")
            seen.add(cur)
            cur = downstream[cur][0]
        terminal[int(rid)] = cur
    return order, downstream, terminal


def route_recent(frame: pd.DataFrame, reach_ids: np.ndarray, order: list[int], downstream: dict[int, tuple[int, float]], terminal: dict[int, int]) -> pd.DataFrame:
    recent = frame.loc[frame.year >= 2016].copy().sort_values(["year", "month", "reach_id"])
    ridx = {rid: i for i, rid in enumerate(reach_ids)}
    cols = ["local_tn_release_kg_n", "local_tn_gt1y_kg_n", "local_tn_gt5y_kg_n", "local_tn_gt10y_kg_n"]
    rows = []
    for (year, month), block in recent.groupby(["year", "month"], sort=True):
        b = block.set_index("reach_id").loc[reach_ids]
        routed = np.column_stack([b[c].to_numpy(float) for c in cols] + [(b.q_local_total_mm * b.catchment_area_km2 * 1000.0).to_numpy(float)])
        for rid in order:
            if rid in downstream:
                down, frac = downstream[rid]
                routed[ridx[down]] += routed[ridx[rid]] * frac
        item = pd.DataFrame(routed, columns=["routed_tn_kg_n", "routed_tn_gt1y_kg_n", "routed_tn_gt5y_kg_n", "routed_tn_gt10y_kg_n", "routed_water_volume_m3"])
        item.insert(0, "reach_id", reach_ids)
        item["year"] = int(year)
        item["month"] = int(month)
        item["terminal_tree_id"] = item.reach_id.map(terminal).astype(int)
        item["raw_tn_mg_l"] = np.divide(item.routed_tn_kg_n * 1000.0, item.routed_water_volume_m3, out=np.full(len(item), np.nan), where=item.routed_water_volume_m3.to_numpy() > 0)
        for label, mass in [("gt1y", "routed_tn_gt1y_kg_n"), ("gt5y", "routed_tn_gt5y_kg_n"), ("gt10y", "routed_tn_gt10y_kg_n")]:
            item[f"fraction_memory_{label}"] = np.divide(item[mass], item.routed_tn_kg_n, out=np.zeros(len(item)), where=item.routed_tn_kg_n.to_numpy() > 0)
        rows.append(item)
    out = pd.concat(rows, ignore_index=True)
    out["model_id"] = frame.model_id.iloc[0]
    return out


def fit_bias(train: pd.DataFrame, station_levels: list[str], ridge: float = 12.0) -> tuple[float, dict[str, float]]:
    raw = np.log1p(train.raw_tn_mg_l.to_numpy(float))
    y = np.log1p(train.tn_mg_l.to_numpy(float))
    residual = y - raw
    index = {s: i for i, s in enumerate(station_levels)}
    x = np.zeros((len(train), 1 + len(station_levels)), dtype=float)
    x[:, 0] = 1.0
    for i, s in enumerate(train.station_key.astype(str)):
        x[i, 1 + index[s]] = 1.0
    penalty = np.diag(np.r_[0.0, np.repeat(ridge, len(station_levels))])
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ residual)
    return float(beta[0]), {s: float(beta[1 + i]) for s, i in index.items()}


def oof_predictions(routed: pd.DataFrame, obs: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    merged = obs.merge(routed[["reach_id", "year", "month", "raw_tn_mg_l", "terminal_tree_id", "fraction_memory_gt1y", "fraction_memory_gt5y", "fraction_memory_gt10y"]], on=["reach_id", "year", "month"], validate="many_to_one")
    fold_defs = folds[["fold_id", "train_start_year", "train_end_year", "evaluation_year"]].drop_duplicates().sort_values("evaluation_year")
    outputs = []
    params = []
    for fd in fold_defs.itertuples():
        train = merged.loc[merged.year.between(fd.train_start_year, fd.train_end_year)].copy()
        test = merged.loc[merged.year.eq(fd.evaluation_year)].copy()
        stations = sorted(train.station_key.astype(str).unique())
        global_bias, effects = fit_bias(train, stations)
        station_effect = test.station_key.astype(str).map(effects).fillna(0.0).to_numpy(float)
        log_pred = np.log1p(test.raw_tn_mg_l.to_numpy(float)) + global_bias + station_effect
        test["pred_tn_mg_l"] = np.maximum(np.expm1(log_pred), 0.0)
        test["fold_id"] = fd.fold_id
        outputs.append(test)
        params.append({"fold_id": fd.fold_id, "global_log_bias": global_bias, "station_effect_count": len(effects), "ridge_lambda": 12.0})
    return pd.concat(outputs, ignore_index=True), params


def metrics(pred: pd.DataFrame) -> dict[str, float]:
    y = pred.tn_mg_l.to_numpy(float)
    p = pred.pred_tn_mg_l.to_numpy(float)
    ly = np.log1p(y)
    lp = np.log1p(p)
    annual = pred.assign(log_obs=ly, log_pred=lp).groupby(["station_key", "year"], as_index=False)[["log_obs", "log_pred"]].median()
    annual_rmse = float(np.sqrt(np.mean((annual.log_pred - annual.log_obs) ** 2)))
    changes = []
    for _, g in annual.groupby("station_key"):
        g = g.sort_values("year")
        if len(g) >= 2:
            changes.extend((np.diff(g.log_pred) - np.diff(g.log_obs)).tolist())
    return {
        "n": len(pred),
        "rmse_log1p": float(np.sqrt(np.mean((lp - ly) ** 2))),
        "rmse_raw_mg_l": float(np.sqrt(np.mean((p - y) ** 2))),
        "pbias_percent": float(100.0 * np.sum(p - y) / np.sum(y)),
        "annual_station_median_log_rmse": annual_rmse,
        "interannual_change_log_rmse": float(np.sqrt(np.mean(np.square(changes)))) if changes else np.nan,
        "correlation": float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan,
    }


def block_bootstrap(candidate: pd.DataFrame, reference: pd.DataFrame, block: str) -> np.ndarray:
    keys = ["station_key", "year", "month"]
    candidate_columns = list(dict.fromkeys(keys + [block, "tn_mg_l", "pred_tn_mg_l"]))
    c = candidate[candidate_columns].rename(columns={"pred_tn_mg_l": "pred_c"})
    r = reference[keys + ["pred_tn_mg_l"]].rename(columns={"pred_tn_mg_l": "pred_r"})
    paired = c.merge(r, on=keys, validate="one_to_one")
    paired["se_c"] = (np.log1p(paired.pred_c) - np.log1p(paired.tn_mg_l)) ** 2
    paired["se_r"] = (np.log1p(paired.pred_r) - np.log1p(paired.tn_mg_l)) ** 2
    stats = paired.groupby(block, as_index=False).agg(n=("se_c", "size"), se_c=("se_c", "sum"), se_r=("se_r", "sum"))
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(stats), size=(N_BOOT, len(stats)))
    n = stats.n.to_numpy(float)[idx].sum(axis=1)
    rc = np.sqrt(stats.se_c.to_numpy(float)[idx].sum(axis=1) / n)
    rr = np.sqrt(stats.se_r.to_numpy(float)[idx].sum(axis=1) / n)
    return rc - rr


def final_cohort_table(model_id: str, pools: dict[str, np.ndarray], reach_ids: np.ndarray, times: list[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for pool_name, values in pools.items():
        rr, cc = np.nonzero(values > 1e-14)
        input_year = np.where(cc == 0, -1, np.array([y for y, _ in times], dtype=int)[np.maximum(cc - 1, 0)])
        input_month = np.where(cc == 0, 0, np.array([m for _, m in times], dtype=int)[np.maximum(cc - 1, 0)])
        rows.append(pd.DataFrame({"model_id": model_id, "pool": pool_name, "reach_id": reach_ids[rr], "input_year": input_year, "input_month": input_month, "cohort_mass_kg_n": values[rr, cc], "cohort_origin": np.where(cc == 0, "pre1961_equilibrium", "observed_history_1961_2022")}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not json.loads(PARENT_AUDIT.read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("20260815_3 did not pass")
    parents = [MONTHLY_PATH, EARLY_PATH, OBS_PATH, FOLD_PATH, TOPOLOGY_PATH, PARENT_AUDIT]
    start_hashes = {str(p): sha256(p) for p in parents}
    dump(REPORTS / "parent_hashes_start.json", start_hashes)
    monthly = pd.read_parquet(MONTHLY_PATH)
    early = pd.read_parquet(EARLY_PATH)
    obs = pd.read_parquet(OBS_PATH)
    folds = pd.read_parquet(FOLD_PATH)
    reach_ids, times, arr = prepare_arrays(monthly)
    early_positive = early.set_index("reach_id").loc[reach_ids, "early_1961_1965_positive_surplus_mean_kg_n_year"].to_numpy(float) / 12.0
    order, downstream, terminal = topology_operators(reach_ids)
    terminal_count = len(set(terminal.values()))

    specs = [("M0", "M0", None), ("S0", "S0", None)] + [(f"S1_tau_{tau:03d}m", "S1", tau) for tau in TAUS]
    routed_frames = []
    oof_frames = []
    metric_rows = []
    readout_params: dict[str, object] = {}
    audits: dict[str, object] = {}
    for model_id, structure, tau in specs:
        local, _, audit = simulate_candidate(model_id, structure, tau, reach_ids, times, arr, early_positive)
        audits[model_id] = audit
        routed = route_recent(local, reach_ids, order, downstream, terminal)
        routed_frames.append(routed)
        pred, params = oof_predictions(routed, obs.loc[obs.year <= 2021], folds)
        pred["model_id"] = model_id
        oof_frames.append(pred)
        metric_rows.append({"model_id": model_id, **metrics(pred)})
        readout_params[model_id] = params
    routed_all = pd.concat(routed_frames, ignore_index=True)
    predictions = pd.concat(oof_frames, ignore_index=True)
    participating_terminal_count = int(predictions.terminal_tree_id.nunique())
    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse_log1p")

    pred_by_model = {m: g.copy() for m, g in predictions.groupby("model_id")}
    comparisons = [("S0", "M0")]
    comparisons += [(f"S1_tau_{tau:03d}m", "S0") for tau in TAUS]
    comparisons += [(f"S1_tau_{tau:03d}m", "M0") for tau in TAUS]
    boot_rows = []
    decisions: dict[str, object] = {}
    metric_map = metrics_df.set_index("model_id").to_dict("index")
    for candidate, reference in comparisons:
        cis = {}
        for block in ["station_key", "terminal_tree_id"]:
            values = block_bootstrap(pred_by_model[candidate], pred_by_model[reference], block)
            lower, upper = np.percentile(values, [2.5, 97.5])
            cis[block] = {"lower": float(lower), "upper": float(upper), "mean": float(values.mean())}
            boot_rows.append(pd.DataFrame({"candidate": candidate, "reference": reference, "block": block, "replicate": np.arange(N_BOOT), "delta_log_rmse": values}))
        process_improvements = {
            key: float(metric_map[reference][key] - metric_map[candidate][key])
            for key in ["annual_station_median_log_rmse", "interannual_change_log_rmse"]
        }
        noninferior = all(cis[b]["upper"] < MARGIN for b in cis)
        process_evidence = any(np.isfinite(v) and v > 1e-6 for v in process_improvements.values())
        decisions[f"{candidate}_vs_{reference}"] = {
            "candidate": candidate,
            "reference": reference,
            "ci": cis,
            "margin_log_rmse": MARGIN,
            "noninferior": noninferior,
            "process_improvements": process_improvements,
            "process_evidence": process_evidence,
            "structurally_supported": noninferior and process_evidence,
            "predictively_superior": all(cis[b]["upper"] < 0.0 for b in cis),
        }

    s0_supported = decisions["S0_vs_M0"]["structurally_supported"]
    eligible_s1 = []
    for tau in TAUS:
        mid = f"S1_tau_{tau:03d}m"
        if decisions[f"{mid}_vs_S0"]["structurally_supported"] and decisions[f"{mid}_vs_M0"]["noninferior"]:
            eligible_s1.append(mid)
    if eligible_s1:
        selected = min(eligible_s1, key=lambda m: (metric_map[m]["rmse_log1p"], int(m.split("_")[-1][:-1])))
        status = "S1_structurally_supported_noninferior"
    elif s0_supported:
        selected = "S0"
        status = "S0_structurally_supported_noninferior"
    else:
        selected = "M0"
        status = "source_memory_non_identifying_retain_simplest"

    selected_spec = next(s for s in specs if s[0] == selected)
    selected_local, selected_pools, selected_audit = simulate_candidate(*selected_spec, reach_ids, times, arr, early_positive)
    selected_local.to_parquet(OUT / "selected_source_model_reach_month_1961_2022.parquet", index=False)
    cohort_output = OUT / "selected_source_model_cohort_state_end_2022.parquet"
    cohort_frame = final_cohort_table(selected, selected_pools, reach_ids, times)
    pq.write_table(
        pa.Table.from_pandas(cohort_frame, preserve_index=False),
        cohort_output,
        compression="zstd",
        row_group_size=65536,
    )
    cohort_metadata = pq.read_metadata(cohort_output)
    if cohort_metadata.num_rows != len(cohort_frame):
        raise RuntimeError("cohort parquet footer/row count verification failed")
    routed_all.to_parquet(OUT / "candidate_routed_reach_month_2016_2022.parquet", index=False)
    predictions.to_parquet(OUT / "source_candidate_oof_predictions_2018_2021.parquet", index=False)
    metrics_df.to_csv(REPORTS / "source_candidate_metrics.csv", index=False)
    pd.concat(boot_rows, ignore_index=True).to_parquet(OUT / "source_structure_bootstrap_distributions.parquet", index=False)
    dump(REPORTS / "readout_parameters.json", readout_params)
    dump(REPORTS / "candidate_mass_and_cohort_audit.json", audits)
    decision = {
        "scenario_id": "20260815_4",
        "selected_source_model_id": selected,
        "source_structure_status": status,
        "eligible_S1": eligible_s1,
        "comparisons": decisions,
        "full_topology_terminal_tree_count": terminal_count,
        "tn_participating_terminal_tree_count": participating_terminal_count,
        "locked_2022_used_for_selection": False,
        "selected_model_audit": selected_audit,
    }
    dump(REPORTS / "source_structure_decision.json", decision)
    end_hashes = {str(p): sha256(p) for p in parents}
    dump(REPORTS / "parent_hashes_end.json", end_hashes)
    if start_hashes != end_hashes:
        raise RuntimeError("parent changed during stage 4")
    print(json.dumps({"selected": selected, "status": status, "eligible_S1": eligible_s1, "full_terminal_trees": terminal_count, "tn_participating_terminal_trees": participating_terminal_count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
