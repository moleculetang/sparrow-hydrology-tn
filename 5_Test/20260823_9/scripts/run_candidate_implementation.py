from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT.parent
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
STAGE2 = TEST / "20260823_2" / "scripts" / "run_corrected_baseline.py"
S000 = TEST / "20260813_52" / "scenarios" / "S000" / "inputs" / "parent_indata.parquet"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
GLHYMPS = TEST / "20260814_9" / "inputs" / "model_ready" / "static" / "soil_tn_glhymps_by_reach.parquet"
EPS = 1.0e-12
SEED = 20260823
BASE_V = float(logit((1.0 - 0.25) / 3.75))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def topology_operators(reaches: list[int]) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    topo = pd.read_csv(TOPOLOGY, encoding="utf-8-sig")
    index = {rid: i for i, rid in enumerate(reaches)}
    upstream_immediate: dict[int, list[int]] = {}
    downstream: dict[int, int | None] = {}
    for row in topo.itertuples(index=False):
        rid = int(row.reach_id)
        tokens = []
        if pd.notna(row.upstream_reaches) and str(row.upstream_reaches).strip():
            tokens = [int(float(v.strip())) for v in str(row.upstream_reaches).replace(";", ",").split(",") if v.strip()]
        upstream_immediate[rid] = [v for v in tokens if v in index]
        value = None
        if pd.notna(row.downstream_reach) and str(row.downstream_reach).strip():
            candidate = int(float(str(row.downstream_reach).split(",")[0]))
            value = candidate if candidate in index else None
        downstream[rid] = value
    aggregate = np.zeros((len(reaches), len(reaches)), float)
    for source in reaches:
        current = source
        seen = set()
        while current is not None:
            if current in seen:
                raise RuntimeError(f"Topology cycle at {source}")
            seen.add(current)
            aggregate[index[current], index[source]] = 1.0
            current = downstream[current]
    terminals = {}
    for rid in reaches:
        current = rid
        while downstream[current] is not None:
            current = int(downstream[current])
        terminals[rid] = current
    return aggregate, index, terminals


def build_states() -> tuple[pd.DataFrame, np.ndarray, dict[int, int], dict[int, int]]:
    core = load_file("stage9_stage2", STAGE2)
    module = core.load_component("stage9_s000", highflow_scale=1.0)
    module.INPUT_PATH = S000
    module.TOPOLOGY_PATH = TOPOLOGY
    forcing = module.load_forcing_panel()
    states = module.add_hydrologic_features(
        forcing,
        rho=core.BASE["rho"], wm=core.BASE["wm"], et_gamma=core.BASE["et_gamma"],
        sas_rho=core.BASE["sas_rho"], young_k=core.BASE["young_k"],
        storage_scale=core.BASE["storage_scale"], prod_capacity=core.BASE["prod_capacity"],
        runoff_gamma=core.BASE["runoff_gamma"], quick_rho=core.BASE["quick_rho"],
        base_rho=core.BASE["base_rho"], base_release=core.BASE["base_release"],
        et_state_operator_mode="baseline_clip",
    )
    states = states.sort_values(["year", "month", "comid"]).reset_index(drop=True)
    reaches = sorted(states["comid"].astype(int).unique())
    if len(reaches) != 230 or len(states) != 46920:
        raise RuntimeError("Full 230 x 204 state panel required")
    aggregate, index, terminals = topology_operators(reaches)
    n_time = states[["year", "month"]].drop_duplicates().shape[0]
    quick_agg = states["routed_quick_cfs"].to_numpy(float).reshape(n_time, len(reaches))
    slow_agg = states["routed_base_cfs"].to_numpy(float).reshape(n_time, len(reaches))
    inverse = np.linalg.inv(aggregate)
    quick_local = quick_agg @ inverse.T
    slow_local = slow_agg @ inverse.T
    scale = max(float(np.nanmax(quick_agg + slow_agg)), 1.0)
    if np.nanmin(quick_local) < -1e-9 * scale or np.nanmin(slow_local) < -1e-9 * scale:
        raise RuntimeError("Local flux inversion produced material negative flow")
    quick_local = np.clip(quick_local, 0.0, None)
    slow_local = np.clip(slow_local, 0.0, None)
    states["local_quick_cfs_reconstructed"] = quick_local.reshape(-1)
    states["local_slow_cfs_reconstructed"] = slow_local.reshape(-1)
    states["terminal_tree"] = states["comid"].astype(int).map(terminals).astype(int)
    return states, aggregate, index, terminals


def static_pcs(states: pd.DataFrame, training_end: int = 2018) -> tuple[pd.DataFrame, dict[str, object]]:
    base_cols = ["comid", "IncAreaKm2", "CumAreaKm2", "LENGTHKM", "SLOPE", "MaxElSmoCm", "L_to_outlet_km"]
    base = pd.read_parquet(S000)[base_cols].drop_duplicates("comid").rename(columns={"comid": "reach_id"})
    glh = pd.read_parquet(GLHYMPS)[["reach_id", "glhymps_log10_permeability_m2", "glhymps_porosity"]]
    climate = states[states["year"].le(training_end)].groupby("comid").agg(
        mean_ppt=("PPT", "mean"), mean_pet=("PET", "mean"), mean_aet=("AET", "mean"),
        ppt_sd=("PPT", "std")
    ).reset_index().rename(columns={"comid": "reach_id"})
    climate["aridity_clim"] = climate["mean_pet"] / climate["mean_ppt"].clip(lower=EPS)
    climate["ppt_cv"] = climate["ppt_sd"] / climate["mean_ppt"].clip(lower=EPS)
    table = base.merge(glh, on="reach_id", how="left", validate="one_to_one").merge(climate, on="reach_id", validate="one_to_one")
    table["log_inc_area"] = np.log1p(table["IncAreaKm2"].clip(lower=0))
    table["log_cum_area"] = np.log1p(table["CumAreaKm2"].clip(lower=0))
    table["log_length"] = np.log1p(table["LENGTHKM"].clip(lower=0))
    table["log_outlet_distance"] = np.log1p(table["L_to_outlet_km"].clip(lower=0))
    table["log_slope"] = np.log1p(table["SLOPE"].clip(lower=0))
    attrs = ["log_inc_area", "log_cum_area", "log_length", "log_outlet_distance", "log_slope", "MaxElSmoCm", "glhymps_log10_permeability_m2", "glhymps_porosity", "aridity_clim", "ppt_cv"]
    x = table[attrs].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    med = x.median()
    x = x.fillna(med).fillna(0.0)
    mean, std = x.mean(), x.std(ddof=0).replace(0, 1.0)
    z = ((x - mean) / std).to_numpy(float)
    if not np.isfinite(z).all():
        raise RuntimeError("Static regionalization attributes remain non-finite after imputation")
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    pcs = z @ vt[:4].T
    for i in range(4):
        table[f"static_pc{i+1}"] = pcs[:, i]
    audit = {
        "attributes": attrs,
        "components": 4,
        "explained_variance_fraction": (singular[:4] ** 2 / np.sum(singular ** 2)).tolist(),
        "training_end": training_end,
        "missing_after_imputation": int(x.isna().sum().sum()),
    }
    return table[["reach_id", *[f"static_pc{i+1}" for i in range(4)]]], audit


def prepare_arrays(states: pd.DataFrame, aggregate: np.ndarray, pcs: pd.DataFrame):
    work = states.merge(pcs, left_on="comid", right_on="reach_id", validate="many_to_one")
    n_time, n_reach = 204, 230
    basis = np.empty((n_time, n_reach, 8), float)
    basis[:, :, 0] = 1.0
    for i in range(4):
        basis[:, :, i + 1] = work[f"static_pc{i+1}"].to_numpy(float).reshape(n_time, n_reach)
    month = work["month"].to_numpy(float).reshape(n_time, n_reach)
    basis[:, :, 5] = np.sin(2 * np.pi * month / 12.0)
    basis[:, :, 6] = np.cos(2 * np.pi * month / 12.0)
    basis[:, :, 7] = work["antecedent_wetness"].to_numpy(float).reshape(n_time, n_reach)
    local_q = work["local_quick_cfs_reconstructed"].to_numpy(float).reshape(n_time, n_reach)
    local_s = work["local_slow_cfs_reconstructed"].to_numpy(float).reshape(n_time, n_reach)
    return work, basis, local_q, local_s


def corrected_panel(theta: np.ndarray, basis: np.ndarray, local_q: np.ndarray, local_s: np.ndarray, aggregate: np.ndarray):
    tq, ts = theta[:8], theta[8:]
    cq = 0.25 + 3.75 * expit(np.einsum("trk,k->tr", basis, tq))
    cs = 0.25 + 3.75 * expit(np.einsum("trk,k->tr", basis, ts))
    q = (local_q * cq) @ aggregate.T
    s = (local_s * cs) @ aggregate.T
    return q, s, cq, cs


def observation_index(work: pd.DataFrame, training_end: int = 2018):
    mask = work["Q_obsv_cfs"].gt(0) & work["year"].le(training_end)
    positions = np.flatnonzero(mask.to_numpy())
    return positions // 230, positions % 230, work.loc[mask, "Q_obsv_cfs"].to_numpy(float)


def fit_regionalized(work, basis, local_q, local_s, aggregate):
    ti, ri, observed = observation_index(work)
    prior_mean = np.zeros(16, float)
    prior_mean[[0, 8]] = BASE_V
    prior_sd = np.full(16, 0.30, float)
    prior_sd[[0, 8]] = 0.50

    def objective(theta: np.ndarray) -> float:
        q, s, _, _ = corrected_panel(theta, basis, local_q, local_s, aggregate)
        pred = q[ti, ri] + s[ti, ri]
        residual = np.log1p(pred.clip(min=0)) - np.log1p(observed)
        prior = (theta - prior_mean) / prior_sd
        return float(np.mean(residual**2) + np.mean(prior**2) / len(observed))

    rng = np.random.default_rng(SEED)
    starts = [prior_mean.copy()] + [prior_mean + rng.normal(0, 0.15, 16) for _ in range(4)]
    fits = [minimize(objective, start, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12}) for start in starts]
    best = min(fits, key=lambda result: result.fun)
    dds_theta = best.x.copy()
    dds_value = float(best.fun)
    for iteration in range(800):
        probability = max(0.05, 1.0 - np.log(iteration + 1) / np.log(801))
        active = rng.random(16) < probability
        if not active.any():
            active[rng.integers(0, 16)] = True
        proposal = dds_theta.copy()
        proposal[active] += rng.normal(0, 0.08, int(active.sum()))
        value = objective(proposal)
        if value < dds_value:
            dds_theta, dds_value = proposal, value
    solver = {
        "lbfgs_objectives": [float(result.fun) for result in fits],
        "lbfgs_success": [bool(result.success) for result in fits],
        "lbfgs_best": float(best.fun),
        "dds_best": float(dds_value),
        "dds_found_materially_better": bool(dds_value < best.fun * (1 - 1e-3)),
        "optimizer_confounded": bool(dds_value < best.fun * (1 - 1e-3) or not best.success),
    }
    chosen = dds_theta if dds_value < best.fun else best.x
    return chosen, solver


def compact_global_basis(work: pd.DataFrame) -> np.ndarray:
    total = work["routed_quick_cfs"].clip(lower=0) + work["routed_base_cfs"].clip(lower=0)
    values = [
        np.ones(len(work)), np.log1p(work["routed_quick_cfs"].clip(lower=0)),
        np.log1p(work["routed_base_cfs"].clip(lower=0)), work["antecedent_wetness"],
        np.sin(2 * np.pi * work["month"] / 12.0), np.cos(2 * np.pi * work["month"] / 12.0),
        np.log1p(total),
    ]
    values += [work[f"static_pc{i}"].to_numpy(float) for i in range(1, 5)]
    return np.column_stack(values)


def fit_compact_layers(work: pd.DataFrame):
    train = work["Q_obsv_cfs"].gt(0) & work["year"].le(2018)
    x_all = compact_global_basis(work)
    x = x_all[train]
    y = np.log1p(work.loc[train, "Q_obsv_cfs"].to_numpy(float))
    penalty = np.diag(np.r_[0.0, np.full(x.shape[1] - 1, 1 / 1.5)])
    beta_p1 = np.linalg.lstsq(np.vstack([x, penalty]), np.r_[y, np.zeros(x.shape[1])], rcond=None)[0]
    station = work.loc[train, "q_site"].astype(str)
    levels = sorted(station.unique())
    level_index = {name: i for i, name in enumerate(levels)}
    local_basis = np.column_stack([
        np.ones(train.sum()),
        np.sin(2 * np.pi * work.loc[train, "month"] / 12.0),
        np.cos(2 * np.pi * work.loc[train, "month"] / 12.0),
        np.log1p((work.loc[train, "routed_quick_cfs"] + work.loc[train, "routed_base_cfs"]).clip(lower=0)),
        work.loc[train, "antecedent_wetness"].to_numpy(float),
    ])
    z = np.zeros((train.sum(), len(levels) * 5), float)
    for row, name in enumerate(station):
        start = level_index[name] * 5
        z[row, start:start + 5] = local_basis[row]
    x2 = np.column_stack([x, z])
    p2_pen = np.r_[np.r_[0.0, np.full(x.shape[1] - 1, 1 / 1.5)], np.full(z.shape[1], 1 / 0.5)]
    beta_p2 = np.linalg.lstsq(np.vstack([x2, np.diag(p2_pen)]), np.r_[y, np.zeros(x2.shape[1])], rcond=None)[0]
    return x_all @ beta_p1, x_all @ beta_p2[:x.shape[1]], beta_p1, beta_p2, levels


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    states, aggregate, index, terminals = build_states()
    pcs, pca_audit = static_pcs(states)
    work, basis, local_q, local_s = prepare_arrays(states, aggregate, pcs)
    theta, solver = fit_regionalized(work, basis, local_q, local_s, aggregate)
    q_corr, s_corr, cq, cs = corrected_panel(theta, basis, local_q, local_s, aggregate)
    p1_log, p2_cold_log, beta_p1, beta_p2, station_levels = fit_compact_layers(work)
    parent_theta = np.zeros(16, float)
    parent_theta[[0, 8]] = BASE_V
    pq, ps, _, _ = corrected_panel(parent_theta, basis, local_q, local_s, aggregate)
    p0q = work["routed_quick_cfs"].to_numpy(float).reshape(204, 230)
    p0s = work["routed_base_cfs"].to_numpy(float).reshape(204, 230)
    reproduction = float(np.max(np.abs((pq + ps) - (p0q + p0s))) / max(np.max(p0q + p0s), 1.0))
    out = work[["comid", "year", "month", "q_site", "Q_obsv_cfs", "terminal_tree", "antecedent_wetness"]].copy()
    out = out.rename(columns={"comid": "reach_id", "Q_obsv_cfs": "observed_cfs"})
    out["q72_quick_cfs"] = p0q.reshape(-1)
    out["q72_slow_cfs"] = p0s.reshape(-1)
    out["q72_total_cfs"] = out["q72_quick_cfs"] + out["q72_slow_cfs"]
    out["corrected_quick_cfs"] = q_corr.reshape(-1)
    out["corrected_slow_cfs"] = s_corr.reshape(-1)
    out["corrected_total_cfs"] = out["corrected_quick_cfs"] + out["corrected_slow_cfs"]
    out["quick_correction_factor"] = cq.reshape(-1)
    out["slow_correction_factor"] = cs.reshape(-1)
    out["p1_global_compact_cfs"] = np.expm1(np.clip(p1_log, -20, 20))
    out["p2_compact_zero_history_coldstart_cfs"] = np.expm1(np.clip(p2_cold_log, -20, 20))
    out.to_parquet(OUTPUTS / "hydrologic_candidate_panel_230_reaches_2006_2022.parquet", index=False)
    parameters = []
    names = ["intercept", "PC1", "PC2", "PC3", "PC4", "month_sin", "month_cos", "antecedent_wetness"]
    for pathway, values in [("quick", theta[:8]), ("slow", theta[8:])]:
        parameters.extend({"model_id": "REGIONALIZED_ROUTED_CORRECTION", "pathway": pathway, "parameter": name, "value": float(value)} for name, value in zip(names, values))
    parameters.extend({"model_id": "P1_GLOBAL_COMPACT", "pathway": "readout", "parameter": f"beta_{i}", "value": float(v)} for i, v in enumerate(beta_p1))
    pd.DataFrame(parameters).to_parquet(OUTPUTS / "candidate_parameters.parquet", index=False)
    verification = {
        "rows": len(out), "reaches": out["reach_id"].nunique(), "months": out[["year", "month"]].drop_duplicates().shape[0],
        "parent_endpoint_relative_error": reproduction,
        "production_mass_balance_max_abs_mm": float(np.nanmax(np.abs(work["production_mass_balance_error_mm"]))),
        "correction_factor_min": float(min(cq.min(), cs.min())),
        "correction_factor_max": float(max(cq.max(), cs.max())),
        "near_boundary_fraction": float(np.mean((cq < 0.275) | (cq > 3.9)) + np.mean((cs < 0.275) | (cs > 3.9))) / 2,
        "finite_predictions": bool(np.isfinite(out[["q72_total_cfs", "corrected_total_cfs", "p1_global_compact_cfs", "p2_compact_zero_history_coldstart_cfs"]]).all().all()),
        "target_station_history_in_eligible_candidate": False,
        "reservoir_equation_modified": False,
        "pca": pca_audit,
        "solver": solver,
        "status": "CANDIDATES_IMPLEMENTED" if reproduction <= 1e-9 and not solver["optimizer_confounded"] else "CANDIDATE_IMPLEMENTATION_CONFOUNDED",
    }
    dump(REPORTS / "candidate_implementation_audit.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
