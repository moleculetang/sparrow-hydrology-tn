"""Run the single conservative SON-pool challenge against M0."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_14"
P12 = ROOT / "5_Test" / "20260824_12"
P13 = ROOT / "5_Test" / "20260824_13"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
SOURCES = P12 / "outputs" / "monthly_source_forcing_1961_2024.parquet"
MONTHLY = P12 / "outputs" / "canonical_tn_bridge_monthly_2010_2024.parquet"
OBS = P12 / "outputs" / "tn_observations_primary_2016_2024.parquet"
FOLDS = P12 / "outputs" / "tn_evaluation_fold_registry.parquet"
EXPANSION = P12 / "outputs" / "tn_natural_expansion_2021_registry.parquet"
M0_PRED = P13 / "outputs" / "m0_carrier_oof_predictions.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
CONTRACT = RUN / "experiment_contract.json"
K_VALUES = (0.05, 0.16, 0.30, 0.75)
CENTRAL_K = 0.16
MARGIN = 0.005
PI_BOUNDS = (1.0e-4, 1.0 - 1.0e-4)
VF_BOUNDS = (0.0, 0.5)
RHO_BOUNDS = (0.0, 0.95)
AG_COLUMNS = ("fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n")
AG_TAGS = ("FERT", "MAN", "BNF")

sys.path.insert(0, str(P13 / "scripts"))
from stage13_model import M0Router, kernel_array, topology_operators  # noqa: E402
from run_stage13 import fold_frames, group_macro_mse, metric_values, objective_arrays  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def spin_and_simulate_unit_son(source: pd.DataFrame, k_year: float) -> tuple[pd.DataFrame, dict[str, object]]:
    """Simulate a unit-allocation SON pool; arbitrary rho scales this solution."""
    src = source.loc[source.calendar_scenario.eq("CENTRAL")].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    reaches = np.sort(src.reach_id.unique().astype(int))
    n = len(reaches)
    early = src.loc[src.year.between(1961, 1965)].groupby(["month", "reach_id"], as_index=False)[list(AG_COLUMNS)].mean()
    state = np.zeros((n, len(AG_TAGS)), dtype=float)
    converged = False
    cycles = 0
    for cycle in range(1, 10001):
        before = state.copy()
        for month in range(1, 13):
            block = early.loc[early.month.eq(month)].sort_values("reach_id")
            days = int(pd.Period(f"2001-{month:02d}").days_in_month)
            release_fraction = 1.0 - math.exp(-k_year * days / 365.25)
            pre = state + block[list(AG_COLUMNS)].to_numpy(float)
            state = (1.0 - release_fraction) * pre
        cycles = cycle
        relative = np.max(np.abs(state - before) / np.maximum(np.abs(state), 1.0))
        if relative <= 1.0e-12:
            converged = True
            break
    if not converged:
        raise RuntimeError(f"SON spin-up failed for k={k_year}")
    initial = state.copy()
    cumulative_input = np.zeros_like(state)
    cumulative_release = np.zeros_like(state)
    records = []
    max_error = 0.0
    max_relative = 0.0
    for (year, month), block in src.groupby(["year", "month"], sort=True):
        block = block.sort_values("reach_id")
        x = block[list(AG_COLUMNS)].to_numpy(float)
        days = int(block.days_in_month.iloc[0])
        release_fraction = 1.0 - math.exp(-k_year * days / 365.25)
        pre = state + x
        release = release_fraction * pre
        state = pre - release
        cumulative_input += x
        cumulative_release += release
        closure = initial + cumulative_input - cumulative_release - state
        max_error = max(max_error, float(np.max(np.abs(closure))))
        max_relative = max(max_relative, float(np.max(np.abs(closure) / np.maximum(initial + cumulative_input, 1.0))))
        for j, tag in enumerate(AG_TAGS):
            records.append(pd.DataFrame({
                "k_L_year_inverse": k_year, "reach_id": reaches, "year": int(year), "month": int(month),
                "source_tag": tag, "unit_son_input_kg_n": x[:, j], "unit_son_release_kg_n": release[:, j],
                "unit_son_end_kg_n": state[:, j],
            }))
    audit = {
        "k_L_year_inverse": k_year, "spinup_cycles": cycles, "spinup_converged": converged,
        "spinup_terminal_relative_delta": relative, "max_abs_mass_error_kg_n": max_error,
        "max_relative_mass_error": max_relative, "minimum_state_kg_n": float(state.min()),
    }
    return pd.concat(records, ignore_index=True), audit


class M1Engine:
    def __init__(self, k_year: float, unit_son: pd.DataFrame, sources: pd.DataFrame, kernels: pd.DataFrame, monthly: pd.DataFrame):
        self.k_year = k_year
        self.monthly = monthly.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        self.reach_ids = np.sort(self.monthly.reach_id.unique().astype(int))
        months = self.monthly[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
        self.month_keys = [(int(y), int(m)) for y, m in months.itertuples(index=False)]
        self.shape = (len(months), len(self.reach_ids))
        self.kernels = kernel_array(kernels.sort_values(["year", "month", "reach_id"]))
        source = sources.loc[sources.calendar_scenario.eq("CENTRAL") & sources.year.ge(2010)].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        unit = unit_son.loc[unit_son.year.ge(2010)].groupby(["year", "month", "reach_id"], as_index=False).unit_son_release_kg_n.sum().sort_values(["year", "month", "reach_id"])
        if not np.array_equal(source[["year", "month", "reach_id"]].to_numpy(), unit[["year", "month", "reach_id"]].to_numpy()):
            raise RuntimeError("SON/source key mismatch")
        self.ag_input = source[list(AG_COLUMNS)].sum(axis=1).to_numpy(float).reshape(self.shape)
        self.legacy_release = unit.unit_son_release_kg_n.to_numpy(float).reshape(self.shape)
        self.dep = source.atmospheric_deposition_kg_n.to_numpy(float).reshape(self.shape)
        self.demand = source.crop_demand_kg_n.to_numpy(float).reshape(self.shape)
        self.local_water = (self.monthly.local_fast_response_volume_m3 + self.monthly.local_slow_response_volume_m3).to_numpy(float).reshape(self.shape)
        self.h_full = self.monthly.h1_exposure_day_per_m.to_numpy(float).reshape(self.shape)
        self.order, self.downstream, _ = topology_operators(TOPOLOGY, self.reach_ids)
        self._router_cache: OrderedDict[float, M0Router] = OrderedDict()

    def available_input(self, rho: float) -> np.ndarray:
        before_demand = self.dep + (1.0 - rho) * self.ag_input + rho * self.legacy_release
        return np.maximum(before_demand - self.demand, 0.0)

    def local_load(self, rho: float) -> np.ndarray:
        x = self.available_input(rho)
        zu = np.zeros(len(self.reach_ids), dtype=float)
        zl = np.zeros(len(self.reach_ids), dtype=float)
        load = np.zeros(self.shape, dtype=float)
        for t in range(self.shape[0]):
            vectors = np.column_stack([zu, zl, x[t]])
            result = np.einsum("roi,ri->ro", self.kernels[t * len(self.reach_ids):(t + 1) * len(self.reach_ids)], vectors)
            zu, zl = result[:, 0], result[:, 1]
            load[t] = result[:, 2] + result[:, 3]
        return load

    def router(self, rho: float) -> M0Router:
        key = round(float(rho), 9)
        if key in self._router_cache:
            router = self._router_cache.pop(key)
            self._router_cache[key] = router
            return router
        router = M0Router(
            local_load=self.local_load(float(rho)), local_water=self.local_water, h_full=self.h_full,
            month_keys=self.month_keys, reach_ids=self.reach_ids, order=self.order, downstream=self.downstream,
        )
        self._router_cache[key] = router
        while len(self._router_cache) > 4:
            self._router_cache.popitem(last=False)
        return router


def fit_m1(engine: M1Engine, train: pd.DataFrame) -> dict[str, object]:
    y, codes = objective_arrays(train)

    def objective(theta: np.ndarray) -> float:
        rho, pi, vf = map(float, theta)
        pred = pi * engine.router(rho).concentration_at_pi1(train, vf)
        error2 = np.square(np.log1p(np.maximum(pred, 0.0)) - y)
        return float(np.mean([group_macro_mse(error2, code) for code in codes]))

    starts = (
        (0.0, 0.02, 0.03), (0.10, 0.05, 0.05), (0.30, 0.10, 0.10),
        (0.55, 0.20, 0.20), (0.80, 0.40, 0.05), (0.95, 0.70, 0.35),
    )
    fits = [minimize(
        objective, np.asarray(start), method="L-BFGS-B",
        bounds=(RHO_BOUNDS, PI_BOUNDS, VF_BOUNDS),
        options={"ftol": 1.0e-11, "gtol": 1.0e-7, "maxiter": 250, "maxls": 30},
    ) for start in starts]
    best = min(fits, key=lambda result: float(result.fun))
    rho, pi, vf = map(float, best.x)
    return {
        "rho_L": rho, "pi_E": pi, "v_f_m_per_day": vf, "objective": float(best.fun),
        "success": bool(best.success), "message": str(best.message), "iterations": int(best.nit),
        "function_evaluations": int(best.nfev),
        "rho_zero_boundary": bool(rho <= 1.0e-6), "rho_upper_boundary": bool(rho >= RHO_BOUNDS[1] - 1.0e-6),
        "pi_boundary": bool(pi <= PI_BOUNDS[0] + 1.0e-6 or pi >= PI_BOUNDS[1] - 1.0e-6),
        "vf_boundary": bool(vf <= 1.0e-7 or vf >= VF_BOUNDS[1] - 1.0e-6),
    }


def fit_fold(engine: M1Engine, obs: pd.DataFrame, fold: pd.Series, expansion: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    train, test = fold_frames(obs, fold, expansion)
    fit = fit_m1(engine, train)
    pred = test.copy()
    pred["pred_tn_mg_l"] = float(fit["pi_E"]) * engine.router(float(fit["rho_L"])).concentration_at_pi1(test, float(fit["v_f_m_per_day"]))
    pred["architecture"] = "M1_SON1"
    pred["k_L_year_inverse"] = engine.k_year
    pred["fold_id"] = str(fold.fold_id)
    pred["holdout_type"] = str(fold.holdout_type)
    pred["holdout_id"] = str(fold.holdout_id)
    pred["evaluation_year"] = int(fold.evaluation_year)
    pars = {
        "architecture": "M1_SON1", "k_L_year_inverse": engine.k_year,
        "fold_id": str(fold.fold_id), "holdout_type": str(fold.holdout_type), "holdout_id": str(fold.holdout_id),
        "train_start_year": int(fold.train_start_year), "train_end_year": int(fold.train_end_year),
        "evaluation_year": int(fold.evaluation_year), "train_rows": len(train), "test_rows": len(test), **fit,
    }
    return pred, pars


def paired_delta(m0: pd.DataFrame, m1: pd.DataFrame, block: str, seed: int) -> dict[str, object]:
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    joined = m0.merge(m1, on=keys, suffixes=("_m0", "_m1"), validate="one_to_one")
    deltas = []
    for _, group in joined.groupby(block):
        y = np.log1p(group.tn_mg_l.to_numpy(float))
        a = np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l_m0.to_numpy(float)) - y)))
        b = np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l_m1.to_numpy(float)) - y)))
        deltas.append(b - a)
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
    upper = float(np.quantile(samples, 0.975))
    return {
        "block": block, "block_count": len(values), "delta_m1_minus_m0": float(values.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)), "ci95_upper": upper,
        "noninferior_0p005": bool(upper < MARGIN), "improved": bool(upper < 0.0),
    }


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    decision13 = json.loads((P13 / "reports" / "stage13_decision.json").read_text(encoding="utf-8"))
    if decision13["status"] != "PASS_STAGE13_CARRIER_LOCKED":
        raise RuntimeError("stage13 carrier not locked")
    selected = str(decision13["selected_carrier"])
    kernel_path = P13 / "outputs" / ("daily_compiled_carrier_kernels.parquet" if selected == "DAILY_COMPILED_CARRIER" else "monthly_balance_carrier_kernels.parquet")
    kernels = pd.read_parquet(kernel_path)
    monthly = pd.read_parquet(MONTHLY)
    sources = pd.read_parquet(SOURCES)
    obs = pd.read_parquet(OBS)
    folds = pd.read_parquet(FOLDS)
    expansion = pd.read_parquet(EXPANSION)
    m0 = pd.read_parquet(M0_PRED).loc[lambda x: x.carrier.eq(selected)]
    temporal_folds = folds.loc[folds.holdout_type.eq("TEMPORAL")].sort_values("evaluation_year")

    unit_frames, audits, engines = [], [], {}
    for k in K_VALUES:
        unit, audit = spin_and_simulate_unit_son(sources, k)
        unit_frames.append(unit)
        audits.append(audit)
        engines[k] = M1Engine(k, unit, sources, kernels, monthly)
    unit_all = pd.concat(unit_frames, ignore_index=True)
    unit_all.to_parquet(OUT / "son_unit_release_1961_2024.parquet", index=False)
    pd.DataFrame(audits).to_parquet(OUT / "son_spinup_mass_audit.parquet", index=False)

    # Exact nesting at rho=0 against the selected M0 local carrier.
    m0_flux = pd.read_parquet(P13 / "outputs" / "m0_local_source_tagged_fluxes.parquet")
    m0_local = m0_flux.loc[m0_flux.carrier.eq(selected)].groupby(["year", "month", "reach_id"], as_index=False).agg(
        load=("local_fast_release_kg_n", "sum"), slow=("local_slow_release_kg_n", "sum")
    ).sort_values(["year", "month", "reach_id"])
    m0_total = (m0_local.load + m0_local.slow).to_numpy(float).reshape(engines[CENTRAL_K].shape)
    nesting_abs = float(np.max(np.abs(engines[CENTRAL_K].local_load(0.0) - m0_total)))
    nesting_rel = float(np.max(np.abs(engines[CENTRAL_K].local_load(0.0) - m0_total) / np.maximum(m0_total, 1.0)))

    predictions, parameters = [], []
    for k, engine in engines.items():
        for row in temporal_folds.itertuples(index=False):
            pred, par = fit_fold(engine, obs, pd.Series(row._asdict()), expansion)
            predictions.append(pred)
            parameters.append(par)
        print(json.dumps({"completed_temporal_k": k}), flush=True)
    pred_frame = pd.concat(predictions, ignore_index=True)
    par_frame = pd.DataFrame(parameters)

    temporal_comparisons = []
    supported_k = []
    for index, k in enumerate(K_VALUES):
        p = pred_frame.loc[pred_frame.k_L_year_inverse.eq(k)]
        comparisons = [paired_delta(m0.loc[m0.holdout_type.eq("TEMPORAL")], p, block, 2026082400 + index * 10 + j) for j, block in enumerate(("station_key", "reach_id", "terminal_tree_id"))]
        for comparison in comparisons:
            temporal_comparisons.append({"k_L_year_inverse": k, "holdout_type": "TEMPORAL", **comparison})
        by_block = {item["block"]: item for item in comparisons}
        noninferior = all(item["noninferior_0p005"] for item in comparisons)
        improved = by_block["station_key"]["improved"] or by_block["reach_id"]["improved"]
        upper_confounded = int(par_frame.loc[par_frame.k_L_year_inverse.eq(k), "rho_upper_boundary"].sum()) >= 2
        if noninferior and improved and not upper_confounded:
            supported_k.append(k)

    temporal_legacy_supported = len(supported_k) >= 3 and CENTRAL_K in supported_k
    spatial_comparisons = []
    if temporal_legacy_supported:
        spatial_folds = folds.loc[folds.holdout_type.isin(["REACH", "TREE"])].sort_values(["evaluation_year", "holdout_type", "holdout_id"])
        for row in spatial_folds.itertuples(index=False):
            pred, par = fit_fold(engines[CENTRAL_K], obs, pd.Series(row._asdict()), expansion)
            predictions.append(pred)
            parameters.append(par)
        pred_frame = pd.concat(predictions, ignore_index=True)
        par_frame = pd.DataFrame(parameters)
        for j, (kind, block) in enumerate((("REACH", "reach_id"), ("TREE", "terminal_tree_id"))):
            comparison = paired_delta(
                m0.loc[m0.holdout_type.eq(kind)],
                pred_frame.loc[pred_frame.k_L_year_inverse.eq(CENTRAL_K) & pred_frame.holdout_type.eq(kind)],
                block, 2026082450 + j,
            )
            spatial_comparisons.append({"k_L_year_inverse": CENTRAL_K, "holdout_type": kind, **comparison})
    spatial_pass = bool(spatial_comparisons) and all(item["noninferior_0p005"] for item in spatial_comparisons)
    legacy_supported = temporal_legacy_supported and spatial_pass
    selected_architecture = "M1_SON1_K016" if legacy_supported else "M0_NO_SON"

    pred_frame.to_parquet(OUT / "m1_son_oof_predictions.parquet", index=False)
    par_frame.to_parquet(OUT / "m1_son_fold_parameters.parquet", index=False)
    comparison_frame = pd.DataFrame(temporal_comparisons + spatial_comparisons)
    comparison_frame.to_parquet(OUT / "m0_m1_paired_bootstrap.parquet", index=False)
    metric_rows = []
    for (k, kind), group in pred_frame.groupby(["k_L_year_inverse", "holdout_type"]):
        metric_rows.append({"architecture": "M1_SON1", "k_L_year_inverse": k, "holdout_type": kind, **metric_values(group)})
    pd.DataFrame(metric_rows).to_parquet(OUT / "m1_son_performance_metrics.parquet", index=False)

    checks = {
        "all_spinups_converged": all(item["spinup_converged"] for item in audits),
        "son_mass_relative_le_1e_12": max(item["max_relative_mass_error"] for item in audits) <= 1.0e-12,
        "rho0_exact_m0_relative_le_1e_12": nesting_rel <= 1.0e-12,
        "all_fits_success": bool(par_frame.success.all()),
        "kL_not_selected_from_TN": True,
        "station_parameters_absent": True,
    }
    status = "PASS_STAGE14_ARCHITECTURE_LOCKED" if all(checks.values()) else "FAIL_STAGE14"
    decision = {
        "stage": "20260824_14", "status": status, "selected_carrier": selected,
        "selected_architecture": selected_architecture, "legacy_supported": legacy_supported,
        "temporal_legacy_supported": temporal_legacy_supported, "spatial_noninferiority_pass": spatial_pass,
        "supported_k_sensitivities": supported_k, "production_k_L_year_inverse": CENTRAL_K if legacy_supported else None,
        "rho0_nesting_max_abs_kg_n": nesting_abs, "rho0_nesting_max_relative": nesting_rel,
        "checks": checks, "temporal_comparisons": temporal_comparisons,
        "spatial_comparisons": spatial_comparisons,
        "parameter_summary": par_frame.groupby("k_L_year_inverse").agg(
            rho_median=("rho_L", "median"), rho_min=("rho_L", "min"), rho_max=("rho_L", "max"),
            pi_median=("pi_E", "median"), vf_median=("v_f_m_per_day", "median"), fits=("fold_id", "size")
        ).reset_index().to_dict("records"),
        "hashes": {"contract": sha256(CONTRACT), "selected_kernel": sha256(kernel_path)},
        "authorized_successor": "20260824_15" if status.startswith("PASS") else None,
    }
    write_json(REPORTS / "stage14_decision.json", decision)
    write_json(REPORTS / "selected_tn_architecture_lock.json", {
        "stage": "20260824_14", "status": "LOCKED" if status.startswith("PASS") else "NOT_LOCKED",
        "selected_architecture": selected_architecture, "selected_carrier": selected,
        "k_L_year_inverse": CENTRAL_K if legacy_supported else None,
    })
    report = f"""# 20260824_14 single-SON Legacy experiment

## Decision

`{status}`  
Selected architecture: **{selected_architecture}**.

The experiment adds one conservative agricultural organic-N pool to FERT, MAN and BNF. Atmospheric
deposition bypasses the pool. Crop demand is deducted only after direct mineral input and SON release
are combined. `rho_L=0` recovers M0 with maximum relative discrepancy `{nesting_rel:.3e}`.

`k_L` was not fitted or selected from TN. Values `{K_VALUES}` are fixed sensitivity members and
`0.16 yr-1` is the only pre-registered production value. Temporal support was obtained for
`{supported_k}`; nested spatial noninferiority was `{spatial_pass}`. Therefore agricultural Legacy
support is `{legacy_supported}`.

This result concerns a parsimonious effective SON-release operator. It does not independently identify
real soil organic-N stock, groundwater age, or a second transport-time distribution.
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    if not status.startswith("PASS"):
        raise RuntimeError(decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
