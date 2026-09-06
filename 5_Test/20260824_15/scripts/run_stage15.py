"""Fit and evaluate static Reach regionalization of land-delivery pi_E."""

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
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_15"
P12 = ROOT / "5_Test" / "20260824_12"
P13 = ROOT / "5_Test" / "20260824_13"
P14 = ROOT / "5_Test" / "20260824_14"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
OBS = P12 / "outputs" / "tn_observations_primary_2016_2024.parquet"
FOLDS = P12 / "outputs" / "tn_evaluation_fold_registry.parquet"
EXPANSION = P12 / "outputs" / "tn_natural_expansion_2021_registry.parquet"
MONTHLY = P12 / "outputs" / "canonical_tn_bridge_monthly_2010_2024.parquet"
M0_FLUX = P13 / "outputs" / "m0_local_source_tagged_fluxes.parquet"
M0_PRED = P13 / "outputs" / "m0_carrier_oof_predictions.parquet"
M0_PAR = P13 / "outputs" / "m0_carrier_fold_parameters.parquet"
COVARIATES = OUT / "static_delivery_covariates_standardized.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
CONTRACT = RUN / "experiment_contract.json"
MARGIN = 0.005
PRIOR_SD = 0.35
GAMMA_BOUND = 1.5
VF_BOUND = 0.5
EPS = 1.0e-12

sys.path.insert(0, str(P13 / "scripts"))
from stage13_model import topology_operators  # noqa: E402
from run_stage13 import fold_frames, metric_values  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def observation_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=float)
    for column in ("station_key", "reach_id", "terminal_tree_id"):
        codes = pd.Categorical(frame[column]).codes
        counts = np.bincount(codes)
        groups = len(counts)
        weights += 1.0 / (3.0 * groups * counts[codes])
    return weights


class RegionalRouter:
    """Differentiable static-pi routing with analytic derivatives."""

    def __init__(self, local_base: np.ndarray, monthly: pd.DataFrame, x: np.ndarray):
        hydro = monthly.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        self.reach_ids = np.sort(hydro.reach_id.unique().astype(int))
        times = hydro[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
        self.month_keys = [(int(y), int(m)) for y, m in times.itertuples(index=False)]
        self.shape = (len(times), len(self.reach_ids))
        self.local_base = local_base
        self.local_water = (hydro.local_fast_response_volume_m3 + hydro.local_slow_response_volume_m3).to_numpy(float).reshape(self.shape)
        self.h = hydro.h1_exposure_day_per_m.to_numpy(float).reshape(self.shape)
        self.x = x
        self.rlookup = {int(r): i for i, r in enumerate(self.reach_ids)}
        self.tlookup = {key: i for i, key in enumerate(self.month_keys)}
        order, downstream, _ = topology_operators(TOPOLOGY, self.reach_ids)
        self.order_idx = [self.rlookup[r] for r in order]
        self.down_idx = {self.rlookup[r]: self.rlookup[d] for r, d in downstream.items()}
        self.water_inlet = np.zeros_like(self.local_water)
        self.water_outlet = np.zeros_like(self.local_water)
        for i in self.order_idx:
            self.water_outlet[:, i] = self.water_inlet[:, i] + self.local_water[:, i]
            if i in self.down_idx:
                self.water_inlet[:, self.down_idx[i]] += self.water_outlet[:, i]

    def evaluate(self, observations: pd.DataFrame, theta: np.ndarray, derivatives: bool = True) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        alpha, gamma, vf = float(theta[0]), np.asarray(theta[1:6], dtype=float), float(theta[6])
        linear = alpha + self.x @ gamma
        pi = sigmoid(linear)
        full_tidx = np.fromiter((self.tlookup[(int(y), int(m))] for y, m in observations[["year", "month"]].itertuples(index=False)), dtype=int, count=len(observations))
        unique_tidx, observation_tidx = np.unique(full_tidx, return_inverse=True)
        base = self.local_base[unique_tidx]
        h_route = self.h[unique_tidx]
        local = base * pi[None, :]
        npar = 7
        inlet = np.zeros_like(local)
        outlet = np.zeros_like(local)
        dinlet = np.zeros((*local.shape, npar), dtype=float) if derivatives else None
        doutlet = np.zeros((*local.shape, npar), dtype=float) if derivatives else None
        dpi_linear = pi * (1.0 - pi)
        design = np.column_stack([np.ones(len(pi)), self.x])
        dlocal_static = base[:, :, None] * (dpi_linear[:, None] * design)[None, :, :]
        survival = np.exp(-vf * h_route)
        midpoint = np.exp(-vf * h_route / 2.0)
        for i in self.order_idx:
            outlet[:, i] = inlet[:, i] * survival[:, i] + local[:, i] * midpoint[:, i]
            if derivatives and dinlet is not None and doutlet is not None:
                doutlet[:, i, :6] = dinlet[:, i, :6] * survival[:, i, None] + dlocal_static[:, i, :] * midpoint[:, i, None]
                doutlet[:, i, 6] = (
                    dinlet[:, i, 6] * survival[:, i]
                    - inlet[:, i] * h_route[:, i] * survival[:, i]
                    - 0.5 * local[:, i] * h_route[:, i] * midpoint[:, i]
                )
            if i in self.down_idx:
                down = self.down_idx[i]
                inlet[:, down] += outlet[:, i]
                if derivatives and dinlet is not None and doutlet is not None:
                    dinlet[:, down, :] += doutlet[:, i, :]

        ridx = observations.reach_id.astype(int).map(self.rlookup).to_numpy(int)
        frac = observations.downstream_fraction_on_reach.to_numpy(float)
        h = h_route[observation_tidx, ridx]
        sf = np.exp(-vf * h * frac)
        sm = np.exp(-vf * h * frac / 2.0)
        local_obs = local[observation_tidx, ridx]
        load = inlet[observation_tidx, ridx] * sf + frac * local_obs * sm
        water = self.water_inlet[full_tidx, ridx] + frac * self.local_water[full_tidx, ridx]
        pred = 1000.0 * load / np.maximum(water, EPS)
        if not derivatives or dinlet is None:
            return pred, None, pi
        dload = np.zeros((len(observations), npar), dtype=float)
        dload[:, :6] = dinlet[observation_tidx, ridx, :6] * sf[:, None] + frac[:, None] * dlocal_static[observation_tidx, ridx, :] * sm[:, None]
        dload[:, 6] = (
            dinlet[observation_tidx, ridx, 6] * sf
            - inlet[observation_tidx, ridx] * h * frac * sf
            - 0.5 * frac * local_obs * h * frac * sm
        )
        dpred = 1000.0 * dload / np.maximum(water[:, None], EPS)
        return pred, dpred, pi


def fit_regional(router: RegionalRouter, train: pd.DataFrame, parent_pi: float, parent_vf: float) -> dict[str, object]:
    y = np.log1p(train.tn_mg_l.to_numpy(float))
    weights = observation_weights(train)
    nblocks = train.station_key.nunique() + train.reach_id.nunique() + train.terminal_tree_id.nunique()

    def value_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        pred, dpred, _ = router.evaluate(train, theta, derivatives=True)
        assert dpred is not None
        error = np.log1p(np.maximum(pred, 0.0)) - y
        loss = float(np.sum(weights * error * error))
        derivative = dpred / (1.0 + pred[:, None])
        grad = 2.0 * np.sum((weights * error)[:, None] * derivative, axis=0)
        loss += float(0.5 * np.sum(np.square(theta[1:6] / PRIOR_SD)) / nblocks)
        grad[1:6] += theta[1:6] / (PRIOR_SD ** 2 * nblocks)
        return loss, grad

    alpha0 = float(np.log(parent_pi / (1.0 - parent_pi)))
    starts = [
        np.r_[alpha0, np.zeros(5), parent_vf],
        np.r_[alpha0 + 0.5, np.array([0.15, -0.10, 0.10, -0.10, 0.05]), min(parent_vf + 0.05, VF_BOUND)],
    ]
    bounds = [(-9.21, 9.21), *[(-GAMMA_BOUND, GAMMA_BOUND)] * 5, (0.0, VF_BOUND)]
    results = [minimize(
        lambda z: value_grad(z), start, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1.0e-9, "gtol": 1.0e-6, "maxiter": 150, "maxls": 30},
    ) for start in starts]
    successful = [result for result in results if result.success and np.isfinite(result.fun)]
    best = min(successful if successful else results, key=lambda result: float(result.fun))
    pred, _, pi = router.evaluate(train, best.x, derivatives=False)
    return {
        "theta": np.asarray(best.x, dtype=float), "objective": float(best.fun), "success": bool(best.success),
        "message": str(best.message), "iterations": int(best.nit), "function_evaluations": int(best.nfev),
        "pi_reach_min": float(pi.min()), "pi_reach_max": float(pi.max()), "pi_reach_median": float(np.median(pi)),
        "pi_extreme_fraction": float(np.mean((pi < 1.0e-4) | (pi > 1.0 - 1.0e-4))),
        "gamma_boundary": bool(np.any(np.abs(best.x[1:6]) >= GAMMA_BOUND - 1.0e-6)),
        "vf_boundary": bool(best.x[6] <= 1.0e-7 or best.x[6] >= VF_BOUND - 1.0e-6),
        "training_prediction_mean": float(np.mean(pred)),
    }


def fit_fold(router: RegionalRouter, obs: pd.DataFrame, fold: pd.Series, expansion: pd.DataFrame, parent_parameter: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    train, test = fold_frames(obs, fold, expansion)
    fit = fit_regional(router, train, float(parent_parameter.pi_E), float(parent_parameter.v_f_m_per_day))
    theta = np.asarray(fit.pop("theta"), dtype=float)
    pred, _, pi = router.evaluate(test, theta, derivatives=False)
    output = test.copy()
    output["pred_tn_mg_l"] = pred
    output["architecture"] = "M2_STATIC_PI_REGIONALIZATION"
    output["fold_id"] = str(fold.fold_id)
    output["holdout_type"] = str(fold.holdout_type)
    output["holdout_id"] = str(fold.holdout_id)
    output["evaluation_year"] = int(fold.evaluation_year)
    parameters = {
        "fold_id": str(fold.fold_id), "holdout_type": str(fold.holdout_type), "holdout_id": str(fold.holdout_id),
        "train_start_year": int(fold.train_start_year), "train_end_year": int(fold.train_end_year),
        "evaluation_year": int(fold.evaluation_year), "alpha": float(theta[0]),
        **{f"gamma_{j + 1}": float(theta[j + 1]) for j in range(5)}, "v_f_m_per_day": float(theta[6]),
        "train_rows": len(train), "test_rows": len(test), **fit,
    }
    return output, parameters


def paired_delta(parent: pd.DataFrame, candidate: pd.DataFrame, block: str, seed: int) -> dict[str, object]:
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    joined = parent.merge(candidate, on=keys, suffixes=("_parent", "_candidate"), validate="one_to_one")
    values = []
    for _, group in joined.groupby(block):
        y = np.log1p(group.tn_mg_l.to_numpy(float))
        a = np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l_parent.to_numpy(float)) - y)))
        b = np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l_candidate.to_numpy(float)) - y)))
        values.append(b - a)
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
    upper = float(np.quantile(samples, 0.975))
    return {
        "block": block, "block_count": len(values), "delta_candidate_minus_parent": float(values.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)), "ci95_upper": upper,
        "noninferior_0p005": bool(upper < MARGIN), "improved": bool(upper < 0.0),
    }


def hydrologic_anomalies(monthly: pd.DataFrame) -> pd.DataFrame:
    features = {
        "upper": "upper_response_storage_start_mm",
        "lower": "lower_slow_storage_start_mm",
        "excess": "effective_excess_to_upper_mm",
        "fast_fraction": "state_consistent_fast_fraction",
    }
    h = monthly[["reach_id", "year", "month", *features.values()]].copy()
    for short, column in features.items():
        if short == "fast_fraction":
            values = np.log(np.clip(h[column], 1.0e-6, 1.0 - 1.0e-6) / np.clip(1.0 - h[column], 1.0e-6, 1.0))
        else:
            values = np.log1p(h[column].to_numpy(float))
        h[f"raw_{short}"] = values
        climatology = h.loc[h.year.between(2010, 2020)].groupby(["reach_id", "month"])[f"raw_{short}"].mean().rename("clim")
        h = h.merge(climatology, on=["reach_id", "month"], how="left", validate="many_to_one")
        h[f"anom_{short}"] = h[f"raw_{short}"] - h.clim
        sd = h.loc[h.year.between(2010, 2020)].groupby("reach_id")[f"anom_{short}"].std(ddof=0).replace(0, 1.0).rename("sd")
        h = h.merge(sd, on="reach_id", how="left", validate="many_to_one")
        h[f"z_anom_{short}"] = h[f"anom_{short}"] / h.sd
        h = h.drop(columns=["clim", "sd"])
    return h[["reach_id", "year", "month", *[f"z_anom_{name}" for name in features]]]


def dynamic_trigger_audit(predictions: pd.DataFrame, monthly: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    frame = predictions.merge(hydrologic_anomalies(monthly), on=["reach_id", "year", "month"], validate="many_to_one")
    frame["residual_log"] = np.log1p(frame.tn_mg_l) - np.log1p(frame.pred_tn_mg_l)
    rng = np.random.default_rng(2026082415)
    rows = []
    station_levels = list(frame.station_key.unique())
    station_indices = {
        station: np.flatnonzero(frame.station_key.to_numpy() == station)
        for station in station_levels
    }
    for feature in ("z_anom_upper", "z_anom_lower", "z_anom_excess", "z_anom_fast_fraction"):
        annual = []
        for year, group in frame.groupby("evaluation_year"):
            rho = float(spearmanr(group[feature], group.residual_log).statistic)
            annual.append((int(year), rho))
            rows.append({"feature": feature, "scope": str(int(year)), "spearman_rho": rho, "ci95_lower": np.nan, "ci95_upper": np.nan})
        boot = []
        x_values = frame[feature].to_numpy(float)
        y_values = frame.residual_log.to_numpy(float)
        for _ in range(2000):
            chosen = rng.choice(station_levels, size=len(station_levels), replace=True)
            index = np.concatenate([station_indices[station] for station in chosen])
            boot.append(float(spearmanr(x_values[index], y_values[index]).statistic))
        combined = float(spearmanr(frame[feature], frame.residual_log).statistic)
        lower, upper = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        signs = [np.sign(rho) for _, rho in annual if abs(rho) >= 0.10]
        same_direction_years = max(signs.count(1.0), signs.count(-1.0)) if signs else 0
        triggered = same_direction_years >= 2 and (lower > 0.0 or upper < 0.0)
        rows.append({"feature": feature, "scope": "COMBINED", "spearman_rho": combined, "ci95_lower": lower, "ci95_upper": upper, "same_direction_abs_ge_0p10_years": same_direction_years, "triggered": triggered})
    result = pd.DataFrame(rows)
    return result, bool(result.loc[result.scope.eq("COMBINED"), "triggered"].fillna(False).any())


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow required")
    decision14 = json.loads((P14 / "reports" / "stage14_decision.json").read_text(encoding="utf-8"))
    if decision14["status"] != "PASS_STAGE14_ARCHITECTURE_LOCKED" or decision14["selected_architecture"] != "M0_NO_SON":
        raise RuntimeError("stage15 implementation expects the actually locked M0 parent")
    selected_carrier = str(decision14["selected_carrier"])
    monthly = pd.read_parquet(MONTHLY)
    obs = pd.read_parquet(OBS)
    folds = pd.read_parquet(FOLDS)
    expansion = pd.read_parquet(EXPANSION)
    cov = pd.read_parquet(COVARIATES).sort_values("reach_id")
    x_columns = [column for column in cov if column.startswith("z_")]
    if len(x_columns) != 5:
        raise RuntimeError("five-covariate contract failed")
    flux = pd.read_parquet(M0_FLUX)
    flux = flux.loc[flux.carrier.eq(selected_carrier)].groupby(["year", "month", "reach_id"], as_index=False).agg(
        fast=("local_fast_release_kg_n", "sum"), slow=("local_slow_release_kg_n", "sum")
    ).sort_values(["year", "month", "reach_id"])
    shape = (monthly[["year", "month"]].drop_duplicates().shape[0], 230)
    router = RegionalRouter((flux.fast + flux.slow).to_numpy(float).reshape(shape), monthly, cov[x_columns].to_numpy(float))
    parent_pred = pd.read_parquet(M0_PRED).loc[lambda x: x.carrier.eq(selected_carrier)]
    parent_par = pd.read_parquet(M0_PAR).loc[lambda x: x.carrier.eq(selected_carrier)].set_index("fold_id")

    predictions, parameters = [], []
    temporal_folds = folds.loc[folds.holdout_type.eq("TEMPORAL")].sort_values("evaluation_year")
    for row in temporal_folds.itertuples(index=False):
        fold = pd.Series(row._asdict())
        pred, par = fit_fold(router, obs, fold, expansion, parent_par.loc[str(row.fold_id)])
        predictions.append(pred); parameters.append(par)
    temporal_pred = pd.concat(predictions, ignore_index=True)
    temporal_comparisons = [
        {"holdout_type": "TEMPORAL", **paired_delta(parent_pred.loc[parent_pred.holdout_type.eq("TEMPORAL")], temporal_pred, block, 20260824150 + j)}
        for j, block in enumerate(("station_key", "reach_id", "terminal_tree_id"))
    ]
    temporal_pass = all(item["noninferior_0p005"] for item in temporal_comparisons) and any(item["improved"] for item in temporal_comparisons if item["block"] in {"station_key", "reach_id"})

    spatial_comparisons = []
    if temporal_pass:
        remaining = folds.loc[~folds.holdout_type.eq("TEMPORAL")].sort_values(["evaluation_year", "holdout_type", "holdout_id"])
        for row in remaining.itertuples(index=False):
            fold = pd.Series(row._asdict())
            pred, par = fit_fold(router, obs, fold, expansion, parent_par.loc[str(row.fold_id)])
            predictions.append(pred); parameters.append(par)
        all_pred = pd.concat(predictions, ignore_index=True)
        for j, (kind, block) in enumerate((("REACH", "reach_id"), ("TREE", "terminal_tree_id"), ("FIRST_OBSERVED_2021", "station_key"), ("FIRST_OBSERVED_2021", "reach_id"))):
            spatial_comparisons.append({"holdout_type": kind, **paired_delta(
                parent_pred.loc[parent_pred.holdout_type.eq(kind)], all_pred.loc[all_pred.holdout_type.eq(kind)], block, 20260824160 + j
            )})
    else:
        all_pred = temporal_pred
    par_frame = pd.DataFrame(parameters)
    spatial_pass = bool(spatial_comparisons) and all(item["noninferior_0p005"] for item in spatial_comparisons)
    boundary_confounded = bool(par_frame.gamma_boundary.any() or (par_frame.pi_extreme_fraction > 0.10).any())
    mpr_supported = temporal_pass and spatial_pass and not boundary_confounded and bool(par_frame.success.all())
    selected_architecture = "M2_STATIC_PI_REGIONALIZATION" if mpr_supported else "M0_NO_SON"

    all_pred.to_parquet(OUT / "static_pi_oof_predictions.parquet", index=False)
    par_frame.to_parquet(OUT / "static_pi_fold_parameters.parquet", index=False)
    pd.DataFrame(temporal_comparisons + spatial_comparisons).to_parquet(OUT / "static_pi_paired_bootstrap.parquet", index=False)
    metrics = pd.DataFrame([
        {"architecture": "M2_STATIC_PI_REGIONALIZATION", "holdout_type": kind, **metric_values(group)}
        for kind, group in all_pred.groupby("holdout_type")
    ])
    metrics.to_parquet(OUT / "static_pi_performance_metrics.parquet", index=False)

    selected_temporal = temporal_pred if mpr_supported else parent_pred.loc[parent_pred.holdout_type.eq("TEMPORAL")].copy()
    trigger_frame, dynamic_triggered = dynamic_trigger_audit(selected_temporal, monthly)
    trigger_frame.to_parquet(OUT / "dynamic_delivery_trigger_audit.parquet", index=False)
    checks = {
        "five_covariates_exact": len(x_columns) == 5, "all_fits_success": bool(par_frame.success.all()),
        "no_station_or_reach_embedding": True, "only_pi_regionalized": True,
        "temporal_gate_pass": temporal_pass, "spatial_gate_pass": spatial_pass,
        "boundary_not_confounded": not boundary_confounded,
    }
    status = "PASS_STAGE15_ARCHITECTURE_LOCKED" if all(value for key, value in checks.items() if key not in {"temporal_gate_pass", "spatial_gate_pass", "boundary_not_confounded"}) else "FAIL_STAGE15"
    decision = {
        "stage": "20260824_15", "status": status, "parent_architecture": "M0_NO_SON",
        "selected_architecture": selected_architecture, "static_pi_supported": mpr_supported,
        "temporal_gate_pass": temporal_pass, "spatial_gate_pass": spatial_pass,
        "boundary_confounded": boundary_confounded, "dynamic_delivery_triggered": dynamic_triggered,
        "dynamic_triggered_features": trigger_frame.loc[trigger_frame.scope.eq("COMBINED") & trigger_frame.triggered.fillna(False), "feature"].tolist(),
        "covariate_columns": x_columns, "comparisons": temporal_comparisons + spatial_comparisons,
        "parameter_medians": par_frame.select_dtypes(include=[np.number]).median().to_dict(),
        "checks": checks, "hashes": {"contract": sha256(CONTRACT), "covariates": sha256(COVARIATES)},
        "authorized_successor": "20260824_16" if dynamic_triggered else "20260824_17",
    }
    write_json(REPORTS / "stage15_decision.json", decision)
    write_json(REPORTS / "selected_static_architecture_lock.json", {
        "stage": "20260824_15", "status": "LOCKED", "selected_architecture": selected_architecture,
        "dynamic_delivery_triggered": dynamic_triggered,
    })
    report = f"""# 20260824_15 static land-delivery regionalization

## Decision

`{status}`  
Selected architecture: **{selected_architecture}**.

Only `pi_E` was allowed to vary by Reach through five frozen catchment attributes. The model has no
station, Reach-ID or tree embedding and cannot alter the canonical water balance. Temporal gate:
`{temporal_pass}`; spatial/natural-expansion gate: `{spatial_pass}`; boundary confounding:
`{boundary_confounded}`. Static regionalization support is therefore `{mpr_supported}`.

The residual audit used de-seasoned anomalies of canonical upper storage, lower storage, effective
excess and routed fast fraction. The pre-registered trigger for a separate dynamic-delivery experiment
is `{dynamic_triggered}`. Stage 16 is {'authorized' if dynamic_triggered else 'closed without fitting'}.
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    if status.startswith("FAIL"):
        raise RuntimeError(decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
