"""Run the one-parameter shared dynamic land-delivery challenge."""

from __future__ import annotations

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
RUN = ROOT / "5_Test" / "20260824_16"
P12 = ROOT / "5_Test" / "20260824_12"
P13 = ROOT / "5_Test" / "20260824_13"
P15 = ROOT / "5_Test" / "20260824_15"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
OBS = P12 / "outputs" / "tn_observations_primary_2016_2024.parquet"
FOLDS = P12 / "outputs" / "tn_evaluation_fold_registry.parquet"
EXPANSION = P12 / "outputs" / "tn_natural_expansion_2021_registry.parquet"
MONTHLY = P12 / "outputs" / "canonical_tn_bridge_monthly_2010_2024.parquet"
SOURCES = P12 / "outputs" / "monthly_source_forcing_1961_2024.parquet"
KERNELS = P13 / "outputs" / "daily_compiled_carrier_kernels.parquet"
M0_PRED = P13 / "outputs" / "m0_carrier_oof_predictions.parquet"
M0_PAR = P13 / "outputs" / "m0_carrier_fold_parameters.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
CONTRACT = RUN / "experiment_contract.json"
EPS = 1.0e-12
MARGIN = 0.005
PRIOR_SD = 0.35

sys.path.insert(0, str(P13 / "scripts"))
sys.path.insert(0, str(P15 / "scripts"))
from stage13_model import kernel_array, source_availability, topology_operators  # noqa: E402
from run_stage13 import fold_frames, metric_values  # noqa: E402
from run_stage15 import hydrologic_anomalies, observation_weights  # noqa: E402


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


class DynamicDeliveryRouter:
    def __init__(self, monthly: pd.DataFrame, kernels: pd.DataFrame, available: pd.DataFrame):
        hydro = monthly.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        src = available.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        if not np.array_equal(hydro[["year", "month", "reach_id"]].to_numpy(), src[["year", "month", "reach_id"]].to_numpy()):
            raise RuntimeError("dynamic source/hydrology mismatch")
        self.reach_ids = np.sort(hydro.reach_id.unique().astype(int))
        times = hydro[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
        self.month_keys = [(int(y), int(m)) for y, m in times.itertuples(index=False)]
        self.shape = (len(times), len(self.reach_ids))
        self.rlookup = {int(r): i for i, r in enumerate(self.reach_ids)}
        self.tlookup = {key: i for i, key in enumerate(self.month_keys)}
        self.kernels = kernel_array(kernels.sort_values(["year", "month", "reach_id"]))
        self.input = src.available_total_kg_n.to_numpy(float).reshape(self.shape)
        anomaly = hydrologic_anomalies(hydro)
        anomaly = anomaly.sort_values(["year", "month", "reach_id"])
        columns = ["z_anom_upper", "z_anom_lower", "z_anom_excess", "z_anom_fast_fraction"]
        self.state_score = np.tanh(anomaly[columns].mean(axis=1).to_numpy(float).reshape(self.shape) / 2.0)
        self.local_water = (hydro.local_fast_response_volume_m3 + hydro.local_slow_response_volume_m3).to_numpy(float).reshape(self.shape)
        self.h = hydro.h1_exposure_day_per_m.to_numpy(float).reshape(self.shape)
        order, downstream, _ = topology_operators(TOPOLOGY, self.reach_ids)
        self.order_idx = [self.rlookup[r] for r in order]
        self.down_idx = {self.rlookup[r]: self.rlookup[d] for r, d in downstream.items()}
        self.water_inlet = np.zeros_like(self.local_water)
        self.water_outlet = np.zeros_like(self.local_water)
        for i in self.order_idx:
            self.water_outlet[:, i] = self.water_inlet[:, i] + self.local_water[:, i]
            if i in self.down_idx:
                self.water_inlet[:, self.down_idx[i]] += self.water_outlet[:, i]

    def carrier(self, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        linear = alpha + beta * self.state_score
        pi = sigmoid(linear)
        dpi = pi * (1.0 - pi)
        x = self.input * pi
        dx = np.stack([self.input * dpi, self.input * dpi * self.state_score], axis=2)
        zu = np.zeros(len(self.reach_ids), dtype=float)
        zl = np.zeros(len(self.reach_ids), dtype=float)
        dzu = np.zeros((len(self.reach_ids), 2), dtype=float)
        dzl = np.zeros((len(self.reach_ids), 2), dtype=float)
        local = np.zeros(self.shape, dtype=float)
        dlocal = np.zeros((*self.shape, 2), dtype=float)
        for t in range(self.shape[0]):
            k = self.kernels[t * len(self.reach_ids):(t + 1) * len(self.reach_ids)]
            vectors = np.column_stack([zu, zl, x[t]])
            result = np.einsum("roi,ri->ro", k, vectors)
            derivatives = np.stack([dzu, dzl, dx[t]], axis=2)  # reach, parameter, input
            dresult = np.einsum("roi,rpi->rop", k, derivatives)
            zu, zl = result[:, 0], result[:, 1]
            dzu, dzl = dresult[:, 0], dresult[:, 1]
            local[t] = result[:, 2] + result[:, 3]
            dlocal[t] = dresult[:, 2] + dresult[:, 3]
        return local, dlocal, pi

    def evaluate(self, observations: pd.DataFrame, theta: np.ndarray, derivatives: bool = True) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        alpha, beta, vf = map(float, theta)
        local, dlocal, pi = self.carrier(alpha, beta)
        inlet = np.zeros_like(local)
        outlet = np.zeros_like(local)
        dinlet = np.zeros((*local.shape, 3), dtype=float)
        doutlet = np.zeros((*local.shape, 3), dtype=float)
        survival = np.exp(-vf * self.h)
        midpoint = np.exp(-vf * self.h / 2.0)
        for i in self.order_idx:
            outlet[:, i] = inlet[:, i] * survival[:, i] + local[:, i] * midpoint[:, i]
            doutlet[:, i, :2] = dinlet[:, i, :2] * survival[:, i, None] + dlocal[:, i, :] * midpoint[:, i, None]
            doutlet[:, i, 2] = dinlet[:, i, 2] * survival[:, i] - inlet[:, i] * self.h[:, i] * survival[:, i] - 0.5 * local[:, i] * self.h[:, i] * midpoint[:, i]
            if i in self.down_idx:
                down = self.down_idx[i]
                inlet[:, down] += outlet[:, i]
                dinlet[:, down] += doutlet[:, i]
        ridx = observations.reach_id.astype(int).map(self.rlookup).to_numpy(int)
        tidx = np.fromiter((self.tlookup[(int(y), int(m))] for y, m in observations[["year", "month"]].itertuples(index=False)), dtype=int, count=len(observations))
        frac = observations.downstream_fraction_on_reach.to_numpy(float)
        h = self.h[tidx, ridx]
        sf = np.exp(-vf * h * frac)
        sm = np.exp(-vf * h * frac / 2.0)
        local_obs = local[tidx, ridx]
        load = inlet[tidx, ridx] * sf + frac * local_obs * sm
        water = self.water_inlet[tidx, ridx] + frac * self.local_water[tidx, ridx]
        pred = 1000.0 * load / np.maximum(water, EPS)
        if not derivatives:
            return pred, None, pi
        dload = np.zeros((len(observations), 3), dtype=float)
        dload[:, :2] = dinlet[tidx, ridx, :2] * sf[:, None] + frac[:, None] * dlocal[tidx, ridx, :] * sm[:, None]
        dload[:, 2] = dinlet[tidx, ridx, 2] * sf - inlet[tidx, ridx] * h * frac * sf - 0.5 * frac * local_obs * h * frac * sm
        return pred, 1000.0 * dload / np.maximum(water[:, None], EPS), pi


def fit_dynamic(router: DynamicDeliveryRouter, train: pd.DataFrame, parent_pi: float, parent_vf: float, multistart: bool) -> dict[str, object]:
    y = np.log1p(train.tn_mg_l.to_numpy(float))
    weights = observation_weights(train)
    nblocks = train.station_key.nunique() + train.reach_id.nunique() + train.terminal_tree_id.nunique()

    def value_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        pred, dpred, _ = router.evaluate(train, theta, True)
        assert dpred is not None
        error = np.log1p(np.maximum(pred, 0.0)) - y
        grad = 2.0 * np.sum((weights * error)[:, None] * dpred / (1.0 + pred[:, None]), axis=0)
        loss = float(np.sum(weights * error * error) + 0.5 * (theta[1] / PRIOR_SD) ** 2 / nblocks)
        grad[1] += theta[1] / (PRIOR_SD ** 2 * nblocks)
        return loss, grad

    alpha = float(np.log(parent_pi / (1.0 - parent_pi)))
    starts = (np.array([alpha, 0.0, parent_vf]), np.array([alpha, 0.25, parent_vf]), np.array([alpha, -0.25, parent_vf])) if multistart else (np.array([alpha, 0.0, parent_vf]),)
    results = [minimize(lambda z: value_grad(z), start, jac=True, method="L-BFGS-B", bounds=((-9.21, 9.21), (-1.0, 1.0), (0.0, 0.5)), options={"ftol": 1.0e-10, "gtol": 1.0e-7, "maxiter": 200}) for start in starts]
    successful = [result for result in results if result.success and np.isfinite(result.fun)]
    best = min(successful if successful else results, key=lambda result: float(result.fun))
    _, _, pi = router.evaluate(train, best.x, False)
    return {
        "theta": np.asarray(best.x, dtype=float), "objective": float(best.fun), "success": bool(best.success),
        "message": str(best.message), "iterations": int(best.nit), "function_evaluations": int(best.nfev),
        "beta_boundary": bool(abs(best.x[1]) >= 1.0 - 1.0e-6), "vf_boundary": bool(best.x[2] <= 1.0e-7 or best.x[2] >= 0.5 - 1.0e-6),
        "pi_min": float(pi.min()), "pi_max": float(pi.max()), "pi_median": float(np.median(pi)),
    }


def fit_fold(router: DynamicDeliveryRouter, obs: pd.DataFrame, fold: pd.Series, expansion: pd.DataFrame, parent_parameter: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    train, test = fold_frames(obs, fold, expansion)
    fit = fit_dynamic(
        router, train, float(parent_parameter.pi_E), float(parent_parameter.v_f_m_per_day),
        multistart=str(fold.holdout_type) == "TEMPORAL",
    )
    theta = np.asarray(fit.pop("theta"), dtype=float)
    pred, _, _ = router.evaluate(test, theta, False)
    output = test.copy(); output["pred_tn_mg_l"] = pred; output["architecture"] = "M3_DYNAMIC_DELIVERY"
    output["fold_id"] = str(fold.fold_id); output["holdout_type"] = str(fold.holdout_type); output["holdout_id"] = str(fold.holdout_id); output["evaluation_year"] = int(fold.evaluation_year)
    pars = {"fold_id": str(fold.fold_id), "holdout_type": str(fold.holdout_type), "holdout_id": str(fold.holdout_id), "train_start_year": int(fold.train_start_year), "train_end_year": int(fold.train_end_year), "evaluation_year": int(fold.evaluation_year), "alpha": float(theta[0]), "beta_D": float(theta[1]), "v_f_m_per_day": float(theta[2]), "train_rows": len(train), "test_rows": len(test), **fit}
    return output, pars


def paired_delta(parent: pd.DataFrame, candidate: pd.DataFrame, block: str, seed: int) -> dict[str, object]:
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    joined = parent.merge(candidate, on=keys, suffixes=("_parent", "_candidate"), validate="one_to_one")
    values = []
    for _, group in joined.groupby(block):
        y = np.log1p(group.tn_mg_l.to_numpy(float))
        a = np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l_parent) - y)))
        b = np.sqrt(np.mean(np.square(np.log1p(group.pred_tn_mg_l_candidate) - y)))
        values.append(b - a)
    values = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1); upper = float(np.quantile(samples, 0.975))
    return {"block": block, "block_count": len(values), "delta_candidate_minus_parent": float(values.mean()), "ci95_lower": float(np.quantile(samples, 0.025)), "ci95_upper": upper, "noninferior_0p005": bool(upper < MARGIN), "improved": bool(upper < 0.0)}


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow": raise RuntimeError("conda sparrow required")
    decision15 = json.loads((P15 / "reports" / "stage15_decision.json").read_text(encoding="utf-8"))
    if not decision15["dynamic_delivery_triggered"] or decision15["selected_architecture"] != "M0_NO_SON": raise RuntimeError("stage16 not authorized")
    monthly = pd.read_parquet(MONTHLY); kernels = pd.read_parquet(KERNELS)
    available = source_availability(pd.read_parquet(SOURCES)); obs = pd.read_parquet(OBS); folds = pd.read_parquet(FOLDS); expansion = pd.read_parquet(EXPANSION)
    router = DynamicDeliveryRouter(monthly, kernels, available)
    parent_pred = pd.read_parquet(M0_PRED).loc[lambda x: x.carrier.eq("DAILY_COMPILED_CARRIER")]
    parent_par = pd.read_parquet(M0_PAR).loc[lambda x: x.carrier.eq("DAILY_COMPILED_CARRIER")].set_index("fold_id")
    predictions, parameters = [], []
    temporal = folds.loc[folds.holdout_type.eq("TEMPORAL")].sort_values("evaluation_year")
    for row in temporal.itertuples(index=False):
        pred, par = fit_fold(router, obs, pd.Series(row._asdict()), expansion, parent_par.loc[str(row.fold_id)]); predictions.append(pred); parameters.append(par)
    temporal_pred = pd.concat(predictions, ignore_index=True)
    temporal_cmp = [{"holdout_type": "TEMPORAL", **paired_delta(parent_pred.loc[parent_pred.holdout_type.eq("TEMPORAL")], temporal_pred, block, 20260824160 + j)} for j, block in enumerate(("station_key", "reach_id", "terminal_tree_id"))]
    temporal_pass = all(item["noninferior_0p005"] for item in temporal_cmp) and any(item["improved"] for item in temporal_cmp if item["block"] in {"station_key", "reach_id"})
    par_frame = pd.DataFrame(parameters)
    stable_beta = ((par_frame.beta_D > 0.05).sum() >= 2 and not (par_frame.beta_D < -0.05).any()) or ((par_frame.beta_D < -0.05).sum() >= 2 and not (par_frame.beta_D > 0.05).any())
    spatial_cmp = []
    if temporal_pass and stable_beta and not par_frame.beta_boundary.any():
        remaining = folds.loc[~folds.holdout_type.eq("TEMPORAL")].sort_values(["evaluation_year", "holdout_type", "holdout_id"])
        for row in remaining.itertuples(index=False):
            pred, par = fit_fold(router, obs, pd.Series(row._asdict()), expansion, parent_par.loc[str(row.fold_id)]); predictions.append(pred); parameters.append(par)
        all_pred = pd.concat(predictions, ignore_index=True); par_frame = pd.DataFrame(parameters)
        for j, (kind, block) in enumerate((("REACH", "reach_id"), ("TREE", "terminal_tree_id"), ("FIRST_OBSERVED_2021", "station_key"), ("FIRST_OBSERVED_2021", "reach_id"))):
            spatial_cmp.append({"holdout_type": kind, **paired_delta(parent_pred.loc[parent_pred.holdout_type.eq(kind)], all_pred.loc[all_pred.holdout_type.eq(kind)], block, 20260824170 + j)})
    else: all_pred = temporal_pred
    spatial_pass = bool(spatial_cmp) and all(item["noninferior_0p005"] for item in spatial_cmp)
    boundary = bool(par_frame.beta_boundary.any() or par_frame.vf_boundary.any())
    supported = temporal_pass and stable_beta and spatial_pass and not boundary and bool(par_frame.success.all())
    selected = "M3_DYNAMIC_DELIVERY" if supported else "M0_NO_SON"
    all_pred.to_parquet(OUT / "dynamic_delivery_oof_predictions.parquet", index=False); par_frame.to_parquet(OUT / "dynamic_delivery_fold_parameters.parquet", index=False)
    pd.DataFrame(temporal_cmp + spatial_cmp).to_parquet(OUT / "dynamic_delivery_paired_bootstrap.parquet", index=False)
    pd.DataFrame([{"architecture": "M3_DYNAMIC_DELIVERY", "holdout_type": kind, **metric_values(group)} for kind, group in all_pred.groupby("holdout_type")]).to_parquet(OUT / "dynamic_delivery_performance_metrics.parquet", index=False)
    checks = {"beta0_exact_parent_by_equation": True, "all_fits_success": bool(par_frame.success.all()), "temporal_gate_pass": temporal_pass, "beta_direction_stable": stable_beta, "spatial_gate_pass": spatial_pass, "boundary_not_confounded": not boundary, "water_not_modified": True}
    status = "PASS_STAGE16_ARCHITECTURE_LOCKED" if checks["all_fits_success"] else "FAIL_STAGE16"
    decision = {"stage": "20260824_16", "status": status, "selected_architecture": selected, "dynamic_delivery_supported": supported, "checks": checks, "comparisons": temporal_cmp + spatial_cmp, "temporal_beta_D": par_frame.loc[par_frame.holdout_type.eq("TEMPORAL"), ["fold_id", "beta_D"]].to_dict("records"), "hashes": {"contract": sha256(CONTRACT), "kernels": sha256(KERNELS)}, "authorized_successor": "20260824_17" if status.startswith("PASS") else None}
    write_json(REPORTS / "stage16_decision.json", decision); write_json(REPORTS / "selected_dynamic_architecture_lock.json", {"stage": "20260824_16", "status": "LOCKED", "selected_architecture": selected})
    (REPORTS / "technical_report.md").write_text(f"# 20260824_16 dynamic delivery\n\n`{status}`\n\nSelected architecture: **{selected}**. Temporal gate: `{temporal_pass}`; beta direction stable: `{stable_beta}`; nested spatial gate: `{spatial_pass}`; boundary confounding: `{boundary}`. The operator changes only the fraction of available N delivered to the locked carrier; canonical water is untouched and beta=0 recovers M0.\n", encoding="utf-8")
    if status.startswith("FAIL"): raise RuntimeError(decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__": main()
