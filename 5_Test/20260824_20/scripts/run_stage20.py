"""Run the registered 2^4 TN structural ablation and optimizer audit."""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_20"
P18 = ROOT / "5_Test" / "20260824_18"
P19 = ROOT / "5_Test" / "20260824_19"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CONTRACT = RUN / "experiment_contract.json"
MONTHLY = P19 / "outputs" / "corrected_tn_bridge_monthly_2006_2024.parquet"
KERNELS = P19 / "outputs" / "corrected_daily_carrier_kernels_2006_2024.parquet"
SOURCE = P19 / "outputs" / "corrected_source_availability_2006_2024.parquet"
PARENT_PRED = P19 / "outputs" / "corrected_m3_full_oof_predictions.parquet"
PARENT_PAR = P19 / "outputs" / "corrected_m3_full_fold_parameters.parquet"
EPS = 1.0e-12
PRIOR_SD = 0.35
RIDGE = 12.0

spec = importlib.util.spec_from_file_location("stage19", P19 / "scripts" / "run_stage19.py")
s19 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(s19)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def station_weights(frame: pd.DataFrame) -> np.ndarray:
    # Stage 20 must reproduce the locked Stage 19 objective exactly.  Its
    # formal macro target gives equal one-third weight to station, Reach and
    # terminal-tree blocks; changing this here would make the ablation unfair.
    return s19.observation_weights(frame)


def configs() -> list[dict[str, object]]:
    result = []
    for calendar in ("MIRCA", "UNIFORM"):
        for pathway in (False, True):
            for hinge in (False, True):
                for lag in (False, True):
                    result.append({
                        "config_id": f"C{int(calendar == 'MIRCA')}_P{int(pathway)}_H{int(hinge)}_L{int(lag)}",
                        "calendar": calendar, "pathway": pathway, "hinge": hinge, "lag": lag,
                    })
    return result


class AblationRouter:
    def __init__(self, monthly: pd.DataFrame, kernels: pd.DataFrame, source_input: np.ndarray, config: dict[str, object]):
        hydro = monthly.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        self.config = config
        self.reach_ids = np.sort(hydro.reach_id.unique().astype(int))
        times = hydro[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
        self.month_keys = [(int(y), int(m)) for y, m in times.itertuples(index=False)]
        self.shape = (len(times), len(self.reach_ids))
        self.rlookup = {int(r): i for i, r in enumerate(self.reach_ids)}
        self.tlookup = {key: i for i, key in enumerate(self.month_keys)}
        self.kernels = s19.kernel_array(kernels.sort_values(["year", "month", "reach_id"]))
        self.input = np.asarray(source_input, dtype=float).reshape(self.shape)
        anomaly = s19.hydrologic_anomalies(hydro).sort_values(["year", "month", "reach_id"])
        cols = ["z_anom_upper", "z_anom_lower", "z_anom_excess", "z_anom_fast_fraction"]
        self.state_score = np.tanh(anomaly[cols].mean(axis=1).to_numpy(float).reshape(self.shape) / 2.0)
        self.local_water = (hydro.local_fast_response_volume_m3 + hydro.local_slow_response_volume_m3).to_numpy(float).reshape(self.shape)
        self.h = hydro.h1_exposure_day_per_m.to_numpy(float).reshape(self.shape)
        self.logq = np.log(np.maximum(hydro.routed_total_m3_s.to_numpy(float).reshape(self.shape), EPS))
        order, downstream, _ = s19.topology_operators(s19.TOPOLOGY, self.reach_ids)
        self.order_idx = [self.rlookup[r] for r in order]
        self.down_idx = {self.rlookup[r]: self.rlookup[d] for r, d in downstream.items()}
        self.water_inlet = np.zeros_like(self.local_water)
        for i in self.order_idx:
            outlet = self.water_inlet[:, i] + self.local_water[:, i]
            if i in self.down_idx:
                self.water_inlet[:, self.down_idx[i]] += outlet

    def parameter_names(self) -> list[str]:
        names = ["alpha_D", "beta_D", "v_f_m_per_day"]
        if self.config["pathway"]:
            names += ["eta_fast", "eta_slow"]
        if self.config["hinge"]:
            names += ["beta_low", "beta_high"]
        return names

    def bounds(self) -> list[tuple[float, float]]:
        bounds = [(-9.21, 9.21), (-1.0, 1.0), (0.0, 0.5)]
        if self.config["pathway"]:
            bounds += [(0.0, 1.0), (0.0, 1.0)]
        if self.config["hinge"]:
            bounds += [(-1.0, 1.0), (-1.0, 1.0)]
        return bounds

    def start(self, variant: int) -> np.ndarray:
        core = [[-4.0, 0.0, 0.05], [-1.5, 0.2, 0.22], [-6.0, -0.2, 0.35]][variant]
        values = list(core)
        if self.config["pathway"]:
            values += [0.65, 0.65]
        if self.config["hinge"]:
            values += [0.0, 0.0]
        return np.asarray(values, dtype=float)

    def carrier(self, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pi = sigmoid(alpha + beta * self.state_score)
        dpi = pi * (1.0 - pi)
        x = self.input * pi
        dx = np.stack([self.input * dpi, self.input * dpi * self.state_score], axis=2)
        if self.config["lag"]:
            rho = math.exp(-1.0 / 36.0)
            z = np.zeros(self.shape[1]); dz = np.zeros((self.shape[1], 2))
            xr = np.zeros_like(x); dxr = np.zeros_like(dx)
            for t in range(self.shape[0]):
                pre, dpre = z + x[t], dz + dx[t]
                xr[t], dxr[t] = (1.0 - rho) * pre, (1.0 - rho) * dpre
                z, dz = rho * pre, rho * dpre
            x, dx = xr, dxr
        n = self.shape[1]
        zu, zl = np.zeros(n), np.zeros(n)
        dzu, dzl = np.zeros((n, 2)), np.zeros((n, 2))
        fast, slow = np.zeros(self.shape), np.zeros(self.shape)
        dfast, dslow = np.zeros((*self.shape, 2)), np.zeros((*self.shape, 2))
        for t in range(self.shape[0]):
            k = self.kernels[t * n:(t + 1) * n]
            result = np.einsum("roi,ri->ro", k, np.column_stack([zu, zl, x[t]]))
            derivatives = np.stack([dzu, dzl, dx[t]], axis=2)
            dresult = np.einsum("roi,rpi->rop", k, derivatives)
            zu, zl = result[:, 0], result[:, 1]
            dzu, dzl = dresult[:, 0], dresult[:, 1]
            fast[t], slow[t] = result[:, 2], result[:, 3]
            dfast[t], dslow[t] = dresult[:, 2], dresult[:, 3]
        return fast, slow, dfast, dslow

    def q_features(self, obs: pd.DataFrame, train_start: int, train_end: int) -> tuple[np.ndarray, np.ndarray]:
        indices = [i for i, (y, _) in enumerate(self.month_keys) if train_start <= y <= train_end]
        center = np.median(self.logq[indices], axis=0)
        ridx = obs.reach_id.astype(int).map(self.rlookup).to_numpy(int)
        tidx = np.fromiter((self.tlookup[(int(y), int(m))] for y, m in obs[["year", "month"]].itertuples(index=False)), dtype=int, count=len(obs))
        z = self.logq[tidx, ridx] - center[ridx]
        return np.minimum(z, 0.0), np.maximum(z, 0.0)

    def log_prediction(self, obs: pd.DataFrame, theta: np.ndarray, train_start: int, train_end: int, derivatives: bool) -> tuple[np.ndarray, np.ndarray | None]:
        values = dict(zip(self.parameter_names(), map(float, theta)))
        fast, slow, dfast, dslow = self.carrier(values["alpha_D"], values["beta_D"])
        eta_f = values.get("eta_fast", 1.0); eta_s = values.get("eta_slow", 1.0)
        local = eta_f * fast + eta_s * slow
        p = len(theta)
        dlocal = np.zeros((*self.shape, p))
        dlocal[:, :, :2] = eta_f * dfast + eta_s * dslow
        cursor = 3
        if self.config["pathway"]:
            dlocal[:, :, cursor] = fast; dlocal[:, :, cursor + 1] = slow; cursor += 2
        vf = values["v_f_m_per_day"]
        inlet, outlet = np.zeros_like(local), np.zeros_like(local)
        dinlet, doutlet = np.zeros((*self.shape, p)), np.zeros((*self.shape, p))
        survival, midpoint = np.exp(-vf * self.h), np.exp(-vf * self.h / 2.0)
        for i in self.order_idx:
            outlet[:, i] = inlet[:, i] * survival[:, i] + local[:, i] * midpoint[:, i]
            doutlet[:, i] = dinlet[:, i] * survival[:, i, None] + dlocal[:, i] * midpoint[:, i, None]
            doutlet[:, i, 2] += -inlet[:, i] * self.h[:, i] * survival[:, i] - 0.5 * local[:, i] * self.h[:, i] * midpoint[:, i]
            if i in self.down_idx:
                down = self.down_idx[i]; inlet[:, down] += outlet[:, i]; dinlet[:, down] += doutlet[:, i]
        ridx = obs.reach_id.astype(int).map(self.rlookup).to_numpy(int)
        tidx = np.fromiter((self.tlookup[(int(y), int(m))] for y, m in obs[["year", "month"]].itertuples(index=False)), dtype=int, count=len(obs))
        frac = obs.downstream_fraction_on_reach.to_numpy(float)
        h = self.h[tidx, ridx]
        sf, sm = np.exp(-vf * h * frac), np.exp(-vf * h * frac / 2.0)
        local_obs = local[tidx, ridx]
        load = inlet[tidx, ridx] * sf + frac * local_obs * sm
        dload = dinlet[tidx, ridx] * sf[:, None] + frac[:, None] * dlocal[tidx, ridx] * sm[:, None]
        dload[:, 2] += -inlet[tidx, ridx] * h * frac * sf - 0.5 * frac * local_obs * h * frac * sm
        water = self.water_inlet[tidx, ridx] + frac * self.local_water[tidx, ridx]
        base = 1000.0 * load / np.maximum(water, EPS)
        dbase = 1000.0 * dload / np.maximum(water[:, None], EPS)
        log_pred = np.log1p(np.maximum(base, 0.0))
        jac = dbase / (1.0 + base[:, None])
        if self.config["hinge"]:
            low, high = self.q_features(obs, train_start, train_end)
            log_pred += values["beta_low"] * low + values["beta_high"] * high
            jac[:, cursor] = low; jac[:, cursor + 1] = high
        return log_pred, jac if derivatives else None


def fit_scipy(router: AblationRouter, train: pd.DataFrame) -> dict[str, object]:
    y = np.log1p(train.tn_mg_l.to_numpy(float)); weights = station_weights(train)
    nblocks = train.station_key.nunique() + train.reach_id.nunique() + train.terminal_tree_id.nunique()
    names = router.parameter_names()
    def value_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        pred, jac = router.log_prediction(train, theta, int(train.year.min()), int(train.year.max()), True)
        assert jac is not None
        error = pred - y
        loss = float(np.sum(weights * error * error))
        grad = 2.0 * np.sum((weights * error)[:, None] * jac, axis=0)
        for name in ("beta_D", "beta_low", "beta_high"):
            if name in names:
                i = names.index(name); loss += 0.5 * (theta[i] / PRIOR_SD) ** 2 / nblocks; grad[i] += theta[i] / (PRIOR_SD**2 * nblocks)
        return loss, grad
    fits = [minimize(value_grad, router.start(i), jac=True, method="L-BFGS-B", bounds=router.bounds(), options={"ftol": 1e-11, "gtol": 1e-8, "maxiter": 250, "maxls": 40}) for i in range(3)]
    valid = [x for x in fits if x.success and np.isfinite(x.fun)]
    best = min(valid if valid else fits, key=lambda x: float(x.fun))
    return {"theta": np.asarray(best.x), "objective": float(best.fun), "success": bool(best.success), "message": str(best.message), "iterations": int(best.nit), "nfev": int(best.nfev)}


def p2_effects(train: pd.DataFrame, train_log_pred: np.ndarray) -> dict[str, float]:
    residual = np.log1p(train.tn_mg_l.to_numpy(float)) - train_log_pred
    temp = pd.DataFrame({"station_key": train.station_key.to_numpy(), "residual": residual})
    out = {}
    for station, block in temp.groupby("station_key"):
        out[str(station)] = float(block.residual.sum() / (len(block) + RIDGE))
    return out


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame.tn_mg_l.to_numpy(float); p = frame.pred_tn_mg_l.to_numpy(float)
    err = p - y; denom = np.sum(np.square(y - y.mean()))
    station = [np.sqrt(np.mean(np.square(np.log1p(g.pred_tn_mg_l) - np.log1p(g.tn_mg_l)))) for _, g in frame.groupby("station_key")]
    reach = [np.sqrt(np.mean(np.square(np.log1p(g.pred_tn_mg_l) - np.log1p(g.tn_mg_l)))) for _, g in frame.groupby("reach_id")]
    tree = [np.sqrt(np.mean(np.square(np.log1p(g.pred_tn_mg_l) - np.log1p(g.tn_mg_l)))) for _, g in frame.groupby("terminal_tree_id")]
    return {"n": len(frame), "stations": frame.station_key.nunique(), "rmse_mg_l": float(np.sqrt(np.mean(err**2))), "nse": float(1 - np.sum(err**2) / denom), "r2": float(np.corrcoef(y, p)[0, 1] ** 2), "station_macro_log_rmse": float(np.mean(station)), "reach_macro_log_rmse": float(np.mean(reach)), "tree_macro_log_rmse": float(np.mean(tree))}


def bootstrap_delta(candidate: pd.DataFrame, parent: pd.DataFrame, reps: int = 4000) -> dict[str, float | bool]:
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l", "layer"]
    joined = parent[keys + ["pred_tn_mg_l"]].merge(candidate[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_p", "_c"), validate="one_to_one")
    by_station = []
    for _, g in joined.groupby("station_key"):
        y = np.log1p(g.tn_mg_l.to_numpy(float)); ep = np.mean((np.log1p(g.pred_tn_mg_l_p) - y) ** 2); ec = np.mean((np.log1p(g.pred_tn_mg_l_c) - y) ** 2)
        by_station.append((ep, ec))
    arr = np.asarray(by_station); point = float(np.mean(np.sqrt(arr[:, 1])) - np.mean(np.sqrt(arr[:, 0])))
    rng = np.random.default_rng(2026082420); idx = rng.integers(0, len(arr), size=(reps, len(arr)))
    draws = np.mean(np.sqrt(arr[idx, 1]), axis=1) - np.mean(np.sqrt(arr[idx, 0]), axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"delta_station_macro_log_rmse": point, "ci95_lower": float(lo), "ci95_upper": float(hi), "improved": bool(hi < 0), "noninferior": bool(hi < 0.005)}


def torch_parent_audit(router: AblationRouter, train: pd.DataFrame, test: pd.DataFrame, scipy_theta: np.ndarray) -> dict[str, object]:
    """Independent float64 torch evaluation at the SciPy optimum.

    Optimization is additionally attempted from the registered cold start. The
    strict equation-equivalence check is separated from optimizer convergence.
    """
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    # The independent torch equation is limited to the exact parent (no optional factors).
    if router.config["pathway"] or router.config["hinge"] or router.config["lag"]:
        raise ValueError("torch audit requires the exact parent")
    device = torch.device("cpu")
    inp = torch.tensor(router.input, device=device); score = torch.tensor(router.state_score, device=device)
    kernels = torch.tensor(router.kernels.reshape(router.shape[0], router.shape[1], 4, 3), device=device)
    h = torch.tensor(router.h, device=device); lw = torch.tensor(router.local_water, device=device); wi = torch.tensor(router.water_inlet, device=device)
    def evaluate(obs: pd.DataFrame, physical: torch.Tensor) -> torch.Tensor:
        alpha, beta, vf = physical[0], physical[1], physical[2]
        x = inp * torch.sigmoid(alpha + beta * score)
        zu = torch.zeros(router.shape[1]); zl = torch.zeros(router.shape[1]); local_rows = []
        for t in range(router.shape[0]):
            result = torch.einsum("roi,ri->ro", kernels[t], torch.stack([zu, zl, x[t]], dim=1))
            zu, zl = result[:, 0], result[:, 1]; local_rows.append(result[:, 2] + result[:, 3])
        local = torch.stack(local_rows)
        inlet = [torch.zeros(router.shape[0]) for _ in range(router.shape[1])]
        outlet = [None for _ in range(router.shape[1])]
        for i in router.order_idx:
            outlet[i] = inlet[i] * torch.exp(-vf * h[:, i]) + local[:, i] * torch.exp(-vf * h[:, i] / 2)
            if i in router.down_idx:
                d = router.down_idx[i]; inlet[d] = inlet[d] + outlet[i]
        ridx = obs.reach_id.astype(int).map(router.rlookup).to_numpy(int)
        tidx = np.fromiter((router.tlookup[(int(y), int(m))] for y, m in obs[["year", "month"]].itertuples(index=False)), dtype=int, count=len(obs))
        frac = torch.tensor(obs.downstream_fraction_on_reach.to_numpy(float))
        ridx_t, tidx_t = torch.tensor(ridx), torch.tensor(tidx)
        hh = h[tidx_t, ridx_t]
        load = torch.stack(inlet)[ridx_t, tidx_t] * torch.exp(-vf * hh * frac) + frac * local[tidx_t, ridx_t] * torch.exp(-vf * hh * frac / 2)
        water = wi[tidx_t, ridx_t] + frac * lw[tidx_t, ridx_t]
        return torch.log1p(1000 * load / torch.clamp(water, min=EPS))
    physical = torch.tensor(scipy_theta)
    torch_train = evaluate(train, physical).detach().numpy(); torch_test = evaluate(test, physical).detach().numpy()
    np_train, _ = router.log_prediction(train, scipy_theta, int(train.year.min()), int(train.year.max()), False)
    np_test, _ = router.log_prediction(test, scipy_theta, int(train.year.min()), int(train.year.max()), False)
    y = torch.tensor(np.log1p(train.tn_mg_l.to_numpy(float))); w = torch.tensor(station_weights(train))
    nblocks = train.station_key.nunique() + train.reach_id.nunique() + train.terminal_tree_id.nunique()
    scipy_obj = float(np.sum(station_weights(train) * (np_train - y.numpy()) ** 2) + 0.5 * (scipy_theta[1] / PRIOR_SD) ** 2 / nblocks)
    torch_obj = float(torch.sum(w * (torch.tensor(torch_train) - y) ** 2) + 0.5 * (physical[1] / PRIOR_SD) ** 2 / nblocks)
    # Cold-start AdamW followed by full-batch LBFGS, using bounded logistic
    # transforms so the physical parameter space is identical to SciPy.
    lower = torch.tensor([-9.21, -1.0, 0.0]); upper = torch.tensor([9.21, 1.0, 0.5])
    def raw_to_physical(raw: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * torch.sigmoid(raw)
    cold = torch.tensor([-4.0, 0.0, 0.05])
    fraction = torch.clamp((cold - lower) / (upper - lower), 1e-8, 1 - 1e-8)
    raw = torch.nn.Parameter(torch.log(fraction / (1 - fraction)))
    def objective() -> torch.Tensor:
        ph = raw_to_physical(raw); lp = evaluate(train, ph)
        return torch.sum(w * (lp - y) ** 2) + 0.5 * (ph[1] / PRIOR_SD) ** 2 / nblocks
    adam = torch.optim.AdamW([raw], lr=0.04, weight_decay=1e-6)
    for _ in range(250):
        adam.zero_grad(); loss = objective(); loss.backward(); torch.nn.utils.clip_grad_norm_([raw], 10.0); adam.step()
    lbfgs = torch.optim.LBFGS([raw], lr=1.0, max_iter=100, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        lbfgs.zero_grad(); loss = objective(); loss.backward(); return loss
    lbfgs.step(closure)
    torch_theta = raw_to_physical(raw).detach()
    optimized_obj = float(objective().detach())
    optimized_test = evaluate(test, torch_theta).detach().numpy()
    return {
        "equation_objective_abs_difference": abs(torch_obj - scipy_obj),
        "equation_train_prediction_max_abs_log_difference": float(np.max(np.abs(torch_train - np_train))),
        "equation_test_prediction_max_abs_log_difference": float(np.max(np.abs(torch_test - np_test))),
        "scipy_objective": scipy_obj, "torch_objective_at_scipy_solution": torch_obj,
        "torch_adamw_lbfgs_objective": optimized_obj,
        "torch_minus_scipy_objective": optimized_obj - scipy_obj,
        "torch_adamw_lbfgs_parameters": torch_theta.numpy().tolist(),
        "optimized_test_prediction_max_abs_log_difference": float(np.max(np.abs(optimized_test - np_test))),
        "equations_equivalent_1e_8": bool(max(abs(torch_obj - scipy_obj), np.max(np.abs(torch_train - np_train)), np.max(np.abs(torch_test - np_test))) <= 1e-8),
        "optimizer_same_minimum_1e_8": bool(abs(optimized_obj - scipy_obj) <= 1e-8),
    }


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    parent = json.loads((P19 / "reports" / "stage19_full_validation.json").read_text(encoding="utf-8"))
    if parent["status"] != "PASS_STAGE19_READY_FOR_20260824_20":
        raise RuntimeError("stage19 parent is not locked")
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_parquet(MONTHLY); kernels = pd.read_parquet(KERNELS)
    source = pd.read_parquet(SOURCE).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    frozen_parent = pd.read_parquet(PARENT_PAR).loc[lambda x: x.replicate.eq("A") & x.holdout_type.eq("TEMPORAL")].set_index("fold_id")
    obs = s19.build_observations(); folds = s19.build_folds(obs, "temporal")
    mirca = source.available_total_kg_n.to_numpy(float)
    uniform_frame = source[["reach_id", "year", "month"]].copy()
    uniform_frame["annual"] = source.groupby(["reach_id", "year"]).available_total_kg_n.transform("sum")
    uniform_frame["nmonth"] = source.groupby(["reach_id", "year"]).month.transform("count")
    uniform = (uniform_frame.annual / uniform_frame.nmonth).to_numpy(float)
    calendar_audit = {"mirca_total_kg_n": float(mirca.sum()), "uniform_total_kg_n": float(uniform.sum()), "absolute_difference_kg_n": float(abs(mirca.sum() - uniform.sum())), "max_reach_year_closure_kg_n": float(pd.DataFrame({"reach_id": source.reach_id, "year": source.year, "delta": mirca - uniform}).groupby(["reach_id", "year"]).delta.sum().abs().max())}
    predictions, parameters = [], []
    router_map = {}
    for ci, config in enumerate(configs(), 1):
        router = AblationRouter(monthly, kernels, mirca if config["calendar"] == "MIRCA" else uniform, config); router_map[config["config_id"]] = router
        for _, fold in folds.iterrows():
            train, test = s19.fold_frames(obs, fold)
            if config["config_id"] == "C1_P0_H0_L0":
                inherited = frozen_parent.loc[str(fold.fold_id)]
                theta = inherited[["alpha", "beta_D", "v_f_m_per_day"]].to_numpy(float)
                fit = {"objective": float(inherited.objective), "success": bool(inherited.success), "message": "exact frozen Stage19 parent", "iterations": int(inherited.iterations), "nfev": int(inherited.function_evaluations)}
            else:
                fit = fit_scipy(router, train); theta = fit.pop("theta")
            train_log, _ = router.log_prediction(train, theta, int(fold.train_start_year), int(fold.train_end_year), False)
            effects = p2_effects(train, train_log)
            test_log, _ = router.log_prediction(test, theta, int(fold.train_start_year), int(fold.train_end_year), False)
            for layer in ("P1", "P2"):
                lp = test_log.copy()
                if layer == "P2":
                    lp += np.asarray([effects.get(str(x), 0.0) for x in test.station_key])
                out = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
                out["pred_tn_mg_l"] = np.maximum(np.expm1(lp), 0.0); out["fold_id"] = str(fold.fold_id); out["layer"] = layer; out["config_id"] = config["config_id"]
                predictions.append(out)
            row = {"config_id": config["config_id"], "calendar": config["calendar"], "pathway": config["pathway"], "hinge": config["hinge"], "lag": config["lag"], "fold_id": str(fold.fold_id), **fit}
            row.update(dict(zip(router.parameter_names(), map(float, theta)))); row["station_effect_count"] = len(effects)
            row["any_boundary"] = bool(any(abs(v - lo) < 1e-6 or abs(v - hi) < 1e-6 for v, (lo, hi) in zip(theta, router.bounds())))
            parameters.append(row)
        print(json.dumps({"completed_config": config["config_id"], "index": ci, "total": 16, **dict(zip(("rss_gib", "peak_rss_gib"), s19.current_memory_gib()))}), flush=True)
    pred = pd.concat(predictions, ignore_index=True); par = pd.DataFrame(parameters)
    metric_rows = []
    for (config_id, layer), block in pred.groupby(["config_id", "layer"]):
        metric_rows.append({"config_id": config_id, "layer": layer, "year": "ALL", **metric_values(block)})
        for year, annual in block.groupby("year"):
            metric_rows.append({"config_id": config_id, "layer": layer, "year": str(int(year)), **metric_values(annual)})
    metrics = pd.DataFrame(metric_rows)
    parent_id = "C1_P0_H0_L0"
    parent_p1 = pred.loc[(pred.config_id == parent_id) & (pred.layer == "P1")]
    comparisons = []
    for config_id in sorted(pred.config_id.unique()):
        candidate = pred.loc[(pred.config_id == config_id) & (pred.layer == "P1")]
        comparisons.append({"config_id": config_id, **bootstrap_delta(candidate, parent_p1)})
    comparisons = pd.DataFrame(comparisons)
    rank = metrics.loc[(metrics.layer == "P1") & (metrics.year == "ALL")].sort_values("station_macro_log_rmse").reset_index(drop=True)
    selected_id = str(rank.iloc[0].config_id)
    p19pred = pd.read_parquet(PARENT_PRED).loc[lambda x: x.holdout_type.eq("TEMPORAL")]
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    reproduction = p19pred[keys + ["pred_tn_mg_l"]].merge(parent_p1[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_19", "_20"), validate="one_to_one")
    parent_reproduction_max = float(np.max(np.abs(reproduction.pred_tn_mg_l_19 - reproduction.pred_tn_mg_l_20)))
    # Independent equation audit on T3, the largest training fold.
    fold = folds.loc[folds.fold_id.eq("T3")].iloc[0]; train, test = s19.fold_frames(obs, fold)
    ptheta = par.loc[(par.config_id == parent_id) & (par.fold_id == "T3"), ["alpha_D", "beta_D", "v_f_m_per_day"]].iloc[0].to_numpy(float)
    optimizer = torch_parent_audit(router_map[parent_id], train, test, ptheta)
    selected_comp = comparisons.loc[comparisons.config_id.eq(selected_id)].iloc[0]
    selected_par = par.loc[par.config_id.eq(selected_id)]
    validation = {
        "stage": "20260824_20", "status": "PASS_STAGE20_READY_FOR_20260824_21",
        "calendar_mass_audit": calendar_audit,
        "parent_reproduction_max_abs_mg_l": parent_reproduction_max,
        "optimizer_audit": optimizer,
        "selected_config_id": selected_id,
        "selected_structure": next(x for x in configs() if x["config_id"] == selected_id),
        "selected_vs_parent": selected_comp.to_dict(),
        "checks": {"sixteen_configs": pred.config_id.nunique() == 16, "three_temporal_folds": pred.fold_id.nunique() == 3, "p1_p2_both_present": set(pred.layer) == {"P1", "P2"}, "calendar_mass_closed": calendar_audit["max_reach_year_closure_kg_n"] <= 1e-8, "parent_exactly_reproduced": parent_reproduction_max <= 1e-8, "all_fits_success": bool(par.success.all()), "optimizer_equations_equivalent": bool(optimizer["equations_equivalent_1e_8"]), "memory_below_hard_stop": s19.current_memory_gib()[1] < 16.0},
        "boundary_confounded": bool(selected_par.any_boundary.sum() >= 2),
        "input_hashes": {str(x): sha256(x) for x in (MONTHLY, KERNELS, SOURCE, PARENT_PRED, PARENT_PAR, CONTRACT)},
        "authorized_successor": "20260824_21"
    }
    if not all(validation["checks"].values()):
        validation["status"] = "FAIL_STAGE20"
        validation["authorized_successor"] = None
    atomic_parquet(pred, OUT / "structural_ablation_oof_predictions.parquet")
    atomic_parquet(par, OUT / "structural_ablation_fold_parameters.parquet")
    atomic_parquet(metrics, OUT / "structural_ablation_metrics.parquet")
    atomic_parquet(comparisons, OUT / "structural_ablation_paired_comparisons.parquet")
    write_json(REPORTS / "stage20_final_validation.json", validation)
    write_json(REPORTS / "optimizer_equation_audit.json", optimizer)
    lines = ["# 20260824_20 公平结构消融与优化器审计", "", f"最终状态：`{validation['status']}`。", "", f"16个预注册组合中，P1 station-macro log-RMSE最低的是 `{selected_id}`。", f"其相对修正M3 parent的差为 `{selected_comp.delta_station_macro_log_rmse:.5f}`，95% CI `{selected_comp.ci95_lower:.5f}`–`{selected_comp.ci95_upper:.5f}`。", f"Stage19 parent逐值复现最大差 `{parent_reproduction_max:.3e} mg/L`；独立PyTorch方程与SciPy最大差不超过 `{max(optimizer['equation_objective_abs_difference'], optimizer['equation_train_prediction_max_abs_log_difference'], optimizer['equation_test_prediction_max_abs_log_difference']):.3e}`。", "", "| rank | config | P1 station-macro log-RMSE | RMSE mg/L | NSE |", "|---:|---|---:|---:|---:|"]
    for i, row in rank.iterrows():
        lines.append(f"| {i+1} | {row.config_id} | {row.station_macro_log_rmse:.4f} | {row.rmse_mg_l:.3f} | {row.nse:.3f} |")
    lines += ["", "`L1`只表示固定36个月通用source-availability memory，不得解释为已识别的地下水年龄。P2没有参与结构选择。"]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
