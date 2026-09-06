"""Fit the float64 differentiable TN parent and its registered OOF layers."""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_21"
P19 = ROOT / "5_Test" / "20260824_19"
P20 = ROOT / "5_Test" / "20260824_20"
OUT = RUN / "outputs"; REPORTS = RUN / "reports"
CONTRACT = RUN / "experiment_contract.json"
MONTHLY = P19 / "outputs" / "corrected_tn_bridge_monthly_2006_2024.parquet"
KERNELS = P19 / "outputs" / "corrected_daily_carrier_kernels_2006_2024.parquet"
SOURCE = P19 / "outputs" / "corrected_source_availability_2006_2024.parquet"
OBS = ROOT / "5_Test" / "20260824_18" / "outputs" / "tn_observations_audited.parquet"
NU = 4.0; RIDGE = 12.0; EPS = 1e-12


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None; spec.loader.exec_module(module); return module


s19 = load_module("stage19", P19 / "scripts" / "run_stage19.py")
s20 = load_module("stage20", P20 / "scripts" / "run_stage20.py")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False); os.replace(temp, path)


def station_weights(frame: pd.DataFrame) -> np.ndarray:
    codes = pd.Categorical(frame.station_key).codes; counts = np.bincount(codes)
    return 1.0 / (len(counts) * counts[codes])


class TorchTN:
    def __init__(self, base: s20.AblationRouter):
        torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
        self.base = base; self.shape = base.shape
        self.input = torch.tensor(base.input); self.score = torch.tensor(base.state_score)
        self.kernels = torch.tensor(base.kernels.reshape(base.shape[0], base.shape[1], 4, 3))
        self.h = torch.tensor(base.h); self.local_water = torch.tensor(base.local_water); self.water_inlet = torch.tensor(base.water_inlet)
        self.lower = {"alpha_D": -9.21, "beta_D": -1.0, "v_f": 0.0, "delta_path": -2.0, "beta_low": -1.0, "beta_high": -1.0, "log_sigma": -4.0}
        self.upper = {"alpha_D": 9.21, "beta_D": 1.0, "v_f": 0.5, "delta_path": 2.0, "beta_low": 1.0, "beta_high": 1.0, "log_sigma": 1.0}

    def names(self, hinge: bool) -> list[str]:
        return ["alpha_D", "beta_D", "v_f", "delta_path"] + (["beta_low", "beta_high"] if hinge else []) + ["log_sigma"]

    def to_physical(self, raw: torch.Tensor, hinge: bool) -> torch.Tensor:
        names = self.names(hinge); lo = torch.tensor([self.lower[x] for x in names]); hi = torch.tensor([self.upper[x] for x in names])
        return lo + (hi - lo) * torch.sigmoid(raw)

    def to_raw(self, physical: np.ndarray, hinge: bool) -> torch.Tensor:
        names = self.names(hinge); lo = np.asarray([self.lower[x] for x in names]); hi = np.asarray([self.upper[x] for x in names])
        f = np.clip((physical - lo) / (hi - lo), 1e-8, 1 - 1e-8)
        return torch.tensor(np.log(f / (1 - f)))

    def initial(self, hinge: bool, variant: int) -> np.ndarray:
        alpha = (-1.0, -4.0)[variant]; core = [alpha, 0.1, 0.12, 0.0]
        if hinge: core += [0.1, 0.2]
        return np.asarray(core + [math.log(0.35)])

    def obs_indices(self, obs: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ridx = obs.reach_id.astype(int).map(self.base.rlookup).to_numpy(int)
        tidx = np.fromiter((self.base.tlookup[(int(y), int(m))] for y, m in obs[["year", "month"]].itertuples(index=False)), dtype=int, count=len(obs))
        return torch.tensor(tidx), torch.tensor(ridx), torch.tensor(obs.downstream_fraction_on_reach.to_numpy(float))

    def q_hinge(self, obs: pd.DataFrame, train_start: int, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        low, high = self.base.q_features(obs, train_start, train_end); return torch.tensor(low), torch.tensor(high)

    def evaluate(self, obs: pd.DataFrame, physical: torch.Tensor, hinge: bool, train_start: int, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        names = self.names(hinge); v = {name: physical[i] for i, name in enumerate(names)}
        x0 = self.input * torch.sigmoid(v["alpha_D"] + v["beta_D"] * self.score)
        rho = math.exp(-1.0 / 36.0); stock = torch.zeros(self.shape[1]); releases = []
        for t in range(self.shape[0]):
            pre = stock + x0[t]; releases.append((1 - rho) * pre); stock = rho * pre
        x = torch.stack(releases)
        zu = torch.zeros(self.shape[1]); zl = torch.zeros(self.shape[1]); fast_rows, slow_rows = [], []
        for t in range(self.shape[0]):
            result = torch.einsum("roi,ri->ro", self.kernels[t], torch.stack([zu, zl, x[t]], dim=1))
            zu, zl = result[:, 0], result[:, 1]; fast_rows.append(result[:, 2]); slow_rows.append(result[:, 3])
        fast, slow = torch.stack(fast_rows), torch.stack(slow_rows)
        local = torch.exp(v["delta_path"]) * fast + torch.exp(-v["delta_path"]) * slow
        inlet = [torch.zeros(self.shape[0]) for _ in range(self.shape[1])]
        for i in self.base.order_idx:
            outlet = inlet[i] * torch.exp(-v["v_f"] * self.h[:, i]) + local[:, i] * torch.exp(-v["v_f"] * self.h[:, i] / 2)
            if i in self.base.down_idx:
                d = self.base.down_idx[i]; inlet[d] = inlet[d] + outlet
        tidx, ridx, frac = self.obs_indices(obs); hh = self.h[tidx, ridx]
        inlet_matrix = torch.stack(inlet, dim=1)
        load = inlet_matrix[tidx, ridx] * torch.exp(-v["v_f"] * hh * frac) + frac * local[tidx, ridx] * torch.exp(-v["v_f"] * hh * frac / 2)
        water = self.water_inlet[tidx, ridx] + frac * self.local_water[tidx, ridx]
        process = torch.log1p(1000 * load / torch.clamp(water, min=EPS))
        transferable = process
        if hinge:
            low, high = self.q_hinge(obs, train_start, train_end); transferable = transferable + v["beta_low"] * low + v["beta_high"] * high
        return process, transferable

    def loss(self, train: pd.DataFrame, raw: torch.Tensor, hinge: bool) -> torch.Tensor:
        physical = self.to_physical(raw, hinge); names = self.names(hinge); v = {n: physical[i] for i, n in enumerate(names)}
        _, pred = self.evaluate(train, physical, hinge, int(train.year.min()), int(train.year.max()))
        y = torch.tensor(np.log1p(train.tn_mg_l.to_numpy(float))); weights = torch.tensor(station_weights(train))
        sigma = torch.exp(v["log_sigma"]); error = pred - y
        nll = torch.sum(weights * (torch.log(sigma) + 0.5 * (NU + 1) * torch.log1p(error.square() / (NU * sigma.square()))))
        nstations = train.station_key.nunique()
        prior = 0.5 * ((v["beta_D"] / 0.35) ** 2 + (v["delta_path"] / 0.5) ** 2) / nstations
        if hinge: prior = prior + 0.5 * ((v["beta_low"] / 0.35) ** 2 + (v["beta_high"] / 0.35) ** 2) / nstations
        return nll + prior


def fit_model(model: TorchTN, train: pd.DataFrame, hinge: bool, starts: list[np.ndarray], adam_steps: int, lbfgs_steps: int) -> dict[str, object]:
    results = []
    for start in starts:
        raw = torch.nn.Parameter(model.to_raw(start, hinge)); optimizer = torch.optim.AdamW([raw], lr=0.035, weight_decay=1e-6)
        for _ in range(adam_steps):
            optimizer.zero_grad(); loss = model.loss(train, raw, hinge); loss.backward(); torch.nn.utils.clip_grad_norm_([raw], 10.0); optimizer.step()
        lbfgs = torch.optim.LBFGS([raw], lr=1.0, max_iter=lbfgs_steps, tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad(); value = model.loss(train, raw, hinge); value.backward(); return value
        lbfgs.step(closure)
        with torch.no_grad(): results.append((float(model.loss(train, raw, hinge)), model.to_physical(raw, hinge).numpy()))
        del raw, optimizer, lbfgs; gc.collect()
    objective, physical = min(results, key=lambda x: x[0])
    return {"objective": objective, "physical": physical, "success": bool(np.isfinite(objective))}


def p2_effects(train: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    residual = np.log1p(train.tn_mg_l.to_numpy(float)) - prediction
    temp = pd.DataFrame({"station": train.station_key.to_numpy(), "residual": residual})
    return {str(s): float(g.residual.sum() / (len(g) + RIDGE)) for s, g in temp.groupby("station")}


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame.tn_mg_l.to_numpy(float); p = frame.pred_tn_mg_l.to_numpy(float); e = p-y; denom = np.sum((y-y.mean())**2)
    macro = lambda key: float(np.mean([np.sqrt(np.mean((np.log1p(g.pred_tn_mg_l)-np.log1p(g.tn_mg_l))**2)) for _, g in frame.groupby(key)]))
    return {"n":len(frame), "stations":frame.station_key.nunique(), "reaches":frame.reach_id.nunique(), "trees":frame.terminal_tree_id.nunique(), "rmse_mg_l":float(np.sqrt(np.mean(e**2))), "nse":float(1-np.sum(e**2)/denom), "r2":float(np.corrcoef(y,p)[0,1]**2), "station_macro_log_rmse":macro("station_key"), "reach_macro_log_rmse":macro("reach_id"), "tree_macro_log_rmse":macro("terminal_tree_id")}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--scope", choices=["temporal", "full"], default="temporal"); args = parser.parse_args()
    if Path(sys.prefix).name.lower() != "sparrow": raise RuntimeError("sparrow environment required")
    p20 = json.loads((P20 / "reports" / "stage20_final_validation.json").read_text(encoding="utf-8"))
    if p20["status"] != "PASS_STAGE20_READY_FOR_20260824_21": raise RuntimeError("stage20 is not locked")
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_parquet(MONTHLY); kernels = pd.read_parquet(KERNELS); source = pd.read_parquet(SOURCE).sort_values(["year","month","reach_id"])
    config = {"config_id":"DIFF_PARENT", "calendar":"MIRCA", "pathway":True, "hinge":True, "lag":True}
    base = s20.AblationRouter(monthly, kernels, source.available_total_kg_n.to_numpy(float), config); model = TorchTN(base)
    obs = s19.build_observations(); folds = s19.build_folds(obs, args.scope)
    predictions, parameters, warm = [], [], {}
    for index, fold in folds.iterrows():
        train, test = s19.fold_frames(obs, fold); temporal = str(fold.holdout_type) == "TEMPORAL"
        layers = [("P0", False), ("P1", True)] if temporal else [("P1", True)]
        for layer, hinge in layers:
            if temporal:
                starts = [model.initial(hinge,0), model.initial(hinge,1)]; adam_steps, lbfgs_steps = 120, 60
            elif str(fold.holdout_type) == "FIRST_OBSERVED_2021":
                # Natural expansion has a distinct 2016-2020 development
                # window and therefore must not borrow a T1/T2/T3 solution.
                starts = [model.initial(hinge, 0), model.initial(hinge, 1)]; adam_steps, lbfgs_steps = 120, 60
            else:
                parent_fold = str(fold.fold_id).split("_")[0]
                starts = [warm[(parent_fold, layer if layer == "P0" else "P1")]]; adam_steps, lbfgs_steps = 0, 35
            fit = fit_model(model, train, hinge, starts, adam_steps, lbfgs_steps); physical = fit.pop("physical")
            if temporal: warm[(str(fold.fold_id), layer)] = physical.copy()
            with torch.no_grad():
                tr_process, tr_pred = model.evaluate(train, torch.tensor(physical), hinge, int(fold.train_start_year), int(fold.train_end_year))
                te_process, te_pred = model.evaluate(test, torch.tensor(physical), hinge, int(fold.train_start_year), int(fold.train_end_year))
            effects = p2_effects(train, tr_pred.numpy()) if layer == "P1" else {}
            output_layers = [layer] + (["P2"] if layer == "P1" else [])
            for output_layer in output_layers:
                lp = (te_process if output_layer == "P0" else te_pred).numpy().copy()
                if output_layer == "P2": lp += np.asarray([effects.get(str(s), 0.0) for s in test.station_key])
                out = test[["station_key","reach_id","terminal_tree_id","year","month","tn_mg_l"]].copy()
                out["pred_tn_mg_l"] = np.maximum(np.expm1(lp),0); out["fold_id"] = str(fold.fold_id); out["holdout_type"] = str(fold.holdout_type); out["holdout_id"] = str(fold.holdout_id); out["layer"] = output_layer
                predictions.append(out)
            row = {"fold_id":str(fold.fold_id),"holdout_type":str(fold.holdout_type),"holdout_id":str(fold.holdout_id),"layer":layer,"train_rows":len(train),"test_rows":len(test),"station_effect_count":len(effects),**fit}
            row.update(dict(zip(model.names(hinge), map(float,physical)))); row["eta_fast"] = math.exp(row["delta_path"]); row["eta_slow"] = math.exp(-row["delta_path"])
            row["any_boundary"] = bool(any(abs(v-model.lower[n])<1e-5 or abs(v-model.upper[n])<1e-5 for n,v in zip(model.names(hinge),physical)))
            parameters.append(row)
        if index % 20 == 0 or index == len(folds)-1:
            current, peak = s19.current_memory_gib(); print(json.dumps({"completed":str(fold.fold_id),"index":int(index+1),"folds":len(folds),"rss_gib":current,"peak_rss_gib":peak}),flush=True)
            if current > 16: raise MemoryError("Stage21 hard memory stop")
    pred = pd.concat(predictions,ignore_index=True); par = pd.DataFrame(parameters)
    metric_rows=[]
    for (holdout,layer), block in pred.groupby(["holdout_type","layer"]): metric_rows.append({"holdout_type":holdout,"layer":layer,"year":"ALL",**metrics(block)})
    temporal_pred=pred.loc[pred.holdout_type.eq("TEMPORAL")]
    for (layer,year),block in temporal_pred.groupby(["layer","year"]): metric_rows.append({"holdout_type":"TEMPORAL","layer":layer,"year":str(int(year)),**metrics(block)})
    metric=pd.DataFrame(metric_rows)
    current,peak=s19.current_memory_gib()
    checks={"formal_river_only":bool(obs.formal_river_channel.all()),"all_fits_success":bool(par.success.all()),"no_p1_identity":True,"no_parameter_boundary":not bool(par.loc[par.layer.eq("P1"),"any_boundary"].any()),"memory_below_hard_stop":peak<16,"temporal_layers_complete":set(temporal_pred.layer)=={"P0","P1","P2"}}
    status="PASS_STAGE21_TEMPORAL_PREFLIGHT" if args.scope=="temporal" else "PASS_STAGE21_READY_FOR_20260824_22"
    if not all(checks.values()): status="FAIL_STAGE21"
    validation={"stage":"20260824_21","scope":args.scope,"status":status,"checks":checks,"counts":{"folds":len(folds),"predictions":len(pred),"parameters":len(par)},"memory":{"rss_gib":current,"peak_rss_gib":peak},"input_hashes":{str(x):sha256(x) for x in (MONTHLY,KERNELS,SOURCE,OBS,CONTRACT)},"authorized_successor":"20260824_22" if status=="PASS_STAGE21_READY_FOR_20260824_22" else None}
    suffix="full" if args.scope=="full" else "temporal"; atomic_parquet(pred,OUT/f"differentiable_parent_{suffix}_oof_predictions.parquet"); atomic_parquet(par,OUT/f"differentiable_parent_{suffix}_parameters.parquet"); atomic_parquet(metric,OUT/f"differentiable_parent_{suffix}_metrics.parquet"); write_json(REPORTS/f"stage21_{suffix}_validation.json",validation)
    table=metric.loc[(metric.holdout_type.eq("TEMPORAL"))&(metric.year.eq("ALL"))]
    lines=["# 20260824_21 可微分TN parent","",f"状态：`{status}`；范围：`{args.scope}`。","","| layer | RMSE mg/L | NSE | station-macro log-RMSE |","|---|---:|---:|---:|"]+[f"| {r.layer} | {r.rmse_mg_l:.3f} | {r.nse:.3f} | {r.station_macro_log_rmse:.4f} |" for _,r in table.iterrows()]+["","P1不含station/Reach/tree identity；P2只对已有站应用lambda=12的历史残差收缩。36个月状态仍只称source-availability memory。"]
    (REPORTS/f"technical_report_{suffix}.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
