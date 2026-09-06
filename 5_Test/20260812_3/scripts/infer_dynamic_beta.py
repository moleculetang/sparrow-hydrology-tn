from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, logit, logsumexp

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "scripts" / "components" / "corrected_dynamic_q72_component.py"
INPUT = ROOT / "inputs" / "scenarios" / "B0_indata.parquet"
TOPOLOGY = ROOT / "inputs" / "topology" / "topology_edges.csv"
GRID = np.linspace(-4.0, 4.0, 81)
PRIOR_SD = 0.5
FOLDS = [
    {"fold_id": "fit_2006_2011_eval_2012_2013", "train_end": 2011, "eval_start": 2012, "eval_end": 2013},
    {"fold_id": "fit_2006_2013_eval_2014_2015", "train_end": 2013, "eval_start": 2014, "eval_end": 2015},
    {"fold_id": "fit_2006_2015_eval_2016_2018", "train_end": 2015, "eval_start": 2016, "eval_end": 2018},
]
FIXED = {
    "rho": 0.70, "wm": 480.0, "et_gamma": 0.75, "sas_rho": 0.93,
    "young_k": 1.5, "storage_scale": 720.0, "prod_capacity": 240.0,
    "runoff_gamma": 2.5, "quick_rho": 0.25, "base_rho": 0.85,
    "base_release": 0.10, "fixed_sigma": 3.0, "production_sigma": 1.5,
    "group_sigma": 1.5, "multistore_sigma": 0.30, "hysteresis_sigma": 3.0,
    "station_sigma": 1.0, "slope_sigma": 0.15, "regime_slope_sigma": 0.25,
    "anomaly_weight": 0.0, "flow_contrast_weight": 1.0,
}


def load_module():
    spec = importlib.util.spec_from_file_location("corrected_q72", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.INPUT_PATH = INPUT
    module.TOPOLOGY_PATH = TOPOLOGY
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    return module


def contrast_specs(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, float]]:
    specs = []
    for _, pos in train.groupby("q_site", sort=False).indices.items():
        pos = np.asarray(pos, dtype=int)
        if len(pos) < 36:
            continue
        flow = train.iloc[pos]["Q_obsv_cfs"].to_numpy(float)
        q25, q75, q90 = np.quantile(flow, [0.25, 0.75, 0.90])
        low, high = pos[flow <= q25], pos[flow >= q75]
        mid, peak = pos[(flow >= q25) & (flow <= q75)], pos[flow >= q90]
        if len(low) >= 6 and len(high) >= 6:
            specs.append((high, low, 1.0))
        if len(peak) >= 3 and len(mid) >= 12:
            specs.append((peak, mid, 1.0))
    return specs


def add_contrasts(values: np.ndarray, specs: list[tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
    if values.ndim == 1:
        return np.asarray([(values[a].mean()-values[b].mean())*w for a,b,w in specs], float)
    return np.vstack([(values[a].mean(axis=0)-values[b].mean(axis=0))*w for a,b,w in specs])


def penalty_vector(module, stations: list[str]) -> np.ndarray:
    n_fixed_base = 1 + len(module.FIXED_FEATURES)
    n_group = len(module.SPATIAL_GROUP_GATES) * len(module.SPATIAL_GROUP_FEATURES)
    n_fixed = n_fixed_base + n_group
    n_station = len(stations)
    n_slope = len(module.RANDOM_SLOPE_FEATURES) * n_station
    n_regime = len(module.REGIME_GATES) * len(module.REGIME_SLOPE_FEATURES) * n_station
    penalty = np.zeros(n_fixed+n_station+n_slope+n_regime)
    penalty[1:n_fixed_base] = 1/FIXED["fixed_sigma"]
    for i, feat in enumerate(module.FIXED_FEATURES, start=1):
        if feat in module.PRODUCTION_FEATURES: penalty[i] = 1/FIXED["production_sigma"]
        if feat in module.MULTISTORE_FEATURES: penalty[i] = 1/FIXED["multistore_sigma"]
        if feat in module.HYSTERESIS_FEATURES: penalty[i] = 1/FIXED["hysteresis_sigma"]
    penalty[n_fixed_base:n_fixed] = 1/FIXED["group_sigma"]
    penalty[n_fixed:n_fixed+n_station] = 1/FIXED["station_sigma"]
    penalty[n_fixed+n_station:n_fixed+n_station+n_slope] = 1/FIXED["slope_sigma"]
    penalty[n_fixed+n_station+n_slope:] = 1/FIXED["regime_slope_sigma"]
    return penalty


@dataclass
class FoldSystem:
    fold: dict
    module: object
    full: pd.DataFrame
    observed: pd.DataFrame
    train: pd.DataFrame
    evaluation: pd.DataFrame
    stations: list[str]
    specs: list
    qcol: int
    penalty: np.ndarray
    B: np.ndarray
    B_eval: np.ndarray
    y: np.ndarray
    chol: tuple
    Ainv_b: np.ndarray
    sse_y: float
    q_vectors: np.ndarray
    q_eval_vectors: np.ndarray
    xres: np.ndarray
    xyres: np.ndarray
    theta_q: np.ndarray
    log_posterior: np.ndarray
    weights: np.ndarray


def q_columns(system_base: dict, beta: float) -> tuple[np.ndarray, np.ndarray]:
    module, full = system_base["module"], system_base["full"]
    fraction = expit(logit(0.35) + beta * full["connectivity_score_fold_z"].to_numpy(float))
    q_full = module._aggregate_local_cfs(full, fraction * full["local_net_cfs"].to_numpy(float))
    q_observed = q_full[system_base["observation_mask"]]
    q_train = q_observed[system_base["train_mask"]]
    mean, std = q_train.copy(), None
    mean = float(np.log1p(q_train).mean())
    std = float(np.log1p(q_train).std(ddof=0))
    if std <= module.EPS: std = 1.0
    z = (np.log1p(q_observed)-mean)/std
    return z[system_base["train_mask"]], z[system_base["eval_mask"]]


def build_fold(fold: dict) -> FoldSystem:
    module = load_module()
    module.CAL_END_YEAR = int(fold["train_end"])
    module.INNER_TRAIN_END_YEAR = int(fold["train_end"])
    module.DYNAMIC_BETA_W = 0.0
    forcing = module.load_forcing_panel()
    full = module.add_hydrologic_features(forcing, **{k: FIXED[k] for k in ["rho","wm","et_gamma","sas_rho","young_k","storage_scale","prod_capacity","runoff_gamma","quick_rho","base_rho","base_release"]})
    observation_mask = module.observation_mask(full).to_numpy()
    observed_raw = full.loc[observation_mask].copy().reset_index(drop=True)
    observed_raw["q_site"] = observed_raw["q_site"].astype(str)
    observed = module.prepare_design(observed_raw)
    train_mask = observed.year.le(int(fold["train_end"])).to_numpy()
    eval_mask = observed.year.between(int(fold["eval_start"]), int(fold["eval_end"])).to_numpy()
    train, evaluation = observed.loc[train_mask].copy(), observed.loc[eval_mask].copy()
    stations = sorted(observed.q_site.unique())
    mean, std = module.standardize_fit(train)
    X, y_obs = module.build_matrix(train, stations, mean, std)
    Xeval, _ = module.build_matrix(evaluation, stations, mean, std)
    qcol = 1 + module.FIXED_FEATURES.index("log_qcalc")
    specs = contrast_specs(train)
    Xc, yc = add_contrasts(X, specs), add_contrasts(y_obs, specs)
    penalty = penalty_vector(module, stations)
    Xaug = np.vstack([X, Xc, np.diag(penalty)])
    y = np.concatenate([y_obs, yc, np.zeros(len(penalty))])
    B = np.delete(Xaug, qcol, axis=1)
    Beval = np.delete(Xeval, qcol, axis=1)
    A = B.T @ B
    chol = cho_factor(A, lower=True, check_finite=False)
    b = B.T @ y
    Ainv_b = cho_solve(chol, b, check_finite=False)
    sse_y = float(y@y - b@Ainv_b)
    base = {"module":module,"full":full,"observation_mask":observation_mask,"train_mask":train_mask,"eval_mask":eval_mask}
    qv, qev, xres, xyres = [], [], [], []
    for beta in GRID:
        qo, qe = q_columns(base, float(beta))
        q_aug = np.concatenate([qo, add_contrasts(qo, specs), np.eye(len(penalty))[:,qcol]*penalty[qcol]])
        g = B.T @ q_aug
        ainv_g = cho_solve(chol, g, check_finite=False)
        xr = float(q_aug@q_aug-g@ainv_g)
        xyr = float(q_aug@y-g@Ainv_b)
        qv.append(q_aug); qev.append(qe); xres.append(xr); xyres.append(xyr)
    qv, qev = np.column_stack(qv), np.column_stack(qev)
    xres, xyres = np.asarray(xres), np.asarray(xyres)
    theta_q = xyres/xres
    sse = sse_y-xyres**2/xres
    logpost = -0.5*sse-0.5*np.log(xres)-0.5*(GRID/PRIOR_SD)**2
    logpost -= logsumexp(logpost)
    weights = np.exp(logpost)
    return FoldSystem(fold,module,full,observed,train,evaluation,stations,specs,qcol,penalty,B,Beval,y,chol,Ainv_b,sse_y,qv,qev,xres,xyres,theta_q,logpost,weights)


def posterior_summary(system: FoldSystem) -> dict[str, float | str | bool]:
    w = system.weights
    mean = float(np.sum(w*GRID)); sd = float(np.sqrt(np.sum(w*(GRID-mean)**2)))
    cdf = np.cumsum(w)
    lo, hi = float(np.interp(0.025,cdf,GRID)), float(np.interp(0.975,cdf,GRID))
    map_beta = float(GRID[np.argmax(w)])
    # Nonlinear score residual fraction against the complete nuisance design.
    i0, i1 = int(np.argmin(abs(GRID))), int(np.argmin(abs(GRID-1)))
    delta = system.q_vectors[:,i1]-system.q_vectors[:,i0]
    g = system.B.T@delta
    residual_ss = float(delta@delta-g@cho_solve(system.chol,g,check_finite=False))
    residual_fraction = float(np.sqrt(max(residual_ss,0))/max(np.linalg.norm(delta),1e-12))
    return {
        "fold_id":system.fold["fold_id"],"posterior_mean":mean,"posterior_map":map_beta,"posterior_sd":sd,
        "ci95_low":lo,"ci95_high":hi,"p_beta_gt_zero":float(w[GRID>0].sum()),"prior_sd":PRIOR_SD,
        "sd_shrink_fraction":1-sd/PRIOR_SD,"dynamic_score_residual_fraction":residual_fraction,
        "mode_hits_boundary":abs(map_beta)>=3.999,"identifiability_pre_gate":bool(residual_fraction>=0.05 and sd<=0.375 and abs(map_beta)<3.999),
    }


def synthetic_recovery(system: FoldSystem, seed: int=2026081203) -> pd.DataFrame:
    rng=np.random.default_rng(seed+int(system.fold["train_end"]))
    n_obs=len(system.train); n_con=len(system.specs); p=len(system.penalty)
    rows=[]
    for truth in [-0.5,0.0,0.5]:
        ti=int(np.argmin(abs(GRID-truth)))
        q=system.q_vectors[:,ti]
        tq=system.theta_q[ti]
        g=system.B.T@q
        tb=cho_solve(system.chol,system.B.T@(system.y-q*tq),check_finite=False)
        mu=system.B[:n_obs]@tb+q[:n_obs]*tq
        resid=system.y[:n_obs]-mu
        sigma=float(np.sqrt(np.mean(resid**2)))
        for rep in range(30):
            ys=mu+rng.normal(0,sigma,n_obs)
            yaug=np.concatenate([ys,add_contrasts(ys,system.specs),np.zeros(p)])
            b=system.B.T@yaug; ainvb=cho_solve(system.chol,b,check_finite=False)
            ssey=float(yaug@yaug-b@ainvb)
            xyy=system.q_vectors.T@yaug-np.sum((system.B.T@system.q_vectors)*ainvb[:,None],axis=0)
            sse=ssey-xyy**2/system.xres
            lp=-0.5*sse-0.5*np.log(system.xres)-0.5*(GRID/PRIOR_SD)**2; lp-=logsumexp(lp); w=np.exp(lp)
            pm=float(np.sum(w*GRID)); c=np.cumsum(w); lo=float(np.interp(.025,c,GRID)); hi=float(np.interp(.975,c,GRID))
            rows.append({"fold_id":system.fold["fold_id"],"beta_true":truth,"replicate":rep,"posterior_mean":pm,"ci95_low":lo,"ci95_high":hi,"covered":lo<=truth<=hi,"sign_correct": (truth==0) or np.sign(pm)==np.sign(truth),"false_positive":truth==0 and not(lo<=0<=hi)})
    return pd.DataFrame(rows)


def posterior_predict(system: FoldSystem) -> pd.DataFrame:
    preds=[]
    for j,beta in enumerate(GRID):
        q=system.q_vectors[:,j]; tq=system.theta_q[j]
        tb=cho_solve(system.chol,system.B.T@(system.y-q*tq),check_finite=False)
        logpred=system.B_eval@tb+system.q_eval_vectors[:,j]*tq
        preds.append(np.exp(np.clip(logpred,-20,20)))
    out=system.evaluation.copy()
    out["predict"]=np.column_stack(preds)@system.weights
    out["actual"]=out["Q_obsv_cfs"]
    out["fold_id"]=system.fold["fold_id"]
    out["scenario_id"]="I1"
    out["beta_posterior_mean"]=np.sum(system.weights*GRID)
    return out


def main() -> None:
    summaries=[]; recoveries=[]; predictions=[]; distributions=[]
    for fold in FOLDS:
        system=build_fold(fold)
        summary=posterior_summary(system); summaries.append(summary)
        for beta,weight,lp in zip(GRID,system.weights,system.log_posterior): distributions.append({"fold_id":fold["fold_id"],"beta_w":beta,"posterior_weight":weight,"log_posterior_normalized":lp})
        if summary["dynamic_score_residual_fraction"]>=0.05:
            recoveries.append(synthetic_recovery(system))
            predictions.append(posterior_predict(system))
    pd.DataFrame(summaries).to_csv(ROOT/"reports"/"dynamic_beta_posterior_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(distributions).to_csv(ROOT/"reports"/"dynamic_beta_posterior_grid.csv",index=False,encoding="utf-8-sig")
    if recoveries: pd.concat(recoveries,ignore_index=True).to_csv(ROOT/"reports"/"synthetic_recovery.csv",index=False,encoding="utf-8-sig")
    if predictions:
        out=pd.concat(predictions,ignore_index=True)
        (ROOT/"outputs"/"I1").mkdir(parents=True,exist_ok=True)
        out.to_parquet(ROOT/"outputs"/"I1"/"q72_three_fold_oof_predictions.parquet",index=False)
    payload={"runtime":RUNTIME,"grid":{"min":-4,"max":4,"points":81},"folds":summaries,"i1_written":bool(predictions)}
    (ROOT/"logs"/"G5_G8_dynamic_inference.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
