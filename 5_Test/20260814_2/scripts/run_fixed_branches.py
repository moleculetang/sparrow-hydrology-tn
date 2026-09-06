from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


warnings.filterwarnings("ignore")
assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT.parent / "20260814_1"
COMPONENT = ROOT / "scripts" / "components" / "q72_prior_semantics_component.py"
INPUT = STAGE1 / "inputs" / "development_indata_2006_2018.parquet"
TOPOLOGY = STAGE1 / "inputs" / "topology" / "topology_edges.csv"
FOLDS = [
    {"fold_id":"fit_2006_2011_eval_2012_2013","train_end":2011,"inner_train_end":2009,"eval_start":2012,"eval_end":2013},
    {"fold_id":"fit_2006_2013_eval_2014_2015","train_end":2013,"inner_train_end":2011,"eval_start":2014,"eval_end":2015},
    {"fold_id":"fit_2006_2015_eval_2016_2018","train_end":2015,"inner_train_end":2013,"eval_start":2016,"eval_end":2018},
]
BRANCHES = {
    "main":{"prod_capacity":240.0,"runoff_gamma":2.5,"quick_rho":0.25,"base_release":0.10,"base_rho":0.85,"highflow_scale":1.00},
    "flash":{"prod_capacity":132.0,"runoff_gamma":1.7,"quick_rho":0.10,"base_release":0.06,"base_rho":0.76,"highflow_scale":1.20},
    "slow":{"prod_capacity":432.0,"runoff_gamma":3.1,"quick_rho":0.48,"base_release":0.07,"base_rho":0.94,"highflow_scale":0.85},
    "buffer":{"prod_capacity":348.0,"runoff_gamma":3.3,"quick_rho":0.66,"base_release":0.12,"base_rho":0.96,"highflow_scale":0.65},
    "wet":{"prod_capacity":288.0,"runoff_gamma":2.2,"quick_rho":0.34,"base_release":0.08,"base_rho":0.90,"highflow_scale":1.10},
}
BASE = {"rho":0.70,"wm":480.0,"et_gamma":0.75,"sas_rho":0.93,"young_k":1.5,"storage_scale":720.0,
        "fixed_sigma":3.0,"production_sigma":1.5,"group_sigma":1.5,"multistore_sigma":0.30,"station_sigma":1.0,
        "slope_sigma":0.15,"regime_slope_sigma":0.25,"anomaly_weight":0.0,"flow_contrast_weight":1.0}
KEY=["comid","q_site","year","month","fold_id"]


def load_component(label: str):
    spec=importlib.util.spec_from_file_location(f"q72_single_{label}",COMPONENT); module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader; spec.loader.exec_module(module)
    module.INPUT_PATH=INPUT; module.TOPOLOGY_PATH=TOPOLOGY; module.STATE_CALENDAR_MODE="full_forcing"
    module.FORCING_SEMANTICS_MODE="prescribed_aet_balance"; module.MASS_ACCOUNTING_MODE="explicit_upstream_volume"
    module.configure_engineering_repair(True); module.configure_full_gaussian_prior_space(True)
    module.ZERO_PRESERVING_FULL_PRIOR_MODE=True; module.DETERMINISTIC_SPINUP_MODE=True; module.ET_STATE_OPERATOR_MODE="baseline_clip"
    original=module.set_et_feature_block_mode
    def single_gate(mode: str):
        original(mode)
        removed=set(module.MULTISTORE_FEATURES)
        module.FIXED_FEATURES=[f for f in module.FIXED_FEATURES if f not in removed]
        module.MULTISTORE_FEATURES=[]
    module.set_et_feature_block_mode=single_gate; module.set_et_feature_block_mode("full")
    return module


def featured(module, forcing: pd.DataFrame, branch: dict[str,float]) -> pd.DataFrame:
    return module.build_featured_observation_panel(
        forcing,rho=BASE["rho"],wm=BASE["wm"],et_gamma=BASE["et_gamma"],sas_rho=BASE["sas_rho"],young_k=BASE["young_k"],storage_scale=BASE["storage_scale"],
        prod_capacity=branch["prod_capacity"],runoff_gamma=branch["runoff_gamma"],quick_rho=branch["quick_rho"],base_rho=branch["base_rho"],base_release=branch["base_release"],state_calendar_mode="full_forcing")


def choose_hysteresis(module, frame: pd.DataFrame, stations: list[str], inner_end: int, train_end: int) -> tuple[float,pd.DataFrame]:
    tr=frame[frame.year<=inner_end].copy(); iv=frame[frame.year.between(inner_end+1,train_end)].copy(); mean,std=module.standardize_fit(tr)
    rows=[]
    for hs in [0.30,0.80,1.50,3.00]:
        beta=module.fit_map_ridge(tr,stations,mean,std,fixed_sigma=BASE["fixed_sigma"],production_sigma=BASE["production_sigma"],group_sigma=BASE["group_sigma"],multistore_sigma=BASE["multistore_sigma"],hysteresis_sigma=hs,station_sigma=BASE["station_sigma"],slope_sigma=BASE["slope_sigma"],regime_slope_sigma=BASE["regime_slope_sigma"],anomaly_weight=0.0,flow_contrast_weight=1.0)
        pred=np.exp(np.clip(module.predict_log(iv,beta,stations,mean,std),-20,20)); obs=iv.Q_obsv_cfs.to_numpy(float)
        md=module.metric_dict(obs,pred); sm=module.station_median_metrics(iv,pred)
        score=(sm["median_NSE_log"]+1.5*sm["median_KGE"]-0.60*min(abs(sm["median_alpha"]-1),2)-0.005*min(sm["median_abs_PBIAS"],9999)+0.01*sm["good_count"])
        rows.append({"hysteresis_sigma":hs,"inner_score":score,"inner_NSE_log":md["NSE_log"],"inner_abs_PBIAS":abs(md["PBIAS_pct"]),**{f"median_{k}":v for k,v in sm.items()}})
    grid=pd.DataFrame(rows).sort_values(["inner_score","hysteresis_sigma"],ascending=[False,True])
    return float(grid.iloc[0].hysteresis_sigma),grid


def run_fold(branch_id: str, fold: dict[str,int|str]) -> pd.DataFrame:
    module=load_component(f"{branch_id}_{fold['train_end']}")
    fold_dir=ROOT/"outputs"/branch_id/str(fold["fold_id"]); fold_dir.mkdir(parents=True,exist_ok=True)
    module.REPORT_DIR=fold_dir/"fit_artifacts"; module.REPORT_DIR.mkdir(parents=True,exist_ok=True)
    module.FIG_DIR=fold_dir/"figures"; module.CAL_END_YEAR=int(fold["train_end"]); module.INNER_TRAIN_END_YEAR=int(fold["inner_train_end"])
    forcing=module.load_forcing_panel(); frame=featured(module,forcing,BRANCHES[branch_id]); stations=sorted(frame.q_site.unique())
    hs,grid=choose_hysteresis(module,frame,stations,int(fold["inner_train_end"]),int(fold["train_end"])); grid.to_csv(fold_dir/"hysteresis_grid.csv",index=False,encoding="utf-8-sig")
    train=frame[frame.year<=int(fold["train_end"])].copy(); mean,std=module.standardize_fit(train)
    beta=module.fit_map_ridge(train,stations,mean,std,fixed_sigma=BASE["fixed_sigma"],production_sigma=BASE["production_sigma"],group_sigma=BASE["group_sigma"],multistore_sigma=BASE["multistore_sigma"],hysteresis_sigma=hs,station_sigma=BASE["station_sigma"],slope_sigma=BASE["slope_sigma"],regime_slope_sigma=BASE["regime_slope_sigma"],anomaly_weight=0.0,flow_contrast_weight=1.0)
    evalf=frame[frame.year.between(int(fold["eval_start"]),int(fold["eval_end"]))].copy()
    evalf["predict"]=np.exp(np.clip(module.predict_log(evalf,beta,stations,mean,std),-20,20)); evalf["actual"]=evalf.Q_obsv_cfs
    evalf["fold_id"]=str(fold["fold_id"]); evalf["branch_id"]=branch_id; evalf["hysteresis_sigma"]=hs
    keep=["comid","q_site","year","month","actual","predict","fold_id","branch_id","hysteresis_sigma","Q_calc_cfs","upstream_positive_input_equivalent_cfs"]
    out=evalf[keep].copy(); out.to_parquet(fold_dir/"evaluation_predictions.parquet",index=False)
    meta=module.design_column_metadata(stations); meta.to_csv(fold_dir/"design_columns.csv",index=False,encoding="utf-8-sig")
    if len(meta)!=3383: raise RuntimeError(f"Single design column gate failed: {len(meta)}")
    return out


def run_branch(branch_id: str) -> dict[str,object]:
    parts=[run_fold(branch_id,fold) for fold in FOLDS]; oof=pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    if len(oof)!=7755 or oof[KEY].duplicated().any(): raise RuntimeError(f"OOF gate failed for {branch_id}")
    path=ROOT/"outputs"/branch_id/"oof.parquet"; path.parent.mkdir(parents=True,exist_ok=True); oof.to_parquet(path,index=False)
    payload={"branch_id":branch_id,"parameters":BRANCHES[branch_id],"rows":len(oof),"design_columns":3383,"hysteresis_by_fold":oof.groupby("fold_id").hysteresis_sigma.first().to_dict(),"performance_type":"fixed_structure_oof"}
    (path.parent/"run_manifest.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False))
    return payload


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("branch",choices=[*BRANCHES,"ALL"]); args=parser.parse_args()
    branches=list(BRANCHES) if args.branch=="ALL" else [args.branch]
    for branch in branches: run_branch(branch)


if __name__=="__main__": main()
