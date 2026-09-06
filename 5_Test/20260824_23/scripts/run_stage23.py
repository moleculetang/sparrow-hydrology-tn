"""Attribute-only hierarchical Bayesian spatial regionalization of TN P1."""
from __future__ import annotations
import os
for _n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_n]="1"
import hashlib, importlib.util, json, math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(r"E:\SPARROW"); RUN=ROOT/"5_Test"/"20260824_23"; P19=ROOT/"5_Test"/"20260824_19"; P20=ROOT/"5_Test"/"20260824_20"; P21=ROOT/"5_Test"/"20260824_21"; P22=ROOT/"5_Test"/"20260824_22"
OUT=RUN/"outputs"; REPORTS=RUN/"reports"; CONTRACT=RUN/"experiment_contract.json"
STATIC=ROOT/"5_Test"/"20260824_15"/"outputs"/"static_delivery_covariates_raw.parquet"; MONTHLY=P19/"outputs"/"corrected_tn_bridge_monthly_2006_2024.parquet"; KERNELS=P19/"outputs"/"corrected_daily_carrier_kernels_2006_2024.parquet"; SOURCE=P19/"outputs"/"corrected_source_availability_2006_2024.parquet"; PAR=P21/"outputs"/"differentiable_parent_full_parameters.parquet"; PRED=P21/"outputs"/"differentiable_parent_full_oof_predictions.parquet"
FEATURES=["cropland_fraction","agricultural_n_intensity","soc_0_30cm","clay_0_20cm","dem_slope","canonical_fast_fraction","logQ_variability","lower_storage_signature"]
def mod(n,p): s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
s19=mod("s19",P19/"scripts"/"run_stage19.py");s20=mod("s20",P20/"scripts"/"run_stage20.py");s21=mod("s21",P21/"scripts"/"run_stage21.py")
def sha(p):
 h=hashlib.sha256();f=p.open("rb")
 for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 f.close();return h.hexdigest()
def pq(x,p):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+".tmp");x.to_parquet(t,index=False);os.replace(t,p)
def js(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda z:z.item() if isinstance(z,np.generic) else str(z))+"\n",encoding="utf-8")
def feature_table(monthly):
 s=pd.read_parquet(STATIC).rename(columns={"cropland_fraction_clcd_multiyear":"cropland_fraction","agricultural_n_intensity_kg_n_km2_year":"agricultural_n_intensity","soc_0_30cm_depth_weighted":"soc_0_30cm"})
 h=monthly.loc[monthly.year.between(2010,2020)].copy(); fast=h.groupby("reach_id").state_consistent_fast_fraction.mean().rename("canonical_fast_fraction"); q=h.assign(logq=np.log(h.routed_total_m3_s.clip(lower=1e-12))).groupby("reach_id").logq.std(ddof=0).rename("logQ_variability"); store=(h.lower_slow_storage_start_mm/(h.soil_storage_start_mm+h.upper_response_storage_start_mm+h.lower_slow_storage_start_mm).clip(lower=1e-12)).groupby(h.reach_id).mean().rename("lower_storage_signature")
 x=s.merge(fast,on="reach_id").merge(q,on="reach_id").merge(store,on="reach_id").sort_values("reach_id"); raw=x.copy()
 for c in FEATURES: x[c]=(x[c]-x[c].mean())/x[c].std(ddof=0)
 if x[FEATURES].isna().any().any() or np.linalg.matrix_rank(np.column_stack([np.ones(len(x)),x[FEATURES]]))<9:raise RuntimeError("feature QA failed")
 return raw,x
def design(obs,x):return np.column_stack([np.ones(len(obs)),x.set_index("reach_id").loc[obs.reach_id,FEATURES].to_numpy(float)])
def weights(obs):return s21.station_weights(obs)
def fit_map(train,residual,x):
 X=design(train,x);w=weights(train);prior_precision=1/.25**2/train.station_key.nunique();A=X.T@(w[:,None]*X)+prior_precision*np.eye(X.shape[1]);b=np.linalg.solve(A,X.T@(w*residual));e=residual-X@b;s2=float(np.sum(w*e**2));cov=s2*np.linalg.inv(A);return b,cov,s2
def skill(frame):
 y=np.log1p(frame.tn_mg_l.to_numpy(float));p=np.log1p(frame.pred_tn_mg_l.to_numpy(float));b=frame.baseline_log_mean.to_numpy(float);return float(1-np.sum((y-p)**2)/np.sum((y-b)**2))
def bootstrap_skill(frame,reps=4000):
 blocks=list(frame.groupby("station_key"));rng=np.random.default_rng(2026082423);vals=[]
 for _ in range(reps):
  draw=pd.concat([blocks[i][1] for i in rng.integers(0,len(blocks),len(blocks))],ignore_index=True);vals.append(skill(draw))
 lo,hi=np.quantile(vals,[.025,.975]);return float(lo),float(hi)
def main():
 if Path(sys.prefix).name.lower()!="sparrow":raise RuntimeError("sparrow required")
 if json.loads((P22/"reports"/"stage22_full_validation.json").read_text(encoding="utf-8"))["status"]!="PASS_STAGE22_READY_FOR_20260824_23":raise RuntimeError("stage22 not locked")
 OUT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True);torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 monthly=pd.read_parquet(MONTHLY);kernels=pd.read_parquet(KERNELS);src=pd.read_parquet(SOURCE).sort_values(["year","month","reach_id"]);raw,x=feature_table(monthly);pq(raw,OUT/"spatial_features_raw.parquet");pq(x,OUT/"spatial_features_standardized.parquet")
 base=s20.AblationRouter(monthly,kernels,src.available_total_kg_n.to_numpy(float),{"config_id":"SPATIAL","calendar":"MIRCA","pathway":True,"hinge":True,"lag":True});model=s21.TorchTN(base);obs=s19.build_observations();folds=s19.build_folds(obs,"full");pars=pd.read_parquet(PAR).loc[lambda z:z.layer.eq("P1")].set_index("fold_id");parentpred=pd.read_parquet(PRED).loc[lambda z:z.layer.eq("P1")]
 preds=[];coef=[];repro=[]
 for i,fold in folds.iterrows():
  train,test=s19.fold_frames(obs,fold);row=pars.loc[str(fold.fold_id)];names=model.names(True);physical=np.array([row[n] for n in names]);combined=pd.concat([train.assign(_part="train"),test.assign(_part="test")],ignore_index=True)
  with torch.no_grad():lp=model.evaluate(combined,torch.tensor(physical),True,int(fold.train_start_year),int(fold.train_end_year))[1].numpy()
  n=len(train);ltr,lt=lp[:n],lp[n:];res=np.log1p(train.tn_mg_l.to_numpy(float))-ltr;b,cov,s2=fit_map(train,res,x);delta=design(test,x)@b;pred=np.maximum(np.expm1(lt+delta),0)
  station_means=train.assign(y=np.log1p(train.tn_mg_l)).groupby("station_key").y.mean();baseline=float(station_means.mean())
  o=test[["station_key","reach_id","terminal_tree_id","year","month","tn_mg_l"]].copy();o["pred_tn_mg_l"]=pred;o["parent_pred_tn_mg_l"]=np.maximum(np.expm1(lt),0);o["baseline_log_mean"]=baseline;o["spatial_delta_log"]=delta;o["fold_id"]=str(fold.fold_id);o["holdout_type"]=str(fold.holdout_type);o["holdout_id"]=str(fold.holdout_id);preds.append(o)
  for j,name in enumerate(["intercept"]+FEATURES):coef.append({"fold_id":str(fold.fold_id),"holdout_type":str(fold.holdout_type),"term":name,"map_coefficient":b[j],"laplace_sd":math.sqrt(max(cov[j,j],0)),"residual_variance":s2})
  pp=parentpred.loc[parentpred.fold_id.eq(str(fold.fold_id))];keys=["station_key","reach_id","terminal_tree_id","year","month","tn_mg_l"];z=pp[keys+["pred_tn_mg_l"]].merge(o[keys+["parent_pred_tn_mg_l"]],on=keys,validate="one_to_one");repro.append(float(np.max(np.abs(z.pred_tn_mg_l-z.parent_pred_tn_mg_l))))
  if i%20==0 or i==len(folds)-1:print(json.dumps({"completed":str(fold.fold_id),"index":int(i+1),"folds":len(folds),"rss_gib":s19.current_memory_gib()[0]}),flush=True)
 pred=pd.concat(preds,ignore_index=True);co=pd.DataFrame(coef);metrics=[];skills=[]
 for hold,g in pred.groupby("holdout_type"):
  c=g.copy();c["pred_tn_mg_l"]=g.pred_tn_mg_l;cm=s21.metrics(c);p=g.copy();p["pred_tn_mg_l"]=g.parent_pred_tn_mg_l;pm=s21.metrics(p);lo,hi=bootstrap_skill(g);metrics.append({"holdout_type":hold,"arm":"regionalized",**cm});metrics.append({"holdout_type":hold,"arm":"parent",**pm});skills.append({"holdout_type":hold,"skill_log":skill(g),"ci95_lower":lo,"ci95_upper":hi,"supported":lo>0})
 mf=pd.DataFrame(metrics);sf=pd.DataFrame(skills);tempc=mf.loc[(mf.holdout_type=="TEMPORAL")&(mf.arm=="regionalized")].iloc[0];tempp=mf.loc[(mf.holdout_type=="TEMPORAL")&(mf.arm=="parent")].iloc[0]
 spatial_ok=bool(sf.loc[sf.holdout_type.isin(["REACH","TREE"]),"supported"].all());improved=tempc.station_macro_log_rmse<tempp.station_macro_log_rmse;delta_bound=float(pred.spatial_delta_log.abs().max());scientific="SPATIAL_REGIONALIZATION_SUPPORTED" if improved and spatial_ok and delta_bound<1 else ("SPATIAL_REGIONALIZATION_EXTRAPOLATION_CONFOUNDED" if delta_bound>=1 else "SPATIAL_REGIONALIZATION_NOT_SUPPORTED")
 peak=s19.current_memory_gib()[1];checks={"eight_features":len(FEATURES)==8,"no_identity_features":not any(z in FEATURES for z in ["station_key","reach_id","terminal_tree_id"]),"parent_reproduction_1e_8":max(repro)<=1e-8,"all_finite":bool(np.isfinite(pred.pred_tn_mg_l).all()),"memory_below_hard_stop":peak<16};status="PASS_STAGE23_READY_FOR_20260824_24" if all(checks.values()) else "FAIL_STAGE23"
 val={"stage":"20260824_23","status":status,"scientific_decision":scientific,"temporal_station_macro_delta":float(tempc.station_macro_log_rmse-tempp.station_macro_log_rmse),"spatial_skill":sf.to_dict("records"),"maximum_abs_spatial_delta_log":delta_bound,"checks":checks,"peak_rss_gib":peak,"input_hashes":{str(p):sha(p) for p in (STATIC,MONTHLY,KERNELS,SOURCE,PAR,PRED,CONTRACT)},"authorized_successor":"20260824_24" if status.startswith("PASS") else None}
 pq(pred,OUT/"spatial_regionalization_oof_predictions.parquet");pq(co,OUT/"spatial_regionalization_coefficients.parquet");pq(mf,OUT/"spatial_regionalization_metrics.parquet");pq(sf,OUT/"station_blind_spatial_skill.parquet");js(REPORTS/"stage23_final_validation.json",val)
 (REPORTS/"technical_report.md").write_text(f"# 20260824_23 层级贝叶斯空间区域化\n\n状态：`{status}`；科学裁决：`{scientific}`。\n\n时间OOF station-macro log-RMSE变化 `{val['temporal_station_macro_delta']:.5f}`；最大属性空间修正 `{delta_bound:.3f}` log unit。\n\n该层只使用八个Reach属性，不使用任何站点、Reach或tree身份。\n",encoding="utf-8");print(json.dumps(val,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
