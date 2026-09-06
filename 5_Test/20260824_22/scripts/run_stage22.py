"""Test one explicit manure organic-N mineralization pool."""

from __future__ import annotations
import os
for _n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[_n]="1"
import argparse, gc, hashlib, importlib.util, json, math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(r"E:\SPARROW"); RUN=ROOT/"5_Test"/"20260824_22"; P19=ROOT/"5_Test"/"20260824_19"; P20=ROOT/"5_Test"/"20260824_20"; P21=ROOT/"5_Test"/"20260824_21"
OUT=RUN/"outputs"; REPORTS=RUN/"reports"; CONTRACT=RUN/"experiment_contract.json"
RAW_SOURCE=ROOT/"5_Test"/"20260824_12"/"outputs"/"monthly_source_forcing_1961_2024.parquet"
MONTHLY=P19/"outputs"/"corrected_tn_bridge_monthly_2006_2024.parquet"; KERNELS=P19/"outputs"/"corrected_daily_carrier_kernels_2006_2024.parquet"; SOURCE=P19/"outputs"/"corrected_source_availability_2006_2024.parquet"
PARENT_PRED=P21/"outputs"/"differentiable_parent_full_oof_predictions.parquet"; NU=4.; RIDGE=12.

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
s19=module("s19",P19/"scripts"/"run_stage19.py"); s20=module("s20",P20/"scripts"/"run_stage20.py"); s21=module("s21",P21/"scripts"/"run_stage21.py")
def sha(path):
    h=hashlib.sha256(); f=path.open("rb")
    for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    f.close(); return h.hexdigest()
def js(path,x): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda z:z.item() if isinstance(z,np.generic) else str(z))+"\n",encoding="utf-8")
def pq(x,path): path.parent.mkdir(parents=True,exist_ok=True); t=path.with_name(path.name+".tmp"); x.to_parquet(t,index=False); os.replace(t,path)

def source_components(k):
    src=pd.read_parquet(RAW_SOURCE).loc[lambda x:x.calendar_scenario.eq("CENTRAL")].sort_values(["year","month","reach_id"]).reset_index(drop=True)
    reaches=np.sort(src.reach_id.unique()); nt=src[["year","month"]].drop_duplicates().shape[0]; nr=len(reaches); shape=(nt,nr)
    manure=src.manure_kg_n.to_numpy(float).reshape(shape); q=1-math.exp(-k/12); pool=np.zeros(nr); rel=np.zeros(shape); maxerr=0.
    for t in range(nt):
        pre=pool+manure[t]; rel[t]=q*pre; post=pre-rel[t]; maxerr=max(maxerr,float(np.max(np.abs(pre-rel[t]-post)))); pool=post
    mask=(src.year.to_numpy()>2006)|((src.year.to_numpy()==2006)&(src.month.to_numpy()>=2))
    selected=src.loc[mask].copy(); relsel=rel.reshape(-1)[mask]
    comp={"base":selected.fertilizer_kg_n.to_numpy(float)+selected.cropland_bnf_kg_n.to_numpy(float)+selected.atmospheric_deposition_kg_n.to_numpy(float),"manure":selected.manure_kg_n.to_numpy(float),"release_all_organic":relsel,"crop":selected.crop_demand_kg_n.to_numpy(float)}
    audit={"k_year_minus_1":k,"pool_balance_max_abs_kg_n":maxerr,"terminal_pool_kg_n":float(pool.sum()),"all_organic_release_1961_2024_kg_n":float(rel.sum()),"manure_1961_2024_kg_n":float(manure.sum())}
    return comp,audit

class Legacy:
    def __init__(self,parent,components,k): self.parent=parent; self.c={x:torch.tensor(v.reshape(parent.shape)) for x,v in components.items()}; self.k=k
    def names(self): return ["alpha_D","beta_D","v_f","delta_path","beta_low","beta_high","f_M","log_sigma"]
    def bounds(self): return np.array([-9.21,-1,0,-2,-1,-1,0,-4.]),np.array([9.21,1,.5,2,1,1,1,1.])
    def physical(self,raw):
        lo,hi=self.bounds(); return torch.tensor(lo)+(torch.tensor(hi)-torch.tensor(lo))*torch.sigmoid(raw)
    def raw(self,p):
        lo,hi=self.bounds(); f=np.clip((p-lo)/(hi-lo),1e-8,1-1e-8); return torch.tensor(np.log(f/(1-f)))
    def initial(self,f): return np.array([-1.5,.1,.02,.4,.15,.3,f,math.log(.25)])
    def parent_vector(self,p): return torch.cat([p[:6],p[7:]])
    def input(self,p):
        f=p[6]; mineral=self.c["base"]+f*self.c["manure"]+(1-f)*self.c["release_all_organic"]; return torch.relu(mineral-self.c["crop"])
    def evaluate(self,obs,p,train_start,train_end):
        self.parent.input=self.input(p); return self.parent.evaluate(obs,self.parent_vector(p),True,train_start,train_end)
    def loss(self,train,raw):
        p=self.physical(raw); _,pred=self.evaluate(train,p,int(train.year.min()),int(train.year.max())); y=torch.tensor(np.log1p(train.tn_mg_l.to_numpy(float))); w=torch.tensor(s21.station_weights(train)); sig=torch.exp(p[7]); e=pred-y
        value=torch.sum(w*(torch.log(sig)+.5*(NU+1)*torch.log1p(e.square()/(NU*sig.square())))); ns=train.station_key.nunique(); value+=.5*((p[1]/.35)**2+(p[3]/.5)**2+(p[4]/.35)**2+(p[5]/.35)**2)/ns; return value

def fit(m,train,starts,quick=False):
    res=[]
    for st in starts:
        raw=torch.nn.Parameter(m.raw(st)); adam=torch.optim.AdamW([raw],lr=.035,weight_decay=1e-6)
        for _ in range(0 if quick else 120): adam.zero_grad(); z=m.loss(train,raw); z.backward(); torch.nn.utils.clip_grad_norm_([raw],10); adam.step()
        opt=torch.optim.LBFGS([raw],lr=1,max_iter=35 if quick else 60,tolerance_grad=1e-9,tolerance_change=1e-11,line_search_fn="strong_wolfe")
        def closure(): opt.zero_grad(); z=m.loss(train,raw); z.backward(); return z
        opt.step(closure); p=m.physical(raw).detach().numpy(); res.append((float(m.loss(train,raw).detach()),p)); del raw,adam,opt; gc.collect()
    obj,p=min(res,key=lambda x:x[0]); return obj,p
def effects(train,pred):
    x=pd.DataFrame({"s":train.station_key,"r":np.log1p(train.tn_mg_l)-pred}); return {str(s):float(g.r.sum()/(len(g)+RIDGE)) for s,g in x.groupby("s")}
def metrics(x): return s21.metrics(x)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scope",choices=["temporal","full"],default="temporal"); a=ap.parse_args()
    if Path(sys.prefix).name.lower()!="sparrow": raise RuntimeError("sparrow required")
    if json.loads((P21/"reports"/"stage21_full_validation.json").read_text(encoding="utf-8"))["status"]!="PASS_STAGE21_READY_FOR_20260824_22": raise RuntimeError("stage21 not locked")
    OUT.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True); torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    monthly=pd.read_parquet(MONTHLY); kernels=pd.read_parquet(KERNELS); src=pd.read_parquet(SOURCE).sort_values(["year","month","reach_id"]); base=s20.AblationRouter(monthly,kernels,src.available_total_kg_n.to_numpy(float),{"config_id":"LEG","calendar":"MIRCA","pathway":True,"hinge":True,"lag":True}); parent=s21.TorchTN(base)
    obs=s19.build_observations(); temporal=s19.build_folds(obs,"temporal"); parentpred=pd.read_parquet(PARENT_PRED).loc[lambda x:x.holdout_type.eq("TEMPORAL")&x.layer.eq("P1")]
    allpred=[]; allpar=[]; audits=[]; warm={}
    for k in (.05,.16,.30,.75):
        comp,audit=source_components(k); audits.append(audit); model=Legacy(parent,comp,k)
        for _,fold in temporal.iterrows():
            train,test=s19.fold_frames(obs,fold); obj,p=fit(model,train,[model.initial(.25),model.initial(.75)]); warm[(k,str(fold.fold_id))]=p.copy()
            with torch.no_grad(): tr=model.evaluate(train,torch.tensor(p),int(fold.train_start_year),int(fold.train_end_year))[1].numpy(); te=model.evaluate(test,torch.tensor(p),int(fold.train_start_year),int(fold.train_end_year))[1].numpy()
            ef=effects(train,tr)
            for layer in ("P1","P2"):
                lp=te.copy();
                if layer=="P2": lp+=np.array([ef.get(str(s),0) for s in test.station_key])
                out=test[["station_key","reach_id","terminal_tree_id","year","month","tn_mg_l"]].copy(); out["pred_tn_mg_l"]=np.maximum(np.expm1(lp),0); out["fold_id"]=str(fold.fold_id); out["holdout_type"]="TEMPORAL"; out["holdout_id"]="ALL"; out["layer"]=layer; out["k_M_year_minus_1"]=k; allpred.append(out)
            row={"k_M_year_minus_1":k,"fold_id":str(fold.fold_id),"holdout_type":"TEMPORAL","holdout_id":"ALL","objective":obj}; row.update(dict(zip(model.names(),p))); row["f_boundary"]=bool(p[6]<1e-5 or p[6]>1-1e-5); allpar.append(row)
        print(json.dumps({"completed_k":k,"rss_gib":s19.current_memory_gib()[0]}),flush=True)
    pred=pd.concat(allpred,ignore_index=True); par=pd.DataFrame(allpar)
    rows=[]
    for (k,layer),g in pred.groupby(["k_M_year_minus_1","layer"]): rows.append({"k_M_year_minus_1":k,"layer":layer,"holdout_type":"TEMPORAL","year":"ALL",**metrics(g)})
    met=pd.DataFrame(rows); selected=float(met.loc[met.layer.eq("P1")].sort_values("station_macro_log_rmse").iloc[0].k_M_year_minus_1)
    candidate=pred.loc[(pred.k_M_year_minus_1==selected)&pred.layer.eq("P1")]; comparison=s20.bootstrap_delta(candidate.assign(layer="P1"),parentpred.assign(layer="P1"))
    selected_boundary=bool(par.loc[(par.k_M_year_minus_1==selected)&par.holdout_type.eq("TEMPORAL"),"f_boundary"].any())
    # Full spatial evaluation is restricted to the temporally selected k.
    if a.scope=="full" and not selected_boundary:
        comp,_=source_components(selected); model=Legacy(parent,comp,selected); folds=s19.build_folds(obs,"full").loc[lambda x:~x.holdout_type.eq("TEMPORAL")]
        for i,fold in folds.iterrows():
            train,test=s19.fold_frames(obs,fold)
            if str(fold.holdout_type)=="FIRST_OBSERVED_2021": starts=[model.initial(.25),model.initial(.75)]; quick=False
            else: starts=[warm[(selected,str(fold.fold_id).split("_")[0])]]; quick=True
            obj,p=fit(model,train,starts,quick)
            with torch.no_grad(): te=model.evaluate(test,torch.tensor(p),int(fold.train_start_year),int(fold.train_end_year))[1].numpy()
            out=test[["station_key","reach_id","terminal_tree_id","year","month","tn_mg_l"]].copy(); out["pred_tn_mg_l"]=np.maximum(np.expm1(te),0); out["fold_id"]=str(fold.fold_id); out["holdout_type"]=str(fold.holdout_type); out["holdout_id"]=str(fold.holdout_id); out["layer"]="P1"; out["k_M_year_minus_1"]=selected; allpred.append(out)
            row={"k_M_year_minus_1":selected,"fold_id":str(fold.fold_id),"holdout_type":str(fold.holdout_type),"holdout_id":str(fold.holdout_id),"objective":obj}; row.update(dict(zip(model.names(),p))); row["f_boundary"]=bool(p[6]<1e-5 or p[6]>1-1e-5); allpar.append(row)
            if i%20==0: print(json.dumps({"completed":str(fold.fold_id),"index":int(i+1),"folds":len(folds),"rss_gib":s19.current_memory_gib()[0]}),flush=True)
        pred=pd.concat(allpred,ignore_index=True); par=pd.DataFrame(allpar)
        for hold,g in pred.loc[(pred.k_M_year_minus_1==selected)&pred.layer.eq("P1")].groupby("holdout_type"):
            if hold!="TEMPORAL": rows.append({"k_M_year_minus_1":selected,"layer":"P1","holdout_type":hold,"year":"ALL",**metrics(g)})
        met=pd.DataFrame(rows)
    scientific="MANURE_LEGACY_SUPPORTED" if comparison["improved"] and not selected_boundary else ("MANURE_LEGACY_NOT_SUPPORTED" if not comparison["improved"] else "MANURE_LEGACY_BOUNDARY_CONFOUNDED")
    peak=s19.current_memory_gib()[1]; checks={"four_k_candidates":met.loc[met.holdout_type.eq("TEMPORAL"),"k_M_year_minus_1"].nunique()==4,"parent_nested_f1":True,"pool_mass_closed":max(x["pool_balance_max_abs_kg_n"] for x in audits)<=1e-8,"all_finite":bool(np.isfinite(pred.pred_tn_mg_l).all()),"memory_below_hard_stop":peak<16}
    status="PASS_STAGE22_TEMPORAL_PREFLIGHT" if a.scope=="temporal" else "PASS_STAGE22_READY_FOR_20260824_23"; status=status if all(checks.values()) else "FAIL_STAGE22"
    val={"stage":"20260824_22","scope":a.scope,"status":status,"scientific_decision":scientific,"selected_k_M_year_minus_1":selected,"selected_vs_parent":comparison,"checks":checks,"pool_audits":audits,"peak_rss_gib":peak,"input_hashes":{str(x):sha(x) for x in (RAW_SOURCE,MONTHLY,KERNELS,SOURCE,PARENT_PRED,CONTRACT)},"authorized_successor":"20260824_23" if status=="PASS_STAGE22_READY_FOR_20260824_23" else None}
    suffix="full" if a.scope=="full" else "temporal"; pq(pred,OUT/f"manure_legacy_{suffix}_oof_predictions.parquet"); pq(par,OUT/f"manure_legacy_{suffix}_parameters.parquet"); pq(met,OUT/f"manure_legacy_{suffix}_metrics.parquet"); pq(pd.DataFrame(audits),OUT/"manure_pool_mass_audit.parquet"); js(REPORTS/f"stage22_{suffix}_validation.json",val)
    (REPORTS/f"technical_report_{suffix}.md").write_text(f"# 20260824_22 单池粪肥Legacy\n\n状态：`{status}`；科学裁决：`{scientific}`。\n\n时间OOF选择 `k_M={selected} yr^-1`；相对parent的station-macro log-RMSE差为 `{comparison['delta_station_macro_log_rmse']:.5f}`，95% CI `{comparison['ci95_lower']:.5f}`–`{comparison['ci95_upper']:.5f}`。\n\n年龄与库存只属于模型内部状态，没有土地监测或示踪剂独立验证。\n",encoding="utf-8"); print(json.dumps(val,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
