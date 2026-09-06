from __future__ import annotations

import importlib.util
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


warnings.filterwarnings("ignore")
assert_sparrow_runtime()
ROOT=Path(__file__).resolve().parents[1]; STAGE1=ROOT.parent/"20260814_1"; STAGE2=ROOT.parent/"20260814_2"
COMPONENT=ROOT/"scripts"/"components"/"q72_prior_semantics_component.py"; INPUT=STAGE1/"inputs"/"development_indata_2006_2018.parquet"; TOPO=STAGE1/"inputs"/"topology"/"topology_edges.csv"
FOLDS=[{"fold_id":"fit_2006_2011_eval_2012_2013","inner_end":2009,"train_end":2011,"eval_start":2012,"eval_end":2013},
       {"fold_id":"fit_2006_2013_eval_2014_2015","inner_end":2011,"train_end":2013,"eval_start":2014,"eval_end":2015},
       {"fold_id":"fit_2006_2015_eval_2016_2018","inner_end":2013,"train_end":2015,"eval_start":2016,"eval_end":2018}]
BASE={"rho":.70,"wm":480.,"et_gamma":.75,"sas_rho":.93,"young_k":1.5,"storage_scale":720.,"fixed_sigma":3.,"production_sigma":1.5,"group_sigma":1.5,"multistore_sigma":.30,"station_sigma":1.,"slope_sigma":.15,"regime_slope_sigma":.25}
ANCHOR={"prod_capacity":240.,"runoff_gamma":2.5,"quick_rho":.25,"k_p":.10,"rho_b":.85,"highflow_scale":1.0}
MULT=[.5,2/3,1.,1.5,2.]; NBOOT=10000; SEED=20260814; EPS=1e-9


def component(label:str):
    spec=importlib.util.spec_from_file_location(f"q72_grid_{label}",COMPONENT); m=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(m)
    m.INPUT_PATH=INPUT; m.TOPOLOGY_PATH=TOPO; m.STATE_CALENDAR_MODE="full_forcing"; m.FORCING_SEMANTICS_MODE="prescribed_aet_balance"; m.MASS_ACCOUNTING_MODE="explicit_upstream_volume"
    m.configure_engineering_repair(True); m.configure_full_gaussian_prior_space(True); m.ZERO_PRESERVING_FULL_PRIOR_MODE=True; m.DETERMINISTIC_SPINUP_MODE=True; m.ET_STATE_OPERATOR_MODE="baseline_clip"
    original=m.set_et_feature_block_mode
    def gate(mode):
        original(mode); removed=set(m.MULTISTORE_FEATURES); m.FIXED_FEATURES=[f for f in m.FIXED_FEATURES if f not in removed]; m.MULTISTORE_FEATURES=[]
    m.set_et_feature_block_mode=gate; m.set_et_feature_block_mode("full")
    return m


def grid()->pd.DataFrame:
    tp=(1-ANCHOR["k_p"])/ANCHOR["k_p"]; tb=ANCHOR["rho_b"]/(1-ANCHOR["rho_b"]); rows=[]
    for mp in MULT:
        for mb in MULT:
            kp=1/(1+mp*tp); rb=(mb*tb)/(1+mb*tb)
            rows.append({"combo_id":f"mp_{mp:.8g}__mb_{mb:.8g}","m_p":mp,"m_b":mb,"k_p":kp,"rho_b":rb,"distance":math.hypot(math.log(mp),math.log(mb)),"is_anchor":abs(mp-1)<1e-12 and abs(mb-1)<1e-12})
    return pd.DataFrame(rows)


def featured(m,forcing,kp,rb):
    return m.build_featured_observation_panel(forcing,rho=BASE["rho"],wm=BASE["wm"],et_gamma=BASE["et_gamma"],sas_rho=BASE["sas_rho"],young_k=BASE["young_k"],storage_scale=BASE["storage_scale"],prod_capacity=ANCHOR["prod_capacity"],runoff_gamma=ANCHOR["runoff_gamma"],quick_rho=ANCHOR["quick_rho"],base_rho=rb,base_release=kp,state_calendar_mode="full_forcing")


def fit_predict(m,frame,stations,train_end,eval_start,eval_end,hs):
    tr=frame[frame.year<=train_end].copy(); ev=frame[frame.year.between(eval_start,eval_end)].copy(); mean,std=m.standardize_fit(tr)
    beta=m.fit_map_ridge(tr,stations,mean,std,fixed_sigma=BASE["fixed_sigma"],production_sigma=BASE["production_sigma"],group_sigma=BASE["group_sigma"],multistore_sigma=BASE["multistore_sigma"],hysteresis_sigma=hs,station_sigma=BASE["station_sigma"],slope_sigma=BASE["slope_sigma"],regime_slope_sigma=BASE["regime_slope_sigma"],anomaly_weight=0.,flow_contrast_weight=1.)
    ev=ev[["comid","q_site","year","month","Q_obsv_cfs"]].copy(); ev["predict"]=np.exp(np.clip(m.predict_log(frame[frame.year.between(eval_start,eval_end)],beta,stations,mean,std),-20,20)); ev=ev.rename(columns={"Q_obsv_cfs":"actual"}); return ev


def nse(obs,pred,log=False):
    o=np.asarray(obs,float); p=np.asarray(pred,float)
    if log:o=np.log(np.clip(o,EPS,None));p=np.log(np.clip(p,EPS,None))
    return float(1-np.sum((p-o)**2)/np.sum((o-o.mean())**2))


def registry(frame,train_end,es,ee):
    obs=frame[["comid","q_site","year","month","Q_obsv_cfs","upstream_positive_input_equivalent_cfs"]].copy(); obs=obs[obs.Q_obsv_cfs.notna()&obs.Q_obsv_cfs.gt(0)]
    q20=obs[obs.year<=train_end].groupby("q_site").Q_obsv_cfs.quantile(.2).rename("q20"); low=obs[obs.year.between(es,ee)].merge(q20,on="q_site"); low=low[low.Q_obsv_cfs<=low.q20][["comid","q_site","year","month","Q_obsv_cfs"]].rename(columns={"Q_obsv_cfs":"actual"})
    events=[]; months=[]
    for site,allsite in obs.groupby("q_site"):
        train=allsite[allsite.year<=train_end]; ev=allsite[allsite.year.between(es,ee)].sort_values(["year","month"]).reset_index(drop=True)
        if train.empty or len(ev)<3:continue
        q75=float(train.Q_obsv_cfs.quantile(.75)); wet75=float(train.upstream_positive_input_equivalent_cfs.quantile(.75))
        for i in range(1,len(ev)-1):
            q=float(ev.at[i,"Q_obsv_cfs"])
            if not(q>=q75 and q>=float(ev.at[i-1,"Q_obsv_cfs"]) and q>float(ev.at[i+1,"Q_obsv_cfs"])):continue
            tail=[];prev=q
            for j in range(i+1,min(i+7,len(ev))):
                p0=pd.Period(f"{int(ev.at[j-1,'year'])}-{int(ev.at[j-1,'month']):02d}");p1=pd.Period(f"{int(ev.at[j,'year'])}-{int(ev.at[j,'month']):02d}");cur=float(ev.at[j,"Q_obsv_cfs"])
                if p1.ordinal-p0.ordinal!=1 or float(ev.at[j,"upstream_positive_input_equivalent_cfs"])>wet75 or cur>prev:break
                tail.append(j);prev=cur
            if len(tail)<2:continue
            eid=f"{site}::{int(ev.at[i,'year'])}-{int(ev.at[i,'month']):02d}"
            events.append({"event_id":eid,"q_site":str(site),"peak_year":int(ev.at[i,"year"]),"peak_month":int(ev.at[i,"month"]),"peak_observed":q})
            for lag,j in enumerate(tail,1):months.append({"event_id":eid,"comid":int(ev.at[j,"comid"]),"q_site":str(site),"year":int(ev.at[j,"year"]),"month":int(ev.at[j,"month"]),"actual":float(ev.at[j,"Q_obsv_cfs"]),"tail_lag":lag})
    return low,pd.DataFrame(events),pd.DataFrame(months)


def diagnostics(pred,low,events,tail):
    keys=["comid","q_site","year","month"]
    lowp=low.merge(pred[keys+["predict"]],on=keys,validate="one_to_one"); tailp=tail.merge(pred[keys+["predict"]],on=keys,validate="one_to_one")
    peaks=events.rename(columns={"peak_year":"year","peak_month":"month"}).merge(pred[keys+["predict"]].rename(columns={"predict":"peak_predict"}),on=["q_site","year","month"],validate="one_to_one")
    tailp=tailp.merge(peaks[["event_id","peak_observed","peak_predict"]],on="event_id",validate="many_to_one")
    logerr=lambda x:(np.log(x.predict.clip(lower=EPS))-np.log(x.actual.clip(lower=EPS)))
    allerr=logerr(pred); lowerr=logerr(lowp); tailerr=logerr(tailp)
    tailp["shape_sq"]=(np.log((tailp.predict+EPS)/(tailp.peak_predict+EPS))-np.log((tailp.actual+EPS)/(tailp.peak_observed+EPS)))**2
    vol=tailp.groupby(["event_id","q_site"]).agg(obs=("actual","sum"),pr=("predict","sum")); volerr=np.log((vol.pr+EPS)/(vol.obs+EPS)).abs()
    return {"raw_nse":nse(pred.actual,pred.predict),"log_nse":nse(pred.actual,pred.predict,True),"pbias":float(100*np.sum(pred.predict-pred.actual)/np.sum(pred.actual)),
            "tail_rmse":float(np.sqrt(np.mean(tailerr**2))),"low_rmse":float(np.sqrt(np.mean(lowerr**2))),"low_abs_log_bias":float(abs(np.log((lowp.predict.sum()+EPS)/(lowp.actual.sum()+EPS)))),"tail_shape":float(np.sqrt(tailp.shape_sq.mean())),"event_volume":float(volerr.mean()),
            "loss_tail":tailp.assign(sq=tailerr**2).groupby("q_site").sq.mean(),"loss_low":lowp.assign(sq=lowerr**2).groupby("q_site").sq.mean(),"loss_log":pred.assign(sq=allerr**2).groupby("q_site").sq.mean(),"loss_raw":pred.assign(sq=(pred.predict-pred.actual)**2).groupby("q_site").sq.mean()}


def one_se(candidates,diag,key,lower=True):
    best=min(candidates,key=lambda c:diag[c][key]) if lower else max(candidates,key=lambda c:diag[c][key]); keep=[]; rng=np.random.default_rng(SEED)
    losskey={"tail_rmse":"loss_tail","low_rmse":"loss_low","log_nse":"loss_log","raw_nse":"loss_raw"}[key]
    for c in candidates:
        a=diag[c][losskey]; b=diag[best][losskey]; common=a.index.intersection(b.index); av=a.loc[common].to_numpy();bv=b.loc[common].to_numpy();idx=rng.integers(0,len(common),size=(NBOOT,len(common)))
        d=np.sqrt(av[idx].mean(1))-np.sqrt(bv[idx].mean(1));se=float(d.std(ddof=1)); degradation=(diag[c][key]-diag[best][key])*(-1 if not lower else 1)
        if degradation<=se+1e-12:keep.append(c)
    return keep


def main():
    (ROOT/"reports").mkdir(parents=True,exist_ok=True); (ROOT/"outputs").mkdir(parents=True,exist_ok=True)
    g=grid(); g.to_csv(ROOT/"reports"/"grid_parameters.csv",index=False,encoding="utf-8-sig"); selections=[]; outer=[]
    anchor_hs=pd.read_parquet(STAGE2/"outputs"/"main"/"oof.parquet").groupby("fold_id").hysteresis_sigma.first().to_dict()
    for fold in FOLDS:
        m=component(str(fold["train_end"])); m.CAL_END_YEAR=fold["train_end"]; m.INNER_TRAIN_END_YEAR=fold["inner_end"]; fold_dir=ROOT/"outputs"/fold["fold_id"]; fold_dir.mkdir(parents=True,exist_ok=True); m.REPORT_DIR=fold_dir/"fit_artifacts";m.REPORT_DIR.mkdir(parents=True,exist_ok=True)
        forcing=m.load_forcing_panel(); hs=float(anchor_hs[fold["fold_id"]]); stations=sorted(forcing.loc[forcing.Q_obsv_cfs.notna()&forcing.Q_obsv_cfs.gt(0),"q_site"].astype(str).unique())
        anchor_frame=featured(m,forcing,ANCHOR["k_p"],ANCHOR["rho_b"]); low,events,tail=registry(anchor_frame,fold["inner_end"],fold["inner_end"]+1,fold["train_end"])
        diag={}; frames={}; inner_rows=[]
        for row in g.itertuples(index=False):
            frame=featured(m,forcing,row.k_p,row.rho_b); frames[row.combo_id]=frame
            pred=fit_predict(m,frame,stations,fold["inner_end"],fold["inner_end"]+1,fold["train_end"],hs); d=diagnostics(pred,low,events,tail);diag[row.combo_id]=d
            inner_rows.append({"combo_id":row.combo_id,"m_p":row.m_p,"m_b":row.m_b,"k_p":row.k_p,"rho_b":row.rho_b,**{k:v for k,v in d.items() if not k.startswith("loss_")}})
        tab=pd.DataFrame(inner_rows); anchor_id=str(g[g.is_anchor].iloc[0].combo_id); a=tab.set_index("combo_id").loc[anchor_id]
        eligible=[]
        for r in tab.itertuples(index=False):
            ok=(r.combo_id==anchor_id) or (r.raw_nse-a.raw_nse>=-.005 and r.log_nse-a.log_nse>=-.005 and abs(r.pbias)<=abs(a.pbias)+3 and r.low_abs_log_bias<=a.low_abs_log_bias+.01 and r.tail_shape<=a.tail_shape+.01 and r.event_volume<=a.event_volume+.01)
            if ok:eligible.append(r.combo_id)
        selected=one_se(eligible,diag,"tail_rmse",True);selected=one_se(selected,diag,"low_rmse",True);selected=one_se(selected,diag,"log_nse",False);selected=one_se(selected,diag,"raw_nse",False)
        if len(selected)>1:
            dist=g.set_index("combo_id").loc[selected].distance; selected=[str(dist.idxmin())]
        winner=selected[0] if selected else anchor_id; wr=g.set_index("combo_id").loc[winner]
        final=fit_predict(m,frames[winner],stations,fold["train_end"],fold["eval_start"],fold["eval_end"],hs);final["fold_id"]=fold["fold_id"];final["combo_id"]=winner;final["k_p"]=wr.k_p;final["rho_b"]=wr.rho_b;outer.append(final)
        tab["eligible"]=tab.combo_id.isin(eligible);tab["selected"]=tab.combo_id.eq(winner);tab.to_csv(fold_dir/"inner_grid_metrics.csv",index=False,encoding="utf-8-sig")
        selections.append({"fold_id":fold["fold_id"],"winner":winner,"k_p":float(wr.k_p),"rho_b":float(wr.rho_b),"eligible_non_anchor":len([x for x in eligible if x!=anchor_id]),"hysteresis_sigma":hs})
        print(json.dumps(selections[-1]))
    oof=pd.concat(outer,ignore_index=True);oof["performance_type"]="conditional_parameter_oof_after_structure_selection";oof.to_parquet(ROOT/"outputs"/"conditional_grid_oof.parquet",index=False);pd.DataFrame(selections).to_csv(ROOT/"reports"/"outer_fold_parameter_selections.csv",index=False,encoding="utf-8-sig")
    gate={"rows":len(oof),"unique_keys":int(oof[["comid","q_site","year","month","fold_id"]].drop_duplicates().shape[0]),"performance_type":"conditional_parameter_oof_after_structure_selection","selections":selections,"pass":len(oof)==7755}
    (ROOT/"reports"/"grid_run_gate.json").write_text(json.dumps(gate,indent=2),encoding="utf-8");print(json.dumps(gate,indent=2))


if __name__=="__main__":main()
