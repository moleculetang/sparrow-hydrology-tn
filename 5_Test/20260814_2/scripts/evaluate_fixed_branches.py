from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


assert_sparrow_runtime()
ROOT=Path(__file__).resolve().parents[1]; STAGE1=ROOT.parent/"20260814_1"
BRANCHES=["main","flash","slow","buffer","wet"]
KEY=["comid","q_site","year","month","fold_id"]
FOLDS=[("fit_2006_2011_eval_2012_2013",2011),("fit_2006_2013_eval_2014_2015",2013),("fit_2006_2015_eval_2016_2018",2015)]
EPS=1e-9; NBOOT=10000; SEED=20260814


def metric(obs: np.ndarray,pred: np.ndarray)->dict[str,float]:
    obs=np.asarray(obs,float); pred=np.asarray(pred,float); good=np.isfinite(obs)&np.isfinite(pred)&(obs>0)&(pred>0); obs=obs[good]; pred=pred[good]
    lo=np.log(obs); lp=np.log(pred)
    return {"n":len(obs),"raw_nse":float(1-np.sum((pred-obs)**2)/np.sum((obs-obs.mean())**2)),
            "log_nse":float(1-np.sum((lp-lo)**2)/np.sum((lo-lo.mean())**2)),
            "pbias_pct":float(100*np.sum(pred-obs)/np.sum(obs)),"log_rmse":float(np.sqrt(np.mean((lp-lo)**2)))}


def terminal_map()->dict[int,int]:
    topo=pd.read_csv(STAGE1/"inputs"/"topology"/"topology_edges.csv")
    down={int(r.reach_id):(None if pd.isna(r.downstream_reach) or str(r.downstream_reach).strip()=="" else int(float(str(r.downstream_reach).split(",")[0]))) for r in topo.itertuples(index=False)}
    result={}
    for rid in down:
        seen=set(); cur=rid
        while down.get(cur) is not None and cur not in seen:
            seen.add(cur); cur=int(down[cur])
        result[rid]=cur
    return result


def build_lowflow(h0:pd.DataFrame)->pd.DataFrame:
    dev=pd.read_parquet(STAGE1/"inputs"/"development_indata_2006_2018.parquet",columns=["comid","q_site","year","month","Q_obsv_cfs"])
    dev=dev[dev.Q_obsv_cfs.notna()&dev.Q_obsv_cfs.gt(0)].copy(); dev.q_site=dev.q_site.astype(str)
    rows=[]
    for fold,train_end in FOLDS:
        q20=dev[dev.year<=train_end].groupby("q_site").Q_obsv_cfs.quantile(.20).rename("train_q20_cfs")
        part=h0[h0.fold_id.eq(fold)].merge(q20,on="q_site",validate="many_to_one")
        part=part[part.actual<=part.train_q20_cfs][KEY+["actual","train_q20_cfs"]].copy(); rows.append(part)
    registry=pd.concat(rows,ignore_index=True); registry.to_parquet(ROOT/"reports"/"lowflow_registry.parquet",index=False)
    return registry


def attach(model:pd.DataFrame,registry:pd.DataFrame)->pd.DataFrame:
    return registry.merge(model[KEY+["predict"]],on=KEY,validate="one_to_one")


def event_metrics(model:pd.DataFrame,tail_months:pd.DataFrame,events:pd.DataFrame)->tuple[float,float,pd.DataFrame]:
    tail=tail_months.merge(model[KEY+["predict"]],on=KEY,validate="one_to_one")
    peaks=events.rename(columns={"peak_year":"year","peak_month":"month","peak_observed_cfs":"peak_observed"})
    peaks=peaks.merge(model[KEY+["predict"]].rename(columns={"predict":"peak_predict"}),on=KEY,validate="one_to_one")
    tail=tail.merge(peaks[["event_id","peak_observed","peak_predict"]],on="event_id",validate="many_to_one")
    tail["shape_sq"]=(np.log((tail.predict+EPS)/(tail.peak_predict+EPS))-np.log((tail.observed_cfs+EPS)/(tail.peak_observed+EPS)))**2
    shape=float(np.sqrt(tail.shape_sq.mean()))
    volume=tail.groupby(["event_id","q_site"],as_index=False).agg(obs=("observed_cfs","sum"),pred=("predict","sum"))
    volume["abs_log_error"]=(np.log((volume.pred+EPS)/(volume.obs+EPS))).abs()
    return shape,float(volume.abs_log_error.mean()),tail


def cluster_mse(frame:pd.DataFrame,obs_col:str="actual")->pd.Series:
    x=frame.copy(); x["sq"]=(np.log(x.predict.clip(lower=EPS))-np.log(x[obs_col].clip(lower=EPS)))**2
    return x.groupby("q_site").sq.mean()


def paired_ci(cand:pd.Series,base:pd.Series,blocks:pd.Series|None=None)->dict[str,float|bool]:
    common=cand.index.intersection(base.index); c=cand.loc[common].to_numpy(float); b=base.loc[common].to_numpy(float)
    rng=np.random.default_rng(SEED); idx=rng.integers(0,len(common),size=(NBOOT,len(common)))
    diff=np.sqrt(c[idx].mean(axis=1))-np.sqrt(b[idx].mean(axis=1)); point=float(np.sqrt(c.mean())-np.sqrt(b.mean()))
    lo,hi=np.quantile(diff,[.025,.975])
    return {"clusters":len(common),"point_difference":point,"ci95_low":float(lo),"ci95_high":float(hi),"stable_improvement":bool(hi<0)}


def spatial_difference(cand:pd.Series,base:pd.Series,station_to_terminal:dict[str,int])->float:
    common=cand.index.intersection(base.index); table=pd.DataFrame({"cand":cand.loc[common],"base":base.loc[common]}); table["terminal"]=[station_to_terminal.get(str(s),-1) for s in table.index]
    block=table.groupby("terminal")[["cand","base"]].mean(); return float(np.sqrt(block.cand.mean())-np.sqrt(block.base.mean()))


def main()->None:
    reports=ROOT/"reports"; reports.mkdir(parents=True,exist_ok=True)
    h0=pd.read_parquet(STAGE1/"outputs"/"P1"/"q72_three_fold_oof_predictions.parquet"); h0.q_site=h0.q_site.astype(str)
    h0=h0.rename(columns={"scenario_id":"model_id"}); h0["model_id"]="H0_hybrid"
    models={"H0_hybrid":h0}
    for branch in BRANCHES:
        x=pd.read_parquet(ROOT/"outputs"/branch/"oof.parquet"); x.q_site=x.q_site.astype(str); models[branch]=x
    lowreg=build_lowflow(h0)
    tails=pd.read_parquet(STAGE1/"inputs"/"registries"/"tail_month_registry.parquet"); events=pd.read_parquet(STAGE1/"inputs"/"registries"/"tail_event_registry.parquet")
    tails.q_site=tails.q_site.astype(str); events.q_site=events.q_site.astype(str)
    tmap=terminal_map(); station_comid=h0.groupby("q_site").comid.first().astype(int); station_terminal={s:tmap.get(int(r),int(r)) for s,r in station_comid.items()}

    summaries=[]; foldrows=[]; loss={}; eventdiag={}
    for name,model in models.items():
        overall=metric(model.actual.to_numpy(float),model.predict.to_numpy(float)); low=attach(model,lowreg); tail=attach(model,tails.rename(columns={"observed_cfs":"actual"})[KEY+["event_id","tail_lag","actual"]])
        shape,volume,taildetail=event_metrics(model,tails,events); eventdiag[name]=(shape,volume)
        low_mse=cluster_mse(low); tail_mse=cluster_mse(tail); loss[name]={"low":low_mse,"tail":tail_mse}
        summaries.append({"model_id":name,**overall,"lowflow_log_rmse":float(np.sqrt(low_mse.mean())),"tail_log_rmse":float(np.sqrt(tail_mse.mean())),"tail_shape_rmse":shape,"event_volume_abs_log_error":volume})
        for fold,train_end in FOLDS:
            mf=model[model.fold_id.eq(fold)]; lf=low[low.fold_id.eq(fold)]; tf=tail[tail.fold_id.eq(fold)]
            foldrows.append({"model_id":name,"fold_id":fold,**metric(mf.actual,mf.predict),"lowflow_log_rmse":metric(lf.actual,lf.predict)["log_rmse"],"tail_log_rmse":metric(tf.actual,tf.predict)["log_rmse"]})
    summary=pd.DataFrame(summaries).set_index("model_id"); fold=pd.DataFrame(foldrows)
    base=summary.loc["H0_hybrid"]; decisions=[]; bootrows=[]
    for name in BRANCHES:
        low_ci=paired_ci(loss[name]["low"],loss["H0_hybrid"]["low"]); tail_ci=paired_ci(loss[name]["tail"],loss["H0_hybrid"]["tail"])
        low_sp=spatial_difference(loss[name]["low"],loss["H0_hybrid"]["low"],station_terminal); tail_sp=spatial_difference(loss[name]["tail"],loss["H0_hybrid"]["tail"],station_terminal)
        low_ci["terminal_point_difference"]=low_sp; tail_ci["terminal_point_difference"]=tail_sp
        low_ci["spatial_direction_consistent"]=bool(np.sign(low_ci["point_difference"])==np.sign(low_sp) or abs(low_sp)<1e-12)
        tail_ci["spatial_direction_consistent"]=bool(np.sign(tail_ci["point_difference"])==np.sign(tail_sp) or abs(tail_sp)<1e-12)
        bootrows += [{"model_id":name,"metric":"lowflow_log_rmse",**low_ci},{"model_id":name,"metric":"tail_log_rmse",**tail_ci}]
        sf=fold[fold.model_id.eq(name)].set_index("fold_id"); bf=fold[fold.model_id.eq("H0_hybrid")].set_index("fold_id")
        low_folds=int((sf.lowflow_log_rmse<bf.lowflow_log_rmse).sum()); tail_folds=int((sf.tail_log_rmse<bf.tail_log_rmse).sum())
        row=summary.loc[name]
        raw_ok=row.raw_nse-base.raw_nse>=-0.005; log_ok=row.log_nse-base.log_nse>=-0.005
        pbias_ok=abs(row.pbias_pct)<=abs(base.pbias_pct)+3.0; shape_ok=row.tail_shape_rmse<=base.tail_shape_rmse+1e-12; volume_ok=row.event_volume_abs_log_error<=base.event_volume_abs_log_error+1e-12
        eligible=bool(raw_ok and log_ok and pbias_ok and low_folds>=2 and tail_folds>=2 and low_ci["stable_improvement"] and tail_ci["stable_improvement"] and low_ci["spatial_direction_consistent"] and tail_ci["spatial_direction_consistent"] and shape_ok and volume_ok)
        anchor_ok=bool(raw_ok and log_ok and pbias_ok and (row.lowflow_log_rmse<base.lowflow_log_rmse or row.tail_log_rmse<base.tail_log_rmse) and row.tail_shape_rmse<=base.tail_shape_rmse+0.01 and row.event_volume_abs_log_error<=base.event_volume_abs_log_error+0.01)
        decisions.append({"model_id":name,"delta_raw_nse":row.raw_nse-base.raw_nse,"delta_log_nse":row.log_nse-base.log_nse,"delta_abs_pbias":abs(row.pbias_pct)-abs(base.pbias_pct),"lowflow_improved_folds":low_folds,"tail_improved_folds":tail_folds,"lowflow_ci95_high":low_ci["ci95_high"],"tail_ci95_high":tail_ci["ci95_high"],"tail_shape_delta":row.tail_shape_rmse-base.tail_shape_rmse,"event_volume_delta":row.event_volume_abs_log_error-base.event_volume_abs_log_error,"fixed_eligible":eligible,"grid_anchor_eligible":anchor_ok})
    decisions=pd.DataFrame(decisions); boot=pd.DataFrame(bootrows)
    qualified=decisions[decisions.fixed_eligible].model_id.tolist()
    if qualified:
        ranked=summary.loc[qualified].sort_values(["tail_log_rmse","lowflow_log_rmse","log_nse","raw_nse"],ascending=[True,True,False,False]); winner=str(ranked.index[0]); action="fixed_winner_skip_grid"
    else:
        anchors=decisions[decisions.grid_anchor_eligible].model_id.tolist()
        if anchors:
            # Deterministic approximation of the pre-registered anchor ordering.
            ad=decisions.set_index("model_id").loc[anchors].join(summary[["tail_log_rmse","lowflow_log_rmse","log_nse","raw_nse"]])
            ad["prefer_main"]=ad.index.to_series().eq("main").astype(int); ad=ad.sort_values(["lowflow_improved_folds","tail_improved_folds","tail_log_rmse","lowflow_log_rmse","log_nse","raw_nse","prefer_main"],ascending=[False,False,True,True,False,False,False])
            winner=str(ad.index[0]); action="run_conditional_grid"
        else: winner="H0_hybrid"; action="retain_H0_skip_grid"
    summary.reset_index().to_csv(reports/"model_metric_summary.csv",index=False,encoding="utf-8-sig"); fold.to_csv(reports/"fold_metric_summary.csv",index=False,encoding="utf-8-sig"); decisions.to_csv(reports/"flow_gate_decisions.csv",index=False,encoding="utf-8-sig"); boot.to_csv(reports/"paired_bootstrap.csv",index=False,encoding="utf-8-sig")
    gate={"performance_type":"fixed_structure_oof","fixed_qualified":qualified,"selected_or_anchor":winner,"next_action":action,"bootstrap_replicates":NBOOT,"pbias_guard_pp":3.0,"groundwater_evaluation_required":bool(qualified),"pass":True}
    (reports/"stage2_flow_gate.json").write_text(json.dumps(gate,indent=2),encoding="utf-8"); print(json.dumps(gate,indent=2))


if __name__=="__main__": main()
