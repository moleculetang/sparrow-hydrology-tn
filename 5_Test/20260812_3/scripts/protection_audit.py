from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["comid","q_site","year","month","fold_id"]


def station_key(value: object) -> str:
    """Normalize a registry/model label without inventing geographic aliases."""
    text = str(value).strip().replace("（", "(").replace("）", ")")
    return text[:-1] if text.endswith("站") else text


def metric(frame: pd.DataFrame) -> dict[str,float|int]:
    o=frame.actual.to_numpy(float); p=frame.predict.to_numpy(float); lo=np.log(o); lp=np.log(p)
    return {"n":len(frame),"raw_nse":1-np.sum((p-o)**2)/np.sum((o-o.mean())**2),"log_nse":1-np.sum((lp-lo)**2)/np.sum((lo-lo.mean())**2),"rmse":np.sqrt(np.mean((p-o)**2)),"log_rmse":np.sqrt(np.mean((lp-lo)**2)),"pbias_pct":100*np.sum(p-o)/np.sum(o),"mae_cfs":np.mean(abs(p-o))}


def station_metric(frame:pd.DataFrame)->pd.DataFrame:
    return pd.DataFrame([{"q_site":s,"comid":int(p.comid.iloc[0]),"observed_mean_cfs":p.actual.mean(),**metric(p)} for s,p in frame.groupby("q_site")])


def lowflow(frame:pd.DataFrame, targets:set[str], scenario:str)->pd.DataFrame:
    rows=[]
    for (site,fold),p in frame[frame.q_site.isin(targets)].groupby(["q_site","fold_id"]):
        threshold=p.actual.quantile(.25); z=p[p.actual<=threshold]; e=np.log(z.predict)-np.log(z.actual)
        rows.append({"scenario":scenario,"q_site":site,"fold_id":fold,"n":len(z),"median_log_bias":np.median(e),"absolute_median_log_bias":abs(np.median(e)),"log_rmse":np.sqrt(np.mean(e**2)),"absolute_error_cfs":abs(z.predict-z.actual).sum(),"observed_volume_cfs_months":z.actual.sum(),"pbias_pct":100*(z.predict-z.actual).sum()/z.actual.sum()})
    return pd.DataFrame(rows)


def path_metrics(frame:pd.DataFrame,pairs:pd.DataFrame,scenario:str)->pd.DataFrame:
    rows=[]
    for r in pairs.itertuples(index=False):
        u=frame[frame.comid.eq(int(r.upstream_reach_id))][["year","month","fold_id","actual","predict"]].rename(columns={"actual":"u_actual","predict":"u_predict"})
        d=frame[frame.comid.eq(int(r.downstream_reach_id))][["year","month","fold_id","actual","predict"]].rename(columns={"actual":"d_actual","predict":"d_predict"})
        j=u.merge(d,on=["year","month","fold_id"])
        if len(j): rows.append({"scenario":scenario,"path_key":r.path_key,"upstream_station":r.upstream_station,"downstream_station":r.downstream_station,"n":len(j),"upstream_mae_cfs":abs(j.u_predict-j.u_actual).mean(),"downstream_mae_cfs":abs(j.d_predict-j.d_actual).mean()})
    return pd.DataFrame(rows)


def main()->None:
    frames={s:pd.read_parquet(ROOT/"outputs"/s/"q72_three_fold_oof_predictions.parquet") for s in ["B0","I0"]}
    for frame in frames.values():
        frame["q_site"] = frame["q_site"].astype(str)
        frame["station_key"] = frame["q_site"].map(station_key)
    registry=pd.read_csv(ROOT/"inputs"/"canonical_signal_registry.csv",encoding="utf-8-sig")
    target_labels=set(registry.loc[registry.legacy_canonical_membership.eq(True),"q_site"].astype(str))
    target_keys={station_key(value) for value in target_labels}
    targets=set(frames["B0"].loc[frames["B0"].station_key.isin(target_keys),"q_site"])
    missing_registry_labels=sorted(
        label for label in target_labels if station_key(label) not in set(frames["B0"].station_key)
    )
    pairs=pd.read_csv(ROOT/"inputs"/"confirmed_evaluable_nearest_downstream_paths.csv",encoding="utf-8-sig")
    low=pd.concat([lowflow(f,targets,s) for s,f in frames.items()],ignore_index=True); low.to_csv(ROOT/"reports"/"legacy_lowflow_metrics.csv",index=False,encoding="utf-8-sig")
    station=pd.concat([station_metric(f).assign(scenario=s) for s,f in frames.items()],ignore_index=True); station.to_csv(ROOT/"reports"/"station_metrics.csv",index=False,encoding="utf-8-sig")
    paths=pd.concat([path_metrics(f,pairs,s) for s,f in frames.items()],ignore_index=True); paths.to_csv(ROOT/"reports"/"topology_path_metrics.csv",index=False,encoding="utf-8-sig")
    b=frames["B0"].sort_values(KEY).reset_index(drop=True); i=frames["I0"].sort_values(KEY).reset_index(drop=True)
    q90=b.groupby("q_site").actual.transform(lambda x:x.quantile(.9)); high_b=b[b.actual>=q90]; high_i=i.loc[high_b.index]
    sm_b=station[station.scenario.eq("B0")].set_index("q_site"); large=set(sm_b.nlargest(max(1,int(np.ceil(len(sm_b)*.2))),"observed_mean_cfs").index)
    large_b=b[b.q_site.isin(large)]; large_i=i[i.q_site.isin(large)]
    lb=low.groupby("q_site").agg(abs_bias=("absolute_median_log_bias","median"),log_rmse=("log_rmse","mean")).loc[lambda x:x.index.isin(targets)]
    pivot_bias=low.pivot_table(index="q_site",columns="scenario",values="absolute_median_log_bias",aggfunc="median")
    pivot_lr=low.pivot_table(index="q_site",columns="scenario",values="log_rmse",aggfunc="mean")
    path_p=paths.pivot(index="path_key",columns="scenario",values="downstream_mae_cfs")
    result={
      "legacy_registry_target_count":len(target_labels),"legacy_model_label_match_count":len(targets),"legacy_evaluable_target_count":int(len(pivot_bias)),"legacy_unmatched_registry_labels":missing_registry_labels,"legacy_abs_median_bias_improved_count":int((pivot_bias.I0<pivot_bias.B0).sum()),
      "legacy_pooled_mean_logrmse_change_pct":float(100*(pivot_lr.I0.mean()/pivot_lr.B0.mean()-1)),
      "highflow_logrmse_change_pct":float(100*(metric(high_i)["log_rmse"]/metric(high_b)["log_rmse"]-1)),
      "large_station_logrmse_change_pct":float(100*(metric(large_i)["log_rmse"]/metric(large_b)["log_rmse"]-1)),
      "overall_abs_pbias_change_points":float(abs(metric(i)["pbias_pct"])-abs(metric(b)["pbias_pct"])),
      "downstream_path_mean_mae_change_pct":float(100*(path_p.I0.mean()/path_p.B0.mean()-1)),
      "shijiao":{s:metric(f[f.q_site.str.replace("站","",regex=False).eq("石角")]) for s,f in frames.items()},
      "protection_pass":False}
    result["protection_pass"]=bool(result["highflow_logrmse_change_pct"]<=2 and result["large_station_logrmse_change_pct"]<=2 and result["overall_abs_pbias_change_points"]<=2 and result["downstream_path_mean_mae_change_pct"]<=2)
    (ROOT/"logs"/"I0_protection_gate.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
