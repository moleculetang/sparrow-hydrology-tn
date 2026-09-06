from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW"); RUN = ROOT / "5_Test" / "20260826_10"; OUT = RUN / "outputs"; REPORT = RUN / "reports"
sys.path[:0] = [str(ROOT / "5_Test" / "20260826_4" / "scripts"), str(ROOT / "5_Test" / "20260825_3" / "scripts"), str(ROOT / "5_Test" / "20260825_5" / "scripts")]
from hydrology_core import load_topology, route_instantaneous  # noqa: E402
from regional_structures import PARAMETER_NAMES, periodic_spinup, run_model  # noqa: E402
from run_stage5 import build_support, kge, nse, station_metrics  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
LOCKED_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_locked_retrospective_2019_2022.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
STATIC = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet"
PARENT_DAILY = ROOT / "5_Test" / "20260825_7" / "outputs" / "tn_hydrology_bridge_daily_2006_2022.parquet"
PARENT_MONTHLY = ROOT / "5_Test" / "20260825_7" / "outputs" / "tn_hydrology_bridge_monthly_2006_2022.parquet"
PARAMETER_FIELDS = ROOT / "5_Test" / "20260826_6" / "outputs" / "spatial_parameter_fields.parquet"
DECISION_LOCK = ROOT / "5_Test" / "20260826_9" / "reports" / "ensemble_and_primary_lock.json"


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): d.update(b)
    return d.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def candidate_frame(model: str, dates: pd.DatetimeIndex, reach_ids: np.ndarray, local: np.ndarray, routed: np.ndarray, length_m: np.ndarray, width: np.ndarray, depth: np.ndarray) -> pd.DataFrame:
    total = routed.sum(axis=2); tau = length_m[None, :] * width[None, :] * depth[None, :] / np.maximum(total, 1e-6) / 86400
    return pd.DataFrame({"date":np.repeat(dates.to_numpy(),len(reach_ids)),"reach_id":np.tile(reach_ids,len(dates)),"model_id":model,"local_fast_response_m3_s":local[:,:,0].reshape(-1),"local_intermediate_response_m3_s":local[:,:,1].reshape(-1),"local_slow_response_m3_s":local[:,:,2].reshape(-1),"local_total_m3_s":local.sum(axis=2).reshape(-1),"routed_fast_response_m3_s":routed[:,:,0].reshape(-1),"routed_intermediate_response_m3_s":routed[:,:,1].reshape(-1),"routed_slow_response_m3_s":routed[:,:,2].reshape(-1),"routed_total_m3_s":total.reshape(-1),"bankfull_hydraulic_exposure_day":tau.reshape(-1),"total_flow_authorized":False,"response_components_identified":False,"TN_role":"STRUCTURAL_SENSITIVITY_ONLY"})


def monthly_from_daily(frame: pd.DataFrame) -> pd.DataFrame:
    temp = frame.copy(); temp["year"] = temp.date.dt.year; temp["month"] = temp.date.dt.month
    numeric = ["local_fast_response_m3_s","local_intermediate_response_m3_s","local_slow_response_m3_s","local_total_m3_s","routed_fast_response_m3_s","routed_intermediate_response_m3_s","routed_slow_response_m3_s","routed_total_m3_s","bankfull_hydraulic_exposure_day"]
    result = temp.groupby(["model_id","year","month","reach_id"],as_index=False)[numeric].mean(); result["total_flow_authorized"] = False; result["response_components_identified"] = False; result["TN_role"] = "STRUCTURAL_SENSITIVITY_ONLY"; return result


def summarize(candidate: str, observed: np.ndarray, predicted: np.ndarray, stations: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    per = station_metrics(candidate, observed, predicted, stations); valid = np.isfinite(observed) & np.isfinite(predicted); o,p=observed[valid],predicted[valid]
    return {"candidate":candidate,"pooled_NSE":nse(o,p),"station_median_NSE":float(per.NSE.median()),"station_mean_NSE":float(per.NSE.mean()),"pooled_log_NSE":nse(np.log1p(o),np.log1p(p)),"pooled_KGE":kge(o,p),"pooled_PBIAS_pct":float(100*(p.sum()-o.sum())/o.sum()),"station_median_absolute_PBIAS_pct":float(per.PBIAS_pct.abs().median()),"pooled_log_RMSE":float(np.sqrt(np.mean((np.log1p(p)-np.log1p(o))**2)))},per


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    decision_lock = json.loads(DECISION_LOCK.read_text(encoding="utf-8")); decision_hash = sha256(DECISION_LOCK)
    access = {"decision_lock_sha256":decision_hash,"decision_locked_before_retrospective_read":True,"parameters_refit_after_retrospective_read":False,"formal_script_retrospective_read_count":0}; write_json(REPORT/"retrospective_access_audit.json",access)
    reach_ids=np.arange(1,231); order,downstream,_=load_topology(TOPOLOGY,reach_ids); area=pd.read_parquet(Q72,columns=["reach_id","catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)
    gauges=pd.read_parquet(GAUGES); stations=gauges.loc[gauges.topology_representative].drop_duplicates("station_norm").sort_values(["terminal_tree","station_norm"]).reset_index(drop=True); support=build_support(stations,reach_ids,order,downstream)
    forcing=pd.read_parquet(FORCING,columns=["reach_id","date","precipitation_daily_mm","pet_fao56_mm_day"]); forcing.date=pd.to_datetime(forcing.date); full_dates=pd.date_range("2006-01-01","2022-12-31"); spin_dates=pd.date_range("2006-01-01","2009-12-31")
    pivot=lambda field,dates:forcing.pivot(index="date",columns="reach_id",values=field).reindex(index=dates,columns=reach_ids).to_numpy(float); full_p,full_pet=pivot("precipitation_daily_mm",full_dates),pivot("pet_fao56_mm_day",full_dates); spin_mask=full_dates.isin(spin_dates)
    geometry=pd.read_parquet(GEOMETRY).set_index("reach_id").reindex(reach_ids); length_m=pd.read_parquet(STATIC,columns=["reach_id","length_km"]).set_index("reach_id").reindex(reach_ids).length_km.to_numpy(float)*1000; width=geometry.bankfull_width_m.to_numpy(float); depth=geometry.bankfull_depth_m.to_numpy(float)
    fields=pd.read_parquet(PARAMETER_FIELDS); candidate_daily={}; spin_rows=[]; local_by_model={}
    for model in ["RAVEN_SACSMA3","MTRS3"]:
        selected=fields.loc[fields.model_id==model].set_index("reach_id").reindex(reach_ids); parameters=selected[PARAMETER_NAMES[model]].to_numpy(float); initial,spin=periodic_spinup(model,full_p[spin_mask],full_pet[spin_mask],parameters,tolerance=1e-8,max_cycles=50); sim=run_model(model,full_p,full_pet,parameters,initial,collect=True); local=sim.response_mm*area[None,:,None]*1000/86400; routed=route_instantaneous(local*86400,reach_ids,order,downstream)/86400; frame=candidate_frame(model,full_dates,reach_ids,local,routed,length_m,width,depth); path=OUT/f"tn_hydrology_sensitivity_daily_{model.lower()}_2006_2022.parquet"; frame.to_parquet(path,index=False); candidate_daily[model]=frame; local_by_model[model]=local; spin_rows.append({"model_id":model,**spin,"full_period_max_abs_mass_error_mm":sim.max_abs_mass_error_mm,"daily_path":str(path),"daily_sha256":sha256(path)})
    pd.DataFrame(spin_rows).to_parquet(OUT/"sensitivity_member_spinup_audit.parquet",index=False)
    parent_month=pd.read_parquet(PARENT_MONTHLY,columns=["year","month","reach_id","local_q0_m3_s","local_q1_m3_s","local_q2_m3_s","local_total_m3_s","routed_q0_m3_s","routed_q1_m3_s","routed_q2_m3_s","routed_total_m3_s","bankfull_travel_time_day"]); parent_month=parent_month.rename(columns={"local_q0_m3_s":"local_fast_response_m3_s","local_q1_m3_s":"local_intermediate_response_m3_s","local_q2_m3_s":"local_slow_response_m3_s","routed_q0_m3_s":"routed_fast_response_m3_s","routed_q1_m3_s":"routed_intermediate_response_m3_s","routed_q2_m3_s":"routed_slow_response_m3_s","bankfull_travel_time_day":"bankfull_hydraulic_exposure_day"}); parent_month["model_id"]="HBV3_PARENT"; parent_month["total_flow_authorized"]=True; parent_month["response_components_identified"]=False; parent_month["TN_role"]="PRIMARY_TOTAL_FLOW_COMPONENTS_SENSITIVITY_ONLY"
    columns=["model_id","year","month","reach_id","local_fast_response_m3_s","local_intermediate_response_m3_s","local_slow_response_m3_s","local_total_m3_s","routed_fast_response_m3_s","routed_intermediate_response_m3_s","routed_slow_response_m3_s","routed_total_m3_s","bankfull_hydraulic_exposure_day","total_flow_authorized","response_components_identified","TN_role"]
    monthly=pd.concat([parent_month[columns]]+[monthly_from_daily(candidate_daily[m])[columns] for m in candidate_daily],ignore_index=True); monthly.to_parquet(OUT/"tn_hydrology_response_ensemble_monthly_2006_2022.parquet",index=False)
    for name in ["fast","intermediate","slow"]: monthly[f"{name}_fraction"]=monthly[f"routed_{name}_response_m3_s"]/monthly.routed_total_m3_s.clip(lower=1e-12)
    grouped=monthly.groupby(["year","month","reach_id"]); uncertainty=grouped.agg(member_count=("model_id","nunique"),total_flow_min_m3_s=("routed_total_m3_s","min"),total_flow_median_m3_s=("routed_total_m3_s","median"),total_flow_max_m3_s=("routed_total_m3_s","max"),fast_fraction_min=("fast_fraction","min"),fast_fraction_median=("fast_fraction","median"),fast_fraction_max=("fast_fraction","max"),intermediate_fraction_min=("intermediate_fraction","min"),intermediate_fraction_median=("intermediate_fraction","median"),intermediate_fraction_max=("intermediate_fraction","max"),slow_fraction_min=("slow_fraction","min"),slow_fraction_median=("slow_fraction","median"),slow_fraction_max=("slow_fraction","max"),hydraulic_exposure_min_day=("bankfull_hydraulic_exposure_day","min"),hydraulic_exposure_median_day=("bankfull_hydraulic_exposure_day","median"),hydraulic_exposure_max_day=("bankfull_hydraulic_exposure_day","max")).reset_index(); uncertainty["slow_fraction_range"]=uncertainty.slow_fraction_max-uncertainty.slow_fraction_min; uncertainty["uncertainty_role"]="STRUCTURAL_SENSITIVITY_NOT_POSTERIOR_CONFIDENCE_INTERVAL"; uncertainty.to_parquet(OUT/"tn_hydrology_structural_uncertainty_monthly_2006_2022.parquet",index=False)
    manifest={"primary_daily_path":str(PARENT_DAILY),"primary_daily_sha256":sha256(PARENT_DAILY),"primary_monthly_path":str(PARENT_MONTHLY),"primary_monthly_sha256":sha256(PARENT_MONTHLY),"sensitivity_daily_members":spin_rows,"combined_monthly_path":str(OUT/"tn_hydrology_response_ensemble_monthly_2006_2022.parquet"),"structural_uncertainty_path":str(OUT/"tn_hydrology_structural_uncertainty_monthly_2006_2022.parquet"),"primary_total_flow_field":"routed_total_m3_s","components_identified":False,"hydraulic_exposure_role":"DIAGNOSTIC_ONLY","member_weights":"none; members are alternative structures, not posterior draws"}; write_json(REPORT/"tn_hydrology_bridge_manifest.json",manifest)
    # The only formal read of locked retrospective discharge occurs here, after all locks.
    locked=pd.read_parquet(LOCKED_Q); access.update({"formal_script_retrospective_read_count":1,"read_at_utc":datetime.now(timezone.utc).isoformat(),"parameters_refit_after_retrospective_read":False}); write_json(REPORT/"retrospective_access_audit.json",access)
    retro_dates=pd.date_range("2019-01-01","2022-12-31"); observed=locked.assign(date=pd.to_datetime(locked.date)).pivot(index="date",columns="station_norm",values="q_m3_s").reindex(index=retro_dates,columns=stations.station_norm).to_numpy(float); retro_mask=full_dates.year>=2019
    parent_daily=pd.read_parquet(PARENT_DAILY,columns=["date","reach_id","local_q0_m3_s","local_q1_m3_s","local_q2_m3_s"]); parent_daily.date=pd.to_datetime(parent_daily.date); parent_local=parent_daily.loc[parent_daily.date.dt.year>=2019].sort_values(["date","reach_id"])[["local_q0_m3_s","local_q1_m3_s","local_q2_m3_s"]].to_numpy(float).reshape(len(retro_dates),230,3); local_by_model["HBV3_PARENT"]=parent_local
    perf_rows=[]; station_rows=[]
    for model,local in local_by_model.items():
        local_retro=local if model=="HBV3_PARENT" else local[retro_mask]; predicted=local_retro.sum(axis=2)@support.T; perf,per=summarize(model,observed,predicted,stations); perf_rows.append(perf); per["model_id"]=model; station_rows.append(per)
    performance=pd.DataFrame(perf_rows); performance.to_parquet(OUT/"locked_retrospective_sensitivity_performance.parquet",index=False); pd.concat(station_rows,ignore_index=True).to_parquet(OUT/"locked_retrospective_sensitivity_station_performance.parquet",index=False)
    final={"stage":"20260826_10","status":"PROGRAM_COMPLETE_PRIMARY_HBV_TOTAL_FLOW_RESPONSE_ENSEMBLE_DIAGNOSTIC","primary_total_flow_model":"HBV3_PARENT_GLOBAL_HBV_R0","primary_total_flow_authorized_for_TN":True,"sensitivity_members":["HBV3_PARENT","RAVEN_SACSMA3","MTRS3"],"sensitivity_total_flow_promoted":False,"component_identifiability_status":"TOTAL_FLOW_SUPPORTED_FAST_INTERMEDIATE_SLOW_NOT_IDENTIFIED","daily_parent_rows":1428070,"monthly_ensemble_rows":int(len(monthly)),"monthly_uncertainty_rows":int(len(uncertainty)),"retrospective_parameters_refit":False,"TN_used_in_hydrology":False,"temperature_used":False,"program_folders":"20260826_1_to_10","repair_slots_remaining":"20260826_11_to_12_only_if_integrity_failure"}; write_json(REPORT/"final_program_decision.json",final)
    stage8=json.loads((ROOT/"5_Test"/"20260826_8"/"reports"/"stage8_decision.json").read_text(encoding="utf-8")); stage7=json.loads((ROOT/"5_Test"/"20260826_7"/"reports"/"stage7_decision.json").read_text(encoding="utf-8")); stage2=json.loads((ROOT/"5_Test"/"20260826_2"/"reports"/"stage2_decision.json").read_text(encoding="utf-8"))
    report=f"""# 20260826 多结构守恒水文响应实验最终报告

## 结论

`{final['status']}`。

本轮没有得到一个可以替代父HBV的全河网快/中/慢水模型。得到的是更精确、也更有用的边界：父HBV总流量继续作为TN唯一正式水量；SAC-SMA与MTRS只作为结构敏感性成员；所有响应分量仍未被独立观测识别。

## 主要成果

1. 修复了DEM静态属性并以零差重放父模型：最大差`{stage2['parent_max_abs_delta_m3_s']}` m3/s。
2. 五结构共同预检通过，SAC-SMA与MTRS通过开发期可行性；MTRS开发期pooled日/月NSE为0.862/0.952。
3. 严格完整河树删除重训中，没有挑战结构通过空间非劣门。SAC-SMA Δlog-RMSE CI95上界`{stage8['spatial_gates']['RAVEN_SACSMA3']['tree_block_ci95_upper']:.4f}`，MTRS为`{stage8['spatial_gates']['MTRS3']['tree_block_ci95_upper']:.4f}`，均高于0.01。
4. 非守恒区域机器学习上限达到开发期留出pooled/站点中位NSE `{stage7['ml_pooled_NSE']:.3f}/{stage7['ml_station_median_NSE']:.3f}`，证明forcing与静态属性仍有未吸收的预测信号，但不能把黑箱状态给TN。
5. 已生成三结构日/月桥与逐Reach-month结构范围。该范围不是posterior置信区间，只能用于TN敏感性传播。

## 锁定回顾诊断（2019–2022）

{performance.to_markdown(index=False)}

回顾期只在所有开发期裁决锁定后读取一次，未再拟合参数。即使候选回顾表现较好，也不能推翻其空间门失败。

## TN使用合同

- 正式水量：父HBV `routed_total_m3_s`。
- 候选成员：必须用各自总量与各自响应分量成套运行，只作结构敏感性；不得把候选比例乘到父总量。
- `fast/intermediate/slow`只表示模型响应速度，不等于地表水/壤中流/地下水、新水/老水或水龄。
- Andreadis只提供宽深；旅行时间使用各成员自己的模拟流量计算，仍为诊断水力暴露，不是已验证河道储量。
- 温度、TN、水库方程、Gauge/P2及WQD参考流量均未参与本轮水文拟合。

## 下一轮真正值得做的技术扩展

不应继续在本阶段增加第六、第七个储库结构。证据指向两件事：一是用质量守恒的可微过程模型吸收机器学习发现的forcing/属性信号；二是补充独立动态状态（遥感土壤湿度、TWS/正式地下水位，最好在代表子流域增加电导率或稳定同位素）。在此之前，快慢分量只能以结构集合和不确定性传给TN。
"""; (REPORT/"technical_report.md").write_text(report,encoding="utf-8")
    integrity_paths=[RUN/"experiment_contract.json",OUT/"sensitivity_member_spinup_audit.parquet",OUT/"tn_hydrology_response_ensemble_monthly_2006_2022.parquet",OUT/"tn_hydrology_structural_uncertainty_monthly_2006_2022.parquet",OUT/"locked_retrospective_sensitivity_performance.parquet",REPORT/"tn_hydrology_bridge_manifest.json",REPORT/"final_program_decision.json",REPORT/"technical_report.md"]; write_json(REPORT/"integrity.json",{str(p):sha256(p) for p in integrity_paths}); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
