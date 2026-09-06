"""Lock the selected TN parent, build final products, and synthesize evidence."""
from __future__ import annotations
import os
for _n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[_n]="1"
import hashlib,importlib.util,json,math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(r"E:\SPARROW");RUN=ROOT/"5_Test"/"20260824_24";P18=ROOT/"5_Test"/"20260824_18";P19=ROOT/"5_Test"/"20260824_19";P20=ROOT/"5_Test"/"20260824_20";P21=ROOT/"5_Test"/"20260824_21";P22=ROOT/"5_Test"/"20260824_22";P23=ROOT/"5_Test"/"20260824_23"
OUT=RUN/"outputs";REPORTS=RUN/"reports";CONTRACT=RUN/"experiment_contract.json";MONTHLY=P19/"outputs"/"corrected_tn_bridge_monthly_2006_2024.parquet";KERNELS=P19/"outputs"/"corrected_daily_carrier_kernels_2006_2024.parquet";SOURCE=P19/"outputs"/"corrected_source_availability_2006_2024.parquet"
def mod(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
s19=mod("s19",P19/"scripts"/"run_stage19.py");s20=mod("s20",P20/"scripts"/"run_stage20.py");s21=mod("s21",P21/"scripts"/"run_stage21.py")
def sha(p):
 h=hashlib.sha256();f=p.open("rb")
 for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 f.close();return h.hexdigest()
def pq(x,p):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+".tmp");x.to_parquet(t,index=False);os.replace(t,p)
def js(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda z:z.item() if isinstance(z,np.generic) else str(z))+"\n",encoding="utf-8")
def residual_month(x):
 y=np.log1p(x.tn_mg_l);p=np.log1p(x.pred_tn_mg_l);z=x.assign(residual_log=y-p);return z.groupby("month",as_index=False).agg(mean_signed_residual_log=("residual_log","mean"),median_signed_residual_log=("residual_log","median"),rmse_log=("residual_log",lambda v:float(np.sqrt(np.mean(v*v)))),n=("residual_log","size"))
def main():
 if Path(sys.prefix).name.lower()!="sparrow":raise RuntimeError("sparrow required")
 locks={"18":json.loads((P18/"reports"/"stage18_final_validation.json").read_text(encoding="utf-8"))["status"],"19":json.loads((P19/"reports"/"stage19_full_validation.json").read_text(encoding="utf-8"))["status"],"20":json.loads((P20/"reports"/"stage20_final_validation.json").read_text(encoding="utf-8"))["status"],"21":json.loads((P21/"reports"/"stage21_full_validation.json").read_text(encoding="utf-8"))["status"],"22":json.loads((P22/"reports"/"stage22_full_validation.json").read_text(encoding="utf-8"))["status"],"23":json.loads((P23/"reports"/"stage23_final_validation.json").read_text(encoding="utf-8"))["status"]}
 if not all(v.startswith("PASS") for v in locks.values()):raise RuntimeError(f"upstream lock failure {locks}")
 OUT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True);torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 monthly=pd.read_parquet(MONTHLY);kernels=pd.read_parquet(KERNELS);src=pd.read_parquet(SOURCE).sort_values(["year","month","reach_id"]);base=s20.AblationRouter(monthly,kernels,src.available_total_kg_n.to_numpy(float),{"config_id":"FINAL","calendar":"MIRCA","pathway":True,"hinge":True,"lag":True});model=s21.TorchTN(base);obs=s19.build_observations();train=obs.loc[obs.year.between(2021,2024)].copy();evaluate=obs.loc[obs.year.between(2016,2024)].copy()
 fit=s21.fit_model(model,train,True,[model.initial(True,0),model.initial(True,1)],160,80);physical=fit["physical"];names=model.names(True)
 with torch.no_grad():tr=model.evaluate(train,torch.tensor(physical),True,2021,2024)[1].numpy();proc,p1=model.evaluate(evaluate,torch.tensor(physical),True,2021,2024);proc,p1=proc.numpy(),p1.numpy()
 effects=s21.p2_effects(train,tr);station=evaluate[["station_key","reach_id","terminal_tree_id","year","month","tn_mg_l"]].copy();station["pred_P0_mg_l"]=np.maximum(np.expm1(proc),0);station["pred_P1_mg_l"]=np.maximum(np.expm1(p1),0);station["pred_P2_mg_l"]=np.maximum(np.expm1(p1+np.array([effects.get(str(s),0) for s in evaluate.station_key])),0);station["period"]=np.where(station.year<=2020,"RETROSPECTIVE_BACKREPORT_2016_2020","APPARENT_FINAL_FIT_2021_2024")
 # Reach-outlet product, driven only by registered source and hydrology.
 reach=monthly[["reach_id","terminal_tree_id","year","month"]].copy();reach["station_key"]="REACH_OUTLET";reach["tn_mg_l"]=0.;reach["downstream_fraction_on_reach"]=1.
 with torch.no_grad():rproc,rp1=model.evaluate(reach,torch.tensor(physical),True,2021,2024)
 product=reach[["reach_id","terminal_tree_id","year","month"]].copy();product["pred_P0_mg_l"]=np.maximum(np.expm1(rproc.numpy()),0);product["pred_P1_mg_l"]=np.maximum(np.expm1(rp1.numpy()),0);product["model_scope"]="transferable_reach_outlet";product["training_period"]="2021-2024"
 metric=[]
 for period,g in station.groupby("period"):
  for layer in ("P0","P1","P2"):
   z=g[["station_key","reach_id","terminal_tree_id","year","month","tn_mg_l"]].copy();z["pred_tn_mg_l"]=g[f"pred_{layer}_mg_l"];metric.append({"program":period,"layer":layer,"year":"ALL",**s21.metrics(z)})
   for year,a in z.groupby("year"):metric.append({"program":period,"layer":layer,"year":str(int(year)),**s21.metrics(a)})
 metric=pd.DataFrame(metric)
 oof=pd.read_parquet(P21/"outputs"/"differentiable_parent_full_oof_predictions.parquet").loc[lambda x:x.holdout_type.eq("TEMPORAL")];month=[]
 for layer,g in oof.groupby("layer"):
  z=residual_month(g);z["layer"]=layer;z["program"]="OOF_2022_2024";month.append(z)
 for layer in ("P0","P1","P2"):
  z=station.loc[station.period.eq("RETROSPECTIVE_BACKREPORT_2016_2020"),["month","tn_mg_l"]].copy();z["pred_tn_mg_l"]=station.loc[station.period.eq("RETROSPECTIVE_BACKREPORT_2016_2020"),f"pred_{layer}_mg_l"].to_numpy();r=residual_month(z);r["layer"]=layer;r["program"]="BACKREPORT_2016_2020";month.append(r)
 month=pd.concat(month,ignore_index=True)
 param=pd.DataFrame([{"objective":fit["objective"],**dict(zip(names,map(float,physical))),"eta_fast":math.exp(float(physical[3])),"eta_slow":math.exp(-float(physical[3])),"station_effect_count":len(effects)}]);eff=pd.DataFrame({"station_key":list(effects),"station_log_intercept":list(effects.values()),"ridge_lambda":12.})
 peak=s19.current_memory_gib()[1];checks={"upstream_locks_pass":True,"river_domain_only":bool(obs.formal_river_channel.all()),"final_train_2021_2024":train.year.min()==2021 and train.year.max()==2024,"backreport_2016_2020":set(station.loc[station.period.str.startswith("RETROSPECTIVE"),"year"])==set(range(2016,2021)),"reach_product_230":product.reach_id.nunique()==230,"reach_month_unique":not product.duplicated(["reach_id","year","month"]).any(),"no_2025_2026":product.year.max()==2024,"all_finite_nonnegative":bool(np.isfinite(product.pred_P1_mg_l).all() and (product.pred_P1_mg_l>=0).all()),"memory_below_hard_stop":peak<16};status="TN_MAINLINE_LOCKED_20260824_24" if all(checks.values()) else "FAIL_STAGE24"
 validation={"stage":"20260824_24","status":status,"selected":"20260824_21_P1","excluded":{"manure_legacy":"boundary_confounded","attribute_regionalization":"tree_and_natural_expansion_not_supported"},"checks":checks,"counts":{"training_observations":len(train),"backreport_observations":int((station.year<=2020).sum()),"reach_month_product_rows":len(product)},"peak_rss_gib":peak,"input_hashes":{str(p):sha(p) for p in (MONTHLY,KERNELS,SOURCE,CONTRACT,P21/"outputs"/"differentiable_parent_full_oof_predictions.parquet")},"authorized_successor":None}
 pq(product,OUT/"canonical_tn_reach_monthly_2006_2024.parquet");pq(station,OUT/"final_station_predictions_2016_2024.parquet");pq(metric,OUT/"final_performance_metrics.parquet");pq(month,OUT/"monthly_residual_diagnostics.parquet");pq(param,OUT/"final_model_parameters.parquet");pq(eff,OUT/"final_station_ridge_effects.parquet");js(REPORTS/"final_validation.json",validation);js(REPORTS/"final_model_lock.json",{"status":status,"architecture":"canonical hydrology -> monthly MIRCA diffuse N -> fixed 36-month availability memory -> Q72 daily-compiled fast/slow carrier -> relative pathway scaling -> bankfull hydraulic exposure -> global C-Q hinge","layers":{"P1":"transferable","P2":"monitored stations only"},"physical_parameters":dict(zip(names,map(float,physical))),"input_hashes":validation["input_hashes"]})
 oofm=pd.read_parquet(P21/"outputs"/"differentiable_parent_full_metrics.parquet");back=metric.loc[(metric.program.str.startswith("RETROSPECTIVE"))&(metric.year.eq("ALL"))];app=metric.loc[(metric.program.str.startswith("APPARENT"))&(metric.year.eq("ALL"))]
 def rows(df):return "\n".join(f"| {r.layer} | {r.rmse_mg_l:.3f} | {r.nse:.3f} | {r.r2:.3f} | {r.station_macro_log_rmse:.4f} |" for _,r in df.iterrows())
 temp=oofm.loc[(oofm.holdout_type.eq("TEMPORAL"))&(oofm.year.eq("ALL"))];reachm=oofm.loc[(oofm.holdout_type.eq("REACH"))&(oofm.layer.eq("P1"))].iloc[0];treem=oofm.loc[(oofm.holdout_type.eq("TREE"))&(oofm.layer.eq("P1"))].iloc[0];nat=oofm.loc[(oofm.holdout_type.eq("FIRST_OBSERVED_2021"))&(oofm.layer.eq("P1"))].iloc[0]
 report=f"""# 20260824_24 TN主线最终技术报告

最终状态：`{status}`。本轮锁定 `_21` P1 作为可迁移TN主线；P2仅服务已有监测站。

## 模型结构

冻结的状态一致快慢流水文模型提供全部流量、快慢分量、库存和月内日尺度carrier；Andreadis只提供河宽/水深几何。TN层使用真实月尺度MIRCA源输入、固定36个月通用availability memory、一个可识别的快慢相对效率、河道水力暴露和全局C–Q hinge。水文参数未被TN反拟合。

## 2022–2024滚动时间OOF

| layer | RMSE mg/L | NSE | R² | station-macro log-RMSE |
|---|---:|---:|---:|---:|
{rows(temp)}

P1的nested留一Reach RMSE为`{reachm.rmse_mg_l:.3f}`、NSE为`{reachm.nse:.3f}`；留一tree RMSE为`{treem.rmse_mg_l:.3f}`、NSE为`{treem.nse:.3f}`。2021自然扩网RMSE为`{nat.rmse_mg_l:.3f}`、NSE为`{nat.nse:.3f}`，仍是明确短板。

## 2016–2020回报

这是用2021–2024拟合后向早期资料的retrospective back-report，不是独立验证。

| layer | RMSE mg/L | NSE | R² | station-macro log-RMSE |
|---|---:|---:|---:|---:|
{rows(back)}

## 2021–2024最终表观拟合

| layer | RMSE mg/L | NSE | R² | station-macro log-RMSE |
|---|---:|---:|---:|---:|
{rows(app)}

## 被淘汰结构

- 单池粪肥Legacy使P1小幅改善，但所有12个候选折均把`f_M`推到0边界，不能识别真实有机粪肥比例，未升级。
- 八属性Bayesian区域化改善时间OOF和留一Reach，但留一tree CI跨0、2021自然扩网skill为负，未升级。
- SciPy与PyTorch在同方程上的目标差为`2.3e-10`，排除了“旧效果差只是优化器未求到解”。

## 仍需保留的限制

P2明显优于P1，说明站点长期偏差仍未完全被可迁移过程解释；P1在2023年及自然扩网仍弱。月份残差表见`monthly_residual_diagnostics.parquet`。下一阶段若重启，必须以新的独立证据注册source identity或观测误差模型，不能自动扩展Legacy、temperature、SAS或Reach-specific参数。
"""; (REPORTS/"technical_report.md").write_text(report,encoding="utf-8")
 # Normalize NumPy scalar values before the final console echo. The JSON lock
 # above already uses the same scalar-safe serializer.
 validation = json.loads(json.dumps(validation, default=lambda z: z.item() if isinstance(z, np.generic) else str(z)))
 literature="""# 文献与结构边界\n\n本轮结构依据包括传统SPARROW的source–delivery–aquatic decay框架、Dynamic SPARROW的月尺度储存与输送、Van Meter与Basu系列关于农田Legacy和土壤有机氮矿化的工作，以及Bayesian SPARROW/stream-network spatial models对空间层级与可迁移性的处理。HBV/PyTorch水文实现只提供冻结水量状态；ELEMeNT、INCA-N、SWAT与HYPE的多池结构只作方程参考，没有被整套移植。\n\n核心识别纪律是：TN浓度只能支持总释放轨迹，不能独立识别真实水龄、土壤库存或矿化比例。因此`f_M`边界改善和内部36个月记忆都不得表述为土地或地下水年龄观测验证。\n""";(REPORTS/"literature_and_structure_synthesis.md").write_text(literature,encoding="utf-8");print(json.dumps(validation,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
