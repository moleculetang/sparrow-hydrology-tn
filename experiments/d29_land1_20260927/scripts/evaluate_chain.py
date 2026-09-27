"""Standard paired statistics for full-chain software controls and compatibility.

No fit, source selection, evaluation-label tuning or scientific model selection.
The synthetic pulse is truth and identity baseline; smooth is the comparison.
Real-label scores, if present, are frozen TRAINING compatibility regressions.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from d29_platform.metrics import CoveragePolicy,evaluate_pair,monthly_views,paired_calendar_bootstrap

OUT=ROOT/"outputs/chain/metrics"
DAILY_POLICY=CoveragePolicy(min_days=120,min_months=6)
MONTHLY_POLICY=CoveragePolicy(min_days=6,min_months=6)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def save_pair(table,name,scope,policy,role):
    station,summary=evaluate_pair(table,policy=policy,scope=scope)
    station["evidence_role"]=role
    summary["evidence_role"]=role
    station.to_csv(OUT/f"{name}_station.csv",index=False)
    summary.to_csv(OUT/f"{name}_paired_summary.csv",index=False)
    return {"rows":len(table),"stations_before_coverage":int(table.station_key.nunique()),
            "coverage":vars(policy),"summary":json.loads(summary.to_json(orient="records"))}


def compatibility():
    snap=ROOT/"vendor/legacy22"
    anchor=json.loads((snap/"input_potential_v2/configs/anchors.json").read_text(encoding="utf-8"))
    records=[]
    for fold,years in (("F23",[2021,2022]),("F24",[2021,2022,2023])):
        file=ROOT/"outputs/legacy"/fold/"training_predictions.parquet"
        if not file.exists():
            records.append({"fold":fold,"status":"missing_frozen_compatibility_predictions"});continue
        source=snap/"data/folds"/anchor[fold+"_U"]["fold"]/"train.parquet"
        prediction=pd.read_parquet(file)
        meta=pd.read_parquet(source)
        if len(prediction)!=len(meta):raise ValueError("Legacy compatibility row count differs")
        for key in ("station_key","year","month"):
            if not np.array_equal(prediction[key].to_numpy(),meta[key].to_numpy()):
                raise ValueError("Legacy row identity mismatch: "+key)
        if not np.array_equal(prediction.observed_mg_l.to_numpy(),meta.tn_mg_l.to_numpy()):
            raise ValueError("Legacy truth mismatch")
        if not set(meta.year).issubset(years):raise ValueError("Non-training years in compatibility table")
        t=prediction.rename(columns={"observed_mg_l":"truth","legacy_prediction_mg_l":"baseline",
                                     "explicit_prediction_mg_l":"candidate"}).copy()
        t["day_index"]=meta.day_index.to_numpy()
        daily=t[t.day_index.ge(0)].copy()
        daily["date"]=pd.Timestamp("1961-01-01")+pd.to_timedelta(daily.day_index,unit="D")
        # Use actual counts from legal fold years, never infer counts from fit weights.
        count_paths=([snap/"data/cohorts/F23_G/hf_days.parquet"] if fold=="F23" else
                     [Path("E:/SPARROW/5_Test/20260917_5/data/cohorts/T24_G/hf_days.parquet")])
        counts=pd.concat([pd.read_parquet(p,columns=["station_key","date","n","y"]) for p in count_paths],ignore_index=True)
        counts["date"]=pd.to_datetime(counts.date)
        if counts.duplicated(["station_key","date"]).any():raise ValueError("Count registry duplicate")
        daily=daily.merge(counts,on=["station_key","date"],how="left",validate="one_to_one")
        if daily.n.isna().any():raise ValueError("Actual read counts missing for compatibility dates")
        if not np.array_equal(daily.truth.to_numpy(),daily.y.to_numpy()):
            raise ValueError("Read-count registry uses different selected readings than frozen training label")
        daily["read_count"]=daily.n
        month=t[t.day_index.lt(0)].copy()
        month["date"]=pd.to_datetime(dict(year=month.year,month=month.month,day=1))
        month["read_count"]=1
        role="frozen_training_compatibility_no_new_fit_not_forecast"
        results={"daily":save_pair(daily,f"compatibility_{fold}_daily","HF_daily",DAILY_POLICY,role)}
        for name,view in monthly_views(daily,month).items():
            results[name]=save_pair(view,f"compatibility_{fold}_{name}",name,MONTHLY_POLICY,role)
        records.append({"fold":fold,"status":"complete","train_years":years,
                        "prediction_sha256":digest(file),"train_metadata_sha256":digest(source),
                        "read_count_files":[{"path":str(p),"sha256":digest(p)} for p in count_paths],
                        "maximum_prediction_change":float(abs(t.candidate-t.baseline).max()),"statistics":results})
    return records


def main():
    implementation_paths=[ROOT/"d29_platform/metrics.py",ROOT/"d29_platform/objective.py",Path(__file__).resolve()]
    implementation_start={str(path.relative_to(ROOT)):digest(path) for path in implementation_paths}
    OUT.mkdir(parents=True,exist_ok=True)
    file=ROOT/"outputs/chain/synthetic_station_days.parquet"
    if not file.exists():
        raise FileNotFoundError("Full-chain synthetic_station_days.parquet must finish before evaluation")
    input_start_sha256=digest(file)
    source=pd.read_parquet(file)
    required={"station_key","date","read_count","pulse_mg_l","smooth_mg_l"}
    if not required.issubset(source):raise ValueError("Chain table missing registered fields")
    t=source[["station_key","date","read_count"]].copy()
    t["truth"]=source.pulse_mg_l
    t["baseline"]=source.pulse_mg_l
    t["candidate"]=source.smooth_mg_l
    role="synthetic_full_daily_field_software_control_not_real_TN"
    stats={"all_years":save_pair(t,"synthetic_2021_2024_daily","synthetic_daily",DAILY_POLICY,role)}
    for year in sorted(pd.to_datetime(t.date).dt.year.unique()):
        frame=t[pd.to_datetime(t.date).dt.year.eq(year)]
        stats[str(year)]=save_pair(frame,f"synthetic_{year}_daily","synthetic_daily",DAILY_POLICY,role)
    empty=pd.DataFrame(columns=["station_key","date","read_count","truth","baseline","candidate"])
    # Reuse the tested read-count month aggregator. Here all fields are synthetic,
    # so do not relabel a 116-station daily field as 116 observed HF records.
    monthly=monthly_views(t,empty,min_hf_days=2)["HF_monthly"]
    stats["synthetic_monthly"]=save_pair(monthly,"synthetic_2021_2024_monthly","synthetic_monthly",MONTHLY_POLICY,role)
    for length in (1,2):
        print("synchronized bootstrap",length,flush=True)
        result=paired_calendar_bootstrap(t,policy=DAILY_POLICY,n_bootstrap=1000,block_months=length,seed=1729)
        for name in ("replicates","intervals","ledger"):
            result[name].to_csv(OUT/f"synthetic_bootstrap_{length}month_{name}.csv",index=False)
    compat=compatibility()
    implementation_end={str(path.relative_to(ROOT)):digest(path) for path in implementation_paths}
    implementation_unchanged=implementation_start==implementation_end
    if not implementation_unchanged:
        raise RuntimeError("Implementation changed during statistics execution; receipt cannot be certified")
    input_unchanged=input_start_sha256==digest(file)
    if not input_unchanged:
        raise RuntimeError("Synthetic chain input changed during statistics execution")
    receipt={"passed":True,"input_sha256":input_start_sha256,"input_unchanged":input_unchanged,
             "implementation_sha256":implementation_start,"implementation_unchanged":implementation_unchanged,
             "implementation_identity_method":"SHA256 captured before computation, rechecked after all statistics",
             "synthetic_comparison":{
             "truth":"pulse_mg_l","baseline":"pulse_mg_l (identity software control)",
             "candidate":"smooth_mg_l","TN_observations_used":False,
             "support":"full 2021-2024 synthetic daily field at frozen station boundaries; not observed 116-station daily TN",
             "daily_weight":"read_count","centered_weight":"read-count demeaning within month, equal months",
             "statistics":stats},"compatibility":compat,
             "bootstrap":{"seed":1729,"replicates":1000,"block_months":[1,2],
                          "synchronized_stations":True,"independent_repeat_month_identity":True,
                          "interpretation":"conditional on fixed synthetic runs; no real predictive or parameter uncertainty claim"}}
    (OUT/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (OUT/"README.md").write_text("# 全链条标准配对统计\n\n"
        "合成脉冲输出为真值和恒等基线，同质量平滑输入输出为对照。116站2021—2024完整日场均为虚构软件结果，"
        "不是116站的实测日水质，也不是改善真实TN预测的证据。\n\n"
        "所有表使用同一 evaluate_pair。日项按 read_count 加权；月内NSE各月按读数权重去均值，月份等权汇总误差和观测方差。"
        "合成月值为同一权重的日场聚合，不冒称原月报。日表采用至少120天、6个月；月表至少6个有效月。"
        "本目录分母只来自合成真值或明确的旧训练标签，零方差不加 epsilon。\n\n"
        "同步月块、连续两月块各1000次、seed1729；重复月独立身份。区间仅描述冻结软件输出的抽样稳定性。\n\n"
        "compatibility 文件为F23/F24原训练支持上的 LAND0 接口回归：旧预测与显式日接口预测相同，NSE差应为零。"
        "它们不属于新拟合、跨年留出或新结构成绩；HF日/HF月/原月报分别制表。读数次数仅取该折合法训练年份。\n",encoding="utf-8")
    print(json.dumps({"passed":True,"synthetic_rows":len(t),"synthetic_stations":t.station_key.nunique(),
                      "compatibility_folds":len(compat)},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
