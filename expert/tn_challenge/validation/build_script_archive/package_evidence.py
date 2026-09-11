import os,sys,json,hashlib,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('E:/SPARROW');REPO=BASE/'repository';PKG=REPO/'expert/tn_challenge';OUT=REPO/'expert/evidence'
os.environ['PYTHONIOENCODING']='utf-8'
import pandas as pd,numpy as np
OUT.mkdir(parents=True,exist_ok=True)
sources=[]
for date in ['20260907_3','20260907_4','20260908_1','20260909_1','20260909_2','20260910_1','20260910_2','20260910_3']:
    folder=ROOT/'5_Test'/date;p=folder/'final_report.md'
    if not p.exists():p=folder/'reports/final_report.md'
    sources.append((p,OUT/date/'final_report.md'))
for date,sub,names in [
 ('20260910_1','reports',['expert_diagnostic_report.md']),
 ('20260910_2','reports',['expert_diagnostic_report.md']),
 ('20260910_3','reports',['expert_diagnostic_report.md']),
 ('20260910_4','reports/exploration_delivery',['final_report.md','expert_diagnostic_report.md','actual_methods_and_deviations.md','scope_decision.json','completion_audit.json']),
 ('20260911_1','reports',['final_report.md','expert_diagnostic_report.md','actual_methods_and_deviations.md','results.json','completion_audit.json','model_validation.json','function_equivalence.json']),
 ('20260911_1','reports/evidence',['evidence_report.md'])]:
    for name in names:sources.append((ROOT/'5_Test'/date/sub/name,OUT/date/name))
manifest=[]
for src,dst in sources:
    assert src.is_file(),src;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    manifest.append({'source':str(src),'published':dst.relative_to(REPO).as_posix(),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'bytes':src.stat().st_size})
(OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
# Full-domain per-station summary tables are explicitly authorized quantitative reports.
for p in (ROOT/'5_Test/20260911_1/outputs/metrics').glob('*stations.parquet'):
    dst=OUT/'20260911_1/station_metrics'/p.with_suffix('.csv').name;dst.parent.mkdir(parents=True,exist_ok=True)
    pd.read_parquet(p).to_csv(dst,index=False,encoding='utf-8')
train=pd.read_csv(PKG/'data/train.csv');full=pd.read_parquet(ROOT/'5_Test/20260911_1/outputs/data/F23_train.parquet')
rows=[]
for name,o in [('sample',train),('full_F23_train',full)]:
    for k,v in o.tn_mg_l.quantile([.01,.1,.5,.9,.99]).items():rows.append({'cohort':name,'metric':'TN_quantile','quantile':k,'value':v})
    rows.extend([{'cohort':name,'metric':'rows','value':len(o)},{'cohort':name,'metric':'stations','value':o.station_key.nunique()}])
pd.DataFrame(rows).to_csv(PKG/'validation/training_distribution_comparison.csv',index=False)
z=np.log1p(train.tn_mg_l.to_numpy());med=np.median(z);mad=np.median(np.abs(z-med));ledger=train[['observation_id','station_key','year','month','tn_mg_l']].copy();flag=np.zeros(len(train),bool)
for key,ii in train.groupby('station_key').indices.items():
    center=np.median(z[ii]) if len(ii)>=12 else med;deviation=np.median(np.abs(z[ii]-center)) if len(ii)>=12 else mad
    flag[ii]=np.abs(z[ii]-center)>4.5*1.4826*max(deviation,.05)
ledger['training_only_suspected']=flag;ledger['RAW_relative_weight']=1.;ledger.to_csv(PKG/'validation/train_anomaly_ledger.csv',index=False)
print('training suspect rows',flag.sum(),'of',len(train))
for name in ['train','development','evaluation','hindcast']:
    frame=pd.read_csv(PKG/f'data/{name}.csv');print(name,len(frame),frame.station_key.nunique())
sys.path.insert(0,str(PKG));from run import metrics
for variant in ['M0','SC']:
    frame=pd.read_csv(PKG/f'reference_only/F23_{variant}_0_predictions.csv')
    summary=[]
    for name,years in [('train',[2021,2022]),('development',[2023]),('evaluation',[2024]),('hindcast',list(range(2016,2021)))]:
        for basin in ['ALL','Beijiang','Dongjiang']:
            f=frame.loc[frame.year.isin(years)&(frame.basin.eq(basin) if basin!='ALL' else True)];m=metrics(f)
            m.to_csv(PKG/f'reference_only/{variant}_{name}_{basin}_stations.csv',index=False)
            summary.append({'variant':variant,'split':name,'basin':basin,'rows':len(f),'stations':len(m),'eligible':int(m.nse.notna().sum()),
                'NSE_median':m.nse.median(),'r_median':m.r.median(),'RMSE_station_mean':m.rmse.mean()})
    pd.DataFrame(summary).to_csv(PKG/f'reference_only/{variant}_summary.csv',index=False)
smoke=json.loads((BASE/'work/smoke_SC/model.json').read_text(encoding='utf-8'))
(PKG/'validation/smoke_fit.json').write_text(json.dumps({k:smoke[k] for k in ['total_calls','total_seconds','MAP','data','prior','projected_gradient','numerical_sufficient','stop_reason','optimizer','initial_preset']},indent=2),encoding='utf-8')
shutil.copyfile(BASE/'work/smoke_SC/trajectory.csv',PKG/'validation/smoke_trajectory.csv')
print('Evidence files',len(sources))
