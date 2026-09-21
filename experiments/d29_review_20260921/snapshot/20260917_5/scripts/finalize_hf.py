"""Freeze and select only on training, then reveal evaluation and report."""
import os,time,gc,subprocess,sys
import native_runtime as rt
from campaign_model import *
from hf_metrics import metric,summaries,monthly_components,clean
from predict_extra import export,rank
R=RUN
def save(df,name):df.to_parquet(R/'reports'/name,index=False)
def mdtable(df):
 if df.empty:return '无合格结果。'
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
 jobs=rt.read(R/'configs/jobs.json');selected={};valid={};states=[]
 for job in jobs:
  root=R/'outputs'/job['tag'];p=root/'audit.json';state=rt.read(R/'work/jobs'/job['tag']/'status.json')
  states.append(dict(tag=job['tag'],status=state['status'],calls=state.get('calls'),active_seconds=state.get('active_seconds')))
  if not p.exists():continue
  a=rt.read(p)
  if a.get('status')!='AUDITED_FIT' or not a.get('physical_reasonable'):continue
  for name,h in a['files'].items():assert rt.sha(root/name)==h,(job['tag'],name)
  valid[job['tag']]=a
  if job['fold'] not in selected or a['objective']<valid[selected[job['fold']]]['objective']:selected[job['fold']]=job['tag']
 rt.write(R/'reports/prediction_freeze_manifest.json',dict(time=time.time(),selected=selected,artifacts={k:v['files'] for k,v in valid.items()},selection='minimum legal training objective only'))
 pd.DataFrame(states).to_csv(R/'reports/path_status.csv',index=False)
 # All parameter and prediction identities are now frozen; scoring starts here.
 pub=pd.read_parquet(R/'data/heldout_labels/monthly_original.parquet');cal=pd.read_parquet(R/'data/prediction_calendar.parquet');coh=cal.drop_duplicates('station_key').set_index('station_key').cohort.to_dict()
 sites=[];components=[];enriched=[];summaryrows=[];fitrows=[];pair_frames={};sensitivityrows=[]
 def evaluate_daily(frame,fold,tag,view='BJT',role='evaluation'):
  labelpath=R/'data/heldout_labels'/f'{fold}_{role}_{view}_days.parquet';labels=pd.read_parquet(labelpath)
  z=labels.merge(frame[['station_key','date','concentration_mg_l','water_m3_day']].rename(columns={'concentration_mg_l':'p'}),on=['station_key','date'],validate='one_to_one');assert len(z)==len(labels)
  z['cohort']=z.station_key.map(coh);z['residual']=z.p-z.y
  cs=z.groupby(['station_key','cohort','year','month']).apply(monthly_components,include_groups=False).reset_index();cs['tag']=tag;cs['fold']=fold;cs['view']=view;cs['role']=role
  local=[]
  for (s,c,y),g in z.groupby(['station_key','cohort','year']):local.append(dict(tag=tag,fold=fold,view=view,role=role,scale='daily',station_key=s,cohort=c,year=int(y),**metric(g,'daily')))
  for (s,c,y),g in cs.groupby(['station_key','cohort','year']):local.append(dict(tag=tag,fold=fold,view=view,role=role,scale='monthly_HF',station_key=s,cohort=c,year=int(y),**metric(g,'monthly_HF')))
  z['week']=z.date-pd.to_timedelta(z.date.dt.dayofweek,unit='D');evyear=int(fold[-2:])+2000
  zz=z[(z.week>=pd.Timestamp(evyear,1,1))&(z.week+pd.Timedelta(days=7)<=pd.Timestamp(evyear+1,1,1))] if role=='evaluation' else z
  for (s,c,y),g in zz.groupby(['station_key','cohort','year']):
   w=g.groupby('week').apply(lambda a:pd.Series(dict(y=np.average(a.y,weights=a.n),p=np.average(a.p,weights=a.n),days=len(a),month=int(a.month.iloc[0]))),include_groups=False);w=w[w.days>=4]
   if len(w):local.append(dict(tag=tag,fold=fold,view=view,role=role,scale='weekly',station_key=s,cohort=c,year=int(y),**metric(w,'weekly')))
  return z,cs,pd.DataFrame(local)
 for job in jobs:
  tag=job['tag']
  if tag not in valid:continue
  root=R/'outputs'/tag;fold=job['fold'][:3];chosen=selected[job['fold']]==tag;rec=rt.read(root/'model.json');fitrows.append(dict(tag=tag,selected=chosen,objective=rec['objective'],pg=rec['pg'],data=rec['terms']['data'],prior=rec['terms']['prior'],numerical_sufficient=valid[tag]['numerical_sufficient'],physical_reasonable=True))
  frame=pd.read_parquet(root/'daily_station_mass_water.parquet');z,cs,ms=evaluate_daily(frame,fold,tag);ms['selected']=chosen;sites.append(ms);components.append(cs);save(z,f'{tag}_HF_evaluated.parquet')
  stat=pd.read_parquet(root/'statistical_products.parquet');yr=2023 if fold=='F23' else 2024
  l=pub[pub.year.eq(yr)].copy();p=stat[stat.year.eq(yr)]
  zz=l[['station_key','year','month','tn_mg_l']].merge(p[['station_key','cohort','year','month','MATCH_mg_l']],on=['station_key','year','month'],validate='one_to_one').rename(columns={'tn_mg_l':'y','MATCH_mg_l':'p'});assert len(zz)==len(l)
  for (s,c,y),g in zz.groupby(['station_key','cohort','year']):sites.append(pd.DataFrame([dict(tag=tag,fold=fold,view='BJT',role='evaluation',scale='monthly_PUB',station_key=s,cohort=c,year=int(y),selected=chosen,**metric(g,'monthly_PUB'))]))
  save(zz,f'{tag}_PUB_evaluated.parquet')
  if chosen:
   pair_frames[job['fold']]=z
   for role,view in [('evaluation','CHM'),('evaluation','CMFD'),('training_raw','BJT'),('evaluation_all_status','BJT')]:
    zs,cc,mm=evaluate_daily(frame,fold,tag,view,role);mm['selected']=True;sensitivityrows.append(mm);save(zs,f'{tag}_{role}_{view}.parquet')
    # Separate common-date comparison so coverage cannot masquerade as a window effect.
    if role=='evaluation':
     shared=z[['station_key','date']].merge(zs[['station_key','date']],on=['station_key','date'],validate='one_to_one')
     for label,base in [('BJT_common_'+view,z),(view+'_common',zs)]:
      q=base.merge(shared,on=['station_key','date'],validate='one_to_one')
      for (s,c,y),g in q.groupby(['station_key','cohort','year']):sensitivityrows.append(pd.DataFrame([dict(tag=tag,fold=fold,view=label,role='evaluation',scale='daily',station_key=s,cohort=c,year=int(y),selected=True,**metric(g,'daily'))]))
   export(np.array(rec['parameters']),rec['design'],R/'sensitivities'/tag/'OS_MIX','OS_MIX')
   replay=rt.read(R/'sensitivities'/tag/'OS_MIX/replay.json')
   if replay.get('physical_reasonable'):
    sf=pd.read_parquet(R/'sensitivities'/tag/'OS_MIX/daily_station_mass_water.parquet');zs,cc,mm=evaluate_daily(sf,fold,tag,'BJT');mm['view']='OS_MIX';mm['selected']=True;sensitivityrows.append(mm);save(zs,f'{tag}_OS_MIX_evaluated.parquet')
   else:
    rt.write(R/'sensitivities'/tag/'OS_MIX/scoring_exclusion.json',dict(reason='Physical tolerance failure; retained for diagnosis, excluded from valid sensitivity scores',replay_hash=rt.sha(R/'sensitivities'/tag/'OS_MIX/replay.json')))
   # Local frozen driving variables for residual diagnostics, explicitly local support.
   des=dict(rec['design'],observation_registry_file='data/prediction_registry.json',observation_registry_hash=sha(R/'data/prediction_registry.json'));pm=make_model(load_data('ALL'),None,'D29_BE',des)
   st=cal.drop_duplicates('station_key').set_index('station_key');di=(z.date-pd.Timestamp('1961-01-01')).dt.days.to_numpy();ri=z.station_key.map(st.reach_id).to_numpy(int)-1
   for k in ['soil_wetness','temperature','upper_water','percolation']:z[k]=np.asarray(getattr(pm.data,k))[di,ri]
   for s,g in frame[frame.date.dt.year.isin([2021,2022] if fold=='F23' else [2021,2022,2023])].groupby('station_key'):
    sel=z.station_key.eq(s);q10=g.water_m3_day.quantile(.1);q90=g.water_m3_day.quantile(.9);z.loc[sel,'flow_class']=np.where(z.loc[sel,'water_m3_day']>=q90,'high',np.where(z.loc[sel,'water_m3_day']<=q10,'low','middle'))
   z['tag']=tag;enriched.append(z);del pm;gc.collect()
 sm=pd.concat(sites,ignore_index=True) if sites else pd.DataFrame();save(sm,'station_metrics.parquet');sm.to_csv(R/'reports/station_metrics.csv',index=False)
 cm=pd.concat(components,ignore_index=True) if components else pd.DataFrame();save(cm,'monthly_error_components.parquet')
 ss=pd.concat(sensitivityrows,ignore_index=True) if sensitivityrows else pd.DataFrame();save(ss,'sensitivity_station_metrics.parquet')
 for keys,g in sm.groupby(['tag','selected','fold','scale','cohort','year']):summaryrows.append(dict(zip(['tag','selected','fold','scale','cohort','year'],keys),**summaries(g)))
 summary=pd.DataFrame(summaryrows);summary.to_csv(R/'reports/region_metrics.csv',index=False);pd.DataFrame(fitrows).to_csv(R/'reports/fit_summary.csv',index=False)
 comparisons=[];boots=[];paired=[];resampling_plans=[];rng=np.random.default_rng(1729);regions=rt.read(R/'data/region_reaches.json')
 def uptake(tag,c,y):
  l=pd.read_parquet(R/'outputs'/tag/'monthly_physical_ledger.parquet');g=l[l.global_reach_id.isin(regions[c])&l.year.eq(y)];den=g.demand_kg.sum();return float(g.uptake_kg.sum()/den) if den>0 else None
 for fold in ['F23','F24']:
  for ma,mb in [('D_HF','M_HF'),('M_HF','M_PUB')]:
   if fold+'_'+ma not in selected or fold+'_'+mb not in selected:continue
   ta=selected[fold+'_'+ma];tb=selected[fold+'_'+mb]
   a=sm[sm.tag.eq(ta)];b=sm[sm.tag.eq(tb)];pair=a.merge(b,on=['station_key','cohort','year','scale'],suffixes=('_a','_b'),validate='one_to_one');pair['contrast']=ma+'-'+mb;pair['fold']=fold;paired.append(pair)
   for (scale,c,y),g in pair.groupby(['scale','cohort','year']):
    sa=summaries(a[a.scale.eq(scale)&a.cohort.eq(c)]);sb=summaries(b[b.scale.eq(scale)&b.cohort.eq(c)])
    row=dict(fold=fold,contrast=ma+'-'+mb,scale=scale,cohort=c,year=int(y),delta_NSE=sa['NSE_median']-sb['NSE_median'],delta_r=sa['r_median']-sb['r_median'],RMSE_ratio=sa['RMSE']/sb['RMSE'] if sb['RMSE'] else np.nan)
    if scale.startswith('monthly'):
     ua=uptake(ta,c,y);ub=uptake(tb,c,y)
     gates=dict(nse=row['delta_NSE']>=.1,correlation=row['delta_r']>=.05 and sa['r_median']>0,lower_quartile=sa['NSE_q25']>=sb['NSE_q25']-.05,negative=sa['negative_r_fraction']<=sb['negative_r_fraction'],undefined=sa['undefined_r_fraction']<=sb['undefined_r_fraction'],uptake=ua is not None and ub is not None and ua>=ub-.1,**{k:sa[k]<=1.05*sb[k] for k in ['RMSE','logRMSE','absolute_bias']});row.update(monthly_gate=all(gates.values()),gate_tests=gates)
    comparisons.append(row)
   # Synchronized month blocks across all eight stations, using per-station month means.
   aa=cm[cm.tag.eq(ta)];bb=cm[cm.tag.eq(tb)];p=aa.merge(bb,on=['station_key','cohort','year','month'],suffixes=('_a','_b'),validate='one_to_one');months=sorted(p.month.unique())
   draws=rng.choice(months,size=(1000,len(months)),replace=True)
   resampling_plans.append(dict(fold=fold,contrast=ma+'-'+mb,months=months,draws=draws.tolist(),synchronized_cohorts=['N','H','X']))
   for c in ['N','H','X']:
    g=p[p.cohort.eq(c)];arr=[]
    for draw in draws:
     sample=pd.concat([g[g.month.eq(m)] for m in draw],ignore_index=True)
     if sample.empty:continue
     station=sample.groupby('station_key')[['daily_SSE_a','daily_SSE_b','within_SSE_a','within_SSE_b']].mean();arr.append([float((station.daily_SSE_a-station.daily_SSE_b).mean()),float((station.within_SSE_a-station.within_SSE_b).mean())])
    vals=np.array(arr);boots.append(dict(fold=fold,contrast=ma+'-'+mb,cohort=c,replicates=len(vals),seed=1729,delta_daily_SSE_quantiles=np.quantile(vals[:,0],[.025,.5,.975]).tolist() if len(vals) else None,delta_within_SSE_quantiles=np.quantile(vals[:,1],[.025,.5,.975]).tolist() if len(vals) else None,interpretation='descriptive synchronized whole-month resampling; not future or mechanism CI'))
 if paired:save(pd.concat(paired,ignore_index=True),'paired_station_changes.parquet')
 rt.write(R/'reports/comparisons.json',clean(comparisons));rt.write(R/'reports/month_block_bootstrap.json',clean(boots))
 rt.write(R/'reports/month_block_resampling_plan.json',clean(resampling_plans))
 if enriched:
  zz=pd.concat(enriched,ignore_index=True);save(zz,'residuals_with_local_frozen_drivers.parquet');corr=[]
  for tag,g in zz.groupby('tag'):
   for variable in ['water_m3_day','soil_wetness','temperature','upper_water','percolation']:
    for kind,keys in [('raw',[]),('within_station',['station_key']),('station_month_anomaly',['station_key','month'])]:
     xy=g[[variable,'residual']].copy()
     if keys:xy-=g.groupby(keys)[[variable,'residual']].transform('mean')
     corr.append(dict(tag=tag,variable=variable,kind=kind,n=len(xy),correlation=float(xy.corr().iloc[0,1])))
  pd.DataFrame(corr).to_csv(R/'reports/residual_driver_correlations.csv',index=False)
  zz.groupby(['tag','cohort','flow_class']).residual.agg(['size','mean','std']).to_csv(R/'reports/flow_event_groups.csv')
 rank(selected)
 # Old parameter forecasts were produced before fitting, scored only now.
 oldmetrics=[]
 for op in ['OU','OS_MIX']:
  path=R/'diagnostics'/f'old_{op}/daily_station_mass_water.parquet'
  if not path.exists():continue
  for fold in ['F23','F24']:
   _,_,mm=evaluate_daily(pd.read_parquet(path),fold,'old_'+op);oldmetrics.append(mm)
 if oldmetrics:save(pd.concat(oldmetrics,ignore_index=True),'old_parameter_diagnostics.parquet')
 chosen=summary[summary.selected&summary.scale.eq('daily')][['fold','cohort','NSE_median','r_median','RMSE','tag']].copy();chosen=chosen.round(5)
 results=mdtable(chosen);details=mdtable(pd.DataFrame(fitrows).round(7))
 caveat='本轮是算法隔离的回顾性检验。F23只有41个共同站点月，F24为122个；七个训练站覆盖五个河段，棉江只有一个高频X站。水文日界混用未被修复；日质量/日水量与四小时算术均值也只是日尺度近似。统计异常不等于测量错误。高频与月报同坐标仍不能认证同一审核血缘。'
 (R/'reports/expert_diagnosis.md').write_text('# 高频TN日约束专家诊断\n\n'+caveat+'\n\n## 留出日尺度结果\n\n'+results+'\n\n## 训练目标与数值状态\n\n'+details+'\n\nM-HF与D-HF使用相同观测和权重，差异仅为月内信息；M-PUB与M-HF同时改变来源及采样支持。训练目标含先验，不能把较小MAP直接当作较好预测。\n\n逐站日周月结果见station_metrics.csv，连续配对差异及原月门槛见comparisons.json；月水平/动态分解见monthly_error_components.parquet。日界、维护值和OS-MIX结果见sensitivity_station_metrics.parquet，必须先核对公共日期后解释日界差异。\n\n灵敏度矩阵见sensitivity_rank.json及两折矩阵文件；小奇异值仅为局部诊断。残差驱动关联有共线性、时间相关和事后解释限制，驱动字段代表对应河段局地支持；C观测乘模拟Q不称实测负荷。同日上下游读数不保证同一水团。\n\n各路径物理、来源和河网账本保存在outputs；OS-MIX完整回放保存在sensitivities，不重新拟合，不选择最好空间方案。月初、高浓度缺口、幅度误差见逐站表，高低流量分组阈值由训练水量定义。\n\n月均与日动态均无收益时不能据此证明高频数据无效，应结合灵敏度、日界与现有月初输入/月内暴露量结构诊断。预测改善不证明legacy来源已识别，不自动扩域或替换主线。\n',encoding='utf8')
 deviations='数据准备首次已成功，但conda捕获输出发生GBK编码异常；重入被已封存保护拒绝，没有重复清洗或改动数据。随后使用完整conda环境及UTF-8、不捕获输出的启动方式。为确保最低合法点，拟合器对每个新的最低目标候选进行完整物理账本检查，这些额外调用也计入同一路径预算。旧参数预测在拟合前冻结，其留出评分延迟到所有新参数冻结后，防止提前揭示评价结果。另发现启动统一哈希检查会读取其他折训练文件的字节，虽仅用于SHA256且未将其解析进数值目标，仍不符合严格文件隔离。控制器已暂停、全部运行路径安全保存，改用每路径独立清单并在校验前安装读取屏障；从原序列化优化状态续算，预算未归零。每条恢复路径另计一次完整目标核验，参数和目标值不变，详见isolation_repair_audit.json。修复前的文件隔离缺陷明确保留，不称从未发生。'
 (R/'reports/actual_methods_and_deviations.md').write_text('# 实际方法与偏离\n\n'+deviations+'\n\n## 固定设计\n\n原30参数、原先验、原26河段无TN标准化未变。仅目标接口接受本折外部权重，并增加日索引读出。每站HF共同月值总体方差及区域正方差10%分位构成共同尺度；F23不能使用旧至少12个月的初始化要求。\n\n'+details+'\n\n启动验收、折内清洗账本、路径完整JSONL、检查点、资源记录和独立审计均保留。数值充分按原坐标梯度与最低合法目标判断，两起点差异不被写成同一全局最优。未完成路径按path_status.csv明确列示，不用已有检查点冒充执行完成。\n',encoding='utf8')
 (R/'reports/short_conclusion.md').write_text('# 简短结论\n\n已完成的配对结果如下；阴性结果、数值不足与日界敏感性需分别判断。\n\n'+results+'\n\n'+caveat+'\n',encoding='utf8')
 (R/'README.md').write_text('# 高频TN日信息约束实验\n\n主报告：[专家诊断](reports/expert_diagnosis.md)、[实际方法与偏离](reports/actual_methods_and_deviations.md)、[简短结论](reports/short_conclusion.md)、[完成审计](reports/completion_audit.json)。\n\n输入只读，所有参数从1961连续演化；没有四小时物理模型、TN状态同化或2025训练。全程sparrow、CPU float64，每worker一线程，无定时任务。12条登记路径见configs/jobs.json。\n\n复现：在完整conda sparrow环境运行scripts/campaign_controller.py；重复启动保护及累计预算生效，不会创建新科学起点。报告脚本为scripts/finalize_hf.py。主数据不得用全历史统计删除标记筛选留出。\n',encoding='utf8')
 supplement=subprocess.run([sys.executable,'-B',str(R/'scripts/supplement_hf.py')],cwd=R,creationflags=subprocess.CREATE_NO_WINDOW)
 if supplement.returncode:raise RuntimeError('EXPERT_SUPPLEMENT_FAILED')
 for script in ['finish_evidence.py','write_final_interpretation.py']:
  completed=subprocess.run([sys.executable,'-B',str(R/'scripts'/script)],cwd=R,creationflags=subprocess.CREATE_NO_WINDOW)
  if completed.returncode:raise RuntimeError('FINAL_EVIDENCE_FAILED: '+script)
 result=subprocess.run([sys.executable,'-B',str(R/'scripts/independent_completion.py')],cwd=R,creationflags=subprocess.CREATE_NO_WINDOW)
 return result.returncode
if __name__=='__main__':sys.exit(main())
