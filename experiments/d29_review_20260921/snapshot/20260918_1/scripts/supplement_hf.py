"""Post-freeze expert evidence: exact month accounting, station pairs and figures."""
import os,json
import numpy as np,pandas as pd
import native_runtime as rt
from hf_metrics import monthly_components,clean
R=rt.RUN
def main():
 freeze=rt.read(R/'reports/prediction_freeze_manifest.json');selected=freeze['selected'];cal=pd.read_parquet(R/'data/prediction_calendar.parquet');st=cal.drop_duplicates('station_key').set_index('station_key');names=st.station_name.to_dict() if 'station_name' in st else st.station.to_dict()
 fits=[];trainparts=[];annual=[];pairs=[];sensitivity=[];events=[];limbs=[];lineage=[]
 for config in rt.read(R/'configs/folds.json'):
  fold=config[:3];mode=config[4:];train=pd.read_parquet(R/'data/folds'/config/'train.parquet');days=pd.read_parquet(R/'data/folds'/fold/'common_days.parquet');months=pd.read_parquet(R/'data/folds'/fold/'common_months.parquet')
  for _,row in train.iterrows():
   if mode=='M_PUB':
    q=months[months.station_key.eq(row.station_key)&months.year.eq(row.year)&months.month.eq(row.month)];assert len(q)==1;ids=[str(q.pub_observation_id.iloc[0])];product='v3 canonical monthly observation_id'
   else:
    q=days[days.station_key.eq(row.station_key)&days.year.eq(row.year)&days.month.eq(row.month)]
    if mode=='D_HF':q=q[q.date.eq(pd.Timestamp('1961-01-01')+pd.Timedelta(days=int(row.day_index)))];assert len(q)==1
    ids=[str(i) for arr in q.selected_record_ids for i in arr];product='auto_4h canonical selected_record_id'
   lineage.append(dict(configuration=config,observation_id=row.observation_id,station_key=row.station_key,year=int(row.year),month=int(row.month),source_product=product,source_record_ids=ids,source_count=len(ids),legacy_template_row_is_not_observation_lineage=True))
 pd.DataFrame(lineage).to_parquet(R/'reports/training_observation_lineage.parquet',index=False)
 for config,tag in selected.items():
  rec=rt.read(R/'outputs'/tag/'model.json');fold=config[:3];frame=pd.read_parquet(R/'outputs'/tag/'daily_station_mass_water.parquet')
  for name,value in zip(rec['names'],rec['parameters']):fits.append(dict(fold=fold,configuration=config,tag=tag,parameter=name,value=value))
  tr=pd.read_parquet(R/'data/folds'/fold/'common_days.parquet');z=tr.merge(frame[['station_key','date','concentration_mg_l']].rename(columns={'concentration_mg_l':'p'}),on=['station_key','date'],validate='one_to_one');assert len(z)==len(tr)
  q=z.groupby(['station_key','year','month']).apply(monthly_components,include_groups=False).reset_index();q['tag']=tag;q['comparison_role']='common HF training diagnostic (M-PUB objective differs)';trainparts.append(q)
  ledger=pd.read_parquet(R/'outputs'/tag/'monthly_physical_ledger.parquet');regions=rt.read(R/'data/region_reaches.json')
  for cohort,reaches in regions.items():
   for y,g in ledger[ledger.global_reach_id.isin(reaches)].groupby('year'):
    row=dict(tag=tag,cohort=cohort,year=int(y))
    for k in ['fast_kg','slow_kg','uptake_kg','demand_kg','mineral_loss_kg','channel_loss_kg','source_kg']:row[k]=float(g[k].sum())
    for k in ['M_end_kg','L_end_kg']:row[k]=float(g[g.month.eq(12)][k].sum())
    row['uptake_demand']=row['uptake_kg']/row['demand_kg'] if row['demand_kg'] else None;annual.append(row)
  ev=pd.read_parquet(R/'reports'/f'{tag}_HF_evaluated.parquet')
  years=[2021,2022] if fold=='F23' else [2021,2022,2023]
  for s,water in frame.groupby('station_key'):
   water=water.sort_values('date').copy();q90=water.loc[water.date.dt.year.isin(years),'water_m3_day'].quantile(.9)
   water['high']=water.water_m3_day>=q90;water['event_id']=(water.high!=water.high.shift(fill_value=False)).cumsum();delta=water.water_m3_day.diff()
   water['limb']=np.where(delta>0,'rising',np.where(delta<0,'falling','steady'))
   g=ev[ev.station_key.eq(s)].merge(water[['date','limb','high','event_id']],on='date',validate='one_to_one')
   for limb,a in g.groupby('limb'):limbs.append(dict(tag=tag,station_key=s,limb=limb,n=len(a),bias=float((a.p-a.y).mean()),RMSE=float(np.sqrt(np.mean((a.p-a.y)**2)))))
   for event,a in g[g.high].groupby('event_id'):
    if len(a)<2:continue
    observed=a.loc[a.y.idxmax(),'date'];predicted=a.loc[a.p.idxmax(),'date']
    events.append(dict(tag=tag,station_key=s,event_id=int(event),observed_days=len(a),first=a.date.min(),last=a.date.max(),q90_m3_day=float(q90),peak_phase_days=int((predicted-observed).days),peak_gap=float(a.p.max()-a.y.max()),support='TN peaks compared only on common observed eligible days within frozen-Q high-flow runs; no lag fitted'))
  for n1,n2 in [('九甸大桥','永昌桥'),('盘溪大桥','禄丰村')]:
   s1=next(s for s,n in names.items() if n==n1);s2=next(s for s,n in names.items() if n==n2)
   a=ev[ev.station_key.eq(s1)];b=ev[ev.station_key.eq(s2)];p=a.merge(b,on='date',suffixes=('_a','_b'),validate='one_to_one')
   for _,r in p.iterrows():pairs.append(dict(tag=tag,fold=fold,station_a=n1,station_b=n2,date=r.date,observed_difference=r.y_b-r.y_a,predicted_difference=r.p_b-r.p_a,difference_residual=(r.p_b-r.p_a)-(r.y_b-r.y_a),observed_ratio=r.y_b/r.y_a if r.y_a>0 else np.nan,predicted_ratio=r.p_b/r.p_a if r.p_a>0 else np.nan,support='common eligible calendar day; not confirmed same water parcel or identical sampling times'))
 if fits:pd.DataFrame(fits).to_csv(R/'reports/selected_parameters.csv',index=False)
 if trainparts:pd.concat(trainparts,ignore_index=True).to_parquet(R/'reports/training_HF_error_components.parquet',index=False)
 pd.DataFrame(annual).to_parquet(R/'reports/annual_physical_summary.parquet',index=False);pd.DataFrame(pairs).to_parquet(R/'reports/same_reach_daily_pairs.parquet',index=False)
 pd.DataFrame(events).to_parquet(R/'reports/event_peak_phase.parquet',index=False);pd.DataFrame(limbs).to_csv(R/'reports/rising_falling_response.csv',index=False)
 if pairs:
  pairframe=pd.DataFrame(pairs);pairframe['month']=pairframe.date.dt.to_period('M').astype(str)
  pairframe.groupby(['tag','station_a','station_b','month'])[['observed_ratio','predicted_ratio','difference_residual']].agg(['count','mean','std','min','max']).to_csv(R/'reports/same_reach_monthly_ratio_variation.csv')
 raw=pd.read_parquet(R/'data/heldout_labels/hf_canonical_selected.parquet');raw.groupby(['station_key','indicator','station_status'],dropna=False).size().reset_index(name='unique_records').to_csv(R/'reports/status_and_indicator_coverage.csv',index=False)
 month=pd.read_parquet(R/'evidence/all_normal_monthly_reconstruction.parquet');month['year']=month.ym.str[:4].astype(int);month['month']=month.ym.str[5:7].astype(int)
 pub=pd.read_parquet(R/'data/heldout_labels/monthly_original.parquet');comparison=month.merge(pub[['station_key','year','month','tn_mg_l']],on=['station_key','year','month'],validate='one_to_one');comparison['HF_minus_PUB']=comparison['mean']-comparison.tn_mg_l
 comparison.to_csv(R/'reports/monthly_source_agreement.csv',index=False)
 # Parameter compensation between starts is reported regardless of the selected start.
 starts=[]
 for config in selected:
  paths=[R/'outputs'/f'{config}_s{i}/model.json' for i in [0,1]]
  if not all(p.exists() for p in paths):continue
  a,b=[rt.read(p) for p in paths]
  for n,u,v in zip(a['names'],a['parameters'],b['parameters']):starts.append(dict(configuration=config,parameter=n,start0=u,start1=v,difference=v-u,MAP0=a['objective'],MAP1=b['objective'],PG0=a['pg'],PG1=b['pg']))
 pd.DataFrame(starts).to_csv(R/'reports/two_start_parameters.csv',index=False)
 # All complete physical readouts carry an explicit sidecar identity.
 identities={}
 for tag in freeze['artifacts']:
  rec=rt.read(R/'outputs'/tag/'model.json');d=rec['design'];identities[tag]=dict(fold_id=d['fold_id'],objective_id=d['objective_id'],operator_id=d['operator_id'],mapping_id='G1',day_window_id='BJT',support_hash=d['support_hash'],observation_registry_hash=d['observation_registry_hash'],model_sha256=rt.sha(R/'outputs'/tag/'model.json'),physical_units={'mass':'kg/day','water':'m3/day','concentration':'mg/L'},scored_window_ids=['BJT','CHM','CMFD'])
 rt.write(R/'reports/prediction_identity_manifest.json',identities)
 # Assess direction changes on common dates, without selecting a best window.
 ss=pd.read_parquet(R/'reports/sensitivity_station_metrics.parquet');main=pd.read_parquet(R/'reports/station_metrics.parquet');rows=[]
 for fold in ['F23','F24']:
  a=selected.get(fold+'_D_HF');b=selected.get(fold+'_M_HF')
  if not a or not b:continue
  for view in ['BJT','CHM','CMFD','BJT_common_CHM','CHM_common','BJT_common_CMFD','CMFD_common','OS_MIX']:
   df=main if view=='BJT' else ss
   aa=df[df.tag.eq(a)&df.scale.eq('daily')&df.view.eq(view)&df.role.eq('evaluation')];bb=df[df.tag.eq(b)&df.scale.eq('daily')&df.view.eq(view)&df.role.eq('evaluation')]
   p=aa.merge(bb,on=['station_key','cohort'],suffixes=('_D','_M'),validate='one_to_one')
   for c,g in p.groupby('cohort'):rows.append(dict(fold=fold,cohort=c,view=view,stations=len(g),delta_mean_RMSE=float((g.RMSE_D-g.RMSE_M).mean()),delta_median_NSE=float(g.NSE_D.median()-g.NSE_M.median()),delta_median_r=float(g.r_D.median()-g.r_M.median())))
 windows=pd.DataFrame(rows);windows.to_csv(R/'reports/day_window_direction_comparison.csv',index=False)
 conclusions=[]
 for fold in ['F23','F24']:
  for c in ['N','H','X']:
   g=windows[windows.fold.eq(fold)&windows.cohort.eq(c)]
   if g.empty:continue
   mainrow=g[g.view.eq('BJT')].iloc[0];flip=False
   for alt in ['CHM','CMFD']:
    a=g[g.view.eq('BJT_common_'+alt)];b=g[g.view.eq(alt+'_common')]
    if len(a) and len(b) and float(a.delta_mean_RMSE.iloc[0])*float(b.delta_mean_RMSE.iloc[0])<0:flip=True
   conclusions.append(f"{fold} {c}：D-HF相对M-HF日RMSE变化 {mainrow.delta_mean_RMSE:+.5f} mg/L，NSE中位变化 {mainrow.delta_median_NSE:+.5f}；"+('公共日期日界比较出现收益方向反转，时间支持下尚不可辨别。' if flip else '已登记公共日期日界比较未发现RMSE收益方向反转；不等于日界已修复。'))
 result='\n'.join('- '+v for v in conclusions)
 for name in ['expert_diagnosis.md','short_conclusion.md']:
  p=R/'reports'/name;p.write_text(p.read_text(encoding='utf8')+'\n\n## 主要配对结论\n\n'+result+'\n',encoding='utf8')
 note='\n\n## 补充证据与解释边界\n\n共同训练读数上的月水平/动态误差另见training_HF_error_components.parquet，M-PUB在此表仅为同资料诊断，不是其实际目标。两入口参数、年度库存和摄取/损失、同河段差值分别见two_start_parameters.csv、annual_physical_summary.parquet、same_reach_daily_pairs.parquet。月报与全部正常高频月均差异见monthly_source_agreement.csv。四来源内部标签限于继承的158、225试点；其他河段保留总库存/通量账本，不将试点来源身份外推全域。\n\n本轮权威逐记录血缘是training_observation_lineage.parquet，逐项指向v3月观测ID或auto_4h的selected_record_id。模型边界元数据中继承的旧模板行号仅是站界参考，不是本轮高频原始行号，也不参与损失或选点。\n\n未取得本轮已登记且时空支持相容的高频实测流量，因此不能量化日内浓度—水量协方差；这不是证明项目内不存在其他流量资料。所有预测须联读prediction_identity_manifest.json。季节/月份距平由评价样本计算，仅为事后残差描述，不回流训练。\n'
 p=R/'reports/expert_diagnosis.md';p.write_text(p.read_text(encoding='utf8')+note,encoding='utf8')
 # Inspectable scientific time-series figures, raw units and all eligible observations.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import matplotlib.dates as mdates
 plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
 for fold in ['F23','F24']:
  if fold+'_M_HF' not in selected or fold+'_D_HF' not in selected:continue
  ta=selected[fold+'_M_HF'];tb=selected[fold+'_D_HF'];obs=pd.read_parquet(R/'reports'/f'{tb}_HF_evaluated.parquet');fa=pd.read_parquet(R/'outputs'/ta/'daily_station_mass_water.parquet');fb=pd.read_parquet(R/'outputs'/tb/'daily_station_mass_water.parquet');year=2023 if fold=='F23' else 2024
  fig,axes=plt.subplots(4,2,figsize=(15,12),constrained_layout=True)
  for ax,(s,g) in zip(axes.ravel(),obs.groupby('station_key')):
   ax.scatter(g.date,g.y,s=6,color='#222222',alpha=.5,label='合格日观测')
   for p,label,color in [(fa,'M-HF','#377eb8'),(fb,'D-HF','#e66101')]:
    q=p[p.station_key.eq(s)&p.date.dt.year.eq(year)];ax.plot(q.date,q.concentration_mg_l,lw=.8,color=color,label=label)
   ax.set_title(names[s]);ax.set_ylabel('TN (mg/L)');ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'));ax.set_ylim(bottom=0)
  axes[0,0].legend(fontsize=8);fig.suptitle(f'{fold}：{year}留出日过程，正常观测未统计删值；冻结水文日界近似')
  fig.savefig(R/'reports'/f'{fold}_daily_dynamics.png',dpi=140);plt.close(fig)
 print('SUPPLEMENTED_EXPERT_EVIDENCE',flush=True)
if __name__=='__main__':main()
