"""Independent event reconstruction, parameter mapping and mass redistribution summaries."""
from runtime import *
def main():
 obs=pd.read_parquet(R/'data/evaluation/observed_days.parquet').set_index(['station_key','date']);events=pd.read_parquet(R/'data/evaluation/events_frozen.parquet');records=[];totals=[];conc=[];MET=['ratio_error','peak_error','base_error']
 d,_=domain('H1');a={k:np.load(R/'data'/f'{k}.npy',mmap_mode='r') for k in ['s','volume_mm','fast_mm','percol_mm']}
 for c in read(R/'data/protocol.json')['configs']:
  arm=c['id'];folder=R/'outputs'/arm;pred=pd.read_parquet(folder/'daily_station.parquet').set_index(['station_key','date']);rows=[]
  for e in events.itertuples():
   if e.cross_period or e.background_start.year<2021 or (e.period=='2024' and e.background_start.year<2024):continue
   if e.station_key not in obs.index.get_level_values(0):continue
   g=obs.loc[e.station_key];base=g[(g.index>=e.background_start)&(g.index<e.start)];peak=g[(g.index>=e.start)&(g.index<=e.end)]
   if len(base)<4 or len(peak)<1:continue
   p=pred.loc[e.station_key];yb=base.y.median();yp=peak.y.max();pb=p.loc[base.index,'p'].median();pp=p.loc[peak.index,'p'].max();assert min(yb,yp,pb,pp)>0
   rows.append(dict(station_key=e.station_key,event_rank=e.event_rank,ratio_error=abs(np.log(pp/pb)-np.log(yp/yb)),peak_error=abs(pp-yp),base_error=abs(pb-yb)))
  out=pd.DataFrame(rows);saved=pd.read_parquet(folder/'event_scores.parquet');z=out.merge(saved,on=['station_key','event_rank'],suffixes=('_ind','_saved'),validate='one_to_one');assert len(z)==127
  error=max(float(abs(z[k+'_ind']-z[k+'_saved']).max()) for k in MET);assert error<1e-12
  records.append(dict(arm=arm,events=len(z),score_max_abs_difference=error))
  if c['family']=='REFERENCE':continue
  ledger=pd.read_parquet(folder/'annual_reach_ledger.parquet')
  for period,mask in [('1961-2024',ledger.year>=1961),('2021-2023',ledger.year.between(2021,2023)),('2024',ledger.year==2024)]:
   g=ledger[mask];row=dict(arm=arm,period=period)
   for key in ledger.columns:
    if key in ['year','reach_id']:continue
    row[key]=float(g.loc[g.year==g.year.max(),key].sum()) if key.endswith('_yearend') else float(g[key].sum())
   totals.append(row)
  land=np.load(folder/'land_history.npy',mmap_mode='r');fpre=land[10]/a['s']+land[11];mpre=land[8]
  vf=np.load(R/'data'/f'C_{c["omega"]:g}_VF.npy',mmap_mode='r') if c['family']=='C' else a['volume_mm'];vp=np.load(R/'data'/f'C_{c["omega"]:g}_VP.npy',mmap_mode='r') if c['family']=='C' else a['volume_mm']
  cf=1000*fpre/(vf*d.area_ha*10);cm=1000*mpre/(vp*d.area_ha*10)
  # These are effective pool concentrations; A/B two pools do not claim two physical volumes.
  frame=[]
  for year in range(1961,2025):
   ix=d.dates.year==year
   for j,reach in enumerate(d.global_reach_ids):
    frame.append(dict(year=year,reach_id=reach,fast_concentration_mean=float(cf[ix,j].mean()),common_concentration_mean=float(cm[ix,j].mean()),fast_concentration_max=float(cf[ix,j].max()),common_concentration_max=float(cm[ix,j].max())))
  pd.DataFrame(frame).to_parquet(folder/'annual_effective_concentrations.parquet',index=False)
  for label,array in [('fast_domain',cf),('common_or_percol_domain',cm)]:
   quant=np.quantile(array,[0,.5,.9,.99,1]);conc.append(dict(arm=arm,domain=label,min=quant[0],median=quant[1],p90=quant[2],p99=quant[3],max=quant[4]))
  np.save(folder/'effective_fast_concentration.npy',cf);np.save(folder/'effective_common_concentration.npy',cm)
 pd.DataFrame(totals).to_csv(R/'reports/mass_redistribution.csv',index=False);pd.DataFrame(conc).to_csv(R/'reports/effective_concentrations.csv',index=False);put(R/'reports/independent_evaluation_audit.json',records)
 model=read(P/'20260916_2/outputs/C0_s1/model.json');rows=[]
 for i,(name,value) in enumerate(zip(model['names'],model['parameters'])):rows.append(dict(index=i,name=name,value=value,active_in_new_land_family=i==3 or 11<=i<=17,active_in_routing=i==2,effect='common upper loss' if i==3 or 11<=i<=17 else ('river removal' if i==2 else 'inactive: replaced mobilization or water split')))
 pd.DataFrame(rows).to_csv(R/'reports/parameter_effects.csv',index=False)
 bootstrap=pd.read_csv(R/'reports/bootstrap_changes.csv');summ=[]
 for (arm,base,period),g in bootstrap.groupby(['arm','reference','period']):
  assert len(g)==1000
  for k in MET:
   v=g[k].quantile([.025,.5,.975]).to_numpy();summ.append(dict(arm=arm,reference=base,period=period,metric=k,q025=v[0],median=v[1],q975=v[2]))
 pd.DataFrame(summ).to_csv(R/'reports/bootstrap_summary.csv',index=False)
 print('DELIVERY_ANALYSIS_PASS')
if __name__=='__main__':main()
