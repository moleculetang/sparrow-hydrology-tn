"""No TN input in build_features. Full-domain frozen H1 water replay only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import shutil,numpy as np,pandas as pd
from mltn.common import ROOT,write,read,sha
BASE=ROOT.parent
def upstream_mean(v,area,up):
    """Unknown propagates only to truly reachable upstream support, not NaN*zero."""
    v=np.asarray(v);mass=np.where(np.isfinite(v),v,0)*area
    result=mass@up.T/(up@area);unknown=(~np.isfinite(v)).astype(np.int16)@up.T>0
    return np.where(unknown,np.nan,result)
def build_features():
    old=BASE/'20260928_1';raw=BASE/'20260917_5/data/domains/FULL24C'
    layout=read(old/'vendor/legacy22/data/domains/FULL24C/arrays.json');top=read(old/'vendor/legacy22/data/domains/FULL24C/topology.json')
    ids={};a={}
    fields=['dates','area_ha','static_raw','fast_water','slow_water','upper_water','percolation','soil_wetness','soil_water_mm','temperature','released_water','enabled']
    for key in fields:
        path=raw/layout[key]['file'];s=sha(path)
        if s!=layout[key]['sha256']:raise ValueError('H1_HASH '+key)
        ids[str(path)]=s;a[key]=np.load(path,mmap_mode='r')
    dates=pd.DatetimeIndex(a['dates']);keep=dates>=pd.Timestamp('2015-01-01');ds=dates[keep];nd=len(ds);nr=230
    area=np.asarray(a['area_ha']);local=np.asarray(a['fast_water'])+np.asarray(a['slow_water']);inlet=np.zeros_like(local);out=np.zeros_like(local)
    down={int(k):int(v) for k,v in top['downstream'].items()};controls={c:i for i,m in enumerate(top['metadata']) for c in m['controls']};seen=[set() for _ in top['metadata']]
    for r in top['order']:
        out[:,r]=inlet[:,r]+local[:,r]
        if r not in controls:
            if r in down:inlet[:,down[r]]+=out[:,r]
        else:
            k=controls[r];m=top['metadata'][k];seen[k].add(r)
            if seen[k]!=set(m['controls']):continue
            bypass=np.zeros(len(dates)) if len(m['controls'])>1 else (1-np.where(a['enabled'][:,k],m['fraction'],1))*local[:,r]
            inlet[:,m['target']]+=a['released_water'][:,k]+bypass
            if len(m['controls'])==1:out[:,r]=a['released_water'][:,k]+bypass
    up=np.eye(nr)
    # Physical reach adjacency plus reservoir links, no label-based adjacency.
    edges=dict(down)
    for m in top['metadata']:
        for c in m['controls']:edges[c]=m['target']
    for r in top['order']:
        if r in edges:up[edges[r]]+=up[r]
    up=(up>0).astype(float);ua=up@area
    adj=np.zeros((nr,nr))
    for r,d in edges.items():adj[d,r]=1
    adj/=np.maximum(1,adj.sum(1,keepdims=True))
    values=[];names=[];units=[]
    def add(name,v,unit):
        if v.shape!=(nd,nr):v=np.broadcast_to(v,(nd,nr))
        values.append(np.asarray(v,dtype=np.float32));names.append(name);units.append(unit)
    for key,unit in [('fast_water','m3/day'),('slow_water','m3/day'),('upper_water','mm storage'),('percolation','mm/day'),('soil_wetness','fraction'),('soil_water_mm','mm storage'),('temperature','degC')]:
        v=np.asarray(a[key][keep]);add('local_'+key,v,unit)
        add('upstream_'+key,upstream_mean(v,area,up),unit+' area mean')
        if key in ['fast_water','slow_water','upper_water','percolation']:
            add('local_log1p_'+key,np.log1p(v),unit+' log1p numerical feature')
    weather_path=Path(read(BASE/'20260916_2/data/drivers/feature_registry.json')['weather_source']);ids[str(weather_path)]=sha(weather_path)
    weather=pd.read_parquet(weather_path,columns=['date','reach_id','precipitation_daily_mm','pet_fao56_mm_day']);weather['date']=pd.to_datetime(weather.date)
    for key in ['precipitation_daily_mm','pet_fao56_mm_day']:
        v=weather.pivot(index='date',columns='reach_id',values=key).reindex(index=ds,columns=np.arange(1,231)).to_numpy()
        if not np.isfinite(v).all():raise ValueError('MISSING_H1_WEATHER '+key)
        add('local_'+key,v,'mm/day');add('upstream_'+key,upstream_mean(v,area,up),'mm/day area mean')
    add('reach_water_log1p',np.log1p(out[keep]),'m3/day log1p');add('reach_inlet_water_log1p',np.log1p(inlet[keep]),'m3/day log1p')
    for i,n in enumerate(top['static_fields']):add(n,a['static_raw'][:,i],n)
    add('log_physical_area_ha',np.log1p(area),'ha log1p');add('log_upstream_area_ha',np.log1p(ua),'ha log1p')
    for n,v in [('annual_sin',np.sin(2*np.pi*(ds.dayofyear-1)/np.where(ds.is_leap_year,366,365))),('annual_cos',np.cos(2*np.pi*(ds.dayofyear-1)/np.where(ds.is_leap_year,366,365)))]:add(n,np.asarray(v)[:,None],'dimensionless calendar')
    # Explicit independent annual/monthly products. Missing entries remain NaN.
    def table(rel):
        p=BASE/rel;ids[str(p)]=sha(p)
        return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
    def annual(col,q,value,divide_area=False):
        z=q.groupby(['year','reach_id'])[value].sum(min_count=1).unstack('reach_id').reindex(index=np.arange(2015,2025),columns=np.arange(1,231)).to_numpy()
        v=z[ds.year-2015];v=v/area if divide_area else v
        unit=value+(' / ha incremental physical area' if divide_area else '')+' annual retrospective'
        add(col,v,unit);add('upstream_'+col,upstream_mean(v,area,up),unit+' upstream area mean')
    q=table('20260928_1/outputs/agriculture_reference/annual_crop_activity_reference.parquet')
    for col in ['harvested_area_ha','harvest_n_kg','residue_n_kg','bnf_plant_kg','bnf_soil_kg']:annual(col,q,col,True)
    q=table('20260928_1/outputs/agriculture_reference/annual_crop_source_reference.parquet')
    for field,g in q.groupby('field'):annual('agriculture_'+str(field),g,'mass_kg',True)
    q=table('20260926_1/outputs/luh3_reach/luh3_reach_state_1961_2024.csv')
    for state,g in q.groupby('state'):annual('luh3_'+str(state),g,'physical_area_m2',True)
    q=table('20260926_1/outputs/mod17_gee_reachcover/reach_clcd_class_annual_npp_qc_2001_2025.csv')
    for col in ['npp_numeric_kg_C_year','npp_missing_area_m2','npp_negative_area_m2']:annual('mod17_'+col,q,col,True)
    q=table('20260926_1/outputs/deposition_landclass/monthly_reach_clcd_class_deposition_1961_2024.parquet')
    q=q[q.clcd_class.ne(5)].copy() # class 5 water excluded, source identity retained
    for col in ['drynhx_kg_n','drynoy_kg_n','wetnhx_kg_n','wetnoy_kg_n']:
        v=q.groupby(['year','month','reach_id'])[col].sum().unstack('reach_id').reindex(index=pd.MultiIndex.from_arrays([ds.year,ds.month]),columns=np.arange(1,231)).to_numpy()/area
        add('deposition_'+col,v,'kg N/ha/month retrospective')
    # Strict trailing windows only, warm-up 2015 retained.
    initial=list(zip(names,values))
    for n,v in initial:
        if n.startswith('local_') and any(x in n for x in ['water','precipitation','pet_','temperature','wetness','percolation']):
            for w in [7,30,90,365]:add(n+'_past_'+str(w),pd.DataFrame(v).shift(1).rolling(w,min_periods=w).mean().to_numpy(),'trailing completed days')
    x=np.stack(values,axis=-1)
    (ROOT/'data').mkdir(exist_ok=True);np.save(ROOT/'data/reach_features.npy',x);np.save(ROOT/'data/adjacency.npy',adj.astype('float32'));np.save(ROOT/'data/upstream.npy',up)
    # Station metadata affects readout geometry only, never a trainable ID.
    reg=pd.read_parquet(old/'data/station_registry.parquet').sort_values('station_key').reset_index(drop=True);support=read(old/'vendor/legacy22/data/spatial_support.json')
    station_water=[]
    for row in reg.itertuples():
        r=int(row.reach_id)-1;f=float(row.downstream_fraction_on_reach)
        if row.station_type=='predam':f=1
        # Current baseline uses uniform OU support (O0); keep audited support identity.
        w=inlet[:,r]+f*local[:,r]
        if row.station_type=='dam_outlet':w=np.asarray(a['released_water'][:,int(row.reservoir_index)])
        elif row.station_type=='postdam_mixed':w=out[:,r]
        station_water.append(w[keep])
    station_water=np.stack(station_water,1)
    np.save(ROOT/'data/station_water.npy',station_water.astype('float64'))
    reg.to_parquet(ROOT/'data/station_registry.parquet',index=False)
    write(ROOT/'data/feature_identity.json',dict(dates=ds.strftime('%Y-%m-%d').tolist(),features=names,units=units,raw_hashes=ids,graph='directed frozen topology; source to downstream; no TN correlations',support='frozen O0 uniform OU, dam readout explicit',source_availability='annual/current-month products retrospective conditional forcings; no realtime claim',excluded=['source.npy','crop.npy','h_day.npy','h_month.npy','D29','LAND1','TN','NH4','DO'],shape=x.shape))
    write(ROOT/'data/topology.json',top)
    shutil.copyfile(old/'vendor/legacy22/data/spatial_blocks.json',ROOT/'data/spatial_blocks.json')
    return dict(shape=x.shape,missing_fraction=float(np.isnan(x).mean()),station_water_nonpositive=int((station_water<=0).sum()))
def observations():
    old=BASE/'20260928_1/data';ident={}
    for name in ['monthly_tn.parquet','hf_daily.parquet','hf_readings.parquet','frozen_evaluation_events.parquet']:
        p=old/name;ident[name]=sha(p);shutil.copyfile(p,ROOT/'data'/name)
    r=pd.read_parquet(ROOT/'data/hf_readings.parquet');r=r[r.indicator.eq('TN')].copy()
    if r.duplicated(['station_key','monitoring_time']).any():raise ValueError('DUPLICATE_TN_READING')
    r.to_parquet(ROOT/'data/hf_readings.parquet',index=False)
    for n in ['monthly_tn','hf_daily']:
        q=pd.read_parquet(ROOT/'data'/f'{n}.parquet');q=q[q.eligible & q.tn_mg_l.notna() & np.isfinite(q.tn_mg_l) & q.tn_mg_l.ge(0)].copy()
        q['date']=pd.to_datetime(q.date);q=q[q.date.dt.year.between(2016,2024)]
        q['year']=q.date.dt.year;q['month']=q.date.dt.month
        if q.duplicated(['station_key','date']).any():raise ValueError('DUPLICATE_LABEL '+n)
        q.to_parquet(ROOT/'data'/f'{n}_accepted.parquet',index=False)
    write(ROOT/'data/observation_identity.json',dict(raw_hashes=ident,monthly='automatic valid-read arithmetic mean; equal-day approximation when counts unknown',hf='frozen deduplicated valid TN readings; daily support gates inherited; no residual trimming',new_stations_admitted=0))
if __name__=='__main__':
    write(ROOT/'evidence/preparation.json',build_features());observations()
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'data').glob('*') if p.is_file()};write(ROOT/'evidence/data_manifest.json',hashes)
