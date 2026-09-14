"""Independent conservative daily MINERAL_LIFETIME model on frozen hydrology."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME, sha256, memory_guard
from audit_inputs import HYDRO, SOURCE, TOPO

import numpy as np
import pandas as pd
from numba import njit


@dataclass
class TNData:
    product: str
    dates: pd.DatetimeIndex
    months: pd.DatetimeIndex
    mid: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    source: np.ndarray
    crop: np.ndarray
    contact: np.ndarray
    fast_fraction: np.ndarray
    lower_release: np.ndarray
    old_lower_release: np.ndarray
    fast_water: np.ndarray
    slow_water: np.ndarray
    official_water: np.ndarray
    h_month: np.ndarray
    h_day: np.ndarray
    temperature: np.ndarray
    upper_water: np.ndarray
    percolation: np.ndarray
    metadata: list
    release_fraction: np.ndarray
    released_water: np.ndarray
    enabled: np.ndarray
    order: list
    downstream: dict
    terminal: list

    def monthly_sum(self,array):
        return np.add.reduceat(array,self.starts,axis=0)


def load_data(product='formal',calendar='CENTRAL',verify_hashes=True):
    if product not in HYDRO:
        raise ValueError(product)
    directory=HYDRO[product]
    if verify_hashes:
        manifest=json.loads((ROOT/'5_Test/20260905_1/reports/input_manifest.json').read_text(encoding='utf-8'))
        expected={r['path']:r['sha256'] for r in manifest['files']}
        for path in [SOURCE[product],TOPO,*[directory/f'tn_hydrology_{name}.parquet' for name in
                       ['reach_daily','reach_monthly','reservoir_daily','reservoir_static_metadata']]]:
            if sha256(path)!=expected[str(path)]:
                raise RuntimeError(f'Frozen input changed: {path}')
    fields=['date','reach_id','local_fast_response_m3_s','local_slow_response_m3_s',
            'routed_total_m3_s','percolation_to_lower_mm_day','upper_response_storage_mm',
            'lower_slow_storage_mm','tmean_c','channel_bankfull_hydraulic_exposure_central_day','bankfull_depth_m']
    daily=pd.read_parquet(directory/'tn_hydrology_reach_daily.parquet',columns=fields).sort_values(['date','reach_id'])
    dates=pd.DatetimeIndex(pd.to_datetime(daily.date.drop_duplicates()))
    def array(name):
        return daily[name].to_numpy(np.float64).reshape(-1,230).copy()
    monthly=pd.read_parquet(directory/'tn_hydrology_reach_monthly.parquet').sort_values(['month','reach_id'])
    months=pd.DatetimeIndex(pd.to_datetime(monthly.month.drop_duplicates()))
    mid=np.asarray((dates.year-1961)*12+dates.month-1,dtype=np.int64)
    starts=np.flatnonzero(np.r_[True,np.diff(mid)!=0]).astype(np.int64)
    stops=np.r_[starts[1:],len(dates)].astype(np.int64)
    if calendar not in ['CENTRAL','OLD']:
        raise ValueError(calendar)
    source_path=ROOT/f'5_Test/20260906_1/outputs/source_{product}.parquet'
    if verify_hashes:
        prepared=json.loads((ROOT/'5_Test/20260906_1/reports/prepared_manifest.json').read_text(encoding='utf-8'))
        if sha256(source_path)!=prepared['derived'][str(source_path)]:
            raise RuntimeError('Prepared forcing changed')
    source=pd.read_parquet(SOURCE[product] if calendar=='OLD' else source_path)
    source=source.loc[source.calendar_scenario.eq('CENTRAL')].sort_values(['year','month','reach_id'])
    if len(source)!=len(months)*230:
        raise ValueError(f'Missing source/calendar coverage: {product} {calendar}')
    area=monthly.catchment_area_km2.to_numpy().reshape(-1,230)[0]
    fast,slow=array('local_fast_response_m3_s')*86400,array('local_slow_response_m3_s')*86400
    fmm,smm=fast/(area*1000),slow/(area*1000)
    perc,upper,lower=array('percolation_to_lower_mm_day'),array('upper_response_storage_mm'),array('lower_slow_storage_mm')
    contact_water=fmm+perc
    contact=np.divide(contact_water,contact_water+upper,out=np.zeros_like(fmm),where=(contact_water+upper)>1e-12)
    fastfrac=np.divide(fmm,contact_water,out=np.zeros_like(fmm),where=contact_water>1e-12)
    release=np.divide(smm,lower+smm,out=np.zeros_like(smm),where=(lower+smm)>1e-12)
    oldavail=np.vstack([lower[:1],lower[:-1]])+perc
    oldrelease=np.divide(smm,oldavail,out=np.zeros_like(smm),where=oldavail>1e-12)
    hd=array('channel_bankfull_hydraulic_exposure_central_day')/array('bankfull_depth_m')
    hm=(monthly.channel_bankfull_hydraulic_exposure_central_day/monthly.bankfull_depth_m).to_numpy().reshape(-1,230)
    # Preserve the registered limiting dry attenuation in the comparison core.
    hd=np.where(np.isfinite(hd),hd,1e6)
    hm=np.where(np.isfinite(hm),hm,1e6)
    graph=pd.read_csv(TOPO)
    downstream={int(r.reach_id)-1:int(r.downstream_reach)-1 for r in graph.itertuples() if pd.notna(r.downstream_reach)}
    degree={r:0 for r in range(230)}
    for target in downstream.values(): degree[target]+=1
    queue=sorted(r for r in degree if degree[r]==0)
    order=[]
    while queue:
        r=queue.pop(0);order.append(r)
        if r in downstream:
            target=downstream[r];degree[target]-=1
            if degree[target]==0:queue.append(target);queue.sort()
    assert len(order)==230
    position={r:i for i,r in enumerate(order)}
    meta=pd.read_parquet(directory/'tn_hydrology_reservoir_static_metadata.parquet')
    metadata=[]
    for r in meta.itertuples():
        controls=[int(v)-1 for v in r.control_reaches]
        assert all(downstream[c]==int(r.outflow_reach)-1 for c in controls)
        metadata.append({'id':str(r.reservoir_entity_id),'controls':controls,
                         'target':int(r.outflow_reach)-1,'fraction':float(r.local_capture_fraction),
                         'position':max(position[c] for c in controls)})
    metadata.sort(key=lambda r:(r['position'],r['id']))
    rid={r['id']:i for i,r in enumerate(metadata)}
    rf=pd.read_parquet(directory/'tn_hydrology_reservoir_daily.parquet',columns=['date','reservoir_entity_id','enabled','storage_m3','total_release_m3'])
    rf['res_idx']=rf.reservoir_entity_id.map(rid)
    rf=rf.sort_values(['date','res_idx'])
    water=rf.total_release_m3.to_numpy().reshape(-1,len(metadata)).copy()
    avail=(rf.total_release_m3+rf.storage_m3).to_numpy().reshape(water.shape)
    frac=np.divide(water,avail,out=np.zeros_like(water),where=avail>1e-12)
    result=TNData(product,dates,months,mid,starts,stops,
        source[['fertilizer_kg_n','manure_kg_n','cropland_bnf_kg_n','atmospheric_deposition_kg_n']].sum(axis=1).to_numpy().reshape(-1,230).copy(),
        source.crop_demand_kg_n.to_numpy().reshape(-1,230).copy(),contact,fastfrac,release,oldrelease,fast,slow,
        array('routed_total_m3_s')*86400,hm,hd,array('tmean_c'),upper,perc,metadata,frac,water,
        rf.enabled.to_numpy().reshape(water.shape).copy(),order,downstream,[r for r in order if r not in downstream])
    memory_guard()
    return result


@njit(cache=False)
def local_daily_kernel(probability,survival,fast_fraction,lower_release,source,crop,mid,days_per_month,uniform,uptake_uniform=None):
    nd,nr=probability.shape
    nm=source.shape[0]
    fast=np.zeros((nd,nr));slow=np.zeros((nd,nr))
    other=np.zeros((nm,nr));uptake=np.zeros((nm,nr))
    mineral_month=np.zeros((nm,nr));lower_month=np.zeros((nm,nr))
    mineral=np.zeros(nr);lower=np.zeros(nr)
    for d in range(nd):
        m=mid[d]
        first=d==0 or mid[d-1]!=m
        for r in range(nr):
            amount=source[m,r]/days_per_month[m] if uniform else (source[m,r] if first else 0.0)
            ud=uniform if uptake_uniform is None else uptake_uniform
            demand=crop[m,r]/days_per_month[m] if ud else (crop[m,r] if first else 0.0)
            pre=mineral[r]+amount
            withdrawn=min(pre,demand)
            available=pre-withdrawn
            mobilized=available*probability[d,r]
            fast[d,r]=mobilized*fast_fraction[d,r]
            lowerpre=lower[r]+mobilized*(1.0-fast_fraction[d,r])
            slow[d,r]=lowerpre*lower_release[d,r]
            lower[r]=lowerpre-slow[d,r]
            mineral[r]=(available-mobilized)*survival[r]
            other[m,r]+=(available-mobilized)*(1.0-survival[r])
            uptake[m,r]+=withdrawn
        if d==nd-1 or mid[d+1]!=m:
            mineral_month[m]=mineral
            lower_month[m]=lower
    return fast,slow,other,uptake,mineral_month,lower_month


def local_daily(data,log_alpha=-8.305930459477558,beta=0.2500000175,tau=1360.174801,
                corrected_lower=True,source_timing='monthly_pulse'):
    if source_timing not in ['monthly_pulse','uniform_daily']:raise ValueError(source_timing)
    alpha=np.broadcast_to(np.exp(np.asarray(log_alpha)),(230,))
    tau=np.broadcast_to(np.asarray(tau),(230,))
    p=-np.expm1(-np.minimum(alpha*data.contact**beta,700.0))
    survival=np.exp(-1.0/tau)
    result=local_daily_kernel(p,survival,data.fast_fraction,
            data.lower_release if corrected_lower else data.old_lower_release,
            data.source,data.crop,data.mid,data.stops-data.starts,source_timing=='uniform_daily')
    return dict(zip(['fast','slow','other','crop','mineral','lower'],result))


def local_monthly_coefficients(data,log_alpha,beta,tau,corrected_lower=False):
    """Independent implementation of the old month coefficient equation."""
    p=-np.expm1(-np.minimum(np.exp(log_alpha)*data.contact**beta,700.0))
    survival=np.exp(-1.0/np.asarray(tau))
    release=data.lower_release if corrected_lower else data.old_lower_release
    mineral=np.zeros(230);lower=np.zeros(230)
    output={k:np.zeros_like(data.source) for k in ['fast','slow','other','crop','mineral','lower']}
    for m,(a,b) in enumerate(zip(data.starts,data.stops)):
        contact=p[a:b]
        retention=(1-contact)*survival
        before=np.cumprod(np.vstack([np.ones((1,230)),retention[:-1]]),axis=0)
        mobilized=before*contact
        percolated=mobilized*(1-data.fast_fraction[a:b])
        tail=np.cumprod((1-release[a:b])[::-1],axis=0)[::-1]
        carry=np.prod(1-release[a:b],axis=0)
        pre=mineral+data.source[m]
        crop=np.minimum(pre,data.crop[m]);available=pre-crop
        output['fast'][m]=available*np.sum(mobilized*data.fast_fraction[a:b],axis=0)
        output['slow'][m]=lower*(1-carry)+available*np.sum(percolated*(1-tail),axis=0)
        output['other'][m]=available*np.sum(before*(1-contact)*(1-survival),axis=0)
        lower=lower*carry+available*np.sum(percolated*tail,axis=0)
        mineral=available*np.prod(retention,axis=0)
        output['crop'][m]=crop;output['mineral'][m]=mineral;output['lower'][m]=lower
    return output


def redistribute_by_water(data,fast_month,slow_month):
    fast=np.zeros_like(data.fast_water);slow=np.zeros_like(data.slow_water)
    for m,(a,b) in enumerate(zip(data.starts,data.stops)):
        fq,sq=data.fast_water[a:b],data.slow_water[a:b]
        fw=np.divide(fq,fq.sum(axis=0),out=np.zeros_like(fq),where=fq.sum(axis=0)>1e-12)
        sw=np.divide(sq,sq.sum(axis=0),out=np.zeros_like(sq),where=sq.sum(axis=0)>1e-12)
        fast[a:b]=fw*fast_month[m];slow[a:b]=sw*slow_month[m]
    return fast,slow


@njit(cache=False)
def reservoir_scan(captured,release_fraction):
    n=len(captured)
    releases=np.empty(n);stocks=np.empty(n)
    storage=0.0
    for d in range(n):
        pre=storage+captured[d]
        releases[d]=pre*release_fraction[d]
        storage=pre-releases[d]
        stocks[d]=storage
    return releases,stocks


def route(data,local,vf=0.021952884993820768,exposure='monthly',water_replay=False):
    """Topological time-series routing. Reservoir scans are causal and conservative."""
    nd,nr=local.shape
    inlet=np.zeros_like(local);preout=np.zeros_like(local);official=np.zeros_like(local)
    removed=np.zeros_like(local)
    nrsv=len(data.metadata)
    captures=np.zeros((nd,nrsv));releases=np.zeros_like(captures);stocks=np.zeros_like(captures)
    by_control={c:i for i,r in enumerate(data.metadata) for c in r['controls']}
    seen=[set() for _ in data.metadata]
    for r in data.order:
        h=data.h_month[data.mid,r] if exposure=='monthly' else data.h_day[:,r]
        attenuation=np.exp(-vf*h)
        local_delivered=local[:,r]*np.exp(-0.5*vf*h)
        preout[:,r]=inlet[:,r]*attenuation+local_delivered
        official[:,r]=preout[:,r]
        removed[:,r]=inlet[:,r]+local[:,r]-preout[:,r]
        if r not in by_control:
            if r in data.downstream:inlet[:,data.downstream[r]]+=preout[:,r]
            continue
        k=by_control[r];meta=data.metadata[k]
        if len(meta['controls'])>1:
            captures[:,k]+=preout[:,r]
            seen[k].add(r)
            if seen[k]!=set(meta['controls']):continue
            bypass=np.zeros(nd)
        else:
            fraction=np.where(data.enabled[:,k],meta['fraction'],1.0)
            bypass=(1-fraction)*local_delivered
            captures[:,k]=preout[:,r]-bypass
        if water_replay:
            releases[:,k]=data.released_water[:,k]
        else:
            releases[:,k],stocks[:,k]=reservoir_scan(captures[:,k],data.release_fraction[:,k])
        inlet[:,meta['target']]+=releases[:,k]+bypass
        # This is precisely the mixed boundary convention of the hydro export.
        if len(meta['controls'])==1:
            official[:,r]=releases[:,k]+bypass
    return {'inlet':inlet,'preout':preout,'official':official,'channel_removed':removed,
            'captures':captures,'releases':releases,'stocks':stocks,
            'terminal':official[:,data.terminal].sum(axis=1)}


def station_predictions(data,local,routed,water,obs,vf):
    local_month=data.monthly_sum(local)
    inlet_month=data.monthly_sum(routed['inlet'])
    water_local_month=data.monthly_sum(data.fast_water+data.slow_water)
    water_inlet_month=data.monthly_sum(water['inlet'])
    ti=((obs.year.to_numpy()-1961)*12+obs.month.to_numpy()-1).astype(int)
    ri=obs.reach_id.to_numpy(int)-1
    f=obs.downstream_fraction_on_reach.to_numpy(float)
    h=data.h_month[ti,ri]
    load=inlet_month[ti,ri]*np.exp(-vf*h*f)+f*local_month[ti,ri]*np.exp(-0.5*vf*h*f)
    volume=water_inlet_month[ti,ri]+f*water_local_month[ti,ri]
    concentration=np.divide(1000*load,volume,out=np.full(len(obs),np.nan),where=volume>0)
    result=obs.copy()
    result['prediction_mg_l']=concentration
    result['station_water_m3_month']=volume
    result['station_load_kg_n_month']=load
    return result
