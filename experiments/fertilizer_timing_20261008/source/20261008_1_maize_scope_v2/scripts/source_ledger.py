"""Independent full-history source-label and network closure, never relaxed."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha
import numpy as np,pandas as pd,hashlib
from timing.chain import TimingChain
from d29_platform.timing_labels import propagate_source_labels
from d29_platform.coupling import route_and_sample,sampling_from_inference_model

def plant_activity_diagnostics(result,return_arrays=False):
    """Demand/capacity/debt and independent uptake balance on actual states.

    Daily target deficits are stock-days, not additional external nitrogen or
    annual harvest debt. Only unmet mutually exclusive outputs form that debt.
    Infinite/maximum-float capacity represents an absent limit, never a sum.
    """
    from d29_platform.precision_transfer import transfer_expansion
    i=result._inputs;nt,nr,nl=i['shape'];before=result.states[:-1].copy()
    for day in i['transition_days']:
        low=np.zeros_like(before[day]);low[...,1:]=-i['compensation_history'][day].reshape(nr,nl,4)
        before[day],_,_=transfer_expansion(before[day],low,i['matrices'][i['event'][day]])
    b=before.reshape(nt,nr*nl,5);source=i['sources'];pa,pp,*_=i['probabilities']
    realized=i['realized_outflows'];planned=i['outflows'];target=i['target'];cap=i['plant_growth_capacity']
    available=b[...,3]+source[...,3]+pa*b[...,1]+pp*b[...,2]
    need=np.maximum(0.,target+planned.sum(-1)-b[...,0]-source[...,0])
    opportunity=np.minimum(need,cap);prescribed=np.minimum(available,opportunity)
    inferred=result.states[1:,...,0].reshape(nt,nr*nl)+realized.sum(-1)-b[...,0]-source[...,0]
    mismatch=float(abs(inferred-prescribed).max(initial=0.))
    negative=float(np.maximum(0.,-inferred).max(initial=0.))
    excess=float(np.maximum(0.,inferred-prescribed).max(initial=0.))
    bounded=np.isfinite(cap)&(cap<np.finfo(float).max/2)
    deficit=np.maximum(0.,target-result.states[1:,...,0].reshape(nt,nr*nl))
    def land_sum(values):return values.reshape(nt,nr,nl).sum(axis=(0,1)).tolist()
    summary=dict(actual_uptake_by_land_kg=land_sum(prescribed),
        potential_daily_demand_with_capacity_by_land_kg=land_sum(opportunity),
        supply_limited_uptake_shortfall_by_land_kg=land_sum(np.maximum(0.,opportunity-prescribed)),
        explicit_growth_capacity_by_land_kg=land_sum(np.where(bounded,cap,0.)),
        target_deficit_stock_days_by_land_kg_day=land_sum(deficit),
        end_year_target_deficit_by_land_kg=deficit[-1].reshape(nr,nl).sum(axis=0).tolist(),
        planned_output_by_land_destination_kg=planned.reshape(nt,nr,nl,3).sum(axis=(0,1)).tolist(),
        realized_output_by_land_destination_kg=realized.reshape(nt,nr,nl,3).sum(axis=(0,1)).tolist(),
        unmet_output_by_land_destination_kg=result.unmet_plant_outflows.sum(axis=(0,1)).tolist(),
        uptake_balance_max_error_kg=mismatch,negative_inferred_uptake_max_kg=negative,uptake_excess_max_kg=excess,
        uptake_constraints_passed=max(mismatch,negative,excess)<=1e-7,
        target_deficit_is_additive_external_source=False,plant_output_is_potential_not_observed_yield=True)
    if not return_arrays:return summary
    arrays=dict(actual_uptake_kg=prescribed.reshape(nt,nr,nl).sum(axis=0),
        potential_demand_with_capacity_kg=opportunity.reshape(nt,nr,nl).sum(axis=0),
        supply_limited_uptake_shortfall_kg=np.maximum(0.,opportunity-prescribed).reshape(nt,nr,nl).sum(axis=0),
        explicit_growth_capacity_kg=np.where(bounded,cap,0.).reshape(nt,nr,nl).sum(axis=0),
        target_deficit_stock_days_kg_day=deficit.reshape(nt,nr,nl).sum(axis=0),
        end_year_target_deficit_kg=deficit[-1].reshape(nr,nl),
        planned_output_kg=planned.reshape(nt,nr,nl,3).sum(axis=0),
        realized_output_kg=realized.reshape(nt,nr,nl,3).sum(axis=0),
        unmet_output_kg=result.unmet_plant_outflows.sum(axis=0))
    return summary,arrays

def ledger(a,x):
    nk=len(a.inputs.labels);st=a.inputs.initial();tag_state=np.zeros((*st.shape[:2],nk,5));tag_state[:,:,-1]=st;tag_corr=None;local=np.zeros((len(a.data.dates),230));localtags=np.zeros((*local.shape,nk));rows=[];maxima=dict(local=0.,source_balance=0.,source_sum=0.);native=TimingChain._run
    localfast=np.zeros_like(local);localslow=np.zeros_like(local);stocks=np.zeros((len(local),13,5));unmet=np.zeros((len(local),13,3));external=np.zeros((len(local),nk));realized=np.zeros_like(unmet)
    def wrapped(chain,year,initial,eta,corr):
        nonlocal tag_state,tag_corr
        r,mask=native(chain,year,initial,eta,corr);idx=np.flatnonzero(a.data.dates.year==year);post=initial[...,0].copy()
        if int(idx[0]) in a.inputs.transitions:post[:,:12]=np.einsum('ri,rij->rj',post[:,:12],a.inputs.transitions[int(idx[0])])
        generated=a.inputs.annual(year,post);tags=generated['tags']*np.exp(eta)
        z=propagate_source_labels(r,tagged_initial=tag_state,tagged_sources=tags,labels=a.inputs.labels,record_history=True,compensated=True,initial_compensation=tag_corr)
        tag_state=z.final.copy();tag_corr=z.compensation.copy();local[idx]=r.fluxes[...,:2].sum(axis=(2,3));localtags[idx]=z.flux_history[...,:2].sum(axis=(2,4))
        maxima['local']=max(maxima['local'],r.max_local_balance_kg);maxima['source_balance']=max(maxima['source_balance'],z.max_local_balance_kg);maxima['source_sum']=max(maxima['source_sum'],z.max_state_sum_error_kg,z.max_flux_sum_error_kg)
        represented=r.states.copy();represented[...,1:]-=r._inputs['compensation_history'].reshape(len(idx)+1,230,13,4)
        localfast[idx]=r.fluxes[...,0].sum(axis=2);localslow[idx]=r.fluxes[...,1].sum(axis=2)
        stocks[idx]=represented[1:].sum(axis=1);unmet[idx]=r.unmet_plant_outflows.sum(axis=1);external[idx]=tags.sum(axis=(1,2,4));realized[idx]=r._inputs['realized_outflows'].reshape(len(idx),230,13,3).sum(axis=1)
        activity,activity_arrays=plant_activity_diagnostics(r,return_arrays=True)
        activity_file=a.folder/'plant_activity_yearly'/f'{year}.npz';activity_file.parent.mkdir(exist_ok=True)
        np.savez_compressed(activity_file,**activity_arrays)
        activity.update(reach_land_file=str(activity_file.relative_to(a.folder)).replace('\\','/'),
            reach_land_sha256=sha(activity_file),physical_land_units_not_crop_replicates=True)
        rows.append(dict(year=year,minimum_state_kg=float(represented.min()),minimum_label_kg=float(z.state_history.min()),stocks_by_land_kg=r.final.sum(axis=0).tolist(),plant_shortfall_by_land_kg=r.unmet_plant_outflows.sum(axis=(0,1)).tolist(),external_by_label_kg=tags.sum(axis=(0,1,2)).tolist(),flux_by_label_kg=z.flux_history.sum(axis=(0,1,2)).tolist(),management_ledger=generated['ledger'],plant_activity=activity))
        write_json(a.folder/'ledger_progress.json',dict(year=year,maxima=maxima));return r,mask
    TimingChain._run=wrapped
    try:v,_=a.value_gradient(x,True)
    finally:TimingChain._run=native
    base,_=a.unpack(x);sampling=sampling_from_inference_model(a.model,a.meta);vf=base[a.base_names.index('v_f')];r=route_and_sample(a.model.daily_data,local,vf,sampling);mass=np.zeros_like(r['sample_mass_kg']);network=[r['network_relative_error']]
    stationtags=np.empty((len(mass),nk),dtype=np.float64)
    prediction_error=float(np.max(abs(r['prediction_mg_l']-a.prediction)))
    prediction_tolerance=1e-8*(1+float(np.max(abs(a.prediction))))
    # Full certified station/date support, including evaluation years and
    # spatial holdouts. Only station geometry and calendar dates are read.
    from temporal_model import clean_metadata
    stations=pd.read_parquet(ROOT/'data/station_registry.parquet')
    fullmeta=clean_metadata(pd.DataFrame([dict(s,year=d.year,month=d.month,observation_id=f'ledger_{s["station_key"]}_{d:%Y%m}') for s in stations.to_dict('records') for d in pd.date_range('2016-01-01','2024-12-01',freq='MS')]))
    previous_registry=a.model.registry;previous_cache=a.model._daily_meta_cache;previous_hash=a.model.design['observation_registry_hash']
    try:
        a.model.registry={oid:dict(day_weights=None,excluded=False) for oid in fullmeta.observation_id};a.model._daily_meta_cache={}
        a.model.design['observation_registry_hash']=hashlib.sha256(json.dumps(a.model.registry,sort_keys=True).encode()).hexdigest()
        sampling=sampling_from_inference_model(a.model,fullmeta)
    finally:
        a.model.registry=previous_registry;a.model._daily_meta_cache=previous_cache;a.model.design['observation_registry_hash']=previous_hash
    r=route_and_sample(a.model.daily_data,local,vf,sampling);network.append(r['network_relative_error']);mass=np.zeros_like(r['sample_mass_kg']);stationtags=np.empty((len(mass),nk),dtype=np.float64)
    for k in range(nk):
        z=route_and_sample(a.model.daily_data,localtags[:,:,k],vf,sampling);network.append(z['network_relative_error']);mass+=z['sample_mass_kg'];stationtags[:,k]=z['sample_mass_kg']
    delta=float(np.max(abs(mass-r['sample_mass_kg'])));scale=float(np.max(abs(r['sample_mass_kg'])));rel=delta/scale if scale else 0. if delta==0 else float('inf')
    passed=max(maxima.values())<=1e-6 and max(network+[rel])<=1e-10 and prediction_error<=prediction_tolerance and min(r['minimum_state_kg'] for r in rows)>=-1e-7 and min(r['minimum_label_kg'] for r in rows)>=-1e-7 and all(r['plant_activity']['uptake_constraints_passed'] for r in rows)
    np.savez_compressed(a.folder/'full_history_physical_source_ledger.npz',local_kg_day=local,local_fast_kg_day=localfast,local_slow_kg_day=localslow,local_source_kg_day=localtags,land_stocks_kg=stocks,land_unmet_plant_kg_day=unmet,land_realized_plant_kg_day=realized,external_by_source_kg_day=external,final_label_stocks=tag_state,final_label_compensation=tag_corr)
    c,records,_,_,_=sampling;record_index=records.numpy()
    np.savez_compressed(a.folder/'station_source_ledger.npz',mass_by_source_kg=stationtags,total_mass_kg=r['sample_mass_kg'],water_m3=r['sample_water_m3'],day_index=c['ti'].numpy(),station_key=fullmeta.station_key.to_numpy()[record_index].astype(str),observation_index=record_index,labels=np.asarray(a.inputs.labels,dtype=str))
    receipt=dict(passed=bool(passed),objective=v,maxima=maxima,network_relative=network,station_source_relative=rel,independent_routing_prediction_error_mg_l=prediction_error,independent_routing_prediction_tolerance_mg_l=prediction_tolerance,year_ledger=rows,labels=a.inputs.labels,history=[1961,2024],reaches=230,land_slots=13,
        parameter_vector_sha256=hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest(),
        full_history_ledger_sha256=sha(a.folder/'full_history_physical_source_ledger.npz'),
        station_source_ledger_sha256=sha(a.folder/'station_source_ledger.npz'))
    write_json(a.folder/'physical_ledger.json',receipt);return receipt
