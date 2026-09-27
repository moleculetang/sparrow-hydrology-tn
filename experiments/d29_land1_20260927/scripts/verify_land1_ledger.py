"""Full-history physical and provenance ledger at a finalized LAND1 checkpoint.

Uses the exact accepted annual input builder and never truncates warmup. Labels
scale only external sources, keeping initial stocks and internal transfers intact.
"""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha,dispatch_allowed
configure()
import numpy as np
import pandas as pd
import d29_training.annual_chain as annual
from d29_platform import precision_candidate as precise
annual.run_land1=precise.run_land1;annual.land1_adjoint=precise.land1_adjoint
from d29_platform.mixture_tags_candidate import propagate_source_labels
from d29_training.land1_adapter import Land1Training
from d29_platform.coupling import route_and_sample,sampling_from_inference_model


def main(job_id):
    ok,_=dispatch_allowed(reserve_bytes=9_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_GATE')
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']==job_id)
    folder=ROOT/'outputs/jobs'/job_id
    status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    if status['status'] in ('running','resource_checkpoint','continuing_same_path_zero_ftol'):raise RuntimeError('LIVE_CHECKPOINT')
    a=Land1Training(job);x=np.load(folder/'best.npy');labels=a.inputs.labels;nk=len(labels)
    initial=a.inputs.initial();tag_initial=np.zeros((*initial.shape[:2],nk,5));tag_initial[:,:,nk-1,:]=initial
    tag_correction=None;rows=[];snapshots=[];tag_min=0.;physical_min=0.;flux_min=0.
    n=len(a.data.dates);local=np.zeros((n,230));localtags=np.zeros((n,230,nk))
    maxima=dict(local=0.,source_balance=0.,source_sum=0.)
    native=annual.AnnualChain._run
    def checked(chain,year,state,eta,correction):
        nonlocal tag_initial,tag_correction,tag_min,physical_min,flux_min
        result,mask=native(chain,year,state,eta,correction)
        ix=np.flatnonzero(a.data.dates.year==year)
        post=state[:,:,0].copy()
        if int(ix[0]) in a.inputs.transitions:
            post[:,:12]=np.einsum('ri,rij->rj',post[:,:12],a.inputs.transitions[int(ix[0])])
        source=a.inputs.annual(year,post)['tags']*np.exp(eta)
        tag=propagate_source_labels(result,tagged_initial=tag_initial,tagged_sources=source,
            labels=labels,record_history=True,compensated=True,initial_compensation=tag_correction)
        local[ix]=result.fluxes[:,:,:,:2].sum(axis=(2,3))
        localtags[ix]=tag.flux_history[:,:,:,:,:2].sum(axis=(2,4))
        represented=result.states.copy()
        represented[:,:,:,1:]-=result._inputs['compensation_history'].reshape(len(ix)+1,230,13,4)
        physical_min=min(physical_min,float(represented.min()))
        # Histories retain high components; final represented low components
        # are checked explicitly at every annual restart.
        tag_end=tag.final.copy();tag_end[:,:,:,1:]-=tag.compensation
        tag_min=min(tag_min,float(tag.state_history.min()),float(tag_end.min()))
        flux_min=min(flux_min,float(result.fluxes.min()),float(tag.flux_history.min()))
        maxima['local']=max(maxima['local'],result.max_local_balance_kg)
        maxima['source_balance']=max(maxima['source_balance'],tag.max_local_balance_kg)
        maxima['source_sum']=max(maxima['source_sum'],tag.max_state_sum_error_kg,tag.max_flux_sum_error_kg)
        rows.append(dict(year=year,external_kg=float(source.sum()),fast_kg=float(result.fluxes[:,:,:,0].sum()),
            slow_kg=float(result.fluxes[:,:,:,1].sum()),loss_kg=float(result.fluxes[:,:,:,2].sum()),
            plant_export_kg=float(result.fluxes[:,:,:,3].sum()),end_inventory_kg=float(represented[-1].sum()),
            unmet_plant_activity_kg=float(result.unmet_plant_outflows.sum()),**maxima))
        snapshots.append(represented[-1].copy())
        tag_initial=tag.final.copy();tag_correction=tag.compensation.copy()
        pd.DataFrame(rows).to_csv(folder/'annual_physical_source_ledger.csv',index=False)
        return result,mask
    annual.AnnualChain._run=checked
    started=time.monotonic()
    try:value,_=a.value_gradient(x,forward_only=True)
    finally:annual.AnnualChain._run=native
    if abs(value-status['best_objective'])>1e-8*(1+abs(value)):raise RuntimeError('CHECKPOINT_OBJECTIVE_CHANGED')
    sampling=sampling_from_inference_model(a.model,a.meta);vf=x[a.names.index('v_f')]
    routed=route_and_sample(a.data,local,vf,sampling);network=routed['network_relative_error']
    tag_mass=np.zeros_like(routed['sample_mass_kg']);network_tag_errors=[]
    for k in range(nk):
        r=route_and_sample(a.data,localtags[:,:,k],vf,sampling)
        tag_mass+=r['sample_mass_kg'];network_tag_errors.append(float(r['network_relative_error']))
    station_error=float(np.max(abs(tag_mass-routed['sample_mass_kg'])))
    # Network/source station summation uses the registered flux-relative
    # tolerance, in addition to absolute local/source ledger tolerances.
    station_scale=float(np.max(abs(routed['sample_mass_kg'])))
    station_relative=station_error/station_scale if station_scale>0 else (0. if station_error==0 else float('inf'))
    passed=max(maxima.values())<=1e-6 and min(physical_min,tag_min,flux_min)>=-1e-7 and max([network,station_relative,*network_tag_errors])<=1e-10
    np.savez_compressed(folder/'full_history_land_ledger.npz',local_kg_day=local,local_source_kg_day=localtags,
        annual_end_stocks=np.stack(snapshots),final_label_stocks=tag_initial,final_label_compensation=tag_correction)
    write_json(folder/'physical_ledger.json',dict(passed=bool(passed),parameter_sha256=sha(folder/'best.npy'),
        local_balance_max_kg=maxima['local'],source_balance_max_kg=maxima['source_balance'],source_sum_max_error_kg=maxima['source_sum'],
        network_relative_error=float(network),source_network_relative_errors=network_tag_errors,
        station_source_mass_sum_error_kg=station_error,station_source_relative_error=station_relative,
        minimum_physical_kg=physical_min,minimum_source_high_or_year_end_represented_kg=tag_min,minimum_flux_kg=flux_min,
        labels=list(labels),days=n,reaches=230,land_slots=13,full_history_continuity=True,
        elapsed_seconds=time.monotonic()-started,forward_calls=1,NSE='not applicable to physical/source ledger'))
    if not passed:raise RuntimeError('FITTED_LAND1_LEDGER_FAILED')


if __name__=='__main__':main(sys.argv[1])
