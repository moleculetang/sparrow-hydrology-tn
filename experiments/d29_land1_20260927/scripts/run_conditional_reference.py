"""Run the explicitly incomplete-boundary research scenario, never a formal fit."""
from pathlib import Path
import sys,json,time,hashlib,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha,dispatch_allowed
configure()
import numpy as np,pandas as pd
from d29_platform.conditional_inputs import ConditionalInputs
from d29_platform.land1 import run_land1,propagate_source_labels,hazard_to_probability,PlantBudgetInfeasible
from d29_platform.legacy import build_legacy,SNAP
from d29_platform.coupling import response_mapping,route_and_sample,sampling_from_inference_model
POTENTIAL='--potential-activity' in sys.argv
OUT=ROOT/('outputs/supply_limited_reference' if POTENTIAL else 'outputs/conditional_reference');OUT.mkdir(parents=True,exist_ok=True)
def ah(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a))).hexdigest()
def main():
    started=time.monotonic();ok,resource=dispatch_allowed(reserve_bytes=4_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_PAUSE '+str(resource))
    m,x,anchor=build_legacy('F23','U',inference_only=True);inp=ConditionalInputs(m.data.dates);cfg=inp.config
    cfg['plant_activity_mode']='potential_with_shortfall' if POTENTIAL else 'strict_prescribed'
    cfg['activity_semantic_change']='potential variant records all unrealized destinations; does not assert observed yields or NPP were attained'
    paths=[Path(__file__),ROOT/'d29_platform/land1.py',ROOT/'d29_platform/conditional_inputs.py',ROOT/'d29_platform/coupling.py',ROOT/'config/conditional_reference.json',ROOT/'config/mineralization_reference.json']
    identities={str(p):sha(p) for p in paths}
    write_json(OUT/'configuration_frozen.json',dict(implementation_hashes=identities,config=cfg,anchor=anchor,
        no_TN_loaded=True,identity='uncalibrated S1 LAND1 conditional scenario',not_a_holdout_prediction=True))
    retired={'log_tau_mineral_days'}|{f'gamma_lifetime_{i}' for i in range(7)}|{'log_source_correction_0'}
    named={n:float(v) for n,v in zip(m.names,x) if n not in retired};h,f=response_mapping(m,named);pm,_=hazard_to_probability(h)
    mini=json.loads((ROOT/'config/mineralization_reference.json').read_text(encoding='utf-8'))
    ka=-np.expm1(-mini['k_active_per_day']);kp=-np.expm1(-1/(270*365.25))
    initial=inp.initial();taginit=np.zeros((230,13,5,5));taginit[:,:,4,:]=initial
    np.save(OUT/'initial_stocks.npy',initial);np.save(OUT/'annual_physical_area_m2.npy',inp.area)
    pd.DataFrame(inp.npp_donors).to_csv(OUT/'npp_group_donor_cases.csv',index=False)
    local=np.zeros((len(m.data.dates),230));fast=local.copy();slow=local.copy();localtags=np.zeros((len(m.data.dates),230,5))
    ledger=[];annualhashes=[];maxbalance=0.;maxtag=0.;state_snapshots=[];paused=False;restart_error=0.;roundoff_continuations=[];correction=None;tagcorrection=None
    for year in range(1961,2025):
        ok,res=dispatch_allowed(paused=paused,reserve_bytes=2_000_000_000)
        while not ok:
            write_json(OUT/'resource_pause.json',dict(year=year,resources=res));time.sleep(5);paused=True
            ok,res=dispatch_allowed(paused=True,reserve_bytes=2_000_000_000)
        paused=False
        ix=np.flatnonzero(m.data.dates.year==year);postplant=initial[:,:,0].copy()
        if int(ix[0]) in inp.transitions:postplant[:,:12]=np.einsum('ri,rij->rj',postplant[:,:12],inp.transitions[int(ix[0])])
        a=inp.annual(year,postplant);spec={k:a[k] for k in ('sources','plant_target','plant_outflows','transitions')}
        # Passive urban and unknown-surface ledger has no biology or transport.
        active=np.ones((1,1,13));active[:,:,4]=0;active[:,:,12]=0
        probs=dict(mineralize_active=ka*active,mineralize_protected=kp*active,mobilize=pm[ix,:,None]*active,
                   available_loss=cfg['available_loss_probability']*active,fast_fraction=f[ix,:,None],lower_release=m.data.lower_release[ix,:,None]*active)
        hashes={k:ah(a[k]) for k in ('sources','plant_target','plant_outflows','tags')}
        hashes['year']=year;hashes['initial']=ah(initial);annualhashes.append(hashes)
        write_json(OUT/'annual_input_hashes.json',annualhashes)
        try:b=run_land1(initial=initial,probabilities=probs,compensated=True,initial_compensation=correction,plant_activity_mode=cfg['plant_activity_mode'],**spec)
        except PlantBudgetInfeasible as e:
            write_json(OUT/'failed_run.json',dict(status='infeasible_plant_activity_no_stock_clipping',year=year,date=str(m.data.dates[ix[e.day]].date()),reach_id=e.reach+1,land=e.land,shortfall_kg=e.shortfall_kg,completed_years=[r['year'] for r in ledger],config_hash=sha(ROOT/'config/conditional_reference.json')))
            raise
        # Source tags are streamed one year at a time with full state continuity.
        if restart_error>1e-6:roundoff_continuations.append(dict(year=year,previous_checkpoint_label_error_kg=restart_error,acceptance_waived=False,diagnostic_only=True))
        tag=propagate_source_labels(b,tagged_initial=taginit,tagged_sources=a['tags'],labels=inp.labels,record_history=True,diagnostic_restart_error_kg=restart_error,compensated=True,initial_compensation=tagcorrection)
        fast[ix]=b.fluxes[:,:,:,0].sum(2);slow[ix]=b.fluxes[:,:,:,1].sum(2);local[ix]=fast[ix]+slow[ix]
        localtags[ix]=tag.flux_history[:,:,:,:,:2].sum(axis=(2,4))
        maxbalance=max(maxbalance,b.max_local_balance_kg,tag.max_local_balance_kg);maxtag=max(maxtag,tag.max_state_sum_error_kg,tag.max_flux_sum_error_kg)
        row={**a['ledger'],'sources_total_kg':float(a['sources'].sum()),'fast_kg':float(fast[ix].sum()),'slow_kg':float(slow[ix].sum()),
             'land_loss_kg':float(b.fluxes[:,:,:,2].sum()),'plant_export_kg':float(b.fluxes[:,:,:,3].sum()),'start_inventory_kg':float(initial.sum()),'end_inventory_kg':float(b.final.sum()),
             'mass_residual_kg':float(initial.sum()+a['sources'].sum()-b.final.sum()-b.fluxes.sum()),'local_max_error_kg':b.max_local_balance_kg,
             'tag_state_max_error_kg':tag.max_state_sum_error_kg,'tag_flux_max_error_kg':tag.max_flux_sum_error_kg,
             'urban_inventory_kg':float(b.final[:,4].sum()),'quarantine_inventory_kg':float(b.final[:,12].sum())}
        row['unmet_plant_export_kg']=float(b.unmet_plant_outflows[:,:,:,0].sum())
        row['unmet_plant_return_kg']=float(b.unmet_plant_outflows[:,:,:,1:].sum())
        shortage=np.argwhere(b.unmet_plant_outflows.sum(-1)>0)
        if len(shortage):
            si,sr,sl=shortage.T
            pd.DataFrame(dict(date=m.data.dates[ix[si]],reach_id=sr+1,land_index=sl,
                unmet_export_kg=b.unmet_plant_outflows[si,sr,sl,0],unmet_return_active_kg=b.unmet_plant_outflows[si,sr,sl,1],unmet_return_protected_kg=b.unmet_plant_outflows[si,sr,sl,2])).to_parquet(OUT/f'plant_shortfall_{year}.parquet',index=False)
        ledger.append(row);state_snapshots.append(b.final.copy());initial=b.final.copy();taginit=tag.final.copy()
        restart_error=float(abs(taginit.sum(2)-initial).max())
        correction=b.compensation.copy();tagcorrection=tag.compensation.copy()
        np.savez(OUT/'resume_checkpoint.npz',year=year,initial=initial,tagged_initial=taginit,compensation=correction,tag_compensation=tagcorrection)
        pd.DataFrame(ledger).to_csv(OUT/'annual_mass_ledger.csv',index=False)
        print(year,'mass',row['mass_residual_kg'],'local',maxbalance,'tag',maxtag,flush=True)
        del a,spec,b,tag;gc.collect()
    np.save(OUT/'local_fast_kg_day.npy',fast);np.save(OUT/'local_slow_kg_day.npy',slow);np.save(OUT/'local_source_labels_kg_day.npy',localtags)
    np.save(OUT/'annual_end_stocks.npy',np.stack(state_snapshots));np.save(OUT/'final_label_stocks.npy',taginit)
    meta=pd.read_parquet(SNAP/'data/prediction_calendar.parquet');meta=meta[meta.year.between(2021,2024)].reset_index(drop=True)
    m.registry=json.loads((SNAP/'data/prediction_registry.json').read_text(encoding='utf-8'))['records'];m._daily_meta_cache={}
    sampling=sampling_from_inference_model(m,meta);coupled=route_and_sample(m.data,local,x[2],sampling)
    c,rec,w,n,op=sampling;r=rec.numpy();water=coupled['sample_water_m3'];mass=coupled['sample_mass_kg']
    valid=water>0;concentration=np.divide(1000*mass,water,out=np.full_like(mass,np.nan),where=valid)
    pd.DataFrame(dict(station_key=meta.station_key.to_numpy()[r],date=m.data.dates[c['ti'].numpy()],read_count=w.numpy(),candidate=concentration,mass_kg=mass,water_m3=water)).to_parquet(OUT/'station_days.parquet',index=False)
    meta['prediction_mg_l']=coupled['prediction_mg_l'];meta.to_parquet(OUT/'sample_predictions.parquet',index=False)
    receipt=dict(status='completed_conditional_forward',days=len(m.data.dates),reaches=230,land_slots=13,
        local_max_balance_kg=maxbalance,label_max_difference_kg=maxtag,network_relative_error=coupled['network_relative_error'],
        strict_mass_pass=bool(maxbalance<=1e-6 and maxtag<=1e-6 and coupled['network_relative_error']<=1e-10),
        no_TN_loaded=True,full_history_continuity=True,roundoff_diagnostic_continuations=roundoff_continuations,gradient_claim='existing full-history kernel validated; annual streaming here is forward only, no truncated-gradient training',
        implementation_unchanged=all(sha(p)==v for p,v in identities.items()),elapsed_seconds=time.monotonic()-started,
        resources_at_start=resource,formal_fit=False,NSE='computed separately against real observations; physical ledger NSE not applicable')
    write_json(OUT/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
