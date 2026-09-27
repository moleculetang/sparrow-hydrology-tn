"""Full 1961–2024 LAND1→frozen H1 routing→OU readout, synthetic only."""
from pathlib import Path
import sys,gc,json,time,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha,dispatch_allowed
configure()
import numpy as np,pandas as pd,torch
from d29_platform.legacy import build_legacy,SNAP
from d29_platform.coupling import response_mapping,route_and_sample,sampling_from_inference_model
from d29_platform.land1 import run_land1,land1_adjoint,hazard_to_probability,PROBABILITY_NAMES

OUT=ROOT/'outputs/chain';OUT.mkdir(parents=True,exist_ok=True)

def main():
    implementation={str(p.relative_to(ROOT)):sha(p) for p in (
        Path(__file__),ROOT/'d29_platform/land1.py',ROOT/'d29_platform/coupling.py',
        ROOT/'d29_platform/legacy.py',ROOT/'d29_platform/runtime.py')}
    ok,resource=dispatch_allowed(reserve_bytes=8_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_GATE_PAUSED '+str(resource))
    start=time.monotonic();m,x,a=build_legacy('F23','U',inference_only=True)
    retired={'log_tau_mineral_days'}|{f'gamma_lifetime_{i}' for i in range(7)}
    named={n:float(v) for n,v in zip(m.names,x) if n not in retired|{'log_source_correction_0'}}
    h,f=response_mapping(m,named)
    with torch.no_grad():oh,_,of,_=m.flux_parameters(torch.tensor(x))
    mapping_error=max(float(abs(h-oh.numpy()).max()),float(abs(f-of.numpy()).max()))
    del oh,of
    T,R=h.shape;L=2;area=m.data.area_ha[:,None]*np.array([.45,.55])[None,:]
    # Explicit test numbers, not inferred source or scientific priors.
    initial=np.broadcast_to(np.array([.05,.4,1.,.02,.03]),(R,L,5)).copy()*area[...,None]
    smooth=np.empty((T,R,L,4),dtype=np.float64)
    for k,v in enumerate((.0001,.0015,.0005,.002)):smooth[...,k]=area[None,:,:]*v
    pulse=smooth.copy();pulse[:,:,:,1]=0.;pulse[:,:,:,3]=0.
    for year in range(1961,2025):
        idx=np.flatnonzero(m.data.dates.year==year);doy=m.data.dates[idx].dayofyear.to_numpy()
        for k,days in ((1,(95,235)),(3,(75,145,225))):
            use=idx[np.isin(doy,days)]
            pulse[use,:,:,k]=smooth[idx,:,:,k].sum(0)/len(use)
    source_hash={}
    for n,arr in [('pulse',pulse),('smooth',smooth),('initial',initial)]:
        import hashlib
        source_hash[n]=hashlib.sha256(memoryview(np.ascontiguousarray(arr))).hexdigest()
    write_json(OUT/'synthetic_identity.json',{'source_hashes':source_hash,'seed':1729,'dates':['1961-01-01','2024-12-31'],
         'years':64,'reaches':R,'lands':L,'source_semantics':'fictional external entries: plant,active organic,protected organic,mineral',
         'uses_tn':False,'assumptions':{'land_fractions':[.45,.55],'daily_entry_intensities':[.0001,.0015,.0005,.002],
         'mineralize_active':.008,'mineralize_protected':.0001,'available_loss':.001}})
    target=np.broadcast_to(area[None,:,:]*.05,(T,R,L))
    outflows=np.broadcast_to(area[None,:,:,None]*np.array([.00015,.00005,.00002]),(T,R,L,3))
    pmob,dpdh=hazard_to_probability(h)
    probs={'mineralize_active':.008,'mineralize_protected':.0001,'mobilize':pmob[:,:,None],
           'available_loss':.001,'fast_fraction':f[:,:,None],'lower_release':m.data.lower_release[:,:,None]}
    spec=dict(initial=initial,sources=pulse,plant_target=target,plant_outflows=outflows,probabilities=probs)
    # Only label-free calendar metadata and read-count registry reach readout.
    meta=pd.read_parquet(SNAP/'data/prediction_calendar.parquet')
    meta=meta[meta.year.between(2021,2024)].reset_index(drop=True)
    m.registry=json.loads((SNAP/'data/prediction_registry.json').read_text(encoding='utf-8'))['records']
    m._daily_meta_cache={};sampling=sampling_from_inference_model(m,meta)
    base=run_land1(**spec);local=base.fluxes[:,:,:,:2].sum(axis=(2,3))
    coupled=route_and_sample(m.data,local,x[2],sampling)
    c,rec,w,n,operator=sampling
    # Independent read-count arithmetic, no reuse of aggregate_daily.
    mass=coupled['sample_mass_kg'];water=coupled['sample_water_m3'];r=rec.numpy();weights=w.numpy()
    daily=1000*mass/water
    independent=np.bincount(r,weights=weights*daily,minlength=n)/np.bincount(r,weights=weights,minlength=n)
    sampling_error=float(abs(independent-coupled['prediction_mg_l']).max())
    station=pd.DataFrame({'station_key':meta.station_key.to_numpy()[r],'date':m.data.dates[c['ti'].numpy()],
              'read_count':weights,'pulse_mg_l':daily,'mass_kg':mass,'water_m3':water})
    # Adjoint through the actual frozen river and OU operator, then LAND1.
    from routing import RiverN,boundary_mass
    lt=torch.tensor(local,requires_grad=True)
    vt=torch.tensor(float(x[2]),requires_grad=True)
    ii,oo,rr=RiverN.apply(lt,vt,m.daily_data,'monthly',True)
    mm=boundary_mass(ii,oo,rr,lt,vt,c)
    coeff=np.sin(np.arange(len(mass))*.03)/len(mass)
    j=torch.sum(mm/c['water']*1000*torch.tensor(coeff));j.backward()
    gf=np.zeros_like(base.fluxes)
    gf[:,:,:,0]=lt.grad.numpy()[:,:,None];gf[:,:,:,1]=lt.grad.numpy()[:,:,None]
    ad=land1_adjoint(base,grad_fluxes=gf)
    directions={'source_scale':float(np.sum(ad['sources']*pulse)),
                'initial_scale':float(np.sum(ad['initial']*initial)),
                'loss_scale':float(np.sum(ad['probabilities']['available_loss'])*.001),
                'target_scale':float(np.sum(ad['plant_target']*target)),
                'outflow_scale':float(np.sum(ad['plant_outflows']*outflows))}
    named_t={k:torch.tensor(v,requires_grad=True) for k,v in named.items()}
    hh,ff=response_mapping(m,named_t,as_numpy=False)
    gh=torch.tensor(ad['probabilities']['mobilize'].sum(axis=2)*dpdh)
    gff=torch.tensor(ad['probabilities']['fast_fraction'].sum(axis=2))
    (torch.sum(hh*gh)+torch.sum(ff*gff)).backward()
    directions['log_alpha_contact']=float(named_t['log_alpha_contact'].grad)
    directions['v_f']=float(vt.grad)
    del named_t,hh,ff,gh,gff,vt
    del ad,gf,lt,ii,oo,rr,mm,j;gc.collect()
    def objective(q,vf=None):
        b=run_land1(**q,keep_history=False)
        v=route_and_sample(m.data,b.fluxes[:,:,:,:2].sum(axis=(2,3)),x[2] if vf is None else vf,sampling)
        return float(np.dot(1000*v['sample_mass_kg']/v['sample_water_m3'],coeff))
    checks=[]
    for key,analytic in directions.items():
        for step in (1e-4,1e-5):
            vals=[]
            for sign in (-1,1):
                q=dict(spec);scale=1+sign*step;vf=None
                if key=='source_scale':q['sources']=pulse*scale
                elif key=='initial_scale':q['initial']=initial*scale
                elif key=='target_scale':q['plant_target']=target*scale
                elif key=='outflow_scale':q['plant_outflows']=outflows*scale
                elif key=='loss_scale':q['probabilities']={**probs,'available_loss':.001*scale}
                elif key=='v_f':vf=x[2]+sign*step
                else:
                    hn,fn=response_mapping(m,{**named,'log_alpha_contact':named['log_alpha_contact']+sign*step})
                    pn,_=hazard_to_probability(hn)
                    q['probabilities']={**probs,'mobilize':pn[:,:,None],'fast_fraction':fn[:,:,None]}
                vals.append(objective(q,vf))
            fd=(vals[1]-vals[0])/(2*step)
            checks.append({'direction':key,'step':step,'analytic':analytic,'finite_difference':fd,
                           'absolute_error':abs(fd-analytic),'passed':abs(fd-analytic)<=1e-6*(1+abs(analytic))})
        print('full-chain gradient',key,checks[-1]['absolute_error'],flush=True)
    smooth_result=run_land1(**{**spec,'sources':smooth},keep_history=False)
    slocal=smooth_result.fluxes[:,:,:,:2].sum(axis=(2,3))
    sc=route_and_sample(m.data,slocal,x[2],sampling)
    station['smooth_mg_l']=1000*sc['sample_mass_kg']/sc['sample_water_m3']
    station.to_parquet(OUT/'synthetic_station_days.parquet',index=False)
    # Authoritative weighted station and paired metrics are generated by
    # scripts/evaluate_chain.py.
    annual=[]
    for year in range(1961,2025):
        z=m.data.dates.year==year
        for rid in range(R):
            annual.append({'year':year,'reach_id':rid+1,'source_kg':float(pulse[z,rid].sum()),
              **{n:float(base.fluxes[z,rid,:,k].sum()) for k,n in enumerate(('fast','slow','loss','plant_export'))},
              'ending_inventory_kg':float(base.states[np.flatnonzero(z)[-1]+1,rid].sum())})
    pd.DataFrame(annual).to_parquet(OUT/'annual_reach_ledger.parquet',index=False)
    layers=[]
    for n,p,s in [('source',pulse.sum(axis=(2,3)),smooth.sum(axis=(2,3))),('local',local,slocal),
                   ('river',coupled['routing']['official'],sc['routing']['official'])]:
        pv=p.std(0);sv=s.std(0);valid=(np.ptp(s,axis=0)>0)&(sv>0)
        ratio=np.divide(pv,sv,out=np.full_like(pv,np.nan),where=valid)
        layers.append({'layer':n,'median_pulse_smooth_sd_ratio':float(np.nanmedian(ratio)) if valid.any() else None,
                       'undefined_reference_constant_reaches':int((~valid).sum()),
                       'NSE':'not applicable: flux attenuation, not concentration prediction'})
    write_json(OUT/'layers.json',layers)
    receipt={'days':T,'reaches':R,'land_classes':L,'mapping_max_error':mapping_error,'sampling_max_error':sampling_error,
        'local_balance_kg':base.max_local_balance_kg,'network_relative_error':coupled['network_relative_error'],
        'gradient_directions':checks,'retired_parameters':sorted(retired),'reads_tn':False,
        'synthetic_only':True,'elapsed_seconds':time.monotonic()-start,'resource_start':resource,
        'implementation_sha256':implementation,
        'implementation_unchanged_during_run':all(sha(ROOT/p)==v for p,v in implementation.items())}
    receipt['passed']=bool(mapping_error<=1e-8 and sampling_error<=1e-8 and base.max_local_balance_kg<=1e-6 and
                   coupled['network_relative_error']<=1e-10 and all(c0['passed'] for c0 in checks) and receipt['implementation_unchanged_during_run'])
    write_json(OUT/'receipt.json',receipt);print('CHAIN',receipt['passed'],flush=True)
    if not receipt['passed']:raise AssertionError('FULL_CHAIN_ACCEPTANCE')

if __name__=='__main__':main()
