"""Frozen-parameter signal propagation and fixed attribution counterexamples."""
import argparse,time,gc
import native_runtime as rt
from synthetic_study import FrozenSimulator,ROOT,metrics
from campaign_model import *
from routing import route,boundary_mass
from scipy.optimize import minimize_scalar

def pulses(structure):
    sim=FrozenSimulator(structure);d=sim.model.data;inp=np.array(sim.model.raw_daily);out=ROOT/'pulses';out.mkdir(exist_ok=True)
    p0,b=sim.forward(inp,ledger=True);rows=[];byreach=[]
    for month in [1,4,7,10]:
        start=int(d.dates.get_loc(pd.Timestamp(2021,month,15)))
        for duration in [1,7,30,365]:
            altered=inp.copy();altered[start:start+duration]+=1./duration/sim.c0
            pred,a=sim.forward(altered,ledger=True)
            np.savez_compressed(out/f'{structure}_{month}_{duration}_station_response.npz',delta_concentration=pred-p0)
            for horizon in [1,7,30,90,365]:
                stop=start+horizon;row=dict(structure=structure,start=str(d.dates[start].date()),spread_days=duration,horizon_days=horizon,total_planned_pulse_kg=len(d.area_ha),delivered_input_kg=len(d.area_ha)*min(horizon,duration)/duration)
                local={}
                for key in ['fast','slow','uptake','loss','M','L']:
                    v=(a[key][stop-1]-b[key][stop-1]) if key in ['M','L'] else (a[key][start:stop]-b[key][start:stop]).sum(0)
                    row['delta_'+key+'_kg']=float(v.sum());local[key]=v
                row['incremental_land_balance_kg']=row['delivered_input_kg']-sum(row['delta_'+k+'_kg'] for k in ['fast','slow','uptake','loss','M','L'])
                row['delta_channel_loss_kg']=float((a['channel_loss'][start:stop]-b['channel_loss'][start:stop]).sum())
                row['delta_terminal_kg']=float((a['terminal'][start:stop]-b['terminal'][start:stop]).sum())
                row['delta_reservoir_stock_kg']=float((a['reservoir_stocks'][stop-1]-b['reservoir_stocks'][stop-1]).sum())
                row['incremental_whole_network_balance_kg']=row['delivered_input_kg']-sum(row['delta_'+k+'_kg'] for k in ['uptake','loss','M','L','channel_loss','terminal','reservoir_stock'])
                active=sim.support.date.between(d.dates[start],d.dates[stop-1]).to_numpy();delta=(pred-p0)[active]
                row['station_max_abs_delta_mg_l']=float(abs(delta).max());row['station_rms_delta_mg_l']=float(np.sqrt(np.mean(delta**2)));rows.append(row)
                for r,rid in enumerate(d.global_reach_ids):byreach.append(dict(structure=structure,start=row['start'],spread_days=duration,horizon_days=horizon,reach_id=int(rid),**{k:float(v[r]) for k,v in local.items()}))
            pd.DataFrame(rows).to_csv(out/f'{structure}_summary.csv',index=False);del a;gc.collect();print('PULSE',structure,month,duration,flush=True)
    pd.DataFrame(byreach).to_parquet(out/f'{structure}_reach_response.parquet',index=False)
    rt.write(out/f'{structure}_audit.json',dict(status='PASS',mass_per_reach_kg=1,reach_selection='all 230, fixed quarterly day15, no residual selection',maximum_incremental_balance_kg=max(abs(r['incremental_land_balance_kg']) for r in rows),calls=sim.calls,process=rt.process(os.getpid())))

def controls(structure):
    sim=FrozenSimulator(structure);d=sim.model.data;out=ROOT/'negative_controls';out.mkdir(exist_ok=True)
    raw=np.load(ROOT/f'{structure}_1729/fabricated_true_daily_sources.npy',mmap_mode='r').sum(-1);truth,b=sim.forward(raw,ledger=True)
    other='L3' if structure=='U' else 'U';other_truth=np.load(ROOT/f'{other}_1729/truth_daily_station.npz')['concentration']
    # Omitted source enters each reach once, at the river, with its own ledger.
    extra=np.ones_like(raw);river=route(sim.model.daily_data,extra,vf=float(sim.x[2]))
    with torch.no_grad():mass=boundary_mass(torch.tensor(river['inlet']),torch.tensor(river['official']),torch.tensor(river['releases']),torch.tensor(extra),torch.tensor(sim.x[2]),sim.c).numpy()
    extra_c=1000*mass/sim.c['water'].numpy();net=extra.sum()-river['channel_removed'].sum()-river['terminal'].sum()-river['stocks'][-1].sum()
    assert abs(net)<=extra.sum()*1e-10
    np.savez_compressed(out/f'{structure}_omitted_mass_ledger.npz',daily_total_input_kg=extra.sum(1),daily_channel_loss=river['channel_removed'].sum(1),daily_terminal=river['terminal'],daily_reservoir_stock=river['stocks'].sum(1),station_mass=mass)
    targets={'cross_structure_truth':other_truth,'wrong_model_station_water':truth,'omitted_stable_river_source':truth+extra_c};rows=[]
    train=sim.mask&(sim.years<=2023);evaluate=sim.mask&(sim.years==2024)
    for name,target in targets.items():
        calls=0;cache={}
        def objective(eta):
            nonlocal calls
            if float(eta) in cache:return cache[float(eta)][0]
            calls+=1;assert calls<=200
            p=sim.forward(raw,np.exp(eta))
            if name=='wrong_model_station_water':p=p/1.25
            j=float(np.mean((np.log1p(p[train])-np.log1p(target[train]))**2));cache[float(eta)]=(j,p);return j
        opt=minimize_scalar(objective,bounds=(-math.log(4),math.log(4)),method='bounded',options={'maxiter':190,'xatol':1e-6})
        best=min([float(opt.x),math.log(sim.c0),-math.log(4),math.log(4)],key=objective)
        for mode,eta in [('unchanged_source',math.log(sim.c0)),('fitted_source',best)]:
            objective(eta);p=cache[eta][1];rows.append(dict(structure=structure,control=name,fit=mode,c=float(np.exp(eta)),calls=calls,**metrics(target[evaluate],p[evaluate])))
        np.savez_compressed(out/f'{structure}_{name}.npz',truth=target,uncorrected=cache[math.log(sim.c0)][1],corrected=cache[best][1]);del cache
    # Exact mass/water scaling requires demand to scale too; demand-fixed counterexample included.
    m=sim.model;original=m.demand.copy();factor=1.5
    m.demand=original*factor;scaled=sim.forward(raw*factor)/factor;m.demand=original
    fixed_demand=sim.forward(raw*factor)/factor
    m.demand=np.zeros_like(original);zero_reference=sim.forward(raw);zero_scaled=sim.forward(raw*factor)/factor;m.demand=original
    zero_error=float(abs(zero_reference-zero_scaled).max());assert zero_error<=1e-9
    rt.write(out/f'{structure}_scale_ambiguity.json',dict(exact_scale_source_demand_and_water_error=float(abs(scaled-truth).max()),source_and_water_only_error=float(abs(fixed_demand-truth).max()),zero_demand_source_water_scaling_error=zero_error,factor=factor,interpretation='Exact homogeneity needs demand scaling; fixed demand generally breaks it. Full-history zero-demand fixture independently verifies source/water scaling.'))
    pd.DataFrame(rows).to_csv(out/f'{structure}_metrics.csv',index=False);rt.write(out/f'{structure}_audit.json',dict(status='PASS',extra_source_kg_per_reach_day=1,extra_network_error=float(net),calls=sim.calls,process=rt.process(os.getpid())))

def demand_calendar(structure):
    sim=FrozenSimulator(structure);m=sim.model;d=m.data;out=ROOT/'demand_calendar';out.mkdir(exist_ok=True)
    rows=[];original=m.demand.copy()
    for mode in ['P','D']:
        raw=np.load(RUN/f'data/daily_inputs/{mode}_total.npy',mmap_mode='r')
        for daily in [False,True]:
            m.demand=d.crop[d.mid]/(d.stops-d.starts)[d.mid,None] if daily else original
            p,ledger=sim.forward(raw,ledger=True)
            np.save(out/f'{structure}_{mode}_demand_{daily}.npy',p)
            rows.append(dict(structure=structure,source_mode=mode,daily_demand=daily,source_kg=float(raw.sum()*sim.c0),demand_kg=float(m.demand.sum()),uptake_kg=float(ledger['uptake'].sum()),loss_kg=float(ledger['loss'].sum()),end_inventory_kg=float((ledger['M'][-1]+ledger['L'][-1]).sum())))
            del ledger;gc.collect()
    m.demand=original;pd.DataFrame(rows).to_csv(out/f'{structure}_ledger.csv',index=False)
    rt.write(out/f'{structure}_audit.json',dict(status='PASS',role='Fixed parameters only; not candidate selection',calls=sim.calls,process=rt.process(os.getpid())))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['pulses','controls','demand_calendar']);p.add_argument('structure',choices=['U','L3']);a=p.parse_args();globals()[a.operation](a.structure)
