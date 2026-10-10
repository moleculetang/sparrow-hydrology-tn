"""Known-input experiments; process parameters frozen, scalar fit train-only."""
import argparse,time,gc
import native_runtime as rt
from campaign_model import *
from closures import scan
from routing import route,boundary_mass
from synthetic_inputs import true_input,version,MODES
from scipy.optimize import minimize_scalar
ROOT=RUN/'outputs/synthetic';ROOT.mkdir(exist_ok=True)

class FrozenSimulator:
    def __init__(self,structure):
        tag='F24_U_P_s1' if structure=='U' else 'F24_L3_P_s0'
        job=next(j for j in rt.read(RUN/'configs/jobs.json') if j['tag']==tag)
        # Freeze to the declared historical legal anchor, no heldout-driven selection.
        old=Path(job['reuse_from']);record=rt.read(old/'model.json');self.x=np.array(record['parameters']);self.model=for_job(job);m=self.model
        m.design=dict(m.design,observation_registry_file='data/prediction_registry.json',observation_registry_hash=rt.sha(RUN/'data/prediction_registry.json'))
        m.registry=rt.read(RUN/'data/prediction_registry.json')['records'];m._daily_meta_cache={}
        obs=pd.read_parquet(RUN/'data/heldout_labels/events/observed_days.parquet')[['station_key','date']].drop_duplicates()
        meta=pd.read_parquet(RUN/'data/prediction_calendar.parquet');meta=meta[meta.year.between(2021,2024)&meta.station_key.isin(obs.station_key.unique())].copy()
        self.meta=meta;self.c,self.records,self.weights=m.daily_metadata(meta)
        self.support=pd.DataFrame(dict(station_key=meta.station_key.to_numpy()[self.records.numpy()],date=m.data.dates[self.c['ti'].numpy()]))
        self.support=self.support.merge(obs.assign(observed=True),how='left',on=['station_key','date'],validate='one_to_one');self.mask=self.support.observed.fillna(False).to_numpy(bool)
        self.years=self.support.date.dt.year.to_numpy();self.c0=float(np.exp(self.x[30]));self.calls=0
        with torch.no_grad():self.h,self.s,self.f,_=[np.ascontiguousarray(v.numpy()) for v in m.flux_parameters(torch.tensor(self.x))]
        self.support.to_parquet(ROOT/'synthetic_date_support.parquet',index=False)
        rt.write(ROOT/f'{structure}_generator.json',dict(model=str(old/'model.json'),sha256=rt.sha(old/'model.json'),parameters=self.x.tolist(),used_TN_values=False,reference='F24 historical training-only selected P anchor'))
    def forward(self,source,c=None,ledger=False):
        m=self.model;d=m.data;c=self.c0 if c is None else c;self.calls+=1
        fast,slow,a,p=scan(self.h,self.s,self.f,np.ones(len(d.area_ha)),d.lower_release,source*c,m.demand,False)
        local=fast+slow;r=route(m.daily_data,local,vf=float(self.x[2]))
        with torch.no_grad():mass=boundary_mass(torch.tensor(r['inlet']),torch.tensor(r['official']),torch.tensor(r['releases']),torch.tensor(local),torch.tensor(self.x[2]),self.c).numpy()
        prediction=1000*mass/self.c['water'].numpy()
        if not ledger:return prediction
        M=a*(1-p)*self.s;L=np.cumsum(a*p*(1-self.f)-slow,axis=0);uptake=np.minimum(np.vstack([np.zeros_like(M[:1]),M[:-1]])+source*c,m.demand);loss=a*(1-p)*(1-self.s)
        balance=source*c-uptake-loss-local-np.diff(M+L,axis=0,prepend=np.zeros_like(M[:1]))
        net=local.sum()-r['channel_removed'].sum()-r['terminal'].sum()-r['stocks'][-1].sum()
        assert np.max(abs(balance))<=1e-6 and abs(net)<=max(1.,local.sum())*1e-10
        return prediction,dict(fast=fast,slow=slow,M=M,L=L,uptake=uptake,loss=loss,station_mass=mass,station_water=self.c['water'].numpy(),terminal=r['terminal'],channel_loss=r['channel_removed'],reservoir_stocks=r['stocks'],network_error=net,local_error=float(abs(balance).max()))

def metrics(y,p):
    error=p-y;v=np.var(y)
    return dict(rmse=float(np.sqrt(np.mean(error**2))),bias=float(error.mean()),nse=float(1-np.mean(error**2)/v) if v>0 else None,std_ratio=float(np.std(p)/np.std(y)) if v>0 else None,n=len(y))

def main(structure,seed):
    path=ROOT/f'{structure}_{seed}';path.mkdir(exist_ok=True);sim=FrozenSimulator(structure);d=sim.model.data
    truth,closure,factor=true_input(d,seed);raw=truth.sum(-1);area=pd.read_parquet(RUN/'data/daily_inputs/harvested_area_ha.parquet').to_numpy()
    if not (path/'fabricated_true_daily_sources.npy').exists():np.save(path/'fabricated_true_daily_sources.npy',truth)
    target,ledger=sim.forward(raw,ledger=True);np.savez_compressed(path/'truth_daily_station.npz',concentration=target,mass=ledger['station_mass'],water=ledger['station_water'])
    # All reaches, full historical monthly budget plus end states; full station daily truth retained.
    np.savez_compressed(path/'truth_monthly_land_ledger.npz',**{k:(v[d.stops-1] if k in ['M','L'] else d.monthly_sum(v)) for k,v in ledger.items() if k in ['fast','slow','M','L','uptake','loss','terminal']})
    rng=np.random.default_rng(seed+10000);sigma=np.sqrt(np.log1p(.1**2));noise=np.exp(rng.normal(-sigma*sigma/2,sigma,len(target)));noisy=target*noise;np.save(path/'noise_multiplier.npy',noise)
    rt.write(path/'truth_identity.json',dict(fabricated=True,seed=seed,monthly_closure_before_annual_factor=closure,annual_factor='1+0.2*sin(2*pi*(year-1961)/5)',noise='mean-one lognormal CV=0.1',generator_source_c=sim.c0,source_sha256=rt.sha(path/'fabricated_true_daily_sources.npy'),local_error=ledger['local_error'],network_error=float(ledger['network_error']),prior_year_1961='same-year boundary retained; no previous year exists'))
    del ledger;gc.collect();rows=[]
    for mode in MODES:
        done=path/(mode+'_summary.json')
        if done.exists():rows+=rt.read(done)['rows'];continue
        if time.time()>=rt.read(RUN/'work/experiment_clock.json')['training_deadline']:break
        tags=version(d,truth,mode,area);inp=tags.sum(-1);del tags
        uncorrected=sim.forward(inp);saved={'uncorrected':uncorrected,'oracle':target};current=[]
        for fold,train_end,evalyear in [('F23',2022,2023),('F24',2023,2024)]:
            train=sim.mask&(sim.years<=train_end);held=sim.years==evalyear
            assert train.any()
            # Equal station weight within training support, then equal valid days.
            stations=sim.support.station_key.to_numpy();counts=pd.Series(stations[train]).value_counts();weights=np.array([1/counts[s] for s in stations[train]]);weights/=weights.sum()
            for target_name,y in [('clean',target),('noise10',noisy)]:
                cache={};calls=0
                def objective(eta):
                    nonlocal calls
                    key=float(eta)
                    if key in cache:return cache[key][0]
                    if calls>=200:raise RuntimeError('SCALAR_CALL_BUDGET')
                    calls+=1;pred=sim.forward(inp,np.exp(key));j=float(np.sum(weights*(np.log1p(pred[train])-np.log1p(y[train]))**2));cache[key]=(j,pred);return j
                opt=minimize_scalar(objective,bounds=(-math.log(4),math.log(4)),method='bounded',options={'maxiter':190,'xatol':1e-6})
                candidates=[float(opt.x),math.log(sim.c0),-math.log(4),math.log(4)]
                best=min(candidates,key=objective);pred=cache[best][1];saved[f'{fold}_{target_name}_corrected']=pred
                for support,mask in [('complete_daily',held),('observed_dates',held&sim.mask)]:
                    for treatment,prediction in [('uncorrected',uncorrected),('scalar_corrected',pred),('true_input_reference',target)]:
                        current.append(dict(structure=structure,seed=seed,input_mode=mode,fold=fold,target=target_name,support=support,treatment=treatment,c=float(np.exp(best)) if treatment=='scalar_corrected' else sim.c0,scalar_calls=calls,training_objective=cache[best][0],**metrics(y[mask],prediction[mask])))
                rt.log('synthetic_calls.jsonl',dict(structure=structure,seed=seed,mode=mode,fold=fold,target=target_name,calls=calls,c=float(np.exp(best))))
                del cache;gc.collect()
        np.savez_compressed(path/(mode+'_predictions.npz'),**saved);rt.write(done,dict(rows=current,full_history_calls=sim.calls,process=rt.process(os.getpid())));rows+=current
        pd.DataFrame(rows).to_csv(path/'metrics.csv',index=False);print('SYNTHETIC',structure,seed,mode,flush=True)
    rt.write(path/'status.json',dict(status='COMPLETE' if len(list(path.glob('*_summary.json')))==8 else 'BUDGET_INCOMPLETE',scenarios=len(list(path.glob('*_summary.json'))),full_history_calls=sim.calls,process=rt.process(os.getpid())))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('structure',choices=['U','L3']);p.add_argument('seed',type=int);a=p.parse_args();main(a.structure,a.seed)
