"""Full-history U core chained to the new train-only observation objective."""
import json
import time
import numpy as np
import pandas as pd
from d29_platform.runtime import ROOT,write_json,sha
from d29_platform.legacy import bootstrap,SNAP
from .objective import StrategyObjective


def fold_design(cm,data,metadata):
    # Same feature formula as campaign_model.build_design, generalized training
    # years. No TN enters the feature design, and held/buffer sites are absent.
    design=cm.fit_design(data,metadata)
    rr=np.unique(metadata.reach_id).astype(int)-1
    years=sorted(map(int,metadata.year.unique()));days=np.isin(data.dates.year,years)
    q=np.log1p((data.fast_water+data.percolation*data.area_ha[None,:]*10)/(10*data.area_ha[None,:]))
    T=data.temperature/10;W=data.soil_wetness;delta=cm.wetting(W)
    raw=np.stack([q,q*q,T,W,delta,delta*W,q*W,T*W],-1)
    ref=raw[days][:,rr].reshape(-1,8);mu=ref.mean(0);sd=np.maximum(ref.std(0),1e-8);basis=(ref-mu)/sd
    st=(np.clip(data.static_raw,design['low'],design['high'])-design['mean'])/design['sd']
    prods=np.stack([st[:,i]*st[:,j] for i in range(7) for j in range(i,7)],-1);center=prods[rr].mean(0)
    _,sv,vt=np.linalg.svd(prods[rr]-center,full_matrices=False);pcs=(prods-center)@vt[:3].T
    gate={'mean':float(data.bfi[rr].mean()),'sd':float(data.bfi[rr].std()),'years':[1991,2020],'training_global_reaches':[int(data.global_reach_ids[i]) for i in rr]}
    if gate['sd']<=1e-12:raise ValueError('DEGENERATE_GATE')
    pi=np.where(data.bfi_water_positive,(1+np.tanh((data.bfi-gate['mean'])/gate['sd']))/2,.5)
    design.update(extra_mean=mu.tolist(),extra_sd=sd.tolist(),scientific=dict(kind='D29',product_center=center.tolist(),components=vt[:3].tolist(),singular_values=sv.tolist(),dynamic_norm=float(np.sqrt(np.mean(np.sum(basis*basis,-1)))),spatial_norm=max(float(np.sqrt(np.mean(np.sum(pcs[rr]**2,-1)))),1e-12),prior_policy='D29 original fixed 17 normalization'),hc2_gate=gate,endpoint_pi_mean=float(pi[rr].mean()),operator_id='OU',support_hash=sha(SNAP/'data/spatial_support.json'),scenario_id='uniform_uniform',observation_operator='MATCH')
    return design


class UTraining:
    def __init__(self,job):
        cm=bootstrap();self.cm=cm;self.job=job
        m=pd.read_parquet(ROOT/'data/monthly_tn.parquet');h=pd.read_parquet(ROOT/'data/hf_daily.parquet')
        excluded=[]
        if job['space_block'] is not None:
            blocks=json.loads((SNAP/'data/spatial_blocks.json').read_text())
            key=str(job['space_block']).removeprefix('S');b=blocks[key]
            excluded=b['held_stations']+b['buffer_stations']
        end=job['train_end']
        self.objective=StrategyObjective(m[m.year.le(end)],h[h.date.dt.year.le(end)],job['strategy'],end,excluded)
        rows=self.objective.rows.copy();station=pd.read_parquet(ROOT/'data/station_registry.parquet')
        fields=['station_key','station','cohort','reach_id','global_reach_id','station_type','downstream_fraction_on_reach','reservoir_index']
        rows=rows.drop(columns=[c for c in fields if c!='station_key' and c in rows]).merge(station[fields],on='station_key',how='left',validate='many_to_one')
        assert rows.reach_id.notna().all()
        rows['observation_id']=[f"{job['id']}_{i}" for i in range(len(rows))]
        rows['day_index']=np.where(rows.kind.eq('HF'),(rows.date-pd.Timestamp('1961-01-01')).dt.days,-1)
        from temporal_model import clean_metadata
        self.meta=clean_metadata(rows)
        self.data=cm.load_data('FULL24C',verify=True)
        namespace=job.get('output_namespace','jobs')
        if namespace not in ('jobs','jobs_numerical_v2'):
            raise ValueError('UNREGISTERED_OUTPUT_NAMESPACE')
        folder=ROOT/'outputs'/namespace/job['id'];folder.mkdir(parents=True,exist_ok=True);self.folder=folder
        reg=folder/'observation_registry.json'
        write_json(reg,{'records':{oid:{'day_weights':None,'excluded':False} for oid in self.meta.observation_id}})
        # Freeze physical feature coordinates within each fold/space support.
        # Historical TN changes observation constraints, not the process map or
        # its effective priors. The common H1 reference uses 2021..train_end,
        # available to every strategy and excluding held/buffer stations.
        self.design=fold_design(cm,self.data,self.meta[self.meta.year.ge(2021)])
        self.design['physical_feature_reference']='common across T0/T1/T2: 2021 through fold training cutoff, admitted training reaches only'
        self.design.update(observation_registry_file=str(reg),observation_registry_hash=sha(reg))
        write_json(folder/'design.json',self.design)
        write_json(folder/'objective_identity.json',{'strategy':job['strategy'],'train_end':end,'excluded':excluded,'long_sites':self.objective.long_sites,'scales':self.objective.scales,'block_weights':[.4,.4,.2] if job['strategy']=='T2' else [.8,.2]})
        self.model=cm.make_model(self.data,None,'SOURCE_UNIFIED',self.design)
        # Inference construction omits native objective state. This count cancels
        # algebraically in Matched.prior, preserving the fixed 17 normalization.
        self.model.nstation=int(rows.station_key.nunique())
        self.bounds=np.asarray(self.model.bounds);self.initial=self.model.initial(job['entry'])
        self.calls=0

    def value_gradient(self,x):
        import torch
        start=time.monotonic();t=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        p=self.model.tensor_predict(t,self.meta)
        value,dp,terms,_=self.objective.evaluate(p.detach().numpy())
        prior=self.model.prior(t);R=.5*torch.dot(prior,prior)
        torch.autograd.backward([p,R],[torch.tensor(dp),None])
        self.calls+=1;self.last={'objective':float(value+R.detach()),'data_terms':terms,'prior':float(R.detach()),'seconds':time.monotonic()-start,'calls':self.calls}
        return self.last['objective'],t.grad.detach().numpy().copy()
