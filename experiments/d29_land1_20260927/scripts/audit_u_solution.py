"""Independent train-only objective recomputation and block-gradient diagnostics."""
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np
import pandas as pd
import torch
from d29_training.u_adapter import UTraining

def recompute(rows,pred,scales,strategy,long_sites):
    f=rows.copy();f['pred']=pred;terms={}
    sets=[('month',f[f.kind.eq('monthly')],.8)] if strategy!='T2' else [('long_month',f[f.kind.eq('monthly')&f.station_key.isin(long_sites)],.4),('other_month',f[f.kind.eq('monthly')&~f.station_key.isin(long_sites)],.4)]
    sets.append(('HF_anomaly',f[f.kind.eq('HF')],.2))
    for name,table,weight in sets:
        stations=[]
        for station,s in table.groupby('station_key'):
            years=[]
            for _,year in s.groupby('year'):
                months=[]
                for _,month in year.groupby('month'):
                    e=month.pred.to_numpy()-month.tn_mg_l.to_numpy()
                    if name=='HF_anomaly':
                        n=month.read_count.to_numpy();alpha=n/n.sum();e=e-np.dot(alpha,e);loss=float(np.dot(alpha,e*e))
                    else:
                        assert len(e)==1;loss=float(e[0]**2)
                    months.append(loss)
                years.append(math.fsum(months)/len(months))
            kind='HF' if name=='HF_anomaly' else 'monthly'
            stations.append(math.fsum(years)/len(years)/scales[kind]['station_variance'][station])
        terms[name]=.5*weight*math.fsum(stations)/len(stations)
    return terms

job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']==sys.argv[1])
a=UTraining(job);x=np.load(a.folder/'best.npy');v,g=a.value_gradient(x)
from serial_solvers import projected_gradient
with torch.no_grad():p=a.model.tensor_predict(torch.tensor(x),a.meta).numpy();prior=.5*float(torch.sum(a.model.prior(torch.tensor(x))**2))
terms=recompute(a.objective.rows,p,a.objective.scales,job['strategy'],a.objective.long_sites)
independent=math.fsum(terms.values())+prior
receipt={'checkpoint_sha256':sha(a.folder/'best.npy'),'objective':v,'independent_objective':independent,'objective_error':abs(v-independent),'objective_passed':abs(v-independent)<=1e-8*(1+abs(v)),'projected_gradient':float(np.max(abs(projected_gradient(x,g,a.bounds)))),'source_multiplier':float(np.exp(x[30])),'NSE':'not applicable to numerical audit','data_terms':terms,'prior':prior}
directions=[]
rng=np.random.default_rng(1729)
for k in range(3):
    d=rng.normal(size=len(x));d/=np.linalg.norm(d)
    # Check the feasible tangent: fixed active-bound coordinates are zero.
    d[(x-a.bounds[:,0]<1e-3)|(a.bounds[:,1]-x<1e-3)]=0
    samples=[]
    for h in [1e-4,3e-5,1e-5]:
        vp,_=a.value_gradient(x+h*d);vm,_=a.value_gradient(x-h*d)
        fd=(vp-vm)/(2*h);samples.append({'step':h,'fd':fd,'analytic':float(g@d),'error':abs(fd-g@d),'passed':bool(abs(fd-g@d)<=1e-6*(1+abs(g@d)))})
    directions.append(samples)
receipt['direction_checks']=directions
receipt['numerically_sufficient']=receipt['projected_gradient']<=1e-5
write_json(a.folder/'independent_audit.json',receipt);print(json.dumps(receipt),flush=True)
