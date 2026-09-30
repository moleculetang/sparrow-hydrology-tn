"""Resolve a failed coordinate with finer steps and recorded physical switches.
Not a replacement acceptance scan; never marks an optimizer converged.
"""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed
configure()
import numpy as np
from d29_training.land1_adapter import Land1Training
from d29_training.annual_chain import AnnualChain
from d29_platform.land1 import _transfer

original=AnnualChain._run
baseline={};seen=set();changes=[];mode='baseline'
def inspect(self,year,initial,eta,correction):
    result,mask=original(self,year,initial,eta,correction)
    if year in seen:return result,mask
    seen.add(year);i=result._inputs;nt,nr,nl=i['shape']
    before=result.states[:-1].reshape(nt,nr*nl,5).copy()
    for t in i['transition_days']:before[t]=_transfer(before[t],i['matrices'][i['event'][t]],nr,nl)
    P=before[:,:,0];X=before[:,:,3]+i['sources'][:,:,3]+i['probabilities'][0]*before[:,:,1]+i['probabilities'][1]*before[:,:,2]
    O=i['outflows'].sum(-1);raw=i['target']+O-P-i['sources'][:,:,0];need=np.maximum(raw,0)
    after=np.where((raw>0)&(X>=need),i['target'],(P+i['sources'][:,:,0]-O)+np.minimum(X,need))
    branch=(raw>0).astype(np.uint8)+2*(X<=need).astype(np.uint8)+4*(after<0).astype(np.uint8)
    if mode=='baseline':baseline[year]=branch
    else:
        changed=np.argwhere(branch!=baseline[year])
        if len(changed):
            witnesses=[]
            for t,u in changed[:20]:witnesses.append({'date':str(np.datetime64(f'{year}-01-01')+np.timedelta64(int(t),'D')),'reach':int(u//nl+1),'land':int(u%nl),'old_branch':int(baseline[year][t,u]),'new_branch':int(branch[t,u]),'supply_minus_need':float(X[t,u]-need[t,u]),'raw_need':float(raw[t,u])})
            changes.append({'year':year,'count':len(changed),'witnesses':witnesses})
    return result,mask
AnnualChain._run=inspect
job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']=='LAND1_F23_T0_s0')
ok,res=dispatch_allowed(reserve_bytes=7_000_000_000)
if not ok:raise RuntimeError('RESOURCE_GATE')
a=Land1Training(job,diagnostic=True);x=a.initial.copy();v,g=a.value_gradient(x)
receipt={'coordinate':3,'name':a.names[3],'base_value':v,'analytic':float(g[3]),'checks':[],'not_optimizer_sufficiency':True,'NSE':'not applicable'}
for step in [3e-6,1e-6,3e-7]:
    row={'step':step};values=[]
    for sign in [1,-1]:
        seen.clear();changes.clear();mode='perturbed';point=x.copy();point[3]+=sign*step
        value,_=a.value_gradient(point);values.append(value)
        row['plus' if sign>0 else 'minus']={'value':value,'switches':list(changes)}
        write_json(ROOT/'outputs/land1_contact_branch_pending.json',row)
    fd=(values[0]-values[1])/(2*step);row.update(fd=fd,error=abs(fd-g[3]),passed=bool(abs(fd-g[3])<=1e-6*(1+abs(g[3]))))
    receipt['checks'].append(row);write_json(ROOT/'outputs/land1_contact_branch.json',receipt)
    print(step,row['error'],row['passed'],flush=True)
