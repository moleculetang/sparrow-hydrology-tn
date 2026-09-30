import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np
import torch
from d29_training.u_adapter import UTraining
from d29_training.experiment_context import ExperimentContext

job_id=sys.argv[1] if len(sys.argv)>1 else 'U_F23_T0_s0'
if len(sys.argv)>1:
    context=ExperimentContext.load();job=context.job(job_id);context.check_worker(job_id)
else:job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']==job_id)
a=UTraining(job);m=a.model
x=np.load(a.folder/'best.npy') if len(sys.argv)>1 else a.initial
v=m.ledger(x)
from balanced_tags import balanced_scan
with torch.no_grad():h,s,f,k=[z.numpy() for z in m.flux_parameters(torch.tensor(x))]
inp=np.zeros_like(h);inp[m.data.starts]=m.corrected_source(torch.tensor(x)).detach().numpy()
tags=np.zeros((*h.shape,4));tags[m.data.starts]=m.original_tags*np.exp(x[30])
tag=balanced_scan(h,s,f,m.data.lower_release,tags,inp,m.demand)
errors={name:float(np.max(abs(tag[i].sum(-1)-v[name]))) for name,i in [('fast',0),('slow',1),('M',3),('L',4),('uptake',5),('mineral_loss',6)]}
errors['per_source_balance']=float(tag[8]);scale=float((v['fast']+v['slow']).sum())
receipt=dict(local_balance_max_kg=float(v['local_balance_max_kg']),network_balance_kg=float(v['network_balance_kg']),network_relative_error=abs(float(v['network_balance_kg']))/scale,source_label_sum_errors=errors,source_minimum_kg=float(tag[9]),reaches=230,days=len(m.data.dates),NSE='not applicable')
receipt['passed']=receipt['local_balance_max_kg']<=1e-6 and receipt['network_relative_error']<=1e-10 and max(errors.values())<=1e-6 and receipt['source_minimum_kg']>=-1e-7
if len(sys.argv)>1:
    receipt['parameter_sha256']=sha(a.folder/'best.npy')
    receipt['final_inventory_M_kg']=float(v['M'][-1].sum())
    receipt['final_inventory_L_kg']=float(v['L'][-1].sum())
    receipt['full_history_uptake_kg']=float(v['uptake'].sum())
    receipt['full_history_loss_kg']=float(v['mineral_loss'].sum())
    np.savez_compressed(a.folder/'full_history_land_ledger.npz',**{k:v[k] for k in ['M','L','fast','slow','uptake','mineral_loss']})
    write_json(a.folder/'physical_ledger.json',receipt)
else:write_json(ROOT/'outputs/u_initial_ledger.json',receipt)
print(json.dumps(receipt),flush=True)
