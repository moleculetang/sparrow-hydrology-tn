"""Training-block parameter gradients; no performance or causal claim."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json
configure()
import numpy as np
import torch
from d29_training.u_adapter import UTraining

j=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']==sys.argv[1])
if j['model']!='U':raise RuntimeError('LAND1_REQUIRES_ITS_ACCEPTED_ADJOINT')
a=UTraining(j);x=np.load(a.folder/'best.npy')
status=json.loads((a.folder/'status.json').read_text(encoding='utf-8'))
if status['status']=='running':raise RuntimeError('WAIT_FOR_FROZEN_TRAINING_POINT')
t=torch.tensor(x,requires_grad=True);prediction=a.model.tensor_predict(t,a.meta)
p=prediction.detach().numpy();gradients={};values={}
for b in a.objective.blocks:
    error=b.center(p[b.indices]-a.objective.truth[b.indices]);dp=np.zeros_like(p)
    dp[b.indices]=b.transpose_center(b.sqrt_weight*b.sqrt_weight*error)
    gradients[b.name]=torch.autograd.grad(prediction,t,torch.tensor(dp),retain_graph=True)[0].detach().numpy()
    values[b.name]=float(.5*np.dot(b.sqrt_weight*error,b.sqrt_weight*error))
prior=a.model.prior(t);R=.5*torch.dot(prior,prior)
gradients['prior']=torch.autograd.grad(R,t)[0].detach().numpy();values['prior']=float(R.detach())
_,total=a.value_gradient(x);sumg=np.sum(list(gradients.values()),axis=0)
error=float(np.max(abs(sumg-total)))
if error>1e-8*(1+np.max(abs(total))):raise RuntimeError('BLOCK_GRADIENT_SUM_MISMATCH')
scale=a.model.variable_scale()
def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>0 else None
names=list(gradients);pairs=[]
for i,k in enumerate(names):
    for l in names[i+1:]:pairs.append({'first':k,'second':l,'raw_coordinate_cosine':cosine(gradients[k],gradients[l]),'fixed_native_scaled_coordinate_cosine':cosine(gradients[k]*scale,gradients[l]*scale)})
write_json(a.folder/'training_block_gradients.json',{'parameter_names':a.model.names,'block_values':values,'raw_gradients':{k:v.tolist() for k,v in gradients.items()},'fixed_parameter_scales':scale.tolist(),'cosines':pairs,'gradient_sum_max_error':error,'NSE':'not applicable to parameter gradients','interpretation':'negative dot products indicate local objective-direction conflict, not proof of incorrect observations or a unique structural cause'})
print(j['id'],'block gradient sum error',error)
