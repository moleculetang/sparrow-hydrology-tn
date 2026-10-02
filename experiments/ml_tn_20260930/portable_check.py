"""One fixed-weight artifact across both PyTorch/CUDA versions, independent targets."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,platform
import numpy as np,torch
from mltn.common import ROOT,write
from mltn.models import Network
def main(create=False):
    torch.set_num_threads(1);folder=ROOT/'evidence/portable';folder.mkdir(parents=True,exist_ok=True);families=['MLP','LSTM','GRU','TCN','Transformer','GraphTCN'];cfg=dict(width=32,dropout=0.)
    if create:
        x=np.random.default_rng(1729).normal(size=(4,90,12)).astype('float32');np.save(folder/'input.npy',x);torch.manual_seed(1729)
        for family in families:
            m=Network(family,12,cfg).eval();torch.save(m.state_dict(),folder/f'{family}.pt');np.save(folder/f'{family}_baseline.npy',m(torch.tensor(x)).detach().numpy())
    x=np.load(folder/'input.npy');rows=[]
    for family in families:
        baseline=np.load(folder/f'{family}_baseline.npy');m=Network(family,12,cfg).eval();m.load_state_dict(torch.load(folder/f'{family}.pt',map_location='cpu',weights_only=True))
        for device in ['cpu','cuda'] if torch.cuda.is_available() else ['cpu']:
            p=m.to(device)(torch.tensor(x,device=device)).detach().cpu().numpy();np.testing.assert_allclose(p,baseline,rtol=3e-5,atol=3e-6);j=float(np.sum((p.astype(float)-np.arange(4)/4)**2));j0=float(np.sum((baseline.astype(float)-np.arange(4)/4)**2));assert abs(j-j0)<1e-4
            rows.append(dict(family=family,device=device,max_prediction_error=float(np.max(np.abs(p-baseline))),objective_difference=j-j0))
    write(ROOT/'evidence'/f'portable_check_{platform.system()}.json',dict(passed=True,torch=torch.__version__,host=platform.node(),checks=rows))
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('--create',action='store_true');a=p.parse_args();main(a.create)
