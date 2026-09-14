"""Data-free publication check; never imports a campaign controller or reads TN."""
from pathlib import Path
import os,sys,ast,json,hashlib,tempfile,types
ROOT=Path(__file__).resolve().parents[2]
assert Path(sys.prefix).name.lower()=='sparrow','Use conda sparrow'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
os.environ['NUMBA_CACHE_DIR']=str(Path(tempfile.gettempdir())/'sparrow_review_20260911_numba')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((Path(__file__).parent/'source_manifest.json').read_text(encoding='utf-8'))
for row in m['source_files']+m['derived_aggregate_tables']:
    p=ROOT/row['path'];assert p.is_file() and sha(p)==row['sha256'],str(p)
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
import numpy as np
import torch
sys.path.insert(0,str(ROOT/'5_Test/20260911_1/scripts'))
from sc_kernel import SharedN,forward
torch.set_num_threads(1)
nd,nr=14,2
mid=np.repeat(np.arange(2),7);source=np.array([[20.,12.],[8.,15.]]);crop=np.array([[3.,2.],[2.,1.]])
water=np.linspace(.2,.8,nd)[:,None]*np.ones((1,nr));delta=np.linspace(-.1,.15,nd)[:,None]*np.ones((1,nr))
scale=np.array([5.,8.]);lower=np.full((nd,nr),.04)
data=types.SimpleNamespace(lower_release=lower,source=source,crop=crop,mid=mid)
h=np.full((nd,nr),.06);s=np.array([.994,.989]);f=np.full((nd,nr),.35);c=np.array([.03,-.04,.06,-.01,.02,-.025])
args=tuple(torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in (h,s,f,c))
assert torch.autograd.gradcheck(lambda hh,ss,ff,cc:SharedN.apply(hh,ss,ff,cc,data,water,delta,scale),args,
 eps=1e-6,atol=3e-6,rtol=3e-5),'FULL_TIME_FEEDBACK_GRADIENT'

def independent(coeff):
    M=np.zeros(nr);L=np.zeros(nr);fast=[];slow=[];balance=[]
    for t in range(nd):
        before=M+L;first=t==0 or mid[t]!=mid[t-1]
        inp=source[mid[t]] if first else np.zeros(nr);d=crop[mid[t]] if first else np.zeros(nr)
        uptake=np.minimum(M+inp,d);A=M+inp-uptake;z=(A-scale)/(A+scale);p2=(3*z*z-1)/2
        B=np.array([z,p2,z*delta[t],p2*delta[t],z*water[t]*delta[t],p2*water[t]*delta[t]])
        lam=h[t]*np.exp(np.log(4)*np.tanh(coeff@B/np.log(4)));E=A*(-np.expm1(-np.minimum(lam,700)))
        ff=E*f[t];L0=L+E*(1-f[t]);ss=L0*lower[t];loss=(A-E)*(1-s);M=(A-E)*s;L=L0-ss
        balance.append(before+inp-uptake-ff-ss-loss-M-L);fast.append(ff);slow.append(ss)
        assert np.all(E<=A) and np.all(uptake<=d) and np.all(M>=0) and np.all(L>=0)
    np.testing.assert_allclose(balance,0,atol=1e-12)
    return np.array(fast),np.array(slow)
for cc in (c,np.zeros(6)):
    out=forward(h,s,f,lower,water,delta,scale,cc,source,crop,mid)
    for a,b in zip(out[:2],independent(cc)):np.testing.assert_allclose(a,b,rtol=2e-13,atol=2e-13)
for hh,src in ((np.zeros_like(h),source),(h,np.zeros_like(source))):
    out=forward(hh,s,f,lower,water,delta,scale,c,src,crop,mid)
    assert not np.any(out[0]) and not np.any(out[1])
original=forward(h,s,f,lower,water,delta,scale,c,source,crop,mid)
changed_h=h.copy();changed_h[10:]*=3
changed=forward(changed_h,s,f,lower,water,delta,scale,c,source,crop,mid)
for a,b in zip(original,changed):np.testing.assert_array_equal(a[:10],b[:10])
print(json.dumps({'status':'PASS_SOURCE_ONLY_REVIEW_CHECK','source_files':len(m['source_files']),
 'aggregate_tables':len(m['derived_aggregate_tables']),'checks':['SHA256','Python syntax','full synthetic temporal autograd gradcheck',
 'independent recurrence and zero extension','nitrogen balance and stock/uptake bounds','zero carrier and source','frozen preprocessing causality'],
 'original_observations_read':False,'training_started':False,'full_experiment_reproduction_claimed':False}))
