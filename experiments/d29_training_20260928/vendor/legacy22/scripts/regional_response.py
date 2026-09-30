"""Frozen spatial modes, local dynamic interactions, unchanged conservative D29.

The low-rank object is the reach-by-feature coefficient matrix, not TN.
Block recomputation below is an exact chain rule over every historical day;
it does not detach or truncate the land or river state adjoints.
"""
import math
import numpy as np
import torch
from source_corrected import SourceCorrected
from campaign_model import Predictor, RUN, sha

KINDS = {'REGIONAL_L1': 1, 'REGIONAL_L3': 3, 'REGIONAL_N3': 3}

def hidden_initial():
    h = np.random.default_rng(1729).normal(size=(2, 8))
    return h / np.linalg.norm(h, axis=1)[:, None]

def added_initial(kind):
    if kind == 'SOURCE_UNIFIED': return np.empty(0)
    k = KINDS[kind]
    return np.r_[np.zeros(k*8), hidden_initial().ravel(), np.zeros(2+6)] if kind == 'REGIONAL_N3' else np.zeros(k*8)

def embed(x, old_kind, new_kind):
    x = np.asarray(x, dtype=np.float64)
    if old_kind == new_kind: return x.copy()
    if new_kind == 'SOURCE_UNIFIED':
        if old_kind != 'SOURCE_UNIFIED': raise ValueError('NOT_NESTED')
        return x.copy()
    y = np.r_[x[:31], added_initial(new_kind)]
    if old_kind in KINDS:
        ko, kn = KINDS[old_kind], KINDS[new_kind]
        if ko > kn or old_kind == 'REGIONAL_N3': raise ValueError('NOT_NESTED')
        y[31:31+ko*8] = x[31:31+ko*8]
    elif old_kind != 'SOURCE_UNIFIED': raise ValueError('NOT_NESTED')
    return y

def unpack(v, k, neural):
    a = v[:k*8].reshape(k,8)
    if not neural: return a, None, None, None
    return a, v[24:40].reshape(2,8), v[40:42], v[42:48].reshape(3,2)

def numpy_delta(v, phi, psi, k, neural):
    a,h,b,w=unpack(v,k,neural)
    out=np.einsum('trj,rj->tr',phi,psi@a,optimize=False)
    if neural:
        z=np.tanh(np.einsum('trj,ij->tri',phi,h,optimize=False)+b)
        out+=np.einsum('tri,ri->tr',z-np.tanh(b),psi@w,optimize=False)
    return out

class RegionalDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, owner):
        vv=v.detach().numpy().copy();phi=owner.dynamic_basis.numpy();psi=owner.psi
        out=np.empty(phi.shape[:2],dtype=np.float64)
        for start in range(0,len(phi),owner.block_days):
            stop=min(start+owner.block_days,len(phi))
            out[start:stop]=numpy_delta(vv,phi[start:stop],psi,owner.rank,owner.neural)
        ctx.v=vv;ctx.owner=owner
        return torch.from_numpy(out)

    @staticmethod
    def backward(ctx, g):
        o=ctx.owner;phi=o.dynamic_basis.numpy();psi=o.psi;gg=g.numpy()
        a,h,b,w=unpack(ctx.v,o.rank,o.neural)
        ga=np.zeros_like(a);gh=np.zeros((2,8));gb=np.zeros(2);gw=np.zeros((3,2))
        if o.neural: pw=psi@w;zb=np.tanh(b)
        for start in range(0,len(phi),o.block_days):
            end=min(start+o.block_days,len(phi));x=phi[start:end];q=gg[start:end]
            ga+=psi.T@np.einsum('tr,trj->rj',q,x,optimize=False)
            if o.neural:
                z=np.tanh(np.einsum('trj,ij->tri',x,h,optimize=False)+b)
                gw+=psi.T@np.einsum('tr,tri->ri',q,z-zb,optimize=False)
                dz=q[:,:,None]*pw[None,:,:]
                inner=dz*(1-z*z)
                gh+=np.einsum('tri,trj->ij',inner,x,optimize=False)
                gb+=np.sum(dz*((1-z*z)-(1-zb*zb)),axis=(0,1))
        grad=np.r_[ga.ravel(),gh.ravel(),gb,gw.ravel()] if o.neural else ga.ravel()
        return torch.from_numpy(grad),None

class RegionalResponse(SourceCorrected):
    def __init__(self,data,train,design,kind):
        super().__init__(data,train,design,'SOURCE_UNIFIED')
        self.regional_kind=kind;self.rank=KINDS[kind];self.neural=kind=='REGIONAL_N3';self.block_days=256
        spec=design['regional_basis'];path=RUN/spec['file']
        if sha(path)!=spec['sha256']:raise ValueError('REGIONAL_BASIS_CHANGED')
        self.psi=np.ascontiguousarray(np.load(path,allow_pickle=False)[:,:self.rank])
        if self.psi.shape!=(len(data.area_ha),self.rank) or not np.isfinite(self.psi).all():raise ValueError('INVALID_SPATIAL_BASIS')
        self.names += [f'regional_A_{i}_{j}' for i in range(self.rank) for j in range(8)]
        self.bounds += [(-12.,12.)]*(self.rank*8)
        if self.neural:
            self.names += [f'neural_H_{i}_{j}' for i in range(2) for j in range(8)]+[f'neural_a_{i}' for i in range(2)]+[f'neural_V_{i}_{j}' for i in range(3) for j in range(2)]
            self.bounds += [(-2.,2.)]*18+[(-12.,12.)]*6

    def initial(self,index=0):return np.r_[super().initial(index),added_initial(self.regional_kind)]
    def variable_scale(self):return np.r_[super().variable_scale()[:31],np.repeat(.5,len(self.names)-31)]
    def prior(self,t):
        extra=[t[31:31+self.rank*8]]
        if self.neural:extra.append(t[73:79])
        return torch.cat([super().prior(t[:31]),*[math.sqrt(.03/17.)*v/.5 for v in extra]])

    def flux_parameters(self,t):
        # Preserve the original operation order at delta=0.
        h,s,f=Predictor.hazard(self,t[:29]);shift=self.pi*(t[29]-t[1])
        h=h*torch.exp(shift[None,:]*self.logcontact)
        u=torch.einsum('trj,j->tr',self.dynamic_basis,t[21:29])
        delta=RegionalDelta.apply(t[31:],self)
        h=h*torch.exp(math.log(10)*torch.tanh((u+delta)/math.log(10)))
        return h,s,f,torch.ones(self.data.source.shape[1])

    def value_gradient(self,x):
        j,g=super().value_gradient(x)
        ra=.5*(.03/17.)*np.sum((np.asarray(x)[31:31+self.rank*8]/.5)**2)
        rv=.5*(.03/17.)*np.sum((np.asarray(x)[73:79]/.5)**2) if self.neural else 0.
        source=.5*(.03/17.)*(x[30]/math.log(2))**2
        self.last_terms.update(source_prior=float(source),mapping_A_prior=float(ra),mapping_V_prior=float(rv),new_prior=float(source+ra+rv),original_prior=float(self.last_terms['prior']-source-ra-rv))
        return j,g
