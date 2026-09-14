"""Deterministic SciPy residual-call replay with immutable, budgeted requests."""
import hashlib
import time
import numpy as np
from scipy.optimize import least_squares
from fc_io import local,read,write,sha
from fc_checkpoint import save,ResourceYield,BudgetStop
from fc_solvers import version_guard,projected_gradient
import torch


class CallReplay:
    def __init__(self,path,identity,guard=lambda:None,max_calls=4000,progress=lambda *_:None):
        self.path=local(path);self.identity=identity;self.guard=guard;self.max_calls=max_calls;self.progress=progress
        self.path.mkdir(exist_ok=True);self.cursor=0
        header=self.path/'index.json'
        if header.exists():
            self.state=read(header)
            if self.state['identity']!=identity:raise RuntimeError('REPLAY_IDENTITY_CHANGED')
        else:
            self.state={'identity':identity,'calls':[],'created_unix':time.time(),'best':None,'attempts':0}
            write(header,self.state)

    def persist(self):write(self.path/'index.json',self.state)

    def call(self,x,fun):
        self.guard();x=np.asarray(x,dtype=np.float64)
        request=hashlib.sha256(x.tobytes()).hexdigest();i=self.cursor
        if i<len(self.state['calls']):
            entry=self.state['calls'][i]
            if entry['request']!=request:raise RuntimeError(f'NONDETERMINISTIC_TRF_REPLAY_AT_{i}')
            path=self.path/entry['file']
            if sha(path)!=entry['sha256']:raise RuntimeError('REPLAY_CACHE_CORRUPTED')
            cached=torch.load(path,weights_only=False,map_location='cpu')
            if not np.array_equal(cached['x'],x):raise RuntimeError('REPLAY_REQUEST_COLLISION')
            self.cursor+=1;return cached['residual'].copy()
        if self.state['attempts']>=self.max_calls:raise BudgetStop('FOUR_THOUSAND_FORWARD_CALLS')
        # Attempt is charged before calling; an interrupted call is never free.
        self.state['attempts']+=1;self.persist()
        residual=np.asarray(fun(x),dtype=np.float64)
        if residual.ndim!=1 or not np.isfinite(residual).all():raise FloatingPointError('INVALID_RESIDUAL')
        name=f'{i:05d}.pt';digest=save(self.path/name,{'x':x.copy(),'residual':residual.copy()})
        value=float(.5*residual@residual)
        self.state['calls'].append({'request':request,'file':name,'sha256':digest,'objective':value})
        if self.state['best'] is None or value<self.state['best']['objective']:
            self.state['best']={'x':x.tolist(),'objective':value,'index':i}
        self.cursor+=1;self.persist();self.progress(self.state)
        return residual


def trf_fit(model,x0,replay):
    version_guard()
    lo=np.array([-np.inf if a is None else a for a,b in model.bounds])
    hi=np.array([np.inf if b is None else b for a,b in model.bounds])
    result=least_squares(lambda x:replay.call(x,model.residual),np.asarray(x0,dtype=float),
        bounds=(lo,hi),jac='3-point',method='trf',tr_solver='exact',x_scale=model.variable_scale(),
        ftol=1e-10,xtol=1e-10,gtol=1e-8,max_nfev=500,loss='linear')
    return {'x':result.x,'objective':float(result.cost),'status':int(result.status),
            'message':result.message,'nfev':int(result.nfev),'njev':int(result.njev),'optimality':float(result.optimality)}


def authoritative_candidate(model,points,guard=lambda:None,evaluate=None):
    evaluated=[]
    for source,x in points:
        xx=np.asarray(x,float)
        if not np.isfinite(xx).all() or any((lo is not None and xx[i]<lo) or (hi is not None and xx[i]>hi) for i,(lo,hi) in enumerate(model.bounds)):
            raise ValueError('INFEASIBLE_FINAL_CANDIDATE '+source)
        guard();f,g=(evaluate or model.value_gradient)(xx)
        pg=float(np.max(np.abs(projected_gradient(np.asarray(x),g,model.bounds))))
        if not np.isfinite(f) or not np.isfinite(g).all():raise FloatingPointError('NONFINITE_FINAL_REPLAY')
        evaluated.append({'source':source,'x':np.asarray(x).tolist(),'objective':f,'projected_gradient':pg})
    best=min(evaluated,key=lambda v:v['objective'])
    return best,evaluated
